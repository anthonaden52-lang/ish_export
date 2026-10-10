"""On-chain and market safety checks (scam / rug filter).

READ-ONLY: only calls Solana RPC read methods. Never touches a wallet.

What IS checked (a token that fails a hard check is rejected):
  - mint authority still set            (creator can print unlimited tokens)
  - freeze authority still set          (creator can freeze your tokens)
  - Token-2022 extensions               (permanent delegate / transfer hook / non-transferable = reject,
                                         transfer fee = warning)
  - holder concentration                (top holder and top-10, with known pool vaults excluded)
  - token age                           (very new pairs are the riskiest)
  - liquidity drop over ~15 minutes     (a rug in progress)

What is NOT checked (reported in "not_checked" every time, never silently skipped):
  - creator / developer wallet activity
  - liquidity lock / burn status
  - wallet clustering (one person holding through many wallets)
  - social / narrative data
A passing result means "no red flag found by these checks", not "safe".
Fail closed: if the RPC cannot be reached, the token is NOT marked safe.
"""
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
if urlparse(RPC_URL).scheme != "https":
    raise SystemExit("SOLANA_RPC_URL must be an https URL.")

MAX_TOP1_PCT = float(os.getenv("MAX_TOP1_PCT", "20"))      # % of supply, pool vaults excluded
MAX_TOP10_PCT = float(os.getenv("MAX_TOP10_PCT", "50"))
MIN_AGE_MINUTES = float(os.getenv("MIN_AGE_MINUTES", "30"))
MAX_LIQ_DROP_15 = float(os.getenv("MAX_LIQ_DROP_15", "0.20"))  # reject if liquidity fell > 20% in 15 min
CACHE_MINUTES = float(os.getenv("SAFETY_CACHE_MIN", "30"))
CACHE_FILE = ROOT / "data" / "safety_cache.json"

# Accounts owned by these addresses are AMM pool vaults, not real holders. Editable:
# verify against a block explorer. The pair address itself is always excluded too.
KNOWN_POOL_OWNERS = {
    "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",  # Raydium AMM v4 authority
    "GpMZbSM2GgvTKHJirzeGfMFoaZ8UR2X7F4v8vHTvxFbL",  # Raydium CPMM authority
} | {a.strip() for a in os.getenv("KNOWN_POOL_OWNERS", "").split(",") if a.strip()}

BAD_EXTENSIONS = {"permanentDelegate", "transferHook", "nonTransferable", "confidentialTransferMint"}
WARN_EXTENSIONS = {"transferFeeConfig", "defaultAccountState", "pausableConfig"}

NOT_CHECKED = [
    "creator/developer wallet activity",
    "liquidity lock/burn status",
    "wallet clustering",
    "social/narrative data",
]


class RpcError(RuntimeError):
    pass


def rpc(method, params, tries=3):
    delay = 1.5
    for _ in range(tries):
        try:
            r = requests.post(RPC_URL, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                              timeout=30, allow_redirects=False)
        except requests.RequestException as e:
            raise RpcError(f"{method}: {e}") from e
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(delay)
            delay *= 2
            continue
        if not r.ok:
            raise RpcError(f"{method}: HTTP {r.status_code}")
        data = r.json()
        if "error" in data:
            raise RpcError(f"{method}: {data['error']}")
        return data["result"]
    raise RpcError(f"{method}: rate limited / server error after {tries} tries")


def _onchain(mint, pair_address):
    """The slow, cacheable part: authorities, extensions, holder distribution."""
    hard, warn, metrics = [], [], {}

    acct = rpc("getAccountInfo", [mint, {"encoding": "jsonParsed"}])["value"]
    if not acct or "parsed" not in (acct.get("data") or {}):
        return {"hard": ["not a readable SPL token mint"], "warn": [], "metrics": {}}
    info = acct["data"]["parsed"]["info"]
    program = acct["data"].get("program", "")
    metrics["program"] = program
    metrics["decimals"] = info.get("decimals")

    if info.get("mintAuthority"):
        hard.append("mint authority not revoked (creator can mint more supply)")
    if info.get("freezeAuthority"):
        hard.append("freeze authority not revoked (creator can freeze holders)")
    for ext in info.get("extensions") or []:
        name = ext.get("extension")
        if name in BAD_EXTENSIONS:
            hard.append(f"Token-2022 extension {name}")
        elif name in WARN_EXTENSIONS:
            warn.append(f"Token-2022 extension {name}")

    supply = int(info.get("supply") or 0)
    if supply <= 0:
        return {"hard": hard + ["zero supply"], "warn": warn, "metrics": metrics}

    top = rpc("getTokenLargestAccounts", [mint])["value"]
    owners = {}
    if top:
        got = rpc("getMultipleAccounts", [[t["address"] for t in top], {"encoding": "jsonParsed"}])["value"]
        for t, a in zip(top, got):
            try:
                owners[t["address"]] = a["data"]["parsed"]["info"]["owner"]
            except (TypeError, KeyError):
                owners[t["address"]] = None
    pool_owners = KNOWN_POOL_OWNERS | ({pair_address} if pair_address else set())
    holders = []
    for t in top:
        owner = owners.get(t["address"])
        if owner in pool_owners:
            continue
        holders.append(int(t["amount"]) / supply * 100)
    holders.sort(reverse=True)
    top1 = holders[0] if holders else 0.0
    top10 = sum(holders[:10])
    metrics.update({"top1_pct": round(top1, 2), "top10_pct": round(top10, 2),
                    "pool_vaults_excluded": len(top) - len(holders)})
    if top1 > MAX_TOP1_PCT:
        hard.append(f"top holder owns {top1:.1f}% of supply (limit {MAX_TOP1_PCT:g}%; "
                    "if this is an unrecognised pool vault, add its owner to KNOWN_POOL_OWNERS)")
    if top10 > MAX_TOP10_PCT:
        hard.append(f"top 10 holders own {top10:.1f}% of supply (limit {MAX_TOP10_PCT:g}%)")
    return {"hard": hard, "warn": warn, "metrics": metrics}


def _cache():
    try:
        return json.loads(CACHE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def cached_onchain(mint, pair_address=""):
    """Returns the on-chain result, or None if it is not cached (call check_token to fetch)."""
    hit = _cache().get(mint)
    if hit and time.time() - hit["ts"] < CACHE_MINUTES * 60:
        return hit["result"]
    return None


def row_checks(row):
    """Cheap checks from market data we already have. Always re-evaluated, never cached."""
    hard, warn = [], []
    age = row.get("pair_age_minutes")
    if age is not None and age == age:
        if age < MIN_AGE_MINUTES:
            hard.append(f"pair is only {age:.0f} min old (minimum {MIN_AGE_MINUTES:g})")
    else:
        warn.append("pair age unknown")
    ch = row.get("liquidity_change_15")
    if ch is not None and ch == ch and ch < -MAX_LIQ_DROP_15:
        hard.append(f"liquidity fell {abs(ch):.0%} in ~15 min")
    return hard, warn


def check_token(row, fetch=True):
    """Full verdict for one indicator row. fetch=False uses only cache (no RPC calls).

    Returns {"passed": bool|None, "hard": [...], "warn": [...], "metrics": {...}, "not_checked": [...]}.
    passed=None means "not verified yet" (cache miss with fetch=False) and must be treated as NOT safe.
    """
    mint = row["token_address"]
    hard, warn = row_checks(row)
    onchain = cached_onchain(mint)
    if onchain is None:
        if not fetch:
            return {"passed": None, "hard": hard, "warn": warn + ["on-chain check pending"],
                    "metrics": {}, "not_checked": NOT_CHECKED}
        try:
            onchain = _onchain(mint, row.get("pair_address", ""))
        except RpcError as e:
            return {"passed": False, "hard": hard + [f"could not verify on-chain data ({e})"],
                    "warn": warn, "metrics": {}, "not_checked": NOT_CHECKED}
        cache = _cache()
        cache[mint] = {"ts": time.time(), "result": onchain}
        CACHE_FILE.parent.mkdir(exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(CACHE_FILE)
    hard = hard + onchain["hard"]
    warn = warn + onchain["warn"]
    return {"passed": not hard, "hard": hard, "warn": warn, "metrics": onchain["metrics"],
            "not_checked": NOT_CHECKED}
