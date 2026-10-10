import os
import math
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from data_quality import normalize_market_row
from process_lock import ProcessLock

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BASE = "https://api.dexscreener.com"
MIN_LIQ = float(os.getenv("MIN_LIQUIDITY_USD", "25000"))
MIN_VOL = float(os.getenv("MIN_VOLUME_24H_USD", "50000"))
MIN_TXNS = int(os.getenv("MIN_TXNS_24H", "100"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS_PER_SCAN", "25"))
MAX_LOOKUPS = int(os.getenv("MAX_LOOKUPS_PER_SCAN", "60"))
API_TIMEOUT = float(os.getenv("DEXSCREENER_TIMEOUT_SECONDS", "20"))
MAX_RETRIES = int(os.getenv("DEXSCREENER_MAX_RETRIES", "4"))
if not 1 <= MAX_TOKENS <= 250 or not 1 <= MAX_LOOKUPS <= 250:
    raise ValueError("MAX_TOKENS_PER_SCAN and MAX_LOOKUPS_PER_SCAN must be between 1 and 250.")
if not 1 <= MAX_RETRIES <= 8 or not 1 <= API_TIMEOUT <= 60:
    raise ValueError("DexScreener timeout must be 1..60 seconds and retries must be 1..8.")

session = requests.Session()
session.headers.update({"User-Agent": "MemeAIResearchBot/1.0"})
RATE_FILE = ROOT / "data" / "dexscreener_rate.json"
RATE_LOCK = ROOT / "data" / "dexscreener_rate.lock"


def _rate_limit(url):
    """Shared process-wide sliding-window cap, including collector and paper-trader calls."""
    discovery = any(urlparse(url).path.startswith(path) for path in SOURCES.values())
    limit = 55 if discovery else 280  # documented ceilings are 60 and 300 requests/minute
    bucket_name = "discovery" if discovery else "pairs"
    while True:
        with ProcessLock(RATE_LOCK):
            now = time.time()
            try:
                payload = json.loads(RATE_FILE.read_text(encoding="utf-8")) if RATE_FILE.exists() else {}
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Cannot safely load DexScreener rate state {RATE_FILE}: {exc}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("DexScreener rate state is not a JSON object; refusing API requests.")
            raw = payload.get(bucket_name, [])
            if not isinstance(raw, list):
                raise RuntimeError("DexScreener rate state has an invalid request bucket.")
            stamps = []
            for stamp in raw:
                if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                    raise RuntimeError("DexScreener rate state has an invalid timestamp.")
                if stamp >= now - 60:
                    stamps.append(float(stamp))
            if len(stamps) < limit:
                stamps.append(now)
                payload[bucket_name] = stamps
                RATE_FILE.parent.mkdir(parents=True, exist_ok=True)
                tmp = RATE_FILE.with_suffix(".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    f.write(json.dumps(payload, allow_nan=False))
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(RATE_FILE)
                return
            oldest = min(stamps)
            wait = 60 - (now - oldest)
            if oldest > now:
                wait = oldest - now + 60
        time.sleep(max(0.05, min(wait, 60)))


class ApiSchemaError(ValueError):
    """An API returned a successful HTTP response with an unexpected shape."""


def _get(url, tries=None):
    """GET a documented DexScreener endpoint with bounded 429/5xx backoff."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "api.dexscreener.com":
        raise ValueError("Refusing a non-DexScreener or non-HTTPS scanner URL.")
    tries = MAX_RETRIES if tries is None else max(1, min(int(tries), 8))
    delay = 2.0
    for attempt in range(tries):
        _rate_limit(url)
        r = session.get(url, timeout=API_TIMEOUT)
        if r.status_code == 429 or r.status_code >= 500:
            try:
                wait = float(r.headers.get("Retry-After") or delay)
            except (TypeError, ValueError):
                wait = delay
            if not math.isfinite(wait) or wait < 0:
                wait = delay
            if attempt + 1 < tries:
                wait = min(wait, 60)
                print(f"[scanner] HTTP {r.status_code}, backing off {wait:.0f}s")
                time.sleep(wait)
            delay = min(delay * 2, 60)
            continue
        r.raise_for_status()
        try:
            return r.json()
        except ValueError as exc:
            raise ApiSchemaError("DexScreener returned invalid JSON.") from exc
    r.raise_for_status()
    raise requests.HTTPError(f"gave up after {tries} tries: {url}")


def get_latest_profiles():
    result = _get(f"{BASE}/token-profiles/latest/v1")
    if result is None:
        return []
    if not isinstance(result, list):
        raise ApiSchemaError("DexScreener token profiles response must be a list.")
    return result


def get_token_pairs(address):
    result = _get(f"{BASE}/token-pairs/v1/solana/{address}")
    if result is None:
        return []
    if not isinstance(result, list):
        raise ApiSchemaError("DexScreener token-pairs response must be a list.")
    return [pair for pair in result if isinstance(pair, dict)]


def get_pairs_by_address(pair_addresses):
    """Fetch specific pairs (up to 30 per call) regardless of filters."""
    out = []
    for i in range(0, len(pair_addresses), 30):
        chunk = ",".join(pair_addresses[i:i + 30])
        data = _get(f"{BASE}/latest/dex/pairs/solana/{chunk}") or {}
        if not isinstance(data, dict) or data.get("pairs") is not None and not isinstance(data.get("pairs"), list):
            raise ApiSchemaError("DexScreener pair lookup response must contain a pairs list.")
        out.extend(pair for pair in (data.get("pairs") or []) if isinstance(pair, dict))
        if i + 30 < len(pair_addresses):
            time.sleep(0.2)
    return out


def num(d, key, default=0):
    if not isinstance(d, dict):
        return default
    try:
        raw = d.get(key, default)
        if raw is None or raw == "":
            return default
        value = float(raw)
        return value if math.isfinite(value) else default
    except (OverflowError, TypeError, ValueError):
        return default


def pick_best_pair(pairs, address):
    good = []
    for p in pairs:
        if not isinstance(p, dict):
            continue
        if p.get("chainId") != "solana":
            continue
        # The token we looked up must be the BASE token, otherwise priceUsd is
        # the price of the other side of the pair (often SOL).
        base = p.get("baseToken")
        if not isinstance(base, dict) or base.get("address") != address:
            continue
        price = num(p, "priceUsd", float("nan"))
        if not math.isfinite(price) or price <= 0:
            continue
        liq = num(p.get("liquidity"), "usd", float("nan"))
        vol = num(p.get("volume"), "h24", float("nan"))
        txns = p.get("txns")
        t24 = txns.get("h24") if isinstance(txns, dict) else None
        if not isinstance(t24, dict):
            continue
        trade_count = num(t24, "buys", float("nan")) + num(t24, "sells", float("nan"))
        if liq >= MIN_LIQ and vol >= MIN_VOL and trade_count >= MIN_TXNS:
            good.append((p, liq, vol))
    if not good:
        return None
    return sorted(good, key=lambda item: (-item[1], -item[2], str(item[0].get("pairAddress", ""))))[0][0]


# Discovery sources. Each returns a list of {"chainId", "tokenAddress", ...} items.
# DexScreener has no official "trending" list; "boosts" are paid promotions, so
# treat them as a lead source only: every candidate still has to pass the same
# liquidity/volume/transaction filters and the safety checks.
SOURCES = {
    "new_profiles": "/token-profiles/latest/v1",
    "latest_boosts": "/token-boosts/latest/v1",
    "top_boosts": "/token-boosts/top/v1",
}
def discover_addresses():
    """Union of all sources, de-duplicated, keeping which sources named each token."""
    found = {}
    for name, path in SOURCES.items():
        try:
            items = _get(f"{BASE}{path}") or []
            if not isinstance(items, list):
                raise ApiSchemaError(f"DexScreener {name} response must be a list.")
        except (requests.RequestException, ApiSchemaError) as e:
            print(f"[scanner] source {name} failed: {e}")
            continue
        for item in items:
            if not isinstance(item, dict) or item.get("chainId") != "solana" or not item.get("tokenAddress"):
                continue
            address = item["tokenAddress"]
            if not isinstance(address, str) or not address.strip():
                continue
            found.setdefault(address.strip(), set()).add(name)
        time.sleep(0.2)
    return found


def momentum_rank(pair):
    """Cheap pre-ranking so the best movers are kept when there are more candidates than slots."""
    vol = pair.get("volume") or {}
    pc = pair.get("priceChange") or {}
    v5, v1 = num(vol, "m5"), num(vol, "h1")
    accel = (v5 / (v1 / 12)) if v1 > 0 else 0.0
    return accel + max(0.0, num(pc, "h1")) / 100.0


def scan():
    """Candidates from all sources that pass the filters right now (best momentum first)."""
    addresses = discover_addresses()
    candidates = []
    for n, address in enumerate(sorted(addresses)):
        if n >= MAX_LOOKUPS:
            break
        try:
            pair = pick_best_pair(get_token_pairs(address), address)
            if pair:
                pair["_sources"] = sorted(addresses[address])
                candidates.append(pair)
        except (requests.RequestException, ApiSchemaError):
            continue
        time.sleep(0.05)
    candidates.sort(key=momentum_rank, reverse=True)
    return candidates[:MAX_TOKENS]


def normalize(pair):
    tx = pair.get("txns") if isinstance(pair.get("txns"), dict) else {}
    vol = pair.get("volume") if isinstance(pair.get("volume"), dict) else {}
    pc = pair.get("priceChange") if isinstance(pair.get("priceChange"), dict) else {}
    liq = pair.get("liquidity") if isinstance(pair.get("liquidity"), dict) else {}
    t24 = tx.get("h24") if isinstance(tx.get("h24"), dict) else {}
    t5 = tx.get("m5") if isinstance(tx.get("m5"), dict) else {}
    t1 = tx.get("h1") if isinstance(tx.get("h1"), dict) else {}
    base = pair.get("baseToken")
    if not isinstance(base, dict):
        raise ApiSchemaError("DexScreener pair omitted its base token.")

    price = num(pair, "priceUsd", float("nan"))
    if not math.isfinite(price) or price <= 0:
        raise ApiSchemaError("DexScreener pair omitted a valid positive USD price.")

    created = num(pair, "pairCreatedAt", float("nan"))
    now = datetime.now(timezone.utc)
    age_min = (now.timestamp() * 1000 - created) / 60000 if math.isfinite(created) else float("nan")
    if math.isfinite(age_min) and age_min < -5:
        raise ApiSchemaError("DexScreener pair creation timestamp is in the future.")
    row = {
        "timestamp": now.isoformat(),
        "token_address": base.get("address"),
        "symbol": base.get("symbol", ""),
        "price_usd": price,
        "liquidity_usd": num(liq, "usd", float("nan")),
        "volume_24h": num(vol, "h24", float("nan")),
        "volume_5m": num(vol, "m5", float("nan")),
        "buys_5m": num(t5, "buys", float("nan")),
        "sells_5m": num(t5, "sells", float("nan")),
        "txns_24h": num(t24, "buys", float("nan")) + num(t24, "sells", float("nan")),
        "price_change_5m": num(pc, "m5", float("nan")),
        "price_change_1h": num(pc, "h1", float("nan")),
        "pair_age_minutes": age_min if math.isfinite(age_min) and age_min >= 0 else float("nan"),
        "pair_address": pair.get("pairAddress", ""),
        "dex": pair.get("dexId", ""),
        "url": pair.get("url", ""),
        "market_cap": num(pair, "marketCap", float("nan")),
        "fdv": num(pair, "fdv", float("nan")),
        "volume_1h": num(vol, "h1", float("nan")),
        "buys_1h": num(t1, "buys", float("nan")),
        "sells_1h": num(t1, "sells", float("nan")),
        "source": "dexscreener",
        "discovery_sources": pair.get("_sources", []),
    }
    return normalize_market_row(row, source_default="dexscreener")


def dead_row(token_address, symbol, pair_address, discovery_sources="unknown"):
    """A tracked pair that vanished: zero marks the exit; unknown metrics stay missing."""
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "token_address": token_address,
            "symbol": symbol, "price_usd": 0.0, "liquidity_usd": float("nan"), "volume_24h": float("nan"),
            "volume_5m": float("nan"), "buys_5m": float("nan"), "sells_5m": float("nan"),
            "txns_24h": float("nan"), "price_change_5m": float("nan"), "price_change_1h": float("nan"),
            "pair_age_minutes": float("nan"),
            "pair_address": pair_address, "dex": "", "url": "", "source": "dexscreener",
            "discovery_sources": discovery_sources,
            "market_cap": float("nan"), "fdv": float("nan"), "volume_1h": float("nan"),
            "buys_1h": float("nan"), "sells_1h": float("nan")}
