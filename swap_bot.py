"""
Jupiter quote and transaction research script for Solana (Swap API v2).

Uses https://api.jup.ag/swap/v2 : quote requests are read-only. Live execution is hard-disabled.

Setup:
    pip install requests solders
    Set JUPITER_API_KEY in your environment if the read-only quote endpoint requires it.

Usage:
    python swap_bot.py --selftest                      # offline checks, no network
    python swap_bot.py --from SOL --to USDC --amount 0.01             # quote only
    python swap_bot.py --from SOL --to <MINT> --amount 0.05 --execute  # refuses before API/key access

The signing/submission code is unreachable because LIVE_EXECUTION_ENABLED is hard-coded False.
Transaction instruction inspection and exact-message simulation remain incomplete; do not change
the gate based on the existing quote or risk checks.
"""

import argparse
import base64
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import requests
from solders.keypair import Keypair
from solders.message import to_bytes_versioned
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import VersionedTransaction

# Only these hosts may be used. Env vars can pick among them but not add new ones.
ALLOWED_JUP_HOSTS = {"api.jup.ag"}
ALLOWED_RPC_HOSTS = {"api.mainnet-beta.solana.com"}  # add your own trusted RPC host here, in code
LIVE_EXECUTION_ENABLED = False


def _checked_url(url, allowed, name):
    u = urlparse(url)
    if u.scheme != "https" or u.hostname not in allowed:
        sys.exit(f"Refusing: {name}={url!r} is not an allowed https host {sorted(allowed)}. "
                 "Edit the allowlist in swap_bot.py if you really trust it.")
    return url.rstrip("/")


BASE = _checked_url(os.environ.get("JUPITER_BASE", "https://api.jup.ag/swap/v2"), ALLOWED_JUP_HOSTS, "JUPITER_BASE")
RPC_URL = _checked_url(os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"), ALLOWED_RPC_HOSTS, "SOLANA_RPC_URL")
API_KEY = os.environ.get("JUPITER_API_KEY", "")
TRADE_LOG = Path(__file__).resolve().parent / "data" / "trades.jsonl"

# symbol: (mint address, decimals)
TOKENS = {
    "SOL": ("So11111111111111111111111111111111111111112", 9),
    "USDC": ("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 6),
}

ORDER_ERRORS = {
    "aggregator": {1: "Insufficient funds (not enough of the input token)",
                   2: "Insufficient SOL for network fees",
                   3: "Swap below the minimum for gasless"},
    "jupiterz": {1: "Insufficient balance to fund the swap",
                 2: "Missing associated token account",
                 3: "Quote could not be built into a transaction"},
}
EXECUTE_ERRORS = {
    -1: "Order expired or not found (request again)",
    -2: "Invalid signed transaction",
    -3: "Invalid message bytes",
    -1000: "Failed to land on-chain",
    -1001: "Unknown aggregator error",
    -1002: "Invalid transaction",
    -1003: "Transaction not fully signed",
    -1004: "Invalid block height",
    -2000: "RFQ route failed to land",
    -2001: "Unknown RFQ error",
    -2002: "Invalid RFQ payload",
    -2003: "Quote expired",
    -2004: "Swap rejected by market maker",
}


def api_headers():
    return {"x-api-key": API_KEY} if API_KEY else {}


def explain_http(r):
    if r.status_code in (401, 403):
        return "Rejected. Set JUPITER_API_KEY (free key at https://portal.jup.ag)."
    if r.status_code == 429:
        return "Rate limited. Wait a bit, or use an API key for higher limits."
    return r.text[:300]


def to_float(v):
    """Jupiter may send numbers as strings. Return a finite float or None."""
    if isinstance(v, bool):
        return None
    try:
        number = float(v) if v is not None and not (isinstance(v, str) and not v.strip()) else None
        return number if number is not None and math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def optional_order_number(order, field):
    """Read optional numeric quote metadata, refusing malformed supplied values."""
    raw = order.get(field)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    number = to_float(raw)
    if number is None:
        sys.exit(f"Refusing: Jupiter returned an invalid {field} value.")
    return number


def validate_limits(a):
    """Reject invalid configured limits before any order can be signed."""
    for name in ("max_usd", "max_daily_usd", "max_loss_pct", "max_fee_bps",
                 "max_impact_pct", "max_slippage_bps"):
        value = to_float(getattr(a, name, None))
        if value is None or value < 0:
            sys.exit(f"Refusing: --{name.replace('_', '-')} must be a finite, non-negative number.")


def required_usd_value(order, field):
    value = to_float(order.get(field))
    if value is None or value <= 0:
        sys.exit(f"Refusing: Jupiter's {field} is missing or invalid; USD risk limits cannot be checked.")
    return value


def rpc(method, params):
    r = requests.post(RPC_URL, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                      timeout=30, allow_redirects=False)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"RPC error: {data['error']}")
    return data["result"]


def resolve_token(s):
    """Return (symbol, mint, decimals) for SOL, USDC, or a mint address."""
    key = s.upper()
    if key in TOKENS:
        mint, dec = TOKENS[key]
        return key, mint, dec
    try:
        Pubkey.from_string(s)
    except Exception:
        sys.exit(f"'{s}' is not SOL, USDC, or a valid mint address.")
    try:
        dec = int(rpc("getTokenSupply", [s])["value"]["decimals"])
    except Exception as e:
        sys.exit(f"Could not look up decimals for {s}: {e}")
    if not 0 <= dec <= 18:
        sys.exit(f"Refusing: implausible decimals {dec} for {s}.")
    return s[:4] + "..." + s[-4:], s, dec


def get_order(in_mint, out_mint, amount_raw, taker=None, slippage_bps=None):
    params = {"inputMint": in_mint, "outputMint": out_mint, "amount": str(amount_raw)}
    if taker:
        params["taker"] = taker  # required to receive a signable transaction
    if slippage_bps is not None:
        params["slippageBps"] = str(slippage_bps)  # switches to "manual" mode
    r = requests.get(f"{BASE}/order", params=params, headers=api_headers(), timeout=30,
                     allow_redirects=False)
    if not r.ok:
        sys.exit(f"/order failed ({r.status_code}): {explain_http(r)}")
    return r.json()


def sign_tx(tx_b64, keypair):
    """Sign our slot only. Some routes need a second signer added by Jupiter."""
    raw = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
    msg = raw.message
    n = msg.header.num_required_signatures
    signers = list(msg.account_keys)[:n]
    me = keypair.pubkey()
    if me not in signers:
        raise RuntimeError("This wallet is not a required signer on the transaction.")
    sigs = list(raw.signatures)
    if len(sigs) < n:
        sigs += [Signature.default()] * (n - len(sigs))
    idx = signers.index(me)
    payload = to_bytes_versioned(msg)
    sigs[idx] = keypair.sign_message(payload)
    if not sigs[idx].verify(me, payload):
        raise RuntimeError("Local signature check failed. Not sending.")
    return VersionedTransaction.populate(msg, sigs), idx


def execute_order(signed, request_id):
    """Returns (result_dict or None, error_text or None). Never exits: caller logs first."""
    body = {"signedTransaction": base64.b64encode(bytes(signed)).decode(), "requestId": request_id}
    try:
        r = requests.post(f"{BASE}/execute", json=body,
                          headers={**api_headers(), "Content-Type": "application/json"},
                          timeout=90, allow_redirects=False)
    except requests.RequestException as e:
        return None, f"network error: {e}"
    if not r.ok:
        return None, f"/execute failed ({r.status_code}): {explain_http(r)}"
    try:
        return r.json(), None
    except ValueError:
        return None, "unreadable /execute response"


# ---------- trade log ----------
def read_log():
    if not TRADE_LOG.exists():
        return []
    out = []
    for line_number, line in enumerate(TRADE_LOG.read_text().splitlines(), start=1):
        try:
            entry = json.loads(line)
        except ValueError:
            sys.exit(f"Refusing: trade log contains invalid JSON on line {line_number}.")
        if not isinstance(entry, dict):
            sys.exit(f"Refusing: trade log entry on line {line_number} is not an object.")
        out.append(entry)
    return out


def write_log(entries):
    TRADE_LOG.parent.mkdir(exist_ok=True)
    tmp = TRADE_LOG.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e) + "\n" for e in entries))
    tmp.replace(TRADE_LOG)


def log_update(entry_id, **fields):
    entries = read_log()
    for e in entries:
        if e["id"] == entry_id:
            e.update(fields)
    write_log(entries)


def chain_status(sig):
    """'landed', 'failed', 'not_found', or 'unknown' (RPC unreachable)."""
    try:
        res = rpc("getSignatureStatuses", [[sig], {"searchTransactionHistory": True}])["value"][0]
    except Exception:
        return "unknown"
    if res is None:
        return "not_found"
    return "failed" if res.get("err") else "landed"


def check_pending(ignore):
    """Resolve earlier trades whose outcome was unclear, so we never double-buy."""
    for e in read_log():
        if e.get("status") not in ("pending", "unclear"):
            continue
        st = chain_status(e["our_signature"]) if e.get("we_are_fee_payer") else "unknown"
        if st in ("landed", "failed"):
            log_update(e["id"], status=st, resolved_at=now_iso())
            print(f"Earlier trade {e['id']} resolved on-chain: {st}.")
            continue
        msg = (f"Earlier trade {e['id']} ({e.get('in_usd')} USD, {e.get('pair')}) has an unclear "
               f"outcome (chain status: {st}). Check https://solscan.io/tx/{e.get('our_signature')} "
               "and your wallet before trading again.")
        if not ignore:
            sys.exit("Refusing: " + msg + "\nUse --ignore-pending once you have checked.")
        print("WARNING: " + msg)
        log_update(e["id"], status="checked_by_user", resolved_at=now_iso())


def spent_today_usd():
    today = datetime.now(timezone.utc).date().isoformat()
    spent = 0.0
    for line_number, entry in enumerate(read_log(), start=1):
        stamp = entry.get("time")
        if not isinstance(stamp, str):
            sys.exit(f"Refusing: trade log entry {line_number} has no valid timestamp.")
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            sys.exit(f"Refusing: trade log entry {line_number} has an invalid timestamp.")
        if parsed.tzinfo is None:
            sys.exit(f"Refusing: trade log entry {line_number} has a timezone-ambiguous timestamp.")
        if parsed.astimezone(timezone.utc).date().isoformat() != today:
            continue
        if entry.get("status") in ("failed", "refused"):
            continue
        amount = to_float(entry.get("in_usd"))
        if amount is None or amount <= 0:
            sys.exit(f"Refusing: today's trade log entry {line_number} has a missing or invalid in_usd value.")
        spent += amount
        if not math.isfinite(spent):
            sys.exit("Refusing: today's trade log total is not finite.")
    return spent


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------- checks ----------
def describe_order_error(order):
    router = order.get("router", "")
    table = ORDER_ERRORS["jupiterz" if router == "jupiterz" else "aggregator"]
    code = order.get("errorCode")
    return f"[{router}] {table.get(code, 'Unknown error')} (code {code}): {order.get('errorMessage', '')}"


def read_impact_pct(order):
    """Price impact as a POSITIVE PERCENT (1.5 = 1.5%), or None if unreadable.

    UNITS ASSUMPTION (verify against current Jupiter docs):
      - `priceImpact`    : already a percent, may be negative (e.g. -0.12 = 0.12% worse)
      - `priceImpactPct` : a FRACTION (0.0012 = 0.12%), as in Jupiter's older quote API
    """
    v = optional_order_number(order, "priceImpact")
    if v is not None:
        return abs(v)
    v = optional_order_number(order, "priceImpactPct")
    if v is not None:
        return abs(v) * 100
    return None


def validate_order(order, in_mint, out_mint, amount_raw, me):
    """Check quote metadata only; this does not inspect or simulate transaction instructions."""
    problems = []
    if order.get("inputMint") != in_mint:
        problems.append(f"inputMint {order.get('inputMint')} != {in_mint}")
    if order.get("outputMint") != out_mint:
        problems.append(f"outputMint {order.get('outputMint')} != {out_mint}")
    if str(order.get("inAmount")) != str(amount_raw):
        problems.append(f"inAmount {order.get('inAmount')} != {amount_raw}")
    if order.get("taker") != me:
        problems.append(f"taker {order.get('taker')} != our wallet {me}")
    try:
        if int(order.get("outAmount", 0)) <= 0:
            problems.append("outAmount is zero or missing")
    except (TypeError, ValueError):
        problems.append("outAmount unreadable")
    if problems:
        sys.exit("Refusing: order does not match the request:\n  - " + "\n  - ".join(problems))


def check_value(order, a, out_dec, spent_today):
    """Size, daily cap, fees, price impact, slippage and total value loss."""
    validate_limits(a)
    in_usd = required_usd_value(order, "inUsdValue")
    out_usd = required_usd_value(order, "outUsdValue")
    fee_bps = optional_order_number(order, "feeBps")
    impact = read_impact_pct(order)
    out_amt = to_float(order.get("outAmount"))
    min_out = optional_order_number(order, "otherAmountThreshold")
    if out_amt is None or out_amt <= 0:
        sys.exit("Refusing: Jupiter returned an invalid outAmount.")
    if min_out is not None and (min_out <= 0 or min_out > out_amt):
        sys.exit("Refusing: Jupiter returned an invalid otherAmountThreshold.")
    slip_bps = (out_amt - min_out) / out_amt * 10000 if min_out is not None else None
    spent_today = to_float(spent_today)
    if spent_today is None or spent_today < 0:
        sys.exit("Refusing: today's recorded USD spend is missing or invalid.")
    if fee_bps is not None and fee_bps < 0:
        sys.exit("Refusing: Jupiter returned a negative feeBps value.")

    print("\n=== PRE-SWAP FEE & SLIPPAGE CHECK ===")
    print(f" Router fee:           {f'{fee_bps:.0f} bps' if fee_bps is not None else 'unknown'}")
    print(f" Price impact:         {f'{impact:.4f}%' if impact is not None else 'unknown'}")
    print(f" Max slippage:         {f'{slip_bps:.0f} bps' if slip_bps is not None else 'unknown'}")
    if in_usd is not None and out_usd is not None:
        print(f" USD in / out:         ${in_usd:,.2f} -> ${out_usd:,.2f}")
        print(f" Total value loss:     {(in_usd - out_usd) / in_usd * 100:.2f}%")
    print(f" Spent today:          ${spent_today:,.2f} of ${a.max_daily_usd:,.2f}")
    print("=====================================\n")

    unknown = [n for n, v in (("router fee", fee_bps), ("price impact", impact),
                              ("slippage threshold", slip_bps)) if v is None]
    if unknown:
        if not a.allow_unknown_value:
            sys.exit(f"Refusing: Jupiter did not provide {', '.join(unknown)}, so the matching "
                     "checks cannot run. Use --allow-unknown-value only if you accept that risk.")
        print(f"WARNING: skipping checks for {', '.join(unknown)} (--allow-unknown-value).")

    if in_usd > a.max_usd:
        sys.exit(f"Refusing: trade is ~${in_usd:,.2f}, above --max-usd ${a.max_usd:,.2f}.")
    projected_spend = spent_today + in_usd
    if not math.isfinite(projected_spend):
        sys.exit("Refusing: projected daily spend is not finite.")
    if projected_spend > a.max_daily_usd:
        sys.exit(f"Refusing: this would bring today's spend to ${projected_spend:,.2f}, "
                 f"above --max-daily-usd ${a.max_daily_usd:,.2f}.")
    loss = (in_usd - out_usd) / in_usd * 100
    if loss > a.max_loss_pct:
        sys.exit(f"Refusing: this swap loses ~{loss:.2f}% to fees and price impact "
                 f"(limit {a.max_loss_pct}%).")
    if fee_bps is not None and fee_bps > a.max_fee_bps:
        sys.exit(f"Refusing: router fee {fee_bps:.0f} bps is above --max-fee-bps {a.max_fee_bps}.")
    if impact is not None and impact > a.max_impact_pct:
        sys.exit(f"Refusing: price impact {impact:.4f}% is above --max-impact-pct {a.max_impact_pct}%.")
    if slip_bps is not None and slip_bps > a.max_slippage_bps:
        sys.exit(f"Refusing: allowed slippage {slip_bps:.0f} bps is above --max-slippage-bps "
                 f"{a.max_slippage_bps}.")
    return in_usd


# ---------- selftest ----------
def selftest():
    """Offline checks: signing slot, impact parsing, order validation, and fail-closed limits."""
    from types import SimpleNamespace
    from tempfile import TemporaryDirectory
    from solders.hash import Hash
    assert LIVE_EXECUTION_ENABLED is False, "Live execution must remain explicitly disabled."
    from solders.instruction import AccountMeta, Instruction
    from solders.message import MessageV0

    ours, other = Keypair(), Keypair()
    for payer, second in ((ours, other), (other, ours)):  # we are slot 0, then slot 1
        ix = Instruction(Pubkey.default(), b"\x00",
                         [AccountMeta(payer.pubkey(), True, True), AccountMeta(second.pubkey(), True, False)])
        msg = MessageV0.try_compile(payer.pubkey(), [ix], [], Hash.default())
        n = msg.header.num_required_signatures
        unsigned = VersionedTransaction.populate(msg, [Signature.default()] * n)
        signed, idx = sign_tx(base64.b64encode(bytes(unsigned)).decode(), ours)
        payload = to_bytes_versioned(signed.message)
        assert signed.signatures[idx].verify(ours.pubkey(), payload), "bad signature"
        assert all(signed.signatures[i] == Signature.default() for i in range(n) if i != idx), \
            "touched a slot that isn't ours"

    assert read_impact_pct({"priceImpact": "-1.5"}) == 1.5
    assert abs(read_impact_pct({"priceImpactPct": "0.015"}) - 1.5) < 1e-9
    def refuses(fn):
        try:
            fn()
        except SystemExit:
            return True
        return False

    assert refuses(lambda: read_impact_pct({"priceImpact": "abc"}))

    me = str(ours.pubkey())
    good = {"inputMint": "A", "outputMint": "B", "inAmount": "100", "outAmount": "50", "taker": me}
    validate_order(good, "A", "B", 100, me)
    assert refuses(lambda: validate_order({**good, "outputMint": "X"}, "A", "B", 100, me))
    assert refuses(lambda: validate_order({**good, "taker": str(other.pubkey())}, "A", "B", 100, me))
    assert refuses(lambda: validate_order({**good, "inAmount": "101"}, "A", "B", 100, me))

    a = SimpleNamespace(max_usd=25, max_daily_usd=100, max_loss_pct=5, max_fee_bps=50,
                        max_impact_pct=3, max_slippage_bps=100, allow_unknown_value=False)
    o = {"inUsdValue": "10", "outUsdValue": 9.9, "feeBps": 10, "priceImpact": -0.2,
         "outAmount": "1000", "otherAmountThreshold": "995"}
    check_value(o, a, 6, 0)
    assert refuses(lambda: check_value({**o, "priceImpact": -4}, a, 6, 0))      # negative impact caught
    assert refuses(lambda: check_value({**o, "feeBps": None}, a, 6, 0))         # missing fee refused
    assert refuses(lambda: check_value({k: v for k, v in o.items() if k != "priceImpact"}, a, 6, 0))
    assert refuses(lambda: check_value({**o, "otherAmountThreshold": "900"}, a, 6, 0))  # 10% slippage
    assert refuses(lambda: check_value({k: v for k, v in o.items() if k != "inUsdValue"}, a, 6, 0))
    assert refuses(lambda: check_value({k: v for k, v in o.items() if k != "outUsdValue"}, a, 6, 0))
    a.allow_unknown_value = True
    assert refuses(lambda: check_value({k: v for k, v in o.items() if k != "inUsdValue"}, a, 6, 0))
    assert refuses(lambda: check_value({k: v for k, v in o.items() if k != "outUsdValue"}, a, 6, 0))
    for field in ("inUsdValue", "outUsdValue"):
        for bad in ("bad", "NaN", "Infinity", "-Infinity", "0", "-1"):
            assert refuses(lambda field=field, bad=bad: check_value({**o, field: bad}, a, 6, 0)), \
                f"accepted invalid {field}={bad!r}"
    assert refuses(lambda: check_value({**o, "inUsdValue": 26}, a, 6, 0))  # max trade size
    assert refuses(lambda: check_value(o, a, 6, 91))                     # daily cap
    assert not refuses(lambda: check_value(o, a, 6, 90))                 # exactly at cap
    for field in ("max_usd", "max_daily_usd"):
        for bad in (None, "bad", float("nan"), float("inf"), -1):
            invalid_limits = SimpleNamespace(**vars(a))
            setattr(invalid_limits, field, bad)
            assert refuses(lambda invalid_limits=invalid_limits: check_value(o, invalid_limits, 6, 0))

    # --allow-unknown-value applies only to absent ancillary metadata, not USD inputs or bad data.
    ancillary_missing = {k: v for k, v in o.items()
                         if k not in ("feeBps", "priceImpact", "otherAmountThreshold")}
    assert refuses(lambda: check_value(ancillary_missing,
                                      SimpleNamespace(**{**vars(a), "allow_unknown_value": False}), 6, 0))
    assert not refuses(lambda: check_value(ancillary_missing, a, 6, 0))
    for field, bad in (("feeBps", "NaN"), ("priceImpact", "Infinity"),
                       ("otherAmountThreshold", "oops")):
        assert refuses(lambda field=field, bad=bad: check_value({**o, field: bad}, a, 6, 0))
    for invalid_spend in (None, "bad", float("nan"), float("inf"), -1):
        assert refuses(lambda invalid_spend=invalid_spend: check_value(o, a, 6, invalid_spend))

    global TRADE_LOG
    original_log = TRADE_LOG
    try:
        with TemporaryDirectory() as tmp:
            TRADE_LOG = Path(tmp) / "trades.jsonl"
            today_entry = {"time": now_iso(), "status": "landed", "in_usd": 10}
            TRADE_LOG.write_text(json.dumps(today_entry) + "\n")
            assert spent_today_usd() == 10
            for bad_entry in (
                    {"time": now_iso(), "status": "landed"},
                    {"time": now_iso(), "status": "landed", "in_usd": "bad"},
                    {"time": now_iso(), "status": "landed", "in_usd": "NaN"},
                    {"time": now_iso(), "status": "landed", "in_usd": "Infinity"},
                    {"time": now_iso(), "status": "landed", "in_usd": "-Infinity"},
                    {"time": now_iso(), "status": "landed", "in_usd": 0},
                    {"time": now_iso(), "status": "landed", "in_usd": -1},
                    {"time": "not-a-time", "status": "landed", "in_usd": 10}):
                TRADE_LOG.write_text(json.dumps(bad_entry) + "\n")
                assert refuses(spent_today_usd), f"accepted invalid trade-log entry: {bad_entry}"
            TRADE_LOG.write_text("{broken json}\n")
            assert refuses(spent_today_usd)
    finally:
        TRADE_LOG = original_log

    print("Selftest passed: signing slot, impact parsing, order validation, and fail-closed limits.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="src")
    p.add_argument("--to", dest="dst")
    p.add_argument("--amount", type=Decimal, help="amount of the FROM token")
    p.add_argument("--slippage-bps", type=int, default=None,
                   help="optional override (50 = 0.5%%). Default lets Jupiter pick")
    p.add_argument("--execute", action="store_true", help="actually send the trade")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt (dangerous)")
    p.add_argument("--max-usd", type=float, default=float(os.environ.get("MAX_USD", "25")))
    p.add_argument("--max-daily-usd", type=float, default=float(os.environ.get("MAX_DAILY_USD", "100")),
                   help="max total USD swapped per UTC day (default 100)")
    p.add_argument("--max-loss-pct", type=float, default=5.0)
    p.add_argument("--max-fee-bps", type=float, default=float(os.environ.get("MAX_FEE_BPS", "50")))
    p.add_argument("--max-impact-pct", type=float, default=float(os.environ.get("MAX_IMPACT_PCT", "3.0")))
    p.add_argument("--max-slippage-bps", type=float, default=float(os.environ.get("MAX_SLIPPAGE_BPS", "100")),
                   help="max gap between quoted and minimum output (default 100 = 1%%)")
    p.add_argument("--allow-unknown-value", action="store_true")
    p.add_argument("--ignore-pending", action="store_true",
                   help="proceed even if an earlier trade's outcome is unclear (after you checked)")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if a.execute and not LIVE_EXECUTION_ENABLED:
        sys.exit("Refusing: live execution is disabled in this build. Paper trading remains enabled by default.")
    validate_limits(a)
    if not (a.src and a.dst and a.amount):
        p.error("--from, --to and --amount are required")
    if a.amount <= 0:
        sys.exit("--amount must be positive.")

    src_sym, in_mint, in_dec = resolve_token(a.src)
    dst_sym, out_mint, out_dec = resolve_token(a.dst)
    if in_mint == out_mint:
        sys.exit("FROM and TO must differ.")
    amount_raw = int(a.amount * 10**in_dec)
    if amount_raw <= 0:
        sys.exit("Amount is too small for this token's decimals.")

    def show(o):
        out_h = int(o["outAmount"]) / 10**out_dec
        print(f"Quote: {a.amount} {src_sym} -> ~{out_h:.6f} {dst_sym} (router: {o.get('router')}, mode: {o.get('mode')})")
        print(f"Decimals: {src_sym}={in_dec}, {dst_sym}={out_dec}  (raw input amount {amount_raw})")
        if o.get("otherAmountThreshold"):
            print(f"Minimum after slippage: {int(o['otherAmountThreshold']) / 10**out_dec:.6f} {dst_sym}")
        imp = read_impact_pct(o)
        print(f"Price impact: {f'{imp:.4f}%' if imp is not None else 'unknown'} | fee: {o.get('feeBps')} bps")
        iu, ou = to_float(o.get("inUsdValue")), to_float(o.get("outUsdValue"))
        if iu is not None:
            print(f"Value: ~${iu:,.2f} in -> ~${(ou or 0):,.2f} out")

    order = get_order(in_mint, out_mint, amount_raw, None, a.slippage_bps)
    if not a.execute:
        show(order)
        print("\nDry run only. Add --execute to send this trade.")
        return

    if (TRADE_LOG.parent / "STOP").exists():
        sys.exit("Refusing: emergency stop file data/STOP exists. Delete it to allow trades again.")
    check_pending(a.ignore_pending)
    key = os.environ.get("BOT_PRIVATE_KEY")
    if not key:
        sys.exit("Set BOT_PRIVATE_KEY (use a dedicated bot wallet).")
    try:
        keypair = Keypair.from_base58_string(key)
    except Exception:
        sys.exit("BOT_PRIVATE_KEY is not a valid base58 keypair.")  # never echo the key
    del key
    me = str(keypair.pubkey())
    print(f"\nWallet: {me}")

    order = get_order(in_mint, out_mint, amount_raw, me, a.slippage_bps)
    if not order.get("transaction"):
        sys.exit("Cannot swap: " + describe_order_error(order))
    validate_order(order, in_mint, out_mint, amount_raw, me)
    show(order)
    in_usd = check_value(order, a, out_dec, spent_today_usd())

    if a.yes:
        print("WARNING: --yes set, sending WITHOUT confirmation. Limits above still apply.",
              file=sys.stderr)
    elif input(f"Swap {a.amount} {src_sym} -> {dst_sym} (decimals {in_dec}->{out_dec}). "
               "Type YES to send: ").strip() != "YES":
        sys.exit("Cancelled.")

    # No local instruction allowlist/inspection or transaction simulation exists here.
    # See live_warning.txt; do not use --execute until those checks are implemented and tested.
    signed, idx = sign_tx(order["transaction"], keypair)  # quotes can become stale after signing
    our_sig = str(signed.signatures[idx])
    entry = {"id": f"{int(time.time())}-{our_sig[:8]}", "time": now_iso(), "status": "pending",
             "pair": f"{src_sym}->{dst_sym}", "in_mint": in_mint, "out_mint": out_mint,
             "amount_raw": amount_raw, "in_usd": in_usd, "request_id": order.get("requestId"),
             "our_signature": our_sig, "we_are_fee_payer": idx == 0}
    write_log(read_log() + [entry])

    result, err = execute_order(signed, order["requestId"])
    sig = (result or {}).get("signature") or (our_sig if idx == 0 else None)
    if sig:
        print(f"https://solscan.io/tx/{sig}")
    if result and result.get("status") == "Success":
        got = int(result.get("totalOutputAmount", 0)) / 10**out_dec
        spent = int(result.get("totalInputAmount", 0)) / 10**in_dec
        log_update(entry["id"], status="landed", signature=sig, resolved_at=now_iso())
        print(f"Success: spent {spent} {src_sym}, received {got} {dst_sym}")
        return
    # Outcome unclear or failed: ask the chain before anyone retries.
    st = chain_status(our_sig) if idx == 0 else "unknown"
    if result and st != "landed":
        code = result.get("code")
        err = f"Swap failed (code {code}): {EXECUTE_ERRORS.get(code, result.get('error', 'unknown'))}"
    final = st if st in ("landed", "failed") else ("failed" if result and st == "not_found" else "unclear")
    log_update(entry["id"], status=final, signature=sig, error=err, resolved_at=now_iso())
    if final == "landed":
        print("Execute reported a problem, but the transaction LANDED on-chain. Do not retry.")
        return
    if final == "unclear":
        sys.exit(f"{err}\nOutcome UNCLEAR. Do not retry until you check the link/wallet. "
                 "The next --execute will refuse until this is resolved.")
    sys.exit(err)


if __name__ == "__main__":
    main()
