"""Jupiter quote verification (READ-ONLY, quote endpoint only - no wallet, no signing).

Asks Jupiter for a buy quote (SOL -> token) and then a sell quote for exactly what the buy would
return (token -> SOL). Units-free round-trip loss is the main number, because the price-impact field
units are unconfirmed (see CHANGES.md #1).

This is a heuristic. It catches "no sell route" and "selling loses a huge amount", which covers many
honeypots. It cannot prove a token is sellable: transfer restrictions can still fail at execution time.
"""
import os
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = os.getenv("JUPITER_BASE", "https://api.jup.ag/swap/v2").rstrip("/")
if urlparse(BASE).scheme != "https" or urlparse(BASE).hostname != "api.jup.ag":
    raise SystemExit("JUPITER_BASE must be https://api.jup.ag/...")
API_KEY = os.getenv("JUPITER_API_KEY", "")
SOL = "So11111111111111111111111111111111111111112"
MAX_ROUND_TRIP_LOSS_PCT = float(os.getenv("MAX_ROUND_TRIP_LOSS_PCT", "8"))
MAX_PRICE_DEVIATION_PCT = float(os.getenv("MAX_PRICE_DEVIATION_PCT", "15"))


def _order(in_mint, out_mint, amount):
    r = requests.get(f"{BASE}/order",
                     params={"inputMint": in_mint, "outputMint": out_mint, "amount": str(int(amount))},
                     headers={"x-api-key": API_KEY} if API_KEY else {}, timeout=30, allow_redirects=False)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
    return r.json()


def verify_round_trip(mint, token_decimals, size_sol, dex_price_usd=None):
    """Returns {"ok": bool, "reasons": [...], "round_trip_loss_pct": float|None}. Fails closed."""
    try:
        lamports = int(size_sol * 1e9)
        buy = _order(SOL, mint, lamports)
        got = int(buy.get("outAmount") or 0)
        if got <= 0:
            return {"ok": False, "reasons": ["no buy route / zero output"], "round_trip_loss_pct": None}
        sell = _order(mint, SOL, got)
        back = int(sell.get("outAmount") or 0)
        if back <= 0:
            return {"ok": False, "reasons": ["NO SELL ROUTE (possible honeypot)"], "round_trip_loss_pct": None}
    except Exception as e:  # network, API key, rate limit, bad JSON: all fail closed
        return {"ok": False, "reasons": [f"quote check failed: {e}"], "round_trip_loss_pct": None}

    loss = (1 - back / lamports) * 100
    reasons = []
    if loss > MAX_ROUND_TRIP_LOSS_PCT:
        reasons.append(f"round trip loses {loss:.1f}% in quotes (limit {MAX_ROUND_TRIP_LOSS_PCT:g}%)")

    in_usd = buy.get("inUsdValue")
    if dex_price_usd and in_usd:
        try:
            quote_price = float(in_usd) / (got / 10 ** token_decimals)
            dev = abs(quote_price / dex_price_usd - 1) * 100
            if dev > MAX_PRICE_DEVIATION_PCT:
                reasons.append(f"Jupiter price differs from DexScreener by {dev:.0f}%")
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return {"ok": not reasons, "reasons": reasons, "round_trip_loss_pct": round(loss, 2)}
