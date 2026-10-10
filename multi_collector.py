#!/usr/bin/env python3
"""Fixed-list collector using Birdeye (optional) + GeckoTerminal.

Writes to data/market_multi.csv so its alternate-source, sparse snapshots do not get mixed into
data/market.csv, which is the DexScreener feed used for training and paper trading.

    export BIRDEYE_API_KEY="..."     # optional; without it only GeckoTerminal is used
    python multi_collector.py                # one scan
    python multi_collector.py --loop 60      # a scan every 60 seconds until Ctrl+C

Do NOT paste API keys into this file.

Limits you should know about: this only records the tokens listed below and only fills price, 24h
volume, market cap, FDV and (if GeckoTerminal returns it) liquidity. Volume 5m/1h and buy/sell counts
stay empty, so most model features are NaN. collector.py (DexScreener) records all of them.
Established coins are survivors, so a model trained only on them is biased.
"""
import argparse
import csv
import io
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from data_quality import MARKET_FIELDS, normalize_market_row
from process_lock import ProcessLock

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
BIRDEYE_KEY = os.environ.get("BIRDEYE_API_KEY", "")
GECKO_BASE = "https://api.geckoterminal.com/api/v2"
BIRDEYE_BASE = "https://public-api.birdeye.so"
_LAST_REQUEST = {"gecko": 0.0, "birdeye": 0.0}
_MIN_INTERVAL = {"gecko": 7.0, "birdeye": 1.1}
TOKENS = {
    "PUMPOWEEN": "5GefefPX1mDs6ZJB1apYmz6fCTCiNJpHturZ9bvFpump",
    "TRUMP": "6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN",
    "PENGU": "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "FARTCOIN": "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump",
    "MELANIA": "FUAfBo2jgks6gB4Z4LfZkqSZgzNucisEHqnNebaRxM1P",
    "PNUT": "2qEHjDLDLbuBgRYvsxhc5D6uDWAivNFZGan56P1tpump",
    "POPCAT": "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",
    "MOODENG": "ED5nyyWEzpPPiWimP8vYm7sD7TD3LAt3Q3gRTWHzPJBY",
    "GOAT": "CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuypump",
    "USELESS": "Dz9mQ9NzkBcCsuGPFJ3r1bS4wgqKMHBPiVuniW8Mbonk",
    "PONKE": "5z3EqYQo9HiCEs3R84RCDMu2n7anpDMxRhdK8PSWmrRC",
}
OUT = ROOT / "data" / "market_multi.csv"
FIELDS = MARKET_FIELDS


def _num(x):
    if isinstance(x, bool):
        return ""
    try:
        v = float(x)
        return v if math.isfinite(v) and v >= 0 else ""
    except (OverflowError, TypeError, ValueError):
        return ""


def _get_json(url, provider, *, params=None, headers=None):
    allowed = {"gecko": "api.geckoterminal.com", "birdeye": "public-api.birdeye.so"}
    parsed = urlparse(url)
    if provider not in allowed or parsed.scheme != "https" or parsed.hostname != allowed[provider]:
        raise ValueError("Refusing an unapproved market-data URL.")
    for attempt in range(3):
        wait = _MIN_INTERVAL[provider] - (time.monotonic() - _LAST_REQUEST[provider])
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST[provider] = time.monotonic()
        response = requests.get(url, params=params, headers=headers, timeout=12)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 2:
                response.raise_for_status()
            try:
                retry_after = float(response.headers.get("Retry-After", "0"))
            except (TypeError, ValueError):
                retry_after = 0
            delay = max(_MIN_INTERVAL[provider], min(retry_after, 60))
            time.sleep(delay)
            continue
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"{provider} response must be a JSON object.")
        return data
    return None


def birdeye_price(addr):
    if not BIRDEYE_KEY:
        return None
    try:
        data = _get_json(f"{BIRDEYE_BASE}/defi/price", "birdeye", params={"address": addr},
                         headers={"X-API-KEY": BIRDEYE_KEY, "x-chain": "solana"})
        return ((data.get("data") or {}).get("value")
                if isinstance(data.get("data"), dict) else None)
    except (requests.RequestException, ValueError, TypeError):
        return None


def gecko_token(addr):
    try:
        data = _get_json(f"{GECKO_BASE}/networks/solana/tokens/{addr}", "gecko",
                         headers={"Accept": "application/json;version=20230203"})
        token = data.get("data")
        a = token.get("attributes") if isinstance(token, dict) else None
        if not isinstance(a, dict):
            return {}
        vol = a.get("volume_usd")
        return {"price_usd": a.get("price_usd"), "volume_24h": vol.get("h24") if isinstance(vol, dict) else vol,
                "fdv": a.get("fdv_usd"), "market_cap": a.get("market_cap_usd"),
                "liquidity": a.get("total_reserve_in_usd")}
    except (requests.RequestException, ValueError, TypeError):
        pass
    return {}


def _rotate_if_old_schema():
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != FIELDS:
            dest = OUT.with_name(f"market_oldschema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            OUT.rename(dest)
            print(f"Old data format found; moved to {dest}.")


def collect_once():
    rows = []
    for sym, addr in TOKENS.items():
        price = birdeye_price(addr)
        birdeye_value = _num(price)
        price_source = "birdeye" if birdeye_value != "" and birdeye_value > 0 else ""
        g = gecko_token(addr)
        gecko_price = _num(g.get("price_usd"))
        if not price_source and gecko_price != "" and gecko_price > 0:
            price_source = "geckoterminal"
        price = birdeye_value if price_source == "birdeye" else gecko_price
        rows.append({"timestamp": datetime.now(timezone.utc).isoformat(), "token_address": addr, "symbol": sym, "price_usd": price,
                     "liquidity_usd": _num(g.get("liquidity")), "volume_24h": _num(g.get("volume_24h")),
                     "dex": "multi", "url": f"https://birdeye.so/token/{addr}",
                     "market_cap": _num(g.get("market_cap")), "fdv": _num(g.get("fdv")),
                     "source": "multi", "discovery_sources":
                         f"fixed_token_list;price:{price_source or 'missing'};metrics:geckoterminal"})
        print(f"{sym}: {price if price != '' else 'no price'}")
    # A token with no price is skipped, not written as an empty row (an empty price breaks the labels).
    rows = [r for r in rows if r["price_usd"] != "" and r["price_usd"] > 0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_old_schema()
    rows = [normalize_market_row(row, source_default="multi") for row in rows]
    exists = OUT.exists()
    buffer = io.StringIO(newline="")
    w = csv.DictWriter(buffer, fieldnames=FIELDS, restval="", extrasaction="raise")
    if not exists:
        w.writeheader()
    w.writerows(rows)
    with OUT.open("a", newline="", encoding="utf-8") as f:
        f.write(buffer.getvalue())
        f.flush()
        os.fsync(f.fileno())
    print(f"Appended {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between scans; 0 = one scan")
    a = ap.parse_args()
    if a.loop < 0:
        ap.error("--loop must be 0 (one scan) or a positive interval")
    try:
        with ProcessLock(ROOT / "data" / "multi_collector.lock"):
            while True:
                collect_once()
                if not a.loop:
                    break
                time.sleep(a.loop)
    except KeyboardInterrupt:
        print("Stopped.")
    except RuntimeError as exc:
        sys.exit(f"Collector not started: {exc}")
