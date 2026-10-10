"""Export a small, public-safe Acorn snapshot for the static Pages site.

Only recent DexScreener market fields are exported. Paper-trading history,
wallet state, model files, logs, and environment settings are never read.
"""
import csv
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parent
MARKET_FILE = ROOT / "data" / "market.csv"
STATUS_FILE = ROOT / "docs" / "status.json"
FRESH_FOR_SECONDS = 180
MAX_PUBLIC_PAIRS = 25
REQUIRED_FIELDS = {"timestamp", "token_address", "price_usd", "url"}


def _number(row, name):
    try:
        value = float(row.get(name, ""))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _dexscreener_url(value):
    try:
        parsed = urlsplit(str(value).strip())
        host = (parsed.hostname or "").lower()
        if (parsed.scheme != "https" or parsed.username or parsed.password
                or not (host == "dexscreener.com" or host.endswith(".dexscreener.com"))):
            return None
        if not parsed.path.startswith("/"):
            return None
        return urlunsplit(("https", host, parsed.path, "", ""))
    except (TypeError, ValueError):
        return None


def _empty(now, status="unavailable"):
    return {
        "schema_version": 1,
        "site": "Acorn 1.0",
        "mode": "PAPER",
        "generated_at": now.isoformat(),
        "market_status": status,
        "last_market_update": None,
        "market_age_seconds": None,
        "known_pairs": 0,
        "source": "DexScreener market snapshots",
        "pairs": [],
    }


def build_export(now=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    if not MARKET_FILE.is_file():
        return _empty(now)

    latest_by_token = {}
    try:
        with MARKET_FILE.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
                return _empty(now)

            for row in reader:
                source = str(row.get("source") or "").strip().lower()
                if source and source not in {"dexscreener", "legacy"}:
                    continue
                url = _dexscreener_url(row.get("url"))
                token = str(row.get("token_address") or "").strip()
                stamp = _timestamp(row.get("timestamp"))
                price = _number(row, "price_usd")
                if not url or not token or len(token) > 64 or stamp is None:
                    continue
                if stamp > now + timedelta(seconds=5) or price is None or price <= 0:
                    continue

                item = {
                    "symbol": str(row.get("symbol") or "")[:24].strip(),
                    "token": token,
                    "price_usd": price,
                    "liquidity_usd": _number(row, "liquidity_usd"),
                    "volume_24h_usd": _number(row, "volume_24h"),
                    "change_5m_pct": _number(row, "price_change_5m"),
                    "change_1h_pct": _number(row, "price_change_1h"),
                    "updated_at": stamp.isoformat(),
                    "url": url,
                }
                previous = latest_by_token.get(token)
                if previous is None or stamp > previous[0]:
                    latest_by_token[token] = (stamp, item)
    except (OSError, UnicodeError, csv.Error):
        return _empty(now)

    if not latest_by_token:
        return _empty(now)

    latest_update = max(stamp for stamp, _ in latest_by_token.values())
    age = max(0, int((now - latest_update).total_seconds()))
    status = "fresh" if age <= FRESH_FOR_SECONDS else "stale"
    pairs = []
    if status == "fresh":
        ranked = [item for stamp, item in latest_by_token.values()
                  if (now - stamp).total_seconds() <= FRESH_FOR_SECONDS]
        ranked.sort(key=lambda item: (
            item["liquidity_usd"] is None,
            -(item["liquidity_usd"] or 0),
            -(item["volume_24h_usd"] or 0),
        ))
        pairs = ranked[:MAX_PUBLIC_PAIRS]

    return {
        "schema_version": 1,
        "site": "Acorn 1.0",
        "mode": "PAPER",
        "generated_at": now.isoformat(),
        "market_status": status,
        "last_market_update": latest_update.isoformat(),
        "market_age_seconds": age,
        "known_pairs": len(latest_by_token),
        "source": "DexScreener market snapshots",
        "pairs": pairs,
    }


def main():
    payload = build_export()
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATUS_FILE.with_name(STATUS_FILE.name + ".tmp")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(STATUS_FILE)
    print(f"Wrote public site snapshot: {payload['market_status']} ({len(payload['pairs'])} pairs shown).")


if __name__ == "__main__":
    main()
