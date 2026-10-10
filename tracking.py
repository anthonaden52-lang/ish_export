"""Survivorship-free tracking: once a token is picked up, keep recording it.

A token found by the scanner stays on the watch list for TRACK_MINUTES even
if it later fails the liquidity/volume filters, so crashes and rugs end up in
the data. If its pair disappears or has no price, a price-0 row is written.
"""
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_quality import DataQualityError
from process_lock import ProcessLock
from scanner import dead_row, get_pairs_by_address, normalize, scan

ROOT = Path(__file__).resolve().parent
TRACK_FILE = ROOT / "data" / "tracked.json"
TRACK_LOCK = ROOT / "data" / "tracking.lock"
TRACK_MINUTES = float(os.getenv("TRACK_MINUTES",
                                str(3 * int(os.getenv("HORIZON_MINUTES", "15")) + 60)))
if not math.isfinite(TRACK_MINUTES) or TRACK_MINUTES <= 0:
    raise ValueError("TRACK_MINUTES must be a finite positive number.")


def _validate_tracked(state):
    if not isinstance(state, dict):
        raise ValueError("Tracked token state must be a JSON object; refusing to reset it.")
    for token, entry in state.items():
        if not isinstance(token, str) or not token.strip() or not isinstance(entry, dict):
            raise ValueError("Tracked token state contains an invalid entry; refusing to reset it.")
        if not isinstance(entry.get("pair"), str) or not entry["pair"].strip():
            raise ValueError(f"Tracked token {token} has no pair address; refusing to reset it.")
        for field in ("first_seen", "last_pass"):
            stamp = entry.get(field)
            if field == "last_pass" and stamp is None:
                stamp = entry.get("first_seen")
            if not isinstance(stamp, str):
                raise ValueError(f"Tracked token {token} has invalid {field}; refusing to reset it.")
            try:
                parsed = datetime.fromisoformat(stamp)
            except ValueError:
                raise ValueError(f"Tracked token {token} has invalid {field}; refusing to reset it.") from None
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError(f"Tracked token {token} timestamp lacks a timezone; refusing to reset it.")
        if not isinstance(entry.get("symbol", ""), str):
            raise ValueError(f"Tracked token {token} has an invalid symbol; refusing to reset it.")
        if not isinstance(entry.get("sources", "unknown"), str):
            raise ValueError(f"Tracked token {token} has invalid source metadata; refusing to reset it.")
    return state


def _load_tracked_unlocked():
    if TRACK_FILE.exists():
        try:
            data = json.loads(TRACK_FILE.read_text(encoding="utf-8"))
            _validate_tracked(data)
            for entry in data.values():
                entry.setdefault("last_pass", entry["first_seen"])
                entry.setdefault("sources", "unknown")
            return data
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Cannot safely load tracked token state from {TRACK_FILE}: {exc}") from exc
    return {}


def load_tracked():
    with ProcessLock(TRACK_LOCK):
        return _load_tracked_unlocked()


def _save_tracked_unlocked(t):
    _validate_tracked(t)
    TRACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = TRACK_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(t, allow_nan=False))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(TRACK_FILE)


def save_tracked(t):
    with ProcessLock(TRACK_LOCK):
        _save_tracked_unlocked(t)


def scan_and_track(tracked):
    """Returns normalized rows for new candidates + every still-tracked token."""
    with ProcessLock(TRACK_LOCK):
        persisted = _load_tracked_unlocked() if TRACK_FILE.exists() else dict(tracked)
        tracked.clear()
        tracked.update(persisted)
        now = datetime.now(timezone.utc)
        rows = {}
        for p in scan():
            if not p.get("priceUsd"):
                continue
            try:
                r = normalize(p)
            except (DataQualityError, ValueError):
                continue
            rows[r["token_address"]] = r
            t = tracked.setdefault(r["token_address"], {"first_seen": now.isoformat(),
                                                        "pair": r["pair_address"], "symbol": r["symbol"],
                                                        "sources": r["discovery_sources"]})
            t.update(pair=r["pair_address"], symbol=r["symbol"], sources=r["discovery_sources"])
            t["last_pass"] = now.isoformat()

        # drop tokens whose tracking window is over
        for tok in list(tracked):
            if now - datetime.fromisoformat(tracked[tok].get("last_pass", tracked[tok]["first_seen"])) \
                    > timedelta(minutes=TRACK_MINUTES):
                del tracked[tok]

        need = {tok: t for tok, t in tracked.items() if tok not in rows}
        if need:
            found = {}
            for p in get_pairs_by_address([t["pair"] for t in need.values()]):
                pair_address = p.get("pairAddress")
                if isinstance(pair_address, str) and pair_address:
                    found[pair_address] = p
            for tok, t in need.items():
                p = found.get(t["pair"])
                base = p.get("baseToken") if p else None
                if p and isinstance(base, dict) and base.get("address") == tok:
                    try:
                        rows[tok] = normalize(p)
                    except (DataQualityError, ValueError):
                        continue  # malformed price is missing data, never a fabricated total loss
                elif p is None:
                    rows[tok] = dead_row(tok, t["symbol"], t["pair"], t.get("sources", "unknown"))
        _save_tracked_unlocked(tracked)
        return list(rows.values())
