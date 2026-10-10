import csv
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from data_quality import DataQualityError, MARKET_FIELDS, clean_market_frame, normalize_market_row
from hotkeys import KeyReader, sleep_with_keys, snapshot
from process_lock import ProcessLock
from tracking import load_tracked, scan_and_track

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "market.csv"
HEALTH = ROOT / "data" / "collector_health.json"
FIELDS = MARKET_FIELDS

interval = int(os.getenv("SCAN_INTERVAL", "60"))
if interval < 1:
    raise ValueError("SCAN_INTERVAL must be at least 1 second.")
MAX_MARKET_MB = float(os.getenv("MAX_MARKET_MB", "2048"))
if not math.isfinite(MAX_MARKET_MB) or MAX_MARKET_MB <= 0:
    raise ValueError("MAX_MARKET_MB must be finite and positive.")
MAX_MARKET_BYTES = int(MAX_MARKET_MB * 1024 * 1024)


def _is_dexscreener_row(row):
    value = row.get("url")
    if not isinstance(value, str):
        return False
    host = (urlparse(value.strip()).hostname or "").lower()
    return host == "dexscreener.com" or host.endswith(".dexscreener.com")


def _repair_partial_csv_tail():
    """Preserve and remove an incomplete final append left by a forced shutdown."""
    if not OUT.exists() or OUT.stat().st_size == 0:
        return
    size = OUT.stat().st_size
    with OUT.open("r+b") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        pos = size
        boundary = None
        while pos > 0 and boundary is None:
            start = max(0, pos - 64 * 1024)
            f.seek(start)
            block = f.read(pos - start)
            index = block.rfind(b"\n")
            if index >= 0:
                boundary = start + index + 1
            pos = start
        if boundary is None:
            raise DataQualityError("market.csv has no complete header/row boundary; archive it for review.")
        f.seek(boundary)
        partial = f.read()
        archive = OUT.with_name(f"market_partial_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.bin")
        archive.write_bytes(partial)
        f.truncate(boundary)
        f.flush()
        os.fsync(f.fileno())
    print(f"Preserved and removed an incomplete final CSV append: {archive}")


def _rotate_if_old_schema():
    """Keep market.csv schema- and source-consistent with the DexScreener collector."""
    if not OUT.exists():
        return

    reason = None
    dex_rows = None
    with OUT.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            reason = "old schema"
            old_rows = list(reader)
            if old_rows and {"timestamp", "token_address", "price_usd"}.issubset(reader.fieldnames or []):
                clean, report = clean_market_frame(pd.DataFrame(old_rows))
                dex_rows = []
                for row in clean.to_dict("records"):
                    if _is_dexscreener_row(row) or str(row.get("source", "")).lower() == "dexscreener":
                        row["source"] = "dexscreener"
                        row["discovery_sources"] = "legacy_import"
                        dex_rows.append(normalize_market_row(row, source_default="dexscreener"))
        else:
            rows = list(reader)
            dex_rows = []
            for row in rows:
                source = str(row.get("source", "")).strip().lower()
                if source == "dexscreener" or source in ("", "legacy") and _is_dexscreener_row(row):
                    row["source"] = "dexscreener"
                    if source != "dexscreener" and not row.get("discovery_sources"):
                        row["discovery_sources"] = "legacy_import"
                    try:
                        dex_rows.append(normalize_market_row(row, source_default="dexscreener"))
                    except DataQualityError:
                        continue
            if len(dex_rows) != len(rows):
                reason = "mixed-source data"
            elif any(str(row.get("source", "")).strip().lower() != "dexscreener" for row in rows):
                reason = "legacy source migration"

    if reason:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dest = OUT.with_name(f"market_archived_{stamp}.csv")
        OUT.rename(dest)
        if dex_rows is not None:
            with OUT.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(dex_rows)
            print(f"{reason} archived to {dest}; kept {len(dex_rows)} DexScreener rows in {OUT}.")
        else:
            print(f"Old schema found; moved to {dest}. Starting a fresh DexScreener market.csv.")


def write_rows(rows):
    _repair_partial_csv_tail()
    exists = OUT.exists()
    normalized = [normalize_market_row(row, source_default="dexscreener") for row in rows]
    if not normalized:
        return
    if any(row["source"] != "dexscreener" for row in normalized):
        raise DataQualityError("collector refuses rows from a non-DexScreener source")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction="raise")
    if not exists:
        writer.writeheader()
    writer.writerows(normalized)
    payload = buffer.getvalue()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    current_size = OUT.stat().st_size if exists else 0
    if current_size + len(payload.encode("utf-8")) > MAX_MARKET_BYTES:
        raise RuntimeError(f"{OUT} reached MAX_MARKET_MB; increase the limit after reviewing disk space.")
    with OUT.open("a", newline="", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def write_health(status, **details):
    HEALTH.parent.mkdir(parents=True, exist_ok=True)
    record = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **details}
    tmp = HEALTH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(HEALTH)


def main():
    print("Starting collector.  Press S = save snapshot, Q = save and quit.")
    print(f"Writing DexScreener snapshots to {OUT} (tracked tokens remain recorded after filters fail)")
    try:
        lock = ProcessLock(ROOT / "data" / "collector.lock")
        lock.acquire()
    except RuntimeError as exc:
        print(f"Collector not started: {exc}")
        return 2
    try:
        _repair_partial_csv_tail()
        _rotate_if_old_schema()
        tracked = load_tracked()
        write_health("starting", interval_seconds=interval)
        with KeyReader() as keys:
            while True:
                try:
                    rows = scan_and_track(tracked)
                    write_rows(rows)
                    dead = sum(1 for r in rows if r["price_usd"] <= 0)
                    write_health("ok", rows=len(rows), tracked_tokens=len(tracked), dead_rows=dead,
                                 market_bytes=OUT.stat().st_size if OUT.exists() else 0,
                                 interval_seconds=interval)
                    print(f"Collected {len(rows)} rows ({len(tracked)} tracked, {dead} dead/vanished).")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    write_health("error", error=str(e)[:300], interval_seconds=interval,
                                 market_bytes=OUT.stat().st_size if OUT.exists() else 0)
                    print("Collector error:", e)
                if sleep_with_keys(interval, keys):
                    break
        snapshot()
        write_health("stopped", interval_seconds=interval)
        print("Stopped.")
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
