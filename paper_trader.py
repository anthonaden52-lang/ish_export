"""Paper trader: scan -> score -> safety -> risk check -> Jupiter quote check -> simulated trade -> log.

NEVER sends a transaction and never reads a private key (it does not import swap_bot).
Works without a trained model: scores are then labelled ANALYTICAL and no confidence is shown.
"""
import csv
import math
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from hotkeys import KeyReader, sleep_with_keys, snapshot
from indicators import add_indicators, window_history, READY
from model_store import MODEL_PATH, load_model
from quote_check import verify_round_trip
from risk import PaperBook, STOP_FILE
from safety import check_token
from score import decide, format_card
from tracking import load_tracked, scan_and_track
from process_lock import ProcessLock

HISTORY = ROOT / "data" / "paper_history.csv"        # persisted so a restart keeps indicator history
PREDICTIONS = ROOT / "data" / "predictions.csv"      # every BUY/WATCH call, traded or not (for evaluate.py)
SAFETY_PER_SCAN = int(os.getenv("SAFETY_MAX_PER_SCAN", "5"))
QUOTE_CHECK = os.getenv("QUOTE_CHECK", "1") == "1"
QUOTE_SIZE_SOL = float(os.getenv("QUOTE_SIZE_SOL", "0.1"))
ALLOW_ANALYTICAL_ENTRIES = os.getenv("PAPER_ENTRIES_ANALYTICAL", "1") == "1"
PRED_LOG_MINUTES = float(os.getenv("PRED_LOG_MINUTES", "10"))
if SAFETY_PER_SCAN < 0:
    raise ValueError("SAFETY_MAX_PER_SCAN must be non-negative.")
if not math.isfinite(QUOTE_SIZE_SOL) or QUOTE_SIZE_SOL <= 0:
    raise ValueError("QUOTE_SIZE_SOL must be finite and positive.")
if not math.isfinite(PRED_LOG_MINUTES) or PRED_LOG_MINUTES <= 0:
    raise ValueError("PRED_LOG_MINUTES must be finite and positive.")
PRED_FIELDS = ["timestamp", "token", "symbol", "verdict", "score", "score_source", "model_prob", "risk",
               "price_usd", "traded", "note"]

bundle = None
history = pd.DataFrame()
tracked = {}
book = None
last_logged = {}


def log_prediction(r, d, traded, note=""):
    now = pd.Timestamp.now(tz="UTC")
    prev = last_logged.get(r["token_address"])
    if prev is not None and (now - prev).total_seconds() < PRED_LOG_MINUTES * 60 and not traded:
        return
    last_logged[r["token_address"]] = now
    new = not PREDICTIONS.exists()
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": now.isoformat(), "token": r["token_address"], "symbol": r.get("symbol", ""),
                    "verdict": d["verdict"], "score": d["score"], "score_source": d["source"],
                    "model_prob": "" if d["prob"] is None else round(d["prob"], 4), "risk": d["risk"],
                    "price_usd": r["price_usd"], "traded": int(traded), "note": note})
        f.flush()
        os.fsync(f.fileno())


def step():
    global history
    rows = scan_and_track(tracked)
    if not rows:
        return
    new = pd.DataFrame(rows)
    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True)
    history = window_history(pd.concat([history, new], ignore_index=True))
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        history.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(HISTORY)

    x = add_indicators(history.copy())
    latest = x.groupby("token_address").tail(1)

    # 1) manage open positions first (stops, targets, vanished pairs)
    for t in book.update({r["token_address"]: float(r["price_usd"]) for _, r in latest.iterrows()}):
        print(f"CLOSED {t['symbol']}: {t['net_return']:+.1%} ({t['exit_reason']}, held {t['held_minutes']:.0f} min, "
              f"score {t['score']} {t['score_source']})")

    # 2) score everything that has enough history
    ready = latest[latest["price_usd"] > 0].dropna(subset=READY)
    if ready.empty:
        print("Waiting for more history...")
        print("BOOK:", book.summary())
        return
    budget = {"n": SAFETY_PER_SCAN}

    def safety_fn(r):
        from safety import cached_onchain
        if cached_onchain(r["token_address"]) is None:
            if budget["n"] <= 0:
                return check_token(r, fetch=False)
            budget["n"] -= 1
        return check_token(r, fetch=True)

    decisions = []
    for _, r in ready.iterrows():
        decisions.append((r, decide(r, bundle, safety_fn)))
    decisions.sort(key=lambda rd: ({"BUY CANDIDATE": 0, "WATCH": 1, "AVOID": 2}[rd[1]["verdict"]], -rd[1]["score"]))

    # 3) act on BUY CANDIDATEs (paper only)
    for r, d in decisions:
        if d["verdict"] != "BUY CANDIDATE":
            if d["verdict"] == "WATCH":
                log_prediction(r, d, False)
            continue
        if d["source"] == "ANALYTICAL" and not ALLOW_ANALYTICAL_ENTRIES:
            log_prediction(r, d, False, "analytical entries disabled")
            continue
        block = book.entry_block_reason(r["token_address"])
        if block:
            log_prediction(r, d, False, block)
            print(f"  skip {r['symbol']}: {block}")
            continue
        if QUOTE_CHECK:
            q = verify_round_trip(r["token_address"], (d["safety"]["metrics"] or {}).get("decimals") or 6,
                                  QUOTE_SIZE_SOL, float(r["price_usd"]))
            if not q["ok"]:
                note = "quote check: " + "; ".join(q["reasons"])
                log_prediction(r, d, False, note)
                print(f"  skip {r['symbol']}: {note}")
                continue
        pos, why = book.enter(r, d)
        log_prediction(r, d, pos is not None, why or "")
        if pos:
            print(f"PAPER BUY {r['symbol']} ${pos['stake']:.2f} @ {pos['entry_price']:.8g} "
                  f"(score {d['score']} {d['source']}, risk {d['risk']})")

    print("\nTOP CANDIDATES")
    for r, d in decisions[:5]:
        print(format_card(r, d), "\n")
    print("BOOK:", book.summary())


def main():
    global bundle, history, tracked, book
    try:
        lock = ProcessLock(ROOT / "data" / "paper_trader.lock")
        lock.acquire()
    except RuntimeError as exc:
        raise SystemExit(f"Paper trader not started: {exc}") from exc
    try:
        if MODEL_PATH.exists():
            bundle = load_model()
            print(f"Loaded trained model (features: {len(bundle['features'])}, "
                  f"calibration: {bundle.get('calibration_method', 'uncalibrated')}).")
        else:
            print("No trained model found: scores will be ANALYTICAL (rule-based, unvalidated). "
                  "Collect data with collector.py, then run train.py.")
        if HISTORY.exists():
            history = pd.read_csv(HISTORY)
            history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
            history = history.dropna(subset=["timestamp", "token_address", "price_usd"])
        tracked = load_tracked()
        book = PaperBook()
        print("ACORN 1.0 | PAPER TRADING - no transactions will be sent.  S = save snapshot, Q = quit.")
        print(f"To halt new entries immediately: create the file {STOP_FILE}")
        print("Signals need about 15 minutes of history per token, so expect a quiet start.")
        with KeyReader() as keys:
            while True:
                try:
                    step()
                    wait = 60
                except KeyboardInterrupt:
                    break
                except Exception as exc:
                    print("Paper trader error:", exc)
                    wait = 15
                if sleep_with_keys(wait, keys):
                    break
        snapshot()
        print("Stopped.")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
