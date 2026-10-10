"""Hotkeys for long-running scripts: press S to save a snapshot, Q to quit.
These keys never trade; nothing here imports swap_bot or touches a wallet.

Click the terminal window first so it has keyboard focus.
Works on Windows (msvcrt) and Mac/Linux (termios).
"""
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from process_lock import ProcessLock

ROOT = Path(__file__).resolve().parent
BACKUP_DIR = ROOT / "backups"
SAVE_FILES = [
    ROOT / "data" / "market.csv", ROOT / "data" / "market_multi.csv",
    ROOT / "models" / "model.joblib", ROOT / "models" / "model.sha256",
    ROOT / "models" / "model_meta.json", ROOT / "data" / "model_history.jsonl",
    ROOT / "data" / "tracked.json", ROOT / "data" / "paper_state.json",
    ROOT / "data" / "paper_trades.csv", ROOT / "data" / "predictions.csv",
]


def snapshot():
    """Copy the collected data and trained model into backups/<timestamp>/."""
    dest = BACKUP_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved = []
    existing = [f for f in SAVE_FILES if f.exists()]
    if existing:
        dest.mkdir(parents=True, exist_ok=False)
    model_files = {ROOT / "models" / name for name in ("model.joblib", "model.sha256", "model_meta.json")}
    for f in existing:
        if f in model_files:
            continue
        shutil.copy2(f, dest / f.name)
        saved.append(f"{f.name} ({f.stat().st_size / 1024:.0f} KB)")
    model_existing = [f for f in existing if f in model_files]
    try:
        model_lock = ProcessLock(ROOT / "models" / ".model.lock") if model_existing else None
        if model_lock:
            model_lock.acquire()
        for f in model_existing:
            shutil.copy2(f, dest / f.name)
            saved.append(f"{f.name} ({f.stat().st_size / 1024:.0f} KB)")
    except RuntimeError as exc:
        print(f"[save] model copy skipped because the model is being updated: {exc}")
    finally:
        if "model_lock" in locals() and model_lock and model_lock.fd is not None:
            model_lock.release()
    if saved:
        print(f"[saved] {dest}: " + ", ".join(saved))
    else:
        print("[save] nothing to save yet (no data/market.csv or model).")


class KeyReader:
    def __enter__(self):
        self.win = sys.platform.startswith("win")
        self.old = None
        if not self.win and sys.stdin.isatty():
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get(self):
        """Return a pressed key (lowercase) or None. Never blocks."""
        if self.win:
            import msvcrt

            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # arrow/Delete/F-keys send 2 codes
                    msvcrt.getwch()           # swallow the second code
                    return None
                return ch.lower()
            return None
        if self.old is not None:
            import os
            import select

            if select.select([self.fd], [], [], 0)[0]:
                data = os.read(self.fd, 1)
                return data.decode(errors="ignore").lower() if data else None
        return None


def sleep_with_keys(seconds, reader):
    """Sleep, but react to keys. Returns True if the user pressed Q."""
    end = time.time() + seconds
    while time.time() < end:
        key = reader.get()
        if key == "s":
            snapshot()
        elif key == "q":
            return True
        time.sleep(0.1)
    return False
