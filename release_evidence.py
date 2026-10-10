"""Collect the on-disk evidence the Acorn 2.0 release gate is judged against.

Read-only and dependency-light: it reads only the retrain log, the cost-aware
backtest summary, and the saved model manifest, so the dashboard can import it
without loading pandas or scikit-learn. Missing artifacts are reported as gaps
rather than raised, because "no evidence" is itself a gate result.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"

HISTORY_FILE = DATA / "model_history.jsonl"
BACKTEST_FILE = DATA / "backtest_summary.json"
MANIFEST_FILE = MODEL_DIR / "model_meta.json"
HASH_FILE = MODEL_DIR / "model.sha256"


def read_json(path):
    """Return a parsed JSON object, or None when the file is missing or malformed."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def last_training_record(path=HISTORY_FILE):
    """Newest complete record in the append-only retrain log, or None."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    for line in reversed(lines):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except ValueError:
            continue
        if isinstance(record, dict):
            return record
    return None


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _span_days(first, last):
    start, end = _timestamp(first), _timestamp(last)
    if start is None or end is None or end < start:
        return None
    return (end - start).total_seconds() / 86400.0


def dataset_span(record):
    """Span of the labelled rows a fit used, falling back to the source rows.

    Source rows are an upper bound on the eligible span, so the answer states
    which of the two it measured.
    """
    record = record or {}
    labelled = _span_days(record.get("first_labeled"), record.get("last_labeled"))
    if labelled is not None:
        return {"span_days": labelled, "span_source": "labeled_rows"}
    quality = record.get("data_quality") or {}
    source = _span_days(quality.get("first_timestamp"), quality.get("last_timestamp"))
    if source is not None:
        return {"span_days": source, "span_source": "source_rows"}
    return {"span_days": None, "span_source": "unavailable"}


def collect(history_path=HISTORY_FILE, backtest_path=BACKTEST_FILE, manifest_path=MANIFEST_FILE):
    """Gather every artifact the release gate reads, without raising on gaps."""
    record = last_training_record(history_path)
    backtest = read_json(backtest_path)
    manifest = read_json(manifest_path)
    dataset = dataset_span(record)
    dataset["usable_rows"] = (record or {}).get("rows")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training": record,
        "backtest": backtest,
        "manifest": manifest,
        "model_sha256": read_text(HASH_FILE) or (manifest or {}).get("sha256"),
        "dataset": dataset,
        "artifacts": {
            "retrain_log": record is not None,
            "backtest_summary": backtest is not None,
            "model_manifest": manifest is not None,
        },
    }
