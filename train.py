"""Leakage-aware model training on the collected market CSV.

All reported model scores are out-of-time. The final test block is untouched by
fit, calibration, and threshold selection. The saved model records its data and
feature versions for reproducibility.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score

from data_quality import DataQualityError, load_market_csv, require_training_source
from indicators import add_indicators, FEATURES, FEATURE_VERSION, READY
from labels import HORIZON_MINUTES, MODEL_THRESHOLD, TARGET_RETURN, add_labels, calibration_split, time_split
from model_store import predict_probability, save_model
from process_lock import ProcessLock
from validation import feature_drift_report, make_classifier, walk_forward_metrics

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "market.csv"
HISTORY = ROOT / "data" / "model_history.jsonl"


def _is_disposable_synthetic_pipeline():
    """Allow synthetic rows only in the explicitly marked temporary self-test copy."""
    expected_root = os.getenv("SELFTEST_PIPELINE_ROOT")
    if os.getenv("ALLOW_SYNTHETIC_DATA") != "1" or not expected_root:
        return False
    try:
        root = ROOT.resolve()
        expected = Path(expected_root).resolve()
        marker = root / ".selftest_synthetic_pipeline"
        return root == expected and marker.is_file()
    except (OSError, RuntimeError, ValueError):
        return False


def main():
    try:
        with ProcessLock(ROOT / "data" / "train.lock"):
            return train()
    except RuntimeError as exc:
        raise SystemExit(f"Training not started: {exc}") from exc


def train():
    if not DATA.exists():
        raise SystemExit(f"No {DATA}. Run collector.py first.")
    try:
        market, quality = load_market_csv(DATA)
    except DataQualityError as exc:
        raise SystemExit(f"Market data rejected: {exc}") from exc
    if quality["future_rows"]:
        raise SystemExit(f"Market data contains {quality['future_rows']} future-dated rows; review the collector clock.")
    try:
        require_training_source(quality, allow_synthetic=_is_disposable_synthetic_pipeline())
    except DataQualityError as exc:
        raise SystemExit(f"Market data rejected: {exc}. Collect DexScreener rows into data/market.csv; "
                         "keep alternate feeds separate.") from exc
    if quality["valid_rows"] < quality["input_rows"]:
        print(f"Data quality: dropped {quality['invalid_rows_dropped']} invalid and "
              f"{quality['duplicate_rows_dropped']} duplicate rows.")
    print("Source rows:", quality["sources"])

    df = add_indicators(market)
    df = add_labels(df, HORIZON_MINUTES, TARGET_RETURN)
    df = df.loc[df["price_usd"] > 0].dropna(subset=READY + ["future_return", "target"])
    df["target"] = df["target"].astype(int)
    if len(df) < 500:
        raise SystemExit(f"Only {len(df)} usable rows. Collect substantially more data before training.")

    try:
        train_rows, test, cut = time_split(df, HORIZON_MINUTES)
    except ValueError as exc:
        raise SystemExit(f"Cannot make a chronological split: {exc}") from exc
    if train_rows.empty or test.empty:
        raise SystemExit("Not enough time span for a train/test split. Collect longer.")
    if train_rows["target"].nunique() < 2:
        raise SystemExit("Training window contains one class only; collect data with both outcomes.")

    fit, calibration, calibration_start = calibration_split(train_rows, HORIZON_MINUTES)
    calibrator = None
    calibration_method = "uncalibrated"
    model = make_classifier()
    fit_rows = train_rows
    calibration_rows = pd.DataFrame()
    if (len(fit) >= 100 and len(calibration) >= 50
            and fit["target"].nunique() > 1 and calibration["target"].nunique() > 1):
        fit_rows = fit
        calibration_rows = calibration
        model.fit(fit_rows[FEATURES], fit_rows["target"].astype(int))
        raw_cal = model.predict_proba(calibration_rows[FEATURES])[:, 1]
        calibrator = LogisticRegression(max_iter=500, random_state=42)
        calibrator.fit(raw_cal.reshape(-1, 1), calibration_rows["target"].astype(int))
        calibration_method = "temporal_platt_logistic"
    else:
        print("Calibration skipped: the pre-test calibration window lacks enough rows or both classes.")
        model.fit(train_rows[FEATURES], train_rows["target"].astype(int))

    feature_drift = feature_drift_report(fit_rows, test, FEATURES)

    bundle = {
        "format_version": 2,
        "feature_version": FEATURE_VERSION,
        "model": model,
        "calibrator": calibrator,
        "calibration_method": calibration_method,
        "features": FEATURES,
        "threshold": MODEL_THRESHOLD,
        "horizon_minutes": HORIZON_MINUTES,
        "target_return": TARGET_RETURN,
        "test_start": cut,
        "calibration_start": calibration_start if calibrator is not None else None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_start": train_rows["timestamp"].min().isoformat(),
        "training_end": train_rows["timestamp"].max().isoformat(),
        "train_rows": int(len(fit_rows)),
        "calibration_rows": int(len(calibration_rows)),
        "training_data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "feature_drift": feature_drift,
    }

    probabilities = predict_probability(bundle, test[FEATURES])
    predictions = (probabilities >= MODEL_THRESHOLD).astype(int)
    y_test = test["target"].astype(int)
    auc = float(roc_auc_score(y_test, probabilities)) if y_test.nunique() > 1 else None
    brier = float(brier_score_loss(y_test, probabilities))
    baseline_probability = float(fit_rows["target"].mean())
    baseline_brier = float(brier_score_loss(y_test, [baseline_probability] * len(test)))

    print(f"Horizon: {HORIZON_MINUTES} min | target: {TARGET_RETURN:.0%} | threshold: {MODEL_THRESHOLD:.0%}")
    print("Usable rows:", len(df), "| Fit:", len(fit_rows), "| Calibration:", len(calibration_rows), "| Test:", len(test))
    print("Test period starts:", cut)
    print("Positive rate fit/calibration/test:", round(fit_rows["target"].mean(), 3), "/",
          (round(calibration_rows["target"].mean(), 3) if len(calibration_rows) else "n/a"), "/",
          round(y_test.mean(), 3))
    print("Probability calibration:", calibration_method)
    print("Test AUC:", "n/a (one outcome class)" if auc is None else round(auc, 4))
    print(f"Test Brier score: {brier:.4f} | prevalence baseline: {baseline_brier:.4f}")
    largest_drift = sorted((item for item in feature_drift if item["psi"] is not None),
                           key=lambda item: item["psi"], reverse=True)[:5]
    print("Largest out-of-time feature drift (PSI; diagnostics only):",
          [(item["feature"], round(item["psi"], 3), round(item["test_missing_rate"], 3))
           for item in largest_drift])
    print(classification_report(y_test, predictions, zero_division=0))

    folds = walk_forward_metrics(train_rows, FEATURES, n_splits=3, horizon_minutes=HORIZON_MINUTES)
    bundle["walk_forward"] = folds
    if folds:
        print("Expanding-window validation (prior to final test block):")
        for fold in folds:
            auc_text = "n/a" if fold["auc"] is None else f"{fold['auc']:.4f}"
            print(f"  fold {fold['fold']}: {fold['test_rows']} rows, AUC {auc_text}, "
                  f"Brier {fold['brier']:.4f} (baseline {fold['baseline_brier']:.4f})")
    else:
        print("Walk-forward validation unavailable: not enough distinct time blocks.")

    save_model(bundle)
    print("Saved hash-checked models/model.joblib and models/model_meta.json")
    history = {
        "trained_at": bundle["trained_at"], "rows": int(len(df)), "fit_rows": len(fit_rows),
        "calibration_rows": len(calibration_rows), "test_rows": int(len(test)),
        "tokens": int(df["token_address"].nunique()), "test_auc": auc, "test_brier": brier,
        "baseline_brier": baseline_brier, "test_positive_rate": float(y_test.mean()),
        "features": FEATURES, "feature_version": FEATURE_VERSION,
        "horizon_minutes": HORIZON_MINUTES, "target_return": TARGET_RETURN,
        "calibration_method": calibration_method, "training_data_sha256": bundle["training_data_sha256"],
        "data_quality": quality, "walk_forward": folds, "feature_drift": feature_drift,
    }
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(history, allow_nan=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"Logged to {HISTORY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
