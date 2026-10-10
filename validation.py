"""Leakage-safe expanding-window evaluation helpers used only for reporting."""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from labels import HORIZON_MINUTES, _tolerance


def make_classifier():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=2.0,
        random_state=42,
    )


def expanding_time_splits(frame, n_splits=3, horizon_minutes=HORIZON_MINUTES):
    """Return expanding train/test blocks with a label-window purge at every boundary."""
    if n_splits < 1 or horizon_minutes <= 0 or frame.empty:
        return []
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    stamps = data["timestamp"].dropna().sort_values().unique()
    if len(stamps) < n_splits + 3:
        return []
    first_test = max(2, int(len(stamps) * 0.4))
    remaining = len(stamps) - first_test
    block = max(1, remaining // n_splits)
    embargo = pd.Timedelta(minutes=horizon_minutes) + _tolerance(horizon_minutes)
    result = []
    for fold in range(n_splits):
        start_idx = first_test + fold * block
        if start_idx >= len(stamps):
            break
        end_idx = len(stamps) if fold == n_splits - 1 else min(len(stamps), start_idx + block)
        start = pd.Timestamp(stamps[start_idx])
        end = pd.Timestamp(stamps[end_idx]) if end_idx < len(stamps) else None
        train = data[data["timestamp"] < start - embargo]
        test = data[data["timestamp"] >= start]
        if end is not None:
            test = test[test["timestamp"] < end]
        if not train.empty and not test.empty and train["target"].nunique() > 1:
            result.append((train, test, start))
    return result


def walk_forward_metrics(frame, features, n_splits=3, horizon_minutes=HORIZON_MINUTES):
    """Train separate models on older windows and report each later block."""
    reports = []
    for fold, (train, test, start) in enumerate(
            expanding_time_splits(frame, n_splits=n_splits, horizon_minutes=horizon_minutes), start=1):
        if train["target"].nunique() < 2:
            continue
        model = make_classifier()
        model.fit(train[features], train["target"].astype(int))
        probabilities = model.predict_proba(test[features])[:, 1]
        if not np.isfinite(probabilities).all():
            continue
        labels = test["target"].astype(int)
        auc = float(roc_auc_score(labels, probabilities)) if labels.nunique() > 1 else None
        prevalence = float(train["target"].mean())
        reports.append({
            "fold": fold,
            "test_start": start.isoformat(),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_positives": int((labels == 1).sum()),
            "test_negatives": int((labels == 0).sum()),
            "train_positive_rate": prevalence,
            "test_positive_rate": float(labels.mean()),
            "auc": auc,
            "brier": float(brier_score_loss(labels, probabilities)),
            "baseline_brier": float(brier_score_loss(labels, np.full(len(labels), prevalence))),
        })
    return reports


def feature_drift_report(reference, current, features):
    """Report PSI and missing-rate shifts without using them to fit or select a model."""
    reports = []
    quantiles = np.linspace(0.0, 1.0, 11)
    for feature in features:
        base = pd.to_numeric(reference[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
        recent = pd.to_numeric(current[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
        base_valid, recent_valid = base[np.isfinite(base)], recent[np.isfinite(recent)]
        base_missing = 1.0 - len(base_valid) / max(1, len(base))
        recent_missing = 1.0 - len(recent_valid) / max(1, len(recent))
        psi = None
        if len(base_valid) >= 20 and len(recent_valid) >= 20:
            edges = np.unique(np.quantile(base_valid, quantiles))
            if len(edges) < 2:
                edges = np.array([base_valid[0] - 1.0, base_valid[0] + 1.0])
            else:
                edges[0], edges[-1] = -np.inf, np.inf
            bins = len(edges) - 1
            expected = np.histogram(base_valid, bins=edges)[0].astype(float)
            actual = np.histogram(recent_valid, bins=edges)[0].astype(float)
            expected = (expected + 0.5) / (expected.sum() + 0.5 * bins)
            actual = (actual + 0.5) / (actual.sum() + 0.5 * bins)
            psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        reports.append({"feature": feature, "psi": psi,
                        "train_missing_rate": float(base_missing),
                        "test_missing_rate": float(recent_missing)})
    return reports
