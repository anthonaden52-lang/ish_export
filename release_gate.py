"""Judge the Acorn 2.0 release gate from evidence that already exists on disk.

The requirements live in MODEL_RELEASE_GATE.md. Every one is evaluated against
saved artifacts and a missing artifact fails the gate rather than being skipped,
because an unverifiable claim is not a passed one. Promotion is a data-quality
result, not a side effect of running the trainer again.
"""
import math
import statistics

# Thresholds mirror MODEL_RELEASE_GATE.md. Change them there first.
MIN_DATASET_DAYS = 7.0
MIN_TEST_ROWS = 500
MIN_TEST_PER_CLASS = 50
MIN_TEST_AUC = 0.60
MIN_BRIER_IMPROVEMENT = 0.05
MIN_CALIBRATION_ROWS = 200
MIN_CALIBRATION_PER_CLASS = 30
MIN_FOLDS = 3
MIN_FOLD_ROWS = 100
MIN_FOLD_PER_CLASS = 20
MIN_MEDIAN_FOLD_AUC = 0.58
MIN_CLOSED_TRADES = 30
# The daily halt latches on the snapshot that breaches it, so one interval of
# overshoot past the configured limit is expected rather than a breach.
DAILY_LOSS_OVERSHOOT_TOLERANCE = 1.25
GATED_NAME = "Acorn 2.0"
HELD_NAME = "Acorn 1.0"
AREA_DATA = "Data and split"
AREA_QUALITY = "Predictive quality"
AREA_PAPER = "Paper-trading evidence"
AREA_RELEASE = "Promotion record"


def number(value):
    """Return a finite numeric value, or None for missing and non-numeric input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def class_pair(counts):
    """Return (positive, negative) for a recorded class-count mapping, or None."""
    if not isinstance(counts, dict):
        return None
    positive, negative = counts.get("positive"), counts.get("negative")
    if not isinstance(positive, int) or not isinstance(negative, int):
        return None
    return positive, negative


def _check(identifier, area, requirement, passed, detail):
    return {"id": identifier, "area": area, "requirement": requirement,
            "passed": bool(passed), "detail": detail}


def _dataset_checks(training, dataset, checks):
    span = number(dataset.get("span_days"))
    checks.append(_check(
        "dataset_span", AREA_DATA,
        f"eligible dataset spans at least {MIN_DATASET_DAYS:.0f} calendar days",
        span is not None and span >= MIN_DATASET_DAYS,
        "no dated training record" if span is None
        else f"{span:.2f} days measured from {dataset.get('span_source', 'unknown')}"))

    test_rows = number(training.get("test_rows"))
    checks.append(_check(
        "test_block_rows", AREA_DATA,
        f"final chronological test block holds at least {MIN_TEST_ROWS} labelled observations",
        test_rows is not None and test_rows >= MIN_TEST_ROWS,
        "no training record" if test_rows is None else f"{int(test_rows)} rows"))

    pair = class_pair((training.get("class_counts") or {}).get("test"))
    checks.append(_check(
        "test_block_classes", AREA_DATA,
        f"the test block holds at least {MIN_TEST_PER_CLASS} of each class",
        pair is not None and min(pair) >= MIN_TEST_PER_CLASS,
        "no class counts recorded" if pair is None
        else f"{pair[0]} positive, {pair[1]} negative"))


def _quality_checks(training, folds, checks):
    auc = number(training.get("test_auc"))
    checks.append(_check(
        "test_auc", AREA_QUALITY,
        f"final-test ROC AUC is at least {MIN_TEST_AUC:.2f}",
        auc is not None and auc >= MIN_TEST_AUC,
        "not computed (one outcome class in the test block)" if auc is None else f"AUC {auc:.4f}"))

    brier, baseline = number(training.get("test_brier")), number(training.get("baseline_brier"))
    if brier is None or baseline is None or baseline <= 0:
        brier_detail, improved = "no usable Brier/prevalence pair recorded", False
    else:
        brier_detail = f"Brier {brier:.4f} vs baseline {baseline:.4f} ({1 - brier / baseline:+.1%})"
        improved = brier <= baseline * (1 - MIN_BRIER_IMPROVEMENT)
    checks.append(_check(
        "test_brier", AREA_QUALITY,
        f"final-test Brier score beats the prevalence baseline by at least "
        f"{MIN_BRIER_IMPROVEMENT:.0%}",
        improved, brier_detail))

    calibration_rows = number(training.get("calibration_rows"))
    calibration = class_pair((training.get("class_counts") or {}).get("calibration"))
    if calibration is None:
        calibration_detail = ("no class counts recorded" if calibration_rows is None
                              else f"{int(calibration_rows)} rows, class counts unavailable")
    else:
        calibration_detail = (f"{int(calibration_rows or 0)} rows, "
                              f"{calibration[0]} positive, {calibration[1]} negative")
    checks.append(_check(
        "calibration_block", AREA_QUALITY,
        f"the calibration block holds at least {MIN_CALIBRATION_ROWS} rows with "
        f"{MIN_CALIBRATION_PER_CLASS} of each class",
        calibration_rows is not None and calibration_rows >= MIN_CALIBRATION_ROWS
        and calibration is not None and min(calibration) >= MIN_CALIBRATION_PER_CLASS,
        calibration_detail))

    method = str(training.get("calibration_method") or "uncalibrated")
    checks.append(_check(
        "calibration_fitted", AREA_QUALITY,
        "the saved artifact uses a fitted calibrator, not uncalibrated probabilities",
        method not in ("", "uncalibrated", "none"), f"calibration method: {method}"))

    qualifying = []
    for fold in folds:
        sizes = class_pair({"positive": fold.get("test_positives"),
                            "negative": fold.get("test_negatives")})
        if (number(fold.get("test_rows")) or 0) >= MIN_FOLD_ROWS and sizes and min(sizes) >= MIN_FOLD_PER_CLASS:
            qualifying.append(fold)
    checks.append(_check(
        "walk_forward_folds", AREA_QUALITY,
        f"at least {MIN_FOLDS} expanding-window folds hold {MIN_FOLD_ROWS} rows and "
        f"{MIN_FOLD_PER_CLASS} of each class",
        len(qualifying) >= MIN_FOLDS,
        f"{len(qualifying)} qualifying of {len(folds)} reported folds"))

    beating = [f for f in qualifying
               if number(f.get("brier")) is not None and number(f.get("baseline_brier")) is not None
               and f["brier"] < f["baseline_brier"]]
    checks.append(_check(
        "walk_forward_brier", AREA_QUALITY,
        "every qualifying fold beats its own prevalence baseline on Brier score",
        bool(qualifying) and len(beating) == len(qualifying),
        f"{len(beating)} of {len(qualifying)} qualifying folds beat their baseline"))

    aucs = [number(fold.get("auc")) for fold in qualifying]
    aucs = [value for value in aucs if value is not None]
    median_auc = statistics.median(aucs) if aucs else None
    checks.append(_check(
        "walk_forward_auc", AREA_QUALITY,
        f"median qualifying-fold AUC is at least {MIN_MEDIAN_FOLD_AUC:.2f}",
        median_auc is not None and median_auc >= MIN_MEDIAN_FOLD_AUC,
        "no fold AUC available" if median_auc is None else f"median AUC {median_auc:.4f}"))
    return qualifying, median_auc


def _paper_checks(backtest, checks):
    risk = backtest.get("risk") or {}
    limits = backtest.get("risk_limits") or {}

    closed = number(backtest.get("closed_trades"))
    checks.append(_check(
        "backtest_trades", AREA_PAPER,
        f"the cost-aware paper backtest closed at least {MIN_CLOSED_TRADES} trades",
        closed is not None and closed >= MIN_CLOSED_TRADES,
        "no backtest summary" if closed is None else f"{int(closed)} closed trades"))

    expectancy = number(backtest.get("mean_net_return"))
    checks.append(_check(
        "backtest_expectancy", AREA_PAPER,
        "net expectancy per trade stays positive after the configured costs",
        expectancy is not None and expectancy > 0,
        "not measurable until trades close" if expectancy is None else f"{expectancy:+.3%} per trade"))

    daily_limit, worst_day = number(limits.get("max_daily_loss")), number(risk.get("worst_day_loss_pct"))
    if daily_limit is None or daily_limit <= 0:
        risk_detail, within = "no configured daily loss limit recorded", False
    elif worst_day is None:
        risk_detail, within = "no intraday equity evidence recorded", False
    else:
        within = worst_day >= -daily_limit * DAILY_LOSS_OVERSHOOT_TOLERANCE
        risk_detail = (f"worst intraday equity change {worst_day:.2%} against a {daily_limit:.0%} "
                       f"halt, allowing {DAILY_LOSS_OVERSHOOT_TOLERANCE:g}x for one snapshot")
    checks.append(_check(
        "backtest_risk_limits", AREA_PAPER,
        "the paper backtest stays inside the configured risk limits", within, risk_detail))

    peak, cap = number(risk.get("max_open_positions")), number(limits.get("max_positions"))
    checks.append(_check(
        "backtest_position_cap", AREA_PAPER,
        "no more positions were open at once than the configured maximum",
        peak is not None and cap is not None and peak <= cap,
        "no position evidence recorded" if peak is None or cap is None
        else f"{int(peak)} concurrent positions against a cap of {int(cap)}"))


def _release_checks(evidence, training, backtest, folds, checks):
    recorded = {
        "model artifact hash": evidence.get("model_sha256"),
        "training data hash": training.get("training_data_sha256"),
        "test metrics": training.get("test_brier"),
        "walk-forward metrics": folds,
        "calibration method": training.get("calibration_method"),
        "backtest results": backtest,
    }
    missing = [name for name, value in recorded.items() if not value]
    checks.append(_check(
        "release_notes_recorded", AREA_RELEASE,
        "artifact hash, data hash, test and walk-forward metrics, calibration method "
        "and backtest results are all recorded",
        not missing, "every required field recorded" if not missing
        else "missing: " + ", ".join(missing)))


def evaluate(evidence):
    """Return the per-requirement results, the verdict, and a metrics snapshot."""
    evidence = evidence or {}
    training = evidence.get("training") or {}
    backtest = evidence.get("backtest") or {}
    dataset = evidence.get("dataset") or {}
    folds = [fold for fold in (training.get("walk_forward") or []) if isinstance(fold, dict)]
    checks = []

    _dataset_checks(training, dataset, checks)
    qualifying, median_auc = _quality_checks(training, folds, checks)
    _paper_checks(backtest, checks)
    _release_checks(evidence, training, backtest, folds, checks)

    failing = [check["id"] for check in checks if not check["passed"]]
    test_pair = class_pair((training.get("class_counts") or {}).get("test"))
    return {
        "generated_at": evidence.get("generated_at"),
        "release_name": HELD_NAME if failing else GATED_NAME,
        "eligible_for_2_0": not failing,
        "checks": checks,
        "summary": {"passed": len(checks) - len(failing), "failed": len(failing),
                    "total": len(checks), "failing": failing},
        "identity": {"model_sha256": evidence.get("model_sha256"),
                     "training_data_sha256": training.get("training_data_sha256")},
        "metrics": {
            "trained_at": training.get("trained_at"),
            "usable_rows": number(training.get("rows")),
            "dataset_span_days": number(dataset.get("span_days")),
            "dataset_span_source": dataset.get("span_source"),
            "test_rows": number(training.get("test_rows")),
            "test_positives": test_pair[0] if test_pair else None,
            "test_negatives": test_pair[1] if test_pair else None,
            "test_auc": number(training.get("test_auc")),
            "test_brier": number(training.get("test_brier")),
            "baseline_brier": number(training.get("baseline_brier")),
            "calibration_rows": number(training.get("calibration_rows")),
            "calibration_method": training.get("calibration_method"),
            "folds_reported": len(folds),
            "folds_qualifying": len(qualifying),
            "median_fold_auc": median_auc,
            "closed_trades": number(backtest.get("closed_trades")),
            "mean_net_return": number(backtest.get("mean_net_return")),
            "final_balance": number(backtest.get("final_balance")),
            "return_pct": number(backtest.get("return_pct")),
            "max_drawdown": number(backtest.get("max_drawdown")),
            "worst_day_loss_pct": number((backtest.get("risk") or {}).get("worst_day_loss_pct")),
        },
    }
