"""Atomic local model storage with hash, manifest, and bundle validation.

WARNING: joblib/pickle files can run arbitrary code when loaded. Only load a
model that YOU trained with train.py on this machine. Never load a model sent
by someone else or downloaded from the internet.
"""
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib

from process_lock import ProcessLock

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
HASH_PATH = MODEL_DIR / "model.sha256"
META_PATH = MODEL_DIR / "model_meta.json"
LOCK_PATH = MODEL_DIR / ".model.lock"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_fraction(value, name, *, minimum=0.0, maximum=1.0):
    if isinstance(value, bool):
        raise ValueError(f"Invalid model {name}.")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"Invalid model {name}.") from None
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"Invalid model {name}.")
    return result


def validate_bundle(bundle):
    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be an object.")
    model = bundle.get("model")
    features = bundle.get("features")
    if model is None or not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Model bundle has no probability model.")
    if (not isinstance(features, (list, tuple)) or not features
            or any(not isinstance(feature, str) or not feature for feature in features)
            or len(set(features)) != len(features)):
        raise ValueError("Model bundle has an invalid feature contract.")
    threshold = _finite_fraction(bundle.get("threshold"), "threshold")
    if threshold <= 0:
        raise ValueError("Invalid model threshold.")
    horizon = bundle.get("horizon_minutes")
    if isinstance(horizon, bool):
        raise ValueError("Model bundle has an invalid horizon.")
    try:
        horizon = float(horizon)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("Model bundle has an invalid horizon.") from None
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Model bundle has an invalid horizon.")
    target_return = _finite_fraction(bundle.get("target_return"), "target return",
                                     minimum=0.0, maximum=float("inf"))
    if target_return <= 0:
        raise ValueError("Invalid model target return.")
    test_start = bundle.get("test_start")
    if test_start is None:
        raise ValueError("Model bundle has no out-of-time test start.")
    try:
        parsed_start = test_start if isinstance(test_start, datetime) else datetime.fromisoformat(
            str(test_start).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise ValueError("Model bundle has an invalid test start.") from None
    if parsed_start.tzinfo is None or parsed_start.utcoffset() is None:
        raise ValueError("Model test start must include a timezone.")
    calibrator = bundle.get("calibrator")
    if calibrator is not None and not callable(getattr(calibrator, "predict_proba", None)):
        raise ValueError("Model bundle has an invalid probability calibrator.")
    return bundle


def predict_probability(bundle, features):
    """Apply the saved calibration model when present; reject invalid outputs."""
    validate_bundle(bundle)
    raw = np.asarray(bundle["model"].predict_proba(features), dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("Model returned an invalid probability matrix.")
    probability = raw[:, 1]
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        probability = np.asarray(calibrator.predict_proba(probability.reshape(-1, 1)), dtype=float)[:, 1]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Model returned a non-finite or out-of-range probability.")
    return probability


def _manifest(bundle, digest):
    def timestamp(name):
        value = bundle.get(name)
        return value.isoformat() if callable(getattr(value, "isoformat", None)) else value

    return {
        "sha256": digest,
        "format_version": int(bundle.get("format_version", 1)),
        "feature_version": bundle.get("feature_version", "legacy"),
        "features": list(bundle["features"]),
        "horizon_minutes": float(bundle["horizon_minutes"]),
        "target_return": float(bundle["target_return"]),
        "threshold": float(bundle["threshold"]),
        "calibration_method": bundle.get("calibration_method", "uncalibrated"),
        "test_start": timestamp("test_start"),
        "calibration_start": timestamp("calibration_start"),
        "training_start": bundle.get("training_start"),
        "training_end": bundle.get("training_end"),
        "train_rows": bundle.get("train_rows"),
        "calibration_rows": bundle.get("calibration_rows"),
        "trained_at": bundle.get("trained_at"),
        "training_data_sha256": bundle.get("training_data_sha256"),
    }


def save_model(bundle):
    validate_bundle(bundle)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessLock(LOCK_PATH):
        model_tmp = None
        hash_tmp = None
        meta_tmp = None
        try:
            with tempfile.NamedTemporaryFile(prefix="model_", suffix=".joblib", dir=MODEL_DIR, delete=False) as f:
                model_tmp = Path(f.name)
            joblib.dump(bundle, model_tmp)
            digest = _sha256(model_tmp)
            with tempfile.NamedTemporaryFile(prefix="model_hash_", dir=MODEL_DIR, mode="w", encoding="utf-8",
                                             delete=False) as f:
                hash_tmp = Path(f.name)
                f.write(digest + "\n")
                f.flush()
                os.fsync(f.fileno())
            with tempfile.NamedTemporaryFile(prefix="model_meta_", dir=MODEL_DIR, mode="w", encoding="utf-8",
                                             delete=False) as f:
                meta_tmp = Path(f.name)
                json.dump(_manifest(bundle, digest), f, allow_nan=False, sort_keys=True, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            model_tmp.replace(MODEL_PATH)
            hash_tmp.replace(HASH_PATH)
            meta_tmp.replace(META_PATH)
        finally:
            for tmp in (model_tmp, hash_tmp, meta_tmp):
                if tmp is not None:
                    tmp.unlink(missing_ok=True)


def load_model(path=MODEL_PATH):
    p = Path(path).resolve()
    if p != MODEL_PATH.resolve():
        raise RuntimeError(f"Refusing to load model from {p}: only {MODEL_PATH} is trusted.")
    if not p.exists() or not HASH_PATH.exists():
        raise RuntimeError("Refusing to load model: missing local model/hash. Train locally with train.py.")
    with ProcessLock(LOCK_PATH):
        expected = HASH_PATH.read_text(encoding="utf-8").strip()
        actual = _sha256(p)
        if not expected or expected != actual:
            raise RuntimeError("Refusing to load model: missing or mismatched models/model.sha256. Retrain locally.")
        if META_PATH.exists():
            try:
                manifest = json.loads(META_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Refusing to load model: invalid model manifest: {exc}") from exc
            if manifest.get("sha256") != actual:
                raise RuntimeError("Refusing to load model: model manifest hash does not match the artifact.")
        bundle = joblib.load(p)
        return validate_bundle(bundle)
