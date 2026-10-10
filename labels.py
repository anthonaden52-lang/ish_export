"""Shared settings, time-based labels, and the chronological split."""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

def _positive_int(name, default):
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer.") from None
    if not math.isfinite(value) or value <= 0 or not value.is_integer():
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _fraction(name, default, *, positive=False):
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must be a finite fraction in [0, 1].") from None
    if not math.isfinite(value) or value < 0 or value > 1 or (positive and value == 0):
        raise ValueError(f"{name} must be a finite fraction in {'(0, 1]' if positive else '[0, 1]'}.")
    return value


HORIZON_MINUTES = _positive_int("HORIZON_MINUTES", 15)
TARGET_RETURN = float(os.getenv("TARGET_RETURN", "0.10"))
if not math.isfinite(TARGET_RETURN) or TARGET_RETURN <= 0:
    raise ValueError("TARGET_RETURN must be a finite positive return fraction.")
MODEL_THRESHOLD = _fraction("MODEL_THRESHOLD", 0.70, positive=True)
# Costs are PER SIDE: charged once on entry and again on exit.
FEE = _fraction("BACKTEST_FEE", 0.005)
SLIPPAGE = _fraction("BACKTEST_SLIPPAGE", 0.02)
if FEE + SLIPPAGE >= 1:
    raise ValueError("BACKTEST_FEE + BACKTEST_SLIPPAGE must be below 1.")
TEST_FRACTION = 0.20


def _tolerance(horizon_minutes):
    # How far past the exact horizon we accept a price (scans are irregular).
    return pd.Timedelta(minutes=max(1.0, horizon_minutes * 0.2))


def add_labels(df, horizon_minutes=HORIZON_MINUTES, target_return=TARGET_RETURN):
    """Label each row with the token's price `horizon_minutes` of REAL TIME later.

    Uses the first snapshot at or after (timestamp + horizon), within a
    tolerance. Rows with no such snapshot get NaN future_return and must be
    dropped by the caller. Only an observed zero-price marker confirms a
    total loss; missing later rows remain censored rather than guessed losses.
    """
    if not isinstance(df, pd.DataFrame) or not {"timestamp", "token_address", "price_usd"}.issubset(df.columns):
        raise ValueError("Label input must contain timestamp, token_address, and price_usd columns.")
    if not math.isfinite(float(horizon_minutes)) or horizon_minutes <= 0:
        raise ValueError("Label horizon must be finite and positive.")
    if not math.isfinite(float(target_return)) or target_return <= 0:
        raise ValueError("Target return must be finite and positive.")
    left = df.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
    left["price_usd"] = pd.to_numeric(left["price_usd"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    left = left.dropna(subset=["timestamp", "token_address", "price_usd"])
    left = left.loc[left["price_usd"] >= 0]
    if left.duplicated(["token_address", "timestamp"]).any():
        raise ValueError("Label input contains duplicate token/timestamp rows; clean it before labeling.")
    horizon = pd.Timedelta(minutes=horizon_minutes)
    left = left.sort_values(["timestamp", "token_address"], kind="mergesort")
    left["_target_ts"] = left["timestamp"] + horizon

    right = (
        left[["timestamp", "token_address", "price_usd"]]
        .rename(columns={"timestamp": "future_ts", "price_usd": "future_price"})
        .sort_values(["future_ts", "token_address"], kind="mergesort")
    )
    left = left.sort_values(["_target_ts", "token_address"], kind="mergesort")
    out = pd.merge_asof(
        left,
        right,
        left_on="_target_ts",
        right_on="future_ts",
        by="token_address",
        direction="forward",
        tolerance=_tolerance(horizon_minutes),
    )
    valid_future = out["future_price"].notna() & (out["future_price"] >= 0) & (out["price_usd"] > 0)
    out["future_return"] = np.nan
    out.loc[valid_future, "future_return"] = out.loc[valid_future, "future_price"] / out.loc[valid_future, "price_usd"] - 1

    # The tracker records an observed zero-price row for a pair that vanished.
    # A token merely missing from later CSV rows is ambiguous (data gap, parse
    # failure, or expired tracking), so keep that outcome censored.
    labeled = out["future_return"].notna() & (out["price_usd"] > 0)
    out["target"] = np.nan
    out.loc[labeled, "target"] = (out.loc[labeled, "future_return"] >= target_return).astype(int)
    out.loc[out["price_usd"] <= 0, ["future_return", "target"]] = np.nan  # can't enter at price 0
    return out.drop(columns="_target_ts")


def time_split(df, horizon_minutes=HORIZON_MINUTES, test_fraction=TEST_FRACTION):
    """Split by TIMESTAMP across all tokens.

    Train rows whose label window reaches into the test period are dropped
    (embargo), so no training label peeks at test-period prices.
    """
    if df.empty or "timestamp" not in df.columns:
        raise ValueError("Time split requires a non-empty dataframe with timestamps.")
    if not math.isfinite(float(horizon_minutes)) or horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be finite and positive.")
    if not math.isfinite(float(test_fraction)) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna().sort_values().unique()
    if len(timestamps) < 3:
        raise ValueError("Time split requires at least three distinct timestamps.")
    cut = pd.Timestamp(timestamps[min(len(timestamps) - 1, max(1, int(len(timestamps) * (1 - test_fraction))))])
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    embargo = pd.Timedelta(minutes=horizon_minutes) + _tolerance(horizon_minutes)
    train = df[df["timestamp"] < cut - embargo]
    test = df[df["timestamp"] >= cut]
    return train, test, cut


def calibration_split(df, horizon_minutes=HORIZON_MINUTES, calibration_fraction=0.20):
    """Split pre-test data into chronological fit/calibration blocks with a purge gap."""
    if df.empty or not 0 < calibration_fraction < 1:
        raise ValueError("Calibration split needs data and a fraction between 0 and 1.")
    if not math.isfinite(float(horizon_minutes)) or horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be finite and positive.")
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    stamps = data["timestamp"].dropna().sort_values().unique()
    if len(stamps) < 3:
        raise ValueError("Calibration split requires at least three distinct timestamps.")
    cut = pd.Timestamp(stamps[max(1, int(len(stamps) * (1 - calibration_fraction)))])
    embargo = pd.Timedelta(minutes=horizon_minutes) + _tolerance(horizon_minutes)
    fit = data[data["timestamp"] < cut - embargo]
    calibration = data[data["timestamp"] >= cut]
    return fit, calibration, cut
