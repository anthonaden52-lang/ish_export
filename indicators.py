import os
import math

import numpy as np
import pandas as pd

# A gap longer than this between two snapshots of a token starts a new
# "segment": indicators restart instead of bridging hours of missing data.
GAP_MINUTES = float(os.getenv("GAP_MINUTES", "5"))
if not math.isfinite(GAP_MINUTES) or GAP_MINUTES <= 0:
    raise ValueError("GAP_MINUTES must be a finite positive number.")
FEATURE_VERSION = "3"

# Rows of history per token used at inference time (paper trader, dashboard).
# Training computes indicators on the full segment; with 300 rows the slowest
# EMA (span 50) keeps < 0.001% weight from older rows, so features match.
HISTORY_ROWS = 300

RETURN_WINDOWS_MIN = (1, 5, 15)


def window_history(df):
    """The ONE history window used everywhere features are computed live."""
    return (df.sort_values(["token_address", "timestamp"])
              .groupby("token_address", group_keys=False).tail(HISTORY_ROWS))


def _time_change(df, col, minutes):
    """Relative change of `col` vs its last value at or before (t - minutes), same token+segment."""
    tol = pd.Timedelta(minutes=max(1.0, minutes * 0.5))
    left = df[["timestamp", "seg"]].copy()
    left["_i"] = np.arange(len(df))
    left["_t"] = left["timestamp"] - pd.Timedelta(minutes=minutes)
    right = df[["timestamp", "seg", col]].rename(columns={"timestamp": "_t", col: "_p"})
    m = pd.merge_asof(left.sort_values("_t"), right.sort_values("_t"), on="_t", by="seg",
                      direction="backward", tolerance=tol).sort_values("_i")
    prev = m["_p"].to_numpy(dtype=float)
    prev = np.where(prev == 0, np.nan, prev)
    return df[col].to_numpy(dtype=float) / prev - 1


def _time_return(df, minutes):
    return _time_change(df, "price_usd", minutes)


# Columns added in v2 of the collector. Older market.csv files lack them; they
# are filled with NaN so old data still loads (the model handles NaN).
V2_COLS = ["market_cap", "fdv", "volume_1h", "buys_1h", "sells_1h"]


def add_indicators(df):
    required = {"timestamp", "token_address", "price_usd"}
    if not isinstance(df, pd.DataFrame) or not required.issubset(df.columns):
        raise ValueError("Indicator input must contain timestamp, token_address, and price_usd.")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["token_address"] = df["token_address"].where(df["token_address"].notna(), "").astype(str).str.strip()
    df = df.sort_values(["token_address", "timestamp"], kind="mergesort")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in ("liquidity_usd", "volume_24h", "volume_5m", "buys_5m", "sells_5m", "txns_24h"):
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["timestamp", "price_usd"])
    df = df.loc[df["token_address"].ne("") & (df["price_usd"] >= 0)]
    df = df.drop_duplicates(["token_address", "timestamp"], keep="last")
    df = df.sort_values(["token_address", "timestamp"], kind="mergesort")
    for c in V2_COLS:
        if c not in df.columns:
            df[c] = np.nan
    gap = df.groupby("token_address")["timestamp"].diff() > pd.Timedelta(minutes=GAP_MINUTES)
    new_tok = df["token_address"] != df["token_address"].shift()
    df["seg"] = (gap | new_tok).cumsum()
    g = df.groupby("seg", group_keys=False)

    def ema(s, n):
        return s.ewm(span=n, adjust=False).mean()

    for n in (9, 21, 50, 12, 26):
        df[f"ema{n}"] = g["price_usd"].transform(lambda s, n=n: ema(s, n))
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df.groupby("seg")["macd"].transform(lambda s: ema(s, 9))

    delta = g["price_usd"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(df["seg"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    avg_loss = loss.groupby(df["seg"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    df["rsi14"] = rsi

    # Returns over REAL time (minutes), not over a number of scans.
    for n in RETURN_WINDOWS_MIN:
        df[f"return_{n}"] = _time_return(df, n)

    df["volatility_15"] = df.groupby("seg")["return_1"].transform(
        lambda s: s.rolling(15, min_periods=10).std())
    df["volume_change_5"] = g["volume_5m"].pct_change(5, fill_method=None)
    df["buy_sell_ratio"] = (df["buys_5m"] + 1) / (df["sells_5m"] + 1)
    df["volume_liquidity"] = df["volume_24h"] / df["liquidity_usd"].replace(0, np.nan)
    df["ema_spread"] = (df["ema9"] - df["ema21"]) / df["price_usd"].replace(0, np.nan)
    df["macd_spread"] = df["macd"] - df["macd_signal"]

    # --- v2 features ---
    df["liquidity_change_15"] = _time_change(df, "liquidity_usd", 15)
    # 5-minute volume vs the average 5-minute volume over the last hour (>1 = accelerating)
    df["volume_accel"] = df["volume_5m"] / (df["volume_1h"] / 12).replace(0, np.nan)
    df["buy_sell_ratio_1h"] = (df["buys_1h"] + 1) / (df["sells_1h"] + 1)
    df["mcap_liquidity"] = df["market_cap"] / df["liquidity_usd"].replace(0, np.nan)
    df["log_fdv"] = np.log10(df["fdv"].where(df["fdv"] > 0))

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    # A price-0 row (dead/vanished pair) is never a valid entry.
    df.loc[df["price_usd"] <= 0, READY] = np.nan
    return df


FEATURES = [
    "rsi14", "ema_spread", "macd_spread", "return_1", "return_5", "return_15",
    "volatility_15", "volume_change_5", "buy_sell_ratio", "volume_liquidity",
    "liquidity_usd", "volume_24h", "txns_24h", "pair_age_minutes",
    "liquidity_change_15", "volume_accel", "buy_sell_ratio_1h", "mcap_liquidity", "log_fdv",
]

# A row is usable once enough history exists for the slow indicators.
READY = ["rsi14", "return_15"]
