"""Shared market CSV schema, normalization, and data-quality reporting."""
import math
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

MARKET_FIELDS = [
    "timestamp", "token_address", "symbol", "price_usd", "liquidity_usd", "volume_24h", "volume_5m",
    "buys_5m", "sells_5m", "txns_24h", "price_change_5m", "price_change_1h", "pair_age_minutes",
    "pair_address", "dex", "url", "market_cap", "fdv", "volume_1h", "buys_1h", "sells_1h",
    "source", "discovery_sources",
]
REQUIRED_MARKET_FIELDS = ("timestamp", "token_address", "price_usd")
NUMERIC_FIELDS = (
    "price_usd", "liquidity_usd", "volume_24h", "volume_5m", "buys_5m", "sells_5m", "txns_24h",
    "price_change_5m", "price_change_1h", "pair_age_minutes", "market_cap", "fdv", "volume_1h",
    "buys_1h", "sells_1h",
)
NONNEGATIVE_FIELDS = set(NUMERIC_FIELDS) - {"price_change_5m", "price_change_1h"}


class DataQualityError(ValueError):
    """Raised when required market columns or values cannot be trusted."""


def require_training_source(report, allow_synthetic=False):
    sources = set((report or {}).get("sources", {}))
    if sources == {"synthetic_test"} and allow_synthetic:
        return
    if sources != {"dexscreener"}:
        raise DataQualityError(f"refusing mixed or unverified market sources: {sorted(sources)}")


def finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _clean_text(value, default=""):
    if value is None:
        return default
    text = str(value)
    text = "".join(" " if ord(char) < 32 or ord(char) == 127 else char for char in text)
    return " ".join(text.split())


def _is_dexscreener_url(value):
    try:
        host = (urlparse(str(value)).hostname or "").lower()
    except (TypeError, ValueError):
        return False
    return host == "dexscreener.com" or host.endswith(".dexscreener.com")


def normalize_market_row(row, source_default="unknown"):
    """Return one schema-shaped row; invalid required identity/price fails closed."""
    raw_token = row.get("token_address")
    token = _clean_text(raw_token) if isinstance(raw_token, str) else ""
    if not token:
        raise DataQualityError("market row has no token address")
    stamp = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
    if pd.isna(stamp):
        raise DataQualityError("market row has an invalid timestamp")
    price = finite_number(row.get("price_usd"))
    if price is None or price < 0:
        raise DataQualityError("market row has an invalid price")

    out = {name: row.get(name, "") for name in MARKET_FIELDS}
    out["timestamp"] = stamp.isoformat()
    out["token_address"] = token
    out["price_usd"] = price
    for name in NUMERIC_FIELDS:
        if name == "price_usd":
            continue
        value = finite_number(row.get(name))
        if value is not None and name in NONNEGATIVE_FIELDS and value < 0:
            value = None
        out[name] = value if value is not None else float("nan")
    for name in ("symbol", "pair_address", "dex", "url"):
        value = out[name]
        out[name] = _clean_text(value)
    source = out["source"]
    if not isinstance(source, str) or not source.strip():
        source = source_default
    out["source"] = _clean_text(source).lower() or source_default
    sources = out["discovery_sources"]
    if isinstance(sources, (list, tuple, set)):
        sources = ",".join(sorted(str(item) for item in sources))
    if not isinstance(sources, str) or not sources.strip():
        sources = "unknown"
    out["discovery_sources"] = _clean_text(sources, "unknown") or "unknown"
    return out


def clean_market_frame(frame, now=None):
    """Normalize a historical frame and return it with an auditable quality summary.

    Rows with a missing/invalid timestamp, token address, or price are discarded.
    A zero price is retained because the collector uses it for a pair observed as
    vanished. Repeated token/timestamp rows keep the last ingested observation.
    """
    if not isinstance(frame, pd.DataFrame):
        raise DataQualityError("market data must be a dataframe")
    missing = [name for name in REQUIRED_MARKET_FIELDS if name not in frame.columns]
    if missing:
        raise DataQualityError("market CSV is missing required columns: " + ", ".join(missing))

    data = frame.copy()
    original_rows = len(data)
    data["_ingest_order"] = np.arange(original_rows)
    string_tokens = data["token_address"].map(lambda value: isinstance(value, str))
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data["token_address"] = data["token_address"].where(data["token_address"].notna(), "").astype(str).str.strip()
    data["price_usd"] = pd.to_numeric(data["price_usd"], errors="coerce")
    price_array = data["price_usd"].to_numpy(dtype=float, na_value=np.nan)
    valid_price = np.isfinite(price_array) & (price_array >= 0)
    valid = data["timestamp"].notna() & string_tokens & data["token_address"].ne("") & valid_price
    invalid_rows = int((~valid).sum())
    data = data.loc[valid].copy()

    for name in NUMERIC_FIELDS:
        if name not in data.columns:
            data[name] = np.nan
        data[name] = pd.to_numeric(data[name], errors="coerce")
        data[name] = data[name].replace([np.inf, -np.inf], np.nan)
        if name in NONNEGATIVE_FIELDS and name != "price_usd":
            data.loc[data[name] < 0, name] = np.nan

    for name in ("source", "discovery_sources", "url", "symbol", "pair_address", "dex"):
        if name not in data.columns:
            data[name] = ""

    for name in MARKET_FIELDS:
        if name not in data.columns:
            data[name] = np.nan if name in NUMERIC_FIELDS else ""
    data["source"] = data["source"].fillna("").astype(str).str.strip().str.lower().replace("", "legacy")
    legacy = data["source"].eq("legacy")
    if legacy.any():
        dex_urls = data["url"].fillna("").astype(str).map(_is_dexscreener_url)
        data.loc[legacy & dex_urls, "source"] = "dexscreener"
    data["discovery_sources"] = data["discovery_sources"].fillna("").astype(str).replace("", "unknown")
    for name in ("symbol", "pair_address", "dex", "url"):
        data[name] = data[name].fillna("").astype(str)

    data = data.sort_values(["timestamp", "_ingest_order"], kind="mergesort")
    before_dedup = len(data)
    data = data.drop_duplicates(["token_address", "timestamp"], keep="last")
    duplicate_rows = before_dedup - len(data)
    data = data.drop(columns="_ingest_order").sort_values(["timestamp", "token_address"], kind="mergesort")
    data = data.reset_index(drop=True)

    as_of = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    last = data["timestamp"].max() if not data.empty else pd.NaT
    age_seconds = None if pd.isna(last) else max(0.0, (as_of - last).total_seconds())
    report = {
        "input_rows": original_rows,
        "valid_rows": int(len(data)),
        "invalid_rows_dropped": invalid_rows,
        "duplicate_rows_dropped": int(duplicate_rows),
        "dead_price_rows": int((data["price_usd"] == 0).sum()),
        "tokens": int(data["token_address"].nunique()),
        "first_timestamp": None if data.empty else data["timestamp"].min().isoformat(),
        "last_timestamp": None if data.empty else last.isoformat(),
        "age_seconds": age_seconds,
        "future_rows": 0 if data.empty else int((data["timestamp"] > as_of).sum()),
        "sources": {str(k): int(v) for k, v in data["source"].value_counts(dropna=False).items()},
        "missing_optional_columns": [name for name in MARKET_FIELDS if name not in frame.columns
                                     and name not in REQUIRED_MARKET_FIELDS],
    }
    return data, report


def load_market_csv(path):
    path = Path(path)
    if not path.exists():
        raise DataQualityError(f"market data file does not exist: {path}")
    try:
        return clean_market_frame(pd.read_csv(path))
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DataQualityError(f"could not read market CSV {path}: {exc}") from exc
