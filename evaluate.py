"""Performance tracking: did the scores actually predict anything?

Reads (whatever exists):
  data/paper_trades.csv   closed paper trades
  data/predictions.csv    every BUY/WATCH call, traded or not
  data/market.csv         price history, used to look up what happened after each call
  models/model.joblib     for permutation importance (only if you trained one)

Usage: python evaluate.py
Small samples are labelled as such; do not trust anything under ~30 trades / ~100 predictions.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data_quality import DataQualityError, load_market_csv, require_training_source
from indicators import FEATURES, READY, add_indicators
from labels import HORIZON_MINUTES, TARGET_RETURN, add_labels

ROOT = Path(__file__).resolve().parent
TRADES, PREDS, MARKET = ROOT / "data" / "paper_trades.csv", ROOT / "data" / "predictions.csv", ROOT / "data" / "market.csv"


def section(t):
    print(f"\n=== {t} ===")


def paper_trades():
    section("PAPER TRADES")
    if not TRADES.exists():
        print("No closed paper trades yet.")
        return
    t = pd.read_csv(TRADES)
    print(f"{len(t)} trades | win rate {(t['net_return'] > 0).mean():.1%} | avg {t['net_return'].mean():+.2%} | "
          f"median {t['net_return'].median():+.2%} | worst {t['net_return'].min():+.2%} | total PnL {t['pnl'].sum():+,.2f}")
    print(f"avg held {t['held_minutes'].mean():.0f} min")
    print("\nBy score source:")
    print(t.groupby("score_source")["net_return"].agg(["count", "mean", lambda s: (s > 0).mean()])
          .rename(columns={"<lambda_0>": "win_rate"}).round(4).to_string())
    print("\nBy exit reason:")
    print(t.groupby("exit_reason")["net_return"].agg(["count", "mean"]).round(4).to_string())
    if len(t) < 30:
        print("\nWarning: fewer than 30 trades. Not statistically meaningful.")


def forward_table():
    if not MARKET.exists():
        return None
    try:
        m, quality = load_market_csv(MARKET)
    except DataQualityError as exc:
        print(f"Market data rejected: {exc}")
        return None
    if quality["future_rows"]:
        print(f"Market data rejected: {quality['future_rows']} future-dated rows.")
        return None
    try:
        require_training_source(quality, allow_synthetic=os.getenv("ALLOW_SYNTHETIC_DATA") == "1")
    except DataQualityError as exc:
        print(f"Market data rejected: {exc}.")
        return None
    print(f"Data rows: {quality['valid_rows']}; sources: {quality['sources']}; "
          f"dropped invalid: {quality['invalid_rows_dropped']}; duplicates: {quality['duplicate_rows_dropped']}")
    return m.sort_values(["token_address", "timestamp"])


def predictions(m):
    section("PREDICTION ACCURACY (all calls, traded or not)")
    if not PREDS.exists() or m is None:
        print("Need data/predictions.csv (run paper_trader.py) and data/market.csv (run collector.py).")
        return
    p = pd.read_csv(PREDS)
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, format="ISO8601", errors="coerce")
    p["price_usd"] = pd.to_numeric(p["price_usd"], errors="coerce")
    p = p.dropna(subset=["timestamp", "token", "price_usd"])
    p = p.loc[np.isfinite(p["price_usd"]) & (p["price_usd"] > 0)]
    right = (m[["timestamp", "token_address", "price_usd"]]
             .rename(columns={"timestamp": "fts", "token_address": "token", "price_usd": "fprice"})
             .sort_values("fts"))
    p["target_ts"] = p["timestamp"] + pd.Timedelta(minutes=HORIZON_MINUTES)
    p = pd.merge_asof(p.sort_values("target_ts"), right, left_on="target_ts", right_on="fts", by="token",
                      direction="forward", tolerance=pd.Timedelta(minutes=max(1.0, HORIZON_MINUTES * 0.2)))
    p["fwd"] = p["fprice"] / p["price_usd"] - 1
    p.loc[p["fprice"] == 0, "fwd"] = -1.0
    done = p.dropna(subset=["fwd"]).copy()
    print(f"{len(p)} calls logged, {len(done)} outcomes observed; "
          f"{len(p) - len(done)} remain censored because no matching future snapshot was recorded.")
    if done.empty:
        return
    done["hit"] = done["fwd"] >= TARGET_RETURN
    base = done["hit"].mean()
    print(f"Base rate (hit +{TARGET_RETURN:.0%} within {HORIZON_MINUTES} min), across logged calls: {base:.1%}")
    print("\nBy verdict:")
    print(done.groupby("verdict")["fwd"].agg(["count", "mean", "median"]).round(4).to_string())
    print(done.groupby("verdict")["hit"].mean().round(3).rename("hit_rate").to_string())
    done["bucket"] = pd.cut(done["score"], [0, 60, 70, 80, 90, 100], include_lowest=True)
    print("\nBy score bucket and source:")
    print(done.groupby(["score_source", "bucket"], observed=True)
          .agg(n=("fwd", "size"), avg_fwd=("fwd", "mean"), hit_rate=("hit", "mean")).round(4).to_string())
    for src, g in done.groupby("score_source"):
        if g["hit"].nunique() > 1 and len(g) >= 30:
            from sklearn.metrics import roc_auc_score
            print(f"\n{src} AUC vs hit/miss: {roc_auc_score(g['hit'], g['score']):.3f}  (0.5 = no better than chance)")
        else:
            print(f"\n{src}: too few calls or only one outcome class for an AUC.")


def indicator_usefulness(m):
    section("WHICH INDICATORS CARRY SIGNAL (rank correlation with future return)")
    if m is None:
        print("No data/market.csv.")
        return
    df = add_labels(add_indicators(m.copy()), HORIZON_MINUTES, TARGET_RETURN)
    df = df[df["price_usd"] > 0].dropna(subset=READY + ["future_return"])
    if len(df) < 200:
        print(f"Only {len(df)} usable rows; collect more.")
        return
    ic = {f: df[f].corr(df["future_return"], method="spearman") for f in FEATURES}
    s = pd.Series(ic).dropna().sort_values(key=abs, ascending=False)
    print("(IC near 0 = no relationship. |IC| > ~0.05 on lots of data is interesting; it is not proof.)")
    print(s.round(4).to_string())
    from model_store import MODEL_PATH, load_model
    if MODEL_PATH.exists():
        from sklearn.inspection import permutation_importance
        b = load_model()
        test = df[df["timestamp"] >= b["test_start"]]
        if len(test) >= 200 and test["target"].nunique() > 1 and "model" in b:
            r = permutation_importance(b["model"], test[b["features"]], test["target"], n_repeats=5,
                                       random_state=0, scoring="roc_auc")
            imp = pd.Series(r.importances_mean, index=b["features"]).sort_values(ascending=False)
            print("\nModel permutation importance on the held-out test period (drop in AUC when shuffled):")
            print(imp.round(4).to_string())


if __name__ == "__main__":
    paper_trades()
    m = forward_table()
    predictions(m)
    indicator_usefulness(m)
