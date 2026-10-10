"""Chronological, cost-aware snapshot backtest. This module has no import-time work."""
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_quality import DataQualityError, load_market_csv, require_training_source
from indicators import READY, add_indicators
from labels import FEE, SLIPPAGE, add_labels
from model_store import load_model, predict_probability
from process_lock import ProcessLock
from risk import (MAX_DAILY_LOSS, MAX_HOLD_MINUTES, MAX_LOSS_PER_TRADE, MAX_POSITIONS, MAX_TOKEN_EXPOSURE,
                  MAX_TRADE_USD, POSITION_FRACTION, STOP_LOSS, TAKE_PROFIT, TRAIL_ACTIVATE,
                  TRAILING_STOP)

ROOT = Path(__file__).resolve().parent
DATA = Path(os.getenv("BACKTEST_DATA", str(ROOT / "data" / "market.csv")))
START_BALANCE = float(os.getenv("BACKTEST_START_BALANCE", "1000"))
POSITION_FRACTION_BT = float(os.getenv("BACKTEST_POSITION_FRACTION", str(POSITION_FRACTION)))
MAX_POSITIONS_BT = int(os.getenv("BACKTEST_MAX_POSITIONS", str(MAX_POSITIONS)))
SIDE_COST = FEE + SLIPPAGE
BACKTEST_SUMMARY = ROOT / "data" / "backtest_summary.json"


def _config():
    for name, value in (("BACKTEST_START_BALANCE", START_BALANCE),
                        ("BACKTEST_POSITION_FRACTION", POSITION_FRACTION_BT),
                        ("BACKTEST_MAX_POSITIONS", MAX_POSITIONS_BT),
                        ("BACKTEST_FEE + BACKTEST_SLIPPAGE", SIDE_COST)):
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if START_BALANCE <= 0 or not 0 < POSITION_FRACTION_BT <= 1 or MAX_POSITIONS_BT < 1:
        raise ValueError("Backtest balance, position fraction, and position count must be positive and bounded.")
    if SIDE_COST < 0 or SIDE_COST >= 1:
        raise ValueError("Combined fee and slippage must be in [0, 1).")


def _net_return(gross_return, side_cost=SIDE_COST):
    # stake is the total cash allocated at entry: buy fill is spot*(1+cost),
    # then exit proceeds are spot*(1+return)*(1-cost).
    return (1 + gross_return) * (1 - side_cost) / (1 + side_cost) - 1


def _exit_for_entry(entry, market, max_hold_minutes=MAX_HOLD_MINUTES):
    """Replay observed prices after entry; fill stop/target at the observed snapshot."""
    token = entry["token_address"]
    start = entry["timestamp"]
    price = float(entry["price_usd"])
    deadline = start + pd.Timedelta(minutes=max_hold_minutes)
    rows = market.loc[(market["token_address"] == token) & (market["timestamp"] > start)].sort_values("timestamp")
    peak = price
    for _, point in rows.iterrows():
        observed = float(point["price_usd"])
        stamp = point["timestamp"]
        if observed <= 0:
            return stamp, -1.0, "pair vanished"
        peak = max(peak, observed)
        ret = observed / price - 1
        reason = None
        if ret <= -STOP_LOSS:
            reason = "stop loss"
        elif ret >= TAKE_PROFIT:
            reason = "take profit"
        elif peak / price - 1 >= TRAIL_ACTIVATE and observed <= peak * (1 - TRAILING_STOP):
            reason = "trailing stop"
        elif stamp >= deadline:
            reason = "max hold"
        if reason:
            return stamp, ret, reason

    # A total loss requires an observed zero-price row. A missing token is
    # censored data: do not use the future label to invent an exit or its time.
    return None


def simulate_portfolio_detailed(entries, market, *, horizon_minutes, initial_balance=START_BALANCE,
                                position_fraction=POSITION_FRACTION_BT, max_positions=MAX_POSITIONS_BT,
                                side_cost=SIDE_COST, max_hold_minutes=MAX_HOLD_MINUTES):
    """Simulate cash, open slots, position risk caps, token exposure and daily stop.

    Returns the closed trades plus the risk evidence the release gate inspects:
    the worst intraday equity drop, how often the daily halt latched, and the
    peak number of concurrent positions.
    """
    numeric = (horizon_minutes, max_hold_minutes, initial_balance, position_fraction, side_cost)
    if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in numeric):
        raise ValueError("Backtest simulation settings must be finite numbers.")
    if isinstance(max_positions, bool):
        raise ValueError("Backtest position count must be a positive integer.")
    try:
        max_positions = float(max_positions)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Backtest position count must be a positive integer.") from None
    if not math.isfinite(max_positions) or max_positions < 1 or not max_positions.is_integer():
        raise ValueError("Backtest position count must be a positive integer.")
    max_positions = int(max_positions)
    if (horizon_minutes <= 0 or max_hold_minutes <= 0 or initial_balance <= 0 or not 0 < position_fraction <= 1
            or side_cost < 0 or side_cost >= 1):
        raise ValueError("Backtest simulation settings are outside safe bounds.")
    entries = entries.sort_values(["timestamp", "token_address"], kind="mergesort")
    market = market.sort_values(["timestamp", "token_address"], kind="mergesort").reset_index(drop=True)
    cash = float(initial_balance)
    open_positions = []
    trades = []
    daily_start = cash
    active_day = None
    daily_halted = False
    latest_prices = {}
    market_cursor = 0
    risk = {"worst_day_loss": 0.0, "worst_day_loss_pct": 0.0, "daily_halt_events": 0,
            "max_open_positions": 0, "entries_skipped": 0}

    def equity():
        value = cash
        for pos in open_positions:
            mark = latest_prices.get(pos["token"], pos["entry_price"])
            gross = mark / pos["entry_price"] - 1
            value += pos["stake"] * (1 + _net_return(gross, side_cost))
        return value

    def mark_daily_risk():
        """Record the worst intraday drop and latch the daily halt exactly once."""
        nonlocal daily_halted
        if daily_start <= 0:
            return
        drop = equity() - daily_start
        risk["worst_day_loss"] = min(risk["worst_day_loss"], drop)
        risk["worst_day_loss_pct"] = min(risk["worst_day_loss_pct"], drop / daily_start)
        if drop <= -MAX_DAILY_LOSS * daily_start and not daily_halted:
            daily_halted = True
            risk["daily_halt_events"] += 1

    def close_due(at):
        nonlocal cash
        due = sorted((p for p in open_positions if p["exit_time"] <= at),
                     key=lambda p: (p["exit_time"], p["token"]))
        for pos in due:
            open_positions.remove(pos)
            pnl = pos["stake"] * pos["net_return"]
            cash += pos["stake"] + pnl
            trades.append({**pos, "pnl": pnl, "balance": cash})

    def advance_market(at):
        nonlocal market_cursor, daily_start, active_day, daily_halted
        while market_cursor < len(market) and market.iloc[market_cursor]["timestamp"] <= at:
            point = market.iloc[market_cursor]
            stamp = point["timestamp"]
            token = point["token_address"]
            price = float(point["price_usd"])
            latest_prices[token] = price
            close_due(stamp)
            day = stamp.date()
            if day != active_day:
                active_day = day
                daily_start = equity()
                daily_halted = False
            mark_daily_risk()
            market_cursor += 1

    for _, row in entries.iterrows():
        stamp, token = row["timestamp"], row["token_address"]
        advance_market(stamp)
        close_due(stamp)
        day = stamp.date()
        if day != active_day:
            active_day, daily_start = day, equity()
            daily_halted = False
        current_equity = equity()
        mark_daily_risk()
        if (len(open_positions) >= max_positions or any(p["token"] == token for p in open_positions)
                or cash <= 0 or daily_halted):
            risk["entries_skipped"] += 1
            continue
        stop_loss_cost = 1 - ((1 - STOP_LOSS) * (1 - side_cost) / (1 + side_cost))
        if stop_loss_cost <= 0 or not math.isfinite(stop_loss_cost):
            continue
        stake = min(cash, current_equity * position_fraction, MAX_TRADE_USD,
                    current_equity * MAX_LOSS_PER_TRADE / stop_loss_cost,
                    current_equity * MAX_TOKEN_EXPOSURE)
        if stake < 1.0:
            risk["entries_skipped"] += 1
            continue
        outcome = _exit_for_entry(row, market, max_hold_minutes)
        if outcome is None:
            risk["entries_skipped"] += 1
            continue
        exit_time, gross_return, reason = outcome
        net = _net_return(gross_return, side_cost)
        open_positions.append({"entry_time": stamp, "exit_time": exit_time, "token": token,
                               "entry_price": float(row["price_usd"]), "stake": stake,
                               "gross_return": gross_return, "net_return": net,
                               "exit_reason": reason})
        risk["max_open_positions"] = max(risk["max_open_positions"], len(open_positions))
        cash -= stake
    close_due(pd.Timestamp.max.tz_localize("UTC"))
    risk["final_balance"] = float(cash)
    risk["open_positions_at_end"] = int(len(open_positions))
    return {"trades": trades, "risk": risk}


def simulate_portfolio(entries, market, **kwargs):
    """Closed trades only; see simulate_portfolio_detailed for the risk evidence."""
    return simulate_portfolio_detailed(entries, market, **kwargs)["trades"]


def run_backtest(bundle=None, market_path=DATA):
    _config()
    bundle = load_model() if bundle is None else bundle
    if bundle is None:
        raise ValueError("No trusted model is saved. Run train.py first.")
    market, quality = load_market_csv(market_path)
    if quality["future_rows"]:
        raise ValueError(f"Market data contains {quality['future_rows']} future-dated rows.")
    require_training_source(quality, allow_synthetic=os.getenv("ALLOW_SYNTHETIC_DATA") == "1")
    horizon = float(bundle["horizon_minutes"])
    enriched = add_indicators(market)
    labeled = add_labels(enriched, horizon, bundle["target_return"])
    test_start = pd.to_datetime(bundle["test_start"], utc=True)
    ready = labeled.loc[(labeled["timestamp"] >= test_start) & (labeled["price_usd"] > 0)]
    ready = ready.dropna(subset=READY + ["target", "future_return"]).copy()
    if ready.empty:
        raise ValueError("No matured, usable rows in the saved out-of-time test period.")
    ready["prob"] = predict_probability(bundle, ready[bundle["features"]])
    selected = ready.loc[ready["prob"] >= float(bundle["threshold"])]
    simulation = simulate_portfolio_detailed(selected, market, horizon_minutes=horizon)
    baseline = [_net_return(float(v)) for v in ready["future_return"]]
    return {"bundle": bundle, "market": market, "quality": quality, "test": ready, "selected": selected,
            "trades": simulation["trades"], "risk": simulation["risk"],
            "baseline_returns": baseline, "test_start": test_start}


def summarize(result):
    """Build the machine-readable cost-aware evidence the release gate reads."""
    test, trades = result["test"], result["trades"]
    baseline = result["baseline_returns"]
    frame = pd.DataFrame(trades)
    baseline_mean = (sum(baseline) / len(baseline)) if baseline else None
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_start": pd.Timestamp(result["test_start"]).isoformat(),
        "test_end": pd.Timestamp(test["timestamp"].max()).isoformat(),
        "matured_rows": int(len(test)),
        "tokens": int(test["token_address"].nunique()),
        "sources": dict(result["quality"]["sources"]),
        "threshold": float(result["bundle"]["threshold"]),
        "horizon_minutes": float(result["bundle"]["horizon_minutes"]),
        "start_balance": float(START_BALANCE),
        "fee_per_side": float(FEE),
        "slippage_per_side": float(SLIPPAGE),
        "side_cost": float(SIDE_COST),
        "baseline_mean_net_return": None if baseline_mean is None else float(baseline_mean),
        "closed_trades": int(len(frame)),
        "risk": dict(result["risk"]),
        "risk_limits": {
            "max_daily_loss": float(MAX_DAILY_LOSS),
            "max_positions": int(MAX_POSITIONS_BT),
            "position_fraction": float(POSITION_FRACTION_BT),
            "max_trade_usd": float(MAX_TRADE_USD),
            "max_token_exposure": float(MAX_TOKEN_EXPOSURE),
            "stop_loss": float(STOP_LOSS),
            "take_profit": float(TAKE_PROFIT),
            "max_hold_minutes": float(MAX_HOLD_MINUTES),
        },
        "win_rate": None, "mean_net_return": None, "median_net_return": None,
        "worst_net_return": None, "expectancy_usd": None, "total_pnl": None,
        "final_balance": float(START_BALANCE), "return_pct": 0.0, "max_drawdown": 0.0,
        "tokens_traded": 0, "exit_reasons": {}, "cost_sensitivity": [],
    }
    if frame.empty:
        return summary
    frame = frame.sort_values(["exit_time", "token"], kind="mergesort")
    equity = pd.Series([START_BALANCE] + frame["balance"].tolist(), dtype=float)
    total_pnl = float(frame["pnl"].sum())
    summary.update(
        win_rate=float((frame["net_return"] > 0).mean()),
        mean_net_return=float(frame["net_return"].mean()),
        median_net_return=float(frame["net_return"].median()),
        worst_net_return=float(frame["net_return"].min()),
        expectancy_usd=total_pnl / len(frame),
        total_pnl=total_pnl,
        final_balance=float(equity.iloc[-1]),
        return_pct=float(equity.iloc[-1] / START_BALANCE - 1),
        max_drawdown=float((equity / equity.cummax() - 1).min()),
        tokens_traded=int(frame["token"].nunique()),
        exit_reasons={str(name): int(count)
                      for name, count in frame["exit_reason"].value_counts().items()},
    )
    return summary


def cost_sensitivity(result, multiples=(0.5, 1.0, 1.5)):
    """Re-run the same signal and price path under lighter and heavier costs."""
    rows = []
    for multiple in multiples:
        scenario = simulate_portfolio_detailed(
            result["selected"], result["market"],
            horizon_minutes=float(result["bundle"]["horizon_minutes"]),
            side_cost=SIDE_COST * multiple)
        final = float(scenario["risk"]["final_balance"])
        rows.append({
            "label": "base" if multiple == 1.0 else f"{multiple:.1f}x costs",
            "cost_multiple": float(multiple),
            "closed_trades": int(len(scenario["trades"])),
            "final_balance": final,
            "return_pct": final / START_BALANCE - 1,
        })
    return rows


def write_summary(summary, path=BACKTEST_SUMMARY):
    """Atomically replace the backtest evidence file read by the release gate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    return path


def main():
    try:
        with ProcessLock(ROOT / "data" / "backtest.lock"):
            result = run_backtest()
    except (RuntimeError, ValueError, DataQualityError) as exc:
        raise SystemExit(f"Backtest stopped: {exc}") from exc
    summary = summarize(result)
    summary["cost_sensitivity"] = cost_sensitivity(result)
    write_summary(summary)
    baseline_text = ("n/a" if summary["baseline_mean_net_return"] is None
                     else f"{summary['baseline_mean_net_return']:.2%}")
    print(f"Test period: {summary['test_start']} -> {summary['test_end']}")
    print(f"Matured rows: {summary['matured_rows']} across {summary['tokens']} tokens")
    print(f"Sources: {summary['sources']}")
    print(f"Round-trip costs: fee {FEE:.2%} + slippage {SLIPPAGE:.2%} each side")
    print(f"Test-row baseline: mean net return {baseline_text}")
    if not summary["closed_trades"]:
        print("No positions met the saved model threshold and risk limits.")
    else:
        print(f"Trades: {summary['closed_trades']} across {summary['tokens_traded']} tokens")
        print(f"Win rate: {summary['win_rate']:.1%}; mean {summary['mean_net_return']:.2%}; "
              f"median {summary['median_net_return']:.2%}; worst {summary['worst_net_return']:.2%}")
        print(f"Final balance: {summary['final_balance']:,.2f} ({summary['return_pct']:+.1%})")
        print(f"Maximum realized drawdown: {summary['max_drawdown']:.1%}")
        print("Exit reasons:", summary["exit_reasons"])
        tr = pd.DataFrame(result["trades"]).sort_values(["exit_time", "token"], kind="mergesort")
        by_token = tr.groupby("token")["net_return"].agg(["count", "mean"]).sort_values("mean")
        print("Worst token averages:")
        print(by_token.head(5).round(4).to_string())
        if summary["closed_trades"] < 30:
            print("Caution: fewer than 30 trades; estimates are highly uncertain.")
    print("Cost sensitivity (same signal/price path, position sizing rerun):")
    for row in summary["cost_sensitivity"]:
        print(f"  {row['label']}: {row['closed_trades']} trades, final balance "
              f"{row['final_balance']:,.2f} ({row['return_pct']:+.1%})")
    print(f"Wrote {BACKTEST_SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
