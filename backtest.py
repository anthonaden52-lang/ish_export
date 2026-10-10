"""Chronological, cost-aware snapshot backtest. This module has no import-time work."""
import math
import os
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


def simulate_portfolio(entries, market, *, horizon_minutes, initial_balance=START_BALANCE,
                       position_fraction=POSITION_FRACTION_BT, max_positions=MAX_POSITIONS_BT,
                       side_cost=SIDE_COST, max_hold_minutes=MAX_HOLD_MINUTES):
    """Simulate cash, open slots, position risk caps, token exposure and daily stop."""
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

    def equity():
        value = cash
        for pos in open_positions:
            mark = latest_prices.get(pos["token"], pos["entry_price"])
            gross = mark / pos["entry_price"] - 1
            value += pos["stake"] * (1 + _net_return(gross, side_cost))
        return value

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
            if daily_start > 0 and equity() - daily_start <= -MAX_DAILY_LOSS * daily_start:
                daily_halted = True
            market_cursor += 1

    for _, row in entries.iterrows():
        stamp, token = row["timestamp"], row["token_address"]
        advance_market(stamp)
        close_due(stamp)
        day = stamp.date()
        if day != active_day:
            active_day, daily_start = day, equity()
            daily_halted = False
        if daily_start > 0 and equity() - daily_start <= -MAX_DAILY_LOSS * daily_start:
            daily_halted = True
        if (len(open_positions) >= max_positions or any(p["token"] == token for p in open_positions)
                or cash <= 0 or daily_halted):
            continue
        current_equity = equity()
        if current_equity - daily_start <= -MAX_DAILY_LOSS * daily_start:
            daily_halted = True
            continue
        stop_loss_cost = 1 - ((1 - STOP_LOSS) * (1 - side_cost) / (1 + side_cost))
        if stop_loss_cost <= 0 or not math.isfinite(stop_loss_cost):
            continue
        stake = min(cash, current_equity * position_fraction, MAX_TRADE_USD,
                    current_equity * MAX_LOSS_PER_TRADE / stop_loss_cost,
                    current_equity * MAX_TOKEN_EXPOSURE)
        if stake < 1.0:
            continue
        outcome = _exit_for_entry(row, market, max_hold_minutes)
        if outcome is None:
            continue
        exit_time, gross_return, reason = outcome
        net = _net_return(gross_return, side_cost)
        open_positions.append({"entry_time": stamp, "exit_time": exit_time, "token": token,
                               "stake": stake, "gross_return": gross_return, "net_return": net,
                               "exit_reason": reason})
        cash -= stake
    close_due(pd.Timestamp.max.tz_localize("UTC"))
    return trades


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
    trades = simulate_portfolio(selected, market, horizon_minutes=horizon)
    baseline = [_net_return(float(v)) for v in ready["future_return"]]
    return {"bundle": bundle, "market": market, "quality": quality, "test": ready, "selected": selected,
            "trades": trades, "baseline_returns": baseline, "test_start": test_start}


def main():
    try:
        with ProcessLock(ROOT / "data" / "backtest.lock"):
            result = run_backtest()
    except (RuntimeError, ValueError, DataQualityError) as exc:
        raise SystemExit(f"Backtest stopped: {exc}") from exc
    test, trades = result["test"], result["trades"]
    print(f"Test period: {test['timestamp'].min()} -> {test['timestamp'].max()}")
    print(f"Matured rows: {len(test)} across {test['token_address'].nunique()} tokens")
    print(f"Sources: {result['quality']['sources']}")
    print(f"Round-trip costs: fee {FEE:.2%} + slippage {SLIPPAGE:.2%} each side")
    print(f"Test-row baseline: mean net return {sum(result['baseline_returns']) / len(test):.2%}")
    if not trades:
        print("No positions met the saved model threshold and risk limits.")
        return 0
    tr = pd.DataFrame(trades).sort_values(["exit_time", "token"], kind="mergesort")
    equity = pd.Series([START_BALANCE] + tr["balance"].tolist())
    drawdown = equity / equity.cummax() - 1
    print(f"Trades: {len(tr)} across {tr['token'].nunique()} tokens")
    print(f"Win rate: {(tr['net_return'] > 0).mean():.1%}; mean {tr['net_return'].mean():.2%}; "
          f"median {tr['net_return'].median():.2%}; worst {tr['net_return'].min():.2%}")
    print(f"Final balance: {equity.iloc[-1]:,.2f} ({equity.iloc[-1] / START_BALANCE - 1:+.1%})")
    print(f"Maximum realized drawdown: {drawdown.min():.1%}")
    print("Exit reasons:", tr["exit_reason"].value_counts().to_dict())
    print("Cost sensitivity (same signal/price path, position sizing rerun):")
    for multiple in (0.5, 1.0, 1.5):
        scenario = simulate_portfolio(result["selected"], result["market"],
                                      horizon_minutes=float(result["bundle"]["horizon_minutes"]),
                                      side_cost=SIDE_COST * multiple)
        final = scenario[-1]["balance"] if scenario else START_BALANCE
        label = "base" if multiple == 1.0 else f"{multiple:.1f}x costs"
        print(f"  {label}: {len(scenario)} trades, final balance {final:,.2f} "
              f"({final / START_BALANCE - 1:+.1%})")
    by_token = tr.groupby("token")["net_return"].agg(["count", "mean"]).sort_values("mean")
    print("Worst token averages:")
    print(by_token.head(5).round(4).to_string())
    if len(tr) < 30:
        print("Caution: fewer than 30 trades; estimates are highly uncertain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
