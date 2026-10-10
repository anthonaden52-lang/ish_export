"""Paper-trading book + risk engine. Simulation only: nothing here sends a transaction or reads a key.

Limits (all in .env, defaults shown):
  PAPER_START_BALANCE=1000     POSITION_FRACTION=0.05      fraction of equity per trade
  MAX_TRADE_USD=50             hard cap per trade
  MAX_LOSS_PER_TRADE=0.01      max fraction of equity lost at the configured stop, including costs
  MAX_POSITIONS=3              MAX_TOKEN_EXPOSURE=0.10      max fraction of equity in one token
  STOP_LOSS=0.15               TAKE_PROFIT=0.30
  TRAIL_ACTIVATE=0.10          TRAILING_STOP=0.10          trailing stop starts after +10%, trails 10% below the peak
  MAX_HOLD_MINUTES=2 x HORIZON_MINUTES
  MAX_DAILY_LOSS=0.05          no new entries for the rest of the UTC day once equity is down 5% on the day
  EMERGENCY STOP: create the file data/STOP to block all new entries (open positions are still managed).

Fills are pessimistic: entry and exit each pay FEE + SLIPPAGE (same numbers as the backtest), and a stop that
gaps through is filled at the price actually observed on the scan, not at the stop level. A pair that vanishes
or shows price 0 is closed at a total loss.
"""
import csv
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from labels import FEE, SLIPPAGE, HORIZON_MINUTES

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "data" / "paper_state.json"
TRADES = ROOT / "data" / "paper_trades.csv"
STOP_FILE = ROOT / "data" / "STOP"


def _config_number(env, name, default, *, minimum=None, maximum=None,
                   minimum_inclusive=False, maximum_inclusive=False):
    raw = env.get(name, str(default))
    try:
        value = float(raw)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"Invalid {name}; expected a finite number.") from None
    if not math.isfinite(value):
        raise ValueError(f"Invalid {name}; expected a finite number.")
    if minimum is not None and (value < minimum if minimum_inclusive else value <= minimum):
        op = ">=" if minimum_inclusive else ">"
        raise ValueError(f"Invalid {name}; expected a value {op} {minimum}.")
    if maximum is not None and (value > maximum if maximum_inclusive else value >= maximum):
        op = "<=" if maximum_inclusive else "<"
        raise ValueError(f"Invalid {name}; expected a value {op} {maximum}.")
    return value


def _load_config(env=None, fee=FEE, slippage=SLIPPAGE):
    """Read and validate paper risk settings before the book can accept trades."""
    env = os.environ if env is None else env
    values = {
        "START_BALANCE": _config_number(env, "PAPER_START_BALANCE", 1000, minimum=0),
        "POSITION_FRACTION": _config_number(env, "POSITION_FRACTION", 0.05, minimum=0, maximum=1,
                                            maximum_inclusive=True),
        "MAX_TRADE_USD": _config_number(env, "MAX_TRADE_USD", 50, minimum=0),
        "MAX_LOSS_PER_TRADE": _config_number(env, "MAX_LOSS_PER_TRADE", 0.01, minimum=0, maximum=1,
                                              maximum_inclusive=True),
        "MAX_TOKEN_EXPOSURE": _config_number(env, "MAX_TOKEN_EXPOSURE", 0.10, minimum=0, maximum=1,
                                              maximum_inclusive=True),
        "STOP_LOSS": _config_number(env, "STOP_LOSS", 0.15, minimum=0, maximum=1),
        "TAKE_PROFIT": _config_number(env, "TAKE_PROFIT", 0.30, minimum=0),
        "TRAIL_ACTIVATE": _config_number(env, "TRAIL_ACTIVATE", 0.10, minimum=0, maximum=1,
                                          maximum_inclusive=True),
        "TRAILING_STOP": _config_number(env, "TRAILING_STOP", 0.10, minimum=0, maximum=1),
        "MAX_HOLD_MINUTES": _config_number(env, "MAX_HOLD_MINUTES", 2 * HORIZON_MINUTES, minimum=0),
        "MAX_DAILY_LOSS": _config_number(env, "MAX_DAILY_LOSS", 0.05, minimum=0, maximum=1),
    }
    raw_positions = env.get("MAX_POSITIONS", "3")
    try:
        positions = float(raw_positions)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("Invalid MAX_POSITIONS; expected a positive integer.") from None
    if not math.isfinite(positions) or positions <= 0 or not positions.is_integer():
        raise ValueError("Invalid MAX_POSITIONS; expected a positive integer.")
    values["MAX_POSITIONS"] = int(positions)

    for name, cost in (("BACKTEST_FEE", fee), ("BACKTEST_SLIPPAGE", slippage)):
        if not math.isfinite(float(cost)) or float(cost) < 0 or float(cost) >= 1:
            raise ValueError(f"Invalid {name}; expected a finite fraction in [0, 1).")
    side_cost = float(fee) + float(slippage)
    if not math.isfinite(side_cost) or side_cost >= 1:
        raise ValueError("Combined fee and slippage must be a finite fraction below 1.")
    values["SIDE_COST"] = side_cost
    return values


globals().update(_load_config())


def _number(value, name, *, minimum=None, minimum_inclusive=False):
    if isinstance(value, bool):
        raise ValueError(f"Invalid {name}; expected a finite number.")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"Invalid {name}; expected a finite number.") from None
    if not math.isfinite(result):
        raise ValueError(f"Invalid {name}; expected a finite number.")
    if minimum is not None and (result < minimum if minimum_inclusive else result <= minimum):
        op = ">=" if minimum_inclusive else ">"
        raise ValueError(f"Invalid {name}; expected a value {op} {minimum}.")
    return result


def _state_number(value, name, *, minimum=None, minimum_inclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {name}; expected a JSON number.")
    return _number(value, name, minimum=minimum, minimum_inclusive=minimum_inclusive)


def _validate_state(state):
    """Reject corrupt paper state instead of silently resetting a trading book."""
    if not isinstance(state, dict):
        raise ValueError("Paper state must be a JSON object.")
    required = {"cash", "start_balance", "positions", "day", "day_start_equity", "last_prices"}
    if not required.issubset(state):
        raise ValueError("Paper state is missing required fields; refusing to trade.")
    if not isinstance(state.get("daily_halted", False), bool):
        raise ValueError("Paper state has an invalid daily_halted flag; refusing to trade.")
    _state_number(state["cash"], "paper state cash", minimum=0, minimum_inclusive=True)
    _state_number(state["start_balance"], "paper state start_balance", minimum=0)
    _state_number(state["day_start_equity"], "paper state day_start_equity", minimum=0, minimum_inclusive=True)
    try:
        if not isinstance(state["day"], str):
            raise ValueError
        if date.fromisoformat(state["day"]).isoformat() != state["day"]:
            raise ValueError
    except ValueError:
        raise ValueError("Paper state has an invalid UTC day; refusing to trade.") from None

    positions = state["positions"]
    prices = state["last_prices"]
    if not isinstance(positions, dict) or not isinstance(prices, dict):
        raise ValueError("Paper state positions and last_prices must be objects.")
    for token, price in prices.items():
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Paper state contains an invalid token key.")
        _state_number(price, "paper state last price", minimum=0, minimum_inclusive=True)
    projected_equity = float(state["cash"])
    for token, position in positions.items():
        if not isinstance(token, str) or not token.strip() or not isinstance(position, dict):
            raise ValueError("Paper state contains an invalid position.")
        for field in ("entry_price", "units", "stake", "peak"):
            _state_number(position.get(field), f"paper position {field}", minimum=0)
        notional = position["units"] * max(position["entry_price"], position["peak"], prices.get(token, 0))
        if not math.isfinite(notional):
            raise ValueError("Paper state position value is outside the supported numeric range.")
        projected_equity += notional
        if not math.isfinite(projected_equity):
            raise ValueError("Paper state total value is outside the supported numeric range.")
        score = _state_number(position.get("score"), "paper position score", minimum=0, minimum_inclusive=True)
        if score > 100:
            raise ValueError("Paper state position score must be between 0 and 100.")
        probability = position.get("model_prob")
        if probability is not None:
            probability = _state_number(probability, "paper position model_prob", minimum=0, minimum_inclusive=True)
            if probability > 1:
                raise ValueError("Paper state model probability must be between 0 and 1.")
        if not isinstance(position.get("entry_time"), str):
            raise ValueError("Paper state position has an invalid entry_time.")
        try:
            entry_time = datetime.fromisoformat(position["entry_time"])
        except ValueError:
            raise ValueError("Paper state position has an invalid entry_time.") from None
        if entry_time.tzinfo is None or entry_time.utcoffset() is None:
            raise ValueError("Paper state position entry_time must include a timezone.")
        for field in ("symbol", "score_source", "risk"):
            if not isinstance(position.get(field), str):
                raise ValueError(f"Paper state position has an invalid {field}.")
    return state

TRADE_FIELDS = ["entry_time", "exit_time", "token", "symbol", "score", "score_source", "model_prob",
                "risk", "entry_price", "exit_price", "stake", "pnl", "net_return", "held_minutes", "exit_reason"]


def _now():
    return datetime.now(timezone.utc)


class PaperBook:
    def __init__(self, path=STATE):
        self.path = path
        self.s = {"cash": START_BALANCE, "start_balance": START_BALANCE, "positions": {},
                  "day": _now().strftime("%Y-%m-%d"), "day_start_equity": START_BALANCE,
                  "last_prices": {}, "daily_halted": False}
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                saved.setdefault("daily_halted", False)  # legacy state is safe to resume
                _validate_state(saved)
                self.s.update(saved)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Cannot safely load paper state from {path}: {exc}") from exc

    # ---- persistence ----
    def save(self):
        _validate_state(self.s)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps(self.s, allow_nan=False))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.path)

    # ---- accounting ----
    def equity(self):
        total = self.s["cash"]
        for tok, p in self.s["positions"].items():
            price = self.s["last_prices"].get(tok, p["entry_price"])
            total += p["units"] * price * (1 - SIDE_COST)  # value if sold now, net of exit cost
        return total

    def daily_pnl(self):
        return self.equity() - self.s["day_start_equity"]

    def _roll_day(self):
        today = _now().strftime("%Y-%m-%d")
        if today != self.s["day"]:
            self.s["day"] = today
            self.s["day_start_equity"] = self.equity()
            self.s["daily_halted"] = False

    def _daily_loss_hit(self):
        start = self.s["day_start_equity"]
        if start > 0 and self.daily_pnl() <= -MAX_DAILY_LOSS * start:
            self.s["daily_halted"] = True
        return self.s["daily_halted"]

    # ---- entry rules ----
    def entry_block_reason(self, token):
        """None if a new entry is allowed, else why not."""
        if not isinstance(token, str) or not token.strip():
            return "invalid token address"
        self._roll_day()
        if STOP_FILE.exists():
            return "EMERGENCY STOP file present (data/STOP)"
        if token in self.s["positions"]:
            return "already holding this token"
        if len(self.s["positions"]) >= MAX_POSITIONS:
            return f"max open positions ({MAX_POSITIONS}) reached"
        eq = self.equity()
        if self._daily_loss_hit():
            return f"daily loss limit hit ({MAX_DAILY_LOSS:.0%}); no new entries until the next UTC day"
        if self.stake(eq) < 1.0:
            return "position size too small / not enough cash"
        return None

    def stake(self, equity):
        try:
            equity = _number(equity, "equity", minimum=0)
        except ValueError:
            return 0.0
        stop_loss_fraction = 1 - ((1 - STOP_LOSS) * (1 - SIDE_COST) / (1 + SIDE_COST))
        stake = min(equity * POSITION_FRACTION,
                    MAX_TRADE_USD,
                    equity * MAX_LOSS_PER_TRADE / stop_loss_fraction,
                    equity * MAX_TOKEN_EXPOSURE,
                    self.s["cash"])
        return max(0.0, stake)

    def enter(self, row, decision):
        try:
            token = row.get("token_address")
            if not isinstance(token, str) or not token.strip():
                return None, "invalid token address"
            price = _number(row.get("price_usd"), "entry price", minimum=0)
            score = _number(decision.get("score"), "decision score", minimum=0, minimum_inclusive=True)
            if score > 100:
                return None, "invalid decision score"
            probability = decision.get("prob")
            if probability is not None:
                probability = _number(probability, "model probability", minimum=0, minimum_inclusive=True)
                if probability > 1:
                    return None, "invalid model probability"
            source = decision.get("source")
            decision_risk = decision.get("risk")
            if not isinstance(source, str) or not source.strip() or not isinstance(decision_risk, str):
                return None, "invalid decision metadata"
        except (AttributeError, TypeError, ValueError) as exc:
            return None, f"invalid trade input: {exc}"
        why = self.entry_block_reason(token)
        if why:
            return None, why
        stake = self.stake(self.equity())
        fill = price * (1 + SIDE_COST)
        units = stake / fill if math.isfinite(fill) and fill > 0 else 0.0
        if not math.isfinite(units) or units <= 0:
            return None, "invalid entry price or calculated position size"
        pos = {"symbol": str(row.get("symbol", "")), "entry_time": _now().isoformat(), "entry_price": price,
               "units": units, "stake": stake, "peak": price, "score": score,
               "score_source": source, "model_prob": probability, "risk": decision_risk}
        self.s["cash"] -= stake
        self.s["positions"][token] = pos
        self.s["last_prices"][token] = price
        self.save()
        return pos, None

    # ---- exits ----
    def update(self, prices):
        """prices: {token: latest price}. Applies stop/target/trailing/time exits. Returns closed trades."""
        if not hasattr(prices, "items"):
            raise ValueError("Prices must be a token-to-price mapping.")
        normalized_prices = {}
        for token, value in prices.items():
            if not isinstance(token, str) or not token.strip():
                raise ValueError("Price update contains an invalid token address.")
            price = _number(value, "market price", minimum=0, minimum_inclusive=True)
            position = self.s["positions"].get(token)
            if position and not math.isfinite(position["units"] * price):
                raise ValueError("Market price would produce an unsupported position value.")
            normalized_prices[token] = price
        prices = normalized_prices
        self._roll_day()
        now = _now()
        closed = []
        for tok, price in prices.items():
            if tok in self.s["positions"] or tok in self.s["last_prices"]:
                self.s["last_prices"][tok] = price
        for tok in list(self.s["positions"]):
            p = self.s["positions"][tok]
            price = prices.get(tok)
            if price is None:
                continue  # not scanned this round; tracking keeps writing rows, so this is rare
            p["peak"] = max(p["peak"], price)
            held = (now - datetime.fromisoformat(p["entry_time"])).total_seconds() / 60
            ret = price / p["entry_price"] - 1
            reason = None
            if price <= 0:
                reason = "pair vanished / price 0 (total loss)"
            elif ret <= -STOP_LOSS:
                reason = "stop loss"
            elif ret >= TAKE_PROFIT:
                reason = "take profit"
            elif p["peak"] / p["entry_price"] - 1 >= TRAIL_ACTIVATE and price <= p["peak"] * (1 - TRAILING_STOP):
                reason = "trailing stop"
            elif held >= MAX_HOLD_MINUTES:
                reason = "max hold time"
            if reason:
                closed.append(self._close(tok, price, reason, now, held))
        self._daily_loss_hit()
        if closed or prices:
            self.save()
        return closed

    def _close(self, tok, price, reason, now, held):
        p = self.s["positions"].pop(tok)
        proceeds = p["units"] * max(price, 0.0) * (1 - SIDE_COST)
        pnl = proceeds - p["stake"]
        self.s["cash"] += proceeds
        row = {"entry_time": p["entry_time"], "exit_time": now.isoformat(), "token": tok, "symbol": p["symbol"],
               "score": p["score"], "score_source": p["score_source"], "model_prob": p["model_prob"],
               "risk": p["risk"], "entry_price": p["entry_price"], "exit_price": price, "stake": round(p["stake"], 4),
               "pnl": round(pnl, 4), "net_return": round(pnl / p["stake"], 5), "held_minutes": round(held, 1),
               "exit_reason": reason}
        TRADES.parent.mkdir(exist_ok=True)
        new = not TRADES.exists()
        with TRADES.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        return row

    def summary(self):
        eq = self.equity()
        return (f"equity ${eq:,.2f} ({eq / self.s['start_balance'] - 1:+.1%} since start) | cash ${self.s['cash']:,.2f} | "
                f"open {len(self.s['positions'])}/{MAX_POSITIONS} | today {self.daily_pnl():+,.2f}")
