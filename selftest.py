"""Offline self-test: no network, no wallet, runs in a temp folder (your data/ is never touched).

    python selftest.py            # offline unit checks; no API, wallet, or saved model access
    python selftest.py --pipeline # also: synthetic data -> train -> backtest -> evaluate

Passing proves the plumbing and the rules behave as documented. It says NOTHING about profitability.
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
TMP = tempfile.mkdtemp(prefix="memeai_selftest_")
os.chdir(TMP)
for _name, _value in {
    "QUOTE_CHECK": "0", "PAPER_START_BALANCE": "1000", "POSITION_FRACTION": "0.05",
    "MAX_TRADE_USD": "50", "MAX_LOSS_PER_TRADE": "0.01", "MAX_POSITIONS": "3",
    "MAX_TOKEN_EXPOSURE": "0.10", "STOP_LOSS": "0.15", "TAKE_PROFIT": "0.30",
    "TRAIL_ACTIVATE": "0.10", "TRAILING_STOP": "0.10", "MAX_HOLD_MINUTES": "30",
    "MAX_DAILY_LOSS": "0.05", "BACKTEST_FEE": "0.005", "BACKTEST_SLIPPAGE": "0.02",
    "HORIZON_MINUTES": "15", "TARGET_RETURN": "0.10", "MODEL_THRESHOLD": "0.70",
    "MIN_LIQUIDITY_USD": "25000", "MIN_VOLUME_24H_USD": "50000", "MIN_TXNS_24H": "100",
    "MAX_TOKENS_PER_SCAN": "25", "MAX_LOOKUPS_PER_SCAN": "60", "DEXSCREENER_TIMEOUT_SECONDS": "20",
    "DEXSCREENER_MAX_RETRIES": "4", "TRACK_MINUTES": "105", "GAP_MINUTES": "5",
    "MAX_TOP1_PCT": "20", "MAX_TOP10_PCT": "50", "MIN_AGE_MINUTES": "30",
    "MAX_LIQ_DROP_15": "0.20", "SAFETY_CACHE_MIN": "30",
    "BACKTEST_START_BALANCE": "1000", "BACKTEST_MAX_POSITIONS": "3",
    "BACKTEST_POSITION_FRACTION": "0.05", "ALLOW_SYNTHETIC_DATA": "0",
}.items():
    os.environ[_name] = _value


def cleanup_temp():
    try:
        os.chdir(PROJECT)
    except OSError:
        pass
    shutil.rmtree(TMP, ignore_errors=True)


atexit.register(cleanup_temp)

import numpy as np
import pandas as pd

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok    {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


def synth_market(n_tokens=40, minutes=420, seed=1, with_rugs=True):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for k in range(n_tokens):
        price, liq, vol1h = 0.001 * (1 + k), rng.uniform(40_000, 400_000), rng.uniform(5_000, 80_000)
        drift = rng.normal(0, 0.0004)
        rug_at = rng.integers(120, minutes - 60) if with_rugs and k % 8 == 0 else None
        for m in range(minutes):
            if rug_at is not None and m >= rug_at:
                price, liq = price * 0.2, liq * 0.1
                if m > rug_at + 10:
                    break
            price = max(1e-9, price * (1 + drift + rng.normal(0, 0.012)))
            v5 = vol1h / 12 * rng.lognormal(0, 0.5)
            buys, sells = rng.integers(5, 60), rng.integers(5, 60)
            rows.append({"timestamp": (t0 + pd.Timedelta(minutes=m)).isoformat(), "token_address": f"TOK{k:03d}",
                         "symbol": f"T{k}", "price_usd": price, "liquidity_usd": liq, "volume_24h": vol1h * 20,
                         "volume_5m": v5, "buys_5m": buys, "sells_5m": sells, "txns_24h": 3000,
                         "price_change_5m": 0, "price_change_1h": 0, "pair_age_minutes": 200 + m,
                         "pair_address": f"PAIR{k:03d}", "dex": "raydium", "url": "",
                         "source": "synthetic_test", "discovery_sources": "selftest",
                         "market_cap": price * 1e9, "fdv": price * 1e9, "volume_1h": vol1h,
                         "buys_1h": buys * 12, "sells_1h": sells * 12})
    return pd.DataFrame(rows)


def test_risk():
    print("\nRisk engine")
    import risk
    risk.STATE = Path("data/paper_state.json")
    risk.TRADES = Path("data/paper_trades.csv")
    risk.STOP_FILE = Path("data/STOP")
    d = {"score": 85, "source": "ANALYTICAL", "prob": None, "risk": "Medium"}
    row = lambda tok, p=1.0: {"token_address": tok, "symbol": tok, "price_usd": p}

    b = risk.PaperBook(Path("data/s1.json"))
    pos, why = b.enter(row("A"), d)
    check("opens a position", pos is not None)
    check("stake respects every cap", pos["stake"] <= min(risk.MAX_TRADE_USD, 1000 * risk.MAX_TOKEN_EXPOSURE) + 1e-9)
    cost_adjusted_stop_loss = 1 - ((1 - risk.STOP_LOSS) * (1 - risk.SIDE_COST) / (1 + risk.SIDE_COST))
    risk_budget_stake = 1000 * risk.MAX_LOSS_PER_TRADE / cost_adjusted_stop_loss
    check("stake cap includes entry and exit costs", pos["stake"] <= risk_budget_stake + 1e-9)
    check("no duplicate position in same token", b.enter(row("A"), d)[0] is None)
    closed = b.update({"A": 1.0 * (1 - risk.STOP_LOSS)})
    check("stop loss closes", len(closed) == 1 and closed[0]["exit_reason"] == "stop loss" and closed[0]["net_return"] < 0)
    check("exact stop loss stays within the configured dollar risk budget",
          abs(closed[0]["pnl"]) <= 1000 * risk.MAX_LOSS_PER_TRADE + 1e-8)

    b = risk.PaperBook(Path("data/s2.json"))
    b.enter(row("B"), d)
    c = b.update({"B": 1.0 + risk.TAKE_PROFIT + 0.01})
    check("take profit closes in profit", len(c) == 1 and c[0]["exit_reason"] == "take profit" and c[0]["net_return"] > 0)

    b = risk.PaperBook(Path("data/s3.json"))
    b.enter(row("C"), d)
    b.update({"C": 1.0 + risk.TRAIL_ACTIVATE + 0.05})            # arms the trail, no exit yet
    peak = 1.0 + risk.TRAIL_ACTIVATE + 0.05
    c = b.update({"C": peak * (1 - risk.TRAILING_STOP) * 0.99})
    check("trailing stop closes after a peak", len(c) == 1 and c[0]["exit_reason"] == "trailing stop")

    b = risk.PaperBook(Path("data/s4.json"))
    b.enter(row("D"), d)
    c = b.update({"D": 0.0})
    check("vanished pair = total loss", len(c) == 1 and abs(c[0]["net_return"] + 1) < 1e-6)

    b = risk.PaperBook(Path("data/s5.json"))
    for t in ("E1", "E2", "E3"):
        b.enter(row(t), d)
    check("max open positions enforced", b.enter(row("E4"), d)[0] is None)

    b = risk.PaperBook(Path("data/s6.json"))
    b.s["day_start_equity"] = b.equity() / (1 - risk.MAX_DAILY_LOSS) * 1.01   # simulate being down > limit today
    check("daily loss limit blocks entries", "daily loss" in (b.entry_block_reason("F") or ""))
    b.s["day_start_equity"] = b.equity()  # simulate a recovery after the intraday limit was hit
    b.save()
    b = risk.PaperBook(Path("data/s6.json"))
    check("daily loss halt stays latched after recovery and restart",
          "daily loss" in (b.entry_block_reason("F") or ""))

    b = risk.PaperBook(Path("data/s7.json"))
    Path("data").mkdir(exist_ok=True)
    Path("data/STOP").write_text("x")
    check("emergency STOP file blocks entries", "EMERGENCY" in (b.entry_block_reason("G") or ""))
    Path("data/STOP").unlink()

    b2 = risk.PaperBook(Path("data/s5.json"))
    check("positions persist across restart", len(b2.s["positions"]) == 3)

    print("\nRisk validation")
    valid_config = risk._load_config({}, fee=risk.FEE, slippage=risk.SLIPPAGE)
    check("default risk configuration is accepted", valid_config["MAX_POSITIONS"] == 3)
    invalid_configs = [
        ({"PAPER_START_BALANCE": "nan"}, "non-finite balance rejected"),
        ({"POSITION_FRACTION": "1.01"}, "fraction above 100% rejected"),
        ({"MAX_LOSS_PER_TRADE": "0"}, "zero per-trade risk rejected"),
        ({"MAX_POSITIONS": "1.5"}, "fractional position count rejected"),
        ({"STOP_LOSS": "1"}, "stop loss at or above 100% rejected"),
        ({"MAX_DAILY_LOSS": "-0.1"}, "negative daily loss limit rejected"),
    ]
    for config, label in invalid_configs:
        try:
            risk._load_config(config, fee=risk.FEE, slippage=risk.SLIPPAGE)
            rejected = False
        except ValueError:
            rejected = True
        check(label, rejected)
    try:
        risk._load_config({}, fee=-0.01, slippage=risk.SLIPPAGE)
        bad_cost_rejected = False
    except ValueError:
        bad_cost_rejected = True
    check("negative fee configuration rejected", bad_cost_rejected)

    b = risk.PaperBook(Path("data/invalid_inputs.json"))
    for price in (0, -1, np.nan, np.inf, 1e-320, "not-a-price"):
        pos, why = b.enter(row("BAD", price), d)
        check(f"invalid entry price {price!r} rejected", pos is None and "invalid" in why)
    bad_decision = dict(d, score=np.nan)
    check("non-finite score rejected", b.enter(row("BAD", 1), bad_decision)[0] is None)
    check("bad entries leave the book unchanged", not b.s["positions"] and b.s["cash"] == risk.START_BALANCE)

    pos, _ = b.enter(row("GOOD", 1), d)
    before = json.dumps(b.s, sort_keys=True)
    for price in (np.nan, np.inf, -0.1, 1e308):
        try:
            b.update({"GOOD": price})
            rejected = False
        except ValueError:
            rejected = True
        check(f"invalid update price {price!r} rejected", rejected)
        check("invalid price update does not mutate book", json.dumps(b.s, sort_keys=True) == before)

    corrupt = Path("data/corrupt_state.json")
    corrupt.parent.mkdir(exist_ok=True)
    corrupt.write_text("{broken", encoding="utf-8")
    try:
        risk.PaperBook(corrupt)
        corrupt_rejected = False
    except ValueError:
        corrupt_rejected = True
    check("corrupt persisted state fails closed", corrupt_rejected)
    wrong_type = Path("data/wrong_type_state.json")
    bad_saved_state = dict(b.s, cash="1000")
    wrong_type.write_text(json.dumps(bad_saved_state), encoding="utf-8")
    try:
        risk.PaperBook(wrong_type)
        wrong_type_rejected = False
    except ValueError:
        wrong_type_rejected = True
    check("wrong-typed persisted state fails closed", wrong_type_rejected)


class FakeRpc:
    def __init__(self, mint_auth=None, freeze=None, holders=(5, 4, 3), ext=(), fail=False):
        self.a = dict(mint_auth=mint_auth, freeze=freeze, holders=holders, ext=ext, fail=fail)

    def __call__(self, method, params, tries=3):
        a = self.a
        import safety
        if a["fail"]:
            raise safety.RpcError("down")
        supply = 1_000_000_000
        if method == "getAccountInfo":
            info = {"mintAuthority": a["mint_auth"], "freezeAuthority": a["freeze"], "supply": str(supply),
                    "decimals": 6, "extensions": [{"extension": e} for e in a["ext"]]}
            return {"value": {"data": {"program": "spl-token", "parsed": {"info": info, "type": "mint"}}}}
        if method == "getTokenLargestAccounts":
            return {"value": [{"address": f"acc{i}", "amount": str(int(supply * p / 100))} for i, p in enumerate(a["holders"])]}
        if method == "getMultipleAccounts":
            return {"value": [{"data": {"parsed": {"info": {"owner": f"own{i}"}}}} for i, _ in enumerate(params[0])]}
        raise AssertionError(method)


def test_safety():
    print("\nSafety filter")
    import safety
    row = {"token_address": "MINT", "pair_address": "P", "pair_age_minutes": 500, "liquidity_change_15": 0.0}

    def run(**kw):
        safety.CACHE_FILE = Path(f"data/sc_{len(os.listdir('data')) if os.path.isdir('data') else 0}.json")
        safety.rpc = FakeRpc(**kw)
        return safety.check_token(dict(row))

    r = run()
    check("clean token passes", r["passed"] is True)
    check("not-checked list is reported", len(r["not_checked"]) >= 3)
    check("mint authority rejects", run(mint_auth="X")["passed"] is False)
    check("freeze authority rejects", run(freeze="X")["passed"] is False)
    check("whale (30% top holder) rejects", run(holders=(30, 5, 4))["passed"] is False)
    check("top-10 concentration rejects", run(holders=(9,) * 10)["passed"] is False)
    check("permanent delegate rejects", run(ext=("permanentDelegate",))["passed"] is False)
    check("RPC failure fails closed", run(fail=True)["passed"] is False)
    young = dict(row, pair_age_minutes=5)
    safety.rpc = FakeRpc()
    safety.CACHE_FILE = Path("data/sc_young.json")
    check("brand-new pair rejects", safety.check_token(young)["passed"] is False)
    drop = dict(row, liquidity_change_15=-0.5)
    safety.CACHE_FILE = Path("data/sc_drop.json")
    check("liquidity collapse rejects", safety.check_token(drop)["passed"] is False)
    safety.CACHE_FILE = Path("data/sc_nofetch.json")
    check("cache miss with fetch=False is 'unverified', not safe", safety.check_token(dict(row), fetch=False)["passed"] is None)


def hot_row():
    return {"token_address": "MINT", "symbol": "HOT", "price_usd": 1.0, "return_5": 0.06, "return_15": 0.12,
            "macd_spread": 0.1, "ema_spread": 0.02, "volume_accel": 3.5, "buy_sell_ratio": 2.2,
            "buy_sell_ratio_1h": 1.6, "liquidity_usd": 300_000, "liquidity_change_15": 0.05, "rsi14": 60,
            "pair_age_minutes": 600, "volatility_15": 0.01, "pair_address": "P"}


def test_score():
    print("\nScoring and honesty of labels")
    import score
    ok_safety = lambda r: {"passed": True, "hard": [], "warn": [], "metrics": {"top10_pct": 20}, "not_checked": []}
    bad_safety = lambda r: {"passed": False, "hard": ["mint authority not revoked"], "warn": [], "metrics": {}, "not_checked": []}
    d = score.decide(hot_row(), None, ok_safety)
    check("strong row -> BUY CANDIDATE", d["verdict"] == "BUY CANDIDATE" and d["score"] >= 80)
    card = score.format_card(hot_row(), d)
    check("no-model card says ANALYTICAL and Confidence: n/a", "Analytical Score" in card and "Confidence: n/a" in card)
    check("no-model card never shows a probability", "probability" not in card.lower())
    d = score.decide(hot_row(), None, bad_safety)
    check("failed safety forces AVOID", d["verdict"] == "AVOID" and d["risk"] == "Extreme")
    d = score.decide(hot_row(), None, lambda r: {"passed": None, "hard": [], "warn": ["on-chain check pending"], "metrics": {}, "not_checked": []})
    check("unverified token cannot be BUY CANDIDATE", d["verdict"] != "BUY CANDIDATE")
    dead = dict(hot_row(), return_5=-0.1, return_15=-0.2, macd_spread=-1, ema_spread=-1, volume_accel=0.2,
                buy_sell_ratio=0.4, buy_sell_ratio_1h=0.5, liquidity_usd=20_000, liquidity_change_15=-0.1, rsi14=15)
    check("weak row -> AVOID", score.decide(dead, None, ok_safety)["verdict"] == "AVOID")

    class M:  # stand-in model
        def predict_proba(self, x):
            return np.array([[0.2, 0.8]] * len(x))
    bundle = {"model": M(), "features": ["rsi14", "return_5"], "threshold": 0.7,
              "horizon_minutes": 15, "target_return": 0.1, "test_start": "2026-01-01T00:00:00+00:00"}
    d = score.decide(hot_row(), bundle, ok_safety)
    card = score.format_card(hot_row(), d)
    check("model card shows uncalibrated probability, not 'confidence'", "uncalibrated" in card and "Confidence" not in card)
    low = {"model": M(), "features": ["rsi14"], "threshold": 0.9,
           "horizon_minutes": 15, "target_return": 0.1, "test_start": "2026-01-01T00:00:00+00:00"}
    check("model below its own threshold is not a BUY", score.decide(hot_row(), low, ok_safety)["verdict"] != "BUY CANDIDATE")


def test_data_quality_and_labels():
    print("\nData quality and leakage controls")
    from data_quality import DataQualityError, clean_market_frame, normalize_market_row, require_training_source
    from labels import add_labels, calibration_split, time_split
    import train
    frame = pd.DataFrame([
        {"timestamp": "2026-09-01T00:00:00Z", "token_address": "A", "price_usd": 1.0, "source": "dexscreener"},
        {"timestamp": "2026-09-01T00:00:00Z", "token_address": "A", "price_usd": 2.0, "source": "dexscreener"},
        {"timestamp": "bad", "token_address": "A", "price_usd": 3.0},
        {"timestamp": "2026-09-01T00:01:00Z", "token_address": "", "price_usd": 3.0},
        {"timestamp": "2026-09-01T00:02:00Z", "token_address": "A", "price_usd": 0.0},
    ])
    clean, report = clean_market_frame(frame, now="2026-09-01T00:03:00Z")
    check("invalid required rows are dropped and duplicates are counted",
          len(clean) == 2 and report["invalid_rows_dropped"] == 2 and report["duplicate_rows_dropped"] == 1)
    check("duplicate timestamps keep the last row and zero-price death sentinel survives",
          float(clean.iloc[0]["price_usd"]) == 2.0 and float(clean.iloc[-1]["price_usd"]) == 0.0)
    malformed = normalize_market_row({"timestamp": "2026-09-01T00:00:00Z", "token_address": "A",
                                      "price_usd": 1, "liquidity_usd": "Infinity"})
    check("malformed optional market fields become missing", np.isnan(malformed["liquidity_usd"]))
    legacy, _ = clean_market_frame(pd.DataFrame([{"timestamp": "2026-09-01T00:00:00Z",
                                                  "token_address": "A", "price_usd": 1.0,
                                                  "url": "https://dexscreener.com/solana/pair"}]))
    check("legacy rows are verified from their documented DexScreener URL", legacy.iloc[0]["source"] == "dexscreener")
    try:
        require_training_source({"sources": {"multi": 10}})
        mixed_rejected = False
    except DataQualityError:
        mixed_rejected = True
    check("training source guard rejects alternate or mixed feeds", mixed_rejected)
    try:
        require_training_source({"sources": {"synthetic_test": 10}})
        synthetic_rejected = False
    except DataQualityError:
        synthetic_rejected = True
    check("synthetic data is rejected unless the disposable test flag is explicit", synthetic_rejected)
    old_root = train.ROOT
    old_allow = os.environ.get("ALLOW_SYNTHETIC_DATA")
    old_pipeline_root = os.environ.get("SELFTEST_PIPELINE_ROOT")
    sandbox = Path("synthetic_guard_test").resolve()
    sandbox.mkdir(exist_ok=True)
    (sandbox / ".selftest_synthetic_pipeline").touch()
    try:
        train.ROOT = sandbox
        os.environ["ALLOW_SYNTHETIC_DATA"] = "1"
        os.environ.pop("SELFTEST_PIPELINE_ROOT", None)
        check("synthetic override alone cannot enable model fitting", not train._is_disposable_synthetic_pipeline())
        os.environ["SELFTEST_PIPELINE_ROOT"] = str(sandbox)
        check("synthetic data is allowed only in the explicitly marked test sandbox",
              train._is_disposable_synthetic_pipeline())
    finally:
        train.ROOT = old_root
        if old_allow is None:
            os.environ.pop("ALLOW_SYNTHETIC_DATA", None)
        else:
            os.environ["ALLOW_SYNTHETIC_DATA"] = old_allow
        if old_pipeline_root is None:
            os.environ.pop("SELFTEST_PIPELINE_ROOT", None)
        else:
            os.environ["SELFTEST_PIPELINE_ROOT"] = old_pipeline_root
        shutil.rmtree(sandbox, ignore_errors=True)

    stamps = pd.date_range("2026-09-01", periods=20, freq="min", tz="UTC")
    prices = [1.0] * 8 + [0.0] + [0.0] * 11
    history = pd.DataFrame({"timestamp": stamps, "token_address": "A", "price_usd": prices})
    labeled = add_labels(history, horizon_minutes=2, target_return=0.1)
    row_before_death = labeled.loc[labeled["timestamp"] == stamps[6]].iloc[0]
    check("observed dead-pair sentinel is labeled as a total loss",
          row_before_death["future_return"] == -1 and row_before_death["target"] == 0)
    sparse = pd.DataFrame({"timestamp": [stamps[0], stamps[10]],
                           "token_address": ["A", "OTHER"], "price_usd": [1.0, 1.0]})
    sparse_labels = add_labels(sparse, horizon_minutes=2, target_return=0.1)
    check("a missing token with later market activity remains censored",
          pd.isna(sparse_labels.loc[sparse_labels["token_address"] == "A", "future_return"]).all())
    fit, test, cut = time_split(labeled.dropna(subset=["target"]), horizon_minutes=2, test_fraction=0.2)
    embargo = pd.Timedelta(minutes=2) + pd.Timedelta(minutes=1)
    check("train labels are purged before chronological test cutoff",
          fit["timestamp"].max() < cut - embargo and test["timestamp"].min() >= cut)
    fit2, calibration, cal_cut = calibration_split(labeled.dropna(subset=["target"]), 2, 0.25)
    check("calibration rows are separated by a temporal purge", fit2["timestamp"].max() < cal_cut - embargo
          and calibration["timestamp"].min() >= cal_cut)
    from validation import feature_drift_report
    drift = feature_drift_report(pd.DataFrame({"f": np.arange(30)}),
                                 pd.DataFrame({"f": np.arange(30, 60)}), ["f"])
    check("feature drift is reported as a diagnostic without changing labels or model inputs",
          drift[0]["psi"] is not None and drift[0]["psi"] > 0)


def test_scanner_and_lock():
    print("\nScanner contracts and process lock")
    import scanner
    from process_lock import ProcessLock
    pair = {"chainId": "solana", "baseToken": {"address": "MINT", "symbol": "T"},
            "priceUsd": "0.01", "liquidity": {"usd": "300000"}, "volume": {"h24": "90000", "m5": "3", "h1": "40"},
            "txns": {"h24": {"buys": 80, "sells": 40}, "m5": {"buys": 2, "sells": 1}},
            "priceChange": {}, "pairCreatedAt": 0, "marketCap": "garbage", "fdv": "Infinity",
            "pairAddress": "PAIR", "url": "https://dexscreener.com/solana/PAIR", "_sources": ["new_profiles"]}
    selected = scanner.pick_best_pair([dict(pair, baseToken={"address": "OTHER"}), pair], "MINT")
    normalized = scanner.normalize(selected)
    check("scanner requires token as the pair base and preserves source provenance",
          normalized["token_address"] == "MINT" and normalized["discovery_sources"] == "new_profiles")
    check("malformed market cap fields are not fabricated", np.isnan(normalized["market_cap"]) and np.isnan(normalized["fdv"]))
    try:
        scanner._get("http://example.invalid")
        rejected = False
    except ValueError:
        rejected = True
    check("scanner rejects non-HTTPS and non-DexScreener URLs", rejected)
    original_rate_file, original_rate_lock = scanner.RATE_FILE, scanner.RATE_LOCK
    try:
        scanner.RATE_FILE = Path("limits/dexscreener_rate.json")
        scanner.RATE_LOCK = Path("limits/dexscreener_rate.lock")
        scanner._rate_limit("https://api.dexscreener.com/token-profiles/latest/v1")
        rate_state = json.loads(scanner.RATE_FILE.read_text(encoding="utf-8"))
        check("DexScreener request cadence is shared through a locked local state file",
              len(rate_state["discovery"]) == 1)
    finally:
        scanner.RATE_FILE, scanner.RATE_LOCK = original_rate_file, original_rate_lock
    path = Path("locks/test.lock")
    first = ProcessLock(path).acquire()
    try:
        child_code = ("import sys\nfrom process_lock import ProcessLock\n"
                      "try:\n    ProcessLock(sys.argv[1]).acquire()\n"
                      "except RuntimeError:\n    sys.exit(0)\n"
                      "sys.exit(3)\n")
        child_env = dict(os.environ, PYTHONPATH=str(PROJECT))
        child = subprocess.run([sys.executable, "-c", child_code, str(path.resolve())],
                               capture_output=True, text=True, env=child_env)
        duplicate_rejected = child.returncode == 0
        check("a second process cannot acquire the same live lock", duplicate_rejected)
    finally:
        first.release()
    with ProcessLock(path):
        check("lock is released and can be acquired again", True)


def test_tracking_collector_and_dashboard():
    print("\nTracking, collector persistence, and dashboard helpers")
    import tracking
    import collector
    import dashboard
    from data_quality import DataQualityError
    original_track = tracking.TRACK_FILE
    original_track_lock = tracking.TRACK_LOCK
    original_scan, original_lookup = tracking.scan, tracking.get_pairs_by_address
    original_out, original_limit = collector.OUT, collector.MAX_MARKET_BYTES
    original_dashboard_data = dashboard.DATA
    original_dashboard_cache = dashboard._cache["csv"]
    original_dashboard_quality = dashboard._cache["quality"]
    original_dashboard_error = dashboard._data_error
    cwd = Path.cwd()
    try:
        tracking.TRACK_FILE = Path("tracking/tracked.json")
        tracking.TRACK_LOCK = Path("tracking/tracking.lock")
        pair = {"chainId": "solana", "baseToken": {"address": "MINT", "symbol": "T"},
                "priceUsd": "0.01", "liquidity": {"usd": "300000"}, "volume": {"h24": "90000"},
                "txns": {"h24": {"buys": 80, "sells": 40}}, "pairCreatedAt": 0,
                "pairAddress": "PAIR", "url": "https://dexscreener.com/solana/PAIR", "_sources": ["latest_boosts"]}
        tracking.scan = lambda: [pair]
        tracking.get_pairs_by_address = lambda addresses: []
        state = tracking.load_tracked()
        first = tracking.scan_and_track(state)
        tracking.save_tracked(state)
        tracking.scan = lambda: []
        dead = tracking.scan_and_track(state)
        check("tracked disappeared token is written as an explicit zero-price row",
              len(first) == 1 and len(dead) == 1 and dead[0]["price_usd"] == 0)
        corrupt_path = tracking.TRACK_FILE
        corrupt_path.write_text("{bad", encoding="utf-8")
        try:
            tracking.load_tracked()
            corrupt_rejected = False
        except ValueError:
            corrupt_rejected = True
        check("corrupt tracking state fails closed", corrupt_rejected)

        collector.OUT = Path("collector/market.csv")
        collector.MAX_MARKET_BYTES = 10_000_000
        collector.write_rows(first)
        written = pd.read_csv(collector.OUT)
        check("collector writes the shared provenance schema", "source" in written and
              written.iloc[0]["source"] == "dexscreener")
        with collector.OUT.open("ab") as f:
            f.write(b"incomplete final row")
        repair_row = dict(first[0], timestamp=(pd.Timestamp(first[0]["timestamp"]) + pd.Timedelta(seconds=1)).isoformat())
        collector.write_rows([repair_row])
        repaired = pd.read_csv(collector.OUT)
        check("collector archives and repairs an interrupted final CSV append",
              len(repaired) == 2 and len(list(collector.OUT.parent.glob("market_partial_*.bin"))) == 1)
        try:
            collector.write_rows([dict(first[0], source="other")])
            wrong_source_rejected = False
        except DataQualityError:
            wrong_source_rejected = True
        check("DexScreener collector refuses mixed-source rows", wrong_source_rejected)
        check("dashboard import leaves the working directory unchanged", Path.cwd() == cwd)
        check("dashboard finite numeric helper rejects infinity", dashboard.finite_number(float("inf")) is None)
        dashboard.DATA = Path("dashboard/future_market.csv")
        dashboard.DATA.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": [pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=1),
                                    pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)],
                      "token_address": ["NOW", "FUTURE"], "price_usd": [1.0, 2.0],
                      "source": ["dexscreener", "dexscreener"]}).to_csv(dashboard.DATA, index=False)
        dashboard._cache["csv"] = (None, None)
        visible = dashboard.load_data()
        check("dashboard ignores future-dated rows when showing current signals",
              len(visible) == 1 and visible.iloc[0]["token_address"] == "NOW"
              and "future-dated" in dashboard._data_error)
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy="prior").fit([[0, 0], [1, 1]], [0, 1])
        dashboard_bundle = {"model": dummy, "features": ["rsi14", "return_15"], "threshold": 0.5,
                           "horizon_minutes": 15, "target_return": 0.1,
                           "test_start": "2026-01-01T00:00:00+00:00"}
        signal_data = pd.DataFrame({"timestamp": pd.date_range("2026-09-01", periods=40, freq="min", tz="UTC"),
                                    "token_address": "DASH", "symbol": "D", "price_usd": np.linspace(1, 1.1, 40),
                                    "liquidity_usd": 100_000, "volume_24h": 200_000, "volume_5m": 500,
                                    "buys_5m": 10, "sells_5m": 5, "txns_24h": 100, "pair_age_minutes": 100})
        signals = dashboard.build_signals(signal_data, dashboard_bundle)
        check("dashboard scores through the validated model probability helper",
              len(signals) == 1 and signals[0]["prob"] == 0.5)
    finally:
        tracking.TRACK_FILE = original_track
        tracking.TRACK_LOCK = original_track_lock
        tracking.scan, tracking.get_pairs_by_address = original_scan, original_lookup
        collector.OUT, collector.MAX_MARKET_BYTES = original_out, original_limit
        dashboard.DATA = original_dashboard_data
        dashboard._cache["csv"] = original_dashboard_cache
        dashboard._cache["quality"] = original_dashboard_quality
        dashboard._data_error = original_dashboard_error


def test_model_store_and_backtest():
    print("\nModel integrity and backtest exits")
    import model_store
    from sklearn.dummy import DummyClassifier
    classifier = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    bundle = {"model": classifier, "features": ["f"], "threshold": 0.5,
              "horizon_minutes": 3, "target_return": 0.1, "test_start": "2026-01-01T00:00:00+00:00",
              "calibration_method": "uncalibrated"}
    paths = {name: getattr(model_store, name) for name in
             ("MODEL_DIR", "MODEL_PATH", "HASH_PATH", "META_PATH", "LOCK_PATH")}
    model_dir = Path("models_test").resolve()
    model_store.MODEL_DIR = model_dir
    model_store.MODEL_PATH = model_dir / "model.joblib"
    model_store.HASH_PATH = model_dir / "model.sha256"
    model_store.META_PATH = model_dir / "model_meta.json"
    model_store.LOCK_PATH = model_dir / ".model.lock"
    try:
        model_store.save_model(bundle)
        loaded = model_store.load_model(model_store.MODEL_PATH)
        check("model artifact and manifest round-trip with hash verification",
              len(model_store.predict_probability(loaded, [[0]])) == 1)
        model_store.MODEL_PATH.write_bytes(model_store.MODEL_PATH.read_bytes() + b"tamper")
        try:
            model_store.load_model(model_store.MODEL_PATH)
            tamper_rejected = False
        except RuntimeError:
            tamper_rejected = True
        check("modified local model artifact is rejected", tamper_rejected)
    finally:
        for name, value in paths.items():
            setattr(model_store, name, value)

    import backtest
    origin = pd.Timestamp("2026-09-01", tz="UTC")
    expected_zero_move = (1 - backtest.SIDE_COST) / (1 + backtest.SIDE_COST) - 1
    check("backtest round-trip cost math matches paper position accounting",
          abs(backtest._net_return(0.0) - expected_zero_move) < 1e-12)
    market = pd.DataFrame({"timestamp": [origin, origin + pd.Timedelta(minutes=1)],
                           "token_address": ["A", "A"], "price_usd": [1.0, 0.8]})
    entries = pd.DataFrame({"timestamp": [origin], "token_address": ["A"], "price_usd": [1.0],
                            "future_return": [-0.2]})
    trades = backtest.simulate_portfolio(entries, market, horizon_minutes=3)
    check("backtest fills a stop at the observed snapshot and charges round-trip costs",
          len(trades) == 1 and trades[0]["exit_reason"] == "stop loss" and trades[0]["net_return"] < -0.15)

    censored_market = pd.DataFrame({"timestamp": [origin, origin + pd.Timedelta(minutes=5)],
                                    "token_address": ["A", "OTHER"], "price_usd": [1.0, 1.0]})
    censored = backtest.simulate_portfolio(entries, censored_market, horizon_minutes=3)
    check("a missing token observation is censored instead of fabricated as a loss", not censored)

    vanished_market = pd.DataFrame({"timestamp": [origin, origin + pd.Timedelta(minutes=1)],
                                    "token_address": ["A", "A"], "price_usd": [1.0, 0.0]})
    vanished = backtest.simulate_portfolio(entries, vanished_market, horizon_minutes=3)
    check("an explicit zero-price observation closes as a confirmed total loss",
          len(vanished) == 1 and vanished[0]["exit_reason"] == "pair vanished"
          and vanished[0]["net_return"] == -1.0)

    late_market = pd.DataFrame({"timestamp": [origin, origin + pd.Timedelta(minutes=3)],
                                "token_address": ["A", "A"], "price_usd": [1.0, 0.95]})
    late = backtest.simulate_portfolio(entries, late_market, horizon_minutes=3, max_hold_minutes=1)
    check("max-hold exits use the next observed price even after a scan gap",
          len(late) == 1 and late[0]["exit_reason"] == "max hold"
          and late[0]["exit_time"] == origin + pd.Timedelta(minutes=3))

    daily_market = pd.DataFrame({"timestamp": [origin, origin + pd.Timedelta(minutes=1),
                                                origin + pd.Timedelta(minutes=2)],
                                 "token_address": ["A", "A", "B"], "price_usd": [1.0, 0.0, 1.0]})
    daily_entries = pd.DataFrame({"timestamp": [origin, origin + pd.Timedelta(minutes=2)],
                                 "token_address": ["A", "B"], "price_usd": [1.0, 1.0],
                                 "future_return": [-1.0, 0.0]})
    daily = backtest.simulate_portfolio(daily_entries, daily_market, horizon_minutes=3)
    check("daily loss limit halts later entries after an observed loss", len(daily) == 1
          and daily[0]["token"] == "A")


def test_live_execution_gate():
    print("\nLive execution safety gate")
    import swap_bot
    original_argv = sys.argv
    try:
        sys.argv = ["swap_bot.py", "--from", "SOL", "--to", "USDC", "--amount", "1", "--execute"]
        try:
            swap_bot.main()
            refused = False
        except SystemExit as exc:
            refused = "live execution is disabled" in str(exc).lower()
        check("--execute is refused before quote, key access, signing, or broadcast", refused
              and swap_bot.LIVE_EXECUTION_ENABLED is False)
    finally:
        sys.argv = original_argv


def test_quote():
    print("\nQuote verification")
    import quote_check as q
    q._order = lambda i, o, a: (_ for _ in ()).throw(RuntimeError("HTTP 500"))
    check("API failure fails closed", q.verify_round_trip("M", 6, 0.1)["ok"] is False)
    q._order = lambda i, o, a: {"outAmount": "1000000"} if o != q.SOL else {"outAmount": "0"}
    r = q.verify_round_trip("M", 6, 0.1)
    check("no sell route detected", r["ok"] is False and "NO SELL ROUTE" in r["reasons"][0])
    q._order = lambda i, o, a: {"outAmount": "1000000"} if o != q.SOL else {"outAmount": str(int(0.1e9 * 0.4))}
    check("60% round-trip loss rejected", q.verify_round_trip("M", 6, 0.1)["ok"] is False)
    q._order = lambda i, o, a: {"outAmount": "1000000"} if o != q.SOL else {"outAmount": str(int(0.1e9 * 0.97))}
    check("3% round-trip loss accepted", q.verify_round_trip("M", 6, 0.1)["ok"] is True)


def test_indicators():
    print("\nIndicators")
    from indicators import add_indicators, FEATURES
    df = synth_market(n_tokens=3, minutes=120, with_rugs=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out = add_indicators(df)
    check("all features computed", all(f in out.columns for f in FEATURES))
    check("new features have values", out["volume_accel"].notna().sum() > 0 and out["liquidity_change_15"].notna().sum() > 0)
    old = df.drop(columns=["market_cap", "fdv", "volume_1h", "buys_1h", "sells_1h"])
    check("old-format data (no v2 columns) still works", add_indicators(old)["rsi14"].notna().sum() > 0)


def pipeline():
    print("\nPipeline on synthetic data (train -> backtest -> evaluate)")
    model_dir = PROJECT / "models"
    model_snapshot = (sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in model_dir.iterdir())
                     if model_dir.exists() else None)
    with tempfile.TemporaryDirectory(prefix="memeai_pipeline_", dir=Path.cwd()) as temp:
        sandbox = Path(temp) / "project"
        shutil.copytree(PROJECT, sandbox, ignore=shutil.ignore_patterns(
            ".git", ".vs", ".venv", "__pycache__", "backups", "data", "models", ".env"))
        data_dir = sandbox / "data"
        data_dir.mkdir(exist_ok=True)
        (sandbox / ".selftest_synthetic_pipeline").touch()
        synth_market(n_tokens=60, minutes=600).to_csv(data_dir / "market.csv", index=False)
        env = dict(os.environ, PYTHONPATH=str(sandbox), MODEL_THRESHOLD="0.5", TARGET_RETURN="0.02",
                   ALLOW_SYNTHETIC_DATA="1", SELFTEST_PIPELINE_ROOT=str(sandbox.resolve()))
        for script in ("train.py", "backtest.py", "evaluate.py"):
            r = subprocess.run([sys.executable, str(sandbox / script)], cwd=sandbox,
                               capture_output=True, text=True, env=env)
            check(f"{script} runs", r.returncode == 0)
            if r.returncode != 0:
                print(r.stdout[-800:], r.stderr[-1500:])
        check("model + hash + history written",
              all(p.exists() for p in (sandbox / "models/model.joblib", sandbox / "models/model.sha256",
                                       sandbox / "data/model_history.jsonl")))
        split_counts_ok = False
        history_file = sandbox / "data/model_history.jsonl"
        if history_file.exists():
            try:
                record = json.loads(history_file.read_text(encoding="utf-8").splitlines()[-1])
                counts = record.get("class_counts", {})
                split_counts_ok = all(
                    isinstance(counts.get(name), dict)
                    and counts[name].get("positive", -1) >= 0
                    and counts[name].get("negative", -1) >= 0
                    and counts[name]["positive"] + counts[name]["negative"] == record[row_key]
                    for name, row_key in (("fit", "fit_rows"), ("calibration", "calibration_rows"),
                                          ("test", "test_rows"))
                )
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                split_counts_ok = False
        check("training history records class counts for every temporal split", split_counts_ok)
    after_snapshot = (sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in model_dir.iterdir())
                      if model_dir.exists() else None)
    check("pipeline leaves project models untouched", after_snapshot == model_snapshot)
    check("pipeline temporary project is cleaned up", not Path(temp).exists())


if __name__ == "__main__":
    test_indicators()
    test_data_quality_and_labels()
    test_scanner_and_lock()
    test_tracking_collector_and_dashboard()
    test_model_store_and_backtest()
    test_live_execution_gate()
    test_risk()
    test_safety()
    test_score()
    test_quote()
    if "--pipeline" in sys.argv:
        pipeline()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
