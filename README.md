# Acorn 1.0

Acorn 1.0 is a paper-first Solana meme-coin scanner and ML research bot.

The public, read-only GitHub Pages site files are in `docs/`; see [PAGES_SETUP.md](PAGES_SETUP.md) for setup.

Pipeline:
1. Discover recent Solana tokens through DEX Screener.
2. Filter by liquidity, volume, transactions and pair age.
3. Record repeated market snapshots to `data/market.csv`.
4. Calculate RSI, EMA, MACD, momentum, volatility and volume features.
5. Create future-return labels (real-time horizon from HORIZON_MINUTES).
6. Train a gradient-boosting classifier.
7. Backtest chronologically.
8. Run a paper-trading loop.

IMPORTANT:
- This is research software, not a profit guarantee.
- Meme coins can go to zero and liquidity can disappear.
- Do not put a main-wallet private key in this project.
- Live execution is hard-disabled in this build. `swap_bot.py --execute` exits before requesting a key,
  fetching a quote, signing, or submitting a transaction. Paper trading is the default.

## Windows / VS Code setup

Open the project folder in VS Code, then open Terminal:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`.

## v2: scanner -> safety -> score -> risk -> quote check -> paper trade -> tracking

`paper_trader.py` now runs the whole chain, still simulation only:

1. **Scan** (`scanner.py`): new profiles + latest boosts + top boosts, de-duplicated, filtered, best momentum first. DexScreener has no official "trending" list; boosts are paid promotion, so they are only leads.
2. **Score** (`score.py`): 0-100. With a trained model, a separate temporal calibrator is applied when available; otherwise the raw output is clearly marked uncalibrated. Neither is called confidence. Without a model it is a rule-based **Analytical Score**. The sources are never mixed in the logs (`score_source`).
3. **Safety** (`safety.py`): mint/freeze authority, Token-2022 extensions, top-holder and top-10 concentration (pool vaults excluded), token age, liquidity collapse. Fail closed: RPC down = not safe. Only run for tokens that otherwise score WATCH or better, cached 30 min, capped per scan (`SAFETY_MAX_PER_SCAN`) to respect RPC limits.
4. **Risk engine** (`risk.py`): position size, max loss per trade, daily loss limit, stop loss, take profit, trailing stop, max open positions, max exposure per token, max hold time, emergency stop (create `data/STOP`; `swap_bot.py` honours it too). Positions persist in `data/paper_state.json`.
5. **Quote check** (`quote_check.py`): read-only Jupiter buy quote + sell-back quote. Rejects missing sell route and large round-trip loss.
6. **Tracking**: `data/paper_trades.csv` (closed trades), `data/predictions.csv` (every BUY/WATCH call, traded or not), `data/model_history.jsonl` (every retrain).
7. **Learn**: `python evaluate.py` shows trade stats, score-bucket hit rates vs base rate, per-indicator rank correlation and model permutation importance. Retrain with `python train.py` as data grows.

Output looks like:

```
🟢 BUY CANDIDATE            (or 🟡 WATCH / 🔴 AVOID)
Token: XYZ
Analytical Score: 91/100  (rule-based, NO trained model yet)
Confidence: n/a
Risk: Low
Reason: ...
```

**Not implemented** (reported as `not_checked`, never silently skipped): creator/developer wallet activity, liquidity lock/burn status, wallet clustering, social/narrative data, transaction simulation before live sends. A token that passes the checks is "no red flag found", not "safe".

`python selftest.py` runs offline checks for data validation, state corruption, model integrity, paper risk,
scanner provenance, backtest exits, dashboard scoring, and the live execution gate. Add `--pipeline` to
run synthetic data through train -> backtest -> evaluate in a disposable project copy. Synthetic data
only checks code paths; it says nothing about profitability or real market performance.

## Dashboard (see everything in a browser)

```powershell
python dashboard.py
```

Open http://127.0.0.1:8000. It shows collector status, data quality counts and sources, the trained model
and calibration method, paper signals, recent job output, and backups. Buttons: Save snapshot, Train model,
Run backtest. It cannot trade and has no wallet access. Non-local binding requires `DASH_TOKEN`; POST
actions also require a same-origin custom header.

On a phone: do not expose it to the internet. Run it on your PC or VPS and reach it through an SSH tunnel (`ssh -L 8000:127.0.0.1:8000 you@server`, then open http://127.0.0.1:8000 in the phone's SSH app browser or on a laptop) or a private network like Tailscale. If you must bind another address, set `DASH_TOKEN` first and use `--host`; the page asks for the token and sends it only in a header.

## 1. Collect data

Run:

```powershell
python collector.py
```

Let it run for several hours or days. More data is better. The DexScreener collector writes:

`data/market.csv`

The collector writes validated DexScreener rows with per-row source provenance to `data/market.csv`,
and reports health in `data/collector_health.json`. If `market.csv` contains rows from multiple sources,
the collector archives the original and keeps only identifiable DexScreener rows in the active file.
`multi_collector.py` writes its separate, sparse feed to `data/market_multi.csv`; it limits GeckoTerminal
requests to roughly 9/minute and records each token at its own fetch time. These feeds are not combined.

The model cannot be meaningfully trained from a few minutes of data.

Hotkeys (click the terminal first): **S** saves snapshots of market data, model plus integrity metadata,
paper state/trades, and predictions into `backups/<timestamp>/`; **Q** saves and quits. The collector
appends after each scan and preserves an incomplete final row before repairing it after an abrupt stop.
The same keys work in `paper_trader.py`.

## 2. Train

After you have data:

```powershell
python train.py
```

The trained model is written to:

`models/model.joblib`

The script splits by timestamp across all tokens and purges overlapping label windows at fit,
calibration, walk-forward, and final-test boundaries. Labels use the first observed price near the real-time
horizon; an explicit zero-price marker confirms a disappeared pair as a total loss, and missing future rows
remain censored and are excluded. It reports
out-of-time AUC and Brier score against a prevalence baseline. A separate chronological block fits a
Platt calibrator when there is enough data and both classes; otherwise the model is clearly marked
uncalibrated. `HORIZON_MINUTES`, `TARGET_RETURN`, and `MODEL_THRESHOLD` are saved with the feature version,
training-data hash, and model hash.

## 3. Backtest

```powershell
python backtest.py
```

This replays only the period after the model's saved test start. It triggers stop, take-profit, trailing,
and max-hold exits only from observed snapshots and uses the observed fill price, including gaps. The
default `MAX_HOLD_MINUTES` is twice `HORIZON_MINUTES`. It charges
fees and slippage on both sides, applies paper risk caps and a daily loss halt, and excludes outcomes the
CSV cannot establish. It reports balance, realized drawdown, win rate, average/median/worst return and exit
reasons. It is still a snapshot-based simulation: intraperiod prices, real order-book depth, execution
latency, MEV, failed fills, and actual transaction costs are unavailable. Fewer than 30 trades is not
meaningful.

## 4. Paper trader

```powershell
python paper_trader.py
```

The paper trader scans candidates and uses the trained model. It never sends a transaction.

## 5. Live execution

Live execution is hard-disabled. `--execute` is refused before a quote or any wallet-key access. The
transaction inspection and exact-message simulation needed for live signing have not been implemented
and reviewed. No private key or seed phrase is needed for this project’s current workflows.

## Data source

DexScreener's documented API provides Solana token/pair information used by the primary collector.
The alternate fixed-list collector uses GeckoTerminal's versioned public API and, when configured,
Birdeye's price API. Both sources are kept in separate CSVs; the alternate feed is sparse and unsuitable
for the current model features. See the [DexScreener API reference](https://docs.dexscreener.com/api/reference),
[GeckoTerminal API spec](https://api.geckoterminal.com/docs/v2/swagger.json), and
[Birdeye API docs](https://docs.birdeye.so/) for current endpoint contracts and limits.

## What the model learns

Target:
"Does this token gain at least TARGET_RETURN within HORIZON_MINUTES?"

Features include:
- RSI 14
- EMA 9/21/50
- MACD
- 1/5/15 minute momentum
- rolling volatility
- volume acceleration
- buy/sell transaction pressure
- liquidity
- volume/liquidity ratio
- pair age

This is not a magic prediction engine. It is a framework for collecting data and testing whether these signals have predictive value.

## Safety notes (patched version)

See CHANGES.md. Only load `models/model.joblib` you trained yourself (model files can run code). Tokens stay tracked after they fail filters so crashes count as losses.
