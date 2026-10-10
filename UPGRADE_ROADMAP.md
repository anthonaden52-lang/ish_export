# Upgrade status

This is the implementation ledger for the 50 upgrade areas. “Implemented” means the code or safeguard
is present; the new offline suite still needs to run in a Python environment. It does not mean any
trading strategy is profitable or validated on live market outcomes.

## Phase 1: security and risk

| # | Upgrade | Status |
|---:|---|---|
| 1 | Validate numeric risk settings and reject non-finite values | Implemented |
| 2 | Fail closed on malformed paper-trading state | Implemented |
| 3 | Fail closed on malformed tracked-token state | Implemented |
| 4 | Size positions against stop loss and both-side costs | Implemented |
| 5 | Enforce a paper emergency stop and daily loss limit | Implemented |
| 6 | Cap position size, token exposure, and open positions | Implemented |
| 7 | Prevent duplicate collector, trader, training, backtest, dashboard-job, and tracking writers | Implemented |
| 8 | Flush and atomically replace important local state and model files | Implemented |
| 9 | Hash-check local model artifacts and record a manifest | Implemented |
| 10 | Hard-disable live execution before quote, key access, signing, or broadcast | Implemented; live transaction inspection remains incomplete |

## Phase 2: market data

| # | Upgrade | Status |
|---:|---|---|
| 11 | Restrict the primary scanner to documented HTTPS DexScreener hosts | Implemented |
| 12 | Retry bounded HTTP 429/5xx failures with backoff | Implemented |
| 13 | Share DexScreener request caps across local processes | Implemented |
| 14 | Use the documented, versioned GeckoTerminal public API at a conservative cadence | Implemented |
| 15 | Validate API response shapes and market row identity, time, and price | Implemented |
| 16 | Record feed and discovery provenance on each row | Implemented |
| 17 | Keep alternate-source data separate and reject unknown/mixed training feeds | Implemented |
| 18 | Preserve unavailable metrics as missing rather than fabricated zeros | Implemented |
| 19 | De-duplicate token/timestamp observations and report invalid rows | Implemented |
| 20 | Preserve and repair a partial final CSV append; cap file growth and expose collector health | Implemented |

## Phase 3: model validation

| # | Upgrade | Status |
|---:|---|---|
| 21 | Build labels from UTC timestamps and real-time horizons | Implemented |
| 22 | Keep zero-price pair-death observations as losses, never entries | Implemented |
| 23 | Count confirmed disappeared pairs as losses (observed zero-price row) | Implemented |
| 24 | Exclude outcomes that remain censored or cannot be established | Implemented |
| 25 | Split train and test chronologically across all tokens | Implemented |
| 26 | Purge label-window overlap at train/test and fit/calibration boundaries | Implemented |
| 27 | Fit calibration only on a separate chronological block | Implemented |
| 28 | Mark small or one-class calibration samples as uncalibrated | Implemented |
| 29 | Apply the saved calibrator consistently in scoring and dashboard paths | Implemented |
| 30 | Record feature version, label settings, test start, and training-data hash | Implemented |
| 31 | Add expanding-window walk-forward metrics | Implemented |
| 32 | Report AUC and Brier score against a prevalence baseline | Implemented |
| 33 | Report out-of-time feature PSI and missing-rate changes as diagnostics | Implemented |
| 34 | Keep threshold selection outside the final test block | Implemented; threshold remains a configured value |
| 35 | Reject future-dated observations before train, evaluation, backtest, or dashboard scoring | Implemented |
| 36 | Reject unverified and alternate-source data for model fitting; confine synthetic override to marked temp pipeline | Implemented |
| 37 | Version feature inputs and validate model bundle contracts | Implemented |
| 38 | Use only a locally trained, hash-checked model | Implemented |
| 39 | Log retraining metrics and data-quality reports | Implemented |
| 40 | Keep synthetic data confined to an explicit disposable pipeline test and add Windows/Linux CI | Implemented; CI run pending |

## Phase 4: paper backtest

| # | Upgrade | Status |
|---:|---|---|
| 41 | Start entries only in the model’s saved out-of-time test period | Implemented |
| 42 | Replay observed stop-loss, take-profit, trailing-stop, and time exits | Implemented |
| 43 | Use observed snapshot prices for fills, including gaps | Implemented |
| 44 | Charge fees and slippage on entry and exit | Implemented |
| 45 | Apply cash, per-trade risk, token exposure, and position-count limits | Implemented |
| 46 | Apply a daily loss halt and skip unresolved/censored outcomes | Implemented |
| 47 | Report realized balance, drawdown, return distribution, exit reasons, and token breakdown | Implemented |
| 48 | Re-run the same signal path at lower, base, and higher cost assumptions | Implemented |
| 49 | Keep snapshot-backtest limitations explicit (no intraperiod tape, order book, latency, or actual fills) | Implemented |
| 50 | Keep live transaction inspection and exact-message simulation outside the trading path | Guarded and deferred; live execution stays disabled |

## Remaining limitations

- The test suite and synthetic pipeline are authored, but could not be executed in this shell because
  `python`, `py`, and a project virtual environment are unavailable. See `README.md` for the test command.
- `requirements.txt` is not a fully pinned lockfile. The CI workflow installs the declared compatible
  ranges; exact tested package versions remain to be recorded after CI runs.
- Missing token rows without an observed zero-price marker remain censored; this avoids treating a
  collector gap or expired tracking window as a real total loss.
- The backtest uses sampled API snapshots, not full tick data or executable order-book fills. It cannot
  model intraperiod stop crossings, failed transactions, MEV, or real slippage.
- Creator-wallet behavior, liquidity-lock verification, wallet clustering, and social/narrative data are
  not implemented. The safety filter reports these checks as unavailable.
- Live transaction instruction decoding, lookup-table resolution, exact-message simulation, and
  post-trade balance reconciliation are incomplete. The hard execution gate must remain off.
