# Acorn 2.0 model release gate

Acorn 2.0 is a model-quality release label, not simply the result of another
training run. Keep the project and model labeled Acorn 1.0 until every gate
below passes on the same saved model artifact.

## Data and split requirements

- Train only on verified DexScreener rows in `data/market.csv`; reject synthetic,
  mixed-source, invalid, and future-dated rows.
- Use the existing chronological splits and label-window embargo in `train.py`.
  Do not tune features, thresholds, or hyperparameters against the final test
  block.
- The eligible dataset must span at least seven calendar days and the final
  chronological test block must contain at least 500 labeled observations,
  including at least 50 positive and 50 negative outcomes.

## Predictive quality requirements

- Final-test ROC AUC must be at least 0.60.
- Final-test Brier score must be at least 5% lower than the prevalence-only
  baseline Brier score computed from the training portion.
- A separate chronological calibration block must contain at least 200 labeled
  observations, with at least 30 examples of each class, and the saved artifact
  must use a fitted calibrator rather than `uncalibrated` probabilities.
- At least three valid expanding-window walk-forward folds must each contain at
  least 100 test observations and 20 examples of each class. Every fold's Brier
  score must beat its own prevalence baseline, and median fold AUC must be at
  least 0.58.

## Paper-trading evidence and promotion

- A cost-aware paper backtest must contain at least 30 closed trades and show
  positive net expectancy after the configured fees and slippage. It must also
  remain within the configured paper risk limits.
- Keep paper trading enabled. Do not broadcast transactions, spend SOL, or
  enable live trading as part of this release gate.
- Record the model artifact hash, data hash, test metrics, walk-forward metrics,
  calibration method, and backtest results in the release notes before changing
  the displayed/versioned name to Acorn 2.0.

## Current status (2026-10-10)

The latest retraining run used 840 eligible rows. Its final chronological test
block contained 84 observations (13 positive, 71 negative), ROC AUC 0.6251,
and Brier score 0.1391 versus a 0.1338 prevalence baseline. The Brier score is
worse than baseline. Temporal Platt calibration was fitted on 121 rows, below
the 200-row and per-class calibration minimums. Only two walk-forward folds
were available; both had Brier scores worse than their prevalence baselines,
and neither met the required test size or positive-class count.

Candidate trained at `2026-10-10T18:45:15Z`:

- Model SHA-256: `143590222dfd81d5a6881d0b03ebf546bcf823ed33a9e1c0869605f83f7b1083`
- Training data SHA-256: `864caab98b835d88b03ab2bd7257280bee89ae10db6780dee42eb7f78ab8c6ba`

The cost-aware paper backtest covered 100 matured rows across 28 tokens, but no
positions met the saved model threshold and risk limits. It therefore has zero
closed trades, below the 30-trade requirement. The full dataset currently
covers less than one day, below the seven-calendar-day minimum. The dashboard's
usable-label count is calculated with the same feature and observed-outcome
filters as `train.py`; it reported 840 when this run completed. The DexScreener
collector and local dashboard are currently running. New snapshots require
indicator warmup and an observed 15-minute future price before they can become
training examples.

**Release decision: keep Acorn 1.0.** The current metrics do not establish a
model-quality improvement, so the 2.0 name remains gated on more chronological
data and passing every requirement above.
