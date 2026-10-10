# Changes (one line per review finding)

## Upgrade pass (2026-10)

- Added `data_quality.py` for a shared market schema, finite-number normalization, source provenance,
  deduplication, future-date checks, and data-quality reports. Training and evaluation reject unknown
  or mixed feeds; only the disposable synthetic pipeline has an explicit synthetic-data override.
- Hardened DexScreener collection with bounded retries, endpoint-aware limits shared across processes,
  response-shape checks, row provenance, disk caps, health status, and preservation of a partial final CSV row.
- Serialized tracked-token reads/scans/writes across collector and paper trader so stale state cannot overwrite
  a newer scan. Made runtime data paths project-relative and paper trading import-safe.
- Added chronological calibration, walk-forward reports, AUC/Brier prevalence baselines, feature drift
  diagnostics, feature/model metadata, and hash-checked model persistence.
- Rebuilt `backtest.py` as an import-safe, test-period-only snapshot simulation with observed stop/target/
  trailing/time exits, both-side costs, risk caps, censored-outcome exclusion, cost sensitivity, and
  realized performance breakdowns.
- Hardened dashboard data/model reads, local job locking, bounded logs, source/quality display, and calibration
  and drift metadata. Expanded `selftest.py` and added a Windows/Linux CI workflow.
- Hard-disabled `swap_bot.py --execute`. It exits before fetching a quote or reading a wallet key; exact
  transaction inspection and simulation remain incomplete. See `live_warning.txt`.
- Added `UPGRADE_ROADMAP.md` for a 50-area implementation ledger and remaining limitations.
- Final review aligned backtest round-trip costs with the paper book, removed inferred missing-token
  exits from backtest fills, marked portfolio equity from observed snapshots, and latched the paper
  daily-loss halt until the next UTC day. Added offline checks for the cost formula, censored exits,
  explicit disappearance, late max-hold fills, and daily halts.
- Training labels and evaluation now treat only an observed zero-price row as confirmed disappearance;
  an absent token row by itself stays censored, since it can also mean a collection gap.
- Dashboard signal generation now ignores future-dated rows and surfaces a collector-clock warning.
- Synthetic training is now allowed only when the explicit test flag, temporary sandbox path, and
  pipeline marker all agree; the environment flag alone cannot authorize synthetic model fitting.
- Tests were not run: this shell has no available `python`, `py`, or project virtual-environment interpreter.

## swap_bot.py (rebuilt from the PDF with proper indentation)
1. (High #2) Price impact is now read as a positive percent with abs(); `priceImpact` is treated as a percent and `priceImpactPct` as a fraction (x100). Unreadable impact refuses the trade. UNITS MUST BE CONFIRMED against current Jupiter docs.
2. (High #3) Missing router fee, price impact, USD values or slippage threshold now refuse the trade unless --allow-unknown-value is passed.
3. (High #4) New --max-slippage-bps (default 100): refuses if the guaranteed minimum output is more than that below the quote.
4. (High #5) Before signing, the order's inputMint, outputMint, inAmount, taker and outAmount must match the request, or the trade is refused.
5. (Med #7) Every trade is logged to data/trades.jsonl before sending (no keys); timeouts/errors check the chain status, and the next --execute refuses while an earlier trade is unresolved (--ignore-pending after you check).
6. (Med #8) JUPITER_BASE and SOLANA_RPC_URL must be https hosts on an in-code allowlist; decimals are sanity-checked and shown in the quote and the YES prompt.
7. (Med #9) New --max-daily-usd cap (default 100, per UTC day, from the trade log); --yes prints a loud warning.
8. (Low #18) All Jupiter numbers go through float()/to_float() before formatting or comparing.
9. (Low #19) Profane error messages replaced with plain wording.
10. Extra: invalid BOT_PRIVATE_KEY gives a message that never echoes the key; requests don't follow redirects; --selftest now also tests the new checks.

## scanner.py / tracking.py (new) / collector.py / labels.py
11. (Critical #1) Tokens stay tracked (data/tracked.json) for TRACK_MINUTES after they last passed the filters and are re-fetched by pair address; a vanished/priceless pair is written as price 0.
12. (Critical #1) labels.py: an observed zero-price row gets future_return = -100%; a token merely missing from later rows stays censored; price-0 rows are never entries.
13. (High #6) scanner.py only accepts pairs whose baseToken is the token looked up.
14. (Low #15) scanner.py backs off and retries on HTTP 429/5xx (honours Retry-After).

## backtest.py / labels.py / .env.example
15. (Med #10) Fee and slippage are charged on entry AND exit; defaults now 0.5% fee + 2% slippage per side.
16. (Med #13) Backtest uses a starting balance, a max number of open positions and a fixed fraction of equity per trade; reports final balance and % drawdown.

## indicators.py
17. (Med #11) return_1/5/15 are now over real minutes (price at or before t-n minutes), and indicators restart after a gap longer than GAP_MINUTES.
18. (Med #12) One history window (HISTORY_ROWS=300 via window_history) used by paper trader and dashboard; long enough that EMAs match training.

## paper_trader.py
19. (Med #12) History is saved to data/paper_history.csv and reloaded on restart; uses the same tracking as the collector.

## model_store.py (new) / train.py / backtest.py / paper_trader.py / dashboard.py
20. (Med #14) Models load only from models/model.joblib and only if its SHA-256 matches models/model.sha256 written by train.py, with a warning that model files can run code.

## dashboard.py
21. (Low #16) Token is accepted only in the X-Token header; the page prompts for it (kept in sessionStorage) and the startup line no longer prints it.

## .gitignore (new) / hotkeys.py / README.md
22. (Low #17) .gitignore excludes .env, key files, data/, models/, backups/.
23. hotkeys snapshot also saves model hash, tracked list and trade log; README updated.

## v2 (from "Meme AI Trader - Problems & Solutions")
24. scanner.py: multiple discovery sources (profiles, latest boosts, top boosts), momentum-ranked.
25. scanner.py / collector.py / indicators.py: new columns (market cap, FDV, 1h volume and txns) and features (liquidity change, volume acceleration, 1h buy/sell pressure, mcap/liquidity, FDV). Old market.csv is moved aside automatically; old-format data still loads.
26. safety.py (new): on-chain and market rug filter, fail closed. quote_check.py (new): read-only Jupiter round-trip quote check.
27. score.py (new): 0-100 score, BUY CANDIDATE / WATCH / AVOID, honest MODEL vs ANALYTICAL labelling.
28. risk.py (new): paper book with sizing, stops, trailing stop, daily loss limit, emergency stop file.
29. paper_trader.py: rewritten around the above; logs predictions and closed trades.
30. evaluate.py (new), train.py (retrain log), selftest.py (new), .env.example and .gitignore (were referenced but missing).
31. swap_bot.py: refuses --execute while data/STOP exists. Nothing else in the live path was changed.
32. multi_collector.py: API key moved to the BIRDEYE_API_KEY environment variable (the old file had it hard-coded: rotate that key); writes the same 21 columns as collector.py; skips tokens with no price; optional --loop. It only records a fixed list and fills few features, so use collector.py for training data.
33. data/market.csv repaired losslessly (117 rows; the old `source` column was dropped from rows that had it). The original is kept as data/market_original_corrupt.csv. Existing data is far too sparse to train on (irregular gaps, no buy/sell counts).
