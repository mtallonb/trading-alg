# trading-alg

Personal Kraken/eToro trading scripts (EUR): read balance, orders and trades, rank assets, suggest or cancel orders.

## Running

- `poetry run python menu.py` (Python 3.11) launches each `src/` script as a subprocess, from the repo root (`./data/...` paths, `src/` on path).
- **Never run `src/orders.py` without asking**: it uses real Kraken keys (`data/keys/kraken.key`) and, with `AUTO_CANCEL_*_ORDER = True`, cancels real orders after an `input()`.
- Safe checks: `python3 -m py_compile src/orders.py`; `poetry run python -c "import sys; sys.path.insert(0, 'src'); import orders"` (import has no side effects).
- `src/summary_trades.py` reads Kraken `TradesHistory` with real keys and appends to `trades_2026.csv`: don't run it (not even with a fake `krakenex`); test by importing it and calling `compute_gain_loss`/`compute_pair_gains` on the CSV read with `read_trades_csv`.
- ruff is not installed; config in `ruff.toml`.

## Layout

- `src/utils/basic.py`: helpers (paginated Kraken, trades CSV, ranking, tables, rounding). `src/utils/classes.py`: `Asset`, `Order`, `Trade`, `MAPPING_NAMES`, `OP_BUY`/`OP_SELL`. `src/backends/`: `kraken.py`, `etoro.py`.
- `data/`: daily prices/volumes per asset, `trades_2026.csv`, smart summary outputs. `src/config.env` and `*.key` are gitignored.
- `src/orders.py`: UPPER_CASE config constants at top; one function per phase (`build_assets` → `fill_prices_and_volumes` → `fill_staking_info` → `fill_orders` → `fill_trades` → `remove_assets_without_trades` → `build_ranking_rows` → `compute_and_print_ranking` → `print_orders_proximity` → `print_orders_to_create` → `print_cash_summary` → `print_time_summary`, plus `print_last_trades`); `main()` only orchestrates, no module-level state. Timing via `with timer(timings=..., label=...)`; shown labels/order in `TIME_SUMMARY_LABELS`, missing phases print `skipped`. `Tee` copies stdout into `captured_output` for the smart summary.
- `src/summary_trades.py`: yearly G/L per pair (FIFO/LIFO) from the trades CSV plus new Kraken trades. `main()` → `read_trades_csv` → `fetch_new_trades` → `compute_pair_gains` (calls `compute_gain_loss(method=FIFO|LIFO)` per pair) → `print_summary` → `print_pair_gains` → `append_trades_to_csv`.

## Conventions

- **Always pass keyword arguments** to project functions/methods (`fill_orders(open_orders=open_orders, assets_dict=assets_dict)`); not for builtins, pandas, `str` methods or `*args` (`Tee`).
- 120-char lines, trailing commas in multi-line calls (ruff `COM`), quotes preserved (mostly single). Comments always in English (older ones are still mixed with Spanish; translate them when touching that code).
- Delete unused variables/constants.
- Refactors preserve behavior: move code as-is, diff against `HEAD` ignoring indentation; fix bugs in separate steps.

## `src/orders.py` refactor (in progress)

Done: phase functions + `main()`, `timer`, unused vars removed, keyword arguments, known bugs fixed (tickers without asset, `close_prices`/`close_volumes` `None`, `last_trade_from_csv` `None`, trades newer than the CSV inserted oldest-first so `trades[0]` is the newest, untracked-pair orders only counted in totals, `load_from_csv` on empty CSV, `Asset.add_trade` fuses partials with `trades[-1]`), averages parametrized with `AVG_SESSIONS` (≥ 2; VOL compares the shortest with the rest) and `EXPECTED_SELLS_DAYS`, ranking rows are dicts. `Asset.trades` is newest first. Pending, in order:

1. Duplication: `XX...` name normalization, prices/volumes loading, buy/sell order accumulation (→ `Asset.add_order`), `check_buys_limit` computed twice, 3 proximity tables, buy/sell cancellation (→ `check_and_cancel`), `warn()`/`fail()` for `BCOLORS`.
2. Idioms: side-effect comprehensions/ternaries, redundant checks, `sys.stdout` restored in one branch only (→ context manager).
3. Move config to its own module.

## `src/balances.py`

Yearly gain % on the average daily balance. Writes `deposits.csv`/`withdrawals.csv` and price files with real keys: test on a copy of `data/` with a fake `krakenex`. `main()` → `load_flows` → `read_trades` → `build_positions` → `add_daily_cash` → `print_summary` → `print_gains_by_year`. Bugs fixed: partial candle of the last stored day kept forever (now refetched, `keep='last'`), year start balance taken from Jan 1 EOD (now previous year's close), empty ledger `KeyError`, daily cash ignored trade/flow fees, excluded assets (WAVES, BSV) dropped out of cash, spot↔earn `transfer` ledger rows counted as flows (lost 1,000 EUR moved to `EUR.M`). Daily cash now matches the Kraken ledger `ZEUR` balance within a few EUR and equals the `CASH` summary. WAVES is valued with local prices until they end (2024-03-31); Kraken converted it to BTC on 2024-10-11/14, added via `ASSET_CONVERSIONS` (zero-cost trades for positions only, not cash); `CASH_ONLY_ASSETS` (BSV airdrop) only count as cash. Realised gain per year is read from `data/realised_gains_by_year.csv` (`REALISED_GAINS_FILE`), where `summary_trades.py` upserts `totals['gl_sell_amount']` for `YEAR` (not on filtered runs); 2021 is the current `gl_sell_amount` (4,689.05; the old hand-typed value was ~FIFO); other years before 2026 were typed by hand and don't match the current calculation, so don't recompute them.

## `src/summary_trades.py` refactor (in progress)

Done: phase functions + `main()` (no module-level state, import has no side effects), UPPER_CASE config, unused vars removed, keyword arguments, English comments, FIFO/LIFO merged into `compute_gain_loss(method=...)` with `find_buy_index` (logs unified: `(FIFO)`/`(LIFO)` headers, `|` separators). Bugs fixed: new Kraken trades added oldest-first, pair with sells but no buys (`buy` unbound), `is_position_closed` same rule for both methods (open buys worth ≤ `CLOSED_POSITION_MAX_AMOUNT` EUR, no `XXBTZEUR` special case), FIFO only uses buys before the sell, `read_trades_csv` returns `None` on empty CSV. Verified: results on the real CSV unchanged. Pending:

1. Pairs iterated from a `set`, so per-pair log order changes between runs (sort them).
2. `append_trades_to_csv` writes no header: on a fully empty file the first trade would be read as the header.
