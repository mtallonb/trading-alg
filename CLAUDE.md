# trading-alg

Personal Kraken/eToro trading scripts (EUR): read balance, orders and trades, rank assets, suggest or cancel orders.

## Running

- `poetry run python menu.py` (Python 3.11) launches each `src/` script as a subprocess, from the repo root (`./data/...` paths, `src/` on path).
- **Never run `src/orders.py` without asking**: it uses real Kraken keys (`data/keys/kraken.key`) and, with `AUTO_CANCEL_*_ORDER = True`, cancels real orders after an `input()`.
- Safe checks: `python3 -m py_compile src/orders.py`; `poetry run python -c "import sys; sys.path.insert(0, 'src'); import orders"` (import has no side effects).
- ruff is not installed; config in `ruff.toml`.

## Layout

- `src/utils/basic.py`: helpers (paginated Kraken, trades CSV, ranking, tables, rounding). `src/utils/classes.py`: `Asset`, `Order`, `Trade`, `MAPPING_NAMES`, `OP_BUY`/`OP_SELL`. `src/backends/`: `kraken.py`, `etoro.py`.
- `data/`: daily prices/volumes per asset, `trades_2026.csv`, smart summary outputs. `src/config.env` and `*.key` are gitignored.
- `src/orders.py`: UPPER_CASE config constants at top; one function per phase (`build_assets` → `fill_prices_and_volumes` → `fill_staking_info` → `fill_orders` → `fill_trades` → `remove_assets_without_trades` → `build_ranking_rows` → `compute_and_print_ranking` → `print_orders_proximity` → `print_orders_to_create` → `print_cash_summary` → `print_time_summary`, plus `print_last_trades`); `main()` only orchestrates, no module-level state. Timing via `with timer(timings=..., label=...)`; shown labels/order in `TIME_SUMMARY_LABELS`, missing phases print `skipped`. `Tee` copies stdout into `captured_output` for the smart summary.

## Conventions

- **Always pass keyword arguments** to project functions/methods (`fill_orders(open_orders=open_orders, assets_dict=assets_dict)`); not for builtins, pandas, `str` methods or `*args` (`Tee`).
- 120-char lines, trailing commas in multi-line calls (ruff `COM`), quotes preserved (mostly single). Mixed Spanish/English comments: follow surroundings.
- Delete unused variables/constants.
- Refactors preserve behavior: move code as-is, diff against `HEAD` ignoring indentation; fix bugs in separate steps.

## `src/orders.py` refactor (in progress)

Done: phase functions + `main()`, `timer`, unused vars removed, keyword arguments, known bugs fixed (tickers without asset, `close_prices`/`close_volumes` `None`, `last_trade_from_csv` `None`, trades newer than the CSV inserted oldest-first so `trades[0]` is the newest, untracked-pair orders only counted in totals, `load_from_csv` on empty CSV, `Asset.add_trade` fuses partials with `trades[-1]`). `Asset.trades` is newest first. Pending, in order:

1. Duplication: `XX...` name normalization, prices/volumes loading, buy/sell order accumulation (→ `Asset.add_order`), `check_buys_limit` computed twice, 200/50/10 averages (→ sessions list), 3 proximity tables, buy/sell cancellation (→ `check_and_cancel`), `warn()`/`fail()` for `BCOLORS`.
2. Idioms: positional ranking rows tied to `ranking_cols` (→ dicts), side-effect comprehensions/ternaries, redundant checks, `sys.stdout` restored in one branch only (→ context manager).
3. Move config to its own module.
