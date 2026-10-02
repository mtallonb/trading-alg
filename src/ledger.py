#!/usr/bin/python3
"""Rebuild the whole Kraken ledger (every type and asset) in LEDGER_FILE, a few pages further back on each run.

Each run reads every record newer than the file and PAGES_PER_RUN pages older than its oldest record, so the file
is always a contiguous range of the ledger that grows back in time until Kraken returns nothing older.
"""

import math

from pathlib import Path

import krakenex
import pandas as pd

from utils.basic import get_paginated_response_from_kraken

LEDGER_FILE = './data/ledger.csv'
KEY_FILE = './data/keys/kraken.key'
PAGES_PER_RUN = 5  # Older pages (50 records each) read per run
LEDGER_COLUMNS = ['LEDGER_ID', 'REFID', 'TIME', 'DATETIME', 'TYPE', 'SUBTYPE', 'ACLASS', 'ASSET', 'AMOUNT', 'FEE', 'BALANCE']  # noqa # fmt: skip


def read_ledger_file(filename: str) -> pd.DataFrame:
    if not Path(filename).exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    # Kraken amounts kept as strings (exact decimals)
    return pd.read_csv(filename, dtype={'AMOUNT': str, 'FEE': str, 'BALANCE': str})


def pages_to_df(pages: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{'ledger_id': ledger_id, **record} for page in pages for ledger_id, record in page.items()])
    if df.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    df.columns = [column.upper() for column in df.columns]
    df['DATETIME'] = pd.to_datetime(df.TIME, unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')  # UTC
    return df[LEDGER_COLUMNS]


def read_ledger_pages(kapi, pages: int | None, start: int | None = None, end: int | None = None) -> pd.DataFrame:
    """Ledger records (all types) with start < time <= end (Kraken: `start` exclusive, `end` inclusive)."""
    params = {'type': 'all'}
    if end is not None:
        params['end'] = end
    pages_read = get_paginated_response_from_kraken(
        kapi=kapi,
        endpoint='Ledgers',
        dict_key='ledger',
        params=params,
        pages=pages,
        timestamp_from=start,
    )
    return pages_to_df(pages=pages_read)


def save_ledger(df_ledger: pd.DataFrame, df_new: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Add df_new to the ledger (records already in it come back from Kraken at the `start`/`end` second)."""
    df_new = df_new[~df_new.LEDGER_ID.isin(df_ledger.LEDGER_ID)]
    df_ledger = pd.concat([df for df in (df_ledger, df_new) if not df.empty] or [df_ledger], ignore_index=True)
    df_ledger = df_ledger.sort_values(by=['TIME', 'LEDGER_ID'], ignore_index=True)
    df_ledger.to_csv(filename, index=False)
    return df_ledger


def update_ledger_file(kapi, filename: str = LEDGER_FILE, pages_per_run: int = PAGES_PER_RUN) -> pd.DataFrame:
    df_ledger = read_ledger_file(filename=filename)
    count_before = len(df_ledger)

    # 1. Every record newer than the file (all pages): saved before going back so the range stays contiguous
    if not df_ledger.empty:
        df_newer = read_ledger_pages(kapi=kapi, pages=None, start=int(df_ledger.TIME.max()))
        df_ledger = save_ledger(df_ledger=df_ledger, df_new=df_newer, filename=filename)
    count_newer = len(df_ledger) - count_before

    # 2. pages_per_run pages older than the file (the newest ones on the first run). `end` is inclusive and rounded
    # up so no record of the oldest second is lost: the oldest one comes back and each page adds 49 new records
    end = math.ceil(df_ledger.TIME.min()) if not df_ledger.empty else None
    df_older = read_ledger_pages(kapi=kapi, pages=pages_per_run, end=end)
    count_older = len(df_older[~df_older.LEDGER_ID.isin(df_ledger.LEDGER_ID)])
    df_ledger = save_ledger(df_ledger=df_ledger, df_new=df_older, filename=filename)

    print(f'Ledger: {count_newer} new records, {count_older} older records, {len(df_ledger)} in {filename}')
    if not df_ledger.empty:
        print(f'From {df_ledger.DATETIME.iloc[0]} to {df_ledger.DATETIME.iloc[-1]} (UTC)')
    if end is not None and count_older == 0:
        print('Complete: Kraken has no records older than the file')
    return df_ledger


def main():
    kapi = krakenex.API()
    kapi.load_key(KEY_FILE)
    update_ledger_file(kapi=kapi)


if __name__ == '__main__':
    main()
