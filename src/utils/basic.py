#!/usr/bin/env python

import operator
import os
import re
import sys
import time

from _csv import writer
from codecs import iterdecode
from csv import DictReader
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytz

from ia_agent import get_smart_summary

from .classes import CSVTrade, PriceOHLC, Trade

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'
DECIMALS = 3
TABLE_COL_WIDTH = 15
TABLE_ALIGN = "^"

# pytzutc = pytz.timezone('UTC')
LOCAL_TZ = pytz.timezone('Europe/Madrid')

# fix pair names
FIX_X_PAIR_NAMES = ['XETHEUR', 'XETH', 'XLTCEUR', 'XLTC', 'XETCEUR', 'XETC']  # 'XBTEUR', 'XDGEUR'
AUTOSTAKING_SUFFIXES = ('.FEUR',)
STAKING_SUFFIXES = ('.S', '.MEUR', '.SEUR', '.BEUR', *AUTOSTAKING_SUFFIXES)
XSTOCKS_SUFFIXES = '.TEUR'

HEADER_PRICES = ["TIMESTAMP", "O", "H", "L", "C", "VOL", "TRADES"]
HEADER_PRICES_KRAKEN = ["TIMESTAMP", "O", "H", "L", "C", "VWAP", "VOL", "TRADES"]
TRADES_CSV_HEADER = ['pair', 'time(UTC)', 'type', 'ordertype', 'price', 'cost', 'fee', 'vol']
HEADER_POSITIONS =['DATE', 'ASSET', 'SHARES', 'PRICE', 'AMOUNT', 'FEE']
RENAME_ASSET_MAPPING = {
    'XBTEUR': 'XXBTZEUR',
    'XRPEUR': 'XXRPZEUR',
    'ETCEUR': 'XETCZEUR',
    'XLMEUR': 'XXLMZEUR',
    'ETHEUR': 'XETHZEUR',
    'LTCEUR': 'XLTCZEUR',
}

OHLCV_DIR = './data/OHLCV_prices/'
PRICES_DIR = './data/prices_with_volume/'
REALISED_GAINS_FILE = './data/realised_gains_by_year.csv'
OHLC_MAX_CANDLES = 720  # Kraken OHLC returns at most the 720 most recent candles, whatever `since` is
OHLC_DAILY_SECONDS = 86400
KRAKEN_RATE_LIMIT_RETRIES = 5
KRAKEN_RATE_LIMIT_WAIT = 6  # Seconds: history calls cost 2 points and the counter decays 0.33 points/s at worst


class KrakenError(Exception):
    """Kraken answered with an error (after the rate limit retries)."""


class BCOLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def from_str_to_date(day: str) -> datetime.timestamp:
    return datetime.strptime(day, DATE_FORMAT).date()


def from_timestamp_to_str(timestamp: datetime.timestamp) -> str:
    return time.strftime(DATETIME_FORMAT, time.localtime(timestamp))


def from_timestamp_to_datetime(timestamp: datetime.timestamp) -> datetime:
    return datetime.fromtimestamp(timestamp)


def from_date_to_timestamp(day: datetime.date) -> datetime.timestamp:
    dt = datetime(year=day.year, month=day.month, day=day.day)
    return int(dt.timestamp())


def from_date_to_datetime_aware(day: datetime.date, hour: int = 0) -> datetime:
    dt = datetime(year=day.year, month=day.month, day=day.day, hour=hour, tzinfo=timezone.utc)
    return dt


def chunks(elem_list, n):
    n = max(1, n)
    return (elem_list[i : i + n] for i in range(0, len(elem_list), n))


def is_staked(name: str):
    return name.endswith(STAKING_SUFFIXES)


def is_auto_staked(name: str):
    return name.endswith(AUTOSTAKING_SUFFIXES)


def remove_staking_suffix(name: str):
    for suffix in STAKING_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    return name


# TODO not working on 5e-05
def count_zeros(value):
    float_str = str(value)
    return len(re.search(r'\d+\.(0*)', float_str).group(1))


def my_round(value: float, decimal_places=DECIMALS):
    if value is None:
        return None

    if abs(value) >= 1:
        return round(value, decimal_places)

    else:
        # decimal_places = count_zeros(value)
        return round(value, decimal_places + 3)


def percentage(a, b):
    """From a to b meaning a is lower than b. Example 100 to 120 is 20 %"""
    return ((float(b) - float(a)) / float(a)) * 100 if float(a) > 0 else 0


def entries_to_remove(entries, the_dict):
    for key in entries:
        if key in the_dict:
            del the_dict[key]


def timestamp_df_to_date_df(df: pd.DataFrame) -> pd.DataFrame:
    df.TIMESTAMP = pd.to_datetime(df.TIMESTAMP, unit='s').dt.date
    df.rename({'TIMESTAMP': 'DATE', 'C': 'PRICE'}, axis=1, inplace=True)
    return df


def read_prices_from_local_file(asset_name: str) -> pd.DataFrame:
    from pathlib import Path

    path = f'{PRICES_DIR}{asset_name}_DAILY_WITH_VOLUME.csv'
    file_path = Path(path)

    # Check if the file exists
    if file_path.exists():
        df = pd.read_csv(path)
        if "TIMESTAMP" in df.columns:
            df = timestamp_df_to_date_df(df=df)
            df = df.drop_duplicates(subset=['DATE'])
        else:
            df.DATE = pd.to_datetime(df.DATE).dt.date
    else:
        print(f"Prices for: {asset_name} taken from OHLC prices")
        df = pd.read_csv(f'{OHLCV_DIR}{asset_name}_1440.csv', names=HEADER_PRICES)[['TIMESTAMP', 'C', 'VOL']]
        df = timestamp_df_to_date_df(df=df)
        df = df.drop_duplicates(subset=['DATE'])
        df.to_csv(f'{PRICES_DIR}{asset_name}_DAILY_WITH_VOLUME.csv', index=False)

    df_prices = df[['DATE', 'PRICE', 'VOL']]
    df['VOL_EUR'] = df.VOL * df.PRICE
    df_volumes = df[['DATE', 'VOL_EUR']]
    return df_prices, df_volumes


def cancel_orders(kapi, order_type, orders) -> list:
    """Cancel on Kraken the orders of order_type. Returns the ones Kraken actually cancelled."""
    cancelled_orders = []
    for order in orders:
        if order.order_type == order_type and cancel_order(kapi=kapi, order=order):
            cancelled_orders.append(order)
    return cancelled_orders


def cancel_order(kapi, order) -> bool:
    """Cancel the order on Kraken. Returns True if Kraken cancelled it (no error and count > 0)."""
    req_data = {'txid': order.txid}
    close_order_result = kapi.query_private('CancelOrder', req_data)
    print_query_result('CancelOrder', close_order_result)
    return not close_order_result.get('error') and close_order_result['result'].get('count', 0) > 0


def get_max_price_since(kapi, pair_name: str, original_name: str, since_datetime: datetime) -> PriceOHLC | None:
    prices = []
    max_price_OHLC = None
    timestamp = since_datetime.timestamp()
    tickers_prices = kapi.query_public('OHLC', {'pair': pair_name, 'interval': 1440, 'since': timestamp})
    if not tickers_prices.get('result'):
        print(f'ERROR: Asset {pair_name} not found')
        return None
    prices_res = tickers_prices['result'].get(original_name) or tickers_prices['result'].get(pair_name)
    if not prices_res:
        print(f'ERROR: Prices are empty {pair_name}')
        return None
    for price in prices_res:
        day = from_timestamp_to_datetime(price[0]).date()
        priceOHLC = PriceOHLC(float(price[1]), float(price[2]), float(price[3]), float(price[4]), day)
        prices.append(priceOHLC)
        if not max_price_OHLC or max_price_OHLC.close < priceOHLC.close:
            max_price_OHLC = priceOHLC
    return max_price_OHLC


def get_max_price_from_csv_since(pair_name: str, since_datetime: datetime) -> float | None:
    df_prices, _ = read_prices_from_local_file(pair_name)
    return df_prices[df_prices.DATE.dt.date >= since_datetime.date()].PRICE.max()


def get_price_shares_from_order(order_string):
    words = order_string.split()
    shares = words[1]
    price = words[-1]
    return float(price), float(shares)


def get_fix_pair_name(pair_name, fix_x_pair_names, currency='EUR'):

    if pair_name.endswith('.T'):
        # return pair_name[:-2]  # Remove '.T' only available on USD
        return pair_name[:-2] + 'USD'

    if pair_name != 'XTZEUR' and pair_name.endswith('Z' + currency):
        pair_name = pair_name[:-4] + currency

    if pair_name[:2] == 'XX' or pair_name in fix_x_pair_names:
        pair_name = pair_name[1:]

    if is_staked(pair_name) or pair_name.endswith(currency):
        return pair_name

    return pair_name + currency


def load_from_csv(filename, assets_dict, fix_x_pair_names):
    """Load the trades CSV into its assets and return its last row as the newest trade (None if empty).

    The CSV must be sorted oldest first (summary_trades.py appends new trades in ascending time): each row is
    inserted on top, which leaves asset.trades newest first, and the last row is taken as the newest trade.
    """
    csv_file = open(filename, mode='rb')
    with csv_file:
        default_header = ['pair', 'time', 'type', 'ordertype', 'price', 'cost', 'fee', 'vol']
        csv_reader = DictReader(iterdecode(csv_file, 'utf-8'), fieldnames=default_header)
        # Skip the header
        next(csv_reader, None)

        trade = None  # Returned as-is when the CSV has no trades
        for asset_csv in csv_reader:
            asset_name = get_fix_pair_name(asset_csv['pair'], fix_x_pair_names)
            asset = assets_dict.get(asset_name)
            execution_time = datetime.strptime(asset_csv['time'], DATETIME_FORMAT)

            # Execution_time: Note CSV data is in UTC
            execution_time_local = pytz.UTC.localize(execution_time).astimezone(LOCAL_TZ)

            trade = Trade(
                asset_csv['type'],
                float(asset_csv['vol']),
                float(asset_csv['price']),
                amount=float(asset_csv['cost']),
                execution_datetime=execution_time_local,
            )
            if asset:
                asset.insert_trade_on_top(trade)

        csv_file.close()
    return trade


def get_trade_from_trade_word(word):
    # Example : 'buy 1.40000000 AVAXEUR @ limit 50.00'
    split_data = word.split('@')
    type_shares = split_data[0].split(' ')
    trade_type = type_shares[0]
    shares = float(type_shares[1])
    price = float(split_data[1].split(' ')[2])
    return Trade(trade_type, shares, price)


def print_query_result(endpoint, result):
    error = result.get('error')
    if error:
        print(f'Error:{error} found on call: {endpoint}')
        return
    print(f'Succeeded: {endpoint} records: {result["result"]["count"]}')


def compute_ranking(df, sessions: list[int]):
    """
    df input COLUMNS: [
        'NAME', 'LAST_TRADE', 'IBS', 'BLR', 'CURR_PRICE', 'AVG_B', 'AVG_S', 'MARGIN_A', 'S_TRADES', 'X_TRADES',
        'AVG_PRICE_<days>' and 'AVG_VOL_<days>' for each days in sessions,
        ]
    With P = CURR_PRICE, P_d = AVG_PRICE_<d>, V_d = AVG_VOL_<d>, s = min(sessions) and d over sessions:

        TREND = (1 + sum_d(P - P_d) / sum_d(|P - P_d|)) / 2
        VOL   = (1 + sum_{d != s}(V_s - V_d) / sum_{d != s}(|V_s - V_d|)) / 2
        TREND_VOL = TREND * VOL

    Both are in [0, 1]: 1 when P (or V_s) is above every other average, 0 when below all of them, 0.5 neutral.
    With a single term the ratio is just ±1, so with 2 sessions VOL is always 0 or 1. VOL needs at least
    2 sessions. If every difference is 0 the ratio is 0/0 = NaN, and the asset is dropped (printed).

    MARGIN_A is the asset result in EUR (Asset.margin_amount = sells + current balance - buys). With k the
    number of assets with MARGIN_A > 0 and r their rank by MARGIN_A (1 = lowest, ties averaged):

        MARGIN_P = 0      if MARGIN_A <= 0
        MARGIN_P = r / k  if MARGIN_A > 0

    So losing assets all score 0 and winning ones are spread evenly in (0, 1] (lowest 1/k, highest 1). Only the
    order counts, not the amount: an outlier (e.g. BTC with 4x the next margin) doesn't squash the rest, so no
    hand-tuned cap is needed (it replaces the old `6 * mean(MARGIN_A)` cap). MARGIN_P is already in [0, 1],
    so it is not min/max normalized like the other terms.

    Assets without sells (AVG_S == 0) or with a NaN RANKING term are dropped before the cross-asset stats
    (MARGIN_P rank and min/max normalization), so they don't change the scale of the ranked ones.
    """
    if len(sessions) < 2:
        raise ValueError(f'At least 2 sessions are needed to compute VOL, got {sessions}')
    price_cols = [f'AVG_PRICE_{days}' for days in sessions]
    vol_cols = [f'AVG_VOL_{days}' for days in sessions]
    shortest_vol_col = f'AVG_VOL_{min(sessions)}'
    longer_vol_cols = [col for col in vol_cols if col != shortest_vol_col]

    # Assets without sells are not ranked: drop them before any cross-asset stat (mean, min/max)
    df = df[df.AVG_S != 0.0].copy()

    df['P_BUY'] = (df.CURR_PRICE - df.AVG_B) / df.CURR_PRICE
    df['P_SELL'] = (df.CURR_PRICE - df.AVG_S) / df.CURR_PRICE
    df['BS_P'] = (df.AVG_S - df.AVG_B) / df.AVG_S
    df['BS_P'] = df['BS_P'].replace([np.inf, -np.inf], 0)
    # Compute TREND
    df['TREND_DIST'] = len(price_cols) * df.CURR_PRICE
    df['TREND_DIST_ABS'] = 0.0
    for col in price_cols:
        df['TREND_DIST'] -= df[col]
        df['TREND_DIST_ABS'] += (df.CURR_PRICE - df[col]).abs()
    df['TREND'] = df.TREND_DIST / df.TREND_DIST_ABS
    df['TREND'] = df['TREND'].replace([np.inf, -np.inf], 0)
    # Rescale from [-1, 1] to [0, 1] instead of truncating negatives to 0,
    # so a slightly negative raw TREND still reflects its relative magnitude.
    df['TREND'] = (df['TREND'] + 1) / 2
    # Compute VOL
    df['VOL_DIST'] = len(longer_vol_cols) * df[shortest_vol_col]
    df['VOL_DIST_ABS'] = 0.0
    for col in longer_vol_cols:
        df['VOL_DIST'] -= df[col]
        df['VOL_DIST_ABS'] += (df[shortest_vol_col] - df[col]).abs()
    df['VOL'] = df.VOL_DIST / df.VOL_DIST_ABS
    df['VOL'] = df['VOL'].replace([np.inf, -np.inf], 0)
    # Rescale from [-1, 1] to [0, 1] instead of truncating negatives to 0,
    # so a slightly negative raw VOL still reflects its relative magnitude.
    df['VOL'] = (df['VOL'] + 1) / 2

    df.loc[df.P_BUY <= -2, 'P_BUY'] = -2.0
    df.loc[df.P_SELL <= -2, 'P_SELL'] = -2.0
    df['TREND_VOL'] = df.TREND * df.VOL

    # A NaN in any RANKING term makes RANKING NaN: drop those assets before any cross-asset stat
    ranking_terms = ['P_BUY', 'P_SELL', 'BS_P', 'S_TRADES', 'MARGIN_A', 'X_TRADES', 'TREND_VOL']
    idx_nan = df[ranking_terms].isna().any(axis=1)
    nan_check_cols = ['CURR_PRICE', 'AVG_B', 'MARGIN_A', 'S_TRADES', 'X_TRADES', *price_cols, *vol_cols, 'TREND', 'VOL']
    for _, row in df[idx_nan].iterrows():
        nan_cols = [col for col in nan_check_cols if pd.isna(row[col])]
        print(f'{BCOLORS.WARNING}Asset {row.NAME} dropped from ranking, NaN in: {", ".join(nan_cols)}{BCOLORS.ENDC}')
    df = df[~idx_nan].copy()

    # Negative margins (losing assets) score 0. Positive ones score their percentile rank among them, already in
    # (0, 1], so an outlier (e.g. BTC) doesn't squash the rest and no cap is needed
    df['MARGIN_P'] = df.MARGIN_A.where(df.MARGIN_A > 0).rank(pct=True)
    df.loc[df.MARGIN_A <= 0, 'MARGIN_P'] = 0.0

    # ------NORMALIZATION--------
    def normalize(x):
        # A constant column would be 0/0 = NaN for every asset and drop them all: use 0 instead (NaN kept)
        x_range = x.max() - x.min()
        return (x - x.min()) / x_range if x_range != 0 else x - x.min()

    COLS_TO_NORM = ['P_BUY', 'P_SELL', 'BS_P', 'S_TRADES', 'X_TRADES']
    df[COLS_TO_NORM] = df[COLS_TO_NORM].apply(normalize)
    # ---------------------------

    df['RANKING'] = (
        df['P_BUY']
        + df['P_SELL']
        + df['BS_P']
        + df['S_TRADES']
        + df['MARGIN_P']
        + df['X_TRADES']
        # + df['TREND']
        # + df['VOL']
        + df['TREND_VOL']
    )

    # idx = df['RANKING'] < -10
    # df.loc[idx, 'RANKING'] = -10
    # Scaled to [0, 10]; a single asset (or all equal) is 0 instead of 0/0 = NaN
    df['RANKING'] = normalize(df['RANKING']) * 10

    df.sort_values(by=['RANKING'], inplace=True, ignore_index=True, ascending=False)
    ranking_df = df[['RANKING', 'NAME', 'LAST_TRADE', 'IBS', 'BLR', 'MARGIN_P', 'S_TRADES', 'X_TRADES', 'P_BUY', 'P_SELL', 'BS_P', 'TREND', 'VOL', 'TREND_VOL']]  # fmt: skip # noqa
    details_df = df[['RANKING', 'NAME', 'CURR_PRICE', 'AVG_B', 'AVG_S', 'MARGIN_A', *price_cols, 'TREND', *vol_cols, 'VOL']]  # fmt: skip # noqa

    return ranking_df, details_df


def read_trades_csv(filename, buy_trades, sell_trades):
    csv_file = open(filename, mode='rb')
    with csv_file:
        default_header = ['pair', 'time', 'type', 'ordertype', 'price', 'cost', 'fee', 'vol']

        csv_reader = DictReader(iterdecode(csv_file, 'utf-8'), fieldnames=default_header)

        next(csv_reader, None)

        trade = None  # Returned as-is when the CSV has no trades
        for asset_csv in csv_reader:
            trade = CSVTrade(
                asset_csv['pair'],
                asset_csv['time'],
                asset_csv['type'],
                asset_csv['price'],
                asset_csv['cost'],
                asset_csv['fee'],
                asset_csv['vol'],
            )
            if trade.type == 'buy':
                buy_trades.append(trade)
            else:
                sell_trades.append(trade)
        csv_file.close()
    return trade


def append_trades_to_csv(filename, trades_to_append):
    """Append trades to the trades CSV, writing the header first when the file is missing or empty.

    Readers (read_trades_csv, load_from_csv) skip the first line, so without the header the first trade of an
    empty file would be lost.
    """
    needs_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, mode='a+', newline='') as csvfile:
        append_writer = writer(csvfile)
        if needs_header:
            append_writer.writerow(TRADES_CSV_HEADER)
        for trade in trades_to_append:
            row = [
                trade.asset_name,
                trade.completed,
                trade.type,
                'limit',
                my_round(trade.price),
                my_round(trade.amount),
                my_round(trade.fee),
                my_round(trade.volume),
            ]
            append_writer.writerow(row)
        csvfile.close()


def read_realised_gains(filename: str) -> dict[int, float]:
    """Realised gain (G/L sell amount) per year, ascending by year."""
    df = pd.read_csv(filename).sort_values(by=['YEAR'])
    return dict(zip(df.YEAR, df.GL_SELL_AMOUNT))


def save_realised_gain(filename: str, year: int, amount: float):
    """Insert or replace the realised gain of year, keeping the other years."""
    gains = read_realised_gains(filename=filename) if os.path.exists(filename) else {}
    gains[year] = round(float(amount), 2)
    df = pd.DataFrame(sorted(gains.items()), columns=['YEAR', 'GL_SELL_AMOUNT'])
    df.to_csv(filename, index=False)


def get_new_prices(
    kapi,
    asset_name: str,
    timestamp_from: datetime.timestamp,
    with_volumes: bool = False,
) -> pd.DataFrame:
    """Daily OHLC candles of the asset since timestamp_from (unix), as a DataFrame with TIMESTAMP, C (and VOL).

    Kraken returns at most the 720 most recent candles (~2 years for daily ones), whatever `since` is, so an older
    timestamp_from leaves a gap: warns when the first candle starts more than one day after timestamp_from.
    """
    if asset_name in RENAME_ASSET_MAPPING:
        asset_name = RENAME_ASSET_MAPPING[asset_name]
    prices = kapi.query_public('OHLC', {'pair': asset_name, 'interval': 1440, 'since': timestamp_from})
    if not prices.get('result') or not prices['result'].get(asset_name):
        print(f'ERROR: OHLC for Asset {asset_name} not found')
        return None
    df_prices = pd.DataFrame.from_dict(prices['result'][asset_name])
    df_prices.columns = HEADER_PRICES_KRAKEN

    first_candle_time = int(df_prices.TIMESTAMP.iloc[0])
    if first_candle_time > timestamp_from + OHLC_DAILY_SECONDS:
        missing_from = datetime.fromtimestamp(timestamp_from, tz=timezone.utc).date()
        first_candle = datetime.fromtimestamp(first_candle_time, tz=timezone.utc).date()
        print(
            f'{BCOLORS.WARNING}OHLC GAP for {asset_name}: asked since {missing_from} but Kraken starts at '
            f'{first_candle} (it only returns the {OHLC_MAX_CANDLES} latest daily candles, or the pair is newer): '
            f'prices from {missing_from} to {first_candle} are missing{BCOLORS.ENDC}',
        )
    columns_to_get = ['TIMESTAMP', 'C']
    if with_volumes:
        columns_to_get = ['TIMESTAMP', 'C', 'VOL']
    df_prices = df_prices[columns_to_get]

    return df_prices


def count_sells_in_range(
    close_prices: pd.DataFrame,
    days: int,
    buy_perc: float,
    sell_perc: float,
    buy_limit: int = 0,
) -> int:
    latest_price = close_prices.DATE.iloc[-1]
    session_start = latest_price - timedelta(days=days)
    ref_df = close_prices[close_prices.DATE >= session_start]
    ref_price = ref_df.PRICE.iloc[0]
    ref_date = session_start
    sell_count = 0
    b_date = None
    s_date = None
    acc_buys = 0
    while 1:
        exp_sells = ref_df[ref_df.PRICE >= ref_price * (1 + sell_perc)]
        exp_buys = ref_df[ref_df.PRICE <= ref_price * (1 - buy_perc)]

        if not exp_sells.empty:
            s_date = exp_sells.DATE.iloc[0]
            s_price = exp_sells.PRICE.iloc[0]
        if not exp_buys.empty:
            b_date = exp_buys.DATE.iloc[0]
            b_price = exp_buys.PRICE.iloc[0]

        if s_date is None and b_date is None:
            break

        if b_date:
            ref_price = b_price
            ref_date = b_date
            acc_buys += 1 if buy_limit and acc_buys < buy_limit else 0

        if b_date is None or (b_date and s_date and s_date < b_date):
            ref_price = s_price
            ref_date = s_date
            if buy_limit and acc_buys:
                sell_count += 1
                acc_buys -= 1
            elif not buy_limit:
                sell_count += 1

        ref_df = ref_df[ref_df.DATE >= ref_date]
        b_date = None
        s_date = None
    return sell_count


def get_paginated_response_from_kraken(
    kapi,
    endpoint: str,
    dict_key: str,
    params: dict,
    pages: int | None,
    is_private: bool = True,
    timestamp_from=None,
) -> list[dict]:
    """Query up to `pages` pages (all of them if None) and return one dict per page.

    Pages and the records inside each dict keep Kraken's order (TradesHistory: most recent first). timestamp_from
    is sent as `start` (exclusive): only newer records. The offset `ofs` advances by the records actually received,
    so it works with any Kraken page size (50 by default). Stops on an empty page or when `count` records are read.
    Rate limit errors are retried; any other error raises KrakenError instead of returning the pages read so far,
    since they are only the newest records and saving them (e.g. to the trades CSV) would leave a gap.
    """
    records = []
    if timestamp_from:
        params['start'] = timestamp_from

    offset = 0
    page = 0
    while pages is None or page < pages:
        params['ofs'] = offset
        response = query_kraken_with_retry(kapi=kapi, endpoint=endpoint, params=params, is_private=is_private)
        if response.get('error'):
            raise KrakenError(f'Kraken {endpoint} error {response["error"]} after reading {offset} records')

        result = response['result']
        results = result.get(dict_key)
        if not results:
            return records

        records.append(results)
        offset += len(results)
        page += 1
        if 'count' in result and offset >= int(result['count']):
            return records

    return records


def query_kraken_with_retry(kapi, endpoint: str, params: dict, is_private: bool) -> dict:
    """Query Kraken, waiting and retrying while it answers with a rate limit error. Returns the last response."""
    for attempt in range(KRAKEN_RATE_LIMIT_RETRIES + 1):
        if is_private:
            response = kapi.query_private(endpoint, params)
        else:
            response = kapi.query_public(endpoint, params)

        is_rate_limit = any('Rate limit' in error for error in response.get('error', []))
        if not is_rate_limit or attempt == KRAKEN_RATE_LIMIT_RETRIES:
            return response
        print(f'{endpoint}: Kraken rate limit, waiting {KRAKEN_RATE_LIMIT_WAIT} s')
        time.sleep(KRAKEN_RATE_LIMIT_WAIT)
    return response


def smart_round(number: float | int | Decimal | None) -> str:
    """
    Intelligently rounds a number for display.
    - Accepts int, float, and Decimal robustly.
    - Uses suffixes K (thousands), M (millions), B (billions).
    - For small numbers (< 1), displays the first significant decimals.
    - For intermediate numbers, uses 2 decimal places by default.

    Args:
        number: The number to format.

    Returns:
        The formatted number as a string.
    """

    if number is None:
        return "N/A"

    try:
        # Safely convert to Decimal to maintain float precision
        # and handle all numeric types uniformly.
        num = Decimal(str(number))
    except Exception:
        # If it cannot be converted, it's not a valid number. Return the original.
        return str(number)

    # Handle NaN and Infinity explicitly before comparisons
    if num.is_nan():
        return "NaN"
    if num.is_infinite():
        return "Infinity" if num > 0 else "-Infinity"
    if num.is_zero():
        return "0"

    # Handle the sign
    sign = "-" if num < 0 else ""
    num = abs(num)

    if num >= 1_000_000_000:
        formatted_num = f"{num / Decimal('1E9'):.2f}B"
    elif num >= 1_000_000:
        formatted_num = f"{num / Decimal('1E6'):.2f}M"
    elif num >= 1_000:
        formatted_num = f"{num / Decimal('1E3'):.2f}K"
    elif num < 1:
        if num > Decimal('1E-12'):  # Avoid errors with extremely small numbers
            # Use Decimal's log10 for precision
            log_val = num.log10()
            # to_integral_value is the equivalent of floor() for Decimal
            decimals = -int(log_val.to_integral_value(rounding='ROUND_FLOOR')) + 1
            formatted_num = f"{num:.{decimals}f}"
        else:
            formatted_num = "0"  # Treat as zero if extremely small
    else:  # Numbers between 1 and 999.99...
        formatted_num = f"{num:.2f}"

    return sign + formatted_num


def get_visual_len(text: str) -> int:
    """Calculates the real visible length of a string, ignoring ANSI color codes."""
    # Regex to match ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', str(text)))


def print_table(
    data: list[dict],
    columns: list[tuple],
    apply_smart_round: bool = False,
    title: str = "REPORT",
    auto_adjust: bool = True,
):
    """
    Renders a professional CLI table with multiline header support and ANSI color awareness.

    Args:
        data (list[dict]): Row data. Example: [{"id": 1, "val": "\033[92m10.5\033[0m"}]
        columns (list[tuple]): (key, label, [alignment]).
            Label can contain '\\n'. Alignment: '<', '>', '^'.
        apply_smart_round (bool): Applies smart_round() to values if True.
        title (str): Table title centered at the top.
        auto_adjust (bool): Dynamic width calculation based on content.

    Example:
        >>> cols = [("id", "ID", "<"), ("val", "Price\\n(USD)", ">")]
        >>> data = [{"id": "01", "val": "50.5"}]
        >>> print_table(data, cols, title="MARKET")
    """

    if not data:
        print(f"\n--- {title} ---\nNo data available.")
        return

    # 1. Configuration & Width Initialization
    col_widths = {}
    col_aligns = {}
    header_lines_map = {}
    max_header_height = 0

    for col in columns:
        key, label = col[0], col[1]
        align = col[2] if len(col) > 2 else "<"
        col_aligns[key] = align

        # Multiline header processing
        lines = str(label).split('\n')
        header_lines_map[key] = lines
        max_header_height = max(max_header_height, len(lines))

        # Base width from header
        col_widths[key] = max(len(line) for line in lines) if auto_adjust else 20

    # 2. Single-pass Data Processing (Rounding + Width Calculation)
    for item in data:
        for key in col_widths:
            # Safety check: if key doesn't exist, use empty string
            raw_val = item.get(key, "")

            if apply_smart_round:
                # Update item directly with rounded value
                item[key] = smart_round(raw_val)

            val_str = str(item.get(key, ""))
            v_len = get_visual_len(val_str)
            if auto_adjust and v_len > col_widths[key]:
                col_widths[key] = v_len

    # Add horizontal padding
    for key in col_widths:
        col_widths[key] += 2

    # Total layout width for separators
    total_width = sum(col_widths.values()) + (3 * (len(columns) - 1))

    # 3. RENDER PHASE
    # Title
    print(f"\n{title.center(total_width, '-')}")

    # Headers
    for i in range(max_header_height):
        header_row = []
        for col in columns:
            key = col[0]
            lines = header_lines_map[key]
            content = lines[i] if i < len(lines) else ""
            header_row.append(f"{content:{col_aligns[key]}{col_widths[key]}}")
        print(" | ".join(header_row))

    print("-" * total_width)

    # Rows
    for item in data:
        row_cells = []
        for col in columns:
            key = col[0]
            # Validation: Handle missing keys gracefully
            val = str(item.get(key, "---"))

            v_len = get_visual_len(val)
            pad_size = col_widths[key] - v_len
            align = col_aligns[key]

            # Manual padding for ANSI string support
            if align == ">":
                cell = (" " * pad_size) + val
            elif align == "^":
                l_pad = pad_size // 2
                cell = (" " * l_pad) + val + (" " * (pad_size - l_pad))
            else:
                cell = val + (" " * pad_size)
            row_cells.append(cell)

        print(" | ".join(row_cells))

    print("-" * total_width)


def print_smart_df(df: pd.DataFrame, exclude_columns: list[str] = [], title: str = "REPORT"):
    """
    Prints a formatted DataFrame with rounded numeric values and a centered title.
    """

    # 1. Identify numeric columns, excluding those specified in the list
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

    # 2. Apply rounding logic using the smart_round function
    printable_df = df.copy()
    printable_df[numeric_columns] = printable_df[numeric_columns].map(smart_round)

    # 3. Convert the DataFrame to string without index to measure its dimensions
    df_string = printable_df.to_string(index=False)

    # 4. Determine the maximum width of the table to center the title
    # We use splitlines() to get each row and max() to find the longest one
    lines = df_string.splitlines()
    table_width = max(len(line) for line in lines) if lines else len(title)

    # 5. Output the centered title and the table
    print(f"\n{title.center(table_width)}")
    print("-" * table_width)  # Header separator
    print(df_string)
    print("-" * table_width + "\n")  # Footer separator


def print_smart_df_multicolor(
    df: pd.DataFrame,
    exclude_columns: list[str] = [],
    title: str = "REPORT",
    highlight_rules: list[dict] | None = None,
):
    """
    Prints a formatted DataFrame with correct alignment and ANSI highlighting.

    Example:
        >>> rules = [{'column': 'ACCUM_B', 'op': '>', 'threshold': 5, 'color': '\033[42;30m'}]
        >>> print_smart_df_multicolor(df, highlight_rules=rules)
    """

    ops = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
    }

    printable_df = df.copy()

    # 1. Numeric formatting
    numeric_cols = printable_df.select_dtypes(include=['number']).columns
    cols_to_format = [col for col in numeric_cols if col not in exclude_columns]
    if cols_to_format:
        printable_df[cols_to_format] = printable_df[cols_to_format].round(2)

    # 2. Get the table as a list of strings WITHOUT color first (to keep alignment)
    # index=False ensures we don't have the row numbers shifting things
    table_lines = printable_df.to_string(index=False).splitlines()

    if not table_lines:
        print(f"\n{title}\nEMPTY\n")
        return

    header = table_lines[0]
    data_lines = table_lines[1:]
    table_width = len(header)

    # 3. Apply colors to the data lines based on the original DataFrame logic
    colored_lines = []
    for i, line in enumerate(data_lines):
        styled_line = line
        if highlight_rules:
            # Get the original row to check conditions
            original_row = printable_df.iloc[i]

            for rule in highlight_rules:
                col = rule.get('column')
                op_sym = rule.get('op')
                threshold = rule.get('threshold')
                ansi_code = rule.get('color')

                if col in original_row and op_sym in ops and ansi_code:
                    if ops[op_sym](original_row[col], threshold):
                        # Wrap the ALREADY PADDED line with the ANSI code
                        styled_line = f"{ansi_code}{line}{BCOLORS.ENDC}"
                        break  # Only the first matching rule is applied

        colored_lines.append(styled_line)

    # 4. Final output with centered title
    print(f"\n{title.center(table_width)}")
    print("-" * table_width)
    print(header)  # Header is never colored in this logic
    print("-" * table_width)
    for cl in colored_lines:
        print(cl)
    print("-" * table_width + "\n")


def print_separator():
    print("\n" + "-" * 100 + "\n")


def run_smart_summary(
    positions,
    death_assets,
    ia_agent,
    captured_output,
    local_tz,
    output_dir,
):
    print(f'\n ***** SMART SUMMARY ({ia_agent}) ***** ')
    smart_summary_time_start = datetime.now(timezone.utc)
    agent_response = get_smart_summary(positions=positions, death_assets=death_assets, ia_agent=ia_agent)
    print(f'Agent response: \n {agent_response}')
    elapsed_time_smart_summary = datetime.now(timezone.utc) - smart_summary_time_start
    print(f'Smart summary latency: {elapsed_time_smart_summary}')

    sys.stdout = sys.stdout.streams[0]
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{datetime.now(local_tz).strftime('%Y%m%d')}.md")
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(captured_output.getvalue())
