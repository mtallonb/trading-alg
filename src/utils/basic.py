#!/usr/bin/env python

import re
import time

from _csv import writer
from codecs import iterdecode
from csv import DictReader
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytz

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

HEADER_PRICES = ["TIMESTAMP", "O", "H", "L", "C", "VOL", "TRADES"]
HEADER_PRICES_KRAKEN = ["TIMESTAMP", "O", "H", "L", "C", "VWAP", "VOL", "TRADES"]
HEADER_POSITIONS = ['DATE', 'ASSET', 'SHARES', 'PRICE', 'AMOUNT', 'FEE']
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


def cancel_orders(kapi, order_type, orders):
    for order in orders:
        if order.order_type == order_type:
            cancel_order(kapi, order)


def cancel_order(kapi, order):
    req_data = {'txid': order.txid}
    close_order_result = kapi.query_private('CancelOrder', req_data)
    print_query_result('CancelOrder', close_order_result)
    return


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
    if pair_name != 'XTZEUR' and pair_name.endswith('Z' + currency):
        pair_name = pair_name[:-4] + currency

    if pair_name[:2] == 'XX' or pair_name in fix_x_pair_names:
        pair_name = pair_name[1:]

    if is_staked(pair_name) or pair_name.endswith(currency):
        return pair_name

    return pair_name + currency


def load_from_csv(filename, assets_dict, fix_x_pair_names):
    csv_file = open(filename, mode='rb')
    with csv_file:
        default_header = ['pair', 'time', 'type', 'ordertype', 'price', 'cost', 'fee', 'vol']
        csv_reader = DictReader(iterdecode(csv_file, 'utf-8'), fieldnames=default_header)
        # Skip the header
        next(csv_reader, None)

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


def compute_ranking(df):
    """
    df input COLUMNS: [
        'NAME', 'LAST_TRADE', 'IBS', 'BLR', 'CURR_PRICE', 'AVG_B', 'AVG_S', 'MARGIN_A', 'S_TRADES', 'X_TRADES',
        'AVG_PRICE_200', 'AVG_PRICE_50', 'AVG_PRICE_10', 'AVG_VOL_200', 'AVG_VOL_50', 'AVG_VOL_10',
        ]
    """

    df['MARGIN_P'] = df.MARGIN_A
    df['P_BUY'] = (df.CURR_PRICE - df.AVG_B) / df.CURR_PRICE
    df['P_SELL'] = (df.CURR_PRICE - df.AVG_S) / df.CURR_PRICE
    df['BS_P'] = (df.AVG_S - df.AVG_B) / df.AVG_S
    df['BS_P'] = df['BS_P'].replace([np.inf, -np.inf], 0)
    # Compute TREND
    df['TREND_DIST'] = 3 * df.CURR_PRICE - df.AVG_PRICE_200 - df.AVG_PRICE_50 - df.AVG_PRICE_10
    df['TREND_DIST_ABS'] = (df.CURR_PRICE - df.AVG_PRICE_200).abs() + (df.CURR_PRICE - df.AVG_PRICE_50).abs() + (df.CURR_PRICE - df.AVG_PRICE_10).abs()  # fmt: skip # noqa
    df['TREND'] = df.TREND_DIST / df.TREND_DIST_ABS
    df['TREND'] = df['TREND'].replace([np.inf, -np.inf], 0)
    # Compute TREND_VOL
    df['VOL_DIST'] = 2 * df.AVG_VOL_10 - df.AVG_VOL_200 - df.AVG_VOL_50
    df['VOL_DIST_ABS'] = (df.AVG_VOL_10 - df.AVG_VOL_200).abs() + (df.AVG_VOL_10 - df.AVG_VOL_50).abs()
    df['VOL'] = df.VOL_DIST / df.VOL_DIST_ABS
    df['VOL'] = df['VOL'].replace([np.inf, -np.inf], 0)

    df.loc[df.VOL < 0, 'VOL'] = 0
    df.loc[df.TREND < 0, 'TREND'] = 0
    df.loc[df.P_BUY <= -2, 'P_BUY'] = -2.0
    df.loc[df.P_SELL <= -2, 'P_SELL'] = -2.0
    df.loc[df.MARGIN_P > 6 * df.MARGIN_A.mean(), 'MARGIN_P'] = 6 * df.MARGIN_A.mean()
    df['TREND_VOL'] = df.TREND * df.VOL

    # ------NORMALIZATION--------
    COLS_TO_NORM = ['P_BUY', 'P_SELL', 'BS_P', 'S_TRADES', 'MARGIN_P', 'X_TRADES']
    df[COLS_TO_NORM] = df[COLS_TO_NORM].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
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

    idx_avg_s_zeros = df['AVG_S'] == 0.0
    df.loc[idx_avg_s_zeros, 'RANKING'] = np.nan
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["RANKING"], how="all", inplace=True)
    # idx = df['RANKING'] < -10
    # df.loc[idx, 'RANKING'] = -10
    df['RANKING'] = df['RANKING'] - df['RANKING'].min()
    df['RANKING'] = (df['RANKING'] / df['RANKING'].max()) * 10

    df.sort_values(by=['RANKING'], inplace=True, ignore_index=True, ascending=False)
    ranking_df = df[['RANKING', 'NAME', 'LAST_TRADE', 'IBS', 'BLR', 'MARGIN_P', 'S_TRADES', 'X_TRADES', 'P_BUY', 'P_SELL', 'BS_P', 'TREND', 'VOL', 'TREND_VOL']]  # fmt: skip # noqa
    details_df = df[['RANKING', 'NAME', 'CURR_PRICE', 'AVG_B', 'AVG_S', 'MARGIN_A', 'AVG_PRICE_200', 'AVG_PRICE_50', 'AVG_PRICE_10', 'TREND', 'AVG_VOL_200', 'AVG_VOL_50', 'AVG_VOL_10','VOL']]  # fmt: skip # noqa

    return ranking_df, details_df


def read_trades_csv(filename, buy_trades, sell_trades):
    csv_file = open(filename, mode='rb')
    with csv_file:
        default_header = ['pair', 'time', 'type', 'ordertype', 'price', 'cost', 'fee', 'vol']

        csv_reader = DictReader(iterdecode(csv_file, 'utf-8'), fieldnames=default_header)

        next(csv_reader, None)

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
    # Write latest trades to CSV
    with open(filename, mode='a+', newline='') as csvfile:
        append_writer = writer(csvfile)
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


def get_new_prices(
    kapi,
    asset_name: str,
    timestamp_from: datetime.timestamp,
    with_volumes: bool = False,
) -> pd.DataFrame:
    RENAME_ASSET_MAPPING
    if asset_name in RENAME_ASSET_MAPPING:
        asset_name = RENAME_ASSET_MAPPING[asset_name]
    # If timestamp_from is higher than 2 years display a warning
    prices = kapi.query_public('OHLC', {'pair': asset_name, 'interval': 1440, 'since': timestamp_from})
    if not prices.get('result'):
        return None
    df_prices = pd.DataFrame.from_dict(prices['result'][asset_name])
    df_prices.columns = HEADER_PRICES_KRAKEN
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
    pages: int,
    records_per_page: int,
    is_private: bool = True,
    timestamp_from=None,
) -> list[dict]:
    records = []
    if timestamp_from:
        params['start'] = timestamp_from

    for page in range(pages):
        params['ofs'] = records_per_page * page
        if is_private:
            response = kapi.query_private(endpoint, params)
        else:
            response = kapi.query_public(endpoint, params)

        results = response.get('result').get(dict_key)
        if results:
            records.append(results)
        else:
            return records

    return records


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


# def print_table(data: list[dict], columns: list[tuple], title: str = "REPORT", auto_adjust: bool = True):
#     """
#     Renders a formatted CLI table. Optimized to avoid redundant iterations when auto_adjust is False.
#     Parameters:
#     - data: A list of dictionaries containing the data to be displayed in the table.
#     - columns: A list of tuples, where each tuple contains a key and label for a column in the table.
#         Addtionally the alignment can be specified as the third element with the default being "^" (centered).
#         But we can use <, >, ^ for left, right, and center alignment.
#     - title: The title of the table (default: "REPORT").
#     - auto_adjust: A boolean indicating whether to automatically adjust column widths (default: True).

#     Example:
#     data = [{"key1": "Value 1", "key2": "Value 2", "key3": "Value 3"}]
#     columns = [("key1", "Column 1"), ("key2", "Column 2", "<"), ("key3", "Column 3", ">")]
#     print_table(data, columns, "Report Title", auto_adjust=False)
#     """

#     DEFAULT_ALIGN = "^"

#     if not data:
#         print(f"\n--- {title} ---\nNo data available.")
#         return

#     col_widths = {}
#     col_aligns = {}

#     # 1. Initialize alignments and widths
#     for col in columns:
#         col_key, label = col[0], col[1]
#         col_aligns[col_key] = col[2] if len(col) > 2 else DEFAULT_ALIGN

#         # If auto_adjust is False, we set the width immediately and move on
#         if not auto_adjust:
#             col_widths[col_key] = TABLE_COL_WIDTH
#         else:
#             # If True, we start with the header length as the minimum width
#             col_widths[col_key] = len(label)

#     # 2. Only run the measurement loop if auto_adjust is True
#     if auto_adjust:
#         for item in data:
#             for col_key in col_widths.keys():
#                 val = item.get(col_key, "")
#                 # Determine string length after formatting
#                 f_val = str(smart_round(val)) if isinstance(val, (int, float, Decimal)) else str(val)
#                 if len(f_val) > col_widths[col_key]:
#                     col_widths[col_key] = len(f_val)

#         # Add padding to all calculated widths
#         for key in col_widths:
#             col_widths[key] += 2

#     # 3. Render Header
#     header_parts = [f"{col[1]:{col_aligns[col[0]]}{col_widths[col[0]]}}" for col in columns]
#     header_line = " | ".join(header_parts)

#     print(f"\n--- {title} ---")
#     print(header_line)
#     print("-" * len(header_line))

#     # 4. Render Rows
#     for item in data:
#         row_values = []
#         for col in columns:
#             col_key = col[0]
#             val = item.get(col_key, "")

#             display_val = smart_round(val) if isinstance(val, (int, float, Decimal)) else str(val)
#             row_values.append(f"{display_val:{col_aligns[col_key]}{col_widths[col_key]}}")

#         print(" | ".join(row_values))


#     print("-" * len(header_line))


def get_visual_len(text: str) -> int:
    """Calculates the real visible length of a string, ignoring ANSI color codes."""
    # Regex to match ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', str(text)))


def print_table(data: list[dict], columns: list[tuple], title: str = "REPORT", auto_adjust: bool = True):
    """
    Renders a formatted CLI table with support for ANSI colors and dynamic alignment.

    Args:
        data (list[dict]): Row data.
        columns (list[tuple]): (key, label, [alignment]).
        title (str): Table title.
        auto_adjust (bool): Dynamic width calculation.

    Example:
        >>> green_val = f"\033[92m{10.5}\033[0m"
        >>> data = [{"val": green_val}]
        >>> cols = [("val", "Price", ">")]
        >>> print_table(data, cols)
    """

    if not data:
        print(f"\n--- {title} ---\nNo data available.")
        return

    col_widths = {}
    col_aligns = {}

    # 1. Initialize configurations
    for col in columns:
        col_key, label = col[0], col[1]
        col_aligns[col_key] = col[2] if len(col) > 2 else TABLE_ALIGN

        # Initial width is the length of the label
        col_widths[col_key] = len(label) if auto_adjust else TABLE_COL_WIDTH

    # 2. Dynamic width measurement (Color-aware)
    if auto_adjust:
        for item in data:
            for col_key in col_widths.keys():
                val = item.get(col_key, "")
                # Use the helper to measure only visible characters
                v_len = get_visual_len(val)
                if v_len > col_widths[col_key]:
                    col_widths[col_key] = v_len

        # Add horizontal padding
        for key in col_widths:
            col_widths[key] += 2

    # 3. Prepare Header
    header_parts = []
    for col in columns:
        key, label = col[0], col[1]
        # Standard formatting works for header as it usually has no colors
        header_parts.append(f"{label:{col_aligns[key]}{col_widths[key]}}")

    header_line = " | ".join(header_parts)
    table_width = len(header_line)

    # 4. Render Table
    print(f"\n{title.center(table_width, '-')}")
    print(header_line)
    print("-" * table_width)

    for item in data:
        row_values = []
        for col in columns:
            col_key = col[0]
            val = str(item.get(col_key, ""))

            # MANUAL PADDING for color strings
            # Since f-strings fail to align strings with hidden ANSI codes,
            # we calculate the padding manually.
            v_len = get_visual_len(val)
            padding = col_widths[col_key] - v_len

            if col_aligns[col_key] == "<":  # Left
                cell = val + (" " * padding)
            elif col_aligns[col_key] == ">":  # Right
                cell = (" " * padding) + val
            else:  # Center
                left_pad = padding // 2
                right_pad = padding - left_pad
                cell = (" " * left_pad) + val + (" " * right_pad)

            row_values.append(cell)

        print(" | ".join(row_values))

    print("-" * table_width)


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
