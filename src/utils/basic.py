#!/usr/bin/env python

import re
import time

from _csv import writer
from codecs import iterdecode
from csv import DictReader
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

from .classes import CSVTrade, PriceOHLC, Trade

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DECIMALS = 3

# pytzutc = pytz.timezone('UTC')
LOCAL_TZ = pytz.timezone('Europe/Madrid')

# fix pair names
FIX_X_PAIR_NAMES = ['XETHEUR', 'XETH', 'XLTCEUR', 'XLTC', 'XETCEUR', 'XETC']
STAKING_SUFFIXES = ('.S', '.MEUR', '.SEUR', '.BEUR')


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


def from_timestamp_to_str(timestamp):
    return time.strftime(DATE_FORMAT, time.localtime(timestamp))


def from_timestamp_to_datetime(timestamp):
    return datetime.fromtimestamp(timestamp)


def chunks(elem_list, n):
    n = max(1, n)
    return (elem_list[i : i + n] for i in range(0, len(elem_list), n))


def is_staked(name):
    return name.endswith(STAKING_SUFFIXES)


# TODO not working on 5e-05
def count_zeros(value):
    float_str = str(value)
    return len(re.search(r'\d+\.(0*)', float_str).group(1))


def my_round(value, decimal_places=DECIMALS):
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


def get_trades_history(request_data, page, RECORDS_PER_PAGE, kapi):
    request_data['ofs'] = RECORDS_PER_PAGE * page
    trades_history = kapi.query_private('TradesHistory', request_data)
    if not trades_history.get('result') or not trades_history['result']['trades']:
        return []
    trades = [order for _, order in trades_history['result']['trades'].items()]
    return trades

def get_flow_from_kraken(kapi, flow_type: str, pages: int, record_p_page=50, timestamp_from=None) -> pd.DataFrame:

    ledger_flow = []
    request_data = {'type': flow_type}
    if timestamp_from:
        request_data['start'] = timestamp_from

    for page in range(pages):
        request_data['ofs'] = record_p_page * page
        ledger_flow_page = kapi.query_private('Ledgers', request_data)
        if not ledger_flow_page.get('result') or not ledger_flow_page['result']['ledger']:
            break

        for _, rec in ledger_flow_page['result']['ledger'].items():
            ledger_flow.append(rec)
    flow = pd.DataFrame(ledger_flow)
    flow.columns = [x.upper() for x in flow.columns]
    flow.TIME = pd.to_datetime(flow.TIME, unit='s')
    return flow


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
        # Pass the header
        next(csv_reader, None)

        for asset_csv in csv_reader:
            asset_name = get_fix_pair_name(asset_csv['pair'], fix_x_pair_names)
            asset = assets_dict.get(asset_name)
            execution_time = datetime.strptime(asset_csv['time'], DATE_FORMAT)

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


def compute_ranking(df, count_sell_trades):
    sum_margin_a = df.MARGIN_A.abs().sum()
    df['MARGIN_PC'] = df.MARGIN_A / sum_margin_a
    df['PB'] = (df.CURR_PRICE - df.AVG_B) / df.CURR_PRICE
    df['PS'] = (df.CURR_PRICE - df.AVG_S) / df.CURR_PRICE
    df['PERC_BS'] = (df.AVG_S - df.AVG_B) / df.AVG_S
    df['S_TRADES'] = (df.S_TRADES / count_sell_trades) * 10

    df['RANKING'] = df['PB'] + df['PS'] + df['PERC_BS'] + df['S_TRADES'] + df['MARGIN_PC']

    idx = df['AVG_S'] == 0.0
    df.loc[idx, 'RANKING'] = np.nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["RANKING"], how="all", inplace=True)
    idx = df['RANKING'] < -10
    df.loc[idx, 'RANKING'] = -10
    df['RANKING'] = df['RANKING'] - df['RANKING'].min()
    df['RANKING'] = (df['RANKING'] / df['RANKING'].max()) * 10

    # df = df.drop(columns=['pb', 'ps'])
    return df


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
