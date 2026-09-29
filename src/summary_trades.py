#!/usr/bin/python3

# Fix decimals

import time

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal as D
from typing import List

import krakenex

from utils.basic import (
    DATETIME_FORMAT,
    FIX_X_PAIR_NAMES,
    REALISED_GAINS_FILE,
    append_trades_to_csv,
    get_fix_pair_name,
    get_paginated_response_from_kraken,
    my_round,
    print_table,
    read_trades_csv,
    save_realised_gain,
    smart_round,
)
from utils.classes import CSVTrade

YEAR = 2026

VERBOSE = True

TRADES_FILE = './data/trades_2026.csv'
KEY_FILE = './data/keys/kraken.key'
PAGES = 2
RECORDS_PER_PAGE = 50
FILTER_ASSET_NAME = ''  #'EOSEUR' 'MATICEUR'
CLOSED_POSITION_MAX_AMOUNT = D(5)  # EUR left in buys (at buy price) still considered dust


FIFO = 'FIFO'
LIFO = 'LIFO'


def find_buy_index(buy_trades: List[CSVTrade], sell: CSVTrade, method: str) -> int | None:
    if method == FIFO:
        # FIFO: the oldest open buy, only if it is BEFORE the sell date (buy_trades are oldest first)
        return 0 if buy_trades[0].completed <= sell.completed else None

    # LIFO: find the latest buy BEFORE the sell date
    for i in range(len(buy_trades) - 1, -1, -1):
        if buy_trades[i].completed <= sell.completed:
            return i
    return None


def compute_gain_loss(
    buy_trades: List[CSVTrade],
    sell_trades: List[CSVTrade],
    year: int,
    asset_name: str,
    method: str,
) -> tuple[D, D, D, bool]:
    total_gain_loss = 0
    gain_loss_year = 0
    fees = 0  # Sell fees since Buy fees are included proportionally on price
    print(f'===== {asset_name} ({method}) =====')
    buy = None
    for sell in sell_trades:
        if VERBOSE:
            print(sell)

        while sell.remaining_volume > 0:
            if not buy_trades:
                print(f'No more buys for: {asset_name} ({method})')
                break

            buy_index = find_buy_index(buy_trades=buy_trades, sell=sell, method=method)
            if buy_index is None:
                print(
                    f'> Missing BUY for current SELL for {asset_name} previous to {sell.completed}.',
                )
                break

            buy = buy_trades[buy_index]

            if VERBOSE:
                print(
                    f'***** buy price: {my_round(value=buy.price)} on {buy.completed.date()} '
                    f'| buy remaining volume before: {my_round(value=buy.remaining_volume)} '
                    f'| sell remaining_volume before: {my_round(value=sell.remaining_volume)}.',
                )

            # Buy to 0
            if sell.remaining_volume > buy.remaining_volume:
                sell.accumulated_buy_amount += (
                    buy.remaining_volume * buy.price + (buy.remaining_volume / buy.volume) * buy.fee
                )
                sell.remaining_volume -= buy.remaining_volume
                buy.remaining_volume = 0
                sell.related_buys.append(buy)
                buy_trades.pop(buy_index)

            # THEN sell.remaining_volume <= buy.remaining_volume:
            else:
                buy.remaining_volume -= sell.remaining_volume

                if buy.remaining_volume == 0:
                    sell.related_buys.append(buy)
                    buy_trades.pop(buy_index)

                sell.accumulated_buy_amount += (
                    sell.remaining_volume * buy.price + (sell.remaining_volume / buy.volume) * buy.fee
                )

                sell.remaining_volume = 0
                gain_loss = sell.amount - sell.accumulated_buy_amount
                if VERBOSE:
                    print(
                        f'sell for year: {sell.completed.year} | completed: {sell.completed.date()} '
                        f'| gain_loss: {my_round(value=gain_loss)} | fee: {my_round(value=sell.fee)}.',
                    )
                total_gain_loss += gain_loss
                if sell.completed.year == year:
                    gain_loss_year += gain_loss
                    fees += sell.fee
                break

            if VERBOSE:
                print(
                    f'***** buy price: {my_round(value=buy.price)} '
                    f'| buy remaining volume after: {my_round(value=buy.remaining_volume)} '
                    f'| sell remaining_volume after: {my_round(value=sell.remaining_volume)}.',
                )

    # Closed when some buy was matched and what is left of the buys is dust
    remaining_amount = sum(open_buy.remaining_volume * open_buy.price for open_buy in buy_trades)
    is_position_closed = buy is not None and remaining_amount <= CLOSED_POSITION_MAX_AMOUNT
    print(f'===== ASSET SUMMARY ({method}) =====')
    print(f'total gain loss: {my_round(value=total_gain_loss)}')
    print(f'gain loss ({year}): {my_round(value=gain_loss_year)}')
    print(f'fees ({year}): {my_round(value=fees)}')
    print(f'is_position_closed: {is_position_closed}')
    print('==========\n')
    return total_gain_loss, gain_loss_year, fees, is_position_closed


def fetch_new_trades(
    kapi,
    latest_trade_csv: CSVTrade | None,
    buy_trades: List[CSVTrade],
    sell_trades: List[CSVTrade],
) -> List[CSVTrade]:
    """Read from Kraken the trades newer than the CSV (all if it is empty), adding them to buy_trades/sell_trades."""
    trades_to_append_to_csv = []
    trade_pages = get_paginated_response_from_kraken(
        kapi=kapi,
        endpoint='TradesHistory',
        dict_key='trades',
        params={'trades': 'false'},
        pages=PAGES,
        records_per_page=RECORDS_PER_PAGE,
    )
    if not trade_pages:
        print('*****No new trades Found*****')

    for trade_page in trade_pages:
        for trade_detail in trade_page.values():
            closetime_str = time.strftime(DATETIME_FORMAT, time.gmtime(trade_detail['time']))
            closetime = datetime.strptime(closetime_str, DATETIME_FORMAT)
            if latest_trade_csv is None or closetime > latest_trade_csv.completed:
                trade = CSVTrade(
                    asset_name=trade_detail['pair'],
                    completed=closetime_str,
                    type=trade_detail['type'],
                    price=trade_detail['price'],
                    cost=trade_detail['cost'],
                    fee=trade_detail['fee'],
                    vol=trade_detail['vol'],
                )
                trades_to_append_to_csv.append(trade)
            else:
                break

    # Sort trades asc: Kraken returns newest first and FIFO/LIFO need buy_trades/sell_trades oldest first
    trades_to_append_to_csv_asc = sorted(trades_to_append_to_csv, key=lambda x: x.completed)
    for trade in trades_to_append_to_csv_asc:
        if trade.type == 'buy':
            buy_trades.append(trade)
        else:
            sell_trades.append(trade)
    return trades_to_append_to_csv_asc


def compute_pair_gains(buy_trades: List[CSVTrade], sell_trades: List[CSVTrade], year: int) -> tuple[list[dict], dict]:
    """G/L FIFO/LIFO per pair sold in year, plus the totals of all pairs."""
    sell_pairs_in_year = set([sell.asset_name for sell in sell_trades if sell.completed.year == year])
    sell_pairs_in_year = set([FILTER_ASSET_NAME]) if FILTER_ASSET_NAME else sell_pairs_in_year

    total_gain_loss = 0
    gain_loss_year = 0
    gain_loss_total_year_lifo = 0
    gain_loss_sell_amount_total_year = 0
    year_fees = 0

    pair_gains = []

    for asset_name in sell_pairs_in_year:
        buy_trades_asset = [buy for buy in buy_trades if buy.asset_name == asset_name]
        sell_trades_asset = [
            sell for sell in sell_trades if sell.asset_name == asset_name and sell.completed.year <= year
        ]

        # --- FIFO ---
        # deepcopy so the original lists are not modified
        total_gain_loss_asset_fifo, gain_loss_year_asset_fifo, fees, is_position_closed = compute_gain_loss(
            buy_trades=deepcopy(buy_trades_asset),
            sell_trades=deepcopy(sell_trades_asset),
            year=year,
            asset_name=asset_name,
            method=FIFO,
        )

        # --- LIFO ---
        _, gain_loss_year_asset_lifo, _, _ = compute_gain_loss(
            buy_trades=deepcopy(buy_trades_asset),
            sell_trades=deepcopy(sell_trades_asset),
            year=year,
            asset_name=asset_name,
            method=LIFO,
        )

        total_gain_loss += total_gain_loss_asset_fifo
        gain_loss_year += gain_loss_year_asset_fifo
        year_fees += fees

        sell_trades_asset_year_amount = [
            sell.amount for sell in sell_trades if sell.asset_name == asset_name and sell.completed.year == year
        ]
        # 1/6 of sell amount is not exactly 20% must be 16.6%
        # Selling at a 20% gain means the gain is 16.6% (0.2 / 1.2) of the sold amount
        sell_amount_asset_year = sum(sell_trades_asset_year_amount) * D('0.166')

        if is_position_closed:
            sell_amount_asset_year = gain_loss_year_asset_fifo
            gain_loss_year_asset_lifo = gain_loss_year_asset_fifo

        gain_loss_sell_amount_total_year += sell_amount_asset_year
        gain_loss_total_year_lifo += gain_loss_year_asset_lifo

        pair_gains.append(
            {
                'name': asset_name,
                'fix_name': get_fix_pair_name(pair_name=asset_name, fix_x_pair_names=FIX_X_PAIR_NAMES),
                'gl': total_gain_loss_asset_fifo,
                'gl_year_fifo': gain_loss_year_asset_fifo,
                'gl_year_lifo': gain_loss_year_asset_lifo,
                'gl_sell_amount': sell_amount_asset_year,
            },
        )

    totals = {
        'gl': total_gain_loss,
        'gl_year_fifo': gain_loss_year,
        'gl_year_lifo': gain_loss_total_year_lifo,
        'gl_sell_amount': gain_loss_sell_amount_total_year,
        'fees_year': year_fees,
    }
    return pair_gains, totals


def print_summary(buy_trades: List[CSVTrade], sell_trades: List[CSVTrade], year: int, totals: dict):
    # Buy/Sells summary
    total_buy_amount = sum([buy.amount for buy in buy_trades])
    total_sell_amount = sum([sell.amount for sell in sell_trades])
    total_fees = sum([trade.fee for trade in buy_trades + sell_trades])

    print('===== BUY/SELLS SUMMARY =====')
    print(f'total buys: {smart_round(number=total_buy_amount)}')
    print(f'total sells: {smart_round(number=total_sell_amount)}')
    print(f'sells - buys: {smart_round(number=(total_sell_amount - total_buy_amount))}')

    print('\n ===== SUMMARY =====')
    print(f'fees ({year}) | total fees: {smart_round(number=totals["fees_year"])} / {smart_round(number=total_fees)}')
    print(f'total gain loss (traded assets): {smart_round(number=totals["gl"])}')
    print(f'G/L FIFO({year}): {smart_round(number=totals["gl_year_fifo"])}')
    print(f'G/L LIFO ({year}): {smart_round(number=totals["gl_year_lifo"])}')
    print(f'G/L Sell amount ({year}): {smart_round(number=totals["gl_sell_amount"])}')


PAIR_COLUMNS = [
    ("fix_name", "PAIR"),
    ("gl", "G/L TOTAL"),
    ("gl_year_fifo", "FIFO"),
    ("gl_year_lifo", "LIFO"),
    ("gl_sell_amount", "SELL AMT"),
]


def print_pair_gains(pair_gains: list[dict]):
    pair_gains.sort(reverse=True, key=lambda x: x['gl'])
    print_table(
        data=deepcopy(pair_gains),
        apply_smart_round=True,
        columns=PAIR_COLUMNS,
        title="G/L PAIRS (SORTED BY G/L TOTAL)",
    )

    pair_gains.sort(reverse=True, key=lambda x: x['gl_year_lifo'])
    print_table(
        data=pair_gains,
        apply_smart_round=True,
        columns=PAIR_COLUMNS,
        title="G/L PAIRS (SORTED BY LIFO)",
    )


def main():
    # configure api
    kapi = krakenex.API()
    kapi.load_key(KEY_FILE)

    read_start = datetime.now(timezone.utc)
    buy_trades = []
    sell_trades = []
    latest_trade_csv = read_trades_csv(filename=TRADES_FILE, buy_trades=buy_trades, sell_trades=sell_trades)
    # latest_trade_csv_completed_tz = pytz.UTC.localize(latest_trade_csv.completed).astimezone(localtz)
    # latest_trade_csv_completed = latest_trade_csv.completed

    # Read Trades from Kraken API
    trades_to_append_to_csv_asc = fetch_new_trades(
        kapi=kapi,
        latest_trade_csv=latest_trade_csv,
        buy_trades=buy_trades,
        sell_trades=sell_trades,
    )

    pair_gains, totals = compute_pair_gains(buy_trades=buy_trades, sell_trades=sell_trades, year=YEAR)
    print_summary(buy_trades=buy_trades, sell_trades=sell_trades, year=YEAR, totals=totals)
    print_pair_gains(pair_gains=pair_gains)

    # Realised gain of the year, read by balances.py (a filtered run only covers one pair)
    if not FILTER_ASSET_NAME:
        save_realised_gain(filename=REALISED_GAINS_FILE, year=YEAR, amount=totals['gl_sell_amount'])

    # Append trades to CSV
    append_trades_to_csv(filename=TRADES_FILE, trades_to_append=trades_to_append_to_csv_asc)
    elapsed_time_read = datetime.now(timezone.utc) - read_start
    print('\n ***** TIME SUMMARY ***** ')
    print(f'Time: {elapsed_time_read}')


if __name__ == '__main__':
    main()
