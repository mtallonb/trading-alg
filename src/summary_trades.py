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
    append_trades_to_csv,
    get_fix_pair_name,
    get_paginated_response_from_kraken,
    my_round,
    read_trades_csv,
    smart_round,
)
from utils.classes import CSVTrade

year = 2025

VERBOSE = True
CSV_READ = False

filename = './data/trades_2025.csv'
file = None
buy_trades = []
sell_trades = []

# configure api
kapi = krakenex.API()
kapi.load_key('./data/keys/kraken.key')
PAGES = 20  # 50 RECORDS per page
RECORDS_PER_PAGE = 50
FILTER_ASSET_NAME = ''  #'EOSEUR' 'MATICEUR'

# prepare request
req_data = {'trades': 'false'}


def compute_gain_loss(
    buy_trades: List[CSVTrade],
    sell_trades: List[CSVTrade],
    year: int,
    asset_name: str,
) -> tuple[D, D, D, bool]:
    total_gain_loss = 0
    gain_loss_year = 0
    fees = 0  # Sell fees since Buy fees are included proportionally on price
    print('===== {} ====='.format(asset_name))
    for sell in sell_trades:
        if VERBOSE:
            print(sell)

        while sell.remaining_volume > 0:
            if buy_trades:
                buy = buy_trades[0]
            else:
                print('No buys for: {}'.format(asset_name))
                break

            if VERBOSE:
                print(
                    f'***** buy price: {my_round(buy.price)} on {buy.completed.date()}, '
                    f'buy remaining volume before: {my_round(buy.remaining_volume)},'
                    f' sell remaining_volume before: {my_round(sell.remaining_volume)}.',
                )

            # Buy to 0
            if sell.remaining_volume > buy.remaining_volume:
                sell.accumulated_buy_amount += (
                    buy.remaining_volume * buy.price + (buy.remaining_volume / buy.volume) * buy.fee
                )
                sell.remaining_volume -= buy.remaining_volume
                buy.remaining_volume = 0
                sell.related_buys.append(buy)
                buy_trades.remove(buy)

            # THEN sell.remaining_volume <= buy.remaining_volume:
            else:
                buy.remaining_volume -= sell.remaining_volume

                if buy.remaining_volume == 0:
                    buy_trades.remove(buy)
                    sell.related_buys.append(buy)

                sell.accumulated_buy_amount += (
                    sell.remaining_volume * buy.price + (sell.remaining_volume / buy.volume) * buy.fee
                )

                sell.remaining_volume = 0
                gain_loss = sell.amount - sell.accumulated_buy_amount
                if VERBOSE:
                    print(
                        f'sell for year: {sell.completed.year}, completed: {sell.completed.date()}.'
                        f'gain_loss: {my_round(gain_loss)}, fee: {my_round(sell.fee)}.',
                    )
                total_gain_loss += gain_loss
                if sell.completed.year == year:
                    gain_loss_year += gain_loss
                    fees += sell.fee
                break

            if VERBOSE:
                print(
                    f'***** buy price: {my_round(buy.price)}, '
                    f'buy remaining volume after: {my_round(buy.remaining_volume)}, '
                    f'sell remaining_volume after: {my_round(sell.remaining_volume)}.',
                )

    is_position_closed = True if buy.remaining_volume <= 0.001 and asset_name != 'XXBTZEUR' else False
    print('===== ASSET SUMMARY =====')
    print(f'total gain loss: {my_round(total_gain_loss)}')
    print(f'gain loss ({year}): {my_round(gain_loss_year)}')
    print(f'fees ({year}): {my_round(fees)}')
    print(f'is_position_closed: {is_position_closed}')
    print('==========\n')
    return total_gain_loss, gain_loss_year, fees, is_position_closed


def compute_gain_loss_lifo(
    buy_trades: List[CSVTrade],
    sell_trades: List[CSVTrade],
    year: int,
    asset_name: str,
) -> tuple[D, D, D, bool]:
    total_gain_loss = 0
    gain_loss_year = 0
    fees = 0  # Sell fees since Buy fees are included proportionally on price
    print(f'===== {asset_name} (LIFO) =====')
    for sell in sell_trades:
        if VERBOSE:
            print(sell)

        while sell.remaining_volume > 0:
            if not buy_trades:
                print(f'No more buys for: {asset_name}')
                break

            # LIFO: Encontrar la última compra ANTERIOR a la fecha de la venta
            buy_index = -1
            for i in range(len(buy_trades) - 1, -1, -1):
                if buy_trades[i].completed <= sell.completed:
                    buy_index = i
                    break

            if buy_index == -1:
                print(
                    f'> Missing BUY for current SELL for {asset_name} previous to {sell.completed}.',
                )
                break

            buy = buy_trades[buy_index]

            if VERBOSE:
                print(
                    f'***** buy price: {my_round(buy.price)} on {buy.completed.date()} '
                    f'| buy remaining volume before: {my_round(buy.remaining_volume)} '
                    f'| sell remaining_volume before: {my_round(sell.remaining_volume)}.',
                )

            # Buy to 0
            if sell.remaining_volume > buy.remaining_volume:
                sell.accumulated_buy_amount += (
                    buy.remaining_volume * buy.price + (buy.remaining_volume / buy.volume) * buy.fee
                )
                sell.remaining_volume -= buy.remaining_volume
                buy.remaining_volume = 0
                sell.related_buys.append(buy)
                buy_trades.pop(buy_index)  # Eliminar por índice es más eficiente

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
                        f'sell for year: {sell.completed.year} | completed: {sell.completed.date()}'
                        f'| gain_loss: {my_round(gain_loss)} | fee: {my_round(sell.fee)}.',
                    )
                total_gain_loss += gain_loss
                if sell.completed.year == year:
                    gain_loss_year += gain_loss
                    fees += sell.fee
                break

            if VERBOSE:
                print(
                    f'***** buy price: {my_round(buy.price)} '
                    f'| buy remaining volume after: {my_round(buy.remaining_volume)} '
                    f'| sell remaining_volume after: {my_round(sell.remaining_volume)}',
                )

    is_position_closed = not buy_trades
    print('===== ASSET SUMMARY (LIFO) =====')
    print(f'total gain loss: {my_round(total_gain_loss)}')
    print(f'gain loss ({year}): {my_round(gain_loss_year)}')
    print(f'fees ({year}): {my_round(fees)}')
    print(f'is_position_closed: {is_position_closed}')
    print('==========\n')
    return total_gain_loss, gain_loss_year, fees, is_position_closed


read_start = datetime.now(timezone.utc)
latest_trade_csv = read_trades_csv(filename, buy_trades, sell_trades)
# latest_trade_csv_completed_tz = pytz.UTC.localize(latest_trade_csv.completed).astimezone(localtz)
# latest_trade_csv_completed = latest_trade_csv.completed

# Read Trades from Kraken API
trades_to_append_to_csv = []
trade_pages = get_paginated_response_from_kraken(
    kapi,
    endpoint='TradesHistory',
    dict_key='trades',
    params={'trades': 'false'},
    pages=2,
    records_per_page=RECORDS_PER_PAGE,
)
if not trade_pages:
    print('*****No new trades Found*****')

for trade_page in trade_pages:
    for trade_detail in trade_page.values():
        closetime_str = time.strftime(DATETIME_FORMAT, time.gmtime(trade_detail['time']))
        if datetime.strptime(closetime_str, DATETIME_FORMAT) > latest_trade_csv.completed:
            # fix_name = REPLACE_NAMES[name] if name in REPLACE_NAMES.keys() else name
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

            if trade.type == 'buy':
                buy_trades.append(trade)
            else:
                sell_trades.append(trade)
        else:
            break

# Sort trades asc
buy_trades_asc = sorted(buy_trades, key=lambda x: x.completed)
sell_trades_asc = sorted(sell_trades, key=lambda x: x.completed)
trades_to_append_to_csv_asc = sorted(trades_to_append_to_csv, key=lambda x: x.completed)

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
    sell_trades_asset = [sell for sell in sell_trades if sell.asset_name == asset_name and sell.completed.year <= year]

    # --- CÁLCULO FIFO ---
    # Usamos deepcopy para no alterar las listas originales
    total_gain_loss_asset_fifo, gain_loss_year_asset_fifo, fees, is_position_closed = compute_gain_loss(
        deepcopy(buy_trades_asset),
        deepcopy(sell_trades_asset),
        year,
        asset_name,
    )

    # --- CÁLCULO LIFO ---
    _, gain_loss_year_asset_lifo, _, _ = compute_gain_loss_lifo(
        deepcopy(buy_trades_asset),
        deepcopy(sell_trades_asset),
        year,
        asset_name,
    )

    total_gain_loss += total_gain_loss_asset_fifo
    gain_loss_year += gain_loss_year_asset_fifo
    year_fees += fees

    sell_trades_asset_year_amount = [
        sell.amount for sell in sell_trades if sell.asset_name == asset_name and sell.completed.year == year
    ]
    sell_amount_asset_year = sum(sell_trades_asset_year_amount) * D('0.2')

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

# Buy/Sells summary
total_buy_amount = sum([buy.amount for buy in buy_trades])
total_sell_amount = sum([sell.amount for sell in sell_trades])
total_fees = sum([trade.fee for trade in buy_trades + sell_trades])

print('===== BUY/SELLS SUMMARY =====')
print(f'total buys: {smart_round(number=total_buy_amount)}')
print(f'total sells: {smart_round(number=total_sell_amount)}')
print(f'sells - buys: {smart_round(number=(total_sell_amount - total_buy_amount))}')

print('\n ===== SUMMARY =====')
print(f'total gain loss (traded assets): {smart_round(total_gain_loss)}')
print(f'G/L FIFO({year}): {smart_round(gain_loss_year)}')
print(f'G/L LIFO ({year}): {smart_round(gain_loss_total_year_lifo)}')
print(f'G/L Sell amount ({year}): {smart_round(gain_loss_sell_amount_total_year)}')
print(f'fees ({year}) | total fees: {smart_round(year_fees)} / {smart_round(total_fees)}')

pair_gains.sort(reverse=True, key=lambda x: x['gl'])


def print_pair_row(pair: dict) -> None:
    print(
        f'{pair["fix_name"]:{10}} {smart_round(pair["gl"]):{10}} {smart_round(pair["gl_year_fifo"]):{10}}'
        f'{smart_round(pair["gl_year_lifo"]):{10}} {smart_round(pair["gl_sell_amount"]):{10}}',
    )


print('\n== G/L PAIRS (TOTAL ACCUM|YEAR FIFO|YEAR LIFO|SELL AMOUNT) ==')
for pair in pair_gains:
    print_pair_row(pair)

pair_gains.sort(reverse=True, key=lambda x: x['gl_year_lifo'])

print('\n== G/L PAIRS SORT BY LIFO (3rd COLUMN) ==')
for pair in pair_gains:
    print_pair_row(pair)

# Append trades to CSV
append_trades_to_csv(filename, trades_to_append_to_csv_asc)
elapsed_time_read = datetime.now(timezone.utc) - read_start
print('\n ***** TIME SUMMARY ***** ')
print(f'Time: {elapsed_time_read}')
