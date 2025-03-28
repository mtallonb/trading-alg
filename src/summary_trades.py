#!/usr/bin/python3

# Fix decimals

import time

from datetime import datetime
from decimal import Decimal as D

import krakenex

from utils.basic import append_trades_to_csv, get_trades_history, my_round, read_trades_csv
from utils.classes import CSVTrade

year = 2025

VERBOSE = True
CSV_READ = False

filename = './data/trades_2025.csv'
file = None
buy_trades = []
sell_trades = []

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# configure api
kapi = krakenex.API()
kapi.load_key('./data/kraken.key')
PAGES = 20  # 50 RECORDS per page
RECORDS_PER_PAGE = 50

# prepare request
req_data = {'trades': 'false'}


def compute_gain_loss(buy_trades, sell_trades, year, asset_name):
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

    print('===== ASSET SUMMARY =====')
    print(f'total gain loss: {my_round(total_gain_loss)}')
    print(f'gain loss ({year}): {my_round(gain_loss_year)}')
    print(f'fees ({year}): {my_round(fees)}')
    print('==========\n')
    return total_gain_loss, gain_loss_year, fees


read_start = datetime.utcnow()
latest_trade_csv = read_trades_csv(filename, buy_trades, sell_trades)
# latest_trade_csv_completed_tz = pytz.UTC.localize(latest_trade_csv.completed).astimezone(localtz)
# latest_trade_csv_completed = latest_trade_csv.completed

# Read Trades from Kraken API
trades_to_append_to_csv = []
for page in range(PAGES):
    request_data = dict(req_data)
    trades = get_trades_history(request_data, page, RECORDS_PER_PAGE, kapi)
    if not trades:
        break

    for trade_detail in trades:
        closetime = time.strftime(DATE_FORMAT, time.gmtime(trade_detail['time']))
        name = trade_detail['pair']
        # fix_name = REPLACE_NAMES[name] if name in REPLACE_NAMES.keys() else name
        trade = CSVTrade(
            name,
            closetime,
            trade_detail['type'],
            trade_detail['price'],
            trade_detail['cost'],
            trade_detail['fee'],
            trade_detail['vol'],
        )
        if trade.completed > latest_trade_csv.completed:
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

total_gain_loss = 0
gain_loss_year = 0
unrealised_p_total_year = 0
year_fees = 0

pair_gains = []

for asset_name in sell_pairs_in_year:
    buy_trades_asset = [buy for buy in buy_trades if buy.asset_name == asset_name]
    sell_trades_asset = [sell for sell in sell_trades if sell.asset_name == asset_name and sell.completed.year <= year]
    total_gain_loss_asset, gain_loss_year_asset, fees = compute_gain_loss(
        buy_trades_asset,
        sell_trades_asset,
        year,
        asset_name,
    )

    sell_trades_asset_year_amount = [
        sell.amount for sell in sell_trades if sell.asset_name == asset_name and sell.completed.year == year
    ]
    sell_total_year = sum(sell_trades_asset_year_amount)
    unrealised_p_year = sell_total_year * D('0.2')
    total_gain_loss += total_gain_loss_asset
    gain_loss_year += gain_loss_year_asset
    unrealised_p_total_year += unrealised_p_year
    year_fees += fees
    pair_gains.append(
        {
            'name': asset_name,
            'gl': total_gain_loss_asset,
            'gl_year': gain_loss_year_asset,
            'unrealised_p_year': unrealised_p_year,
        },
    )

# Buy /Sells summary
total_buy_amount = sum([buy.amount for buy in buy_trades])
total_sell_amount = sum([sell.amount for sell in sell_trades])
total_fees = sum([trade.fee for trade in buy_trades + sell_trades])

print('===== BUY/SELLS SUMMARY =====')
print(f'total buys: {my_round(total_buy_amount)}')
print(f'total sells: {my_round(total_sell_amount)}')
print(f'sells - buys: {my_round(total_sell_amount - total_buy_amount)}')
# print(f'Computed cash: {my_round(D(7091) + total_sell_amount - total_buy_amount) - total_fees}')

print('\n ===== SUMMARY =====')
print(f'total gain loss (traded assets): {my_round(total_gain_loss)}')
print(f'gain loss ({year}): {my_round(gain_loss_year)}')
print(f'unrealised gain ({year}): {my_round(unrealised_p_total_year)}')
print(f'fees ({year})/  total fees : {my_round(year_fees)} / {my_round(total_fees)}')

pair_gains.sort(reverse=True, key=lambda x: x['gl'])

print('\n== G/L PAIRS (TOTAL ACCUM/CURRENT YEAR (FIFO)/(LIFO)) ==')
for pair in pair_gains:
    print(
        f'{pair["name"]:{10}} {my_round(pair["gl"]):{10}} {my_round(pair["gl_year"]):{10}}'
        f'{my_round(pair["unrealised_p_year"]):{10}}',
    )

pair_gains.sort(reverse=True, key=lambda x: x['unrealised_p_year'])

print('\n== G/L PAIRS SORT BY LIFO (3rd COLUMN) ==')
for pair in pair_gains:
    print(
        f'{pair["name"]:{10}} {my_round(pair["gl"]):{10}} {my_round(pair["gl_year"]):{10}}'
        f'{my_round(pair["unrealised_p_year"]):{10}}',
    )

# Append trades to CSV
append_trades_to_csv(filename, trades_to_append_to_csv_asc)
elapsed_time_read = datetime.utcnow() - read_start
print('\n ***** TIME SUMMARY ***** ')
print(f'Time: {elapsed_time_read}')
