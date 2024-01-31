# coding=utf-8
#!/usr/bin/env python
# TODO
# Métricas para conocer valor de compra y cuantos asset podemos tener en cartera
# Umbral de perdidas y ganancia max percentile 10 esperar a que suba tras bajar para comprar
# Notificaciones telegram con telebot

# BUG si hay más de 50 trades sin actualizar en los trades
# TWRR en trades y más métricas
# Mostrar si esta bloqueado el que esta a punto de vender

# Dinero invertido en los asset muertos
# Compensar ganancias con las perdidas de las muertas.
# Add support to staking asset
# Añadir media y varianza a cada asset de 30 dias por ejemplo parametrizable


import pandas as pd

import krakenex

from utils.classes import Order, Asset
from utils.basic import *

LAST_ORDERS = 200

BUY_LIMIT = 4  # Number of consecutive buy trades
GAIN_PERCENTAGE = 0.2  # Gain percentage to sell/buy 20%
ORDER_THR = 0.25  # Umbral que consideramos error en la compra o venta a eliminar
MINIMUM_BUY_AMOUNT = 70
BUY_LIMIT_AMOUNT = BUY_LIMIT * 1.5 * MINIMUM_BUY_AMOUNT  # Computed as asset.trades_buy_amount - asset.trades_sell_amount

PAGES = 20  # 50 RECORDS per page
RECORDS_PER_PAGE = 50

# fix pair names
FIX_X_PAIR_NAMES = ['XETHEUR', 'XLTCEUR', 'XETCEUR']

# Exclude
EXCLUDE_PAIR_NAMES = ['ZEUREUR', 'BSVEUR', 'LUNAEUR', 'SHIBEUR', 'ETH2EUR']
# auto remove *.SEUR 'ATOM.SEUR', 'DOT.SEUR', 'XTZ.SEUR', 'EUR.MEUR']

ASSETS_TO_EXCLUDE_AMOUNT = ['SCEUR', 'DASHEUR', 'SGBEUR', 'SHIBEUR', 'LUNAEUR', 'LUNA2EUR']

# PAIR_TO_LAST_TRADES = ['SCEUR', 'SNXEUR', 'SHIBEUR', 'SOLEUR', 'ETCEUR']
# PAIR_TO_LAST_TRADES = ['XDGEUR', 'EOSEUR']
# PAIR_TO_LAST_TRADES = ['MATICEUR']
# PAIR_TO_LAST_TRADES = ['XBTEUR']
# PAIR_TO_LAST_TRADES = ['LINKEUR']
# PAIR_TO_LAST_TRADES = ['ETCEUR']
# PAIR_TO_LAST_TRADES = ['ATOMEUR']
# PAIR_TO_LAST_TRADES = ['LTCEUR']
PAIR_TO_LAST_TRADES = ['SNXEUR']
# PAIR_TO_LAST_TRADES = ['LUNAEUR', 'SOLEUR', ]
# PAIR_TO_LAST_TRADES = []

# PAIR_TO_FORCE_INFO = ['ETCEUR']
# PAIR_TO_FORCE_INFO = ['XLMEUR']
PAIR_TO_FORCE_INFO = ['XBTEUR', 'MINAEUR']

PRINT_LAST_TRADES = False
PRINT_ORDERS_SUMMARY = True
PRINT_BUYS_WARN_CONSECUTIVE = False
PRINT_PERCENTAGE_TO_EXECUTE_ORDERS = True
SHOW_COUNT_BUYS = False

AUTO_CANCEL_BUY_ORDER = True
AUTO_CANCEL_SELL_ORDER = True

GET_FULL_TRADE_HISTORY = True
TRADE_FILE = './data/trades_2024.csv'

# configure api
kapi = krakenex.API()
kapi.load_key('./data/kraken.key')

# prepare request
# req_data = {'docalcs': 'true'}
req_data = {'trades': 'false'}

# query servers
# start = kapi.query_public('Time')
start = datetime.utcnow()

balance = kapi.query_private('Balance')
open_orders = kapi.query_private('OpenOrders', data=req_data)
# stacked_assets = kapi.query_private('Earn/Allocations')#, data={'hide_zero_allocations': True})
# trade_balance = kapi.query_private('TradeBalance')
# close_orders = kapi.query_private('CloseOrders', req_data)
# trades_history = kapi.query_private('TradesHistory', req_data)

# end = kapi.query_public('Time')
# latency = end['result']['unixtime'] - start['result']['unixtime']
latency = datetime.utcnow() - start
currency = 'EUR'

# EUR balance
cash_eur = float(balance['result']['ZEUR'])

elapsed_time_open_orders = None
elapsed_time_last_trades = None
elapsed_time_orders_summary = None
processing_time_start = datetime.utcnow()

# Pandas
# Float output format
pd.options.display.float_format = '{:.3f}'.format

assets_dict = {}
ledger_deposit = []
sells_amount = 0
buys_amount = 0
# ------------------------------------------------
initialization_time_start = datetime.utcnow()

# Create pair dicts
for key, value in balance['result'].items():
    key_name = key[1:] + currency if key[0] == key[1] == 'X' else key + currency
    original_name = key + 'Z' + currency if key[0] == 'X' else key + currency
    key_name = get_fix_pair_name(key_name, FIX_X_PAIR_NAMES)
    if key_name not in EXCLUDE_PAIR_NAMES and not is_staked(key_name):
        asset = Asset(name=key_name, original_name=original_name, shares=float(value))
        assets_dict[key_name] = asset

elapsed_time_initialization = datetime.utcnow() - initialization_time_start

# Fill price and balance
name_list = assets_dict.keys()
concatenate_names = ','.join(name_list)
tickers_info = kapi.query_public('Ticker', {'pair': concatenate_names.lower()})
# Watch-out is returning all assets
for name, ticker_info in tickers_info['result'].items():
    ticker_info = tickers_info['result'].get(name)
    fixed_pair_name = get_fix_pair_name(name, FIX_X_PAIR_NAMES)
    asset = assets_dict.get(fixed_pair_name)
    if asset:
        asset.fill_ticker_info(ticker_info)

print(f'\n *****PAIR NAMES BY BALANCE TOTAL: {len(assets_dict)} *****')
# Sort dict by balance descending
sorted_pair_names_list_balance = sorted(assets_dict.items(), key=lambda x: x[1].balance, reverse=True)
name_list = [ele[0] for ele in sorted_pair_names_list_balance]
[print(ele) for ele in chunks(name_list, 5)]

open_orders_time_start = datetime.utcnow()

orders = []
print('\n *****OPEN ORDERS READ*****')
for txid, order_dict in open_orders['result']['open'].items():
    order_detail = order_dict['descr']
    pair_name = order_detail['pair']
    asset = assets_dict.get(pair_name)
    if not asset:
        print(f'Missing order pair. Adding pair: {pair_name}')
        asset = Asset(name=pair_name, original_name=pair_name)

    price, shares = get_price_shares_from_order(order_detail['order'])
    amount = price * shares
    order = Order(txid, order_detail['type'], shares, price)
    order.creation_datetime = datetime.fromtimestamp(order_dict['opentm'])
    asset.orders.append(order)
    if order.order_type == 'buy':
        asset.orders_buy_amount += amount
        buys_amount += amount
        asset.orders_buy_count += 1
        asset.update_orders_buy_higher_price(price)
    else:
        asset.orders_sell_amount += amount
        sells_amount += amount
        asset.orders_sell_count += 1
        asset.update_orders_sell_lower_price(price)

    # This array is used exclusively for Pandas stats
    orders.append({'asset': pair_name, 'order_type': order.order_type, 'price': price, 'current_price': asset.price})

if PRINT_PERCENTAGE_TO_EXECUTE_ORDERS:
    df = pd.DataFrame(orders)
    df['percentage'] = 100 * (df['price']-df['current_price'])/df['current_price']
    df['percentage_abs'] = abs(df['percentage'])
    df = df.reindex(df.percentage.abs().sort_values().index)

    df_closer = df[df['percentage_abs'] <= 10].drop(columns=['percentage_abs'])
    df_middle = df[(df['percentage_abs'] > 10) & (df['percentage_abs'] <= 100)].drop(columns=['percentage_abs'])
    df_last = df[df['percentage_abs'] > 100].drop(columns=['percentage_abs'])

    print(f'\n***** ({df_closer.shape[0]}) < 10% *****\n')
    print(df_closer.to_string(index=False))
    print(f'\n***** ({df_middle.shape[0]}) > 10% *****\n')
    print(df_middle.to_string(index=False))
    print(f'\n***** ({df_last.shape[0]}) > 100% ****\n')
    print(df_last.to_string(index=False))

elapsed_time_open_orders = datetime.utcnow() - open_orders_time_start

csv_trades_time_start = datetime.utcnow()
last_trade_from_csv = None

if GET_FULL_TRADE_HISTORY:
    # Load trades from CSV
    last_trade_from_csv = load_from_csv(TRADE_FILE, assets_dict, FIX_X_PAIR_NAMES)

elapsed_time_csv_trades = datetime.utcnow() - csv_trades_time_start

trades_time_start = datetime.utcnow()
print('\n *****TRADES*****')
asset_name = ''
trade = None
for page in range(PAGES):
    request_data = dict(req_data)
    trades = get_trades_history(request_data, page, RECORDS_PER_PAGE, kapi)
    # time.sleep(1)
    if not trades:
        print(BCOLORS.WARNING + 'No trades Found' + BCOLORS.ENDC)
        break

    for trade_detail in trades:
        asset_name = get_fix_pair_name(trade_detail['pair'], FIX_X_PAIR_NAMES)
        asset = assets_dict.get(asset_name)
        if asset:
            execution_datetime = datetime.fromtimestamp(trade_detail['time'])
            execution_datetime_tz = localtz.localize(execution_datetime.replace(microsecond=0))

            trade = Trade(trade_detail['type'], float(trade_detail['vol']), float(trade_detail['price']),
                          amount=float(trade_detail['cost']), execution_datetime=execution_datetime_tz)
            if GET_FULL_TRADE_HISTORY and last_trade_from_csv and \
                    trade.execution_datetime > last_trade_from_csv.execution_datetime:
                asset.insert_trade_on_top(trade)
                print(BCOLORS.WARNING + 'CSV not updated' + BCOLORS.ENDC)

            elif trade.execution_datetime <= last_trade_from_csv.execution_datetime:
                print(f'CSV is updated from here on so we can leave the loop: {asset_name} {trade.execution_datetime}, '
                      f'last_trade from CSV: {last_trade_from_csv.execution_datetime}')
                break
            else:
                asset.add_trade(trade)
        else:
            print(f'Missing trade pair: {asset_name}')

    if GET_FULL_TRADE_HISTORY and trade and trade.execution_datetime <= last_trade_from_csv.execution_datetime:
        print(f'Leaving main loop. CSV is UPDATED: {trade.execution_datetime}, '
              f'last_trade: {last_trade_from_csv.execution_datetime}')
        break


# Oldest trade read
if asset_name and trade:
    print(BCOLORS.OKGREEN+f"Oldest trade date read for {asset_name}: {trade}"+BCOLORS.ENDC)

# exit(0)

# Fill latest trade date
for _, asset in assets_dict.items():
    if asset.trades:
        asset.latest_trade_date = asset.trades[0].execution_datetime.date()

# Last trades
if PRINT_LAST_TRADES:
    for asset_name in PAIR_TO_LAST_TRADES:
        asset = assets_dict.get(asset_name)
        print(f'\n**** Open orders for asset: {asset_name}.')
        for order in asset.orders[:LAST_ORDERS]:
            print(f'\n {order} ')

        print('\n**** Trades for asset: {}.'.format(asset_name))
        for trade in asset.trades[:LAST_ORDERS]:
            print(f'\n {trade}')

elapsed_time_last_trades = datetime.utcnow() - trades_time_start

print('\n*****PAIR NAMES BY LATEST TRADE:*****')
# Sort dict by balance descending
sorted_pair_names_list_latest = sorted(assets_dict.items(), key=lambda x: x[1].latest_trade_date, reverse=False)

assets_by_last_trade = []
count_sell_trades = 0
for _, asset in sorted_pair_names_list_latest:

    if not asset.trades:
        continue

    asset.fill_last_shares()

    if asset.latest_trade_date != DEFAULT_TRADE_DATE:
        sell_trades_count = asset.trades_sell_count
        count_sell_trades += sell_trades_count
        last_buy_amount = asset.last_buys_shares * asset.last_buys_avg_price
        buy_limit_reached = asset.check_buys_limit(BUY_LIMIT, MINIMUM_BUY_AMOUNT * BUY_LIMIT, last_buy_amount)
        buy_limit_amount_reached, margin_amount = asset.check_buys_amount_limit(BUY_LIMIT_AMOUNT)
        buy_limit_reached = 1 if buy_limit_reached or buy_limit_amount_reached else 0
        margin_amount = asset.trades_sell_amount - asset.trades_buy_amount
        assets_by_last_trade.append([asset.name,  asset.latest_trade_date, asset.orders_buy_count, buy_limit_reached,
                                     my_round(asset.price), my_round(asset.avg_buys), my_round(asset.avg_sells),
                                     sell_trades_count, my_round(margin_amount)])


# ------ RANKING ----------
# ibs is_buy_set, blr buy limit reached
ranking_col = ['Name', 'Last trd', 'ibs', 'blr', 'curr_price', 'avg_buys', 'avg_sells', 's_trades', 'margin_a']
df = pd.DataFrame(assets_by_last_trade, columns=ranking_col)
df = compute_ranking(df, count_sell_trades)
print(df.to_string(index=False))
for record in df[['Name', 'ranking']].to_dict('records'):
    # set ranking on the asset
    assets_dict[record['Name']].ranking = record['ranking']

print(
    '\n*****PAIR NAMES BY RANKING: (ibs: is buy set. blr: buy limit reached. '
    'margin_a: sells_amount - buys_amount)*****'
)
print(df.sort_values(by='ranking', ascending=False).to_string(index=False))

count_valid_asset = 0
count_remaining_buys = 0
count_missing_buys = 0
count_all_remaining_buys = 0
count_valid_asset -= len(ASSETS_TO_EXCLUDE_AMOUNT)

if PRINT_ORDERS_SUMMARY:
    orders_summary_time_start = datetime.utcnow()
    print('\n *****ORDERS TO CREATE*****')

    for _, asset in sorted_pair_names_list_balance:
        asset_name = asset.name
        if not asset.trades:
            continue

        last_trade_price = asset.trades[0].price
        thr_sell = last_trade_price * (1+ORDER_THR)
        thr_buy = last_trade_price * (1-ORDER_THR)

        remaining_buys = max(BUY_LIMIT - asset.last_buys_count, 0)
        last_buy_amount = asset.last_buys_shares * asset.last_buys_avg_price
        buy_limit_reached = asset.check_buys_limit(BUY_LIMIT, MINIMUM_BUY_AMOUNT*BUY_LIMIT, last_buy_amount)
        buy_limit_amount_reached, margin_amount = asset.check_buys_amount_limit(BUY_LIMIT_AMOUNT)

        if asset_name not in ASSETS_TO_EXCLUDE_AMOUNT and remaining_buys:
            count_all_remaining_buys += remaining_buys
            if asset.orders_buy_amount:
                print(f'BUY order already set. Subtracting 1.') if SHOW_COUNT_BUYS else None
                count_all_remaining_buys -= 1

            if SHOW_COUNT_BUYS:
                print(f'Remaining buys: {remaining_buys} for pair: {asset_name}.')
                print(f'Count ALL buys: {count_all_remaining_buys}.\n')

        if asset_name in PAIR_TO_FORCE_INFO:
            print(BCOLORS.WARNING+f'FORCE INFO ON PAIR: {asset_name}'+BCOLORS.ENDC)

        if asset.orders_sell_lower_price and asset.orders_sell_lower_price >= thr_sell:
            perc = percentage(last_trade_price, asset.orders_sell_lower_price,)
            print(
                BCOLORS.WARNING + 'Watch-out sell order greater than THR for pair: {}. Last lowest sell order: {}, '
                'last trade price: {}, perc: {} %'.format(
                    asset.name, my_round(asset.orders_sell_lower_price), my_round(last_trade_price), my_round(perc)
                )
                + BCOLORS.ENDC
            )
            if AUTO_CANCEL_SELL_ORDER:
                print(BCOLORS.WARNING + f'Going to delete SELL orders from pair: {asset_name}.' + BCOLORS.ENDC)
                input("Press Enter to continue or Ctrl+D to exit")
                cancel_orders(kapi, Order.SELL, asset.orders)

        if asset.orders_buy_higher_price and asset.orders_buy_higher_price <= thr_buy:
            perc = percentage(last_trade_price, asset.orders_buy_higher_price,)
            print(
                BCOLORS.WARNING + 'Watch-out buy order lower than THR for pair: {}. Last highest buy order: {}, '
                'last trade price: {}, perc: {} %'.format(
                    asset_name, my_round(asset.orders_buy_higher_price), my_round(last_trade_price), my_round(perc)
                )
                + BCOLORS.ENDC
            )

            if AUTO_CANCEL_BUY_ORDER:
                print(BCOLORS.WARNING + f'Going to delete BUY orders from pair: {asset_name}.' + BCOLORS.ENDC)
                # time.sleep(5)
                input("Press Enter to continue or Ctrl+D to exit")
                cancel_orders(kapi, Order.BUY, asset.orders)

        if buy_limit_amount_reached:
            print(
                BCOLORS.WARNING +
                f'Watch-out BUY LIMIT AMOUNT of {BUY_LIMIT_AMOUNT} reached on asset: {asset_name}. '
                f'Margin amount: {my_round(margin_amount)}'
                + BCOLORS.ENDC
            )

        if buy_limit_reached:
            print(
                BCOLORS.WARNING +
                f'Watch-out {BUY_LIMIT} consecutive BUYS on asset: {asset_name}. '
                f'Total buy amount: {my_round(asset.last_buys_shares * asset.last_buys_avg_price)}'
                + BCOLORS.ENDC
            )

        if asset.orders_buy_count >= 2:
            print(BCOLORS.FAIL + 'Buy duplicated for asset: {}'.format(asset_name) + BCOLORS.ENDC)

        if asset.orders_sell_count >= 2:
            print(BCOLORS.OKCYAN + 'Sell duplicated for asset: {}'.format(asset_name) + BCOLORS.ENDC)

        if not asset.orders_buy_amount or asset_name in PAIR_TO_FORCE_INFO:
            if not buy_limit_reached or PRINT_BUYS_WARN_CONSECUTIVE or asset_name in PAIR_TO_FORCE_INFO:
                print(asset.print_buy_message(GAIN_PERCENTAGE))

            if not asset.orders_buy_amount and not buy_limit_reached and asset_name not in ASSETS_TO_EXCLUDE_AMOUNT:
                count_remaining_buys += remaining_buys
                count_missing_buys += 1

        if not asset.orders_sell_amount or asset_name in PAIR_TO_FORCE_INFO:
            print(asset.print_sell_message(kapi, GAIN_PERCENTAGE, MINIMUM_BUY_AMOUNT))

        print('\n')

    elapsed_time_orders_summary = datetime.utcnow() - orders_summary_time_start
    print('\n ***** SUMMARY ***** ')
    cash_needed_missing_buy = count_missing_buys * MINIMUM_BUY_AMOUNT
    cash_needed = count_remaining_buys * MINIMUM_BUY_AMOUNT
    all_cash_needed = count_all_remaining_buys * MINIMUM_BUY_AMOUNT
    print(
        f'Total Sell amount: {my_round(sells_amount)}.\n'
        f'Total Buys amount: {my_round(buys_amount)}.\n'
        f'Remaining Cash (EUR): {my_round(cash_eur - buys_amount)}.\n'
        f'Count missing buys: {count_missing_buys}.\n'
        f'Needed cash missing buys: {cash_needed_missing_buy}.\n'
        f'Count remaining buys: {count_remaining_buys}.\n'
        f'Needed cash: {cash_needed}.\n'
        f'Count ALL remaining buys (worst case): {count_all_remaining_buys}.\n'
        f'ALL Needed cash: {all_cash_needed}'
    )

elapsed_time_since_begining = datetime.utcnow() - processing_time_start

print('\n ***** TIME SUMMARY ***** ')
print(f'Endpoints latency: {latency}')
print(f'Load CSV time: {elapsed_time_csv_trades}')
print(f'Initialization time: {elapsed_time_initialization}')
print(f'Open orders time: {elapsed_time_open_orders}')
print(f'Last trades time: {elapsed_time_last_trades}')
print(f'Orders summary time: {elapsed_time_orders_summary}')
print(f'Total time: {elapsed_time_since_begining}')
