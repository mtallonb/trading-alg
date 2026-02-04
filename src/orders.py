# coding=utf-8
#!/usr/bin/env python
# TODO
# Métricas para conocer valor de compra y cuantos asset podemos tener en cartera
# Dinero invertido en los asset muertos
# Compensar ganancias con las perdidas de las muertas.
# Ejecutar las pérdidas si hay mucha ganancia este año
# Mostrar si esta bloqueado el que esta a punto de vender
# DEFAULT_SESSIONS = [10, 50, 200]
# Add accumulated B/S on LIST percentage to execute

# RENAMING OF ASSETS:
# MATICEUR -> POLEUR
# EOSEUR -> AEUR

# DELISTING:
# LUNA, LUNA2, ETHW,

from datetime import datetime, timedelta, timezone

import krakenex
import pandas as pd

from ia_agent import get_smart_summary
from utils.basic import (
    BCOLORS,
    FIX_X_PAIR_NAMES,
    LOCAL_TZ,
    cancel_orders,
    chunks,
    compute_ranking,
    count_sells_in_range,
    get_fix_pair_name,
    get_paginated_response_from_kraken,
    get_price_shares_from_order,
    is_auto_staked,
    is_staked,
    load_from_csv,
    my_round,
    percentage,
    read_prices_from_local_file,
    remove_staking_suffix,
)
from utils.classes import MAPPING_NAMES, OP_BUY, OP_SELL, Asset, Order, Trade

# -----ALG PARAMS------------------------------------------------------------------------------------------------------
BUY_LIMIT = 4  # Number of consecutive buy trades
BUY_PERCENTAGE = SELL_PERCENTAGE = 0.2  # Risk percentage to sell/buy 20%
MINIMUM_BUY_AMOUNT = 70
BUY_LIMIT_AMOUNT = (
    BUY_LIMIT * 0.5 * MINIMUM_BUY_AMOUNT
)  # Computed as asset.trades_buy_amount - asset.trades_sell_amount
ORDER_THR = 0.35  # Umbral que consideramos error en la compra o venta a eliminar
USE_ORDER_THR = False
SHOW_SMART_SUMMARY = False
IA_AGENT = "groq"  # ['groq', 'gemini', 'openai']
# ----------------------------------------------------------------------------------------------------------------------
PAGES = 20  # 50 RECORDS per page
RECORDS_PER_PAGE = 50
LAST_ORDERS = 10
DEFAULT_SESSIONS = [10, 50, 200]
EXCLUDE_PAIR_NAMES = [
    'ZEUREUR', 'BSVEUR', 'LUNAEUR', 'SHIBEUR', 'ETH2EUR', 'WAVESEUR', 'XMREUR', 'EUR', 'EIGENEUR', 'APENFTEUR',
    'MATICEUR', 'EOSEUR',
]  # fmt: off
# auto remove *.SEUR 'ATOM.SEUR', 'DOT.SEUR', 'XTZ.SEUR', 'EUR.MEUR']
ASSETS_TO_EXCLUDE_AMOUNT = [
    'SCEUR', 'DASHEUR', 'SGBEUR', 'SHIBEUR', 'LUNAEUR', 'LUNA2EUR', 'WAVESEUR', 'EIGENEUR', 'APENFTEUR',
    'MATICEUR',
]  # fmt: off
MAPPING_STAKING_NAME = {'BTC': 'XBTEUR'}
# DUAL_ASSETS_NAME = {'MATICEUR': 'POLEUR'}

# PAIR NAMES: [
# 'SCEUR', 'ATOMEUR', 'ETCEUR', 'ETHEUR', 'BCHEUR', 'TIAEUR', 'TRXEUR', 'XRPEUR', 'LINKEUR', 'XBTEUR',
# 'FLREUR', 'SNXEUR', 'EOSEUR', 'SOLEUR', 'XDGEUR', 'MINAEUR', 'LTCEUR', 'APTEUR', 'XLMEUR', 'UNIEUR', 'BATEUR',
# 'FLOWEUR', 'AAVEEUR', 'XTZEUR', 'ADAEUR', 'AVAXEUR', 'ALGOEUR', 'MATICEUR', 'SUIEUR', 'TRUMPEUR']
PAIR_TO_LAST_TRADES = []

PAIR_TO_FORCE_INFO = ['XBTEUR']  # ['ADAEUR', 'SOLEUR']

PRINT_LAST_TRADES = False
PRINT_ORDERS_SUMMARY = True
PRINT_PERCENTAGE_TO_EXECUTE_ORDERS = True

AUTO_CANCEL_BUY_ORDER = True
AUTO_BUY_ORDER = False
AUTO_CANCEL_SELL_ORDER = True
AUTO_SELL_ORDER = False
PRINT_BUYS_WARN_CONSECUTIVE = False
SHOW_COUNT_BUYS = False

GET_FULL_TRADE_HISTORY = True
LOAD_ALL_CLOSE_PRICES = True
TRADE_FILE = './data/trades_2026.csv'
KEY_FILE = './data/keys/kraken.key'

TREND_THR = 0.2

# configure api
kapi = krakenex.API()
kapi.load_key(KEY_FILE)

# prepare request
# req_data = {'docalcs': 'true'}

# time to query servers
start = datetime.now(timezone.utc)

# CALLS TO KRAKEN API
# kapi.query_public('Ticker', {'pair': concatenate_names.lower()})
balance = kapi.query_private('Balance')
# open_orders = kapi.query_private('OpenOrders', data={'trades': 'false'})
# staked_assets = kapi.query_private('Earn/Allocations', data={'hide_zero_allocations': 'true'})
# trade_balance = kapi.query_private('TradeBalance')
# close_orders = kapi.query_private('CloseOrders', req_data)
# trades_history = kapi.query_private('TradesHistory', req_data)

# end = kapi.query_public('Time')
# latency = end['result']['unixtime'] - start['result']['unixtime']
elapsed_time_query_server = datetime.now(timezone.utc) - start
currency = 'EUR'

# EUR balance
cash_eur = float(balance['result']['ZEUR'])
staked_eur = 0.0

elapsed_time_open_orders = None
elapsed_time_last_trades = None
elapsed_time_orders_summary = None
processing_time_start = datetime.now(timezone.utc)

# Pandas conf
# Float output format
PANDAS_FLOAT_FORMAT = '{:.3f}'.format
pd.options.display.float_format = PANDAS_FLOAT_FORMAT

assets_dict: dict[str, Asset] = {}
ledger_deposit = []
sells_amount = 0
buys_amount = 0

yesterday = (datetime.today() - timedelta(days=1)).date()
# ------------------------------------------------
initialization_time_start = datetime.now(timezone.utc)

# Assets with balance or open order
asset_original_names = list(balance['result'].keys())
open_orders = kapi.query_private('OpenOrders', data={'trades': 'false'})
asset_original_names.extend(set([order['descr']['pair'] for order in open_orders['result']['open'].values()]))
asset_original_names = set(asset_original_names)

# ----------INITIALIZE PAIRS DICT-------------------------------------------------------------------
for name in asset_original_names:
    key_name = name[1:] if len(name) > 2 and name[0] == name[1] == 'X' else name
    original_name = name + 'Z' if name[0] == 'X' else name
    original_name = original_name if original_name.endswith(currency) else original_name + currency
    key_name = get_fix_pair_name(pair_name=key_name, fix_x_pair_names=FIX_X_PAIR_NAMES)
    if key_name not in EXCLUDE_PAIR_NAMES and not is_staked(key_name) and not assets_dict.get(key_name, False):
        asset = Asset(name=key_name, original_name=original_name)
        assets_dict[key_name] = asset

# ----------FILL BALANCE-------------------------------------------------------------------
for key, value in balance['result'].items():
    key_name = key[1:] if len(key) > 2 and key[0] == key[1] == 'X' else key
    key_name = get_fix_pair_name(key_name, FIX_X_PAIR_NAMES)
    if not is_staked(key_name) and key_name not in EXCLUDE_PAIR_NAMES and not assets_dict.get(key_name, False):
        print(f'Missing balance for pair: {key_name}')
        continue
    if key_name not in EXCLUDE_PAIR_NAMES:
        if not is_staked(key_name):
            assets_dict[key_name].shares = float(value)

        if is_auto_staked(key_name):
            asset_name_clean = f'{remove_staking_suffix(key_name)}EUR'
            if not assets_dict.get(asset_name_clean, False):
                print(f'Cannot fill autostaking balance for pair: {key_name} and clean pair: {asset_name_clean}')
                continue
            else:
                assets_dict[asset_name_clean].autostaked_shares = float(value)


# ----------FILL PRICES and VOLUMES-------------------------------------------------------------------
name_list = list(assets_dict.keys())
concatenate_names = ','.join(name_list)
# Watch-out is returning all assets with the latest price
tickers_info = kapi.query_public('Ticker', {'pair': concatenate_names.lower()})

for name, ticker_info in tickers_info['result'].items():
    fixed_pair_name = get_fix_pair_name(name, FIX_X_PAIR_NAMES)
    asset = assets_dict.get(fixed_pair_name)
    if asset:
        asset.fill_ticker_info(ticker_info)
    if LOAD_ALL_CLOSE_PRICES:
        df_prices, df_volumes = read_prices_from_local_file(asset_name=fixed_pair_name)
        if not df_prices.empty:
            asset.close_prices = df_prices
            latest_price_date = df_prices.DATE.iloc[-1]
            if latest_price_date < yesterday:
                print(f'Local PRICES of asset {fixed_pair_name} not updated since: {latest_price_date}')
        else:
            print(f'None prices found for asset: {fixed_pair_name}')

        if not df_volumes.empty:
            asset.close_volumes = df_volumes
            latest_volume_date = df_volumes.DATE.iloc[-1]
            if latest_volume_date < yesterday:
                print(f'Local VOLUMES of asset {fixed_pair_name} not updated since: {latest_volume_date}')
        else:
            print(f'None volumes found for asset: {fixed_pair_name}')

# ----------FILL STACKING INFO-------------------------------------------------------------------
staked_assets = kapi.query_private('Earn/Allocations', data={'hide_zero_allocations': 'true'})
# Watch-out is returning all assets
for staking_info in staked_assets['result']['items']:
    staking_name = staking_info['native_asset']
    if staking_name == 'EUR':
        staked_eur = float(staking_info['amount_allocated']['total']['native'])
        continue
    name = MAPPING_STAKING_NAME.get(staking_name, f"{staking_name}EUR")
    asset = assets_dict.get(name)
    if asset:
        asset.fill_staking_info(staking_info)

elapsed_time_initialization = datetime.now(timezone.utc) - initialization_time_start

# ----------SORTING BY BALANCE-------------------------------------------------------------------
print(f'\n *****PAIR NAMES SORTED BY BALANCE TOTAL: {len(assets_dict)} *****')
# Sort dict by balance descending
sorted_pair_names_list_balance = sorted(assets_dict.items(), key=lambda x: x[1].balance, reverse=True)
name_list = [ele[0] for ele in sorted_pair_names_list_balance]
[print(ele) for ele in chunks(name_list, 5)]

# ----------FILL ORDERS-------------------------------------------------------------------
open_orders_time_start = datetime.now(timezone.utc)
orders = []
print('\n *****OPEN ORDERS READ*****')

for txid, order_dict in open_orders.get('result').get('open').items():
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
    orders.append(
        {
            'asset': pair_name,
            'order_type': order.order_type,
            'price': price,
            'current_price': asset.price,
        },
    )

elapsed_time_open_orders = datetime.now(timezone.utc) - open_orders_time_start

# ----------FILL TRADES-------------------------------------------------------------------
csv_trades_time_start = datetime.now(timezone.utc)
last_trade_from_csv = None

if GET_FULL_TRADE_HISTORY:
    # Load trades from CSV
    last_trade_from_csv = load_from_csv(TRADE_FILE, assets_dict, FIX_X_PAIR_NAMES)

elapsed_time_csv_trades = datetime.now(timezone.utc) - csv_trades_time_start

trades_time_start = datetime.now(timezone.utc)
print('\n *****TRADES*****')
asset_name = ''
trade_pages = get_paginated_response_from_kraken(
    kapi,
    endpoint='TradesHistory',
    dict_key='trades',
    params={'trades': 'false'},
    pages=2,
    records_per_page=RECORDS_PER_PAGE,
)
if not trade_pages:
    print(BCOLORS.WARNING + 'No trades Found' + BCOLORS.ENDC)

for trade_page in trade_pages:
    for trade_detail in trade_page.values():
        asset_name = get_fix_pair_name(trade_detail['pair'], FIX_X_PAIR_NAMES)
        asset = assets_dict.get(asset_name)
        if asset:
            execution_datetime = datetime.fromtimestamp(trade_detail['time'])
            execution_datetime_tz = LOCAL_TZ.localize(execution_datetime.replace(microsecond=0))

            trade = Trade(
                trade_type=trade_detail['type'],
                shares=float(trade_detail['vol']),
                price=float(trade_detail['price']),
                amount=float(trade_detail['cost']),
                execution_datetime=execution_datetime_tz,
            )
            if (
                GET_FULL_TRADE_HISTORY
                and last_trade_from_csv
                and trade.execution_datetime > last_trade_from_csv.execution_datetime
            ):
                asset.insert_trade_on_top(trade)
                print(BCOLORS.WARNING + 'CSV not updated' + BCOLORS.ENDC)

            elif trade.execution_datetime <= last_trade_from_csv.execution_datetime:
                print(
                    f'CSV is updated from here on so we can leave the loop: {asset_name} {trade.execution_datetime}, '
                    f'last_trade from CSV: {last_trade_from_csv.execution_datetime}',
                )
                break
            else:
                asset.add_trade(trade)
        else:
            print(f'Missing trade pair: {asset_name}')

    if GET_FULL_TRADE_HISTORY and trade and trade.execution_datetime <= last_trade_from_csv.execution_datetime:
        print(
            f'Leaving main loop. CSV is UPDATED: {trade.execution_datetime}, '
            f'last_trade: {last_trade_from_csv.execution_datetime}',
        )
        break


# Oldest trade read
if asset_name and trade:
    print(BCOLORS.OKGREEN + f"Oldest trade date read for {asset_name}: {trade}" + BCOLORS.ENDC)

# ----------FILL LATEST TRADE OR DELETE ASSET-------------------------------------------------------------------
keys_to_delete = []
for key, asset in assets_dict.items():
    if asset.trades:
        asset.latest_trade_date = asset.trades[0].execution_datetime.date()
    else:
        keys_to_delete.append(key)

for key in keys_to_delete:
    del assets_dict[key]

# ----------PRINT LAST TRADES-------------------------------------------------------------------
if PRINT_LAST_TRADES:
    for asset_name in PAIR_TO_LAST_TRADES:
        asset = assets_dict.get(asset_name)
        print(f'\n**** Open orders for asset: {asset.output_name}.')
        for order in asset.orders[:LAST_ORDERS]:
            print(f'\n {order} ')

        print('\n**** Trades for asset: {}.'.format(asset.output_name))
        for trade in asset.trades[:LAST_ORDERS]:
            print(f'\n {trade}')

elapsed_time_last_trades = datetime.now(timezone.utc) - trades_time_start

# ----------FILL CALCULATIONS FROM LAST TRADES-------------------------------------------------------------------
# print('\n*****PAIR NAMES BY LATEST TRADE:*****')
# Sort dict by last trade
# sorted_pair_names_list_latest = sorted(assets_dict.items(), key=lambda x: x[1].latest_trade_date, reverse=False)

assets_by_last_trade = []
count_sell_trades = 0
for _, asset in assets_dict.items():
    if not asset.trades:
        continue

    asset.compute_last_buy_sell_avg()

    if asset.latest_trade_date:
        sell_trades_count = asset.trades_sell_count
        last_buy_amount = asset.last_buys_shares * asset.last_buys_avg_price
        buy_limit_reached = asset.check_buys_limit(BUY_LIMIT, MINIMUM_BUY_AMOUNT * BUY_LIMIT, last_buy_amount)
        buy_limit_amount_reached, margin_amount = asset.check_buys_amount_limit(BUY_LIMIT_AMOUNT)
        buy_limit_reached = 1 if buy_limit_reached or buy_limit_amount_reached else 0
        margin_amount = asset.margin_amount
        expected_sells_200 = avg_sessions_200 = avg_sessions_50 = avg_sessions_10 = None
        avg_volumes_200 = avg_volumes_50 = avg_volumes_10 = None
        if not asset.close_prices.empty:
            expected_sells_200 = count_sells_in_range(
                close_prices=asset.close_prices,
                days=200,
                buy_perc=BUY_PERCENTAGE,
                sell_perc=SELL_PERCENTAGE,
            )
            avg_sessions_200 = asset.avg_session_price(days=200)
            avg_sessions_50 = asset.avg_session_price(days=50)
            avg_sessions_10 = asset.avg_session_price(days=10)

        if not asset.close_volumes.empty:
            avg_volumes_200 = asset.avg_session_volume(days=200)
            avg_volumes_50 = asset.avg_session_volume(days=50)
            avg_volumes_10 = asset.avg_session_volume(days=10)

        # This list will be loaded to a DataFrame see ranking_cols
        assets_by_last_trade.append(
            [
                asset.name,
                asset.latest_trade_date,
                asset.orders_buy_count,
                buy_limit_reached,
                my_round(asset.price),
                my_round(asset.avg_buys),
                my_round(asset.avg_sells),
                my_round(margin_amount),
                sell_trades_count,
                expected_sells_200,
                my_round(avg_sessions_200),
                my_round(avg_sessions_50),
                my_round(avg_sessions_10),
                my_round(avg_volumes_200),
                my_round(avg_volumes_50),
                my_round(avg_volumes_10),
            ],
        )


# ------ RANKING ----------------------------------------------------------------------------------
ranking_cols = [
    'NAME',
    'LAST_TRADE',
    'IBS',
    'BLR',
    'CURR_PRICE',
    'AVG_B',
    'AVG_S',
    'MARGIN_A',
    'S_TRADES',
    'X_TRADES',
    'AVG_PRICE_200',
    'AVG_PRICE_50',
    'AVG_PRICE_10',
    'AVG_VOL_200',
    'AVG_VOL_50',
    'AVG_VOL_10',
]
df = pd.DataFrame(assets_by_last_trade, columns=ranking_cols)
ranking_df, detailed_ranking_df = compute_ranking(df)
# Print RANKING sorted by latest trade
# print(df.to_string(index=False))
for record in ranking_df[['NAME', 'RANKING']].to_dict('records'):
    assets_dict[record['NAME']].ranking = record['RANKING']

print(
    '\n*****PAIR NAMES BY RANKING: (IBD: Is Buy Set. BLR: Buy Limit Reached. '
    'S_TRADES and X_TRADES: Sell trades and Expected Sell trades on 200 sessions)*****',
)
# ranking_df['NAME'] = ranking_df['NAME'].replace(MAPPING_NAMES)
ranking_df.loc[:, 'NAME'] = ranking_df['NAME'].replace(MAPPING_NAMES)
# detailed_ranking_df['NAME'] = detailed_ranking_df['NAME'].replace(MAPPING_NAMES)
detailed_ranking_df.loc[:, 'NAME'] = detailed_ranking_df['NAME'].replace(MAPPING_NAMES)
pd.options.display.float_format = '{:.1f}'.format
print(ranking_df.to_string(index=False))
ranking_df_trending = ranking_df[ranking_df.TREND >= TREND_THR]
print(f'\n*****PAIR NAMES with TREND >= {TREND_THR} *****')
print(ranking_df_trending.to_string(index=False))
print('\n*****PAIR NAMES BY RANKING DETAILS: MARGIN_A: sells_amount - buys_amount.')
pd.options.display.float_format = PANDAS_FLOAT_FORMAT
print(detailed_ranking_df.to_string(index=False))
# -------------------------------------------------------------------------------------------------
live_asset_names = list(ranking_df[ranking_df.IBS == 1].NAME)
death_asset_names = list(ranking_df[ranking_df.IBS == 0].NAME)
print(f'\n*** LIVE ASSET NAMES ({len(live_asset_names)}): {live_asset_names}')
print(f'\n*** DEATH ASSET NAMES ({len(death_asset_names)}): {death_asset_names}')

if PRINT_PERCENTAGE_TO_EXECUTE_ORDERS:
    for order in orders:
        asset = assets_dict.get(order['asset'])
        if not asset:
            print(f'Missing asset from order with asset name {order["asset"]}')
        else:
            order['accum_b'] = asset.last_buys_count
            order['accum_s'] = asset.last_sells_count

    df = pd.DataFrame(orders)
    df.columns = df.columns.str.upper()
    df['ACCUM_B'] = df['ACCUM_B'].fillna(0).astype(int)
    df['ACCUM_S'] = df['ACCUM_S'].fillna(0).astype(int)
    df['PERCENTAGE'] = (100 * (df['PRICE'] - df['CURRENT_PRICE']) / df['CURRENT_PRICE']).round(1)
    df['PERCENTAGE_ABS'] = abs(df['PERCENTAGE'])
    df = df.reindex(df.PERCENTAGE.abs().sort_values().index)

    df_closer = df[df['PERCENTAGE_ABS'] <= 10].drop(columns=['PERCENTAGE_ABS'])
    df_middle = df[(df['PERCENTAGE_ABS'] > 10) & (df['PERCENTAGE_ABS'] <= 100)].drop(columns=['PERCENTAGE_ABS'])
    df_last = df[df['PERCENTAGE_ABS'] > 100].drop(columns=['PERCENTAGE_ABS'])

    print(f'\n***** ({df_closer.shape[0]}) < 10% *****\n')
    if not df_closer.empty:
        print(df_closer.to_string(index=False))
    else:
        print('EMPTY')

    print(f'\n***** ({df_middle.shape[0]}) > 10% *****\n')
    if not df_middle.empty:
        print(df_middle.to_string(index=False))
    else:
        print('EMPTY')

    print(f'\n***** ({df_last.shape[0]}) > 100% ****\n')
    if not df_last.empty:
        print(df_last.to_string(index=False))
    else:
        print('EMPTY')

# -----------SUMMARY INITIALIZATION----------------------------------------------------------------
count_valid_asset = 0
count_remaining_buys = 0
count_missing_buys = 0
count_all_remaining_buys = 0
count_valid_asset -= len(ASSETS_TO_EXCLUDE_AMOUNT)

# ------ SUMMARY ----------------------------------------------------------------------------------
if PRINT_ORDERS_SUMMARY:
    orders_summary_time_start = datetime.now(timezone.utc)
    print('\n *****ORDERS TO CREATE*****')

    for _, asset in sorted_pair_names_list_balance:
        asset_name = asset.output_name
        if not asset.trades:
            continue

        last_trade_price = asset.trades[0].price
        thr_sell = last_trade_price * (1 + ORDER_THR)
        thr_buy = last_trade_price * (1 - ORDER_THR)

        remaining_buys = max(BUY_LIMIT - asset.last_buys_count, 0)
        last_buy_amount = asset.last_buys_shares * asset.last_buys_avg_price
        buy_limit_reached = asset.check_buys_limit(BUY_LIMIT, MINIMUM_BUY_AMOUNT * BUY_LIMIT, last_buy_amount)
        buy_limit_amount_reached, margin_amount = asset.check_buys_amount_limit(BUY_LIMIT_AMOUNT)

        if asset.name not in ASSETS_TO_EXCLUDE_AMOUNT and remaining_buys:
            count_all_remaining_buys += remaining_buys
            if asset.orders_buy_amount:
                print('BUY order already set. Subtracting 1.') if SHOW_COUNT_BUYS else None
                count_all_remaining_buys -= 1

            if SHOW_COUNT_BUYS:
                print(f'Remaining buys: {remaining_buys} for pair: {asset_name}.')
                print(f'Count ALL buys: {count_all_remaining_buys}.\n')

        if asset_name in PAIR_TO_FORCE_INFO:
            print(BCOLORS.WARNING + f'FORCE INFO ON PAIR: {asset_name}' + BCOLORS.ENDC)

        oldest_sell_order = asset.oldest_order(type=OP_SELL)
        last_trade_execution = asset.trades[0].execution_datetime.replace(tzinfo=None)
        sell_lower_price = asset.orders_sell_lower_price
        cancel_condition = (USE_ORDER_THR and sell_lower_price and sell_lower_price >= thr_sell) or (
            oldest_sell_order and last_trade_execution > oldest_sell_order.creation_datetime
        )
        if cancel_condition:
            # SELL ORDERS
            perc = percentage(last_trade_price, asset.orders_sell_lower_price)
            print(
                BCOLORS.WARNING + f'Watch-out sell order greater than THR for pair: {asset.name}.'
                f'Order price: {my_round(asset.orders_sell_lower_price)}, last trade price: {my_round(last_trade_price)}, perc: {my_round(perc)} %. \n'  # noqa: E501
                f'Or is outdated, last_trade execution: {last_trade_execution}, oldest order creation: {oldest_sell_order.creation_datetime}.'  # noqa: E501
                 + BCOLORS.ENDC,
            )
            if AUTO_CANCEL_SELL_ORDER:
                print(BCOLORS.WARNING + f'Going to delete SELL orders from pair: {asset_name}.' + BCOLORS.ENDC)
                input("Press Enter to continue or Ctrl+D to exit")
                cancel_orders(kapi, OP_SELL, asset.orders)

        oldest_buy_order = asset.oldest_order(type=OP_BUY)
        buy_higher_price = asset.orders_buy_higher_price
        cancel_condition = (USE_ORDER_THR and buy_higher_price and buy_higher_price <= thr_buy) or (
            oldest_buy_order and last_trade_execution > oldest_buy_order.creation_datetime
        )
        if cancel_condition:
            # BUY ORDERS
            perc = percentage(last_trade_price, asset.orders_buy_higher_price)
            print(
                BCOLORS.WARNING + f'Watch-out buy order lower than THR for pair: {asset_name}.'
                f'Order price: {my_round(asset.orders_buy_higher_price)}, last trade price: {my_round(last_trade_price)}, perc: {my_round(perc)} %. \n'  # noqa: E501
                f'Or is outdated, last_trade execution: {last_trade_execution}, last order creation: {oldest_buy_order.creation_datetime}.'  # noqa: E501
                 + BCOLORS.ENDC,
            )

            if AUTO_CANCEL_BUY_ORDER:
                print(BCOLORS.WARNING + f'Going to delete BUY orders from pair: {asset_name}.' + BCOLORS.ENDC)
                input("Press Enter to continue or Ctrl+D to exit")
                cancel_orders(kapi, OP_BUY, asset.orders)

        if buy_limit_amount_reached:
            print(
                BCOLORS.WARNING + f'Watch-out BUY LIMIT AMOUNT of {BUY_LIMIT_AMOUNT} reached on asset: {asset_name}. '
                f'Margin amount: {my_round(margin_amount)}' + BCOLORS.ENDC,
            )

        if buy_limit_reached:
            print(
                BCOLORS.WARNING + f'Watch-out {BUY_LIMIT} consecutive BUYS on asset: {asset_name}. '
                f'Total buy amount: {my_round(asset.last_buys_shares * asset.last_buys_avg_price)}' + BCOLORS.ENDC,
            )

        if asset.orders_buy_count >= 2:
            print(BCOLORS.FAIL + 'Buy order duplicated for asset: {}'.format(asset_name) + BCOLORS.ENDC)

        if asset.orders_sell_count >= 2:
            print(BCOLORS.OKCYAN + 'Sell order duplicated for asset: {}'.format(asset_name) + BCOLORS.ENDC)

        if not asset.orders_buy_amount or asset.name in PAIR_TO_FORCE_INFO:
            if not buy_limit_reached or PRINT_BUYS_WARN_CONSECUTIVE or asset.name in PAIR_TO_FORCE_INFO:
                print(asset.print_buy_message(BUY_PERCENTAGE))

                if AUTO_BUY_ORDER:
                    asset.print_set_order_message(
                        order_type=OP_BUY,
                        order_percentage=BUY_PERCENTAGE,
                        minimum_order_amount=MINIMUM_BUY_AMOUNT,
                    )
                    # input("Press Enter to continue or Ctrl+D to exit")
                    # buy_order(kapi, OP_BUY)

            if not any(
                [
                    asset.orders_buy_amount,
                    buy_limit_reached,
                    buy_limit_amount_reached,
                    asset.name in ASSETS_TO_EXCLUDE_AMOUNT,
                ],
            ):
                count_remaining_buys += remaining_buys
                count_missing_buys += 1

        if not asset.orders_sell_amount or asset.name in PAIR_TO_FORCE_INFO:
            print(asset.print_sell_message(SELL_PERCENTAGE, MINIMUM_BUY_AMOUNT))

            if AUTO_SELL_ORDER:
                asset.print_set_order_message(
                    order_type=OP_SELL,
                    order_percentage=SELL_PERCENTAGE,
                    minimum_order_amount=MINIMUM_BUY_AMOUNT,
                )

        print('\n')

    elapsed_time_orders_summary = datetime.now(timezone.utc) - orders_summary_time_start
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
        f'ALL Needed cash (excluding existing buy orders): {all_cash_needed}.\n'
        f'Staked cash: {my_round(staked_eur)}',
    )

elapsed_time_since_begining = datetime.now(timezone.utc) - processing_time_start

if SHOW_SMART_SUMMARY:
    print('\n ***** SMART SUMMARY ***** ')
    smart_summary_time_start = datetime.now(timezone.utc)
    positions = [asset.to_dict() for asset in assets_dict.values()]
    agent_response = get_smart_summary(positions=positions, death_assets=death_asset_names, ia_agent=IA_AGENT)
    print(f'Agent response: \n {agent_response}')
    elapsed_time_smart_summary = datetime.now(timezone.utc) - smart_summary_time_start
    print(f'Smart summary latency: {elapsed_time_smart_summary}')

print('\n ***** TIME SUMMARY ***** ')
print(f'Endpoints latency: {elapsed_time_query_server}')
print(f'Load CSV time: {elapsed_time_csv_trades}')
print(f'Initialization time: {elapsed_time_initialization}')
print(f'Open orders time: {elapsed_time_open_orders}')
print(f'Last trades time: {elapsed_time_last_trades}')
print(f'Orders summary time: {elapsed_time_orders_summary}')
print(f'Total time: {elapsed_time_since_begining}')
