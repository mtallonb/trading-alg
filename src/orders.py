# coding=utf-8
#!/usr/bin/env python
# TODO
# Métricas para conocer valor de compra y cuantos asset podemos tener en cartera
# Dinero invertido en los asset muertos
# Compensar ganancias con las perdidas de las muertas.
# Ejecutar las pérdidas si hay mucha ganancia este año
# Mostrar si esta bloqueado el que esta a punto de vender
# Add accumulated B/S on LIST percentage to execute
# Meter todos los trades e indicarle cuando salir de los muertos. Voy a abrir una posición de compra o venta
# de acuerdo a la volatilidad en tal activo te paso mi ranking. Critica mi ranking.
# Ayudar al usuario en los parámetros de configuración
# Add sell count al comprar
# Guardar precios para cambio de moneda USDEUR o EURUSD


# RENAMING OF ASSETS:
# MATICEUR -> POLEUR
# EOSEUR -> AEUR

# DELISTING:
# LUNA, LUNA2, ETHW,

import io
import sys

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import krakenex
import pandas as pd

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
    is_staked,
    load_from_csv,
    my_round,
    percentage,
    print_smart_df,
    print_smart_df_multicolor,
    print_table,
    read_prices_from_local_file,
    run_smart_summary,
    smart_round,
)
from utils.classes import MAPPING_NAMES, OP_BUY, OP_SELL, Asset, Order, Trade

# -----ALG PARAMS------------------------------------------------------------------------------------------------------
BUY_LIMIT = 4  # Number of consecutive buy trades
BUY_PERCENTAGE = SELL_PERCENTAGE = 0.2  # Risk percentage to sell/buy 20%
MINIMUM_BUY_AMOUNT = 70
BUY_LIMIT_AMOUNT = (
    BUY_LIMIT * 0.5 * MINIMUM_BUY_AMOUNT
)  # Reached when buys - sells - balance (= -MARGIN_A, EUR lost on the asset) exceeds it
ORDER_THR = 0.35  # Umbral que consideramos error en la compra o venta a eliminar
USE_ORDER_THR = False  # Use ORDER_THR to cancel orders
IA_AGENT = "gemini"  # ['groq', 'gemini', 'openai']
SHOW_SMART_SUMMARY = False
# ----------------------------------------------------------------------------------------------------------------------
TRADE_PAGES = 2  # Latest Kraken trade pages (50 trades each) read when there is no CSV
LAST_ORDERS = 10
EXCLUDE_PAIR_NAMES = [
    'ZEUREUR', 'BSVEUR', 'LUNAEUR', 'SHIBEUR', 'ETH2EUR', 'WAVESEUR', 'XMREUR', 'EUR', 'EIGENEUR', 'APENFTEUR',
    'MATICEUR', 'EOSEUR', 'GOOGLxUSD', 'USDCEUR', 'ZUSDEUR', 'XLTCZEUR', 'XETHZEUR', 'XXRPZEUR', 'XXBTZEUR',
    'XETCZEUR', 'XXLMZEUR',
]  # fmt: off
# auto remove *.SEUR 'ATOM.SEUR', 'DOT.SEUR', 'XTZ.SEUR', 'EUR.MEUR']
ASSETS_TO_EXCLUDE_AMOUNT = [
    'SCEUR', 'DASHEUR', 'SGBEUR', 'SHIBEUR', 'LUNAEUR', 'LUNA2EUR', 'WAVESEUR', 'EIGENEUR', 'APENFTEUR',
    'MATICEUR',
]  # fmt: off

SMART_OUTPUT_DIR = './data/smart_outputs'


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


MAPPING_STAKING_NAME = {'BTC': 'XBTEUR'}
# DUAL_ASSETS_NAME = {'MATICEUR': 'POLEUR'}

# PAIR NAMES: [
# 'SCEUR', 'ATOMEUR', 'ETCEUR', 'ETHEUR', 'BCHEUR', 'TIAEUR', 'TRXEUR', 'XRPEUR', 'LINKEUR', 'XBTEUR',
# 'FLREUR', 'SNXEUR', 'EOSEUR', 'SOLEUR', 'XDGEUR', 'MINAEUR', 'LTCEUR', 'APTEUR', 'XLMEUR', 'UNIEUR', 'BATEUR',
# 'FLOWEUR', 'AAVEEUR', 'XTZEUR', 'ADAEUR', 'AVAXEUR', 'ALGOEUR', 'MATICEUR', 'SUIEUR', 'TRUMPEUR']
PAIR_TO_LAST_TRADES = []

PAIR_TO_FORCE_INFO = []  # ['XBTEUR', 'ADAEUR', 'SOLEUR']

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

TREND_THR = 0.6  # rescaled from 0.2 on the old [-1,1]-truncated-to-0 TREND scale, now [0,1] with 0.5 = neutral
AVG_SESSIONS = [200, 50, 10]  # Days of the price/volume averages used by TREND and VOL (at least 2)
EXPECTED_SELLS_DAYS = 200  # Days used to count expected sell trades (X_TRADES)


def build_assets(balance, open_orders, currency) -> dict[str, Asset]:
    """Create one Asset per pair with balance or open order, and fill its shares from the balance."""
    assets_dict: dict[str, Asset] = {}

    # Assets with balance or open order
    asset_original_names = list(balance['result'].keys())
    asset_original_names.extend(set([order['descr']['pair'] for order in open_orders['result']['open'].values()]))
    asset_original_names = set(asset_original_names)

    # ----------INITIALIZE PAIRS DICT-------------------------------------------------------------------
    for name in asset_original_names:
        key_name = name[1:] if len(name) > 2 and name[0] == name[1] == 'X' else name
        original_name = name + 'Z' if name[0] == 'X' else name
        original_name = original_name if original_name.endswith(currency) else original_name + currency
        key_name = get_fix_pair_name(pair_name=key_name, fix_x_pair_names=FIX_X_PAIR_NAMES)
        if key_name not in EXCLUDE_PAIR_NAMES and not is_staked(name=key_name) and not assets_dict.get(key_name, False):
            asset = Asset(name=key_name, original_name=original_name)
            assets_dict[key_name] = asset

    # ----------FILL BALANCE-------------------------------------------------------------------
    for key, value in balance['result'].items():
        key_name = key[1:] if len(key) > 2 and key[0] == key[1] == 'X' else key
        key_name = get_fix_pair_name(pair_name=key_name, fix_x_pair_names=FIX_X_PAIR_NAMES)
        if not is_staked(name=key_name) and key_name not in EXCLUDE_PAIR_NAMES and not assets_dict.get(key_name, False):
            print(f'Missing balance for pair: {key_name}')
            continue

        if key_name not in EXCLUDE_PAIR_NAMES:
            if not is_staked(name=key_name):
                assets_dict[key_name].shares = float(value)

    return assets_dict


def fill_prices_and_volumes(kapi, assets_dict: dict[str, Asset]):
    """Fill latest ticker info from Kraken and, optionally, local close prices and volumes."""
    yesterday = (datetime.today() - timedelta(days=1)).date()
    name_list = list(assets_dict.keys())
    concatenate_names = ','.join(name_list)
    # Watch-out is returning all assets with the latest price
    tickers_info = kapi.query_public('Ticker', {'pair': concatenate_names.lower()})
    # Example GOOGLxUSD tiene precios y habria que convertirlo a GOOGLxEUR que es el activo real que tengo
    # xstocks_info = kapi.query_public('Ticker', {'tokenized_asset': concatenate_xstock_names.lower()})
    # Usar la conversion del pair USDCEUR

    # print(f'Extra tickers on response: {set(tickers_info["result"].keys()) - set(name_list)}')

    for name, ticker_info in tickers_info['result'].items():
        fixed_pair_name = get_fix_pair_name(pair_name=name, fix_x_pair_names=FIX_X_PAIR_NAMES)
        asset = assets_dict.get(fixed_pair_name)
        if not asset:
            print(f'Ticker without asset: {fixed_pair_name}')
            continue

        asset.fill_ticker_info(ticker_info=ticker_info)
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


def fill_staking_info(kapi, assets_dict: dict[str, Asset]) -> float:
    """Fill staking info on each asset. Returns the staked EUR amount."""
    staked_eur = 0.0
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
            asset.fill_staking_info(staking_info=staking_info)
    return staked_eur


def fill_orders(open_orders, assets_dict: dict[str, Asset]):
    """Attach open orders to their assets. Returns (orders rows for pandas stats, buys_amount, sells_amount).

    Kraken OpenOrders is a dict keyed by txid with no documented order, so asset.orders keeps the response order
    and anything order-dependent must sort by creation_datetime (see Asset.oldest_order).
    """
    orders = []
    buys_amount = 0
    sells_amount = 0
    print('\n *****OPEN ORDERS READ*****')

    for txid, order_dict in open_orders.get('result').get('open').items():
        order_detail = order_dict['descr']
        # Same name normalization as build_assets so the lookup matches
        pair_name = get_fix_pair_name(pair_name=order_detail['pair'], fix_x_pair_names=FIX_X_PAIR_NAMES)

        price, shares = get_price_shares_from_order(order_string=order_detail['order'])
        amount = price * shares
        order = Order(txid=txid, order_type=order_detail['type'], shares=shares, price=price)
        order.creation_datetime = datetime.fromtimestamp(order_dict['opentm'])

        # Totals include every open order: it is cash committed even if the pair is not tracked
        if order.order_type == 'buy':
            buys_amount += amount
        else:
            sells_amount += amount

        asset = assets_dict.get(pair_name)
        if not asset:
            print(f'Order for untracked pair (excluded or unknown), only counted in totals: {pair_name}')
            continue

        asset.orders.append(order)
        if order.order_type == 'buy':
            asset.orders_buy_amount += amount
            asset.orders_buy_count += 1
            asset.update_orders_buy_higher_price(price=price)
        else:
            asset.orders_sell_amount += amount
            asset.orders_sell_count += 1
            asset.update_orders_sell_lower_price(price=price)

        # This array is used exclusively for Pandas stats
        orders.append(
            {
                'asset': pair_name,
                'order_type': order.order_type,
                'price': price,
                'current_price': asset.price,
            },
        )

    return orders, buys_amount, sells_amount


def fill_trades(kapi, assets_dict: dict[str, Asset], last_trade_from_csv):
    """Add the latest Kraken trades (not yet in the CSV) to their assets.

    With a CSV every trade after it is read (Kraken `start` = its last trade time, all pages); without it only
    the TRADE_PAGES latest pages. Kraken TradesHistory returns the most recent trades first (documented), so
    reading stops at the first trade not newer than last_trade_from_csv (`start` has second precision, so the
    last CSV trade can come back). Asset.trades must stay newest first: with a CSV (already loaded, newest
    first) the new trades are inserted on top oldest first; without it they are appended in Kraken order.
    """
    print('\n *****TRADES*****')
    asset_name = ''
    trade_pages = get_paginated_response_from_kraken(
        kapi=kapi,
        endpoint='TradesHistory',
        dict_key='trades',
        params={'trades': 'false'},
        pages=None if last_trade_from_csv else TRADE_PAGES,
        timestamp_from=int(last_trade_from_csv.execution_datetime.timestamp()) if last_trade_from_csv else None,
    )
    if not trade_pages:
        print(BCOLORS.WARNING + 'No trades Found' + BCOLORS.ENDC)

    trade = None
    csv_reached = False
    new_trades = []  # (asset, trade) newer than the CSV, newest first as returned by Kraken
    for trade_page in trade_pages:
        for trade_detail in trade_page.values():
            asset_name = get_fix_pair_name(pair_name=trade_detail['pair'], fix_x_pair_names=FIX_X_PAIR_NAMES)
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
                if last_trade_from_csv and trade.execution_datetime <= last_trade_from_csv.execution_datetime:
                    print(
                        f'CSV is updated from here on so we can leave the loop: {asset_name} '
                        f'{trade.execution_datetime}, last_trade from CSV: {last_trade_from_csv.execution_datetime}',
                    )
                    csv_reached = True
                    break

                if last_trade_from_csv:
                    # Kept in Kraken order (newest first); reversed when inserted at the end of this function
                    new_trades.append((asset, trade))
                else:
                    # No CSV: trades come newest first, so appending keeps trades[0] as the newest
                    asset.add_trade(trade=trade)
            else:
                print(f'Missing trade pair: {asset_name}')

        if csv_reached:
            print(
                f'Leaving main loop. CSV is UPDATED: {trade.execution_datetime}, '
                f'last_trade: {last_trade_from_csv.execution_datetime}',
            )
            break

    # Oldest trade read
    if asset_name and trade:
        print(BCOLORS.OKGREEN + f"Oldest trade date read for {asset_name}: {trade}" + BCOLORS.ENDC)

    if new_trades:
        print(BCOLORS.WARNING + f'CSV not updated: {len(new_trades)} trades newer than the CSV' + BCOLORS.ENDC)
    # Kraken order is reversed here: insert_trade_on_top puts each trade at trades[0], so inserting
    # oldest first leaves the newest trade at trades[0], on top of the trades loaded from the CSV
    # (inserting in Kraken order would leave the oldest new trade on top)
    for new_trade_asset, new_trade in reversed(new_trades):
        new_trade_asset.insert_trade_on_top(trade=new_trade)


def remove_assets_without_trades(assets_dict: dict[str, Asset]):
    """Fill latest_trade_date on each asset and delete (in place) the assets without trades."""
    keys_to_delete = []
    for key, asset in assets_dict.items():
        if asset.trades:
            asset.latest_trade_date = asset.trades[0].execution_datetime.date()
        else:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del assets_dict[key]


def print_last_trades(assets_dict: dict[str, Asset]):
    for asset_name in PAIR_TO_LAST_TRADES:
        asset = assets_dict.get(asset_name)
        if not asset:
            print(f'PAIR_TO_LAST_TRADES asset without balance, orders or trades: {asset_name}')
            continue
        print(f'\n**** Open orders for asset: {asset.output_name}.')
        for order in asset.orders[:LAST_ORDERS]:
            print(f'\n {order} ')

        print('\n**** Trades for asset: {}.'.format(asset.output_name))
        for trade in asset.trades[:LAST_ORDERS]:
            print(f'\n {trade}')


def build_ranking_rows(assets_dict: dict[str, Asset]) -> list[dict]:
    """Compute per-asset stats used for the ranking, one dict per asset keyed by ranking column."""
    assets_by_last_trade = []
    for _, asset in assets_dict.items():
        if not asset.trades:
            continue

        asset.compute_last_buy_sell_avg()

        if asset.latest_trade_date:
            sell_trades_count = asset.trades_sell_count
            last_buy_amount = asset.last_buys_shares * asset.last_buys_avg_price
            buy_limit_reached = asset.check_buys_limit(
                buy_limit=BUY_LIMIT,
                buy_limit_amount=MINIMUM_BUY_AMOUNT * BUY_LIMIT,
                buy_amount=last_buy_amount,
            )
            buy_limit_amount_reached, _ = asset.check_buys_amount_limit(buy_limit_amount=BUY_LIMIT_AMOUNT)
            buy_limit_reached = 1 if buy_limit_reached or buy_limit_amount_reached else 0
            margin_amount = asset.margin_amount
            has_prices = asset.close_prices is not None and not asset.close_prices.empty
            has_volumes = asset.close_volumes is not None and not asset.close_volumes.empty
            expected_sells = None
            if has_prices:
                expected_sells = count_sells_in_range(
                    close_prices=asset.close_prices,
                    days=EXPECTED_SELLS_DAYS,
                    buy_perc=BUY_PERCENTAGE,
                    sell_perc=SELL_PERCENTAGE,
                )

            # Column order matters: it is the order the ranking DataFrame is built with
            row = {
                'NAME': asset.name,
                'LAST_TRADE': asset.latest_trade_date,
                'IBS': asset.orders_buy_count,
                'BLR': buy_limit_reached,
                'CURR_PRICE': my_round(value=asset.price),
                'AVG_B': my_round(value=asset.avg_buys),
                'AVG_S': my_round(value=asset.avg_sells),
                'MARGIN_A': my_round(value=margin_amount),
                'S_TRADES': sell_trades_count,
                'X_TRADES': expected_sells,
            }
            for days in AVG_SESSIONS:
                avg_price = asset.avg_session_price(days=days) if has_prices else None
                row[f'AVG_PRICE_{days}'] = my_round(value=avg_price)
            for days in AVG_SESSIONS:
                avg_volume = asset.avg_session_volume(days=days) if has_volumes else None
                row[f'AVG_VOL_{days}'] = my_round(value=avg_volume)
            assets_by_last_trade.append(row)

    return assets_by_last_trade


def compute_and_print_ranking(assets_dict: dict[str, Asset], assets_by_last_trade: list[dict]) -> list[str]:
    """Compute the ranking, store it on each asset and print the ranking tables. Returns death asset names."""
    df = pd.DataFrame(assets_by_last_trade)
    ranking_df, detailed_ranking_df = compute_ranking(df=df, sessions=AVG_SESSIONS)

    for record in ranking_df[['NAME', 'RANKING']].to_dict('records'):
        assets_dict[record['NAME']].ranking = record['RANKING']

    table_title = (
        'PAIR NAMES BY RANKING: \n(IBD: Is Buy Set. BLR: Buy Limit Reached. '
        f'S_TRADES and X_TRADES: Sell trades and Expected Sell trades on {EXPECTED_SELLS_DAYS} sessions'
    )
    ranking_df.loc[:, 'NAME'] = ranking_df['NAME'].replace(MAPPING_NAMES)
    print_smart_df(df=ranking_df, exclude_columns=['IBS', 'BLR'], title=table_title)

    ranking_df_trending = ranking_df[ranking_df.TREND >= TREND_THR]
    if ranking_df_trending.empty:
        print(f'*** NO TRENDING PAIRS WITH TREND >= {TREND_THR} ***')
    else:
        table_title = f'PAIR NAMES with TREND >= {TREND_THR}'
        print_smart_df(df=ranking_df_trending, title=table_title)

    table_title = 'PAIR NAMES BY RANKING DETAILS: MARGIN_A: sells_amount + balance - buys_amount.'
    detailed_ranking_df.loc[:, 'NAME'] = detailed_ranking_df['NAME'].replace(MAPPING_NAMES)
    print_smart_df(df=detailed_ranking_df, title=table_title)

    # -------------------------------------------------------------------------------------------------
    live_asset_names = list(ranking_df[ranking_df.IBS == 1].NAME)
    death_asset_names = list(ranking_df[ranking_df.IBS == 0].NAME)
    print(f'\n*** LIVE ASSET NAMES ({len(live_asset_names)}): {live_asset_names}')
    print(f'\n*** DEATH ASSET NAMES ({len(death_asset_names)}): {death_asset_names}')

    return death_asset_names


def print_orders_proximity(orders: list[dict], assets_dict: dict[str, Asset]):
    """Print open orders grouped by distance (%) to the current price."""
    rules = [{'column': 'ACCUM_B', 'op': '>=', 'threshold': BUY_LIMIT, 'color': BCOLORS.WARNING}]

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

    table_title = f'({df_closer.shape[0]}) < 10%'
    if not df_closer.empty:
        print_smart_df_multicolor(
            df=df_closer,
            exclude_columns=['ACCUM_B', 'ACCUM_S'],
            title=table_title,
            highlight_rules=rules,
        )
    else:
        print(table_title)
        print('EMPTY')

    table_title = f'({df_middle.shape[0]}) > 10%'
    if not df_middle.empty:
        print_smart_df_multicolor(
            df=df_middle,
            exclude_columns=['ACCUM_B', 'ACCUM_S'],
            title=table_title,
            highlight_rules=rules,
        )
    else:
        print(table_title)
        print('EMPTY')

    table_title = f'({df_last.shape[0]}) > 100%'
    if not df_last.empty:
        print_smart_df(df=df_last, exclude_columns=['ACCUM_B', 'ACCUM_S'], title=table_title)
    else:
        print(table_title)
        print('EMPTY')


def print_orders_to_create(kapi, sorted_pair_names_list_balance):
    """Print buy/sell warnings and order suggestions per asset, cancelling outdated orders if enabled.

    Returns (count_missing_buys, count_remaining_buys, count_all_remaining_buys).
    """
    count_remaining_buys = 0
    count_missing_buys = 0
    count_all_remaining_buys = 0

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
        buy_limit_reached = asset.check_buys_limit(
            buy_limit=BUY_LIMIT,
            buy_limit_amount=MINIMUM_BUY_AMOUNT * BUY_LIMIT,
            buy_amount=last_buy_amount,
        )
        buy_limit_amount_reached, _ = asset.check_buys_amount_limit(buy_limit_amount=BUY_LIMIT_AMOUNT)

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
            perc = percentage(a=last_trade_price, b=asset.orders_sell_lower_price)
            print(
                BCOLORS.WARNING + f'Watch-out sell order greater than THR for pair: {asset.name}.'
                f'Order price: {my_round(value=asset.orders_sell_lower_price)}, last trade price: {my_round(value=last_trade_price)}, perc: {my_round(value=perc)} %. \n'  # noqa: E501
                f'Or is outdated, last_trade execution: {last_trade_execution}, oldest order creation: {oldest_sell_order.creation_datetime}.'  # noqa: E501
                 + BCOLORS.ENDC,
            )
            if AUTO_CANCEL_SELL_ORDER:
                print(BCOLORS.WARNING + f'Going to delete SELL orders from pair: {asset_name}.' + BCOLORS.ENDC)
                input("Press Enter to continue or Ctrl+D to exit")
                cancel_orders(kapi=kapi, order_type=OP_SELL, orders=asset.orders)

        oldest_buy_order = asset.oldest_order(type=OP_BUY)
        buy_higher_price = asset.orders_buy_higher_price
        cancel_condition = (USE_ORDER_THR and buy_higher_price and buy_higher_price <= thr_buy) or (
            oldest_buy_order and last_trade_execution > oldest_buy_order.creation_datetime
        )
        if cancel_condition:
            # BUY ORDERS
            perc = percentage(a=last_trade_price, b=asset.orders_buy_higher_price)
            print(
                BCOLORS.WARNING + f'Watch-out buy order lower than THR for pair: {asset_name}.'
                f'Order price: {my_round(value=asset.orders_buy_higher_price)}, last trade price: {my_round(value=last_trade_price)}, perc: {my_round(value=perc)} %. \n'  # noqa: E501
                f'Or is outdated, last_trade execution: {last_trade_execution}, last order creation: {oldest_buy_order.creation_datetime}.'  # noqa: E501
                 + BCOLORS.ENDC,
            )

            if AUTO_CANCEL_BUY_ORDER:
                print(BCOLORS.WARNING + f'Going to delete BUY orders from pair: {asset_name}.' + BCOLORS.ENDC)
                input("Press Enter to continue or Ctrl+D to exit")
                cancel_orders(kapi=kapi, order_type=OP_BUY, orders=asset.orders)

        if buy_limit_amount_reached:
            print(
                BCOLORS.WARNING + f'Watch-out BUY LIMIT AMOUNT of {BUY_LIMIT_AMOUNT} reached on asset: {asset_name}. '
                f'Net margin (MARGIN_A): {my_round(value=asset.margin_amount)}' + BCOLORS.ENDC,
            )

        if buy_limit_reached:
            print(
                BCOLORS.WARNING + f'Watch-out {BUY_LIMIT} consecutive BUYS on asset: {asset_name}. '
                f'Total buy amount: {my_round(value=asset.last_buys_shares * asset.last_buys_avg_price)}'
                + BCOLORS.ENDC,
            )

        if asset.orders_buy_count >= 2:
            print(BCOLORS.FAIL + 'Buy order duplicated for asset: {}'.format(asset_name) + BCOLORS.ENDC)

        if asset.orders_sell_count >= 2:
            print(BCOLORS.OKCYAN + 'Sell order duplicated for asset: {}'.format(asset_name) + BCOLORS.ENDC)

        if not asset.orders_buy_amount or asset.name in PAIR_TO_FORCE_INFO:
            if (
                (not buy_limit_reached and not buy_limit_amount_reached)
                or PRINT_BUYS_WARN_CONSECUTIVE
                or asset.name in PAIR_TO_FORCE_INFO
            ):
                asset.print_buy_message(
                    gain_perc=BUY_PERCENTAGE,
                    minimum_buy_amount=MINIMUM_BUY_AMOUNT,
                    sessions=AVG_SESSIONS,
                )

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
            asset.print_sell_message(gain_perc=SELL_PERCENTAGE, minimum_amount=MINIMUM_BUY_AMOUNT)

            if AUTO_SELL_ORDER:
                asset.print_set_order_message(
                    order_type=OP_SELL,
                    order_percentage=SELL_PERCENTAGE,
                    minimum_order_amount=MINIMUM_BUY_AMOUNT,
                )

    return count_missing_buys, count_remaining_buys, count_all_remaining_buys


def print_cash_summary(
    sells_amount,
    buys_amount,
    cash_eur,
    staked_eur,
    count_missing_buys,
    count_remaining_buys,
    count_all_remaining_buys,
):
    # Common column configuration using multiline headers for better fit
    # Format: (key, label, alignment)
    cols_config = [
        ("desc", "Description", "<"),
        ("val", "Value", ">"),
    ]

    # --- TABLE 1: TRADING ACTIVITY ---
    trading_activity_data = [
        {"desc": "Total Sell amount", "val": smart_round(number=sells_amount)},
        {"desc": "Total Buys amount", "val": smart_round(number=buys_amount)},
    ]
    print_table(data=trading_activity_data, columns=cols_config, title="1. TRADING ACTIVITY")

    # --- TABLE 2: CASH STATUS ---
    remaining_val = smart_round(number=cash_eur - buys_amount)
    highlighted_cash = BCOLORS.WARNING + f"{remaining_val}" + BCOLORS.ENDC

    cash_status_data = [
        {"desc": "Remaining Cash (EUR)", "val": highlighted_cash},
        {"desc": "Staked cash", "val": smart_round(number=staked_eur)},
    ]
    print_table(data=cash_status_data, columns=cols_config, title="2. CASH STATUS")

    # --- TABLE 3: FUTURE PROJECTIONS ---
    cash_needed_missing_buy = count_missing_buys * MINIMUM_BUY_AMOUNT
    cash_needed = count_remaining_buys * MINIMUM_BUY_AMOUNT
    all_cash_needed = count_all_remaining_buys * MINIMUM_BUY_AMOUNT
    # We provide raw values for counts and rounded values for currency
    projections_data = [
        {"desc": "Count missing buys", "val": count_missing_buys},
        {"desc": "Needed cash (missing)", "val": smart_round(number=cash_needed_missing_buy)},
        {"desc": "Count remaining buys", "val": count_remaining_buys},
        {"desc": "Needed cash (remaining)", "val": smart_round(number=cash_needed)},
        {"desc": "Count ALL remaining buys", "val": count_all_remaining_buys},
        {"desc": "ALL Needed (Worst Case)", "val": smart_round(number=all_cash_needed)},
    ]
    print_table(data=projections_data, columns=cols_config, title="3. FUTURE PROJECTIONS")


@contextmanager
def timer(timings: dict, label: str):
    """Store in timings[label] the elapsed time of the with-block."""
    start = datetime.now(timezone.utc)
    try:
        yield
    finally:
        timings[label] = datetime.now(timezone.utc) - start


TIME_SUMMARY_LABELS = [
    'Endpoints latency',
    'Load CSV time',
    'Initialization time',
    'Open orders time',
    'Last trades time',
    'Orders summary time',
    'Total time',
]


def print_time_summary(timings: dict):
    """Print timings in TIME_SUMMARY_LABELS order ('skipped' if the phase didn't run), then any extra label."""
    print('\n ***** TIME SUMMARY ***** ')
    for label in TIME_SUMMARY_LABELS:
        print(f'{label}: {timings.get(label, "skipped")}')
    for label, elapsed_time in timings.items():
        if label not in TIME_SUMMARY_LABELS:
            print(f'{label}: {elapsed_time}')


def main():
    captured_output = io.StringIO()
    sys.stdout = Tee(sys.stdout, captured_output)

    # configure api
    kapi = krakenex.API()
    kapi.load_key(KEY_FILE)

    # prepare request
    # req_data = {'docalcs': 'true'}

    timings = {}

    # CALLS TO KRAKEN API
    # kapi.query_public('Ticker', {'pair': concatenate_names.lower()})
    with timer(timings=timings, label='Endpoints latency'):
        balance = kapi.query_private('Balance')
    # open_orders = kapi.query_private('OpenOrders', data={'trades': 'false'})
    # staked_assets = kapi.query_private('Earn/Allocations', data={'hide_zero_allocations': 'true'})
    # trade_balance = kapi.query_private('TradeBalance')
    # close_orders = kapi.query_private('CloseOrders', req_data)
    # trades_history = kapi.query_private('TradesHistory', req_data)

    # end = kapi.query_public('Time')
    # latency = end['result']['unixtime'] - start['result']['unixtime']
    currency = 'EUR'

    # EUR balance
    cash_eur = float(balance['result']['ZEUR'])

    processing_time_start = datetime.now(timezone.utc)

    # Pandas conf
    # Float output format
    PANDAS_FLOAT_FORMAT = '{:.3f}'.format
    pd.options.display.float_format = PANDAS_FLOAT_FORMAT

    # ------------------------------------------------
    with timer(timings=timings, label='Initialization time'):
        open_orders = kapi.query_private('OpenOrders', data={'trades': 'false'})
        assets_dict = build_assets(balance=balance, open_orders=open_orders, currency=currency)
        fill_prices_and_volumes(kapi=kapi, assets_dict=assets_dict)
        staked_eur = fill_staking_info(kapi=kapi, assets_dict=assets_dict)

    # ----------SORTING BY BALANCE-------------------------------------------------------------------
    print(f'\n *****PAIR NAMES SORTED BY BALANCE TOTAL: {len(assets_dict)} *****')
    # Sort dict by balance descending
    sorted_pair_names_list_balance = sorted(assets_dict.items(), key=lambda x: x[1].balance, reverse=True)
    name_list = [ele[0] for ele in sorted_pair_names_list_balance]
    [print(ele) for ele in chunks(elem_list=name_list, n=5)]

    # ----------FILL ORDERS-------------------------------------------------------------------
    with timer(timings=timings, label='Open orders time'):
        orders, buys_amount, sells_amount = fill_orders(open_orders=open_orders, assets_dict=assets_dict)

    # ----------FILL TRADES-------------------------------------------------------------------
    last_trade_from_csv = None
    with timer(timings=timings, label='Load CSV time'):
        if GET_FULL_TRADE_HISTORY:
            # Load trades from CSV
            last_trade_from_csv = load_from_csv(
                filename=TRADE_FILE,
                assets_dict=assets_dict,
                fix_x_pair_names=FIX_X_PAIR_NAMES,
            )

    with timer(timings=timings, label='Last trades time'):
        fill_trades(kapi=kapi, assets_dict=assets_dict, last_trade_from_csv=last_trade_from_csv)
        remove_assets_without_trades(assets_dict=assets_dict)
        if PRINT_LAST_TRADES:
            print_last_trades(assets_dict=assets_dict)

    # ----------FILL CALCULATIONS FROM LAST TRADES-------------------------------------------------------------------
    assets_by_last_trade = build_ranking_rows(assets_dict=assets_dict)

    # ------ RANKING ----------------------------------------------------------------------------------
    death_asset_names = compute_and_print_ranking(assets_dict=assets_dict, assets_by_last_trade=assets_by_last_trade)

    if PRINT_PERCENTAGE_TO_EXECUTE_ORDERS:
        print_orders_proximity(orders=orders, assets_dict=assets_dict)

    # ------ SUMMARY ----------------------------------------------------------------------------------
    if PRINT_ORDERS_SUMMARY:
        with timer(timings=timings, label='Orders summary time'):
            count_missing_buys, count_remaining_buys, count_all_remaining_buys = print_orders_to_create(
                kapi=kapi,
                sorted_pair_names_list_balance=sorted_pair_names_list_balance,
            )

        print_cash_summary(
            sells_amount=sells_amount,
            buys_amount=buys_amount,
            cash_eur=cash_eur,
            staked_eur=staked_eur,
            count_missing_buys=count_missing_buys,
            count_remaining_buys=count_remaining_buys,
            count_all_remaining_buys=count_all_remaining_buys,
        )

    timings['Total time'] = datetime.now(timezone.utc) - processing_time_start

    if SHOW_SMART_SUMMARY:
        positions = [asset.to_dict() for asset in assets_dict.values()]
        run_smart_summary(
            positions=positions,
            death_assets=death_asset_names,
            ia_agent=IA_AGENT,
            captured_output=captured_output,
            local_tz=LOCAL_TZ,
            output_dir=SMART_OUTPUT_DIR,
        )
    else:
        sys.stdout = sys.stdout.streams[0]

    print_time_summary(timings=timings)


if __name__ == '__main__':
    main()
