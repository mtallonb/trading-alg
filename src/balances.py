#!/usr/bin/python3

from datetime import date, datetime
from decimal import Decimal

import krakenex
import pandas as pd

from utils.basic import (
    FIX_X_PAIR_NAMES,
    from_timestamp_to_str,
    get_deposit_wd_info,
    get_fix_pair_name,
    my_round,
    read_trades_csv,
)

# Invested on each asset and current balance -> result No muy util
# balance a principio de año y actual quitando los flujos de IO

# date_from = date(2022, 1, 1)
# date_to = date(2023, 1, 1)

VERBOSE = True

filename = './data/trades_2024.csv'
file = None
buy_trades = []
sell_trades = []

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# REPLACE_NAMES = {'ETHEUR': 'XETHZEUR',
#                  'LTCEUR': 'XLTCEUR',
#                  'ETCEUR': 'XETCZEUR',
#                  'XMREUR': 'XXMRZEUR',
#                  'XRPEUR': 'XXRPZEUR',
#                  }

# configure api
kapi = krakenex.API()
kapi.load_key('./data/kraken.key')
PAGES = 5  # 50 RECORDS per page
RECORDS_PER_PAGE = 50

# prepare request
# req_data = {'trades': 'false'}

df_trades = pd.read_csv(filename)

df_trades.drop(columns=['ordertype'], inplace=True)
df_trades.rename({'pair': 'Asset', 'time': 'Datetime', 'cost': 'Amount'}, axis=1, inplace=True)
df_trades.columns = [x.upper() for x in df_trades.columns]


# Operation types D|W|B|S stands for Deposit, Withdrawal, Buy, Sell
df_trades['TYPE'].replace('buy', 'B', inplace=True)
df_trades['TYPE'].replace('sell', 'S', inplace=True)
df_trades.DATETIME = pd.to_datetime(df_trades.DATETIME)
df_trades['DATE'] = df_trades['DATETIME'].dt.floor('d')

# Get prices
# ---------------------------------------------------------------------------------------------------------
date_from = date(2017, 10, 1)
date_to = date(2019, 10, 1)
timestamp_from = datetime(date_from.year, date_from.month, date_from.day).timestamp()
timestamp_to = datetime(date_to.year, date_to.month, date_to.day).timestamp()

# ticker_prices = kapi.query_public('OHLC', {'pair': 'XETHZEUR', 'interval': 1440, 'since': timestamp_from})
# ticker_prices = cw.markets.get(f"kraken:{pair_name}", ohlc=True, ohlc_before=int(timestamp_to), periods=["1d"])
# df_prices = pd.DataFrame.from_dict(ticker_prices['result']['XETHZEUR'])

asset_names = df_trades[~df_trades.ASSET.isin(['XXLMXXBT', 'BSVEUR'])].ASSET.dropna().unique()
print(asset_names)
header = ["TIMESTAMP", "O", "H", "L", "C", "VOL", "TRADES"]
asset_names = [asset_names[0]]
for asset_name in asset_names:
    fix_asset_name = get_fix_pair_name(asset_name, FIX_X_PAIR_NAMES)
    df_prices = pd.read_csv(f'./data/OHLC_prices/{fix_asset_name}_1440.csv', names=header)[['TIMESTAMP', 'C']]
    df_prices.rename({'C': 'PRICE'}, axis=1, inplace=True)
    df_prices['DATE'] = pd.to_datetime(df_prices.TIMESTAMP, unit='s').dt.floor('d')
    # print(asset_name)
    # print(fix_asset_name)
    # print(df_prices)

# print(f"Last day: {from_timestamp_to_str(df_prices["DATETIME"].iloc[-1])}")
initial_date = df_prices["DATE"].iloc[0]
print(f'Initial day: {initial_date}')
last_date = df_prices["DATE"].iloc[-1]
print(f'Last day: {last_date}')

# ---------------------------------------------------------------------------------------------------------

# Fill prices
# ---------------------------------------------------------------------------------------------------------
# df_trades.sort_values(by=['DATETIME'], inplace=True)

df_trades_asset = df_trades[df_trades.ASSET == asset_name]

trades_initial_date = df_trades_asset["DATE"].iloc[0]

dates = pd.date_range(start=trades_initial_date, end=date_to, freq='d')
df_pos_temp = pd.DataFrame(columns=['DATE','ASSET','SHARES','PRICE'])
df_pos_temp.DATE = dates
df_pos_temp.ASSET = fix_asset_name
df_pos_temp.SHARES = 0.0

df_pos_temp.PRICE = df_pos_temp.DATE.map(df_prices.set_index('DATE')['PRICE'])

df_trades_asset[df_trades_asset.TYPE == 'S'].VOL *= -1
df_trades_asset['TOTAL_SHARES'] = df_trades_asset.VOL.cumsum()
# df_pos_temp.SHARES = 
# df_trades_with_prices = df_trades[df_trades.ASSET.isin([asset_name])].merge(df_prices, on='DATE')


# ---------------------------------------------------------------------------------------------------------
exit()
# ---------------------------------------------------------------------------------------------------------
total_deposit_amount, total_wd_amount, deposit_list, wd_list = get_deposit_wd_info(
    kapi,
    PAGES,
    RECORDS_PER_PAGE,
    verbose=False,
)
deposit_wd = Decimal(total_deposit_amount + total_wd_amount)
print('\n INVESTED (DEPOSITS-WD): {}'.format(my_round(deposit_wd)))
# ----------------------------------------------------------------------------------------------------------

df_deposit = pd.DataFrame.from_dict(deposit_list)
df_deposit = df_deposit[df_deposit['asset'] == 'ZEUR']
df_deposit = df_deposit.assign(TYPE='D')

df_wd = pd.DataFrame.from_dict(wd_list)
df_wd = df_wd[df_wd['asset'] == 'ZEUR']
df_wd = df_wd.assign(TYPE='W')

df_flows = pd.concat([df_deposit, df_wd])
df_flows.columns = [x.upper() for x in df_flows.columns]

df_flows = df_flows.assign(PRICE=1.0)
df_flows = df_flows.assign(FEE=0.0)
df_flows['VOL'] = df_flows.loc[:, 'AMOUNT']

df_movs = pd.concat([df_trades, df_flows])
df_movs.sort_values(by=['DATETIME'], inplace=True)
# df_movs.reset_index(drop=True)
# print(df_movs.to_markdown())
print(df_movs)
print(df_movs.dtypes)
# TABLE COLUMNS: ASSET | DATETIME | TYPE | PRICE | AMOUNT | FEE | VOL

df_asset_sum = df_movs.groupby(['ASSET', 'TYPE'])['VOL'].sum()

# for asset in df_asset_sum:
#     prices = cw.markets.get(f"kraken:{pair_name}", ohlc=True, ohlc_before=int(timestamp_to), periods=["1d"])

#     prices_df = pd.DataFrame(prices.of_1d)
#     # We take only dates and Close prices
#     close_prices = prices_df[[0, 4]]
#     close_prices.columns = ["date", "price"]
#     close_prices['date'] = pd.to_datetime(close_prices['date'], unit='s')
#     for close_price in close_prices.to_dict('records'):
#         day = close_price['date'].date()
#         price = close_price['price']
#         priceOHLC = PriceOHLC(float(price), float(price), float(price), float(price), day)
#         pp.prices.append(priceOHLC)

#     print(f"Initial price for ({pair_name}): {pp.prices[0]}")
#     print(f"Last price for ({pair_name}): {pp.prices[-1]}")


# for asset in df_movs.ASSET:

# def calculate_positions(df: pd.DataFrame, date_to: date):
#     df.groupby('ASSET')['']

exit()

latest_trade_csv = read_trades_csv(filename, buy_trades, sell_trades)

# Sort trades asc
buy_trades_asc = sorted(buy_trades, key=lambda x: x.completed)
sell_trades_asc = sorted(sell_trades, key=lambda x: x.completed)

# buy_trades_df = pd.DataFrame.from_dict()

# Buy /Sells summary
total_buy_amount = sum([buy.amount for buy in buy_trades])
total_sell_amount = sum([sell.amount for sell in sell_trades])
total_fees = sum([trade.fee for trade in buy_trades + sell_trades])


cash = deposit_wd - total_buy_amount + total_sell_amount - total_fees
print('\n CASH: {}'.format(my_round(cash)))

# balance on date = position(date) + cash(date) + stacked(date)
# balance = kapi.query_private('Balance')
# Comparar el último con el que obtengo de Kraken
# print(balance)
# El balance contiene los stacked tb asi que solo queda multiplicar por el precio cada position
# Calcular position diarios del fichero de trades y luego asignar precios diarios para calcular el balance diario
# OJO Hay que tener en cuenta los deposits y wd. En realidad los stacked se pueden obviar.
# gain_loss = balance / deposit_wd

# first_deposit_date = date(2017, 10, 31)
# first_deposit_date =
