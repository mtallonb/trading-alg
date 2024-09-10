#!/usr/bin/python3

from datetime import date
from decimal import Decimal

import krakenex
import pandas as pd

from utils.basic import get_deposit_wd_info, my_round, read_trades_csv

# Invested on each asset and current balance -> result No muy util
# balance a principio de año y actual quitando los flujos de IO

date_from = date(2022, 1, 1)
date_to = date(2023, 1, 1)

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
req_data = {'trades': 'false'}

df_trades = pd.read_csv(filename)


df_trades.drop(columns=['ordertype'], inplace=True)
df_trades.rename({'pair': 'Asset', 'time': 'Datetime', 'cost': 'Amount'}, axis=1, inplace=True)
df_trades.columns = [x.upper() for x in df_trades.columns]


# Operation types D|W|B|S stands for Deposit, Withdrawal, Buy, Sell
df_trades['TYPE'].replace('buy', 'B', inplace=True)
df_trades['TYPE'].replace('sell', 'S', inplace=True)
df_trades.DATETIME = pd.to_datetime(df_trades.DATETIME)

# ---------------------------------------------------------------------------------------------------------
total_deposit_amount, total_wd_amount, deposit_list, wd_list = get_deposit_wd_info(
    kapi,
    PAGES,
    RECORDS_PER_PAGE,
    verbose=False,
)
deposit_wd = Decimal(total_deposit_amount + total_wd_amount)
print('\n INVESTED (DEPOSITS-WD): {}'.format(my_round(deposit_wd)))
# --------------------------------------------------------------------------------------------------------------------

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

df_movs.groupby(['ASSET', 'TYPE'])['VOL'].sum()

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
