#!/usr/bin/python3

from datetime import datetime

import krakenex
import pandas as pd

from utils.basic import FIX_X_PAIR_NAMES, get_fix_pair_name, get_flow_from_kraken, my_round

# Invested on each asset and current balance -> result No muy util
# balance a principio de año y actual quitando los flujos de IO

# date_from = date(2022, 1, 1)
# date_to = date(2023, 1, 1)

# PANDAS CONF
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = "{:,.4f}".format

VERBOSE = True
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
PAGES = 4  # 50 RECORDS per page
RECORDS_PER_PAGE = 50  # Watchout is not working for higher values than 50
FLOW_TYPE_DEPOSIT = 'deposit'
FLOW_TYPE_WD = 'withdrawal'

filename = './data/trades_2024.csv'
deposit_filename = './data/deposits.csv'
wd_filename = './data/withdrawals.csv'
file = None
buy_trades = []
sell_trades = []

# date_to = date(2024, 11, 7)
date_to = datetime.today().date()
timestamp_to = datetime(date_to.year, date_to.month, date_to.day).timestamp()

# REPLACE_NAMES = {'ETHEUR': 'XETHZEUR',
#                  'LTCEUR': 'XLTCEUR',
#                  'ETCEUR': 'XETCZEUR',
#                  'XMREUR': 'XXMRZEUR',
#                  'XRPEUR': 'XXRPZEUR',
#                  }

# configure api
kapi = krakenex.API()
kapi.load_key('./data/kraken.key')

header_prices = ["TIMESTAMP", "O", "H", "L", "C", "VOL", "TRADES"]
header_prices_kraken = ["TIMESTAMP", "O", "H", "L", "C", "VWAP", "VOL", "TRADES"]
header_positions = ['DATE', 'ASSET', 'SHARES', 'PRICE', 'AMOUNT', 'FEE']

# prepare request
# req_data = {'trades': 'false'}


# -----------------------------------------
def get_asset_positions(
    asset_name: str,
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    date_to: datetime,
) -> pd.DataFrame:
    df_cash_pos = df_trades.copy()
    df_cash_pos.ASSET = 'ZEUR'
    df_cash_pos.PRICE = 1.0
    df_cash_pos.loc[df_trades.TYPE == 'B', 'AMOUNT'] *= -1
    # df_cash_pos['TOTAL_AMOUNT'] = df_cash_pos['AMOUNT'].cumsum()
    # df_cash_pos['AMOUNT'] = df_cash_pos.TOTAL_AMOUNT
    df_cash_pos['SHARES'] = df_cash_pos.AMOUNT
    df_cash_pos['FEE'] = 0.0
    df_cash_pos.drop(['VOL', 'DATETIME', 'TYPE'], axis=1, inplace=True)
    # df_cash_pos.drop_duplicates(subset=['DATE'], keep='last', inplace=True)

    dates = pd.date_range(start=df_trades["DATE"].iloc[0], end=date_to, freq='d')
    df_pos_temp = pd.DataFrame(columns=['DATE', 'ASSET', 'SHARES', 'PRICE'])
    df_pos_temp.DATE = dates
    df_pos_temp.ASSET = asset_name
    df_pos_temp.SHARES = 0.0

    df_pos_temp.PRICE = df_pos_temp.DATE.map(df_prices.set_index('DATE')['PRICE'])
    df_pos_temp.PRICE = pd.to_numeric(df_pos_temp.PRICE)

    # df_trades_asset[df_trades_asset.TYPE == 'S']['VOL'] *= -1
    df_trades.loc[df_trades.TYPE == 'S', 'VOL'] *= -1
    df_trades['TOTAL_SHARES'] = df_trades['VOL'].cumsum()

    df_pos_temp = pd.merge(df_pos_temp, df_trades.drop(['PRICE', 'ASSET'], axis=1), on='DATE', how='left')
    df_pos_temp.SHARES = df_pos_temp.TOTAL_SHARES
    df_pos_temp.drop(['VOL', 'DATETIME', 'TYPE', 'TOTAL_SHARES'], axis=1, inplace=True)
    df_pos_temp.SHARES.ffill(inplace=True)
    df_pos_temp.AMOUNT = df_pos_temp.SHARES * df_pos_temp.PRICE
    df_pos_temp.FEE.fillna(0, inplace=True)
    df_pos_temp.drop_duplicates(subset=['DATE'], keep='last', inplace=True)

    return pd.concat([df_pos_temp, df_cash_pos])


def get_new_prices(asset_name: str, timestamp_from: datetime.timestamp) -> pd.DataFrame:
    # If timestamp_from is higher than 2 years display a warning
    prices = kapi.query_public('OHLC', {'pair': asset_name, 'interval': 1440, 'since': timestamp_from})
    df_prices = pd.DataFrame.from_dict(prices['result'][asset_name])
    df_prices.columns = header_prices_kraken
    df_prices = df_prices[['TIMESTAMP', 'C']]

    return df_prices


def clean_flows_df(df_flow: pd.DataFrame) -> pd.DataFrame:
    df_flow.drop(['ACLASS', 'REFID', 'TYPE', 'SUBTYPE'], axis=1, inplace=True)
    df_flow.rename({'TIME': 'DATE', 'BALANCE': 'SHARES'}, axis=1, inplace=True)
    df_flow = df_flow[df_flow.ASSET == 'ZEUR']
    df_flow.DATE = pd.to_datetime(df_flow.DATE).dt.floor('d')
    df_flow.loc[df_flow.ASSET == 'ZEUR', 'SHARES'] = df_flow.AMOUNT
    df_flow.AMOUNT = pd.to_numeric(df_flow.AMOUNT)
    df_flow.SHARES = pd.to_numeric(df_flow.SHARES)
    df_flow['PRICE'] = df_flow.AMOUNT / df_flow.SHARES
    # df_flow.PRICE.fillna(1, inplace=True)

    return df_flow


def drop_cash_rows(df: pd.DataFrame) -> pd.DataFrame:
    df.reset_index(inplace=True)
    i = df[df.ASSET == 'ZEUR'].index
    df.drop(i, inplace=True)

    return df

def update_get_flow_file(flow_type: str):
    flow_filename = deposit_filename if flow_type == FLOW_TYPE_DEPOSIT else wd_filename
    df_flows = pd.read_csv(flow_filename)
    latest_flow_datetime = df_flows.TIME.iloc[-1]
    flow_datetime = datetime.fromisoformat(latest_flow_datetime)
    # deposit_datetime += timedelta(days=1)

    df_new_flows = get_flow_from_kraken(
        kapi=kapi,
        flow_type=flow_type,
        pages=PAGES,
        record_p_page=RECORDS_PER_PAGE,
        timestamp_from=flow_datetime.timestamp(),
    )

    df_new_flows = df_new_flows[df_new_flows.TIME > latest_flow_datetime]
    if not df_new_flows.empty:
        df_flows = pd.concat([df_flows, df_new_flows])
        df_flows['TIME'] = pd.to_datetime(df_flows.TIME)
        df_flows.sort_values(by=['TIME'], ascending=True, inplace=True, ignore_index=True)
        df_flows.to_csv(flow_filename, index=False)

    return df_flows

# ----GET DEPOSITS and WD-------------------------------------------------------------------------------------------
df_deposits = update_get_flow_file(flow_type=FLOW_TYPE_DEPOSIT)
df_wd = update_get_flow_file(flow_type=FLOW_TYPE_WD)
df_deposits = clean_flows_df(df_flow=df_deposits)
df_wd = clean_flows_df(df_flow=df_wd)

df_list = [df_deposits, df_wd]
# ----------GET TRADES-------------------------------------------------------------------
df_trades = pd.read_csv(filename)

df_trades.drop(columns=['ordertype'], inplace=True)
df_trades.rename({'pair': 'Asset', 'time': 'Datetime', 'cost': 'Amount'}, axis=1, inplace=True)
df_trades.columns = [x.upper() for x in df_trades.columns]

# Operation types D|W|B|S stands for Deposit, Withdrawal, Buy, Sell
df_trades['TYPE'].replace('buy', 'B', inplace=True)
df_trades['TYPE'].replace('sell', 'S', inplace=True)
df_trades.DATETIME = pd.to_datetime(df_trades.DATETIME)
df_trades['DATE'] = df_trades['DATETIME'].dt.floor('d')

# -------GET PRICES-------------------------------------------------------------------------------------------------
# ticker_prices = kapi.query_public('OHLC', {'pair': 'XETHZEUR', 'interval': 1440, 'since': timestamp_from})
# ticker_prices = cw.markets.get(f"kraken:{pair_name}", ohlc=True, ohlc_before=int(timestamp_to), periods=["1d"])
# df_prices = pd.DataFrame.from_dict(ticker_prices['result']['XETHZEUR'])

# Remove trades from 'XXLMXXBT', 'BSVEUR'
asset_names = df_trades[~df_trades.ASSET.isin(['XXLMXXBT', 'BSVEUR', 'WAVESEUR'])].ASSET.dropna().unique()
# asset_names = ['XXBTZEUR']

for asset_name in asset_names:
    fix_asset_name = get_fix_pair_name(asset_name, FIX_X_PAIR_NAMES)
    df_prices = pd.read_csv(f'./data/OHLC_prices/{fix_asset_name}_1440.csv', names=header_prices)[['TIMESTAMP', 'C']]
    latest_timestamp = df_prices.TIMESTAMP.iloc[-1]
    if latest_timestamp < timestamp_to:
        new_prices = get_new_prices(asset_name=asset_name, timestamp_from=latest_timestamp)
        df_prices = pd.concat([df_prices, new_prices])

    df_prices.rename({'C': 'PRICE'}, axis=1, inplace=True)
    df_prices['DATE'] = pd.to_datetime(df_prices.TIMESTAMP, unit='s').dt.floor('d')
    df_trades_asset = df_trades[df_trades.ASSET == asset_name]
    df_asset_pos = get_asset_positions(
        asset_name=fix_asset_name,
        df_trades=df_trades_asset,
        df_prices=df_prices,
        date_to=date_to,
    )
    df_list.append(df_asset_pos)

df_positions = pd.concat(df_list)
df_positions.sort_values(by=['DATE'], inplace=True)
df_positions = df_positions[df_positions.DATE.dt.date <= date_to]

df_positions['TOTAL_SHARES'] = df_positions['SHARES'].cumsum()
df_positions.loc[df_positions.ASSET == 'ZEUR', 'SHARES'] = df_positions.loc[df_positions.ASSET == 'ZEUR', 'SHARES'].cumsum()  # noqa # fmt: skip
df_positions.drop(columns=['TOTAL_SHARES'], inplace=True)
df_no_duplicates = df_positions.loc[df_positions.ASSET == 'ZEUR', :].drop_duplicates(subset=['DATE'], keep='last')

# df_positions.reset_index(inplace=True)
# i = df_positions[df_positions.ASSET == 'ZEUR'].index
# df_positions.drop(i, inplace=True)
df_positions = drop_cash_rows(df_positions)
df_positions = pd.concat([df_positions, df_no_duplicates])
df_positions['AMOUNT'] = df_positions.SHARES * df_positions.PRICE

dates = pd.date_range(start=df_positions["DATE"].iloc[0], end=date_to, freq='d')
df_cash_daily = pd.DataFrame(columns=['DATE'])
df_cash_daily.DATE = dates
df_cash_daily = pd.merge(df_cash_daily, df_positions.loc[df_positions.ASSET == 'ZEUR', :], on='DATE', how='left')
df_cash_daily.ASSET = 'ZEUR'
df_cash_daily.FEE = 0.0
df_cash_daily.PRICE = 1.0
df_cash_daily.SHARES.ffill(inplace=True)
df_cash_daily.AMOUNT.ffill(inplace=True)


# df_positions.reset_index(inplace=True)
# i = df_positions[df_positions.ASSET == 'ZEUR'].index
# df_positions.drop(i, inplace=True)
df_positions = drop_cash_rows(df_positions)
df_positions = pd.concat([df_positions, df_cash_daily])

df_positions.sort_values(by=['DATE'], inplace=True)

# # print(f"Last day: {from_timestamp_to_str(df_prices["DATETIME"].iloc[-1])}")
# initial_date = df_prices["DATE"].iloc[0]
# print(f'Initial day: {initial_date}')
# last_date = df_prices["DATE"].iloc[-1]
# print(f'Last day: {last_date}')

# ---SUMMARY------------------------------------------------------------------------

total_buy_amount = df_trades[df_trades.TYPE == 'B'].AMOUNT.sum()
total_sell_amount = df_trades[df_trades.TYPE == 'S'].AMOUNT.sum()
total_fees = df_trades.FEE.sum()

print('\n BUYS: {}'.format(my_round(total_buy_amount)))
print('\n SELLS: {}'.format(my_round(total_sell_amount)))
print('\n SELLS - BUYS: {}'.format(my_round(total_sell_amount - total_buy_amount)))
print('\n FEES: {}'.format(my_round(total_fees)))

total_deposit = df_deposits.AMOUNT.sum()
total_wd = -df_wd.AMOUNT.sum()
print('\n DEPOSIT: {}'.format(my_round(total_deposit)))
print('\n WD: {}'.format(my_round(total_wd)))
print('\n DEPOSIT - WD: {}'.format(my_round(total_deposit - total_wd)))

cash = total_deposit - total_wd - total_buy_amount + total_sell_amount - total_fees
print('\n CASH: {}'.format(my_round(cash)))

# AVG BALANCE
df_avg_balances_per_day = df_positions.groupby('DATE').AMOUNT.sum().reset_index()
# print(df_avg_balances_per_day.to_string())


# GAIN per year
def year_gain_perc(
    df_deposits: pd.DataFrame,
    df_wd: pd.DataFrame,
    df_balances_avg: pd.DataFrame,
    year: int,
    unrealised: float,
    verbose: bool = VERBOSE,
) -> float:
    deposit_amount_year = df_deposits[df_deposits.DATE.dt.year == year].AMOUNT.sum()
    wd_amount_year = -df_wd[df_wd.DATE.dt.year == year].AMOUNT.sum()
    daily_balances_year = df_balances_avg[df_balances_avg.DATE.dt.year == year]
    balance_0 = daily_balances_year.AMOUNT.iloc[0]
    balance_365 = daily_balances_year.AMOUNT.iloc[-1]
    flows = wd_amount_year - deposit_amount_year
    mean_balance = daily_balances_year.AMOUNT.mean()
    gain_numerator = balance_365 - balance_0 + flows
    gain = 100 * gain_numerator / mean_balance
    if verbose:
        print(
            f'\n***YEAR: {year}| balance_0: {my_round(balance_0)}| balance_365: {my_round(balance_365)}| mean_balance: {my_round(mean_balance)} \n'  # noqa: E501
            f'flows: {my_round(flows)}| gain_numerator: {my_round(gain_numerator)}| gain: {my_round(gain)}.',
        )
        print(f'Unrealised gain (perc): {my_round(100*unrealised/mean_balance)} \n')
    return gain


years = [2019, 2020, 2021, 2022, 2023, 2024]
gains_by_year = [246.0, 1154.7, 8533.0, 2421.2, 2700.0, 5000.0]
for idx, year in enumerate(years):
    gain = year_gain_perc(
        df_deposits=df_deposits,
        df_wd=df_wd,
        df_balances_avg=df_avg_balances_per_day,
        year=year,
        unrealised=gains_by_year[idx],
    )

# print(df_positions[df_positions.DATE.dt.date == date(2021, 5, 19)].to_string())

# df_positions[df_positions.DATE.dt.date == date_to - timedelta(days=1)]
# df_positions[df_positions.DATE.dt.date == date_to]

# _________________________________
exit()


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
