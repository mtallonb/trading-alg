#!/usr/bin/python3

from datetime import datetime, timedelta

import krakenex
import pandas as pd

from utils.basic import (
    FIX_X_PAIR_NAMES,
    PRICES_DIR,
    from_date_to_datetime_aware,
    from_date_to_timestamp,
    get_fix_pair_name,
    get_new_prices,
    get_paginated_response_from_kraken,
    my_round,
    read_prices_from_local_file,
    timestamp_df_to_date_df,
)

# Invested on each asset and current balance -> result No muy util
# Fix unrealised gain on asset delisting. Venta forzosa no se gana el 20%
# Dates instead of timestamps csv close files

# date_from = date(2022, 1, 1)
# date_to = date(2023, 1, 1)

# PANDAS CONF
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = "{:,.4f}".format

VERBOSE = True
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
PAGES = 4  # 50 RECORDS per page
RECORDS_PER_PAGE = 50  # Watch-out is not working for higher values than 50
FLOW_TYPE_DEPOSIT = 'deposit'
FLOW_TYPE_WD = 'withdrawal'

year = 2025
filename = f'./data/trades_{year}.csv'
deposit_filename = './data/deposits.csv'
wd_filename = './data/withdrawals.csv'
file = None
buy_trades = []
sell_trades = []

yesterday = (datetime.today() - timedelta(days=1)).date()
date_to = yesterday
datetime_to = from_date_to_datetime_aware(day=yesterday)
timestamp_to = datetime_to.timestamp()

# REPLACE_NAMES = {'ETHEUR': 'XETHZEUR',
#                  'LTCEUR': 'XLTCEUR',
#                  'ETCEUR': 'XETCZEUR',
#                  'XMREUR': 'XXMRZEUR',
#                  'XRPEUR': 'XXRPZEUR',
#                  }

# configure api
kapi = krakenex.API()
kapi.load_key('./data/keys/kraken.key')

# prepare request
# req_data = {'trades': 'false'}


# -----Func definitions--------------------------------------------------
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
    df_pos_temp.DATE = df_pos_temp.DATE.dt.date
    df_pos_temp.ASSET = asset_name
    df_pos_temp.SHARES = 0.0

    df_pos_temp.PRICE = df_pos_temp.DATE.map(df_prices.set_index('DATE')['PRICE'])
    df_pos_temp.PRICE = pd.to_numeric(df_pos_temp.PRICE)

    # df_trades_asset[df_trades_asset.TYPE == 'S']['VOL'] *= -1
    df_trades.loc[df_trades.TYPE == 'S', 'VOL'] *= -1
    df_trades['TOTAL_SHARES'] = df_trades['VOL'].cumsum()
    df_trades = df_trades.drop(columns=['PRICE', 'ASSET'])

    df_pos_temp = pd.merge(df_pos_temp, df_trades, on='DATE', how='left')
    df_pos_temp.SHARES = df_pos_temp.TOTAL_SHARES
    df_pos_temp.drop(['VOL', 'DATETIME', 'TYPE', 'TOTAL_SHARES'], axis=1, inplace=True)
    df_pos_temp['SHARES'] = df_pos_temp['SHARES'].ffill()
    df_pos_temp.AMOUNT = df_pos_temp.SHARES * df_pos_temp.PRICE
    df_pos_temp['FEE'] = df_pos_temp['FEE'].fillna(0)
    df_pos_temp.drop_duplicates(subset=['DATE'], keep='last', inplace=True)

    return pd.concat([df_pos_temp, df_cash_pos])


def clean_flows_df(df_flow: pd.DataFrame) -> pd.DataFrame:
    df_flow.drop(['ACLASS', 'REFID', 'TYPE', 'SUBTYPE'], axis=1, inplace=True)
    df_flow.rename({'TIME': 'DATE', 'BALANCE': 'SHARES'}, axis=1, inplace=True)
    df_flow = df_flow[df_flow.ASSET == 'ZEUR']
    df_flow.DATE = pd.to_datetime(df_flow.DATE).dt.date
    df_flow.loc[df_flow.ASSET == 'ZEUR', 'SHARES'] = df_flow.AMOUNT
    df_flow.AMOUNT = pd.to_numeric(df_flow.AMOUNT)
    df_flow.SHARES = pd.to_numeric(df_flow.SHARES)
    df_flow['PRICE'] = df_flow.AMOUNT / df_flow.SHARES
    # df_flow.PRICE.fillna(1, inplace=True)

    return df_flow


def drop_cash_rows(df: pd.DataFrame) -> pd.DataFrame:
    # df.reset_index(inplace=True)
    i = df[df.ASSET == 'ZEUR'].index
    df.drop(i, inplace=True)

    return df


def update_get_flow_file(flow_type: str):
    flow_filename = deposit_filename if flow_type == FLOW_TYPE_DEPOSIT else wd_filename
    df_flows = pd.read_csv(flow_filename)
    latest_flow_datetime = df_flows.TIME.iloc[-1]
    flow_datetime = pd.to_datetime(latest_flow_datetime)
    if isinstance(flow_datetime, pd.Timestamp):
        flow_datetime = flow_datetime.to_pydatetime(warn=False)

    new_flow_pages = get_paginated_response_from_kraken(
        kapi=kapi,
        endpoint='Ledgers',
        dict_key='ledger',
        params={'type': flow_type},
        pages=PAGES,
        records_per_page=RECORDS_PER_PAGE,
        is_private=True,
        timestamp_from=flow_datetime.timestamp(),
    )
    df_new_flows = pd.DataFrame([rec for page in new_flow_pages for rec in page.values()])
    df_new_flows.columns = [x.upper() for x in df_new_flows.columns]
    df_new_flows.TIME = pd.to_datetime(df_new_flows.TIME, unit='s')
    df_new_flows = df_new_flows[df_new_flows.TIME > latest_flow_datetime]
    if not df_new_flows.empty:
        df_flows = pd.concat([df_flows, df_new_flows])
        df_flows['TIME'] = pd.to_datetime(df_flows.TIME)
        df_flows.sort_values(by=['TIME'], ascending=True, inplace=True, ignore_index=True)
        df_flows.to_csv(flow_filename, index=False)

    return df_flows


# GAIN per year
def year_gain_perc(
    df_deposits: pd.DataFrame,
    df_wd: pd.DataFrame,
    df_balances_avg: pd.DataFrame,
    year: int,
    unrealised: float,
    verbose: bool = VERBOSE,
) -> float:
    df_deposits.DATE = pd.to_datetime(df_deposits.DATE)
    df_wd.DATE = pd.to_datetime(df_wd.DATE)
    df_balances_avg.DATE = pd.to_datetime(df_balances_avg.DATE)
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
            f'flows: {my_round(flows)}| gain_numerator: {my_round(gain_numerator)}| gain: {my_round(gain)} %.',
        )
        print(f'Unrealised gain (perc): {my_round(100 * unrealised / mean_balance)} %.\n')
    return gain


# --------End of functions------------------------------------------------------------------------------------------

# ----GET DEPOSITS and WD-------------------------------------------------------------------------------------------
df_deposits = update_get_flow_file(flow_type=FLOW_TYPE_DEPOSIT)
df_wd = update_get_flow_file(flow_type=FLOW_TYPE_WD)
df_deposits = clean_flows_df(df_flow=df_deposits)
df_wd = clean_flows_df(df_flow=df_wd)

df_list = [df_deposits, df_wd]
# ----------GET TRADES-------------------------------------------------------------------
df_trades = pd.read_csv(filename)

df_trades.drop(columns=['ordertype'], inplace=True)
df_trades.rename({'pair': 'Asset', 'time(UTC)': 'Datetime', 'cost': 'Amount'}, axis=1, inplace=True)
df_trades.columns = [x.upper() for x in df_trades.columns]

# Operation types D|W|B|S stands for Deposit, Withdrawal, Buy, Sell
df_trades['TYPE'] = df_trades['TYPE'].replace('buy', 'B')
df_trades['TYPE'] = df_trades['TYPE'].replace('sell', 'S')
df_trades.DATETIME = pd.to_datetime(df_trades.DATETIME)
df_trades['DATE'] = df_trades['DATETIME'].dt.date

# Ojo con este trade
# XXLMXXBT,2018-02-01 17:33:11,buy,limit,0.00004885,0.014990794,0.000038976,306.8739771
# reemplazar por
# XXBTZEUR,2018-02-01 17:33:11,sell,limit,7163,107.3,0,0.014990794
# XXLMZEUR,2018-02-01 17:33:11,buy,limit,0.35,107.3,0.2,306.8739771


# -------GET PRICES-------------------------------------------------------------------------------------------------
# ticker_prices = kapi.query_public('OHLC', {'pair': 'XETHZEUR', 'interval': 1440, 'since': timestamp_from})
# ticker_prices = cw.markets.get(f"kraken:{pair_name}", ohlc=True, ohlc_before=int(timestamp_to), periods=["1d"])
# df_prices = pd.DataFrame.from_dict(ticker_prices['result']['XETHZEUR'])

# Remove trades from 'XXLMXXBT', 'BSVEUR'
asset_names = df_trades[~df_trades.ASSET.isin(['XXLMXXBT', 'BSVEUR', 'WAVESEUR'])].ASSET.dropna().unique()
# asset_names = ['XXBTZEUR']
# asset_names = ['TRUMPEUR']

for asset_name in asset_names:
    latest_date = yesterday - timedelta(days=600)
    fix_asset_name = get_fix_pair_name(asset_name, FIX_X_PAIR_NAMES)
    df_prices, _ = read_prices_from_local_file(asset_name=fix_asset_name)
    if not df_prices.empty:
        latest_date = df_prices.DATE.iloc[-1]

    if latest_date < date_to:
        new_prices = get_new_prices(
            kapi=kapi,
            asset_name=asset_name,
            timestamp_from=from_date_to_timestamp(day=latest_date),
            with_volumes=True,
        )
        if new_prices is not None:
            new_prices = timestamp_df_to_date_df(df=new_prices)
            df_prices = pd.concat([df_prices, new_prices])
            df_prices = df_prices.drop_duplicates(subset=['DATE'])

            df_prices.to_csv(f'{PRICES_DIR}{fix_asset_name}_DAILY_WITH_VOLUME.csv', index=False)

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
df_positions.dropna(subset=['AMOUNT'], inplace=True)
df_positions.reset_index(inplace=True)
df_positions = df_positions[df_positions.DATE <= date_to]

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
df_cash_daily.DATE = df_cash_daily.DATE.dt.date
df_cash_daily = pd.merge(df_cash_daily, df_positions.loc[df_positions.ASSET == 'ZEUR', :], on='DATE', how='left')
df_cash_daily.ASSET = 'ZEUR'
df_cash_daily.FEE = 0.0
df_cash_daily.PRICE = 1.0
df_cash_daily['SHARES'] = df_cash_daily['SHARES'].ffill()
df_cash_daily['AMOUNT'] = df_cash_daily['AMOUNT'].ffill()

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
# print(df_positions[df_positions.ASSET == 'XBTEUR'].to_string())


years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
gains_by_year = [246.0, 1154.7, 8533.0, 2421.2, 2700.0, 6000.0, 3120.0]
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
