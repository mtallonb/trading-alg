#!/usr/bin/python3

from datetime import date, datetime, timedelta, timezone

import krakenex
import pandas as pd

from utils.basic import (
    FIX_X_PAIR_NAMES,
    FX_PAIR,
    REALISED_GAINS_FILE,
    get_fix_pair_name,
    get_paginated_response_from_kraken,
    is_eur_pair,
    is_xstock_pair,
    my_round,
    print_table,
    read_realised_gains,
    smart_round,
    update_asset_prices,
)

# Invested on each asset and current balance -> result not very useful
# Fix unrealised gain on asset delisting. Forced sale: the 20% is not gained

# PANDAS CONF
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = "{:,.4f}".format

VERBOSE = True
TRADES_FILE = './data/trades_2026.csv'
DEPOSITS_FILE = './data/deposits.csv'
WITHDRAWALS_FILE = './data/withdrawals.csv'
KEY_FILE = './data/keys/kraken.key'
FLOW_TYPE_DEPOSIT = 'deposit'
FLOW_TYPE_WD = 'withdrawal'
CASH_ONLY_ASSETS = ['BSVEUR']  # Sold airdropped coins without buys: only their cash is counted, no position
USD_ASSET = 'ZUSD'
# Balance moves without a trade (no cash involved), from the Kraken ledger
ASSET_CONVERSIONS = [
    # WAVES delisted and converted to BTC (refids LABGJBC-TH6PA-CGKUYI, LA72OX5-EHUGC-B3OXJE)
    {'ASSET': 'WAVESEUR', 'DATETIME': '2024-10-11 09:39:58', 'TYPE': 'S', 'VOL': 84.9999999900},
    {'ASSET': 'XXBTZEUR', 'DATETIME': '2024-10-14 08:30:59', 'TYPE': 'B', 'VOL': 0.0013214450},
]
# Trades only in the Kraken ledger (not in TradesHistory, so not in the CSV): they move cash and positions
MANUAL_TRADES = [
    # EUR -> USD -> GOOGLx conversion (refid TSSVWVV-TQSBX-DRQRZW): 96.68 EUR for 0.3 GOOGLx
    {'ASSET': 'GOOGLxUSD', 'DATETIME': '2026-04-30 10:34:16', 'TYPE': 'B', 'PRICE': 96.68 / 0.3, 'AMOUNT': 96.68, 'FEE': 0.0, 'VOL': 0.3},  # noqa # fmt: skip
]


# -----Func definitions--------------------------------------------------
def get_cash_positions(df_trades: pd.DataFrame) -> pd.DataFrame:
    """ZEUR cash movement of every trade: buys pay cost + fee, sells receive cost - fee."""
    df_cash_pos = df_trades.copy()
    df_cash_pos.ASSET = 'ZEUR'
    df_cash_pos.PRICE = 1.0
    df_cash_pos.loc[df_trades.TYPE == 'B', 'AMOUNT'] *= -1
    df_cash_pos['AMOUNT'] -= df_cash_pos.FEE
    df_cash_pos['SHARES'] = df_cash_pos.AMOUNT
    df_cash_pos['FEE'] = 0.0
    df_cash_pos.drop(['VOL', 'DATETIME', 'TYPE'], axis=1, inplace=True)

    return df_cash_pos


def get_fx_cash_positions(df_fx_trades: pd.DataFrame) -> pd.DataFrame:
    """ZEUR cash movement of every EURUSD trade: selling EUR pays VOL, buying EUR receives VOL.

    VOL is in EUR (the base); the fee is charged in USD (the quote), so it is taken from the USD position instead.
    """
    df_cash_pos = df_fx_trades.copy()
    df_cash_pos.ASSET = 'ZEUR'
    df_cash_pos.PRICE = 1.0
    df_cash_pos['AMOUNT'] = df_cash_pos.VOL.where(df_cash_pos.TYPE == 'B', -df_cash_pos.VOL)
    df_cash_pos['SHARES'] = df_cash_pos.AMOUNT
    df_cash_pos['FEE'] = 0.0
    df_cash_pos.drop(['VOL', 'DATETIME', 'TYPE'], axis=1, inplace=True)

    return df_cash_pos


def get_usd_positions(df_fx_trades: pd.DataFrame, df_fx_prices: pd.DataFrame, date_to: date) -> pd.DataFrame:
    """Daily USD position, valued in EUR with 1 / EURUSD (the last known rate on days without a candle).

    Selling EUR is buying USD (cost - fee received); buying EUR is selling USD (cost + fee paid).
    """
    df_usd_trades = df_fx_trades.copy()
    is_eur_sell = df_usd_trades.TYPE == 'S'
    df_usd_trades['VOL'] = (df_usd_trades.AMOUNT - df_usd_trades.FEE).where(
        is_eur_sell,
        df_usd_trades.AMOUNT + df_usd_trades.FEE,
    )
    df_usd_trades['TYPE'] = is_eur_sell.map({True: 'B', False: 'S'})
    df_usd_trades['FEE'] = 0.0  # Already taken from VOL, and the FEE column of the positions is in EUR
    df_usd_trades['ASSET'] = USD_ASSET

    return get_asset_positions(
        asset_name=USD_ASSET,
        df_trades=df_usd_trades,
        df_prices=get_eur_prices(df_fx_prices=df_fx_prices, date_to=date_to),
        date_to=date_to,
    )


def get_eur_prices(
    df_fx_prices: pd.DataFrame,
    date_to: date,
    df_usd_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Daily EUR price of df_usd_prices (of 1 USD when None): USD price / EURUSD.

    Days without a candle (in either file) keep the last known value.
    """
    dates = pd.date_range(start=df_fx_prices.DATE.min(), end=date_to, freq='d').date
    fx_rate = pd.to_numeric(df_fx_prices.set_index('DATE').PRICE).reindex(dates).ffill()
    usd_price = 1.0
    if df_usd_prices is not None:
        usd_price = pd.to_numeric(df_usd_prices.set_index('DATE').PRICE).reindex(dates).ffill()

    return pd.DataFrame({'DATE': dates, 'PRICE': (usd_price / fx_rate).values})


def get_asset_positions(
    asset_name: str,
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    date_to: date,
) -> pd.DataFrame:
    dates = pd.date_range(start=df_trades["DATE"].iloc[0], end=date_to, freq='d')
    df_pos_temp = pd.DataFrame(columns=['DATE', 'ASSET', 'SHARES', 'PRICE'])
    df_pos_temp.DATE = dates
    df_pos_temp.DATE = df_pos_temp.DATE.dt.date
    df_pos_temp.ASSET = asset_name
    df_pos_temp.SHARES = 0.0

    df_pos_temp.PRICE = df_pos_temp.DATE.map(df_prices.set_index('DATE')['PRICE'])
    df_pos_temp.PRICE = pd.to_numeric(df_pos_temp.PRICE)

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

    return df_pos_temp


def clean_flows_df(df_flow: pd.DataFrame) -> pd.DataFrame:
    # Transfers are internal moves (spot <-> staking/earn), not money in or out
    df_flow = df_flow[(df_flow.ASSET == 'ZEUR') & (df_flow.TYPE != 'transfer')]
    df_flow.drop(['ACLASS', 'TYPE', 'SUBTYPE'], axis=1, inplace=True)
    df_flow.rename({'TIME': 'DATE', 'BALANCE': 'SHARES'}, axis=1, inplace=True)
    df_flow.DATE = pd.to_datetime(df_flow.DATE).dt.date
    df_flow.AMOUNT = pd.to_numeric(df_flow.AMOUNT)
    df_flow.FEE = pd.to_numeric(df_flow.FEE)
    # AMOUNT is the flow itself; the cash balance (SHARES) also loses the fee
    df_flow['SHARES'] = df_flow.AMOUNT - df_flow.FEE
    df_flow['PRICE'] = 1.0

    return df_flow


def drop_cash_rows(df: pd.DataFrame) -> pd.DataFrame:
    i = df[df.ASSET == 'ZEUR'].index
    df.drop(i, inplace=True)

    return df


def update_get_flow_file(kapi, flow_type: str) -> pd.DataFrame:
    flow_filename = DEPOSITS_FILE if flow_type == FLOW_TYPE_DEPOSIT else WITHDRAWALS_FILE
    df_flows = pd.read_csv(flow_filename)
    latest_flow_datetime = df_flows.TIME.iloc[-1]
    flow_datetime = pd.to_datetime(latest_flow_datetime)
    if isinstance(flow_datetime, pd.Timestamp):
        flow_datetime = flow_datetime.to_pydatetime(warn=False)

    # Every flow after the last one in the file (all pages). Its TIME is naive UTC; Kraken returns it again
    # (`start` has second precision) and the TIME filter below drops it
    new_flow_pages = get_paginated_response_from_kraken(
        kapi=kapi,
        endpoint='Ledgers',
        dict_key='ledger',
        params={'type': flow_type},
        pages=None,
        is_private=True,
        timestamp_from=int(flow_datetime.replace(tzinfo=timezone.utc).timestamp()),
    )
    if not new_flow_pages:
        return df_flows

    df_new_flows = pd.DataFrame([rec for page in new_flow_pages for rec in page.values()])
    df_new_flows.columns = [x.upper() for x in df_new_flows.columns]
    df_new_flows.drop(['REFID'], axis=1, inplace=True)
    df_new_flows.TIME = pd.to_datetime(df_new_flows.TIME, unit='s')
    df_new_flows = df_new_flows[df_new_flows.TIME > latest_flow_datetime]
    if not df_new_flows.empty:
        df_flows = pd.concat([df_flows, df_new_flows])
        # Mixed column: file rows are strings (with or without fractional seconds), new ones Timestamps
        df_flows['TIME'] = pd.to_datetime(df_flows.TIME, format='ISO8601')
        df_flows.sort_values(by=['TIME'], ascending=True, inplace=True, ignore_index=True)
        df_flows.to_csv(flow_filename, index=False)

    return df_flows


def load_flows(kapi) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Update the deposits and withdrawals files from Kraken and return them cleaned (ZEUR only)."""
    df_deposits = update_get_flow_file(kapi=kapi, flow_type=FLOW_TYPE_DEPOSIT)
    df_wd = update_get_flow_file(kapi=kapi, flow_type=FLOW_TYPE_WD)
    df_deposits = clean_flows_df(df_flow=df_deposits)
    df_wd = clean_flows_df(df_flow=df_wd)

    return df_deposits, df_wd


def read_trades(filename: str) -> pd.DataFrame:
    df_trades = pd.read_csv(filename)

    df_trades.drop(columns=['ordertype'], inplace=True)
    df_trades.rename({'pair': 'Asset', 'time(UTC)': 'Datetime', 'cost': 'Amount'}, axis=1, inplace=True)
    df_trades.columns = [x.upper() for x in df_trades.columns]

    # Cash and positions are in EUR: a USD cost (xStocks) would be counted as ZEUR. FX_PAIR is handled apart
    is_kept = df_trades.ASSET.map(lambda name: is_eur_pair(pair_name=name) or name == FX_PAIR)
    if not is_kept.all():
        print(f'Trades of pairs not quoted in EUR left out: {sorted(df_trades[~is_kept].ASSET.unique())}')
    df_trades = df_trades[is_kept]
    df_trades = pd.concat([df_trades, pd.DataFrame(MANUAL_TRADES)], ignore_index=True)

    # Operation types D|W|B|S stands for Deposit, Withdrawal, Buy, Sell
    df_trades['TYPE'] = df_trades['TYPE'].replace('buy', 'B')
    df_trades['TYPE'] = df_trades['TYPE'].replace('sell', 'S')
    df_trades.DATETIME = pd.to_datetime(df_trades.DATETIME)
    df_trades['DATE'] = df_trades['DATETIME'].dt.date
    df_trades = df_trades.sort_values(by=['DATETIME'], kind='stable', ignore_index=True)

    # Watch out for this trade
    # XXLMXXBT,2018-02-01 17:33:11,buy,limit,0.00004885,0.014990794,0.000038976,306.8739771
    # replace it with
    # XXBTZEUR,2018-02-01 17:33:11,sell,limit,7163,107.3,0,0.014990794
    # XXLMZEUR,2018-02-01 17:33:11,buy,limit,0.35,107.3,0.2,306.8739771

    return df_trades


def add_conversions(df_trades: pd.DataFrame) -> pd.DataFrame:
    """Trades plus ASSET_CONVERSIONS as zero-cost trades, to be used for positions only (not for cash)."""
    df_conversions = pd.DataFrame(ASSET_CONVERSIONS)
    df_conversions.DATETIME = pd.to_datetime(df_conversions.DATETIME)
    df_conversions['DATE'] = df_conversions['DATETIME'].dt.date
    df_conversions[['PRICE', 'AMOUNT', 'FEE']] = 0.0
    df_trades = pd.concat([df_trades, df_conversions], ignore_index=True)

    return df_trades.sort_values(by=['DATETIME'], kind='stable', ignore_index=True)


# GAIN per year
def year_gain_perc(
    df_deposits: pd.DataFrame,
    df_wd: pd.DataFrame,
    df_balances_avg: pd.DataFrame,
    year: int,
    realised: float,
    current_balance: float | None = None,
    verbose: bool = VERBOSE,
) -> float:
    df_deposits.DATE = pd.to_datetime(df_deposits.DATE)
    df_wd.DATE = pd.to_datetime(df_wd.DATE)
    df_balances_avg.DATE = pd.to_datetime(df_balances_avg.DATE)
    deposit_amount_year = df_deposits[df_deposits.DATE.dt.year == year].AMOUNT.sum()
    wd_amount_year = -df_wd[df_wd.DATE.dt.year == year].AMOUNT.sum()
    daily_balances_year = df_balances_avg[df_balances_avg.DATE.dt.year == year]
    previous_year_balances = df_balances_avg[df_balances_avg.DATE.dt.year == year - 1]
    # Start from the previous year's closing balance, so January 1st gains and flows are counted once
    if previous_year_balances.empty:
        balance_0 = daily_balances_year.AMOUNT.iloc[0]
    else:
        balance_0 = previous_year_balances.AMOUNT.iloc[-1]
    balance_365 = daily_balances_year.AMOUNT.iloc[-1]
    flows = wd_amount_year - deposit_amount_year
    mean_balance = daily_balances_year.AMOUNT.mean()
    gain_numerator = balance_365 - balance_0 + flows
    gain = 100 * gain_numerator / mean_balance
    if verbose:
        table_data = [
            {
                "current_balance": smart_round(number=current_balance) if current_balance is not None else None,
                "balance_0": smart_round(number=balance_0),
                "balance_365": smart_round(number=balance_365),
                "mean_balance": smart_round(number=mean_balance),
                "flows": smart_round(number=flows),
                "gain": smart_round(number=gain),
                "realised_perc": smart_round(number=(100 * realised / mean_balance)) if mean_balance != 0 else 0,
            },
        ]

        # 2. Define the columns mapping (key, display_label)
        table_columns = [
            ("balance_0", "START BALANCE"),
            ("balance_365", "END BALANCE"),
            ("mean_balance", "MEAN BALANCE"),
            ("flows", "FLOWS"),
            ("gain", "GAIN (%)"),
            ("realised_perc", "REALISED GAIN (%)"),
        ]
        if current_balance is not None:
            # Today's positions at the latest price (today's candle, not closed yet)
            table_columns.insert(2, ("current_balance", "CURRENT BALANCE"))
        print_table(
            data=table_data,
            columns=table_columns,
            title=f"YEAR: {year}",
        )
    return gain


def build_positions(
    kapi,
    df_trades: pd.DataFrame,
    df_deposits: pd.DataFrame,
    df_wd: pd.DataFrame,
    date_to: date,
) -> pd.DataFrame:
    """Daily position (shares, price, amount) of every traded asset, plus the ZEUR cash movements."""
    df_fx_trades = df_trades[df_trades.ASSET == FX_PAIR]
    df_trades = df_trades[df_trades.ASSET != FX_PAIR]
    asset_names = df_trades[~df_trades.ASSET.isin(CASH_ONLY_ASSETS)].ASSET.dropna().unique()

    # Update prices of USDCEUR
    update_asset_prices(asset_name='USDCEUR', kapi=kapi, date_to=date_to)
    df_list = [df_deposits, df_wd, get_cash_positions(df_trades=df_trades)]
    df_fx_prices = None
    if not df_fx_trades.empty or any(is_xstock_pair(pair_name=name) for name in asset_names):
        df_fx_prices = update_asset_prices(asset_name=FX_PAIR, kapi=kapi, date_to=date_to)
    if not df_fx_trades.empty:
        df_list.append(get_fx_cash_positions(df_fx_trades=df_fx_trades))
        df_list.append(get_usd_positions(df_fx_trades=df_fx_trades, df_fx_prices=df_fx_prices, date_to=date_to))
    df_trades = add_conversions(df_trades=df_trades)

    for asset_name in asset_names:
        if is_xstock_pair(pair_name=asset_name):
            # Priced in USD (get_fix_pair_name would give e.g. GOOGLxUSDEUR): valued at USD price / EURUSD
            fix_asset_name = asset_name
            df_prices = get_eur_prices(
                df_fx_prices=df_fx_prices,
                date_to=date_to,
                df_usd_prices=update_asset_prices(asset_name=asset_name, kapi=kapi, date_to=date_to),
            )
        else:
            fix_asset_name = get_fix_pair_name(pair_name=asset_name, fix_x_pair_names=FIX_X_PAIR_NAMES)
            df_prices = update_asset_prices(asset_name=fix_asset_name, kapi=kapi, date_to=date_to)

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

    return df_positions


def add_daily_cash(df_positions: pd.DataFrame, date_to: date) -> pd.DataFrame:
    """Replace the ZEUR cash movements with one accumulated cash balance row per day."""
    df_positions.loc[df_positions.ASSET == 'ZEUR', 'SHARES'] = df_positions.loc[df_positions.ASSET == 'ZEUR', 'SHARES'].cumsum()  # noqa # fmt: skip
    df_no_duplicates = df_positions.loc[df_positions.ASSET == 'ZEUR', :].drop_duplicates(subset=['DATE'], keep='last')

    df_positions = drop_cash_rows(df=df_positions)
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

    df_positions = drop_cash_rows(df=df_positions)
    df_positions = pd.concat([df_positions, df_cash_daily])

    df_positions.sort_values(by=['DATE'], inplace=True)

    return df_positions


def print_summary(df_trades: pd.DataFrame, df_deposits: pd.DataFrame, df_wd: pd.DataFrame):
    # EURUSD is a currency exchange, not an investment: out of BUYS/SELLS, its EUR (VOL) only moves the cash
    df_fx_trades = df_trades[df_trades.ASSET == FX_PAIR]
    df_trades = df_trades[df_trades.ASSET != FX_PAIR]
    fx_eur_sold = df_fx_trades[df_fx_trades.TYPE == 'S'].VOL.sum()
    fx_eur_bought = df_fx_trades[df_fx_trades.TYPE == 'B'].VOL.sum()
    if not df_fx_trades.empty:
        print(
            f'\n FX EUR->USD: {my_round(value=fx_eur_sold)} EUR sold, {my_round(value=fx_eur_bought)} EUR bought '
            f'(fees {my_round(value=df_fx_trades.FEE.sum())} USD)',
        )

    total_buy_amount = df_trades[df_trades.TYPE == 'B'].AMOUNT.sum()
    total_sell_amount = df_trades[df_trades.TYPE == 'S'].AMOUNT.sum()
    total_fees = df_trades.FEE.sum()

    print('\n BUYS: {}'.format(my_round(value=total_buy_amount)))
    print('\n SELLS: {}'.format(my_round(value=total_sell_amount)))
    print('\n SELLS - BUYS: {}'.format(my_round(value=total_sell_amount - total_buy_amount)))
    print('\n FEES: {}'.format(my_round(value=total_fees)))

    total_deposit = df_deposits.AMOUNT.sum()
    total_wd = -df_wd.AMOUNT.sum()
    print('\n DEPOSIT: {}'.format(my_round(value=total_deposit)))
    print('\n WD: {}'.format(my_round(value=total_wd)))
    print('\n DEPOSIT - WD: {}'.format(my_round(value=total_deposit - total_wd)))

    flow_fees = df_deposits.FEE.sum() + df_wd.FEE.sum()
    cash = (
        total_deposit - total_wd - total_buy_amount + total_sell_amount - total_fees - flow_fees
        - fx_eur_sold + fx_eur_bought
    )  # fmt: skip
    print('\n CASH: {}'.format(my_round(value=cash)))


def print_gains_by_year(
    df_deposits: pd.DataFrame,
    df_wd: pd.DataFrame,
    df_positions: pd.DataFrame,
    current_balance: float,
    current_year: int,
):
    # AVG BALANCE
    df_avg_balances_per_day = df_positions.groupby('DATE').AMOUNT.sum().reset_index()

    print('\n ***** GAINS BY YEAR ***** ')
    # G/L sell amount per year, updated by summary_trades.py
    for year, realised in read_realised_gains(filename=REALISED_GAINS_FILE).items():
        year_gain_perc(
            df_deposits=df_deposits,
            df_wd=df_wd,
            df_balances_avg=df_avg_balances_per_day,
            year=year,
            realised=realised,
            current_balance=current_balance if year == current_year else None,
        )


def main():
    # configure api
    kapi = krakenex.API()
    kapi.load_key(KEY_FILE)

    today = datetime.today().date()
    date_to = today - timedelta(days=1)

    df_deposits, df_wd = load_flows(kapi=kapi)
    df_trades = read_trades(filename=TRADES_FILE)
    df_positions = build_positions(
        kapi=kapi,
        df_trades=df_trades,
        df_deposits=df_deposits,
        df_wd=df_wd,
        date_to=today,
    )
    df_positions = add_daily_cash(df_positions=df_positions, date_to=today)
    # Today only for the current balance: the gains use the closed days (until date_to)
    current_balance = df_positions[df_positions.DATE == today].AMOUNT.sum()
    df_positions = df_positions[df_positions.DATE <= date_to]

    print_summary(df_trades=df_trades, df_deposits=df_deposits, df_wd=df_wd)
    print_gains_by_year(
        df_deposits=df_deposits,
        df_wd=df_wd,
        df_positions=df_positions,
        current_balance=current_balance,
        current_year=today.year,
    )


if __name__ == '__main__':
    main()
