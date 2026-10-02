from datetime import date

import pandas as pd

import balances

from fake_kraken import FakeKraken

LAST_FLOW_TIME = 1_780_000_000.1234  # Kraken ledger times are unix floats with 4 decimals


def ledger_record(time: float, type: str = 'deposit', amount: float = 100.0) -> dict:
    """Ledgers record as Kraken returns it."""
    return {
        'refid': f'R{time}',
        'time': time,
        'type': type,
        'subtype': '',
        'aclass': 'currency',
        'asset': 'ZEUR',
        'amount': f'{amount:.4f}',
        'fee': '0.0000',
        'balance': '1000.0000',
    }


def write_deposits_csv(path, times: list[float]):
    # Same format update_get_flow_file saves: Kraken columns upper-cased, TIME as naive UTC with nanoseconds
    df = pd.DataFrame([ledger_record(time=time) for time in times]).drop(columns=['refid'])
    df.columns = [column.upper() for column in df.columns]
    df.TIME = pd.to_datetime(df.TIME, unit='s')
    df.to_csv(path, index=False)


def test_update_flow_file_reads_every_page_after_the_last_flow(tmp_path, monkeypatch):
    deposits_file = tmp_path / 'deposits.csv'
    write_deposits_csv(path=deposits_file, times=[LAST_FLOW_TIME - 86400, LAST_FLOW_TIME])
    monkeypatch.setattr(balances, 'DEPOSITS_FILE', str(deposits_file))

    kraken = FakeKraken()
    kraken.add_ledger(
        [ledger_record(time=LAST_FLOW_TIME - 86400 * i) for i in range(1, 6)]  # older, already in the file
        + [ledger_record(time=LAST_FLOW_TIME)]  # the last one in the file
        # 250 new: more than the old 4-page (200 records) limit
        + [ledger_record(time=LAST_FLOW_TIME + 3600 * i) for i in range(1, 251)]
        + [ledger_record(time=LAST_FLOW_TIME + 60 * i, type='withdrawal', amount=-50) for i in range(1, 4)],
    )

    df_flows = balances.update_get_flow_file(kapi=kraken, flow_type=balances.FLOW_TYPE_DEPOSIT)

    calls = kraken.calls_to('Ledgers')
    assert all(params['start'] == int(LAST_FLOW_TIME) and params['type'] == 'deposit' for params in calls)
    assert [params['ofs'] for params in calls] == [0, 50, 100, 150, 200, 250]  # 251: the last flow comes back
    saved = pd.read_csv(deposits_file)
    for df in (df_flows, saved):
        assert len(df) == 2 + 250  # the repeated last flow is not added again
        assert pd.to_datetime(df.TIME, format='ISO8601').is_monotonic_increasing
        assert pd.to_datetime(df.TIME, format='ISO8601').is_unique
        assert set(df.TYPE) == {'deposit'}


# ADA buy, the agreed EUR -> USD example (1,000 EUR at 1.12452, 2.25 USD fee) and an xStock in USD
FX_TRADES_CSV = """pair,time(UTC),type,ordertype,price,cost,fee,vol
ADAEUR,2026-09-01 10:00:00,buy,limit,0.5,500,1,1000
EURUSD,2026-09-02 10:00:00,sell,limit,1.12452,1124.52,2.25,1000
AAPLxUSD,2026-09-03 10:00:00,buy,limit,200,200,0.5,1
"""


def read_fx_trades(tmp_path) -> pd.DataFrame:
    trades_file = tmp_path / 'trades.csv'
    trades_file.write_text(FX_TRADES_CSV)
    return balances.read_trades(filename=str(trades_file))


def test_read_trades_keeps_eurusd_and_drops_other_non_eur_pairs(tmp_path, capsys):
    df_trades = read_fx_trades(tmp_path=tmp_path)

    assert list(df_trades.ASSET) == ['ADAEUR', 'EURUSD']
    assert "['AAPLxUSD']" in capsys.readouterr().out


def test_eur_to_usd_exchange_is_neither_gain_nor_loss(tmp_path):
    df_trades = read_fx_trades(tmp_path=tmp_path)
    df_fx_trades = df_trades[df_trades.ASSET == balances.FX_PAIR]
    # No candle on 2026-09-04: the last known rate is used
    df_fx_prices = pd.DataFrame({'DATE': [date(2026, 9, 2), date(2026, 9, 3)], 'PRICE': ['1.12452', '1.13']})

    df_cash = balances.get_fx_cash_positions(df_fx_trades=df_fx_trades)
    df_usd = balances.get_usd_positions(df_fx_trades=df_fx_trades, df_fx_prices=df_fx_prices, date_to=date(2026, 9, 4))

    assert list(df_cash.SHARES) == [-1000.0]  # the EUR sold, the fee is in USD
    assert list(df_usd.DATE) == [date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4)]
    assert set(df_usd.ASSET) == {balances.USD_ASSET}
    assert [round(shares, 2) for shares in df_usd.SHARES] == [1122.27] * 3  # cost - fee
    # Exchange day: only the fee is lost (1,000 EUR -> 998.00 EUR); then the EUR/USD move
    assert [round(amount, 2) for amount in df_usd.AMOUNT] == [998.0, 993.16, 993.16]


def test_summary_cash_moves_the_exchanged_eur_out_of_buys_and_sells(tmp_path, capsys):
    df_trades = read_fx_trades(tmp_path=tmp_path)
    df_deposits = pd.DataFrame({'AMOUNT': [5000.0], 'FEE': [0.0]})
    df_wd = pd.DataFrame({'AMOUNT': [], 'FEE': []})

    balances.print_summary(df_trades=df_trades, df_deposits=df_deposits, df_wd=df_wd)

    out = capsys.readouterr().out
    assert 'BUYS: 500' in out and 'SELLS: 0' in out and 'FEES: 1' in out  # EURUSD is not an investment
    assert 'FX EUR->USD: 1000' in out
    # 5000 - 500 ADA - 1 fee - 1000 exchanged, the same the daily cash adds up from the movements
    assert 'CASH: 3499' in out
    df_fx_trades = df_trades[df_trades.ASSET == balances.FX_PAIR]
    cash_movements = pd.concat(
        [
            balances.get_cash_positions(df_trades=df_trades[df_trades.ASSET != balances.FX_PAIR]),
            balances.get_fx_cash_positions(df_fx_trades=df_fx_trades),
        ],
    )
    assert df_deposits.AMOUNT.sum() + cash_movements.SHARES.sum() == 3499


def test_current_balance_column_only_when_given(capsys):
    df_balances = pd.DataFrame({'DATE': [date(2025, 12, 31), date(2026, 1, 1), date(2026, 10, 1)], 'AMOUNT': [100.0, 100.0, 110.0]})  # noqa # fmt: skip
    flows = pd.DataFrame({'DATE': [], 'AMOUNT': []})

    for current_balance in (None, 123.0):
        balances.year_gain_perc(
            df_deposits=flows.copy(),
            df_wd=flows.copy(),
            df_balances_avg=df_balances.copy(),
            year=2026,
            realised=0,
            current_balance=current_balance,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert ('CURRENT BALANCE' in out) == (current_balance is not None)
        assert current_balance is None or '123' in out
