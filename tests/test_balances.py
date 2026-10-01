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
