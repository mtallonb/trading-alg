import pandas as pd

import ledger

from fake_kraken import FakeKraken

FIRST_TIME = 1_700_000_000.1234  # Kraken ledger times are unix floats with 4 decimals


def ledger_record(time: float, type: str = 'staking', asset: str = 'ADA.S') -> dict:
    """Ledgers record as Kraken returns it."""
    return {
        'refid': f'R{time}',
        'time': time,
        'type': type,
        'subtype': '',
        'aclass': 'currency',
        'asset': asset,
        'amount': '0.1234',
        'fee': '0.0000',
        'balance': '10.0000',
    }


def kraken_with_ledger(records: int) -> FakeKraken:
    kraken = FakeKraken()
    kraken.add_ledger([ledger_record(time=FIRST_TIME + 3600 * i) for i in range(records)])
    return kraken


def test_each_run_goes_further_back_until_the_ledger_is_complete(tmp_path, capsys):
    ledger_file = str(tmp_path / 'ledger.csv')
    kraken = kraken_with_ledger(records=120)

    counts = []
    for _ in range(4):
        df_ledger = ledger.update_ledger_file(kapi=kraken, filename=ledger_file, pages_per_run=1)
        counts.append(len(df_ledger))

    # Newest page first, then one older page per run (49 new: `end` is inclusive, the oldest one comes back)
    assert counts == [50, 99, 120, 120]
    saved = pd.read_csv(ledger_file)
    assert len(saved) == 120 and saved.LEDGER_ID.is_unique
    assert saved.TIME.is_monotonic_increasing
    assert saved.TIME.iloc[0] == FIRST_TIME and saved.TIME.iloc[-1] == FIRST_TIME + 3600 * 119
    assert 'Complete: Kraken has no records older than the file' in capsys.readouterr().out.split('Ledger:')[-1]


def test_new_records_are_all_read_before_going_back(tmp_path):
    ledger_file = str(tmp_path / 'ledger.csv')
    kraken = kraken_with_ledger(records=120)
    ledger.update_ledger_file(kapi=kraken, filename=ledger_file, pages_per_run=1)  # the newest 50

    # 75 new records (more than a page) after the first run
    kraken.add_ledger([ledger_record(time=FIRST_TIME + 3600 * (120 + i), type='earn') for i in range(75)])
    df_ledger = ledger.update_ledger_file(kapi=kraken, filename=ledger_file, pages_per_run=1)

    assert len(df_ledger) == 50 + 75 + 49  # every new one plus one older page (minus the oldest, read again)
    times = df_ledger.TIME
    # Contiguous: every record between the oldest and the newest in the file
    expected = [FIRST_TIME + 3600 * i for i in range(21, 195)]
    assert list(times) == expected
    assert (df_ledger.TYPE == 'earn').sum() == 75
