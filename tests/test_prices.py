from datetime import date, datetime, timedelta, timezone

from fake_kraken import FakeKraken, make_daily_candles
from utils.basic import OHLC_MAX_CANDLES, from_date_to_timestamp, get_new_prices

FIRST_DAY = date(2023, 1, 1)
DAYS = 1000  # more than the OHLC_MAX_CANDLES Kraken returns
LAST_DAY = FIRST_DAY + timedelta(days=DAYS - 1)


def kraken_with_candles(pair: str = 'ADAEUR', first_day: date = FIRST_DAY, days: int = DAYS) -> FakeKraken:
    kraken = FakeKraken()
    first_day_time = int(datetime(first_day.year, first_day.month, first_day.day, tzinfo=timezone.utc).timestamp())
    kraken.ohlc[pair] = make_daily_candles(first_day_time=first_day_time, days=days)
    return kraken


def test_recent_since_reads_the_new_candles_without_warning(capsys):
    since_day = LAST_DAY - timedelta(days=3)

    df_prices = get_new_prices(
        kapi=kraken_with_candles(),
        asset_name='ADAEUR',
        timestamp_from=from_date_to_timestamp(day=since_day),  # as update_asset_prices calls it
        with_volumes=True,
    )

    assert 'OHLC GAP' not in capsys.readouterr().out
    assert list(df_prices.columns) == ['TIMESTAMP', 'C', 'VOL']
    first_day = datetime.fromtimestamp(df_prices.TIMESTAMP.iloc[0], tz=timezone.utc).date()
    assert first_day in (since_day, since_day + timedelta(days=1))  # `since` inclusivity is not documented


def test_since_older_than_the_candles_kraken_keeps_warns_with_the_missing_dates(capsys):
    since_day = LAST_DAY - timedelta(days=800)

    df_prices = get_new_prices(
        kapi=kraken_with_candles(),
        asset_name='ADAEUR',
        timestamp_from=from_date_to_timestamp(day=since_day),
    )

    assert len(df_prices) == OHLC_MAX_CANDLES
    out = capsys.readouterr().out
    first_kept = LAST_DAY - timedelta(days=OHLC_MAX_CANDLES - 1)
    assert 'OHLC GAP for ADAEUR' in out
    assert f'Kraken starts at {first_kept}' in out


def test_pair_newer_than_since_warns(capsys):
    kraken = kraken_with_candles(first_day=LAST_DAY - timedelta(days=9), days=10)

    since = from_date_to_timestamp(day=LAST_DAY - timedelta(days=30))

    get_new_prices(kapi=kraken, asset_name='ADAEUR', timestamp_from=since)

    assert 'OHLC GAP for ADAEUR' in capsys.readouterr().out
