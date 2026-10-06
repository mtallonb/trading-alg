from datetime import date, datetime, timedelta, timezone

from fake_kraken import FakeKraken, make_daily_candles
from utils import basic
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


def test_new_pair_prices_are_downloaded_and_eurusd_volume_is_already_eur(tmp_path, monkeypatch):
    # No prices file nor OHLCV file: it raised FileNotFoundError
    monkeypatch.setattr(basic, 'PRICES_DIR', f'{tmp_path}/')
    monkeypatch.setattr(basic, 'OHLCV_DIR', f'{tmp_path}/')
    df_prices, _ = basic.read_prices_from_local_file(asset_name='EURUSD')
    assert df_prices.empty

    # Kraken answers the EURUSD OHLC under its internal name
    date_to = LAST_DAY
    kraken = kraken_with_candles(pair='ZEURZUSD', first_day=date_to - timedelta(days=9), days=10)
    basic.update_asset_prices(asset_name='EURUSD', kapi=kraken, date_to=date_to)

    df_prices, df_volumes = basic.read_prices_from_local_file(asset_name='EURUSD')
    assert len(df_prices) == 10 and df_prices.DATE.iloc[-1] == date_to
    # VOL is in EUR (the base): not multiplied by the USD price
    assert list(df_volumes.VOL_EUR) == list(df_prices.VOL)


def test_xstock_ohlc_asks_the_tokenized_asset_class():
    # Without asset_class Kraken answers 'EQuery:Invalid asset pair' for GOOGLxUSD
    kraken = kraken_with_candles(pair='GOOGLxUSD', first_day=LAST_DAY - timedelta(days=9), days=10)
    since = from_date_to_timestamp(day=LAST_DAY - timedelta(days=30))

    get_new_prices(kapi=kraken, asset_name='GOOGLxUSD', timestamp_from=since)
    get_new_prices(kapi=kraken, asset_name='ADAEUR', timestamp_from=since)

    asset_classes = [params.get('asset_class') for params in kraken.calls_to('OHLC')]
    assert asset_classes == ['tokenized_asset', None]
