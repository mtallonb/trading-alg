from datetime import datetime, timezone

import pytest

import orders
import summary_trades
import utils.basic as basic

from fake_kraken import FakeKraken, make_trade
from utils.basic import DATETIME_FORMAT, LOCAL_TZ, KrakenError
from utils.classes import Asset, CSVTrade, Trade

NEWEST = 1_788_000_000.0


def kraken_with_trades(count: int) -> FakeKraken:
    """count BTC trades one hour apart, newest at NEWEST (+0.25 s like Kraken's fractional times)."""
    kraken = FakeKraken()
    kraken.add_trades([make_trade(time=NEWEST - i * 3600 + 0.25, price=70000 + i) for i in range(count)])
    return kraken


def local_datetime(unix_time: float) -> datetime:
    # Same conversion as fill_trades and load_from_csv
    return LOCAL_TZ.localize(datetime.fromtimestamp(unix_time).replace(microsecond=0))


def btc_asset() -> Asset:
    return Asset(name='XBTEUR', original_name='XXBTZEUR')


def test_fill_trades_with_csv_reads_every_trade_after_it():
    kraken = kraken_with_trades(count=300)
    asset = btc_asset()
    # The CSV ends at Kraken trade #150: 150 newer trades, more than the old 100-trade limit
    csv_trade = Trade(
        trade_type='sell',
        shares=1,
        price=1,
        amount=1,
        execution_datetime=local_datetime(unix_time=NEWEST - 150 * 3600),
    )
    asset.trades.append(csv_trade)

    orders.fill_trades(kapi=kraken, assets_dict={'XBTEUR': asset}, last_trade_from_csv=csv_trade)

    assert kraken.calls_to('TradesHistory')[0]['start'] == int(csv_trade.execution_datetime.timestamp())
    assert len(asset.trades) == 151  # 150 new + the CSV one (its Kraken copy is not added again)
    assert asset.trades[-1] is csv_trade
    times = [trade.execution_datetime for trade in asset.trades]
    assert times == sorted(times, reverse=True)  # newest first


def test_fill_trades_without_csv_reads_trade_pages():
    kraken = kraken_with_trades(count=300)
    asset = btc_asset()

    orders.fill_trades(kapi=kraken, assets_dict={'XBTEUR': asset}, last_trade_from_csv=None)

    assert len(kraken.calls_to('TradesHistory')) == orders.TRADE_PAGES
    assert 'start' not in kraken.calls_to('TradesHistory')[0]
    assert len(asset.trades) == orders.TRADE_PAGES * 50
    assert asset.trades[0].execution_datetime == local_datetime(unix_time=NEWEST)


def mixed_kraken_trades(count: int) -> FakeKraken:
    """count trades one hour apart (newest at NEWEST + 0.25 s), alternating buy/sell and 3 pairs."""
    kraken = FakeKraken()
    pairs = ['XXBTZEUR', 'ADAEUR', 'SOLEUR']
    kraken.add_trades(
        [
            make_trade(time=NEWEST - i * 3600 + 0.25, pair=pairs[i % 3], type='buy' if i % 2 else 'sell', price=10 + i)
            for i in range(count)
        ],
    )
    return kraken


def csv_trade_at(unix_time: float, type: str) -> CSVTrade:
    completed = datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime(DATETIME_FORMAT)  # CSV is UTC
    return CSVTrade(asset_name='XXBTZEUR', completed=completed, type=type, price='1', cost='1', fee='0', vol='1')


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(basic.time, 'sleep', lambda seconds: None)


def test_fetch_new_trades_reads_every_page_after_the_csv():
    kraken = mixed_kraken_trades(count=300)
    # The CSV already has Kraken trades #299..#150 (oldest first); #150 is the last one
    csv_trades = [
        csv_trade_at(unix_time=NEWEST - i * 3600, type='buy' if i % 2 else 'sell') for i in range(299, 149, -1)
    ]
    buy_trades = [trade for trade in csv_trades if trade.type == 'buy']
    sell_trades = [trade for trade in csv_trades if trade.type == 'sell']

    new_trades = summary_trades.fetch_new_trades(
        kapi=kraken,
        latest_trade_csv=csv_trades[-1],
        buy_trades=buy_trades,
        sell_trades=sell_trades,
    )

    # 151 records after `start` (#150 comes back: second precision) in 4 pages, the last one with 1 record
    calls = kraken.calls_to('TradesHistory')
    assert [params['ofs'] for params in calls] == [0, 50, 100, 150]
    assert all(params['start'] == int(NEWEST - 150 * 3600) for params in calls)
    assert len(new_trades) == 150  # #150 skipped: already in the CSV
    new_times = [trade.completed for trade in new_trades]
    assert new_times == sorted(new_times) and len(set(new_times)) == 150  # oldest first, no duplicates
    assert new_trades[-1].completed == datetime.fromtimestamp(NEWEST, tz=timezone.utc).replace(tzinfo=None)
    assert {trade.asset_name for trade in new_trades} == {'XXBTZEUR', 'ADAEUR', 'SOLEUR'}
    # CSV + new trades, still oldest first, split by type
    assert len(buy_trades) == len(sell_trades) == 150
    for trades in (buy_trades, sell_trades):
        assert [trade.completed for trade in trades] == sorted(trade.completed for trade in trades)


def test_fetch_new_trades_with_empty_csv_reads_whole_history():
    kraken = mixed_kraken_trades(count=230)

    new_trades = summary_trades.fetch_new_trades(kapi=kraken, latest_trade_csv=None, buy_trades=[], sell_trades=[])

    assert [params['ofs'] for params in kraken.calls_to('TradesHistory')] == [0, 50, 100, 150, 200]
    assert 'start' not in kraken.calls_to('TradesHistory')[0]
    assert len(new_trades) == 230


def test_fetch_new_trades_retries_rate_limit_in_a_middle_page(no_sleep):
    kraken = mixed_kraken_trades(count=300)
    kraken.queue_error(endpoint='TradesHistory', after=2, times=2)

    new_trades = summary_trades.fetch_new_trades(
        kapi=kraken,
        latest_trade_csv=csv_trade_at(unix_time=NEWEST - 150 * 3600, type='buy'),
        buy_trades=[],
        sell_trades=[],
    )

    assert len(new_trades) == 150


def test_fetch_new_trades_error_in_a_middle_page_adds_nothing():
    kraken = mixed_kraken_trades(count=300)
    kraken.queue_error(endpoint='TradesHistory', error='EGeneral:Internal error', after=2)
    buy_trades, sell_trades = [], []

    with pytest.raises(KrakenError):
        summary_trades.fetch_new_trades(
            kapi=kraken,
            latest_trade_csv=csv_trade_at(unix_time=NEWEST - 150 * 3600, type='buy'),
            buy_trades=buy_trades,
            sell_trades=sell_trades,
        )

    # Nothing to append to the CSV: the 100 newest trades alone would leave a gap
    assert buy_trades == [] and sell_trades == []
