import pytest

import utils.basic as basic

from fake_kraken import RATE_LIMIT_ERROR, FakeKraken, make_trade
from utils.basic import KrakenError, get_paginated_response_from_kraken

NEWEST = 1_788_000_000.0


@pytest.fixture
def kraken():
    fake = FakeKraken()
    fake.add_trades([make_trade(time=NEWEST - i * 3600 + 0.25, price=70000 + i) for i in range(230)])
    return fake


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(basic.time, 'sleep', sleeps.append)
    return sleeps


def read(kraken, **kwargs) -> list[dict]:
    pages = get_paginated_response_from_kraken(kapi=kraken, endpoint='TradesHistory', dict_key='trades', **kwargs)
    return [record for page in pages for record in page.values()]


def test_pages_limit_reads_latest_pages(kraken):
    records = read(kraken, params={}, pages=2)
    assert len(records) == 100
    assert [params['ofs'] for params in kraken.calls_to('TradesHistory')] == [0, 50]
    assert records[0]['time'] == NEWEST + 0.25  # most recent first


def test_pages_none_reads_everything_and_stops_on_count(kraken):
    records = read(kraken, params={}, pages=None)
    assert len(records) == 230
    assert len({record['time'] for record in records}) == 230
    assert len(kraken.calls_to('TradesHistory')) == 5  # stops on count, no extra empty page


def test_offset_follows_records_received_with_any_page_size(kraken):
    records = read(kraken, params={'limit': 100}, pages=None)
    assert len({record['time'] for record in records}) == 230
    assert [params['ofs'] for params in kraken.calls_to('TradesHistory')] == [0, 100, 200]


def test_timestamp_from_is_sent_as_start(kraken):
    records = read(kraken, params={}, pages=None, timestamp_from=int(NEWEST - 120 * 3600))
    assert all(params['start'] == int(NEWEST - 120 * 3600) for params in kraken.calls_to('TradesHistory'))
    # start is exclusive and has second precision: trade #120 (.25 s after it) comes back
    assert len(records) == 121


def test_rate_limit_is_retried(kraken, no_sleep):
    kraken.queue_error(endpoint='TradesHistory', error=RATE_LIMIT_ERROR, times=2)
    assert len(read(kraken, params={}, pages=None)) == 230
    assert no_sleep == [basic.KRAKEN_RATE_LIMIT_WAIT] * 2


def test_rate_limit_in_a_middle_page_is_retried(kraken, no_sleep):
    kraken.queue_error(endpoint='TradesHistory', error=RATE_LIMIT_ERROR, after=2)
    records = read(kraken, params={}, pages=None)
    assert len({record['time'] for record in records}) == 230
    assert [params['ofs'] for params in kraken.calls_to('TradesHistory')] == [0, 50, 100, 100, 150, 200]


def test_error_in_a_middle_page_raises_instead_of_returning_newest_pages(kraken):
    kraken.queue_error(endpoint='TradesHistory', error='EGeneral:Internal error', after=2)
    with pytest.raises(KrakenError, match='after reading 100 records'):
        read(kraken, params={}, pages=None)


def test_rate_limit_retries_are_bounded(kraken):
    kraken.queue_error(endpoint='TradesHistory', times=basic.KRAKEN_RATE_LIMIT_RETRIES + 1)
    with pytest.raises(KrakenError, match='Rate limit'):
        read(kraken, params={}, pages=2)
    assert len(kraken.calls_to('TradesHistory')) == basic.KRAKEN_RATE_LIMIT_RETRIES + 1
