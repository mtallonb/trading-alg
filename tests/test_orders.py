from datetime import datetime

import pytest

from fake_kraken import FakeKraken, make_open_order

import orders

from utils.basic import LOCAL_TZ, KrakenError
from utils.classes import Asset, Trade


@pytest.fixture
def no_local_prices(monkeypatch):
    # read_prices_from_local_file reads ./data and can write a prices file when it is missing
    monkeypatch.setattr(orders, 'LOAD_ALL_CLOSE_PRICES', False)


def asset_with_trade(name: str) -> Asset:
    asset = Asset(name=name, original_name=name)
    asset.shares = 100.0
    asset.add_trade(
        trade=Trade(
            trade_type='sell',
            shares=10,
            price=1.0,
            amount=10.0,
            execution_datetime=LOCAL_TZ.localize(datetime(2026, 9, 1)),
        ),
    )
    asset.latest_trade_date = asset.trades[0].execution_datetime.date()
    return asset


def asset_with_outdated_orders(kraken: FakeKraken) -> Asset:
    """Asset whose last trade (2026-09-10) is newer than its open buy and sell orders (2026-09-01/02)."""
    kraken.add_open_orders(
        [
            make_open_order(opentm=datetime(2026, 9, 1).timestamp(), pair='ADAEUR', type='buy', price=0.4, vol=100),
            make_open_order(opentm=datetime(2026, 9, 2).timestamp(), pair='ADAEUR', type='sell', price=0.6, vol=100),
        ],
    )
    asset = Asset(name='ADAEUR', original_name='ADAEUR')
    asset.price = 0.5
    asset.shares = 1000.0
    # add_trade appends older trades: newest (sell) first
    asset.add_trade(
        trade=Trade(
            trade_type='sell',
            shares=100,
            price=0.5,
            amount=50.0,
            execution_datetime=LOCAL_TZ.localize(datetime(2026, 9, 10)),
        ),
    )
    asset.add_trade(
        trade=Trade(
            trade_type='buy',
            shares=100,
            price=0.45,
            amount=45.0,
            execution_datetime=LOCAL_TZ.localize(datetime(2026, 9, 5)),
        ),
    )
    asset.latest_trade_date = asset.trades[0].execution_datetime.date()
    orders.fill_orders(open_orders=kraken.query_private('OpenOrders'), assets_dict={'ADAEUR': asset})
    asset.compute_last_buy_sell_avg()
    return asset


def test_cancelled_orders_are_removed_so_new_ones_are_suggested(monkeypatch, capsys):
    monkeypatch.setattr('builtins.input', lambda prompt='': '')
    monkeypatch.setattr(orders, 'AUTO_CANCEL_BUY_ORDER', True)
    monkeypatch.setattr(orders, 'AUTO_CANCEL_SELL_ORDER', True)
    monkeypatch.setattr(orders, 'ASSETS_TO_EXCLUDE_AMOUNT', [])
    kraken = FakeKraken()
    asset = asset_with_outdated_orders(kraken=kraken)
    assert asset.orders_buy_count == asset.orders_sell_count == 1

    count_missing_buys, _, count_all_remaining_buys = orders.print_orders_to_create(
        kapi=kraken,
        sorted_pair_names_list_balance=[('ADAEUR', asset)],
    )

    assert len(kraken.cancelled) == 2 and not kraken.open_orders
    assert asset.orders == [] and asset.orders_buy_amount == asset.orders_sell_amount == 0
    assert asset.orders_buy_count == asset.orders_sell_count == 0
    assert count_missing_buys == 1  # the cancelled buy is counted as missing
    assert count_all_remaining_buys == orders.BUY_LIMIT  # last trade is a sell: no consecutive buys
    out = capsys.readouterr().out
    assert 'ACCUMULATED' in out or 'MARKET STATUS' in out  # buy/sell suggestions printed


def test_orders_kraken_does_not_cancel_are_kept(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda prompt='': '')
    monkeypatch.setattr(orders, 'AUTO_CANCEL_BUY_ORDER', True)
    monkeypatch.setattr(orders, 'AUTO_CANCEL_SELL_ORDER', True)
    kraken = FakeKraken()
    asset = asset_with_outdated_orders(kraken=kraken)
    kraken.queue_error(endpoint='CancelOrder', error='EOrder:Unknown order', times=2)

    orders.print_orders_to_create(kapi=kraken, sorted_pair_names_list_balance=[('ADAEUR', asset)])

    assert asset.orders_buy_count == asset.orders_sell_count == 1
    assert len(asset.orders) == 2


def test_asset_without_ticker_is_reported_and_not_ranked(no_local_prices, capsys):
    kraken = FakeKraken()
    kraken.set_tickers(prices={'XXBTZEUR': 70000.0})  # no ticker for ADAEUR
    assets_dict = {'XBTEUR': asset_with_trade(name='XBTEUR'), 'ADAEUR': asset_with_trade(name='ADAEUR')}

    orders.fill_prices_and_volumes(kapi=kraken, assets_dict=assets_dict)
    rows = orders.build_ranking_rows(assets_dict=assets_dict)

    assert "No Kraken ticker price (price 0, not ranked): ['ADAEUR']" in capsys.readouterr().out
    curr_prices = {row['NAME']: row['CURR_PRICE'] for row in rows}
    assert curr_prices == {'XBTEUR': 70000.0, 'ADAEUR': None}  # None: compute_ranking drops and prints it


def test_unknown_pair_in_ticker_keeps_the_other_prices(no_local_prices, capsys):
    # Real Kraken: a delisted pair (e.g. ETHWEUR) returns the error together with the known pairs' tickers
    kraken = FakeKraken()
    kraken.set_tickers(prices={'XXBTZEUR': 70000.0})
    kraken.unknown_pairs = {'ETHWEUR'}
    assets_dict = {'XBTEUR': asset_with_trade(name='XBTEUR'), 'ETHWEUR': asset_with_trade(name='ETHWEUR')}

    orders.fill_prices_and_volumes(kapi=kraken, assets_dict=assets_dict)

    assert assets_dict['XBTEUR'].price == 70000.0
    out = capsys.readouterr().out
    assert (
        "No Kraken ticker price (price 0, not ranked): ['ETHWEUR']. "
        "Kraken Ticker error ['EQuery:Unknown asset pair'], add them to EXCLUDE_PAIR_NAMES if delisted."
    ) in out


def test_ticker_error_without_any_price_raises_clear_error(no_local_prices):
    kraken = FakeKraken()
    kraken.queue_error(endpoint='Ticker', error='EGeneral:Internal error')

    with pytest.raises(KrakenError, match='Internal error'):
        orders.fill_prices_and_volumes(kapi=kraken, assets_dict={'XBTEUR': asset_with_trade(name='XBTEUR')})


def test_oldest_order_uses_creation_time_not_response_order():
    kraken = FakeKraken()
    # Added newest first, so the response order is not the creation order
    kraken.add_open_orders(
        [
            make_open_order(opentm=datetime(2026, 9, 20).timestamp(), type='sell', price=80000),
            make_open_order(opentm=datetime(2026, 9, 5).timestamp(), type='sell', price=90000),
            make_open_order(opentm=datetime(2026, 9, 1).timestamp(), type='buy', price=50000),
        ],
    )
    asset = Asset(name='XBTEUR', original_name='XXBTZEUR')

    orders.fill_orders(open_orders=kraken.query_private('OpenOrders'), assets_dict={'XBTEUR': asset})

    assert asset.oldest_order(type='sell').creation_datetime == datetime(2026, 9, 5)
    assert asset.oldest_order(type='buy').creation_datetime == datetime(2026, 9, 1)
    assert asset.oldest_order().creation_datetime == datetime(2026, 9, 1)
    assert Asset(name='ADAEUR', original_name='ADAEUR').oldest_order(type='buy') is None


def test_print_last_trades_skips_unknown_pair(monkeypatch, capsys):
    monkeypatch.setattr(orders, 'PAIR_TO_LAST_TRADES', ['NOPEEUR'])
    orders.print_last_trades(assets_dict={})
    assert 'NOPEEUR' in capsys.readouterr().out


def test_build_assets_skips_eurusd_order():
    # An open EURUSD order (EUR -> USD) became the 'EURUSDEUR' asset and Kraken Ticker failed for every pair
    kraken = FakeKraken()
    kraken.add_open_orders([make_open_order(opentm=datetime(2026, 10, 1).timestamp(), pair='EURUSD', price=1.12)])

    assets_dict = orders.build_assets(
        balance={'result': {'ZEUR': '100.0', 'ADA': '10.0', 'AAPLx.T': '1.0'}},  # a stock in USD: no asset
        open_orders=kraken.query_private('OpenOrders'),
        currency='EUR',
    )

    assert list(assets_dict) == ['ADAEUR']


def test_eurusd_sell_order_holds_its_eur_out_of_the_remaining_cash(capsys):
    kraken = FakeKraken()
    opentm = datetime(2026, 10, 1).timestamp()
    kraken.add_open_orders(
        [
            make_open_order(opentm=opentm, pair='ADAEUR', type='buy', price=0.5, vol=100),  # holds 50 EUR
            make_open_order(opentm=opentm, pair='EURUSD', type='sell', price=1.12, vol=1000),  # holds 1,000 EUR
            make_open_order(opentm=opentm, pair='EURUSD', type='buy', price=1.10, vol=500),  # holds USD
            make_open_order(opentm=opentm, pair='AAPLxUSD', type='buy', price=200, vol=1),  # holds USD
        ],
    )
    asset = Asset(name='ADAEUR', original_name='ADAEUR')

    _, buys_amount, sells_amount, fx_eur_committed = orders.fill_orders(
        open_orders=kraken.query_private('OpenOrders'),
        assets_dict={'ADAEUR': asset},
    )

    # Before, the USD amounts (1,120 + 550 + 200) were added to the EUR totals and the 1,000 EUR were not held
    assert (buys_amount, sells_amount, fx_eur_committed) == (50, 0, 1000)
    assert len(asset.orders) == 1

    orders.print_cash_summary(
        sells_amount=sells_amount,
        buys_amount=buys_amount,
        cash_eur=5000,
        fx_eur_committed=fx_eur_committed,
        staked_eur=0,
        count_missing_buys=0,
        count_remaining_buys=0,
        count_all_remaining_buys=0,
    )
    out = capsys.readouterr().out
    assert '3.95K' in out  # Remaining Cash: 5,000 - 50 - 1,000
    assert 'EUR held by EURUSD sells' in out
