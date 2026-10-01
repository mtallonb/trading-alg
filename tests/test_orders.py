from datetime import datetime

import orders

from fake_kraken import FakeKraken, make_open_order
from utils.classes import Asset


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
