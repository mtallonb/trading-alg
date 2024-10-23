import krakenex

from my_examples.backends.base import BaseBrokerBackend

req_data = {'trades': 'false'}


class KrakenBackend(BaseBrokerBackend):
    api: krakenex.API
    balance: dict
    open_orders: dict
    cash_eur: float

    def __init__(self, broker):
        super().__init__(broker)
        api = krakenex.API()
        api.load_key('./data/kraken.key')

        # query servers
        start = api.query_public('Time')

        balance = api.query_private('Balance')
        open_orders = api.query_private('OpenOrders', req_data)

        end = api.query_public('Time')

        cash_eur = float(balance['result']['ZEUR'])

    def get_deposits_wd(self):
        pass
