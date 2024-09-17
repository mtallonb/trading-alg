# TODO
# Add fees

import cryptowatch as cw
import krakenex
import pandas as pd

from pandas import DataFrame

from utils.basic import *
from utils.classes import Experiment, PairPrices

# configure api
kapi = krakenex.API()

# GET ALL PRICES FROM https://github.com/uoshvis/python-cryptowatch
# https://pypi.org/project/cryptowatch-sdk/

# prepare request
# req_data = {'docalcs': 'true'}
req_data = {'trades': 'false'}

# PAIRS = ['XXBTZEUR', 'XETHZEUR', 'ADAEUR']
PAIRS = ['btceur', 'etheur', 'adaeur']
ENTRY_POINTS = 20
AMOUNT_BS = 80
EXPECTED_GL = 0.20
SELL_HOLD_PERC = 0.9  # 1 means we sell all, 0.5 we preserve 0.5 of the gain
CONSECUTIVE_TRADES_LIMIT = 4

PRICES_INTERVAL = 1440  # 1440 for a day. Available values (minutes): 1 5 15 30 60 240 1440 10080 21600

pairs_dict = {}

# concatenate_names = ','.join(PAIRS)
default_timestamp = 1548111600
print(f'default_timestamp: {default_timestamp}, {from_timestamp_to_str(default_timestamp)}')
timestamp = datetime(2018, 1, 1).timestamp()
print(f'New timestamp: {timestamp}, {from_timestamp_to_str(default_timestamp)}')


pairs = []
# for pair_name in PAIRS:
#     pp = PairPrices(pair_name)
#     pairs.append(pp)
#     tickers_prices = kapi.query_public('OHLC', {'pair': pair_name, 'interval': 1440})
#     if not tickers_prices.get('result'):
#         print(f'ERROR: Asset {pair_name} not found')
#         exit(-1)
#     for price in tickers_prices['result'][pair_name]:
#         day = from_timestamp_to_datetime(price[0]).date()
#         priceOHLC = PriceOHLC(float(price[1]), float(price[2]), float(price[3]), float(price[4]), day)
#         pp.prices.append(priceOHLC)
#     print(f"Initial price for ({pair_name}): {pp.prices[0]}")
#     print(f"Last price for ({pair_name}): {pp.prices[-1]}")

# date_to = date(2023, 1, 1)
date_to = date.today()

# timestamp_from = datetime.datetime.combine(date_from, datetime.time()).timestamp()
timestamp_to = datetime(date_to.year, date_to.month, date_to.day).timestamp()



for pair_name in PAIRS:
    pp = PairPrices(pair_name)
    pairs.append(pp)
    prices = cw.markets.get(f"kraken:{pair_name}", ohlc=True, ohlc_before=int(timestamp_to), periods=["1d"])

    prices_df = pd.DataFrame(prices.of_1d)
    # We take only dates and Close prices
    close_prices = prices_df[[0, 4]]
    close_prices.columns = ["date", "price"]
    close_prices['date'] = pd.to_datetime(close_prices['date'], unit='s')
    for close_price in close_prices.to_dict('records'):
        day = close_price['date'].date()
        price = close_price['price']
        priceOHLC = PriceOHLC(float(price), float(price), float(price), float(price), day)
        pp.prices.append(priceOHLC)

    print(f"Initial price for ({pair_name}): {pp.prices[0]}")
    print(f"Last price for ({pair_name}): {pp.prices[-1]}")


stats_list = []
for _ in range(ENTRY_POINTS):
    e = Experiment(pairs, AMOUNT_BS, SELL_HOLD_PERC, CONSECUTIVE_TRADES_LIMIT, EXPECTED_GL)
    stats_pair_list = e.execute()
    stats_list.extend(stats_pair_list)

stats_df = DataFrame([o.__dict__ for o in stats_list])
print(f'Avg: {stats_df["unrealisedPL"].mean()}')
