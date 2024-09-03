#!/usr/bin/python3

from datetime import date
import krakenex
from decimal import Decimal

from utils.basic import *

# Invested on each asset and current balance -> result No muy util
# balance a principio de año y actual quitando los flujos de IO

date_from = date(2022, 1, 1)
date_to = date(2023, 1, 1)

VERBOSE = True

filename = './data/trades_2023.csv'
file = None
buy_trades = []
sell_trades = []

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# REPLACE_NAMES = {'ETHEUR': 'XETHZEUR',
#                  'LTCEUR': 'XLTCEUR',
#                  'ETCEUR': 'XETCZEUR',
#                  'XMREUR': 'XXMRZEUR',
#                  'XRPEUR': 'XXRPZEUR',
#                  }

# configure api
kapi = krakenex.API()
kapi.load_key('./data/kraken.key')
PAGES = 5  # 50 RECORDS per page
RECORDS_PER_PAGE = 50

# prepare request
req_data = {'trades': 'false'}

latest_trade_csv = read_trades_csv(filename, buy_trades, sell_trades)

# Sort trades asc
buy_trades_asc = sorted(buy_trades, key=lambda x: x.completed)
sell_trades_asc = sorted(sell_trades, key=lambda x: x.completed)

# Buy /Sells summary
total_buy_amount = sum([buy.amount for buy in buy_trades])
total_sell_amount = sum([sell.amount for sell in sell_trades])
total_fees = sum([trade.fee for trade in buy_trades + sell_trades])


# ---------------------------------------------------------------------------------------------------------
total_deposit_amount, total_wd_amount = get_deposit_wd_info(kapi, PAGES, RECORDS_PER_PAGE, verbose=VERBOSE)
deposit_wd = Decimal(total_deposit_amount + total_wd_amount)
print('\n INVESTED (DEPOSITS-WD): {}'.format(my_round(deposit_wd)))
# --------------------------------------------------------------------------------------------------------------------

cash = deposit_wd - total_buy_amount + total_sell_amount - total_fees
print('\n CASH: {}'.format(my_round(cash)))

# balance on date = position(date) + cash(date) + stacked(date)
balance = kapi.query_private('Balance')
# El balance contiene los stacked tb asi que solo queda multiplicar por el precio cada position
# gain_loss = balance / deposit_wd

