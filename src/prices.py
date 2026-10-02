#!/usr/bin/python3
"""Update the local daily prices/volumes files (PRICES_DIR) of PAIRS from Kraken OHLC (public, no keys)."""

from datetime import datetime, timedelta

import krakenex

from utils.basic import update_asset_prices

# Not updated by balances.py (it only updates the traded assets and USDCEUR): exchange rates and xStocks (in USD)
PAIRS = ['EURUSD', 'GOOGLxUSD']


def main():
    kapi = krakenex.API()
    date_to = (datetime.today() - timedelta(days=1)).date()

    for pair in PAIRS:
        df_prices = update_asset_prices(asset_name=pair, kapi=kapi, date_to=date_to)
        if df_prices.empty:
            print(f'{pair}: no prices')
            continue
        print(
            f'{pair}: {len(df_prices)} days from {df_prices.DATE.iloc[0]} to {df_prices.DATE.iloc[-1]}, '
            f'last price {df_prices.PRICE.iloc[-1]}',
        )


if __name__ == '__main__':
    main()
