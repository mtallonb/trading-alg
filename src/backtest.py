# TODO
# Add fees

from random import randint

import pandas as pd

from utils.basic import count_sells_in_range, my_round, read_prices_from_local_file

# Para las semillas usar un rango de dias igual en todas las
# ejecuciones por ejemplo 200 sesiones
# habrá que empezar dejando ese número de sesiones en la última

# prepare request
# req_data = {'docalcs': 'true'}
req_data = {'trades': 'false'}

ASSET_NAMES = ['XBTEUR', 'ETHEUR', 'ADAEUR', 'TRXEUR', 'SOLEUR', 'MINAEUR']
ENTRY_POINTS = 20
SESSIONS = 200
BUY_PERCENTAGE = SELL_PERCENTAGE = 0.15  # Risk percentage to sell/buy 20%
MINIMUM_BUY_AMOUNT = 70
SELL_HOLD_PERC = 0.9  # 1 means we sell all, 0.5 we preserve 0.5 of the gain
CONSECUTIVE_TRADES_LIMIT = 4


pairs_info = []
for asset_name in ASSET_NAMES:
    df_prices = read_prices_from_local_file(asset_name=asset_name)
    if df_prices.empty:
        print(f'No prices found for {asset_name}')
        continue

    latest_timestamp = df_prices.TIMESTAMP.iloc[-1]

    df_prices.rename({'C': 'PRICE'}, axis=1, inplace=True)
    df_prices['DATE'] = pd.to_datetime(df_prices.TIMESTAMP, unit='s').dt.date

    print(f"ASSET: {asset_name}|Initial price: {df_prices.PRICE.iloc[0]}|Last price: {df_prices.PRICE.iloc[-1]}")
    num_prices = len(df_prices)
    total_sells = 0
    for _ in range(ENTRY_POINTS):
        start_point = randint(0, num_prices - SESSIONS)
        df_prices_200 = df_prices.iloc[start_point : start_point + SESSIONS]
        sells_count = count_sells_in_range(
            close_prices=df_prices_200,
            days=SESSIONS,
            buy_perc=BUY_PERCENTAGE,
            sell_perc=SELL_PERCENTAGE,
        )
        total_sells += sells_count
    avg_sells = my_round(total_sells / ENTRY_POINTS)
    print(f'AVG sells for asset: {asset_name}|{avg_sells}')
    pairs_info.append({'name': asset_name, 'avg_sells': avg_sells})
    # Sort by avg_sells descending

sorted_asset = sorted(pairs_info, key=lambda x: x['avg_sells'], reverse=True)
for ele in sorted_asset:
    print(f"{ele['name']} -> {ele['avg_sells']}")


# stats_list = []
# for _ in range(ENTRY_POINTS):
#     e = Experiment(pairs, AMOUNT_BS, SELL_HOLD_PERC, CONSECUTIVE_TRADES_LIMIT, EXPECTED_GL)
#     stats_pair_list = e.execute()
#     stats_list.extend(stats_pair_list)

# stats_df = DataFrame([o.__dict__ for o in stats_list])
# print(f'Avg: {stats_df["unrealisedPL"].mean()}')
