# TODO
# Add fees

from random import randint

from utils.basic import count_sells_in_range, my_round, read_prices_from_local_file

# Para las semillas usar un rango de dias igual en todas las
# ejecuciones por ejemplo 200 sesiones
# habrá que empezar dejando ese número de sesiones en la última

# prepare request
# req_data = {'docalcs': 'true'}
req_data = {'trades': 'false'}

ASSET_NAMES = ['XBTEUR', 'ETHEUR', 'ADAEUR', 'TRXEUR', 'SOLEUR', 'MINAEUR']
ENTRY_POINTS = 20
SESSIONS = 600
BUY_PERCENTAGE = SELL_PERCENTAGE = 0.15  # Risk percentage to sell/buy 20%
MINIMUM_BUY_AMOUNT = 70
SELL_HOLD_PERC = 0.9  # 1 means we sell all, 0.5 we preserve 0.5 of the gain
CONSECUTIVE_TRADES_LIMIT = 4
FEES_PERCENTAGE = 0.01


pairs_info = []
for asset_name in ASSET_NAMES:
    df_prices, _ = read_prices_from_local_file(asset_name=asset_name)
    if df_prices.empty:
        print(f'No prices found for {asset_name}')
        continue

    latest_date = df_prices.DATE.iloc[-1]

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
            buy_limit=CONSECUTIVE_TRADES_LIMIT,
        )
        total_sells += sells_count
    avg_sells = my_round(total_sells / ENTRY_POINTS)
    gain_amount = my_round(avg_sells * MINIMUM_BUY_AMOUNT * SELL_PERCENTAGE)
    fee_amount = my_round(avg_sells * MINIMUM_BUY_AMOUNT * (1 + SELL_PERCENTAGE) * FEES_PERCENTAGE)
    print(f'AVG sells for asset: {asset_name}|{avg_sells}|{gain_amount}')
    pairs_info.append(
        {'name': asset_name, 'avg_sells': avg_sells, 'gain_amount': gain_amount, 'fee_amount': fee_amount},
    )
    # Sort by avg_sells descending

sorted_asset = sorted(pairs_info, key=lambda x: x['avg_sells'], reverse=True)
total_amount = 0
print("NAME -> AVG_SELLS -> GAIN_AMOUNT -> FEE_AMOUNT")
for ele in sorted_asset:
    print(f"{ele['name']} -> {ele['avg_sells']} -> {ele['gain_amount']} -> {ele['fee_amount']}")
    total_amount += ele['gain_amount'] - ele['fee_amount']

print(f'TOTAL AMOUNT: {total_amount}')


# stats_list = []
# for _ in range(ENTRY_POINTS):
#     e = Experiment(pairs, AMOUNT_BS, SELL_HOLD_PERC, CONSECUTIVE_TRADES_LIMIT, EXPECTED_GL)
#     stats_pair_list = e.execute()
#     stats_list.extend(stats_pair_list)

# stats_df = DataFrame([o.__dict__ for o in stats_list])
# print(f'Avg: {stats_df["unrealisedPL"].mean()}')
