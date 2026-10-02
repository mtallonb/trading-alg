import summary_trades

from utils.classes import CSVTrade

PAIRS = ['XXBTZEUR', 'SOLEUR', 'ADAEUR', 'DOTEUR', 'LINKEUR', 'AAVEEUR']


def trade(pair: str, completed: str, type: str, price: str) -> CSVTrade:
    vol = '10'
    cost = str(float(price) * float(vol))
    return CSVTrade(asset_name=pair, completed=completed, type=type, price=price, cost=cost, fee='0.1', vol=vol)


def test_pairs_are_processed_in_name_order(capsys):
    buy_trades = [trade(pair=pair, completed='2026-01-10 10:00:00', type='buy', price='1') for pair in PAIRS]
    sell_trades = [trade(pair=pair, completed='2026-02-10 10:00:00', type='sell', price='1.2') for pair in PAIRS]

    pair_gains, _ = summary_trades.compute_pair_gains(buy_trades=buy_trades, sell_trades=sell_trades, year=2026)

    assert [pair_gain['name'] for pair_gain in pair_gains] == sorted(PAIRS)
    # Per-pair logs follow the same order
    out = capsys.readouterr().out
    log_positions = [out.find(pair) for pair in sorted(PAIRS)]
    assert -1 not in log_positions and log_positions == sorted(log_positions)
