from utils.basic import TRADES_CSV_HEADER, append_trades_to_csv, read_trades_csv
from utils.classes import CSVTrade


def csv_trade(completed: str, type: str = 'buy') -> CSVTrade:
    return CSVTrade(
        asset_name='ADAEUR',
        completed=completed,
        type=type,
        price='0.5',
        cost='50',
        fee='0.1',
        vol='100',
    )


def read_back(path) -> tuple[list[CSVTrade], list[CSVTrade]]:
    buy_trades, sell_trades = [], []
    read_trades_csv(filename=path, buy_trades=buy_trades, sell_trades=sell_trades)
    return buy_trades, sell_trades


def test_append_to_empty_file_writes_header_so_first_trade_is_kept(tmp_path):
    path = tmp_path / 'trades.csv'
    path.touch()

    append_trades_to_csv(
        filename=path,
        trades_to_append=[csv_trade(completed='2026-09-01 10:00:00'), csv_trade(completed='2026-09-02 10:00:00')],
    )

    assert path.read_text().splitlines()[0] == ','.join(TRADES_CSV_HEADER)
    buy_trades, _ = read_back(path=path)
    assert [str(trade.completed) for trade in buy_trades] == ['2026-09-01 10:00:00', '2026-09-02 10:00:00']


def test_append_to_missing_file_writes_header(tmp_path):
    path = tmp_path / 'new_trades.csv'

    append_trades_to_csv(filename=path, trades_to_append=[csv_trade(completed='2026-09-01 10:00:00')])

    assert len(read_back(path=path)[0]) == 1


def test_append_to_existing_file_does_not_repeat_header(tmp_path):
    path = tmp_path / 'trades.csv'
    append_trades_to_csv(filename=path, trades_to_append=[csv_trade(completed='2026-09-01 10:00:00')])

    append_trades_to_csv(filename=path, trades_to_append=[csv_trade(completed='2026-09-02 10:00:00', type='sell')])

    lines = path.read_text().splitlines()
    assert len(lines) == 3 and lines.count(','.join(TRADES_CSV_HEADER)) == 1
    buy_trades, sell_trades = read_back(path=path)
    assert len(buy_trades) == 1 and len(sell_trades) == 1
