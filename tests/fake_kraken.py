"""In-memory stand-in for krakenex.API, so code that talks to Kraken can be tested without keys or network.

Behaves like the Kraken REST API (https://docs.kraken.com/api-reference/) where the project relies on it:
- TradesHistory / Ledgers: most recent first, `start` exclusive and `end` inclusive (unix time), `ofs` offset,
  `limit` page size (default 50, max 100; Ledgers always 50), `count` with the total matching records.
- OpenOrders: `open` dict keyed by txid, in the order the orders were added (the real order is undocumented).
- OHLC: only the 720 most recent candles of `ohlc[pair]` (oldest first, like Kraken), then filtered by `since`.
- Ticker: all tickers, whatever pairs are asked; if one of them is in `unknown_pairs`, 'EQuery:Unknown asset pair'
  together with the tickers (as the real API does).
- Balance, Earn/Allocations: return the data set.
- CancelOrder: removes the open order and records the txid in `cancelled`.
Errors (e.g. rate limit) can be queued per endpoint with `queue_error`. Every call is logged in `calls`.
"""

from collections import defaultdict, deque

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
OHLC_MAX_CANDLES = 720
RATE_LIMIT_ERROR = 'EAPI:Rate limit exceeded'


def make_trade(
    time: float,
    pair: str = 'XXBTZEUR',
    type: str = 'buy',
    price: float = 70000.0,
    vol: float = 0.001,
    fee: float = 0.1,
    ordertype: str = 'limit',
) -> dict:
    """TradesHistory record as Kraken returns it (numbers as strings, time as unix float)."""
    return {
        'pair': pair,
        'time': time,
        'type': type,
        'ordertype': ordertype,
        'price': str(price),
        'cost': str(price * vol),
        'fee': str(fee),
        'vol': str(vol),
    }


def make_daily_candles(first_day_time: int, days: int, price: float = 100.0) -> list[list]:
    """`days` daily OHLC candles from first_day_time (unix, 00:00 UTC), oldest first, as Kraken returns them."""
    return [
        [first_day_time + day * 86400, str(price), str(price), str(price), str(price + day), str(price), '10.0', 5]
        for day in range(days)
    ]


def make_open_order(opentm: float, pair: str = 'XBTEUR', type: str = 'buy', price: float = 60000.0, vol=0.001):
    """OpenOrders record as Kraken returns it."""
    return {
        'opentm': opentm,
        'status': 'open',
        'vol': str(vol),
        'descr': {'pair': pair, 'type': type, 'ordertype': 'limit', 'order': f'{type} {vol} {pair} @ limit {price}'},
    }


class FakeKraken:
    def __init__(self):
        self.trades: dict[str, dict] = {}  # txid -> record, any order (sorted by time when queried)
        self.ledger: dict[str, dict] = {}
        self.open_orders: dict[str, dict] = {}
        self.balance: dict[str, str] = {}
        self.tickers: dict[str, dict] = {}
        self.unknown_pairs: set[str] = set()  # asked in Ticker: error + the known tickers (e.g. delisted pairs)
        self.allocations: list[dict] = []
        self.ohlc: dict[str, list] = {}
        self.cancelled: list[str] = []
        self.calls: list[tuple[str, dict]] = []
        self._error_plans: dict[str, deque] = defaultdict(deque)  # endpoint -> [calls to let pass, error, times]

    # --- setup helpers --------------------------------------------------------------------------------------------
    def add_trades(self, trades: list[dict]):
        for trade in trades:
            self.trades[f'T{len(self.trades):05d}-{trade["time"]}'] = trade

    def add_ledger(self, records: list[dict]):
        for record in records:
            self.ledger[f'L{len(self.ledger):05d}-{record["time"]}'] = record

    def add_open_orders(self, orders: list[dict]):
        for order in orders:
            self.open_orders[f'O{len(self.open_orders):05d}'] = order

    def set_tickers(self, prices: dict[str, float]):
        self.tickers = {pair: {'c': [str(price), '1.0']} for pair, price in prices.items()}

    def queue_error(self, endpoint: str, error: str = RATE_LIMIT_ERROR, times: int = 1, after: int = 0):
        """After `after` successful calls to endpoint, the next `times` calls answer with error instead of a result."""
        self._error_plans[endpoint].append([after, error, times])

    def calls_to(self, endpoint: str) -> list[dict]:
        return [params for name, params in self.calls if name == endpoint]

    # --- krakenex.API interface -----------------------------------------------------------------------------------
    def load_key(self, path: str):
        pass

    def query_private(self, method: str, data: dict | None = None, timeout=None) -> dict:
        return self._query(method=method, data=data)

    def query_public(self, method: str, data: dict | None = None, timeout=None) -> dict:
        return self._query(method=method, data=data)

    # --- endpoints ------------------------------------------------------------------------------------------------
    def _query(self, method: str, data: dict | None) -> dict:
        params = dict(data or {})
        self.calls.append((method, params))
        plans = self._error_plans[method]
        if plans:
            plan = plans[0]
            if plan[0] > 0:
                plan[0] -= 1
            else:
                plan[2] -= 1
                if plan[2] == 0:
                    plans.popleft()
                return {'error': [plan[1]]}

        handlers = {
            'TradesHistory': lambda: self._history(records=self.trades, key='trades', params=params, limit=True),
            'Ledgers': lambda: self._history(records=self._filtered_ledger(params=params), key='ledger', params=params),
            'OpenOrders': lambda: {'open': dict(self.open_orders)},
            'Balance': lambda: dict(self.balance),
            # Like the real one for this project: every ticker set, whatever pairs are asked
            'Ticker': lambda: dict(self.tickers),
            'Earn/Allocations': lambda: {'items': list(self.allocations)},
            'OHLC': lambda: self._ohlc(params=params),
            'CancelOrder': lambda: self._cancel(params=params),
        }
        asked_pairs = {pair.upper() for pair in params.get('pair', '').split(',') if pair}
        if method == 'Ticker' and asked_pairs & self.unknown_pairs:
            # Real Kraken: the error comes together with the known pairs' tickers
            return {'error': ['EQuery:Unknown asset pair'], 'result': dict(self.tickers)}
        if method not in handlers:
            return {'error': [f'EGeneral:Unknown method {method}']}
        return {'error': [], 'result': handlers[method]()}

    def _history(self, records: dict[str, dict], key: str, params: dict, limit: bool = False) -> dict:
        start, end = params.get('start'), params.get('end')
        selected = [
            (txid, record)
            for txid, record in records.items()
            if (start is None or record['time'] > float(start)) and (end is None or record['time'] <= float(end))
        ]
        selected.sort(key=lambda item: item[1]['time'], reverse=True)
        page_size = min(int(params.get('limit', DEFAULT_PAGE_SIZE)), MAX_PAGE_SIZE) if limit else DEFAULT_PAGE_SIZE
        offset = int(params.get('ofs', 0))
        return {key: dict(selected[offset : offset + page_size]), 'count': len(selected)}

    def _ohlc(self, params: dict) -> dict:
        # Only the OHLC_MAX_CANDLES most recent candles exist for the API, then `since` filters them
        pair = params.get('pair')
        candles = self.ohlc.get(pair, [])[-OHLC_MAX_CANDLES:]
        since = params.get('since')
        if since is not None:
            candles = [candle for candle in candles if candle[0] > int(since)]
        return {pair: candles, 'last': candles[-1][0] if candles else 0}

    def _filtered_ledger(self, params: dict) -> dict[str, dict]:
        ledger_type = params.get('type')
        return {txid: rec for txid, rec in self.ledger.items() if ledger_type in (None, 'all', rec.get('type'))}

    def _cancel(self, params: dict) -> dict:
        txid = params.get('txid')
        count = 1 if self.open_orders.pop(txid, None) else 0
        if count:
            self.cancelled.append(txid)
        return {'count': count}
