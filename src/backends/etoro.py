import uuid

import requests

from backends.base import BaseBrokerBackend

BASE_URL = 'https://public-api.etoro.com/api/v1'
KEY_FILE = './data/keys/etoro.key'


def _load_key(filepath: str) -> tuple[str, str]:
    """Load public_key and user_key from a key file (one per line)."""
    with open(filepath) as f:
        lines = [line.strip() for line in f.readlines()]
    return lines[0], lines[1]


class EtoroBackend(BaseBrokerBackend):
    session: requests.Session
    login_id: str

    def __init__(self):
        public_key, user_key = _load_key(KEY_FILE)
        self.login_id = user_key

        self.session = requests.Session()
        self.session.headers.update(
            {
                'x-api-key': public_key,
                'x-user-key': user_key,
                'Content-Type': 'application/json',
            },
        )

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f'{BASE_URL}{path}'
        headers = {'x-request-id': str(uuid.uuid4())}
        response = self.session.get(url, params=params, headers=headers)
        if not response.ok:
            raise requests.HTTPError(
                f'{response.status_code} {response.reason} — {response.text}',
                response=response,
            )
        return response.json()

    def get_deposits_wd(self):
        """Return deposits and withdrawals for the account."""

        deposits = self._get(f'/accounts/{self.login_id}/deposits')
        withdrawals = self._get(f'/accounts/{self.login_id}/withdrawals')
        return {
            'deposits': deposits,
            'withdrawals': withdrawals,
        }

    def balances(self, rec_ids=None):
        """Return portfolio balances. Optionally filter by rec_ids (instrument ids)."""

        portfolio = self._get(f'/v1/accounts/{self.login_id}/portfolio')
        if rec_ids is None:
            return portfolio
        positions = portfolio.get('positions', [])
        return [p for p in positions if p.get('instrumentId') in rec_ids]

    def fill_pairs(self, rec_ids=None):
        """Return instrument/pair details. Optionally filter by rec_ids."""

        params = {}
        if rec_ids:
            params['instrumentIds'] = ','.join(str(i) for i in rec_ids)
        return self._get('/instruments', params=params or None)

    def open_orders(self, rec_ids=None):
        """Return open orders/positions PnL. Optionally filter by rec_ids."""

        data = self._get('/trading/info/real/pnl')
        orders = data if isinstance(data, list) else data.get('positions', [])
        if rec_ids is not None:
            orders = [o for o in orders if o.get('instrumentId') in rec_ids]
        return orders
