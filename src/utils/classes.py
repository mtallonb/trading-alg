import string

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal as D
from random import randint
from textwrap import dedent
from typing import List

from pandas import DataFrame

OP_BUY = 'buy'
OP_SELL = 'sell'
MAPPING_NAMES = {'XBTEUR': 'BTCEUR', 'XDGEUR': 'DOGEUR'}


class Order:
    def __init__(self, txid: str, order_type: str, shares: float, price: float):
        self.txid = txid
        self.order_type = order_type
        self.shares = shares
        self.price = price
        self.amount = 0
        self.creation_datetime = None

    def __str__(self):
        from utils.basic import my_round

        return (
            f'{self.order_type} {my_round(self.shares)} @ {my_round(self.price)} '
            f'Amount: {my_round(self.shares * self.price)} -creation time: {self.creation_datetime.date()}'
        )


class Trade:
    def __init__(
        self,
        trade_type: str,
        shares: float,
        price: float,
        amount: float = 0.0,
        execution_datetime: datetime | None = None,
    ):
        self.trade_type = trade_type
        self.shares = shares
        self.price = price
        self.amount = amount
        self.execution_datetime = execution_datetime

    def __str__(self):
        from utils.basic import my_round

        return (
            f'{self.trade_type} {my_round(self.shares)} @ {my_round(self.price)} '
            f'Amount: {my_round(self.shares * self.price)} -closed time: {self.execution_datetime}'
        )

    def to_dict(self):
        return {
            'trade_type': self.trade_type,
            'shares': self.shares,
            'price': self.price,
            'amount': self.amount,
            'execution_datetime': self.execution_datetime,
        }

    def is_partial(self, trade):
        # Return true if is a partial trade
        return (
            self.trade_type == trade.trade_type
            and self.price == trade.price
            and self.execution_datetime.date() == trade.execution_datetime.date()
        )

    def sum_trade(self, trade):
        self.shares += trade.shares
        self.amount += trade.amount


@dataclass
class Currency:
    name: string

    # Optional fields
    exc_rate_to_eur: float = 1.0


@dataclass
class Asset:
    name: string
    original_name: string
    orders: list[Order] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)

    # Optional fields
    description: string = ''
    currency: Currency = None

    # Dataframe with columns DATE and PRICE
    close_prices: DataFrame | None = None
    # Dataframe with columns DATE and VOL_EUR
    close_volumes: DataFrame | None = None

    price: float = 0.0
    shares: float = 0.0  # Spot shares

    orders_buy_count: int = 0
    orders_buy_amount: float = 0.0
    orders_buy_higher_price: float = 0.0

    orders_sell_count: int = 0
    orders_sell_amount: float = 0.0
    orders_sell_lower_price: float = 0.0

    latest_trade_date: date = None

    trades_buy_shares: float = 0.0
    trades_buy_amount: float = 0.0
    trades_buy_count: int = 0

    trades_sell_shares: float = 0.0
    trades_sell_amount: float = 0.0
    trades_sell_count: int = 0

    last_buys_count: int = 0
    last_buys_shares: float = 0.0
    last_buys_avg_price: float = 0.0

    last_sells_count: int = 0
    last_sells_shares: float = 0.0
    last_sells_avg_price: float = 0.0

    ranking: float = 0.0

    is_staking: bool = False
    staked_shares: float = 0.0  # Incl. autostaked
    autostaked_shares: float = 0.0

    def to_dict(self):
        return {
            'name': self.output_name,
            'original_name': self.original_name,
            'price': self.price,
            # 'shares': self.shares,
            'balance (EUR)': self.balance,
        }

    @property
    def avg_buys(self):
        return self.trades_buy_amount / self.trades_buy_shares if self.trades_buy_shares else 0

    @property
    def avg_sells(self):
        return self.trades_sell_amount / self.trades_sell_shares if self.trades_sell_shares else 0

    @property
    def spot_balance(self) -> float:
        return self.shares * self.price

    @property
    def balance(self) -> float:
        return self.spot_balance + self.stacked_balance

    @property
    def stacked_balance(self) -> float:
        return self.staked_shares * self.price

    @property
    def autostacked_balance(self) -> float:
        return self.autostaked_shares * self.price

    @property
    def manual_stacked_shares(self) -> float:
        return self.staked_shares - self.autostaked_shares

    @property
    def manual_stacked_balance(self) -> float:
        return self.manual_stacked_shares * self.price

    @property
    def margin_amount(self) -> float:
        return self.trades_sell_amount + self.balance - self.trades_buy_amount

    @property
    def output_name(self):
        return MAPPING_NAMES.get(self.name, self.name)

    def avg_session_price(self, days: int) -> float:
        if self.close_prices is None:
            return 0.0
        latest_price = self.close_prices.DATE.iloc[-1]
        session_start = latest_price - timedelta(days=days)
        return self.close_prices[self.close_prices.DATE >= session_start].PRICE.mean()

    def avg_session_volume(self, days: int) -> float:
        if self.close_volumes is None:
            return 0.0
        latest_volume = self.close_volumes.DATE.iloc[-1]
        session_start = latest_volume - timedelta(days=days)
        return self.close_volumes[self.close_volumes.DATE >= session_start].VOL_EUR.mean()

    def get_ranking_message(self) -> str:
        from utils.basic import BCOLORS, my_round

        if self.ranking > 5:
            return f'RANKING: {BCOLORS.OKGREEN}{my_round(self.ranking)}{BCOLORS.ENDC}'
        else:
            return f'RANKING: {BCOLORS.WARNING}{my_round(self.ranking)}{BCOLORS.ENDC}'

    def oldest_order(self, type: str | None = None) -> Order | None:
        if type is None:
            return self.orders[0] if self.orders else None

        for order in self.orders:
            if order.order_type == type:
                return order
        return None

    def latest_trade(self, type: str | None = None) -> Trade | None:
        if type is None:
            return self.trades[0] if self.trades else None
        for trade in self.trades:
            if trade.trade_type == type:
                return trade
        return None

    def latest_max_price_since(self, day: date) -> float | None:
        if not self.close_prices.empty:
            return self.close_prices[self.close_prices.DATE >= day].PRICE.max()
        return None

    def compute_last_buy_sell_avg(self):
        trades = self.trades
        if not trades:
            return

        if trades[0].trade_type == OP_BUY:
            for trade in trades:
                if trade.trade_type == OP_BUY:
                    self.last_buys_count += 1
                    self.last_buys_shares += trade.shares
                    self.last_buys_avg_price += trade.price * trade.shares
                else:
                    break

        else:
            for trade in trades:
                if trade.trade_type == OP_SELL:
                    self.last_sells_count += 1
                    self.last_sells_shares += trade.shares
                    self.last_sells_avg_price += trade.price * trade.shares
                else:
                    break

        self.last_buys_avg_price /= self.last_buys_shares if self.last_buys_shares > 0 else 1
        self.last_sells_avg_price /= self.last_sells_shares if self.last_sells_shares > 0 else 1

    def update_calc(self, trade):
        if trade.trade_type == OP_BUY:
            self.trades_buy_shares += trade.shares
            self.trades_buy_amount += trade.amount
            self.trades_buy_count += 1
        else:
            self.trades_sell_shares += trade.shares
            self.trades_sell_amount += trade.amount
            self.trades_sell_count += 1

    def fill_ticker_info(self, ticker_info):
        if ticker_info:
            self.price = float(ticker_info['c'][0])
        else:
            from utils.basic import BCOLORS

            print(f'{BCOLORS.FAIL}Missing price for asset: {self.name}{BCOLORS.ENDC}')

    def fill_staking_info(self, staking_info):
        if staking_info:
            self.is_staking = True
            self.staked_shares += float(staking_info['amount_allocated']['total']['native'])
        else:
            from utils.basic import BCOLORS

            print(f'{BCOLORS.FAIL}Missing staking info: {self.name}{BCOLORS.ENDC}')

    def update_orders_sell_lower_price(self, price):
        if not self.orders_sell_lower_price:
            self.orders_sell_lower_price = price

        elif price < self.orders_sell_lower_price:
            self.orders_sell_lower_price = price

    def update_orders_buy_higher_price(self, price):
        if not self.orders_buy_higher_price:
            self.orders_buy_higher_price = price

        elif price > self.orders_buy_higher_price:
            self.orders_buy_higher_price = price

    def add_trade(self, trade):
        if self.trades and self.trades[0].is_partial(trade):
            self.trades[0].sum_trade(trade)
        else:
            self.trades.append(trade)
        self.update_calc(trade)

    def insert_trade_on_top(self, trade):
        # if trade is the same we fuse instead
        if self.trades and self.trades[0].is_partial(trade):
            self.trades[0].sum_trade(trade)
        else:
            self.trades.insert(0, trade)
        self.update_calc(trade)

    def check_buys_limit(self, buy_limit, buy_limit_amount, buy_amount):
        trades_list = self.trades
        if (
            len(trades_list) >= buy_limit
            and trades_list[buy_limit - 1].trade_type == OP_BUY
            and buy_amount > buy_limit_amount
        ):
            # Iterate to check remaining are also buys
            for trade in trades_list[: buy_limit - 1]:
                if trade.trade_type != OP_BUY:
                    return False
            return True
        return False

    def check_buys_amount_limit(self, buy_limit_amount):
        margin_amount = self.trades_buy_amount - self.trades_sell_amount - self.balance
        return (margin_amount > buy_limit_amount), margin_amount

    def print_staking_info(self):
        from utils.basic import smart_round

        msg = f"""
        ***** staking info *****
        Shares (incl. autostaked shares): {smart_round(self.staked_shares)}| Balance: {smart_round(self.stacked_balance)}| Total balance (staked+spot): {smart_round(self.balance)},
        AutoStaked shares: {smart_round(self.autostaked_shares)}| AutoStaked balance: {smart_round(self.autostacked_balance)},
        Manual stacked shares: {smart_round(self.manual_stacked_shares)}| Manual stacked balance: {smart_round(self.manual_stacked_balance)},
        """  # noqa: E501

        return dedent(msg)

    def get_buy_avg_msg(self) -> str:
        from utils.basic import BCOLORS, my_round, percentage, smart_round

        perc = -my_round(percentage(self.avg_buys, self.price))

        if self.avg_buys < self.price:
            return f'{BCOLORS.OKGREEN}{smart_round(self.avg_buys)!s} Perc: {perc!s} %{BCOLORS.ENDC}'
        else:
            return f'{BCOLORS.WARNING}{smart_round(self.avg_buys)!s} Perc: {perc!s} %{BCOLORS.ENDC}'

    def print_buy_message(self, gain_perc):
        from utils.basic import BCOLORS, my_round, smart_round

        latest_trade = self.trades[0]
        last_price = latest_trade.price
        next_buy_price = last_price * (1 - gain_perc)
        next_buy_price_half = last_price * (1 - gain_perc / 2)

        optional_price_msg = BCOLORS.OKGREEN + str(my_round(next_buy_price_half)) + BCOLORS.ENDC
        amount_msg = BCOLORS.WARNING + str(my_round(self.last_buys_shares * self.last_buys_avg_price)) + BCOLORS.ENDC

        message = f"""
            Missing buy: {self.name}| Price to set: {smart_round(next_buy_price)}| {self.get_ranking_message()},
            Curr. Price: {smart_round(self.price)}| latest trade price: {smart_round(last_price)},
            Spot + Autostaking. Shares: {smart_round(self.shares + self.autostaked_shares)}| Spot + Autostaking. Balance: {smart_round(self.spot_balance + self.autostacked_balance)},
            Latest trade: Amount: {smart_round(latest_trade.amount)}| Vol: {smart_round(latest_trade.shares)}| Exec date: {latest_trade.execution_datetime.date()},
            ALL buys: Avg price: {self.get_buy_avg_msg()}| Amount: {smart_round(self.trades_buy_amount)},
            ALL sells amount: {smart_round(self.trades_sell_amount)},
            Margin amount(Sells-Buys): {smart_round(self.margin_amount)},
            Accum. sell vol: {smart_round(self.last_sells_shares)},
            AVG sell price {smart_round(self.last_sells_avg_price)},
            Accum. sell amount: {smart_round(self.last_sells_shares * self.last_sells_avg_price)},
            Accum. buy vol: {smart_round(self.last_buys_shares)},
            AVG buy price {smart_round(self.last_buys_avg_price)},
            Accum buy count|amount: {self.last_buys_count}|{amount_msg},
            Optionally price to set (half perc / {gain_perc / 2}): {optional_price_msg},
            Prices AVG (200)(50)(10): {smart_round(self.avg_session_price(days=200))}| {smart_round(self.avg_session_price(days=50))}| {smart_round(self.avg_session_price(days=10))}
            Volumes AVG (200)(50)(10): {smart_round(self.avg_session_volume(days=200))}| {smart_round(self.avg_session_volume(days=50))}| {smart_round(self.avg_session_volume(days=10))}
            """  # noqa: E501

        message = dedent(message)
        if self.is_staking:
            message += self.print_staking_info()
        return (BCOLORS.BOLD + message + BCOLORS.ENDC) if self.price <= next_buy_price else message

    def print_set_order_message(self, order_type: str, order_percentage: float, minimum_order_amount: float) -> None:
        from utils.basic import BCOLORS, my_round

        last_trade = self.latest_trade(type=order_type)
        if last_trade:
            price_to_trade = (
                last_trade.price * (1 - order_percentage)
                if order_type == OP_BUY
                else last_trade.price * (1 + order_percentage)
            )
            shares_to_trade = minimum_order_amount / price_to_trade
            print(
                BCOLORS.WARNING
                + f'Going to create {order_type} order from pair: {self.name}.'
                + f'Price: {my_round(price_to_trade)}, Shares: {my_round(shares_to_trade)}'
                + f'| Total amount: {my_round(price_to_trade * shares_to_trade)}.'
                + BCOLORS.ENDC,
            )

    def get_sell_avg_msg(self) -> str:
        from utils.basic import BCOLORS, percentage, smart_round

        perc = smart_round(percentage(self.price, self.avg_sells))

        if self.avg_sells > self.price:
            return f'{BCOLORS.OKGREEN}{smart_round(self.avg_sells)!s} Perc: {perc!s} %{BCOLORS.ENDC}'
        else:
            return f'{BCOLORS.WARNING}{smart_round(self.avg_sells)!s} Perc: {perc!s} %{BCOLORS.ENDC}'

    def print_sell_message(self, gain_perc, minimum_amount):
        from utils.basic import BCOLORS, smart_round

        latest_trade = self.trades[0]
        last_price = latest_trade.price
        next_price = last_price * (1 + gain_perc)
        suggested_buy_price = None

        if self.balance < 1.5 * minimum_amount:
            max_close_price_after_trade = 0
            if self.close_prices is not None:
                max_close_price_after_trade = self.latest_max_price_since(day=latest_trade.execution_datetime.date())

            if max_close_price_after_trade:
                suggested_buy_price = max_close_price_after_trade * (1 - gain_perc)

        sell_amount = smart_round(self.last_sells_shares * self.last_sells_avg_price)

        message = f"""
        Missing sell: {self.name}| price to set: {smart_round(next_price)}| {self.get_ranking_message()},
        Curr. price: {smart_round(self.price)}| latest trade price: {smart_round(last_price)},
        Spot + Autostaking. Shares: {smart_round(self.shares + self.autostaked_shares)}| Spot + Autostaking. Balance: {smart_round(self.spot_balance + self.autostacked_balance)},
        Latest trade amount: {smart_round(latest_trade.amount)}| latest trade vol: {smart_round(latest_trade.shares)},
        Execution date: {latest_trade.execution_datetime.date()},
        ALL sells  Avg price: {self.get_sell_avg_msg()},
        Accum. buy vol: {smart_round(self.last_buys_shares)},
        AVG buy price {smart_round(self.last_buys_avg_price)},
        Accum. buy amount: {smart_round(self.last_buys_shares * self.last_buys_avg_price)},
        Suggested buy price to set based on max after last trade: {smart_round(suggested_buy_price)},
        Accum. sell vol: {smart_round(self.last_sells_shares)},
        AVG sell price {smart_round(self.last_sells_avg_price)},
        Accum sell count|Amount: {self.last_sells_count}|{smart_round(sell_amount)}
        """  # noqa
        message = dedent(message)
        if self.is_staking:
            message += self.print_staking_info()
        return BCOLORS.BOLD + message + BCOLORS.ENDC


class PriceOHLC:
    def __init__(self, open, high, low, close, day):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.day = day

    def __str__(self):
        return f'Close: {self.close} on: {self.day}'

    def avg_price(self):
        return (self.high + self.low) / 2 or self.close

    def is_order_executed_today(self, order_type, price):
        if order_type == OP_SELL:
            return price <= self.close
        else:
            return self.close <= price


@dataclass
class Stats:
    realisedPL: float = 0.0
    unrealisedPL: float = 0.0
    realisedPL_perc: float = 0.0  # Base 0-1
    unrealisedPL_perc: float = 0.0  # Base 0-1

    def to_dict(self):
        return {
            'realisedPL': self.realisedPL,
            'unrealisedPL': self.unrealisedPL,
            'realisedPL_perc': self.realisedPL_perc,
            'unrealisedPL_perc': self.unrealisedPL_perc,
        }


@dataclass
class PairPrices:
    code: str = ''

    consecutive_buys: int = 0
    consecutive_sells: int = 0
    shares: float = 0.0

    prices: List[PriceOHLC] = field(default_factory=list)
    buy_trades: List[Trade] = field(default_factory=list)
    sell_trades: List[Trade] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)

    realisedPL: float = 0.0
    unrealisedPL: float = 0.0
    realisedPL_perc: float = 0.0  # Base 0-1
    unrealisedPL_perc: float = 0.0  # Base 0-1

    @property
    def last_price(self):
        return self.prices[-1].close

    @property
    def last_buy_trade(self):
        return self.buy_trades[-1]

    @property
    def last_sell_trade(self):
        return self.sell_trades[-1]

    @property
    def last_trade(self):
        return self.trades[-1]

    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        if trade.trade_type == OP_BUY:
            self.buy_trades.append(trade)
            self.consecutive_sells = 0
            self.consecutive_buys += 1
            self.shares += trade.shares
        else:
            self.sell_trades.append(trade)
            self.consecutive_sells += 1
            self.consecutive_buys = 0
            self.shares -= trade.shares


class Experiment:
    def __init__(self, pairs: List[PairPrices], amount_bs, sell_perc, consecutive_trade_limit, expected_gl):
        self.pairs = pairs
        self.amount_bs = amount_bs
        self.sell_perc = sell_perc
        self.consecutive_trade_limit = consecutive_trade_limit
        self.expected_gl = expected_gl

    def can_buy(self, pprices):
        return pprices.consecutive_buys < self.consecutive_trade_limit

    def can_sell(self, pprices, trade):
        return pprices.shares > trade.shares

    def compute_next_buy_price(self, price):
        return price * (1.0 - self.expected_gl)

    def compute_next_sell_price(self, price):
        return price * (1.0 + self.expected_gl)

    def execute(self):
        stats_list = []
        for pair in self.pairs:
            entry_point = randint(1, len(pair.prices) - 30)
            priceOHLC = pair.prices[entry_point]
            # create buy trade @ avg price of the day
            avg_price = priceOHLC.avg_price()
            try:
                shares = self.amount_bs / avg_price
            except ZeroDivisionError:
                print(f'Zero division on price: {priceOHLC}')
            initial_trade = Trade(OP_BUY, shares, avg_price, self.amount_bs, priceOHLC.day)
            pair.add_trade(initial_trade)
            print(f'Initial trade: {initial_trade} for pair: {pair.code}')
            self.simulate_pair(pair, entry_point)
            stats = self.compute_PL(pair)
            stats_list.append(stats)

        return stats_list

    def simulate_pair(self, pair: PairPrices, entry_point):
        for price in pair.prices[entry_point:]:
            last_trade_price = pair.last_trade.price
            next_buy_price = self.compute_next_buy_price(last_trade_price)
            next_sell_price = self.compute_next_sell_price(last_trade_price)

            if price.is_order_executed_today(OP_BUY, next_buy_price):
                # Create new Trade
                shares = self.amount_bs / next_buy_price
                trade = Trade(OP_BUY, shares, next_buy_price, self.amount_bs, price.day)
                if self.can_buy(pair):
                    print(trade)
                    pair.add_trade(trade)

            if price.is_order_executed_today(OP_SELL, next_sell_price):
                # Create new Trade
                shares = self.amount_bs / next_sell_price
                trade = Trade(OP_SELL, shares, next_sell_price, self.amount_bs, price.day)
                if self.can_sell(pair, trade):
                    print(trade)
                    pair.add_trade(trade)

    # PL means profit/loses
    def compute_PL(self, pp: PairPrices) -> Stats:
        buys_df = DataFrame([o.__dict__ for o in pp.buy_trades])

        if not pp.sell_trades:
            return Stats()

        sell_df = DataFrame([o.__dict__ for o in pp.sell_trades])

        sell_amount = sell_df['amount'].sum()
        invested = buys_df['amount'].sum()
        balance = pp.shares * pp.last_price

        pp.unrealisedPL = sell_amount + balance - invested

        return Stats(unrealisedPL=pp.unrealisedPL)


class CSVTrade:
    def __init__(self, asset_name: str, completed: str, type: str, price: str, cost: str, fee: str, vol: str):
        from utils.basic import DATETIME_FORMAT

        # --- FIX: Initialize as instance attributes ---
        self.related_buys: List[CSVTrade] = []
        self.accumulated_buy_amount: D = D(0)
        # -------------------------------------------

        self.asset_name = asset_name
        self.completed = datetime.strptime(completed, DATETIME_FORMAT)
        self.type = type
        self.price = D(price)
        self.amount = D(cost)
        self.fee = D(fee)
        self.volume = D(vol)
        self.remaining_volume = self.volume

    def __str__(self):
        from utils.basic import my_round

        return (
            f'\n TRADE INFO: Pair: {self.asset_name}, Volume: {my_round(self.volume)}, '
            f'Price: {my_round(self.price)}, Amount {my_round(self.amount)}'
        )

    def to_dict(self):
        return {
            'asset': self.asset_name,
            'amount': self.amount,
            'date_time': self.completed,
            'fee': self.fee,
            'volume': self.volume,
            'type': self.type,
        }
