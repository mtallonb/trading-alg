import string
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal as D
from random import randint

from typing import List

from pandas import DataFrame


@dataclass
class Asset:

    name: string
    original_name: string
    orders: list = field(default_factory=list)
    trades: list = field(default_factory=list)

    price: float = 0.0
    shares: float = 0.0
    balance: float = 0.0

    orders_buy_count:  int = 0
    orders_buy_amount: float = 0.0
    orders_buy_higher_price: float = 0.0

    orders_sell_count: int = 0
    orders_sell_amount: float = 0.0
    orders_sell_lower_price: float = 0.0

    latest_trade_date: date = date(2000, 1, 1)

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
    is_stacking: bool = False

    def to_dict(self):
        return {
            'name': self.name,
            'original_name': self.original_name,
            'price': self.price,
        }

    @property
    def avg_buys(self):
        return self.trades_buy_amount / self.trades_buy_shares if self.trades_buy_shares else 0

    @property
    def avg_sells(self):
        return self.trades_sell_amount / self.trades_sell_shares if self.trades_sell_shares else 0

    def fill_last_shares(self):
        trades = self.trades
        if not trades:
            return

        if trades[0].trade_type == Trade.BUY:
            for trade in trades:
                if trade.trade_type == Trade.BUY:
                    self.last_buys_count += 1
                    self.last_buys_shares += trade.shares
                    self.last_buys_avg_price += trade.price * trade.shares
                else:
                    break

        else:
            for trade in trades:
                if trade.trade_type == Trade.SELL:
                    self.last_sells_count += 1
                    self.last_sells_shares += trade.shares
                    self.last_sells_avg_price += trade.price * trade.shares
                else:
                    break

        self.last_buys_avg_price /= self.last_buys_shares if self.last_buys_shares > 0 else 1
        self.last_sells_avg_price /= self.last_sells_shares if self.last_sells_shares > 0 else 1

    def update_calc(self, trade):
        if trade.trade_type == Trade.BUY:
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
            self.balance = self.price * self.shares
        else:
            from utils.basic import BCOLORS
            print(BCOLORS.FAIL + f'Missing price for asset: {self.name}' + BCOLORS.ENDC)

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
        if len(trades_list) >= buy_limit and trades_list[buy_limit - 1].trade_type == Trade.BUY and \
                buy_amount > buy_limit_amount:
            # Iterate to check remaining are also buys
            for trade in trades_list[:buy_limit - 1]:
                if trade.trade_type != Trade.BUY:
                    return False
            return True
        return False

    def check_buys_amount_limit(self, buy_limit_amount):
        margin_amount = self.trades_buy_amount - self.trades_sell_amount
        return (margin_amount > buy_limit_amount), margin_amount


    def print_buy_message(self, gain_perc):
        from utils.basic import BCOLORS, my_round, percentage

        latest_trade = self.trades[0]
        last_price = latest_trade.price
        next_buy_price = last_price * (1 - gain_perc)
        next_buy_price_half = last_price * (1 - gain_perc / 2)
        buy_avg_price = self.avg_buys
        perc = -my_round(percentage(buy_avg_price, self.price))

        if buy_avg_price < self.price:
            buy_avg_msg = (
                BCOLORS.OKGREEN + str(my_round(buy_avg_price)) + ' Perc: ' + str(perc) + ' %' + BCOLORS.ENDC
            )
        else:
            buy_avg_msg = BCOLORS.WARNING + str(my_round(buy_avg_price)) + ' Perc: ' + str(perc) + ' %' + BCOLORS.ENDC
            
        amount_msg = BCOLORS.WARNING + str(my_round(self.last_buys_shares * self.last_buys_avg_price)) + BCOLORS.ENDC
        message = f"""Missing buy: {self.name}, price to set: {my_round(next_buy_price)}, RANKING: {my_round(self.ranking)}, 
            curr. price: {my_round(self.price)}, latest trade price: {my_round(last_price)},
            latest trade amount: {my_round(latest_trade.amount)}, latest trade vol: {my_round(latest_trade.shares)},
            execution date: {latest_trade.execution_datetime.date()},
            ALL buys Avg price: {buy_avg_msg},
            ALL buys amount: {my_round(self.trades_buy_amount)} - All sells amount: {my_round(self.trades_sell_amount)},
            Margin amount(Buys - Sells): {my_round(self.trades_buy_amount - self.trades_sell_amount)} 
            accum. sell vol: {my_round(self.last_sells_shares)},
            avg. sell price {my_round(self.last_sells_avg_price)}, 
            accum. sell amount: {my_round(self.last_sells_shares * self.last_sells_avg_price)}
            accum. buy vol: {my_round(self.last_buys_shares)}, 
            avg. buy price {my_round(self.last_buys_avg_price)}, 
            accum buy count|amount: {self.last_buys_count}|{amount_msg},
            Optionally price to set (half perc / {gain_perc / 2}): {my_round(next_buy_price_half)}"""  # noqa
        return (BCOLORS.BOLD + message + BCOLORS.ENDC) if self.price <= next_buy_price else message

    def print_sell_message(self, kapi,  gain_perc, minimum_amount):
        from utils.basic import BCOLORS, my_round, percentage, get_max_price_since
        latest_trade = self.trades[0]
        last_price = latest_trade.price
        next_price = last_price * (1 + gain_perc)
        suggested_buy_price = 0

        if self.balance < minimum_amount:
            max_priceOHLC_after_trade = get_max_price_since(kapi, self.original_name, latest_trade.execution_datetime)

            if max_priceOHLC_after_trade:
                suggested_buy_price = max_priceOHLC_after_trade.close * (1 - gain_perc)

        sell_avg_price = self.avg_sells
        perc = my_round(percentage(self.price, sell_avg_price))
        sell_amount = my_round(self.last_sells_shares * self.last_sells_avg_price)

        if sell_avg_price > self.price:
            sell_avg_message = (
                BCOLORS.OKGREEN + str(my_round(sell_avg_price)) + ' Perc: ' + str(perc) + ' %' + BCOLORS.ENDC
            )
        else: 
            sell_avg_message = BCOLORS.WARNING + str(my_round(sell_avg_price)) + ' Perc: ' + str(perc) + ' %' + BCOLORS.ENDC
        
        message = (
            f"""Missing sell: {self.name}, price to set: {my_round(next_price)}, RANKING: {my_round(self.ranking)}, 
            current price: {my_round(self.price)}, latest trade price: {my_round(last_price)},
            latest trade amount: {my_round(latest_trade.amount)}, latest trade vol: {my_round(latest_trade.shares)},
            execution date: {latest_trade.execution_datetime.date()},
            ALL sells  Avg price: {sell_avg_message},
            accum. buy vol: {my_round(self.last_buys_shares)}, 
            avg. buy price {my_round(self.last_buys_avg_price)}, 
            accum. buy amount: {my_round(self.last_buys_shares * self.last_buys_avg_price)},
            Suggested buy price to set based on max after last trade: {my_round(suggested_buy_price)},
            accum. sell vol: {my_round(self.last_sells_shares)}, 
            avg. sell price {my_round(self.last_sells_avg_price)}, 
            accum sell count|Amount: {self.last_sells_count}|{sell_amount}"""  # noqa
        )
        return BCOLORS.BOLD + message + BCOLORS.ENDC


class Trade:
    BUY = 'buy'
    SELL = 'sell'

    def __init__(self, trade_type, shares, price, amount=0.0, execution_datetime=None):
        self.trade_type = trade_type
        self.shares = shares
        self.price = price
        self.amount = amount
        self.execution_datetime = execution_datetime

    def __str__(self):
        from utils.basic import my_round
        return f'{self.trade_type} {my_round(self.shares)} @ {my_round(self.price)} ' \
               f'Amount: {my_round(self.shares * self.price)} -closed time: {self.execution_datetime}'

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
        return (self.trade_type == trade.trade_type and self.price == trade.price and
                self.execution_datetime.date() == trade.execution_datetime.date())

    def sum_trade(self, trade):
        self.shares += trade.shares
        self.amount += trade.amount


class Order:
    BUY = 'buy'
    SELL = 'sell'

    def __init__(self, txid, order_type, shares, price):
        self.txid = txid
        self.order_type = order_type
        self.shares = shares
        self.price = price
        self.amount = 0
        self.creation_datetime = None

    def __str__(self):
        from utils.basic import my_round
        return f'{self.order_type} {my_round(self.shares)} @ {my_round(self.price)} ' \
               f'Amount: {my_round(self.shares*self.price)} -creation time: {self.creation_datetime.date()}'


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
        return (self.high + self.low)/2 or self.close

    def is_order_executed_today(self, order_type, price):
        if order_type == Trade.SELL:
            return price <= self.close
        else:
            return self.close <= price

@dataclass
class Stats:
    realisedPL: float = 0.0
    unrealisedPL: float = 0.0
    realisedPL_perc: float = 0.0 # Base 0-1
    unrealisedPL_perc: float = 0.0 # Base 0-1

    def to_dict(self):
        return {
            'realisedPL': self.realisedPL,
            'unrealisedPL': self.unrealisedPL,
            'realisedPL_perc': self.realisedPL_perc,
            'unrealisedPL_perc': self.unrealisedPL_perc
        }

@dataclass
class PairPrices:

    code: str = ''

    consecutive_buys: int = 0
    consecutive_sells = int = 0
    shares = float = 0.0

    prices: List[PriceOHLC] = field(default_factory=list)
    buy_trades: List[Trade] = field(default_factory=list)
    sell_trades: List[Trade] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)

    realisedPL: float = 0.0
    unrealisedPL: float = 0.0
    realisedPL_perc: float = 0.0 # Base 0-1
    unrealisedPL_perc: float = 0.0 # Base 0-1

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
        if trade.trade_type == Trade.BUY:
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
    def __init__(self, pairs: List[PairPrices], amount_bs, sell_perc, consecutive_trade_limit,
                 expected_gl):
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
            entry_point = randint(1, len(pair.prices)-30)
            priceOHLC = pair.prices[entry_point]
            # create buy trade @ avg price of the day
            avg_price = priceOHLC.avg_price()
            try:
                shares = self.amount_bs / avg_price
            except ZeroDivisionError:
                print(f'Zero division on price: {priceOHLC}')
            initial_trade = Trade(Trade.BUY, shares, avg_price, self.amount_bs, priceOHLC.day)
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

            if price.is_order_executed_today(Order.BUY, next_buy_price):
                # Create new Trade
                shares = self.amount_bs/next_buy_price
                trade = Trade(Trade.BUY, shares, next_buy_price, self.amount_bs, price.day)
                if self.can_buy(pair):
                    print(trade)
                    pair.add_trade(trade)

            if price.is_order_executed_today(Order.SELL, next_sell_price):
                # Create new Trade
                shares = self.amount_bs/next_sell_price
                trade = Trade(Trade.SELL, shares, next_sell_price, self.amount_bs, price.day)
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
    related_buys = []
    accumulated_buy_amount = 0

    def __init__(self, asset_name, completed, type, price, cost, fee, vol):
        from utils.basic import DATE_FORMAT
        self.asset_name = asset_name
        self.completed = datetime.strptime(completed, DATE_FORMAT)
        self.type = type
        self.price = D(price)
        self.amount = D(cost)
        self.fee = D(fee)
        self.volume = D(vol)
        self.remaining_volume = self.volume

    def __str__(self):
        from utils.basic import my_round
        return (f'\n TRADE INFO: Pair: {self.asset_name}, Volume: {my_round(self.volume)}, '
                f'Price: {my_round(self.price)}, Amount {my_round(self.amount)}')

    def to_dict(self):
        return





