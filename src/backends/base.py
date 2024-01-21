from abc import ABCMeta, abstractmethod


class BaseBrokerBackend(metaclass=ABCMeta):
    broker = None

    def __init__(self, broker):
        self.broker = broker

    @abstractmethod
    def get_deposits_wd(self):
        pass

    @abstractmethod
    def balances(self, rec_ids=None):
        pass

    @abstractmethod
    def fill_pairs(self, rec_ids=None):
        pass

    @abstractmethod
    def open_orders(self, rec_ids=None):
        pass



