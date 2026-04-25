from backends.etoro import EtoroBackend

etoro_broker = EtoroBackend()

# flows = etoro_broker.get_deposits_wd()
# print(f"\nDeposits: {flows['deposits']}\nWithdrawals: {flows['withdrawals']}")
# etoro_broker.balances()
# etoro_broker.fill_pairs()
open_orders = etoro_broker.open_orders()
print(open_orders)
