from backends.etoro import EtoroBackend

etoro_broker = EtoroBackend()

flows = etoro_broker.get_deposits_wd()
print(f"\nDeposits: {flows['deposits']}\nWithdrawals: {flows['withdrawals']}")
# etoro_broker.balances()
# etoro_broker.fill_pairs()
# etoro_broker.open_orders()
