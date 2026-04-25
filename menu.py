#!/usr/bin/env python3
import subprocess
import sys

from pathlib import Path

SRC = Path(__file__).parent / "src"

SCRIPTS = [
    ("Balances", SRC / "balances.py"),
    ("Summary Trades", SRC / "summary_trades.py"),
    ("Orders", SRC / "orders.py"),
    ("eToro Orders", SRC / "etoro_orders.py"),
    ("Backtest", SRC / "backtest.py"),
]


def print_menu():
    title = "Trading Dashboard"
    entries = [(str(i), name) for i, (name, _) in enumerate(SCRIPTS, 1)] + [("0", "Exit")]
    inner = max(len(title) + 4, max(len(n) for _, n in entries) + 7)
    h = "═" * inner
    print(f"\n╔{h}╗")
    print(f"║{title.center(inner)}║")
    print(f"╠{h}╣")
    for num, name in entries:
        row = f"  {num}. {name}"
        print(f"║{row:<{inner}}║")
    print(f"╚{h}╝")


def main():
    while True:
        print_menu()
        try:
            choice = input("\nChoose an option: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

        if choice == "0":
            print("Exiting...")
            sys.exit(0)

        if not choice.isdigit() or not (1 <= int(choice) <= len(SCRIPTS)):
            print(f"Invalid option. Choose between 0 and {len(SCRIPTS)}.")
            continue

        name, script = SCRIPTS[int(choice) - 1]
        print(f"\n▶ Running {name}...\n{'─' * 34}")
        result = subprocess.run([sys.executable, str(script)])
        print(f"\n{'─' * 34}")
        print(f"✓ {name} finished (exit code: {result.returncode})")
        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()
