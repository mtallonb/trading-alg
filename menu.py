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
    print("\n╔══════════════════════════════╗")
    print("║       Trading Dashboard       ║")
    print("╠══════════════════════════════╣")
    for i, (name, _) in enumerate(SCRIPTS, 1):
        print(f"║  {i}. {name:<26}║")
    print("║  0. Salir                    ║")
    print("╚══════════════════════════════╝")


def main():
    while True:
        print_menu()
        try:
            choice = input("\nElige una opción: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo...")
            sys.exit(0)

        if choice == "0":
            print("Saliendo...")
            sys.exit(0)

        if not choice.isdigit() or not (1 <= int(choice) <= len(SCRIPTS)):
            print(f"Opción no válida. Elige entre 0 y {len(SCRIPTS)}.")
            continue

        name, script = SCRIPTS[int(choice) - 1]
        print(f"\n▶ Ejecutando {name}...\n{'─' * 34}")
        result = subprocess.run([sys.executable, str(script)])
        print(f"\n{'─' * 34}")
        print(f"✓ {name} finalizado (código: {result.returncode})")
        input("\nPulsa Enter para volver al menú...")


if __name__ == "__main__":
    main()
