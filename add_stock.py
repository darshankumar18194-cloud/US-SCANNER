"""Add one or more symbols to shared-data/symbols.csv after checking that Yahoo really has data for them.

    python add_stock.py NVDA
    python add_stock.py NVDA --name "NVIDIA" --type Stock
    python add_stock.py SAMSUNG --yahoo 005930.KS --name "Samsung Electronics"
    python add_stock.py --bulk "NVDA, PLTR, SPY"          # what the GitHub 'Run workflow' box uses
    python add_stock.py FOO --no-check                     # add without asking Yahoo

The new symbol is scanned on the next run (or run `python scan.py --symbols NVDA` now).
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional, Tuple

import market_data
import symbols as sym

MIN_CANDLES = 60


def check_yahoo(yahoo: str, history_fn=None) -> Tuple[bool, str]:
    kw = {"history_fn": history_fn} if history_fn else {}
    r = market_data.fetch_history(yahoo, drop_forming=False, **kw)
    if r.status != "OK":
        return False, f"Yahoo has no usable 1h data for {yahoo} ({r.status}: {r.error})"
    if len(r.df) < MIN_CANDLES:
        return False, f"only {len(r.df)} hourly candles for {yahoo} (need {MIN_CANDLES}+) - too new to scan"
    return True, ""


def add_one(symbol: str, name: str = "", type_: str = "Stock", yahoo: str = "", check: bool = True,
            path=None, history_fn=None) -> Tuple[bool, str]:
    s = sym.clean_symbol(symbol)
    if not s or not re.fullmatch(r"[A-Z0-9.\-^=]{1,20}", s):
        return False, f"'{symbol}' is not a valid ticker"
    target = sym.clean_symbol(yahoo) if yahoo else sym.default_yahoo(s)
    if check:
        ok, why = check_yahoo(target, history_fn)
        if not ok:
            return False, why
    try:
        sym.append_symbol(s, name, type_, yahoo, path=path)
    except ValueError as e:
        return False, str(e)
    return True, f"added {s}" + (f" (Yahoo: {target})" if target != s else "")


def split_bulk(text: str) -> List[str]:
    return [t for t in re.split(r"[\s,;]+", text or "") if t]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol", nargs="?")
    ap.add_argument("--name", default="")
    ap.add_argument("--type", default="Stock", choices=["Stock", "ETF", "stock", "etf"])
    ap.add_argument("--yahoo", default="", help="Yahoo ticker when it differs from the symbol, e.g. 005930.KS")
    ap.add_argument("--bulk", default="", help="comma/space separated list of symbols")
    ap.add_argument("--no-check", action="store_true", help="do not ask Yahoo first")
    args = ap.parse_args(argv)
    if not args.symbol and not args.bulk:
        ap.error("give a symbol or --bulk")
    jobs = [(args.symbol, args.name, args.type, args.yahoo)] if args.symbol else []
    jobs += [(t, "", "Stock", "") for t in split_bulk(args.bulk)]
    added, failed = 0, 0
    for s, name, typ, yahoo in jobs:
        ok, msg = add_one(s, name, typ, yahoo, check=not args.no_check)
        if ok:
            added += 1
            print(msg)
        else:
            failed += 1
            print(f"::warning::{s} not added: {msg}")           # shows as a yellow warning in GitHub Actions
            print(f"SKIPPED {s}: {msg}")
    print(f"{added} added, {failed} skipped")
    return 0 if added or not failed else 1


if __name__ == "__main__":
    sys.exit(main())
