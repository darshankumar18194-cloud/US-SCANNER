"""Read and write shared-data/symbols.csv - the single source of truth for what gets scanned.

Columns:  Symbol, Name, Type, Yahoo Symbol, Active, Notes
  Symbol        your own label (CoinSwitch name)
  Yahoo Symbol  real Yahoo Finance ticker; leave blank when it is the same as Symbol
  Active        Y / N  - N rows are kept in the file but never scanned
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from typing import List, Tuple

import config

HEADER = ["Symbol", "Name", "Type", "Yahoo Symbol", "Active", "Notes"]
_TRUE = {"", "y", "yes", "true", "1", "t"}
_FALSE = {"n", "no", "false", "0", "f"}


@dataclass
class SymbolRow:
    symbol: str
    name: str
    type: str
    yahoo: str
    active: bool
    notes: str = ""


def clean_symbol(s) -> str:
    return str(s or "").strip().upper().replace(" ", "")


def default_yahoo(symbol: str) -> str:
    """US share classes use a dash on Yahoo (BRK.B -> BRK-B)."""
    return symbol.replace(".", "-") if "." in symbol else symbol


def _get(row: dict, *names: str) -> str:
    low = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    for n in names:
        if n.lower() in low and low[n.lower()] is not None:
            return str(low[n.lower()]).strip()
    return ""


def load_symbols(path=None) -> Tuple[List[SymbolRow], List[str]]:
    """Return (rows, problems). Never raises on a bad row - it reports it and carries on."""
    path = path or config.SYMBOLS_CSV
    problems: List[str] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - create it (see README)")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "symbol" not in [h.strip().lower() for h in reader.fieldnames]:
            raise ValueError(f"{path} must have a header row containing 'Symbol'")
        rows: List[SymbolRow] = []
        seen = {}
        for lineno, raw in enumerate(reader, start=2):
            sym = clean_symbol(_get(raw, "Symbol"))
            if not sym:
                continue
            if sym in seen:
                problems.append(f"line {lineno}: duplicate symbol {sym} ignored (first one on line {seen[sym]} is used)")
                continue
            seen[sym] = lineno
            active_txt = _get(raw, "Active").lower()
            if active_txt in _TRUE:
                active = True
            elif active_txt in _FALSE:
                active = False
            else:
                problems.append(f"line {lineno}: {sym} has Active='{active_txt}' - treated as Y")
                active = True
            typ = _get(raw, "Type", "Type (best guess)") or "Stock"
            typ = "ETF" if typ.strip().upper() == "ETF" else "Stock"
            yahoo = _get(raw, "Yahoo Symbol", "Yahoo", "Ticker").upper().replace(" ", "") or default_yahoo(sym)
            rows.append(SymbolRow(sym, _get(raw, "Name") or sym, typ, yahoo, active, _get(raw, "Notes")))
    return rows, problems


def append_symbol(symbol: str, name: str = "", type_: str = "Stock", yahoo: str = "",
                  notes: str = "", path=None) -> SymbolRow:
    """Append one row (creating the file if needed). Raises ValueError for a duplicate."""
    path = path or config.SYMBOLS_CSV
    sym = clean_symbol(symbol)
    if not sym:
        raise ValueError("empty symbol")
    if os.path.exists(path):
        existing, _ = load_symbols(path)
        if any(r.symbol == sym for r in existing):
            raise ValueError(f"{sym} is already in {os.path.basename(str(path))}")
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    typ = "ETF" if str(type_).strip().upper() == "ETF" else "Stock"
    ysym = clean_symbol(yahoo) if yahoo else ""
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(
        [sym, name or sym, typ, ysym if ysym and ysym != sym else "", "Y", notes])
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    needs_nl = False
    if not new_file:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_nl = f.read(1) not in (b"\n", b"\r")
    with open(path, "a", encoding="utf-8", newline="") as f:
        if new_file:
            f.write(",".join(HEADER) + "\n")
        elif needs_nl:
            f.write("\n")
        f.write(buf.getvalue())
    return SymbolRow(sym, name or sym, typ, ysym or default_yahoo(sym), True, notes)
