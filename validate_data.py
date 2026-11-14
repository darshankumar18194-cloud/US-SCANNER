"""Sanity-check shared-data/stocks.json BEFORE it is committed. Exit code 1 = do not publish.

    python validate_data.py            # check stocks.json against symbols.csv
    python validate_data.py --strict   # also fail on warnings
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import List

import config
import symbols as sym

VALID_STATUS = {"OK", "STALE", "NO_DATA", "ERROR", "INSUFFICIENT_DATA", "INACTIVE", "PENDING_SCAN"}
DIRS = {"bull", "bear", "neutral"}


def _bad_constant(name):
    raise ValueError(f"stocks.json contains the illegal JSON token {name} (the browser would refuse to load it)")


def validate(path=None, csv_path=None) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    warns: List[str] = []
    path = path or config.STOCKS_JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f, parse_constant=_bad_constant)
    except FileNotFoundError:
        return [f"{path} does not exist"], warns
    except (ValueError, json.JSONDecodeError) as e:
        return [f"cannot read stocks.json: {e}"], warns
    if not isinstance(data, dict) or not isinstance(data.get("stocks"), list) or not isinstance(data.get("meta"), dict):
        return ["stocks.json must be an object with 'meta' and 'stocks'"], warns
    meta, stocks = data["meta"], data["stocks"]
    for k in ("generated_at", "run_type", "counts", "total_symbols"):
        if k not in meta:
            errors.append(f"meta.{k} missing")

    seen = set()
    for s in stocks:
        name = s.get("symbol") or "?"
        if not s.get("symbol"):
            errors.append("a record has no symbol")
            continue
        if name in seen:
            errors.append(f"{name}: duplicate record")
        seen.add(name)
        st = s.get("scanner_status")
        if st not in VALID_STATUS:
            errors.append(f"{name}: unknown scanner_status {st!r}")
            continue
        if st in ("OK", "STALE"):
            for k in ("price", "smc", "signals", "context", "scores", "candles", "last_bar_time"):
                if s.get(k) is None:
                    errors.append(f"{name}: {st} record is missing '{k}'")
            if not isinstance(s.get("price"), (int, float)) or (isinstance(s.get("price"), float) and not math.isfinite(s["price"])) or (s.get("price") or 0) <= 0:
                errors.append(f"{name}: bad price {s.get('price')!r}")
            for g in s.get("signals") or []:
                if g.get("dir") not in DIRS or not isinstance(g.get("since"), int) or g["since"] < 0 \
                        or g["since"] >= config.STORE_BARS or not (0 <= g.get("score", -1) <= 100) \
                        or not g.get("name") or not g.get("time"):
                    errors.append(f"{name}: malformed signal {g!r}"[:200])
                    break
            if len(s.get("candles") or []) < 5:
                errors.append(f"{name}: too few candles saved")
        if st == "STALE":
            warns.append(f"{name}: STALE ({s.get('scan_error', '')})")

    counts = meta.get("counts") or {}
    if sum(counts.values()) != len(stocks):
        errors.append(f"meta.counts adds up to {sum(counts.values())} but there are {len(stocks)} records")
    try:
        rows, _ = sym.load_symbols(csv_path)
        want = {r.symbol for r in rows}
        missing = want - seen
        extra = seen - want
        if missing:
            errors.append(f"{len(missing)} symbols.csv rows are missing from stocks.json: {', '.join(sorted(missing)[:8])}")
        if extra:
            warns.append(f"{len(extra)} records are not in symbols.csv: {', '.join(sorted(extra)[:8])}")
        active = [r for r in rows if r.active]
    except Exception as e:                                          # noqa: BLE001
        warns.append(f"could not compare with symbols.csv: {e}")
        active = []
    n_active = max(1, len(active) or meta.get("active_symbols", 1))
    bad = counts.get("STALE", 0) + counts.get("ERROR", 0) + counts.get("NO_DATA", 0)
    ok = counts.get("OK", 0)
    if ok == 0 and n_active > 0 and meta.get("total_symbols", 0) > 0:
        errors.append("not a single symbol scanned OK - Yahoo is probably unreachable; refusing to publish")
    elif bad / n_active > config.STALE_WARN_FRACTION:
        warns.append(f"{bad} of {n_active} active symbols are STALE / ERROR / NO_DATA "
                     f"(> {int(config.STALE_WARN_FRACTION * 100)}%)")
    return errors, warns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    errors, warns = validate()
    for w in warns:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR:   {e}")
    if errors or (args.strict and warns):
        print(f"validation FAILED ({len(errors)} errors, {len(warns)} warnings)")
        return 1
    print(f"validation passed ({len(warns)} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
