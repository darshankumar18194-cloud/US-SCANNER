"""Backtest the three exit rules on the history the scanner already downloads (about two years of 1H candles).

    python backtest.py --shard 0 --shards 8 --out shard_0.json      # replay one eighth of the symbols
    python backtest.py --merge shard_*.json --out shared-data/backtest.json     # combine the shards into the summary

How it works - there is no second copy of the rules:
  * for every symbol it calls the REAL scan.analyse_frame on the history as it looked at an earlier moment
    (candles 0..k only), every `step` candles, starting at candle `start` so the levels have a long past to work with;
  * a plan is frozen the first time it is seen, exactly like the live scanner (same id, same levels, same rules);
  * every frozen plan is then followed to the END of the data with plans.evaluate (stop first, half off at TP1, ...).

The fair test is a SHUFFLED-PRICES BASELINE. Each symbol is replayed twice: on its real candles and on the same candles in
random order (each candle keeps its own shape and volume, the order is destroyed, so volatility stays but every real pattern,
trend and support / resistance is gone). The same rules run on both. A rule has an edge only if its real result is better than
its own shuffled result, by clearly more than the error margin. (Comparing different rules with each other proves nothing:
they have different stop / target shapes, which changes the result even on pure noise.)

Honest limits (they are printed into backtest.json too):
  * a plan is made at the first replay step that sees the setup, up to step-1 candles after the entry candle, so its levels
    know a few candles that the entry did not. The live scanner has the same delay but only one candle;
  * Yahoo gives ~2 years of hourly history: one market regime, today's symbol list (survivorship), no costs or slippage;
  * neighbouring setups overlap in time and are correlated, so the error margin is optimistic;
  * plans that are still running when the data ends are counted as open, not as results.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config
import market_data
import plans as plans_mod
import scan
import symbols as sym
from common import atomic_write_text, dumps_strict, get_logger

log = get_logger("backtest")

START_BARS = 1000          # first replay point (about 6 months of candles before the first plan)
STEP = 4                   # replay every N candles; plans live 10 candles, so nothing is missed
FINISHED = ("sl", "be", "tp2", "expired")
KEEP_FIELDS = ("symbol", "side", "pattern", "variant", "entry_time", "rr1", "r", "status", "grade_sl", "sl_basis", "id")


# ----------------------------------------------------------------------------- the shuffled baseline
def shuffle_frame(df, seed: int = 0):
    """Same candles, random order. Each candle keeps its shape relative to the previous close (open, high, low, close) and its
    volume; the new price path is rebuilt from them, so the volatility is the same and all real structure is destroyed."""
    import numpy as np
    n = len(df)
    pc = df["Close"].shift(1).bfill().to_numpy(float)
    rel = np.column_stack([df[c].to_numpy(float) / pc for c in ("Open", "High", "Low", "Close")])
    vol = df["Volume"].to_numpy(float)
    order = np.random.default_rng(seed).permutation(n)
    rel, vol = rel[order], vol[order]
    close = np.empty(n)
    prev = float(df["Close"].iloc[0])
    for i in range(n):
        close[i] = prev * rel[i, 3]
        prev = close[i]
    base = np.r_[float(df["Close"].iloc[0]), close[:-1]]
    out = df.copy()
    out["Open"], out["High"], out["Low"], out["Close"] = base * rel[:, 0], base * rel[:, 1], base * rel[:, 2], close
    out["Volume"] = vol
    return out


# ----------------------------------------------------------------------------- replay one symbol
def replay_symbol(row: sym.SymbolRow, df, tz: str = config.MARKET_TZ_NAME, step: int = STEP,
                  start: int = START_BARS) -> List[Dict[str, Any]]:
    """All plans the scanner would have made on this symbol's history, followed to the end of the data."""
    n = len(df)
    found: Dict[str, Dict[str, Any]] = {}
    labels = df["Date"].dt.strftime(config.DATE_FMT).to_numpy()
    for k in range(start, n - 1, step):
        sub = df.iloc[:k + 1].reset_index(drop=True)
        rec = scan.analyse_frame(row, market_data.Fetch(status="OK", df=sub, tz=tz), labels[k])
        for p in rec.get("plan_new", []):
            found.setdefault(p["id"], p)                           # frozen on first sight, like the live scanner
    out = []
    for p in found.values():
        upd = plans_mod.evaluate(df, p)
        if upd is None:
            continue
        p.update(upd)
        out.append({k: p.get(k) for k in KEEP_FIELDS})
    return out


def _work(args):
    row, step, start = args
    t0 = time.time()
    f = market_data.fetch_history(row.yahoo)
    if f.status != "OK" or f.df is None or len(f.df) < start + 50:
        return row.symbol, None, f"{f.status} {len(f.df) if f.df is not None else 0} candles"
    try:
        res = replay_symbol(row, f.df, f.tz, step, start)
        null = replay_symbol(row, shuffle_frame(f.df, seed=zlib.crc32(row.symbol.encode()) % 100000), f.tz, step, start)
    except Exception as e:                                          # noqa: BLE001 - one bad symbol must not stop the shard
        return row.symbol, None, f"{type(e).__name__}: {e}"
    return row.symbol, {"real": res, "null": null}, f"{len(res)} real / {len(null)} shuffled plans in {time.time() - t0:.0f}s"


# ----------------------------------------------------------------------------- statistics
def stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fin = [p for p in rows if p.get("status") in FINISHED and p.get("r") is not None]
    n = len(fin)
    out: Dict[str, Any] = {"n": n, "open": len(rows) - n, "avg_r": None, "se": None, "win": None,
                           "sl": 0, "be": 0, "tp2": 0, "expired": 0}
    for s in ("sl", "be", "tp2", "expired"):
        out[s] = sum(1 for p in fin if p["status"] == s)
    if n:
        avg = sum(p["r"] for p in fin) / n
        out["avg_r"] = round(avg, 3)
        out["win"] = round(sum(1 for p in fin if p["r"] > 0) / n, 3)
        if n > 1:
            sd = math.sqrt(sum((p["r"] - avg) ** 2 for p in fin) / (n - 1))
            out["se"] = round(sd / math.sqrt(n), 3)
    return out


def edge(real: Dict[str, Any], null: Dict[str, Any]) -> Dict[str, Any]:
    """Real average R minus the same rule's average R on shuffled prices, with the combined error margin."""
    if real["avg_r"] is None or null["avg_r"] is None:
        return {"diff": None, "se": None, "z": None}
    diff = real["avg_r"] - null["avg_r"]
    se = math.sqrt((real["se"] or 0) ** 2 + (null["se"] or 0) ** 2) or None
    return {"diff": round(diff, 3), "se": None if se is None else round(se, 3), "z": None if not se else round(diff / se, 1)}


def summarize(rows: List[Dict[str, Any]], meta: Dict[str, Any], null_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    rules = {v: [p for p in rows if p["variant"] == v] for v in plans_mod.VARIANTS}
    lv = rules["levels"]
    bucket = lambda p: "below 1" if p["rr1"] < 1 else "1 to 1.5" if p["rr1"] < 1.5 else "1.5 or more"
    entries = sorted(p["entry_time"] for p in rows) or [None]
    top = {}
    for p in lv:
        top.setdefault(p["pattern"], []).append(p)
    return {
        "schema": 1, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **meta,
        "first_entry": entries[0], "last_entry": entries[-1], "setups": len(lv) or len(rules["atr"]),
        "rules": {v: stats(r) for v, r in rules.items()},
        "shuffled": {v: stats([p for p in (null_rows or []) if p["variant"] == v]) for v in plans_mod.VARIANTS},
        "edge": {v: edge(stats(r), stats([p for p in (null_rows or []) if p["variant"] == v])) for v, r in rules.items()},
        "by_side": {v: {s: stats([p for p in r if p["side"] == s]) for s in ("long", "short")} for v, r in rules.items()},
        "levels_by_rr": {b: stats([p for p in lv if bucket(p) == b]) for b in ("below 1", "1 to 1.5", "1.5 or more")},
        "levels_by_sl_grade": {g: stats([p for p in lv if p.get("grade_sl") == g]) for g in ("A", "B", "C", "D")},
        "levels_by_pattern": {k: stats(v) for k, v in sorted(top.items(), key=lambda kv: -len(kv[1])) if len(v) >= 10},
        "limits": ["about 2 years of hourly history: one market regime", "today's symbol list (survivorship)",
                   "no costs or slippage", "overlapping setups are correlated: the error margins are optimistic",
                   "a plan is made up to `step`-1 candles after its entry candle",
                   "the baseline is ONE random reordering of each symbol: it shows what no-structure prices give, not a full distribution"],
    }


# ----------------------------------------------------------------------------- command line
def run_shard(args) -> int:
    rows, problems = sym.load_symbols(args.csv)
    rows = [r for r in rows if r.active][args.shard::args.shards]
    if args.limit:
        rows = rows[:args.limit]
    log.info(f"shard {args.shard}/{args.shards}: {len(rows)} symbols, step {args.step}, start {args.start}")
    done, failed, all_plans, null_plans = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_work, (r, args.step, args.start)): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            s, res, note = fut.result()
            log.info(f"  {i}/{len(rows)} {s:<12} {note}")
            (done if res is not None else failed).append(s)
            all_plans += (res or {}).get("real", [])
            null_plans += (res or {}).get("null", [])
    atomic_write_text(args.out, dumps_strict({"shard": args.shard, "shards": args.shards, "step": args.step,
                                              "start": args.start, "done": done, "failed": failed, "plans": all_plans,
                                              "null_plans": null_plans}, False))
    log.info(f"shard {args.shard}: {len(done)} symbols done, {len(failed)} skipped, {len(all_plans)} plans")
    return 0


def run_merge(args) -> int:
    files = sorted(f for pat in args.merge for f in glob.glob(pat))
    if not files:
        log.error("no shard files found")
        return 2
    rows, null_rows, done, failed, step, start = [], [], 0, 0, None, None
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        rows += d["plans"]
        null_rows += d.get("null_plans", [])
        done += len(d["done"])
        failed += len(d["failed"])
        step, start = d.get("step"), d.get("start")
    out = summarize(rows, {"symbols_done": done, "symbols_skipped": failed, "step": step, "start_bars": start,
                           "shards": len(files)}, null_rows)
    atomic_write_text(args.out, dumps_strict(out, True))
    r = out["rules"]
    log.info(f"{len(files)} shards, {done} symbols, {out['setups']} setups | " +
             " | ".join(f"{v}: n={r[v]['n']} avg R {r[v]['avg_r']} (shuffled {out['shuffled'][v]['avg_r']}, edge {out['edge'][v]['diff']})" for v in r))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--start", type=int, default=START_BARS)
    ap.add_argument("--limit", type=int, default=0, help="only the first N symbols of the shard (quick test)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default="backtest_shard.json")
    ap.add_argument("--merge", nargs="+", help="shard files to combine into the summary")
    args = ap.parse_args(argv)
    return run_merge(args) if args.merge else run_shard(args)


if __name__ == "__main__":
    sys.exit(main())
