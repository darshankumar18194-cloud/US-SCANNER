"""Scan every ACTIVE symbol in shared-data/symbols.csv on closed 1-hour candles and write shared-data/stocks.json.

    python scan.py                       # the normal run (what GitHub Actions does)
    python scan.py --symbols NVDA AAPL   # only these; every other symbol keeps its previous result
    python scan.py --limit 20            # quick test on the first 20 active symbols
    python scan.py --workers 2           # gentler on Yahoo

Rules that protect your data:
  * Only CLOSED candles are analysed - the candle still forming when the scan runs is dropped, so a signal can
    never appear mid-candle and then vanish.
  * A failed download never wipes good data: the previous result is kept and marked STALE.
  * stocks.json is written atomically and only after every symbol has been processed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import numpy as np

import config
import levels as levels_mod
import market_data
import signals
import smc_engine as eng
import symbols as sym
from common import (IST, LockBusy, MARKET_TZ, file_lock, get_logger, is_market_open, load_store, now_market,
                    run_type_for, safe_float, save_store)

log = get_logger("scan")

SMC_KEYS = ["trend", "structure_event", "last_structure_time", "top_signal", "bottom_signal", "top_signal_time",
            "top_signal_price", "bottom_signal_time", "bottom_signal_price", "top_bars_since", "bottom_bars_since",
            "active_swing_high", "active_swing_low", "signal", "signal_strength", "bars_since_signal",
            "last_signal_time", "bearish_rank_score", "bullish_rank_score", "signal_rank_score"]
HAS_ANALYSIS = ("OK", "STALE")
SMC_PRICE_KEYS = {"top_signal_price", "bottom_signal_price", "active_swing_high", "active_swing_low"}


def clean_smc(fields: Dict[str, Any]) -> Dict[str, Any]:
    """The SMC engine writes prices as text and 'NA' for missing. The dashboard wants real numbers and nulls."""
    out: Dict[str, Any] = {}
    for k in SMC_KEYS:
        v = fields[k]
        if v in ("NA", "", None):
            out[k] = None
        elif k in SMC_PRICE_KEYS:
            out[k] = safe_float(v)
        else:
            out[k] = v
    return out


def tradingview_url(yahoo: str) -> str:
    y = yahoo.upper()
    code, _, suffix = y.rpartition(".")
    if suffix in ("KS", "KQ") and code:
        return "https://www.tradingview.com/chart/?symbol=" + quote(f"KRX:{code}", safe="")
    if suffix == "HK" and code.isdigit():
        return "https://www.tradingview.com/chart/?symbol=" + quote(f"HKEX:{int(code)}", safe="")
    if suffix == "SS" and code:
        return "https://www.tradingview.com/chart/?symbol=" + quote(f"SSE:{code}", safe="")
    if suffix == "SZ" and code:
        return "https://www.tradingview.com/chart/?symbol=" + quote(f"SZSE:{code}", safe="")
    return "https://www.tradingview.com/chart/?symbol=" + quote(y.replace("-", "."), safe="")


def base_record(row: sym.SymbolRow) -> Dict[str, Any]:
    return {"symbol": row.symbol, "yahoo_symbol": row.yahoo, "name": row.name, "type": row.type,
            "active": row.active, "notes": row.notes, "tradingview_url": tradingview_url(row.yahoo)}


def analyse_frame(row: sym.SymbolRow, fetch: market_data.Fetch, scanned_at: str) -> Dict[str, Any]:
    """Turn closed candles into one stocks.json record. Raises on a genuine bug (caller catches)."""
    df = fetch.df
    n = len(df)
    rec = base_record(row)
    rec.update({"tz": fetch.tz, "scanned_at": scanned_at, "bars": n,
                "last_bar_time": df["Date"].iloc[-1].strftime(config.DATE_FMT)})
    if n < config.MIN_BARS:
        rec.update({"scanner_status": "INSUFFICIENT_DATA",
                    "scan_error": f"only {n} closed candles, need {config.MIN_BARS}"})
        return rec

    labels = df["Date"].dt.strftime(config.DATE_FMT).to_numpy()
    a = eng.analyze(df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy(), config.SWING_LEN)
    fields = eng.to_fields(a, labels, "", "1H")
    smc = clean_smc(fields)
    smc["structure_bars_since"] = (n - 1 - a.event_bar) if a.event_bar >= 0 else -1
    smc["status"] = fields["scanner_status"]                       # OK | NO_PIVOTS

    sigs, ctx, scores = signals.analyze(df, smc["trend"])

    dates = df["Date"].dt.date
    prev = df["Close"][dates < dates.iloc[-1]]
    price = float(df["Close"].iloc[-1])
    change = (price / float(prev.iloc[-1]) - 1) * 100 if len(prev) else None
    tail = df.tail(config.RECENT_CANDLES)
    candles = [[t.strftime(config.DATE_FMT), round(o, 4), round(h, 4), round(l, 4), round(c, 4), int(v)]
               for t, o, h, l, c, v in zip(tail["Date"], tail["Open"], tail["High"], tail["Low"], tail["Close"], tail["Volume"])]
    rec.update({"scanner_status": "OK", "price": round(price, 4),
                "change_pct": None if change is None else round(change, 2),
                "smc": smc, "signals": sigs, "context": ctx, "scores": scores, "candles": candles})
    if getattr(config, "LEVELS_ON", False):                         # support / resistance levels (levels.py)
        try:
            lv = levels_mod.analyse(df)
            rec["levels"], rec["levels_meta"] = lv["levels"], lv["meta"]
        except Exception as e:                                      # noqa: BLE001 - a level bug must never lose the scan
            log.warning(f"{row.symbol}: levels failed - {type(e).__name__}: {e}")
            rec["levels"], rec["levels_meta"] = [], {"error": f"{type(e).__name__}: {e}"[:120]}
    return rec


def scan_symbol(row: sym.SymbolRow, previous: Optional[Dict[str, Any]], scanned_at: str,
                history_fn=None, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """Download + analyse one symbol. Never raises. Failures keep the last good analysis and mark it STALE."""
    kw: Dict[str, Any] = {}
    if history_fn is not None:
        kw["history_fn"] = history_fn
    if now_utc is not None:
        kw["now_utc"] = now_utc
    err, status = "", "ERROR"
    try:
        fetch = market_data.fetch_history(row.yahoo, **kw)
        if fetch.status == "OK":
            rec = analyse_frame(row, fetch, scanned_at)
            if rec["scanner_status"] == "OK":
                return rec
            status, err = rec["scanner_status"], rec.get("scan_error", "")
        else:
            status, err = fetch.status, fetch.error
    except Exception as e:                                          # noqa: BLE001 - one bad symbol must not stop the run
        status, err = "ERROR", f"{type(e).__name__}: {e}"

    if previous and previous.get("scanner_status") in HAS_ANALYSIS and previous.get("smc"):
        kept = dict(previous)                                       # keep the last good analysis
        kept.update(base_record(row))
        kept.update({"scanner_status": "STALE", "scan_error": f"{status}: {err}"[:200]})
        return kept
    rec = base_record(row)
    rec.update({"scanner_status": status, "scan_error": err[:200], "scanned_at": scanned_at})
    return rec


def inactive_record(row: sym.SymbolRow) -> Dict[str, Any]:
    rec = base_record(row)
    rec["scanner_status"] = "INACTIVE"
    return rec


def run(rows: List[sym.SymbolRow], previous: Dict[str, Dict[str, Any]], only: Optional[set], workers: int,
        history_fn=None, now_utc: Optional[datetime] = None, limit: int = 0) -> Dict[str, Dict[str, Any]]:
    scanned_at = (now_utc or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    targets = [r for r in rows if r.active and (only is None or r.symbol in only)]
    if limit > 0:
        targets = targets[:limit]
    results: Dict[str, Dict[str, Any]] = {}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(scan_symbol, r, previous.get(r.symbol), scanned_at, history_fn, now_utc): r for r in targets}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                results[r.symbol] = fut.result()
            except Exception as e:                                  # noqa: BLE001 - defensive: scan_symbol already catches
                log.warning(f"{r.symbol}: crashed - {type(e).__name__}: {e}")
                results[r.symbol] = {**base_record(r), "scanner_status": "ERROR", "scan_error": str(e)[:200]}
            done += 1
            if done % 50 == 0 or done == len(targets):
                log.info(f"  {done}/{len(targets)} symbols  ({time.time() - t0:.0f}s)")
    return results


def assemble(rows: List[sym.SymbolRow], results: Dict[str, Dict[str, Any]], previous: Dict[str, Dict[str, Any]],
             problems: List[str], now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """Merge fresh results, untouched previous results and inactive rows, in CSV order, plus the meta block."""
    stocks: List[Dict[str, Any]] = []
    for r in rows:
        if not r.active:
            stocks.append(inactive_record(r))
        elif r.symbol in results:
            stocks.append(results[r.symbol])
        elif r.symbol in previous and previous[r.symbol].get("scanner_status") != "INACTIVE":
            kept = dict(previous[r.symbol])                         # partial run: leave other symbols as they were
            kept.update(base_record(r))
            stocks.append(kept)
        else:
            rec = base_record(r)
            rec["scanner_status"] = "PENDING_SCAN"
            stocks.append(rec)
    counts: Dict[str, int] = {}
    for s in stocks:
        counts[s["scanner_status"]] = counts.get(s["scanner_status"], 0) + 1
    ok_times = [s["last_bar_time"] for s in stocks if s["scanner_status"] == "OK" and s.get("last_bar_time")]
    now_utc = now_utc or datetime.now(timezone.utc)
    meta = {
        "schema": 1,
        "generated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_et": now_utc.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "generated_ist": now_utc.astimezone(IST).strftime("%Y-%m-%d %H:%M IST"),
        "run_type": run_type_for(now_utc.astimezone(MARKET_TZ)),
        "market_open_at_run": is_market_open(now_utc.astimezone(MARKET_TZ)),
        "trigger": os.environ.get("GITHUB_EVENT_NAME", "local"),
        "interval": config.INTERVAL, "session": "regular hours only",
        "total_symbols": len(rows), "active_symbols": sum(1 for r in rows if r.active),
        "counts": counts,
        "latest_candle": max(ok_times) if ok_times else None,
        "store_bars": config.STORE_BARS, "fresh_fade_bars": config.FRESH_FADE_BARS,
        "window_options": config.WINDOW_OPTIONS, "top_board_window": config.TOP_BOARD_WINDOW,
        "top_board_size": config.TOP_BOARD_SIZE,
        "csv_problems": problems,
    }
    return {"meta": meta, "stocks": stocks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", help="scan only these CSV symbols (others keep their previous result)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=config.SCAN_WORKERS)
    ap.add_argument("--csv", default=None, help="alternative symbols.csv")
    ap.add_argument("--pretty", action="store_true", help="indent stocks.json (bigger file, easier to read)")
    args = ap.parse_args(argv)
    try:
        with file_lock(config.PIPELINE_LOCK, wait=0):
            rows, problems = sym.load_symbols(args.csv)
            for p in problems:
                log.warning(f"symbols.csv: {p}")
            store = load_store()
            previous = {s.get("symbol"): s for s in store["stocks"]}
            only = {sym.clean_symbol(s) for s in args.symbols} if args.symbols else None
            if only:
                missing = only - {r.symbol for r in rows}
                if missing:
                    log.error(f"not in symbols.csv: {', '.join(sorted(missing))}")
                    return 2
            log.info(f"{now_market():%Y-%m-%d %H:%M %Z} | market open: {is_market_open()} | "
                     f"csv rows: {len(rows)} | active: {sum(r.active for r in rows)}")
            t0 = time.time()
            results = run(rows, previous, only, args.workers, limit=args.limit)
            out = assemble(rows, results, previous, problems)
            save_store(out["meta"], out["stocks"], pretty=args.pretty)
            counts = out["meta"]["counts"]
            log.info(f"Done in {time.time() - t0:.0f}s | " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
            top = sorted((s for s in out["stocks"] if s["scanner_status"] == "OK"),
                         key=lambda s: -s["scores"]["best"])[:8]
            for s in top:
                best = max(s["signals"], key=lambda g: g["score"], default=None)
                if best:
                    log.info(f"  {s['symbol']:<14} score={s['scores']['best']:<3} {best['name']} ({best['dir']}, {best['since']} ago)")
            return 0
    except LockBusy:
        log.error(f"Another run holds {config.PIPELINE_LOCK}. Wait for it, or delete that file if it is left over.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
