import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

import add_stock
import config
import market_data
import scan
import symbols as sym
import validate_data
from tests.helpers import frame, random_walk

CSV = """Symbol,Name,Type,Yahoo Symbol,Active,Notes
GOOD,Good Co,Stock,,Y,
ETFX,Some ETF,ETF,,Y,
SAMSUNG,Samsung,Stock,005930.KS,Y,
AMDSTOCK,AMD,Stock,AMD,Y,
GONE,Delisted Inc,Stock,,Y,
BOOM,Flaky Inc,Stock,,Y,
NEWBIE,Just Listed,Stock,,Y,
PRIV,Private Co,Stock,,N,Private company
"""
AFTER_CLOSE = datetime(2026, 10, 3, 0, 2, tzinfo=timezone.utc)          # Fri 20:02 ET
MID_SESSION = datetime(2026, 10, 2, 17, 30, tzinfo=timezone.utc)        # Fri 13:30 EDT


def yf_like(df: pd.DataFrame, tz="America/New_York") -> pd.DataFrame:
    """Shape a frame the way yfinance returns it: tz-aware index, capitalised columns."""
    out = df.set_index(pd.DatetimeIndex(df["Date"]).tz_localize(tz))[["Open", "High", "Low", "Close", "Volume"]]
    out.index.name = "Datetime"
    return out


class Env(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "shared-data").mkdir()
        (self.tmp / "shared-data" / "symbols.csv").write_text(CSV, encoding="utf-8")
        self.patches = [
            mock.patch.object(config, "DATA_DIR", self.tmp / "shared-data"),
            mock.patch.object(config, "STOCKS_JSON", self.tmp / "shared-data" / "stocks.json"),
            mock.patch.object(config, "SYMBOLS_CSV", self.tmp / "shared-data" / "symbols.csv"),
            mock.patch.object(config, "PIPELINE_LOCK", self.tmp / "shared-data" / ".lock"),
            mock.patch.object(config, "LOG_DIR", self.tmp / "logs"),
            mock.patch.object(config, "RETRY_BASE", 0.0),
        ]
        for p in self.patches:
            p.start()
        self.calls = []
        self.mode = {}                    # ticker -> "fail" to force an error on a later run

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake_history(self, ticker, interval, start):
        self.calls.append(ticker)
        if ticker == "GONE":
            raise Exception("No data found, symbol may be delisted")
        if ticker == "BOOM" or self.mode.get(ticker) == "fail":
            raise RuntimeError("Yahoo exploded")
        if ticker == "NEWBIE":
            return yf_like(random_walk(50, seed=9))
        seeds = {"GOOD": 1, "ETFX": 2, "AMD": 3}
        if ticker == "005930.KS":
            idx = pd.date_range("2026-08-03 09:00", periods=1, freq="h")          # replaced below
            from tests.helpers import SLOTS
            days = pd.bdate_range(end="2026-10-02", periods=140)
            ts = [d + pd.Timedelta(hours=h) for d in days for h in range(9, 16)]   # 9:00..15:00 KST candles
            df = random_walk(len(ts), seed=7)
            df["Date"] = pd.DatetimeIndex(ts)
            return yf_like(df, "Asia/Seoul")
        return yf_like(random_walk(900, seed=seeds.get(ticker, 5), vol=0.008))

    def run_scan(self, argv=()):
        with mock.patch.object(market_data, "_history", self.fake_history), \
             mock.patch("scan.run_type_for", return_value="after-close"):
            return scan.main(list(argv))

    def store(self):
        return json.loads((self.tmp / "shared-data" / "stocks.json").read_text(encoding="utf-8"),
                          parse_constant=lambda c: self.fail(f"illegal JSON token {c}"))

    def rec(self, symbol):
        return {s["symbol"]: s for s in self.store()["stocks"]}[symbol]


class TestScanRun(Env):
    def test_end_to_end(self):
        self.assertEqual(self.run_scan(), 0)
        data = self.store()
        meta, by = data["meta"], {s["symbol"]: s for s in data["stocks"]}
        self.assertEqual(len(by), 8)
        self.assertEqual(meta["total_symbols"], 8)
        self.assertEqual(meta["active_symbols"], 7)
        self.assertEqual([s["symbol"] for s in data["stocks"]],
                         ["GOOD", "ETFX", "SAMSUNG", "AMDSTOCK", "GONE", "BOOM", "NEWBIE", "PRIV"])   # CSV order
        status = {k: v["scanner_status"] for k, v in by.items()}
        self.assertEqual(status, {"GOOD": "OK", "ETFX": "OK", "SAMSUNG": "OK", "AMDSTOCK": "OK", "GONE": "NO_DATA",
                                  "BOOM": "ERROR", "NEWBIE": "INSUFFICIENT_DATA", "PRIV": "INACTIVE"})
        self.assertEqual(sum(meta["counts"].values()), 8)
        self.assertNotIn("PRIV", self.calls)                                   # inactive rows are never downloaded
        self.assertIn("AMD", self.calls)                                       # Yahoo Symbol column is honoured
        self.assertNotIn("AMDSTOCK", self.calls)

    def test_ok_record_has_everything_the_dashboard_reads(self):
        self.run_scan()
        r = self.rec("GOOD")
        for k in ("price", "change_pct", "smc", "signals", "context", "scores", "candles", "last_bar_time",
                  "scanned_at", "tradingview_url", "yahoo_symbol", "type", "name", "bars"):
            self.assertIn(k, r)
        self.assertEqual(len(r["candles"]), config.RECENT_CANDLES)
        self.assertEqual(r["candles"][-1][0], r["last_bar_time"])
        self.assertIn(r["smc"]["trend"], ("BULLISH", "BEARISH", "UNKNOWN"))
        self.assertEqual(r["type"], "Stock")
        self.assertEqual(self.rec("ETFX")["type"], "ETF")
        self.assertAlmostEqual(r["price"], r["candles"][-1][4], places=3)

    def test_smc_fields_are_real_numbers_or_null_never_text(self):
        self.run_scan()
        for s in self.store()["stocks"]:
            if s["scanner_status"] != "OK":
                continue
            m = s["smc"]
            for k in ("top_signal_price", "bottom_signal_price", "active_swing_high", "active_swing_low"):
                self.assertTrue(m[k] is None or isinstance(m[k], (int, float)), f"{s['symbol']} {k}={m[k]!r}")
            self.assertNotIn("NA", m.values())
            self.assertIn(m["trend"], ("BULLISH", "BEARISH", "UNKNOWN"))
            self.assertIsInstance(m["structure_bars_since"], int)

    def test_clean_smc(self):
        raw = {k: "NA" for k in scan.SMC_KEYS}
        raw.update(active_swing_low="96.78", top_bars_since=0, trend="BULLISH")
        out = scan.clean_smc(raw)
        self.assertIsNone(out["active_swing_high"])
        self.assertEqual(out["active_swing_low"], 96.78)
        self.assertEqual(out["top_bars_since"], 0)
        self.assertEqual(out["trend"], "BULLISH")

    def test_meta_block(self):
        self.run_scan()
        m = self.store()["meta"]
        self.assertEqual(m["run_type"], "after-close")
        self.assertEqual(m["window_options"], config.WINDOW_OPTIONS)
        self.assertTrue(m["generated_at"].endswith("Z"))
        self.assertIn("IST", m["generated_ist"])
        self.assertEqual(m["latest_candle"], max(s["last_bar_time"] for s in self.store()["stocks"] if s["scanner_status"] == "OK"))

    def test_korean_symbol_keeps_exchange_local_time(self):
        self.run_scan()
        r = self.rec("SAMSUNG")
        self.assertEqual(r["scanner_status"], "OK")
        self.assertEqual(r["tz"], "Asia/Seoul")
        self.assertTrue(r["last_bar_time"].endswith("15:00"))
        self.assertIn("KRX", r["tradingview_url"])

    def test_failure_keeps_previous_analysis_and_marks_stale(self):
        self.run_scan()
        before = self.rec("GOOD")
        self.mode["GOOD"] = "fail"
        self.run_scan()
        after = self.rec("GOOD")
        self.assertEqual(after["scanner_status"], "STALE")
        self.assertIn("Yahoo exploded", after["scan_error"])
        self.assertEqual(after["signals"], before["signals"])                  # old analysis kept
        self.assertEqual(after["price"], before["price"])
        self.mode.clear()
        self.run_scan()
        self.assertEqual(self.rec("GOOD")["scanner_status"], "OK")             # recovers by itself
        self.assertNotIn("scan_error", self.rec("GOOD"))

    def test_symbol_that_never_worked_is_not_stale(self):
        self.run_scan()
        self.run_scan()
        self.assertEqual(self.rec("GONE")["scanner_status"], "NO_DATA")
        self.assertEqual(self.rec("BOOM")["scanner_status"], "ERROR")

    def test_partial_run_leaves_everything_else_untouched(self):
        self.run_scan()
        before = {s["symbol"]: s for s in self.store()["stocks"]}
        self.calls.clear()
        self.run_scan(["--symbols", "GOOD"])
        after = {s["symbol"]: s for s in self.store()["stocks"]}
        self.assertEqual(set(self.calls), {"GOOD"})
        self.assertEqual(after["ETFX"], before["ETFX"])
        self.assertEqual(after["PRIV"]["scanner_status"], "INACTIVE")

    def test_unknown_symbol_argument_is_an_error(self):
        self.assertEqual(self.run_scan(["--symbols", "NOPE"]), 2)

    def test_new_csv_row_appears_on_next_run_and_removed_row_disappears(self):
        self.run_scan()
        sym.append_symbol("NVDA", "Nvidia", path=config.SYMBOLS_CSV)
        self.run_scan()
        self.assertEqual(self.rec("NVDA")["scanner_status"], "OK")
        text = config.SYMBOLS_CSV.read_text(encoding="utf-8").replace("ETFX,Some ETF,ETF,,Y,\n", "")
        config.SYMBOLS_CSV.write_text(text, encoding="utf-8")
        self.run_scan()
        self.assertNotIn("ETFX", {s["symbol"] for s in self.store()["stocks"]})

    def test_output_passes_validation(self):
        self.run_scan()
        errors, warns = validate_data.validate()
        self.assertEqual(errors, [])

    def test_validation_catches_corruption(self):
        self.run_scan()
        p = config.STOCKS_JSON
        txt = p.read_text(encoding="utf-8")
        p.write_text(txt.replace('"price":', '"price":NaN,"x":', 1), encoding="utf-8")
        errors, _ = validate_data.validate()
        self.assertTrue(any("illegal JSON token" in e or "cannot read" in e for e in errors))
        # all symbols failing -> refuse to publish
        d = json.loads(txt)
        for s in d["stocks"]:
            if s["scanner_status"] == "OK":
                s["scanner_status"] = "ERROR"
        d["meta"]["counts"] = {"ERROR": 7, "INACTIVE": 1}
        p.write_text(json.dumps(d), encoding="utf-8")
        errors, _ = validate_data.validate()
        self.assertTrue(any("not a single symbol" in e for e in errors))

    def test_lock_prevents_two_runs(self):
        config.PIPELINE_LOCK.write_text("x")
        self.assertEqual(self.run_scan(), 2)


class TestYahooRangeLimit(unittest.TestCase):
    def test_lookback_stays_inside_yahoos_730_day_limit_for_every_time_zone(self):
        # start = local-midnight DATE of (now - lookback) in New York; the same date at midnight in UTC+14 is 14h earlier
        worst_case_days = config.HOURLY_LOOKBACK_DAYS + 1 + 14 / 24
        self.assertLess(worst_case_days, 730)

    def test_the_requested_start_is_a_plain_date(self):
        seen = {}
        def fake(ticker, interval, start):
            seen["start"] = start
            return yf_like(random_walk(300, seed=1))
        market_data.fetch_history("X", history_fn=fake, now_utc=AFTER_CLOSE)
        self.assertRegex(seen["start"], r"^\d{4}-\d\d-\d\d$")


class TestClosedCandleRule(Env):
    def test_forming_candle_is_dropped_mid_session(self):
        df = random_walk(900, seed=1, vol=0.008, last="2026-10-02 13:30")           # last candle STARTS 13:30
        row = sym.SymbolRow("GOOD", "Good", "Stock", "GOOD", True)
        rec = scan.scan_symbol(row, None, "x", history_fn=lambda *a: yf_like(df), now_utc=MID_SESSION)
        self.assertEqual(rec["last_bar_time"], "2026-10-02 12:30")                   # 13:30 candle still forming
        rec = scan.scan_symbol(row, None, "x", history_fn=lambda *a: yf_like(df), now_utc=AFTER_CLOSE)
        self.assertEqual(rec["last_bar_time"], "2026-10-02 13:30")                   # closed by then

    def test_half_hour_closing_candle_counts_as_closed_at_16_00(self):
        df = random_walk(900, seed=1, vol=0.008, last="2026-10-02 15:30")
        row = sym.SymbolRow("GOOD", "Good", "Stock", "GOOD", True)
        at = lambda h, m: datetime(2026, 10, 2, h, m, tzinfo=timezone.utc)
        early = scan.scan_symbol(row, None, "x", history_fn=lambda *a: yf_like(df), now_utc=at(19, 59))
        late = scan.scan_symbol(row, None, "x", history_fn=lambda *a: yf_like(df), now_utc=at(20, 0))
        self.assertEqual(early["last_bar_time"], "2026-10-02 14:30")
        self.assertEqual(late["last_bar_time"], "2026-10-02 15:30")

    def test_signal_never_comes_from_the_forming_candle(self):
        df = random_walk(900, seed=4, vol=0.008, last="2026-10-02 13:30")
        row = sym.SymbolRow("GOOD", "Good", "Stock", "GOOD", True)
        rec = scan.scan_symbol(row, None, "x", history_fn=lambda *a: yf_like(df), now_utc=MID_SESSION)
        for s in rec["signals"]:
            self.assertLessEqual(s["time"], "2026-10-02 12:30")

    def test_only_candle_is_forming_gives_no_data_not_a_crash(self):
        df = random_walk(1, seed=1, last="2026-10-02 13:30")
        r = market_data.fetch_history("X", history_fn=lambda *a: yf_like(df), now_utc=MID_SESSION)
        self.assertEqual(r.status, "NO_DATA")


class TestSymbolsAndAddStock(Env):
    def test_loader_defaults_and_problems(self):
        p = self.tmp / "s.csv"
        p.write_text("Symbol,Name,Type (best guess),Yahoo Symbol,Active\nbrk.b,Berkshire,Stock,,\nAAPL,Apple,ETF,,maybe\nAAPL,dup,Stock,,Y\n,blank,Stock,,Y\n", encoding="utf-8")
        rows, problems = sym.load_symbols(p)
        self.assertEqual([r.symbol for r in rows], ["BRK.B", "AAPL"])
        self.assertEqual(rows[0].yahoo, "BRK-B")                     # Yahoo wants a dash
        self.assertTrue(rows[0].active)
        self.assertEqual(rows[1].type, "ETF")
        self.assertEqual(len(problems), 2)                           # bad Active value + duplicate

    def test_real_symbols_csv_is_loadable_and_sane(self):
        rows, problems = sym.load_symbols(Path(__file__).resolve().parents[1] / "shared-data" / "symbols.csv")
        self.assertEqual(problems, [])
        self.assertEqual(len(rows), 241)
        by = {r.symbol: r for r in rows}
        self.assertEqual(by["AMDSTOCK"].yahoo, "AMD")
        self.assertEqual(by["SAMSUNG"].yahoo, "005930.KS")
        self.assertFalse(by["OPENAI"].active)
        self.assertFalse(by["NOW"].active)
        self.assertTrue(by["AAPL"].active)
        self.assertEqual(sum(r.active for r in rows), 228)

    def test_add_stock_validates_with_yahoo_first(self):
        ok, msg = add_stock.add_one("nvda", "Nvidia", history_fn=self.fake_history)
        self.assertTrue(ok, msg)
        self.assertIn("NVDA", [r.symbol for r in sym.load_symbols()[0]])
        ok, msg = add_stock.add_one("NVDA", history_fn=self.fake_history)
        self.assertFalse(ok)
        self.assertIn("already", msg)
        ok, msg = add_stock.add_one("GONE", history_fn=self.fake_history)
        self.assertFalse(ok)
        self.assertNotIn("GONE", [r.symbol for r in sym.load_symbols()[0] if r.symbol == "GONE" and r.name == "GONE"])
        ok, msg = add_stock.add_one("NEWBIE", history_fn=self.fake_history)
        self.assertFalse(ok)
        self.assertIn("too new", msg)
        ok, msg = add_stock.add_one("bad symbol!", history_fn=self.fake_history)
        self.assertFalse(ok)

    def test_add_stock_keeps_existing_rows_intact_and_handles_missing_newline(self):
        before = config.SYMBOLS_CSV.read_text(encoding="utf-8").rstrip("\n")
        config.SYMBOLS_CSV.write_text(before, encoding="utf-8")                 # no trailing newline
        add_stock.add_one("NVDA", history_fn=self.fake_history)
        rows, problems = sym.load_symbols()
        self.assertEqual(problems, [])
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[-1].symbol, "NVDA")

    def test_bulk_split(self):
        self.assertEqual(add_stock.split_bulk("NVDA, PLTR;spy  AAPL"), ["NVDA", "PLTR", "spy", "AAPL"])
        self.assertEqual(add_stock.split_bulk(""), [])

    def test_added_symbol_is_scanned_next_run(self):
        add_stock.add_one("NVDA", "Nvidia", history_fn=self.fake_history)
        self.run_scan()
        self.assertEqual(self.rec("NVDA")["scanner_status"], "OK")


if __name__ == "__main__":
    unittest.main()
