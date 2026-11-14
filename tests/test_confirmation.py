"""Confirmation rule for the pattern tables: patStatus() in assets/common.js, run under Node on real scanner records."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

import market_data as md
import scan
import symbols as sym
from tests.helpers import K, frame, prefix
from tests.test_patterns import CASES

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

RUNNER = r"""
const vm = require('vm'), fs = require('fs');
const ctx = vm.createContext({console});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8') + '\n;globalThis.__api = {patStatus};', ctx);
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(cases.map(c => ctx.__api.patStatus(c.stock, c.sig))));
"""


def record(cs):
    row = sym.SymbolRow("TST", "Test", "Stock", "TST", True)
    return scan.analyse_frame(row, md.Fetch("OK", df=frame(cs), ticker="TST", tz="America/New_York"), "x")


def statuses(pairs):
    out = subprocess.run([NODE, "-e", RUNNER, str(ROOT / "assets" / "common.js")], input=json.dumps(
        [{"stock": s, "sig": g} for s, g in pairs]), capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout)


def pick(rec, name, since):
    return next(g for g in rec["signals"] if g["name"] == name and g["since"] == since)


@unittest.skipUnless(NODE, "node is not installed")
class TestPatStatus(unittest.TestCase):
    def test_candlestick_pattern_on_the_latest_candle_is_pending(self):
        pre, p = prefix("down", n=200)
        rec = record(pre + CASES["Morning Star"][1](p))
        self.assertEqual(statuses([(rec, pick(rec, "Morning Star", 0))]), ["pending"])

    def test_candlestick_pattern_confirmed_by_a_later_close_beyond_it(self):
        pre, p = prefix("down", n=200)
        rec = record(pre + CASES["Morning Star"][1](p) + [K(p - 0.1, p + 0.9)])
        self.assertEqual(statuses([(rec, pick(rec, "Morning Star", 1))]), ["confirmed"])

    def test_candlestick_pattern_that_failed_is_marked_failed(self):
        pre, p = prefix("down", n=200)
        rec = record(pre + CASES["Morning Star"][1](p) + [K(p - 1.0, p - 1.8, 0.05, 0.05)])
        self.assertEqual(statuses([(rec, pick(rec, "Morning Star", 1))]), ["failed"])

    def test_bearish_pattern_is_the_mirror_image(self):
        pre, p = prefix("up", n=200)
        pending = record(pre + CASES["Evening Star"][1](p))
        confirmed = record(pre + CASES["Evening Star"][1](p) + [K(p + 0.1, p - 0.9)])
        self.assertEqual(statuses([(pending, pick(pending, "Evening Star", 0)),
                                   (confirmed, pick(confirmed, "Evening Star", 1))]), ["pending", "confirmed"])

    def _turtle_soup(self, extra):
        pre, p = prefix("flat", n=200)
        lvl = min(c[2] for c in pre[-20:])
        sweep = [K(lvl + 0.05, lvl + 0.6, 0.05, 0.4)]            # sweeps the 20-candle low and closes back above it
        rec = record(pre + sweep + list(extra))
        since = len(extra)
        return rec, pick(rec, "Turtle Soup", since), lvl

    def test_turtle_soup_is_decided_from_the_saved_candles(self):
        rec, g, lvl = self._turtle_soup([])
        self.assertEqual(statuses([(rec, g)]), ["pending"])
        high = rec["candles"][-1][2]
        rec2, g2, _ = self._turtle_soup([K(lvl + 0.55, high + 0.8, 0.05, 0.05)])           # a later close above the signal candle's high
        self.assertEqual(statuses([(rec2, g2)]), ["confirmed"])
        rec3, g3, _ = self._turtle_soup([K(lvl + 0.5, lvl - 0.9, 0.05, 0.05)])             # a later close below its low
        self.assertEqual(statuses([(rec3, g3)]), ["failed"])

    def test_breakout_events_are_triggers_not_pending(self):
        from tests.test_signals import quiet_then_breakout
        import config
        from tests.test_scan import yf_like
        rec = scan.analyse_frame(sym.SymbolRow("Q", "Q", "Stock", "Q", True),
                                 md.Fetch("OK", df=quiet_then_breakout(up=True), ticker="Q", tz="America/New_York"), "x")
        sigs = [g for g in rec["signals"] if g["name"] in ("TTM Squeeze Fired", "Volume Spike Breakout", "ATR Expansion Breakout") and g["since"] == 0]
        self.assertGreaterEqual(len(sigs), 2)
        self.assertEqual(set(statuses([(rec, g) for g in sigs])), {"trigger"})


if __name__ == "__main__":
    unittest.main()
