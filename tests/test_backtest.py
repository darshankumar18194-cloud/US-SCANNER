"""Backtest (backtest.py): synthetic data with known answers. No network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import backtest
import config
import plans
import scan
import symbols as sym
from tests.test_levels import frame, ranging


def row():
    return sym.SymbolRow(symbol="T", name="t", type="Stock", yahoo="T", active=True, notes="")


def rec_plan(label, variant="levels", entry=100.0, sl=96.0, tp1=104.0, tp2=108.0):
    pid = f"T|long|Hammer|p1" + ("" if variant == "levels" else f"|{variant}")
    return {"id": pid, "variant": variant, "symbol": "T", "side": "long", "pattern": "Hammer", "pattern_time": label,
            "entry_time": label, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr1": 1.0, "rr2": 2.0, "status": "open",
            "tp1_time": None, "tp2_time": None, "closed_time": None, "r": None, "r_live": None, "grade_sl": "A", "sl_basis": "level"}


class TestReplay(unittest.TestCase):
    def test_plan_seen_at_many_steps_is_frozen_once_and_followed_to_the_end(self):
        c = np.full(1100, 100.0)
        c[1060:] = 105.0                                              # price jumps after the entry candle: TP1 and then TP2
        c[1090:] = 109.0
        df = frame(c, wick=0.3)
        label = df["Date"].dt.strftime(config.DATE_FMT).iloc[1050]
        seen = []

        def fake(row_, fetch, scanned_at, open_plans=None):
            seen.append(len(fetch.df))
            p = rec_plan(label)
            if len(fetch.df) > 1060:
                p["tp1"] = 999.0                                      # a LATER sighting must not change the frozen plan
            return {"plan_new": [p]}

        with mock.patch.object(scan, "analyse_frame", side_effect=fake):
            out = backtest.replay_symbol(row(), df, step=10, start=1000)
        self.assertGreater(len(seen), 3)                              # it looked at several moments in time
        self.assertEqual(len(out), 1)                                 # ... but there is exactly one plan
        self.assertEqual(out[0]["status"], "tp2")                     # and it was followed to the end of the data
        self.assertAlmostEqual(out[0]["r"], 0.5 * 1.0 + 0.5 * 2.0)

    def test_every_replay_step_sees_only_the_past(self):
        df = frame(ranging(1100, noise=0.1))
        lengths = []
        with mock.patch.object(scan, "analyse_frame", side_effect=lambda r, f, s, open_plans=None: lengths.append(len(f.df)) or {}):
            backtest.replay_symbol(row(), df, step=25, start=1000)
        self.assertEqual(lengths, list(range(1001, 1100, 25)))        # candles 0..k, growing: never the future

    def test_real_pipeline_runs_end_to_end_on_synthetic_prices(self):
        df = frame(ranging(1150, noise=0.3, seed=5))
        out = backtest.replay_symbol(row(), df, step=40, start=1000)  # real analyse_frame, no mocks
        self.assertIsInstance(out, list)
        for p in out:
            self.assertIn(p["variant"], plans.VARIANTS)


def P(variant, r, status="tp2", side="long", i=0, rr1=1.2, grade="A", pattern="Hammer"):
    pid = f"T|{side}|{pattern}|{i}" + ("" if variant == "levels" else f"|{variant}")
    return {"id": pid, "variant": variant, "symbol": "T", "side": side, "pattern": pattern, "entry_time": f"2026-01-{i % 28 + 1:02d} 10:30",
            "rr1": rr1, "r": r, "status": status, "grade_sl": grade, "sl_basis": "level"}


class TestShuffle(unittest.TestCase):
    def test_shuffled_prices_keep_the_candles_but_lose_the_order(self):
        df = frame(ranging(600, noise=0.2, seed=2), wick=0.3)
        df["Volume"] = np.arange(600, dtype=float)
        sh = backtest.shuffle_frame(df, seed=1)
        self.assertEqual(len(sh), 600)
        self.assertTrue((sh["Date"].to_numpy() == df["Date"].to_numpy()).all())          # same calendar
        self.assertEqual(sorted(sh["Volume"]), sorted(df["Volume"]))                     # same candles ...
        self.assertFalse((sh["Volume"].to_numpy() == df["Volume"].to_numpy()).all())     # ... in a different order
        self.assertTrue((sh["High"] >= sh[["Open", "Close"]].max(axis=1) - 1e-9).all())  # still valid candles
        self.assertTrue((sh["Low"] <= sh[["Open", "Close"]].min(axis=1) + 1e-9).all())
        r_real = np.diff(np.log(df["Close"]))
        r_sh = np.diff(np.log(sh["Close"]))
        self.assertAlmostEqual(r_sh.std(), r_real.std(), delta=r_real.std() * 0.25)      # volatility is kept
        self.assertAlmostEqual(abs(np.corrcoef(sh["Close"].to_numpy()[:-1], df["Close"].to_numpy()[:-1])[0, 1]), 0.0, delta=0.5)


class TestStats(unittest.TestCase):
    def test_stats_known_numbers(self):
        s = backtest.stats([P("levels", -1.0, "sl"), P("levels", 2.0, "tp2", i=1), P("levels", 0.5, "be", i=2), P("levels", None, "open", i=3)])
        self.assertEqual((s["n"], s["open"], s["sl"], s["be"], s["tp2"]), (3, 1, 1, 1, 1))
        self.assertAlmostEqual(s["avg_r"], 0.5)
        self.assertAlmostEqual(s["win"], 2 / 3, places=3)
        self.assertAlmostEqual(s["se"], 0.866, places=2)           # sd = 1.5, n = 3

    def test_stats_with_nothing_finished(self):
        s = backtest.stats([P("levels", None, "open")])
        self.assertEqual((s["n"], s["open"], s["avg_r"]), (0, 1, None))

    def test_edge_is_real_minus_shuffled_with_a_combined_error(self):
        e = backtest.edge({"avg_r": 0.30, "se": 0.04}, {"avg_r": -0.10, "se": 0.03})
        self.assertAlmostEqual(e["diff"], 0.40)
        self.assertAlmostEqual(e["se"], 0.05)
        self.assertAlmostEqual(e["z"], 8.0)
        self.assertIsNone(backtest.edge({"avg_r": None, "se": None}, {"avg_r": 0.1, "se": 0.1})["diff"])

    def test_summary_groups(self):
        rows = []
        for i in range(12):
            rows += [P("levels", 1.0, i=i, rr1=0.8), P("fixed", -1.0, "sl", i=i), P("atr", 0.0, "be", i=i)]
        null = [P("levels", -1.0, "sl", i=i) for i in range(12)]
        s = backtest.summarize(rows, {"symbols_done": 1}, null)
        self.assertEqual(s["rules"]["levels"]["n"], 12)
        self.assertEqual(s["levels_by_rr"]["below 1"]["n"], 12)
        self.assertEqual(s["levels_by_rr"]["1.5 or more"]["n"], 0)
        self.assertIn("Hammer", s["levels_by_pattern"])               # 12 >= 10 setups
        self.assertAlmostEqual(s["edge"]["levels"]["diff"], 2.0)       # real +1.0 vs shuffled -1.0
        json.dumps(s, allow_nan=False)


class TestMerge(unittest.TestCase):
    def test_merge_two_shards(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for k in (0, 1):
                (d / f"s{k}.json").write_text(json.dumps({"shard": k, "shards": 2, "step": 4, "start": 1000, "done": ["A"], "failed": [],
                                                           "plans": [P("levels", 1.0, i=k), P("atr", -1.0, "sl", i=k)],
                                                           "null_plans": [P("levels", -1.0, "sl", i=k)]}), encoding="utf-8")
            out = d / "bt.json"
            self.assertEqual(backtest.main(["--merge", str(d / "s*.json"), "--out", str(out)]), 0)
            s = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual((s["symbols_done"], s["rules"]["levels"]["n"], s["shards"]), (2, 2, 2))
            self.assertEqual(s["shuffled"]["levels"]["n"], 2)
            self.assertEqual(backtest.main(["--merge", str(d / "nothing*.json"), "--out", str(out)]), 2)


if __name__ == "__main__":
    unittest.main()
