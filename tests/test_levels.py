"""Support / resistance levels (levels.py): synthetic prices with known answers. No network."""
import time
import unittest

import numpy as np
import pandas as pd

import levels
from tests.helpers import session_index


def frame(close, wick=0.3, vol=1000.0):
    close = np.asarray(close, float)
    n = len(close)
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) + wick
    lo = np.minimum(op, close) - wick
    return pd.DataFrame({"Date": session_index(n), "Open": op, "High": hi, "Low": lo, "Close": close, "Volume": vol})


def ranging(n=700, lo=100.5, hi=109.5, period=40, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    return mid + amp * np.sin(2 * np.pi * t / period) + rng.normal(0, noise, n)


class TestLevels(unittest.TestCase):
    def test_finds_range_edges(self):
        df = frame(ranging(noise=0.05))
        out = levels.analyse(df, extended_hours=False)
        self.assertGreater(len(out["levels"]), 0)
        near = lambda target: [lv for lv in out["levels"] if abs(lv["price"] - target) < 1.0]
        top, bottom = near(109.8), near(100.2)
        self.assertTrue(top, out["levels"])
        self.assertTrue(bottom, out["levels"])
        self.assertTrue(all(lv["role"] == "R" for lv in top))
        self.assertTrue(all(lv["role"] == "S" for lv in bottom))
        best = max(out["levels"], key=lambda lv: lv["score"])
        self.assertGreaterEqual(best["score"], 40)
        self.assertGreaterEqual(best["visits"], 3)

    def test_confirmed_break_flips_the_role(self):
        c = ranging(n=600, noise=0.05)
        # price then closes above the top edge and stays there: resistance (about 109.8) becomes support
        up = np.linspace(105, 116, 60)
        df = frame(np.r_[c, up, np.full(40, 116.0)])
        out = levels.analyse(df, extended_hours=False)
        flipped = [lv for lv in out["levels"] if lv["flipped"] and abs(lv["price"] - 109.8) < 1.0]
        self.assertTrue(flipped, out["levels"])
        self.assertEqual(flipped[0]["role"], "S")

    def test_single_poke_is_not_a_break(self):
        c = ranging(n=700, noise=0.05)
        c[-30] = 112.0                                       # one close beyond the edge, then back inside
        out = levels.analyse(frame(c), extended_hours=False)
        top = [lv for lv in out["levels"] if abs(lv["price"] - 109.8) < 1.0]
        self.assertTrue(top)
        self.assertTrue(all(lv["role"] == "R" for lv in top))

    def test_too_few_bars_and_flat_price(self):
        self.assertEqual(levels.analyse(frame(np.full(50, 100.0)))["levels"], [])
        flat = frame(np.full(300, 100.0), wick=0.0)           # zero range -> ATR 0 -> nothing, and no crash
        self.assertEqual(levels.analyse(flat)["levels"], [])

    def test_output_is_json_safe(self):
        import json
        out = levels.analyse(frame(ranging(noise=0.2)), extended_hours=False)
        json.dumps(out, allow_nan=False)                      # NaN / inf anywhere would raise

    def test_speed_on_a_long_history(self):
        rng = np.random.default_rng(7)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 3500)))
        df = frame(close, wick=0.2)
        t0 = time.time()
        out = levels.analyse(df, extended_hours=False)
        self.assertLess(time.time() - t0, 3.0)                # 240 symbols must stay in the minutes, not hours
        self.assertEqual(out["meta"]["bars"], 3500)


class TestConfluenceFixes(unittest.TestCase):
    def test_round_number_rule_is_selective(self):
        p = levels.Params()
        # price ~277, ATR 5.4 -> step 100/5 = 20 -> multiples of 20 only
        self.assertTrue(levels._round_hit(280.0, 277.0, 5.4, p))
        self.assertTrue(levels._round_hit(281.5, 277.0, 5.4, p))
        self.assertFalse(levels._round_hit(270.0, 277.0, 5.4, p))
        self.assertFalse(levels._round_hit(290.0, 277.0, 5.4, p))
        book = levels.Params(round_div=20.0, round_cap=None)      # the book's plain rule counts far more prices as round
        hits_new = sum(levels._round_hit(x / 10, 277.0, 5.4, p) for x in range(2500, 3000))
        hits_book = sum(levels._round_hit(x / 10, 277.0, 5.4, book) for x in range(2500, 3000))
        self.assertLess(hits_new, hits_book * 0.5)

    def test_htf_swings_come_from_daily_candles(self):
        df = frame(ranging(n=700, period=40, noise=0.05))
        pts = levels._htf_swings(df, df["High"].to_numpy(), df["Low"].to_numpy(), levels.Params())
        day = df["Date"].dt.normalize()
        g = df.groupby(day).agg(High=("High", "max"), Low=("Low", "min"))
        allowed = set(np.round(g["High"], 9)) | set(np.round(g["Low"], 9))
        self.assertTrue(pts)
        self.assertLessEqual(len(pts), levels.Params().htf_keep)
        self.assertTrue(all(round(x, 9) in allowed for x in pts))   # every point is a daily high or low, not an hourly one


if __name__ == "__main__":
    unittest.main()
