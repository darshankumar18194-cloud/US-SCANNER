import unittest

import numpy as np

import patterns as P
from tests.helpers import K, frame, prefix

# name -> (trend before the pattern, function(p) -> list of candles, p = last close of the prefix)
CASES = {
    "Hammer": ("down", lambda p: [K(p, p + 0.15, 0.02, 0.8)]),
    "Hanging Man": ("up", lambda p: [K(p, p + 0.15, 0.02, 0.8)]),
    "Inverted Hammer": ("down", lambda p: [K(p, p + 0.15, 0.8, 0.02)]),
    "Shooting Star": ("up", lambda p: [K(p, p - 0.15, 0.8, 0.02)]),
    "Dragonfly Doji": ("down", lambda p: [K(p, p + 0.02, 0.0, 0.9)]),
    "Gravestone Doji": ("up", lambda p: [K(p, p - 0.02, 0.9, 0.0)]),
    "Doji": ("flat", lambda p: [K(p, p + 0.01, 0.25, 0.25)]),
    "Long-Legged Doji": ("flat", lambda p: [K(p, p + 0.02, 0.6, 0.6)]),
    "Bullish Marubozu": ("flat", lambda p: [K(p, p + 1.0, 0, 0)]),
    "Bearish Marubozu": ("flat", lambda p: [K(p, p - 1.0, 0, 0)]),
    "Bullish Belt Hold": ("flat", lambda p: [K(p, p + 1.0, 0.2, 0)]),
    "Bearish Belt Hold": ("flat", lambda p: [K(p, p - 1.0, 0, 0.2)]),
    "High Wave Candle": ("flat", lambda p: [K(p, p + 0.3, 0.8, 0.8)]),
    "Bullish Spinning Top": ("flat", lambda p: [K(p, p + 0.2, 0.25, 0.25)]),
    "Bearish Spinning Top": ("flat", lambda p: [K(p, p - 0.2, 0.25, 0.25)]),

    "Bullish Engulfing": ("down", lambda p: [K(p, p - 0.5), K(p - 0.55, p + 0.3)]),
    "Bearish Engulfing": ("up", lambda p: [K(p, p + 0.5), K(p + 0.55, p - 0.3)]),
    "Piercing Line": ("down", lambda p: [K(p, p - 1.0), K(p - 1.05, p - 0.4)]),
    "Dark Cloud Cover": ("up", lambda p: [K(p, p + 1.0), K(p + 1.05, p + 0.4)]),
    "Bullish Harami": ("down", lambda p: [K(p, p - 1.2), K(p - 0.9, p - 0.6)]),
    "Bearish Harami": ("up", lambda p: [K(p, p + 1.2), K(p + 0.9, p + 0.6)]),
    "Homing Pigeon": ("down", lambda p: [K(p, p - 1.2), K(p - 0.6, p - 0.9)]),
    "Bullish Harami Cross": ("down", lambda p: [K(p, p - 1.2), K(p - 0.6, p - 0.61, 0.3, 0.3)]),
    "Bearish Harami Cross": ("up", lambda p: [K(p, p + 1.2), K(p + 0.6, p + 0.61, 0.3, 0.3)]),
    "Tweezer Bottom": ("down", lambda p: [K(p, p - 0.6), K(p - 0.6, p - 0.3, 0.1, 0.1)]),
    "Tweezer Top": ("up", lambda p: [K(p, p + 0.6), K(p + 0.6, p + 0.3, 0.1, 0.1)]),
    "Bullish Kicker": ("flat", lambda p: [K(p, p - 0.6), K(p + 0.3, p + 1.0)]),
    "Bearish Kicker": ("flat", lambda p: [K(p, p + 0.6), K(p - 0.3, p - 1.0)]),
    "Rising Window": ("up", lambda p: [K(p, p + 0.4, 0.05, 0.1), K(p + 0.75, p + 1.1, 0.1, 0.05)]),
    "Falling Window": ("down", lambda p: [K(p, p - 0.4, 0.1, 0.05), K(p - 0.75, p - 1.1, 0.05, 0.1)]),
    "Bullish Doji Star": ("down", lambda p: [K(p, p - 1.2), K(p - 1.3, p - 1.29, 0.3, 0.3)]),
    "Bearish Doji Star": ("up", lambda p: [K(p, p + 1.2), K(p + 1.3, p + 1.29, 0.3, 0.3)]),
    "Bullish Meeting Lines": ("down", lambda p: [K(p, p - 1.0), K(p - 1.6, p - 1.0)]),
    "Bearish Meeting Lines": ("up", lambda p: [K(p, p + 1.0), K(p + 1.6, p + 1.0)]),
    "Matching Low": ("down", lambda p: [K(p, p - 0.6), K(p - 0.2, p - 0.6)]),
    "Bullish Separating Lines": ("up", lambda p: [K(p, p - 0.6), K(p, p + 0.7)]),
    "Bearish Separating Lines": ("down", lambda p: [K(p, p + 0.6), K(p, p - 0.7)]),
    "On-Neck Line": ("down", lambda p: [K(p, p - 1.0), K(p - 1.2, p - 1.1, 0.05, 0.0)]),
    "In-Neck Line": ("down", lambda p: [K(p, p - 1.0), K(p - 1.2, p - 0.95)]),
    "Thrusting Line": ("down", lambda p: [K(p, p - 1.0), K(p - 1.2, p - 0.6)]),

    "Morning Star": ("down", lambda p: [K(p, p - 1.2), K(p - 1.3, p - 1.15), K(p - 1.1, p - 0.1)]),
    "Evening Star": ("up", lambda p: [K(p, p + 1.2), K(p + 1.3, p + 1.15), K(p + 1.1, p + 0.1)]),
    "Bullish Abandoned Baby": ("down", lambda p: [K(p, p - 1.2), K(p - 1.6, p - 1.59, 0.1, 0.1),
                                                  K(p - 1.4, p - 0.4, 0.1, 0.05)]),
    "Bearish Abandoned Baby": ("up", lambda p: [K(p, p + 1.2), K(p + 1.6, p + 1.59, 0.1, 0.1),
                                                K(p + 1.4, p + 0.4, 0.05, 0.1)]),
    "Three White Soldiers": ("down", lambda p: [K(p, p + 0.9, 0.05, 0.1), K(p + 0.3, p + 1.2, 0.05, 0.1),
                                                K(p + 0.6, p + 1.5, 0.05, 0.1)]),
    "Three Black Crows": ("up", lambda p: [K(p, p - 0.9, 0.1, 0.05), K(p - 0.3, p - 1.2, 0.1, 0.05),
                                           K(p - 0.6, p - 1.5, 0.1, 0.05)]),
    "Three Inside Up": ("down", lambda p: [K(p, p - 1.2), K(p - 0.9, p - 0.5), K(p - 0.4, p + 0.3)]),
    "Three Inside Down": ("up", lambda p: [K(p, p + 1.2), K(p + 0.9, p + 0.5), K(p + 0.4, p - 0.3)]),
    "Three Outside Up": ("down", lambda p: [K(p, p - 0.5), K(p - 0.55, p + 0.4), K(p + 0.4, p + 1.0)]),
    "Three Outside Down": ("up", lambda p: [K(p, p + 0.5), K(p + 0.55, p - 0.4), K(p - 0.4, p - 1.0)]),
    "Bullish Tri-Star": ("down", lambda p: [K(p, p + 0.01, 0.2, 0.2), K(p - 0.8, p - 0.79, 0.1, 0.1),
                                            K(p - 0.1, p - 0.09, 0.2, 0.2)]),
    "Bearish Tri-Star": ("up", lambda p: [K(p, p + 0.01, 0.2, 0.2), K(p + 0.8, p + 0.81, 0.1, 0.1),
                                          K(p + 0.1, p + 0.11, 0.2, 0.2)]),
    "Stick Sandwich": ("down", lambda p: [K(p, p - 0.6), K(p - 0.6, p + 0.2), K(p + 0.2, p - 0.6)]),
    "Two Black Gapping": ("down", lambda p: [K(p, p - 0.5), K(p - 1.0, p - 1.4, 0.1, 0.1), K(p - 1.4, p - 1.9, 0.1, 0.1)]),

    "Rising Three Methods": ("flat", lambda p: [K(p, p + 1.4, 0.05, 0.05), K(p + 1.3, p + 1.1, 0.05, 0.05),
                                                K(p + 1.15, p + 0.95, 0.05, 0.05), K(p + 1.0, p + 0.8, 0.05, 0.05),
                                                K(p + 0.85, p + 1.9, 0.05, 0.05)]),
    "Falling Three Methods": ("flat", lambda p: [K(p, p - 1.4, 0.05, 0.05), K(p - 1.3, p - 1.1, 0.05, 0.05),
                                                 K(p - 1.15, p - 0.95, 0.05, 0.05), K(p - 1.0, p - 0.8, 0.05, 0.05),
                                                 K(p - 0.85, p - 1.9, 0.05, 0.05)]),
    "Mat Hold": ("flat", lambda p: [K(p, p + 1.4, 0.05, 0.05), K(p + 1.55, p + 1.35, 0.05, 0.05),
                                    K(p + 1.35, p + 1.15, 0.05, 0.05), K(p + 1.2, p + 1.0, 0.05, 0.05),
                                    K(p + 1.05, p + 2.2, 0.05, 0.05)]),
    "Ladder Bottom": ("down", lambda p: [K(p, p - 0.8, 0.05, 0.05), K(p - 0.4, p - 1.2, 0.05, 0.05),
                                         K(p - 1.0, p - 1.8, 0.05, 0.05), K(p - 1.7, p - 1.8, 0.6, 0.05),
                                         K(p - 1.5, p - 0.8, 0.05, 0.05)]),
    "Ladder Top": ("up", lambda p: [K(p, p + 0.8, 0.05, 0.05), K(p + 0.4, p + 1.2, 0.05, 0.05),
                                    K(p + 1.0, p + 1.8, 0.05, 0.05), K(p + 1.7, p + 1.8, 0.05, 0.6),
                                    K(p + 1.5, p + 0.8, 0.05, 0.05)]),
    "Concealing Baby Swallow": ("down", lambda p: [K(p, p - 0.9, 0, 0), K(p - 0.95, p - 1.9, 0, 0),
                                                   K(p - 2.1, p - 2.4, 0.5, 0.05), K(p - 1.5, p - 2.6, 0.05, 0.05)]),
}


def hits_for(trend, build):
    pre, p = prefix(trend)
    cs = pre + build(p)
    df = frame(cs)
    x = P.Candles(df)
    return {h.name: h.length for h in P.candle_hits(x, x.n - 1)}


class TestCandlestickPatterns(unittest.TestCase):
    def test_every_pattern_has_a_test_case(self):
        self.assertEqual(set(CASES), set(P.META))

    def test_each_pattern_detected_on_its_textbook_shape(self):
        for name, (trend, build) in CASES.items():
            with self.subTest(pattern=name):
                got = hits_for(trend, build)
                self.assertIn(name, got, f"{name} not detected; got {sorted(got)}")

    def test_trend_dependent_patterns_need_the_trend(self):
        # The same candles after a sideways market must NOT count as reversal patterns.
        for name in ["Hammer", "Hanging Man", "Shooting Star", "Bullish Engulfing", "Bearish Engulfing",
                     "Morning Star", "Evening Star", "Piercing Line", "Dark Cloud Cover", "Bullish Harami"]:
            trend, build = CASES[name]
            with self.subTest(pattern=name):
                self.assertNotIn(name, hits_for("flat", build))

    def test_wrong_trend_gives_the_mirror_name_not_the_same(self):
        # A hammer shape after an UPtrend is a Hanging Man, never a Hammer.
        got = hits_for("up", CASES["Hammer"][1])
        self.assertIn("Hanging Man", got)
        self.assertNotIn("Hammer", got)

    def test_no_pattern_on_dead_flat_candles(self):
        df = frame([K(100, 100, 0, 0)] * 80)
        x = P.Candles(df)
        for i in range(60, x.n):
            self.assertEqual(P.candle_hits(x, i), [])

    def test_stop_reference_follows_the_handbook(self):
        pre, p = prefix("down")
        x = P.Candles(frame(pre + CASES["Morning Star"][1](p)))
        i = x.n - 1
        # Handbook: stop below the low of the MIDDLE candle for a Morning Star
        self.assertAlmostEqual(P.stop_price(x, "Morning Star", i, 3), x.l[i - 1])
        pre, p = prefix("down")
        x = P.Candles(frame(pre + CASES["Bullish Engulfing"][1](p)))
        i = x.n - 1
        self.assertAlmostEqual(P.stop_price(x, "Bullish Engulfing", i, 2), min(x.l[i - 1], x.l[i]))
        self.assertIsNone(P.stop_price(x, "Doji", i, 1))

    def test_random_data_never_crashes_and_names_are_known(self):
        from tests.helpers import random_walk
        x = P.Candles(random_walk(800, seed=3, vol=0.01))
        seen = 0
        for i in range(60, x.n):
            for h in P.candle_hits(x, i):
                self.assertIn(h.name, P.META)
                self.assertGreaterEqual(i - h.length + 1, 0)
                seen += 1
        self.assertGreater(seen, 0)


if __name__ == "__main__":
    unittest.main()
