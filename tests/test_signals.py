import unittest

import numpy as np
import pandas as pd

import config
import indicators as ind
import signals as S
from tests.helpers import K, frame, prefix, random_walk, session_index


def quiet_then_breakout(seed=5, up=True, trend_after=0):
    """500 normal candles, 60 very quiet low-volume ones, then one big high-volume breakout candle
    (optionally followed by `trend_after` steadily trending candles)."""
    rng = np.random.default_rng(seed)
    rows, price = [], 100.0

    def bar(vol, volume, drift=0.0):
        nonlocal price
        o = price
        c = o * (1 + drift + rng.normal(0, vol))
        h = max(o, c) * (1 + abs(rng.normal(0, vol / 3)))
        l = min(o, c) * (1 - abs(rng.normal(0, vol / 3)))
        rows.append((o, h, l, c, volume * rng.uniform(0.9, 1.1)))
        price = c
    for _ in range(500):
        bar(0.004, 1000)
    base = price                                     # the quiet phase is a tight, mean-reverting range
    for _ in range(60):                              # (a random walk drifts too far for a true TTM squeeze)
        o = price
        c = base * (1 + rng.normal(0, 0.0006))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0004)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.0004)))
        rows.append((o, h, l, c, 500 * rng.uniform(0.9, 1.1)))
        price = c
    sign = 1 if up else -1
    bar(0.0005, 3000, drift=sign * 0.03)
    for _ in range(trend_after):
        bar(0.0008, 1500, drift=sign * 0.004)
    return frame(rows)


def event_names(df, since=None):
    sigs, ctx, sc = S.analyze(df, "UNKNOWN")
    return [(s["name"], s["dir"], s["since"]) for s in sigs if since is None or s["since"] == since], ctx


class TestEvents(unittest.TestCase):
    def test_quiet_then_up_breakout_fires_the_compression_release_family(self):
        got, ctx = event_names(quiet_then_breakout(up=True), since=0)
        names = {n for n, d, s in got if d == "bull"}
        for expected in ["Bollinger Squeeze Breakout", "TTM Squeeze Fired", "ATR Expansion Breakout",
                         "Donchian Squeeze Breakout", "Volume Spike Breakout"]:
            self.assertIn(expected, names, f"{expected} missing; got {sorted(names)}")
        self.assertEqual(ctx["squeeze_state"], "fired")

    def test_quiet_then_down_breakout_is_bearish(self):
        got, _ = event_names(quiet_then_breakout(up=False), since=0)
        names = {n for n, d, s in got if d == "bear"}
        for expected in ["Bollinger Squeeze Breakout", "ATR Expansion Breakout", "Volume Spike Breakout"]:
            self.assertIn(expected, names)
        self.assertNotIn("Bollinger Squeeze Breakout", {n for n, d, s in got if d == "bull"})

    def test_no_breakout_events_in_a_plain_random_walk_tail_when_nothing_is_compressed(self):
        # a noisy walk has no squeeze right before its last candle: volume-spike breakouts need a dry-up first
        got, _ = event_names(random_walk(800, seed=11, vol=0.01), since=0)
        self.assertNotIn("Volume Spike Breakout", [n for n, d, s in got])

    def test_trend_after_breakout_starts_adx_and_ribbon_fan_out(self):
        df = quiet_then_breakout(seed=5, up=True, trend_after=20)
        sigs, ctx, _ = S.analyze(df, "BULLISH")
        names = {(s["name"], s["dir"]) for s in sigs}
        self.assertIn(("MA Ribbon Fan-Out", "bull"), names)
        self.assertIn(("ADX Trend Start", "bull"), names)

    def test_inside_bar_and_nr7(self):
        pre, p = prefix("flat")
        got, _ = event_names(frame(pre + [K(p, p + 1.2, 0.1, 0.1), K(p + 0.5, p + 0.7, 0.05, 0.05)]), since=0)
        names = [n for n, d, s in got]
        self.assertIn("Inside Bar", names)
        self.assertIn("NR7", names)
        self.assertIn("NR4", names)

    def test_inside_bar_needs_a_meaningful_mother_bar(self):
        pre, p = prefix("flat")
        got, _ = event_names(frame(pre + [K(p, p + 0.05, 0.02, 0.02), K(p + 0.02, p + 0.03, 0.005, 0.005)]), since=0)
        self.assertNotIn("Inside Bar", [n for n, d, s in got])

    def test_fair_value_gap_bull_and_status(self):
        pre, p = prefix("flat")
        cs = pre + [K(p, p + 0.3, 0.1, 0.1), K(p + 0.3, p + 1.5, 0.05, 0.05), K(p + 1.5, p + 1.7, 0.1, 0.1)]
        sigs, _, _ = S.analyze(frame(cs), "UNKNOWN")
        fvg = [s for s in sigs if s["name"] == "Fair Value Gap" and s["since"] == 0]
        self.assertEqual(len(fvg), 1)
        self.assertEqual(fvg[0]["dir"], "bull")
        self.assertEqual(fvg[0]["status"], "open")
        lo, hi = fvg[0]["zone"]
        self.assertAlmostEqual(lo, p + 0.4, places=3)
        self.assertAlmostEqual(hi, p + 1.4, places=3)
        # price later trades back into the gap -> partial; through it -> filled
        cs2 = cs + [K(p + 1.7, p + 1.2, 0.05, 0.05)]
        s2 = [s for s in S.analyze(frame(cs2), "UNKNOWN")[0] if s["name"] == "Fair Value Gap" and s["since"] == 1]
        self.assertEqual(s2[0]["status"], "partial")
        cs3 = cs + [K(p + 1.7, p + 0.1, 0.05, 0.05)]
        s3 = [s for s in S.analyze(frame(cs3), "UNKNOWN")[0] if s["name"] == "Fair Value Gap" and s["since"] == 1]
        self.assertEqual(s3[0]["status"], "filled")

    def test_tiny_gap_is_not_a_fair_value_gap(self):
        pre, p = prefix("flat")
        cs = pre + [K(p, p + 0.3, 0.1, 0.1), K(p + 0.3, p + 1.5, 0.05, 0.05), K(p + 0.4, p + 0.9, 0.1, 0.1)]
        sigs, _, _ = S.analyze(frame(cs), "UNKNOWN")
        self.assertFalse([s for s in sigs if s["name"] == "Fair Value Gap" and s["since"] == 0])

    def test_turtle_soup_bull_and_bear(self):
        pre, p = prefix("flat")
        df = frame(pre)
        lvl = df["Low"].iloc[-20:].min()
        got, _ = event_names(frame(pre + [K(lvl + 0.05, lvl + 0.5, 0.05, 0.35)]), since=0)
        self.assertIn(("Turtle Soup", "bull", 0), got)
        hi = df["High"].iloc[-20:].max()
        got, _ = event_names(frame(pre + [K(hi - 0.05, hi - 0.5, 0.35, 0.05)]), since=0)
        self.assertIn(("Turtle Soup", "bear", 0), got)

    def test_breakout_that_holds_is_not_turtle_soup(self):
        pre, p = prefix("flat")
        lvl = frame(pre)["Low"].iloc[-20:].min()
        got, _ = event_names(frame(pre + [K(lvl + 0.05, lvl - 0.6, 0.05, 0.1)]), since=0)
        self.assertNotIn("Turtle Soup", [n for n, d, s in got])

    def test_undercut_and_rally(self):
        pre, p = prefix("flat", n=80)
        base = frame(pre)["Low"].iloc[-41:].min()
        cs = pre + [K(base + 0.1, base - 0.2, 0.05, 0.3), K(base - 0.2, base + 0.4, 0.1, 0.1, v=1600)]
        got, _ = event_names(frame(cs), since=0)
        self.assertIn(("Undercut & Rally", "bull", 0), got)

    def test_pocket_pivot_needs_volume_above_every_recent_down_bar(self):
        pre, p = prefix("flat")
        got, _ = event_names(frame(pre + [K(p, p + 0.6, 0.05, 0.05, v=1800)]), since=0)
        self.assertIn(("Pocket Pivot", "bull", 0), got)
        # same candle on ordinary volume -> not a pocket pivot
        got, _ = event_names(frame(pre + [K(p, p + 0.6, 0.05, 0.05, v=1000)]), since=0)
        self.assertNotIn("Pocket Pivot", [n for n, d, s in got])

    def test_ichimoku_twist_found_on_a_v_shaped_market(self):
        down = [K(100 - 0.5 * i, 100 - 0.5 * (i + 1), 0.2, 0.2) for i in range(120)]
        p = 100 - 0.5 * 120
        upc = [K(p + 0.5 * i, p + 0.5 * (i + 1), 0.2, 0.2) for i in range(160)]
        e = S.Engine(frame(down + upc))
        found = [(i, ev["dir"]) for i in range(60, e.n) for ev in S.event_hits(e, i) if ev["name"] == "Ichimoku Cloud Twist"]
        self.assertTrue(any(d == "bull" for _, d in found), f"no bullish twist; got {found}")
        self.assertFalse([1 for i, d in found if d == "bear" and i > 120 + 100])

    def test_every_event_name_is_registered(self):
        e = S.Engine(quiet_then_breakout(trend_after=10))
        for i in range(60, e.n):
            for ev in S.event_hits(e, i):
                self.assertIn(ev["name"], S.EVENT_META)


class TestContext(unittest.TestCase):
    def test_obv_bullish_divergence_when_up_bars_carry_the_volume(self):
        cs, price = [], 100.0
        for i in range(120):
            d = 0.1 if i % 2 == 0 else -0.1
            cs.append(K(price, price + d, 0.1, 0.1, v=3000 if d > 0 else 1000))
            price += d
        sigs, ctx, _ = S.analyze(frame(cs), "UNKNOWN")
        self.assertEqual(ctx["obv_divergence"], "bullish")

    def test_obv_bearish_divergence_when_down_bars_carry_the_volume(self):
        cs, price = [], 100.0
        for i in range(120):
            d = 0.1 if i % 2 == 0 else -0.1
            cs.append(K(price, price + d, 0.1, 0.1, v=1000 if d > 0 else 3000))
            price += d
        self.assertEqual(S.analyze(frame(cs), "UNKNOWN")[1]["obv_divergence"], "bearish")

    def test_no_divergence_on_balanced_volume(self):
        self.assertEqual(S.analyze(random_walk(300, seed=2), "UNKNOWN")[1]["obv_divergence"], "none")

    def test_slot_relative_volume_ignores_the_heavy_open_candle(self):
        idx = session_index(210)
        vol = np.where((idx.hour == 9) & (idx.minute == 30), 3000.0, 1000.0)
        rv = ind.slot_rel_volume(idx, vol)
        ok = rv[~np.isnan(rv)]
        self.assertTrue(len(ok) > 100)
        self.assertTrue(np.allclose(ok, 1.0))

    def test_context_fields_are_json_safe_and_complete(self):
        from common import dumps_strict
        _, ctx, _ = S.analyze(random_walk(500, seed=4), "BULLISH")
        dumps_strict(ctx)
        for key in ["squeeze_state", "bb_squeeze", "ttm_squeeze", "atr_contraction", "hv_contraction", "volume_dry_up",
                    "obv_divergence", "ad_divergence", "cmf", "adx", "choppiness", "ichimoku_cloud",
                    "rel_volume", "compression_layers"]:
            self.assertIn(key, ctx)

    def test_quiet_market_reports_compression_layers(self):
        df = quiet_then_breakout()
        df = df.iloc[:-1]                                     # stop before the breakout candle
        _, ctx, _ = S.analyze(df, "UNKNOWN")
        for layer in ["bb", "ttm", "atr", "volume", "donchian"]:
            self.assertIn(layer, ctx["compression_layers"])
        self.assertIn(ctx["squeeze_state"], ("on", "fired"))      # a compressed market is never "off"


class TestScoring(unittest.TestCase):
    def sig(self, d="bull", stars=4, since=0, vol=1.8):
        return {"dir": d, "stars": stars, "since": since, "vol": vol}

    def test_perfect_bullish_setup(self):
        r = S.score_signal(self.sig(stars=5, vol=1.8), "BULLISH", "fired")
        self.assertEqual(r["parts"], {"pattern": 30, "smc": 25, "fresh": 20, "volume": 15, "squeeze": 10})
        self.assertEqual(r["total"], 100)

    def test_smc_alignment(self):
        agree = S.score_signal(self.sig("bull"), "BULLISH", "off")["parts"]["smc"]
        disagree = S.score_signal(self.sig("bull"), "BEARISH", "off")["parts"]["smc"]
        unknown = S.score_signal(self.sig("bull"), "UNKNOWN", "off")["parts"]["smc"]
        self.assertEqual((agree, disagree, unknown), (25, 9, 12))
        self.assertEqual(S.score_signal(self.sig("bear"), "BEARISH", "off")["parts"]["smc"], 25)

    def test_neutral_patterns_get_no_smc_credit_and_a_discount(self):
        r = S.score_signal(self.sig("neutral", stars=2), "BULLISH", "off")
        self.assertEqual(r["parts"]["smc"], 0)
        self.assertEqual(r["parts"]["pattern"], round(12 * 0.6))

    def test_freshness_fades_to_zero_at_the_configured_window(self):
        f = lambda since: S.score_signal(self.sig(since=since), "BULLISH", "off")["parts"]["fresh"]
        self.assertEqual(f(0), 20)
        self.assertEqual(f(10), 10)
        self.assertEqual(f(config.FRESH_FADE_BARS), 0)
        self.assertEqual(f(34), 0)

    def test_volume_credit_clamps(self):
        f = lambda v: S.score_signal(self.sig(vol=v), "BULLISH", "off")["parts"]["volume"]
        self.assertEqual((f(0.5), f(0.8), f(1.3), f(5.0), f(None)), (0, 0, 8, 15, 0))

    def test_squeeze_credit(self):
        f = lambda st: S.score_signal(self.sig(), "BULLISH", st)["parts"]["squeeze"]
        self.assertEqual((f("fired"), f("on"), f("off")), (10, 6, 0))


class TestAnalyzeOutput(unittest.TestCase):
    def test_signals_are_sorted_recent_first_and_capped(self):
        sigs, ctx, sc = S.analyze(random_walk(900, seed=1, vol=0.012), "BULLISH")
        self.assertLessEqual(len(sigs), config.MAX_SIGNALS)
        sinces = [s["since"] for s in sigs]
        self.assertEqual(sinces, sorted(sinces))
        self.assertTrue(all(0 <= s <= config.STORE_BARS - 1 for s in sinces))

    def test_range_triggers_are_only_kept_for_a_few_candles(self):
        for seed in range(6):
            sigs, _, _ = S.analyze(random_walk(900, seed=seed, vol=0.01), "UNKNOWN")
            for s in sigs:
                if s["group"] == "range":
                    self.assertLessEqual(s["since"], config.RANGE_KEEP_BARS)

    def test_row_scores_match_best_signal(self):
        sigs, _, sc = S.analyze(random_walk(900, seed=8, vol=0.01), "BULLISH")
        self.assertEqual(sc["bull"], max([s["score"] for s in sigs if s["dir"] == "bull"], default=0))
        self.assertEqual(sc["best"], max(sc["bull"], sc["bear"]))

    def test_signal_times_are_real_candle_starts_inside_the_session(self):
        sigs, _, _ = S.analyze(random_walk(900, seed=2, vol=0.01), "UNKNOWN")
        for s in sigs:
            t = pd.Timestamp(s["time"])
            self.assertIn((t.hour, t.minute), [(9, 30), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30)])
            self.assertLess(t.weekday(), 5)

    def test_confirmation_status(self):
        pre, p = prefix("down")
        eng = K(p, p - 0.5), K(p - 0.55, p + 0.3)         # bullish engulfing, high of pattern ~ p+0.4
        fresh = S.analyze(frame(pre + list(eng)), "BULLISH")[0]
        be = [s for s in fresh if s["name"] == "Bullish Engulfing" and s["since"] == 0][0]
        self.assertEqual(be["confirm"], "pending")
        ok = S.analyze(frame(pre + list(eng) + [K(p + 0.3, p + 1.0)]), "BULLISH")[0]
        self.assertEqual([s for s in ok if s["name"] == "Bullish Engulfing" and s["since"] == 1][0]["confirm"], "confirmed")
        bad = S.analyze(frame(pre + list(eng) + [K(p + 0.3, p - 1.5, 0.05, 0.05)]), "BULLISH")[0]
        self.assertEqual([s for s in bad if s["name"] == "Bullish Engulfing" and s["since"] == 1][0]["confirm"], "failed")
        wait = S.analyze(frame(pre + list(eng) + [K(p + 0.3, p + 0.25)]), "BULLISH")[0]
        self.assertEqual([s for s in wait if s["name"] == "Bullish Engulfing" and s["since"] == 1][0]["confirm"], "pending")


if __name__ == "__main__":
    unittest.main()
