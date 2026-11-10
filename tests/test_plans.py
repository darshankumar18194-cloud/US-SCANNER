"""Trade plans (plans.py): hand-made candles with known outcomes. No network."""
import json
import unittest

import numpy as np
import pandas as pd

import config
import plans
from tests.helpers import session_index


def frame(rows):
    """rows = [(open, high, low, close), ...] -> candle frame on a real session calendar."""
    n = len(rows)
    a = np.array(rows, float)
    return pd.DataFrame({"Date": session_index(n), "Open": a[:, 0], "High": a[:, 1], "Low": a[:, 2], "Close": a[:, 3], "Volume": 1000.0})


def long_plan(df, entry_idx, entry=100.0, sl=96.0, tp1=104.0, tp2=108.0, side="long"):
    label = df["Date"].dt.strftime(config.DATE_FMT).iloc[entry_idx]
    risk = abs(entry - sl)
    return {"id": "T|long|Hammer|x", "symbol": "T", "side": side, "entry_time": label, "entry": entry, "sl": sl, "tp1": tp1,
            "tp2": tp2, "rr1": abs(tp1 - entry) / risk, "rr2": None if tp2 is None else abs(tp2 - entry) / risk,
            "status": "open", "tp1_time": None, "tp2_time": None, "closed_time": None, "r": None, "r_live": None}


FLAT = (100, 100.5, 99.5, 100)


class TestEvaluate(unittest.TestCase):
    def test_stop_hit_is_minus_one_r(self):
        df = frame([FLAT, FLAT, (100, 100.5, 95.5, 96)])
        u = plans.evaluate(df, long_plan(df, 1))
        self.assertEqual((u["status"], u["r"]), ("sl", -1.0))

    def test_tp1_then_tp2(self):
        df = frame([FLAT, FLAT, (100, 104.5, 99.8, 104), (104, 108.5, 103.5, 108)])
        u = plans.evaluate(df, long_plan(df, 1))
        self.assertEqual(u["status"], "tp2")
        self.assertAlmostEqual(u["r"], 0.5 * 1.0 + 0.5 * 2.0)          # rr1 = 1, rr2 = 2

    def test_tp1_then_stop_moves_to_entry(self):
        df = frame([FLAT, FLAT, (100, 104.5, 99.8, 104), (104, 104.2, 99.5, 100)])   # back through entry
        u = plans.evaluate(df, long_plan(df, 1))
        self.assertEqual(u["status"], "be")
        self.assertAlmostEqual(u["r"], 0.5)                             # half sold at TP1, half stopped at entry

    def test_same_candle_touching_stop_and_target_counts_as_stop(self):
        df = frame([FLAT, FLAT, (100, 105, 95, 100)])
        self.assertEqual(plans.evaluate(df, long_plan(df, 1))["status"], "sl")

    def test_tp1_candle_does_not_also_judge_the_breakeven_stop(self):
        df = frame([FLAT, FLAT, (100, 104.5, 99.0, 104)])              # tp1 hit, low is below entry: order unknown, no stop
        u = plans.evaluate(df, long_plan(df, 1))
        self.assertEqual(u["status"], "tp1")
        self.assertIsNone(u["closed_time"])

    def test_still_open_reports_live_r(self):
        df = frame([FLAT, FLAT, (100, 101, 99.5, 101)])
        u = plans.evaluate(df, long_plan(df, 1))
        self.assertEqual(u["status"], "open")
        self.assertAlmostEqual(u["r_live"], 0.25)

    def test_short_mirror(self):
        df = frame([FLAT, FLAT, (100, 100.2, 95.5, 96), (96, 96.5, 91.5, 92)])
        p = long_plan(df, 1, entry=100, sl=104, tp1=96, tp2=92, side="short")
        u = plans.evaluate(df, p)
        self.assertEqual(u["status"], "tp2")
        self.assertAlmostEqual(u["r"], 0.5 * 1.0 + 0.5 * 2.0)
        df2 = frame([FLAT, FLAT, (100, 104.5, 99.9, 104)])
        self.assertEqual(plans.evaluate(df2, long_plan(df2, 1, entry=100, sl=104, tp1=96, tp2=92, side="short"))["status"], "sl")

    def test_unknown_entry_candle_returns_none(self):
        df = frame([FLAT, FLAT, FLAT])
        p = long_plan(df, 1)
        p["entry_time"] = "1999-01-01 10:30"
        self.assertIsNone(plans.evaluate(df, p))

    def test_expires_after_max_age(self):
        df = frame([FLAT, FLAT] + [(100, 100.4, 99.6, 100.1)] * (plans.MAX_AGE + 2))
        u = plans.evaluate(df, long_plan(df, 1))
        self.assertEqual(u["status"], "expired")


def lv(price, grade="B"):
    return {"price": price, "grade": grade, "role": "S"}


class TestBuildPlan(unittest.TestCase):
    def setUp(self):
        self.df = frame([FLAT] * 5)
        self.setup = {"low": True, "conf_idx": 3, "ext": 99.0, "level": 99.0, "dir": "bull",
                      "g": {"name": "Hammer", "time": "2026-11-06 10:30", "score": 70}}

    def test_long_uses_next_levels_and_a_buffer(self):
        p = plans.build_plan("T", self.setup, self.df, [lv(90), lv(96), lv(104), lv(110), lv(118)], atr=2.0, scanned_at="x")
        self.assertEqual((p["tp1"], p["tp2"]), (104.0, 110.0))
        self.assertAlmostEqual(p["sl"], 96 - 0.2 * 2.0)                 # nearest level below entry, minus the buffer
        self.assertEqual(p["side"], "long")
        self.assertAlmostEqual(p["rr1"], (104 - 100) / (100 - p["sl"]), places=2)

    def test_levels_closer_than_half_an_atr_are_skipped(self):
        p = plans.build_plan("T", self.setup, self.df, [lv(90), lv(99.5), lv(100.6), lv(106), lv(112)], atr=2.0, scanned_at="x")
        self.assertEqual(p["tp1"], 106.0)                               # 100.6 is only 0.3 ATR away
        self.assertLess(p["sl"], 90.0 + 1)                              # 99.5 skipped, so the stop sits under 90

    def test_stop_is_never_inside_the_pattern(self):
        s = dict(self.setup, ext=94.0)                                  # the pattern's low is below the level-based stop
        p = plans.build_plan("T", s, self.df, [lv(96), lv(104), lv(110)], atr=2.0, scanned_at="x")
        self.assertAlmostEqual(p["sl"], 94.0 - 0.2)
        self.assertEqual(p["sl_basis"], "pattern")

    def test_no_plan_without_a_level_on_both_sides(self):
        self.assertIsNone(plans.build_plan("T", self.setup, self.df, [lv(104), lv(110)], atr=2.0, scanned_at="x"))
        self.assertIsNone(plans.build_plan("T", self.setup, self.df, [lv(90), lv(95)], atr=2.0, scanned_at="x"))

    def test_short_is_the_mirror(self):
        s = dict(self.setup, low=False, ext=101.0, dir="bear")
        p = plans.build_plan("T", s, self.df, [lv(90), lv(96), lv(104), lv(110)], atr=2.0, scanned_at="x")
        self.assertEqual(p["side"], "short")
        self.assertEqual((p["tp1"], p["tp2"]), (96.0, 90.0))
        self.assertAlmostEqual(p["sl"], 104 + 0.2 * 2.0)


class TestStore(unittest.TestCase):
    def test_existing_plan_is_never_redrawn(self):
        old = {"id": "a", "symbol": "T", "entry_time": "2026-11-06 10:30", "status": "open", "sl": 96.0, "tp1": 104.0}
        redrawn = dict(old, sl=90.0, tp1=120.0)
        store = plans.merge({"plans": [old]}, [redrawn], {}, "now")
        self.assertEqual(store["plans"][0]["sl"], 96.0)
        self.assertEqual(store["plans"][0]["tp1"], 104.0)

    def test_open_plans_move_forward_and_closed_ones_stay_closed(self):
        a = {"id": "a", "symbol": "T", "entry_time": "1", "status": "open"}
        b = {"id": "b", "symbol": "T", "entry_time": "2", "status": "sl", "closed_time": "3", "r": -1.0}
        store = plans.merge({"plans": [a, b]}, [], {"a": {"status": "tp1"}, "b": {"status": "tp2", "r": 5.0}}, "now")
        by = {p["id"]: p for p in store["plans"]}
        self.assertEqual(by["a"]["status"], "tp1")
        self.assertEqual((by["b"]["status"], by["b"]["r"]), ("sl", -1.0))

    def test_old_closed_plans_are_trimmed_but_open_ones_never(self):
        many = [{"id": f"c{i}", "symbol": "T", "entry_time": f"{i:05d}", "status": "sl", "closed_time": f"{i:05d}"}
                for i in range(plans.KEEP_CLOSED + 50)]
        keep = {"id": "o", "symbol": "T", "entry_time": "00000", "status": "open"}
        store = plans.merge({"plans": many + [keep]}, [], {}, "now")
        ids = {p["id"] for p in store["plans"]}
        self.assertIn("o", ids)
        self.assertEqual(len(store["plans"]), plans.KEEP_CLOSED + 1)
        self.assertNotIn("c0", ids)


class TestVariants(unittest.TestCase):
    def setUp(self):
        self.df = frame([FLAT] * 5)
        self.setup = {"low": True, "conf_idx": 3, "ext": 99.0, "level": 99.0, "dir": "bull",
                      "g": {"name": "Hammer", "time": "2026-11-06 10:30", "score": 70}}
        self.lv = [lv(90), lv(96), lv(104), lv(110), lv(118)]

    def test_levels_rule_keeps_its_old_id_and_the_others_get_a_suffix(self):
        ids = {v: plans.build_plan("T", self.setup, self.df, self.lv, 2.0, "x", v)["id"] for v in plans.VARIANTS}
        self.assertEqual(ids["levels"], "T|long|Hammer|2026-11-06 10:30")      # unchanged: old plans.json stays valid
        self.assertTrue(ids["fixed"].endswith("|fixed"))
        self.assertTrue(ids["atr"].endswith("|atr"))
        self.assertEqual(len(set(ids.values())), 3)

    def test_fixed_rule_has_the_level_stop_and_targets_at_1_5r_and_3r(self):
        a = plans.build_plan("T", self.setup, self.df, self.lv, 2.0, "x", "levels")
        f = plans.build_plan("T", self.setup, self.df, self.lv, 2.0, "x", "fixed")
        self.assertEqual(f["sl"], a["sl"])
        risk = 100.0 - f["sl"]
        self.assertAlmostEqual(f["tp1"], 100 + 1.5 * risk, places=3)
        self.assertAlmostEqual(f["tp2"], 100 + 3.0 * risk, places=3)
        self.assertEqual((f["rr1"], f["rr2"], f["variant"]), (1.5, 3.0, "fixed"))

    def test_atr_rule_needs_no_levels_at_all(self):
        a = plans.build_plan("T", self.setup, self.df, [], 2.0, "x", "atr")
        self.assertIsNotNone(a)
        self.assertAlmostEqual(a["sl"], 100 - 1.5 * 2.0)                      # 3 below entry, pattern low 99 is closer
        self.assertAlmostEqual(a["tp1"], 100 + 1.5 * 3.0)
        self.assertIsNone(plans.build_plan("T", self.setup, self.df, [], 2.0, "x", "levels"))
        self.assertIsNone(plans.build_plan("T", self.setup, self.df, [], 2.0, "x", "fixed"))

    def test_atr_stop_is_never_inside_the_pattern(self):
        s = dict(self.setup, ext=95.0)                                         # a long pattern: low is further than 1.5 ATR
        a = plans.build_plan("T", s, self.df, [], 2.0, "x", "atr")
        self.assertAlmostEqual(a["sl"], 95.0 - 0.2)

    def test_short_atr_rule_is_the_mirror(self):
        s = dict(self.setup, low=False, ext=101.0, dir="bear")
        a = plans.build_plan("T", s, self.df, [], 2.0, "x", "atr")
        self.assertAlmostEqual(a["sl"], 100 + 1.5 * 2.0)
        self.assertAlmostEqual(a["tp1"], 100 - 1.5 * 3.0)
        self.assertEqual(a["side"], "short")

    def test_variants_are_followed_like_any_other_plan(self):
        df = frame([FLAT, FLAT, (100, 104.6, 99.9, 104.5), (104.5, 110.2, 104, 110)])
        f = plans.build_plan("T", dict(self.setup, conf_idx=1), df, self.lv, 2.0, "x", "atr")
        u = plans.evaluate(df, f)
        self.assertEqual(u["status"], "tp2" if 110 >= f["tp2"] else "tp1")


from tests.test_scan import Env   # noqa: E402  (reuses the temporary-folder scan harness)


class TestScanWritesPlans(Env):
    def plans_path(self):
        return self.tmp / "shared-data" / "plans.json"

    def test_scan_writes_plans_json_and_keeps_stocks_json_clean(self):
        self.assertEqual(self.run_scan(), 0)
        d = json.loads(self.plans_path().read_text(encoding="utf-8"))
        self.assertEqual(d["schema"], 1)
        self.assertIsInstance(d["plans"], list)
        for s in self.store()["stocks"]:
            self.assertNotIn("plan_new", s)                       # plan data lives in plans.json only
            self.assertNotIn("plan_eval", s)
            if s["scanner_status"] == "OK":
                self.assertIn("levels", s)

    def test_open_plan_is_followed_on_the_next_scan(self):
        self.run_scan()
        c = self.rec("GOOD")["candles"]
        entry_row = c[-30]
        plan = {"id": "GOOD|long|Hammer|seed", "symbol": "GOOD", "side": "long", "pattern": "Hammer", "pattern_time": entry_row[0],
                "entry_time": entry_row[0], "entry": entry_row[4], "sl": entry_row[4] * 0.5, "tp1": entry_row[4] * 2.0, "tp2": None,
                "rr1": 2.0, "rr2": None, "status": "open", "tp1_time": None, "tp2_time": None, "closed_time": None,
                "r": None, "r_live": None, "created": "seed"}
        self.plans_path().write_text(json.dumps({"schema": 1, "updated": None, "plans": [plan]}), encoding="utf-8")
        self.assertEqual(self.run_scan(), 0)
        saved = {p["id"]: p for p in json.loads(self.plans_path().read_text(encoding="utf-8"))["plans"]}
        p = saved["GOOD|long|Hammer|seed"]
        self.assertEqual(p["status"], "open")
        self.assertIsNotNone(p["r_live"])                           # moved forward with the candles after entry
        self.assertEqual((p["sl"], p["created"]), (entry_row[4] * 0.5, "seed"))   # and was not redrawn

    def test_unreadable_plans_file_is_left_alone_and_the_scan_still_succeeds(self):
        self.plans_path().write_text("{ not json", encoding="utf-8")
        self.assertEqual(self.run_scan(), 0)
        self.assertEqual(self.plans_path().read_text(encoding="utf-8"), "{ not json")
        self.assertEqual(self.rec("GOOD")["scanner_status"], "OK")


if __name__ == "__main__":
    unittest.main()
