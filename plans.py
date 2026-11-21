"""Trade plans for confirmed SMC + pattern setups, tracked from entry to outcome.

The setup rules are a straight port of the dashboard's "SMC + Pattern" tables (main-dashboard/index.html, comboRows), so a
plan exists exactly for the rows you see there. A plan is made ONCE, on the first scan that sees the confirmed setup, and
then frozen - later scans only move it forward (open -> TP1 hit -> finished). That makes the record honest: the levels it
used cannot be redrawn after the fact.

LONG  (Strong/Weak Low + bullish pattern)            SHORT (Strong/Weak High + bearish pattern) is the mirror image.
  entry = close of the candle that CONFIRMED the pattern
  SL    = the nearest level below entry that is at least MIN_RISK_ATR away, minus STOP_BUF_ATR (never inside the pattern:
          at most pattern low minus 0.1 ATR)
  TP1   = the nearest level above entry (at least MIN_RISK_ATR away),   TP2 = the next level after it
  after TP1 is hit the stop moves to entry; TP2 ends the trade.

Three RULES are tracked side by side on the very same setups, so the record can show whether levels help at all:
  levels  the rule above (SL and TPs at the next support / resistance)
  fixed   the same level-based SL, but fixed targets: TP1 = 1.5R, TP2 = 3R         (do level-based TARGETS add anything?)
  atr     no levels at all: SL = 1.5 ATR beyond entry (or the pattern's extreme if that is further), TP1 = 1.5R, TP2 = 3R
          (a baseline: do levels add anything at all?)
Only "levels" is shown in the dashboard's Trade plan column; the comparison table shows all three.
NOTE: the three rules have different stop / target SHAPES, so a gap between them proves nothing about levels (on pure noise
"levels" beats "atr" by itself). The honest test is backtest.py: every rule against its own result on SHUFFLED prices.

Replay rules (conservative, like the book's signal book):
  * the stop is checked BEFORE the targets on every candle: a candle that touches both counts as a stop;
  * on the candle that hits TP1 nothing else is judged (the order inside one candle is unknown);
  * a plan still running after MAX_AGE candles is closed at the last close ("expired").
Results are in R (1R = the distance entry -> SL) assuming half the position is sold at TP1 and half at TP2 / the stop.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import config
from common import atomic_write_text, dumps_strict

REV = {
    "bull": ["Hammer", "Inverted Hammer", "Dragonfly Doji", "Bullish Engulfing", "Piercing Line", "Bullish Kicker",
             "Bullish Harami Cross", "Bullish Doji Star", "Morning Star", "Bullish Abandoned Baby", "Three White Soldiers",
             "Three Inside Up", "Three Outside Up", "Bullish Tri-Star", "Turtle Soup", "Undercut & Rally"],
    "bear": ["Shooting Star", "Hanging Man", "Gravestone Doji", "Bearish Engulfing", "Dark Cloud Cover", "Bearish Kicker",
             "Bearish Harami Cross", "Bearish Doji Star", "Evening Star", "Bearish Abandoned Baby", "Three Black Crows",
             "Three Inside Down", "Three Outside Down", "Bearish Tri-Star", "Turtle Soup"],
}
SMC_SIGNALS = {"STRONG_HIGH", "WEAK_HIGH", "STRONG_LOW", "WEAK_LOW"}
COMBO_ATR = 1.0        # the pattern's extreme must be within this many ATR of the SMC level
COMBO_TREND = 1.0      # the 8 closes before the pattern must average this many ATR beyond it
COMBO_RUN = 2.0        # drop the setup once price has run this many ATR away from the pattern
WINDOW = 10            # only patterns formed within the last N candles (the dashboard's default window)
MIN_RISK_ATR = 0.5     # a level closer than this to entry is skipped as an SL / TP anchor
STOP_BUF_ATR = 0.2     # stop beyond the level by this much
MAX_AGE = 100          # candles after entry before a still-running plan is closed as "expired"
KEEP_CLOSED = 1500
MIN_RR = 1.5           # the book's minimum reward-to-risk; plans below it are kept but flagged
OPEN_STATES = ("open", "tp1")
VARIANTS = ("levels", "fixed", "atr")
FIXED_R = (1.5, 3.0)   # TP1 / TP2 of the "fixed" and "atr" rules, in R
ATR_STOP = 1.5         # SL distance of the "atr" rule, in ATR


# ----------------------------------------------------------------------------- 1. which setups exist
def setups(df: pd.DataFrame, smc: Dict[str, Any], sigs: List[Dict[str, Any]], atr: float, price: float) -> List[Dict[str, Any]]:
    """Best confirmed setup per side (same filters as the dashboard's comboRows). Returns 0, 1 or 2 dicts."""
    n = len(df)
    if n < 20 or not (atr and atr > 0):
        return []
    H, L, C = df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)
    out: List[Dict[str, Any]] = []
    for low in (True, False):
        key, d = ("bottom", "bull") if low else ("top", "bear")
        sig, level, lv_since = smc.get(key + "_signal"), smc.get(key + "_signal_price"), smc.get(key + "_bars_since")
        if sig not in SMC_SIGNALS or not sig.endswith("LOW" if low else "HIGH") or level is None:
            continue
        cands = []
        for g in sigs:
            if g["since"] > WINDOW or g["dir"] != d or g["name"] not in REV[d] or g.get("confirm") == "failed":
                continue
            if lv_since is not None and lv_since >= 0 and g["since"] > lv_since:
                continue                                           # the pattern must come at or after the level was set
            end = n - 1 - g["since"]
            a = max(0, end - (g["len"] - 1))
            if end < 0:
                continue
            lo, hi = float(L[a:end + 1].min()), float(H[a:end + 1].max())
            ext = lo if low else hi
            if abs(ext - level) > COMBO_ATR * atr:
                continue                                           # not at the level
            pre = C[max(0, a - 8):a]
            if len(pre) < 4:
                continue
            avg = float(pre.mean())
            if not (((avg - ext) if low else (ext - avg)) >= COMBO_TREND * atr):
                continue                                           # no real move into the level
            if ((price - ext) if low else (ext - price)) > COMBO_RUN * atr:
                continue                                           # already ran away
            status = g.get("confirm") or "pending"
            failed, conf_idx = False, -1
            for i in range(end + 1, n):
                if (L[i] < ext) if low else (H[i] > ext):
                    failed = True                                  # a new low (high) after the pattern: the setup is dead
                    break
                if conf_idx < 0 and ((C[i] > hi) if low else (C[i] < lo)):
                    conf_idx = i                                   # the first later candle that closed beyond the pattern
                if status == "pending" and conf_idx == i:
                    status = "confirmed"
            if failed or status != "confirmed" or conf_idx < 0:
                continue
            cands.append({"g": g, "ext": ext, "conf_idx": conf_idx, "end": end, "low": low, "dir": d, "level": level})
        if cands:
            cands.sort(key=lambda c: (-c["g"]["score"], c["g"]["since"]))
            out.append(cands[0])
    return out


# ----------------------------------------------------------------------------- 2. build a plan from the levels
def build_plan(symbol: str, setup: Dict[str, Any], df: pd.DataFrame, lvls: List[Dict[str, Any]], atr: float,
               scanned_at: str, variant: str = "levels") -> Optional[Dict[str, Any]]:
    long_ = setup["low"]
    labels = df["Date"].dt.strftime(config.DATE_FMT)
    ci = setup["conf_idx"]
    entry = float(df["Close"].iloc[ci])
    sign = 1.0 if long_ else -1.0
    tag = variant
    if variant == "atr":                                           # no levels needed
        pattern_stop = setup["ext"] - 0.1 * atr if long_ else setup["ext"] + 0.1 * atr
        sl = min(entry - ATR_STOP * atr, pattern_stop) if long_ else max(entry + ATR_STOP * atr, pattern_stop)
        return _fixed_targets(symbol, setup, labels, ci, entry, sl, atr, scanned_at, "atr", None, None)
    if not lvls:
        return None
    gap = MIN_RISK_ATR * atr
    prices = sorted(lvls, key=lambda l: l["price"])
    below = [l for l in prices if l["price"] <= entry - gap]
    above = [l for l in prices if l["price"] >= entry + gap]
    stops, targets = (below, above) if long_ else (above, below)
    if not stops or not targets:
        return None
    sl_lv = stops[-1] if long_ else stops[0]                       # the nearest level on the wrong side
    tps = targets if long_ else targets[::-1]                      # nearest first
    tp1_lv, tp2_lv = tps[0], (tps[1] if len(tps) > 1 else None)
    buf = STOP_BUF_ATR * atr
    pattern_stop = setup["ext"] - 0.1 * atr if long_ else setup["ext"] + 0.1 * atr
    sl_level = sl_lv["price"] - buf if long_ else sl_lv["price"] + buf
    sl = min(sl_level, pattern_stop) if long_ else max(sl_level, pattern_stop)   # never tighter than the pattern's own extreme
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    tp1, tp2 = float(tp1_lv["price"]), (float(tp2_lv["price"]) if tp2_lv else None)
    if variant == "fixed":                                         # level-based stop, fixed targets
        return _fixed_targets(symbol, setup, labels, ci, entry, sl, atr, scanned_at, "fixed", sl_lv.get("grade"),
                              "pattern" if sl == pattern_stop and sl != sl_level else "level")
    rr1 = abs(tp1 - entry) / risk
    g = setup["g"]
    return {
        "variant": tag,
        "id": f"{symbol}|{'long' if long_ else 'short'}|{g['name']}|{g['time']}" + ("" if tag == "levels" else f"|{tag}"),
        "symbol": symbol, "side": "long" if long_ else "short", "pattern": g["name"], "pattern_time": g["time"],
        "entry_time": labels.iloc[ci], "entry": round(entry, 4), "sl": round(sl, 4), "tp1": round(tp1, 4),
        "tp2": None if tp2 is None else round(tp2, 4), "atr": round(atr, 4),
        "rr1": round(rr1, 2), "rr2": None if tp2 is None else round(abs(tp2 - entry) / risk, 2),
        "sl_basis": "pattern" if sl == pattern_stop and sl != sl_level else "level",
        "grade_sl": sl_lv.get("grade"), "grade_tp1": tp1_lv.get("grade"), "grade_tp2": tp2_lv.get("grade") if tp2_lv else None,
        "smc": setup.get("level"), "score": g.get("score"), "created": scanned_at,
        "status": "open", "tp1_time": None, "tp2_time": None, "closed_time": None, "r": None, "r_live": None,
    }


def _fixed_targets(symbol, setup, labels, ci, entry, sl, atr, scanned_at, variant, grade_sl, basis) -> Optional[Dict[str, Any]]:
    long_ = setup["low"]
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    sign = 1.0 if long_ else -1.0
    tp1, tp2 = entry + sign * FIXED_R[0] * risk, entry + sign * FIXED_R[1] * risk
    g = setup["g"]
    return {
        "variant": variant, "id": f"{symbol}|{'long' if long_ else 'short'}|{g['name']}|{g['time']}|{variant}",
        "symbol": symbol, "side": "long" if long_ else "short", "pattern": g["name"], "pattern_time": g["time"],
        "entry_time": labels.iloc[ci], "entry": round(entry, 4), "sl": round(sl, 4), "tp1": round(tp1, 4), "tp2": round(tp2, 4),
        "atr": round(atr, 4), "rr1": FIXED_R[0], "rr2": FIXED_R[1], "sl_basis": basis or "atr",
        "grade_sl": grade_sl, "grade_tp1": None, "grade_tp2": None,
        "smc": setup.get("level"), "score": g.get("score"), "created": scanned_at,
        "status": "open", "tp1_time": None, "tp2_time": None, "closed_time": None, "r": None, "r_live": None,
    }


# ----------------------------------------------------------------------------- 3. follow a plan with the candles
def evaluate(df: pd.DataFrame, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Replay the candles after entry. Returns the fields to update, or None when the entry candle is no longer in `df`."""
    labels = df["Date"].dt.strftime(config.DATE_FMT).to_numpy()
    hit = np.flatnonzero(labels == plan["entry_time"])
    if not len(hit):
        return None
    e = int(hit[0])
    H, L, C = df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)
    n = len(df)
    long_ = plan["side"] == "long"
    entry, sl0, tp1, tp2 = plan["entry"], plan["sl"], plan["tp1"], plan.get("tp2")
    risk = abs(entry - sl0)
    rr1 = abs(tp1 - entry) / risk
    rr2 = None if tp2 is None else abs(tp2 - entry) / risk
    sign = 1.0 if long_ else -1.0
    stop, tp1_hit = sl0, False
    upd: Dict[str, Any] = {"status": "open", "tp1_time": plan.get("tp1_time"), "tp2_time": plan.get("tp2_time"),
                           "closed_time": None, "r": None}
    for i in range(e + 1, n):
        t = labels[i]
        stopped = (L[i] <= stop) if long_ else (H[i] >= stop)
        if stopped:                                                # the stop comes first, always
            upd.update(status="be" if tp1_hit else "sl", closed_time=t, r=round(0.5 * rr1, 2) if tp1_hit else -1.0)
            return upd
        if not tp1_hit:
            if (H[i] >= tp1) if long_ else (L[i] <= tp1):
                tp1_hit, stop, upd["status"], upd["tp1_time"] = True, entry, "tp1", t
                continue                                           # nothing else is judged on this candle
        elif tp2 is not None and ((H[i] >= tp2) if long_ else (L[i] <= tp2)):
            upd.update(status="tp2", tp2_time=t, closed_time=t, r=round(0.5 * rr1 + 0.5 * rr2, 2))
            return upd
        if i - e >= MAX_AGE:                                       # too old: close at the last close
            mark = float(sign * (C[i] - entry) / risk)
            upd.update(status="expired", closed_time=t, r=round((0.5 * rr1 + 0.5 * mark) if tp1_hit else mark, 2))
            return upd
    mark = float(sign * (C[-1] - entry) / risk)
    upd["r_live"] = round((0.5 * rr1 + 0.5 * mark) if tp1_hit else mark, 2)
    return upd


# ----------------------------------------------------------------------------- 4. what scan.py calls per symbol
def make_plans(symbol: str, df: pd.DataFrame, smc: Dict[str, Any], sigs: List[Dict[str, Any]], atr: float, price: float,
               lvls: List[Dict[str, Any]], scanned_at: str) -> List[Dict[str, Any]]:
    out = []
    for st in setups(df, smc, sigs, atr, price):
        for variant in VARIANTS:                                   # the same setup, three different exit rules
            p = build_plan(symbol, st, df, lvls, atr, scanned_at, variant)
            if p is None:
                continue
            upd = evaluate(df, p)                                  # it may already have moved since the confirming candle
            if upd:
                p.update(upd)
            out.append(p)
    return out


def evaluate_open(df: pd.DataFrame, open_plans: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    res = {}
    for p in open_plans:
        upd = evaluate(df, p)
        if upd:
            res[p["id"]] = upd
    return res


# ----------------------------------------------------------------------------- 5. the saved file
def load(path=None) -> Dict[str, Any]:
    import json
    import os
    path = path or config.PLANS_JSON
    if not os.path.exists(path):
        return {"schema": 1, "updated": None, "plans": []}
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict) or not isinstance(d.get("plans"), list):
        raise ValueError(f"{path} must be an object with a 'plans' array")
    return d


def open_by_symbol(store: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p in store["plans"]:
        if p.get("status") in OPEN_STATES:
            out.setdefault(p["symbol"], []).append(p)
    return out


def merge(store: Dict[str, Any], new: List[Dict[str, Any]], evals: Dict[str, Dict[str, Any]], scanned_at: str) -> Dict[str, Any]:
    """Add plans seen for the first time (frozen), move the open ones forward, trim the old closed ones."""
    by_id = {p["id"]: p for p in store["plans"]}
    for p in new:
        if p["id"] not in by_id:                                   # an existing plan is NEVER redrawn
            by_id[p["id"]] = p
    for pid, upd in evals.items():
        if pid in by_id and by_id[pid].get("status") in OPEN_STATES:
            by_id[pid].update(upd)
    plans = list(by_id.values())
    opened = [p for p in plans if p.get("status") in OPEN_STATES]
    closed = sorted((p for p in plans if p.get("status") not in OPEN_STATES), key=lambda p: p.get("closed_time") or "")
    store["plans"] = sorted(opened + closed[-KEEP_CLOSED:], key=lambda p: p["entry_time"])
    store["updated"] = scanned_at
    return store


def save(store: Dict[str, Any], path=None) -> None:
    atomic_write_text(path or config.PLANS_JSON, dumps_strict(store, False))
