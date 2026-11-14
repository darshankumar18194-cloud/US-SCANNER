"""Candlestick pattern detectors (Group A of the Candlestick Pattern Handbook).

All size tests are relative to the 14-candle ATR, so they work for a $5 stock and a $500 stock alike.
Hourly US-equity candles almost never gap inside the session, so patterns that the Handbook describes with a gap
(Morning/Evening Star, Piercing Line, Dark Cloud Cover, Meeting Lines ...) are relaxed to "opens at or beyond the
previous close". Patterns whose DEFINITION is a gap (Kicker, Rising/Falling Window, Abandoned Baby, Tri-Star,
Two Black Gapping, Mat Hold) stay strict and will mostly fire on the 9:30 candle.

Not implemented (they are aliases of patterns already here, per the Handbook's own table):
  * "Last Engulfing Pattern"  = Bullish/Bearish Engulfing
  * "Counterattack Lines"     = Meeting Lines
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

import indicators as ind


class Meta(NamedTuple):
    group: str          # 1-candle | 2-candle | 3-candle | extended
    direction: str      # bull | bear | neutral
    stars: int          # Handbook reliability (extended-group patterns have no rating -> 2, 3 where it quotes 65-70%)
    stop: str           # all | middle | first | gap | open  (where the Handbook puts the stop)


META: Dict[str, Meta] = {}


def _m(name, group, direction, stars, stop="all"):
    META[name] = Meta(group, direction, stars, stop)


# ---- single candle
_m("Dragonfly Doji", "1-candle", "bull", 3); _m("Gravestone Doji", "1-candle", "bear", 3)
_m("Hammer", "1-candle", "bull", 4); _m("Hanging Man", "1-candle", "bear", 3)
_m("Inverted Hammer", "1-candle", "bull", 3); _m("Shooting Star", "1-candle", "bear", 3)
_m("Bullish Marubozu", "1-candle", "bull", 3); _m("Bearish Marubozu", "1-candle", "bear", 3)
_m("Bullish Spinning Top", "1-candle", "neutral", 2); _m("Bearish Spinning Top", "1-candle", "neutral", 2)
_m("Bullish Belt Hold", "1-candle", "bull", 2); _m("Bearish Belt Hold", "1-candle", "bear", 2)
_m("Doji", "1-candle", "neutral", 2); _m("Long-Legged Doji", "1-candle", "neutral", 2)
_m("High Wave Candle", "1-candle", "neutral", 2)
# ---- two candles
_m("Bullish Engulfing", "2-candle", "bull", 4); _m("Bearish Engulfing", "2-candle", "bear", 4)
_m("Piercing Line", "2-candle", "bull", 3); _m("Dark Cloud Cover", "2-candle", "bear", 3)
_m("Bullish Harami", "2-candle", "bull", 2); _m("Bearish Harami", "2-candle", "bear", 2)
_m("Tweezer Bottom", "2-candle", "bull", 2); _m("Tweezer Top", "2-candle", "bear", 2)
_m("Bullish Kicker", "2-candle", "bull", 4, "open"); _m("Bearish Kicker", "2-candle", "bear", 4, "open")
_m("Rising Window", "2-candle", "bull", 3, "gap"); _m("Falling Window", "2-candle", "bear", 3, "gap")
_m("Bullish Harami Cross", "2-candle", "bull", 3); _m("Bearish Harami Cross", "2-candle", "bear", 3)
# ---- three candles
_m("Morning Star", "3-candle", "bull", 4, "middle"); _m("Evening Star", "3-candle", "bear", 4, "middle")
_m("Bullish Abandoned Baby", "3-candle", "bull", 5, "middle"); _m("Bearish Abandoned Baby", "3-candle", "bear", 5, "middle")
_m("Three White Soldiers", "3-candle", "bull", 4, "first"); _m("Three Black Crows", "3-candle", "bear", 4, "first")
_m("Three Inside Up", "3-candle", "bull", 3); _m("Three Inside Down", "3-candle", "bear", 3)
_m("Three Outside Up", "3-candle", "bull", 4); _m("Three Outside Down", "3-candle", "bear", 4)
_m("Bullish Tri-Star", "3-candle", "bull", 3, "middle"); _m("Bearish Tri-Star", "3-candle", "bear", 3, "middle")
_m("Stick Sandwich", "3-candle", "bull", 2)
# ---- extended (4-5 candles and the Handbook's cheat-sheet patterns)
_m("Rising Three Methods", "extended", "bull", 3); _m("Falling Three Methods", "extended", "bear", 3)
_m("Mat Hold", "extended", "bull", 3)
_m("Ladder Bottom", "extended", "bull", 2); _m("Ladder Top", "extended", "bear", 2)
_m("Concealing Baby Swallow", "extended", "bull", 2)
_m("Bullish Meeting Lines", "extended", "bull", 2); _m("Bearish Meeting Lines", "extended", "bear", 2)
_m("On-Neck Line", "extended", "bear", 2); _m("In-Neck Line", "extended", "bear", 2)
_m("Thrusting Line", "extended", "bear", 2)
_m("Bullish Doji Star", "extended", "bull", 3); _m("Bearish Doji Star", "extended", "bear", 3)
_m("Homing Pigeon", "extended", "bull", 2); _m("Matching Low", "extended", "bull", 2)
_m("Bullish Separating Lines", "extended", "bull", 2); _m("Bearish Separating Lines", "extended", "bear", 2)
_m("Two Black Gapping", "extended", "bear", 3, "gap")

MIN_CONTEXT = 60          # candles of history needed before we evaluate a candle


class Candles:
    """OHLCV arrays plus the helpers every detector shares."""

    def __init__(self, df: pd.DataFrame):
        self.times = pd.DatetimeIndex(df["Date"])
        self.o = df["Open"].to_numpy(float)
        self.h = df["High"].to_numpy(float)
        self.l = df["Low"].to_numpy(float)
        self.c = df["Close"].to_numpy(float)
        self.v = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.zeros(len(df))
        self.n = len(df)
        self.tr = ind.true_range(self.h, self.l, self.c)
        self.atr = ind.atr(self.h, self.l, self.c, 14)
        self.volr = ind.slot_rel_volume(self.times, self.v)

    def trend_before(self, start: int, length: int = 8) -> str:
        """'down' / 'up' / 'flat' judged on the candles BEFORE `start`."""
        j = start - 1
        if j - length < 0 or np.isnan(self.atr[j]):
            return "flat"
        move = self.c[j] - self.c[j - length]
        avg = float(np.mean(self.c[j - length:j]))
        if move <= -1.0 * self.atr[j] and self.c[j] < avg:
            return "down"
        if move >= 1.0 * self.atr[j] and self.c[j] > avg:
            return "up"
        return "flat"


class Hit(NamedTuple):
    name: str
    length: int


def candle_hits(x: Candles, i: int) -> List[Hit]:
    """Every candlestick pattern whose LAST candle is index i."""
    hits: List[Hit] = []
    a = x.atr[i]
    if np.isnan(a) or a <= 0 or i < 4:
        return hits
    eps = 0.03 * a
    o, h, l, c = x.o, x.h, x.l, x.c

    def body(k): return abs(c[k] - o[k])
    def rng(k): return h[k] - l[k]
    def up(k): return c[k] > o[k]
    def dn(k): return c[k] < o[k]
    def uw(k): return h[k] - max(o[k], c[k])
    def lw(k): return min(o[k], c[k]) - l[k]
    def top(k): return max(o[k], c[k])
    def bot(k): return min(o[k], c[k])
    def is_doji(k, min_range=0.15): return rng(k) > 0 and body(k) <= 0.1 * rng(k) and rng(k) >= min_range * a
    def add(name, n): hits.append(Hit(name, n))

    # ------------------------------------------------------------------ 1 candle
    B, R, UW, LW = body(i), rng(i), uw(i), lw(i)
    tr1 = x.trend_before(i)
    if R > 0:
        if B <= 0.1 * R and R >= 0.3 * a:                                  # the doji family (exclusive)
            if UW <= 0.1 * R and tr1 == "down":
                add("Dragonfly Doji", 1)
            elif LW <= 0.1 * R and tr1 == "up":
                add("Gravestone Doji", 1)
            elif UW >= 0.3 * R and LW >= 0.3 * R and R >= 1.0 * a:
                add("Long-Legged Doji", 1)
            else:
                add("Doji", 1)
        elif UW <= 0.05 * R and LW <= 0.05 * R and B >= 0.8 * a:
            add("Bullish Marubozu" if up(i) else "Bearish Marubozu", 1)
        elif B >= 0.8 * a and up(i) and LW <= 0.05 * R and 0.05 * R < UW <= 0.25 * R:
            add("Bullish Belt Hold", 1)
        elif B >= 0.8 * a and dn(i) and UW <= 0.05 * R and 0.05 * R < LW <= 0.25 * R:
            add("Bearish Belt Hold", 1)
        elif B > 0.1 * R and R >= 0.5 * a and LW >= 2 * B and UW <= 0.15 * R:
            if tr1 == "down":
                add("Hammer", 1)
            elif tr1 == "up":
                add("Hanging Man", 1)
        elif B > 0.1 * R and R >= 0.5 * a and UW >= 2 * B and LW <= 0.15 * R:
            if tr1 == "down":
                add("Inverted Hammer", 1)
            elif tr1 == "up":
                add("Shooting Star", 1)
        elif B <= 0.25 * R and UW >= 0.3 * R and LW >= 0.3 * R and R >= 1.2 * a:
            add("High Wave Candle", 1)
        elif 0.1 * R < B <= 0.3 * R and UW >= 0.2 * R and LW >= 0.2 * R and R >= 0.4 * a \
                and min(UW, LW) >= 0.5 * max(UW, LW):
            add("Bullish Spinning Top" if up(i) else "Bearish Spinning Top", 1)

    # ------------------------------------------------------------------ 2 candles  (p = i-1, k = i)
    p, k = i - 1, i
    tr2 = x.trend_before(p)
    B1, B2, R2 = body(p), body(k), rng(k)
    if tr2 == "down" and dn(p) and up(k) and o[k] <= c[p] + eps and c[k] >= o[p] - eps and B2 > B1 and B2 >= 0.5 * a:
        add("Bullish Engulfing", 2)
    if tr2 == "up" and up(p) and dn(k) and o[k] >= c[p] - eps and c[k] <= o[p] + eps and B2 > B1 and B2 >= 0.5 * a:
        add("Bearish Engulfing", 2)
    if tr2 == "down" and dn(p) and B1 >= 0.5 * a and up(k) and B2 >= 0.3 * a and o[k] <= c[p] + eps \
            and (o[p] + c[p]) / 2 < c[k] < o[p] - eps:
        add("Piercing Line", 2)
    if tr2 == "up" and up(p) and B1 >= 0.5 * a and dn(k) and B2 >= 0.3 * a and o[k] >= c[p] - eps \
            and o[p] + eps < c[k] < (o[p] + c[p]) / 2:
        add("Dark Cloud Cover", 2)
    inside = lambda: top(k) <= top(p) + eps and bot(k) >= bot(p) - eps
    if R2 > 0 and B1 >= 0.6 * a and inside():
        if B2 <= 0.1 * R2:                                                  # a doji inside the body
            if tr2 == "down" and dn(p):
                add("Bullish Harami Cross", 2)
            elif tr2 == "up" and up(p):
                add("Bearish Harami Cross", 2)
        elif B2 <= 0.6 * B1:
            if tr2 == "down" and dn(p) and up(k):
                add("Bullish Harami", 2)
            elif tr2 == "up" and up(p) and dn(k):
                add("Bearish Harami", 2)
            elif tr2 == "down" and dn(p) and dn(k):
                add("Homing Pigeon", 2)
    if tr2 == "down" and abs(l[p] - l[k]) <= 0.08 * a and rng(p) >= 0.3 * a and R2 >= 0.3 * a:
        add("Tweezer Bottom", 2)
    if tr2 == "up" and abs(h[p] - h[k]) <= 0.08 * a and rng(p) >= 0.3 * a and R2 >= 0.3 * a:
        add("Tweezer Top", 2)
    if dn(p) and up(k) and B1 >= 0.4 * a and B2 >= 0.4 * a and o[k] >= o[p] + 0.1 * a:
        add("Bullish Kicker", 2)
    if up(p) and dn(k) and B1 >= 0.4 * a and B2 >= 0.4 * a and o[k] <= o[p] - 0.1 * a:
        add("Bearish Kicker", 2)
    if tr2 == "up" and l[k] - h[p] >= 0.1 * a:
        add("Rising Window", 2)
    if tr2 == "down" and l[p] - h[k] >= 0.1 * a:
        add("Falling Window", 2)
    if R2 > 0 and is_doji(k) and B1 >= 0.6 * a and not inside():            # gapped doji (precursor of a star)
        if tr2 == "down" and dn(p) and top(k) <= c[p] + eps:
            add("Bullish Doji Star", 2)
        elif tr2 == "up" and up(p) and bot(k) >= c[p] - eps:
            add("Bearish Doji Star", 2)
    if tr2 == "down" and dn(p) and up(k) and B1 >= 0.5 * a and B2 >= 0.5 * a and o[k] <= c[p] - 0.1 * a \
            and abs(c[k] - c[p]) <= 0.1 * a:
        add("Bullish Meeting Lines", 2)
    if tr2 == "up" and up(p) and dn(k) and B1 >= 0.5 * a and B2 >= 0.5 * a and o[k] >= c[p] + 0.1 * a \
            and abs(c[k] - c[p]) <= 0.1 * a:
        add("Bearish Meeting Lines", 2)
    if tr2 == "down" and dn(p) and dn(k) and B1 >= 0.3 * a and B2 >= 0.3 * a and abs(c[p] - c[k]) <= 0.08 * a:
        add("Matching Low", 2)
    if B1 >= 0.4 * a and B2 >= 0.4 * a and abs(o[p] - o[k]) <= 0.08 * a:
        if tr2 == "up" and dn(p) and up(k):
            add("Bullish Separating Lines", 2)
        elif tr2 == "down" and up(p) and dn(k):
            add("Bearish Separating Lines", 2)
    if tr2 == "down" and dn(p) and B1 >= 0.6 * a and up(k) and o[k] <= c[p] + eps:
        if abs(c[k] - l[p]) <= 0.1 * a and c[k] <= c[p]:
            add("On-Neck Line", 2)
        elif c[p] < c[k] <= c[p] + 0.15 * a:
            add("In-Neck Line", 2)
        elif c[p] + 0.15 * a < c[k] < (o[p] + c[p]) / 2:
            add("Thrusting Line", 2)

    # ------------------------------------------------------------------ 3 candles  (s = i-2, m = i-1, e = i)
    s, m, e = i - 2, i - 1, i
    tr3 = x.trend_before(s)
    Bs, Bm, Be = body(s), body(m), body(e)
    if tr3 == "down" and dn(s) and Bs >= 0.5 * a and is_doji(m, 0.1) and h[m] < l[s] and h[m] < l[e] \
            and up(e) and Be >= 0.5 * a:
        add("Bullish Abandoned Baby", 3)
    elif tr3 == "down" and dn(s) and Bs >= 0.6 * a and Bm <= 0.35 * Bs and top(m) <= c[s] + 0.15 * a \
            and up(e) and Be >= 0.5 * a and c[e] > (o[s] + c[s]) / 2:
        add("Morning Star", 3)
    if tr3 == "up" and up(s) and Bs >= 0.5 * a and is_doji(m, 0.1) and l[m] > h[s] and l[m] > h[e] \
            and dn(e) and Be >= 0.5 * a:
        add("Bearish Abandoned Baby", 3)
    elif tr3 == "up" and up(s) and Bs >= 0.6 * a and Bm <= 0.35 * Bs and bot(m) >= c[s] - 0.15 * a \
            and dn(e) and Be >= 0.5 * a and c[e] < (o[s] + c[s]) / 2:
        add("Evening Star", 3)
    trio = (s, m, e)
    if tr3 != "up" and all(up(q) and body(q) >= 0.5 * a and uw(q) <= 0.25 * body(q) for q in trio) \
            and c[m] > c[s] and c[e] > c[m] and o[s] - eps <= o[m] <= c[s] + eps and o[m] - eps <= o[e] <= c[m] + eps:
        add("Three White Soldiers", 3)
    if tr3 != "down" and all(dn(q) and body(q) >= 0.5 * a and lw(q) <= 0.25 * body(q) for q in trio) \
            and c[m] < c[s] and c[e] < c[m] and c[s] - eps <= o[m] <= o[s] + eps and c[m] - eps <= o[e] <= o[m] + eps:
        add("Three Black Crows", 3)
    if tr3 == "down" and dn(s) and Bs >= 0.6 * a and up(m) and top(m) <= o[s] + eps and bot(m) >= c[s] - eps \
            and up(e) and c[e] > h[s]:
        add("Three Inside Up", 3)
    if tr3 == "up" and up(s) and Bs >= 0.6 * a and dn(m) and top(m) <= c[s] + eps and bot(m) >= o[s] - eps \
            and dn(e) and c[e] < l[s]:
        add("Three Inside Down", 3)
    if tr3 == "down" and dn(s) and up(m) and o[m] <= c[s] + eps and c[m] >= o[s] - eps and Bm > Bs \
            and Bm >= 0.5 * a and up(e) and c[e] > c[m]:
        add("Three Outside Up", 3)
    if tr3 == "up" and up(s) and dn(m) and o[m] >= c[s] - eps and c[m] <= o[s] + eps and Bm > Bs \
            and Bm >= 0.5 * a and dn(e) and c[e] < c[m]:
        add("Three Outside Down", 3)
    if all(is_doji(q) for q in trio):
        if tr3 == "down" and h[m] < l[s] and h[m] < l[e]:
            add("Bullish Tri-Star", 3)
        elif tr3 == "up" and l[m] > h[s] and l[m] > h[e]:
            add("Bearish Tri-Star", 3)
    if tr3 == "down" and dn(s) and up(m) and dn(e) and Bs >= 0.3 * a and Be >= 0.3 * a \
            and c[m] > o[s] and abs(c[e] - c[s]) <= 0.1 * a:
        add("Stick Sandwich", 3)
    if tr3 == "down" and dn(m) and dn(e) and h[m] < l[s] - 0.05 * a and l[e] < l[m] and Bm >= 0.3 * a and Be >= 0.3 * a:
        add("Two Black Gapping", 3)

    # ------------------------------------------------------------------ 4-5 candles
    if i >= 4:
        q = [i - 4, i - 3, i - 2, i - 1, i]
        c1, c2, c3, c4, c5 = q
        tr5 = x.trend_before(c1)
        # Rising / Falling Three Methods
        if tr5 != "down" and up(c1) and body(c1) >= 0.6 * a and up(c5) and body(c5) >= 0.6 * a and c[c5] > c[c1] \
                and all(dn(z) and h[z] <= h[c1] + eps and l[z] >= l[c1] - eps and body(z) <= 0.5 * body(c1)
                        for z in (c2, c3, c4)):
            add("Rising Three Methods", 5)
        if tr5 != "up" and dn(c1) and body(c1) >= 0.6 * a and dn(c5) and body(c5) >= 0.6 * a and c[c5] < c[c1] \
                and all(up(z) and h[z] <= h[c1] + eps and l[z] >= l[c1] - eps and body(z) <= 0.5 * body(c1)
                        for z in (c2, c3, c4)):
            add("Falling Three Methods", 5)
        # Mat Hold: gap up, small pullback that holds above the first body's midpoint, then a new high
        mid1 = (o[c1] + c[c1]) / 2
        if tr5 != "down" and up(c1) and body(c1) >= 0.6 * a and o[c2] > c[c1] + 0.1 * a \
                and all(dn(z) and body(z) <= 0.5 * body(c1) and l[z] >= mid1 for z in (c2, c3, c4)) \
                and up(c5) and body(c5) >= 0.6 * a and c[c5] > max(h[c2], h[c3], h[c4]):
            add("Mat Hold", 5)
        # Ladder Bottom / Top
        if tr5 != "up" and all(dn(z) and body(z) >= 0.4 * a for z in (c1, c2, c3)) and c[c2] < c[c1] and c[c3] < c[c2] \
                and c[c1] - eps <= o[c2] <= o[c1] + eps and c[c2] - eps <= o[c3] <= o[c2] + eps \
                and dn(c4) and uw(c4) >= body(c4) and up(c5) and o[c5] > o[c4] and c[c5] > h[c4]:
            add("Ladder Bottom", 5)
        if tr5 != "down" and all(up(z) and body(z) >= 0.4 * a for z in (c1, c2, c3)) and c[c2] > c[c1] and c[c3] > c[c2] \
                and o[c1] - eps <= o[c2] <= c[c1] + eps and o[c2] - eps <= o[c3] <= c[c2] + eps \
                and up(c4) and lw(c4) >= body(c4) and dn(c5) and o[c5] < o[c4] and c[c5] < l[c4]:
            add("Ladder Top", 5)
        # Concealing Baby Swallow (classical definition: a bullish bottom reversal of four black candles)
        a1, a2, a3, a4 = c2, c3, c4, c5
        tr4 = x.trend_before(a1)
        maru = lambda z: dn(z) and uw(z) <= 0.1 * rng(z) and lw(z) <= 0.1 * rng(z) and body(z) >= 0.5 * a
        if tr4 == "down" and maru(a1) and maru(a2) and c[a2] < c[a1] and dn(a3) and o[a3] < c[a2] \
                and c[a2] < h[a3] <= o[a2] and dn(a4) and o[a4] > h[a3] and c[a4] < l[a3]:
            add("Concealing Baby Swallow", 4)
    return hits


def stop_price(x: Candles, name: str, i: int, length: int) -> Optional[float]:
    """Stop reference from the Handbook for a pattern that ends at candle i."""
    meta = META[name]
    if meta.direction == "neutral":
        return None
    s = i - length + 1
    rng_ = range(s, i + 1)
    lows = [x.l[z] for z in rng_]
    highs = [x.h[z] for z in rng_]
    bull = meta.direction == "bull"
    if meta.stop == "middle" and length >= 3:
        z = s + 1
        return float(x.l[z] if bull else x.h[z])
    if meta.stop == "first":
        return float(x.l[s] if bull else x.h[s])
    if meta.stop == "open":                                     # Kicker: the open of the 2nd candle
        return float(x.o[i])
    if meta.stop == "gap":                                      # Windows: the far edge of the gap
        return float(x.h[i - 1] if bull else x.l[i - 1]) if length == 2 else float(x.h[s] if bull else x.l[s])
    return float(min(lows) if bull else max(highs))
