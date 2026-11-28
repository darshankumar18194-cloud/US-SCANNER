"""Everything that is NOT a plain candlestick pattern, plus the glue that turns raw detections into stored signals.

EVENTS (stored with a 'since' = candles ago, like patterns):
  range      Inside Bar, NR4, NR7
  squeeze    Bollinger Squeeze Breakout, TTM Squeeze Fired, ATR Expansion Breakout, Donchian Squeeze Breakout
  volume     Volume Spike Breakout, Pocket Pivot
  trend      ADX Trend Start, MA Ribbon Fan-Out
  structure  Fair Value Gap, Turtle Soup, Undercut & Rally, Ichimoku Cloud Twist

STATES (current readings, stored once per symbol in `context`):
  Bollinger / TTM squeeze on, ATR contraction, historical-volatility contraction, volume dry-up,
  OBV and A/D divergence, CMF, Donchian squeeze, ADX / Choppiness, MA-ribbon squeeze, Ichimoku position.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import indicators as ind
import patterns as P

EVENT_META: Dict[str, Tuple[str, int]] = {      # name -> (group, stars)
    "Inside Bar": ("range", 1), "NR4": ("range", 2), "NR7": ("range", 2),
    "Bollinger Squeeze Breakout": ("squeeze", 3), "TTM Squeeze Fired": ("squeeze", 4),
    "ATR Expansion Breakout": ("squeeze", 3), "Donchian Squeeze Breakout": ("squeeze", 3),
    "Volume Spike Breakout": ("volume", 3), "Pocket Pivot": ("volume", 3),
    "ADX Trend Start": ("trend", 3), "MA Ribbon Fan-Out": ("trend", 3),
    "Fair Value Gap": ("structure", 2), "Turtle Soup": ("structure", 3),
    "Undercut & Rally": ("structure", 3), "Ichimoku Cloud Twist": ("structure", 2),
}
GROUPS = ["1-candle", "2-candle", "3-candle", "extended", "squeeze", "volume", "trend", "structure", "range"]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _r(v, nd=4):
    """Round to nd places; NaN / inf / None -> None (strict JSON has no NaN)."""
    try:
        if v is None or not np.isfinite(v):
            return None
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


class Engine:
    """Precomputes every series once for one symbol."""

    def __init__(self, df: pd.DataFrame):
        self.x = x = P.Candles(df)
        o, h, l, c, v = x.o, x.h, x.l, x.c, x.v
        self.n = x.n
        self.atrp = x.atr / c
        self.bb_mid, self.bb_up, self.bb_lo, self.bw = ind.bollinger(c)
        _, self.kc_up, self.kc_lo = ind.keltner(h, l, c)
        with np.errstate(invalid="ignore"):
            self.ttm_on = (self.bb_up < self.kc_up) & (self.bb_lo > self.kc_lo)
        self.mom = ind.ttm_momentum(h, l, c)
        self.adx, self.pdi, self.mdi = ind.adx(h, l, c)
        self.chop = ind.choppiness(h, l, c)
        self.hv = ind.hist_vol(c)
        self.hh = ind.rolling_max(h, config.BREAKOUT_LOOKBACK)
        self.ll = ind.rolling_min(l, config.BREAKOUT_LOOKBACK)
        self.hh_prev = np.roll(self.hh, 1)
        self.ll_prev = np.roll(self.ll, 1)
        self.hh_prev[0] = self.ll_prev[0] = np.nan
        self.dc_width = (self.hh - self.ll) / c
        self.ma = {k: ind.sma(c, k) for k in (5, 10, 20, 50)}
        stack = np.vstack(list(self.ma.values()))
        with np.errstate(invalid="ignore"):
            self.rib_w = (stack.max(axis=0) - stack.min(axis=0)) / c     # NaN wherever any MA is NaN
        self.tenkan, self.kijun, self.span_a, self.span_b = ind.ichimoku(h, l, c)
        self.obv = ind.obv(c, v)
        self.adl = ind.ad_line(h, l, c, v)
        self.cmf = ind.cmf(h, l, c, v)
        self._cache: Dict[Tuple[str, int], bool] = {}

    # ---- compression flags (cached)
    def _low(self, key: str, series: np.ndarray, j: int) -> bool:
        if j < 0:
            return False
        k = (key, j)
        if k not in self._cache:
            self._cache[k] = ind.is_low(series, j)
        return self._cache[k]

    def bb_sq(self, j): return self._low("bb", self.bw, j)
    def atr_low(self, j): return self._low("atr", self.atrp, j)
    def dc_sq(self, j): return self._low("dc", self.dc_width, j)
    def rib_sq(self, j): return self._low("rib", self.rib_w, j)
    def hv_low(self, j): return self._low("hv", self.hv, j)

    def vdu(self, j) -> bool:
        if j < 4:
            return False
        w = self.x.volr[j - 4: j + 1]
        w = w[~np.isnan(w)]
        return len(w) >= 3 and float(np.median(w)) <= config.VDU_RATIO

    def ordered(self, j) -> int:
        """+1 when MA5>MA10>MA20>MA50, -1 when reversed, else 0."""
        m = [self.ma[k][j] for k in (5, 10, 20, 50)]
        if any(np.isnan(m)):
            return 0
        if m[0] > m[1] > m[2] > m[3]:
            return 1
        if m[0] < m[1] < m[2] < m[3]:
            return -1
        return 0


# ----------------------------------------------------------------------------- events at one candle
def event_hits(e: Engine, i: int) -> List[dict]:
    """Events whose trigger candle is i. Each: {name, dir, extra?}."""
    x = e.x
    o, h, l, c = x.o, x.h, x.l, x.c
    a = x.atr[i]
    out: List[dict] = []
    if np.isnan(a) or a <= 0 or i < 55:
        return out
    n = e.n

    def add(name, d, **extra):
        out.append({"name": name, "dir": d, **extra})

    # ---- range
    if h[i] < h[i - 1] and l[i] > l[i - 1] and (h[i - 1] - l[i - 1]) >= 0.5 * a:
        add("Inside Bar", "neutral")
    rg = h - l
    if rg[i] > 0:
        if rg[i] < np.min(rg[i - 3:i]):
            add("NR4", "neutral")
        if rg[i] < np.min(rg[i - 6:i]):
            add("NR7", "neutral")

    # ---- squeeze family
    if any(e.bb_sq(j) for j in range(i - 3, i)) and not np.isnan(e.bb_up[i]):
        if c[i] > e.bb_up[i] and c[i - 1] <= e.bb_up[i - 1]:
            add("Bollinger Squeeze Breakout", "bull")
        elif c[i] < e.bb_lo[i] and c[i - 1] >= e.bb_lo[i - 1]:
            add("Bollinger Squeeze Breakout", "bear")
    if e.ttm_on[i - 1] and not e.ttm_on[i] and not np.isnan(e.mom[i]) and e.mom[i] != 0:
        add("TTM Squeeze Fired", "bull" if e.mom[i] > 0 else "bear")
    if any(e.atr_low(j) for j in range(i - 10, i)) and not np.isnan(x.atr[i - 1]) and x.atr[i - 1] > 0 \
            and x.tr[i] >= 1.5 * x.atr[i - 1] and not np.isnan(e.hh_prev[i]):
        if c[i] > e.hh_prev[i]:
            add("ATR Expansion Breakout", "bull")
        elif c[i] < e.ll_prev[i]:
            add("ATR Expansion Breakout", "bear")
    if any(e.dc_sq(j) for j in range(i - 5, i)) and not np.isnan(e.hh_prev[i]):
        if c[i] > e.hh_prev[i] and c[i - 1] <= e.hh_prev[i]:
            add("Donchian Squeeze Breakout", "bull")
        elif c[i] < e.ll_prev[i] and c[i - 1] >= e.ll_prev[i]:
            add("Donchian Squeeze Breakout", "bear")

    # ---- volume
    vr = x.volr[i]
    if not np.isnan(vr) and vr >= config.VOL_SPIKE_RATIO and not np.isnan(e.hh_prev[i]) \
            and any(e.vdu(j) for j in range(i - 10, i)):
        if c[i] > e.hh_prev[i]:
            add("Volume Spike Breakout", "bull")
        elif c[i] < e.ll_prev[i]:
            add("Volume Spike Breakout", "bear")
    sma50 = e.ma[50][i]
    if not np.isnan(vr) and vr >= 1.0 and not np.isnan(sma50) and c[i] > o[i] and c[i] > c[i - 1] \
            and c[i] >= sma50 and c[i] - sma50 <= 3 * a:
        lb = config.PIVOT_VOL_LOOKBACK
        down = [x.volr[j] for j in range(i - lb, i) if c[j] < c[j - 1] and not np.isnan(x.volr[j])]
        if down and vr > max(down):
            add("Pocket Pivot", "bull")

    # ---- trend
    ad = e.adx
    if not np.isnan(ad[i]) and not np.isnan(ad[i - 1]) and ad[i - 1] < 25 <= ad[i] \
            and np.nanmin(ad[i - 10:i]) < 20 and not np.isnan(e.pdi[i]):
        add("ADX Trend Start", "bull" if e.pdi[i] > e.mdi[i] else "bear")
    od, op = e.ordered(i), e.ordered(i - 1)
    if od != 0 and op != od and e.rib_w[i] > e.rib_w[i - 1] and any(e.rib_sq(j) for j in range(i - 15, i)):
        add("MA Ribbon Fan-Out", "bull" if od > 0 else "bear")

    # ---- structure
    body = lambda k: abs(c[k] - o[k])
    if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= config.FVG_MIN_ATR * a and c[i - 1] > o[i - 1] and body(i - 1) >= 0.5 * a:
        lo_, hi_ = h[i - 2], l[i]
        later = l[i + 1:] if i + 1 < n else np.array([])
        st = "open" if later.size == 0 or later.min() > hi_ else ("filled" if later.min() <= lo_ else "partial")
        add("Fair Value Gap", "bull", zone=[_r(lo_), _r(hi_)], status=st)
    if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= config.FVG_MIN_ATR * a and c[i - 1] < o[i - 1] and body(i - 1) >= 0.5 * a:
        lo_, hi_ = h[i], l[i - 2]
        later = h[i + 1:] if i + 1 < n else np.array([])
        st = "open" if later.size == 0 or later.max() < lo_ else ("filled" if later.max() >= hi_ else "partial")
        add("Fair Value Gap", "bear", zone=[_r(lo_), _r(hi_)], status=st)
    W = config.BREAKOUT_LOOKBACK
    lvl1_lo, lvl1_hi = np.min(l[i - W:i]), np.max(h[i - W:i])            # one-candle sweep
    lvl2_lo, lvl2_hi = np.min(l[i - W - 1:i - 1]), np.max(h[i - W - 1:i - 1])   # two-candle version
    if (l[i] < lvl1_lo - 0.05 * a and c[i] > lvl1_lo and c[i] > o[i]) or \
            (c[i - 1] < lvl2_lo and c[i] > lvl2_lo and c[i] > o[i]):
        add("Turtle Soup", "bull")
    if (h[i] > lvl1_hi + 0.05 * a and c[i] < lvl1_hi and c[i] < o[i]) or \
            (c[i - 1] > lvl2_hi and c[i] < lvl2_hi and c[i] < o[i]):
        add("Turtle Soup", "bear")
    if c[i] > o[i] and not np.isnan(vr) and vr >= 1.2:                    # Undercut & Rally
        for j in range(i - 3, i):
            base = np.min(l[j - 40:j])
            if l[j] < base - 0.1 * a and c[i] > base and c[i - 1] <= base:
                add("Undercut & Rally", "bull")
                break
    d0, d1 = e.span_a[i] - e.span_b[i], e.span_a[i - 1] - e.span_b[i - 1]
    if not np.isnan(d0) and not np.isnan(d1) and d1 != 0 and np.sign(d0) != np.sign(d1) and d0 != 0:
        ca, cb = e.span_a[i - 26], e.span_b[i - 26]
        pos = "unknown" if np.isnan(ca) or np.isnan(cb) else (
            "above" if c[i] > max(ca, cb) else "below" if c[i] < min(ca, cb) else "inside")
        add("Ichimoku Cloud Twist", "bull" if d0 > 0 else "bear", price_vs_cloud=pos)
    return out


# ----------------------------------------------------------------------------- stored signals
def _confirm(x: P.Candles, d: str, i: int, ph: float, pl: float, stop: Optional[float]) -> Optional[str]:
    """Handbook confirmation rule: a later candle must CLOSE beyond the pattern's extreme."""
    if d == "neutral":
        return None
    if i == x.n - 1:
        return "pending"
    inval = stop if stop is not None else (pl if d == "bull" else ph)
    for k in range(i + 1, x.n):
        if d == "bull":
            if x.c[k] > ph:
                return "confirmed"
            if x.c[k] < inval:
                return "failed"
        else:
            if x.c[k] < pl:
                return "confirmed"
            if x.c[k] > inval:
                return "failed"
    return "pending"


def build_signals(e: Engine) -> List[dict]:
    x = e.x
    n = x.n
    first = max(P.MIN_CONTEXT, n - config.STORE_BARS)
    sigs: List[dict] = []
    for i in range(first, n):
        since = n - 1 - i
        t = x.times[i].strftime(config.DATE_FMT)
        base = {"since": since, "time": t, "price": _r(x.c[i]), "vol": _r(x.volr[i], 2)}
        for hit in P.candle_hits(x, i):
            m = P.META[hit.name]
            s0 = i - hit.length + 1
            stop = P.stop_price(x, hit.name, i, hit.length)
            ph, pl = float(np.max(x.h[s0:i + 1])), float(np.min(x.l[s0:i + 1]))
            sigs.append({"name": hit.name, "group": m.group, "dir": m.direction,
                         "len": hit.length, "stars": m.stars, "stop": _r(stop), **base,
                         "trend_before": x.trend_before(s0),
                         "confirm": _confirm(x, m.direction, i, ph, pl, stop)})
        for ev in event_hits(e, i):
            group, stars = EVENT_META[ev["name"]]
            if group == "range" and since > config.RANGE_KEEP_BARS:
                continue
            extra = {k: v for k, v in ev.items() if k not in ("name", "dir")}
            sigs.append({"name": ev["name"], "group": group, "dir": ev["dir"],
                         "len": 1, "stars": stars, "stop": None, **base, "confirm": None, **extra})
    return sigs


# ----------------------------------------------------------------------------- current-state readings
def _divergence(price: np.ndarray, flow: np.ndarray, vol: np.ndarray, lookback: int, atr: float) -> str:
    """Price drifting sideways/slightly against the flow line -> hidden accumulation / distribution."""
    if len(price) <= lookback or atr <= 0:
        return "none"
    dp = price[-1] - price[-1 - lookback]
    net = (flow[-1] - flow[-1 - lookback]) / max(float(np.mean(vol[-lookback:])) * lookback, 1e-9)
    if -4 * atr <= dp <= 1.0 * atr and net >= 0.15:
        return "bullish"
    if -1.0 * atr <= dp <= 4 * atr and net <= -0.15:
        return "bearish"
    return "none"


def build_context(e: Engine, sigs: List[dict]) -> dict:
    x = e.x
    i = e.n - 1
    lb = config.DIV_LOOKBACK

    def streak(flags):
        k = 0
        while i - k >= 0 and flags(i - k):
            k += 1
        return k

    recent = [s for s in sigs if s["since"] <= 3]
    fired = any(s["name"] in ("TTM Squeeze Fired", "Bollinger Squeeze Breakout") for s in recent)
    bb_on, ttm_on = bool(e.bb_sq(i)), bool(e.ttm_on[i])
    squeeze_state = "fired" if fired else "on" if (bb_on or ttm_on) else "off"
    cmf_v = e.cmf[i]
    adx_v = e.adx[i]
    chop_v = e.chop[i]
    cloud_a, cloud_b = e.span_a[i - 26] if i >= 26 else np.nan, e.span_b[i - 26] if i >= 26 else np.nan
    cloud = "unknown" if np.isnan(cloud_a) or np.isnan(cloud_b) else (
        "above" if x.c[i] > max(cloud_a, cloud_b) else "below" if x.c[i] < min(cloud_a, cloud_b) else "inside")
    ctx = {
        "squeeze_state": squeeze_state,
        "bb_squeeze": {"on": bb_on, "bars": streak(e.bb_sq), "bandwidth_pct": _r(ind.pct_rank(e.bw, i), 2)},
        "ttm_squeeze": {"on": ttm_on, "bars": streak(lambda j: bool(e.ttm_on[j])),
                        "momentum": _r(e.mom[i]), "momentum_dir": None if np.isnan(e.mom[i]) else
                        ("bull" if e.mom[i] > 0 else "bear")},
        "atr_contraction": bool(e.atr_low(i)),
        "hv_contraction": bool(e.hv_low(i)),
        "volume_dry_up": bool(e.vdu(i)),
        "donchian_squeeze": bool(e.dc_sq(i)),
        "ma_ribbon_squeeze": bool(e.rib_sq(i)),
        "obv_divergence": _divergence(x.c, e.obv, x.v, lb, x.atr[i]),
        "ad_divergence": _divergence(x.c, e.adl, x.v, lb, x.atr[i]),
        "cmf": _r(cmf_v, 3),
        "cmf_state": None if np.isnan(cmf_v) else ("positive" if cmf_v > 0.05 else "negative" if cmf_v < -0.05 else "flat"),
        "adx": _r(adx_v, 1),
        "adx_state": None if np.isnan(adx_v) else ("weak" if adx_v < 20 else "trending" if adx_v >= 25 else "building"),
        "choppiness": _r(chop_v, 1),
        "ichimoku_cloud": cloud,
        "rel_volume": _r(x.volr[i], 2),
        "atr": _r(x.atr[i]),
        "atr_pct": _r(e.atrp[i] * 100, 3),
        "hv_annualised": _r(e.hv[i], 3),
    }
    layers = [k for k, on in (("bb", bb_on), ("ttm", ttm_on), ("atr", ctx["atr_contraction"]),
                              ("hv", ctx["hv_contraction"]), ("volume", ctx["volume_dry_up"]),
                              ("donchian", ctx["donchian_squeeze"]), ("ribbon", ctx["ma_ribbon_squeeze"]),
                              ("adx", ctx["adx_state"] == "weak")) if on]
    ctx["compression_layers"] = layers
    return ctx


# ----------------------------------------------------------------------------- confluence score
PART_ORDER = ["pattern", "smc", "fresh", "volume", "squeeze"]    # order of the stored `parts` list
SCORE_MAX = {"pattern": 30, "smc": 25, "fresh": 20, "volume": 15, "squeeze": 10}


def score_signal(sig: dict, smc_trend: str, squeeze_state: str) -> dict:
    """0-100 ranking number. It ranks setups against each other - it is NOT a probability of success."""
    d = sig["dir"]
    pattern = round(sig["stars"] / 5 * SCORE_MAX["pattern"])
    if d == "neutral":
        pattern = round(pattern * 0.6)
        smc = 0
    elif smc_trend not in ("BULLISH", "BEARISH"):
        smc = round(SCORE_MAX["smc"] * 0.5)
    else:
        smc = SCORE_MAX["smc"] if (d == "bull") == (smc_trend == "BULLISH") else round(SCORE_MAX["smc"] * 0.35)
    fresh = round(max(0.0, 1 - sig["since"] / config.FRESH_FADE_BARS) * SCORE_MAX["fresh"])
    v = sig.get("vol")
    vol = round(min(1.0, max(0.0, ((v if v is not None else 0.8) - 0.8) / 1.0)) * SCORE_MAX["volume"])
    sq = 10 if squeeze_state == "fired" else 6 if squeeze_state == "on" else 0
    parts = {"pattern": pattern, "smc": smc, "fresh": fresh, "volume": vol, "squeeze": sq}
    return {"total": int(sum(parts.values())), "parts": parts}


def analyze(df: pd.DataFrame, smc_trend: str = "UNKNOWN") -> Tuple[List[dict], dict, dict]:
    """-> (signals, context, scores). `df` = closed candles only, columns Date/Open/High/Low/Close/Volume."""
    e = Engine(df.tail(config.CALC_BARS).reset_index(drop=True))
    sigs = build_signals(e)
    ctx = build_context(e, sigs)
    for s in sigs:
        sc = score_signal(s, smc_trend, ctx["squeeze_state"])
        s["score"], s["parts"] = sc["total"], [sc["parts"][k] for k in PART_ORDER]
    sigs.sort(key=lambda s: (-s["score"], s["since"], s["name"]))
    sigs = sigs[:config.MAX_SIGNALS]                       # lowest-scoring (oldest, weakest) are dropped first
    sigs.sort(key=lambda s: (s["since"], -s["score"], s["name"]))
    for s in sigs:                                         # stars were only needed for scoring
        s.pop("stars", None)
    bull = max([s["score"] for s in sigs if s["dir"] == "bull"], default=0)
    bear = max([s["score"] for s in sigs if s["dir"] == "bear"], default=0)
    return sigs, ctx, {"bull": bull, "bear": bear, "best": max(bull, bear)}
