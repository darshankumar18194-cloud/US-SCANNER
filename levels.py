"""Support / resistance levels - a Python port of the rules in the book "Auto Touch S/R v4".

    result = levels.analyse(df)          # df = closed 1H candles [Date, Open, High, Low, Close, Volume]
    result["levels"]                     # the strongest levels, best first per side (see _record)
    result["meta"]                       # bars used, candidates found, scored, ATR

What is ported (book chapter in brackets):
  swing points -> 0.1-ATR histogram -> candidate levels, half above / half below price (ch. 4)
  touches, visits, reaction, hold rate, wick rejection, arrival, sweeps (ch. 4, 7)
  confirmed breaks (2 closes beyond the level by 0.2 ATR) and role flips (ch. 7)
  confluence with VWAP, EMA 50/200, Fibonacci, previous day/week, day open, opening range, round numbers,
  higher-timeframe swings (ch. 8)
  the 12-measure score, grade A-D, percentile of visits, next-level targets and reward-to-risk (ch. 4, 6)

What is deliberately NOT ported: news windows, market presets, the daily signal/loss limits and the alerts - they belong
to a TradingView chart, not to an hourly scan of many symbols.

Every distance is measured in ATR and every candle is judged by its own ATR (book 4.7), the grid by today's ATR (4.5).
Levels are computed on the newest closed candle, exactly like the indicator: they are NOT a history of what existed
earlier (book 4.13). To keep a trade plan stable, the scanner must save the levels at the moment the plan is made.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import config


@dataclass
class Params:
    # 1. Levels
    left: int = 2
    right: int = 2
    tol_mult: float = 0.3            # touch zone half-width (x ATR) ...
    cluster_mult: float = 0.6        # ... but never below the cluster width; also the histogram window
    merge_mult: float = 1.5          # keep levels at least this far apart (x ATR)
    min_piv_hits: int = 2            # swing points needed to become a candidate
    max_cand: int = 40
    min_touch: int = 3               # min visits
    max_dist_pct: float = 60.0       # ignore swing points farther than this % from price (book 4.4)
    bin_mult: float = 0.1            # histogram bin width (x ATR)
    # 2. Touch quality
    min_gap: int = 3                 # candles this close belong to the same visit
    react_bars: int = 10
    react_min: float = 0.8           # a bounce = moved at least this far away (x ATR)
    react_full: float = 2.0
    wick_ratio: float = 1.0
    off_sess_w: float = 0.5          # weight of touches outside the active session (extended-hours data only)
    fresh_bars: int = 60
    # 3. Breaks and sweeps
    brk_buf: float = 0.2
    brk_closes: int = 2
    max_breaks: int = 8
    break_win: int = 300
    broken_bars: int = 30            # "recently flipped"
    sweep_min: float = 0.1
    sweep_max: float = 1.5
    # 4. Confluence
    conf_tol: float = 0.4
    fib_bars: int = 300
    or_bars: int = 6
    htf_left: int = 5
    htf_right: int = 5
    htf_keep: int = 30               # newest N daily swing points (about 3 months of major structure; the book keeps 300 on a chart far below its HTF)
    htf_source: str = "daily"        # "daily" = swings of daily candles built from the hourly data; "hourly" = old behaviour
    round_div: float = 5.0           # round-number step = (power of ten at the price) / round_div   (book: 20)
    round_cap: Optional[float] = 0.25  # round-number tolerance never above this x step (None = the book's plain 0.4 ATR)
    # 6/7. Score
    weights: Dict[str, float] = field(default_factory=lambda: dict(
        touch=15, react=20, hold=15, conf=12, recent=8, tight=5, wick=5, days=6, sess=4, sweep=4, div=2, arrive=4))
    min_score: float = 40.0
    # Grades are calibrated against NOISE, not the book's 70/55/40: on simulated random prices (5,367 levels) the score
    # has median 75, 70th percentile 77, 85th 80, 95th 82. So A = better than ~95% of levels that pure noise produces,
    # B = better than ~85%, C = better than ~70%. The book's cut-offs made 97% of real levels grade A (= no information).
    grade_a: float = 82.0
    grade_b: float = 80.0
    grade_c: float = 77.0
    fresh_bonus: float = 5.0
    break_penalty: float = 1.5
    near_bias: float = 0.3           # score points lost per ATR of distance when ranking
    stop_buf: float = 0.2            # stop beyond the zone by (x ATR)
    # output
    max_bars: int = 4990
    min_bars: int = 100
    store_per_side: int = 8


P = Params()


# ----------------------------------------------------------------------------- small helpers
def _rma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def _atr(h, l, c, n: int = 14) -> np.ndarray:
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return _rma(tr, n)


def _rsi(c, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up, dn = _rma(np.maximum(d, 0), n), _rma(np.maximum(-d, 0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = up / dn
    out = 100 - 100 / (1 + rs)
    return np.where(dn == 0, 100.0, out)


def _pivots(x: np.ndarray, left: int, right: int, high: bool) -> np.ndarray:
    """Boolean array: bar i is a swing high (low) when it is the highest (lowest) of `left` bars before it and `right`
    bars after it. An equal value on the left is allowed, on the right it is not, so a flat top gives ONE swing."""
    n, w = len(x), left + right + 1
    out = np.zeros(n, bool)
    if n < w:
        return out
    win = sliding_window_view(x, w)
    c = win[:, left]
    if high:
        ok = (c >= win[:, :left].max(1)) & (c > win[:, left + 1:].max(1))
    else:
        ok = (c <= win[:, :left].min(1)) & (c < win[:, left + 1:].min(1))
    out[left:n - right] = ok
    return out


def _num(v, nd: int = 4) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    return round(v, nd) if math.isfinite(v) else None


# ----------------------------------------------------------------------------- step 1-4: candidates
def _candidates(swing_prices: np.ndarray, close: float, atr_now: float, p: Params):
    """Book 4.5 / 4.6. Returns a list of (price, span, n_swings) sorted by price."""
    bin_size = atr_now * p.bin_mult
    keys_arr = np.floor(swing_prices / bin_size).astype(np.int64)
    cnt: Dict[int, int] = {}
    sm: Dict[int, float] = {}
    mx: Dict[int, float] = {}
    mn: Dict[int, float] = {}
    for k, v in zip(keys_arr.tolist(), swing_prices.tolist()):
        cnt[k] = cnt.get(k, 0) + 1
        sm[k] = sm.get(k, 0.0) + v
        mx[k] = v if k not in mx else max(mx[k], v)
        mn[k] = v if k not in mn else min(mn[k], v)
    keys = list(cnt.keys())
    K = len(keys)
    if K == 0:
        return []
    wB = max(1, round(p.cluster_mult / p.bin_mult))
    sepB = max(wB, round(p.merge_mult / p.bin_mult))
    wc = [0] * K
    ws = [0.0] * K
    wsp = [0.0] * K
    wmx = [math.nan] * K
    wmn = [math.nan] * K
    for x, k0 in enumerate(keys):
        sc, ss = 0, 0.0
        kmn, kmx = k0 + 10 ** 9, k0 - 10 ** 9
        a, b = math.nan, math.nan
        for d in range(-wB, wB + 1):
            c = cnt.get(k0 + d, 0)
            if c > 0:
                sc += c
                ss += sm[k0 + d]
                kmn, kmx = min(kmn, k0 + d), max(kmx, k0 + d)
                a = mx[k0 + d] if math.isnan(a) else max(a, mx[k0 + d])
                b = mn[k0 + d] if math.isnan(b) else min(b, mn[k0 + d])
        wc[x], ws[x] = sc, ss
        wsp[x] = (kmx - kmn) * bin_size if sc > 0 else 0.0
        wmx[x], wmn[x] = a, b
    suppressed = [False] * K
    half = math.ceil(p.max_cand / 2.0)
    c_up = c_dn = 0
    out = []
    for _ in range(p.max_cand):
        bi, bw = -1, 0
        for x in range(K):
            if not suppressed[x] and wc[x] > bw:
                up = ws[x] / max(wc[x], 1) > close
                if (c_up < half) if up else (c_dn < half):
                    bw, bi = wc[x], x
        if bi < 0 or bw < p.min_piv_hits:
            break
        centroid = ws[bi] / wc[bi]
        if centroid > close:
            c_up += 1
        else:
            c_dn += 1
        out.append((centroid, wsp[bi], wc[bi]))
        kb = keys[bi]
        for x in range(K):
            if abs(keys[x] - kb) <= sepB:
                suppressed[x] = True
    out.sort(key=lambda t: t[0])
    return out


# ----------------------------------------------------------------------------- step 5: scan one level
def _scan_level(lvl: float, span: float, n_piv: int, ctx: Dict[str, Any], p: Params) -> Dict[str, Any]:
    O, H, L, C, atr = ctx["O"], ctx["H"], ctx["L"], ctx["C"], ctx["atr"]
    n = ctx["n"]
    idx = ctx["idx"]
    tol = max(p.tol_mult, p.cluster_mult) * atr
    is_res = lvl > C[-1]

    # ---- confirmed breaks (book 7.1): runs of closes beyond the level by brk_buf x ATR
    buf = p.brk_buf * atr
    s = np.where(C > lvl + buf, 1, np.where(C < lvl - buf, -1, 0))
    change = np.r_[True, s[1:] != s[:-1]] | (s == 0)
    start = np.maximum.accumulate(np.where(change, idx, 0))
    streak = np.where(s != 0, idx - start + 1, 0)
    brks = brks_recent = 0
    last_brk = 10 ** 6
    conf_side = 0
    for e in np.flatnonzero(streak == p.brk_closes).tolist():
        side = int(s[e])
        if side != conf_side:
            if conf_side != 0:
                brks += 1
                off = n - 1 - e
                if off < p.break_win:
                    brks_recent += 1
                last_brk = off
            conf_side = side

    # ---- sweeps (book 7.5)
    top_s, bot_s = H - lvl, lvl - L
    bear = (top_s >= p.sweep_min * atr) & (top_s <= p.sweep_max * atr) & (np.maximum(O, C) < lvl)
    bull = ~bear & (bot_s >= p.sweep_min * atr) & (bot_s <= p.sweep_max * atr) & (np.minimum(O, C) > lvl)
    sweeps = int(bear.sum() + bull.sum())
    sweep_dir = (-1 if bear[-1] else 1 if bull[-1] else 0)

    # ---- touches and visits (book 4.7): a candle's high or low inside the zone; close candles = one visit
    low_t = np.abs(L - lvl) <= tol
    high_t = np.abs(H - lvl) <= tol
    t = low_t | high_t
    t[0] = False                                  # the oldest bar has no "previous close" to judge the approach
    tidx = np.flatnonzero(t)
    visits: List[int] = []
    groups: List[List[int]] = []
    last = -10 ** 9
    for ix in tidx.tolist():
        if ix - last >= p.min_gap:
            visits.append(ix)
            groups.append([ix, ix, 1])
        else:
            groups[-1][1] = ix
            groups[-1][2] += 1
        last = ix
    vi = np.array(visits, dtype=np.int64)
    c = len(vi)

    meas = bounces = wick_ok = arrive = sess_t = 0
    react_sum = rec = w_touch = 0.0
    days = 0
    diverg = False
    last_visit_off = 10 ** 6
    touch_now = False
    if c:
        last_visit_off = n - 1 - int(vi[-1])
        touch_now = int(vi[-1]) == n - 1
        a = atr[vi]
        from_above = np.where(low_t[vi] & ~high_t[vi], True, np.where(high_t[vi] & ~low_t[vi], False, C[vi - 1] > lvl))
        # reaction within the next react_bars candles (needs at least 3 candles after the visit)
        kk = np.arange(1, p.react_bars + 1)
        i2 = vi[:, None] + kk[None, :]
        valid = i2 <= n - 1
        i2c = np.minimum(i2, n - 1)
        up_away = np.where(valid, H[i2c], -np.inf).max(1) - lvl
        dn_away = lvl - np.where(valid, L[i2c], np.inf).min(1)
        away = np.maximum(np.where(from_above, up_away, dn_away), 0.0)
        away = np.where(np.isfinite(away), away, 0.0)
        measurable = (n - 1 - vi) >= 3
        react = np.minimum(away / a, 3.0)
        meas = int(measurable.sum())
        react_sum = float(react[measurable].sum())
        bounces = int((react[measurable] >= p.react_min).sum())
        # wick rejection: the wick on the approach side is at least as long as the body
        body = np.abs(C[vi] - O[vi])
        wick = np.where(from_above, np.minimum(O[vi], C[vi]) - L[vi], H[vi] - np.maximum(O[vi], C[vi]))
        wick_ok = int(((wick > 0) & (wick >= body * p.wick_ratio)).sum())
        # arrival: a sharp 3-candle move in, and / or a stretch from the 20 EMA
        ok4 = vi >= 4
        v4 = np.maximum(vi - 4, 0)
        sharp = ok4 & (np.abs(C[vi - 1] - C[v4]) / a >= 1.0)
        stretch = (vi >= 1) & (np.abs(C[vi - 1] - ctx["ema20"][vi - 1]) / a >= 1.5)
        arrive = int(sharp.sum() + stretch.sum())
        act = ctx["active"][vi]
        sess_t = int(act.sum())
        w_touch = float(np.where(act, 1.0, p.off_sess_w).sum())
        rec = float((1.0 - (n - 1 - vi) / n).sum())
        days = int(pd.Series(ctx["day"][vi]).nunique())
        if c >= 2:
            o1, o2 = int(vi[-2]), int(vi[-1])
            rsi = ctx["rsi"]
            diverg = bool((H[o2] >= H[o1] and rsi[o2] < rsi[o1]) if is_res else (L[o2] <= L[o1] and rsi[o2] > rsi[o1]))

    # ---- confluence (book 8): every reference within conf_tol x ATR adds one point
    atr_now = ctx["atr_now"]
    tags: List[str] = []
    for name, v in ctx["refs"]:
        if abs(v - lvl) <= p.conf_tol * atr_now:
            tags.append(name)
    if _round_hit(lvl, ctx["C"][-1], atr_now, p):
        tags.append("RND")
    tol_now = max(p.tol_mult, p.cluster_mult) * atr_now
    if any(abs(hp - lvl) <= tol_now for hp in ctx["htf"]):
        tags.append("HTF")

    return dict(price=lvl, spread=span, pivots=n_piv, touches=c, candles=int(len(tidx)), w_touch=w_touch,
                measured=meas, bounces=bounces, react_sum=react_sum, wick_ok=wick_ok, arrive=arrive, sess_t=sess_t,
                days=days, recency=rec, sweeps=sweeps, breaks=brks, breaks_recent=brks_recent,
                last_break_off=last_brk, conf_side=conf_side, last_visit_off=last_visit_off, diverg=diverg,
                tags=tags, touch_now=touch_now, sweep_now=sweep_dir != 0, sweep_dir=sweep_dir, is_res=bool(is_res),
                fresh=(c > 0 and last_visit_off >= p.fresh_bars))


# ----------------------------------------------------------------------------- step 6/7: score
def _score(lv: Dict[str, Any], tol_now: float, p: Params) -> None:
    w = p.weights
    vis = max(lv["touches"], 1)
    parts = dict(
        touch=min(1.0, lv["w_touch"] / 10.0),
        react=min(1.0, (lv["react_sum"] / lv["measured"]) / p.react_full) if lv["measured"] > 0 else 0.0,
        hold=lv["bounces"] / lv["measured"] if lv["measured"] > 0 else 0.5,
        conf=min(1.0, len(lv["tags"]) / 3.0),
        recent=lv["recency"] / vis,
        tight=(1.0 - min(1.0, lv["spread"] / (3.0 * tol_now))) if lv["pivots"] >= 2 else 0.5,
        wick=lv["wick_ok"] / vis,
        days=min(1.0, lv["days"] / 3.0),
        sess=lv["sess_t"] / vis,
        sweep=min(1.0, lv["sweeps"] / 2.0),
        div=1.0 if lv["diverg"] else 0.0,
        arrive=lv["arrive"] / (2.0 * vis))
    wsum = max(1.0, sum(w.values()))
    raw = 100.0 * sum(w[k] * parts[k] for k in w) / wsum
    raw += p.fresh_bonus if lv["fresh"] else 0.0
    raw -= p.break_penalty * lv["breaks_recent"]
    lv["score"] = max(0.0, min(100.0, raw))
    lv["grade"] = "A" if lv["score"] >= p.grade_a else "B" if lv["score"] >= p.grade_b else "C" if lv["score"] >= p.grade_c else "D"


# ----------------------------------------------------------------------------- confluence helpers
def _htf_swings(df: pd.DataFrame, H: np.ndarray, L: np.ndarray, p: Params) -> List[float]:
    """Higher-timeframe swing points (book 8.6). Built from DAILY candles resampled from the hourly data, so a level
    is only tagged HTF when a swing of a genuinely longer timeframe sits next to it. (Comparing with swings of the SAME
    hourly series, as the first port did, is circular: the levels are made of those swings, so ~98% matched.)"""
    if p.htf_source == "hourly":
        hi, lo = H, L
    else:
        day = df["Date"].dt.normalize()
        g = df.groupby(day, sort=True).agg(High=("High", "max"), Low=("Low", "min"))
        hi, lo = g["High"].to_numpy(float), g["Low"].to_numpy(float)
    ih = np.flatnonzero(_pivots(hi, p.htf_left, p.htf_right, True))
    il = np.flatnonzero(_pivots(lo, p.htf_left, p.htf_right, False))
    pts = sorted([(int(i), float(hi[i])) for i in ih] + [(int(i), float(lo[i])) for i in il])   # oldest -> newest
    return [price for _, price in pts[-p.htf_keep:]]


def _round_hit(lvl: float, close: float, atr_now: float, p: Params) -> bool:
    """Book 8.5, tightened: near a multiple of (power of ten / round_div), within 0.4 ATR but never more than
    round_cap x the step. With the book's plain rule ~83% of levels counted as 'round'."""
    step = (10 ** math.floor(math.log10(close))) / p.round_div
    tol = p.conf_tol * atr_now
    if p.round_cap is not None:
        tol = min(tol, p.round_cap * step)
    return abs(lvl - round(lvl / step) * step) <= tol


# ----------------------------------------------------------------------------- the whole pipeline
def _reference_prices(df: pd.DataFrame, close: np.ndarray, p: Params) -> List[tuple]:
    """Book 8.2: today's reference prices (taken at the newest candle)."""
    refs: List[tuple] = []
    dates = df["Date"].dt.normalize()
    today = dates.iloc[-1]
    prev_days = dates[dates < today]
    t_mask = (dates == today).to_numpy()
    H, L, O, V = (df[c].to_numpy(float) for c in ("High", "Low", "Open", "Volume"))
    # VWAP of today's session
    if t_mask.any() and V[t_mask].sum() > 0:
        tp = (H[t_mask] + L[t_mask] + close[t_mask]) / 3.0
        refs.append(("VWAP", float((tp * V[t_mask]).sum() / V[t_mask].sum())))
    refs.append(("EMA50", float(pd.Series(close).ewm(span=50, adjust=False).mean().iloc[-1])))
    refs.append(("EMA200", float(pd.Series(close).ewm(span=200, adjust=False).mean().iloc[-1])))
    k = min(p.fib_bars, len(close))
    hh, ll = float(H[-k:].max()), float(L[-k:].min())
    rng = hh - ll
    refs += [("FIB38", hh - rng * 0.382), ("FIB50", hh - rng * 0.5), ("FIB62", hh - rng * 0.618)]
    if len(prev_days):
        pd_day = prev_days.iloc[-1]
        m = (dates == pd_day).to_numpy()
        refs += [("PDH", float(H[m].max())), ("PDL", float(L[m].min()))]
    week = df["Date"].dt.isocalendar()
    wkey = (week["year"].astype(int) * 100 + week["week"].astype(int)).to_numpy()
    prev_w = wkey < wkey[-1]
    if prev_w.any():
        last_pw = wkey[prev_w][-1]
        m = wkey == last_pw
        refs += [("PWH", float(H[m].max())), ("PWL", float(L[m].min()))]
    if t_mask.any():
        refs.append(("DOPEN", float(O[t_mask][0])))
        first = np.flatnonzero(t_mask)[:p.or_bars]
        refs += [("ORH", float(H[first].max())), ("ORL", float(L[first].min()))]
    return refs


def analyse(df: pd.DataFrame, p: Params = P, extended_hours: Optional[bool] = None) -> Dict[str, Any]:
    """Find, scan, score and rank the levels of one symbol. Never needs the network."""
    if extended_hours is None:
        extended_hours = bool(getattr(config, "EXTENDED_HOURS", False))
    df = df.tail(p.max_bars).reset_index(drop=True)
    n = len(df)
    empty = {"levels": [], "meta": {"bars": n, "candidates": 0, "scored": 0, "atr": None}}
    if n < p.min_bars:
        return empty
    O, H, L, C = (df[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
    atr = _atr(H, L, C)
    atr_now = float(atr[-1])
    close = float(C[-1])
    if not math.isfinite(atr_now) or atr_now <= 0:
        return empty
    atr = np.where(np.isfinite(atr) & (atr > 0), atr, atr_now)

    ph, pl = _pivots(H, p.left, p.right, True), _pivots(L, p.left, p.right, False)
    swings = np.r_[H[ph], L[pl]]
    swings = swings[np.abs(swings - close) <= p.max_dist_pct / 100.0 * close]          # book 4.4
    cands = _candidates(swings, close, atr_now, p)

    htf = _htf_swings(df, H, L, p)                                  # higher-timeframe swing points (book 8.6)

    dt = df["Date"]
    minutes = (dt.dt.hour * 60 + dt.dt.minute).to_numpy()
    active = np.ones(n, bool)
    if extended_hours:                                              # only regular hours count fully (book 4.7)
        active = (minutes >= config.MARKET_OPEN[0] * 60 + config.MARKET_OPEN[1]) & (minutes < config.MARKET_CLOSE[0] * 60)
    ctx = dict(O=O, H=H, L=L, C=C, atr=atr, atr_now=atr_now, n=n, idx=np.arange(n),
               ema20=pd.Series(C).ewm(span=20, adjust=False).mean().to_numpy(), rsi=_rsi(C), active=active,
               day=dt.dt.normalize().to_numpy().astype("datetime64[D]").astype(np.int64),
               refs=_reference_prices(df, C, p), htf=htf)

    tol_now = max(p.tol_mult, p.cluster_mult) * atr_now
    lvs: List[Dict[str, Any]] = []
    for price, span, npiv in cands:
        lv = _scan_level(price, span, npiv, ctx, p)
        lv["valid"] = lv["touches"] >= p.min_touch and lv["breaks_recent"] <= p.max_breaks
        if lv["valid"]:
            _score(lv, tol_now, p)
        lvs.append(lv)

    valid = [v for v in lvs if v["valid"]]
    for v in valid:                                                  # percentile of visits among scored levels
        below = sum(1 for m in valid if m["touches"] < v["touches"])
        v["pct"] = below / (len(valid) - 1) if len(valid) > 1 else 1.0
    stop_dist = tol_now + p.stop_buf * atr_now
    for j, v in enumerate(valid):                                    # next scored level on each side -> reward-to-risk
        v["tgt_dn"] = valid[j - 1]["price"] if j > 0 else None
        v["tgt_up"] = valid[j + 1]["price"] if j < len(valid) - 1 else None
        tg = v["tgt_dn"] if v["is_res"] else v["tgt_up"]
        v["rr"] = abs(tg - v["price"]) / stop_dist if tg is not None else None

    # choose what to keep: best rank first (score minus a distance penalty), levels 1.5 ATR apart (book 4.11)
    chosen: List[Dict[str, Any]] = []
    pool = [v for v in valid if v["score"] >= p.min_score]
    while pool:
        best = max(pool, key=lambda v: v["score"] - p.near_bias * abs(v["price"] - close) / atr_now)
        pool.remove(best)
        if all(abs(c_["price"] - best["price"]) > p.merge_mult * atr_now for c_ in chosen):
            chosen.append(best)
    keep: List[Dict[str, Any]] = []
    for side in (True, False):
        side_levels = [v for v in chosen if v["is_res"] == side][:p.store_per_side]
        keep += side_levels
    keep.sort(key=lambda v: -v["price"])
    return {"levels": [_record(v, close, atr_now, n, p) for v in keep],
            "meta": {"bars": n, "candidates": len(lvs), "scored": len(valid), "atr": _num(atr_now)}}


def _record(v: Dict[str, Any], close: float, atr_now: float, n: int, p: Params) -> Dict[str, Any]:
    role_res = v["is_res"] if v["conf_side"] == 0 else v["conf_side"] == -1
    flipped = v["breaks"] >= 1
    return {
        "price": _num(v["price"]), "role": "R" if role_res else "S", "grade": v["grade"], "score": int(round(v["score"])),
        "visits": v["touches"], "candles": v["candles"],
        "hold": _num(100.0 * v["bounces"] / v["measured"], 0) if v["measured"] else None,
        "breaks": v["breaks_recent"],
        "flipped": bool(flipped), "flip_ago": int(v["last_break_off"]) if flipped and v["last_break_off"] <= p.broken_bars else None,
        "rr": _num(v["rr"], 1), "tags": v["tags"], "sweeps": v["sweeps"], "fresh": bool(v["fresh"]),
        "pct": _num(100.0 * v["pct"], 0), "dist_atr": _num((v["price"] - close) / atr_now, 1),
        "last_visit_ago": int(v["last_visit_off"]) if v["touches"] else None,
        "tgt_up": _num(v["tgt_up"]), "tgt_dn": _num(v["tgt_dn"]),
    }
