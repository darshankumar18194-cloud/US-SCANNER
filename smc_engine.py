"""Swing market-structure engine (pure functions, no network, fully unit-tested).

Follows LuxAlgo "Smart Money Concepts" swing logic bar by bar, in chronological order:

  * leg / pivot detection : high[i-size] > highest(high, size)  ->  bearish leg (pivot HIGH)
                            low[i-size]  < lowest(low, size)    ->  bullish leg (pivot LOW)
  * only the CURRENT swing high / low is tracked; a new pivot replaces the old one
  * BOS / CHoCH           : close crosses the current unbroken pivot level. The tag is CHOCH when it
                            flips the trend, BOS when it continues it. Events are applied in the order
                            they happen in time.
  * Strong/Weak High/Low  : trailing extreme, reset to the pivot on every new pivot and extended by
                            each new high/low afterwards.  "Strong" = it protects the current trend.

Differences from the previous implementation (see README): the old code processed each pivot's
first break in pivot order rather than time order, kept superseded pivots alive, and could
therefore end with the wrong trend / BOS-vs-CHoCH tag / structure date.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

import config

BULLISH, BEARISH, UNKNOWN = "BULLISH", "BEARISH", "UNKNOWN"
STRONG_HIGH, WEAK_HIGH = "STRONG_HIGH", "WEAK_HIGH"
STRONG_LOW, WEAK_LOW = "STRONG_LOW", "WEAK_LOW"
NO_SIGNAL = "NO_SIGNAL"


@dataclass
class Pivot:
    kind: str        # "HIGH" | "LOW"
    bar: int         # bar index of the pivot candle
    confirmed: int   # bar index where the pivot became known (bar + size)
    price: float


@dataclass
class Analysis:
    n: int
    trend: str
    event: str                      # BOS | CHOCH | NONE
    event_bar: int
    pivots: list
    swing_high: Optional[Pivot]     # current (latest) swing high pivot
    swing_low: Optional[Pivot]
    swing_high_broken: bool
    swing_low_broken: bool
    top_price: Optional[float]      # trailing extreme (Strong/Weak High)
    top_bar: int
    bottom_price: Optional[float]
    bottom_bar: int
    last_close: float


# ---------------------------------------------------------------------------
def leg_series(highs: np.ndarray, lows: np.ndarray, size: int) -> np.ndarray:
    """0 = bearish leg, 1 = bullish leg (initial leg is 0, as in Pine). Vectorised."""
    n = len(highs)
    ev = np.full(n, -1, dtype=np.int8)
    if n > size:
        roll_hi = pd.Series(highs).rolling(size).max().to_numpy()   # window = bars i-size+1 .. i
        roll_lo = pd.Series(lows).rolling(size).min().to_numpy()
        new_low = np.zeros(n, dtype=bool)
        new_high = np.zeros(n, dtype=bool)
        new_high[size:] = highs[:n - size] > roll_hi[size:]
        new_low[size:] = lows[:n - size] < roll_lo[size:]
        ev[new_low] = 1
        ev[new_high] = 0                                            # high has priority (Pine if/else if)
    leg = pd.Series(ev).replace(-1, np.nan).ffill().fillna(0).to_numpy().astype(np.int8)
    return leg


def find_pivots(highs: np.ndarray, lows: np.ndarray, size: int) -> list:
    leg = leg_series(highs, lows, size)
    pivots = []
    for i in np.flatnonzero(np.diff(leg) != 0) + 1:
        p = int(i) - size
        if p < 0:
            continue
        if leg[i] == 0:
            pivots.append(Pivot("HIGH", p, int(i), float(highs[p])))
        else:
            pivots.append(Pivot("LOW", p, int(i), float(lows[p])))
    return pivots


def analyze(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
            size: int = config.SWING_LEN) -> Analysis:
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    pivots = find_pivots(highs, lows, size)
    by_confirm = {pv.confirmed: pv for pv in pivots}

    trend, event, event_bar = UNKNOWN, "NONE", -1
    hi: Optional[Pivot] = None
    lo: Optional[Pivot] = None
    hi_broken = lo_broken = False
    top = bot = None
    top_bar = bot_bar = -1

    for i in range(n):
        pv = by_confirm.get(i)
        if pv is not None:
            if pv.kind == "HIGH":
                hi, hi_broken = pv, False
                top, top_bar = pv.price, pv.bar
            else:
                lo, lo_broken = pv, False
                bot, bot_bar = pv.price, pv.bar
        if i > 0:
            c, pc = closes[i], closes[i - 1]
            if hi is not None and not hi_broken and pc <= hi.price < c:
                event = "CHOCH" if trend == BEARISH else "BOS"
                trend, hi_broken, event_bar = BULLISH, True, i
            if lo is not None and not lo_broken and pc >= lo.price > c:
                event = "CHOCH" if trend == BULLISH else "BOS"
                trend, lo_broken, event_bar = BEARISH, True, i
        if top is not None and highs[i] >= top:
            top, top_bar = float(highs[i]), i
        if bot is not None and lows[i] <= bot:
            bot, bot_bar = float(lows[i]), i

    return Analysis(n, trend, event, event_bar, pivots, hi, lo, hi_broken, lo_broken,
                    top, top_bar, bot, bot_bar, float(closes[-1]) if n else float("nan"))


# ---------------------------------------------------------------------------
def rank_score(signal: str, bars_since: Optional[int], event: str) -> float:
    base = 100 if signal in (STRONG_HIGH, STRONG_LOW) else 60 if signal in (WEAK_HIGH, WEAK_LOW) else 0
    bars = 999 if (bars_since is None or bars_since < 0) else bars_since
    freshness = min(bars * 1.5, 45)
    bonus = 10 if event == "CHOCH" else 5 if event == "BOS" else 0
    return round(max(base + bonus - freshness, 0), 2)


def _px(x: Optional[float]) -> str:
    return "NA" if x is None else str(round(float(x), 2))


def empty_fields(suffix: str, tf_label: str, status: str) -> Dict[str, Any]:
    s = suffix
    return {
        f"trend{s}": UNKNOWN, f"trend_source{s}": "NONE", f"unknown_reason{s}": status,
        f"structure_event{s}": "NONE", f"top_signal{s}": NO_SIGNAL, f"bottom_signal{s}": NO_SIGNAL,
        f"top_signal_time{s}": "NA", f"top_signal_price{s}": "NA",
        f"bottom_signal_time{s}": "NA", f"bottom_signal_price{s}": "NA",
        f"top_pivot_time{s}": "NA", f"bottom_pivot_time{s}": "NA", f"last_structure_time{s}": "NA",
        f"last_swing_high{s}": "NA", f"last_swing_low{s}": "NA",
        f"active_swing_high{s}": "NA", f"active_swing_low{s}": "NA",
        f"current_price{s}": "NA", f"timeframe{s}": tf_label, f"scan_date{s}": "NA",
        f"top_bars_since{s}": -1, f"bottom_bars_since{s}": -1,
        f"bearish_rank_score{s}": 0.0, f"bullish_rank_score{s}": 0.0,
        f"signal{s}": NO_SIGNAL, f"signal_side{s}": "NONE", f"signal_strength{s}": "NONE",
        f"last_signal_time{s}": "NA", f"signal_rank_score{s}": 0.0, f"bars_since_signal{s}": -1,
        f"scanner_status{s}": status,
    }


def to_fields(a: Analysis, labels: Sequence[str], suffix: str, tf_label: str) -> Dict[str, Any]:
    """Convert an Analysis into the flat record fields the dashboards read (schema unchanged)."""
    s = suffix
    if not a.pivots:
        return empty_fields(s, tf_label, "NO_PIVOTS")

    has_top, has_bot = a.top_price is not None, a.bottom_price is not None
    top_signal = (STRONG_HIGH if a.trend == BEARISH else WEAK_HIGH) if has_top else NO_SIGNAL
    bot_signal = (STRONG_LOW if a.trend == BULLISH else WEAK_LOW) if has_bot else NO_SIGNAL
    top_since = a.n - 1 - a.top_bar if has_top else -1
    bot_since = a.n - 1 - a.bottom_bar if has_bot else -1

    if a.trend == BULLISH:
        signal, side, since, when = bot_signal, "LOW", bot_since, labels[a.bottom_bar] if has_bot else "NA"
        score = rank_score(bot_signal, bot_since, a.event)
    elif a.trend == BEARISH:
        signal, side, since, when = top_signal, "HIGH", top_since, labels[a.top_bar] if has_top else "NA"
        score = rank_score(top_signal, top_since, a.event)
    else:
        signal, side, since, when, score = NO_SIGNAL, "NONE", -1, "NA", 0.0

    last_hi = [p for p in a.pivots if p.kind == "HIGH"]
    last_lo = [p for p in a.pivots if p.kind == "LOW"]
    return {
        f"trend{s}": a.trend,
        f"trend_source{s}": "STRUCTURE" if a.trend != UNKNOWN else "NONE",
        f"unknown_reason{s}": "NONE" if a.trend != UNKNOWN else "NO_CONFIRMED_STRUCTURE",
        f"structure_event{s}": a.event,
        f"top_signal{s}": top_signal, f"bottom_signal{s}": bot_signal,
        f"top_signal_time{s}": labels[a.top_bar] if has_top else "NA",
        f"top_signal_price{s}": _px(a.top_price),
        f"bottom_signal_time{s}": labels[a.bottom_bar] if has_bot else "NA",
        f"bottom_signal_price{s}": _px(a.bottom_price),
        f"top_pivot_time{s}": labels[last_hi[-1].bar] if last_hi else "NA",
        f"bottom_pivot_time{s}": labels[last_lo[-1].bar] if last_lo else "NA",
        f"last_structure_time{s}": labels[a.event_bar] if a.event_bar >= 0 else "NA",
        f"last_swing_high{s}": _px(a.swing_high.price) if a.swing_high else "NA",
        f"last_swing_low{s}": _px(a.swing_low.price) if a.swing_low else "NA",
        # "active" = the current swing level that has NOT been broken yet
        f"active_swing_high{s}": _px(a.swing_high.price) if a.swing_high and not a.swing_high_broken else "NA",
        f"active_swing_low{s}": _px(a.swing_low.price) if a.swing_low and not a.swing_low_broken else "NA",
        f"current_price{s}": _px(a.last_close),
        f"timeframe{s}": tf_label,
        f"scan_date{s}": labels[-1],
        f"top_bars_since{s}": top_since, f"bottom_bars_since{s}": bot_since,
        f"bearish_rank_score{s}": rank_score(top_signal, top_since, a.event),
        f"bullish_rank_score{s}": rank_score(bot_signal, bot_since, a.event),
        f"signal{s}": signal, f"signal_side{s}": side,
        f"signal_strength{s}": "STRONG" if "STRONG" in signal else "WEAK" if "WEAK" in signal else "NONE",
        f"last_signal_time{s}": when, f"signal_rank_score{s}": score, f"bars_since_signal{s}": since,
        f"scanner_status{s}": "OK",
    }


def analyze_frame(df: pd.DataFrame, suffix: str, tf_label: str, date_fmt: str,
                  min_bars: int = config.MIN_BARS, size: int = config.SWING_LEN) -> Dict[str, Any]:
    """df needs columns Date, High, Low, Close (sorted, no NaN)."""
    if df is None or len(df) < min_bars:
        return empty_fields(suffix, tf_label, "INSUFFICIENT_DATA")
    labels = df["Date"].dt.strftime(date_fmt).to_numpy()
    a = analyze(df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy(), size)
    return to_fields(a, labels, suffix, tf_label)
