"""Synthetic-candle helpers for the tests. No network, fully deterministic."""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS = [(9, 30), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30)]


def session_index(n: int, last: str = "2026-10-02 15:30") -> pd.DatetimeIndex:
    """n regular-session 1H candle start times (7 per weekday) ending at `last`."""
    day = pd.Timestamp(last).normalize()
    slot = SLOTS.index((pd.Timestamp(last).hour, pd.Timestamp(last).minute))
    out = []
    while len(out) < n:
        out.append(day + pd.Timedelta(hours=SLOTS[slot][0], minutes=SLOTS[slot][1]))
        slot -= 1
        if slot < 0:
            slot = 6
            day -= pd.Timedelta(days=1)
            while day.weekday() >= 5:
                day -= pd.Timedelta(days=1)
    return pd.DatetimeIndex(out[::-1])


def K(o, c, uw=0.1, lw=0.1, v=1000.0):
    """One candle (o, h, l, c, v) from open, close and wick lengths."""
    return (o, max(o, c) + uw, min(o, c) - lw, c, v)


def prefix(direction: str, n: int = 60, start: float = 100.0):
    """n candles ending in a clear down / up trend ('flat' = sideways). Candle range is about 0.8."""
    out, price = [], start
    flat = n - 12 if direction != "flat" else n
    for i in range(flat):
        d = 0.15 if i % 2 == 0 else -0.15
        out.append(K(price, price + d, 0.25, 0.25))
        price += d
    if direction != "flat":
        step = -0.6 if direction == "down" else 0.6
        for _ in range(12):
            out.append(K(price, price + step, 0.1, 0.1))
            price += step
    return out, price


def frame(candles, last: str = "2026-10-02 15:30") -> pd.DataFrame:
    idx = session_index(len(candles), last)
    arr = np.array(candles, dtype=float)
    return pd.DataFrame({"Date": idx, "Open": arr[:, 0], "High": arr[:, 1], "Low": arr[:, 2],
                         "Close": arr[:, 3], "Volume": arr[:, 4]})


def random_walk(n=600, seed=0, start=100.0, vol=0.004, base_volume=1000.0, last="2026-10-02 15:30"):
    rng = np.random.default_rng(seed)
    price, rows = start, []
    for _ in range(n):
        o = price
        c = o * (1 + rng.normal(0, vol))
        h = max(o, c) * (1 + abs(rng.normal(0, vol / 2)))
        l = min(o, c) * (1 - abs(rng.normal(0, vol / 2)))
        rows.append((o, h, l, c, base_volume * rng.uniform(0.7, 1.3)))
        price = c
    return frame(rows, last)
