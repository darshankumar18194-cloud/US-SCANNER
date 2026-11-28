"""Plain numpy/pandas indicator math. No network, no state."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

NAN = np.nan


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).mean().to_numpy()


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def rolling_max(x, n):
    return pd.Series(x).rolling(n).max().to_numpy()


def rolling_min(x, n):
    return pd.Series(x).rolling(n).min().to_numpy()


def true_range(h, l, c) -> np.ndarray:
    pc = np.roll(c, 1)
    pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def atr(h, l, c, n: int = 14) -> np.ndarray:
    """Simple moving average of true range (the same smoothing TradingView's 'ATR (SMA)' uses)."""
    return sma(true_range(h, l, c), n)


def bollinger(c, n: int = config.BB_LEN, k: float = config.BB_MULT):
    mid = sma(c, n)
    sd = pd.Series(c).rolling(n).std(ddof=0).to_numpy()
    up, lo = mid + k * sd, mid - k * sd
    with np.errstate(divide="ignore", invalid="ignore"):
        bw = (up - lo) / mid
    return mid, up, lo, bw


def keltner(h, l, c, n: int = 20, k: float = config.KC_MULT):
    mid = ema(c, n)
    rng = atr(h, l, c, n)
    return mid, mid + k * rng, mid - k * rng


def pct_rank(series: np.ndarray, i: int, lookback: int = config.PCT_LOOKBACK) -> float:
    """Share of the last `lookback` values (ending at i) that are <= series[i]. NaN if not enough history."""
    if i < 0 or i >= len(series) or np.isnan(series[i]):
        return NAN
    w = series[max(0, i - lookback + 1): i + 1]
    w = w[~np.isnan(w)]
    if len(w) < max(30, lookback // 4):
        return NAN
    return float(np.sum(w <= series[i]) / len(w))


def is_low(series: np.ndarray, i: int, pct: float = config.SQUEEZE_PCT) -> bool:
    r = pct_rank(series, i)
    return (not np.isnan(r)) and r <= pct


def adx(h, l, c, n: int = 14):
    """Wilder ADX with +DI / -DI. Returns (adx, plus_di, minus_di)."""
    N = len(c)
    up = np.zeros(N)
    dn = np.zeros(N)
    up[1:] = h[1:] - h[:-1]
    dn[1:] = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(h, l, c)
    tr[0] = h[0] - l[0]
    out_adx = np.full(N, NAN)
    pdi = np.full(N, NAN)
    mdi = np.full(N, NAN)
    if N <= 2 * n:
        return out_adx, pdi, mdi
    atr_s = tr[1:n + 1].sum()
    p_s = plus_dm[1:n + 1].sum()
    m_s = minus_dm[1:n + 1].sum()
    dx = np.full(N, NAN)
    for i in range(n, N):
        if i > n:
            atr_s = atr_s - atr_s / n + tr[i]
            p_s = p_s - p_s / n + plus_dm[i]
            m_s = m_s - m_s / n + minus_dm[i]
        if atr_s <= 0:
            continue
        pdi[i] = 100 * p_s / atr_s
        mdi[i] = 100 * m_s / atr_s
        den = pdi[i] + mdi[i]
        dx[i] = 100 * abs(pdi[i] - mdi[i]) / den if den > 0 else 0.0
    first = 2 * n - 1
    out_adx[first] = np.nanmean(dx[n:first + 1])
    for i in range(first + 1, N):
        if not np.isnan(dx[i]):
            out_adx[i] = (out_adx[i - 1] * (n - 1) + dx[i]) / n
    return out_adx, pdi, mdi


def choppiness(h, l, c, n: int = 14) -> np.ndarray:
    tr = pd.Series(true_range(h, l, c)).rolling(n).sum().to_numpy()
    rng = rolling_max(h, n) - rolling_min(l, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 * np.log10(tr / rng) / np.log10(n)
    out[~np.isfinite(out)] = NAN
    return out


def obv(c, v) -> np.ndarray:
    d = np.sign(np.diff(c, prepend=c[0]))
    return np.cumsum(d * v)


def clv(h, l, c) -> np.ndarray:
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ((c - l) - (h - c)) / rng
    out[~np.isfinite(out)] = 0.0
    return out


def ad_line(h, l, c, v) -> np.ndarray:
    return np.cumsum(clv(h, l, c) * v)


def cmf(h, l, c, v, n: int = 20) -> np.ndarray:
    num = pd.Series(clv(h, l, c) * v).rolling(n).sum().to_numpy()
    den = pd.Series(v).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = NAN
    return out


def hist_vol(c, n: int = 20) -> np.ndarray:
    """Annualised stdev of log returns (7 one-hour candles a day, 252 days)."""
    r = np.diff(np.log(c), prepend=np.nan)
    sd = pd.Series(r).rolling(n).std(ddof=1).to_numpy()
    return sd * np.sqrt(config.BARS_PER_DAY * 252)


def ichimoku(h, l, c):
    """Unshifted span A / span B (the 'future cloud'), plus tenkan / kijun."""
    tenkan = (rolling_max(h, 9) + rolling_min(l, 9)) / 2
    kijun = (rolling_max(h, 26) + rolling_min(l, 26)) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (rolling_max(h, 52) + rolling_min(l, 52)) / 2
    return tenkan, kijun, span_a, span_b


def linreg_end(y: np.ndarray, n: int = 20) -> np.ndarray:
    """Value at the last point of an n-bar least-squares line, for every bar."""
    N = len(y)
    out = np.full(N, NAN)
    if N < n:
        return out
    x = np.arange(n, dtype=float)
    xm = x.mean()
    den = ((x - xm) ** 2).sum()
    win = np.lib.stride_tricks.sliding_window_view(y, n)
    ym = win.mean(axis=1)
    slope = ((win - ym[:, None]) * (x - xm)).sum(axis=1) / den
    out[n - 1:] = ym + slope * (n - 1 - xm)
    return out


def ttm_momentum(h, l, c, n: int = 20) -> np.ndarray:
    mid = (rolling_max(h, n) + rolling_min(l, n)) / 2
    delta = c - (mid + sma(c, n)) / 2
    ok = ~np.isnan(delta)
    out = np.full(len(c), NAN)
    if ok.sum() >= n:
        first = int(np.argmax(ok))
        out[first:] = linreg_end(delta[first:], n)
    return out


def slot_rel_volume(times: pd.DatetimeIndex, v: np.ndarray, days: int = config.SLOT_VOL_DAYS) -> np.ndarray:
    """Volume divided by the average volume of the SAME time-of-day candle over the previous `days` sessions.

    Hourly volume is far heavier at the open and close, so comparing a 12:30 candle with a 9:30 candle
    would mislabel normal activity as 'spikes' and 'dry-ups'. NaN until 5 prior same-slot candles exist.
    """
    key = np.asarray(times.hour * 60 + times.minute)
    out = np.full(len(v), NAN)
    hist: dict = {}
    for i in range(len(v)):
        k = int(key[i])
        h = hist.setdefault(k, [])
        if len(h) >= 5:
            avg = float(np.mean(h[-days:]))
            if avg > 0:
                out[i] = v[i] / avg
        h.append(float(v[i]))
    return out
