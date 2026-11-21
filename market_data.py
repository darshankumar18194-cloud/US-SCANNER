"""Yahoo Finance access: one place, thread-safe (Ticker.history, not the global-state yf.download).

Candles are returned as a DataFrame[Date, Open, High, Low, Close, Volume] where Date is the candle START
time in the exchange's own local time (tz-naive). The exchange time zone is returned separately.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

import config
from common import (is_retryable_yf_error, now_market, retry, yf_cooldown_trigger, yf_cooldown_wait)

_OHLC = ["Open", "High", "Low", "Close"]
BAR = pd.Timedelta(hours=1)


@dataclass
class Fetch:
    status: str                         # OK | NO_DATA | ERROR
    df: Optional[pd.DataFrame] = None
    ticker: str = ""
    tz: str = config.MARKET_TZ_NAME
    error: str = ""
    dropped_forming: bool = False


def normalize(raw: Optional[pd.DataFrame]):
    """-> (DataFrame[Date, O, H, L, C, Volume] tz-naive local time, tz name) or (None, '')."""
    if raw is None or len(raw) == 0:
        return None, ""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if not all(c in df.columns for c in _OHLC):
        return None, ""
    idx = pd.to_datetime(df.index)
    tz = getattr(idx, "tz", None)
    tz_name = str(tz) if tz is not None else config.MARKET_TZ_NAME
    if tz is not None:
        idx = idx.tz_localize(None)                       # keep wall-clock time in the exchange's own zone
    out = df[_OHLC].apply(pd.to_numeric, errors="coerce")
    out["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0) if "Volume" in df.columns else 0.0
    out.insert(0, "Date", idx)
    out = out.dropna(subset=_OHLC).reset_index(drop=True)
    out = out.drop_duplicates(subset="Date", keep="last").sort_values("Date").reset_index(drop=True)
    return (out, tz_name) if len(out) else (None, "")


def bar_end(start: pd.Timestamp, tz_name: str) -> pd.Timestamp:
    """When a candle that STARTS at `start` (naive local time) closes (naive local time).

    US regular session ends 16:00, so the 15:30 candle is only 30 minutes long.
    Other exchanges are treated as full 1-hour candles (a few minutes of error right after their
    close is harmless for a scan that runs hours later).
    """
    end = start + BAR
    if tz_name == config.MARKET_TZ_NAME:
        close = start.normalize() + pd.Timedelta(hours=config.MARKET_CLOSE[0], minutes=config.MARKET_CLOSE[1])
        if start < close < end:
            end = close
        opn = start.normalize() + pd.Timedelta(hours=config.MARKET_OPEN[0], minutes=config.MARKET_OPEN[1])
        if start < opn < end:                              # extended hours: the 9:00 pre-market candle ends at the 9:30 open
            end = opn
    return end


def is_bar_closed(start: pd.Timestamp, tz_name: str, now_utc: Optional[datetime] = None) -> bool:
    now_utc = now_utc or datetime.now(timezone.utc)
    end = bar_end(start, tz_name)
    try:
        end_aware = end.tz_localize(tz_name, ambiguous=True, nonexistent="shift_forward")
    except Exception:                                      # unknown tz name -> assume US
        end_aware = end.tz_localize(config.MARKET_TZ_NAME, ambiguous=True, nonexistent="shift_forward")
    return now_utc >= end_aware.tz_convert("UTC").to_pydatetime()


def drop_forming_candle(df: pd.DataFrame, tz_name: str, now_utc: Optional[datetime] = None):
    """Remove the last candle if it has not closed yet. Returns (df, dropped?)."""
    if df is None or len(df) == 0:
        return df, False
    if is_bar_closed(df["Date"].iloc[-1], tz_name, now_utc):
        return df, False
    return df.iloc[:-1].reset_index(drop=True), True


def _history(ticker: str, interval: str, start: str):
    import yfinance as yf                                  # lazy import: tests run without it installed
    t = yf.Ticker(ticker)
    kw = dict(start=start, interval=interval, auto_adjust=False, actions=False,
              prepost=not config.REGULAR_HOURS_ONLY, timeout=config.REQUEST_TIMEOUT)
    try:
        return t.history(raise_errors=True, **kw)
    except TypeError:                                      # very old yfinance without raise_errors
        return t.history(**kw)


def _is_no_data(e: Exception) -> bool:
    n = type(e).__name__
    msg = str(e).lower()
    return (any(k in n for k in ("PricesMissing", "TickerMissing", "TzMissing", "InvalidPeriod"))
            or "delisted" in msg or "no data found" in msg or "no price data" in msg)


def fetch_history(ticker: str, lookback_days: int = config.HOURLY_LOOKBACK_DAYS,
                  interval: str = config.INTERVAL, history_fn=None,
                  now_utc: Optional[datetime] = None, drop_forming: bool = True) -> Fetch:
    history_fn = history_fn or _history                    # looked up at call time (patchable in tests)
    start = (now_market() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    def call():
        yf_cooldown_wait()
        return history_fn(ticker, interval, start)

    def on_retry(e: Exception):
        if is_retryable_yf_error(e):
            yf_cooldown_trigger(20.0)

    try:
        raw = retry(call, tries=config.MAX_RETRIES, base=config.RETRY_BASE,
                    retryable=lambda e: not _is_no_data(e), on_retry=on_retry,
                    sleep=lambda s: time.sleep(s + random.uniform(0, config.RETRY_BASE)))
    except Exception as e:                                 # noqa: BLE001
        if _is_no_data(e):
            return Fetch("NO_DATA", ticker=ticker, error=str(e)[:160])
        return Fetch("ERROR", ticker=ticker, error=f"{type(e).__name__}: {str(e)[:160]}")
    df, tz_name = normalize(raw)
    if df is None:
        return Fetch("NO_DATA", ticker=ticker, error="empty history")
    dropped = False
    if drop_forming:
        df, dropped = drop_forming_candle(df, tz_name, now_utc)
        if len(df) == 0:
            return Fetch("NO_DATA", ticker=ticker, tz=tz_name, error="no closed candles")
    return Fetch("OK", df=df, ticker=ticker, tz=tz_name, dropped_forming=dropped)
