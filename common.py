"""Shared helpers: market clock, strict JSON I/O, locks, logging, retry/cooldown."""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import config

for _stream in (sys.stdout, sys.stderr):          # Windows consoles default to cp1252
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

MARKET_TZ = ZoneInfo(config.MARKET_TZ_NAME)       # observes DST - never use a fixed UTC offset
IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------- time
def now_market() -> datetime:
    return datetime.now(MARKET_TZ)


def _at(day: datetime, hm: tuple) -> datetime:
    return day.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)


def is_market_open(now: Optional[datetime] = None) -> bool:
    now = now or now_market()
    return now.weekday() < 5 and _at(now, config.MARKET_OPEN) <= now < _at(now, config.MARKET_CLOSE)


def run_type_for(now: Optional[datetime] = None) -> str:
    """'mid-session' while the US market is open, otherwise 'after-close'."""
    return "mid-session" if is_market_open(now) else "after-close"


# ---------------------------------------------------------------------------- numbers / json
def safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, "", "NA"):
            return None
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def sanitize(obj: Any) -> Any:
    """NaN / Infinity are legal for Python's json module but break the browser's JSON.parse."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):    # numpy scalar
        try:
            return sanitize(obj.item())
        except Exception:                                             # noqa: BLE001
            pass
    return obj


def dumps_strict(obj: Any, pretty: bool = False) -> str:
    kw = {"indent": 2} if pretty else {"separators": (",", ":")}
    return json.dumps(sanitize(obj), allow_nan=False, ensure_ascii=False, **kw)


def atomic_write_text(path, text: str) -> None:
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------- store
def load_store(path=None) -> dict:
    """Return {"meta": {...}, "stocks": [...]}. A missing file is a brand-new repo, not an error."""
    path = path or config.STOCKS_JSON
    if not os.path.exists(path):
        return {"meta": {}, "stocks": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("stocks"), list):
        raise ValueError(f"{path} must be an object with a 'stocks' array")
    data.setdefault("meta", {})
    return data


def save_store(meta: dict, stocks: list, path=None, pretty: bool = False) -> None:
    atomic_write_text(path or config.STOCKS_JSON, dumps_strict({"meta": meta, "stocks": stocks}, pretty))


# ---------------------------------------------------------------------------- locking
class LockBusy(Exception):
    pass


@contextlib.contextmanager
def file_lock(path, wait: int = 0):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    deadline = time.time() + wait
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() >= deadline:
                raise LockBusy(str(path))
            time.sleep(0.5)
    try:
        yield
    finally:
        try:
            os.remove(str(path))
        except OSError:
            pass


# ---------------------------------------------------------------------------- logging
def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(config.LOG_DIR / f"{name}_{datetime.now():%Y-%m-%d}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError:
        pass
    log.propagate = False
    return log


# ---------------------------------------------------------------------------- retry / rate limits
def retry(fn: Callable, tries: int = config.MAX_RETRIES, base: float = 1.5,
          retryable: Callable[[Exception], bool] = lambda e: True, sleep=time.sleep,
          on_retry: Optional[Callable[[Exception], None]] = None):
    last: Optional[Exception] = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                                        # noqa: BLE001
            last = e
            if not retryable(e) or i == tries - 1:
                raise
            if on_retry:
                on_retry(e)
            sleep(base ** (i + 1) + (i * 0.37))
    raise last  # pragma: no cover


def is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "RateLimit" in type(e).__name__ or "429" in msg or "too many requests" in msg


def is_auth_hiccup(e: Exception) -> bool:
    """Yahoo's crumb/session token going bad mid-run - a flaky moment, not a code problem."""
    msg = str(e).lower()
    return "invalid crumb" in msg or ("401" in msg and "unauthorized" in msg)


def is_retryable_yf_error(e: Exception) -> bool:
    return is_rate_limit(e) or is_auth_hiccup(e)


_COOLDOWN_LOCK = threading.Lock()
_COOLDOWN_UNTIL = 0.0


def yf_cooldown_wait() -> None:
    """Block until any cooldown has elapsed, so no thread piles onto a broken Yahoo session."""
    remaining = _COOLDOWN_UNTIL - time.time()
    if remaining > 0:
        time.sleep(remaining)


def yf_cooldown_trigger(seconds: float = 20.0) -> None:
    global _COOLDOWN_UNTIL
    with _COOLDOWN_LOCK:
        _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.time() + seconds)
