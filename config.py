"""Central settings. Everything tunable lives here."""
import os
from pathlib import Path

# SCANNER_HOME lets the tests (or you) point the whole system at another folder.
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.environ.get("SCANNER_HOME") or CODE_DIR)
DATA_DIR = BASE_DIR / "shared-data"
LOG_DIR = BASE_DIR / "logs"
STOCKS_JSON = DATA_DIR / "stocks.json"        # scan output - the dashboards read this
SYMBOLS_CSV = DATA_DIR / "symbols.csv"        # YOUR list - the only place symbols are defined
PIPELINE_LOCK = DATA_DIR / ".pipeline.lock"

# ---- Market data -------------------------------------------------------------
INTERVAL = "1h"
HOURLY_LOOKBACK_DAYS = 700        # Yahoo refuses ranges over 730 days. The start is a local-midnight date, which for Asian
                                  # exchanges lies up to ~1.6 days earlier than for New York, so 729 failed for every KS/HK/SZ ticker.
DATE_FMT = "%Y-%m-%d %H:%M"       # candle START time, in the exchange's own local time
CALC_BARS = 700                   # bars fed to the pattern/indicator engine (SMC uses everything)
# ONE SWITCH for extended hours (EXPERIMENT).  False = regular session only (the original, tuned behaviour).
# True = also pre-market and after-hours candles from Yahoo (about 4:00 AM - 8:00 PM New York time, ~16 candles a day).
# Yahoo has no overnight (8 PM - 4 AM ET) candles, so this is NOT a full 24-hour feed. Every candle-count setting below
# scales automatically with BARS_PER_DAY, so flipping this back to False restores the original numbers exactly.
EXTENDED_HOURS = False
REGULAR_HOURS_ONLY = not EXTENDED_HOURS      # passed to yfinance as prepost=not REGULAR_HOURS_ONLY
BARS_PER_DAY = 16 if EXTENDED_HOURS else 7   # 1H candles per day per symbol

# ---- Market-structure (SMC) engine - unchanged from your original project -----
SWING_LEN = 50                    # LuxAlgo "swings length"
MIN_BARS = SWING_LEN * 2 + 20     # fewer bars than this -> INSUFFICIENT_DATA

# ---- US market clock ---------------------------------------------------------
MARKET_TZ_NAME = "America/New_York"
MARKET_OPEN = (9, 30)
MARKET_CLOSE = (16, 0)

# ---- Signals -------------------------------------------------------------------
STORE_BARS = 5 * BARS_PER_DAY     # keep every signal that formed within the last N candles (5 trading days) = 35 regular / 80 extended
FRESH_FADE_BARS = 20              # freshness score falls from full marks to 0 over this many candles
RANGE_KEEP_BARS = BARS_PER_DAY    # Inside Bar / NR4 / NR7 are one-candle triggers: keep them only this long (one day)
MAX_SIGNALS = 60                  # hard cap per symbol (keeps stocks.json small)
RECENT_CANDLES = STORE_BARS + 5   # candles saved per symbol (mini chart + confirmation time/price): must cover STORE_BARS
WINDOW_OPTIONS = [3, 6, 7, 10, 14, 20, 35] if not EXTENDED_HOURS else [3, 8, 16, 24, 32, 48, 80]   # dashboard "formed within last" filter
TOP_BOARD_WINDOW = 20             # the Top bullish / Top bearish boards look at this many candles
TOP_BOARD_SIZE = 20

PCT_LOOKBACK = 63 * 7 if not EXTENDED_HOURS else 28 * BARS_PER_DAY   # 441 candles regular (63 days) / 448 extended (28 days); must stay <= CALC_BARS
SQUEEZE_PCT = 0.20                # compression = in the lowest 20% of its own recent history
BB_LEN, BB_MULT = 20, 2.0
KC_MULT = 1.5                     # Keltner channel width for the TTM squeeze
BREAKOUT_LOOKBACK = 20            # "recent range" used by breakout rules
VDU_RATIO = 0.6                   # volume dry-up: typical relative volume below this
VOL_SPIKE_RATIO = 2.0             # volume spike: relative volume at or above this
SLOT_VOL_DAYS = 50                # relative volume = volume / average of the same time-of-day slot over this many sessions
PIVOT_VOL_LOOKBACK = 10           # pocket pivot: compare against down bars in the last N candles
FVG_MIN_ATR = 0.25                # fair value gap must be at least this many ATRs wide
DIV_LOOKBACK = 30                 # OBV / A-D divergence window (candles)

# ---- Downloading -------------------------------------------------------------
SCAN_WORKERS = 4
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_BASE = float(os.environ.get("SCANNER_RETRY_BASE", 2.0))   # seconds; exponential backoff base
STALE_WARN_FRACTION = 0.25        # validate step warns when more than this share of symbols is STALE/ERROR

# ---- Support / resistance levels (levels.py, ported from the book "Auto Touch S/R v4") ----------------------------
LEVELS_ON = True                  # False = skip the level finder entirely (stocks.json then has no "levels")

# ---- Trade plans (plans.py): entry / SL / TP1 / TP2 for confirmed SMC + pattern setups, tracked until they finish ---
PLANS_ON = True
PLANS_JSON = DATA_DIR / "plans.json"     # kept across scans (never trimmed): the record of every plan and its outcome
