# Pattern Scanner

Scans **only the symbols in `shared-data/symbols.csv`** (your CoinSwitch US-stock-futures list) on **closed 1-hour candles**, twice a day, and publishes a dashboard. Nothing to host, no server, no paid service.

Each symbol gets
* **SMC market structure** (BOS / CHoCH, strong and weak highs and lows) from your original engine,
* **60 candlestick patterns** from the Candlestick Handbook (Morning Star, Evening Star, Engulfing, ... all of them),
* **15 compression / volume / trend / structure events** from the Pre-Breakout guides (squeeze breakouts, pocket pivot, fair value gaps, turtle soup ...),
* a **confluence score** from 0 to 100 that ranks setups. It is not a win probability.

See [docs/SIGNALS.md](docs/SIGNALS.md) for every signal and its exact rule.

---

## Set it up once (about 10 minutes)

You need a free GitHub account. Everything happens in the browser.

**1. Create a new repository.** On github.com click **+** then **New repository**. Name it, for example, `pattern-scanner`. Choose **Public** (GitHub Pages and unlimited Actions minutes are free for public repositories). Tick **Add a README file** so the repository is not empty. Click **Create repository**.

**2. Upload this project.** Unzip the download. In your repository click **Add file**, then **Upload files**. Open the unzipped `pattern-scanner` folder and drag **everything inside it** into the page, including the hidden `.github` folder. If your computer hides it, show hidden files first (Mac: Cmd+Shift+. in Finder; Windows: View, Show, Hidden items). Without `.github/workflows/scan.yml` nothing will run. Click **Commit changes**.

**3. Let the workflow save results.** In the repository go to **Settings, Actions, General**. Scroll to **Workflow permissions**, choose **Read and write permissions**, click **Save**.

**4. Turn on the website.** Go to **Settings, Pages**. Under **Build and deployment** set **Source: Deploy from a branch**, **Branch: main**, folder **/ (root)**, click **Save**. After a minute the page shows your address, `https://YOUR-NAME.github.io/pattern-scanner/`.

**5. Run the first scan by hand.** Go to the **Actions** tab. If GitHub asks, click the green button to enable workflows. Open **Scan market** in the list on the left, click **Run workflow**, then the green **Run workflow** button. It takes a few minutes. A green tick means it worked.

**6. Open your dashboard.** Visit the Pages address from step 4. Bookmark it. From now on it updates by itself.

### What runs automatically

| When | IST | What |
|---|---|---|
| Mon to Fri 17:30 UTC | 23:00 | **Mid-session look.** The candle still forming is excluded, so nothing flickers. |
| Every day 00:02 UTC | 05:32 | **Final run** after the close, with the whole day's candles. Overwrites the mid-session result. |

Both times are inside or after the US session in summer and winter time. GitHub can start a scheduled run 10 to 60 minutes late. That is harmless.

---

## Changing your symbol list

`shared-data/symbols.csv` is the **only** place symbols are defined. Columns:

| Column | Meaning |
|---|---|
| `Symbol` | Your own label (the CoinSwitch name) |
| `Name`, `Type` | Display name; `Stock` or `ETF` |
| `Yahoo Symbol` | The real Yahoo Finance ticker. Leave blank when it is the same as `Symbol`. Example: `AMDSTOCK` has `AMD`, `SAMSUNG` has `005930.KS` |
| `Active` | `Y` to scan, `N` to keep the row but skip it |
| `Notes` | Free text |

Three ways to change it:

1. **Edit the file on GitHub.** Open `shared-data/symbols.csv`, click the pencil icon, edit, **Commit changes**. The next run picks it up. To scan immediately, use **Actions, Scan market, Run workflow**.
2. **Add stocks from the Run workflow box.** In **Actions, Scan market, Run workflow** type `NVDA, PLTR, SPY` in the *symbols to ADD* box. Each ticker is checked against Yahoo first. A ticker Yahoo does not know is skipped with a warning and does not stop the scan.
3. **On your computer:** `python add_stock.py NVDA --name "NVIDIA"`

### Rows that need your attention
Open the dashboard's **Data issues** panel (bottom of the page). In the supplied CSV:
* **13 rows are switched off** (`Active = N`): 7 private companies with no public price (ANTHROPIC, OPENAI, SPCX ...), `NOW` (the CSV calls it a crypto token, not ServiceNow), and a few unclear rows (DRAM, MINIMAX, ZHIPU, TMX, SKHY).
* **Korean, Hong Kong and Shenzhen tickers** (e.g. `005930.KS`, `0700.HK`) and the `...STOCK` names were mapped to Yahoo tickers by me. **Please spot-check them.** A wrong mapping scans the wrong company without any error.
* Any row Yahoo cannot serve shows as *No Yahoo data* in the issues panel instead of failing silently.

---

## Running it on your own computer (optional)

```
pip install -r requirements.txt
python scan.py                 # scan everything
python scan.py --symbols NVDA  # just one
python validate_data.py        # check the output
python -m http.server          # then open http://localhost:8000/
python -W ignore -m unittest discover -s tests -t .    # run the tests
```

Open the dashboard through `http://localhost:8000/`, not by double-clicking the HTML file (browsers block file access from `file://`).

---

## Settings

Everything is in `config.py`. The ones people change:

| Setting | Default | Meaning |
|---|---|---|
| `STORE_BARS` | 35 | A signal is kept for this many candles (5 trading days) |
| `FRESH_FADE_BARS` | 20 | Freshness score falls to 0 over this many candles |
| `WINDOW_OPTIONS` | 3, 6, 7, 10, 14, 20, 35 | Choices in the "formed within last" filter |
| `SWING_LEN` | 50 | SMC swing length |
| `SQUEEZE_PCT` | 0.20 | "Compressed" = lowest 20% of its own recent history |

Change the times in `.github/workflows/scan.yml` (cron is UTC).

---

## How to read the dashboard

* **Candles**: a candle is labelled by its **start** time in the exchange's own time zone (ET for US stocks). "Fri 2:30 PM" means the candle from 2:30 to 3:30 PM ET.
* **"2 candles ago"**: counted in candles, not hours. A trading day has 7 candles, so 7 is about one day. Overnight and weekends do not count.
* **Latest candle**: 0 ago = the most recent closed candle.
* **Pending / confirmed / failed**: the Handbook's rule is to wait for a later close beyond the pattern. Pending means that has not happened yet.
* **SMC + pattern tables**: a Strong/Weak Low (or High) with a strong reversal pattern (3 stars or more, plus Turtle Soup and Undercut & Rally) whose lowest (highest) point is within 1 ATR of the level, formed at or after the level was set, with no new low (high) since, after a real move into the level (the 8 closes before the pattern at least 1 ATR beyond it), and with price not already more than 2 ATR away. If nothing qualifies the table stays empty on purpose; the Top 20 boards look at the last 10 candles. Ranked by SMC rank, ties by pattern score. Only **confirmed** setups are listed: a later candle has already closed beyond the pattern, so the turn has started. A pattern still waiting for that close is not shown. Dashboard-only logic, no scanner change. The Bullish and Bearish pattern tables use the same confirmation idea: *Confirmed only* by default (squeeze, volume and trend events show as *Trigger*), switchable with the **Pattern confirmation** filter.
* **Stale**: the last download for that symbol failed, so the previous good result is shown. It recovers by itself on the next good run.
* **Not scanned** symbols are listed in *Data issues*.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Dashboard says "No scan results yet" | Run **Scan market** once (step 5). |
| Dashboard shows a red "could not load stocks.json" banner | Pages is not on yet (step 4), or you opened the HTML file from your computer. |
| Actions run fails at **Commit results** with a permission error | Step 3: set workflow permissions to **Read and write**. |
| Yellow banner "data is N hours old" | The scheduled run stopped. Open **Actions**. GitHub switches schedules off after about 60 days without repository activity. Click **Enable workflow** and press **Run workflow**. |
| Many symbols **Stale** | Yahoo was rate-limiting. Re-run later. The scan retries and backs off automatically. |
| A symbol is *No Yahoo data* | Its `Yahoo Symbol` is wrong, or Yahoo has no 1-hour data for it. Fix it in the CSV. |
| Validation failed, nothing was committed | The previous dashboard data stays online. The log says why (for example Yahoo unreachable). |

## Limits you should know
* Data comes from Yahoo Finance's free feed (unofficial, can have gaps). Regular trading hours only; pre- and after-market are not scanned.
* The scan reads Yahoo prices, not CoinSwitch prices. CoinSwitch futures may trade around the clock and can differ from the US session.
* Candlestick reliability figures in the Handbook come from daily charts. On 1-hour charts the same shapes give more false signals. Use the score to rank, then check the chart.
* The repository grows by roughly 0.4 MB per scan commit. That is fine for years; if you ever want it smaller, delete old history or re-create the repository.
* Not financial advice.
