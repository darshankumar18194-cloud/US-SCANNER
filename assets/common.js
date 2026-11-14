/* Shared by the dashboard and the stock page. No build step, no dependencies. */
const DATA_URL = '../shared-data/stocks.json';
const DAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
const PART_LABELS = [['pattern','Pattern strength',30],['smc','SMC alignment',25],['fresh','Freshness',20],['volume','Volume',15],['squeeze','Squeeze',10]];

const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const store = {
  get(k){ try { return localStorage.getItem(k); } catch(e){ return null; } },
  set(k,v){ try { localStorage.setItem(k,v); } catch(e){} }
};
async function loadData(){
  const r = await fetch(DATA_URL + '?t=' + Math.floor(Date.now() / 60000), {cache: 'no-store'});
  if (!r.ok) throw new Error('stocks.json returned HTTP ' + r.status);
  const d = await r.json();
  if (!d || !Array.isArray(d.stocks) || !d.meta) throw new Error('stocks.json has an unexpected shape');
  setRef(d.meta.latest_candle);
  return d;
}
function safeUrl(u){ return typeof u === 'string' && /^https:\/\/www\.tradingview\.com\//.test(u) ? u : '#'; }
function dirLabel(d){ return d === 'bull' ? 'Bullish' : d === 'bear' ? 'Bearish' : 'Neutral'; }
function money(v){ return v == null ? '–' : Number(v).toFixed(v >= 1 ? 2 : 4); }
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
let REF_DAY = null;            // the day of the latest candle; older times are shown with their date
function setRef(t){ const m = /^(\d{4})-(\d\d)-(\d\d)/.exec(t || ''); if (m) REF_DAY = Date.UTC(+m[1], +m[2]-1, +m[3]); }
function fmtTime(t){            // "2026-10-02 14:30" (exchange-local candle start) -> "Fri 2:30 PM", or "18 Sep, 12:30 PM" if older than 5 days
  if (!t) return '–';
  const m = /^(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d)$/.exec(t); if (!m) return t;
  const day = Date.UTC(+m[1], +m[2]-1, +m[3]), d = new Date(day); let h = +m[4]; const ap = h >= 12 ? 'PM' : 'AM'; h = h % 12 || 12;
  const clock = h + ':' + m[5] + ' ' + ap;
  if (REF_DAY != null && (REF_DAY - day) / 864e5 >= 6) return (+m[3]) + ' ' + MONTHS[+m[2]-1] + (REF_DAY - day > 300 * 864e5 ? ' ' + m[1] : '') + ', ' + clock;
  return DAYS[d.getUTCDay()] + ' ' + clock;
}
function tzShort(tz){ return tz === 'America/New_York' ? 'ET' : (tz || '').split('/').pop().replace('_',' '); }
function agoText(n){ return n === 0 ? 'latest candle' : n + ' candle' + (n === 1 ? '' : 's') + ' ago'; }
function isLive(s){ return s.scanner_status === 'OK' || s.scanner_status === 'STALE'; }
function signalsIn(stock, win){ return (stock.signals || []).filter(g => g.since <= win); }
function bestOf(sigs){ return sigs.slice().sort((a,b) => b.score - a.score || a.since - b.since)[0] || null; }
function colorVar(d){ return d === 'bear' ? 'bear' : d === 'bull' ? 'bull' : 'amber'; }

function strip(stock, sig, n){
  n = n || 14;
  const cs = (stock.candles || []).slice(-n); if (!cs.length) return '';
  const W = 8, H = 30, hi = Math.max(...cs.map(c => c[2])), lo = Math.min(...cs.map(c => c[3]));
  const y = v => 2 + (hi - v) / (hi - lo || 1) * (H - 4);
  const width = cs.length * W;
  let s = '<svg class="strip" width="' + width + '" height="' + H + '" viewBox="0 0 ' + width + ' ' + H + '" role="img" aria-label="Last ' + cs.length + ' hourly candles">';
  if (sig){
    const end = cs.length - 1 - sig.since, a = end - (sig.len - 1);
    if (end >= 0) { const x0 = Math.max(0, a); s += '<rect x="' + (x0*W) + '" y="0" width="' + ((end - x0 + 1)*W) + '" height="' + H + '" rx="2" fill="var(--' + colorVar(sig.dir) + '-soft)"/>'; }
  }
  cs.forEach((c,i) => {
    const col = c[4] >= c[1] ? 'var(--bull)' : 'var(--bear)', x = i*W + W/2;
    s += '<line x1="' + x + '" x2="' + x + '" y1="' + y(c[2]) + '" y2="' + y(c[3]) + '" stroke="' + col + '"/>';
    s += '<rect x="' + (i*W+1.5) + '" y="' + y(Math.max(c[1],c[4])) + '" width="' + (W-3) + '" height="' + Math.max(1.2, Math.abs(y(c[1]) - y(c[4]))) + '" fill="' + col + '"/>';
  });
  return s + '</svg>';
}

/* theme: follows the device until the person picks one */
(function(){
  const saved = store.get('scanner-theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);
})();
function bindTheme(btn){
  btn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next); store.set('scanner-theme', next);
  });
}
/* watchlist lives in this browser only */
function getWatch(){ try { const a = JSON.parse(store.get('scanner-watch') || '[]'); return Array.isArray(a) ? a : []; } catch(e){ return []; } }
function setWatch(a){ store.set('scanner-watch', JSON.stringify(a)); }

/* ---------- confirmation rule for the pattern tables ----------
   Same rule as the SMC + pattern tables (Candlestick Handbook): a candle pattern only counts once a LATER candle has closed
   beyond it. Returns 'confirmed', 'pending', 'failed' or 'trigger'.
   - candlestick patterns: the scanner already decided (g.confirm)
   - Turtle Soup / Undercut & Rally: decided here from the saved candles. Bullish: a later close above the signal candle's
     high confirms it, a later close below its low kills it (bearish is the mirror image). Whichever comes first wins.
   - squeeze / volume / trend / structure events: the close beyond a level IS the trigger, so there is nothing left to confirm */
function patStatus(s, g){
  if (g.confirm) return g.confirm;
  if (g.name === 'Turtle Soup' || g.name === 'Undercut & Rally') {
    const cs = s.candles || [], n = cs.length, end = n - 1 - g.since;
    if (end < 0 || end >= n) return 'pending';
    const bull = g.dir === 'bull';
    for (let i = end + 1; i < n; i++) {
      if (bull ? cs[i][4] > cs[end][2] : cs[i][4] < cs[end][3]) return 'confirmed';
      if (bull ? cs[i][4] < cs[end][3] : cs[i][4] > cs[end][2]) return 'failed';
    }
    return 'pending';
  }
  return 'trigger';
}
