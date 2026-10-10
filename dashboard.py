"""
Local dashboard for Acorn 1.0.  Run:  python dashboard.py
Then open http://127.0.0.1:8000

It READS data/market.csv and models/model.joblib and can run snapshot, train
and backtest for you. It never touches a wallet key and cannot trade.

Remote access (e.g. from your phone): don't expose it to the internet.
Use an SSH tunnel or a private network such as Tailscale instead. If you
must bind to another address, set DASH_TOKEN and pass --host.
"""
import argparse
import hmac
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from indicators import READY, add_indicators, window_history  # noqa: E402
from data_quality import clean_market_frame  # noqa: E402
from model_store import load_model as trusted_load, predict_probability  # noqa: E402
from process_lock import ProcessLock  # noqa: E402

DATA = ROOT / "data" / "market.csv"
MODEL = ROOT / "models" / "model.joblib"
HEALTH = ROOT / "data" / "collector_health.json"
BACKUPS = ROOT / "backups"
LOGS = {"train": ROOT / "data" / "last_train.txt", "backtest": ROOT / "data" / "last_backtest.txt"}
SCRIPTS = {"train": "train.py", "backtest": "backtest.py"}
TOKEN = os.environ.get("DASH_TOKEN", "")
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
STALE_AFTER_S = 180  # collector scans every 60s by default
NUMERIC_COLUMNS = (
    "price_usd", "liquidity_usd", "volume_24h", "volume_5m", "buys_5m", "sells_5m",
    "txns_24h", "price_change_5m", "price_change_1h", "pair_age_minutes", "market_cap",
    "fdv", "volume_1h", "buys_1h", "sells_1h",
)

_cache = {"csv": (None, None), "model": (None, None), "sig": (None, None), "quality": None}
_data_error = None
_model_error = None
_job_lock = threading.Lock()
_job = {"name": None, "started": None}


# ---------- data ----------
def load_data():
    global _data_error
    if not DATA.exists():
        _data_error = "market.csv does not exist yet"
        return None
    mtime = DATA.stat().st_mtime
    if _cache["csv"][0] == mtime:
        return _cache["csv"][1]
    try:
        df, quality = clean_market_frame(pd.read_csv(DATA))
    except Exception:
        _data_error = "market.csv is temporarily unreadable or has invalid schema"
        return _cache["csv"][1]  # file mid-write; keep the last good copy
    if quality["future_rows"]:
        count = quality["future_rows"]
        df = df.loc[df["timestamp"] <= pd.Timestamp.now(tz="UTC")].copy()
        _data_error = f"Ignored {count} future-dated market rows; check the collector clock."
    else:
        _data_error = None
    _cache["csv"] = (mtime, df)
    _cache["quality"] = quality
    return df


def load_model():
    global _model_error
    if not MODEL.exists():
        _model_error = "No local trained model is saved."
        return None
    mtime = MODEL.stat().st_mtime
    if _cache["model"][0] == mtime:
        return _cache["model"][1]
    try:
        bundle = trusted_load(MODEL)  # only the local, hash-checked model
    except Exception:
        _model_error = "Saved model failed its integrity or bundle checks. Retrain locally."
        return None
    _cache["model"] = (mtime, bundle)
    _model_error = None
    return bundle


def iso(ts):
    if isinstance(ts, str):
        return ts
    return ts.isoformat() if ts is not None and not pd.isna(ts) else None


def finite_number(value):
    """Return a finite float, or None for missing and malformed values."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def build_signals(df, bundle):
    cutoff = df["timestamp"].max() - pd.Timedelta(minutes=30)
    recent = df.loc[df["timestamp"] >= cutoff].sort_values("timestamp")
    latest = recent.groupby("token_address", sort=False).tail(1)
    latest_prices = pd.to_numeric(latest["price_usd"], errors="coerce")
    valid_latest = latest_prices.notna() & np.isfinite(latest_prices) & (latest_prices > 0)
    active = latest.loc[valid_latest, "token_address"].unique()
    if not len(active):
        return []

    sub = df[df["token_address"].isin(active)].copy()
    prices = pd.to_numeric(sub["price_usd"], errors="coerce")
    valid_prices = prices.notna() & np.isfinite(prices) & (prices > 0)
    sub = sub.loc[valid_prices].copy()
    sub["price_usd"] = prices.loc[valid_prices]
    sub = window_history(sub)  # same window as paper trader
    x = add_indicators(sub.copy())
    last = x.groupby("token_address").tail(1).copy()
    counts = sub.groupby("token_address").size()

    last["prob"] = float("nan")
    if bundle:
        feats = bundle["features"]
        ok = last.dropna(subset=READY)
        if not ok.empty:
            finite_features = np.isfinite(ok[feats].to_numpy(dtype=float)).all(axis=1)
            ok = ok.loc[finite_features]
        if not ok.empty:
            last.loc[ok.index, "prob"] = predict_probability(bundle, ok[feats])

    rows = []
    for _, r in last.iterrows():
        tok = r["token_address"]
        spark = [finite_number(v) for v in x.loc[x["token_address"] == tok, "price_usd"].tail(60)]
        spark = [v for v in spark if v is not None]
        rows.append({
            "symbol": str(r.get("symbol", "")),
            "token": str(tok),
            "price": finite_number(r["price_usd"]),
            "liquidity": finite_number(r.get("liquidity_usd")),
            "ret5": finite_number(r.get("return_5")),
            "rsi": finite_number(r.get("rsi14")),
            "prob": finite_number(r.get("prob")),
            "rows": int(counts.get(tok, 0)),
            "spark": spark,
            "url": str(r.get("url", "")),
        })
    rows.sort(key=lambda d: (d["prob"] is None, -(d["prob"] or 0)))
    return rows[:40]


def build_backups():
    out = []
    if BACKUPS.exists():
        for d in sorted((p for p in BACKUPS.iterdir() if p.is_dir()), reverse=True)[:10]:
            files = {f.name: round(f.stat().st_size / 1024) for f in d.iterdir() if f.is_file()}
            out.append({"name": d.name, "files": files})
    return out


def read_log(name):
    p = LOGS[name]
    if not p.exists():
        return None
    with p.open("rb") as f:
        f.seek(max(0, p.stat().st_size - 6000))
        text = f.read().decode("utf-8", errors="replace")
    return {"text": text,
            "when": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()}


def build_state():
    now = datetime.now(timezone.utc)
    bundle = load_model()
    df = load_data()
    state = {"now": now.isoformat(), "data": None, "signals": [], "model": None,
             "model_error": _model_error,
             "backups": build_backups(),
             "data_quality": _cache.get("quality"), "data_error": _data_error,
             "collector": None,
             "logs": {k: read_log(k) for k in LOGS},
             "job": {"name": _job["name"],
                     "seconds": int(time.time() - _job["started"]) if _job["name"] else None}}

    if HEALTH.exists():
        try:
            health = json.loads(HEALTH.read_text(encoding="utf-8"))
            updated = pd.to_datetime(health.get("updated_at"), utc=True, errors="coerce")
            age = (pd.Timestamp(now) - updated).total_seconds() if not pd.isna(updated) else None
            health["age_s"] = None if age is None else max(0, int(age))
            health["running"] = health.get("status") in ("starting", "ok") and age is not None and 0 <= age < STALE_AFTER_S
            state["collector"] = health
        except (OSError, ValueError, TypeError):
            state["collector"] = {"status": "invalid health record", "running": False}

    horizon = bundle["horizon_minutes"] if bundle else 15
    if df is not None and not df.empty:
        last, first = df["timestamp"].max(), df["timestamp"].min()
        age = (now - last.to_pydatetime()).total_seconds()
        labelable = int((df["timestamp"] <= last - pd.Timedelta(minutes=horizon)).sum())
        state["data"] = {
            "rows": int(len(df)), "tokens": int(df["token_address"].nunique()),
            "first": iso(first), "last": iso(last), "age_s": int(age),
            "running": 0 <= age < STALE_AFTER_S, "labelable": labelable, "need": 500,
            "size_kb": round(DATA.stat().st_size / 1024),
        }
        try:
            state["signals"] = build_signals(df, bundle)
        except Exception as e:  # never let the page die on a bad row
            state["signals_error"] = str(e)[:200]

    if bundle:
        drift = [item for item in bundle.get("feature_drift", [])
                 if isinstance(item, dict) and finite_number(item.get("psi")) is not None]
        drift.sort(key=lambda item: float(item["psi"]), reverse=True)
        trained = bundle.get("trained_at")
        if hasattr(trained, "isoformat"):
            trained = trained.isoformat()
        if not isinstance(trained, str):
            try:
                trained = datetime.fromtimestamp(MODEL.stat().st_mtime, timezone.utc).isoformat()
            except OSError:
                trained = None
        state["model"] = {
            "horizon": bundle.get("horizon_minutes"), "target": bundle.get("target_return"),
            "threshold": bundle.get("threshold"), "test_start": iso(bundle.get("test_start")),
            "trained": trained,
            "calibration": bundle.get("calibration_method", "uncalibrated"),
            "feature_version": bundle.get("feature_version", "legacy"),
            "feature_drift": [{"feature": item["feature"], "psi": item["psi"],
                               "test_missing_rate": item.get("test_missing_rate")}
                              for item in drift[:3]],
        }
    return state


# ---------- actions ----------
def run_snapshot():
    try:
        r = subprocess.run([sys.executable, "-c", "from hotkeys import snapshot; snapshot()"],
                           capture_output=True, text=True, timeout=120, cwd=ROOT)
        return (r.stdout + r.stderr).strip()[-6000:]
    except subprocess.TimeoutExpired:
        return "Snapshot timed out after 2 minutes."
    except OSError as exc:
        return f"Could not start snapshot: {type(exc).__name__}: {exc}"


def start_job(name):
    if not _job_lock.acquire(blocking=False):
        return False

    def work():
        try:
            _job.update(name=name, started=time.time())
            LOGS[name].parent.mkdir(parents=True, exist_ok=True)
            try:
                with ProcessLock(ROOT / "data" / "dashboard_jobs.lock"):
                    r = subprocess.run([sys.executable, str(ROOT / SCRIPTS[name])],
                                       capture_output=True, text=True, timeout=3600, cwd=ROOT)
                    text = r.stdout + (("\n" + r.stderr) if r.stderr.strip() else "")
                    text += f"\n[exit code {r.returncode}]"
            except subprocess.TimeoutExpired:
                text = "[timed out after 1 hour]"
            except RuntimeError as exc:
                text = f"[job refused: {exc}]"
            except Exception as exc:
                text = f"[job failed: {type(exc).__name__}: {exc}]"
            tmp = LOGS[name].with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                f.write(text[-500_000:])
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(LOGS[name])
        finally:
            _job.update(name=None, started=None)
            _job_lock.release()

    threading.Thread(target=work, daemon=True).start()
    return True


# ---------- web ----------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                         "connect-src 'self'; img-src data:")
        self.end_headers()
        self.wfile.write(raw)

    def _ok(self, need_token=True):
        raw = self.headers.get("Host") or ""
        if raw.startswith("["):
            host = raw.split("]")[0] + "]"
        else:
            host = raw.rsplit(":", 1)[0] if ":" in raw else raw
        if self.server.local_only and host not in ALLOWED_HOSTS:
            self._send(403, '{"error":"bad host"}')
            return False
        if TOKEN and need_token:
            # Header only: a token in the URL leaks into history and logs.
            given = self.headers.get("X-Token", "")
            if not hmac.compare_digest(given, TOKEN):
                self._send(401, '{"error":"token required"}')
                return False
        return True

    def do_GET(self):
        path = urlparse(self.path).path
        if not self._ok(need_token=(path != "/")):
            return
        if path == "/":
            self._send(200, PAGE, "text/html")  # static page, no data; API calls need the token
        elif path == "/api/state":
            self._send(200, json.dumps(build_state(), allow_nan=False))
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        if not self._ok():
            return
        if self.headers.get("X-Dash") != "1":  # blocks cross-site form posts
            self._send(403, '{"error":"missing header"}')
            return
        path = urlparse(self.path).path
        if path == "/api/snapshot":
            self._send(200, json.dumps({"message": run_snapshot()}))
        elif path in ("/api/train", "/api/backtest"):
            name = path.rsplit("/", 1)[1]
            started = start_job(name)
            self._send(200 if started else 409,
                       json.dumps({"started": started, "message": "started" if started else "another job is running"}))
        else:
            self._send(404, '{"error":"not found"}')


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Acorn 1.0</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#14171c;--mut:#667085;--line:#e4e7ec;--good:#12803c;--bad:#c4321f;--warn:#b26a00;--acc:#3b5bdb}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--fg:#e8eaee;--mut:#9aa3b2;--line:#262b36;--good:#3ecf74;--bad:#ff6b5a;--warn:#f0b24a;--acc:#7c93ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:max(12px,env(safe-area-inset-top)) 12px 40px}
.wrap{max-width:980px;margin:0 auto}
header{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:6px 0 14px;flex-wrap:wrap}
h1{font-size:20px;margin:0}h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 10px}
.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);background:var(--card);border-radius:999px;padding:4px 11px;font-size:13px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--mut)}.dot.on{background:var(--good)}.dot.off{background:var(--bad)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.stat .v{font-size:22px;font-weight:650}.stat .l{color:var(--mut);font-size:12px}
.bar{height:9px;background:var(--line);border-radius:6px;overflow:hidden;margin:6px 0 4px}.bar>i{display:block;height:100%;background:var(--acc)}
.mut{color:var(--mut);font-size:13px}.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:560px}
th,td{padding:8px 6px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;font-size:12px}
.prob{display:inline-block;width:46px;height:7px;background:var(--line);border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:6px}.prob>i{display:block;height:100%;background:var(--acc)}
.hit td:first-child{box-shadow:inset 3px 0 var(--good)}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:9px;padding:9px 14px;cursor:pointer}
button.p{background:var(--acc);border-color:var(--acc);color:#fff}button:disabled{opacity:.5;cursor:default}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
pre{background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:10px;overflow:auto;max-height:260px;margin:6px 0 0;font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap}
.msg{min-height:20px;font-size:13px}
</style></head><body><div class="wrap">
<header><h1>Acorn 1.0</h1><div><span class="pill"><span id="dot" class="dot"></span><span id="status">connecting…</span></span></div></header>
<div id="err" class="card bad" style="display:none"></div>

<div class="card"><h2>Data</h2>
 <div class="grid" id="stats"></div>
 <div id="prog"></div>
</div>

<div class="card"><h2>Model</h2><div id="model" class="mut">Loading…</div></div>

<div class="card"><h2>Signals <span class="mut" id="sigmeta"></span></h2>
 <div class="scroll"><table><thead><tr><th>Token</th><th>Win prob</th><th>Trend</th><th>Price</th><th>Last 5 scans</th><th>RSI</th><th>Liquidity</th></tr></thead><tbody id="sigs"></tbody></table></div>
 <div class="mut" id="signote" style="margin-top:8px"></div>
</div>

<div class="card"><h2>Actions</h2>
 <div class="row"><button id="b-snap">Save snapshot</button><button id="b-train">Train model</button><button id="b-back">Run backtest</button></div>
 <div class="msg mut" id="msg"></div>
 <div class="mut" id="trainlbl" style="margin-top:8px"></div><pre id="trainlog" style="display:none"></pre>
 <div class="mut" id="backlbl" style="margin-top:8px"></div><pre id="backlog" style="display:none"></pre>
</div>

<div class="card"><h2>Backups</h2><div id="backups" class="mut">None yet.</div></div>
<p class="mut">Read-only for trading: this page has no wallet access and cannot place trades. Signals are model scores on collected data, not advice.</p>
</div>
<script>
let TOKEN=sessionStorage.getItem('dash_token')||'';
const H=(t,a,...k)=>{const e=document.createElement(t);for(const[x,y]of Object.entries(a||{}))e.setAttribute(x,y);for(const c of k)e.append(c instanceof Node?c:document.createTextNode(String(c)));return e};
const $=id=>document.getElementById(id);
const hdr=extra=>Object.assign(TOKEN?{'X-Token':TOKEN}:{},extra||{});
const fmtP=p=>p==null?'–':p<0.01?p.toPrecision(4):p<1?p.toFixed(5):p.toFixed(3);
const fmtM=n=>n==null?'–':n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':n.toFixed(0);
const pct=v=>v==null?'–':(v*100).toFixed(2)+'%';
const ago=s=>s<90?s+'s ago':s<5400?Math.round(s/60)+' min ago':s<172800?(s/3600).toFixed(1)+' h ago':Math.round(s/86400)+' d ago';
const when=iso=>iso?new Date(iso).toLocaleString():'–';
function spark(a){const w=84,h=24;if(!a||a.length<2)return H('span',{class:'mut'},'–');
 const lo=Math.min(...a),hi=Math.max(...a),r=(hi-lo)||1;
 const pts=a.map((v,i)=>(i/(a.length-1)*w).toFixed(1)+','+(h-2-(v-lo)/r*(h-4)).toFixed(1)).join(' ');
 const up=a[a.length-1]>=a[0];const s=document.createElementNS('http://www.w3.org/2000/svg','svg');
 s.setAttribute('width',w);s.setAttribute('height',h);s.setAttribute('viewBox','0 0 '+w+' '+h);
 const p=document.createElementNS('http://www.w3.org/2000/svg','polyline');p.setAttribute('points',pts);p.setAttribute('fill','none');
 p.setAttribute('stroke',up?'#12a150':'#d93f2a');p.setAttribute('stroke-width','1.6');s.append(p);return s}
function render(s){
 $('err').style.display='none';
 const d=s.data;
 $('dot').className='dot '+(d?(d.running?'on':'off'):'');
 $('status').textContent=!d?'No data yet — run collector.py':d.running?'Collector running · updated '+ago(d.age_s):'Collector not running · last row '+ago(d.age_s);
 const st=$('stats');st.replaceChildren();
 if(d){[['Rows',d.rows.toLocaleString()],['Tokens',d.tokens],['Collecting since',when(d.first)],['File size',d.size_kb.toLocaleString()+' KB']].forEach(([l,v])=>st.append(H('div',{class:'stat'},H('div',{class:'v'},v),H('div',{class:'l'},l))));
  const pc=Math.min(100,d.labelable/d.need*100);const ready=d.labelable>=d.need;
  $('prog').replaceChildren(H('div',{class:'bar'},H('i',{style:'width:'+pc.toFixed(0)+'%'})),H('div',{class:'mut'},ready?'Enough data to train ('+d.labelable.toLocaleString()+' labelled rows). More days of data = better.':'About '+d.labelable.toLocaleString()+' of '+d.need+' rows needed before training works. Keep the collector running.'));
  const q=s.data_quality;if(q)$('prog').append(H('div',{class:'mut'},'Sources: '+JSON.stringify(q.sources)+' · dropped invalid: '+q.invalid_rows_dropped+' · duplicates: '+q.duplicate_rows_dropped));
  if(s.data_error)$('prog').append(H('div',{class:'warn'},s.data_error));
 }else{st.append(H('div',{class:'mut'},s.data_error||'Start python collector.py, then refresh.'));$('prog').replaceChildren()}
 const m=s.model,me=$('model');me.replaceChildren();
 if(m){[['Trained',when(m.trained)],['Predicts',(m.target*100).toFixed(0)+'%+ gain within '+m.horizon+' min'],['Signal threshold',(m.threshold*100).toFixed(0)+'% probability'],['Calibration',m.calibration||'uncalibrated'],['Feature version',m.feature_version||'legacy'],['Test period starts',when(m.test_start)]].forEach(([l,v])=>me.append(H('div',{},H('span',{class:'mut'},l+': '),v)));if(m.feature_drift&&m.feature_drift.length)me.append(H('div',{class:'mut'},'Largest test-period feature PSI: '+m.feature_drift.map(x=>x.feature+' '+Number(x.psi).toFixed(2)).join(', ')))}
 else me.textContent=s.model_error||'No trained model yet. Collect data, then press “Train model”.';
 const tb=$('sigs');tb.replaceChildren();const thr=m?m.threshold:null;let hits=0;
 for(const r of s.signals){const hit=r.prob!=null&&thr!=null&&r.prob>=thr;if(hit)hits++;
  const tr=H('tr',hit?{class:'hit'}:{});
  const name=H('td',{},H('div',{},r.symbol||'?'),H('div',{class:'mut'},r.token.slice(0,4)+'…'+r.token.slice(-4)));
  const pb=r.prob==null?H('span',{class:'mut'},r.rows<16?'warming up ('+r.rows+')':'–'):H('span',{},H('span',{class:'prob'},H('i',{style:'width:'+Math.round(r.prob*100)+'%'})),(r.prob*100).toFixed(0)+'%');
  const rc=r.ret5==null?'mut':r.ret5>=0?'good':'bad';
  tr.append(name,H('td',{},pb),H('td',{},spark(r.spark)),H('td',{},fmtP(r.price)),H('td',{class:rc},pct(r.ret5)),H('td',{},r.rsi==null?'–':r.rsi.toFixed(0)),H('td',{},'$'+fmtM(r.liquidity)));
  tb.append(tr)}
 $('sigmeta').textContent=s.signals.length?'· '+s.signals.length+' active'+(m?' · '+hits+' above threshold':''):'';
 $('signote').textContent=s.signals_error?'Could not score: '+s.signals_error:!s.signals.length?'Nothing to show yet.':!m?'Train a model to see win probabilities.':'Green edge = above your threshold. These are paper signals only.';
 for(const k of ['train','back']){const key=k==='back'?'backtest':'train';const L=s.logs[key];const pre=$(k+'log'),lb=$(k+'lbl');
  if(L){pre.style.display='block';pre.textContent=L.text;lb.textContent=(key==='train'?'Last training':'Last backtest')+' · '+when(L.when)}else{pre.style.display='none';lb.textContent=''}}
 const busy=!!s.job.name;['b-train','b-back'].forEach(i=>$(i).disabled=busy);
 if(busy)$('msg').textContent='Running '+s.job.name+'… '+s.job.seconds+'s';
 const bk=$('backups');bk.replaceChildren();
 if(s.backups.length)for(const b of s.backups)bk.append(H('div',{},H('b',{},b.name),'  ',Object.entries(b.files).map(([f,k])=>f+' ('+k+' KB)').join(', ')));else bk.textContent='None yet. Press “Save snapshot”.';
}
async function poll(){try{const r=await fetch('/api/state',{headers:hdr()});if(!r.ok){if(r.status===401){const t=prompt('Dashboard token (DASH_TOKEN):');if(t){TOKEN=t;sessionStorage.setItem('dash_token',t)}}throw new Error(r.status===401?'Token required':'Server error '+r.status)};render(await r.json())}
 catch(e){$('err').style.display='block';$('err').textContent=String(e.message||e)}}
async function act(path,label){$('msg').textContent=label+'…';
 try{const r=await fetch(path,{method:'POST',headers:hdr({'X-Dash':'1'})});const j=await r.json();$('msg').textContent=j.message||'Done';}catch(e){$('msg').textContent='Failed: '+e}
 poll()}
$('b-snap').onclick=()=>act('/api/snapshot','Saving snapshot');
$('b-train').onclick=()=>act('/api/train','Starting training');
$('b-back').onclick=()=>act('/api/backtest','Starting backtest');
poll();setInterval(poll,10000);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.port < 1 or args.port > 65535:
        sys.exit("Port must be between 1 and 65535.")
    local_only = args.host in ("127.0.0.1", "localhost", "::1")
    if not local_only and not TOKEN:
        sys.exit("Refusing to listen on a non-local address without DASH_TOKEN set.\n"
                 "Safer: keep the default (127.0.0.1) and use an SSH tunnel or Tailscale.")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.local_only = local_only
    note = "  (enter DASH_TOKEN when the page asks)" if TOKEN else ""
    print(f"Dashboard: http://{args.host}:{args.port}{note}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
