#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screener Nasdaq — compression de volatilité puis cassure (version B)
=====================================================================

Ce que fait le script, à chaque lancement (idéalement chaque soir après la clôture US) :
  1. Récupère la liste officielle des actions cotées au Nasdaq (fichier Nasdaq Trader).
  2. Télécharge ~2 ans de bougies quotidiennes sur Yahoo Finance (bibliothèque yfinance).
  3. Calcule :
       - Version B (SIGNAL)  : compression des bandes de Bollinger la veille, puis cassure du jour
                               avec volume > 1,5 x la moyenne 50 jours.
       - Version A (SETUP)   : tendance + momentum + compression, à moins de 5 % du plus haut 50 jours,
                               pas encore cassé (liste de surveillance).
  4. Génère une page HTML (screener_AAAA-MM-JJ.html) avec un tableau et un graphique par action
     (bougies, Bollinger 20/2, EMA 20/50, volume), et un fichier CSV des résultats.

Installation (une seule fois) :
    pip install yfinance pandas numpy

Utilisation :
    python nasdaq_screener.py                 # tout le Nasdaq, signaux du dernier jour
    python nasdaq_screener.py --days 5        # signaux B des 5 dernières séances
    python nasdaq_screener.py --tickers AAPL,NVDA,AMD   # test rapide sur quelques titres
    python nasdaq_screener.py --limit 300     # test sur les 300 premiers titres de la liste

Remarques :
  - Yahoo Finance n'a pas d'API officielle : yfinance peut ralentir, être limité ou cesser de
    fonctionner. Le script réessaie automatiquement et ignore les titres en échec.
  - Les formules reproduisent la version B testée (largeur de Bollinger = (haute - basse) / moyenne,
    écart-type de population). De petits écarts avec ProRealTime sont possibles.
  - Outil d'aide à l'analyse, pas un conseil en investissement.
"""

import argparse
import datetime as dt
import io
import json
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Paramètres (identiques au code ProRealTime "version B")
# --------------------------------------------------------------------------------------
P = dict(
    len_bb=20,             # période Bollinger
    len_range=65,          # fenêtre du range de largeur de bandes
    sqz_pct=0.30,          # compression = largeur dans les 30 % bas du range
    min_price=5.0,         # prix minimum ($)
    min_dollar_vol=5e6,    # volume moyen 50 j x prix minimum ($)
    vol_mult=1.5,          # volume du jour > 1,5 x moyenne 50 j (veille)
    min_day_gain=1.5,      # hausse du jour minimum (%)
    hh_len=20,             # cassure du plus haut N jours
    near_high_pct=5.0,     # version A : à moins de X % du plus haut 50 j
    chart_bars=180,        # nombre de bougies affichées par graphique
    max_setups=60,         # nombre max de setups (version A) affichés
)

NASDAQ_LIST_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
LWC_URL = "https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"


# --------------------------------------------------------------------------------------
# 1. Univers
# --------------------------------------------------------------------------------------
def parse_nasdaq_list(text: str) -> pd.DataFrame:
    """Parse le fichier nasdaqlisted.txt (séparateur '|', dernière ligne = date de création)."""
    lines = [l for l in text.splitlines() if l and not l.startswith("File Creation Time")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
    # On écarte warrants, droits, unités et actions de préférence
    bad = df["Security Name"].str.contains(r"\b(?:Warrant|Right|Unit|Preferred|Depositary Share)s?\b",
                                           case=False, regex=True, na=False)
    df = df[~bad]
    df = df[df["Symbol"].str.fullmatch(r"[A-Z]{1,5}", na=False)]
    return df[["Symbol", "Security Name"]].rename(columns={"Symbol": "ticker", "Security Name": "name"})


def get_universe() -> pd.DataFrame:
    req = urllib.request.Request(NASDAQ_LIST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode("utf-8", errors="replace")
    return parse_nasdaq_list(text)


# --------------------------------------------------------------------------------------
# 2. Données Yahoo Finance
# --------------------------------------------------------------------------------------
def download(tickers, period="2y", chunk=100, pause=1.5):
    import yfinance as yf
    out = {}
    n = len(tickers)
    for i in range(0, n, chunk):
        part = tickers[i:i + chunk]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(part, period=period, interval="1d", group_by="ticker",
                                 auto_adjust=False, threads=True, progress=False)
                break
            except Exception as e:  # réseau, limitation Yahoo...
                print(f"  ! erreur lot {i // chunk + 1} (tentative {attempt + 1}) : {e}")
                time.sleep(10 * (attempt + 1))
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                lvl0 = set(df.columns.get_level_values(0))
                for t in part:
                    if t in lvl0:
                        sub = df[t].dropna(subset=["Close"])
                        if len(sub):
                            out[t] = sub
            else:
                out[part[0]] = df.dropna(subset=["Close"])
        print(f"  téléchargé {min(i + chunk, n)}/{n} titres ({len(out)} avec données)", flush=True)
        time.sleep(pause)
    return out


# --------------------------------------------------------------------------------------
# 3. Calcul des conditions
# --------------------------------------------------------------------------------------
def compute(df: pd.DataFrame, p=P) -> pd.DataFrame:
    o, h, l, c, v = (df[k].astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    tp = (h + l + c) / 3
    mid = tp.rolling(p["len_bb"]).mean()
    sd = tp.rolling(p["len_bb"]).std(ddof=0)
    up, dn = mid + 2 * sd, mid - 2 * sd
    bbw = (up - dn) / mid
    hB = bbw.rolling(p["len_range"]).max()
    lB = bbw.rolling(p["len_range"]).min()
    sqz = (bbw < lB + p["sqz_pct"] * (hB - lB)).fillna(False)

    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    av50 = v.rolling(50).mean()

    liq = (c > p["min_price"]) & (av50 * c > p["min_dollar_vol"])
    r3 = (c / c.shift(63) - 1) * 100
    r6 = (c / c.shift(126) - 1) * 100
    r10 = (c / c.shift(200) - 1) * 100
    mom = (r3 > 0) & (r6 > 0) & ((r3 > 5) | (r6 > 5) | (r10 > 5))
    trend = (c > e50) & (e20 > e50)

    gain = (c / c.shift(1) - 1) * 100
    hhN = h.rolling(p["hh_len"]).max()
    bo = (c >= up) & (c > hhN.shift(1)) & (gain > p["min_day_gain"])
    vol_ok = v > p["vol_mult"] * av50.shift(1)

    push = (liq & trend & mom & sqz.shift(1, fill_value=False) & bo & vol_ok).fillna(False)
    sig_b = push & ~push.shift(1, fill_value=False)

    hh50 = h.rolling(50).max()
    setup_a = (liq & trend & (e50 > e50.shift(10)) & mom & sqz
               & (c >= (1 - p["near_high_pct"] / 100) * hh50) & (c < up)).fillna(False)

    return pd.DataFrame(dict(
        open=o, high=h, low=l, close=c, volume=v, bb_up=up, bb_mid=mid, bb_dn=dn,
        ema20=e20, ema50=e50, gain=gain, vol_x=v / av50.shift(1), r3=r3, r6=r6, r10=r10,
        dist_hh50=(hh50 - c) / hh50 * 100, bbw=bbw, sig_b=sig_b, setup_a=setup_a,
    ))


# --------------------------------------------------------------------------------------
# 4. Page HTML
# --------------------------------------------------------------------------------------
def chart_payload(ind: pd.DataFrame, n_bars: int):
    d = ind.tail(n_bars)
    t = [x.strftime("%Y-%m-%d") for x in d.index]
    r2 = lambda s: [None if pd.isna(x) else round(float(x), 4) for x in s]
    return dict(
        t=t, o=r2(d["open"]), h=r2(d["high"]), l=r2(d["low"]), c=r2(d["close"]),
        v=[0 if pd.isna(x) else int(x) for x in d["volume"]],
        up=r2(d["bb_up"]), mid=r2(d["bb_mid"]), dn=r2(d["bb_dn"]),
        e20=r2(d["ema20"]), e50=r2(d["ema50"]),
        sig=[x for x, s in zip(t, d["sig_b"]) if s],
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Screener Nasdaq – __DATE__</title>
<script src="__LWC__"></script>
<style>
body{margin:0;background:#0f1218;color:#e6e8ec;font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:21px;margin:0 0 4px}h2{font-size:17px;margin:26px 0 8px}
.mut{color:#9aa3b2}a{color:#5aa0ff}
#tbl{border-collapse:collapse;width:100%;font-size:13px}#tbl th,#tbl td{padding:5px 8px;border-bottom:1px solid #2a303b;text-align:right;white-space:nowrap}
#tbl th{color:#9aa3b2;font-weight:500;cursor:pointer;position:sticky;top:0;background:#0f1218}#tbl td:first-child,#tbl th:first-child,#tbl td:nth-child(2),#tbl th:nth-child(2){text-align:left}
#tbl td:nth-child(2){max-width:260px;overflow:hidden;text-overflow:ellipsis}
.pos{color:#26a69a}.neg{color:#ef5350}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px}
.card{background:#171b23;border:1px solid #2a303b;border-radius:8px;padding:8px}
.card h3{margin:0 0 2px;font-size:14px}.card .chart{height:300px}
.tag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;margin-left:6px}
.tB{background:#26a69a33;color:#26a69a}.tA{background:#f0b42933;color:#f0b429}
.tabs button{background:#171b23;color:#e6e8ec;border:1px solid #2a303b;padding:6px 12px;border-radius:6px;cursor:pointer;margin-right:6px}
.tabs button.on{border-color:#f0b429;color:#f0b429}
</style></head><body><div class="wrap">
<h1>Screener Nasdaq – compression puis cassure</h1>
<div class="mut">Dernière séance : <b>__DATE__</b> · Univers analysé : __NUNIV__ titres (__NDATA__ avec données) · Généré le __GEN__ · Données Yahoo Finance (non officielles)</div>
<div class="mut">B = cassure du jour après compression (signal) · A = compression près du plus haut, pas encore cassé (surveillance)</div>
<h2>Résultats</h2>
<div class="tabs"><button data-f="all" class="on">Tous</button><button data-f="B">Signaux B</button><button data-f="A">Setups A</button></div>
<table id="tbl"></table>
<h2>Graphiques</h2>
<div id="grid" class="grid"></div>
<p class="mut" style="font-size:12px;margin-top:24px">Graphiques : <a href="https://www.tradingview.com/" target="_blank" rel="noopener">TradingView Lightweight Charts™</a> (licence Apache 2.0). Bougies quotidiennes, Bollinger 20/2 (prix typique), EMA 20 (jaune) et EMA 50 (violet), volume. Flèche = signal B. Outil d'aide à l'analyse, pas un conseil en investissement.</p>
</div>
<script>
const ROWS=__ROWS__;const CH=__CHARTS__;
let F='all',SK='rank',SD=1;
const cols=[['ticker','Ticker'],['name','Nom'],['type','Type'],['date','Date'],['close','Clôture'],['gain','Var. jour %'],['vol_x','Volume x moy.'],['r3','Perf 3 m %'],['r6','Perf 6 m %'],['dist','Dist. haut 50 j %']];
const f=(v,d=1)=>v==null?'–':(+v).toFixed(d);const pc=v=>v==null?'–':`<span class="${v>=0?'pos':'neg'}">${v>0?'+':''}${(+v).toFixed(1)}</span>`;
function table(){const r=ROWS.filter(x=>F==='all'||x.type===F).sort((a,b)=>{const x=a[SK],y=b[SK];return (x>y?1:x<y?-1:0)*SD;});
document.getElementById('tbl').innerHTML='<tr>'+cols.map(c=>`<th data-k="${c[0]}">${c[1]}</th>`).join('')+'</tr>'+r.map(x=>`<tr><td><a href="#c-${x.ticker}">${x.ticker}</a> · <a href="https://www.tradingview.com/chart/?symbol=NASDAQ:${x.ticker}" target="_blank" rel="noopener">TV</a></td><td title="${x.name}">${x.name}</td><td><span class="tag t${x.type}">${x.type}</span></td><td>${x.date}</td><td>${f(x.close,2)}</td><td>${pc(x.gain)}</td><td>${f(x.vol_x)}</td><td>${pc(x.r3)}</td><td>${pc(x.r6)}</td><td>${f(x.dist)}</td></tr>`).join('');
document.querySelectorAll('#tbl th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;SD=(SK===k)?-SD:1;SK=k;table();});}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('on'));b.classList.add('on');F=b.dataset.f;table();});
table();
const grid=document.getElementById('grid');
const seen=new Set();
ROWS.forEach(x=>{if(seen.has(x.ticker))return;seen.add(x.ticker);const d=document.createElement('div');d.className='card';d.id='c-'+x.ticker;
d.innerHTML=`<h3>${x.ticker} <span class="tag t${x.type}">${x.type}</span> <span class="mut" style="font-weight:400">${x.name}</span></h3><div class="chart"></div>`;grid.appendChild(d);});
function draw(card){const t=card.id.slice(2);const c=CH[t];if(!c||card.dataset.done)return;card.dataset.done=1;const el=card.querySelector('.chart');
const ch=LightweightCharts.createChart(el,{width:el.clientWidth,height:300,layout:{background:{color:'#171b23'},textColor:'#c8ccd6'},grid:{vertLines:{color:'#20252e'},horzLines:{color:'#20252e'}},rightPriceScale:{borderColor:'#2a303b'},timeScale:{borderColor:'#2a303b'}});
const cs=ch.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',borderVisible:false,wickUpColor:'#26a69a',wickDownColor:'#ef5350'});
cs.setData(c.t.map((d,i)=>({time:d,open:c.o[i],high:c.h[i],low:c.l[i],close:c.c[i]})));
const line=(arr,color,w=1)=>{const s=ch.addLineSeries({color,lineWidth:w,priceLineVisible:false,lastValueVisible:false});s.setData(c.t.map((d,i)=>arr[i]==null?{time:d}:{time:d,value:arr[i]}));};
line(c.up,'#8a93a6');line(c.dn,'#8a93a6');line(c.mid,'#5c6578');line(c.e20,'#f0b429',2);line(c.e50,'#b388ff',2);
const vs=ch.addHistogramSeries({priceScaleId:'',priceFormat:{type:'volume'},priceLineVisible:false,lastValueVisible:false});ch.priceScale('').applyOptions({scaleMargins:{top:0.82,bottom:0}});
vs.setData(c.t.map((d,i)=>({time:d,value:c.v[i],color:c.c[i]>=c.o[i]?'#26a69a55':'#ef535055'})));
cs.setMarkers(c.sig.map(d=>({time:d,position:'belowBar',color:'#26a69a',shape:'arrowUp',text:'B'})));ch.timeScale().fitContent();new ResizeObserver(()=>{ch.applyOptions({width:el.clientWidth});}).observe(el);}
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)draw(e.target);}),{rootMargin:'400px'});
document.querySelectorAll('.card').forEach(c=>io.observe(c));
</script></body></html>"""


def build_html(rows, charts, last_date, n_univ, n_data, path: Path):
    html = (HTML_TEMPLATE
            .replace("__LWC__", LWC_URL)
            .replace("__DATE__", last_date)
            .replace("__NUNIV__", str(n_univ))
            .replace("__NDATA__", str(n_data))
            .replace("__GEN__", dt.datetime.now().strftime("%d/%m/%Y %H:%M"))
            .replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
            .replace("__CHARTS__", json.dumps(charts, separators=(",", ":"))))
    path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------------------
# 5. Programme principal
# --------------------------------------------------------------------------------------
def run(data: dict, names: dict, days: int, n_univ: int, out_dir: Path, open_browser=True):
    if not data:
        sys.exit("Aucune donnée téléchargée.")
    # Dernière séance commune (évite les titres suspendus / données en retard)
    last_dates = pd.Series({t: df.index.max() for t, df in data.items()})
    market_last = last_dates.mode().iloc[0]
    recent_days = sorted({d for df in data.values() for d in df.index if d <= market_last})[-days:]

    rows, charts = [], {}
    for t, df in data.items():
        if len(df) < 260 or df.index.max() != market_last:
            continue
        try:
            ind = compute(df)
        except Exception as e:
            print(f"  ! calcul impossible pour {t} : {e}")
            continue
        last = ind.iloc[-1]
        hit_b = ind.loc[ind.index.isin(recent_days), "sig_b"]
        hit_b = hit_b[hit_b]
        typ, ref = None, None
        if len(hit_b):
            typ, ref = "B", ind.loc[hit_b.index[-1]]
        elif bool(last["setup_a"]):
            typ, ref = "A", last
        if typ is None:
            continue
        g = lambda k: None if pd.isna(ref[k]) else round(float(ref[k]), 2)
        rows.append(dict(ticker=t, name=names.get(t, ""), type=typ,
                         date=ref.name.strftime("%Y-%m-%d"), close=g("close"), gain=g("gain"),
                         vol_x=g("vol_x"), r3=g("r3"), r6=g("r6"), dist=g("dist_hh50")))
        charts[t] = chart_payload(ind, P["chart_bars"])

    # Tri : signaux B d'abord (volume décroissant), puis setups A (plus proches du plus haut)
    b = sorted([r for r in rows if r["type"] == "B"], key=lambda r: -(r["vol_x"] or 0))
    a = sorted([r for r in rows if r["type"] == "A"], key=lambda r: (r["dist"] or 99))[:P["max_setups"]]
    rows = b + a
    for i, r in enumerate(rows):
        r["rank"] = i
    charts = {r["ticker"]: charts[r["ticker"]] for r in rows}

    stamp = market_last.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"screener_{stamp}.html"
    csv_path = out_dir / f"screener_{stamp}.csv"
    build_html(rows, charts, stamp, n_univ, len(data), html_path)
    pd.DataFrame(rows).drop(columns=["rank"], errors="ignore").to_csv(csv_path, index=False, sep=";", decimal=",")
    print(f"\nSéance analysée : {stamp} — {len(b)} signal(aux) B, {len(a)} setup(s) A")
    print(f"Page : {html_path}\nCSV  : {csv_path}")
    if open_browser:
        webbrowser.open(html_path.resolve().as_uri())
    return rows, html_path, csv_path


def main():
    ap = argparse.ArgumentParser(description="Screener Nasdaq – compression puis cassure")
    ap.add_argument("--tickers", help="Liste de tickers séparés par des virgules (sinon : tout le Nasdaq)")
    ap.add_argument("--limit", type=int, help="Ne prendre que les N premiers titres (test)")
    ap.add_argument("--days", type=int, default=1, help="Signaux B des N dernières séances (défaut 1)")
    ap.add_argument("--out", default="resultats_screener", help="Dossier de sortie")
    ap.add_argument("--no-open", action="store_true", help="Ne pas ouvrir la page automatiquement")
    a = ap.parse_args()

    if a.tickers:
        uni = pd.DataFrame({"ticker": [x.strip().upper() for x in a.tickers.split(",") if x.strip()], "name": ""})
    else:
        print("Récupération de la liste Nasdaq…")
        uni = get_universe()
    if a.limit:
        uni = uni.head(a.limit)
    print(f"{len(uni)} titres à analyser. Téléchargement Yahoo Finance (quelques minutes pour tout le Nasdaq)…")
    data = download(list(uni["ticker"]))
    run(data, dict(zip(uni["ticker"], uni["name"])), a.days, len(uni), Path(a.out), not a.no_open)


if __name__ == "__main__":
    main()
