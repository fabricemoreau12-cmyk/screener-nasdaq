#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest de la version B du screener Nasdaq (compression puis cassure)
======================================================================

À lancer une fois (GitHub Actions ou PC). Produit :
  - backtest/trades.csv   : une ligne par signal B, avec
        * les critères mesurés le jour du signal (pour tester un classement),
        * le résultat de plusieurs règles de sortie,
        * l'excès de rendement par rapport à la moyenne équipondérée des actions liquides
          du Nasdaq sur la même période (retire l'effet "marché").
  - backtest/resume.txt   : statistiques par règle de sortie.
  - backtest_resultats.zip : les deux fichiers compressés.

Hypothèses : entrée à la clôture du jour du signal ; stops intrajournaliers exécutés au prix du
stop, ou à l'ouverture si le cours ouvre en dessous ; frais + glissement de 0,10 % aller-retour.
Univers = actions actuellement cotées au Nasdaq (biais du survivant : les sociétés disparues
depuis ne sont pas dans les données).
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import nasdaq_screener as ns

COST = 0.001          # frais + glissement aller-retour (0,10 %)
PERIOD = "10y"        # historique Yahoo
MIN_BARS = 300


# ------------------------------------------------------------------ indicateurs supplémentaires
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    ind = ns.compute(df)
    h, l, c = ind["high"], ind["low"], ind["close"]
    c1 = c.shift(1)
    tr = pd.concat([h - l, (h - c1).abs(), (l - c1).abs()], axis=1).max(axis=1)
    ind["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    ind["ema10"] = c.ewm(span=10, adjust=False).mean()
    p = ns.P
    hB = ind["bbw"].rolling(p["len_range"]).max()
    lB = ind["bbw"].rolling(p["len_range"]).min()
    ind["bbw_pos"] = (ind["bbw"] - lB) / (hB - lB)
    sqz = ind["bbw"] < lB + p["sqz_pct"] * (hB - lB)
    grp = (~sqz).cumsum()
    ind["sqz_len"] = sqz.astype(int).groupby(grp).cumsum()
    ind["r1"] = (c / c.shift(21) - 1) * 100
    ind["r12"] = (c / c.shift(252) - 1) * 100
    ind["dist_hh252"] = (h.rolling(252).max() - c) / h.rolling(252).max() * 100
    rng = (h - l).replace(0, np.nan)
    ind["close_pos"] = (c - l) / rng
    ind["ext_ema20"] = (c / ind["ema20"] - 1) * 100
    ind["dollar_vol"] = ind["volume"].rolling(50).mean() * c
    ind["liq"] = (c > ns.P["min_price"]) & (ind["dollar_vol"] > ns.P["min_dollar_vol"])
    return ind


# ------------------------------------------------------------------ simulation des sorties
def simulate(o, h, l, c, e10, e20, e50, atr, i, rule, max_hold):
    """Retourne (indice de sortie, prix de sortie, risque initial en % ou NaN). Entrée à c[i]."""
    n = len(c)
    entry = c[i]
    if rule == "fixe20":
        j = min(i + 20, n - 1)
        return j, c[j], np.nan, j - i < 20
    stop = l[i]
    if rule == "atr_chandelier":
        stop = entry - 2 * atr[i]
    risk = (entry - stop) / entry if entry > stop else np.nan
    target = entry + 2 * (entry - stop) if rule == "cible2R" else None
    highest = entry
    for j in range(i + 1, min(i + 1 + max_hold, n)):
        # 1) stop intrajournalier
        if o[j] <= stop:
            return j, o[j], risk, False
        if l[j] <= stop:
            return j, stop, risk, False
        # 2) objectif
        if target is not None and h[j] >= target:
            return j, max(o[j], target), risk, False
        # 3) règles de clôture
        if rule == "stop_ema10" and c[j] < e10[j]:
            return j, c[j], risk, False
        if rule == "stop_ema20" and c[j] < e20[j]:
            return j, c[j], risk, False
        if rule == "stop_ema50" and c[j] < e50[j]:
            return j, c[j], risk, False
        if rule == "cible2R" and j - i >= 20:
            return j, c[j], risk, False
        if rule == "atr_chandelier":
            highest = max(highest, c[j])
            stop = max(stop, highest - 3 * atr[j])
    j = min(i + max_hold, n - 1)
    still_open = (i + max_hold) >= n
    return j, c[j], risk, still_open


RULES = {  # nom : durée maximale (séances)
    "fixe20": 20,
    "stop_ema10": 120,
    "stop_ema20": 120,
    "stop_ema50": 250,
    "atr_chandelier": 250,
    "cible2R": 20,
}


# ------------------------------------------------------------------ programme principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", help="liste de tickers (test)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="backtest")
    a = ap.parse_args()

    if a.tickers:
        uni = pd.DataFrame({"ticker": a.tickers.split(","), "name": ""})
    else:
        uni = ns.get_universe()
    if a.limit:
        uni = uni.head(a.limit)
    tickers = list(uni["ticker"])
    print(f"{len(tickers)} titres. Téléchargement {PERIOD}…", flush=True)
    data = ns.download(tickers + ["QQQ"], period=PERIOD)
    run(data, a.out)


def run(data: dict, out: str):
    t0 = time.time()
    qqq = data.pop("QQQ", None)
    inds = {}
    for t, df in data.items():
        if len(df) < MIN_BARS:
            continue
        try:
            inds[t] = enrich(df)
        except Exception as e:
            print(f"  ! {t}: {e}")
    print(f"{len(inds)} titres exploitables.", flush=True)

    # Indice équipondéré des actions liquides (rendement du jour des titres liquides la veille)
    rets = pd.DataFrame({t: d["close"].pct_change() for t, d in inds.items()})
    liqm = pd.DataFrame({t: d["liq"].shift(1, fill_value=False) for t, d in inds.items()})
    rets = rets.where(liqm.reindex_like(rets).fillna(False).astype(bool))
    rets = rets.clip(-0.5, 1.0)
    ew = (1 + rets.mean(axis=1).fillna(0)).cumprod()

    # Régime de marché (QQQ)
    if qqq is not None and len(qqq):
        qc = qqq["Close"].astype(float)
        q_reg = (qc > qc.rolling(200).mean())
        q_r3 = (qc / qc.shift(63) - 1) * 100
    else:
        q_reg = pd.Series(dtype=bool)
        q_r3 = pd.Series(dtype=float)

    # Nombre de signaux par jour (signal d'euphorie collective ?)
    sig_count = pd.DataFrame({t: d["sig_b"] for t, d in inds.items()}).sum(axis=1)

    rows = []
    for t, d in inds.items():
        idx = np.where(d["sig_b"].values)[0]
        if not len(idx):
            continue
        o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
        e10, e20, e50, atr = (d[k].values for k in ("ema10", "ema20", "ema50", "atr14"))
        dates = d.index
        for i in idx:
            if i < 260:
                continue
            dt_ = dates[i]
            r = dict(
                ticker=t, date=dt_.strftime("%Y-%m-%d"), close=round(c[i], 4),
                gain=d["gain"].iat[i], vol_x=d["vol_x"].iat[i],
                r1=d["r1"].iat[i], r3=d["r3"].iat[i], r6=d["r6"].iat[i], r10=d["r10"].iat[i],
                r12=d["r12"].iat[i], dist_hh50=d["dist_hh50"].iat[i], dist_hh252=d["dist_hh252"].iat[i],
                bbw_prev=d["bbw"].iat[i - 1], bbw_pos_prev=d["bbw_pos"].iat[i - 1],
                sqz_len_prev=int(d["sqz_len"].iat[i - 1]), atr_pct=atr[i] / c[i] * 100,
                close_pos=d["close_pos"].iat[i], ext_ema20=d["ext_ema20"].iat[i],
                log_dollar_vol=float(np.log10(d["dollar_vol"].iat[i])) if d["dollar_vol"].iat[i] > 0 else np.nan,
                stop_bar_pct=(c[i] - l[i]) / c[i] * 100,
                qqq_above_200=bool(q_reg.get(dt_, np.nan)) if dt_ in q_reg.index else None,
                qqq_r3=q_r3.get(dt_, np.nan),
                n_signals_day=int(sig_count.get(dt_, 0)),
            )
            r["rs3"] = r["r3"] - r["qqq_r3"] if pd.notna(r["qqq_r3"]) else np.nan
            ew_i = ew.get(dt_, np.nan)
            # Horizons fixes (clôture à clôture, sans frais)
            for hz in (5, 10, 20, 60):
                j = i + hz
                if j < len(c):
                    rr = c[j] / c[i] - 1
                    r[f"ret{hz}"] = rr * 100
                    ew_j = ew.get(dates[j], np.nan)
                    r[f"exc{hz}"] = ((1 + rr) / (ew_j / ew_i) - 1) * 100 if ew_i and ew_j else np.nan
            # Règles de sortie
            for rule, mh in RULES.items():
                j, px, risk, opn = simulate(o, h, l, c, e10, e20, e50, atr, i, rule, mh)
                rr = px / c[i] - 1 - COST
                r[f"{rule}_ret"] = rr * 100
                r[f"{rule}_days"] = j - i
                r[f"{rule}_R"] = rr / risk if risk and risk > 0 else np.nan
                ew_j = ew.get(dates[j], np.nan)
                r[f"{rule}_exc"] = ((1 + rr) / (ew_j / ew_i) - 1) * 100 if ew_i and ew_j else np.nan
                r[f"{rule}_open"] = bool(opn)
            rows.append(r)

    tr = pd.DataFrame(rows).sort_values(["date", "ticker"])
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    tr.to_csv(outp / "trades.csv", index=False)
    ew.rename("ew_index").to_csv(outp / "indice_equipondere.csv")

    lines = [f"Backtest version B — {len(tr)} signaux, {tr['ticker'].nunique() if len(tr) else 0} titres, "
             f"{tr['date'].min() if len(tr) else '-'} → {tr['date'].max() if len(tr) else '-'}",
             f"Frais+glissement {COST*100:.2f} % A/R inclus dans les règles de sortie.", ""]
    for rule in RULES:
        if not len(tr):
            break
        s = tr[~tr[f"{rule}_open"]]
        x = s[f"{rule}_ret"]
        win, loss = x[x > 0], x[x <= 0]
        pf = win.sum() / -loss.sum() if loss.sum() < 0 else np.nan
        lines.append(f"{rule:15s} n={len(s):5d}  moy={x.mean():6.2f}%  méd={x.median():6.2f}%  gagnants={len(win)/len(x)*100:5.1f}%  "
                     f"gain moy={win.mean():6.2f}%  perte moy={loss.mean():6.2f}%  PF={pf:4.2f}  "
                     f"durée moy={s[f'{rule}_days'].mean():5.1f} j  excès moy={s[f'{rule}_exc'].mean():6.2f}%")
    (outp / "resume.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    with zipfile.ZipFile("backtest_resultats.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("trades.csv", "resume.txt", "indice_equipondere.csv"):
            z.write(outp / f, arcname=f)
    print(f"Terminé en {time.time() - t0:.0f} s (hors téléchargement). Fichier : backtest_resultats.zip")


if __name__ == "__main__":
    main()
