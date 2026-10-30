#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, time, json, os, threading
import numpy as np
import pandas as pd
import urllib.request

import backtest_pro_v2 as bt
bt.FEE_RT = 0.0

START_MS = 1672531200000
OUTROOT = "/home/user/results_entry"
TF_SEC = {"3m": 180, "5m": 300, "15m": 900}
TFS = ["3m", "5m", "15m"]
OKX = "https://www.okx.com/api/v5/market/history-candles"
SESSIONS = ("LONDRES", "NEW YORK")

def okx_page(inst, bar, after_ms, tries=12):
    last = None
    for t in range(tries):
        try:
            url = f"{OKX}?instId={inst}&bar={bar}&after={after_ms}&limit=100"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                j = json.loads(r.read().decode())
            if j.get("code") == "0":
                return j["data"]
            last = j.get("code")
            time.sleep(0.5 * (t + 1))
        except Exception as e:
            last = str(e)
            time.sleep(0.5 * (t + 1))
    raise RuntimeError(f"OKX {inst} {bar}: {last}")

def _chunk(inst, bar, lo, hi, out, compact):
    after = hi
    while after > lo:
        data = okx_page(inst, bar, after)
        if not data:
            break
        out.extend(data)
        oldest = int(data[-1][0])
        if oldest <= lo or len(data) < 100:
            break
        after = oldest
        time.sleep(0.01)
    if out:
        d = pd.DataFrame(out)
        del out[:]
        compact.append(d)

def download_tf(inst, bar, workers=6):
    end = int(time.time() * 1000)
    bounds = np.linspace(end, START_MS, workers + 1).astype(np.int64)
    lists = [[] for _ in range(workers)]
    compacts = [[] for _ in range(workers)]
    ths = []
    for w in range(workers):
        lo, hi = int(bounds[w + 1]), int(bounds[w])
        if w > 0:
            lo += 1
        th = threading.Thread(target=_chunk, args=(inst, bar, lo, hi, lists[w], compacts[w]))
        ths.append(th)
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    parts = [c[0] for c in compacts if len(c) > 0]
    full = pd.concat(parts, ignore_index=True)
    df = full.iloc[:, :8].copy()
    df.columns = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volQ"]
    if full.shape[1] >= 9:
        keep = full.iloc[:, 8].astype(str) != "0"
        df = df[keep.to_numpy()].reset_index(drop=True)
    df["ts"] = df["ts"].astype(np.int64)
    for c in ("open", "high", "low", "close", "vol"):
        df[c] = df[c].astype(np.float64)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close", "vol"]]

def method_B(sym, tf, df, A, w1_h=2.0, w2_h=2.0, expire_tp1=True, funnel=None):
    ts = df["ts"].to_numpy(np.int64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    n = len(ts)
    w1 = int(w1_h * 3600_000 / (TF_SEC[tf] * 1000))
    w2 = int(w2_h * 3600_000 / (TF_SEC[tf] * 1000))
    out = []
    n_swept = 0
    sweep_bars = []
    return_bars = []
    for t in A:
        i = int(np.searchsorted(ts, t["ts"]))
        if i >= n or ts[i] != t["ts"]:
            continue
        d = t["dir"]
        E = t["entry"]
        sl0 = t["sl"]
        j = -1
        end1 = min(n, i + 1 + w1)
        for m in range(i + 1, end1):
            if d == "LONG":
                if l[m] <= sl0:
                    j = m
                    break
                if expire_tp1 and h[m] >= t["tp1"]:
                    break
            else:
                if h[m] >= sl0:
                    j = m
                    break
                if expire_tp1 and l[m] <= t["tp1"]:
                    break
        if j < 0:
            continue
        n_swept += 1
        sweep_bars.append(j - i)
        sln = l[j] if d == "LONG" else h[j]
        k = -1
        end2 = min(n, j + 1 + w2)
        for m in range(j, end2):
            if d == "LONG":
                if l[m] < sln:
                    sln = l[m]
                if h[m] >= E:
                    k = m
                    break
            else:
                if h[m] > sln:
                    sln = h[m]
                if l[m] <= E:
                    k = m
                    break
        if k < 0:
            continue
        return_bars.append(k - j)
        risk = (E - sln) if d == "LONG" else (sln - E)
        if risk <= 0:
            continue
        if d == "LONG":
            tp3 = E + 3 * risk
            tp6 = E + 6 * risk
            s_i = bt.first_where(l[k:] <= sln)
            t3 = bt.first_where(h[k:] >= tp3)
            t6 = bt.first_where(h[k:] >= tp6)
        else:
            tp3 = E - 3 * risk
            tp6 = E - 6 * risk
            s_i = bt.first_where(h[k:] >= sln)
            t3 = bt.first_where(l[k:] <= tp3)
            t6 = bt.first_where(l[k:] <= tp6)
        R3 = -1.0 if (s_i != -1 and (t3 == -1 or s_i <= t3)) else (3.0 if t3 != -1 else np.nan)
        R6 = -1.0 if (s_i != -1 and (t6 == -1 or s_i <= t6)) else (6.0 if t6 != -1 else np.nan)
        if np.isnan(R3) and np.isnan(R6):
            continue
        ek = ts[k]
        hhmm = (ek // 3600000) % 24 * 100 + (ek // 60000) % 60
        entry_ses = int((700 <= hhmm < 1000) or (1300 <= hhmm < 1600))
        out.append(dict(sym=sym, tf=tf, ts=int(t["ts"]), ts_entry=int(ek), dir=d,
                        session=t["session"], entry_ses=entry_ses,
                        entry=E, sl0=round(sl0, 6), sl_new=round(sln, 6),
                        risk=round(risk, 6), risk_pct=round(risk / E * 100, 4),
                        R3=R3, R6=R6, sweep_bars=int(j - i), return_bars=int(k - j)))
    if funnel is not None:
        funnel.update(n_sig=len(A), n_swept=n_swept, n_trades=len(out),
                      med_sweep_bars=float(np.median(sweep_bars)) if sweep_bars else np.nan,
                      med_return_bars=float(np.median(return_bars)) if return_bars else np.nan,
                      avg_risk_pct=float(np.mean([x["risk_pct"] for x in out])) if out else np.nan)
    return out

def apply_rule(tdf, rcol, keycol="ts"):
    t = tdf.sort_values(keycol).reset_index(drop=True)
    if len(t) == 0:
        return t
    key = (t["ts"] // 86_400_000).astype(str) + "_" + t["session"]
    keep = np.ones(len(t), dtype=bool)
    prev = None
    consec = 0
    stopped = False
    Rs = t[rcol].to_numpy(float)
    for idx in range(len(t)):
        k = key.iloc[idx]
        if k != prev:
            prev = k
            consec = 0
            stopped = False
        if stopped:
            keep[idx] = False
            continue
        if Rs[idx] < 0:
            consec += 1
        else:
            consec = 0
        if consec >= 2:
            stopped = True
    return t[keep].reset_index(drop=True)

def stats(tdf, rcol):
    if tdf is None or len(tdf) == 0:
        return dict(n=0)
    t = tdf[tdf[rcol].notna()]
    R = t[rcol].to_numpy(float)
    ts = t["ts"].to_numpy(np.int64)
    n = len(R)
    wins = R[R > 0].sum()
    losses = -R[R < 0].sum()
    pf = wins / losses if losses > 1e-12 else (99.0 if wins > 0 else np.nan)
    wr = 100.0 * (R > 0).mean()
    cs = np.cumsum(R)
    pk = np.maximum.accumulate(cs)
    maxDD = float((cs - pk).min())
    streak = cur = 0
    for r in R:
        cur = cur + 1 if r < 0 else 0
        streak = max(streak, cur)
    dfd = pd.DataFrame({"d": ts // 86_400_000, "R": R})
    pire_jour = float(dfd.groupby("d")["R"].sum().min())
    mon = pd.Series(pd.to_datetime(ts, unit="ms")).dt.to_period("M").astype(str).to_numpy()
    dfm = pd.DataFrame({"m": mon, "R": R})
    mm = dfm.groupby("m")["R"].sum()
    return dict(n=n, wr=round(float(wr), 1), pf=round(float(pf), 2), totalR=round(float(R.sum()), 1),
                avgR=round(float(R.mean()), 3), maxDD=round(maxDD, 1), streak=streak,
                pire_jour=round(pire_jour, 1), worst_month=round(float(mm.min()), 1),
                best_month=round(float(mm.max()), 1), npos_months=int((mm > 0).sum()),
                nmonths=int(len(mm)), avg_month=round(float(mm.mean()), 2))

def collect(tdf, rcol, rname, method, scope, tf, rows):
    t = tdf.copy()
    if scope in ("LNY", "RULE"):
        t = t[t["session"].isin(SESSIONS)]
    if scope == "RULE":
        t = t[t["entry_ses"] == 1]
        t = apply_rule(t, rcol, keycol=("ts_entry" if str(method).startswith("B") else "ts"))
    st = stats(t, rcol)
    st.update(tf=tf, method=method, R=rname, scope=scope)
    rows.append(st)

def main():
    sym = sys.argv[1]
    outd = f"{OUTROOT}/{sym}"
    os.makedirs(outd, exist_ok=True)
    summary_rows = []
    funnel_rows = []
    for tf in TFS:
        t0 = time.time()
        for attempt in (1, 2):
            try:
                df = download_tf(sym, tf)
                break
            except Exception as e:
                print(f"  !! {sym} {tf} tent {attempt}: {e} -> retry 60s", flush=True)
                if attempt == 2:
                    raise
                time.sleep(60)
        print(f"  {sym} {tf}: {len(df)} bars  {pd.to_datetime(df.ts.iloc[0], unit='ms')} -> "
              f"{pd.to_datetime(df.ts.iloc[-1], unit='ms')}  ({time.time() - t0:.0f}s)", flush=True)
        A = bt.run_symbol_tf(sym, tf, df, None, prot_mode="reversal", entry_mode="close", allUT=False)
        A_df = pd.DataFrame(A)
        A_df["entry_ses"] = 1
        A_df.to_csv(f"{outd}/{tf}_A_trades.csv", index=False)

        AUT = bt.run_symbol_tf(sym, tf, df, None, prot_mode="reversal", entry_mode="close", allUT=True)
        AUT_df = pd.DataFrame(AUT)
        AUT_df["entry_ses"] = 1
        AUT_df.to_csv(f"{outd}/{tf}_A_UT_trades.csv", index=False)

        for rcol, rname in (("R_tp1", "3R"), ("R_tp2", "6R")):
            for scope in ("LNY", "RULE"):
                collect(A_df, rcol, rname, "A", scope, tf, summary_rows)
                collect(AUT_df, rcol, rname, "A_UT", scope, tf, summary_rows)
        print(f"  {sym} {tf}: A={len(A_df)} trades | A_UT={len(AUT_df)} trades ({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(summary_rows).to_csv(f"{outd}/summary.csv", index=False)
    pd.DataFrame(funnel_rows).to_csv(f"{outd}/funnel.csv", index=False)

    chk = [r for r in summary_rows if r["tf"] == "3m" and r["method"] == "A"
           and r["R"] == "6R" and r["scope"] == "RULE"]
    if chk:
        r = chk[0]
        print(f"  SANITY A 3m RULE 6R: n={r['n']} WR={r['wr']}% PF={r['pf']} totalR={r['totalR']} "
              f"maxDD={r['maxDD']}R streak={r['streak']} pireJour={r['pire_jour']}R "
              f"mois={r['npos_months']}/{r['nmonths']} + (moy {r['avg_month']}R/mois)", flush=True)
    print(f"TERMINÉ {sym}", flush=True)

if __name__ == "__main__":
    main()
