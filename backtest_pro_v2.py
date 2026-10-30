#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os, sys, json, time

DATA = "/home/user/data"
FOUR_H_MS = 4 * 3600 * 1000
M15_MS = 15 * 60 * 1000

TF_MIN = {"3m": 3, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "2H": 120, "4H": 240}
TFS = ["3m", "5m", "15m", "30m", "1H", "2H", "4H"]
SYMS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
        "ADA-USDT", "DOGE-USDT", "AVAX-USDT", "LINK-USDT"]

FVG_TF_MS = FOUR_H_MS
HP_FVG = True
MIN_GAP = 0.3
FVG_ONLY = True
PROT_ON = True
USE_BOS = True
USE_MTF = True
MTF_TF = 15
COOL_BARS = 3
RR = 3.0
ATR_MULT = 1.5
FEE_RT = 0.001

def ema_seeded(x, length):
    n = len(x); out = np.full(n, np.nan)
    if n >= length:
        a = 2.0 / (length + 1.0)
        e = np.mean(x[:length]); out[length - 1] = e
        for i in range(length, n):
            e += a * (x[i] - e); out[i] = e
    return out

def rma_seeded(x, length):
    n = len(x); out = np.full(n, np.nan)
    if n >= length:
        a = 1.0 / length
        r = np.mean(x[:length]); out[length - 1] = r
        for i in range(length, n):
            r += a * (x[i] - r); out[i] = r
    return out

def first_where(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else -1

def run_symbol_tf(sym, tf, df, df15=None, prot_mode="faithful", verbose=False, entry_mode="close",
                  allUT=False, rr=None, coolBars=None, minGap=None, pinOn=False, arOn=False,
                  pinRatio=2.0):
    rr_ = RR if rr is None else float(rr)
    cool_ = COOL_BARS if coolBars is None else int(coolBars)
    gap_ = MIN_GAP if minGap is None else float(minGap)

    ts = df["ts"].to_numpy(np.int64)
    o = df["open"].to_numpy(np.float64); h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64); c = df["close"].to_numpy(np.float64)
    v = df["vol"].to_numpy(np.float64) if "vol" in df.columns else np.ones(len(df))
    n = len(ts)
    tf_min = TF_MIN[tf]

    hour = (ts // 3600000) % 24
    minute = (ts // 60000) % 60
    hhmm = hour * 100 + minute
    inAsia = (hhmm >= 0) & (hhmm < 400)
    inLond = (hhmm >= 700) & (hhmm < 1000)
    inNY = (hhmm >= 1300) & (hhmm < 1600)
    inSes = inAsia | inLond | inNY
    sess_tag = np.where(inAsia, "ASIE", np.where(inLond, "LONDRES", np.where(inNY, "NEW YORK", "FERME")))

    w4 = ts // FOUR_H_MS
    dg = pd.DataFrame({"w4": w4, "h": h, "l": l, "c": c})
    htf_high = dg.groupby("w4")["h"].cummax().to_numpy()
    htf_low = dg.groupby("w4")["l"].cummin().to_numpy()

    win = dg.groupby("w4").agg(c_end=("c", "last"), h_end=("h", "last"),
                               l_end=("l", "last")).sort_index()
    W = win.index.to_numpy(); C_end = win["c_end"].to_numpy()
    H_end = win["h_end"].to_numpy(); L_end = win["l_end"].to_numpy()
    nw = len(win)
    k = np.searchsorted(W, w4)

    ema50_fin = ema_seeded(C_end, 50)
    TR4 = np.empty(nw)
    for j in range(nw):
        pc = C_end[j - 1] if j > 0 else C_end[j]
        TR4[j] = max(H_end[j] - L_end[j], abs(H_end[j] - pc), abs(L_end[j] - pc))
    atr14_fin = rma_seeded(TR4, 14)

    valid = k >= 1
    ema_prev = np.full(n, np.nan); ema_prev[valid] = ema50_fin[k[valid] - 1]
    ema50_4h = np.full(n, np.nan)
    ok = ~np.isnan(ema_prev)
    ema50_4h[ok] = ema_prev[ok] + (2.0 / 51.0) * (c[ok] - ema_prev[ok])
    htfBull = c > ema50_4h

    atr_prev = np.full(n, np.nan); atr_prev[valid] = atr14_fin[k[valid] - 1]
    prevC = np.full(n, np.nan); prevC[valid] = C_end[k[valid] - 1]
    tr_in = np.maximum.reduce([htf_high - htf_low, np.abs(htf_high - prevC), np.abs(htf_low - prevC)])
    atr4 = np.full(n, np.nan)
    ok = ~np.isnan(atr_prev)
    atr4[ok] = atr_prev[ok] + (1.0 / 14.0) * (tr_in[ok] - atr_prev[ok])

    ph = pd.Series(htf_high).rolling(5).max().to_numpy()
    pl = pd.Series(htf_low).rolling(5).min().to_numpy()
    ph_u = np.roll(ph, 1); ph_u[0] = np.nan
    pl_u = np.roll(pl, 1); pl_u[0] = np.nan
    c_1 = np.roll(c, 1); c_1[0] = np.nan
    c_2 = np.roll(c, 2); c_2[:2] = np.nan
    hh_2 = np.roll(htf_high, 2); hh_2[:2] = np.nan
    ll_2 = np.roll(htf_low, 2); ll_2[:2] = np.nan

    bos_bull = ~np.isnan(ph_u) & (c > ph_u) & (c_1 <= ph_u)
    bos_bear = ~np.isnan(pl_u) & (c < pl_u) & (c_1 >= pl_u)
    choch_bear = ~np.isnan(pl_u) & (c < pl_u) & (c_1 >= pl_u) & (c_2 > hh_2)
    choch_bull = ~np.isnan(ph_u) & (c > ph_u) & (c_1 <= ph_u) & (c_2 < ll_2)

    if prot_mode == "faithful":
        prot_long = bos_bull | ((c > pl_u) & ~choch_bear)
        prot_short = prot_long.copy()
    elif prot_mode == "corrected":
        prot_long = bos_bull | ((c > pl_u) & ~choch_bear)
        prot_short = bos_bear | ((c < ph_u) & ~choch_bull)
    else:
        prot_long = (c < ph_u) | choch_bull
        prot_short = (c > pl_u) | choch_bear

    okTrade_L = inSes & prot_long
    okTrade_S = inSes & prot_short

    if tf_min <= 15:
        w15 = ts // M15_MS
        dg15 = pd.DataFrame({"w15": w15, "h": h, "l": l, "c": c, "o": o})
        mtf_open = dg15.groupby("w15")["o"].transform("first").to_numpy()
        win15 = dg15.groupby("w15").agg(c_end=("c", "last")).sort_index()
        W15 = win15.index.to_numpy(); C15 = win15["c_end"].to_numpy()
        ema20_15 = ema_seeded(C15, 20)
        k15 = np.searchsorted(W15, w15)
        v15 = k15 >= 1
        ema20_prev = np.full(n, np.nan); ema20_prev[v15] = ema20_15[k15[v15] - 1]
        mtf_ema = np.full(n, np.nan)
        ok = ~np.isnan(ema20_prev)
        mtf_ema[ok] = ema20_prev[ok] + (2.0 / 21.0) * (c[ok] - ema20_prev[ok])
        mtf_close = c
    else:
        ts15 = df15["ts"].to_numpy(np.int64)
        c15 = df15["close"].to_numpy(np.float64); o15 = df15["open"].to_numpy(np.float64)
        ema20_15 = ema_seeded(c15, 20)
        target = ts + (tf_min - 15) * 60000
        idx = np.searchsorted(ts15, target)
        exact = np.zeros(n, dtype=bool)
        valid = idx < len(ts15)
        exact[valid] = ts15[idx[valid]] == target[valid]
        idx = np.where(exact, idx, np.maximum(idx - 1, 0))
        mtf_close = np.where(target < ts15[0], np.nan, c15[idx])
        mtf_open = np.where(target < ts15[0], np.nan, o15[idx])
        mtf_ema = np.where(target < ts15[0], np.nan, ema20_15[idx])
    mtf_bull = np.nan_to_num((mtf_close > mtf_open) & (mtf_close > mtf_ema), nan=False)
    mtf_bear = np.nan_to_num((mtf_close < mtf_open) & (mtf_close < mtf_ema), nan=False)

    tr_c = np.empty(n)
    for i in range(n):
        pc = c[i - 1] if i > 0 else c[i]
        tr_c[i] = max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc))
    atr_chart = rma_seeded(tr_c, 14)

    if allUT:
        gB = np.full(n, np.nan)
        gS = np.full(n, np.nan)
        if n >= 3:
            gB[2:] = l[2:] - h[:-2]
            gS[2:] = l[:-2] - h[2:]
        okB_ltf = (~np.isnan(gB)) & (gB >= atr_chart * gap_) & htfBull
        okS_ltf = (~np.isnan(gS)) & (gS >= atr_chart * gap_) & (~htfBull)
    else:
        okB_ltf = np.zeros(n, dtype=bool)
        okS_ltf = np.zeros(n, dtype=bool)

    body = np.maximum(np.abs(c - o), 1e-9)
    c1 = np.roll(c, 1); o1 = np.roll(o, 1); c1[0] = np.nan; o1[0] = np.nan
    body1 = np.abs(c1 - o1)
    engBull = (c > o) & (c1 < o1) & (c >= o1) & (o <= c1) & (body > body1)
    engBear = (c < o) & (c1 > o1) & (c <= o1) & (o >= c1) & (body > body1)
    engBull[0] = False; engBear[0] = False

    if pinOn:
        upper = np.maximum(o, c); lower = np.minimum(o, c)
        upperW = h - upper; lowerW = lower - l
        pinBull = (c > o) & (lowerW >= pinRatio * body) & (upperW <= body)
        pinBear = (c < o) & (upperW >= pinRatio * body) & (lowerW <= body)
        pinBull[0] = False; pinBear[0] = False
    else:
        pinBull = np.zeros(n, dtype=bool); pinBear = np.zeros(n, dtype=bool)

    if arOn:
        day = ts // 86400000
        asiH_v = np.where(inAsia, h, np.nan)
        asiL_v = np.where(inAsia, l, np.nan)
        asiH = pd.Series(asiH_v).groupby(day).transform("max").to_numpy()
        asiL = pd.Series(asiL_v).groupby(day).transform("min").to_numpy()
        arBull = (l < asiL) & (c > asiL)
        arBear = (h > asiH) & (c < asiH)
    else:
        arBull = np.zeros(n, dtype=bool); arBear = np.zeros(n, dtype=bool)

    is_new = np.zeros(n, dtype=bool); is_new[0] = True
    is_new[1:] = w4[1:] != w4[:-1]
    okB = np.zeros(n, dtype=bool); okS = np.zeros(n, dtype=bool)
    bullT_v = np.full(n, np.nan); bullB_v = np.full(n, np.nan)
    bearT_v = np.full(n, np.nan); bearB_v = np.full(n, np.nan)
    for i in np.where(is_new)[0]:
        ki = k[i]
        if ki >= 3:
            Lm1, Hm1 = L_end[ki - 1], H_end[ki - 1]
            Lm3, Hm3 = L_end[ki - 3], H_end[ki - 3]
            if not np.isnan(atr4[i]) and not np.isnan(ema50_4h[i]):
                thr = atr4[i] * gap_
                if (Lm1 - Hm3) >= thr and htfBull[i]:
                    okB[i] = True; bullT_v[i] = Lm1; bullB_v[i] = Hm3
                if (Lm3 - Hm1) >= thr and not htfBull[i]:
                    okS[i] = True; bearT_v[i] = Lm3; bearB_v[i] = Hm1

    sig = []
    bullT = np.nan; bullB = np.nan; bearT = np.nan; bearB = np.nan
    lastB = 0; lastS = 0
    for i in range(n):
        if okB[i]:
            bullT = bullT_v[i]; bullB = bullB_v[i]
        if okS[i]:
            bearT = bearT_v[i]; bearB = bearB_v[i]
        if allUT:
            if okB_ltf[i]:
                bullT = l[i]; bullB = h[i - 2]
            if okS_ltf[i]:
                bearT = l[i - 2]; bearB = h[i]
        if not np.isnan(bullT) and c[i] < bullB:
            bullT = np.nan
        if not np.isnan(bearT) and c[i] > bearT:
            bearT = np.nan
        inBull = (not np.isnan(bullT)) and l[i] <= bullT and c[i] >= bullB
        inBear = (not np.isnan(bearT)) and h[i] >= bearB and c[i] <= bearT
        okZoneB = (not FVG_ONLY) or inBull
        okZoneS = (not FVG_ONLY) or inBear
        canB = (i - lastB) > cool_
        canS = (i - lastS) > cool_
        sigB = (engBull[i] or pinBull[i] or arBull[i])
        sigS = (engBear[i] or pinBear[i] or arBear[i])
        if sigB and okTrade_L[i] and okZoneB and canB and mtf_bull[i]:
            lastB = i; sig.append((i, "LONG", sess_tag[i], ts[i]))
        if sigS and okTrade_S[i] and okZoneS and canS and mtf_bear[i]:
            lastS = i; sig.append((i, "SHORT", sess_tag[i], ts[i]))

    trades = []
    for (i, d, sess, tsi) in sig:
        if i + 1 >= n:
            continue
        if entry_mode == "next_open":
            entry = o[i + 1]
        else:
            entry = c[i]
        if d == "LONG":
            sl = l[i] - atr_chart[i] * ATR_MULT
            risk = entry - sl
            tp1 = entry + risk * rr_
            tp2 = entry + risk * rr_ * 2
        else:
            sl = h[i] + atr_chart[i] * ATR_MULT
            risk = sl - entry
            tp1 = entry - risk * rr_
            tp2 = entry - risk * rr_ * 2
        if risk <= 0:
            trades.append(dict(sym=sym, tf=tf, ts=int(tsi), dir=d, session=sess,
                               entry=round(float(entry), 6), sl=round(float(sl), 6),
                               tp1=round(float(tp1), 6), tp2=round(float(tp2), 6),
                               risk=round(abs(float(entry - (l[i] if d == "LONG" else h[i]))), 6),
                               fee=0.0, R_tp1=-1.0, flag_tp1="GAP_SL", R_tp2=-1.0,
                               flag_tp2="GAP_SL", R_part=-1.0, flag_part="GAP_SL",
                               tp1_hit=0, tp2_hit=0, sl_hit=1))
            continue
        lows = l[i + 1:]; highs = h[i + 1:]; closes = c[i + 1:]
        if d == "LONG":
            sl_idx = first_where(lows <= sl)
            tp1_idx = first_where(highs >= tp1)
            tp2_idx = first_where(highs >= tp2)
            be_idx = first_where(lows <= entry)
            r_end = (closes[-1] - entry) / risk
        else:
            sl_idx = first_where(highs >= sl)
            tp1_idx = first_where(lows <= tp1)
            tp2_idx = first_where(lows <= tp2)
            be_idx = first_where(highs >= entry)
            r_end = (entry - closes[-1]) / risk

        def r_scale(p):
            return (p - entry) / risk if d == "LONG" else (entry - p) / risk

        if sl_idx == -1 and tp1_idx == -1:
            R1, f1 = r_end, "OPEN"
        elif sl_idx != -1 and (tp1_idx == -1 or sl_idx <= tp1_idx):
            R1, f1 = -1.0, "SL"
        else:
            R1, f1 = rr_, "TP1"
        if sl_idx == -1 and tp2_idx == -1:
            R2, f2 = r_end, "OPEN"
        elif sl_idx != -1 and (tp2_idx == -1 or sl_idx <= tp2_idx):
            R2, f2 = -1.0, "SL"
        else:
            R2, f2 = rr_ * 2, "TP2"
        if tp1_idx == -1 and sl_idx == -1:
            Rp, fp = r_end, "OPEN"
        elif sl_idx != -1 and (tp1_idx == -1 or sl_idx <= tp1_idx):
            Rp, fp = -1.0, "SL"
        else:
            tp1_abs = i + 1 + tp1_idx
            be_after = first_where(lows[tp1_idx + 1:] <= entry) if d == "LONG" \
                else first_where(highs[tp1_idx + 1:] >= entry)
            be_abs = (tp1_abs + 1 + be_after) if be_after >= 0 else -1
            if tp2_idx > tp1_idx:
                tp2_abs = i + 1 + tp2_idx
            elif tp2_idx == tp1_idx:
                tp2_abs = tp1_abs
            else:
                tp2_abs = -1
            if tp2_abs != -1 and (be_abs == -1 or tp2_abs <= be_abs):
                Rp, fp = 1.5 + 0.5 * (rr_ * 2), "PART_TP2"
            elif be_abs != -1:
                Rp, fp = 1.5, "PART_BE"
            else:
                rem = r_scale(closes[-1])
                Rp, fp = 1.5 + 0.5 * rem, "PART_OPEN"

        fee = FEE_RT * entry / risk
        trades.append(dict(sym=sym, tf=tf, ts=int(tsi), dir=d, session=sess,
                           entry=round(float(entry), 6), sl=round(float(sl), 6),
                           tp1=round(float(tp1), 6), tp2=round(float(tp2), 6),
                           risk=round(float(risk), 6), fee=round(float(fee), 6),
                           R_tp1=round(float(R1), 4), flag_tp1=f1,
                           R_tp2=round(float(R2), 4), flag_tp2=f2,
                           R_part=round(float(Rp), 4), flag_part=fp,
                           tp1_hit=int(tp1_idx >= 0), tp2_hit=int(tp2_idx >= 0),
                           sl_hit=int(sl_idx >= 0),
                           sig_atr=float(atr_chart[i]), sig_vol=float(v[i])))
    return trades

def metrics(R_list, fee_list):
    arr = np.asarray(R_list, float) - np.asarray(fee_list, float)
    n = len(arr)
    if n == 0:
        return dict(n=0, wr=float("nan"), pf=float("nan"), ret=float("nan"),
                    totalR=0.0, avgR=float("nan"), maxDD=float("nan"))
    wins = arr[arr > 0]; losses = arr[arr < 0]
    pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 1e-12 else (np.inf if wins.sum() > 0 else 0.0)
    wr = 100.0 * (arr > 0).mean()
    eqc = np.cumprod(1 + 0.01 * arr)
    ret = (eqc[-1] - 1) * 100.0
    peak = np.maximum.accumulate(eqc)
    maxDD = ((eqc - peak) / peak).min() * 100.0
    return dict(n=n, wr=wr, pf=pf, ret=ret, totalR=arr.sum(), avgR=arr.mean(), maxDD=maxDD)

def main():
    t0 = time.time()
    prot_mode = sys.argv[1] if len(sys.argv) > 1 else "faithful"
    out_trades = f"/home/user/prov2_trades_{prot_mode}.csv"
    rows_trades = []
    rows_sum = []
    rows_sess = []

    df15_cache = {}
    for sym in SYMS:
        p15 = f"{DATA}/{sym}_15m.csv"
        if os.path.exists(p15):
            df15_cache[sym] = pd.read_csv(p15)
        for tf in TFS:
            p = f"{DATA}/{sym}_{tf}.csv"
            if not os.path.exists(p):
                print(f"  MANQUANT {p}", flush=True); continue
            df = pd.read_csv(p)
            if len(df) < 200:
                continue
            tr = run_symbol_tf(sym, tf, df, df15_cache.get(sym), prot_mode=prot_mode)
            if not tr:
                continue
            tdf = pd.DataFrame(tr)
            rows_trades.append(tdf)
            for mode, col in (("TP1_3R", "R_tp1"), ("TP2_6R", "R_tp2"), ("PARTIAL", "R_part")):
                m = metrics(tdf[col].values, tdf["fee"].values)
                rows_sum.append(dict(sym=sym, tf=tf, mode=mode, n=m["n"], win_rate=round(m["wr"], 1),
                                     pf=round(m["pf"], 2), net_return=round(m["ret"], 2),
                                     totalR=round(m["totalR"], 2), avgR=round(m["avgR"], 3),
                                     maxDD=round(m["maxDD"], 2)))
            for sess in ["ASIE", "LONDRES", "NEW YORK"]:
                sub = tdf[tdf["session"] == sess]
                for mode, col in (("TP1_3R", "R_tp1"), ("TP2_6R", "R_tp2"), ("PARTIAL", "R_part")):
                    if len(sub) == 0:
                        continue
                    m = metrics(sub[col].values, sub["fee"].values)
                    rows_sess.append(dict(sym=sym, tf=tf, mode=mode, session=sess, n=m["n"],
                                          win_rate=round(m["wr"], 1), pf=round(m["pf"], 2),
                                          net_return=round(m["ret"], 2), totalR=round(m["totalR"], 2)))
            print(f"  {sym} {tf:4s}: {len(tr)} trades ({time.time()-t0:.0f}s)", flush=True)

    allt = pd.concat(rows_trades, ignore_index=True)
    allt.to_csv(out_trades, index=False)
    pd.DataFrame(rows_sum).to_csv(f"/home/user/prov2_summary_{prot_mode}.csv", index=False)
    pd.DataFrame(rows_sess).to_csv(f"/home/user/prov2_sessions_{prot_mode}.csv", index=False)
    print(f"\nTrades: {len(allt)} | summary + sessions sauvegardés ({prot_mode})")
    print("Terminé en %.0f s" % (time.time() - t0))

if __name__ == "__main__":
    main()
