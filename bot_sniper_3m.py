#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import backtest_pro_v2 as bt
bt.FEE_RT = 0.0

TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN_SNIPER') or os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID_SNIPER') or os.getenv('TELEGRAM_CHAT_ID')

SYMS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
RR_MAP = {"BTC/USDT": 3.0, "ETH/USDT": 5.0, "BNB/USDT": 5.0, "SOL/USDT": 3.0}

TF = "3m"
TF_MS = 180_000
N_BARS = 9600
SESSIONS_TRADE = ("LONDRES", "NEW YORK")
MAX_LOSSES = 2
STALE_MS = 600_000
LOOP_SECONDS = 60
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sniper3m_state.json")

SES_HOURS = {"LONDRES": "07:00-10:00", "NEW YORK": "13:00-16:00"}

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def session_of(ts):
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    hhmm = dt.hour * 100 + dt.minute
    if 700 <= hhmm < 1000:
        return "LONDRES"
    if 1300 <= hhmm < 1600:
        return "NEW YORK"
    if 0 <= hhmm < 400:
        return "ASIE"
    return "FERME"

def fmt_px(p):
    return f"{p:,.2f}".replace(",", " ")

def fmt_day(day):
    return datetime.fromtimestamp(day * 86_400, tz=timezone.utc).strftime("%d/%m")

def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(st):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE_FILE)

def sym_state(st, sym):
    return st.setdefault(sym, {
        "pos": None,
        "last_entry_ts": 0,
        "rule": {"key": None, "consec": 0, "stopped": False},
    })

def send_telegram_message(message, token, chat_id):
    if not token or not chat_id:
        log(f"(pas de Telegram configuré)\n{message}")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=15)
        if r.status_code != 200:
            log(f"Telegram {r.status_code} : {r.text[:200]}")
    except Exception as e:
        log(f"Erreur Telegram : {e}")

def fetch_mexc(ex, sym, n=N_BARS):
    now = ex.milliseconds()
    since = now - (n + 2) * TF_MS
    rows = []
    while since < now:
        batch = ex.fetch_ohlcv(sym, TF, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + TF_MS
        if nxt <= since:
            break
        since = nxt
        time.sleep(max(getattr(ex, "rateLimit", 100), 100) / 1000)
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df[df["ts"] + TF_MS <= now].reset_index(drop=True)
    return df.tail(n).reset_index(drop=True)

def fetch_okx(inst, n=N_BARS):
    from entry_methods import okx_page
    out = []
    after = int(time.time() * 1000)
    while len(out) < n + 10:
        data = okx_page(inst, "3m", after)
        if not data:
            break
        out.extend(data)
        if len(data) < 100:
            break
        after = int(data[-1][0])
        time.sleep(0.15)
    if not out:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])
    df = pd.DataFrame(out).iloc[:, :6]
    df.columns = ["ts", "open", "high", "low", "close", "vol"]
    df["ts"] = df["ts"].astype(np.int64)
    for c in ("open", "high", "low", "close", "vol"):
        df[c] = df[c].astype(np.float64)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    now = int(time.time() * 1000)
    df = df[df["ts"] + TF_MS <= now].reset_index(drop=True)
    return df.tail(n).reset_index(drop=True)

def run_engine(sym, df):
    return bt.run_symbol_tf(
        sym, TF, df,
        prot_mode="reversal",
        allUT=False,
        rr=RR_MAP[sym],
        coolBars=3, minGap=0.3,
        pinOn=True, arOn=True, pinRatio=2.0,
        entry_mode="close",
    )

def rule_key_of(day, session):
    return f"{day}_{session}"

def rule_on_close(rule, key, r):
    if rule.get("key") != key:
        rule.update(key=key, consec=0, stopped=False)
    if r < 0:
        rule["consec"] += 1
    else:
        rule["consec"] = 0
    if rule["consec"] >= MAX_LOSSES and not rule["stopped"]:
        rule["stopped"] = True
        return True
    return False

def rule_blocks(rule, key):
    return rule.get("key") == key and rule.get("stopped")

def resolve_pos(pos, df):
    ts = df["ts"].to_numpy(np.int64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    i0 = int(np.searchsorted(ts, pos["checked_ts"], side="right"))
    tp1_event = None
    for i in range(i0, len(ts)):
        if pos["dir"] == "LONG":
            if l[i] <= pos["sl"]:
                return (-1.0, "SL", pos["sl"], int(ts[i])), tp1_event
            if tp1_event is None and h[i] >= pos["tp1"]:
                tp1_event = (pos["tp1"], int(ts[i]))
            if h[i] >= pos["tp2"]:
                return (pos["rr"] * 2, "TP2", pos["tp2"], int(ts[i])), tp1_event
        else:
            if h[i] >= pos["sl"]:
                return (-1.0, "SL", pos["sl"], int(ts[i])), tp1_event
            if tp1_event is None and l[i] <= pos["tp1"]:
                tp1_event = (pos["tp1"], int(ts[i]))
            if l[i] <= pos["tp2"]:
                return (pos["rr"] * 2, "TP2", pos["tp2"], int(ts[i])), tp1_event
    if len(ts):
        pos["checked_ts"] = int(ts[-1])
    return None, tp1_event

def msg_setup(sym, t, rule):
    dt = datetime.fromtimestamp(t["ts"] / 1000, tz=timezone.utc)
    risk = abs(t["entry"] - t["sl"])
    rr = RR_MAP[sym]
    return "\n".join([
        f"🚀 SNIPER 3m — {t['dir']}",
        f"{sym} | {t['session']} ({SES_HOURS[t['session']]} GMT) — {fmt_day(t['ts'] // 86_400_000)}",
        f"Signal 3m de {dt.strftime('%H:%M')} GMT (close de bougie)",
        f"Entrée : {fmt_px(t['entry'])}",
        f"SL     : {fmt_px(t['sl'])}   (1R = {risk / t['entry'] * 100:.3f}%)",
        f"TP1    : {fmt_px(t['tp1'])}   (+{rr:g}R)",
        f"TP2    : {fmt_px(t['tp2'])}   (+{rr * 2:g}R)",
        f"Pertes session : {rule['consec']}/{MAX_LOSSES}",
    ])

def msg_tp1(sym, pos, px, tp1_ts):
    dt = datetime.fromtimestamp(tp1_ts / 1000, tz=timezone.utc)
    return "\n".join([
        f"🎯 SNIPER 3m — {sym} TP1 TOUCHÉ (+{pos['rr']:g}R)",
        f"Entrée {fmt_px(pos['entry'])} → TP1 {fmt_px(px)} — {dt.strftime('%H:%M')} GMT",
        f"Position continue : TP2 {fmt_px(pos['tp2'])} (+{pos['rr'] * 2:g}R) | SL {fmt_px(pos['sl'])}",
    ])

def msg_exit(sym, pos, r, flag, px, rule):
    if r < 0:
        head = f"❌ {sym} — SL TOUCHÉ (-1R)"
        tail = f"Pertes ({pos['session']} {fmt_day(pos['day'])}) : {rule['consec']}/{MAX_LOSSES}"
        tp1line = " | TP1 avait été touché (retour à l'entrée puis SL — classique, -1R compté)" if pos.get("tp1_hit") else ""
    else:
        head = f"✅ {sym} — TP2 TOUCHÉ (+{pos['rr'] * 2:g}R)"
        tail = "Compteur de pertes remis à zéro (0/2)"
        tp1line = " | TP1 touché avant, la queue a fait le travail" if pos.get("tp1_hit") else " | TP1 jamais touché (gap direct TP2)"
    return "\n".join([
        head,
        f"Entrée {fmt_px(pos['entry'])} → sortie {fmt_px(px)}{tp1line}",
        tail,
    ])

def msg_stop(sym, rule):
    day, ses = rule["key"].rsplit("_", 1)
    return "\n".join([
        f"🛑 {sym} — SESSION STOPPÉE",
        f"{MAX_LOSSES} pertes consécutives ({ses}, {fmt_day(int(day))}).",
        f"Plus de trade sur {sym} avant la prochaine (jour, session).",
    ])

def msg_heartbeat(st):
    dt = datetime.now(timezone.utc)
    lines = [f"🤖 Sniper 3m — OK ({dt.strftime('%d/%m %H:%M')} GMT)"]
    for sym in SYMS:
        s = st.get(sym)
        if not s:
            continue
        rule = s["rule"]
        if s.get("pos"):
            p = s["pos"]
            when = datetime.fromtimestamp(p["ts"] / 1000, tz=timezone.utc).strftime('%d/%m %H:%M')
            lines.append(f"{sym} : {p['dir']} depuis {when} ({p['session']}) | pertes {rule['consec']}/{MAX_LOSSES}")
        else:
            stop = "  🛑 SESSION STOPPÉE" if rule.get("stopped") else ""
            lines.append(f"{sym} : flat | pertes {rule['consec']}/{MAX_LOSSES}{stop}")
    return "\n".join(lines)

def process_symbol(st, sym, df, notify):
    s = sym_state(st, sym)
    now = int(time.time() * 1000)

    pos = s.get("pos")
    if pos is not None:
        res, tp1_event = resolve_pos(pos, df)
        if tp1_event is not None:
            tp1_px, tp1_ts = tp1_event
            pos["tp1_hit"] = True
            notify(msg_tp1(sym, pos, tp1_px, tp1_ts))
        if res is not None:
            r, flag, px, _ets = res
            key = rule_key_of(pos["day"], pos["session"])
            stopped = rule_on_close(s["rule"], key, r)
            s["pos"] = None
            notify(msg_exit(sym, pos, r, flag, px, s["rule"]))
            if stopped:
                notify(msg_stop(sym, s["rule"]))

    trades = [t for t in run_engine(sym, df) if t["session"] in SESSIONS_TRADE]
    trades.sort(key=lambda t: t["ts"])
    for t in trades:
        if t["ts"] <= s.get("last_entry_ts", 0):
            continue
        s["last_entry_ts"] = t["ts"]
        if s.get("pos") is not None:
            continue
        if now - t["ts"] > STALE_MS:
            continue
        key = rule_key_of(t["ts"] // 86_400_000, t["session"])
        if rule_blocks(s["rule"], key):
            continue
        s["pos"] = {
            "ts": t["ts"], "dir": t["dir"], "entry": t["entry"], "sl": t["sl"],
            "tp1": t["tp1"], "tp2": t["tp2"], "session": t["session"],
            "day": t["ts"] // 86_400_000, "rr": RR_MAP[sym],
            "checked_ts": t["ts"], "tp1_hit": False,
        }
        notify(msg_setup(sym, t, s["rule"]))
        break

def run_cycle(st, ex, use_mexc, notify):
    for sym in SYMS:
        try:
            if use_mexc:
                df = fetch_mexc(ex, sym)
            else:
                df = fetch_okx(sym.replace("/USDT", "-USDT"))
            if len(df) < 300:
                log(f"{sym} : données insuffisantes ({len(df)} bougies) — cycle ignoré")
                continue
            if len(df) < 5000:
                log(f"{sym} : attention, seulement {len(df)} bougies")
            process_symbol(st, sym, df, notify)
        except Exception as e:
            log(f"{sym} : erreur {e!r}")
    save_state(st)

    hd = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if st.get("last_hb") != hd:
        st["last_hb"] = hd
        notify(msg_heartbeat(st))
        save_state(st)

def main():
    if "--test" in sys.argv:
        run_test()
        return

    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        print("ERREUR : identifiants Telegram non configurés.")
        sys.exit(1)

    ex, use_mexc = None, False
    try:
        import ccxt
        ex = ccxt.mexc({"enableRateLimit": True})
        ex.load_markets()
        probe = fetch_mexc(ex, "BTC/USDT", n=100)
        if len(probe) >= 50:
            use_mexc = True
    except Exception as e:
        log(f"MEXC inaccessible ({e!r}) → fallback OKX")

    st = load_state()
    notify = lambda m: send_telegram_message(m, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    src = "MEXC" if use_mexc else "OKX"

    if "--once" in sys.argv:
        log(f"Mode --once — source {src} — un cycle puis sortie")
        run_cycle(st, ex, use_mexc, notify)
        return

    log(f"Démarrage du bot Sniper 3m (boucle {LOOP_SECONDS}s) — source {src}")
    notify("🤖 Sniper 3m démarré — BTC/ETH/BNB/SOL — config V2.3 validée")

    while True:
        t0 = time.time()
        run_cycle(st, ex, use_mexc, notify)
        time.sleep(max(5, LOOP_SECONDS - (time.time() - t0)))

if __name__ == "__main__":
    main()
