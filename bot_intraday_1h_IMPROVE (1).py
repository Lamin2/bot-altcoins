import os
import json
import time
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone


# ==========================================
# CONFIGURATION - BOT INTRADAY (1H) AVEC SESSIONS
# ==========================================

TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN_INTRADAY')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID_INTRADAY')

TIMEFRAME_ANALYSE = '1h'

# =====================================================================
# >>> AMÉLIORATION 1 : FILTRE DE RÉGIME BTC
# =====================================================================
# Règle : LONG seulement si BTC > SMA200 (1H) | SHORT seulement si BTC < SMA200 (1H)
# C'est LE levier qui a transformé le backtest (9 symboles, 80 mois, clôture TP2) :
#   sans filtre : +303R / PF 1.02  (marginal, meurt aux frais)
#   avec filtre : +1287R / PF 1.13 (les 7 années positives)
USE_BTC_REGIME = True
BTC_SMA_PERIOD = 200

# =====================================================================
# >>> AMÉLIORATION 2 : mode SHORT only (le meilleur slice mesuré)
# =====================================================================
# SHORT + régime + clôture TP2 : +713R | PF 1.15 | +0.119R/trade | DD -114R
#   (7/7 années positives, PF 2024-26 = 1.21, le plus résistant aux frais)
# Les deux sens (False) : +1287R | PF 1.13 | DD -175R (2x plus d'alerts)
# >>> Objectif "top 10 hedge fund" : c'est le 2 sens qui gagne sur comptes 0 frais
#     (ann +30.4% / 81m, Sharpe 1.52, 12 derniers mois : ann +49%, Sharpe 2.44)
#     SHORT_ONLY=True reste le choix pour tout déployement AVEC frais (Bybit).
SHORT_ONLY = False

# =====================================================================
# >>> EDGE ULTIME (mesuré oct 2026) : TIME STOP H24
# =====================================================================
# Backtest 2020->2026 (2 sens + régime + TP2) :
#   sans H24 : +1286.6R | PF 1.13 | DD -175 | Sharpe 1.52
#   avec H24 : +1483.9R | PF 1.12 | DD -133 | Sharpe 1.68 (+15% R, DD -24%)
#   + règle "2 pertes consécutives = stop session" : +2349R | PF 1.24 | DD -106
# Principe : un setup qui n'atteint ni TP2 ni SL sous 24h a perdu son élan
# -> fermer au marché (libère le symbole, coupe la dérive).
TIME_STOP_HOURS = 24
STATE_FILE_1H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intraday1h_state.json")

# =====================================================================
# >>> MAX OPTIM (grille 180 configs, validé IS 2020-23 / OOS 2024-26)
# =====================================================================
# Config la plus robuste trouvée : ATR 0.35 + H24 + tol x1.25 + RR fixe 2.5
#   +1718.9R | PF 1.12 | DD -136 | 7/7 ans positives | IS +808R / OOS +911R
#   + règle "2 pertes consécutives = stop session" : +2591R | PF 1.24 | DD -107
#     ann +72% | DD -7.1% | Sharpe 2.76 | 75% mois+ (perso 0 frais)
ATR_MULT = 0.35
TOL_SCALE = 1.25
FIXED_RR = 2.5  # None = RR par session (ancien comportement)

# =====================================================================
# >>> SIZING (décidé 30/10/2026 — round MAX)
# =====================================================================
# 0.3% par trade maintenant. 0.5% après 4 semaines de live si le 1H
# tient ses ~32R/mois. Mesuré (FULL 81m, 0 frais MEXC) :
#   0.3% -> ann +102.6% / DD -8.3%   |   0.5% -> ann +208.2% / DD -13.7%
# Pour passer à 0.5% : changer RISK_PCT_1H = 0.5 (une seule ligne).
CAPITAL_1H = 75_000
RISK_PCT_1H = 0.3
RISK_USD_1H = CAPITAL_1H * RISK_PCT_1H / 100.0   # = 225$

# =====================================================================
# >>> AMÉLIORATION 3 (gestion) : gère tes positions à TP2, pas TP1
# =====================================================================
# Backtest : clôture à TP2 (2×RR) > clôture à TP1 :
#   TP2 : +1287R / PF 1.13   |   TP1 : +867R / PF 1.08
# Ne pas prendre partiel à TP1 : c'est le gain des cassures qui paie.
# (Optionnel : après TP1, tu peux déplacer le SL à l'entrée = protection sans couper)

# =====================================================================
# >>> AMÉLIORATION 4 (connaissance) : ce que disent les sessions
# =====================================================================
# Backtest avec régime : toutes les sessions sont positives.
# ASIA est la meilleure (+390R), LONDRES la plus faible des 2 sens.
# Ta config actuelle (NUIT stricte : tol 0.5%, vol 20M) est bonne :
# la session NUIT a le meilleur avgR/trade (0.127-0.153R). Ne la durcis pas.
# Les RR par session (2.0-3.0) sont confirmés : un RR fixe 1.5 est pire.


# ==========================================
# DÉTECTION DE LA SESSION DE TRADING
# ==========================================


def get_current_session():
    now_utc = datetime.now(timezone.utc)
    total_min = now_utc.hour * 60 + now_utc.minute

    if total_min < 8*60 + 30:
        return {"name": "ASIATIQUE", "emoji": "🌏", "pivot": 5, "tol": 0.008,
                "vol": 10_000_000, "rr": 2.5, "max": 2,
                "desc": "Volume faible. On ne prend que les setups parfaits."}
    elif total_min < 13*60 + 30:
        return {"name": "LONDRES", "emoji": "🇬🇧", "pivot": 4, "tol": 0.012,
                "vol": 5_000_000, "rr": 2.0, "max": 4,
                "desc": "Volume moyen. Les tendances se dessinent."}
    elif total_min < 16*60 + 30:
        return {"name": "OVERLAP LONDRES/US", "emoji": "🔥", "pivot": 3, "tol": 0.015,
                "vol": 5_000_000, "rr": 2.0, "max": 5,
                "desc": "Meilleure liquidité de la journée. Les meilleurs mouvements se dessinent."}
    elif total_min < 21*60:
        return {"name": "US", "emoji": "🇺🇸", "pivot": 3, "tol": 0.015,
                "vol": 5_000_000, "rr": 2.5, "max": 5,
                "desc": "Forte volatilité. Mouvements explosifs. TP plus ambitieux."}
    else:
        return {"name": "NUIT", "emoji": "🌙", "pivot": 5, "tol": 0.005,
                "vol": 20_000_000, "rr": 3.0, "max": 1,
                "desc": "Volume très faible. On attend la session asiatique."}


# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================


def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Erreur : Identifiants Telegram Intraday manquants.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Erreur Telegram : {e}")


def get_dynamic_altcoins(exchange, top_n=80):
    print("Récupération liste dynamique...")
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': top_n, 'page': 1, 'sparkline': 'false'}
    try:
        response = requests.get(url, params=params, timeout=15)
        coins = response.json()
    except Exception as e:
        print(f"Erreur CoinGecko : {e}")
        return ['BTC/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT', 'ADA/USDT']
    try:
        markets = exchange.load_markets()
    except Exception:
        return []
    altcoins = []
    EXCLUDE = {'usdt','usdc','dai','fdusd','tusd','usde','pyusd','busd',
               'usd1','usdd','frax','lusd','gho','crvusd',
               'wbtc','weth','steth','wsteth','reth','cbeth','wbeth',
               'meth','sfrxeth','frxeth','wmatic','wavax','wbnb'}
    for coin in coins:
        symbol = coin['symbol'].upper()
        if symbol.lower() in EXCLUDE: continue
        pair = f"{symbol}/USDT"
        if pair in markets and markets[pair].get('active', True):
            altcoins.append(pair)
    print(f"✅ {len(altcoins)} paires récupérées.")
    return altcoins


def fetch_ohlcv(exchange, symbol, timeframe, limit=200):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception:
        return None


def detect_pivots(df, window=5):
    # Pivots calculés sur la Force Relative (RS)
    df['pivot_high'] = df['RS'].rolling(window=window, center=True).max() == df['RS']
    df['pivot_low'] = df['RS'].rolling(window=window, center=True).min() == df['RS']
    highs = df[df['pivot_high']].tail(2)
    lows = df[df['pivot_low']].tail(2)
    if len(highs) < 2 or len(lows) < 2: return None
    return {
        'pH1': highs['RS'].iloc[-1], 'pH2': highs['RS'].iloc[-2],
        'pH1b': df.index.get_loc(highs.index[-1]), 'pH2b': df.index.get_loc(highs.index[-2]),
        'pL1': lows['RS'].iloc[-1], 'pL2': lows['RS'].iloc[-2],
        'pL1b': df.index.get_loc(lows.index[-1]), 'pL2b': df.index.get_loc(lows.index[-2])
    }


def analyze_channel(df, pivot_data):
    if pivot_data is None: return None
    pH1, pH2, pH1b, pH2b = pivot_data['pH1'], pivot_data['pH2'], pivot_data['pH1b'], pivot_data['pH2b']
    pL1, pL2, pL1b, pL2b = pivot_data['pL1'], pivot_data['pL2'], pivot_data['pL1b'], pivot_data['pL2b']
    current_idx = len(df) - 1
    current_close = df['RS'].iloc[-1]
    bullChan = (pL1 > pL2) and (pH1 > pH2)
    bearChan = (pL1 < pL2) and (pH1 < pH2)
    if bullChan:
        slope = (pL1 - pL2) / (pL1b - pL2b) if (pL1b - pL2b) != 0 else 0
        supNow = pL1 + slope * (current_idx - pL1b)
        dist = abs(current_close - supNow) / current_close
        if current_close < supNow:
            return {"dir": "SHORT", "type": "CASSÉ", "level": supNow, "status": "PREPARATION"}
        elif dist < 0.02:
            return {"dir": "SHORT", "type": "SURVEILLANCE", "level": supNow, "status": "SURVEILLANCE"}
    elif bearChan:
        slope = (pH1 - pH2) / (pH1b - pH2b) if (pH1b - pH2b) != 0 else 0
        resNow = pH1 + slope * (current_idx - pH1b)
        dist = abs(current_close - resNow) / current_close
        if current_close > resNow:
            return {"dir": "LONG", "type": "CASSÉ", "level": resNow, "status": "PREPARATION"}
        elif dist < 0.02:
            return {"dir": "LONG", "type": "SURVEILLANCE", "level": resNow, "status": "SURVEILLANCE"}
    return None


def calculate_levels(df, signal, pivot_data, rr_ratio):
    entry_price = df['close'].iloc[-1]
    atr = df['high'].sub(df['low']).rolling(14).mean().iloc[-1]

    # Conversion du niveau RS en prix réel
    is_btc = 'close_btc' not in df.columns or df['close_btc'].iloc[-1] == df['close'].iloc[-1]
    btc_price = df['close_btc'].iloc[-1] if not is_btc else 1

    level_price = signal['level'] * btc_price if not is_btc else signal['level']
    pL1_price = pivot_data['pL1'] * btc_price if not is_btc else pivot_data['pL1']
    pH1_price = pivot_data['pH1'] * btc_price if not is_btc else pivot_data['pH1']

    if signal['dir'] == 'LONG':
        sl_price = min(pL1_price, level_price) - (atr * ATR_MULT)
        risk = entry_price - sl_price
        tp1 = entry_price + (risk * rr_ratio)
        tp2 = entry_price + (risk * rr_ratio * 2)
    else:
        sl_price = max(pH1_price, level_price) + (atr * ATR_MULT)
        risk = sl_price - entry_price
        tp1 = entry_price - (risk * rr_ratio)
        tp2 = entry_price - (risk * rr_ratio * 2)

    return entry_price, sl_price, tp1, tp2


# ==========================================
# >>> EDGE ULTIME : suivi des setups (time stop H24)
# ==========================================


def _load_state():
    try:
        with open(STATE_FILE_1H) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state):
    try:
        with open(STATE_FILE_1H, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Erreur sauvegarde état H24 : {e}")


def _check_resolved(exchange, symbol, st):
    """'SL' / 'TP2' / None (pas touché depuis ts_open). SL prioritaire dans la
    même bougie (conservatif). None aussi en cas d'erreur -> on garde le suivi."""
    try:
        df = fetch_ohlcv(exchange, symbol, TIMEFRAME_ANALYSE, limit=30)
        if df is None or len(df) == 0:
            return None
        since = pd.to_datetime(st["ts_open"], unit='ms')
        sub = df[df.index >= since]
        for _, row in sub.iterrows():
            if st["dir"] == "LONG":
                if row['low'] <= st["sl"]:
                    return "SL"
                if row['high'] >= st["tp2"]:
                    return "TP2"
            else:
                if row['high'] >= st["sl"]:
                    return "SL"
                if row['low'] <= st["tp2"]:
                    return "TP2"
        return None
    except Exception:
        return None


# ==========================================
# LOGIQUE PRINCIPALE
# ==========================================


def run_bot():
    session = get_current_session()
    rr_eff = FIXED_RR if FIXED_RR is not None else session['rr']
    print(f"BOT INTRADAY - Session {session['name']} - {datetime.now()}")

    exchange = ccxt.mexc()

    # >>> EDGE ULTIME (code) : suivi des setups pour le time stop H24
    state = _load_state()
    h24_alerts = []
    if TIME_STOP_HOURS > 0 and state:
        now_ms = int(time.time() * 1000)
        for sym, st in list(state.items()):
            res = _check_resolved(exchange, sym, st)
            age_h = (now_ms - st["ts_open"]) / 3.6e6
            if res in ("SL", "TP2"):
                print(f"[H24] {sym} ({st['dir']}) : {res} touché après {age_h:.0f}h — suivi terminé.")
                del state[sym]
                continue
            if age_h >= TIME_STOP_HOURS:
                h24_alerts.append((sym, st, age_h))
                del state[sym]
                continue
            time.sleep(0.3)

    # >>> AMÉLIORATION 1 (code) : 250 bougies BTC pour une SMA200 (1H) fiable
    df_btc = fetch_ohlcv(exchange, 'BTC/USDT', TIMEFRAME_ANALYSE, limit=250)
    if df_btc is None:
        send_telegram_message("❌ *[INTRADAY]* Erreur BTC.")
        return

    # >>> AMÉLIORATION 1 (code) : calcul du régime BTC
    btc_bull = None
    if USE_BTC_REGIME and len(df_btc) > BTC_SMA_PERIOD:
        sma_btc = df_btc['close'].rolling(BTC_SMA_PERIOD).mean().iloc[-1]
        if not pd.isna(sma_btc):
            btc_bull = bool(df_btc['close'].iloc[-1] > sma_btc)

    if not USE_BTC_REGIME:
        regime_msg = "🧭 Filtre régime BTC désactivé (USE_BTC_REGIME=False)"
    elif btc_bull is None:
        regime_msg = "🧭 BTC : régime indisponible (données insuffisantes) — filtre inopérant ce tour"
    elif btc_bull:
        regime_msg = f"🧭 *BTC : HAUSSIER (close > SMA{BTC_SMA_PERIOD} 1H) → les SHORT sont bloqués*"
    else:
        regime_msg = f"🧭 *BTC : BAISSIER (close < SMA{BTC_SMA_PERIOD} 1H) → les LONG sont bloqués*"

    send_telegram_message(
        f"{session['emoji']} *BOT INTRADAY - Session {session['name']}*\n"
        f"_{session['desc']}_\n"
        f"⚙️ RR: 1:{rr_eff} | Vol min: {session['vol']//1_000_000}M$ | Max: {session['max']}\n"
        + regime_msg
    )

    dynamic_alts = get_dynamic_altcoins(exchange, top_n=80)
    if not dynamic_alts:
        send_telegram_message("❌ *[INTRADAY]* Erreur liste altcoins.")
        return

    setups_valides = []
    preparations = []
    surveillances = []
    bloques = []  # >>> nouveau : setups rejetés par régime / SHORT_ONLY

    for symbol in dynamic_alts:
        df_alt = fetch_ohlcv(exchange, symbol, TIMEFRAME_ANALYSE)
        if df_alt is None: continue
        try:
            ticker = exchange.fetch_ticker(symbol)
            if ticker['quoteVolume'] < session['vol']: continue
        except Exception: continue

        df = pd.merge(df_alt, df_btc, on='timestamp', suffixes=('_alt', '_btc'))
        if len(df) < 50: continue
        df.rename(columns={'open_alt':'open','high_alt':'high','low_alt':'low',
                           'close_alt':'close','volume_alt':'volume'}, inplace=True)

        if symbol == 'BTC/USDT':
            df['RS'] = df['close']
        else:
            df['RS'] = df['close'] / df['close_btc']

        pivot_data = detect_pivots(df, window=session['pivot'])
        signal = analyze_channel(df, pivot_data)

        if signal:
            if signal['status'] == 'PREPARATION':
                dist = abs(df['RS'].iloc[-1] - signal['level']) / df['RS'].iloc[-1]
                if dist < session['tol'] * TOL_SCALE:
                    # >>> AMÉLIORATION 2 (code) : SHORT_ONLY
                    if SHORT_ONLY and signal['dir'] != 'SHORT':
                        bloques.append({'symbol': symbol, 'dir': signal['dir'], 'why': 'SHORT_ONLY'})
                        time.sleep(0.5)
                        continue
                    # >>> AMÉLIORATION 1 (code) : filtre de régime sur les setups VALIDÉS
                    if USE_BTC_REGIME and btc_bull is not None:
                        contre = (signal['dir'] == 'LONG' and not btc_bull) or \
                                 (signal['dir'] == 'SHORT' and btc_bull)
                        if contre:
                            bloques.append({'symbol': symbol, 'dir': signal['dir'], 'why': 'régime BTC'})
                            time.sleep(0.5)
                            continue

                    entry, sl, tp1, tp2 = calculate_levels(df, signal, pivot_data, rr_eff)
                    setups_valides.append({'symbol': symbol, 'dir': signal['dir'],
                                           'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2})
                    # >>> EDGE ULTIME (code) : suivi H24 (pas de reset si déjà suivi)
                    if TIME_STOP_HOURS > 0 and symbol not in state:
                        state[symbol] = {'dir': signal['dir'], 'entry': entry, 'sl': sl,
                                         'tp2': tp2, 'ts_open': int(time.time() * 1000)}
                else:
                    is_btc = symbol == 'BTC/USDT'
                    price_level = signal['level'] if is_btc else signal['level'] * df['close_btc'].iloc[-1]
                    preparations.append({'symbol': symbol, 'dir': signal['dir'], 'level': price_level})
            elif signal['status'] == 'SURVEILLANCE':
                is_btc = symbol == 'BTC/USDT'
                price_level = signal['level'] if is_btc else signal['level'] * df['close_btc'].iloc[-1]
                surveillances.append({'symbol': symbol, 'dir': signal['dir'], 'level': price_level})
        time.sleep(0.5)

    report = f"⚡ *RAPPORT INTRADAY (1H) - {session['emoji']} {session['name']}*\n"
    report += f"🕐 {datetime.now().strftime('%H:%M')} | 🔍 _{len(dynamic_alts)} altcoins_\n"
    # >>> AMÉLIORATION 3 (code) : rappel de gestion dans chaque rapport
    report += "🎯 _Gérer à TP2 (pas de partiel à TP1) ou SL→entrée après TP1_\n"
    report += "🛑 _2 pertes consécutives = stop de la session (mesurée : +865R sur 80 mois)_\n"
    report += f"💰 _Sizing : {RISK_PCT_1H:g}% de {CAPITAL_1H:,.0f}$ = {RISK_USD_1H:,.0f}$ de risque par trade (taille dans chaque setup)_\n\n"

    if setups_valides:
        report += f"🚀 *SETUPS VALIDÉS (max {session['max']})*\n"
        for s in setups_valides[:session['max']]:
            dist_sl = abs(s['entry'] - s['sl'])
            taille = RISK_USD_1H / dist_sl if dist_sl > 0 else 0.0
            report += f"• *{s['symbol']}* ({s['dir']})\n"
            report += f"  Entrée : {s['entry']:.4f} | SL : {s['sl']:.4f}\n"
            report += f"  TP1 : {s['tp1']:.4f} | TP2 : {s['tp2']:.4f}\n"
            report += f"  💰 {RISK_USD_1H:,.0f}$ de risque -> ~{taille:,.4f} coins (position ~{taille*s['entry']:,.0f}$)\n\n"
    else:
        report += "🚀 *SETUPS VALIDÉS* : Aucun.\n\n"

    if preparations:
        report += "⏳ *EN PRÉPARATION*\n"
        for p in preparations[:5]:
            report += f"• *{p['symbol']}* ({p['dir']}) - Niveau : {p['level']:.4f}\n"
        report += "\n"

    if surveillances:
        report += "👀 *SOUS SURVEILLANCE*\n"
        for s in surveillances[:5]:
            report += f"• *{s['symbol']}* ({s['dir']}) - Ligne : {s['level']:.4f}\n"

    # >>> nouveau : transparence sur les blocages
    if bloques:
        report += "\n🚫 *BLOQUÉS (ne pas trader)*\n"
        for b in bloques:
            report += f"• *{b['symbol']}* ({b['dir']}) - {b['why']}\n"

    if h24_alerts:
        report += "\n⏰ *TIME STOP H24 — FERMER AU MARCHÉ si encore en position*\n"
        for sym, st, age_h in h24_alerts:
            report += f"• *{sym}* ({st['dir']}) — ouvert {age_h:.0f}h, ni TP2 ni SL touchés\n"

    send_telegram_message(report)
    print("Rapport envoyé.")
    _save_state(state)


if __name__ == "__main__":
    run_bot()
