import os
import json
import time
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime


# ==========================================
# CONFIGURATION
# ==========================================

TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

TIMEFRAME_ANALYSE = '4h'
MIN_VOLUME_24H_USDT = 10_000_000

# TP1 = RISK_REWARD_RATIO R | TP2 = 2 x RISK_REWARD_RATIO R
# =====================================================================
# >>> AMÉLIORATION 1 (commentaire) : choix du RR, chiffres backtestés
# =====================================================================
# Backtest fidèle 6.8 ans (9 symboles, filtre régime BTC activé) :
#   RR = 1.5 -> WR 43.2% | PF 1.14 | DD -54R   (meilleur compromis WR/récompense)
#   RR = 2.0 -> WR 36.6% | PF 1.16 | DD -71R   (le plus de R par trade)
#   RR = 0.5 -> WR 66.4% | PF 0.99  (PIÈGE : WR massif mais PERTE nette)
# => 2.0 par défaut ; passe à 1.5 si tu veux plus de trades gagnants.
# >>> MAX OPTIM (grille 216 configs, validé IS 2020-23 / OOS 2024-26) :
# Config la plus robuste : RR 3.0 + tol 2.2% + pivot 7 + time stop 48 bougies (8j)
#   +399.2R | PF 1.29 | DD -41 | OOS 2024-26 : +122R (ancre : +83R)
#   + règle "2 pertes = stop" : +388R | PF 1.31 | DD -38
RISK_REWARD_RATIO = 3.0

PIVOT_WINDOW = 7
TOLERANCE_RETEST = 0.022
ATR_MULT = 0.5

# =====================================================================
# >>> SIZING (décidé 30/10/2026 — round MAX)
# =====================================================================
# 0.3% par trade maintenant. 0.5% après 4 semaines de live si le 1H
# tient ses ~32R/mois. Mesuré (FULL 81m, 0 frais MEXC) :
#   0.3% -> ann +102.6% / DD -8.3%   |   0.5% -> ann +208.2% / DD -13.7%
# Pour passer à 0.5% : changer RISK_PCT_4H = 0.5 (une seule ligne).
CAPITAL_4H = 50_000
RISK_PCT_4H = 0.3
RISK_USD_4H = CAPITAL_4H * RISK_PCT_4H / 100.0   # = 150$

# >>> TIME STOP 48 bougies 4H (8 jours) — même logique que le H24 du bot 1H
TIME_STOP_HOURS_4H = 48
STATE_FILE_4H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intraday4h_state.json")

# =====================================================================
# >>> AMÉLIORATION 2 : FILTRE DE RÉGIME BTC (le seul levier qui a ajouté
#                       de l'edge réel dans le backtest)
# =====================================================================
# Règle : LONG seulement si BTC > SMA200 (4H) | SHORT seulement si BTC < SMA200 (4H)
# Effet mesuré (2R) : +246R/PF 1.06  ->  +337R/PF 1.16, DD -107R -> -71R,
#                     LONG passe à WR 44.7% / PF 1.61 / DD -10R
USE_BTC_REGIME = True
BTC_SMA_PERIOD = 200

# =====================================================================
# >>> AMÉLIORATION 3 : paires à exclure (mesurées sur le backtest)
# =====================================================================
# BNB/USDT = -41R / PF 0.78 sur 6.8 ans (seul symbole perdant du panier)
SKIP_PAIRS = {'BNB/USDT'}

# =====================================================================
# >>> AMÉLIORATION 4 : mode SHORT only (mesuré sur la série vérifiée R2R)
# =====================================================================
# SHORT only : n=1085 | PF 1.23 | +154R | DD -17 | positif 7/7 années
#   12 derniers mois : PF 1.26 | DD -7 (le côté le plus propre)
# Les deux sens (False) : PF 1.16 | +260R | DD -23
#   → +1.3R/mois de plus en revenu brut (0 frais), mais LONG : -18R en 2022,
#     DD -35 sur les 12 derniers mois. Sur Bybit (frais) : préférer True.
# >>> Objectif "top 10 hedge fund" : sur 0 frais, le 2 sens gagne aussi en 4H
#     (ann +5.8% / Sharpe 1.07 / Calmar 1.63 vs SHORT only ann +3.4% / 0.93).
SHORT_ONLY = False


# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================


def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Erreur : Les identifiants Telegram ne sont pas configurés.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Erreur Telegram : {e}")


def fmt_price(p):
    """Format adaptatif : les micro-prix (PEPE ~0.000005) ne s'affichent pas à 0.0000"""
    if p >= 100: return f"{p:,.2f}"
    if p >= 1: return f"{p:.4f}"
    if p >= 0.01: return f"{p:.5f}"
    return f"{p:.8f}"


def fmt_qty(q):
    """Quantité de coins lisible quel que soit l'ordre de grandeur"""
    if q >= 1000: return f"{q:,.0f}"
    if q >= 1: return f"{q:,.2f}"
    return f"{q:,.4f}"


def get_dynamic_altcoins(exchange, top_n=100):
    print("Récupération de la liste dynamique des altcoins...")
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd', 'order': 'market_cap_desc',
        'per_page': top_n, 'page': 1, 'sparkline': 'false'
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        coins = response.json()
    except Exception as e:
        print(f"Erreur CoinGecko : {e}")
        return ['BTC/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT', 'ADA/USDT']

    try:
        markets = exchange.load_markets()
    except Exception as e:
        print(f"Erreur chargement MEXC : {e}")
        return []

    altcoins = []
    EXCLUDE = {
        'usdt', 'usdc', 'dai', 'fdusd', 'tusd', 'usde', 'pyusd', 'busd',
        'usd1', 'usdd', 'frax', 'lusd', 'gho', 'crvusd',
        'wbtc', 'weth', 'steth', 'wsteth', 'reth', 'cbeth', 'wbeth',
        'meth', 'sfrxeth', 'frxeth', 'wmatic', 'wavax', 'wbnb'
    }

    for coin in coins:
        symbol = coin['symbol'].upper()
        if symbol.lower() in EXCLUDE:
            continue
        pair = f"{symbol}/USDT"
        if pair in markets and markets[pair].get('active', True):
            altcoins.append(pair)

    print(f"✅ {len(altcoins)} paires tradables récupérées dynamiquement.")
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
    if len(highs) < 2 or len(lows) < 2:
        return None
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
        dist_to_sup = abs(current_close - supNow) / current_close
        if current_close < supNow:
            return {"dir": "SHORT", "type": "CASSÉ", "level": supNow, "status": "PREPARATION"}
        elif dist_to_sup < 0.02:
            return {"dir": "SHORT", "type": "SURVEILLANCE", "level": supNow, "status": "SURVEILLANCE"}

    elif bearChan:
        slope = (pH1 - pH2) / (pH1b - pH2b) if (pH1b - pH2b) != 0 else 0
        resNow = pH1 + slope * (current_idx - pH1b)
        dist_to_res = abs(current_close - resNow) / current_close
        if current_close > resNow:
            return {"dir": "LONG", "type": "CASSÉ", "level": resNow, "status": "PREPARATION"}
        elif dist_to_res < 0.02:
            return {"dir": "LONG", "type": "SURVEILLANCE", "level": resNow, "status": "SURVEILLANCE"}

    return None


def calculate_levels(df, signal, pivot_data):
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
        tp1 = entry_price + (risk * RISK_REWARD_RATIO)
        tp2 = entry_price + (risk * RISK_REWARD_RATIO * 2)
    else:
        sl_price = max(pH1_price, level_price) + (atr * ATR_MULT)
        risk = sl_price - entry_price
        tp1 = entry_price - (risk * RISK_REWARD_RATIO)
        tp2 = entry_price - (risk * RISK_REWARD_RATIO * 2)

    return entry_price, sl_price, tp1, tp2


# ==========================================
# LOGIQUE PRINCIPALE
# ==========================================


def _load_state():
    try:
        with open(STATE_FILE_4H) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state):
    try:
        with open(STATE_FILE_4H, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Erreur sauvegarde état H48 : {e}")


def _check_resolved(exchange, symbol, st):
    """'SL' / 'TP1' / None (pas touché depuis ts_open). SL prioritaire (conservatif).
    Le régime validé 4H gère à TP1 (3R) — c'est lui le signal de sortie."""
    try:
        df = fetch_ohlcv(exchange, symbol, TIMEFRAME_ANALYSE, limit=60)
        if df is None or len(df) == 0:
            return None
        since = pd.to_datetime(st["ts_open"], unit='ms')
        sub = df[df.index >= since]
        for _, row in sub.iterrows():
            if st["dir"] == "LONG":
                if row['low'] <= st["sl"]:
                    return "SL"
                if row['high'] >= st["tp1"]:
                    return "TP1"
            else:
                if row['high'] >= st["sl"]:
                    return "SL"
                if row['low'] <= st["tp1"]:
                    return "TP1"
        return None
    except Exception:
        return None


def run_bot():
    print(f"Démarrage du bot Swing - {datetime.now()}")
    exchange = ccxt.mexc()

    # >>> MAX OPTIM (code) : suivi des setups pour le time stop H48 (8 jours)
    state = _load_state()
    h48_alerts = []
    if TIME_STOP_HOURS_4H > 0 and state:
        now_ms = int(time.time() * 1000)
        for sym, st in list(state.items()):
            res = _check_resolved(exchange, sym, st)
            age_h = (now_ms - st["ts_open"]) / 3.6e6
            if res in ("SL", "TP1"):
                print(f"[H48] {sym} ({st['dir']}) : {res} touché après {age_h:.0f}h — suivi terminé.")
                del state[sym]
                continue
            if age_h >= TIME_STOP_HOURS_4H * 4:
                h48_alerts.append((sym, st, age_h))
                del state[sym]
                continue
            time.sleep(0.3)

    # >>> AMÉLIORATION 4 : 250 bougies pour avoir une SMA200 4H fiable
    # (200 bougies = 1 seule valeur de SMA200, zéro marge si une bougie manque)
    df_btc = fetch_ohlcv(exchange, 'BTC/USDT', TIMEFRAME_ANALYSE, limit=250)
    if df_btc is None:
        send_telegram_message("❌ *Erreur* : Impossible de récupérer les données BTC.")
        return

    # =====================================================================
    # >>> AMÉLIORATION 2 (code) : calcul du régime BTC
    # =====================================================================
    btc_bull = None
    if USE_BTC_REGIME and len(df_btc) > BTC_SMA_PERIOD:
        sma_btc = df_btc['close'].rolling(BTC_SMA_PERIOD).mean().iloc[-1]
        if not pd.isna(sma_btc):
            btc_bull = bool(df_btc['close'].iloc[-1] > sma_btc)

    dynamic_alts = get_dynamic_altcoins(exchange, top_n=100)
    if not dynamic_alts:
        send_telegram_message("❌ *Erreur* : Impossible de récupérer la liste des altcoins.")
        return

    setups_valides = []
    preparations = []
    surveillances = []
    bloques_regime = []   # >>> nouveau : setups existants mais bloqués par le filtre BTC

    for symbol in dynamic_alts:

        # >>> AMÉLIORATION 3 (code) : on saute les paires non rentables
        if symbol in SKIP_PAIRS:
            continue

        df_alt = fetch_ohlcv(exchange, symbol, TIMEFRAME_ANALYSE)
        if df_alt is None: continue

        try:
            ticker = exchange.fetch_ticker(symbol)
            if ticker['quoteVolume'] < MIN_VOLUME_24H_USDT: continue
        except Exception: continue

        df = pd.merge(df_alt, df_btc, on='timestamp', suffixes=('_alt', '_btc'))
        if len(df) < 50: continue

        df.rename(columns={
            'open_alt': 'open', 'high_alt': 'high', 'low_alt': 'low',
            'close_alt': 'close', 'volume_alt': 'volume'
        }, inplace=True)

        if symbol == 'BTC/USDT':
            df['RS'] = df['close']
        else:
            df['RS'] = df['close'] / df['close_btc']

        pivot_data = detect_pivots(df, window=PIVOT_WINDOW)
        signal = analyze_channel(df, pivot_data)

        if signal:
            if signal['status'] == 'PREPARATION':
                dist = abs(df['RS'].iloc[-1] - signal['level']) / df['RS'].iloc[-1]
                if dist < TOLERANCE_RETEST:
                    # >>> AMÉLIORATION 2 (code) : filtre de régime sur les setups VALIDÉS
                    # (LONG seulement si BTC haussier, SHORT seulement si BTC baissier)
                    if USE_BTC_REGIME and btc_bull is not None:
                        contre_régime = (
                            (signal['dir'] == 'LONG' and not btc_bull) or
                            (signal['dir'] == 'SHORT' and btc_bull)
                        )
                        if contre_régime:
                            bloques_regime.append({'symbol': symbol, 'dir': signal['dir']})
                            time.sleep(0.5)
                            continue

                    # >>> AMÉLIORATION 4 (code) : mode SHORT only
                    if SHORT_ONLY and signal['dir'] != 'SHORT':
                        bloques_regime.append({'symbol': symbol,
                                               'dir': signal['dir'] + " (ignoré SHORT_ONLY)"})
                        time.sleep(0.5)
                        continue

                    entry, sl, tp1, tp2 = calculate_levels(df, signal, pivot_data)
                    setups_valides.append({
                        'symbol': symbol, 'dir': signal['dir'], 'entry': entry,
                        'sl': sl, 'tp1': tp1, 'tp2': tp2
                    })
                    # >>> MAX OPTIM (code) : suivi H48 (pas de reset si déjà suivi)
                    if TIME_STOP_HOURS_4H > 0 and symbol not in state:
                        state[symbol] = {'dir': signal['dir'], 'entry': entry, 'sl': sl,
                                         'tp1': tp1, 'tp2': tp2,
                                         'ts_open': int(time.time() * 1000)}
                else:
                    is_btc = symbol == 'BTC/USDT'
                    price_level = signal['level'] if is_btc else signal['level'] * df['close_btc'].iloc[-1]
                    preparations.append({'symbol': symbol, 'dir': signal['dir'], 'level': price_level})
            elif signal['status'] == 'SURVEILLANCE':
                is_btc = symbol == 'BTC/USDT'
                price_level = signal['level'] if is_btc else signal['level'] * df['close_btc'].iloc[-1]
                surveillances.append({'symbol': symbol, 'dir': signal['dir'], 'level': price_level})

        time.sleep(0.5)

    # Ligne régime BTC dans le rapport (>>> AMÉLIORATION 2)
    if btc_bull is None:
        regime_ligne = "🧭 *BTC : régime indisponible (données insuffisantes) — filtre désactivé ce tour*"
    elif btc_bull:
        regime_ligne = "🧭 *BTC : HAUSSIER (close > SMA200 4H) → SHORT bloqués*"
    else:
        regime_ligne = "🧭 *BTC : BAISSIER (close < SMA200 4H) → LONG bloqués*"

    report = f"📊 *RAPPORT SWING (4H) ({datetime.now().strftime('%H:%M')})*\n"
    report += regime_ligne + "\n"
    report += f"🔍 _Analyse sur {len(dynamic_alts)} altcoins_\n"
    report += "🎯 _Gérer à TP1 (3R) — LE régime validé (pas de partiel, pas de BE)_\n"
    report += "🛑 _2 pertes consécutives = stop de la session (ta règle, les 2 comptes)_\n"
    report += f"💰 _Sizing : {RISK_PCT_4H:g}% de {CAPITAL_4H:,.0f}$ = {RISK_USD_4H:,.0f}$ de risque par trade (taille dans chaque setup)_\n\n"

    if setups_valides:
        report += "🚀 *SETUPS VALIDÉS (Prêts à trader)*\n"
        for s in setups_valides:
            dist_sl = abs(s['entry'] - s['sl'])
            taille = RISK_USD_4H / dist_sl if dist_sl > 0 else 0.0
            report += f"• *{s['symbol']}* ({s['dir']})\n"
            report += f"  Entrée : {fmt_price(s['entry'])} | SL : {fmt_price(s['sl'])}\n"
            report += f"  TP1 : {fmt_price(s['tp1'])} | TP2 : {fmt_price(s['tp2'])}\n"
            report += f"  💰 {RISK_USD_4H:,.0f}$ de risque -> ~{fmt_qty(taille)} coins (position ~{taille*s['entry']:,.0f}$)\n\n"
    else:
        report += "🚀 *SETUPS VALIDÉS* : Aucun pour le moment.\n\n"

    if preparations:
        report += "⏳ *EN PRÉPARATION (Attente retest)*\n"
        for p in preparations:
            report += f"• *{p['symbol']}* ({p['dir']}) - Niveau : {fmt_price(p['level'])}\n"
        report += "\n"

    if surveillances:
        report += "👀 *SOUS SURVEILLANCE (Proche cassure)*\n"
        for s in surveillances:
            report += f"• *{s['symbol']}* ({s['dir']}) - Ligne : {fmt_price(s['level'])}\n"
        report += "\n"

    # >>> nouveau : transparence sur ce que le filtre a bloqué
    if bloques_regime:
        report += "🚫 *BLOQUÉS PAR LE RÉGIME BTC (ne pas trader)*\n"
        for b in bloques_regime:
            report += f"• *{b['symbol']}* ({b['dir']})\n"
        report += "\n"

    if not (setups_valides or preparations or surveillances or bloques_regime):
        report += "😴 *Marché calme.* Aucun canal intéressant détecté pour le moment."

    if h48_alerts:
        report += "\n⏰ *TIME STOP 8 JOURS — FERMER AU MARCHÉ si encore en position*\n"
        for sym, st, age_h in h48_alerts:
            report += f"• *{sym}* ({st['dir']}) — ouvert {age_h/24:.0f}j, ni TP1 ni SL touchés\n"

    send_telegram_message(report)
    print("Rapport envoyé avec succès.")
    _save_state(state)


if __name__ == "__main__":
    run_bot()
