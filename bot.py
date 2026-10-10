import os
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

# Paramètres de trading
TIMEFRAME_ANALYSE = '4h'
MIN_VOLUME_24H_USDT = 10_000_000
RISK_REWARD_RATIO = 2.0
PIVOT_WINDOW = 5
TOLERANCE_RETEST = 0.015  # 1.5% de tolérance pour considérer qu'on est sur un retest

ALTCOINS = [
    'AERO/USDT', 'LIT/USDT', 'WLD/USDT', 'ZRO/USDT', 'APT/USDT',
    'NEAR/USDT', 'INJ/USDT', 'SUI/USDT', 'ARB/USDT', 'OP/USDT',
    'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'MATIC/USDT', 'ATOM/USDT',
    'FIL/USDT', 'ETC/USDT', 'DOGE/USDT', 'SHIB/USDT', 'TON/USDT'
]

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
    df['pivot_high'] = df['high'].rolling(window=window, center=True).max() == df['high']
    df['pivot_low'] = df['low'].rolling(window=window, center=True).min() == df['low']
    highs = df[df['pivot_high']].tail(2)
    lows = df[df['pivot_low']].tail(2)
    if len(highs) < 2 or len(lows) < 2:
        return None
    return {
        'pH1': highs['high'].iloc[-1], 'pH2': highs['high'].iloc[-2],
        'pH1b': df.index.get_loc(highs.index[-1]), 'pH2b': df.index.get_loc(highs.index[-2]),
        'pL1': lows['low'].iloc[-1], 'pL2': lows['low'].iloc[-2],
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
        resNow = pH1 + slope * (current_idx - pH1b)
        # Vérifier si on est proche de la cassure (le prix s'approche du bas du canal)
        dist_to_sup = abs(current_close - supNow) / current_close
        if current_close < supNow: # Cassure baissière
            return {"dir": "SHORT", "type": "CASSÉ", "level": supNow, "status": "PREPARATION"}
        elif dist_to_sup < 0.02: # À 2% du support
            return {"dir": "SHORT", "type": "SURVEILLANCE", "level": supNow, "status": "SURVEILLANCE"}
            
    elif bearChan:
        slope = (pH1 - pH2) / (pH1b - pH2b) if (pH1b - pH2b) != 0 else 0
        resNow = pH1 + slope * (current_idx - pH1b)
        supNow = pL1 + slope * (current_idx - pL1b)
        # Vérifier si on est proche de la cassure (le prix s'approche du haut du canal)
        dist_to_res = abs(current_close - resNow) / current_close
        if current_close > resNow: # Cassure haussière
            return {"dir": "LONG", "type": "CASSÉ", "level": resNow, "status": "PREPARATION"}
        elif dist_to_res < 0.02: # À 2% de la résistance
            return {"dir": "LONG", "type": "SURVEILLANCE", "level": resNow, "status": "SURVEILLANCE"}
            
    return None

def calculate_levels(df, signal, pivot_data):
    entry_price = df['close_alt'].iloc[-1]
    atr = df['high_alt'].sub(df['low_alt']).rolling(14).mean().iloc[-1]
    
    if signal['dir'] == 'LONG':
        sl_price = min(pivot_data['pL1'], signal['level']) - (atr * 0.5)
        risk = entry_price - sl_price
        tp1 = entry_price + (risk * RISK_REWARD_RATIO)
        tp2 = entry_price + (risk * RISK_REWARD_RATIO * 2)
    else:
        sl_price = max(pivot_data['pH1'], signal['level']) + (atr * 0.5)
        risk = sl_price - entry_price
        tp1 = entry_price - (risk * RISK_REWARD_RATIO)
        tp2 = entry_price - (risk * RISK_REWARD_RATIO * 2)
        
    return entry_price, sl_price, tp1, tp2

# ==========================================
# LOGIQUE PRINCIPALE
# ==========================================

def run_bot():
    print(f"Démarrage du bot - {datetime.now()}")
    exchange = ccxt.mexc()
    
    df_btc = fetch_ohlcv(exchange, 'BTC/USDT', TIMEFRAME_ANALYSE)
    if df_btc is None:
        send_telegram_message("❌ *Erreur* : Impossible de récupérer les données BTC.")
        return

    setups_valides = []
    preparations = []
    surveillances = []
    
    for symbol in ALTCOINS:
        df_alt = fetch_ohlcv(exchange, symbol, TIMEFRAME_ANALYSE)
        if df_alt is None: continue
            
        try:
            ticker = exchange.fetch_ticker(symbol)
            if ticker['quoteVolume'] < MIN_VOLUME_24H_USDT: continue
        except Exception: continue

        df = pd.merge(df_alt, df_btc, on='timestamp', suffixes=('_alt', '_btc'))
        df['RS'] = df['close_alt'] / df['close_btc']
        
        pivot_data = detect_pivots(df, window=PIVOT_WINDOW)
        signal = analyze_channel(df, pivot_data)
        
        if signal:
            if signal['status'] == 'PREPARATION':
                # On vérifie si le prix est revenu tester la ligne (Retest)
                # Pour simplifier, on considère que si le prix est à moins de 1.5% de la ligne, c'est un retest
                dist = abs(df['close_alt'].iloc[-1] - signal['level']) / df['close_alt'].iloc[-1]
                if dist < TOLERANCE_RETEST:
                    entry, sl, tp1, tp2 = calculate_levels(df, signal, pivot_data)
                    setups_valides.append({
                        'symbol': symbol, 'dir': signal['dir'], 'entry': entry, 
                        'sl': sl, 'tp1': tp1, 'tp2': tp2, 'level': signal['level']
                    })
                else:
                    preparations.append({'symbol': symbol, 'dir': signal['dir'], 'level': signal['level']})
            elif signal['status'] == 'SURVEILLANCE':
                surveillances.append({'symbol': symbol, 'dir': signal['dir'], 'level': signal['level']})
                
        time.sleep(1)

    # --- CONSTRUCTION DU RAPPORT TELEGRAM ---
    report = f"📊 *RAPPORT DE MARCHÉ ({datetime.now().strftime('%H:%M')})*\n\n"
    
    if setups_valides:
        report += "🚀 *SETUPS VALIDÉS (Prêts à trader)*\n"
        for s in setups_valides:
            report += f"• *{s['symbol']}* ({s['dir']})\n"
            report += f"  Entrée : {s['entry']:.4f} | SL : {s['sl']:.4f}\n"
            report += f"  TP1 : {s['tp1']:.4f} | TP2 : {s['tp2']:.4f}\n"
        report += "\n"
    else:
        report += "🚀 *SETUPS VALIDÉS* : Aucun pour le moment.\n\n"

    if preparations:
        report += "⏳ *EN PRÉPARATION (Attente retest)*\n"
        for p in preparations:
            report += f"• *{p['symbol']}* ({p['dir']}) - Niveau à surveiller : {p['level']:.4f}\n"
        report += "\n"
        
    if surveillances:
        report += "👀 *SOUS SURVEILLANCE (Proche cassure)*\n"
        for s in surveillances:
            report += f"• *{s['symbol']}* ({s['dir']}) - Ligne : {s['level']:.4f}\n"
        report += "\n"

    if not (setups_valides or preparations or surveillances):
        report += "😴 *Marché calme.* Aucun canal intéressant détecté pour le moment."

    send_telegram_message(report)
    print("Rapport envoyé.")

if __name__ == "__main__":
    run_bot()
