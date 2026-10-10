import os
import time
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# CONFIGURATION (Ne jamais mettre les clés ici)
# ==========================================
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Paramètres de trading
TIMEFRAME_ANALYSE = '4h'
TIMEFRAME_ENTREE = '15m'
MIN_VOLUME_24H_USDT = 10_000_000  # 10 Millions de dollars
MAX_SIGNALS_PAR_JOUR = 3
RISK_REWARD_RATIO = 2.0  # Ratio Risque/Récompense (1:2)
PIVOT_WINDOW = 5  # Sensibilité des pivots

# Liste des altcoins à scanner (Modifie cette liste selon tes préférences)
# Tu peux aussi la récupérer dynamiquement via l'API de Binance
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
    """Envoie un message sur Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Erreur : Les identifiants Telegram ne sont pas configurés.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Erreur lors de l'envoi Telegram : {e}")

def fetch_ohlcv(exchange, symbol, timeframe, limit=200):
    """Récupère les données OHLCV."""
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Erreur de récupération pour {symbol} : {e}")
        return None

def detect_pivots(df, window=5):
    """Détecte les pivots hauts et bas."""
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

def check_wave_theory_module9(df, pivot_data):
    """
    Applique la logique du Module 9 (Canaux et Cassures) sur la Force Relative.
    """
    if pivot_data is None:
        return None
        
    pH1, pH2, pH1b, pH2b = pivot_data['pH1'], pivot_data['pH2'], pivot_data['pH1b'], pivot_data['pH2b']
    pL1, pL2, pL1b, pL2b = pivot_data['pL1'], pivot_data['pL2'], pivot_data['pL1b'], pivot_data['pL2b']
    
    current_idx = len(df) - 1
    current_close = df['RS'].iloc[-1]
    prev_close = df['RS'].iloc[-2]
    
    # Conditions de canal (comme dans le Pine Script)
    bullChan = (pL1 > pL2) and (pH1 > pH2)
    bearChan = (pL1 < pL2) and (pH1 < pH2)
    
    if not (bullChan or bearChan):
        return None
        
    # Calcul des pentes et rails
    if bullChan:
        slopeB = (pL1 - pL2) / (pL1b - pL2b) if (pL1b - pL2b) != 0 else 0
        supNow = pL1 + slopeB * (current_idx - pL1b)
        resNow = pH1 + slopeB * (current_idx - pH1b)
        chanDir = 1
    else: # bearChan
        slopeS = (pH1 - pH2) / (pH1b - pH2b) if (pH1b - pH2b) != 0 else 0
        resNow = pH1 + slopeS * (current_idx - pH1b)
        supNow = pL1 + slopeS * (current_idx - pL1b)
        chanDir = -1
        
    # Détection des signaux (Crossovers)
    brkUp = (chanDir == -1) and (current_close > resNow) and (prev_close <= resNow)
    brkDown = (chanDir == 1) and (current_close < supNow) and (prev_close >= supNow)
    outUp = (chanDir == 1) and (current_close > resNow) and (prev_close <= resNow)
    outDn = (chanDir == -1) and (current_close < supNow) and (prev_close >= supNow)
    
    if brkUp:
        return {"signal": "LONG", "type": "Cassure Baissière (Breakout)", "sup": supNow, "res": resNow}
    elif brkDown:
        return {"signal": "SHORT", "type": "Cassure Haussière (Breakdown)", "sup": supNow, "res": resNow}
    elif outUp:
        return {"signal": "LONG", "type": "Accélération Haussière", "sup": supNow, "res": resNow}
    elif outDn:
        return {"signal": "SHORT", "type": "Accélération Baissière", "sup": supNow, "res": resNow}
    
    return None

# ==========================================
# LOGIQUE PRINCIPALE
# ==========================================

def run_bot():
    print(f"Démarrage du bot - {datetime.now()}")
    exchange = ccxt.binance()
    
    # 1. Récupérer les données BTC
    df_btc = fetch_ohlcv(exchange, 'BTC/USDT', TIMEFRAME_ANALYSE)
    if df_btc is None:
        print("Impossible de récupérer les données BTC. Arrêt.")
        return

    signals_envoyes = 0
    
    for symbol in ALTCOINS:
        if signals_envoyes >= MAX_SIGNALS_PAR_JOUR:
            print("Limite de signaux atteinte pour aujourd'hui.")
            break
            
        print(f"Analyse de {symbol}...")
        
        # 2. Récupérer les données de l'Alt
        df_alt = fetch_ohlcv(exchange, symbol, TIMEFRAME_ANALYSE)
        if df_alt is None:
            continue
            
        # 3. Vérifier le volume 24h
        try:
            ticker = exchange.fetch_ticker(symbol)
            volume_24h = ticker['quoteVolume']
            if volume_24h < MIN_VOLUME_24H_USDT:
                print(f"{symbol} ignoré : Volume 24h trop faible ({volume_24h:,.0f} USDT)")
                continue
        except Exception as e:
            print(f"Impossible de récupérer le ticker pour {symbol} : {e}")
            continue

        # 4. Fusionner les données et calculer la Force Relative (RS)
        df = pd.merge(df_alt, df_btc, on='timestamp', suffixes=('_alt', '_btc'))
        df['RS'] = df['close_alt'] / df['close_btc']
        
        # 5. Détection des pivots sur la ligne RS
        pivot_data = detect_pivots(df, window=PIVOT_WINDOW)
        
        # 6. Appliquer le Module 9
        signal = check_wave_theory_module9(df, pivot_data)
        
        if signal:
            # 7. Calcul de l'Entrée, SL et TP
            entry_price = df['close_alt'].iloc[-1]
            atr = df['high_alt'].sub(df['low_alt']).rolling(14).mean().iloc[-1]
            
            if signal['signal'] == 'LONG':
                # SL sous le dernier pivot bas ou sous le support du canal
                sl_price = min(pivot_data['pL1'], signal['sup']) - (atr * 0.5)
                risk = entry_price - sl_price
                tp1 = entry_price + (risk * RISK_REWARD_RATIO)
                tp2 = entry_price + (risk * RISK_REWARD_RATIO * 2)
            else: # SHORT
                # SL au-dessus du dernier pivot haut ou au-dessus de la résistance du canal
                sl_price = max(pivot_data['pH1'], signal['res']) + (atr * 0.5)
                risk = sl_price - entry_price
                tp1 = entry_price - (risk * RISK_REWARD_RATIO)
                tp2 = entry_price - (risk * RISK_REWARD_RATIO * 2)
                
            # 8. Formater le message Telegram
            message = f"""
🚀 *SIGNAL ALTCOIN (Force Relative vs BTC)*

📊 *Actif* : {symbol}
📈 *Direction* : {signal['signal']}
⚡ *Type* : {signal['type']}

📍 *Entrée* : {entry_price:.4f}
🛑 *Stop Loss* : {sl_price:.4f} ({(abs(entry_price-sl_price)/entry_price)*100:.2f}%)
🎯 *TP1* : {tp1:.4f} (+{((tp1-entry_price)/entry_price)*100:.2f}%)
🎯 *TP2* : {tp2:.4f} (+{((tp2-entry_price)/entry_price)*100:.2f}%)

📉 *Contexte* : Force Relative vs BTC positive sur 4H. Cassure détectée.
            """
            
            send_telegram_message(message)
            print(f"Signal envoyé pour {symbol} !")
            signals_envoyes += 1
            
        time.sleep(1) # Éviter le rate limit de l'API

if __name__ == "__main__":
    run_bot()
