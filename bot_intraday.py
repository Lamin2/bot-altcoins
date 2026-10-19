import os
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
    # ⚠️ CORRECTION : on calcule les pivots sur RS
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
    
    # ⚠️ CORRECTION : Conversion du niveau RS en prix réel
    # BTC : RS = prix (egal a close). Alts : RS = ratio. Test robuste, independant des 2 telechargements de BTC.
    is_btc = df['RS'].iloc[-1] == df['close'].iloc[-1]
    btc_price = df['close_btc'].iloc[-1] if not is_btc else 1
    
    level_price = signal['level'] * btc_price if not is_btc else signal['level']
    pL1_price = pivot_data['pL1'] * btc_price if not is_btc else pivot_data['pL1']
    pH1_price = pivot_data['pH1'] * btc_price if not is_btc else pivot_data['pH1']
    
    if signal['dir'] == 'LONG':
        sl_price = min(pL1_price, level_price) - (atr * 0.5)
        risk = entry_price - sl_price
        tp1 = entry_price + (risk * rr_ratio)
        tp2 = entry_price + (risk * rr_ratio * 2)
    else:
        sl_price = max(pH1_price, level_price) + (atr * 0.5)
        risk = sl_price - entry_price
        tp1 = entry_price - (risk * rr_ratio)
        tp2 = entry_price - (risk * rr_ratio * 2)
    return entry_price, sl_price, tp1, tp2

# ==========================================
# LOGIQUE PRINCIPALE
# ==========================================

def run_bot():
    session = get_current_session()
    print(f"BOT INTRADAY - Session {session['name']} - {datetime.now()}")
    
    send_telegram_message(
        f"{session['emoji']} *BOT INTRADAY - Session {session['name']}*\n"
        f"_{session['desc']}_\n"
        f"⚙️ RR: 1:{session['rr']} | Vol min: {session['vol']//1_000_000}M$ | Max: {session['max']}"
    )
    
    exchange = ccxt.mexc()
    df_btc = fetch_ohlcv(exchange, 'BTC/USDT', TIMEFRAME_ANALYSE)
    if df_btc is None:
        send_telegram_message("❌ *[INTRADAY]* Erreur BTC.")
        return
    
    dynamic_alts = get_dynamic_altcoins(exchange, top_n=80)
    if not dynamic_alts:
        send_telegram_message("❌ *[INTRADAY]* Erreur liste altcoins.")
        return
    
    setups_valides = []
    preparations = []
    surveillances = []
    
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
                # ⚠️ CORRECTION : comparer RS avec RS
                dist = abs(df['RS'].iloc[-1] - signal['level']) / df['RS'].iloc[-1]
                if dist < session['tol']:
                    entry, sl, tp1, tp2 = calculate_levels(df, signal, pivot_data, session['rr'])
                    setups_valides.append({'symbol': symbol, 'dir': signal['dir'],
                                           'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2})
                else:
                    # ⚠️ CORRECTION : afficher le prix converti
                    is_btc = symbol == 'BTC/USDT'
                    price_level = signal['level'] if is_btc else signal['level'] * df['close_btc'].iloc[-1]
                    preparations.append({'symbol': symbol, 'dir': signal['dir'], 'level': price_level})
            elif signal['status'] == 'SURVEILLANCE':
                is_btc = symbol == 'BTC/USDT'
                price_level = signal['level'] if is_btc else signal['level'] * df['close_btc'].iloc[-1]
                surveillances.append({'symbol': symbol, 'dir': signal['dir'], 'level': price_level})
        time.sleep(0.5)
    
    report = f"⚡ *RAPPORT INTRADAY (1H) - {session['emoji']} {session['name']}*\n"
    report += f"🕐 {datetime.now().strftime('%H:%M')} | 🔍 _{len(dynamic_alts)} altcoins_\n\n"
    
    if setups_valides:
        report += f"🚀 *SETUPS VALIDÉS (max {session['max']})*\n"
        for s in setups_valides[:session['max']]:
            report += f"• *{s['symbol']}* ({s['dir']})\n"
            report += f"  Entrée : {s['entry']:.4f} | SL : {s['sl']:.4f}\n"
            report += f"  TP1 : {s['tp1']:.4f} | TP2 : {s['tp2']:.4f}\n\n"
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
    
    send_telegram_message(report)
    print("Rapport envoyé.")

if __name__ == "__main__":
    run_bot()