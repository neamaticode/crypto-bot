# -*- coding: utf-8 -*-
"""
VIP Crypto Signal Bot v3.0 — Ultra Sniper Edition
===================================================
Author : VIP Bot
License: MIT

Single-file professional cryptocurrency futures signal bot featuring:
  • Ultra-strict "Sniper" signal logic (High Win-Rate, low SL frequency)
  • Multi-Timeframe (MTF) alignment across 15m / 1h / 4h / 1d
  • ADX > 25 trend-strength gate + clear MACD momentum
  • Order Block (OB) + Fair Value Gap (FVG) confluence required
  • Confidence threshold >= 91% before emitting any signal
  • Wide ATR-based SL (2.5x ATR) to avoid premature stop-outs
  • Telegram reply system: TP/SL messages reply to the original signal
  • Daily + Weekly performance reports
  • SQLite persistence layer
"""

# ============================================================
# CONFIGURATION
# ============================================================
import os as _os

TELEGRAM_TOKEN          = _os.environ.get('TELEGRAM_TOKEN',   '')
TELEGRAM_CHAT_ID        = _os.environ.get('TELEGRAM_CHAT_ID', '')
DB_NAME                 = 'crypto_bot_data.db'
INITIAL_ACCOUNT_BALANCE = 5000.0
LEVERAGE                = 20

API_CONFIG = {
    'apiKey'     : _os.environ.get('BINANCE_API_KEY', ''),
    'secret'     : _os.environ.get('BINANCE_SECRET',  ''),
    'defaultType': 'future',
    'options'    : {'defaultType': 'future'},
}
del _os

# Runtime limits
MIN_CONFIDENCE  = 91     # >= 91% confidence required before emitting a signal
MIN_AGREEMENTS  = 7      # >= 7 independent indicator categories must agree
MAX_OPEN_TRADES = 5      # maximum simultaneous open positions
SIGNAL_COOLDOWN = 360    # minutes between signals on the same symbol
SCAN_INTERVAL   = 120    # seconds between main-loop cycles

# ============================================================
# IMPORTS
# ============================================================
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytz
from sklearn.ensemble import (GradientBoostingClassifier,
                               RandomForestClassifier,
                               VotingClassifier)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

# Guard for division-by-zero
EPSILON = 1e-10

# ============================================================
# LOGGING
# ============================================================
def _setup_logging() -> logging.Logger:
    fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('crypto_bot.log', encoding='utf-8'),
        ],
    )
    for noisy in ('ccxt', 'telegram', 'httpx', 'hpack', 'urllib3'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger('VIPBot')


logger = _setup_logging()

# UTC+4:30 — Afghanistan timezone
AFG_TZ = pytz.timezone('Asia/Kabul')


# ============================================================
# DATABASE MANAGER
# ============================================================
class DatabaseManager:
    """All aiosqlite persistence (async-safe)."""

    _SCHEMA = '''
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT    NOT NULL,
            direction       TEXT    NOT NULL,
            entry           REAL    NOT NULL,
            sl              REAL    NOT NULL,
            original_sl     REAL    NOT NULL,
            tp1             REAL    NOT NULL,
            tp2             REAL    NOT NULL,
            tp3             REAL    NOT NULL,
            position_size   REAL    NOT NULL,
            status          TEXT    DEFAULT 'ACTIVE',
            pnl_percent     REAL    DEFAULT 0.0,
            pnl_usdt        REAL    DEFAULT 0.0,
            telegram_msg_id INTEGER,
            created_at      TEXT    NOT NULL,
            closed_at       TEXT,
            breakeven_moved INTEGER DEFAULT 0,
            trailing_active INTEGER DEFAULT 0,
            trailing_sl     REAL,
            exit_price      REAL,
            confidence      REAL    NOT NULL,
            reasons         TEXT,
            features        TEXT,
            tp1_hit         INTEGER DEFAULT 0,
            tp2_hit         INTEGER DEFAULT 0,
            tp3_hit         INTEGER DEFAULT 0,
            max_pnl_reached REAL    DEFAULT 0.0,
            session_type    TEXT,
            market_regime   TEXT,
            leverage        INTEGER DEFAULT 20,
            funding_rate    REAL    DEFAULT 0.0,
            atr             REAL    DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS ai_data (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol            TEXT,
            rsi               REAL,
            macd              REAL,
            macd_hist         REAL,
            obv               REAL,
            atr               REAL,
            bb_width          REAL,
            bb_position       REAL,
            ema_alignment     REAL,
            volume_ratio      REAL,
            order_block_score REAL,
            fvg_score         REAL,
            liquidity_score   REAL,
            mtf_trend_15m     REAL,
            mtf_trend_4h      REAL,
            mtf_trend_1d      REAL,
            funding_rate      REAL,
            target            INTEGER,
            created_at        TEXT
        );

        CREATE TABLE IF NOT EXISTS reports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type  TEXT    NOT NULL,
            period_start TEXT,
            period_end   TEXT,
            content      TEXT,
            created_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS bot_state (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at TEXT
        );
    '''

    def __init__(self, db_name: str = DB_NAME) -> None:
        self.db_name = db_name

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_name) as db:
            await db.executescript(self._SCHEMA)
            await db.commit()
            logger.info('Database initialised OK')

    # signals
    async def save_signal(self, sig: Dict) -> int:
        try:
            async with aiosqlite.connect(self.db_name) as db:
                cur = await db.execute(
                    '''INSERT INTO signals
                       (symbol, direction, entry, sl, original_sl, tp1, tp2, tp3,
                        position_size, confidence, reasons, features, session_type,
                        market_regime, leverage, funding_rate, atr, created_at, status)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (
                        sig['symbol'], sig['direction'], sig['entry'],
                        sig['sl'], sig['sl'], sig['tp1'], sig['tp2'], sig['tp3'],
                        sig['position_size'], sig['confidence'],
                        json.dumps(sig.get('reasons', [])),
                        json.dumps(sig.get('features', {})),
                        sig.get('session_type', ''), sig.get('market_regime', ''),
                        LEVERAGE, sig.get('funding_rate', 0.0), sig.get('atr', 0.0),
                        datetime.now(AFG_TZ).isoformat(), 'ACTIVE',
                    ),
                )
                await db.commit()
                return cur.lastrowid
        except Exception as exc:
            logger.error('save_signal error: %s', exc)
            return -1

    async def update_signal(self, signal_id: int, updates: Dict) -> None:
        if not updates:
            return
        try:
            async with aiosqlite.connect(self.db_name) as db:
                clause = ', '.join(f'{k}=?' for k in updates)
                vals   = list(updates.values()) + [signal_id]
                await db.execute(f'UPDATE signals SET {clause} WHERE id=?', vals)
                await db.commit()
        except Exception as exc:
            logger.error('update_signal error: %s', exc)

    async def get_active_signals(self) -> List[Dict]:
        try:
            async with aiosqlite.connect(self.db_name) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM signals WHERE status='ACTIVE'"
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(r) for r in rows]
        except Exception as exc:
            logger.error('get_active_signals error: %s', exc)
            return []

    async def get_signals_by_period(self, start: datetime, end: datetime) -> List[Dict]:
        try:
            async with aiosqlite.connect(self.db_name) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM signals WHERE created_at>=? AND created_at<? "
                    "AND status!='ACTIVE'",
                    (start.isoformat(), end.isoformat()),
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(r) for r in rows]
        except Exception as exc:
            logger.error('get_signals_by_period error: %s', exc)
            return []

    # AI data
    async def save_ai_data(self, features: Dict, target: int) -> None:
        try:
            async with aiosqlite.connect(self.db_name) as db:
                await db.execute(
                    '''INSERT INTO ai_data
                       (symbol, rsi, macd, macd_hist, obv, atr, bb_width,
                        bb_position, ema_alignment, volume_ratio,
                        order_block_score, fvg_score, liquidity_score,
                        mtf_trend_15m, mtf_trend_4h, mtf_trend_1d,
                        funding_rate, target, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (
                        features.get('symbol', ''),
                        features.get('rsi', 50),       features.get('macd', 0),
                        features.get('macd_hist', 0),  features.get('obv', 0),
                        features.get('atr', 0),        features.get('bb_width', 0),
                        features.get('bb_position', 0.5),
                        features.get('ema_alignment', 0),
                        features.get('volume_ratio', 1),
                        features.get('order_block_score', 0),
                        features.get('fvg_score', 0),
                        features.get('liquidity_score', 0),
                        features.get('mtf_trend_15m', 0),
                        features.get('mtf_trend_4h', 0),
                        features.get('mtf_trend_1d', 0),
                        features.get('funding_rate', 0),
                        target, datetime.now(AFG_TZ).isoformat(),
                    ),
                )
                await db.commit()
        except Exception as exc:
            logger.error('save_ai_data error: %s', exc)

    async def get_ai_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            async with aiosqlite.connect(self.db_name) as db:
                async with db.execute(
                    '''SELECT rsi, macd, macd_hist, obv, atr, bb_width,
                              bb_position, ema_alignment, volume_ratio,
                              order_block_score, fvg_score, liquidity_score,
                              mtf_trend_15m, mtf_trend_4h, mtf_trend_1d,
                              funding_rate, target
                       FROM ai_data'''
                ) as cursor:
                    rows = await cursor.fetchall()
            if len(rows) < 50:
                return None, None
            X = np.array([[r[i] for i in range(16)] for r in rows], dtype=float)
            y = np.array([r[16] for r in rows], dtype=int)
            return X, y
        except Exception as exc:
            logger.error('get_ai_training_data error: %s', exc)
            return None, None

    # bot state
    async def set_state(self, key: str, value: Any) -> None:
        try:
            async with aiosqlite.connect(self.db_name) as db:
                await db.execute(
                    'INSERT OR REPLACE INTO bot_state (key,value,updated_at) VALUES (?,?,?)',
                    (key, json.dumps(value), datetime.now(AFG_TZ).isoformat()),
                )
                await db.commit()
        except Exception as exc:
            logger.error('set_state error: %s', exc)

    async def get_state(self, key: str, default: Any = None) -> Any:
        try:
            async with aiosqlite.connect(self.db_name) as db:
                async with db.execute(
                    'SELECT value FROM bot_state WHERE key=?', (key,)
                ) as cursor:
                    row = await cursor.fetchone()
            return json.loads(row[0]) if row else default
        except Exception:
            return default


# ============================================================
# TECHNICAL ANALYSIS ENGINE
# ============================================================
class TechnicalAnalysisEngine:
    """Computes 16+ technical indicators."""

    @staticmethod
    def compute_all(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or len(df) < 200:
            return None
        try:
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]

            # RSI
            rsi = ta.rsi(df['close'], length=14)
            df['rsi'] = rsi.fillna(50) if rsi is not None else 50.0

            # MACD
            macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                cols = macd_df.columns.tolist()
                df['macd']        = macd_df[cols[0]].fillna(0)
                df['macd_signal'] = macd_df[cols[2]].fillna(0)
                df['macd_hist']   = macd_df[cols[1]].fillna(0)
            else:
                df['macd'] = df['macd_signal'] = df['macd_hist'] = 0.0

            # Bollinger Bands
            bb = ta.bbands(df['close'], length=20, std=2)
            if bb is not None and not bb.empty:
                bc = bb.columns.tolist()
                df['bb_lower'] = bb[bc[0]].fillna(df['close'])
                df['bb_mid']   = bb[bc[1]].fillna(df['close'])
                df['bb_upper'] = bb[bc[2]].fillna(df['close'])
            else:
                df['bb_lower'] = df['bb_mid'] = df['bb_upper'] = df['close']
            span = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
            df['bb_width']    = (span / df['bb_mid'].replace(0, np.nan)).fillna(0)
            df['bb_position'] = ((df['close'] - df['bb_lower']) / span).fillna(0.5).clip(0, 1)

            # EMAs
            for p in [9, 21, 50, 100, 200]:
                e = ta.ema(df['close'], length=p)
                df[f'ema{p}'] = e.fillna(df['close']) if e is not None else df['close']

            # EMA alignment score  -4 ... +4
            ema_align = 0
            periods = [9, 21, 50, 100, 200]
            for i in range(len(periods) - 1):
                v1 = df[f'ema{periods[i]}'].iloc[-1]
                v2 = df[f'ema{periods[i+1]}'].iloc[-1]
                if not (np.isnan(v1) or np.isnan(v2)):
                    ema_align += (1 if v1 > v2 else -1)
            df['ema_alignment'] = ema_align

            # ATR
            atr = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr'] = atr.fillna(df['high'] - df['low']) if atr is not None \
                        else (df['high'] - df['low'])

            # ADX
            adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_df is not None and not adx_df.empty:
                ac = adx_df.columns.tolist()
                df['adx'] = adx_df[ac[0]].fillna(20)
                df['dmp'] = adx_df[ac[1]].fillna(20)
                df['dmn'] = adx_df[ac[2]].fillna(20)
            else:
                df['adx'] = df['dmp'] = df['dmn'] = 20.0

            # Stochastic RSI
            srsi = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
            if srsi is not None and not srsi.empty:
                sc = srsi.columns.tolist()
                df['stochrsi_k'] = srsi[sc[0]].fillna(50)
                df['stochrsi_d'] = srsi[sc[1]].fillna(50)
            else:
                df['stochrsi_k'] = df['stochrsi_d'] = 50.0

            # OBV
            obv = ta.obv(df['close'], df['volume'])
            df['obv'] = obv.fillna(0) if obv is not None else 0.0
            obv_sma = df['obv'].rolling(20, min_periods=1).mean().replace(0, np.nan)
            df['obv_signal'] = ((df['obv'] - obv_sma) / obv_sma.abs()).fillna(0)

            # Volume ratio
            vol_sma = df['volume'].rolling(20, min_periods=1).mean().replace(0, np.nan)
            df['volume_ratio'] = (df['volume'] / vol_sma).fillna(1)

            # VWAP (session approximation)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / (df['volume'].cumsum() + EPSILON)

            # Ichimoku Cloud
            try:
                ich_result = ta.ichimoku(df['high'], df['low'], df['close'])
                ich = ich_result[0] if isinstance(ich_result, tuple) else ich_result
                if ich is not None and not ich.empty:
                    ic = ich.columns.tolist()
                    df['tenkan']   = ich[ic[0]].fillna(df['close']) if len(ic) > 0 else df['close']
                    df['kijun']    = ich[ic[1]].fillna(df['close']) if len(ic) > 1 else df['close']
                    df['senkou_a'] = ich[ic[2]].fillna(df['close']) if len(ic) > 2 else df['close']
                    df['senkou_b'] = ich[ic[3]].fillna(df['close']) if len(ic) > 3 else df['close']
                else:
                    df['tenkan'] = df['kijun'] = df['senkou_a'] = df['senkou_b'] = df['close']
            except Exception:
                df['tenkan'] = df['kijun'] = df['senkou_a'] = df['senkou_b'] = df['close']

            # Supertrend
            try:
                st = ta.supertrend(df['high'], df['low'], df['close'],
                                   length=10, multiplier=3.0)
                if st is not None and not st.empty:
                    dir_cols = [c for c in st.columns if 'SUPERTd' in c]
                    val_cols = [c for c in st.columns
                                if c.startswith('SUPERT_') and 'SUPERTd' not in c
                                and 'SUPERTl' not in c and 'SUPERTs' not in c]
                    df['supertrend_dir'] = st[dir_cols[0]].fillna(1) if dir_cols else 1
                    df['supertrend']     = st[val_cols[0]].fillna(df['close']) if val_cols else df['close']
                else:
                    df['supertrend_dir'] = 1
                    df['supertrend']     = df['close']
            except Exception:
                df['supertrend_dir'] = 1
                df['supertrend']     = df['close']

            df = df.ffill().bfill().fillna(0)
            return df

        except Exception as exc:
            logger.error('compute_all error: %s\n%s', exc, traceback.format_exc())
            return None

    @staticmethod
    def compute_fibonacci(df: pd.DataFrame, lookback: int = 100) -> Dict:
        try:
            recent = df.tail(lookback)
            hi = float(recent['high'].max())
            lo = float(recent['low'].min())
            diff = hi - lo
            return {
                'fib_0':      lo,
                'fib_236':    lo + 0.236 * diff,
                'fib_382':    lo + 0.382 * diff,
                'fib_500':    lo + 0.500 * diff,
                'fib_618':    lo + 0.618 * diff,
                'fib_786':    lo + 0.786 * diff,
                'fib_1':      hi,
                'swing_high': hi,
                'swing_low':  lo,
            }
        except Exception:
            return {}

    @staticmethod
    def compute_support_resistance(df: pd.DataFrame, lookback: int = 100) -> Dict:
        try:
            recent = df.tail(lookback).reset_index(drop=True)
            supports, resistances = [], []
            for i in range(2, len(recent) - 2):
                h = recent['high'].iloc[i]
                l = recent['low'].iloc[i]
                if (h > recent['high'].iloc[i-1] and h > recent['high'].iloc[i-2]
                        and h > recent['high'].iloc[i+1] and h > recent['high'].iloc[i+2]):
                    resistances.append(h)
                if (l < recent['low'].iloc[i-1] and l < recent['low'].iloc[i-2]
                        and l < recent['low'].iloc[i+1] and l < recent['low'].iloc[i+2]):
                    supports.append(l)
            close = float(recent['close'].iloc[-1])
            ns = max((s for s in supports    if s < close), default=close * 0.98)
            nr = min((r for r in resistances if r > close), default=close * 1.02)
            return {
                'nearest_support':    ns,
                'nearest_resistance': nr,
                'all_supports':       sorted(supports,    reverse=True)[:5],
                'all_resistances':    sorted(resistances)[:5],
            }
        except Exception:
            return {'nearest_support': 0, 'nearest_resistance': 0}


# ============================================================
# SMART MONEY CONCEPTS
# ============================================================
class SmartMoneyConcepts:

    @staticmethod
    def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> Dict:
        result = {'bullish_ob': None, 'bearish_ob': None,
                  'bull_ob_score': 0.0, 'bear_ob_score': 0.0}
        try:
            recent = df.tail(lookback).reset_index(drop=True)
            close  = float(recent['close'].iloc[-1])
            for i in range(1, len(recent) - 1):
                curr = recent.iloc[i]
                prev = recent.iloc[i - 1]
                # Bullish OB: bearish candle just before strong up-move
                if prev['close'] < prev['open']:
                    move = (curr['close'] - prev['close']) / (prev['close'] + EPSILON)
                    if move > 0.003 and prev['low'] <= close <= prev['high'] * 1.02:
                        result['bullish_ob']    = {'high': float(prev['high']), 'low': float(prev['low'])}
                        result['bull_ob_score'] = min(1.0, move * 100)
                # Bearish OB: bullish candle just before strong down-move
                if prev['close'] > prev['open']:
                    move = (prev['close'] - curr['close']) / (prev['close'] + EPSILON)
                    if move > 0.003 and prev['low'] * 0.98 <= close <= prev['high']:
                        result['bearish_ob']    = {'high': float(prev['high']), 'low': float(prev['low'])}
                        result['bear_ob_score'] = min(1.0, move * 100)
        except Exception as exc:
            logger.debug('detect_order_blocks: %s', exc)
        return result

    @staticmethod
    def detect_fair_value_gaps(df: pd.DataFrame, lookback: int = 50) -> Dict:
        result = {'bullish_fvg': [], 'bearish_fvg': [],
                  'bull_fvg_score': 0.0, 'bear_fvg_score': 0.0}
        try:
            recent = df.tail(lookback).reset_index(drop=True)
            close  = float(recent['close'].iloc[-1])
            for i in range(1, len(recent) - 1):
                prev = recent.iloc[i - 1]
                nxt  = recent.iloc[i + 1]
                # Bullish FVG
                if nxt['low'] > prev['high']:
                    gap = float(nxt['low'] - prev['high'])
                    result['bullish_fvg'].append({'high': float(nxt['low']), 'low': float(prev['high'])})
                    if float(prev['high']) <= close <= float(nxt['low']) * 1.01:
                        result['bull_fvg_score'] = max(result['bull_fvg_score'], gap / (close + EPSILON))
                # Bearish FVG
                if nxt['high'] < prev['low']:
                    gap = float(prev['low'] - nxt['high'])
                    result['bearish_fvg'].append({'high': float(prev['low']), 'low': float(nxt['high'])})
                    if float(nxt['high']) * 0.99 <= close <= float(prev['low']):
                        result['bear_fvg_score'] = max(result['bear_fvg_score'], gap / (close + EPSILON))
        except Exception as exc:
            logger.debug('detect_fair_value_gaps: %s', exc)
        return result

    @staticmethod
    def detect_liquidity_zones(df: pd.DataFrame, lookback: int = 100) -> Dict:
        result = {'buy_side_liq': [], 'sell_side_liq': [],
                  'bull_liq_score': 0.0, 'bear_liq_score': 0.0}
        try:
            recent = df.tail(lookback).reset_index(drop=True)
            close  = float(recent['close'].iloc[-1])
            sh, sl = [], []
            for i in range(3, len(recent) - 3):
                h = float(recent['high'].iloc[i])
                l = float(recent['low'].iloc[i])
                if (h > recent['high'].iloc[i-1] and h > recent['high'].iloc[i-2]
                        and h > recent['high'].iloc[i+1] and h > recent['high'].iloc[i+2]):
                    sh.append(h)
                if (l < recent['low'].iloc[i-1] and l < recent['low'].iloc[i-2]
                        and l < recent['low'].iloc[i+1] and l < recent['low'].iloc[i+2]):
                    sl.append(l)
            result['buy_side_liq']  = sorted(sh, reverse=True)[:5]
            result['sell_side_liq'] = sorted(sl)[:5]
            for v in sh:
                if close * 0.999 <= v <= close * 1.002:
                    result['bear_liq_score'] += 0.5
            for v in sl:
                if close * 0.998 <= v <= close * 1.001:
                    result['bull_liq_score'] += 0.5
        except Exception as exc:
            logger.debug('detect_liquidity_zones: %s', exc)
        return result

    @staticmethod
    def detect_market_structure(df: pd.DataFrame, lookback: int = 50) -> Dict:
        result = {'last_bos': None, 'last_choch': None,
                  'structure_trend': 'neutral', 'bos_score': 0, 'choch_score': 0}
        try:
            recent = df.tail(lookback).reset_index(drop=True)
            close  = float(recent['close'].iloc[-1])
            sh, sl = [], []
            for i in range(2, len(recent) - 2):
                h = float(recent['high'].iloc[i])
                l = float(recent['low'].iloc[i])
                if (h > recent['high'].iloc[i-1] and h > recent['high'].iloc[i-2]
                        and h > recent['high'].iloc[i+1] and h > recent['high'].iloc[i+2]):
                    sh.append((i, h))
                if (l < recent['low'].iloc[i-1] and l < recent['low'].iloc[i-2]
                        and l < recent['low'].iloc[i+1] and l < recent['low'].iloc[i+2]):
                    sl.append((i, l))
            if sh and close > sh[-1][1]:
                result['last_bos']        = 'bullish'
                result['bos_score']       = 1
                result['structure_trend'] = 'bullish'
            if sl and close < sl[-1][1]:
                if result['last_bos'] == 'bullish':
                    result['last_choch']      = 'bearish'
                    result['choch_score']     = 1
                    result['structure_trend'] = 'reversal_bearish'
                else:
                    result['last_bos']        = 'bearish'
                    result['bos_score']       = -1
                    result['structure_trend'] = 'bearish'
        except Exception as exc:
            logger.debug('detect_market_structure: %s', exc)
        return result

    @staticmethod
    def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 50) -> Dict:
        result = {'regular_bullish': False, 'regular_bearish': False,
                  'hidden_bullish': False,  'hidden_bearish': False,
                  'divergence_score': 0.0}
        try:
            recent = df.tail(lookback).reset_index(drop=True)
            if 'rsi' not in recent.columns:
                return result
            price_highs, price_lows, rsi_highs, rsi_lows = [], [], [], []
            for i in range(3, len(recent) - 3):
                c   = float(recent['close'].iloc[i])
                rsi = float(recent['rsi'].iloc[i])
                if np.isnan(c) or np.isnan(rsi):
                    continue
                neighbors = [recent['close'].iloc[i+j] for j in (-2, -1, 1, 2)]
                if c > max(neighbors):
                    price_highs.append((i, c)); rsi_highs.append((i, rsi))
                if c < min(neighbors):
                    price_lows.append((i, c));  rsi_lows.append((i, rsi))

            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                p1, p2 = price_highs[-2][1], price_highs[-1][1]
                r1, r2 = rsi_highs[-2][1],   rsi_highs[-1][1]
                if p2 > p1 and r2 < r1:
                    result['regular_bearish']   = True
                    result['divergence_score'] -= 1.0
                if p2 < p1 and r2 > r1:
                    result['hidden_bearish']    = True
                    result['divergence_score'] -= 0.5

            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                p1, p2 = price_lows[-2][1], price_lows[-1][1]
                r1, r2 = rsi_lows[-2][1],   rsi_lows[-1][1]
                if p2 < p1 and r2 > r1:
                    result['regular_bullish']   = True
                    result['divergence_score'] += 1.0
                if p2 > p1 and r2 < r1:
                    result['hidden_bullish']    = True
                    result['divergence_score'] += 0.5
        except Exception as exc:
            logger.debug('detect_rsi_divergence: %s', exc)
        return result


# ============================================================
# MULTI-TIMEFRAME ANALYSIS
# ============================================================
class MultiTimeframeAnalysis:

    @staticmethod
    def get_trend_score(df: pd.DataFrame) -> float:
        """Return trend score: -1.0 (strong bear) ... +1.0 (strong bull)."""
        if df is None or len(df) < 200:
            return 0.0
        try:
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            close = float(df['close'].iloc[-1])
            e50   = ta.ema(df['close'], length=50)
            e200  = ta.ema(df['close'], length=200)
            if e50 is None or e200 is None:
                return 0.0
            v50  = float(e50.iloc[-1])
            v200 = float(e200.iloc[-1])
            if np.isnan(v50) or np.isnan(v200):
                return 0.0
            if close > v50 > v200:  return  1.0
            if close > v200 > v50:  return  0.5
            if close < v200 < v50:  return -0.5
            if close < v50 < v200:  return -1.0
            return 0.0
        except Exception:
            return 0.0


# ============================================================
# AI ENGINE
# ============================================================
class AIEngine:

    FEATURE_ORDER = [
        'rsi', 'macd', 'macd_hist', 'obv', 'atr', 'bb_width', 'bb_position',
        'ema_alignment', 'volume_ratio', 'order_block_score', 'fvg_score',
        'liquidity_score', 'mtf_trend_15m', 'mtf_trend_4h', 'mtf_trend_1d',
        'funding_rate',
    ]

    def __init__(self, db: DatabaseManager) -> None:
        self.db           = db
        self.scaler       = StandardScaler()
        self.model        = self._build_model()
        self.is_trained   = False
        self.last_trained: Optional[datetime] = None
        self.accuracy     = 0.0

    @staticmethod
    def _build_model() -> VotingClassifier:
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                        max_depth=4, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        return VotingClassifier(estimators=[('gb', gb), ('rf', rf)], voting='soft')

    async def train(self) -> bool:
        try:
            X, y = await self.db.get_ai_training_data()
            if X is None:
                return False
            if len(set(y)) < 2:
                logger.warning('AI training skipped: only one target class')
                return False
            X    = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_sc = self.scaler.fit_transform(X)
            cv   = min(5, len(X) // 20)
            if cv >= 2:
                scores = cross_val_score(self.model, X_sc, y, cv=cv, scoring='accuracy')
                self.accuracy = float(np.mean(scores))
            self.model.fit(X_sc, y)
            self.is_trained   = True
            self.last_trained = datetime.now(AFG_TZ)
            logger.info('AI trained  accuracy=%.3f  samples=%d', self.accuracy, len(X))
            return True
        except Exception as exc:
            logger.error('AI train error: %s', exc)
            return False

    def predict(self, features: Dict) -> Tuple[int, float]:
        if not self.is_trained:
            return 0, 0.5
        try:
            X = np.array([[features.get(f, 0.0) for f in self.FEATURE_ORDER]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_sc    = self.scaler.transform(X)
            probas  = self.model.predict_proba(X_sc)[0]
            classes = self.model.classes_
            best    = int(np.argmax(probas))
            return int(classes[best]), float(probas[best])
        except Exception:
            return 0, 0.5

    def needs_retraining(self) -> bool:
        if not self.is_trained or self.last_trained is None:
            return True
        return (datetime.now(AFG_TZ) - self.last_trained).days >= 1


# ============================================================
# RISK MANAGER
# ============================================================
class RiskManager:

    @staticmethod
    def calculate_position_size(balance: float, entry: float,
                                sl: float, leverage: int = LEVERAGE) -> Dict:
        try:
            risk_usdt = balance * 0.01
            sl_pct    = abs(entry - sl) / (entry + EPSILON)
            if sl_pct <= 0:
                return {'position_size_usdt': 100, 'risk_amount': balance * 0.01}
            pos     = risk_usdt / sl_pct
            max_pos = balance * leverage * 0.10
            pos     = min(pos, max_pos)
            return {
                'position_size_usdt': round(pos, 2),
                'quantity'          : round(pos / entry, 6),
                'risk_amount'       : round(risk_usdt, 2),
                'risk_percent'      : 1.0,
                'sl_percent'        : round(sl_pct * 100, 3),
                'leverage'          : leverage,
            }
        except Exception:
            return {'position_size_usdt': 100, 'risk_amount': balance * 0.01}

    @staticmethod
    def calculate_tp_sl(entry: float, direction: str, atr: float) -> Dict:
        """
        Use a wide SL (2.5x ATR) to avoid premature stop-outs.
        TPs maintain a favourable risk-to-reward ratio.
        """
        try:
            m_sl  = 2.5   # wide SL
            m_tp1 = 2.0
            m_tp2 = 4.5
            m_tp3 = 8.0
            if direction == 'LONG':
                sl  = entry - atr * m_sl
                tp1 = entry + atr * m_tp1
                tp2 = entry + atr * m_tp2
                tp3 = entry + atr * m_tp3
            else:
                sl  = entry + atr * m_sl
                tp1 = entry - atr * m_tp1
                tp2 = entry - atr * m_tp2
                tp3 = entry - atr * m_tp3
            sl_dist = abs(sl - entry)
            return {
                'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
                'rr1': round(abs(tp1 - entry) / (sl_dist + EPSILON), 2),
                'rr2': round(abs(tp2 - entry) / (sl_dist + EPSILON), 2),
                'rr3': round(abs(tp3 - entry) / (sl_dist + EPSILON), 2),
            }
        except Exception:
            return {}


# ============================================================
# SIGNAL SCORING ENGINE  (Ultra-Strict Sniper Mode)
# ============================================================
class SignalScoringEngine:
    """
    Scores a directional signal across 16 categories.

    Three hard gates must ALL pass before any scoring occurs:
      1. ADX > 25 (confirmed trend strength)
      2. All three MTF timeframes (15m, 4h, 1d) align with direction
      3. At least one OB and one FVG present in the direction

    Any gate failure returns confidence = 0 immediately.
    """

    def __init__(self, ai: AIEngine) -> None:
        self.ai = ai

    def score(self, df: pd.DataFrame, direction: str,
              smc: Dict, mtf: Dict,
              funding: float, ob_imbalance: float,
              features: Dict) -> Tuple[float, int, List[str]]:
        """Returns (confidence 0-99, agreement_count, reasons list)."""

        if df is None or len(df) < 5:
            return 0.0, 0, []

        is_long = direction == 'LONG'

        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]

            def _v(col: str) -> float:
                val = last.get(col, np.nan) if hasattr(last, 'get') else getattr(last, col, np.nan)
                return float(val) if not (val is None or (isinstance(val, float) and np.isnan(val))) else np.nan

            # ========================================
            # HARD GATES — all must pass
            # ========================================

            # Gate 1: ADX > 25 required
            adx = _v('adx')
            if np.isnan(adx) or adx <= 25:
                return 0.0, 0, []

            # Gate 2: All three MTF timeframes must align with direction
            s15 = mtf.get('15m', 0.0)
            s4h = mtf.get('4h',  0.0)
            s1d = mtf.get('1d',  0.0)
            if is_long:
                if not (s15 > 0 and s4h > 0 and s1d > 0):
                    return 0.0, 0, []
            else:
                if not (s15 < 0 and s4h < 0 and s1d < 0):
                    return 0.0, 0, []

            # Gate 3: OB + FVG confluence required
            ob_data  = smc.get('order_blocks', {})
            fvg_data = smc.get('fair_value_gaps', {})
            if is_long:
                if ob_data.get('bull_ob_score', 0) <= 0 or fvg_data.get('bull_fvg_score', 0) <= 0:
                    return 0.0, 0, []
            else:
                if ob_data.get('bear_ob_score', 0) <= 0 or fvg_data.get('bear_fvg_score', 0) <= 0:
                    return 0.0, 0, []

            # ========================================
            # SCORING
            # ========================================
            score      = 0.0
            agreements = 0
            reasons: List[str] = []

            # 1  EMA Alignment  +/-15
            ea = _v('ema_alignment')
            if not np.isnan(ea):
                if is_long:
                    if ea >= 3:    score += 15; agreements += 1; reasons.append('EMA stack fully bullish')
                    elif ea >= 1:  score +=  8; reasons.append('EMA alignment bullish')
                    elif ea < 0:   score -= 15
                else:
                    if ea <= -3:   score += 15; agreements += 1; reasons.append('EMA stack fully bearish')
                    elif ea <= -1: score +=  8; reasons.append('EMA alignment bearish')
                    elif ea > 0:   score -= 15

            # 2  RSI  +/-12
            rsi = _v('rsi')
            if not np.isnan(rsi):
                if is_long:
                    if rsi < 35:    score += 12; agreements += 1; reasons.append(f'RSI oversold ({rsi:.1f})')
                    elif rsi <= 55: score +=  6; reasons.append(f'RSI neutral-bullish ({rsi:.1f})')
                    elif rsi > 70:  score -= 12
                else:
                    if rsi > 65:    score += 12; agreements += 1; reasons.append(f'RSI overbought ({rsi:.1f})')
                    elif rsi >= 45: score +=  6; reasons.append(f'RSI neutral-bearish ({rsi:.1f})')
                    elif rsi < 30:  score -= 12

            # 3  MACD  +/-10
            macd_v   = _v('macd'); ms_v = _v('macd_signal'); mh_v = _v('macd_hist')
            pmacd_v  = float(prev['macd']); pms_v = float(prev['macd_signal'])
            if not any(np.isnan(x) for x in [macd_v, ms_v, mh_v, pmacd_v, pms_v]):
                bull_cross = macd_v > ms_v and pmacd_v <= pms_v
                bear_cross = macd_v < ms_v and pmacd_v >= pms_v
                if is_long:
                    if bull_cross:                   score += 10; agreements += 1; reasons.append('MACD bullish crossover')
                    elif macd_v > ms_v and mh_v > 0: score +=  6; reasons.append('MACD bullish momentum')
                    elif macd_v < ms_v:              score -= 10
                else:
                    if bear_cross:                   score += 10; agreements += 1; reasons.append('MACD bearish crossover')
                    elif macd_v < ms_v and mh_v < 0: score +=  6; reasons.append('MACD bearish momentum')
                    elif macd_v > ms_v:              score -= 10

            # 4  Stochastic RSI  +/-8
            sk = _v('stochrsi_k'); sd = _v('stochrsi_d')
            if not any(np.isnan(x) for x in [sk, sd]):
                if is_long:
                    if sk < 20 and sk > sd: score += 8; agreements += 1; reasons.append(f'StochRSI oversold bullish ({sk:.1f})')
                    elif sk < 30:           score += 4
                else:
                    if sk > 80 and sk < sd: score += 8; agreements += 1; reasons.append(f'StochRSI overbought bearish ({sk:.1f})')
                    elif sk > 70:           score += 4

            # 5  Bollinger Bands  +/-8
            bp = _v('bb_position'); bw = _v('bb_width')
            if not any(np.isnan(x) for x in [bp, bw]):
                if is_long:
                    if bp < 0.10:   score += 8; agreements += 1; reasons.append(f'Price at lower BB ({bp:.2f})')
                    elif bp < 0.30: score += 4
                else:
                    if bp > 0.90:   score += 8; agreements += 1; reasons.append(f'Price at upper BB ({bp:.2f})')
                    elif bp > 0.70: score += 4
                if bw < 0.02:
                    reasons.append('BB squeeze — breakout imminent')

            # 6  ADX  +/-8  (gate already ensures >25)
            dmp = _v('dmp'); dmn = _v('dmn')
            if not any(np.isnan(x) for x in [adx, dmp, dmn]):
                if is_long  and dmp > dmn: score += 8; agreements += 1; reasons.append(f'Strong bull trend ADX={adx:.0f}')
                elif not is_long and dmn > dmp: score += 8; agreements += 1; reasons.append(f'Strong bear trend ADX={adx:.0f}')
                elif is_long  and dmn > dmp: score -= 8
                elif not is_long and dmp > dmn: score -= 8

            # 7  Volume  +/-10
            vr = _v('volume_ratio')
            if not np.isnan(vr):
                if vr > 2.0:   score += 10; agreements += 1; reasons.append(f'High volume {vr:.1f}x avg')
                elif vr > 1.3: score +=  5; reasons.append(f'Above-avg volume {vr:.1f}x')
                elif vr < 0.7: score -=  5

            # 8  Multi-Timeframe  (gate ensures all aligned)
            mtf_sum = s15 + s4h * 1.5 + s1d * 2.0
            if is_long:
                if mtf_sum > 3.5:    score += 15; agreements += 1; reasons.append('Strong MTF bullish alignment (all TFs)')
                elif mtf_sum > 2.0:  score += 10; reasons.append('MTF bullish alignment confirmed')
            else:
                if mtf_sum < -3.5:   score += 15; agreements += 1; reasons.append('Strong MTF bearish alignment (all TFs)')
                elif mtf_sum < -2.0: score += 10; reasons.append('MTF bearish alignment confirmed')

            # 9  Market Structure  +/-12
            ms_data = smc.get('market_structure', {})
            bos_s   = ms_data.get('bos_score', 0)
            if is_long:
                if bos_s > 0:   score += 12; agreements += 1; reasons.append('Bullish BOS confirmed')
                elif bos_s < 0: score -= 12
            else:
                if bos_s < 0:   score += 12; agreements += 1; reasons.append('Bearish BOS confirmed')
                elif bos_s > 0: score -= 12
            if ms_data.get('last_choch') == 'bullish' and is_long:
                score += 8; reasons.append('CHoCH bullish reversal')
            if ms_data.get('last_choch') == 'bearish' and not is_long:
                score += 8; reasons.append('CHoCH bearish reversal')

            # 10  Order Blocks  (gate ensures present)
            if is_long  and ob_data.get('bull_ob_score', 0) > 0:
                score += 12; agreements += 1; reasons.append('Bullish Order Block confluence')
            if not is_long and ob_data.get('bear_ob_score', 0) > 0:
                score += 12; agreements += 1; reasons.append('Bearish Order Block confluence')

            # 11  Fair Value Gaps  (gate ensures present)
            if is_long  and fvg_data.get('bull_fvg_score', 0) > 0:
                score += 10; agreements += 1; reasons.append('Bullish FVG confluence')
            if not is_long and fvg_data.get('bear_fvg_score', 0) > 0:
                score += 10; agreements += 1; reasons.append('Bearish FVG confluence')

            # 12  RSI Divergence  +/-15
            div = smc.get('divergence', {})
            if is_long:
                if div.get('regular_bullish'):  score += 15; agreements += 1; reasons.append('Regular Bullish RSI Divergence')
                elif div.get('hidden_bullish'): score +=  8; reasons.append('Hidden Bullish RSI Divergence')
                if div.get('regular_bearish'):  score -= 15
            else:
                if div.get('regular_bearish'):  score += 15; agreements += 1; reasons.append('Regular Bearish RSI Divergence')
                elif div.get('hidden_bearish'): score +=  8; reasons.append('Hidden Bearish RSI Divergence')
                if div.get('regular_bullish'):  score -= 15

            # 13  Funding Rate  +/-8
            fr = funding if funding is not None and not np.isnan(funding) else 0.0
            if is_long  and fr < -0.001: score += 8; agreements += 1; reasons.append(f'Negative funding rate {fr*100:.4f}%')
            elif not is_long and fr > 0.001: score += 8; agreements += 1; reasons.append(f'Positive funding rate {fr*100:.4f}%')
            elif is_long  and fr > 0.003: score -= 8
            elif not is_long and fr < -0.003: score -= 8

            # 14  Orderbook Imbalance  +/-6
            if ob_imbalance is not None and not np.isnan(ob_imbalance):
                if is_long  and ob_imbalance > 0.60: score += 6; reasons.append(f'Bid imbalance {ob_imbalance:.0%}')
                elif not is_long and ob_imbalance < 0.40: score += 6; reasons.append(f'Ask imbalance {1-ob_imbalance:.0%}')

            # 15  Supertrend  +/-5
            std = _v('supertrend_dir')
            if not np.isnan(std):
                if is_long  and std == 1:    score += 5; reasons.append('Supertrend bullish')
                elif not is_long and std == -1: score += 5; reasons.append('Supertrend bearish')
                elif is_long  and std == -1: score -= 5
                elif not is_long and std == 1:  score -= 5

            # 16  AI Prediction bonus
            ai_pred, ai_conf = self.ai.predict(features)
            ai_dir = 1 if is_long else -1
            if ai_pred == ai_dir and ai_conf > 0.60:
                bonus = (ai_conf - 0.50) * 20
                score += bonus
                reasons.append(f'AI confirms signal ({ai_conf:.0%})')
            elif ai_pred == -ai_dir and ai_conf > 0.65:
                score -= (ai_conf - 0.50) * 15

            confidence = min(99.0, abs(score) + 50.0)
            return round(confidence, 1), agreements, reasons

        except Exception as exc:
            logger.error('score error: %s\n%s', exc, traceback.format_exc())
            return 0.0, 0, []


# ============================================================
# TELEGRAM FORMATTER  (Quant Analyst style)
# ============================================================
class TelegramFormatter:

    @staticmethod
    def _clean_symbol(raw: str) -> str:
        """Convert exchange symbol to hashtag form.

        Examples:
            'DRIFT/USDT:USDT' -> '#DRIFTUSDT'
            'BTC/USDT'        -> '#BTCUSDT'
        """
        return '#' + raw.split(':')[0].replace('/', '')

    @staticmethod
    def _session() -> str:
        h = datetime.utcnow().hour
        if  8 <= h < 16: return 'London'
        if 13 <= h < 21: return 'New York'
        if  0 <= h <  8: return 'Tokyo'
        return 'Off-Hours'

    @staticmethod
    def _regime(df: pd.DataFrame) -> str:
        try:
            adx   = float(df['adx'].iloc[-1])
            close = float(df['close'].iloc[-1])
            e50   = float(df['ema50'].iloc[-1])
            bw    = float(df['bb_width'].iloc[-1])
            if adx > 30 and close > e50: return 'Trending Up'
            if adx > 30 and close < e50: return 'Trending Down'
            if bw  > 0.04:               return 'Volatile'
            return 'Ranging'
        except Exception:
            return 'Unknown'

    @staticmethod
    def signal(sig: Dict, sig_id: int) -> str:
        d       = sig['direction']
        is_long = d == 'LONG'
        setup   = '📈 LONG SETUP'  if is_long else '📉 SHORT SETUP'
        market  = '📈 Upward'      if is_long else '📉 Downward'
        conf    = sig['confidence']
        entry   = sig['entry']
        sl      = sig['sl']
        tp1     = sig['tp1']
        tp2     = sig['tp2']
        tp3     = sig['tp3']
        symbol  = TelegramFormatter._clean_symbol(sig['symbol'])
        lev     = sig.get('leverage', LEVERAGE)

        reas = sig.get('reasons', [])
        if isinstance(reas, str):
            try:
                reas = json.loads(reas)
            except Exception:
                reas = []
        catalyst = '; '.join(reas[:3]) if reas else 'Strong MTF Confirmation & OB/FVG Confluence.'

        return (
            f'{setup}: <b>{symbol}</b>\n'
            f'<b>━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'🔻 Entry Zone : <code>{entry:.4f}</code>\n'
            f'⛔️ Stop Loss  : <code>{sl:.4f}</code>\n\n'
            f'🎯 Take Profit Targets:\n'
            f'  • TP1 (Safe): <code>{tp1:.4f}</code>\n'
            f'  • TP2 (Mid):  <code>{tp2:.4f}</code>\n'
            f'  • TP3 (Max):  <code>{tp3:.4f}</code>\n\n'
            f'📐 Trade Info:\n'
            f'  Leverage: {lev}x\n'
            f'  Win Prob: {conf:.0f}%\n'
            f'  Market:   {market}\n\n'
            f'💡 Signal Catalyst:\n'
            f'  {catalyst}\n\n'
            f'@NovaCryptoSignal'
        )

    @staticmethod
    def tp_hit(sig: Dict, tp_num: int, price: float, pnl: float) -> str:
        e      = {1: '🥇', 2: '🥈', 3: '🏆'}.get(tp_num, '🎯')
        symbol = TelegramFormatter._clean_symbol(sig['symbol'])
        return (
            f'{e} <b>TP{tp_num} HIT!</b>\n'
            f'<b>━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'📌 {symbol}  {sig["direction"]}\n'
            f'💰 Price:  <code>{price:.4f}</code>\n'
            f'📊 Profit: <code>+{pnl:.2f}%</code>\n\n'
            f'@NovaCryptoSignal'
        )

    @staticmethod
    def sl_hit(sig: Dict, price: float, pnl: float) -> str:
        symbol = TelegramFormatter._clean_symbol(sig['symbol'])
        return (
            f'🛡 <b>STOP LOSS HIT</b>\n'
            f'<b>━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'📌 {symbol}  {sig["direction"]}\n'
            f'💔 Price: <code>{price:.4f}</code>\n'
            f'📉 Loss:  <code>{pnl:.2f}%</code>\n'
            f'Risk managed successfully.\n\n'
            f'@NovaCryptoSignal'
        )

    @staticmethod
    def daily_report(signals: List[Dict], date: datetime) -> str:
        date_str = date.strftime('%Y-%m-%d')
        if not signals:
            return (
                f'📊 <b>DAILY PERFORMANCE REPORT</b>\n'
                f'📅 {date_str}\n'
                f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
                f'😴 No completed signals today.\n'
                f'<i>VIP Crypto Signal Bot v3.0</i>'
            )
        wins    = [s for s in signals if s.get('pnl_percent', 0) > 0]
        tot_pct = sum(s.get('pnl_percent', 0) for s in signals)
        tot_usd = sum(s.get('pnl_usdt',   0) for s in signals)
        wr      = len(wins) / len(signals) * 100
        best    = max(signals, key=lambda x: x.get('pnl_percent', 0))
        worst   = min(signals, key=lambda x: x.get('pnl_percent', 0))
        rows    = ''.join(
            f"  {'✅' if s.get('pnl_percent', 0) > 0 else '❌'} "
            f"{TelegramFormatter._clean_symbol(s['symbol'])} {s['direction']}: "
            f"{s.get('pnl_percent', 0):+.2f}%\n"
            for s in signals
        )
        return (
            f'📊 <b>DAILY PERFORMANCE REPORT</b>\n'
            f'📅 {date_str}\n'
            f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'📈 Total Trades: <b>{len(signals)}</b>\n'
            f'✅ Wins: <b>{len(wins)}</b>  ❌ Losses: <b>{len(signals)-len(wins)}</b>\n'
            f'🎯 Win Rate: <b>{wr:.1f}%</b>\n'
            f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'💰 Total PnL: <code>{tot_pct:+.2f}%</code>  (<code>{tot_usd:+.2f} USDT</code>)\n'
            f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'📋 <b>Trades:</b>\n{rows}'
            f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'🏆 Best:  {TelegramFormatter._clean_symbol(best["symbol"])} '
            f'<code>{best.get("pnl_percent", 0):+.2f}%</code>\n'
            f'💔 Worst: {TelegramFormatter._clean_symbol(worst["symbol"])} '
            f'<code>{worst.get("pnl_percent", 0):+.2f}%</code>\n'
            f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'🤖 <i>VIP Crypto Signal Bot v3.0</i>'
        )

    @staticmethod
    def weekly_report(signals: List[Dict], ws: datetime, we: datetime,
                      daily: Dict) -> str:
        if not signals:
            return (
                f'🏆 <b>WEEKLY PERFORMANCE REPORT</b>\n'
                f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
                f'😴 No signals this week.\n'
                f'<i>VIP Crypto Signal Bot v3.0</i>'
            )
        wins    = [s for s in signals if s.get('pnl_percent', 0) > 0]
        wr      = len(wins) / len(signals) * 100
        tot_pct = sum(s.get('pnl_percent', 0) for s in signals)
        tot_usd = sum(s.get('pnl_usdt',   0) for s in signals)
        sym_pnl: Dict[str, float] = defaultdict(float)
        for s in signals:
            sym_pnl[s['symbol']] += s.get('pnl_percent', 0)
        top3    = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)[:3]
        top_txt = '\n'.join(
            f'  {i+1}. {TelegramFormatter._clean_symbol(sym)}: {pnl:+.2f}%'
            for i, (sym, pnl) in enumerate(top3)
        )
        day_txt = ''.join(
            f"  {'📈' if sum(s.get('pnl_percent', 0) for s in ds) >= 0 else '📉'} "
            f"{day}: {sum(s.get('pnl_percent', 0) for s in ds):+.2f}% "
            f"({len(ds)} trades)\n"
            for day, ds in daily.items()
        )
        return (
            f'🏆 <b>WEEKLY PERFORMANCE REPORT</b>\n'
            f'📅 {ws.strftime("%m/%d")}–{we.strftime("%m/%d/%Y")}\n'
            f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'📈 Total Trades: <b>{len(signals)}</b>  ✅ {len(wins)}  ❌ {len(signals)-len(wins)}\n'
            f'🎯 Win Rate: <b>{wr:.1f}%</b>\n'
            f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'💰 Total PnL: <code>{tot_pct:+.2f}%</code>  (<code>{tot_usd:+.2f} USDT</code>)\n'
            f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'📅 <b>Day-by-Day:</b>\n{day_txt}'
            f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'🏆 <b>Top Symbols:</b>\n{top_txt}\n'
            f'<b>━━━━━━━━━━━━━━━━━━━━━━━</b>\n'
            f'🤖 <i>VIP Crypto Signal Bot v3.0</i>'
        )


# ============================================================
# MAIN BOT
# ============================================================
class CryptoSignalBot:

    def __init__(self) -> None:
        self.db      = DatabaseManager()
        self.ta      = TechnicalAnalysisEngine()
        self.smc_eng = SmartMoneyConcepts()
        self.mtf_eng = MultiTimeframeAnalysis()
        self.ai      = AIEngine(self.db)
        self.rm      = RiskManager()
        self.fmt     = TelegramFormatter()
        self.scorer  = SignalScoringEngine(self.ai)
        self.tg      = Bot(token=TELEGRAM_TOKEN)

        self.exchange = ccxt_async.binance({
            **API_CONFIG,
            'enableRateLimit': True,
            'timeout'        : 30000,
        })

        self.account_balance    = INITIAL_ACCOUNT_BALANCE
        self.last_daily_report:  Optional[str] = None
        self.last_weekly_report: Optional[str] = None

        self._cooldowns:    Dict[str, datetime]            = {}
        self._ticker_cache: Dict[str, Tuple[float, float]] = {}
        self._CACHE_TTL = 30  # seconds

        logger.info('CryptoSignalBot (v3.0 Sniper) initialised')

    # Telegram helpers
    async def _send(self, text: str,
                    reply_to_message_id: Optional[int] = None) -> Optional[int]:
        """Send a Telegram message and return its message_id for reply threading."""
        try:
            msg = await self.tg.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_to_message_id=reply_to_message_id,
            )
            return msg.message_id
        except TelegramError as exc:
            logger.error('Telegram send error: %s', exc)
            return None

    # Exchange helpers
    async def _fetch_ohlcv(self, symbol: str, tf: str,
                           limit: int = 500) -> Optional[pd.DataFrame]:
        try:
            raw = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
            if not raw or len(raw) < 50:
                return None
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except ccxt_async.RateLimitExceeded:
            logger.warning('Rate limit hit for %s, sleeping 15s', symbol)
            await asyncio.sleep(15)
            return None
        except (ccxt_async.NetworkError, ccxt_async.ExchangeError) as exc:
            logger.debug('OHLCV fetch %s %s: %s', symbol, tf, exc)
            return None

    async def _price(self, symbol: str) -> Optional[float]:
        now = time.monotonic()
        if symbol in self._ticker_cache:
            p, ts = self._ticker_cache[symbol]
            if now - ts < self._CACHE_TTL:
                return p
        try:
            t = await self.exchange.fetch_ticker(symbol)
            p = t.get('last') or t.get('close')
            if p:
                self._ticker_cache[symbol] = (float(p), now)
            return float(p) if p else None
        except Exception:
            return None

    async def _funding(self, symbol: str) -> float:
        try:
            r = await self.exchange.fetch_funding_rate(symbol)
            return float(r.get('fundingRate', 0) or 0)
        except Exception:
            return 0.0

    async def _ob_imbalance(self, symbol: str) -> float:
        try:
            ob   = await self.exchange.fetch_order_book(symbol, limit=20)
            bids = sum(b[1] for b in (ob.get('bids') or [])[:10])
            asks = sum(a[1] for a in (ob.get('asks') or [])[:10])
            tot  = bids + asks
            return bids / tot if tot > 0 else 0.5
        except Exception:
            return 0.5

    async def _futures_symbols(self) -> List[str]:
        try:
            mkts = await self.exchange.load_markets()
            return [
                s for s, m in mkts.items()
                if m.get('quote') == 'USDT'
                and m.get('type') in ('swap', 'future')
                and m.get('active', True)
                and ':USDT' in s
            ]
        except Exception as exc:
            logger.error('load_markets error: %s', exc)
            return []

    def _in_cooldown(self, symbol: str) -> bool:
        if symbol not in self._cooldowns:
            return False
        mins = (datetime.now() - self._cooldowns[symbol]).total_seconds() / 60
        return mins < SIGNAL_COOLDOWN

    async def _active_count(self) -> int:
        return len(await self.db.get_active_signals())

    def _correlated(self, symbol: str, active: List[Dict]) -> int:
        base = symbol.split('/')[0].split(':')[0]
        return sum(1 for s in active if s['symbol'].split('/')[0].split(':')[0] == base)

    # Full symbol analysis
    async def _analyze(self, symbol: str) -> Optional[Dict]:
        try:
            df = await self._fetch_ohlcv(symbol, '1h', 500)
            if df is None or len(df) < 200:
                return None

            df = self.ta.compute_all(df)
            if df is None:
                return None

            close = float(df['close'].iloc[-1])
            atr   = float(df['atr'].iloc[-1])
            if close <= 0 or atr <= 0 or np.isnan(close) or np.isnan(atr):
                return None

            # Multi-timeframe scores
            mtf_scores: Dict[str, float] = {}
            for tf in ['15m', '4h', '1d']:
                mtf_df = await self._fetch_ohlcv(symbol, tf, 250)
                mtf_scores[tf] = self.mtf_eng.get_trend_score(mtf_df)
                await asyncio.sleep(0.15)

            # Smart Money Concepts
            smc_data = {
                'order_blocks'    : self.smc_eng.detect_order_blocks(df),
                'fair_value_gaps' : self.smc_eng.detect_fair_value_gaps(df),
                'liquidity_zones' : self.smc_eng.detect_liquidity_zones(df),
                'market_structure': self.smc_eng.detect_market_structure(df),
                'divergence'      : self.smc_eng.detect_rsi_divergence(df),
            }

            funding  = await self._funding(symbol)
            ob_imbal = await self._ob_imbalance(symbol)

            ob_score  = (smc_data['order_blocks'].get('bull_ob_score', 0)
                         - smc_data['order_blocks'].get('bear_ob_score', 0))
            fvg_score = (smc_data['fair_value_gaps'].get('bull_fvg_score', 0)
                         - smc_data['fair_value_gaps'].get('bear_fvg_score', 0))
            liq_score = (smc_data['liquidity_zones'].get('bull_liq_score', 0)
                         - smc_data['liquidity_zones'].get('bear_liq_score', 0))

            features = {
                'symbol'            : symbol,
                'rsi'               : float(df['rsi'].iloc[-1]),
                'macd'              : float(df['macd'].iloc[-1]),
                'macd_hist'         : float(df['macd_hist'].iloc[-1]),
                'obv'               : float(df['obv_signal'].iloc[-1]),
                'atr'               : atr / close,
                'bb_width'          : float(df['bb_width'].iloc[-1]),
                'bb_position'       : float(df['bb_position'].iloc[-1]),
                'ema_alignment'     : float(df['ema_alignment'].iloc[-1]),
                'volume_ratio'      : float(df['volume_ratio'].iloc[-1]),
                'order_block_score' : float(ob_score),
                'fvg_score'         : float(fvg_score),
                'liquidity_score'   : float(liq_score),
                'mtf_trend_15m'     : float(mtf_scores.get('15m', 0)),
                'mtf_trend_4h'      : float(mtf_scores.get('4h',  0)),
                'mtf_trend_1d'      : float(mtf_scores.get('1d',  0)),
                'funding_rate'      : float(funding),
            }
            # Sanitise NaN / Inf values
            features = {
                k: (float(np.nan_to_num(v, 0.0)) if isinstance(v, float) else v)
                for k, v in features.items()
            }

            best_sig:  Optional[Dict] = None
            best_conf  = 0.0

            for direction in ('LONG', 'SHORT'):
                conf, agr, reas = self.scorer.score(
                    df, direction, smc_data, mtf_scores,
                    funding, ob_imbal, features,
                )
                if conf >= MIN_CONFIDENCE and agr >= MIN_AGREEMENTS and conf > best_conf:
                    best_conf = conf
                    tp_sl = self.rm.calculate_tp_sl(close, direction, atr)
                    if not tp_sl:
                        continue
                    pos = self.rm.calculate_position_size(self.account_balance, close, tp_sl['sl'])
                    best_sig = {
                        'symbol'       : symbol,
                        'direction'    : direction,
                        'entry'        : close,
                        'sl'           : tp_sl['sl'],
                        'tp1'          : tp_sl['tp1'],
                        'tp2'          : tp_sl['tp2'],
                        'tp3'          : tp_sl['tp3'],
                        'confidence'   : conf,
                        'agreements'   : agr,
                        'reasons'      : reas,
                        'position_size': pos.get('position_size_usdt', 100),
                        'session_type' : self.fmt._session(),
                        'market_regime': self.fmt._regime(df),
                        'funding_rate' : funding,
                        'features'     : features,
                        'atr'          : atr,
                        'leverage'     : LEVERAGE,
                    }

            return best_sig

        except Exception as exc:
            logger.debug('_analyze %s error: %s', symbol, exc)
            return None

    # Trade monitor
    async def _monitor_trades(self) -> None:
        for sig in await self.db.get_active_signals():
            try:
                price = await self._price(sig['symbol'])
                if price is None:
                    continue

                entry       = float(sig['entry'])
                direction   = sig['direction']
                sl          = float(sig['sl'])
                tp1         = float(sig['tp1'])
                tp2         = float(sig['tp2'])
                tp3         = float(sig['tp3'])
                is_long     = direction == 'LONG'
                orig_msg_id: Optional[int] = sig.get('telegram_msg_id')

                raw_pnl = (price - entry) / entry if is_long else (entry - price) / entry
                pnl_pct = raw_pnl * 100 * LEVERAGE
                pnl_usd = float(sig.get('position_size', 100)) * raw_pnl * LEVERAGE
                updates = {'pnl_percent': pnl_pct, 'pnl_usdt': pnl_usd}
                if pnl_pct > float(sig.get('max_pnl_reached', 0)):
                    updates['max_pnl_reached'] = pnl_pct

                # SL hit
                if (is_long and price <= sl) or (not is_long and price >= sl):
                    updates.update(
                        status='CLOSED_SL', exit_price=price,
                        closed_at=datetime.now(AFG_TZ).isoformat(),
                        pnl_percent=pnl_pct, pnl_usdt=pnl_usd,
                    )
                    await self.db.update_signal(sig['id'], updates)
                    await self._send(
                        self.fmt.sl_hit(sig, price, pnl_pct),
                        reply_to_message_id=orig_msg_id,
                    )
                    try:
                        feat = json.loads(sig.get('features', '{}'))
                        await self.db.save_ai_data(feat, -1 if is_long else 1)
                    except Exception:
                        pass
                    continue

                # TP3 hit — close position
                if (is_long and price >= tp3) or (not is_long and price <= tp3):
                    updates.update(
                        tp3_hit=1, status='CLOSED_TP3',
                        exit_price=price, closed_at=datetime.now(AFG_TZ).isoformat(),
                        pnl_percent=pnl_pct, pnl_usdt=pnl_usd,
                    )
                    await self.db.update_signal(sig['id'], updates)
                    await self._send(
                        self.fmt.tp_hit(sig, 3, price, pnl_pct),
                        reply_to_message_id=orig_msg_id,
                    )
                    try:
                        feat = json.loads(sig.get('features', '{}'))
                        await self.db.save_ai_data(feat, 1 if is_long else -1)
                    except Exception:
                        pass
                    continue

                # TP2 hit
                if ((is_long and price >= tp2) or (not is_long and price <= tp2)) \
                        and not sig.get('tp2_hit'):
                    updates['tp2_hit'] = 1
                    await self.db.update_signal(sig['id'], updates)
                    await self._send(
                        self.fmt.tp_hit(sig, 2, price, pnl_pct),
                        reply_to_message_id=orig_msg_id,
                    )

                # TP1 hit
                if ((is_long and price >= tp1) or (not is_long and price <= tp1)) \
                        and not sig.get('tp1_hit'):
                    updates['tp1_hit'] = 1
                    # Move SL to breakeven (entry price) once TP1 is reached
                    if not sig.get('breakeven_moved'):
                        updates['sl'] = entry
                        updates['breakeven_moved'] = 1
                    await self.db.update_signal(sig['id'], updates)
                    await self._send(
                        self.fmt.tp_hit(sig, 1, price, pnl_pct),
                        reply_to_message_id=orig_msg_id,
                    )

                await self.db.update_signal(sig['id'], updates)

            except Exception as exc:
                logger.error('monitor trade %s: %s', sig.get('id'), exc)

    # Reports
    async def _check_reports(self) -> None:
        now = datetime.now(AFG_TZ)

        # Daily report at 23:00-23:04 AFT
        if now.hour == 23 and now.minute < 5:
            ds = now.strftime('%Y-%m-%d')
            if self.last_daily_report != ds:
                await self._send_daily(now)
                self.last_daily_report = ds
                await self.db.set_state('last_daily_report', ds)

        # Weekly report on Friday 23:00-23:04 AFT
        if now.weekday() == 4 and now.hour == 23 and now.minute < 5:
            ws = now.strftime('%Y-W%W')
            if self.last_weekly_report != ws:
                await self._send_weekly(now)
                self.last_weekly_report = ws
                await self.db.set_state('last_weekly_report', ws)

    async def _send_daily(self, now: datetime) -> None:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        sigs  = await self.db.get_signals_by_period(start, now)
        await self._send(self.fmt.daily_report(sigs, now))
        logger.info('Daily report sent')

    async def _send_weekly(self, now: datetime) -> None:
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0)
        sigs  = await self.db.get_signals_by_period(start, now)
        daily: Dict[str, List] = defaultdict(list)
        for s in sigs:
            try:
                day = datetime.fromisoformat(s['created_at']).strftime('%A')
                daily[day].append(s)
            except Exception:
                pass
        await self._send(self.fmt.weekly_report(sigs, start, now, dict(daily)))
        logger.info('Weekly report sent')

    # AI retraining
    async def _retrain_if_needed(self) -> None:
        if self.ai.needs_retraining():
            logger.info('Starting AI retraining ...')
            success = await self.ai.train()
            if success:
                logger.info('AI retrained  accuracy=%.3f', self.ai.accuracy)

    # Main loop
    async def run(self) -> None:
        logger.info('=' * 60)
        logger.info('VIP Crypto Signal Bot v3.0 - Sniper Mode Starting ...')
        logger.info('=' * 60)

        await self.db.init_db()
        self.account_balance    = await self.db.get_state('account_balance', INITIAL_ACCOUNT_BALANCE)
        self.last_daily_report  = await self.db.get_state('last_daily_report',  None)
        self.last_weekly_report = await self.db.get_state('last_weekly_report', None)

        cycle = 0
        try:
            while True:
                try:
                    cycle += 1
                    logger.info('-- Cycle #%d --', cycle)

                    # Phase 1: AI retraining
                    await self._retrain_if_needed()

                    # Phase 2: Signal generation
                    if await self._active_count() < MAX_OPEN_TRADES:
                        symbols = await self._futures_symbols()
                        if symbols:
                            random.shuffle(symbols)
                            logger.info(
                                'Scanning %d symbols (Sniper: conf>=%d%%, agreements>=%d) ...',
                                len(symbols), MIN_CONFIDENCE, MIN_AGREEMENTS,
                            )
                            emitted = 0

                            for symbol in symbols:
                                if await self._active_count() >= MAX_OPEN_TRADES:
                                    break
                                if self._in_cooldown(symbol):
                                    continue
                                try:
                                    sig = await self._analyze(symbol)
                                    if sig and sig['confidence'] >= MIN_CONFIDENCE:
                                        active = await self.db.get_active_signals()
                                        if self._correlated(symbol, active) >= 2:
                                            logger.info('Skip %s – too many correlated', symbol)
                                            continue
                                        sig_id = await self.db.save_signal(sig)
                                        if sig_id < 0:
                                            continue
                                        msg_id = await self._send(self.fmt.signal(sig, sig_id))
                                        if msg_id:
                                            await self.db.update_signal(sig_id, {'telegram_msg_id': msg_id})
                                        self._cooldowns[symbol] = datetime.now()
                                        emitted += 1
                                        await self.db.save_ai_data(
                                            sig['features'],
                                            1 if sig['direction'] == 'LONG' else -1,
                                        )
                                        logger.info(
                                            'SIGNAL %s %s  conf=%.1f%%  agr=%d',
                                            symbol, sig['direction'],
                                            sig['confidence'], sig['agreements'],
                                        )
                                    await asyncio.sleep(1.0)
                                except Exception as exc:
                                    logger.debug('analyze %s: %s', symbol, exc)
                                    await asyncio.sleep(0.5)

                            logger.info('Cycle #%d -> emitted %d signals', cycle, emitted)

                    # Phase 3: Trade monitoring
                    await self._monitor_trades()

                    # Phase 4: Reports
                    await self._check_reports()

                    logger.info('Cycle #%d done -- sleeping %ds', cycle, SCAN_INTERVAL)
                    await asyncio.sleep(SCAN_INTERVAL)

                except KeyboardInterrupt:
                    logger.info('Bot stopped by user')
                    break
                except Exception as exc:
                    logger.error('Main loop error: %s\n%s', exc, traceback.format_exc())
                    await asyncio.sleep(30)
        finally:
            try:
                await self.exchange.close()
            except Exception as exc:
                logger.warning('Error closing exchange: %s', exc)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    bot = CryptoSignalBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info('Shutdown complete')
