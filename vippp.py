# -*- coding: utf-8 -*-
"""
VIP Crypto Signal Bot v2.0 — Ultra Professional Edition
=========================================================
Author : VIP Bot
License: MIT

Single-file professional cryptocurrency futures signal bot featuring:
  • 16+ technical indicators
  • Smart Money Concepts (Order Blocks, FVG, Liquidity, BOS/CHoCH, Divergence)
  • Multi-timeframe analysis (15m / 1h / 4h / 1d)
  • AI ensemble model (GradientBoosting + RandomForest)
  • Advanced risk management (1% risk, trailing stop, breakeven, partial TPs)
  • Professional Telegram signals with HTML formatting
  • Daily / Weekly / Monthly reports (Afghanistan timezone UTC+4:30)
  • SQLite persistence layer
"""

# ============================================================
# CONFIGURATION  ─ edit these before running
# You may also set the corresponding environment variables
# (TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API_KEY,
#  BINANCE_SECRET) and the values below are used as fallbacks.
# ============================================================
import os as _os
TELEGRAM_TOKEN          = _os.environ.get('TELEGRAM_TOKEN',   '')
TELEGRAM_CHAT_ID        = _os.environ.get('TELEGRAM_CHAT_ID', '')
DB_NAME                 = 'crypto_bot_data.db'
INITIAL_ACCOUNT_BALANCE = 5000.0
LEVERAGE                = 20

API_CONFIG = {
    'apiKey'     : _os.environ.get('BINANCE_API_KEY', ''),    # ← Set BINANCE_API_KEY env var or fill in here
    'secret'     : _os.environ.get('BINANCE_SECRET',  ''),    # ← Set BINANCE_SECRET env var or fill in here
    'defaultType': 'future',
    'options'    : {'defaultType': 'future'},
}
del _os  # clean up the temporary import alias

# ── Runtime limits ───────────────────────────────────────────
MIN_CONFIDENCE  = 75     # minimum confidence % to emit a signal
MIN_AGREEMENTS  = 5      # minimum indicator categories that must agree
MAX_OPEN_TRADES = 8      # maximum simultaneous positions
SIGNAL_COOLDOWN = 240    # minutes between signals on the same symbol
SCAN_INTERVAL   = 120    # seconds between main-loop cycles

# ============================================================
# IMPORTS
# ============================================================
import asyncio
import json
import logging
import os
import random
import sys
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

# Division-by-zero guard used throughout numeric calculations
EPSILON = 1e-10
from sklearn.ensemble import (GradientBoostingClassifier,
                               RandomForestClassifier,
                               VotingClassifier)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

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

# Afghanistan timezone  UTC+4:30
AFG_TZ = pytz.timezone('Asia/Kabul')


# ============================================================
# DATABASE MANAGER
# ============================================================
class DatabaseManager:
    """All aiosqlite persistence (async-safe, no database is locked errors)."""

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
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT,
            rsi             REAL,
            macd            REAL,
            macd_hist       REAL,
            obv             REAL,
            atr             REAL,
            bb_width        REAL,
            bb_position     REAL,
            ema_alignment   REAL,
            volume_ratio    REAL,
            order_block_score REAL,
            fvg_score       REAL,
            liquidity_score REAL,
            mtf_trend_15m   REAL,
            mtf_trend_4h    REAL,
            mtf_trend_1d    REAL,
            funding_rate    REAL,
            target          INTEGER,
            created_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS reports (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT    NOT NULL,
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

    # ── schema ──────────────────────────────────────────────
    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_name) as db:
            await db.executescript(self._SCHEMA)
            await db.commit()
            logger.info('Database initialised OK')

    # ── signals ─────────────────────────────────────────────
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

    # ── AI data ─────────────────────────────────────────────
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

    # ── bot state ────────────────────────────────────────────
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

    # ── main ────────────────────────────────────────────────
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

            # EMA alignment score  −4 … +4
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
                df['adx'] = adx_df[ac[0]].fillna(25)
                df['dmp'] = adx_df[ac[1]].fillna(25)
                df['dmn'] = adx_df[ac[2]].fillna(25)
            else:
                df['adx'] = df['dmp'] = df['dmn'] = 25.0

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
            tp  = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp * df['volume']).cumsum() / (df['volume'].cumsum() + EPSILON)

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

            # Final NaN cleanup
            df = df.ffill()
            df = df.bfill()
            df = df.fillna(0)
            return df

        except Exception as exc:
            logger.error('compute_all error: %s\n%s', exc, traceback.format_exc())
            return None

    # ── Fibonacci ────────────────────────────────────────────
    @staticmethod
    def compute_fibonacci(df: pd.DataFrame, lookback: int = 100) -> Dict:
        try:
            recent = df.tail(lookback)
            hi = float(recent['high'].max())
            lo = float(recent['low'].min())
            diff = hi - lo
            return {
                'fib_0':   lo,
                'fib_236': lo + 0.236 * diff,
                'fib_382': lo + 0.382 * diff,
                'fib_500': lo + 0.500 * diff,
                'fib_618': lo + 0.618 * diff,
                'fib_786': lo + 0.786 * diff,
                'fib_1':   hi,
                'swing_high': hi,
                'swing_low':  lo,
            }
        except Exception:
            return {}

    # ── Support / Resistance ─────────────────────────────────
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

    # ── Order Blocks ─────────────────────────────────────────
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
                        result['bullish_ob']     = {'high': float(prev['high']), 'low': float(prev['low'])}
                        result['bull_ob_score']  = min(1.0, move * 100)
                # Bearish OB: bullish candle just before strong down-move
                if prev['close'] > prev['open']:
                    move = (prev['close'] - curr['close']) / (prev['close'] + EPSILON)
                    if move > 0.003 and prev['low'] * 0.98 <= close <= prev['high']:
                        result['bearish_ob']    = {'high': float(prev['high']), 'low': float(prev['low'])}
                        result['bear_ob_score'] = min(1.0, move * 100)
        except Exception as exc:
            logger.debug('detect_order_blocks: %s', exc)
        return result

    # ── Fair Value Gaps ──────────────────────────────────────
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

    # ── Liquidity Zones ──────────────────────────────────────
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

    # ── Market Structure ─────────────────────────────────────
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
                result['last_bos']       = 'bullish'
                result['bos_score']      = 1
                result['structure_trend'] = 'bullish'
            if sl and close < sl[-1][1]:
                if result['last_bos'] == 'bullish':
                    result['last_choch']      = 'bearish'
                    result['choch_score']     = 1
                    result['structure_trend'] = 'reversal_bearish'
                else:
                    result['last_bos']       = 'bearish'
                    result['bos_score']      = -1
                    result['structure_trend'] = 'bearish'
        except Exception as exc:
            logger.debug('detect_market_structure: %s', exc)
        return result

    # ── RSI Divergence ───────────────────────────────────────
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
                ph = [recent['close'].iloc[i+j] for j in (-2,-1,1,2)]
                pl = ph
                if c > max(ph):
                    price_highs.append((i, c)); rsi_highs.append((i, rsi))
                if c < min(pl):
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
        """−1 (Strong Bear) … +1 (Strong Bull)."""
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
            if close > v50 > v200:   return  1.0
            if close > v200 > v50:   return  0.5
            if close < v200 < v50:   return -0.5
            if close < v50 < v200:   return -1.0
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
            if len(set(y)) <= 1:
                logger.warning('AI training skipped: target array has only one unique class')
                return False
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
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
            X_sc = self.scaler.transform(X)
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
            # Risk 1% of account per trade – a widely accepted conservative
            # risk-management rule (Kelly Criterion / fixed-fractional).
            # Keeps drawdowns manageable across a long losing streak.
            risk_usdt   = balance * 0.01
            sl_pct      = abs(entry - sl) / entry
            if sl_pct <= 0:
                return {'position_size_usdt': 100, 'risk_amount': balance * 0.01}
            pos         = risk_usdt / sl_pct
            max_pos     = balance * leverage * 0.10
            pos         = min(pos, max_pos)
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
        try:
            m_sl  = 1.5
            m_tp1 = 2.0
            m_tp2 = 4.0
            m_tp3 = 7.0
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
                'rr1': round(abs(tp1 - entry) / sl_dist, 2) if sl_dist > 0 else 0,
                'rr2': round(abs(tp2 - entry) / sl_dist, 2) if sl_dist > 0 else 0,
                'rr3': round(abs(tp3 - entry) / sl_dist, 2) if sl_dist > 0 else 0,
            }
        except Exception:
            return {}


# ============================================================
# SIGNAL SCORING ENGINE
# ============================================================
class SignalScoringEngine:

    def __init__(self, ai: AIEngine) -> None:
        self.ai = ai

    def score(self, df: pd.DataFrame, direction: str,
              smc: Dict, mtf: Dict,
              funding: float, ob_imbalance: float,
              features: Dict) -> Tuple[float, int, List[str]]:
        """
        Score a signal across 16 categories.
        Returns (confidence 0-99, agreement_count, reasons list).
        """
        score      = 0.0
        agreements = 0
        reasons: List[str] = []

        if df is None or len(df) < 5:
            return 0.0, 0, []

        is_long = direction == 'LONG'
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]

            def _v(col: str) -> float:
                val = last.get(col, np.nan) if hasattr(last, 'get') else getattr(last, col, np.nan)
                return float(val) if not (val is None or (isinstance(val, float) and np.isnan(val))) else np.nan

            # ── 1  EMA Alignment  ±15 ──────────────────────────
            ea = _v('ema_alignment')
            if not np.isnan(ea):
                if is_long:
                    if ea >= 3:   score += 15; agreements += 1; reasons.append('✅ Strong bullish EMA stack')
                    elif ea >= 1: score +=  8; reasons.append('✅ Moderate bullish EMA alignment')
                    elif ea < 0:  score -= 15
                else:
                    if ea <= -3:   score += 15; agreements += 1; reasons.append('✅ Strong bearish EMA stack')
                    elif ea <= -1: score +=  8; reasons.append('✅ Moderate bearish EMA alignment')
                    elif ea > 0:   score -= 15

            # ── 2  RSI  ±12 ────────────────────────────────────
            rsi = _v('rsi')
            if not np.isnan(rsi):
                if is_long:
                    if rsi < 35:          score += 12; agreements += 1; reasons.append(f'✅ RSI oversold ({rsi:.1f})')
                    elif rsi <= 60:       score +=  6; reasons.append(f'✅ RSI neutral-bullish ({rsi:.1f})')
                    elif rsi > 70:        score -= 12; reasons.append(f'⚠️ RSI overbought for LONG ({rsi:.1f})')
                else:
                    if rsi > 65:          score += 12; agreements += 1; reasons.append(f'✅ RSI overbought ({rsi:.1f})')
                    elif rsi >= 40:       score +=  6; reasons.append(f'✅ RSI neutral-bearish ({rsi:.1f})')
                    elif rsi < 30:        score -= 12; reasons.append(f'⚠️ RSI oversold for SHORT ({rsi:.1f})')

            # ── 3  MACD  ±10 ───────────────────────────────────
            macd  = _v('macd');  ms = _v('macd_signal'); mh = _v('macd_hist')
            pmacd = float(prev['macd']); pms = float(prev['macd_signal'])
            if not any(np.isnan(x) for x in [macd, ms, mh, pmacd, pms]):
                bull_cross = macd > ms and pmacd <= pms
                bear_cross = macd < ms and pmacd >= pms
                if is_long:
                    if bull_cross:        score += 10; agreements += 1; reasons.append('✅ MACD bullish crossover')
                    elif macd > ms and mh > 0: score += 5; reasons.append('✅ MACD bullish momentum')
                    elif macd < ms:       score -= 10
                else:
                    if bear_cross:        score += 10; agreements += 1; reasons.append('✅ MACD bearish crossover')
                    elif macd < ms and mh < 0: score += 5; reasons.append('✅ MACD bearish momentum')
                    elif macd > ms:       score -= 10

            # ── 4  Stochastic RSI  ±8 ──────────────────────────
            sk = _v('stochrsi_k'); sd = _v('stochrsi_d')
            if not any(np.isnan(x) for x in [sk, sd]):
                if is_long:
                    if sk < 20 and sk > sd: score += 8; agreements += 1; reasons.append(f'✅ StochRSI oversold bullish ({sk:.1f})')
                    elif sk < 30:           score += 4
                else:
                    if sk > 80 and sk < sd: score += 8; agreements += 1; reasons.append(f'✅ StochRSI overbought bearish ({sk:.1f})')
                    elif sk > 70:           score += 4

            # ── 5  Bollinger Bands  ±8 ─────────────────────────
            bp = _v('bb_position'); bw = _v('bb_width')
            if not any(np.isnan(x) for x in [bp, bw]):
                if is_long:
                    if bp < 0.10:   score += 8; agreements += 1; reasons.append(f'✅ Price at lower BB ({bp:.2f})')
                    elif bp < 0.30: score += 4
                else:
                    if bp > 0.90:   score += 8; agreements += 1; reasons.append(f'✅ Price at upper BB ({bp:.2f})')
                    elif bp > 0.70: score += 4
                if bw < 0.02:
                    reasons.append('⚡ BB squeeze — breakout imminent')

            # ── 6  ADX  ±8 ─────────────────────────────────────
            adx = _v('adx'); dmp = _v('dmp'); dmn = _v('dmn')
            if not any(np.isnan(x) for x in [adx, dmp, dmn]):
                if adx > 25:
                    if is_long  and dmp > dmn: score += 8; agreements += 1; reasons.append(f'✅ Strong bull trend ADX={adx:.0f}')
                    elif not is_long and dmn > dmp: score += 8; agreements += 1; reasons.append(f'✅ Strong bear trend ADX={adx:.0f}')
                    elif is_long  and dmn > dmp: score -= 8
                    elif not is_long and dmp > dmn: score -= 8
                else:
                    reasons.append(f'⚠️ Weak trend ADX={adx:.0f}')

            # ── 7  Volume  ±10 ─────────────────────────────────
            vr = _v('volume_ratio')
            if not np.isnan(vr):
                if vr > 2.0:   score += 10; agreements += 1; reasons.append(f'✅ High volume {vr:.1f}x avg')
                elif vr > 1.3: score +=  5; reasons.append(f'✅ Above-avg volume {vr:.1f}x')
                elif vr < 0.7: score -=  5; reasons.append(f'⚠️ Low volume {vr:.1f}x avg')

            # ── 8  Multi-Timeframe  ±15 ────────────────────────
            s15 = mtf.get('15m', 0.0); s4h = mtf.get('4h', 0.0); s1d = mtf.get('1d', 0.0)
            mtf_sum = s15 + s4h * 1.5 + s1d * 2.0
            if is_long:
                if mtf_sum > 2.5:   score += 15; agreements += 1; reasons.append('✅ Strong MTF bullish confirmation')
                elif mtf_sum > 0.5: score +=  8; reasons.append('✅ Moderate MTF bullish')
                elif mtf_sum < -1.5: score -= 15
            else:
                if mtf_sum < -2.5:   score += 15; agreements += 1; reasons.append('✅ Strong MTF bearish confirmation')
                elif mtf_sum < -0.5: score +=  8; reasons.append('✅ Moderate MTF bearish')
                elif mtf_sum > 1.5:  score -= 15

            # ── 9  Market Structure  ±12 ───────────────────────
            ms_data = smc.get('market_structure', {})
            bos_s   = ms_data.get('bos_score', 0)
            if is_long:
                if bos_s > 0:  score += 12; agreements += 1; reasons.append('✅ Bullish BOS confirmed')
                elif bos_s < 0: score -= 12
            else:
                if bos_s < 0:  score += 12; agreements += 1; reasons.append('✅ Bearish BOS confirmed')
                elif bos_s > 0: score -= 12
            if ms_data.get('last_choch') == 'bearish' and not is_long:
                score += 8; reasons.append('✅ CHoCH bearish reversal')
            if ms_data.get('last_choch') == 'bullish' and is_long:
                score += 8; reasons.append('✅ CHoCH bullish reversal')

            # ── 10  Order Blocks  ±12 ──────────────────────────
            ob = smc.get('order_blocks', {})
            if is_long  and ob.get('bull_ob_score', 0) > 0:
                score += 12; agreements += 1; reasons.append('✅ Bullish Order Block nearby')
            if not is_long and ob.get('bear_ob_score', 0) > 0:
                score += 12; agreements += 1; reasons.append('✅ Bearish Order Block nearby')

            # ── 11  Fair Value Gaps  ±10 ───────────────────────
            fvg = smc.get('fair_value_gaps', {})
            if is_long  and fvg.get('bull_fvg_score', 0) > 0:
                score += 10; agreements += 1; reasons.append('✅ Bullish FVG present')
            if not is_long and fvg.get('bear_fvg_score', 0) > 0:
                score += 10; agreements += 1; reasons.append('✅ Bearish FVG present')

            # ── 12  RSI Divergence  ±15 ────────────────────────
            div = smc.get('divergence', {})
            if is_long:
                if div.get('regular_bullish'): score += 15; agreements += 1; reasons.append('✅ Regular Bullish RSI Divergence')
                elif div.get('hidden_bullish'): score += 8; reasons.append('✅ Hidden Bullish RSI Divergence')
                if div.get('regular_bearish'): score -= 15
            else:
                if div.get('regular_bearish'): score += 15; agreements += 1; reasons.append('✅ Regular Bearish RSI Divergence')
                elif div.get('hidden_bearish'): score += 8; reasons.append('✅ Hidden Bearish RSI Divergence')
                if div.get('regular_bullish'): score -= 15

            # ── 13  Funding Rate  ±8 ───────────────────────────
            fr = funding if funding is not None and not np.isnan(funding) else 0.0
            if is_long  and fr < -0.001: score += 8; agreements += 1; reasons.append(f'✅ Negative funding {fr*100:.4f}%')
            elif not is_long and fr > 0.001: score += 8; agreements += 1; reasons.append(f'✅ Positive funding {fr*100:.4f}%')
            elif is_long  and fr > 0.003: score -= 8
            elif not is_long and fr < -0.003: score -= 8

            # ── 14  Orderbook Imbalance  ±6 ────────────────────
            if ob_imbalance is not None and not np.isnan(ob_imbalance):
                if is_long  and ob_imbalance > 0.60: score += 6; reasons.append(f'✅ Bid imbalance {ob_imbalance:.0%}')
                elif not is_long and ob_imbalance < 0.40: score += 6; reasons.append(f'✅ Ask imbalance {1-ob_imbalance:.0%}')

            # ── 15  Supertrend  ±5 ─────────────────────────────
            std = _v('supertrend_dir')
            if not np.isnan(std):
                if is_long  and std == 1:  score += 5; reasons.append('✅ Supertrend bullish')
                elif not is_long and std == -1: score += 5; reasons.append('✅ Supertrend bearish')
                elif is_long  and std == -1: score -= 5
                elif not is_long and std == 1:  score -= 5

            # ── 16  AI Prediction bonus ────────────────────────
            ai_pred, ai_conf = self.ai.predict(features)
            ai_dir = 1 if is_long else -1
            if ai_pred == ai_dir and ai_conf > 0.60:
                bonus = (ai_conf - 0.50) * 20
                score += bonus
                reasons.append(f'✅ AI confirms signal ({ai_conf:.0%})')
            elif ai_pred == -ai_dir and ai_conf > 0.65:
                score -= (ai_conf - 0.50) * 15

            confidence = min(99.0, abs(score) + 50.0)
            return round(confidence, 1), agreements, reasons

        except Exception as exc:
            logger.error('score error: %s\n%s', exc, traceback.format_exc())
            return 0.0, 0, []


# ============================================================
# TELEGRAM FORMATTER
# ============================================================
class TelegramFormatter:

    @staticmethod
    def _bar(pct: float) -> str:
        filled = max(0, min(10, int(pct / 10)))
        return f"[{'█'*filled}{'░'*(10-filled)}] {pct:.0f}%"

    @staticmethod
    def _session() -> str:
        h = datetime.utcnow().hour
        if  8 <= h < 16: return '🇬🇧 London'
        if 13 <= h < 21: return '🇺🇸 New York'
        if  0 <= h <  8: return '🇯🇵 Tokyo'
        return '🌐 Off-Hours'

    @staticmethod
    def _regime(df: pd.DataFrame) -> str:
        try:
            adx   = float(df['adx'].iloc[-1])
            close = float(df['close'].iloc[-1])
            e50   = float(df['ema50'].iloc[-1])
            bw    = float(df['bb_width'].iloc[-1])
            if adx > 30 and close > e50: return '📈 Trending Up'
            if adx > 30 and close < e50: return '📉 Trending Down'
            if bw  > 0.04:               return '⚡ Volatile'
            return '↔️ Ranging'
        except Exception:
            return '❓ Unknown'

    # ── Signal ──────────────────────────────────────────────
    @staticmethod
    def signal(sig: Dict, sig_id: int) -> str:
        d      = sig['direction']
        emoji  = '🟢 LONG' if d == 'LONG' else '🔴 SHORT'
        conf   = sig['confidence']
        entry  = sig['entry']
        sl     = sig['sl']
        tp1    = sig['tp1']; tp2 = sig['tp2']; tp3 = sig['tp3']
        pos    = sig.get('position_size', 0)
        fr     = sig.get('funding_rate', 0.0)
        sess   = sig.get('session_type', TelegramFormatter._session())
        regime = sig.get('market_regime', '❓')
        sl_d   = abs(sl - entry)
        rr1    = round(abs(tp1 - entry) / sl_d, 1) if sl_d > 0 else 0
        rr2    = round(abs(tp2 - entry) / sl_d, 1) if sl_d > 0 else 0
        rr3    = round(abs(tp3 - entry) / sl_d, 1) if sl_d > 0 else 0
        reas   = sig.get('reasons', [])
        if isinstance(reas, str):
            reas = json.loads(reas)
        reas_txt = '\n'.join(f'  {r}' for r in reas[:7])
        bar    = TelegramFormatter._bar(conf)
        return (
            f'💎 <b>VIP SIGNAL #{sig_id:04d}</b> ��\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'🎯 <b>{sig["symbol"]}</b>  |  {emoji}\n'
            f'📊 Confidence: <code>{bar}</code>\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'💰 <b>Entry :</b> <code>${entry:.4f}</code>\n'
            f'🎯 <b>TP 1  :</b> <code>${tp1:.4f}</code>  <i>RR {rr1}x</i>\n'
            f'🎯 <b>TP 2  :</b> <code>${tp2:.4f}</code>  <i>RR {rr2}x</i>\n'
            f'🏆 <b>TP 3  :</b> <code>${tp3:.4f}</code>  <i>RR {rr3}x</i>\n'
            f'🛡 <b>Stop  :</b> <code>${sl:.4f}</code>\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'⚡ <b>Leverage :</b> {LEVERAGE}x\n'
            f'💵 <b>Position :</b> ${pos:.0f} USDT\n'
            f'📈 <b>Funding  :</b> {fr*100:.4f}%\n'
            f'🕐 <b>Session  :</b> {sess}\n'
            f'📊 <b>Regime   :</b> {regime}\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'📋 <b>Analysis:</b>\n{reas_txt}\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'⚠️ <i>Always manage your risk. Not financial advice.</i>\n'
            f'🤖 <i>VIP Crypto Signal Bot v2.0</i>'
        )

    @staticmethod
    def tp_hit(sig: Dict, tp_num: int, price: float, pnl: float) -> str:
        e = {1: '🥇', 2: '🥈', 3: '🏆'}.get(tp_num, '🎯')
        return (
            f'{e} <b>TP{tp_num} HIT!</b> {e}\n'
            f'━━━━━━━━━━━━━━━━\n'
            f'🎯 <b>{sig["symbol"]}</b>  {sig["direction"]}\n'
            f'💰 Price: <code>${price:.4f}</code>\n'
            f'📊 Profit: <code>+{pnl:.2f}%</code>\n'
            f'🎉 Signal #{sig["id"]:04d} — Congratulations!'
        )

    @staticmethod
    def sl_hit(sig: Dict, price: float, pnl: float) -> str:
        return (
            f'🛑 <b>STOP LOSS HIT</b>\n'
            f'━━━━━━━━━━━━━━━━\n'
            f'📊 <b>{sig["symbol"]}</b>  {sig["direction"]}\n'
            f'💔 Price: <code>${price:.4f}</code>\n'
            f'📉 Loss: <code>{pnl:.2f}%</code>\n'
            f'🔄 Signal #{sig["id"]:04d} closed\n'
            f'<i>Cut losses, preserve capital 💪</i>'
        )

    @staticmethod
    def breakeven(sig: Dict) -> str:
        return (
            f'🔐 <b>BREAKEVEN ACTIVATED</b>\n'
            f'━━━━━━━━━━━━━━━━\n'
            f'🎯 <b>{sig["symbol"]}</b>  {sig["direction"]}\n'
            f'✅ SL moved to entry: <code>${sig["entry"]:.4f}</code>\n'
            f'🛡 <b>Risk-free trade!</b>  Signal #{sig["id"]:04d}'
        )

    @staticmethod
    def trailing_update(sig: Dict, new_sl: float) -> str:
        return (
            f'📡 <b>TRAILING STOP UPDATE</b>\n'
            f'━━━━━━━━━━━━━━━━\n'
            f'🎯 <b>{sig["symbol"]}</b>  {sig["direction"]}\n'
            f'🔄 New SL: <code>${new_sl:.4f}</code>\n'
            f'🛡 Protecting profits  Signal #{sig["id"]:04d}'
        )

    # ── Reports ─────────────────────────────────────────────
    @staticmethod
    def daily_report(signals: List[Dict], date: datetime) -> str:
        if not signals:
            return (
                f'📊 <b>DAILY REPORT</b>  |  {date.strftime("%Y-%m-%d")}\n'
                f'━━━━━━━━━━━━━━━━━━━━━━━\n'
                f'😴 No completed signals today\n'
                f'<i>VIP Crypto Signal Bot v2.0</i>'
            )
        wins     = [s for s in signals if s.get('pnl_percent', 0) > 0]
        tot_pct  = sum(s.get('pnl_percent', 0) for s in signals)
        tot_usd  = sum(s.get('pnl_usdt',   0) for s in signals)
        wr       = len(wins) / len(signals) * 100
        best     = max(signals, key=lambda x: x.get('pnl_percent', 0))
        worst    = min(signals, key=lambda x: x.get('pnl_percent', 0))
        rows     = ''.join(
            f"  {'✅' if s.get('pnl_percent',0)>0 else '❌'} "
            f"{s['symbol']} {s['direction']}: {s.get('pnl_percent',0):+.2f}%\n"
            for s in signals
        )
        return (
            f'📊 <b>DAILY REPORT</b>  |  {date.strftime("%Y-%m-%d")}\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'📈 Signals: <b>{len(signals)}</b>\n'
            f'✅ Wins: <b>{len(wins)}</b>  ❌ Losses: <b>{len(signals)-len(wins)}</b>\n'
            f'🎯 Win Rate: <b>{wr:.1f}%</b>\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'💰 PnL: <code>{tot_pct:+.2f}%</code>  (<code>{tot_usd:+.2f} USDT</code>)\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'📋 <b>Trades:</b>\n{rows}'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'🏆 Best:  {best["symbol"]} {best["direction"]} <code>{best.get("pnl_percent",0):+.2f}%</code>\n'
            f'💔 Worst: {worst["symbol"]} {worst["direction"]} <code>{worst.get("pnl_percent",0):+.2f}%</code>\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'🤖 <i>VIP Crypto Signal Bot v2.0</i>'
        )

    @staticmethod
    def weekly_report(signals: List[Dict], ws: datetime, we: datetime,
                      daily: Dict) -> str:
        if not signals:
            return (
                '📊 <b>WEEKLY REPORT</b>\n'
                '━━━━━━━━━━━━━━━━━━━━━━━\n'
                '😴 No signals this week\n'
                '<i>VIP Crypto Signal Bot v2.0</i>'
            )
        wins    = [s for s in signals if s.get('pnl_percent', 0) > 0]
        wr      = len(wins) / len(signals) * 100
        tot_pct = sum(s.get('pnl_percent', 0) for s in signals)
        tot_usd = sum(s.get('pnl_usdt',   0) for s in signals)
        sym_pnl: Dict[str, float] = defaultdict(float)
        for s in signals:
            sym_pnl[s['symbol']] += s.get('pnl_percent', 0)
        top3 = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)[:3]
        top_txt = '\n'.join(f'  {i+1}. {sym}: {pnl:+.2f}%' for i,(sym,pnl) in enumerate(top3))
        day_txt = ''.join(
            f"  {'📈' if sum(s.get('pnl_percent',0) for s in ds)>=0 else '📉'} "
            f"{day}: {sum(s.get('pnl_percent',0) for s in ds):+.2f}% "
            f"({len(ds)} trades)\n"
            for day, ds in daily.items()
        )
        return (
            f'📊 <b>WEEKLY REPORT</b>  |  {ws.strftime("%m/%d")}–{we.strftime("%m/%d/%Y")}\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'📈 Trades: <b>{len(signals)}</b>  ✅ {len(wins)}  ❌ {len(signals)-len(wins)}\n'
            f'🎯 Win Rate: <b>{wr:.1f}%</b>\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'💰 PnL: <code>{tot_pct:+.2f}%</code>  (<code>{tot_usd:+.2f} USDT</code>)\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'📅 <b>Day-by-Day:</b>\n{day_txt}'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'🏆 <b>Top Symbols:</b>\n{top_txt}\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'🤖 <i>VIP Crypto Signal Bot v2.0</i>'
        )

    @staticmethod
    def monthly_report(signals: List[Dict], month: datetime, weekly: Dict) -> str:
        if not signals:
            return (
                f'📊 <b>MONTHLY REPORT</b>  |  {month.strftime("%B %Y")}\n'
                '━━━━━━━━━━━━━━━━━━━━━━━\n'
                '😴 No signals this month\n'
                '<i>VIP Crypto Signal Bot v2.0</i>'
            )
        wins    = [s for s in signals if s.get('pnl_percent', 0) > 0]
        wr      = len(wins) / len(signals) * 100
        tot_pct = sum(s.get('pnl_percent', 0) for s in signals)
        tot_usd = sum(s.get('pnl_usdt',   0) for s in signals)
        wk_txt  = ''
        best_w_pnl = float('-inf'); best_w = ''
        worst_w_pnl = float('inf'); worst_w = ''
        for wl, ws in weekly.items():
            wp = sum(s.get('pnl_percent', 0) for s in ws)
            e  = '📈' if wp >= 0 else '📉'
            wk_txt += f'  {e} {wl}: {wp:+.2f}% ({len(ws)} trades)\n'
            if wp > best_w_pnl:  best_w_pnl  = wp; best_w  = wl
            if wp < worst_w_pnl: worst_w_pnl = wp; worst_w = wl
        return (
            f'📊 <b>MONTHLY REPORT</b>  |  {month.strftime("%B %Y")}\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'📈 Trades: <b>{len(signals)}</b>  ✅ {len(wins)}  ❌ {len(signals)-len(wins)}\n'
            f'🎯 Win Rate: <b>{wr:.1f}%</b>\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'💰 PnL: <code>{tot_pct:+.2f}%</code>  (<code>{tot_usd:+.2f} USDT</code>)\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'📅 <b>Weekly Breakdown:</b>\n{wk_txt}'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'🏆 Best Week : {best_w}  {best_w_pnl:+.2f}%\n'
            f'💔 Worst Week: {worst_w}  {worst_w_pnl:+.2f}%\n'
            f'━━━━━━━━━━━━━━━━━━━━━━━\n'
            f'🤖 <i>VIP Crypto Signal Bot v2.0</i>'
        )


# ============================================================
# MAIN BOT
# ============================================================
class CryptoSignalBot:

    def __init__(self) -> None:
        self.db       = DatabaseManager()
        self.ta       = TechnicalAnalysisEngine()
        self.smc      = SmartMoneyConcepts()
        self.mtf      = MultiTimeframeAnalysis()
        self.ai       = AIEngine(self.db)
        self.rm       = RiskManager()
        self.fmt      = TelegramFormatter()
        self.scorer   = SignalScoringEngine(self.ai)
        self.tg       = Bot(token=TELEGRAM_TOKEN)

        self.exchange = ccxt_async.binance({
            **API_CONFIG,
            'enableRateLimit': True,
            'timeout': 30000,
        })

        self.account_balance     = INITIAL_ACCOUNT_BALANCE
        self.last_daily_report:   Optional[str] = None
        self.last_weekly_report:  Optional[str] = None
        self.last_monthly_report: Optional[str] = None

        self._cooldowns: Dict[str, datetime] = {}
        self._ticker_cache: Dict[str, Tuple[float, float]] = {}
        self._CACHE_TTL = 30  # seconds

        logger.info('CryptoSignalBot ready')

    # ── Telegram helpers ────────────────────────────────────
    async def _send(self, text: str, reply_to_message_id: Optional[int] = None) -> Optional[int]:
        try:
            msg = await self.tg.send_message(
                chat_id=TELEGRAM_CHAT_ID, text=text,
                parse_mode=ParseMode.HTML, disable_web_page_preview=True,
                reply_to_message_id=reply_to_message_id,
            )
            return msg.message_id
        except TelegramError as exc:
            logger.error('Telegram send error: %s', exc)
            return None

    async def _edit(self, msg_id: int, text: str) -> bool:
        try:
            await self.tg.edit_message_text(
                chat_id=TELEGRAM_CHAT_ID, message_id=msg_id, text=text,
                parse_mode=ParseMode.HTML, disable_web_page_preview=True,
            )
            return True
        except TelegramError:
            return False

    # ── Exchange helpers ─────────────────────────────────────
    async def _fetch_ohlcv(self, symbol: str, tf: str, limit: int = 500) -> Optional[pd.DataFrame]:
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
        import time as _time
        now = _time.monotonic()
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

    # ── Full symbol analysis ─────────────────────────────────
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

            # MTF
            mtf_scores: Dict[str, float] = {}
            for tf in ['15m', '4h', '1d']:
                mtf_df = await self._fetch_ohlcv(symbol, tf, 250)
                mtf_scores[tf] = self.mtf.get_trend_score(mtf_df)
                await asyncio.sleep(0.15)

            # SMC
            smc_data = {
                'order_blocks'    : self.smc.detect_order_blocks(df),
                'fair_value_gaps' : self.smc.detect_fair_value_gaps(df),
                'liquidity_zones' : self.smc.detect_liquidity_zones(df),
                'market_structure': self.smc.detect_market_structure(df),
                'divergence'      : self.smc.detect_rsi_divergence(df),
            }

            funding    = await self._funding(symbol)
            ob_imbal   = await self._ob_imbalance(symbol)

            ob_score = (smc_data['order_blocks'].get('bull_ob_score', 0)
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
            # Sanitise
            features = {k: (float(np.nan_to_num(v, 0.0)) if isinstance(v, float) else v)
                        for k, v in features.items()}

            best_sig:   Optional[Dict] = None
            best_conf   = 0.0

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
                    }

            return best_sig

        except Exception as exc:
            logger.debug('_analyze %s error: %s', symbol, exc)
            return None

    # ── Trade monitor ────────────────────────────────────────
    async def _monitor_trades(self) -> None:
        for sig in await self.db.get_active_signals():
            try:
                price = await self._price(sig['symbol'])
                if price is None:
                    continue

                entry     = float(sig['entry'])
                direction = sig['direction']
                sl        = float(sig['sl'])
                tp1       = float(sig['tp1'])
                tp2       = float(sig['tp2'])
                tp3       = float(sig['tp3'])
                is_long   = direction == 'LONG'
                orig_msg_id: Optional[int] = sig.get('telegram_msg_id')

                raw_pnl   = (price - entry) / entry if is_long else (entry - price) / entry
                pnl_pct   = raw_pnl * 100 * LEVERAGE
                pnl_usd   = float(sig.get('position_size', 100)) * raw_pnl * LEVERAGE
                updates   = {'pnl_percent': pnl_pct, 'pnl_usdt': pnl_usd}
                if pnl_pct > float(sig.get('max_pnl_reached', 0)):
                    updates['max_pnl_reached'] = pnl_pct

                # ── SL hit ─────────────────────────────────
                sl_hit = (is_long and price <= sl) or (not is_long and price >= sl)
                if sl_hit:
                    updates.update(status='CLOSED_SL', exit_price=price,
                                   closed_at=datetime.now(AFG_TZ).isoformat())
                    updates.update(pnl_percent=pnl_pct, pnl_usdt=pnl_usd)
                    await self.db.update_signal(sig['id'], updates)
                    await self._send(self.fmt.sl_hit(sig, price, pnl_pct),
                                     reply_to_message_id=orig_msg_id)
                    try:
                        feat = json.loads(sig.get('features', '{}'))
                        await self.db.save_ai_data(feat, -1 if is_long else 1)
                    except Exception:
                        pass
                    continue

                # ── TP3 ────────────────────────────────────
                tp3_hit = (is_long and price >= tp3) or (not is_long and price <= tp3)
                if tp3_hit:
                    updates.update(tp3_hit=1, status='CLOSED_TP3',
                                   exit_price=price, closed_at=datetime.now(AFG_TZ).isoformat(),
                                   pnl_percent=pnl_pct, pnl_usdt=pnl_usd)
                    await self.db.update_signal(sig['id'], updates)
                    await self._send(self.fmt.tp_hit(sig, 3, price, pnl_pct),
                                     reply_to_message_id=orig_msg_id)
                    try:
                        feat = json.loads(sig.get('features', '{}'))
                        await self.db.save_ai_data(feat, 1 if is_long else -1)
                    except Exception:
                        pass
                    continue

                # ── TP2 ────────────────────────────────────
                tp2_hit = (is_long and price >= tp2) or (not is_long and price <= tp2)
                if tp2_hit and not sig.get('tp2_hit'):
                    updates['tp2_hit'] = 1
                    await self.db.update_signal(sig['id'], updates)
                    await self._send(self.fmt.tp_hit(sig, 2, price, pnl_pct),
                                     reply_to_message_id=orig_msg_id)

                # ── TP1 ────────────────────────────────────
                tp1_hit = (is_long and price >= tp1) or (not is_long and price <= tp1)
                if tp1_hit and not sig.get('tp1_hit'):
                    updates['tp1_hit'] = 1
                    await self.db.update_signal(sig['id'], updates)
                    await self._send(self.fmt.tp_hit(sig, 1, price, pnl_pct),
                                     reply_to_message_id=orig_msg_id)

                # ── Breakeven ──────────────────────────────
                if not sig.get('breakeven_moved'):
                    dist_done = abs(price - entry)
                    dist_tp1  = abs(tp1 - entry)
                    if dist_done >= dist_tp1 * 0.40:
                        updates['breakeven_moved'] = 1
                        updates['sl']              = entry
                        await self.db.update_signal(sig['id'], updates)
                        await self._send(self.fmt.breakeven(sig),
                                         reply_to_message_id=orig_msg_id)
                        continue

                # ── Trailing stop ─────────────────────────
                if pnl_pct > 3.0:
                    atr_val      = float(sig.get('atr', abs(entry - float(sig.get('original_sl', sl))) / 1.5))
                    trail_dist   = atr_val * 0.5
                    cur_trail_sl = float(sig.get('trailing_sl') or sl)
                    if is_long:
                        new_tsl = price - trail_dist
                        if new_tsl > cur_trail_sl:
                            updates.update(trailing_active=1, trailing_sl=new_tsl, sl=new_tsl)
                            await self.db.update_signal(sig['id'], updates)
                            if sig.get('trailing_active'):
                                await self._send(self.fmt.trailing_update(sig, new_tsl),
                                                 reply_to_message_id=orig_msg_id)
                            continue
                    else:
                        new_tsl = price + trail_dist
                        if new_tsl < cur_trail_sl:
                            updates.update(trailing_active=1, trailing_sl=new_tsl, sl=new_tsl)
                            await self.db.update_signal(sig['id'], updates)
                            if sig.get('trailing_active'):
                                await self._send(self.fmt.trailing_update(sig, new_tsl),
                                                 reply_to_message_id=orig_msg_id)
                            continue

                await self.db.update_signal(sig['id'], updates)

            except Exception as exc:
                logger.error('monitor trade %s: %s', sig.get('id'), exc)

    # ── Reports ─────────────────────────────────────────────
    async def _check_reports(self) -> None:
        now = datetime.now(AFG_TZ)
        # Daily at 23:xx AFT
        if now.hour == 23 and now.minute < 5:
            ds = now.strftime('%Y-%m-%d')
            if self.last_daily_report != ds:
                await self._send_daily(now)
                self.last_daily_report = ds
                await self.db.set_state('last_daily_report', ds)
        # Weekly on Friday 23:xx
        if now.weekday() == 4 and now.hour == 23 and now.minute < 5:
            ws = now.strftime('%Y-W%W')
            if self.last_weekly_report != ws:
                await self._send_weekly(now)
                self.last_weekly_report = ws
                await self.db.set_state('last_weekly_report', ws)
        # Monthly on 1st 23:xx
        if now.day == 1 and now.hour == 23 and now.minute < 5:
            ms = now.strftime('%Y-%m')
            if self.last_monthly_report != ms:
                await self._send_monthly(now)
                self.last_monthly_report = ms
                await self.db.set_state('last_monthly_report', ms)

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

    async def _send_monthly(self, now: datetime) -> None:
        end   = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start = (end - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        sigs  = await self.db.get_signals_by_period(start, end)
        weekly: Dict[str, List] = defaultdict(list)
        for s in sigs:
            try:
                wn = datetime.fromisoformat(s['created_at']).isocalendar()[1]
                weekly[f'Week {wn}'].append(s)
            except Exception:
                pass
        await self._send(self.fmt.monthly_report(sigs, start, dict(weekly)))
        logger.info('Monthly report sent')

    # ── AI retraining ────────────────────────────────────────
    async def _retrain_if_needed(self) -> None:
        if self.ai.needs_retraining():
            logger.info('Starting AI retraining …')
            success = await self.ai.train()
            if success:
                logger.info('AI retrained  accuracy=%.3f', self.ai.accuracy)

    # ── Main loop ────────────────────────────────────────────
    async def run(self) -> None:
        logger.info('=' * 60)
        logger.info('VIP Crypto Signal Bot v2.0 Starting …')
        logger.info('=' * 60)

        # Initialise DB schema and load persisted state
        await self.db.init_db()
        self.account_balance     = await self.db.get_state('account_balance', INITIAL_ACCOUNT_BALANCE)
        self.last_daily_report   = await self.db.get_state('last_daily_report',   None)
        self.last_weekly_report  = await self.db.get_state('last_weekly_report',  None)
        self.last_monthly_report = await self.db.get_state('last_monthly_report', None)

        await self._send(
            '🚀 <b>VIP Crypto Signal Bot v2.0 Started!</b>\n'
            '━━━━━━━━━━━━━━━━━━━━━━━\n'
            '✅ Scanning ALL Binance USDT Futures\n'
            '🤖 AI Engine: Active\n'
            '📊 16+ Technical Indicators: Active\n'
            '💎 Smart Money Concepts: Active\n'
            '⏱ Multi-Timeframe Analysis: Active\n'
            '⏰ Afghanistan TZ Reports: Active\n'
            '━━━━━━━━━━━━━━━━━━━━━━━\n'
            '🔍 Starting market scan …'
        )

        cycle = 0
        try:
            while True:
                try:
                    cycle += 1
                    logger.info('── Cycle #%d ──────────────────────────', cycle)

                    # Phase 1 – AI retraining
                    await self._retrain_if_needed()

                    # Phase 2 – Signal generation
                    if await self._active_count() < MAX_OPEN_TRADES:
                        symbols = await self._futures_symbols()
                        if symbols:
                            # Shuffle symbols to distribute API load fairly across the
                            # full universe; avoids always analysing the same symbols first.
                            random.shuffle(symbols)
                            logger.info('Scanning %d symbols …', len(symbols))
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

                            logger.info('Cycle #%d → emitted %d signals', cycle, emitted)

                    # Phase 3 – Trade monitoring
                    await self._monitor_trades()

                    # Phase 4 – Reports
                    await self._check_reports()

                    logger.info('Cycle #%d done — sleeping %ds', cycle, SCAN_INTERVAL)
                    await asyncio.sleep(SCAN_INTERVAL)

                except KeyboardInterrupt:
                    logger.info('Bot stopped by user')
                    await self._send('⛔ <b>Bot stopped by operator</b>')
                    break
                except Exception as exc:
                    logger.error('Main loop error: %s\n%s', exc, traceback.format_exc())
                    await asyncio.sleep(30)
        finally:
            await self.exchange.close()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    bot = CryptoSignalBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info('Shutdown complete')
