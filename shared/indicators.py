# shared/indicators.py
"""
Shared indicator calculation module.
Used identically by Engine (live) and Trainer (simulation).
If this file diverges between the two: that is a bug.

All functions follow the same contract:
  Input:  pandas DataFrame with OHLCV columns
  Output: pandas Series or dict of Series
  Rules:
    - Never modifies the input DataFrame
    - Returns same length as input
    - Returns NaN for warmup rows (insufficient history)
    - Raises IndicatorError on invalid input
"""

import pandas as pd
import pandas_ta as ta
import numpy as np

class IndicatorError(Exception):
    pass

def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """Simple Moving Average."""
    if period < 1:
        raise IndicatorError(f"SMA period must be >= 1, got {period}")
    return ta.sma(df['close'], length=period)

def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average."""
    if period < 1:
        raise IndicatorError(f"EMA period must be >= 1, got {period}")
    return ta.ema(df['close'], length=period)

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    return ta.rsi(df['close'], length=period)

def calculate_bbands(df: pd.DataFrame,
                     period: int = 20,
                     std: float = 2.0) -> dict:
    """Bollinger Bands. Returns dict of Series."""
    result = ta.bbands(df['close'], length=period, std=std)
    return {
        'bb_upper': result[f'BBU_{period}_{std}'],
        'bb_mid':   result[f'BBM_{period}_{std}'],
        'bb_lower': result[f'BBL_{period}_{std}'],
        'bb_width': result[f'BBB_{period}_{std}'],
        'bb_pct':   result[f'BBP_{period}_{std}'],
    }

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    return ta.atr(df['high'], df['low'], df['close'], length=period)

def calculate_adx(df: pd.DataFrame, period: int = 14) -> dict:
    """Average Directional Index. Returns dict of Series."""
    result = ta.adx(df['high'], df['low'], df['close'], length=period)
    return {
        'adx':    result[f'ADX_{period}'],
        'dmp':    result[f'DMP_{period}'],
        'dmn':    result[f'DMN_{period}'],
    }

def calculate_macd(df: pd.DataFrame,
                   fast: int = 12,
                   slow: int = 26,
                   signal: int = 9) -> dict:
    """MACD. Returns dict of Series."""
    result = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
    return {
        'macd':        result[f'MACD_{fast}_{slow}_{signal}'],
        'macd_signal': result[f'MACDs_{fast}_{slow}_{signal}'],
        'macd_hist':   result[f'MACDh_{fast}_{slow}_{signal}'],
    }

def calculate_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume Simple Moving Average."""
    return ta.sma(df['volume'], length=period)

def calculate_session(df: pd.DataFrame) -> pd.Series:
    """
    Session label for each candle based on UTC timestamp.
    Priority order: DEAD_ZONE → LONDON_NY_OVERLAP → LONDON_OPEN
                    → NY_OPEN → LONDON → NEW_YORK → ASIAN
    """
    def get_session(ts):
        hour = ts.hour
        if 21 <= hour or hour < 0:
            return 'DEAD_ZONE'
        if 13 <= hour < 16:
            return 'LONDON_NY_OVERLAP'
        if 7 <= hour < 9:
            return 'LONDON_OPEN'
        if 13 <= hour < 15:
            return 'NY_OPEN'
        if 8 <= hour < 16:
            return 'LONDON'
        if 13 <= hour < 21:
            return 'NEW_YORK'
        return 'ASIAN'

    return df.index.to_series().apply(get_session)

def calculate_regime(df: pd.DataFrame,
                     adx_series: pd.Series,
                     atr_series: pd.Series) -> pd.Series:
    """
    Regime label for each candle.
    Requires pre-computed ADX and ATR series.
    Order: QUIET → VOLATILE → TRENDING → RANGING → TRANSITIONING
    """
    atr_sma20 = ta.sma(atr_series, length=20)
    atr_ratio = atr_series / atr_sma20

    def classify(row):
        adx = row['adx']
        ratio = row['atr_ratio']
        if pd.isna(adx) or pd.isna(ratio):
            return None
        if ratio < 0.7:
            return 'QUIET'
        if ratio > 1.5:
            return 'VOLATILE'
        if adx > 25:
            return 'TRENDING'
        if adx < 20:
            return 'RANGING'
        return 'TRANSITIONING'

    temp = pd.DataFrame({'adx': adx_series, 'atr_ratio': atr_ratio})
    return temp.apply(classify, axis=1)

# Crossover detection utilities
def crossed_above(fast_current: float, fast_prev: float,
                  slow_current: float, slow_prev: float) -> bool:
    """True ONLY on the candle where fast crosses above slow."""
    return (fast_prev <= slow_prev) and (fast_current > slow_current)

def crossed_below(fast_current: float, fast_prev: float,
                  slow_current: float, slow_prev: float) -> bool:
    """True ONLY on the candle where fast crosses below slow."""
    return (fast_prev >= slow_prev) and (fast_current < slow_current)