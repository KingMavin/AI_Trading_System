"""
Shared indicator calculation module.
Used identically by Engine (live) and Trainer (simulation).

CRITICAL: This file is the single source of truth for all
indicator calculations. Both Engine and Trainer import from
here. Any divergence between the two is a bug.

Contract for all functions:
  Input:  pandas DataFrame with OHLCV columns and DatetimeIndex
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


# ── MOVING AVERAGES ────────────────────────────────────

def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """Simple Moving Average on close price."""
    if period < 1:
        raise IndicatorError(f"SMA period must be >= 1, got {period}")
    return ta.sma(df['close'], length=period)


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average on close price."""
    if period < 1:
        raise IndicatorError(f"EMA period must be >= 1, got {period}")
    return ta.ema(df['close'], length=period)


# ── OSCILLATORS ────────────────────────────────────────

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    Returns values 0-100. NaN for first (period) rows.
    """
    if period < 2:
        raise IndicatorError(f"RSI period must be >= 2, got {period}")
    return ta.rsi(df['close'], length=period)


def calculate_macd(df: pd.DataFrame,
                   fast: int = 12,
                   slow: int = 26,
                   signal: int = 9) -> dict:
    """
    MACD — Moving Average Convergence Divergence.
    Returns dict with keys: macd, macd_signal, macd_hist
    """
    result = ta.macd(df['close'], fast=fast, slow=slow,
                     signal=signal)
    if result is None:
        raise IndicatorError("MACD calculation returned None")
    return {
        'macd':        result[f'MACD_{fast}_{slow}_{signal}'],
        'macd_signal': result[f'MACDs_{fast}_{slow}_{signal}'],
        'macd_hist':   result[f'MACDh_{fast}_{slow}_{signal}'],
    }


# ── VOLATILITY ─────────────────────────────────────────

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.
    Always positive. NaN for first (period) rows.
    """
    if period < 1:
        raise IndicatorError(f"ATR period must be >= 1, got {period}")
    return ta.atr(df['high'], df['low'], df['close'],
                  length=period)


def calculate_bbands(df: pd.DataFrame,
                     period: int = 20,
                     std: float = 2.0) -> dict:
    """
    Bollinger Bands.
    Returns dict with keys:
      bb_upper, bb_mid, bb_lower, bb_width, bb_pct
    """
    if period < 2:
        raise IndicatorError(f"BB period must be >= 2, got {period}")
    if std <= 0:
        raise IndicatorError(f"BB std must be > 0, got {std}")

    result = ta.bbands(df['close'], length=period, std=std)
    if result is None:
        raise IndicatorError("Bollinger Bands calculation returned None")

    # pandas_ta column names vary by version — handle both
    cols = result.columns.tolist()
    upper = [c for c in cols if c.startswith('BBU')][0]
    mid   = [c for c in cols if c.startswith('BBM')][0]
    lower = [c for c in cols if c.startswith('BBL')][0]
    width = [c for c in cols if c.startswith('BBB')][0]
    pct   = [c for c in cols if c.startswith('BBP')][0]

    return {
        'bb_upper': result[upper],
        'bb_mid':   result[mid],
        'bb_lower': result[lower],
        'bb_width': result[width],
        'bb_pct':   result[pct],
    }


# ── TREND ──────────────────────────────────────────────

def calculate_adx(df: pd.DataFrame, period: int = 14) -> dict:
    """
    Average Directional Index.
    Returns dict with keys: adx, dmp (+DI), dmn (-DI)
    Values 0-100. Measures trend strength not direction.
    """
    if period < 2:
        raise IndicatorError(f"ADX period must be >= 2, got {period}")

    result = ta.adx(df['high'], df['low'], df['close'],
                    length=period)
    if result is None:
        raise IndicatorError("ADX calculation returned None")

    cols = result.columns.tolist()
    adx_col = [c for c in cols if c.startswith('ADX')][0]
    dmp_col = [c for c in cols if c.startswith('DMP')][0]
    dmn_col = [c for c in cols if c.startswith('DMN')][0]

    return {
        'adx': result[adx_col],
        'dmp': result[dmp_col],
        'dmn': result[dmn_col],
    }


# ── VOLUME ─────────────────────────────────────────────

def calculate_volume_sma(df: pd.DataFrame,
                          period: int = 20) -> pd.Series:
    """Simple Moving Average of volume."""
    if period < 1:
        raise IndicatorError(
            f"Volume SMA period must be >= 1, got {period}"
        )
    return ta.sma(df['volume'], length=period)


# ── CONTEXT ────────────────────────────────────────────

def calculate_session(df: pd.DataFrame) -> pd.Series:
    """
    Trading session label for each candle.
    Based on UTC timestamp of candle close.

    Priority order (highest to lowest):
      DEAD_ZONE → LONDON_NY_OVERLAP → LONDON_OPEN
      → NY_OPEN → LONDON → NEW_YORK → ASIAN

    Returns pd.Series of session label strings.
    """
    def get_session(hour: int) -> str:
        if hour >= 21:
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

    hours = df.index.hour
    return pd.Series(
        [get_session(h) for h in hours],
        index=df.index,
        name='session'
    )


def calculate_regime(df: pd.DataFrame,
                     adx_series: pd.Series,
                     atr_series: pd.Series) -> pd.Series:
    """
    Market regime label for each candle.
    Requires pre-computed ADX and ATR Series.

    Classification order:
      QUIET → VOLATILE → TRENDING → RANGING → TRANSITIONING

    Returns pd.Series of regime label strings.
    """
    atr_sma20 = ta.sma(atr_series, length=20)
    atr_ratio = atr_series / atr_sma20

    regimes = []
    for i in range(len(df)):
        adx = adx_series.iloc[i]
        ratio = atr_ratio.iloc[i]

        if pd.isna(adx) or pd.isna(ratio):
            regimes.append(None)
            continue

        if ratio < 0.7:
            regimes.append('QUIET')
        elif ratio > 1.5:
            regimes.append('VOLATILE')
        elif adx > 25:
            regimes.append('TRENDING')
        elif adx < 20:
            regimes.append('RANGING')
        else:
            regimes.append('TRANSITIONING')

    return pd.Series(regimes, index=df.index, name='regime')


# ── CROSSOVER DETECTION ────────────────────────────────

def crossed_above(fast_current: float,
                  fast_prev: float,
                  slow_current: float,
                  slow_prev: float) -> bool:
    """
    True ONLY on the exact candle where fast crosses above slow.
    Previous candle: fast was at or below slow.
    Current candle:  fast is above slow.

    Usage:
        signal = crossed_above(
            cache['sma_fast'], cache['sma_fast_prev'],
            cache['sma_slow'], cache['sma_slow_prev']
        )
    """
    return (fast_prev <= slow_prev) and (fast_current > slow_current)


def crossed_below(fast_current: float,
                  fast_prev: float,
                  slow_current: float,
                  slow_prev: float) -> bool:
    """
    True ONLY on the exact candle where fast crosses below slow.
    Previous candle: fast was at or above slow.
    Current candle:  fast is below slow.
    """
    return (fast_prev >= slow_prev) and (fast_current < slow_current)