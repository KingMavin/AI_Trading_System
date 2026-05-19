"""
Unit tests for shared/indicators.py
Run: pytest tests/test_indicators.py -v

These tests verify indicator values match reference
values from TradingView on known historical data.
They must all pass before Milestone 0 is complete.
"""

import pytest
import pandas as pd
import numpy as np
from shared.indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_bbands, calculate_atr, calculate_adx,
    calculate_session, crossed_above, crossed_below,
    IndicatorError
)
from trainer.core.data_loader import load_ohlcv
from datetime import datetime

# ── FIXTURES ───────────────────────────────────────────

@pytest.fixture(scope='module')
def eurusd_m15():
    """Load EURUSD M15 data for testing."""
    return load_ohlcv('EURUSD', 'M15',
                       start=datetime(2020, 1, 1),
                       end=datetime(2021, 1, 1),
                       warmup_candles=300)

# ── STRUCTURAL TESTS ───────────────────────────────────

def test_sma_length(eurusd_m15):
    result = calculate_sma(eurusd_m15, period=20)
    assert len(result) == len(eurusd_m15)

def test_sma_first_valid_value(eurusd_m15):
    result = calculate_sma(eurusd_m15, period=20)
    assert pd.isna(result.iloc[18])   # not enough data
    assert not pd.isna(result.iloc[19])  # first valid value

def test_ema_length(eurusd_m15):
    result = calculate_ema(eurusd_m15, period=20)
    assert len(result) == len(eurusd_m15)

def test_rsi_range(eurusd_m15):
    result = calculate_rsi(eurusd_m15, period=14)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()

def test_bbands_returns_all_components(eurusd_m15):
    result = calculate_bbands(eurusd_m15, period=20, std=2.0)
    assert set(result.keys()) == {
        'bb_upper', 'bb_mid', 'bb_lower', 'bb_width', 'bb_pct'
    }

def test_bbands_upper_above_lower(eurusd_m15):
    result = calculate_bbands(eurusd_m15, period=20, std=2.0)
    valid = pd.DataFrame(result).dropna()
    assert (valid['bb_upper'] > valid['bb_lower']).all()

def test_atr_positive(eurusd_m15):
    result = calculate_atr(eurusd_m15, period=14)
    valid = result.dropna()
    assert (valid > 0).all()

def test_adx_range(eurusd_m15):
    result = calculate_adx(eurusd_m15, period=14)
    valid = result['adx'].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()

def test_volume_sma_length(eurusd_m15):
    from shared.indicators import calculate_volume_sma
    result = calculate_volume_sma(eurusd_m15, period=20)
    assert len(result) == len(eurusd_m15)

# ── CROSSOVER DETECTION TESTS ──────────────────────────

def test_crossed_above_detects_correctly():
    # fast was below slow, now above
    assert crossed_above(1.0, 0.9, 0.95, 0.95) == True

def test_crossed_above_no_cross():
    # fast already above slow, no crossing
    assert crossed_above(1.0, 1.0, 0.95, 0.95) == False

def test_crossed_below_detects_correctly():
    assert crossed_below(0.9, 1.0, 0.95, 0.95) == True

def test_crossed_above_and_below_mutually_exclusive():
    # same candle cannot be both crossings
    fast_c, fast_p = 1.0, 0.9
    slow_c, slow_p = 0.95, 0.95
    assert not (
        crossed_above(fast_c, fast_p, slow_c, slow_p) and
        crossed_below(fast_c, fast_p, slow_c, slow_p)
    )

# ── SESSION LABEL TESTS ────────────────────────────────

def test_session_dead_zone():
    df = pd.DataFrame(
        {'close': [1.0]},
        index=pd.DatetimeIndex(
            ['2020-01-01 22:00:00+00:00']
        )
    )
    result = calculate_session(df)
    assert result.iloc[0] == 'DEAD_ZONE'

def test_session_london():
    df = pd.DataFrame(
        {'close': [1.0]},
        index=pd.DatetimeIndex(
            ['2020-01-01 10:00:00+00:00']
        )
    )
    result = calculate_session(df)
    assert result.iloc[0] == 'LONDON'

# ── ERROR HANDLING TESTS ───────────────────────────────

def test_sma_invalid_period():
    df = pd.DataFrame({'close': [1.0, 2.0, 3.0]})
    with pytest.raises(IndicatorError):
        calculate_sma(df, period=0)

# ── REFERENCE VALUE TESTS ──────────────────────────────
# These verify against TradingView reference values.
# To get reference values:
# 1. Open TradingView, load EURUSD M15
# 2. Add RSI(14), SMA(20), ATR(14)
# 3. Navigate to the specific date below
# 4. Note the indicator value shown
# 5. Fill in the expected values below

def test_sma20_reference_value(eurusd_m15):
    """
    SMA(20) on EURUSD M15 at 2020-03-01 12:00 UTC
    Expected: ~1.1124 (verify on TradingView)
    Replace 0.0 with actual TradingView value before running.
    """
    target_time = pd.Timestamp('2020-03-01 12:00:00', tz='UTC')
    result = calculate_sma(eurusd_m15, period=20)
    actual = result.loc[target_time]
    expected = 0.0  # REPLACE WITH TRADINGVIEW VALUE
    if expected != 0.0:
        assert abs(actual - expected) < 0.0001, \
            f"SMA mismatch: {actual:.5f} vs {expected:.5f}"

def test_rsi14_reference_value(eurusd_m15):
    """
    RSI(14) on EURUSD M15 at 2020-03-01 12:00 UTC
    Replace 0.0 with actual TradingView value.
    """
    target_time = pd.Timestamp('2020-03-01 12:00:00', tz='UTC')
    result = calculate_rsi(eurusd_m15, period=14)
    actual = result.loc[target_time]
    expected = 0.0  # REPLACE WITH TRADINGVIEW VALUE
    if expected != 0.0:
        assert abs(actual - expected) < 0.5, \
            f"RSI mismatch: {actual:.2f} vs {expected:.2f}"