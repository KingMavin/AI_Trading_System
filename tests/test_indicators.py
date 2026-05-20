"""
Unit tests for shared/indicators.py
Run: pytest tests/test_indicators.py -v

Completion gate for Milestone 0:
All tests must pass before moving to Milestone 1.
"""

import sys
from pathlib import Path

# Add project root to path so shared/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from shared.indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_bbands, calculate_atr, calculate_adx,
    calculate_macd, calculate_volume_sma,
    calculate_session, calculate_regime,
    crossed_above, crossed_below,
    IndicatorError
)


# ── FIXTURES ───────────────────────────────────────────

@pytest.fixture(scope='module')
def eurusd_m15():
    """Load EURUSD M15 data for indicator testing."""
    parquet_path = (
        Path(__file__).parent.parent
        / 'trainer' / 'trainer_data' / 'historical'
        / 'EURUSD' / 'M15.parquet'
    )
    df = pd.read_parquet(parquet_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').sort_index()

    # Use a specific known period for reproducibility
    df = df.loc['2020-01-01':'2021-01-01']
    return df


@pytest.fixture(scope='module')
def small_df():
    """Small synthetic DataFrame for edge case testing."""
    dates = pd.date_range(
        '2020-01-01', periods=100, freq='15min', tz='UTC'
    )
    np.random.seed(42)
    close = 1.1000 + np.cumsum(np.random.randn(100) * 0.0005)
    high  = close + np.abs(np.random.randn(100) * 0.0003)
    low   = close - np.abs(np.random.randn(100) * 0.0003)
    open_ = close + np.random.randn(100) * 0.0002
    vol   = np.random.randint(100, 1000, 100).astype(float)

    return pd.DataFrame({
        'open':   open_,
        'high':   high,
        'low':    low,
        'close':  close,
        'volume': vol,
    }, index=dates)


# ── SMA TESTS ──────────────────────────────────────────

class TestSMA:
    def test_returns_series(self, eurusd_m15):
        result = calculate_sma(eurusd_m15, period=20)
        assert isinstance(result, pd.Series)

    def test_same_length_as_input(self, eurusd_m15):
        result = calculate_sma(eurusd_m15, period=20)
        assert len(result) == len(eurusd_m15)

    def test_warmup_period_is_nan(self, small_df):
        result = calculate_sma(small_df, period=20)
        # First 19 rows should be NaN
        assert result.iloc[:19].isna().all()

    def test_first_valid_value(self, small_df):
        result = calculate_sma(small_df, period=20)
        # Row 19 (index 19) should be the first valid value
        assert not pd.isna(result.iloc[19])

    def test_value_is_mean_of_period(self, small_df):
        period = 10
        result = calculate_sma(small_df, period=period)
        # Manually compute expected value at row 9
        expected = small_df['close'].iloc[:period].mean()
        actual = result.iloc[period - 1]
        assert abs(actual - expected) < 1e-8

    def test_invalid_period_raises(self, small_df):
        with pytest.raises(IndicatorError):
            calculate_sma(small_df, period=0)


# ── EMA TESTS ──────────────────────────────────────────

class TestEMA:
    def test_returns_series(self, eurusd_m15):
        result = calculate_ema(eurusd_m15, period=20)
        assert isinstance(result, pd.Series)

    def test_same_length(self, eurusd_m15):
        result = calculate_ema(eurusd_m15, period=20)
        assert len(result) == len(eurusd_m15)

    def test_invalid_period_raises(self, small_df):
        with pytest.raises(IndicatorError):
            calculate_ema(small_df, period=0)


# ── RSI TESTS ──────────────────────────────────────────

class TestRSI:
    def test_returns_series(self, eurusd_m15):
        result = calculate_rsi(eurusd_m15, period=14)
        assert isinstance(result, pd.Series)

    def test_values_in_range(self, eurusd_m15):
        result = calculate_rsi(eurusd_m15, period=14)
        valid = result.dropna()
        assert (valid >= 0).all(), "RSI below 0 found"
        assert (valid <= 100).all(), "RSI above 100 found"

    def test_same_length(self, eurusd_m15):
        result = calculate_rsi(eurusd_m15, period=14)
        assert len(result) == len(eurusd_m15)

    def test_warmup_is_nan(self, small_df):
        result = calculate_rsi(small_df, period=14)
        # pandas_ta RSI only produces NaN on row 0
        # (uses Wilder smoothing which initialises from first value)
        assert pd.isna(result.iloc[0])
        # Values after warmup should be valid 0-100
        assert not pd.isna(result.iloc[20])

    def test_invalid_period_raises(self, small_df):
        with pytest.raises(IndicatorError):
            calculate_rsi(small_df, period=1)


# ── BOLLINGER BANDS TESTS ──────────────────────────────

class TestBBands:
    def test_returns_dict(self, eurusd_m15):
        result = calculate_bbands(eurusd_m15, period=20, std=2.0)
        assert isinstance(result, dict)

    def test_all_keys_present(self, eurusd_m15):
        result = calculate_bbands(eurusd_m15, period=20, std=2.0)
        assert set(result.keys()) == {
            'bb_upper', 'bb_mid', 'bb_lower', 'bb_width', 'bb_pct'
        }

    def test_upper_above_lower(self, eurusd_m15):
        result = calculate_bbands(eurusd_m15, period=20, std=2.0)
        valid = pd.DataFrame(result).dropna()
        assert (valid['bb_upper'] > valid['bb_lower']).all()

    def test_mid_between_bands(self, eurusd_m15):
        result = calculate_bbands(eurusd_m15, period=20, std=2.0)
        valid = pd.DataFrame(result).dropna()
        assert (valid['bb_mid'] <= valid['bb_upper']).all()
        assert (valid['bb_mid'] >= valid['bb_lower']).all()

    def test_invalid_period_raises(self, small_df):
        with pytest.raises(IndicatorError):
            calculate_bbands(small_df, period=1)

    def test_invalid_std_raises(self, small_df):
        with pytest.raises(IndicatorError):
            calculate_bbands(small_df, period=20, std=0)


# ── ATR TESTS ──────────────────────────────────────────

class TestATR:
    def test_returns_series(self, eurusd_m15):
        result = calculate_atr(eurusd_m15, period=14)
        assert isinstance(result, pd.Series)

    def test_always_positive(self, eurusd_m15):
        result = calculate_atr(eurusd_m15, period=14)
        valid = result.dropna()
        assert (valid > 0).all(), "ATR must always be positive"

    def test_same_length(self, eurusd_m15):
        result = calculate_atr(eurusd_m15, period=14)
        assert len(result) == len(eurusd_m15)


# ── ADX TESTS ──────────────────────────────────────────

class TestADX:
    def test_returns_dict(self, eurusd_m15):
        result = calculate_adx(eurusd_m15, period=14)
        assert isinstance(result, dict)

    def test_all_keys_present(self, eurusd_m15):
        result = calculate_adx(eurusd_m15, period=14)
        assert set(result.keys()) == {'adx', 'dmp', 'dmn'}

    def test_adx_in_range(self, eurusd_m15):
        result = calculate_adx(eurusd_m15, period=14)
        valid = result['adx'].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_same_length(self, eurusd_m15):
        result = calculate_adx(eurusd_m15, period=14)
        assert len(result['adx']) == len(eurusd_m15)


# ── VOLUME SMA TESTS ───────────────────────────────────

class TestVolumeSMA:
    def test_returns_series(self, eurusd_m15):
        result = calculate_volume_sma(eurusd_m15, period=20)
        assert isinstance(result, pd.Series)

    def test_same_length(self, eurusd_m15):
        result = calculate_volume_sma(eurusd_m15, period=20)
        assert len(result) == len(eurusd_m15)

    def test_values_positive(self, eurusd_m15):
        result = calculate_volume_sma(eurusd_m15, period=20)
        valid = result.dropna()
        assert (valid >= 0).all()


# ── SESSION TESTS ──────────────────────────────────────

class TestSession:
    def test_returns_series(self, eurusd_m15):
        result = calculate_session(eurusd_m15)
        assert isinstance(result, pd.Series)

    def test_same_length(self, eurusd_m15):
        result = calculate_session(eurusd_m15)
        assert len(result) == len(eurusd_m15)

    def test_valid_labels_only(self, eurusd_m15):
        valid_sessions = {
            'ASIAN', 'LONDON_OPEN', 'LONDON',
            'NY_OPEN', 'NEW_YORK', 'LONDON_NY_OVERLAP',
            'DEAD_ZONE'
        }
        result = calculate_session(eurusd_m15)
        assert set(result.unique()).issubset(valid_sessions)

    def test_dead_zone_at_22_utc(self):
        dates = pd.DatetimeIndex(
            ['2020-01-01 22:00:00+00:00']
        )
        df = pd.DataFrame(
            {'open': [1.0], 'high': [1.0],
             'low': [1.0], 'close': [1.0], 'volume': [100.0]},
            index=dates
        )
        result = calculate_session(df)
        assert result.iloc[0] == 'DEAD_ZONE'

    def test_london_at_10_utc(self):
        dates = pd.DatetimeIndex(
            ['2020-01-01 10:00:00+00:00']
        )
        df = pd.DataFrame(
            {'open': [1.0], 'high': [1.0],
             'low': [1.0], 'close': [1.0], 'volume': [100.0]},
            index=dates
        )
        result = calculate_session(df)
        assert result.iloc[0] == 'LONDON'

    def test_asian_at_02_utc(self):
        dates = pd.DatetimeIndex(
            ['2020-01-01 02:00:00+00:00']
        )
        df = pd.DataFrame(
            {'open': [1.0], 'high': [1.0],
             'low': [1.0], 'close': [1.0], 'volume': [100.0]},
            index=dates
        )
        result = calculate_session(df)
        assert result.iloc[0] == 'ASIAN'

    def test_london_ny_overlap_at_14_utc(self):
        dates = pd.DatetimeIndex(
            ['2020-01-01 14:00:00+00:00']
        )
        df = pd.DataFrame(
            {'open': [1.0], 'high': [1.0],
             'low': [1.0], 'close': [1.0], 'volume': [100.0]},
            index=dates
        )
        result = calculate_session(df)
        assert result.iloc[0] == 'LONDON_NY_OVERLAP'


# ── CROSSOVER TESTS ────────────────────────────────────

class TestCrossover:
    def test_crossed_above_detects_correctly(self):
        # fast was below slow, now above
        assert crossed_above(
            fast_current=1.10, fast_prev=0.90,
            slow_current=1.00, slow_prev=1.00
        ) is True

    def test_crossed_above_no_cross_stays_above(self):
        # fast already above slow last candle — no cross
        assert crossed_above(
            fast_current=1.10, fast_prev=1.05,
            slow_current=1.00, slow_prev=1.00
        ) is False

    def test_crossed_above_no_cross_stays_below(self):
        # fast stays below slow — no cross
        assert crossed_above(
            fast_current=0.90, fast_prev=0.85,
            slow_current=1.00, slow_prev=1.00
        ) is False

    def test_crossed_below_detects_correctly(self):
        # fast was above slow, now below
        assert crossed_below(
            fast_current=0.90, fast_prev=1.10,
            slow_current=1.00, slow_prev=1.00
        ) is True

    def test_crossed_below_no_cross_stays_below(self):
        assert crossed_below(
            fast_current=0.90, fast_prev=0.85,
            slow_current=1.00, slow_prev=1.00
        ) is False

    def test_mutually_exclusive(self):
        # Same candle cannot trigger both crossovers
        fc, fp = 1.10, 0.90
        sc, sp = 1.00, 1.00
        above = crossed_above(fc, fp, sc, sp)
        below = crossed_below(fc, fp, sc, sp)
        assert not (above and below)

    def test_crossed_above_at_equal_values(self):
        # fast equals slow on prev — touching counts as cross
        assert crossed_above(
            fast_current=1.05, fast_prev=1.00,
            slow_current=1.00, slow_prev=1.00
        ) is True

    def test_crossed_below_at_equal_values(self):
        assert crossed_below(
            fast_current=0.95, fast_prev=1.00,
            slow_current=1.00, slow_prev=1.00
        ) is True


# ── REGIME TESTS ───────────────────────────────────────

class TestRegime:
    def test_returns_series(self, eurusd_m15):
        atr = calculate_atr(eurusd_m15, period=14)
        adx = calculate_adx(eurusd_m15, period=14)['adx']
        result = calculate_regime(eurusd_m15, adx, atr)
        assert isinstance(result, pd.Series)

    def test_valid_labels_only(self, eurusd_m15):
        atr = calculate_atr(eurusd_m15, period=14)
        adx = calculate_adx(eurusd_m15, period=14)['adx']
        result = calculate_regime(eurusd_m15, adx, atr)
        valid_strings = {
            'TRENDING', 'RANGING', 'VOLATILE',
            'QUIET', 'TRANSITIONING'
        }
        # Filter out NaN/None values, check only string values
        string_values = set(
            v for v in result.unique()
            if isinstance(v, str)
        )
        assert string_values.issubset(valid_strings)

    def test_same_length(self, eurusd_m15):
        atr = calculate_atr(eurusd_m15, period=14)
        adx = calculate_adx(eurusd_m15, period=14)['adx']
        result = calculate_regime(eurusd_m15, adx, atr)
        assert len(result) == len(eurusd_m15)

    def test_no_unexpected_values(self, eurusd_m15):
        atr = calculate_atr(eurusd_m15, period=14)
        adx = calculate_adx(eurusd_m15, period=14)['adx']
        result = calculate_regime(eurusd_m15, adx, atr)
        # After warmup all values should be valid strings
        after_warmup = result.dropna()
        valid = {
            'TRENDING', 'RANGING', 'VOLATILE',
            'QUIET', 'TRANSITIONING'
        }
        assert set(after_warmup.unique()).issubset(valid)