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
        after_warmup = result.dropna()
        valid = {
            'TRENDING', 'RANGING', 'VOLATILE',
            'QUIET', 'TRANSITIONING'
        }
        assert set(after_warmup.unique()).issubset(valid)


# ── SHARPE RATIO TESTS ─────────────────────────────────

class TestSharpeRatio:

    def test_sharpe_annualization_with_trades_per_year(self):
        from shared.metrics import _calculate_sharpe
        # 100 trades over 1.0 year -> trades_per_year = 100
        # PnLs: mean = 10.0, std = 10.0 -> per-trade Sharpe = 1.0
        # Expected annualized Sharpe = 1.0 * sqrt(100) = 10.0
        pnls = [15.0, 5.0] * 50  # mean=10, std=5.025...
        arr = np.array(pnls)
        expected_mean = np.mean(arr)
        expected_std = np.std(arr, ddof=1)
        expected_sharpe = (expected_mean / expected_std) * np.sqrt(100 / 1.0)
        
        calc_sharpe = _calculate_sharpe(pnls, years=1.0)
        assert calc_sharpe == pytest.approx(expected_sharpe, rel=1e-4)

    def test_sharpe_degenerate_cases(self):
        from shared.metrics import _calculate_sharpe
        assert _calculate_sharpe([]) == 0.0
        assert _calculate_sharpe([10.0]) == 0.0
        assert _calculate_sharpe([10.0, 10.0]) == 0.0  # std == 0
        assert _calculate_sharpe([10.0, -5.0], years=0.0) == 0.0

    def test_sharpe_hand_calculated_values(self):
        from shared.metrics import _calculate_sharpe
        # Hand Calculation:
        # pnls = [100.0, 100.0, 100.0, -50.0] (4 trades), years = 0.5
        # mean = (100 + 100 + 100 - 50) / 4 = 62.5
        # diffs from mean = [37.5, 37.5, 37.5, -112.5]
        # sq_diffs = [1406.25, 1406.25, 1406.25, 12656.25]
        # sum_sq_diffs = 16875.0
        # variance (ddof=1) = 16875.0 / 3 = 5625.0
        # std = sqrt(5625.0) = 75.0
        # per-trade Sharpe = 62.5 / 75.0 = 5/6 = 0.8333333333333334
        # trades_per_year = 4 / 0.5 = 8.0
        # sqrt(trades_per_year) = sqrt(8.0) = 2.8284271247461903
        # annualized_sharpe = (5/6) * sqrt(8.0) = 2.3570226039551585
        pnls = [100.0, 100.0, 100.0, -50.0]
        calc = _calculate_sharpe(pnls, years=0.5)
        assert calc == pytest.approx(2.3570226039551585, rel=1e-6)

    def test_sharpe_annualization_scales_with_years(self):
        from shared.metrics import _calculate_sharpe
        pnls = [100.0, 100.0, 100.0, -50.0]
        s1 = _calculate_sharpe(pnls, years=1.0)
        s2 = _calculate_sharpe(pnls, years=2.0)
        # Doubling years halves trades_per_year, so Sharpe scales by 1/sqrt(2)
        assert s2 == pytest.approx(s1 / np.sqrt(2.0), rel=1e-6)

    def test_sharpe_regression_guard_old_candles_formula(self):
        from shared.metrics import _calculate_sharpe
        # 50 trades in 1 year
        pnls = [100.0, 100.0, 100.0, -50.0] * 12 + [100.0, 100.0]
        calc = _calculate_sharpe(pnls, years=1.0)
        # Old bug formula: (mean / std) * sqrt(24192) = (62.5 / 75.0) * 155.53778 = 129.6148
        # Correct new formula: (mean / std) * sqrt(50) = (62.5 / 75.0) * 7.0710678 = 5.892556
        old_inflated_sharpe = (62.5 / 75.0) * np.sqrt(96 * 252)
        assert calc < 0.1 * old_inflated_sharpe


@pytest.mark.parametrize("t0", [14, 15, 20, 21, 50])
@pytest.mark.parametrize("future_offset", [1, 5])
def test_look_ahead_bias_in_all_indicators(small_df, t0, future_offset):
    """
    Parametrized look-ahead audit test for all indicators in shared/indicators.py.
    Tests multiple cutoff rows (t0 = 14, 15, 20, 21, 50) and future offsets (t+1 immediate next candle, t+5).
    Asserts bitwise exact identity (check_exact=True) for all prior rows t <= t0.
    """
    t_future = t0 + future_offset
    if t_future >= len(small_df):
        pytest.skip("t_future out of bounds")

    # Mutate a copy of small_df at t_future
    df_mutated = small_df.copy()
    df_mutated.iloc[t_future, df_mutated.columns.get_loc('open')] *= 2.0
    df_mutated.iloc[t_future, df_mutated.columns.get_loc('high')] *= 2.0
    df_mutated.iloc[t_future, df_mutated.columns.get_loc('low')] *= 0.5
    df_mutated.iloc[t_future, df_mutated.columns.get_loc('close')] *= 2.0
    df_mutated.iloc[t_future, df_mutated.columns.get_loc('volume')] *= 10.0

    # 1. SMA
    sma_orig = calculate_sma(small_df, 14)
    sma_mut = calculate_sma(df_mutated, 14)
    pd.testing.assert_series_equal(sma_orig.iloc[:t0+1], sma_mut.iloc[:t0+1], check_exact=True)

    # 2. EMA
    ema_orig = calculate_ema(small_df, 14)
    ema_mut = calculate_ema(df_mutated, 14)
    pd.testing.assert_series_equal(ema_orig.iloc[:t0+1], ema_mut.iloc[:t0+1], check_exact=True)

    # 3. RSI
    rsi_orig = calculate_rsi(small_df, 14)
    rsi_mut = calculate_rsi(df_mutated, 14)
    pd.testing.assert_series_equal(rsi_orig.iloc[:t0+1], rsi_mut.iloc[:t0+1], check_exact=True)

    # 4. MACD
    macd_orig = calculate_macd(small_df, 12, 26, 9)
    macd_mut = calculate_macd(df_mutated, 12, 26, 9)
    for k in macd_orig:
        pd.testing.assert_series_equal(macd_orig[k].iloc[:t0+1], macd_mut[k].iloc[:t0+1], check_exact=True)

    # 5. ATR
    atr_orig = calculate_atr(small_df, 14)
    atr_mut = calculate_atr(df_mutated, 14)
    pd.testing.assert_series_equal(atr_orig.iloc[:t0+1], atr_mut.iloc[:t0+1], check_exact=True)

    # 6. Bollinger Bands
    bb_orig = calculate_bbands(small_df, 20, 2.0)
    bb_mut = calculate_bbands(df_mutated, 20, 2.0)
    for k in bb_orig:
        pd.testing.assert_series_equal(bb_orig[k].iloc[:t0+1], bb_mut[k].iloc[:t0+1], check_exact=True)

    # 7. ADX
    adx_orig = calculate_adx(small_df, 14)
    adx_mut = calculate_adx(df_mutated, 14)
    for k in adx_orig:
        pd.testing.assert_series_equal(adx_orig[k].iloc[:t0+1], adx_mut[k].iloc[:t0+1], check_exact=True)

    # 8. Volume SMA
    v_orig = calculate_volume_sma(small_df, 20)
    v_mut = calculate_volume_sma(df_mutated, 20)
    pd.testing.assert_series_equal(v_orig.iloc[:t0+1], v_mut.iloc[:t0+1], check_exact=True)

    # 9. Session
    sess_orig = calculate_session(small_df)
    sess_mut = calculate_session(df_mutated)
    pd.testing.assert_series_equal(sess_orig.iloc[:t0+1], sess_mut.iloc[:t0+1], check_exact=True)

    # 10. Regime
    reg_orig = calculate_regime(small_df, adx_orig['adx'], atr_orig)
    reg_mut = calculate_regime(df_mutated, adx_mut['adx'], atr_mut)
    pd.testing.assert_series_equal(reg_orig.iloc[:t0+1], reg_mut.iloc[:t0+1], check_exact=True)

