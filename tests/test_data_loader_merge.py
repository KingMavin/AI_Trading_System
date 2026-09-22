"""
Unit tests for TrainerManifest data merge logic (WAVE 3.4).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from trainer.core.data_loader import merge_historical_and_live_data


def test_merge_disjoint_datasets():
    """Verify merging disjoint historical and live datasets produces a sorted unified DataFrame."""
    times_hist = pd.date_range('2026-01-01', periods=10, freq='15min', tz='UTC')
    df_hist = pd.DataFrame({
        'open': [1.1000] * 10,
        'high': [1.1010] * 10,
        'low': [1.0990] * 10,
        'close': [1.1000] * 10,
        'volume': [100] * 10,
    }, index=times_hist)

    times_live = pd.date_range('2026-01-02', periods=10, freq='15min', tz='UTC')
    df_live = pd.DataFrame({
        'open': [1.1020] * 10,
        'high': [1.1030] * 10,
        'low': [1.1010] * 10,
        'close': [1.1020] * 10,
        'volume': [200] * 10,
    }, index=times_live)

    merged = merge_historical_and_live_data(df_hist, df_live)

    assert len(merged) == 20
    assert merged.index.is_monotonic_increasing
    assert merged.index[0] == pd.Timestamp('2026-01-01 00:00:00+00:00')
    assert merged.index[-1] == times_live[-1]


def test_merge_overlapping_datasets_live_precedence():
    """Verify live dataset values overwrite historical values on overlapping timestamps."""
    times_hist = pd.date_range('2026-01-01 00:00', periods=5, freq='15min', tz='UTC')
    df_hist = pd.DataFrame({
        'open': [1.1000] * 5,
        'high': [1.1010] * 5,
        'low': [1.0990] * 5,
        'close': [1.1000] * 5,
        'volume': [100] * 5,
    }, index=times_hist)

    # Overlaps on last 2 candles of df_hist (00:45, 01:00)
    times_live = pd.date_range('2026-01-01 00:45', periods=4, freq='15min', tz='UTC')
    df_live = pd.DataFrame({
        'open': [1.2000] * 4,
        'high': [1.2010] * 4,
        'low': [1.1990] * 4,
        'close': [1.2000] * 4,
        'volume': [999] * 4,
    }, index=times_live)

    merged = merge_historical_and_live_data(df_hist, df_live)

    # 3 unique hist candles + 4 live candles = 7 total candles
    assert len(merged) == 7
    assert merged.index.is_monotonic_increasing
    # Overlapping timestamp 00:45 should take live close price 1.2000 and volume 999
    assert merged.loc[pd.Timestamp('2026-01-01 00:45:00+00:00', tz='UTC'), 'close'] == 1.2000
    assert merged.loc[pd.Timestamp('2026-01-01 00:45:00+00:00', tz='UTC'), 'volume'] == 999


def test_merge_price_contradiction_warning(caplog):
    """Verify price contradiction warning is logged when close price diff exceeds tolerance, and live value wins."""
    times = pd.date_range('2026-01-01 00:00', periods=3, freq='15min', tz='UTC')
    df_hist = pd.DataFrame({
        'open': [1.1000] * 3,
        'high': [1.1010] * 3,
        'low': [1.0990] * 3,
        'close': [1.1000] * 3,
        'volume': [100] * 3,
    }, index=times)

    df_live = pd.DataFrame({
        'open': [1.1000] * 3,
        'high': [1.1010] * 3,
        'low': [1.0990] * 3,
        'close': [1.1050] * 3,  # 50 pips diff > 0.001 tolerance
        'volume': [100] * 3,
    }, index=times)

    with caplog.at_level('WARNING'):
        merged = merge_historical_and_live_data(df_hist, df_live, price_tolerance=0.001)

    assert "Price contradiction detected" in caplog.text
    assert len(merged) == 3
    # Assert resolved value matches live dataset value (1.1050), not historical (1.1000)
    assert merged.loc[times[0], 'close'] == 1.1050


def test_merge_naive_timestamp_index():
    """Verify naive (tz-naive) DatetimeIndex is correctly localized to UTC and merged without error."""
    times_naive = pd.date_range('2026-01-01 00:00', periods=5, freq='15min')  # Naive (no tz)
    df_hist = pd.DataFrame({
        'open': [1.1000] * 5,
        'high': [1.1010] * 5,
        'low': [1.0990] * 5,
        'close': [1.1000] * 5,
        'volume': [100] * 5,
    }, index=times_naive)

    times_live = pd.date_range('2026-01-01 01:15', periods=5, freq='15min', tz='UTC')  # UTC aware
    df_live = pd.DataFrame({
        'open': [1.1020] * 5,
        'high': [1.1030] * 5,
        'low': [1.1010] * 5,
        'close': [1.1020] * 5,
        'volume': [200] * 5,
    }, index=times_live)

    merged = merge_historical_and_live_data(df_hist, df_live)

    assert merged.index.tz == timezone.utc
    assert len(merged) == 10
    assert merged.index.is_monotonic_increasing
    assert merged.loc[pd.Timestamp('2026-01-01 00:00:00', tz='UTC'), 'close'] == 1.1000


def test_merge_fails_loudly_on_nan_values():
    """Verify loud failure (ValueError) if merged dataset contains NaN values per AGENTS.md §3."""
    times = pd.date_range('2026-01-01 00:00', periods=3, freq='15min', tz='UTC')
    df_hist = pd.DataFrame({
        'open': [1.1000, np.nan, 1.1000],
        'high': [1.1010] * 3,
        'low': [1.0990] * 3,
        'close': [1.1000] * 3,
        'volume': [100] * 3,
    }, index=times)

    df_live = pd.DataFrame({
        'open': [1.1000] * 3,
        'high': [1.1010] * 3,
        'low': [1.0990] * 3,
        'close': [1.1000] * 3,
        'volume': [100] * 3,
    }, index=pd.date_range('2026-01-02', periods=3, freq='15min', tz='UTC'))

    with pytest.raises(ValueError, match="NaN values detected in merged OHLCV dataset"):
        merge_historical_and_live_data(df_hist, df_live)


def test_merge_fails_on_missing_ohlcv_columns():
    """Verify ValueError is raised if input datasets miss core OHLCV columns."""
    df_bad = pd.DataFrame({'close': [1.1000]}, index=pd.date_range('2026-01-01', periods=1, tz='UTC'))
    df_good = pd.DataFrame({
        'open': [1.1000], 'high': [1.1010], 'low': [1.0990], 'close': [1.1000], 'volume': [100]
    }, index=pd.date_range('2026-01-01', periods=1, tz='UTC'))

    with pytest.raises(ValueError, match="Required OHLCV column"):
        merge_historical_and_live_data(df_bad, df_good)
