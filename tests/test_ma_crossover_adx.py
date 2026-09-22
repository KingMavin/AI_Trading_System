import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from trainer.signals.ma_crossover import generate_signal


def test_adx_threshold_disabled():
    """When adx_threshold == 0, ADX filter is disabled and signals fire normally."""
    cache = {
        'sma_fast': 1.1050,
        'sma_fast_prev': 1.1000,
        'sma_slow': 1.1020,
        'sma_slow_prev': 1.1020,
        'atr': 0.0010,
        'close': 1.1050,
        'adx': 10.0,  # Low ADX, but threshold is 0
    }
    params = {
        'fast_ma_period': 10,
        'slow_ma_period': 50,
        'sl_atr_multiple': 1.5,
        'tp_rr_ratio': 2.0,
        'adx_threshold': 0,
        'pip_size': 0.0001,
    }
    res = generate_signal(cache, params)
    assert res['signal'] == 'BUY'
    assert 'MA_CROSS_BULL' in res['reason']


def test_adx_threshold_suppresses_weak_trend():
    """When adx_threshold == 25 and ADX < 25, signal is suppressed to HOLD."""
    cache = {
        'sma_fast': 1.1050,
        'sma_fast_prev': 1.1000,
        'sma_slow': 1.1020,
        'sma_slow_prev': 1.1020,
        'atr': 0.0010,
        'close': 1.1050,
        'adx': 18.5,  # 18.5 < 25.0
    }
    params = {
        'fast_ma_period': 10,
        'slow_ma_period': 50,
        'sl_atr_multiple': 1.5,
        'tp_rr_ratio': 2.0,
        'adx_threshold': 25,
        'pip_size': 0.0001,
    }
    res = generate_signal(cache, params)
    assert res['signal'] == 'HOLD'
    assert 'below threshold' in res['reason']


def test_adx_threshold_allows_strong_trend():
    """When adx_threshold == 25 and ADX >= 25, signal passes normally."""
    cache = {
        'sma_fast': 1.1050,
        'sma_fast_prev': 1.1000,
        'sma_slow': 1.1020,
        'sma_slow_prev': 1.1020,
        'atr': 0.0010,
        'close': 1.1050,
        'adx': 28.2,  # 28.2 >= 25.0
    }
    params = {
        'fast_ma_period': 10,
        'slow_ma_period': 50,
        'sl_atr_multiple': 1.5,
        'tp_rr_ratio': 2.0,
        'adx_threshold': 25,
        'pip_size': 0.0001,
    }
    res = generate_signal(cache, params)
    assert res['signal'] == 'BUY'


def test_adx_threshold_nan_during_warmup():
    """When ADX is NaN under active adx_threshold, signal is fail-safe suppressed to HOLD."""
    cache = {
        'sma_fast': 1.1050,
        'sma_fast_prev': 1.1000,
        'sma_slow': 1.1020,
        'sma_slow_prev': 1.1020,
        'atr': 0.0010,
        'close': 1.1050,
        'adx': np.nan,
    }
    params = {
        'fast_ma_period': 10,
        'slow_ma_period': 50,
        'sl_atr_multiple': 1.5,
        'tp_rr_ratio': 2.0,
        'adx_threshold': 20,
        'pip_size': 0.0001,
    }
    res = generate_signal(cache, params)
    assert res['signal'] == 'HOLD'
    assert 'NaN ADX indicator' in res['reason']
