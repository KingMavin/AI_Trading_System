import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from trainer.signals.rsi_reversion import generate_signal, check_exit
from trainer.core.parameter_grid import get_grid, TEMPLATE_SPACES


def test_rsi_reversion_buy_signal():
    """Verify BUY signal when RSI crosses back above oversold threshold."""
    cache = {
        'rsi': 32.0,
        'rsi_prev': 28.0,
        'atr': 0.0010,
        'close': 1.1000,
        'adx': 25.0,
    }
    params = {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'sl_atr_multiple': 1.5,
        'tp_rr_ratio': 2.0,
        'pip_size': 0.0001,
        'adx_threshold': 0,
    }
    res = generate_signal(cache, params)
    assert res['signal'] == 'BUY'
    assert 'RSI_OVERSOLD_REVERSION' in res['reason']
    assert pytest.approx(res['sl_price']) == 1.1000 - 0.0015
    assert pytest.approx(res['tp_price']) == 1.1000 + 0.0030


def test_rsi_reversion_sell_signal():
    """Verify SELL signal when RSI crosses back below overbought threshold."""
    cache = {
        'rsi': 68.0,
        'rsi_prev': 72.0,
        'atr': 0.0010,
        'close': 1.1000,
        'adx': 25.0,
    }
    params = {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'sl_atr_multiple': 1.5,
        'tp_rr_ratio': 2.0,
        'pip_size': 0.0001,
        'adx_threshold': 0,
    }
    res = generate_signal(cache, params)
    assert res['signal'] == 'SELL'
    assert 'RSI_OVERBOUGHT_REVERSION' in res['reason']
    assert pytest.approx(res['sl_price']) == 1.1000 + 0.0015
    assert pytest.approx(res['tp_price']) == 1.1000 - 0.0030


def test_rsi_reversion_midline_exit():
    """Verify check_exit returns should_exit=True when RSI crosses midline 50.0."""
    position_buy = {'direction': 'BUY'}
    position_sell = {'direction': 'SELL'}

    params_exit_true = {'exit_on_midline': True}
    params_exit_false = {'exit_on_midline': False}

    # BUY position exits when RSI >= 50
    res1 = check_exit({'rsi': 51.0}, position_buy, params_exit_true)
    assert res1['should_exit'] is True
    assert res1['reason'] == 'EXIT_RSI_MIDLINE'

    res2 = check_exit({'rsi': 48.0}, position_buy, params_exit_true)
    assert res2['should_exit'] is False

    # SELL position exits when RSI <= 50
    res3 = check_exit({'rsi': 49.0}, position_sell, params_exit_true)
    assert res3['should_exit'] is True
    assert res3['reason'] == 'EXIT_RSI_MIDLINE'

    res4 = check_exit({'rsi': 52.0}, position_sell, params_exit_true)
    assert res4['should_exit'] is False

    # Disabled exit_on_midline
    res5 = check_exit({'rsi': 55.0}, position_buy, params_exit_false)
    assert res5['should_exit'] is False


def test_rsi_reversion_nan_fail_closed():
    """Verify HOLD signal returned on NaN indicator inputs or low ADX."""
    params = {'rsi_oversold': 30, 'rsi_overbought': 70, 'adx_threshold': 20}

    # NaN RSI
    res_nan = generate_signal({'rsi': None, 'rsi_prev': 28.0, 'atr': 0.0010, 'close': 1.1000}, params)
    assert res_nan['signal'] == 'HOLD'
    assert 'NaN indicator' in res_nan['reason']

    # ADX below threshold
    res_adx = generate_signal({'rsi': 32.0, 'rsi_prev': 28.0, 'atr': 0.0010, 'close': 1.1000, 'adx': 15.0}, params)
    assert res_adx['signal'] == 'HOLD'
    assert 'below threshold' in res_adx['reason']


def test_rsi_reversion_parameter_grid():
    """Verify parameter grid generation and constraints for rsi_reversion."""
    grid = get_grid('rsi_reversion')
    assert len(grid) > 0
    assert 'rsi_reversion' in TEMPLATE_SPACES

    for combo in grid:
        assert combo['rsi_oversold'] < combo['rsi_overbought']
        assert combo['risk_per_trade_pct'] == 1.0
        assert combo['exit_on_opposite_crossover'] is False
        assert combo['warmup_candles'] == 250
