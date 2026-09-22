"""
Walk-Forward Data Leakage Audit Test (WAVE 4.3).
Verifies that mutating out-of-sample (OOS) data does NOT alter
in-sample (IS) indicator calculations, parameter optimization choices,
or IS performance metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from trainer.core.walk_forward import WalkForwardValidator
from shared.instrument_spec import InstrumentSpec


@pytest.fixture
def mock_spec():
    spec = InstrumentSpec.build_simulation_default("EURUSD")
    spec.sanity_ok = True
    with patch("shared.instrument_spec.get_spec", return_value=spec):
        yield spec


def test_walk_forward_oos_data_isolation(mock_spec):
    """
    Construct a 4.5-month hourly dataset.
    Run WalkForwardValidator on original data.
    Mutate candles in the last 750 rows (OOS period) with extreme price spikes.
    Run WalkForwardValidator on mutated data.
    Assert IS parameters, IS composite scores, IS trades, and IS metrics are bitwise identical.
    """
    dates = pd.date_range('2020-01-01 00:00', periods=3300, freq='1h', tz='UTC')
    N = len(dates)
    t = np.linspace(0, 60 * np.pi, N)
    closes = 1.1000 + 0.0050 * np.sin(t)

    df_orig = pd.DataFrame({
        'open':   closes - 0.0001,
        'high':   closes + 0.0002,
        'low':    closes - 0.0002,
        'close':  closes,
        'volume': [100.0] * N,
    }, index=dates)

    df_mutated = df_orig.copy()
    # Mutate all candles in the last 750 rows (OOS period) with massive price spikes
    df_mutated.iloc[N-750:, df_mutated.columns.get_loc('open')] *= 2.5
    df_mutated.iloc[N-750:, df_mutated.columns.get_loc('high')] *= 3.0
    df_mutated.iloc[N-750:, df_mutated.columns.get_loc('low')] *= 0.5
    df_mutated.iloc[N-750:, df_mutated.columns.get_loc('close')] *= 2.5
    df_mutated.iloc[N-750:, df_mutated.columns.get_loc('volume')] *= 100.0

    test_param = {
        'fast_ma_period': 5,
        'slow_ma_period': 15,
        'sl_atr_multiple': 1.0,
        'tp_rr_ratio': 1.5,
        'warmup_candles': 50,
        'pip_size': 0.0001
    }

    wfv_orig = WalkForwardValidator(
        symbol='EURUSD',
        df=df_orig,
        opt_months=2,
        test_months=1,
        min_trades_opt=1,
        min_trades_test=1
    )
    wfv_orig.param_grid = [test_param]
    res_orig = wfv_orig.run()

    wfv_mutated = WalkForwardValidator(
        symbol='EURUSD',
        df=df_mutated,
        opt_months=2,
        test_months=1,
        min_trades_opt=1,
        min_trades_test=1
    )
    wfv_mutated.param_grid = [test_param]
    res_mutated = wfv_mutated.run()

    assert len(res_orig) > 0
    assert len(res_orig) == len(res_mutated)

    for i in range(len(res_orig)):
        r1 = res_orig[i]
        r2 = res_mutated[i]

        # IS metrics must be 100% identical regardless of OOS mutations
        assert r1.best_params == r2.best_params
        assert r1.is_trades == r2.is_trades
        assert r1.is_profit_factor == r2.is_profit_factor
        assert r1.is_calmar == r2.is_calmar
        assert r1.is_net_profit == r2.is_net_profit
        assert r1.is_composite_score == r2.is_composite_score

        # For last window (where OOS hits mutated region), OOS metrics differ, proving mutation took effect
        if i == len(res_orig) - 1:
            assert r1.oos_net_profit != r2.oos_net_profit


def test_walk_forward_disjoint_window_boundaries(mock_spec):
    """
    Verify that opt_end and test_start window boundaries created by _build_windows
    are non-overlapping (opt_end == test_start, and test_end = test_start + test_months).
    """
    dates = pd.date_range('2020-01-01 00:00', periods=3300, freq='1h', tz='UTC')
    N = len(dates)
    df = pd.DataFrame({
        'open': [1.1000]*N, 'high': [1.1010]*N,
        'low': [1.0990]*N, 'close': [1.1000]*N,
        'volume': [100.0]*N
    }, index=dates)

    wfv = WalkForwardValidator(symbol='EURUSD', df=df, opt_months=2, test_months=1)
    windows = wfv._build_windows()

    assert len(windows) > 0
    for w in windows:
        assert w['opt_end'] == w['test_start']
        assert w['opt_start'] < w['opt_end']
        assert w['test_start'] < w['test_end']


def test_walk_forward_forward_leak_isolation(mock_spec):
    """
    Forward-leak audit test (WAVE 4.3).
    Mutates an early window's OOS region (e.g. rows 1440-2160 in Month 3),
    and asserts that an early window's IS optimization results are bitwise identical
    between original and mutated datasets.
    """
    dates = pd.date_range('2020-01-01 00:00', periods=5000, freq='1h', tz='UTC')
    N = len(dates)
    t = np.linspace(0, 80 * np.pi, N)
    closes = 1.1000 + 0.0050 * np.sin(t)

    df_orig = pd.DataFrame({
        'open':   closes - 0.0001,
        'high':   closes + 0.0002,
        'low':    closes - 0.0002,
        'close':  closes,
        'volume': [100.0] * N,
    }, index=dates)

    df_mutated = df_orig.copy()
    # Mutate rows 1440 to 2160 (Month 3 OOS test window for Window 1)
    df_mutated.iloc[1440:2160, df_mutated.columns.get_loc('open')] *= 4.0
    df_mutated.iloc[1440:2160, df_mutated.columns.get_loc('high')] *= 4.0
    df_mutated.iloc[1440:2160, df_mutated.columns.get_loc('close')] *= 4.0

    test_param = {
        'fast_ma_period': 5,
        'slow_ma_period': 15,
        'sl_atr_multiple': 1.0,
        'tp_rr_ratio': 1.5,
        'warmup_candles': 50,
        'pip_size': 0.0001
    }

    wfv_orig = WalkForwardValidator(symbol='EURUSD', df=df_orig, opt_months=2, test_months=1, min_trades_opt=1, min_trades_test=1)
    wfv_orig.param_grid = [test_param]
    res_orig = wfv_orig.run()

    wfv_mutated = WalkForwardValidator(symbol='EURUSD', df=df_mutated, opt_months=2, test_months=1, min_trades_opt=1, min_trades_test=1)
    wfv_mutated.param_grid = [test_param]
    res_mutated = wfv_mutated.run()

    # Window 1's OOS results will differ (because window 1 OOS hits the mutated region)
    assert res_orig[0].oos_net_profit != res_mutated[0].oos_net_profit

    # Window 1's IS results must be bitwise identical
    assert res_orig[0].is_net_profit == res_mutated[0].is_net_profit
    assert res_orig[0].best_params == res_mutated[0].best_params
