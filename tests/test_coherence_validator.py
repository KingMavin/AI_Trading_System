"""
Unit tests for WAVE 11.1 — Coherence Validator.
Verifies rejection of degenerate parameter configurations and gating prior to WalkForwardValidator backtest execution.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.core.coherence_validator import validate_coherence
from shared.instrument_spec import InstrumentSpec
from trainer.core.walk_forward import WalkForwardValidator


class TestCoherenceValidator:

    def test_fast_ma_greater_equal_slow_ma_rejected(self):
        # fast >= slow
        params = {'fast_ma_period': 50, 'slow_ma_period': 20, 'sl_atr_multiple': 1.5, 'tp_rr_ratio': 2.0}
        valid, reason = validate_coherence(params)
        assert not valid
        assert "Fast MA period (50) must be strictly less than Slow MA period (20)" in reason

        # fast == slow
        params_equal = {'fast_ma_period': 20, 'slow_ma_period': 20, 'sl_atr_multiple': 1.5, 'tp_rr_ratio': 2.0}
        valid_eq, reason_eq = validate_coherence(params_equal)
        assert not valid_eq
        assert "Fast MA period (20) must be strictly less than Slow MA period (20)" in reason_eq

    def test_non_positive_parameters_rejected(self):
        # sl_atr <= 0
        params_sl = {'fast_ma_period': 10, 'slow_ma_period': 50, 'sl_atr_multiple': 0.0, 'tp_rr_ratio': 2.0}
        valid_sl, reason_sl = validate_coherence(params_sl)
        assert not valid_sl
        assert "sl_atr_multiple must be > 0" in reason_sl

        # tp_rr <= 0
        params_tp = {'fast_ma_period': 10, 'slow_ma_period': 50, 'sl_atr_multiple': 1.5, 'tp_rr_ratio': -0.5}
        valid_tp, reason_tp = validate_coherence(params_tp)
        assert not valid_tp
        assert "tp_rr_ratio must be > 0" in reason_tp

    def test_stops_level_violation_rejected(self):
        spec = InstrumentSpec(
            symbol="EURUSD",
            digits=5,
            point=0.00001,
            contract_size=100000.0,
            tick_size=0.00001,
            tick_value=1.0,
            volume_min=0.01,
            volume_step=0.01,
            volume_max=100.0,
            stops_level=500,  # 50 pips minimum stop distance
            spread_typical=10,
            swap_long=0.0,
            swap_short=0.0,
            currency_base="EUR",
            currency_profit="USD",
            currency_margin="EUR",
            account_currency="USD",
            captured_at="2026-07-31T00:00:00Z",
            sanity_ok=True
        )
        # Very tiny SL multiple (0.01 ATR * 0.0010 = 0.00001 = 1 point < 500 points)
        params = {'fast_ma_period': 10, 'slow_ma_period': 50, 'sl_atr_multiple': 0.01, 'tp_rr_ratio': 2.0}
        valid, reason = validate_coherence(params, instrument_spec=spec, historical_atr=0.0010)
        assert not valid
        assert "violates broker stops_level" in reason

    def test_valid_parameters_pass(self):
        params = {'fast_ma_period': 10, 'slow_ma_period': 50, 'sl_atr_multiple': 1.5, 'tp_rr_ratio': 2.0}
        valid, reason = validate_coherence(params)
        assert valid
        assert reason is None

    def test_walk_forward_optimise_skips_coherence_rejected_candidates(self, monkeypatch):
        # Create minimal dummy dataframe
        dates = pd.date_range('2020-01-01', periods=100, freq='h', tz='UTC')
        df = pd.DataFrame({
            'open': 1.1000, 'high': 1.1050, 'low': 1.0950, 'close': 1.1010, 'volume': 100
        }, index=dates)

        wf = WalkForwardValidator('EURUSD', df)
        wf.param_grid = [
            {'fast_ma_period': 50, 'slow_ma_period': 20, 'sl_atr_multiple': 1.5, 'tp_rr_ratio': 2.0},  # Degenerate (fast >= slow)
            {'fast_ma_period': 10, 'slow_ma_period': 50, 'sl_atr_multiple': 1.5, 'tp_rr_ratio': 2.0},  # Valid
            {'fast_ma_period': 10, 'slow_ma_period': 50, 'sl_atr_multiple': 0.0, 'tp_rr_ratio': 2.0},  # Degenerate (sl <= 0)
        ]

        run_one_mock = MagicMock(return_value={
            'total_trades': 30, 'profit_factor': 1.5, 'calmar_ratio': 1.2, 'win_rate': 0.55
        })
        monkeypatch.setattr(wf, '_run_one', run_one_mock)

        best_params, best_score, best_metrics = wf._optimise(df, window_num=1)

        # Confirm _run_one was executed ONLY ONCE (for the single valid parameter set)
        assert run_one_mock.call_count == 1
        assert best_params['fast_ma_period'] == 10
        assert best_params['slow_ma_period'] == 50
