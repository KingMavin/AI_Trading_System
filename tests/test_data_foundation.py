"""
Unit tests for WAVE 12 — Data Foundation Completion.
Verifies pip_size resolution for XAUUSD, USDJPY, and EURUSD via InstrumentSpec without hardcoded fallbacks.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.instrument_spec import get_spec
from trainer.core.adaptation import ParameterSearch
from trainer.core.walk_forward import WalkForwardValidator
from trainer.core.knowledge_base import KnowledgeBase
from engine.core.strategy_loader import StrategySpec


class TestDataFoundation:

    def test_instrument_spec_pip_size_values(self):
        """Verify captured InstrumentSpec digits, point, and pip_size values for XAUUSD, USDJPY, EURUSD."""
        xau = get_spec("XAUUSD")
        assert xau is not None and xau.sanity_ok
        assert xau.digits == 2
        assert xau.point == 0.01
        assert xau.pip_size == 0.01  # Gold: pip_size == point

        jpy = get_spec("USDJPY")
        assert jpy is not None and jpy.sanity_ok
        assert jpy.digits == 3
        assert jpy.point == 0.001
        assert jpy.pip_size == 0.01  # JPY: pip_size == point * 10

        eur = get_spec("EURUSD")
        assert eur is not None and eur.sanity_ok
        assert eur.digits == 5
        assert eur.point == 0.00001
        assert eur.pip_size == 0.0001  # 5-digit FX: pip_size == point * 10

    def test_parameter_search_uses_correct_pip_size_for_xauusd_and_usdjpy(self, tmp_path):
        """Verify ParameterSearch grid search sources pip_size directly from InstrumentSpec."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='15min', tz='UTC')
        df = pd.DataFrame({'open': 1800.0, 'high': 1810.0, 'low': 1790.0, 'close': 1805.0, 'volume': 100}, index=dates)

        kb = KnowledgeBase(base_path=str(tmp_path / "kb.json"))

        search_xau = ParameterSearch(symbol="XAUUSD", template="ma_crossover", df=df, knowledge_base=kb, data_hash="hash1", run_id="run1")
        grid_xau = search_xau.run_grid_search(df, candidate_id_prefix="cand_xau")
        assert len(grid_xau) > 0
        for cand in grid_xau:
            assert cand.parameters['pip_size'] == 0.01

        search_jpy = ParameterSearch(symbol="USDJPY", template="ma_crossover", df=df, knowledge_base=kb, data_hash="hash2", run_id="run2")
        grid_jpy = search_jpy.run_grid_search(df, candidate_id_prefix="cand_jpy")
        assert len(grid_jpy) > 0
        for cand in grid_jpy:
            assert cand.parameters['pip_size'] == 0.01

    def test_walk_forward_validator_uses_correct_pip_size(self, monkeypatch):
        """Verify WalkForwardValidator uses XAUUSD's real pip_size (0.01) instead of 0.0001."""
        dates = pd.date_range('2020-01-01', periods=100, freq='h', tz='UTC')
        df = pd.DataFrame({
            'open': 1800.0, 'high': 1810.0, 'low': 1790.0, 'close': 1805.0, 'volume': 100
        }, index=dates)

        wf = WalkForwardValidator('XAUUSD', df)
        wf.param_grid = [
            {'fast_ma_period': 10, 'slow_ma_period': 50, 'sl_atr_multiple': 1.5, 'tp_rr_ratio': 2.0}
        ]

        run_one_mock = MagicMock(return_value={
            'total_trades': 25, 'profit_factor': 1.5, 'calmar_ratio': 1.2, 'win_rate': 0.55,
            'max_drawdown_pct': 4.0, 'net_profit': 200.0, 'mandate_compliant': True, 'mandate_breach_reason': None
        })
        monkeypatch.setattr(wf, '_run_one', run_one_mock)

        best_params, score, metrics = wf._optimise(df, window_num=1)
        assert best_params['pip_size'] == 0.01

        # Check call arguments passed to _run_one
        passed_params = run_one_mock.call_args[0][1]
        assert passed_params['pip_size'] == 0.01

    def test_missing_instrument_spec_fails_closed(self, monkeypatch, tmp_path):
        """Confirm missing or un-sanitized InstrumentSpec raises ValueError."""
        monkeypatch.setattr("shared.instrument_spec.get_spec", lambda sym: None)
        kb = KnowledgeBase(base_path=str(tmp_path / "kb.json"))
        df = pd.DataFrame({'open': [1.0], 'high': [1.1], 'low': [0.9], 'close': [1.0], 'volume': [10]})

        with pytest.raises(ValueError) as exc_info:
            ParameterSearch(symbol="NONEXISTENT_PAIR", template="ma_crossover", df=df, knowledge_base=kb, data_hash="h", run_id="r").run_grid_search(df, "test")
        assert "INSTRUMENT_SPEC_MISSING" in str(exc_info.value)

    def test_strategy_loader_resolves_pip_size_from_spec(self):
        """Confirm StrategySpec resolves pip_size from get_spec if omitted from parameters."""
        config = StrategySpec(
            strategy_id="STRAT_TEST",
            template="ma_crossover",
            symbol="XAUUSD",
            timeframe="M15",
            parameters={},
            filters={},
            composite_score=0.85,
            promoted_at="2026-07-31T00:00:00Z",
            file_hash="1234567890abcdef"
        )
        assert config.pip_size == 0.01
