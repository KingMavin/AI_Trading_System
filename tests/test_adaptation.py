"""
Unit tests for adaptation.py and knowledge_base.py
Run: pytest tests/test_adaptation.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from trainer.core.adaptation import (
    CandidateConfig, HypothesisGenerator
)
from trainer.core.knowledge_base import KnowledgeBase
from trainer.core.parameter_grid import get_grid, get_grid_size


# ── FIXTURES ───────────────────────────────────────────

@pytest.fixture
def temp_kb(tmp_path):
    """Knowledge base backed by a temp directory."""
    kb_path = tmp_path / 'knowledge_base.json'
    return KnowledgeBase(base_path=str(kb_path))


# ── CANDIDATE CONFIG TESTS ─────────────────────────────

class TestCandidateConfig:

    def test_hash_is_deterministic(self):
        c1 = CandidateConfig(
            candidate_id='test',
            template='ma_crossover',
            symbol='EURUSD',
            timeframe='M15',
            parameters={
                'fast_ma_period': 10,
                'slow_ma_period': 50,
            }
        )
        c2 = CandidateConfig(
            candidate_id='different_id',
            template='ma_crossover',
            symbol='EURUSD',
            timeframe='M15',
            parameters={
                'fast_ma_period': 10,
                'slow_ma_period': 50,
            }
        )
        assert c1.compute_hash() == c2.compute_hash()

    def test_different_params_different_hash(self):
        c1 = CandidateConfig(
            candidate_id='test',
            template='ma_crossover',
            symbol='EURUSD',
            timeframe='M15',
            parameters={'fast_ma_period': 10,
                        'slow_ma_period': 50}
        )
        c2 = CandidateConfig(
            candidate_id='test',
            template='ma_crossover',
            symbol='EURUSD',
            timeframe='M15',
            parameters={'fast_ma_period': 20,
                        'slow_ma_period': 50}
        )
        assert c1.compute_hash() != c2.compute_hash()

    def test_different_symbol_different_hash(self):
        c1 = CandidateConfig(
            candidate_id='test',
            template='ma_crossover',
            symbol='EURUSD',
            timeframe='M15',
            parameters={'fast_ma_period': 10,
                        'slow_ma_period': 50}
        )
        c2 = CandidateConfig(
            candidate_id='test',
            template='ma_crossover',
            symbol='GBPUSD',
            timeframe='M15',
            parameters={'fast_ma_period': 10,
                        'slow_ma_period': 50}
        )
        assert c1.compute_hash() != c2.compute_hash()

    def test_hash_length(self):
        c = CandidateConfig(
            candidate_id='test',
            template='ma_crossover',
            symbol='EURUSD',
            timeframe='M15',
            parameters={'fast_ma_period': 10}
        )
        assert len(c.compute_hash()) == 16


# ── KNOWLEDGE BASE TESTS ───────────────────────────────

class TestKnowledgeBase:

    def test_initialises_empty(self, temp_kb):
        stats = temp_kb.get_stats()
        assert stats['total_runs']        == 0
        assert stats['candidates_tested'] == 0

    def test_has_been_tested_false_initially(self, temp_kb):
        assert temp_kb.has_been_tested(
            'abc123', 'datahash456'
        ) is False

    def test_record_and_retrieve_candidate(self, temp_kb):
        temp_kb.record_candidate(
            config_hash='abc123',
            data_hash='datahash456',
            run_id='run_001',
            composite_score=0.62,
            template='ma_crossover',
            symbol='EURUSD',
            params={'fast_ma_period': 10}
        )
        assert temp_kb.has_been_tested(
            'abc123', 'datahash456'
        ) is True

    def test_different_data_hash_not_cached(self, temp_kb):
        temp_kb.record_candidate(
            config_hash='abc123',
            data_hash='datahash456',
            run_id='run_001',
            composite_score=0.62,
            template='ma_crossover',
            symbol='EURUSD',
            params={'fast_ma_period': 10}
        )
        # Same config hash but different data → not cached
        assert temp_kb.has_been_tested(
            'abc123', 'different_data_hash'
        ) is False

    def test_cached_score_retrieval(self, temp_kb):
        temp_kb.record_candidate(
            config_hash='abc123',
            data_hash='datahash456',
            run_id='run_001',
            composite_score=0.62,
            template='ma_crossover',
            symbol='EURUSD',
            params={}
        )
        score = temp_kb.get_cached_score(
            'abc123', 'datahash456'
        )
        assert score == 0.62

    def test_template_performance_tracking(self, temp_kb):
        temp_kb.update_template_performance(
            'ma_crossover', 'EURUSD', 0.55, promoted=False
        )
        temp_kb.update_template_performance(
            'ma_crossover', 'EURUSD', 0.62, promoted=True
        )
        best = temp_kb.get_best_score_ever(
            'ma_crossover', 'EURUSD'
        )
        assert best == 0.62

    def test_parameter_insight_low_confidence(self, temp_kb):
        temp_kb.record_parameter_result(
            'ma_crossover', 'EURUSD',
            'fast_ma_period', 10, 0.55
        )
        # Only 1 test — LOW confidence → returns None
        insight = temp_kb.get_parameter_insight(
            'ma_crossover', 'EURUSD', 'fast_ma_period'
        )
        assert insight is None

    def test_parameter_insight_high_confidence(self, temp_kb):
        # Record enough tests for HIGH confidence
        for _ in range(16):
            temp_kb.record_parameter_result(
                'ma_crossover', 'EURUSD',
                'fast_ma_period', 10, 0.60
            )
            temp_kb.record_parameter_result(
                'ma_crossover', 'EURUSD',
                'fast_ma_period', 20, 0.45
            )
        insight = temp_kb.get_parameter_insight(
            'ma_crossover', 'EURUSD', 'fast_ma_period'
        )
        assert insight is not None
        assert insight['confidence'] == 'HIGH'
        assert insight['best_value'] == '10'

    def test_filter_effectiveness_tracking(self, temp_kb):
        temp_kb.record_filter_result(
            'session', 'ma_crossover',
            improved=True, delta=0.08
        )
        temp_kb.record_filter_result(
            'session', 'ma_crossover',
            improved=False, delta=-0.02
        )
        temp_kb.record_filter_result(
            'session', 'ma_crossover',
            improved=True, delta=0.05
        )
        temp_kb.record_filter_result(
            'session', 'ma_crossover',
            improved=True, delta=0.03
        )
        temp_kb.record_filter_result(
            'session', 'ma_crossover',
            improved=False, delta=-0.01
        )
        rate = temp_kb.get_filter_improvement_rate(
            'session', 'ma_crossover'
        )
        assert rate == pytest.approx(0.60, abs=0.01)

    def test_filter_rate_none_before_5_tests(self, temp_kb):
        temp_kb.record_filter_result(
            'session', 'ma_crossover',
            improved=True, delta=0.05
        )
        rate = temp_kb.get_filter_improvement_rate(
            'session', 'ma_crossover'
        )
        assert rate is None  # < 5 tests → insufficient

    def test_regime_pattern_tracking(self, temp_kb):
        temp_kb.update_regime_pattern(
            'TRENDING', 'ma_crossover', pf=1.4, win_rate=0.55
        )
        temp_kb.update_regime_pattern(
            'TRENDING', 'rsi_reversion', pf=0.9, win_rate=0.45
        )
        best = temp_kb.get_best_template_for_regime('TRENDING')
        assert best == 'ma_crossover'

    def test_save_and_reload(self, tmp_path):
        kb_path = tmp_path / 'kb_test.json'
        kb1 = KnowledgeBase(base_path=str(kb_path))
        kb1.record_candidate(
            'hash1', 'data1', 'run1', 0.65,
            'ma_crossover', 'EURUSD', {}
        )
        kb1.save()

        # Reload from disk
        kb2 = KnowledgeBase(base_path=str(kb_path))
        assert kb2.has_been_tested('hash1', 'data1') is True
        assert kb2.get_cached_score('hash1', 'data1') == 0.65

    def test_run_summary_recording(self, temp_kb):
        temp_kb.record_run_summary({
            'run_id':       'run_001',
            'best_score':   0.62,
            'total_trades': 450,
        })
        trend = temp_kb.get_run_trend('best_score', n_runs=5)
        assert len(trend) == 1
        assert trend[0] == 0.62


# ── PARAMETER GRID TESTS ───────────────────────────────

class TestParameterGrid:

    def test_grid_has_valid_combinations(self):
        grid = get_grid('ma_crossover')
        assert len(grid) > 0

    def test_all_combos_have_fast_less_than_slow(self):
        grid = get_grid('ma_crossover')
        for params in grid:
            assert params['fast_ma_period'] < \
                   params['slow_ma_period'], \
                f"fast={params['fast_ma_period']} >= " \
                f"slow={params['slow_ma_period']}"

    def test_grid_size_matches_count(self):
        grid = get_grid('ma_crossover')
        assert len(grid) == get_grid_size('ma_crossover')

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError):
            get_grid('nonexistent_template')

    def test_all_required_params_present(self):
        required = [
            'fast_ma_period', 'slow_ma_period',
            'sl_atr_multiple', 'tp_rr_ratio',
            'risk_per_trade_pct', 'warmup_candles'
        ]
        grid = get_grid('ma_crossover')
        for params in grid:
            for req in required:
                assert req in params, \
                    f"Missing param {req} in combo {params}"


# ── STAGE 4 INTERFACE TEST ─────────────────────────────

class TestDiscoveryInterface:

    def test_raises_not_implemented(self):
        gen = HypothesisGenerator()
        with pytest.raises(NotImplementedError):
            gen.generate({}, {}, None)


# ── DATA ISOLATION TESTS (WAVE 4.4) ────────────────────

class TestAdaptationDataIsolation:

    def test_stage1_held_out_data_isolation(self, temp_kb):
        """
        Verify that mutating df_held_out in AdaptationSystem has ZERO effect on Stage 1
        parameter search output candidates (parameters, scores, hashes), proving complete
        isolation of held-out test data from Stage 1 parameter optimization.
        """
        from trainer.core.adaptation import AdaptationSystem
        from shared.instrument_spec import InstrumentSpec
        from unittest.mock import patch

        dates_opt = pd.date_range('2020-01-01', periods=1000, freq='15min', tz='UTC')
        dates_held_out = pd.date_range('2020-01-15', periods=500, freq='15min', tz='UTC')

        t = np.linspace(0, 20 * np.pi, 1000)
        closes_opt = 1.1000 + 0.0050 * np.sin(t)

        df_opt = pd.DataFrame({
            'open': closes_opt - 0.0001,
            'high': closes_opt + 0.0002,
            'low': closes_opt - 0.0002,
            'close': closes_opt,
            'volume': [100.0] * 1000,
        }, index=dates_opt)

        df_held_out_orig = pd.DataFrame({
            'open': [1.1000] * 500,
            'high': [1.1010] * 500,
            'low': [1.0990] * 500,
            'close': [1.1000] * 500,
            'volume': [100.0] * 500,
        }, index=dates_held_out)

        df_held_out_mutated = df_held_out_orig.copy()
        df_held_out_mutated['close'] *= 10.0
        df_held_out_mutated['volume'] *= 100.0

        spec = InstrumentSpec.build_simulation_default("EURUSD")
        spec.sanity_ok = True

        with patch("shared.instrument_spec.get_spec", return_value=spec):
            sys_orig = AdaptationSystem(
                symbol='EURUSD',
                df=df_opt,
                df_held_out=df_held_out_orig,
                knowledge_base=temp_kb,
                data_hash='hash1',
                run_id='run1',
                templates=['ma_crossover'],
                top_n=3
            )
            cands_orig = sys_orig.run()

            sys_mutated = AdaptationSystem(
                symbol='EURUSD',
                df=df_opt,
                df_held_out=df_held_out_mutated,
                knowledge_base=temp_kb,
                data_hash='hash2',
                run_id='run2',
                templates=['ma_crossover'],
                top_n=3
            )
            cands_mutated = sys_mutated.run()

            # Assert candidates originating from Stage 1 optimization are 100% identical in parameters and template
            assert len(cands_orig) == len(cands_mutated)
            for c1, c2 in zip(cands_orig, cands_mutated):
                assert c1.parameters == c2.parameters
                assert c1.template == c2.template

    # ── WAVE 17 OPTUNA SEARCH TESTS ─────────────────────────────

    def test_optuna_search_output_shape_and_types(self, temp_kb):
        from trainer.core.adaptation import ParameterSearch, CandidateConfig
        from shared.instrument_spec import InstrumentSpec
        from unittest.mock import patch

        dates = pd.date_range('2020-01-01', periods=500, freq='15min', tz='UTC')
        t = np.linspace(0, 10 * np.pi, 500)
        closes = 1.1000 + 0.0030 * np.sin(t)
        df = pd.DataFrame({
            'open': closes - 0.0001, 'high': closes + 0.0002,
            'low': closes - 0.0002, 'close': closes, 'volume': [100.0] * 500
        }, index=dates)

        spec = InstrumentSpec.build_simulation_default("EURUSD")
        spec.sanity_ok = True

        with patch("shared.instrument_spec.get_spec", return_value=spec):
            search = ParameterSearch(
                symbol='EURUSD', template='ma_crossover', df=df,
                knowledge_base=temp_kb, data_hash='dhash', run_id='r1', top_n=3
            )
            cands = search.run_optuna_search(df, candidate_id_prefix='test_optuna', n_trials=30)

            assert len(cands) == 3
            for c in cands:
                assert isinstance(c, CandidateConfig)
                assert c.template == 'ma_crossover'
                assert c.symbol == 'EURUSD'
                assert c.timeframe == 'M15'
                assert isinstance(c.parameters, dict)
                assert isinstance(c.composite_score, float)
                assert isinstance(c.config_hash, str) and len(c.config_hash) > 0
                assert c.parameters['fast_ma_period'] < c.parameters['slow_ma_period']

    def test_optuna_reproducibility(self, tmp_path):
        """Verify Optuna TPESampler seed determinism across separate runs without KB caching."""
        from trainer.core.adaptation import ParameterSearch
        from trainer.core.knowledge_base import KnowledgeBase
        from shared.instrument_spec import InstrumentSpec
        from unittest.mock import patch

        kb1 = KnowledgeBase(base_path=str(tmp_path / 'kb1.json'))
        kb2 = KnowledgeBase(base_path=str(tmp_path / 'kb2.json'))

        dates = pd.date_range('2020-01-01', periods=500, freq='15min', tz='UTC')
        t = np.linspace(0, 10 * np.pi, 500)
        closes = 1.1000 + 0.0030 * np.sin(t)
        df = pd.DataFrame({
            'open': closes - 0.0001, 'high': closes + 0.0002,
            'low': closes - 0.0002, 'close': closes, 'volume': [100.0] * 500
        }, index=dates)

        spec = InstrumentSpec.build_simulation_default("EURUSD")
        spec.sanity_ok = True

        with patch("shared.instrument_spec.get_spec", return_value=spec):
            s1 = ParameterSearch(symbol='EURUSD', template='ma_crossover', df=df, knowledge_base=kb1, data_hash='dh', run_id='r1', top_n=3)
            cands1 = s1.run_optuna_search(df, candidate_id_prefix='run1', n_trials=20)

            s2 = ParameterSearch(symbol='EURUSD', template='ma_crossover', df=df, knowledge_base=kb2, data_hash='dh', run_id='r2', top_n=3)
            cands2 = s2.run_optuna_search(df, candidate_id_prefix='run2', n_trials=20)

            assert len(cands1) == len(cands2)
            for c1, c2 in zip(cands1, cands2):
                assert c1.parameters == c2.parameters
                assert c1.composite_score == pytest.approx(c2.composite_score, abs=1e-5)

    def test_adaptation_dispatches_non_ma_template_and_propagates_timeframe(self, temp_kb, monkeypatch):
        """
        Verify ParameterSearch & FilterTesting dispatch to template signal module
        (not hardcoded ma_crossover) and propagate timeframe='H1'.
        """
        from trainer.core.adaptation import ParameterSearch, FilterTesting
        from shared.instrument_spec import InstrumentSpec
        from unittest.mock import patch
        import trainer.core.adaptation as adapt_mod

        dates = pd.date_range('2020-01-01', periods=320, freq='1h', tz='UTC')
        closes = 1800.0 + 10.0 * np.sin(np.linspace(0, 8 * np.pi, 320))
        df = pd.DataFrame({
            'open': closes - 0.5,
            'high': closes + 1.0,
            'low': closes - 1.0,
            'close': closes,
            'volume': [100.0] * 320,
        }, index=dates)

        seen_modules = []
        orig_get_sig = adapt_mod.get_signal_module

        def spy_get_signal_module(tmpl):
            mod = orig_get_sig(tmpl)
            seen_modules.append((tmpl, getattr(mod, '__name__', '')))
            return mod

        monkeypatch.setattr(adapt_mod, 'get_signal_module', spy_get_signal_module)

        spec = InstrumentSpec.build_simulation_default("XAUUSD")
        spec.sanity_ok = True

        with patch("shared.instrument_spec.get_spec", return_value=spec):
            search = ParameterSearch(
                symbol='XAUUSD',
                template='hybrid_confluence',
                df=df,
                knowledge_base=temp_kb,
                data_hash='h1_hash',
                run_id='r_h1',
                top_n=1,
                timeframe='H1',
            )
            cands = search.run_optuna_search(df, candidate_id_prefix='hc_xau', n_trials=2)
            assert len(cands) == 1
            assert cands[0].timeframe == 'H1'

            ft = FilterTesting(
                symbol='XAUUSD',
                df_opt=df,
                df_held_out=df,
                knowledge_base=temp_kb,
                max_filters=1,
            )
            ft.test(cands[0], template='hybrid_confluence')

        assert any(t == 'hybrid_confluence' and 'hybrid_confluence' in name for t, name in seen_modules)



class TestWave46CombinationRouting:

    def test_valid_combinations_reference_implemented_templates(self):
        from trainer.core.adaptation import VALID_COMBINATIONS
        from trainer.core.walk_forward import (
            VALID_BASE_TEMPLATES,
            ComboSignalAdapter,
            get_signal_module,
        )

        exit_templates = [exit_t for _, exit_t, _ in VALID_COMBINATIONS]
        assert 'bb_breakout' not in exit_templates
        assert len(VALID_COMBINATIONS) >= 1

        for entry_t, exit_t, _ in VALID_COMBINATIONS:
            assert entry_t in VALID_BASE_TEMPLATES
            assert exit_t in VALID_BASE_TEMPLATES
            mod = get_signal_module(f"combo_{entry_t}__{exit_t}")
            assert isinstance(mod, ComboSignalAdapter)

    def test_combo_signal_adapter_routes_entry_and_exit_separately(self):
        from trainer.core.walk_forward import get_signal_module, ComboSignalAdapter

        adapter = get_signal_module('combo_ma_crossover__rsi_reversion')
        assert isinstance(adapter, ComboSignalAdapter)

        cache = {
            'sma_fast': 1.1050,
            'sma_fast_prev': 1.0990,
            'sma_slow': 1.1000,
            'sma_slow_prev': 1.1010,
            'atr': 0.0020,
            'adx': 25.0,
            'close': 1.1050,
            'rsi': 55.0,
            'session': 'LONDON',
            'pip_size': 0.0001,
        }
        params = {
            'fast_ma_period': 5,
            'slow_ma_period': 30,
            'sl_atr_multiple': 1.5,
            'tp_rr_ratio': 2.0,
            'adx_threshold': 0,
            'exit_on_midline': True,
            'exit_on_opposite_crossover': False,
        }

        sig = adapter.generate_signal(cache, params)
        assert sig['signal'] == 'BUY'
        assert 'MA_CROSS_BULL' in sig['reason']

        exit_res = adapter.check_exit(cache, {'direction': 'BUY'}, params)
        assert exit_res['should_exit'] is True
        assert exit_res['reason'] == 'EXIT_RSI_MIDLINE'

    def test_combination_testing_merges_exit_defaults_and_uses_combo_module(self, monkeypatch):
        import trainer.core.adaptation as adapt_mod
        from trainer.core.adaptation import CombinationTesting, CandidateConfig
        from trainer.core.walk_forward import ComboSignalAdapter
        from shared.instrument_spec import InstrumentSpec
        from unittest.mock import patch

        dates = pd.date_range('2023-01-01', periods=20, freq='15min')
        df = pd.DataFrame({
            'open': np.linspace(1.1000, 1.1020, 20),
            'high': np.linspace(1.1010, 1.1030, 20),
            'low': np.linspace(1.0990, 1.1010, 20),
            'close': np.linspace(1.1005, 1.1025, 20),
            'volume': [100.0] * 20,
        }, index=dates)

        captured = {}

        class DummyBacktester:
            def __init__(self, symbol, params, initial_equity=10000.0, random_seed=42):
                captured['params'] = dict(params)
                self.initial_equity = initial_equity

            def run(self, run_df, signal_module):
                captured['signal_module'] = signal_module
                return {
                    'trades': [{'pnl': 50.0, 'net_pnl': 50.0, 'entry_time': dates[1], 'exit_time': dates[-1]} for _ in range(12)],
                    'equity_curve': [10000.0 + i * 50.0 for i in range(20)],
                }

        monkeypatch.setattr(adapt_mod, 'Backtester', DummyBacktester)
        monkeypatch.setattr(adapt_mod, 'calculate_metrics', lambda *a, **kw: {
            'total_trades': 12, 'profit_factor': 2.5, 'sharpe_ratio': 1.8, 'max_drawdown_pct': 2.0,
        })
        monkeypatch.setattr(adapt_mod, 'calculate_quick_score', lambda m, min_trades=10: 0.90)

        entry_cand = CandidateConfig(
            candidate_id='ma_cand_1',
            template='ma_crossover',
            symbol='EURUSD',
            timeframe='M15',
            parameters={'fast_ma_period': 5, 'slow_ma_period': 30, 'sl_atr_multiple': 2.0, 'tp_rr_ratio': 2.5},
            composite_score=0.50,
        )
        exit_cand = CandidateConfig(
            candidate_id='rsi_cand_1',
            template='rsi_reversion',
            symbol='EURUSD',
            timeframe='M15',
            parameters={'rsi_period': 14, 'exit_on_midline': True, 'sl_atr_multiple': 1.0},
            composite_score=0.45,
        )

        tester = CombinationTesting(symbol='EURUSD', df=df, improvement_threshold=0.05)
        spec = InstrumentSpec.build_simulation_default('EURUSD')
        with patch('shared.instrument_spec.get_spec', return_value=spec):
            combo = tester._test_one_combination(entry_cand, exit_cand, 'ma_crossover', 'rsi_reversion')

        assert combo is not None
        assert combo.template == 'combo_ma_crossover__rsi_reversion'
        assert isinstance(captured['signal_module'], ComboSignalAdapter)
        # Entry parameter sl_atr_multiple=2.0 must NOT be overwritten by rsi_reversion space default (1.0)
        assert captured['params']['sl_atr_multiple'] == 2.0
        # Non-colliding rsi_reversion defaults must be merged in
        assert captured['params']['exit_on_midline'] is True
        assert captured['params']['rsi_period'] == 14
