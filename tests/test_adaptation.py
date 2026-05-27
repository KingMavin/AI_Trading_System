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