"""
Unit tests for scoring.py and promotion.py
Run: pytest tests/test_scoring.py -v
All should pass without any data files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime, timezone, timedelta
from trainer.core.scoring import (
    check_hard_gates, calculate_composite_score,
    aggregate_wf_results, rank_candidates,
    CompositeScore, GATES
)
from trainer.core.promotion import PromotionEngine


# ── FIXTURES ───────────────────────────────────────────

def _make_trades(n: int, win_pnl: float = 20.0, loss_pnl: float = -14.0) -> list:
    """Build a minimal representative OOS trade list, 1 trade per calendar day."""
    base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    trades = []
    for i in range(n):
        day = base + timedelta(days=i)
        pnl = win_pnl if i % 2 == 0 else loss_pnl
        trades.append({
            'direction': 'BUY',
            'entry_price': 1.1000,
            'entry_time': day,
            'exit_price': 1.1050,
            'exit_time': day + timedelta(minutes=30),
            'lots': 0.01,
            'pnl': pnl,
        })
    return trades


def good_wf_summary():
    """A walk-forward summary that passes all gates."""
    return {
        'valid_windows':             20,
        'total_trades':              450,
        'median_trades_per_window':  22.0,
        'median_profit_factor':      1.35,
        'median_calmar_ratio':       1.20,
        'median_avg_win_loss':       1.80,
        'median_max_drawdown_pct':   12.0,
        'profitable_window_rate':    0.70,
        'max_consecutive_losses':    6,
        'regime_consistency':        0.65,
        'temporal_stable':           True,
        'temporal_trend_pct':        5.0,
        'total_oos_pnl':             4500.0,
        'avg_oos_pnl_per_window':    225.0,
        'regime_windows_min':        4,
        'mandate_compliant':         True,
        'all_oos_trades':            _make_trades(450),
    }

def failing_wf_summary():
    """A summary that fails multiple gates."""
    return {
        'valid_windows':             5,
        'total_trades':              40,
        'median_trades_per_window':  8.0,
        'median_profit_factor':      0.85,
        'median_calmar_ratio':       0.30,
        'median_avg_win_loss':       1.10,
        'median_max_drawdown_pct':   35.0,
        'profitable_window_rate':    0.30,
        'max_consecutive_losses':    15,
        'regime_consistency':        0.30,
        'temporal_stable':           False,
        'temporal_trend_pct':       -40.0,
        'total_oos_pnl':            -2000.0,
        'avg_oos_pnl_per_window':   -400.0,
        'regime_windows_min':        1,
    }


# ── HARD GATE TESTS ────────────────────────────────────

class TestHardGates:

    def test_good_summary_passes_all_gates(self):
        result = check_hard_gates(good_wf_summary(), 'test')
        assert result.passed is True
        assert result.failed_gate is None

    def test_failing_summary_fails_gates(self):
        result = check_hard_gates(failing_wf_summary(), 'test')
        assert result.passed is False
        assert result.failed_gate is not None

    def test_gate_1_total_trades(self):
        summary = good_wf_summary()
        summary['total_trades'] = 50  # below 100
        result = check_hard_gates(summary, 'test')
        assert result.passed is False
        assert 'gate_1' in result.failed_gate

    def test_gate_2_trades_per_window(self):
        summary = good_wf_summary()
        summary['median_trades_per_window'] = 5  # below 20
        result = check_hard_gates(summary, 'test')
        assert result.passed is False
        assert 'gate_2' in result.failed_gate

    def test_gate_3_valid_windows(self):
        summary = good_wf_summary()
        summary['valid_windows'] = 5  # below 12
        result = check_hard_gates(summary, 'test')
        assert result.passed is False
        assert 'gate_3' in result.failed_gate

    def test_gate_4_max_drawdown(self):
        summary = good_wf_summary()
        summary['median_max_drawdown_pct'] = 25.0  # above 20%
        result = check_hard_gates(summary, 'test')
        assert result.passed is False
        assert 'gate_4' in result.failed_gate

    def test_gate_5_profit_factor(self):
        summary = good_wf_summary()
        summary['median_profit_factor'] = 1.05  # below 1.1
        result = check_hard_gates(summary, 'test')
        assert result.passed is False
        assert 'gate_5' in result.failed_gate

    def test_gate_6_profitable_windows(self):
        summary = good_wf_summary()
        summary['profitable_window_rate'] = 0.40  # below 0.55
        result = check_hard_gates(summary, 'test')
        assert result.passed is False
        assert 'gate_6' in result.failed_gate

    def test_gate_7_consecutive_losses(self):
        summary = good_wf_summary()
        summary['max_consecutive_losses'] = 12  # above 10
        result = check_hard_gates(summary, 'test')
        assert result.passed is False
        assert 'gate_7' in result.failed_gate

    def test_exactly_at_threshold_passes(self):
        summary = good_wf_summary()
        summary['total_trades'] = GATES['min_total_trades']
        summary['median_profit_factor'] = \
            GATES['min_median_profit_factor']
        result = check_hard_gates(summary, 'test')
        assert result.passed is True


# ── COMPOSITE SCORE TESTS ──────────────────────────────

class TestCompositeScore:

    def test_score_in_range(self):
        score = calculate_composite_score(
            good_wf_summary(), 'test', trial_count=1
        )
        assert 0.0 <= score.composite <= 1.0

    def test_good_summary_scores_well(self):
        score = calculate_composite_score(
            good_wf_summary(), 'test', trial_count=1
        )
        assert score.composite >= 0.50

    def test_weights_sum_to_one(self):
        from trainer.core.scoring import WEIGHTS
        total = sum(WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_fragile_penalty_applied(self):
        score_moderate = calculate_composite_score(
            good_wf_summary(), 'test',
            sensitivity_status='MODERATE',
            trial_count=1,
        )
        score_fragile = calculate_composite_score(
            good_wf_summary(), 'test',
            sensitivity_status='FRAGILE',
            trial_count=1,
        )
        assert score_fragile.composite < score_moderate.composite

    def test_robust_bonus_applied(self):
        score_moderate = calculate_composite_score(
            good_wf_summary(), 'test',
            sensitivity_status='MODERATE',
            trial_count=1,
        )
        score_robust = calculate_composite_score(
            good_wf_summary(), 'test',
            sensitivity_status='ROBUST',
            trial_count=1,
        )
        assert score_robust.composite > score_moderate.composite

    def test_no_tier2_data_neutral(self):
        score = calculate_composite_score(
            good_wf_summary(), 'test',
            tier2_robustness=None,
            trial_count=1,
        )
        # Neutral robustness = 0.5 → contributes 0.5 * 0.05 = 0.025
        assert score.robustness_score == 0.5

    def test_high_tier2_robustness(self):
        score = calculate_composite_score(
            good_wf_summary(), 'test',
            tier2_robustness=0.80,
            trial_count=1,
        )
        assert score.robustness_score == 1.0

    def test_score_never_negative(self):
        score = calculate_composite_score(
            failing_wf_summary(), 'test', trial_count=1
        )
        assert score.composite >= 0.0

    def test_score_never_above_one(self):
        # Give perfect values
        summary = good_wf_summary()
        summary['median_calmar_ratio']     = 10.0
        summary['median_profit_factor']    = 10.0
        summary['median_avg_win_loss']     = 10.0
        summary['regime_consistency']      = 1.0
        score = calculate_composite_score(
            summary, 'test',
            tier2_robustness=1.0,
            sensitivity_status='ROBUST',
            trial_count=1,
        )
        assert score.composite <= 1.0

    def test_trial_count_required_keyword_only(self):
        with pytest.raises(TypeError):
            calculate_composite_score(good_wf_summary(), 'test')

    def test_trial_count_invalid_raises(self):
        for bad in (0, -5, True):
            with pytest.raises(ValueError, match="trial_count"):
                calculate_composite_score(good_wf_summary(), 'test', trial_count=bad)

    def test_deflation_penalty_monotonically_reduces_score(self):
        s1 = calculate_composite_score(good_wf_summary(), 'test', trial_count=1)
        s10 = calculate_composite_score(good_wf_summary(), 'test', trial_count=10)
        s100 = calculate_composite_score(good_wf_summary(), 'test', trial_count=100)
        assert s1.deflation_penalty == 0.0
        assert s10.deflation_penalty > s1.deflation_penalty
        assert s100.deflation_penalty > s10.deflation_penalty
        assert s100.composite < s10.composite < s1.composite

    def test_prop_firm_penalty_near_dd_ceiling(self):
        safe_summary = good_wf_summary()
        safe_summary['worst_oos_drawdown_pct'] = 5.0
        warn_summary = good_wf_summary()
        warn_summary['worst_oos_drawdown_pct'] = 7.6  # > 85% of 8.0% = 6.8%

        s_safe = calculate_composite_score(
            safe_summary, 'test', trial_count=1, max_drawdown_limit_pct=8.0
        )
        s_warn = calculate_composite_score(
            warn_summary, 'test', trial_count=1, max_drawdown_limit_pct=8.0
        )
        assert s_safe.prop_firm_penalty == 0.0
        assert s_warn.prop_firm_penalty > 0.0
        assert s_warn.composite < s_safe.composite


# ── RANKING TESTS ──────────────────────────────────────

class TestRanking:

    def test_ranks_by_score_descending(self):
        scores = [
            CompositeScore('A', composite=0.45),
            CompositeScore('B', composite=0.72),
            CompositeScore('C', composite=0.38),
            CompositeScore('D', composite=0.61),
        ]
        ranked = rank_candidates(scores)
        assert ranked[0].candidate_id == 'B'
        assert ranked[1].candidate_id == 'D'
        assert ranked[2].candidate_id == 'A'
        assert ranked[3].candidate_id == 'C'

    def test_single_candidate(self):
        scores = [CompositeScore('A', composite=0.55)]
        ranked = rank_candidates(scores)
        assert len(ranked) == 1


# ── PROMOTION ENGINE TESTS ─────────────────────────────

class TestPromotionEngine:

    def test_first_run_auto_promote(self, tmp_path):
        engine = PromotionEngine(
            run_id='test_run',
            log_path=str(tmp_path / 'decision_log.jsonl'),
        )
        score = CompositeScore('test', composite=0.55)
        gate  = check_hard_gates(good_wf_summary(), 'test')
        decision = engine.decide(score, good_wf_summary(), gate)
        # First run with score >= 0.50 → AUTO_PROMOTE
        assert decision.decision_path == 'AUTO_PROMOTE'
        assert decision.decision_outcome == 'PROMOTED'

    def test_first_run_flag(self, tmp_path):
        engine = PromotionEngine(
            run_id='test_run',
            log_path=str(tmp_path / 'decision_log.jsonl'),
        )
        score = CompositeScore('test', composite=0.44)
        gate  = check_hard_gates(good_wf_summary(), 'test')
        decision = engine.decide(score, good_wf_summary(), gate)
        # Score 0.40-0.50 → FLAG
        assert decision.decision_path == 'FLAG_REVIEW'

    def test_first_run_reject(self, tmp_path):
        engine = PromotionEngine(
            run_id='test_run',
            log_path=str(tmp_path / 'decision_log.jsonl'),
        )
        score = CompositeScore('test', composite=0.30)
        gate  = check_hard_gates(good_wf_summary(), 'test')
        decision = engine.decide(score, good_wf_summary(), gate)
        # Score < 0.40 → REJECT
        assert decision.decision_path == 'AUTO_REJECT'

    def test_gate_failure_rejects(self, tmp_path):
        engine = PromotionEngine(
            run_id='test_run',
            log_path=str(tmp_path / 'decision_log.jsonl'),
        )
        score = CompositeScore('test', composite=0.70)
        gate  = check_hard_gates(failing_wf_summary(), 'test')
        decision = engine.decide(score, failing_wf_summary(), gate)
        assert decision.decision_path == 'AUTO_REJECT'
        assert decision.decision_outcome == 'REJECTED'

    def test_decision_has_recommendation(self, tmp_path):
        engine = PromotionEngine(
            run_id='test_run',
            log_path=str(tmp_path / 'decision_log.jsonl'),
        )
        score = CompositeScore('test', composite=0.55)
        gate  = check_hard_gates(good_wf_summary(), 'test')
        decision = engine.decide(score, good_wf_summary(), gate)
        assert len(decision.recommendation) > 0

    def test_first_run_auto_promote_deploys_loadable_strategy(self, tmp_path):
        from engine.core.strategy_loader import StrategyLoader

        engine = PromotionEngine(
            run_id='test_deploy_run',
            deploy_path=str(tmp_path),
            log_path=str(tmp_path / 'decision_log.jsonl'),
        )
        summary = good_wf_summary()
        summary['best_params'] = {'fast_ma_period': 10, 'slow_ma_period': 50}
        summary['filters'] = {'adx_threshold': 20}
        summary['median_win_rate'] = 0.58

        score = CompositeScore('cand_xau', composite=0.62)
        gate = check_hard_gates(summary, 'cand_xau')
        decision = engine.decide(
            score, summary, gate,
            template='ma_crossover', symbol='XAUUSD', timeframe='H1',
        )
        assert decision.decision_outcome == 'PROMOTED'
        active_file = tmp_path / 'active_strategy.json'
        assert active_file.exists()

        spec = StrategyLoader(str(active_file)).load_strategy()
        assert spec is not None
        assert spec.strategy_id == 'cand_xau'
        assert spec.template == 'ma_crossover'
        assert spec.symbol == 'XAUUSD'
        assert spec.timeframe == 'H1'
        assert spec.parameters == summary['best_params']
        assert spec.filters == summary['filters']

    def test_deploy_without_template_or_symbol_raises(self, tmp_path):
        engine = PromotionEngine(
            run_id='test_missing_meta',
            deploy_path=str(tmp_path),
            log_path=str(tmp_path / 'decision_log.jsonl'),
        )
        score = CompositeScore('cand_bad', composite=0.62)
        gate = check_hard_gates(good_wf_summary(), 'cand_bad')
        with pytest.raises(ValueError, match="template"):
            engine.decide(score, good_wf_summary(), gate)


# ── CANONICAL QUICK / COMPOSITE SCORE TESTS ────────────

class TestCanonicalQuickScore:

    def test_canonical_identity_with_walk_forward(self):
        from trainer.core.scoring import calculate_quick_score, composite_score
        from trainer.core import walk_forward as wf_mod
        assert composite_score is calculate_quick_score
        assert wf_mod.composite_score is calculate_quick_score

    def test_static_dd_limit_and_partial_trade_ramp(self):
        from trainer.core.scoring import calculate_quick_score

        base_metrics = {
            'total_trades': 15,
            'profit_factor': 2.0,
            'calmar_ratio': 1.5,
            'sharpe_ratio': 1.5,
            'win_rate': 0.55,
            'net_profit': 750.0,
            'max_drawdown_pct': 4.0,
            'mandate_compliant': True,
        }
        # Strict mode rejects < min_trades
        assert calculate_quick_score(base_metrics, min_trades=30, allow_partial_trades=False) == 0.0

        # Partial trade mode provides monotonic gradient
        s_partial = calculate_quick_score(base_metrics, min_trades=30, allow_partial_trades=True)
        full_metrics = dict(base_metrics, total_trades=30, net_profit=1500.0)
        s_full = calculate_quick_score(full_metrics, min_trades=30, allow_partial_trades=True)
        assert 0.0 < s_partial < s_full

        # Breaching static 8.0% DD returns 0.0
        breach_metrics = dict(full_metrics, max_drawdown_pct=8.1)
        assert calculate_quick_score(breach_metrics, min_trades=30) == 0.0

        # expectancy_pips overrides net_profit/total_trades fallback when supplied
        exp_metrics = dict(full_metrics, expectancy_pips=20.0, net_profit=1.0)
        assert calculate_quick_score(exp_metrics, min_trades=30) > calculate_quick_score(
            dict(full_metrics, net_profit=1.0), min_trades=30
        )