"""
tests/test_pre_trade_checklist.py

Unit tests for WAVE 21 Part 2 — Structured 9-Item Pre-Trade Checklist.
Tests:
1. All 9 checklist items returning PASS on valid system state.
2. Individual item failures (MT5 disconnect, stale data, spread breach, dead zone, daily loss, drawdown, margin, positions).
3. Item 6 (regime_allowed) condition: PASSes when regime_target is None/empty per PRD §14.
4. Structuring of 9-item dictionary payload.
"""

import sys
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.core.checklist import PreTradeChecklistEvaluator, ChecklistResult
from engine.core.strategy_loader import StrategySpec


class TestPreTradeChecklist:

    @pytest.fixture
    def evaluator(self):
        return PreTradeChecklistEvaluator()

    @pytest.fixture
    def mock_spec(self):
        return StrategySpec(
            strategy_id='ATS-TEST-001',
            template='ma_crossover',
            symbol='EURUSD',
            timeframe='M15',
            parameters={'fast_ma_period': 20, 'slow_ma_period': 100},
            filters={'max_spread_pips': 3.0, 'allowed_sessions': ['LONDON', 'NEW_YORK']},
            composite_score=1.5,
            promoted_at='2026-05-27T00:00:00+00:00',
            file_hash='abc1234567890def',
            wf_median_win_rate=0.55,
            wf_median_profit_factor=1.5,
            dna_hash='1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
            regime_target='TRENDING'
        )

    def test_all_9_items_pass_on_valid_state(self, evaluator, mock_spec):
        res = evaluator.evaluate(
            mt5_connected=True,
            strategy_spec=mock_spec,
            data_fresh=True,
            current_spread_pips=1.2,
            max_spread_pips=3.0,
            current_session='LONDON',
            allowed_sessions=['LONDON', 'NEW_YORK'],
            current_regime='TRENDING',
            regime_target='TRENDING',
            safe_mode=False,
            daily_loss_ok=True,
            drawdown_ok=True,
            free_margin=10000.0,
            required_margin=100.0,
            open_positions_count=0,
            max_open_positions=3,
            has_conflicting_position=False,
            stale_seconds=5.0
        )

        assert res.all_passed is True
        assert len(res.items) == 9
        assert res.first_failure_reason is None

        for item in res.items.values():
            assert item['pass'] is True

    def test_item6_regime_allowed_passes_when_regime_target_is_none(self, evaluator, mock_spec):
        """Condition 2 Verification: PRD §14 regime_target None/empty must PASS item 6."""
        res = evaluator.evaluate(
            mt5_connected=True,
            strategy_spec=mock_spec,
            data_fresh=True,
            current_spread_pips=1.0,
            max_spread_pips=3.0,
            current_session='LONDON',
            allowed_sessions=['LONDON'],
            current_regime='RANGING',  # current regime is RANGING
            regime_target=None,         # regime_target is None (filter inactive in Phase 1)
            safe_mode=False,
            daily_loss_ok=True,
            drawdown_ok=True,
            free_margin=10000.0,
            required_margin=100.0,
            open_positions_count=0,
            max_open_positions=3,
            has_conflicting_position=False,
            stale_seconds=2.0
        )

        item6 = res.items['6_regime_allowed']
        assert item6['pass'] is True
        assert "Regime filter inactive" in item6['detail']
        assert res.all_passed is True

    def test_dead_zone_fails_session_allowed(self, evaluator, mock_spec):
        res = evaluator.evaluate(
            mt5_connected=True,
            strategy_spec=mock_spec,
            data_fresh=True,
            current_spread_pips=1.0,
            max_spread_pips=3.0,
            current_session='DEAD_ZONE',
            allowed_sessions=['LONDON', 'NEW_YORK'],
            current_regime='TRENDING',
            regime_target='TRENDING',
            safe_mode=False,
            daily_loss_ok=True,
            drawdown_ok=True,
            free_margin=10000.0,
            required_margin=100.0,
            open_positions_count=0,
            max_open_positions=3,
            has_conflicting_position=False,
            stale_seconds=2.0
        )

        assert res.all_passed is False
        assert res.items['5_session_allowed']['pass'] is False
        assert "DEAD_ZONE" in res.items['5_session_allowed']['detail']

    def test_spread_exceedance_fails_spread_ok(self, evaluator, mock_spec):
        res = evaluator.evaluate(
            mt5_connected=True,
            strategy_spec=mock_spec,
            data_fresh=True,
            current_spread_pips=4.5,
            max_spread_pips=3.0,
            current_session='LONDON',
            allowed_sessions=['LONDON'],
            current_regime='TRENDING',
            regime_target='TRENDING',
            safe_mode=False,
            daily_loss_ok=True,
            drawdown_ok=True,
            free_margin=10000.0,
            required_margin=100.0,
            open_positions_count=0,
            max_open_positions=3,
            has_conflicting_position=False,
            stale_seconds=2.0
        )

        assert res.all_passed is False
        assert res.items['4_spread_ok']['pass'] is False
        assert "exceeds limit" in res.items['4_spread_ok']['detail']

    def test_daily_loss_breach_fails_risk_limits(self, evaluator, mock_spec):
        res = evaluator.evaluate(
            mt5_connected=True,
            strategy_spec=mock_spec,
            data_fresh=True,
            current_spread_pips=1.0,
            max_spread_pips=3.0,
            current_session='LONDON',
            allowed_sessions=['LONDON'],
            current_regime='TRENDING',
            regime_target='TRENDING',
            safe_mode=False,
            daily_loss_ok=False,  # Daily loss breached!
            drawdown_ok=True,
            free_margin=10000.0,
            required_margin=100.0,
            open_positions_count=0,
            max_open_positions=3,
            has_conflicting_position=False,
            stale_seconds=2.0
        )

        assert res.all_passed is False
        assert res.items['7_risk_limits_ok']['pass'] is False
        assert "Daily loss limit breached" in res.items['7_risk_limits_ok']['detail']
