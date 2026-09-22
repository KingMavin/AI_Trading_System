"""
Unit tests for WAVE 11.2 — Prop-Rule-Aware Mandate Evaluator & Hard Gate.
Verifies daily loss breach detection, overall drawdown breach detection, fail-closed missing mandate behavior, and Gate 9 hard rejection.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.core.mandate import Mandate
from trainer.core.mandate_evaluator import evaluate_backtest_mandate
from trainer.core.scoring import check_hard_gates


class TestMandateEvaluator:

    @pytest.fixture
    def sample_mandate(self):
        return Mandate(
            mandate_id="MANDATE_TEST",
            firm_name="TestFirm",
            account_id="ACC_123",
            max_daily_loss_pct=5.0,
            max_overall_drawdown_pct=10.0,
            drawdown_type="static",
            min_trading_days=5,
            max_daily_risk_pct=1.0,
            symbol_universe=["EURUSD", "GBPUSD"]
        )

    def test_clean_equity_curve_passes_mandate(self, sample_mandate):
        curve = [
            {'timestamp': '2026-07-30T10:00:00+00:00', 'equity': 10000.0},
            {'timestamp': '2026-07-30T14:00:00+00:00', 'equity': 10100.0},
            {'timestamp': '2026-07-31T10:00:00+00:00', 'equity': 10200.0},
        ]
        is_compliant, reason = evaluate_backtest_mandate(curve, mandate=sample_mandate)
        assert is_compliant
        assert reason is None

    def test_daily_loss_breach_hard_rejected(self, sample_mandate):
        # 6% drop in a single day (10000 -> 9400 = 6% loss)
        curve = [
            {'timestamp': '2026-07-30T00:00:00+00:00', 'equity': 10000.0},
            {'timestamp': '2026-07-30T12:00:00+00:00', 'equity': 9400.0},
        ]
        is_compliant, reason = evaluate_backtest_mandate(curve, mandate=sample_mandate)
        assert not is_compliant
        assert "MANDATE_DAILY_LOSS_BREACH" in reason

    def test_overall_drawdown_breach_hard_rejected(self, sample_mandate):
        # Peak $10,000 -> drops to $8,800 (12% overall DD > 10% limit) across multiple days
        curve = [
            {'timestamp': '2026-07-30T00:00:00+00:00', 'equity': 10000.0},
            {'timestamp': '2026-07-31T00:00:00+00:00', 'equity': 9600.0},
            {'timestamp': '2026-08-01T00:00:00+00:00', 'equity': 8800.0},
        ]
        is_compliant, reason = evaluate_backtest_mandate(curve, mandate=sample_mandate)
        assert not is_compliant
        assert "MANDATE_OVERALL_DD_BREACH" in reason

    def test_missing_mandate_fails_closed(self):
        missing_path = Path("/nonexistent/mandate_file_12345.json")
        with pytest.raises(RuntimeError) as exc_info:
            evaluate_backtest_mandate([], mandate=None, mandate_path=missing_path)
        assert "MANDATE_CONFIG_MISSING" in str(exc_info.value)

    def test_scoring_gate_9_hard_rejects_non_compliant_candidate(self):
        wf_summary = {
            'total_trades': 150,
            'median_trades_per_window': 25,
            'valid_windows': 14,
            'median_max_drawdown_pct': 5.0,
            'median_profit_factor': 1.8,
            'profitable_window_rate': 0.80,
            'max_consecutive_losses': 4,
            'mandate_compliant': False,
            'mandate_breach_reason': 'Daily loss reached 6.2% on 2026-07-30'
        }

        gate_res = check_hard_gates(wf_summary, candidate_id='cand_mandate_fail')
        assert not gate_res.passed
        assert gate_res.failed_gate == 'gate_9_mandate_compliance'
        assert "Mandate compliance breach" in gate_res.failure_reason

    def test_walk_forward_multi_window_mandate_rollup(self):
        """Confirm aggregate_wf_results sets mandate_compliant=False if ANY window breached."""
        from datetime import datetime, timezone
        from trainer.core.walk_forward import WindowResult
        from trainer.core.scoring import aggregate_wf_results

        now = datetime.now(timezone.utc)
        win1 = WindowResult(
            window_num=1, opt_start=now, opt_end=now, test_start=now, test_end=now,
            oos_trades=60, oos_profit_factor=1.5, oos_calmar=1.2, oos_win_rate=0.55, oos_max_drawdown=5.0, oos_net_profit=100.0,
            oos_mandate_compliant=True, status='OK'
        )
        win2 = WindowResult(
            window_num=2, opt_start=now, opt_end=now, test_start=now, test_end=now,
            oos_trades=60, oos_profit_factor=1.4, oos_calmar=1.1, oos_win_rate=0.52, oos_max_drawdown=12.0, oos_net_profit=80.0,
            oos_mandate_compliant=False, oos_mandate_breach_reason='MANDATE_DAILY_LOSS_BREACH', status='OK'
        )

        wf_summary = aggregate_wf_results([win1, win2])
        # Override valid_windows/consec losses for minimal test window list
        wf_summary['valid_windows'] = 15
        wf_summary['max_consecutive_losses'] = 3
        assert wf_summary['mandate_compliant'] is False
        assert wf_summary['mandate_breach_reason'] == 'MANDATE_DAILY_LOSS_BREACH'

        gate_res = check_hard_gates(wf_summary, candidate_id='multi_win_candidate')
        assert not gate_res.passed
        assert gate_res.failed_gate == 'gate_9_mandate_compliance'

    def test_trainer_cli_mandate_load_failure_crashes(self, monkeypatch):
        """Confirm TrainerRunner instantiation fails loudly with RuntimeError when mandate config is missing."""
        from trainer.core.trainer_runner import TrainerRunner
        missing_mandate = Path("/nonexistent/mandate_missing_999.json")
        monkeypatch.setattr("engine.core.mandate.MANDATE_FILE", missing_mandate)

        with pytest.raises(RuntimeError) as exc_info:
            TrainerRunner(quick_mode=True)
        assert "does not exist" in str(exc_info.value)
