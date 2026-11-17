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

    def test_mandate_evaluator_prague_midnight_split(self, sample_mandate):
        # Europe/Prague in January is CET (UTC+1). 
        # Midnight Prague = 23:00:00 UTC of the previous day.
        # Limit is 5% daily loss.
        
        curve_pass = [
            # Day 1 (Jan 1st Prague): 22:50 UTC = 23:50 Prague. Equity: 100k -> 97k (3% loss)
            {'timestamp': '2026-01-01T22:50:00Z', 'equity': 100000},
            {'timestamp': '2026-01-01T22:55:00Z', 'equity': 97000},  
            
            # Day 2 (Jan 2nd Prague): 23:10 UTC = 00:10 Prague. Day rolls over.
            # Starting equity for Day 2 is 97k. Drops to 93500 (3.6% loss).
            {'timestamp': '2026-01-01T23:10:00Z', 'equity': 93500},  
        ]
        # Passes because Day 1 loss is 3%, Day 2 loss is 3.6%. Neither hits 5%.
        is_comp, _ = evaluate_backtest_mandate(curve_pass, mandate=sample_mandate)
        assert is_comp is True

        # Also verify real FTMO 4.0% daily limit with Europe/Prague rollover:
        # Day 1 (-3.0%): 100k -> 97k; Day 2 (-3.09% of 97k): 97k -> 94k -> passes
        sample_mandate.max_daily_loss_pct = 4.0
        curve_ftmo_pass = [
            {'timestamp': '2026-01-01T22:50:00Z', 'equity': 100000},
            {'timestamp': '2026-01-01T22:55:00Z', 'equity': 97000},
            {'timestamp': '2026-01-01T23:10:00Z', 'equity': 94000},
        ]
        is_comp, _ = evaluate_backtest_mandate(curve_ftmo_pass, mandate=sample_mandate)
        assert is_comp is True

        # Same-day 4.2% drop violates 4.0% limit
        curve_ftmo_fail = [
            {'timestamp': '2026-01-01T10:00:00Z', 'equity': 100000},
            {'timestamp': '2026-01-01T15:00:00Z', 'equity': 95800},
        ]
        is_comp, reason = evaluate_backtest_mandate(curve_ftmo_fail, mandate=sample_mandate)
        assert is_comp is False
        assert "MANDATE_DAILY_LOSS_BREACH" in reason

        sample_mandate.max_daily_loss_pct = 5.0
        curve_fail = [
            # Day 1: 100k -> 94.8k (5.2% loss) on a single calendar day
            {'timestamp': '2026-01-01T10:00:00Z', 'equity': 100000},
            {'timestamp': '2026-01-01T15:00:00Z', 'equity': 94800},
        ]
        is_comp, reason = evaluate_backtest_mandate(curve_fail, mandate=sample_mandate)
        assert is_comp is False
        assert "MANDATE_DAILY_LOSS_BREACH" in reason

    def test_mandate_evaluator_drawdown_type(self, sample_mandate):
        # Initial: 100k -> Peak: 110k -> Drop: 100k
        # Drop from 110k to 100k is 10k drop. 10k / 110k = 9.09% trailing DD.
        sample_mandate.max_overall_drawdown_pct = 8.0
        curve = [
            {'timestamp': '2026-01-01T10:00:00Z', 'equity': 100000},
            {'timestamp': '2026-01-05T10:00:00Z', 'equity': 110000},
            {'timestamp': '2026-01-10T10:00:00Z', 'equity': 100000},
        ]
        
        # Static 8% (from initial) should pass because 100k >= 92k (8% of 100k)
        is_comp, _ = evaluate_backtest_mandate(curve, mandate=sample_mandate)
        assert is_comp is True
        
        # Trailing 8% should fail because 110k -> 100k is a 9.09% drop (> 8.0%).
        sample_mandate.drawdown_type = 'trailing'
        is_comp, reason = evaluate_backtest_mandate(curve, mandate=sample_mandate)
        assert is_comp is False
        assert "MANDATE_OVERALL_DD_BREACH" in reason

    def test_mandate_evaluator_rejects_malformed_timestamp(self, sample_mandate):
        curve = [
            {'timestamp': '2026-01-01T10:00:00Z', 'equity': 10000.0},
            {'timestamp': 'not-a-timestamp', 'equity': 10050.0},
        ]
        with pytest.raises(ValueError, match="Unparseable ISO timestamp"):
            evaluate_backtest_mandate(curve, mandate=sample_mandate)

    def test_mandate_evaluator_rejects_missing_timestamp_key(self, sample_mandate):
        curve = [
            {'equity': 10000.0},
            {'equity': 9900.0},
        ]
        with pytest.raises(ValueError, match="Missing 'timestamp' key"):
            evaluate_backtest_mandate(curve, mandate=sample_mandate)

    def test_mandate_evaluator_rejects_non_dict_point(self, sample_mandate):
        curve = [
            {'timestamp': '2026-01-01T10:00:00Z', 'equity': 10000.0},
            "invalid_point",
        ]
        with pytest.raises(ValueError, match="Invalid equity curve point"):
            evaluate_backtest_mandate(curve, mandate=sample_mandate)

    def test_metrics_fails_loudly_without_timestamps(self):
        from shared.metrics import calculate_metrics
        with pytest.raises(ValueError, match="wall-clock annualisation"):
            calculate_metrics([{'net_pnl': 100.0}], [10000.0, 10100.0])

    def test_daily_risk_pct_breach_rejected(self, sample_mandate):
        curve = [
            {'timestamp': '2026-01-01T10:00:00Z', 'equity': 10000.0},
            {'timestamp': '2026-01-02T10:00:00Z', 'equity': 10100.0},
        ]
        ok, reason = evaluate_backtest_mandate(curve, mandate=sample_mandate, risk_per_trade_pct=1.0)
        assert ok is True and reason is None

        bad, reason = evaluate_backtest_mandate(curve, mandate=sample_mandate, risk_per_trade_pct=1.5)
        assert bad is False
        assert "MANDATE_DAILY_RISK_BREACH" in reason

        bad_tr, tr_reason = evaluate_backtest_mandate(
            curve,
            mandate=sample_mandate,
            trades=[{'entry_time': '2026-01-01T10:00:00Z', 'risk_pct': 1.2}],
        )
        assert bad_tr is False
        assert "MANDATE_DAILY_RISK_BREACH" in tr_reason

    def test_min_trading_days_breach_and_fail_loud(self, sample_mandate):
        curve = [
            {'timestamp': '2026-01-01T10:00:00Z', 'equity': 10000.0},
            {'timestamp': '2026-01-10T10:00:00Z', 'equity': 10200.0},
        ]
        with pytest.raises(ValueError, match="enforce_min_trading_days=True"):
            evaluate_backtest_mandate(curve, mandate=sample_mandate, enforce_min_trading_days=True)

        two_day_trades = [
            {'entry_time': '2026-01-01T10:00:00Z'},
            {'entry_time': '2026-01-02T10:00:00Z'},
        ]
        ok_sparse, reason = evaluate_backtest_mandate(
            curve, mandate=sample_mandate, trades=two_day_trades, enforce_min_trading_days=True
        )
        assert ok_sparse is False
        assert "MANDATE_MIN_TRADING_DAYS_BREACH" in reason

        five_day_trades = [
            {'entry_time': f'2026-01-0{d}T10:00:00Z'} for d in range(1, 6)
        ]
        ok_five, reason = evaluate_backtest_mandate(
            curve, mandate=sample_mandate, trades=five_day_trades, enforce_min_trading_days=True
        )
        assert ok_five is True and reason is None

    def test_gate_9_enforces_min_trading_days_and_risk_at_promotion(self, sample_mandate):
        from datetime import datetime, timezone, timedelta
        sample_mandate.min_trading_days = 5
        sample_mandate.max_daily_loss_pct = 50.0
        sample_mandate.max_overall_drawdown_pct = 50.0
        sample_mandate.max_daily_risk_pct = 1.0

        base = datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc)
        # 120 trades crammed into 2 calendar days -> violates min_trading_days=5
        sparse_day_trades = [
            {
                'entry_time': base + timedelta(days=i % 2, minutes=i),
                'exit_time': base + timedelta(days=i % 2, minutes=i + 15),
                'lots': 0.01,
                'pnl': 10.0,
            }
            for i in range(120)
        ]
        summary = {
            'total_trades': 120,
            'median_trades_per_window': 25,
            'valid_windows': 14,
            'median_max_drawdown_pct': 5.0,
            'median_profit_factor': 1.8,
            'profitable_window_rate': 0.80,
            'max_consecutive_losses': 4,
            'mandate_compliant': True,
            'all_oos_trades': sparse_day_trades,
        }
        res_days = check_hard_gates(summary, candidate_id='cand_sparse_days', mandate=sample_mandate)
        assert res_days.passed is False
        assert res_days.failed_gate == 'gate_9_mandate_compliance'
        assert "MANDATE_MIN_TRADING_DAYS_BREACH" in res_days.failure_reason

        # Spread across 10 days, but configured risk_per_trade_pct exceeds limit
        spread_trades = [
            {
                'entry_time': base + timedelta(days=i % 10, hours=1),
                'exit_time': base + timedelta(days=i % 10, hours=2),
                'lots': 0.01,
                'pnl': 10.0,
            }
            for i in range(120)
        ]
        summary['all_oos_trades'] = spread_trades
        summary['risk_per_trade_pct'] = 1.5
        res_risk = check_hard_gates(summary, candidate_id='cand_high_risk', mandate=sample_mandate)
        assert res_risk.passed is False
        assert res_risk.failed_gate == 'gate_9_mandate_compliance'
        assert "MANDATE_DAILY_RISK_BREACH" in res_risk.failure_reason
