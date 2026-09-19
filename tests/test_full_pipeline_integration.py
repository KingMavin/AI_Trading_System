"""
WAVE 13 — Capstone Full Pipeline Integration Test Suite.
Exercises the entire ATS Trainer pipeline end-to-end using real historical Parquet data.
Verifies data loading, coherence validation, adaptation parameter search, walk-forward validation,
mandate compliance evaluation, hard gate scoring, promotion decisions, and report generation.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.core.trainer_runner import TrainerRunner, DEFAULT_CONFIG
from engine.core.mandate import Mandate, MANDATE_FILE
from shared.instrument_spec import get_spec
from trainer.core.scoring import check_hard_gates


class TestFullPipelineIntegration:

    @pytest.fixture(autouse=True)
    def setup_small_grid(self, monkeypatch):
        """Monkeypatch TEMPLATE_SPACES and GATES thresholds for fast end-to-end integration testing."""
        small_space = {
            'fast_ma_period': [10],
            'slow_ma_period': [20],
            'ma_type': ['SMA'],
            'sl_atr_multiple': [1.5],
            'tp_rr_ratio': [2.0]
        }
        from trainer.core import parameter_grid
        monkeypatch.setitem(parameter_grid.TEMPLATE_SPACES, 'ma_crossover', small_space)
        from trainer.core import scoring
        monkeypatch.setitem(scoring.GATES, 'min_total_trades', 1)
        monkeypatch.setitem(scoring.GATES, 'min_trades_per_window', 1)
        monkeypatch.setitem(scoring.GATES, 'min_valid_windows', 1)
        monkeypatch.setitem(scoring.GATES, 'min_median_profit_factor', 0.0)
        monkeypatch.setitem(scoring.GATES, 'min_profitable_window_rate', 0.0)
        monkeypatch.setitem(scoring.GATES, 'max_median_drawdown_pct', 100.0)
        monkeypatch.setitem(scoring.GATES, 'max_consecutive_losses', 100)

        # Ensure in-sample walk-forward optimization selects best parameter set during integration test
        def test_composite_score(metrics):
            if metrics.get('total_trades', 0) < 1:
                return 0.0
            return max(0.001, float(metrics.get('profit_factor', 0.1)))
        monkeypatch.setattr("trainer.core.walk_forward.composite_score", test_composite_score)

    def test_full_pipeline_eurusd_real_data(self, tmp_path):
        """
        13.2: Real EURUSD Full-Pipeline Integration Test.
        Runs TrainerRunner directly on a real 4-month EURUSD historical Parquet slice.
        """
        config = DEFAULT_CONFIG.copy()
        config['symbols'] = ['EURUSD']
        config['timeframe'] = 'M15'
        config['data_start'] = '2019-01-01'
        config['data_end'] = '2020-05-01'
        config['opt_months'] = 2
        config['test_months'] = 1
        config['top_n_candidates'] = 1
        config['templates'] = ['ma_crossover']

        test_mandate = Mandate(
            mandate_id="TEST_MANDATE",
            firm_name="TestFirm",
            account_id="ACC_TEST",
            max_daily_loss_pct=50.0,  # 50% limit so test walk-forward search finds valid parameter sets
            max_overall_drawdown_pct=50.0,
            drawdown_type="static",
            min_trading_days=1,
            max_daily_risk_pct=5.0,
            symbol_universe=["EURUSD", "XAUUSD"]
        )

        runner = TrainerRunner(config=config, quick_mode=True)
        runner.kb_path = tmp_path / "test_kb.json"
        runner.mandate = test_mandate

        record = runner.run()

        # Assertion 1: Pipeline completes without unhandled exception
        assert record.outcome in ['COMPLETED', 'PARTIAL_FAILURE']
        assert 'EURUSD' in record.symbol_status
        assert record.symbol_status['EURUSD']['status'] == 'COMPLETED'

        # Assertion 2: Verify every candidate has all 9 gate results present in gate_results for this run_id
        log_path = Path(runner.log_path)
        assert log_path.exists()
        import json
        all_lines = [json.loads(line) for line in log_path.read_text(encoding='utf-8').strip().split('\n') if line]
        decisions = [d for d in all_lines if d.get('run_id') == runner.run_id]
        assert len(decisions) > 0, f"Expected decision records for run_id {runner.run_id} in decision_log.jsonl"
        for dec in decisions:
            gate_results = dec.get('gate_results', {})
            expected_gates = [
                'gate_1_total_trades',
                'gate_2_trades_per_window',
                'gate_3_valid_windows',
                'gate_4_max_drawdown',
                'gate_5_profit_factor',
                'gate_6_profitable_windows',
                'gate_7_consecutive_losses',
                'gate_8_regime_coverage',
                'gate_9_mandate_compliant',
                'gate_10_monte_carlo_stress',
            ]
            for g in expected_gates:
                assert g in gate_results, f"Missing gate '{g}' in decision gate_results"

        # Assertion 3: Verify WAVE 4.2 fill-timing (entries execute at next-candle open, not same-candle close)
        from trainer.core.data_loader import load_ohlcv
        from datetime import datetime as dt
        df = load_ohlcv('EURUSD', 'M15', dt(2019, 1, 1), dt(2020, 5, 1))
        from trainer.core.backtester import Backtester
        from trainer.signals import ma_crossover
        backtester = Backtester(
            symbol='EURUSD',
            params={'fast_ma_period': 10, 'slow_ma_period': 50, 'ma_type': 'SMA', 'sl_atr_multiple': 1.5, 'tp_rr_ratio': 2.0, 'pip_size': 0.0001},
            initial_equity=10000.0
        )
        res = backtester.run(df, ma_crossover)
        trades = res.get('trades', [])
        if trades:
            for tr in trades[:5]:  # Spot check first 5 trades
                entry_time = tr['entry_time']
                entry_price = tr['entry_price']
                candle_open = df.loc[entry_time, 'open']
                assert abs(entry_price - candle_open) < 0.0005, f"Trade entry {entry_price} not executing at next-candle open {candle_open} (diff={abs(entry_price - candle_open)})"

        # Assertion 4: Verify persistent report generated and contains outcome
        from trainer.core.report import DEFAULT_REPORT_DIR
        reports_dir = Path(DEFAULT_REPORT_DIR)
        assert reports_dir.exists()
        report_files = list(reports_dir.glob(f"report_{runner.run_id}*.md"))
        assert len(report_files) > 0, f"Expected report file for run {runner.run_id}"
        report_content = report_files[0].read_text(encoding='utf-8')
        assert "Pipeline Outcome:" in report_content
        assert "COMPLETED" in report_content or "PARTIAL_FAILURE" in report_content

        # Assertion 5: Verify pip_size == 0.0001 for EURUSD
        eur_spec = get_spec('EURUSD')
        assert eur_spec.pip_size == 0.0001

    def test_full_pipeline_xauusd_real_data(self, tmp_path):
        """
        13.3: Cross-Instrument Real-Data Integration Test (XAUUSD Gold).
        Runs TrainerRunner directly on a real 4-month XAUUSD historical Parquet slice.
        """
        config = DEFAULT_CONFIG.copy()
        config['symbols'] = ['XAUUSD']
        config['timeframe'] = 'M15'
        config['data_start'] = '2019-01-01'
        config['data_end'] = '2020-05-01'
        config['opt_months'] = 2
        config['test_months'] = 1
        config['top_n_candidates'] = 1
        config['templates'] = ['ma_crossover']

        test_mandate = Mandate(
            mandate_id="TEST_MANDATE_XAU",
            firm_name="TestFirm",
            account_id="ACC_TEST",
            max_daily_loss_pct=50.0,
            max_overall_drawdown_pct=50.0,
            drawdown_type="static",
            min_trading_days=1,
            max_daily_risk_pct=5.0,
            symbol_universe=["XAUUSD"]
        )

        runner = TrainerRunner(config=config, quick_mode=True)
        runner.kb_path = tmp_path / "test_xau_kb.json"
        runner.mandate = test_mandate

        record = runner.run()

        # Assertion 1: Pipeline completes for XAUUSD
        assert record.outcome in ['COMPLETED', 'PARTIAL_FAILURE']
        assert record.symbol_status['XAUUSD']['status'] == 'COMPLETED'

        # Assertion 2: Verify XAUUSD spec parameters (contract size 100.0, pip_size 0.01, sanity_ok True)
        xau_spec = get_spec('XAUUSD')
        assert xau_spec.pip_size == 0.01
        assert xau_spec.contract_size == 100.0
        assert xau_spec.sanity_ok is True

        # Assertion 3: Verify Gold mandate evaluation on actual run decisions and sample equity curve
        log_path = Path(runner.log_path)
        assert log_path.exists()
        import json
        all_lines = [json.loads(line) for line in log_path.read_text(encoding='utf-8').strip().split('\n') if line]
        xau_decisions = [d for d in all_lines if d.get('run_id') == runner.run_id]
        assert len(xau_decisions) > 0, f"Expected decision records for run_id {runner.run_id} in decision_log.jsonl"
        for dec in xau_decisions:
            assert 'gate_9_mandate_compliant' in dec.get('gate_results', {})
            assert dec['gate_results']['gate_9_mandate_compliant'] is True

        from trainer.core.mandate_evaluator import evaluate_backtest_mandate
        sample_gold_equity = [10000.0, 10050.0, 9980.0, 10120.0]
        compliant, breach_reason = evaluate_backtest_mandate(sample_gold_equity, runner.mandate)
        assert compliant is True
        assert breach_reason is None

    def test_full_pipeline_failure_path_mandate_breach(self, tmp_path):
        """
        13.4: Failure-Path Integration Test.
        Drives an artificial tight mandate limit (0.00001% max daily loss), confirming candidates
        trip Gate 9, receive candidate.gate_status = 'FAILED', and are rejected with breach reason preserved in decision log & report.
        """
        strict_mandate = Mandate(
            mandate_id="STRICT_TEST",
            firm_name="TestFirm",
            account_id="ACC_STRICT",
            max_daily_loss_pct=0.00001,  # 0.00001% daily loss limit (guaranteed to breach on any loss)
            max_overall_drawdown_pct=0.00001,
            drawdown_type="static",
            min_trading_days=5,
            max_daily_risk_pct=1.0,
            symbol_universe=["EURUSD"]
        )

        config = DEFAULT_CONFIG.copy()
        config['symbols'] = ['EURUSD']
        config['timeframe'] = 'M15'
        config['data_start'] = '2019-01-01'
        config['data_end'] = '2020-05-01'
        config['opt_months'] = 2
        config['test_months'] = 1
        config['top_n_candidates'] = 1
        config['templates'] = ['ma_crossover']

        runner = TrainerRunner(config=config, quick_mode=True)
        runner.kb_path = tmp_path / "test_fail_kb.json"
        runner.mandate = strict_mandate

        record = runner.run()

        # Assertion 1: Candidate failed gate and was rejected
        assert record.outcome in ['COMPLETED', 'PARTIAL_FAILURE']
        assert record.candidates_rejected >= 1

        # Assertion 2: Decision log records gate_9_mandate_compliant == False and captures verbatim mandate breach reason
        log_path = Path(runner.log_path)
        assert log_path.exists()
        import json
        all_lines = [json.loads(line) for line in log_path.read_text(encoding='utf-8').strip().split('\n') if line]
        decisions = [d for d in all_lines if d.get('run_id') == runner.run_id]
        assert len(decisions) > 0
        breached_decision = decisions[-1]
        assert breached_decision['gate_results']['gate_9_mandate_compliant'] is False

        # Capture the actual breach reason string produced by evaluate_backtest_mandate() and passed into decision record flags
        flags_list = breached_decision.get('flags', [])
        assert len(flags_list) > 0, "Expected non-empty flags list in decision record"
        gate_flag = str(flags_list[0])
        assert "Mandate compliance breach:" in gate_flag, f"Expected Mandate compliance breach in flag, got: {gate_flag}"
        captured_breach_reason = gate_flag.split("Mandate compliance breach:")[-1].strip()
        assert len(captured_breach_reason) > 0, "Expected non-empty mandate breach reason"
        assert "MANDATE_" in captured_breach_reason

        rec_str = str(breached_decision.get('recommendation', ''))
        combined_decision_text = f"{rec_str} {gate_flag}"
        assert captured_breach_reason in combined_decision_text, \
            f"Expected captured breach reason '{captured_breach_reason}' verbatim in decision log, got: {combined_decision_text}"

        # Assertion 3: Generated report contains the exact captured breach reason verbatim
        from trainer.core.report import DEFAULT_REPORT_DIR
        reports_dir = Path(DEFAULT_REPORT_DIR)
        report_files = list(reports_dir.glob(f"report_{runner.run_id}*.md"))
        assert len(report_files) > 0
        report_content = report_files[0].read_text(encoding='utf-8')
        assert captured_breach_reason in report_content, \
            f"Expected captured breach reason '{captured_breach_reason}' verbatim in report content, got report snippet: {report_content[:500]}"
