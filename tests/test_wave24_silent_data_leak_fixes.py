"""
tests/test_wave24_silent_data_leak_fixes.py

Unit tests for WAVE 24 — Fixing Remaining 10 Silent Data Leaks:
1. Corrupt CUSUM state file raises RuntimeError (CUSUM_STATE_CORRUPT).
2. Position reconciliation error engages SAFE_MODE and retains ticket tracking.
3. Signal generation crash returns signal: ERROR and trips circuit breaker.
4. Strategy file hash read error raises RuntimeError (STRATEGY_FILE_READ_FAILED).
5. Coherence validator exception logs COHERENCE_VALIDATOR_ERROR.
6. Candidate scoring crash in adaptation logs CANDIDATE_SCORING_CRASH and returns -1.0.
7. Composite scoring crash in trainer_runner sets composite_score = None and scoring_status = 'ERROR'.
8. Filter test quick score crash logs CANDIDATE_SCORING_CRASH.
9. Combination test evaluation crash logs COMBINATION_TEST_ERROR.
10. Checkpoint write failure raises RuntimeError (CHECKPOINT_WRITE_FAILED).
"""

import sys
import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.core.degradation import DegradationMonitor
from engine.core.strategy_loader import StrategyLoader
from engine.core.signal_engine import SignalEngine
from engine.core.engine import Engine
from trainer.core.adaptation import ParameterSearch, FilterTesting, CombinationTesting, CandidateConfig
from trainer.core.checkpoint import CheckpointManager


class TestWave24SilentDataLeakFixes:

    # ── Item 1: CUSUM State Load ──────────────────────────────
    def test_cusum_corrupt_state_raises_runtime_error(self, tmp_path):
        """Item 1: Corrupt CUSUM state file must raise RuntimeError."""
        state_file = tmp_path / 'cusum_state.json'
        state_file.write_text("{ INVALID_JSON_DATA }", encoding='utf-8')

        with pytest.raises(RuntimeError, match="CUSUM_STATE_CORRUPT"):
            DegradationMonitor(
                strategy_id="ATS-TEST-001",
                wf_median_pf=1.5,
                wf_profitable_rate=0.6,
                cusum_state_file=state_file
            )

    # ── Item 2: Closed-Trade Reconciliation ───────────────────
    def test_closed_trade_reconciliation_failure_engages_safe_mode_and_retains_ticket(self, tmp_path):
        """Item 2: Closed-trade reconciliation failure engages SAFE_MODE and retains ticket in tracking."""
        engine = Engine(symbol="EURUSD", paper_mode=True)
        engine.last_known_tickets = [1001, 1002]

        with patch('engine.core.engine.get_open_positions', return_value=[]), \
             patch.object(engine, '_record_closed_trade', side_effect=RuntimeError("RECONCILIATION_DB_ERROR")), \
             patch('engine.core.engine.get_account_info', return_value={'equity': 10000.0}), \
             patch('engine.core.engine.mt5') as mock_mt5:

            mock_deal = MagicMock()
            mock_deal.ticket = 2001
            mock_deal.position_id = 1001
            mock_deal.entry = 1
            mock_deal.price = 1.0850
            mock_deal.volume = 0.1
            mock_deal.profit = 50.0
            mock_mt5.history_deals_get.return_value = [mock_deal]

            engine._reconcile_closed_positions([])

            # Assert SAFE_MODE was engaged
            assert engine.safe_mode is True
            # Assert ticket 1001 was retained so reconciliation tracking is not dropped
            assert 1001 in engine.last_known_tickets

    # ── Item 3: Signal Generation Fallback ────────────────────
    def test_signal_generation_crash_returns_error_payload(self):
        """Item 3: Exception during signal generation returns signal: ERROR."""
        mock_spec = MagicMock(parameters={'ma_type': 'SMA'})
        signal_engine = SignalEngine(spec=mock_spec)
        signal_engine.candles_processed = 2  # Bypass warmup

        mock_df = pd.DataFrame({'close': range(60)})
        with patch.object(signal_engine, '_compute_indicators', side_effect=ValueError("INDICATOR_COMPUTATION_CRASH")):
            res = signal_engine.process_candle(mock_df)

            assert res['signal'] == 'ERROR'
            assert 'SIGNAL_EVALUATION_FAILED' in res['reason']

    def test_engine_trips_circuit_breaker_on_signal_error(self, tmp_path):
        """Item 3 Engine Handling: Engine trips circuit breaker when signal: ERROR is returned."""
        engine = Engine(symbol="EURUSD", paper_mode=True)
        engine.healthy = True
        engine.strategy_loader = MagicMock()
        engine.strategy_loader.get_spec.return_value = MagicMock(parameters={})
        engine.strategy_loader.reload_if_changed.return_value = False
        engine.signal_engine = MagicMock()
        engine.signal_engine.process_candle.return_value = {'signal': 'ERROR', 'reason': 'SIGNAL_EVALUATION_FAILED: Test crash'}

        mock_df = pd.DataFrame({'close': [1.0850, 1.0860]}, index=pd.date_range('2026-01-01', periods=2, freq='15min'))

        with patch('engine.core.engine.fetch_candles', return_value=mock_df), \
             patch('engine.core.engine.get_account_info', return_value={'equity': 10000.0, 'balance': 10000.0}):

            engine._process_candle()
            assert engine.circuit_breaker_tripped is True

    # ── Item 4: Strategy File Hash ─────────────────────────────
    def test_strategy_file_hash_read_error_raises_runtime_error(self, tmp_path):
        """Item 4: File read error on existing file in _compute_file_hash raises RuntimeError."""
        strat_file = tmp_path / 'strategy.json'
        strat_file.write_text("{}", encoding='utf-8')
        loader = StrategyLoader(strategy_path=str(strat_file))

        with patch.object(Path, 'read_bytes', side_effect=OSError("Disk read failure")):
            with pytest.raises(RuntimeError, match="STRATEGY_FILE_READ_FAILED"):
                loader._compute_file_hash()

    # ── Item 5: Coherence Validation Skip ──────────────────────
    def test_coherence_validator_crash_logs_error(self, tmp_path, caplog):
        """Item 5: Exception during validate_coherence logs COHERENCE_VALIDATOR_ERROR."""
        import logging
        caplog.set_level(logging.ERROR)

        kb = MagicMock()
        kb.get_cached_score.return_value = None

        search = ParameterSearch(
            symbol="EURUSD",
            template="ma_crossover",
            df=MagicMock(),
            knowledge_base=kb,
            data_hash="hash123",
            run_id="run_123"
        )

        with patch('trainer.core.coherence_validator.validate_coherence', side_effect=RuntimeError("VALIDATOR_CRASH")), \
             patch('shared.instrument_spec.get_spec', return_value=MagicMock(sanity_ok=True, pip_size=0.0001)):

            try:
                search.run_optuna_search(MagicMock(), "test_prefix", n_trials=1)
            except RuntimeError as e:
                assert "SEARCH_FAILED" in str(e)

            assert any("COHERENCE_VALIDATOR_ERROR" in record.message for record in caplog.records)

    # ── Item 6 & 7: Composite Scoring Crash ────────────────────
    def test_candidate_scoring_crash_returns_negative_one(self):
        """Item 6: Candidate quick backtest crash returns -1.0 score and logs error."""
        search = ParameterSearch(
            symbol="EURUSD",
            template="ma_crossover",
            df=MagicMock(),
            knowledge_base=MagicMock(),
            data_hash="hash123",
            run_id="run_123"
        )

        with patch('trainer.core.adaptation.Backtester', side_effect=RuntimeError("BACKTEST_ENGINE_CRASH")):
            score = search._run_quick_backtest({'fast_ma': 10}, MagicMock())
            assert score == -1.0

    def test_trainer_runner_scoring_crash_sets_error_status_and_none_score(self, tmp_path):
        """Item 7 Real Method Invocation: Invoke TrainerRunner._run_symbol with calculate_composite_score mocked to raise, verifying actual method execution."""
        from trainer.core.trainer_runner import TrainerRunner, RunRecord
        import pandas as pd

        log_file = tmp_path / "decision_log.jsonl"
        runner = TrainerRunner(config={
            'symbols': ['EURUSD'],
            'top_n_candidates': 1,
            'initial_equity': 10000.0,
            'data_start': '2020-01-01',
            'data_end': '2023-01-01',
            'templates': ['ma_crossover'],
            'opt_months': 12,
            'test_months': 3,
            'timeframe': 'M15',
            'min_trades_opt': 10,
            'min_trades_test': 5
        })
        runner.log_path = log_file

        record = RunRecord(run_id="test_run", config=runner.config)
        kb = MagicMock()

        candidate = CandidateConfig(
            candidate_id="ATS-TEST-ERR",
            template="ma_crossover",
            symbol="EURUSD",
            timeframe="M15",
            parameters={'fast_ma': 10},
            composite_score=1.5
        )

        dates = pd.date_range("2020-01-01", periods=500, freq="15min")
        dummy_df = pd.DataFrame({
            'open': 1.1000, 'high': 1.1050, 'low': 1.0950, 'close': 1.1010, 'tick_volume': 100
        }, index=dates)

        wf_summary = {'metrics': {'calmar_ratio': 2.0, 'win_rate': 0.6}, 'trades': [1]*30, 'profit_factor': 1.8}

        with patch('trainer.core.trainer_runner.load_ohlcv', return_value=dummy_df), \
             patch('trainer.core.trainer_runner.compute_data_hash', return_value="hash123"), \
             patch('trainer.core.trainer_runner.AdaptationSystem') as MockAdaptation, \
             patch('trainer.core.trainer_runner.WalkForwardValidator') as MockWF, \
             patch('trainer.core.trainer_runner.aggregate_wf_results', return_value=wf_summary), \
             patch('trainer.core.trainer_runner.check_hard_gates', return_value=MagicMock(passed=True)), \
             patch('trainer.core.trainer_runner.calculate_composite_score', side_effect=RuntimeError("SCORING_FORMULA_CRASH: Formula calculation failure")):

            MockAdaptation.return_value.run.return_value = [candidate]
            MockWF.return_value.run.return_value = [MagicMock()]

            decisions, returned_candidates = runner._run_symbol('EURUSD', record, kb)

            assert len(returned_candidates) == 1
            returned_cand = returned_candidates[0]
            assert returned_cand.composite_score is None
            assert returned_cand.scoring_status == 'ERROR'
            assert returned_cand.gate_status == 'SCORING_ERROR'
            assert "SCORING_FORMULA_CRASH" in returned_cand.scoring_error

            assert len(decisions) == 1
            assert decisions[0].decision_path == 'AUTO_REJECT'
            assert decisions[0].decision_outcome == 'REJECTED'

    # ── Item 8: Filter Test Quick Score Crash ──────────────────
    def test_filter_test_quick_score_crash_logs_error(self, caplog):
        """Item 8: Exception during filter test quick score returns -1.0."""
        import logging
        caplog.set_level(logging.ERROR)

        filter_testing = FilterTesting(
            symbol="EURUSD",
            df_opt=MagicMock(),
            df_held_out=MagicMock(),
            knowledge_base=MagicMock()
        )

        score = filter_testing._quick_score({'fast_ma': 10}, MagicMock())
        assert score == -1.0
        assert any("CANDIDATE_SCORING_CRASH" in record.message for record in caplog.records)

    # ── Item 9: Combination Test Evaluation Crash ──────────────
    def test_combination_test_crash_logs_error(self, caplog):
        """Item 9: Exception in combination test logs COMBINATION_TEST_ERROR."""
        import logging
        caplog.set_level(logging.ERROR)

        engine = CombinationTesting(
            symbol="EURUSD",
            df=MagicMock()
        )

        ec = CandidateConfig(candidate_id="c1", template="ma_crossover", symbol="EURUSD", timeframe="M15", parameters={})
        xc = CandidateConfig(candidate_id="c2", template="ma_crossover", symbol="EURUSD", timeframe="M15", parameters={})

        with patch('trainer.core.adaptation.Backtester', side_effect=RuntimeError("BACKTESTER_CRASH")):
            res = engine._test_one_combination(ec, xc, "tmpl1", "tmpl2")
            assert res is None
            assert any("COMBINATION_TEST_ERROR" in record.message for record in caplog.records)

    # ── Item 10: Checkpoint Write Failure ─────────────────────
    def test_checkpoint_write_failure_raises_runtime_error(self, tmp_path):
        """Item 10: Atomic write failure in save must raise RuntimeError."""
        bad_path = tmp_path / 'non_existent_dir' / 'checkpoint.json'
        mgr = CheckpointManager(checkpoint_path=bad_path)

        with patch('trainer.core.checkpoint.atomic_write_json', side_effect=OSError("Disk full or permission denied")):
            with pytest.raises(RuntimeError, match="CHECKPOINT_WRITE_FAILED"):
                mgr.save("run_123")
