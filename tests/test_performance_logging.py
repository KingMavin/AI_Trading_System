"""
Unit tests for WAVE 10.3 & 10.4 — Performance Observation & Continuous Logging.
Verifies continuous resource usage logging, Engine candle-processing timing observation, and non-blocking failure isolation.
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.core.resource_monitor import ResourceWatchdog, log_performance_record, PERFORMANCE_LOG_FILE
from trainer.core.checkpoint import CheckpointManager


class TestPerformanceLogging:

    def test_continuous_performance_logging_in_resource_watchdog(self, tmp_path, monkeypatch):
        perf_log = tmp_path / "performance.jsonl"
        monkeypatch.setattr("trainer.core.resource_monitor.PERFORMANCE_LOG_FILE", perf_log)

        cp = CheckpointManager(tmp_path / "cp.json")
        watchdog = ResourceWatchdog(cp, path_for_disk=tmp_path)

        import trainer.core.resource_monitor as rm_mod
        monkeypatch.setattr(rm_mod, 'check_resource_status', lambda path: ('GREEN', 45.0, 30.0))

        status = watchdog.evaluate("RUN-PERF-001")
        assert status == "GREEN"

        assert perf_log.exists()
        lines = perf_log.read_text(encoding='utf-8').strip().split('\n')
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record['source_system'] == "TRAINER"
        assert record['metric_type'] == "RESOURCE_USAGE"
        assert record['ram_pct'] == 45.0
        assert record['disk_pct'] == 30.0
        assert record['status'] == "GREEN"

    def test_engine_candle_timing_performance_logging_real_execution(self, tmp_path, monkeypatch):
        """Real end-to-end integration test of Engine._process_candle() writing timing performance log."""
        from engine.core.engine import Engine

        perf_log = tmp_path / "performance.jsonl"
        monkeypatch.setattr("trainer.core.resource_monitor.PERFORMANCE_LOG_FILE", perf_log)

        # Mock Engine dependencies
        monkeypatch.setattr("engine.core.engine.KILL_SWITCH_FILE", MagicMock(exists=lambda: False))
        monkeypatch.setattr("engine.core.engine.Engine._check_connection_health", lambda self: True)
        monkeypatch.setattr("engine.core.engine.fetch_candles", lambda s, t, c: MagicMock(index=[MagicMock(to_pydatetime=lambda: MagicMock(date=lambda: MagicMock()))]))
        monkeypatch.setattr("engine.core.engine.get_account_info", lambda: {'equity': 10000.0, 'balance': 10000.0})
        monkeypatch.setattr("engine.core.engine.get_open_positions", lambda s: [])
        monkeypatch.setattr("engine.core.engine.Engine.save_snapshot", lambda self: None)
        monkeypatch.setattr("engine.core.engine.write_state", lambda d: None)
        monkeypatch.setattr("engine.core.engine.DegradationMonitor.write_status_file", lambda self: None)

        engine = Engine('EURUSD', paper_mode=True)
        engine.strategy_loader = MagicMock()
        engine.strategy_loader.get_spec.return_value = MagicMock(parameters={})
        engine.strategy_loader.reload_if_changed.return_value = False
        engine.signal_engine = MagicMock()
        engine.signal_engine.generate_signal.return_value = {'signal': 'NONE'}
        engine.degradation = MagicMock()

        # Run _process_candle() end-to-end through instrumented timing logic
        engine._process_candle()

        assert perf_log.exists()
        record = json.loads(perf_log.read_text(encoding='utf-8').strip())

        assert record['source_system'] == "ENGINE"
        assert record['metric_type'] == "CANDLE_TIMING"
        assert record['symbol'] == "EURUSD"
        assert record['candles_processed'] == 1

        # Real timing float assertions and sanity bounds
        total_ms = record['total_duration_ms']
        mt5_ms = record['mt5_duration_ms']
        eval_ms = record['eval_duration_ms']

        assert isinstance(total_ms, float)
        assert isinstance(mt5_ms, float)
        assert isinstance(eval_ms, float)

        assert total_ms >= mt5_ms
        assert eval_ms >= 0.0

    def test_engine_performance_logging_skipped_when_df_is_none(self, tmp_path, monkeypatch):
        """Confirm performance logging is correctly skipped (not crashed) when fetch_candles returns None."""
        from engine.core.engine import Engine

        perf_log = tmp_path / "performance.jsonl"
        monkeypatch.setattr("trainer.core.resource_monitor.PERFORMANCE_LOG_FILE", perf_log)

        monkeypatch.setattr("engine.core.engine.KILL_SWITCH_FILE", MagicMock(exists=lambda: False))
        monkeypatch.setattr("engine.core.engine.Engine._check_connection_health", lambda self: True)
        monkeypatch.setattr("engine.core.engine.fetch_candles", lambda s, t, c: None)

        engine = Engine('EURUSD', paper_mode=True)
        engine.strategy_loader = MagicMock()
        engine.strategy_loader.get_spec.return_value = MagicMock(parameters={})
        engine.strategy_loader.reload_if_changed.return_value = False

        # Run _process_candle() — must exit early on df is None without raising or logging performance timing
        engine._process_candle()

        assert not perf_log.exists() or len(perf_log.read_text(encoding='utf-8').strip()) == 0

    def test_engine_continues_even_if_performance_logging_raises(self, monkeypatch):
        from engine.core.engine import Engine

        def buggy_logger(*args, **kwargs):
            raise IOError("Simulated disk error in performance logger")

        monkeypatch.setattr("trainer.core.resource_monitor.log_performance_record", buggy_logger)

        # Mock Engine methods so _process_candle runs cleanly
        monkeypatch.setattr("engine.core.engine.KILL_SWITCH_FILE", MagicMock(exists=lambda: False))
        monkeypatch.setattr("engine.core.engine.Engine._check_connection_health", lambda self: True)
        monkeypatch.setattr("engine.core.engine.fetch_candles", lambda s, t, c: MagicMock(index=[MagicMock(to_pydatetime=lambda: MagicMock(date=lambda: MagicMock()))]))
        monkeypatch.setattr("engine.core.engine.get_account_info", lambda: {'equity': 10000.0, 'balance': 10000.0})
        monkeypatch.setattr("engine.core.engine.get_open_positions", lambda s: [])
        monkeypatch.setattr("engine.core.engine.Engine.save_snapshot", lambda self: None)
        monkeypatch.setattr("engine.core.engine.write_state", lambda d: None)
        monkeypatch.setattr("engine.core.engine.DegradationMonitor.write_status_file", lambda self: None)

        engine = Engine('EURUSD', paper_mode=True)
        engine.strategy_loader = MagicMock()
        engine.strategy_loader.get_spec.return_value = MagicMock(parameters={})
        engine.strategy_loader.reload_if_changed.return_value = False
        engine.signal_engine = MagicMock()
        engine.signal_engine.generate_signal.return_value = {'signal': 'NONE'}
        engine.degradation = MagicMock()

        # Execute _process_candle() — must NOT raise even when log_performance_record fails
        engine._process_candle()
        assert engine.candles_processed == 1
