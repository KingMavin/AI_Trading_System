"""
Unit tests for WAVE 9 — Trainer Long-Run Robustness.
Tests Checkpointing, Resource Watchdog, Bounded Parallelism, Per-Symbol Isolation, and Concurrency Locks.
Run: pytest tests/test_trainer_robustness.py -v
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.core.checkpoint import CheckpointManager


class TestTrainerCheckpointing:

    def test_checkpoint_atomic_write_and_resume_boundary(self, tmp_path):
        checkpoint_file = tmp_path / 'trainer_checkpoint.json'
        cp1 = CheckpointManager(checkpoint_file)

        run_id = 'RUN-OVERNIGHT-001'
        symbol = 'EURUSD'
        window_idx = 0
        candidates = ['CAND-001', 'CAND-002', 'CAND-003', 'CAND-004', 'CAND-005']

        # Process N=3 candidates out of M=5
        for i in range(3):
            cand_id = candidates[i]
            cp1.mark_candidate_completed(symbol, window_idx, cand_id, {'score': 0.8 + i*0.05})

        cp1.save(run_id)
        assert checkpoint_file.exists()

        # Simulate Crash & Restart -> Load Checkpoint in fresh manager
        cp2 = CheckpointManager(checkpoint_file)
        assert cp2.load() is True
        assert cp2.run_id == run_id

        # Determine remaining candidates to process
        remaining_candidates = []
        for cand_id in candidates:
            if not cp2.is_candidate_completed(symbol, window_idx, cand_id):
                remaining_candidates.append(cand_id)

        # Assert exactly M - N = 5 - 3 = 2 candidates remain
        assert len(remaining_candidates) == 2
        assert remaining_candidates == ['CAND-004', 'CAND-005']

        # Confirm completed candidates are NOT re-processed
        for i in range(3):
            assert cp2.is_candidate_completed(symbol, window_idx, candidates[i]) is True

    def test_corrupt_checkpoint_fails_loudly(self, tmp_path):
        checkpoint_file = tmp_path / 'trainer_checkpoint.json'

        # Case A: Invalid JSON string
        checkpoint_file.write_text("{corrupt_json: true,", encoding='utf-8')
        cp = CheckpointManager(checkpoint_file)
        with pytest.raises(RuntimeError, match="Corrupt checkpoint file"):
            cp.load()

        # Case B: Valid JSON but missing required schema keys
        checkpoint_file.write_text(json.dumps({'run_id': 'TEST'}), encoding='utf-8')
        with pytest.raises(RuntimeError, match="Corrupt checkpoint file"):
            cp.load()


class TestResourceWatchdog:

    def test_resource_watchdog_thresholds_and_emergency_checkpoint(self, tmp_path, monkeypatch):
        from trainer.core.resource_monitor import ResourceWatchdog, check_resource_status
        import trainer.core.resource_monitor as rm_mod

        checkpoint_file = tmp_path / 'trainer_checkpoint.json'
        cp = CheckpointManager(checkpoint_file)
        cp.completed_symbols.add('EURUSD')
        watchdog = ResourceWatchdog(cp, path_for_disk=tmp_path)

        run_id = 'RUN-RESOURCE-001'

        # Test GREEN (RAM 50%, Disk 50%)
        monkeypatch.setattr(rm_mod, 'check_resource_status', lambda path: ('GREEN', 50.0, 50.0))
        assert watchdog.evaluate(run_id) == 'GREEN'

        # Test AMBER (RAM 82%, Disk 50%)
        monkeypatch.setattr(rm_mod, 'check_resource_status', lambda path: ('AMBER', 82.0, 50.0))
        assert watchdog.evaluate(run_id) == 'AMBER'

        # Test RED (RAM 92%, Disk 50%)
        monkeypatch.setattr(rm_mod, 'check_resource_status', lambda path: ('RED', 92.0, 50.0))
        assert watchdog.evaluate(run_id) == 'RED'

        # Test CRITICAL (RAM 97%, Disk 50%) -> Emergency atomic checkpoint & RuntimeError
        monkeypatch.setattr(rm_mod, 'check_resource_status', lambda path: ('CRITICAL', 97.0, 50.0))

        with pytest.raises(RuntimeError, match="RESOURCE_MONITOR_CRITICAL"):
            watchdog.evaluate(run_id)

        # Assert checkpoint was saved before RuntimeError was raised
        assert checkpoint_file.exists()
        res_cp = CheckpointManager(checkpoint_file)
        assert res_cp.load() is True
        assert 'EURUSD' in res_cp.completed_symbols


class TestBoundedProcessPool:

    def test_get_bounded_pool_size_leaves_one_core_free(self, monkeypatch):
        from trainer.core.adaptation import get_bounded_pool_size

        monkeypatch.setattr('os.cpu_count', lambda: 8)
        assert get_bounded_pool_size() == 7
        assert get_bounded_pool_size(max_workers=100) == 7
        assert get_bounded_pool_size(max_workers=4) == 4

    def test_execute_isolated_task_catches_worker_exceptions(self):
        from trainer.core.adaptation import execute_isolated_task

        def buggy_worker(x):
            if x == 0:
                raise ValueError("Division by zero in worker")
            return 100 / x

        res_ok = execute_isolated_task(buggy_worker, 5)
        assert res_ok['status'] == 'SUCCESS'
        assert res_ok['result'] == 20.0

        res_err = execute_isolated_task(buggy_worker, 0)
        assert res_err['status'] == 'FAILED'
        assert "Division by zero" in res_err['error']


class TestPerSymbolIsolation:

    def test_per_symbol_failure_isolation(self, monkeypatch):
        from trainer.core.trainer_runner import TrainerRunner

        config = {
            'symbols': ['EURUSD', 'BADSYMBOL', 'GBPUSD'],
            'timeframe': 'M15',
            'templates': ['ma_crossover'],
            'data_start': '2020-01-01',
            'data_end': '2022-12-31',
            'initial_equity': 10000.0,
        }
        runner = TrainerRunner(config=config, quick_mode=True)

        monkeypatch.setattr('trainer.core.trainer_runner.KnowledgeBase', MagicMock)
        monkeypatch.setattr('trainer.core.trainer_runner.print_run_header', lambda c, r: None)
        monkeypatch.setattr('trainer.core.trainer_runner.print_final_report', lambda r, d: None)

        def mock_run_symbol(symbol, record, kb):
            if symbol == 'BADSYMBOL':
                raise ValueError("Corrupt OHLCV data for BADSYMBOL")
            return [], [MagicMock()]

        monkeypatch.setattr(runner, '_run_symbol', mock_run_symbol)

        record = runner.run()

        # Outcome MUST be PARTIAL_FAILURE, not a crash!
        assert record.outcome == 'PARTIAL_FAILURE'
        assert record.symbol_status['EURUSD']['status'] == 'COMPLETED'
        assert record.symbol_status['BADSYMBOL']['status'] == 'FAILED'
        assert "Corrupt OHLCV data" in record.symbol_status['BADSYMBOL']['error']
        assert record.symbol_status['GBPUSD']['status'] == 'COMPLETED'


class TestConcurrencyLock:

    def test_active_lock_blocks_second_trainer_instance(self, tmp_path, monkeypatch):
        from trainer.core.lock import ConcurrencyLock
        import trainer.core.lock as lock_mod

        lock_file = tmp_path / 'trainer.lock'
        lock1 = ConcurrencyLock(lock_file)
        lock1.acquire()
        assert lock_file.exists()

        monkeypatch.setattr(lock_mod, 'is_pid_running', lambda pid: True)

        lock2 = ConcurrencyLock(lock_file)
        with pytest.raises(RuntimeError, match="TRAINER_CONCURRENCY_LOCK_ACTIVE"):
            lock2.acquire()

        lock1.release()
        assert not lock_file.exists()

    def test_stale_lock_detected_and_cleared(self, tmp_path, monkeypatch):
        from trainer.core.lock import ConcurrencyLock
        import trainer.core.lock as lock_mod

        lock_file = tmp_path / 'trainer.lock'
        stale_data = {'pid': 999999, 'started_at': '2026-07-28T10:00:00+00:00'}
        lock_file.write_text(json.dumps(stale_data), encoding='utf-8')

        # Mock pid 999999 as NOT running (stale lock)
        monkeypatch.setattr(lock_mod, 'is_pid_running', lambda pid: False if pid == 999999 else True)

        lock2 = ConcurrencyLock(lock_file)
        lock2.acquire() # Must detect staleness, remove stale file, and acquire cleanly!

        assert lock2.acquired is True
        data = json.loads(lock_file.read_text(encoding='utf-8'))
        assert data['pid'] == os.getpid()

        lock2.release()
        assert not lock_file.exists()
