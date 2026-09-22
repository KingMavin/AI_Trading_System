"""
tests/test_silent_data_leak_fixes.py

Unit tests for WAVE 23 — Fixing Highest-Priority Silent Data Leaks:
1. Corrupt strategy_bench.json raises RuntimeError (BENCH_STORE_CORRUPT) and prevents promotion.
2. write_audit_event failure raises RuntimeError (AUDIT_LEDGER_WRITE_FAILED) and aborts safe mode override before process modification.
3. Missing bench file initializes cleanly via _ensure_init().
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.core.bench import StrategyBenchStore
from engine.dashboard import process_helpers as ph
from engine.dashboard import app as app_mod


class TestSilentDataLeakFixes:

    def test_corrupt_bench_store_raises_runtime_error_and_prevents_promotion(self, tmp_path):
        """Item 1: Corrupt strategy_bench.json must raise RuntimeError, failing closed."""
        bench_file = tmp_path / 'strategy_bench.json'
        bench_file.write_text("{ CORRUPT_JSON_DATA }", encoding='utf-8')

        store = StrategyBenchStore(bench_file=bench_file)

        # 1. Direct _read_file, get_by_id, and get_all must raise RuntimeError
        with pytest.raises(RuntimeError, match="BENCH_STORE_CORRUPT"):
            store.get_all()

        with pytest.raises(RuntimeError, match="BENCH_STORE_CORRUPT"):
            store.get_by_id('ATS-MA-MANUAL-001')

        with pytest.raises(RuntimeError, match="BENCH_STORE_CORRUPT"):
            store.get_by_status('DEPLOYED')

    def test_valid_missing_bench_file_initializes_cleanly(self, tmp_path):
        """Item 1 Validation: Missing bench file initializes stub cleanly without error."""
        bench_file = tmp_path / 'strategy_bench.json'
        assert not bench_file.exists()

        store = StrategyBenchStore(bench_file=bench_file)
        active = store.get_by_status('DEPLOYED')

        assert len(active) == 1
        assert active[0]['strategy_id'] == 'ATS-MA-MANUAL-001'
        assert bench_file.exists()

    def test_write_audit_event_raises_runtime_error_on_write_failure(self, tmp_path, monkeypatch):
        """Item 2: write_audit_event must raise RuntimeError if file write fails."""
        bad_ledger = tmp_path / 'non_existent_dir' / 'read_only_ledger.jsonl'
        monkeypatch.setattr(ph, 'AUDIT_LEDGER', bad_ledger)

        # Force write to raise PermissionError/OSError
        def mock_open_fail(*args, **kwargs):
            raise OSError("Permission denied / Disk error simulation")

        monkeypatch.setattr('builtins.open', mock_open_fail)

        with pytest.raises(RuntimeError, match="AUDIT_LEDGER_WRITE_FAILED"):
            ph.write_audit_event('SAFE_MODE_MANUAL_OVERRIDE', 'User cleared safe mode')

    def test_safe_mode_override_aborts_when_audit_write_fails(self, monkeypatch):
        """Item 2 Real-Path Verification: Invoking cmd_clear_safe_mode() via TestClient aborts before process changes when write_audit_event fails."""
        pytest.importorskip('fastapi')
        pytest.importorskip('httpx')
        from fastapi.testclient import TestClient

        mock_stop_engine = MagicMock(return_value=(True, "Stopped"))
        mock_spawn_engine = MagicMock(return_value=12345)

        monkeypatch.setattr(app_mod, 'graceful_stop_engine', mock_stop_engine)
        monkeypatch.setattr(app_mod, '_spawn_engine', mock_spawn_engine)
        monkeypatch.setattr(app_mod, 'get_engine_lock_info', lambda: {'pid': 9999, 'symbol': 'EURUSD'})

        def mock_write_audit_fail(action, reason, metadata=None):
            raise RuntimeError(f"AUDIT_LEDGER_WRITE_FAILED: Simulated write failure for '{action}'")

        monkeypatch.setattr(app_mod, 'write_audit_event', mock_write_audit_fail)

        client = TestClient(app_mod.app, raise_server_exceptions=False)
        response = client.post("/api/command/engine/clear_safe_mode", json={"reason": "User cleared safe mode"})

        # Assert exception caused server 500 error or exception response
        assert response.status_code == 500

        # Assert downstream process modification functions were NEVER called by the route handler
        assert mock_stop_engine.call_count == 0
        assert mock_spawn_engine.call_count == 0
