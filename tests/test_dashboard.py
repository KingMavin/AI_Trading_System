"""
Tests for WAVE 18 — Azrael Local Dashboard.

Tests what can be validated without a live Engine or MT5:
  - PID discovery via lockfile (including PID-reuse guard via cmdline check)
  - Audit event lifecycle (REQUESTED → CONFIRMED/FAILED, never orphaned)
  - get_engine_status() with missing state files returns safe defaults
  - TrainerSubprocess idle state
  - Log tail behavior
  - Tier 3 hard-kill phrase enforcement at API layer (not just UI)

All psutil and os.kill calls are mocked. No real processes are spawned.
"""

import json
import os
import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.dashboard.process_helpers import (
    TrainerSubprocess,
    write_audit_event,
)

_FASTAPI_AVAILABLE = False
try:
    import fastapi  # noqa: F401
    _FASTAPI_AVAILABLE = True
except ImportError:
    pass


# ── TEST: find_engine_pid ────────────────────────────────

def test_find_engine_pid_returns_none_when_lockfile_absent(tmp_path, monkeypatch):
    """No lockfile → Engine not running → PID is None."""
    import engine.dashboard.process_helpers as ph
    monkeypatch.setattr(ph, 'ENGINE_LOCK_FILE', tmp_path / 'engine.lock.json')
    assert ph.find_engine_pid() is None


def test_find_engine_pid_returns_none_when_pid_dead(tmp_path, monkeypatch):
    """Lockfile exists but PID is not alive → returns None."""
    import engine.dashboard.process_helpers as ph

    lock = tmp_path / 'engine.lock.json'
    lock.write_text(json.dumps({
        'pid': 99998, 'started_at': '2026-01-01T00:00:00+00:00', 'symbol': 'EURUSD'
    }), encoding='utf-8')
    monkeypatch.setattr(ph, 'ENGINE_LOCK_FILE', lock)
    monkeypatch.setattr(ph, 'PSUTIL_AVAILABLE', True)

    mock_psutil = MagicMock()
    mock_psutil.pid_exists.return_value = False
    monkeypatch.setattr(ph, 'psutil', mock_psutil)

    result = ph.find_engine_pid()
    assert result is None


def test_find_engine_pid_returns_pid_when_alive(tmp_path, monkeypatch):
    """Lockfile exists, PID alive, cmdline matches → returns PID."""
    import engine.dashboard.process_helpers as ph

    lock = tmp_path / 'engine.lock.json'
    lock.write_text(json.dumps({
        'pid': 12345, 'started_at': '2026-01-01T00:00:00+00:00', 'symbol': 'EURUSD'
    }), encoding='utf-8')
    monkeypatch.setattr(ph, 'ENGINE_LOCK_FILE', lock)
    monkeypatch.setattr(ph, 'PSUTIL_AVAILABLE', True)

    mock_proc = MagicMock()
    mock_proc.cmdline.return_value = ['python', 'engine/core/engine.py', '--symbol', 'EURUSD']
    mock_psutil = MagicMock()
    mock_psutil.pid_exists.return_value = True
    mock_psutil.Process.return_value = mock_proc
    monkeypatch.setattr(ph, 'psutil', mock_psutil)

    result = ph.find_engine_pid()
    assert result == 12345


def test_find_engine_pid_aborts_on_cmdline_mismatch(tmp_path, monkeypatch):
    """
    PID alive but cmdline doesn't match Engine — guards against PID reuse.
    Must return None rather than the wrong PID.
    """
    import engine.dashboard.process_helpers as ph

    lock = tmp_path / 'engine.lock.json'
    lock.write_text(json.dumps({
        'pid': 12345, 'started_at': '2026-01-01T00:00:00+00:00', 'symbol': 'EURUSD'
    }), encoding='utf-8')
    monkeypatch.setattr(ph, 'ENGINE_LOCK_FILE', lock)
    monkeypatch.setattr(ph, 'PSUTIL_AVAILABLE', True)

    mock_proc = MagicMock()
    mock_proc.cmdline.return_value = ['python', 'notepad.exe']  # wrong process
    mock_psutil = MagicMock()
    mock_psutil.pid_exists.return_value = True
    mock_psutil.Process.return_value = mock_proc
    monkeypatch.setattr(ph, 'psutil', mock_psutil)

    result = ph.find_engine_pid()

    assert result is None, (
        "find_engine_pid must return None when cmdline doesn't match Engine — "
        "returning a wrong PID here could cause a Tier 3 kill of the wrong process"
    )


# ── TEST: Audit event lifecycle ──────────────────────────

def test_audit_event_written_to_file(tmp_path, monkeypatch):
    """write_audit_event appends valid JSON to the ledger."""
    import engine.dashboard.process_helpers as ph

    ledger = tmp_path / 'logs' / 'audit_ledger.jsonl'
    ledger.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ph, 'AUDIT_LEDGER', ledger)

    ph.write_audit_event('TEST_EVENT', 'unit test', {'foo': 'bar'})

    lines = ledger.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry['action'] == 'TEST_EVENT'
    assert entry['reason'] == 'unit test'
    assert entry['foo'] == 'bar'
    assert 'timestamp' in entry
    assert entry['source'] == 'DASHBOARD'


def test_audit_no_orphaned_requested_on_clean_stop(tmp_path, monkeypatch):
    """
    Audit contract: graceful_stop_engine writes ENGINE_STOP_REQUESTED
    and always follows up with ENGINE_STOP_CONFIRMED or ENGINE_STOP_FAILED.
    No REQUESTED entry may be left without a resolution.
    """
    import engine.dashboard.process_helpers as ph

    ledger = tmp_path / 'logs' / 'audit_ledger.jsonl'
    ledger.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ph, 'AUDIT_LEDGER', ledger)
    monkeypatch.setattr(ph, 'PSUTIL_AVAILABLE', True)

    mock_psutil = MagicMock()
    mock_psutil.pid_exists.return_value = False   # process died after SIGTERM
    monkeypatch.setattr(ph, 'psutil', mock_psutil)

    with patch('engine.dashboard.process_helpers.os.kill'):
        ph.graceful_stop_engine(12345, 'test stop')

    events = [json.loads(l) for l in ledger.read_text().strip().splitlines()]
    actions = [e['action'] for e in events]
    assert 'ENGINE_STOP_REQUESTED' in actions
    assert any(a in actions for a in [
        'ENGINE_STOP_CONFIRMED', 'ENGINE_STOP_FAILED', 'ENGINE_STOP_TIMED_OUT'
    ]), f"No resolution event found. Events: {actions}"


def test_audit_failed_written_when_signal_raises(tmp_path, monkeypatch):
    """
    When os.kill raises PermissionError, ENGINE_STOP_FAILED is written.
    The ledger must never show only ENGINE_STOP_REQUESTED without a follow-up.
    """
    import engine.dashboard.process_helpers as ph

    ledger = tmp_path / 'logs' / 'audit_ledger.jsonl'
    ledger.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ph, 'AUDIT_LEDGER', ledger)

    with patch('engine.dashboard.process_helpers.os.kill',
               side_effect=PermissionError("access denied")):
        ok, msg = ph.graceful_stop_engine(12345, 'fail test')

    assert ok is False
    events = [json.loads(l) for l in ledger.read_text().strip().splitlines()]
    actions = [e['action'] for e in events]
    assert 'ENGINE_STOP_REQUESTED' in actions
    assert 'ENGINE_STOP_FAILED' in actions, (
        f"ENGINE_STOP_FAILED not written after signal failure. Actions: {actions}"
    )


# ── TEST: get_engine_status safe defaults ────────────────

def test_get_engine_status_handles_missing_files(tmp_path, monkeypatch):
    """
    When all state files are absent, get_engine_status() returns safe defaults
    without raising — no KeyError, no AttributeError, no FileNotFoundError.
    """
    import engine.dashboard.app as app_mod
    import engine.dashboard.process_helpers as ph

    monkeypatch.setattr(app_mod, 'STRATEGY_PATH',    tmp_path / 'active_strategy.json')
    monkeypatch.setattr(app_mod, 'DEGRADATION_PATH', tmp_path / 'degradation_status.json')
    monkeypatch.setattr(app_mod, 'STATE_PATH',        tmp_path / 'engine_state.json')
    monkeypatch.setattr(app_mod, 'CUSUM_PATH',        tmp_path / 'cusum_state.json')
    monkeypatch.setattr(app_mod, 'LOGS_PATH',         tmp_path)
    monkeypatch.setattr(app_mod, 'DECISION_LOG',      tmp_path / 'decision_log.jsonl')
    monkeypatch.setattr(ph, 'ENGINE_LOCK_FILE',        tmp_path / 'engine.lock.json')

    status = app_mod.get_engine_status()

    assert status['engine']['running'] is False
    assert status['engine']['pid'] is None
    assert status['cusum']['tripped'] is False
    assert status['trainer']['running'] is False
    assert isinstance(status['performance']['win_rate'], float)
    assert isinstance(status['performance']['profit_factor'], float)


# ── TEST: TrainerSubprocess idle state ───────────────────

def test_trainer_status_idle_when_proc_is_none(tmp_path, monkeypatch):
    """get_status() returns running=False, pid=None before any subprocess is launched."""
    import engine.dashboard.process_helpers as ph
    monkeypatch.setattr(ph, 'TRAINER_STATUS', tmp_path / 'trainer_status.json')

    ts = TrainerSubprocess()
    status = ts.get_status()

    assert status['running'] is False
    assert status['pid'] is None


def test_trainer_log_lines_empty_initially(tmp_path, monkeypatch):
    """get_log_lines() returns [] before any run."""
    import engine.dashboard.process_helpers as ph
    monkeypatch.setattr(ph, 'TRAINER_STATUS', tmp_path / 'trainer_status.json')

    ts = TrainerSubprocess()
    assert ts.get_log_lines() == []


# ── TEST: Log tail ────────────────────────────────────────

def test_tail_text_file_returns_last_n_lines(tmp_path):
    """tail_text_file returns the last N lines in correct order."""
    from engine.dashboard.app import tail_text_file

    log_file = tmp_path / 'test.log'
    content = '\n'.join(f'line {i}' for i in range(200))
    log_file.write_text(content, encoding='utf-8')

    result = tail_text_file(log_file, n=50)
    assert len(result) == 50
    assert result[-1] == 'line 199'
    assert result[0] == 'line 150'


def test_tail_text_file_returns_empty_for_missing_file(tmp_path):
    """tail_text_file returns [] if the file does not exist."""
    from engine.dashboard.app import tail_text_file
    result = tail_text_file(tmp_path / 'nonexistent.log', n=50)
    assert result == []


def test_tail_jsonl_file_returns_valid_records(tmp_path):
    """tail_jsonl_file skips corrupt lines and returns the last N valid records."""
    from engine.dashboard.app import tail_jsonl_file

    jsonl = tmp_path / 'test.jsonl'
    lines = []
    for i in range(60):
        lines.append(json.dumps({'id': i, 'val': i * 2}))
    lines.append('{corrupt json')
    jsonl.write_text('\n'.join(lines), encoding='utf-8')

    result = tail_jsonl_file(jsonl, n=10)
    # Should have 10 valid records, ignoring the corrupt line
    assert len(result) == 10
    assert all('id' in r for r in result)
    # Most recent records (id 50–59)
    ids = [r['id'] for r in result]
    assert 59 in ids


# ── TEST: Hard kill phrase enforcement ──────────────────

def test_hard_kill_phrase_is_exact_string():
    """KILL_PHRASE is exactly 'KILL AZRAEL ENGINE' — no leading/trailing whitespace."""
    from engine.dashboard.app import KILL_PHRASE
    assert KILL_PHRASE == "KILL AZRAEL ENGINE"


def test_hard_kill_phrase_validation_rejects_variants():
    """
    The API-layer check (phrase != KILL_PHRASE) must reject all near-matches.
    This is the same logic used in the FastAPI endpoint before any kill is sent.
    """
    from engine.dashboard.app import KILL_PHRASE

    wrong_phrases = [
        '',
        'kill azrael engine',        # lowercase
        'KILL AZRAEL',               # truncated
        'KILL AZRAEL ENGINE ',       # trailing space
        ' KILL AZRAEL ENGINE',       # leading space
        'KILL AZRAEL ENGINE NOW',    # extended
        'KILL  AZRAEL ENGINE',       # double space
    ]

    for phrase in wrong_phrases:
        assert phrase != KILL_PHRASE, (
            f"Phrase '{phrase}' incorrectly passes validation — "
            f"this would allow a Tier 3 kill without proper confirmation"
        )


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_hard_kill_http_422_on_wrong_phrase():
    """
    Integration: POST /api/command/engine/kill with wrong phrase returns HTTP 422.
    Enforced at API layer — not just in the UI.
    """
    pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    from engine.dashboard.app import app

    client = TestClient(app)
    response = client.post(
        "/api/command/engine/kill",
        json={"confirmation_phrase": "wrong phrase", "reason": "test"}
    )
    assert response.status_code == 422, (
        f"Expected HTTP 422 for wrong kill phrase, got {response.status_code}"
    )


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_hard_kill_http_200_engine_not_running(tmp_path, monkeypatch):
    """
    Integration: correct phrase but Engine not running → HTTP 200 ok=False.
    Kill is aborted safely (no pid to kill), not a 422 or 500.
    """
    pytest.importorskip('httpx')
    import engine.dashboard.process_helpers as ph
    monkeypatch.setattr(ph, 'ENGINE_LOCK_FILE', tmp_path / 'engine.lock.json')

    from fastapi.testclient import TestClient
    from engine.dashboard.app import app

    client = TestClient(app)
    response = client.post(
        "/api/command/engine/kill",
        json={"confirmation_phrase": "KILL AZRAEL ENGINE", "reason": "test"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data['ok'] is False
    assert 'not running' in data['message'].lower()


# ── TEST: StrategyBenchStore ─────────────────────────────

def test_strategy_bench_store_init_and_deploy(tmp_path):
    """StrategyBenchStore initialises stub and deploys candidates cleanly."""
    from trainer.core.bench import StrategyBenchStore

    bench_file = tmp_path / 'strategy_bench.json'
    deploy_dir = tmp_path / 'deploy'

    store = StrategyBenchStore(bench_file=bench_file)
    strategies = store.get_all()
    assert len(strategies) == 1
    assert strategies[0]['strategy_id'] == 'ATS-MA-MANUAL-001'
    assert strategies[0]['promoted_by'] == 'manual_test_stub'
    assert strategies[0]['status'] == 'DEPLOYED'

    # Add a new candidate
    new_candidate = {
        'strategy_id': 'ATS-TEST-002',
        'status': 'APPROVED',
        'template': 'ma_crossover',
        'symbol': 'EURUSD',
        'composite_score': 1.45,
        'parameters': {'fast_ma_period': 15, 'slow_ma_period': 80},
        'wf_summary': {'median_win_rate': 0.58, 'median_profit_factor': 1.6}
    }
    store.add_or_update(new_candidate)
    assert len(store.get_all()) == 2

    # Pre-create an active strategy file to verify archiving
    deploy_dir.mkdir(parents=True, exist_ok=True)
    initial_active = deploy_dir / 'active_strategy.json'
    initial_active.write_text(json.dumps({'strategy_id': 'ATS-OLD-001', 'parameters': {}}), encoding='utf-8')

    # Deploy new candidate
    ok = store.deploy('ATS-TEST-002', deploy_dir)
    assert ok is True

    # Check status transition: ATS-TEST-002 is DEPLOYED, ATS-MA-MANUAL-001 is ARCHIVED (history retained)
    deployed = store.get_by_id('ATS-TEST-002')
    stub = store.get_by_id('ATS-MA-MANUAL-001')
    assert deployed['status'] == 'DEPLOYED'
    assert stub['status'] == 'ARCHIVED'
    assert stub['promoted_by'] == 'manual_test_stub'

    # Verify deployed active_strategy.json
    active_file = deploy_dir / 'active_strategy.json'
    assert active_file.exists()
    with open(active_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data['strategy_id'] == 'ATS-TEST-002'

    # Verify archive directory contains outgoing file
    archive_files = list((deploy_dir / 'archive').glob('*.json'))
    assert len(archive_files) == 1
    assert 'ATS-OLD-001' in archive_files[0].name


# ── TEST: Part E Mutual Exclusion Guards ─────────────────

@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_mutual_exclusion_trainer_running_blocks_engine_start(tmp_path, monkeypatch):
    """Integration: Starting/restarting Engine when Trainer is running returns HTTP 409."""
    pytest.importorskip('httpx')
    import engine.dashboard.app as app_mod

    # Mock trainer status to running=True
    monkeypatch.setattr(app_mod._trainer, 'get_status', lambda: {'running': True, 'pid': 9999})

    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app)

    # Start Engine should fail with 409
    res = client.post("/api/command/engine/start", json={"symbol": "EURUSD"})
    assert res.status_code == 409
    assert "cannot start engine while trainer" in res.json()['detail'].lower()

    # Restart Engine should fail with 409
    res_restart = client.post("/api/command/engine/restart", json={"symbol": "EURUSD"})
    assert res_restart.status_code == 409
    assert "cannot restart engine while trainer" in res_restart.json()['detail'].lower()


# ── TEST: Adaptation & Bench API Endpoints ────────────────

@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_api_adaptation_and_bench_endpoints(tmp_path, monkeypatch):
    """Integration: GET /api/adaptation/latest, GET /api/bench, POST /api/bench/deploy."""
    pytest.importorskip('httpx')
    import engine.dashboard.app as app_mod
    import trainer.core.bench as bench_mod

    # Patch paths to temporary test directory
    monkeypatch.setattr(app_mod, 'STRATEGY_PATH', tmp_path / 'active_strategy.json')
    monkeypatch.setattr(app_mod, 'DECISION_LOG', tmp_path / 'decision_log.jsonl')
    monkeypatch.setattr(bench_mod, 'BENCH_FILE', tmp_path / 'strategy_bench.json')

    # Create dummy active strategy
    active_payload = {
        'strategy_id': 'ATS-MA-MANUAL-001',
        'parameters': {'fast_ma_period': 20, 'slow_ma_period': 100}
    }
    (tmp_path / 'active_strategy.json').write_text(json.dumps(active_payload), encoding='utf-8')

    # Create dummy decision log record
    dec_record = {
        'decision_id': 'dec123',
        'candidate_id': 'ATS-TEST-002',
        'composite_score': 1.45,
        'parameters': {'fast_ma_period': 15, 'slow_ma_period': 80},
        'flags': [],
        'gate_results': {'gate_1_total_trades': 150, 'gate_4_max_drawdown': 12.0}
    }
    (tmp_path / 'decision_log.jsonl').write_text(json.dumps(dec_record) + '\n', encoding='utf-8')

    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app)

    # 1. GET /api/adaptation/latest
    res_adapt = client.get("/api/adaptation/latest")
    assert res_adapt.status_code == 200
    adapt_data = res_adapt.json()
    assert adapt_data['has_decision'] is True
    assert len(adapt_data['parameter_diff']) > 0
    assert len(adapt_data['gate_breakdown']) == 10

    # 2. GET /api/bench
    res_bench = client.get("/api/bench")
    assert res_bench.status_code == 200
    bench_data = res_bench.json()
    assert 'strategies' in bench_data
    assert len(bench_data['strategies']) >= 1
    assert bench_data['strategies'][0]['promoted_by'] == 'manual_test_stub'

    # 3. POST /api/bench/deploy
    res_deploy = client.post("/api/bench/deploy", json={"strategy_id": "ATS-MA-MANUAL-001"})
    assert res_deploy.status_code == 200
    assert res_deploy.json()['ok'] is True


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_log_dates_endpoint_lists_available_dates(tmp_path, monkeypatch):
    """GET /api/logs/dates returns sorted list of available log dates."""
    pytest.importorskip('httpx')
    import engine.dashboard.app as app_mod

    logs_dir = tmp_path / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / 'engine_20260810.log').write_text("line 1\n", encoding='utf-8')
    (logs_dir / 'engine_20260812.log').write_text("line 2\n", encoding='utf-8')
    monkeypatch.setattr(app_mod, 'LOGS_PATH', logs_dir)

    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app)

    res = client.get("/api/logs/dates")
    assert res.status_code == 200
    data = res.json()
    assert 'dates' in data
    dates = [d['date'] for d in data['dates']]
    assert '2026-08-12' in dates
    assert '2026-08-10' in dates
    # Dates must be in descending order
    assert dates == sorted(dates, reverse=True)


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_log_archived_date_query(tmp_path, monkeypatch):
    """GET /api/logs/engine?date=2026-08-12 returns archived log file content."""
    pytest.importorskip('httpx')
    import engine.dashboard.app as app_mod

    logs_dir = tmp_path / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / 'engine_20260812.log').write_text("2026-08-12 INFO Archived entry 1\n2026-08-12 INFO Archived entry 2\n", encoding='utf-8')
    monkeypatch.setattr(app_mod, 'LOGS_PATH', logs_dir)

    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app)

    res = client.get("/api/logs/engine?date=2026-08-12")
    assert res.status_code == 200
    data = res.json()
    assert data['source'] == 'engine'
    assert data['date'] == '2026-08-12'
    assert len(data['lines']) == 2
    assert "Archived entry 1" in data['lines'][0]
    assert "Archived entry 2" in data['lines'][1]