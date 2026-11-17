"""
Unit and integration tests for WAVE 6 — Prop-Firm Compliance & Mandate Safety.
Run: pytest tests/test_mandate.py -v
Pure logic — no MT5 required.
"""

import sys
import json
import pytest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import engine.core.mandate as mandate_mod
from engine.core.mandate import Mandate, load_mandate, log_audit_event
import engine.core.engine as engine_mod


# ── FIX 6.1: MANDATE SCHEMA & CONFIGURATION TESTS ───────

class TestMandateConfiguration:

    def test_mandate_load_valid_config(self, tmp_path):
        config_file = tmp_path / 'mandate.json'
        data = {
            'mandate_id': 'TEST-100K',
            'firm_name': 'FTMO',
            'account_id': 'paper_1',
            'max_daily_loss_pct': 4.0,
            'max_overall_drawdown_pct': 8.0,
            'drawdown_type': 'trailing',
            'min_trading_days': 5,
            'max_daily_risk_pct': 1.0,
            'symbol_universe': ['EURUSD', 'GBPUSD'],
            'prohibited_patterns': []
        }
        config_file.write_text(json.dumps(data))

        mandate = load_mandate(config_file)
        assert mandate.mandate_id == 'TEST-100K'
        assert mandate.max_daily_loss_pct == 4.0
        assert mandate.max_overall_drawdown_pct == 8.0
        assert mandate.drawdown_type == 'trailing'
        assert mandate.symbol_universe == ['EURUSD', 'GBPUSD']

    def test_mandate_missing_file_fails_loudly(self, tmp_path):
        missing_file = tmp_path / 'missing_mandate.json'
        with pytest.raises(RuntimeError, match="does not exist"):
            load_mandate(missing_file)

    def test_mandate_corrupt_json_fails_loudly(self, tmp_path):
        corrupt_file = tmp_path / 'mandate.json'
        corrupt_file.write_text("invalid json {{{")
        with pytest.raises(RuntimeError, match="corrupted or unparseable"):
            load_mandate(corrupt_file)

    def test_mandate_missing_required_field_fails_loudly(self, tmp_path):
        incomplete_file = tmp_path / 'mandate.json'
        data = {'mandate_id': 'TEST-100K'} # Missing other fields
        incomplete_file.write_text(json.dumps(data))
        with pytest.raises(RuntimeError, match="missing required fields"):
            load_mandate(incomplete_file)

    def test_mandate_invalid_drawdown_type_fails_loudly(self, tmp_path):
        invalid_file = tmp_path / 'mandate.json'
        data = {
            'mandate_id': 'TEST-100K',
            'firm_name': 'FTMO',
            'account_id': 'paper_1',
            'max_daily_loss_pct': 4.0,
            'max_overall_drawdown_pct': 8.0,
            'drawdown_type': 'invalid_type',
            'min_trading_days': 5,
            'max_daily_risk_pct': 1.0,
            'symbol_universe': ['EURUSD'],
            'prohibited_patterns': []
        }
        invalid_file.write_text(json.dumps(data))
        with pytest.raises(RuntimeError, match="drawdown_type .* invalid"):
            load_mandate(invalid_file)


# ── FIX 6.2: PRE-TRADE MANDATE GATE TESTS ───────────────

class TestPreTradeMandateGate:

    def test_pre_trade_gate_symbol_not_in_universe_blocks_entry(self, tmp_path, monkeypatch):
        mandate = Mandate(
            mandate_id='TEST-100K', firm_name='FTMO', account_id='1',
            max_daily_loss_pct=4.0, max_overall_drawdown_pct=8.0, drawdown_type='trailing',
            min_trading_days=5, max_daily_risk_pct=1.0, symbol_universe=['EURUSD'], prohibited_patterns=[]
        )

        audit_log = tmp_path / 'audit_ledger.jsonl'
        monkeypatch.setattr(mandate_mod, 'AUDIT_LOG_FILE', audit_log)
        monkeypatch.setattr(engine_mod, 'load_mandate', lambda *args: mandate)

        engine = engine_mod.Engine(symbol='GBPUSD', paper_mode=True) # GBPUSD not in ['EURUSD']
        engine.mandate = mandate

        dates = pd.date_range('2026-01-01 12:15', periods=5, freq='15min', tz='UTC')
        df_mock = pd.DataFrame({
            'open': [1.1000]*5, 'high': [1.1010]*5, 'low': [1.0990]*5, 'close': [1.1000]*5, 'volume': [100.0]*5
        }, index=dates)

        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df_mock)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 10000.0, 'balance': 10000.0})
        monkeypatch.setattr(engine_mod, 'get_open_positions', lambda sym: [])

        mock_loader = MagicMock()
        mock_spec = MagicMock()
        mock_spec.strategy_id = 'ATS-001'
        mock_spec.risk_per_trade_pct = 1.0
        mock_spec.pip_size = 0.0001
        mock_spec.parameters = {}
        mock_spec.parameters = {}
        mock_loader.get_spec.return_value = mock_spec
        mock_loader.reload_if_changed.return_value = False
        engine.strategy_loader = mock_loader

        mock_signal_engine = MagicMock()
        mock_signal_engine.process_candle.return_value = {'signal': 'BUY', 'reason': 'MA Crossover', 'sl_pips': 20.0, 'sl_price': 1.0980, 'tp_price': 1.1040}
        mock_signal_engine.get_cache.return_value = {'session': 'LONDON', 'regime': 'TRENDING', 'close': 1.1000}
        engine.signal_engine = mock_signal_engine

        mock_degradation = MagicMock()
        mock_degradation.get_summary.return_value = {'status': 'OK'}
        engine.degradation = mock_degradation
        engine.live_logger = MagicMock()

        # Execute real _process_candle method
        engine._process_candle()

        # Assert entry was blocked
        assert engine.orders_placed == 0

        # Assert audit log recorded PRE_TRADE_BLOCKED
        assert audit_log.exists()
        lines = audit_log.read_text().strip().split('\n')
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record['event_type'] == 'PRE_TRADE_BLOCKED'
        assert 'not in mandate symbol universe' in record['details']['reason']

    def test_pre_trade_gate_risk_exceeds_max_blocks_entry(self, tmp_path, monkeypatch):
        mandate = Mandate(
            mandate_id='TEST-100K', firm_name='FTMO', account_id='1',
            max_daily_loss_pct=4.0, max_overall_drawdown_pct=8.0, drawdown_type='trailing',
            min_trading_days=5, max_daily_risk_pct=1.0, symbol_universe=['EURUSD'], prohibited_patterns=[]
        )

        audit_log = tmp_path / 'audit_ledger.jsonl'
        monkeypatch.setattr(mandate_mod, 'AUDIT_LOG_FILE', audit_log)
        monkeypatch.setattr(engine_mod, 'load_mandate', lambda *args: mandate)

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        engine.mandate = mandate

        dates = pd.date_range('2026-01-01 12:15', periods=5, freq='15min', tz='UTC')
        df_mock = pd.DataFrame({
            'open': [1.1000]*5, 'high': [1.1010]*5, 'low': [1.0990]*5, 'close': [1.1000]*5, 'volume': [100.0]*5
        }, index=dates)

        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df_mock)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 10000.0, 'balance': 10000.0})
        monkeypatch.setattr(engine_mod, 'get_open_positions', lambda sym: [])

        mock_loader = MagicMock()
        mock_spec = MagicMock()
        mock_spec.strategy_id = 'ATS-001'
        mock_spec.risk_per_trade_pct = 2.5 # Exceeds mandate max risk 1.0%
        mock_spec.pip_size = 0.0001
        mock_spec.parameters = {}
        mock_loader.get_spec.return_value = mock_spec
        mock_loader.reload_if_changed.return_value = False
        engine.strategy_loader = mock_loader

        mock_signal_engine = MagicMock()
        mock_signal_engine.process_candle.return_value = {'signal': 'BUY', 'reason': 'MA Crossover', 'sl_pips': 20.0, 'sl_price': 1.0980, 'tp_price': 1.1040}
        mock_signal_engine.get_cache.return_value = {'session': 'LONDON', 'regime': 'TRENDING', 'close': 1.1000}
        engine.signal_engine = mock_signal_engine

        mock_degradation = MagicMock()
        mock_degradation.get_summary.return_value = {'status': 'OK'}
        engine.degradation = mock_degradation
        engine.live_logger = MagicMock()

        engine._process_candle()

        assert engine.orders_placed == 0
        assert audit_log.exists()
        record = json.loads(audit_log.read_text().strip().split('\n')[0])
        assert record['event_type'] == 'PRE_TRADE_BLOCKED'
        assert 'exceeds mandate max risk' in record['details']['reason']


# ── FIX 6.3: VIOLATION DETECTION & KILL SWITCH TESTS ───

class TestMandateViolationAndKillSwitch:

    def test_overall_dd_breach_flattens_positions(self, tmp_path, monkeypatch):
        mandate = Mandate(
            mandate_id='TEST-100K', firm_name='FTMO', account_id='1',
            max_daily_loss_pct=4.0, max_overall_drawdown_pct=8.0, drawdown_type='trailing',
            min_trading_days=5, max_daily_risk_pct=1.0, symbol_universe=['EURUSD'], prohibited_patterns=[]
        )

        audit_log = tmp_path / 'audit_ledger.jsonl'
        monkeypatch.setattr(mandate_mod, 'AUDIT_LOG_FILE', audit_log)
        monkeypatch.setattr(engine_mod, 'load_mandate', lambda *args: mandate)

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        engine.mandate = mandate
        engine.peak_equity = 10000.0
        engine.daily_start_equity = 10000.0

        class MockPos:
            ticket = 777
            symbol = 'EURUSD'
            type = 0
            volume = 0.1
            profit = -900.0

        closed_positions = []
        monkeypatch.setattr(engine_mod, 'close_position', lambda pos, paper: closed_positions.append(pos) or True)
        monkeypatch.setattr(engine_mod, 'get_open_positions', lambda sym: [MockPos()])
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 9100.0, 'balance': 9100.0}) # 9% DD > 8% limit

        dates = pd.date_range('2026-01-01 12:15', periods=5, freq='15min', tz='UTC')
        df_mock = pd.DataFrame({
            'open': [1.1000]*5, 'high': [1.1010]*5, 'low': [1.0990]*5, 'close': [1.1000]*5, 'volume': [100.0]*5
        }, index=dates)
        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df_mock)

        mock_loader = MagicMock()
        mock_spec = MagicMock()
        mock_spec.strategy_id = 'ATS-001'
        mock_spec.risk_per_trade_pct = 1.0
        mock_spec.parameters = {}
        mock_spec.parameters = {}
        mock_loader.get_spec.return_value = mock_spec
        mock_loader.reload_if_changed.return_value = False
        engine.strategy_loader = mock_loader

        mock_signal_engine = MagicMock()
        mock_signal_engine.process_candle.return_value = {'signal': 'NONE', 'reason': 'None'}
        mock_signal_engine.get_cache.return_value = {'session': 'LONDON', 'regime': 'TRENDING', 'close': 1.1000}
        engine.signal_engine = mock_signal_engine
        mock_degradation = MagicMock()
        mock_degradation.get_summary.return_value = {'status': 'OK'}
        engine.degradation = mock_degradation
        engine.live_logger = MagicMock()

        # Run real _process_candle method
        engine._process_candle()

        # Assert position was FLATTENED (close_position was called)
        assert len(closed_positions) == 1
        assert closed_positions[0].ticket == 777
        assert engine.circuit_breaker_tripped is True

        # Confirm audit log written
        lines = audit_log.read_text().strip().split('\n')
        events = [json.loads(line)['event_type'] for line in lines]
        assert 'MANDATE_OVERALL_DD_BREACH' in events
        assert 'POSITIONS_FLATTENED' in events

    def test_kill_switch_file_blocks_entries(self, tmp_path, monkeypatch):
        mandate = Mandate(
            mandate_id='TEST-100K', firm_name='FTMO', account_id='1',
            max_daily_loss_pct=4.0, max_overall_drawdown_pct=8.0, drawdown_type='trailing',
            min_trading_days=5, max_daily_risk_pct=1.0, symbol_universe=['EURUSD'], prohibited_patterns=[]
        )

        audit_log = tmp_path / 'audit_ledger.jsonl'
        kill_file = tmp_path / 'KILL_SWITCH'
        kill_file.write_text("HALT")

        monkeypatch.setattr(mandate_mod, 'AUDIT_LOG_FILE', audit_log)
        monkeypatch.setattr(engine_mod, 'KILL_SWITCH_FILE', kill_file)
        monkeypatch.setattr(engine_mod, 'load_mandate', lambda *args: mandate)

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        engine.mandate = mandate

        dates = pd.date_range('2026-01-01 12:15', periods=5, freq='15min', tz='UTC')
        df_mock = pd.DataFrame({
            'open': [1.1000]*5, 'high': [1.1010]*5, 'low': [1.0990]*5, 'close': [1.1000]*5, 'volume': [100.0]*5
        }, index=dates)

        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df_mock)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 10000.0, 'balance': 10000.0})
        monkeypatch.setattr(engine_mod, 'get_open_positions', lambda sym: [])

        mock_loader = MagicMock()
        mock_spec = MagicMock()
        mock_spec.strategy_id = 'ATS-001'
        mock_spec.risk_per_trade_pct = 1.0
        mock_spec.parameters = {}
        mock_loader.get_spec.return_value = mock_spec
        mock_loader.reload_if_changed.return_value = False
        engine.strategy_loader = mock_loader

        mock_signal_engine = MagicMock()
        mock_signal_engine.process_candle.return_value = {'signal': 'BUY', 'reason': 'MA Crossover', 'sl_pips': 20.0, 'sl_price': 1.0980, 'tp_price': 1.1040}
        mock_signal_engine.get_cache.return_value = {'session': 'LONDON', 'regime': 'TRENDING', 'close': 1.1000}
        engine.signal_engine = mock_signal_engine
        mock_degradation = MagicMock()
        mock_degradation.get_summary.return_value = {'status': 'OK'}
        engine.degradation = mock_degradation
        engine.live_logger = MagicMock()

        engine._process_candle()

        assert engine.circuit_breaker_tripped is True
        assert engine.orders_placed == 0

        lines = audit_log.read_text().strip().split('\n')
        events = [json.loads(line)['event_type'] for line in lines]
        assert 'KILL_SWITCH_ENGAGED' in events

    def test_kill_switch_on_startup_blocks_trading(self, tmp_path, monkeypatch):
        mandate = Mandate(
            mandate_id='TEST-100K', firm_name='FTMO', account_id='1',
            max_daily_loss_pct=4.0, max_overall_drawdown_pct=8.0, drawdown_type='trailing',
            min_trading_days=5, max_daily_risk_pct=1.0, symbol_universe=['EURUSD'], prohibited_patterns=[]
        )

        audit_log = tmp_path / 'audit_ledger.jsonl'
        kill_file = tmp_path / 'KILL_SWITCH'
        flag_file = tmp_path / 'shutdown_flag.json'
        kill_file.write_text("HALT")
        flag_file.write_text(json.dumps({'clean_shutdown': True}))

        monkeypatch.setattr(mandate_mod, 'AUDIT_LOG_FILE', audit_log)
        monkeypatch.setattr(engine_mod, 'KILL_SWITCH_FILE', kill_file)
        monkeypatch.setattr(engine_mod, 'SHUTDOWN_FLAG_FILE', flag_file)
        monkeypatch.setattr(engine_mod, 'load_mandate', lambda *args: mandate)

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        main_loop_called = False
        def mock_main_loop():
            nonlocal main_loop_called
            main_loop_called = True

        monkeypatch.setattr(engine, '_main_loop', mock_main_loop)

        engine.start()

        assert engine.running is False
        assert main_loop_called is False
        assert engine.circuit_breaker_tripped is True
        lines = audit_log.read_text().strip().split('\n')
        events = [json.loads(line)['event_type'] for line in lines]
        assert 'KILL_SWITCH_ENGAGED' in events

    def test_mandate_load_lowercase_symbols_normalized(self, tmp_path):
        config_file = tmp_path / 'mandate.json'
        data = {
            'mandate_id': 'TEST-100K',
            'firm_name': 'FTMO',
            'account_id': 'paper_1',
            'max_daily_loss_pct': 4.0,
            'max_overall_drawdown_pct': 8.0,
            'drawdown_type': 'trailing',
            'min_trading_days': 5,
            'max_daily_risk_pct': 1.0,
            'symbol_universe': ['eurusd', 'gbpusd '], # lowercase with space
            'prohibited_patterns': []
        }
        config_file.write_text(json.dumps(data))

        mandate = load_mandate(config_file)
        assert mandate.symbol_universe == ['EURUSD', 'GBPUSD']

    def test_mandate_and_wave5_daily_equity_independence(self):
        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        engine.daily_start_equity = 10000.0
        engine.mandate_daily_start_equity = 10000.0

        # Corrupt / reset WAVE 5 daily start equity
        engine.daily_start_equity = 5000.0

        # Confirm mandate_daily_start_equity remains independent and uncorrupted
        assert engine.mandate_daily_start_equity == 10000.0

    def test_runtime_kill_switch_file_exits_main_loop(self, tmp_path, monkeypatch):
        mandate = Mandate(
            mandate_id='TEST-100K', firm_name='FTMO', account_id='1',
            max_daily_loss_pct=4.0, max_overall_drawdown_pct=8.0, drawdown_type='trailing',
            min_trading_days=5, max_daily_risk_pct=1.0, symbol_universe=['EURUSD'], prohibited_patterns=[]
        )

        audit_log = tmp_path / 'audit_ledger.jsonl'
        kill_file = tmp_path / 'KILL_SWITCH'
        flag_file = tmp_path / 'shutdown_flag.json'
        flag_file.write_text(json.dumps({'clean_shutdown': True}))

        monkeypatch.setattr(mandate_mod, 'AUDIT_LOG_FILE', audit_log)
        monkeypatch.setattr(engine_mod, 'KILL_SWITCH_FILE', kill_file)
        monkeypatch.setattr(engine_mod, 'SHUTDOWN_FLAG_FILE', flag_file)
        monkeypatch.setattr(engine_mod, 'load_mandate', lambda *args: mandate)

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        engine.mandate = mandate
        engine.running = True

        dates = pd.date_range('2026-01-01 12:15', periods=5, freq='15min', tz='UTC')
        df_mock = pd.DataFrame({
            'open': [1.1000]*5, 'high': [1.1010]*5, 'low': [1.0990]*5, 'close': [1.1000]*5, 'volume': [100.0]*5
        }, index=dates)

        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df_mock)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 10000.0, 'balance': 10000.0})
        monkeypatch.setattr(engine_mod, 'get_open_positions', lambda sym: [])

        wait_called = 0
        def mock_wait(sym, tf):
            nonlocal wait_called
            wait_called += 1
            # Create kill switch sentinel file during first wait cycle
            kill_file.write_text("HALT")

        monkeypatch.setattr(engine_mod, 'wait_for_next_candle', mock_wait)

        mock_loader = MagicMock()
        mock_spec = MagicMock()
        mock_spec.strategy_id = 'ATS-001'
        mock_spec.risk_per_trade_pct = 1.0
        mock_spec.parameters = {}
        mock_loader.get_spec.return_value = mock_spec
        mock_loader.reload_if_changed.return_value = False
        engine.strategy_loader = mock_loader

        mock_signal_engine = MagicMock()
        mock_signal_engine.process_candle.return_value = {'signal': 'NONE', 'reason': 'None'}
        mock_signal_engine.get_cache.return_value = {'session': 'LONDON', 'regime': 'TRENDING', 'close': 1.1000}
        engine.signal_engine = mock_signal_engine
        mock_degradation = MagicMock()
        mock_degradation.get_summary.return_value = {'status': 'OK'}
        engine.degradation = mock_degradation
        engine.live_logger = MagicMock()

        # Run real _main_loop()
        engine._main_loop()

        # Assert main loop exited when kill switch appeared
        assert engine.running is False
        assert wait_called == 1
        assert engine.circuit_breaker_tripped is True
        lines = audit_log.read_text().strip().split('\n')
        events = [json.loads(line)['event_type'] for line in lines]
        assert 'KILL_SWITCH_ENGAGED' in events


# ── FIX 6.4: AUDIT LEDGER TESTS ────────────────────────

class TestAuditLedger:

    def test_audit_ledger_append_only(self, tmp_path):
        audit_file = tmp_path / 'audit_ledger.jsonl'

        log_audit_event('EVENT_ONE', 'MANDATE-1', {'detail': 'first'}, audit_file)
        log_audit_event('EVENT_TWO', 'MANDATE-1', {'detail': 'second'}, audit_file)

        assert audit_file.exists()
        lines = audit_file.read_text().strip().split('\n')
        assert len(lines) == 2

        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])

        assert r1['event_type'] == 'EVENT_ONE'
        assert r1['details']['detail'] == 'first'
        assert r2['event_type'] == 'EVENT_TWO'
        assert r2['details']['detail'] == 'second'


# ── WAVE 45: ENGINE PARITY TESTS ────────────────────────

class TestWave45EngineParity:

    def _build_engine_for_candle_test(self, tmp_path, monkeypatch, drawdown_type='static'):
        from engine.core.mandate import Mandate
        mandate = Mandate(
            mandate_id='WAVE45-MANDATE',
            firm_name='FTMO_TEST',
            account_id='ACC-001',
            max_daily_loss_pct=4.0,
            max_overall_drawdown_pct=8.0,
            drawdown_type=drawdown_type,
            min_trading_days=5,
            max_daily_risk_pct=1.0,
            symbol_universe=['EURUSD', 'XAUUSD'],
        )
        audit_log = tmp_path / 'audit_ledger.jsonl'
        kill_file = tmp_path / 'KILL_SWITCH'
        snapshot_file = tmp_path / 'engine_snapshot.json'

        monkeypatch.setattr(mandate_mod, 'AUDIT_LOG_FILE', audit_log)
        monkeypatch.setattr(engine_mod, 'KILL_SWITCH_FILE', kill_file)
        monkeypatch.setattr(engine_mod, 'SNAPSHOT_FILE', snapshot_file)
        monkeypatch.setattr(engine_mod, 'write_state', lambda *a, **kw: None)

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True, magic=123456)
        engine.mandate = mandate
        engine.running = True

        mock_loader = MagicMock()
        mock_spec = MagicMock()
        mock_spec.strategy_id = 'ATS-W45'
        mock_spec.template = 'ma_crossover'
        mock_spec.timeframe = 'M15'
        mock_spec.risk_per_trade_pct = 1.0
        mock_spec.parameters = {}
        mock_spec.filters = {}
        mock_loader.get_spec.return_value = mock_spec
        mock_loader.reload_if_changed.return_value = False
        engine.strategy_loader = mock_loader

        mock_signal_engine = MagicMock()
        mock_signal_engine.process_candle.return_value = {'signal': 'HOLD', 'reason': 'No signal'}
        mock_signal_engine.get_cache.return_value = {'session': 'LONDON', 'regime': 'TRENDING', 'close': 1.1000}
        engine.signal_engine = mock_signal_engine
        mock_degradation = MagicMock()
        mock_degradation.get_summary.return_value = {'status': 'OK'}
        engine.degradation = mock_degradation
        engine.live_logger = MagicMock()
        return engine

    def test_foreign_magic_positions_are_ignored(self, monkeypatch):
        from types import SimpleNamespace

        own_pos = SimpleNamespace(ticket=101, magic=123456, type=0, volume=0.1, profit=5.0)
        manual_pos = SimpleNamespace(ticket=102, magic=0, type=0, volume=0.5, profit=-200.0)
        other_ea_pos = SimpleNamespace(ticket=103, magic=999888, type=1, volume=0.2, profit=15.0)

        monkeypatch.setattr(
            engine_mod,
            'get_open_positions',
            lambda sym, magic=None: [
                p for p in [own_pos, manual_pos, other_ea_pos]
                if engine_mod._position_matches_magic(p, magic)
            ],
        )

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True, magic=123456)
        own = engine._get_own_open_positions()
        assert [p.ticket for p in own] == [101]

        # Even if an external mock returns unfiltered positions from get_open_positions,
        # _get_own_open_positions still excludes foreign magic positions.
        monkeypatch.setattr(engine_mod, 'get_open_positions', lambda sym: [own_pos, manual_pos, other_ea_pos])
        own_strict = engine._get_own_open_positions()
        assert [p.ticket for p in own_strict] == [101]

    def test_static_drawdown_measured_from_initial_equity_not_peak(self, tmp_path, monkeypatch):
        engine = self._build_engine_for_candle_test(tmp_path, monkeypatch, drawdown_type='static')
        monkeypatch.setattr(engine_mod, 'get_open_positions', lambda sym, magic=None: [])

        base = pd.DataFrame({'open': [1.1]*5, 'high': [1.101]*5, 'low': [1.099]*5, 'close': [1.1]*5, 'volume': [10.0]*5})

        # Candle 1: initial_equity = 10000, peak_equity = 10000
        df1 = base.copy()
        df1.index = pd.date_range('2026-01-15 10:00', periods=5, freq='15min', tz='UTC')
        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df1)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 10000.0, 'balance': 10000.0})
        engine._process_candle()
        assert engine.initial_equity == 10000.0
        assert engine.circuit_breaker_tripped is False

        # Candle 2: equity grows to 12000 (peak_equity = 12000, initial_equity stays 10000)
        df2 = base.copy()
        df2.index = pd.date_range('2026-01-15 10:15', periods=5, freq='15min', tz='UTC')
        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df2)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 12000.0, 'balance': 12000.0})
        engine._process_candle()
        assert engine.peak_equity == 12000.0
        assert engine.initial_equity == 10000.0

        # Candle 3 (next Prague day so daily loss doesn't interfere): equity pulls back to 11000.
        # Trailing DD from 12000 peak is 8.33% (>= 8.0%), but static DD from 10000 initial is 0.0%.
        df3 = base.copy()
        df3.index = pd.date_range('2026-01-16 10:00', periods=5, freq='15min', tz='UTC')
        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df3)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 11000.0, 'balance': 11000.0})
        engine._process_candle()
        assert engine.circuit_breaker_tripped is False

        # Candle 4 (next Prague day): equity falls to 9190 -> 8.1% static DD from 10000 initial -> BREACH!
        df4 = base.copy()
        df4.index = pd.date_range('2026-01-17 10:00', periods=5, freq='15min', tz='UTC')
        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df4)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 9190.0, 'balance': 9190.0})
        engine._process_candle()
        assert engine.circuit_breaker_tripped is True

    def test_mandate_daily_start_equity_rolls_over_at_prague_midnight(self, tmp_path, monkeypatch):
        engine = self._build_engine_for_candle_test(tmp_path, monkeypatch, drawdown_type='static')
        monkeypatch.setattr(engine_mod, 'get_open_positions', lambda sym, magic=None: [])

        base = pd.DataFrame({'open': [1.1]*5, 'high': [1.101]*5, 'low': [1.099]*5, 'close': [1.1]*5, 'volume': [10.0]*5})

        # 2026-01-15 22:45 UTC = 2026-01-15 23:45 Europe/Prague (winter CET = UTC+1)
        df1 = base.copy()
        df1.index = pd.date_range('2026-01-15 21:45', periods=5, freq='15min', tz='UTC')
        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df1)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 10000.0, 'balance': 10000.0})
        engine._process_candle()
        assert engine.mandate_daily_start_equity == 10000.0

        # 2026-01-15 23:00 UTC = 2026-01-16 00:00 Europe/Prague (SAME UTC date, NEW Prague date) -> MUST reset
        df2 = base.copy()
        df2.index = pd.date_range('2026-01-15 22:00', periods=5, freq='15min', tz='UTC')
        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df2)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 10250.0, 'balance': 10250.0})
        engine._process_candle()
        assert engine.mandate_daily_start_equity == 10250.0

        # 2026-01-16 00:00 UTC = 2026-01-16 01:00 Europe/Prague (NEW UTC date, SAME Prague date) -> MUST NOT overwrite
        df3 = base.copy()
        df3.index = pd.date_range('2026-01-15 23:00', periods=5, freq='15min', tz='UTC')
        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df3)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 10400.0, 'balance': 10400.0})
        engine._process_candle()
        assert engine.mandate_daily_start_equity == 10250.0
        assert engine.daily_start_equity == 10400.0
