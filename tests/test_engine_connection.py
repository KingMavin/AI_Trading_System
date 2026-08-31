"""
Unit tests for WAVE 8.1 — Engine Reconnection, Heartbeat, and SAFE_MODE.
Run: pytest tests/test_engine_connection.py -v
Pure logic — no MT5 required.
"""

import sys
import json
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import engine.core.mandate as mandate_mod
from engine.core.mandate import Mandate
import engine.core.engine as engine_mod


class TestEngineReconnectionAndSafeMode:

    def test_reconnection_backoff_and_safe_mode(self, tmp_path, monkeypatch):
        audit_log = tmp_path / 'audit_ledger.jsonl'
        monkeypatch.setattr(mandate_mod, 'AUDIT_LOG_FILE', audit_log)
        monkeypatch.setattr(engine_mod, 'time', MagicMock()) # Mock sleep

        mandate = Mandate(
            mandate_id='TEST-100K', firm_name='FTMO', account_id='1',
            max_daily_loss_pct=4.0, max_overall_drawdown_pct=8.0, drawdown_type='trailing',
            min_trading_days=5, max_daily_risk_pct=1.0, symbol_universe=['EURUSD'], prohibited_patterns=[]
        )

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        engine.mandate = mandate

        # Mock terminal_info returning disconnected (None)
        monkeypatch.setattr(engine_mod, 'MT5_AVAILABLE', True)
        mock_mt5 = MagicMock()
        mock_mt5.terminal_info.return_value = None
        mock_mt5.initialize.return_value = False
        monkeypatch.setattr(engine_mod, 'mt5', mock_mt5)

        # Call _check_connection_health() 3 times
        assert engine._check_connection_health() is False # attempt 1
        assert engine.failed_reconnect_attempts == 1
        assert engine.safe_mode is False

        assert engine._check_connection_health() is False # attempt 2
        assert engine.failed_reconnect_attempts == 2
        assert engine.safe_mode is False

        assert engine._check_connection_health() is False # attempt 3 -> SAFE_MODE
        assert engine.failed_reconnect_attempts == 3
        assert engine.safe_mode is True
        assert engine.circuit_breaker_tripped is True

        # Verify audit log recorded SAFE_MODE_ENGAGED
        assert audit_log.exists()
        lines = audit_log.read_text().strip().split('\n')
        record = json.loads(lines[0])
        assert record['event_type'] == 'SAFE_MODE_ENGAGED'
        assert record['details']['failed_attempts'] == 3

    def test_safe_mode_blocks_entries_allows_exits(self, tmp_path, monkeypatch):
        mandate = Mandate(
            mandate_id='TEST-100K', firm_name='FTMO', account_id='1',
            max_daily_loss_pct=4.0, max_overall_drawdown_pct=8.0, drawdown_type='trailing',
            min_trading_days=5, max_daily_risk_pct=1.0, symbol_universe=['EURUSD'], prohibited_patterns=[]
        )

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        engine.mandate = mandate
        engine.safe_mode = True

        mock_spec = MagicMock()
        mock_spec.risk_per_trade_pct = 1.0

        # Assert Pre-Trade Gate BLOCKS entries when safe_mode is active
        allowed, reason = engine._check_mandate_pre_trade_gate({'signal': 'BUY'}, mock_spec)
        assert allowed is False
        assert 'SAFE_MODE' in reason

        # Assert Position Management runs and processes exits normally
        closed_positions = []
        monkeypatch.setattr(engine_mod, 'close_position', lambda pos, paper: closed_positions.append(pos) or True)

        class MockPos:
            ticket = 888
            type = 0 # BUY
            profit = 50.0

        mock_spec.parameters = {'exit_on_opposite_crossover': True}
        account = {'equity': 10000.0}
        signal = {'signal': 'SELL'} # Opposite signal

        monkeypatch.setattr(engine, '_reconcile_closed_positions', lambda pos: None)
        monkeypatch.setattr(engine, '_record_closed_trade', lambda pos, reason, acct: None)

        engine._manage_positions([MockPos()], signal, None, mock_spec, account)

        # Position management successfully executed close!
        assert len(closed_positions) == 1
        assert closed_positions[0].ticket == 888

    def test_stale_data_guard_triggers_reconnection(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine_mod, 'time', MagicMock())

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        engine.stale_data_threshold_minutes = 45

        # Set last_candle_time 50 minutes in the past
        past_time = datetime.now(timezone.utc) - timedelta(minutes=50)
        engine.last_candle_time = past_time

        # Mock MT5 terminal_info as connected
        monkeypatch.setattr(engine_mod, 'MT5_AVAILABLE', True)
        mock_mt5 = MagicMock()
        mock_term = MagicMock()
        mock_term.connected = True
        mock_mt5.terminal_info.return_value = mock_term
        mock_mt5.initialize.return_value = False
        monkeypatch.setattr(engine_mod, 'mt5', mock_mt5)

        # Stale data guard must trigger health failure despite connected=True
        healthy = engine._check_connection_health()
        assert healthy is False
        assert engine.failed_reconnect_attempts == 1
        assert mock_mt5.initialize.called is True


class TestOrderSubmissionRobustness:

    def test_order_submission_deviation_cap(self, monkeypatch):
        monkeypatch.setattr(engine_mod, 'MT5_AVAILABLE', True)
        mock_mt5 = MagicMock()
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_FILLING_FOK = 0

        mock_info = MagicMock()
        mock_info.filling_mode = 1
        mock_info.trade_stops_level = 0
        mock_info.point = 0.00001
        mock_mt5.symbol_info.return_value = mock_info

        mock_tick = MagicMock()
        mock_tick.ask = 1.08500
        mock_mt5.symbol_info_tick.return_value = mock_tick

        sent_requests = []
        def mock_order_send(req):
            sent_requests.append(req)
            res = MagicMock()
            res.retcode = 10009
            res.order = 999
            res.price = req['price']
            return res

        mock_mt5.order_send = mock_order_send
        monkeypatch.setattr(engine_mod, 'mt5', mock_mt5)

        signal = {'signal': 'BUY', 'sl_price': 1.08000, 'tp_price': 1.09000, 'close': 1.08500}

        # Test EURUSD (default deviation 10)
        order_eur = engine_mod.place_order('EURUSD', signal, lots=1.0, paper_mode=False)
        assert order_eur is not None
        assert sent_requests[0]['deviation'] == 10

        # Test XAUUSD (Gold deviation 50)
        order_gold = engine_mod.place_order('XAUUSD', signal, lots=1.0, paper_mode=False)
        assert order_gold is not None
        assert sent_requests[1]['deviation'] == 50

    def test_order_submission_filling_mode_negotiation(self, monkeypatch):
        monkeypatch.setattr(engine_mod, 'MT5_AVAILABLE', True)
        mock_mt5 = MagicMock()
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_FILLING_IOC = 1

        mock_tick = MagicMock()
        mock_tick.ask = 1.08500
        mock_mt5.symbol_info_tick.return_value = mock_tick
        monkeypatch.setattr(engine_mod, 'mt5', mock_mt5)

        signal = {'signal': 'BUY', 'sl_price': 1.08000, 'tp_price': 1.09000, 'close': 1.08500}

        # Case A: filling_mode = 2 (IOC)
        mock_info = MagicMock()
        mock_info.filling_mode = 2
        mock_info.trade_stops_level = 0
        mock_info.point = 0.00001
        mock_mt5.symbol_info.return_value = mock_info

        sent_requests = []
        def mock_order_send(req):
            sent_requests.append(req)
            res = MagicMock()
            res.retcode = 10009
            res.order = 1001
            res.price = req['price']
            return res
        mock_mt5.order_send = mock_order_send

        order = engine_mod.place_order('EURUSD', signal, lots=1.0, paper_mode=False)
        assert order is not None
        assert sent_requests[0]['type_filling'] == mock_mt5.ORDER_FILLING_IOC

        # Case B: Unsupported filling_mode = 16
        mock_info.filling_mode = 16
        order_fail = engine_mod.place_order('EURUSD', signal, lots=1.0, paper_mode=False)
        assert order_fail is None

    def test_order_submission_stops_level_validation(self, monkeypatch):
        monkeypatch.setattr(engine_mod, 'MT5_AVAILABLE', True)
        mock_mt5 = MagicMock()
        mock_mt5.ORDER_TYPE_BUY = 0

        mock_info = MagicMock()
        mock_info.filling_mode = 1
        mock_info.trade_stops_level = 50 # 50 points = 0.00050 min stop distance
        mock_info.point = 0.00001
        mock_mt5.symbol_info.return_value = mock_info

        mock_tick = MagicMock()
        mock_tick.ask = 1.08500
        mock_mt5.symbol_info_tick.return_value = mock_tick

        mock_mt5.order_send = MagicMock() # Should NEVER be called if stops level violated
        monkeypatch.setattr(engine_mod, 'mt5', mock_mt5)

        # Invalid SL: distance is 0.00020 (less than min stop distance 0.00050)
        invalid_signal = {'signal': 'BUY', 'sl_price': 1.08480, 'tp_price': 1.09000, 'close': 1.08500}
        order = engine_mod.place_order('EURUSD', invalid_signal, lots=1.0, paper_mode=False)

        assert order is None
        assert mock_mt5.order_send.called is False


class TestExecutionOrdering:

    def test_closes_run_before_entries_in_same_candle(self, tmp_path, monkeypatch):
        """
        Verify that position closes (SL/TP reconciliation or opposite crossover exit)
        execute and update open position state BEFORE new entry logic evaluates.
        """
        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)

        mock_spec = MagicMock()
        mock_spec.strategy_id = 'MA-CROSS'
        mock_spec.parameters = {'exit_on_opposite_crossover': True}
        mock_spec.fast_ma = 10
        mock_spec.slow_ma = 20
        mock_spec.sl_atr_multiple = 1.5
        mock_spec.tp_rr_ratio = 2.0
        mock_spec.risk_per_trade_pct = 1.0
        mock_spec.pip_size = 0.0001

        engine.strategy_loader = MagicMock()
        engine.strategy_loader.get_spec.return_value = mock_spec
        engine.strategy_loader.reload_if_changed.return_value = False
        mock_deg = MagicMock()
        mock_deg.get_summary.return_value = {'status': 'OK'}
        engine.degradation = mock_deg

        monkeypatch.setattr(engine_mod, 'write_state', lambda s: None)
        monkeypatch.setattr(engine, 'save_snapshot', lambda: None)

        engine.signal_engine = MagicMock()
        engine.signal_engine.process_candle.return_value = {
            'signal': 'SELL', 'sl_price': 1.08800, 'tp_price': 1.08200, 'sl_pips': 30, 'reason': 'Cross down'
        }
        engine.signal_engine.get_cache.return_value = {'session': 'LONDON', 'regime': 'TRENDING', 'close': 1.08500}

        # Mock candles fetch
        import pandas as pd
        mock_df = pd.DataFrame(
            {'open': [1.08], 'high': [1.09], 'low': [1.07], 'close': [1.085], 'volume': [100]},
            index=[pd.Timestamp('2026-07-29 10:00:00', tz='UTC')]
        )
        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda s, tf, c: mock_df)
        monkeypatch.setattr(engine, '_check_connection_health', lambda: True)

        mock_mandate = MagicMock()
        mock_mandate.symbol_universe = ['EURUSD']
        mock_mandate.max_daily_risk_pct = 2.0
        mock_mandate.max_daily_loss_pct = 4.0
        mock_mandate.max_overall_drawdown_pct = 8.0
        engine.mandate = mock_mandate

        class MockPos:
            ticket = 555
            type = 0 # BUY (opposite of SELL signal)
            profit = -10.0
            volume = 1.0

        current_positions = [MockPos()]
        execution_sequence = []

        def mock_get_open_positions(symbol):
            return list(current_positions)

        def mock_close_position(pos, paper_mode):
            execution_sequence.append(('CLOSE', pos.ticket))
            current_positions.clear() # Position closed!
            return True

        def mock_enter_trade(signal, spec, balance):
            execution_sequence.append(('ENTER', signal['signal']))

        monkeypatch.setattr(engine_mod, 'get_open_positions', mock_get_open_positions)
        monkeypatch.setattr(engine_mod, 'close_position', mock_close_position)
        monkeypatch.setattr(engine, '_enter_trade', mock_enter_trade)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 10000.0, 'balance': 10000.0})

        # Process candle
        engine._process_candle()

        # Sequence MUST be: CLOSE first, then ENTER second!
        assert len(execution_sequence) == 2
        assert execution_sequence[0] == ('CLOSE', 555)
        assert execution_sequence[1] == ('ENTER', 'SELL')

    def test_candle_gap_processing_only_evaluates_latest_candle(self, monkeypatch):
        """
        Verify Fix 8.4: If a multi-hour candle gap occurs (e.g. laptop sleep or outage),
        _process_candle evaluates ONLY the latest candle close and does NOT iterate or replay backlog.
        """
        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)

        mock_spec = MagicMock()
        mock_spec.strategy_id = 'MA-CROSS'
        mock_spec.parameters = {}
        mock_spec.risk_per_trade_pct = 1.0

        engine.strategy_loader = MagicMock()
        engine.strategy_loader.get_spec.return_value = mock_spec
        engine.strategy_loader.reload_if_changed.return_value = False

        mock_deg = MagicMock()
        mock_deg.get_summary.return_value = {'status': 'OK'}
        engine.degradation = mock_deg

        monkeypatch.setattr(engine_mod, 'write_state', lambda s: None)
        monkeypatch.setattr(engine, 'save_snapshot', lambda: None)
        monkeypatch.setattr(engine, '_check_connection_health', lambda: True)

        processed_signals = []
        def mock_process_candle(df, **kwargs):
            latest_time = df.index[-1]
            processed_signals.append(latest_time)
            return {'signal': 'NONE', 'reason': 'No cross'}

        engine.signal_engine = MagicMock()
        engine.signal_engine.process_candle = mock_process_candle
        engine.signal_engine.get_cache.return_value = {'session': 'NEW_YORK', 'regime': 'RANGE', 'close': 1.08500}

        # Mock candles with a 4-hour gap between 10:00 and 14:00
        import pandas as pd
        t1 = pd.Timestamp('2026-07-29 10:00:00', tz='UTC')
        t2 = pd.Timestamp('2026-07-29 14:00:00', tz='UTC') # 4-hour gap!
        gap_df = pd.DataFrame(
            {'open': [1.0800, 1.0850], 'high': [1.0860, 1.0890], 'low': [1.0790, 1.0840], 'close': [1.0850, 1.0880], 'volume': [100, 200]},
            index=[t1, t2]
        )
        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda s, tf, c: gap_df)
        monkeypatch.setattr(engine_mod, 'get_open_positions', lambda s: [])
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 10000.0, 'balance': 10000.0})

        # Process candle
        engine._process_candle()

        # Signal Engine MUST be evaluated exactly ONCE for the latest candle timestamp t2
        assert len(processed_signals) == 1
        assert processed_signals[0] == t2
        assert engine.last_candle_time == t2.to_pydatetime()

    # ── WAVE 16 PART B: MT5 ALGO TRADING TOGGLE TESTS ──────────────

    def test_connect_mt5_fails_when_algo_trading_disabled(self, monkeypatch):
        import engine.core.engine as engine_mod

        monkeypatch.setattr(engine_mod, 'MT5_AVAILABLE', True)
        mock_mt5 = MagicMock()
        mock_mt5.initialize.return_value = True

        class MockAccountInfo:
            login = 123456
            server = "TestServer"
            balance = 10000.0
            currency = "USD"

        class MockTerminalInfo:
            connected = True
            trade_allowed = False # ALGO TRADING BUTTON IS OFF!

        mock_mt5.account_info.return_value = MockAccountInfo()
        mock_mt5.terminal_info.return_value = MockTerminalInfo()
        monkeypatch.setattr(engine_mod, 'mt5', mock_mt5)

        # connect_mt5 MUST fail and return False when terminal_info().trade_allowed == False
        res = engine_mod.connect_mt5()
        assert res is False

    def test_check_connection_health_fails_when_algo_trading_disabled(self, monkeypatch):
        import engine.core.engine as engine_mod

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)

        monkeypatch.setattr(engine_mod, 'MT5_AVAILABLE', True)
        mock_mt5 = MagicMock()

        class MockTerminalInfo:
            connected = True
            trade_allowed = False # ALGO TRADING BUTTON OFF MID-SESSION!

        mock_mt5.terminal_info.return_value = MockTerminalInfo()
        mock_mt5.initialize.return_value = False
        monkeypatch.setattr(engine_mod, 'mt5', mock_mt5)

        res = engine._check_connection_health()
        assert res is False
        assert engine.failed_reconnect_attempts == 1
