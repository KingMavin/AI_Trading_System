"""
Unit tests for engine integration components.
Run: pytest tests/test_engine_integration.py -v
Pure logic — no MT5, no live data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import json
import tempfile
from datetime import datetime, timezone, timedelta
from engine.core.strategy_loader import StrategyLoader, StrategySpec
from shared.dna import compute_dna_hash
from engine.core.degradation import (
    DegradationMonitor, TradeRecord, THRESHOLDS
)
from engine.core.live_logger import LiveLogger, TrainerManifest


# ── STRATEGY LOADER TESTS ──────────────────────────────

class TestStrategyLoader:

    def test_no_file_returns_none(self, tmp_path):
        loader = StrategyLoader(
            strategy_path=str(tmp_path / 'missing.json')
        )
        spec = loader.load_strategy()
        assert spec is None
        assert loader.is_loaded() is False

    def test_valid_file_loads(self, tmp_path):
        params = {
            'fast_ma_period': 10,
            'slow_ma_period': 50,
            'sl_atr_multiple': 1.5,
            'tp_rr_ratio': 2.0,
            'risk_per_trade_pct': 1.0,
            'pip_size': 0.0001,
        }
        filters = {}
        strategy = {
            'strategy_id':    'ATS-MA-001',
            'dna_hash':       compute_dna_hash(params, filters),
            'template':       'ma_crossover',
            'symbol':         'EURUSD',
            'timeframe':      'M15',
            'composite_score': 0.65,
            'promoted_at':    '2026-01-01T00:00:00+00:00',
            'parameters':     params,
            'filters':        filters,
            'wf_summary': {'median_win_rate': 0.55, 'median_profit_factor': 1.5},
        }
        path = tmp_path / 'active_strategy.json'
        path.write_text(json.dumps(strategy))

        loader = StrategyLoader(strategy_path=str(path))
        spec   = loader.load_strategy()

        assert spec is not None
        assert spec.strategy_id    == 'ATS-MA-001'
        assert spec.template       == 'ma_crossover'
        assert spec.composite_score == 0.65
        assert spec.fast_ma        == 10
        assert spec.slow_ma        == 50
        assert loader.is_loaded()  is True

    def test_invalid_json_returns_none(self, tmp_path):
        path = tmp_path / 'active_strategy.json'
        path.write_text("this is not json {{{")

        loader = StrategyLoader(strategy_path=str(path))
        spec   = loader.load_strategy()
        assert spec is None

    def test_missing_required_field_returns_none(self,
                                                  tmp_path):
        strategy = {
            'template': 'ma_crossover',
            # Missing strategy_id and composite_score
        }
        path = tmp_path / 'active_strategy.json'
        path.write_text(json.dumps(strategy))

        loader = StrategyLoader(strategy_path=str(path))
        spec   = loader.load_strategy()
        assert spec is None

    def test_reload_detects_change(self, tmp_path):
        params = {}
        filters = {}
        strategy = {
            'strategy_id':    'ATS-MA-001',
            'dna_hash':       compute_dna_hash(params, filters),
            'template':       'ma_crossover',
            'composite_score': 0.65,
            'parameters':     params,
            'filters':        filters,
            'wf_summary': {'median_win_rate': 0.55, 'median_profit_factor': 1.5},
        }
        path = tmp_path / 'active_strategy.json'
        path.write_text(json.dumps(strategy))

        loader = StrategyLoader(strategy_path=str(path))
        loader.load_strategy()

        # No change — should return False
        assert loader.reload_if_changed() is False

        # Modify file
        strategy['strategy_id'] = 'ATS-MA-002'
        strategy['dna_hash'] = compute_dna_hash(params, filters)
        path.write_text(json.dumps(strategy))

        # Should detect change and return True
        assert loader.reload_if_changed() is True
        assert loader.get_spec().strategy_id == 'ATS-MA-002'

    def test_strategy_spec_properties(self, tmp_path):
        params = {
            'fast_ma_period':     15,
            'slow_ma_period':     80,
            'sl_atr_multiple':    2.0,
            'tp_rr_ratio':        2.5,
            'risk_per_trade_pct': 0.5,
            'pip_size':           0.01,
        }
        filters = {}
        strategy = {
            'strategy_id':    'test',
            'dna_hash':       compute_dna_hash(params, filters),
            'template':       'ma_crossover',
            'composite_score': 0.60,
            'parameters':     params,
            'filters':        filters,
            'wf_summary': {'median_win_rate': 0.55, 'median_profit_factor': 1.5},
        }
        path = tmp_path / 'active_strategy.json'
        path.write_text(json.dumps(strategy))

        loader = StrategyLoader(strategy_path=str(path))
        spec   = loader.load_strategy()

        assert spec.fast_ma            == 15
        assert spec.slow_ma            == 80
        assert spec.sl_atr_multiple    == 2.0
        assert spec.tp_rr_ratio        == 2.5
        assert spec.risk_per_trade_pct == 0.5
        assert spec.pip_size           == 0.01

    def test_get_status_returns_dict(self, tmp_path):
        loader = StrategyLoader(
            strategy_path=str(tmp_path / 'missing.json')
        )
        loader.load_strategy()
        status = loader.get_status()
        assert isinstance(status, dict)
        assert 'strategy_id'  in status
        assert 'load_errors'  in status
        assert 'strategy_path' in status


# ── DEGRADATION MONITOR TESTS ──────────────────────────

def make_trade(pnl: float, n: int = 0) -> TradeRecord:
    """Helper to create a test trade record."""
    return TradeRecord(
        trade_id       = f"trade_{n}",
        direction      = 'BUY' if pnl > 0 else 'SELL',
        entry_price    = 1.1000,
        exit_price     = 1.1000 + pnl * 0.0001,
        net_pnl        = pnl,
        close_reason   = 'TP_HIT' if pnl > 0 else 'SL_HIT',
        closed_at      = datetime.now(timezone.utc).isoformat(),
        strategy_id    = 'ATS-MA-001',
    )


class TestDegradationMonitor:

    @pytest.fixture(autouse=True)
    def clean_cusum_state(self):
        from engine.core.degradation import CUSUM_STATE_FILE
        if CUSUM_STATE_FILE.exists():
            try:
                CUSUM_STATE_FILE.unlink()
            except Exception:
                pass
        yield
        if CUSUM_STATE_FILE.exists():
            try:
                CUSUM_STATE_FILE.unlink()
            except Exception:
                pass

    def test_initialises_ok(self):
        monitor = DegradationMonitor(
            'ATS-MA-001', wf_median_pf=1.3
        )
        assert monitor.state.status == 'OK'
        assert monitor.state.alert_count == 0

    def test_insufficient_trades_no_alert(self):
        monitor = DegradationMonitor(
            'ATS-MA-001', wf_median_pf=1.3
        )
        # Add fewer than min_trades_for_assessment
        for i in range(5):
            monitor.record_trade(make_trade(-10.0, i))
        assert monitor.state.status == 'OK'

    def test_good_performance_stays_ok(self):
        monitor = DegradationMonitor(
            'ATS-MA-001',
            wf_median_pf=1.3,
            wf_profitable_rate=0.55
        )
        # 25 profitable trades
        for i in range(25):
            pnl = 50.0 if i % 3 != 0 else -20.0
            monitor.record_trade(make_trade(pnl, i))
        assert monitor.state.status == 'OK'
        assert monitor.state.alert_count == 0

    def test_consecutive_losses_triggers_warning(self):
        monitor = DegradationMonitor(
            'ATS-MA-001',
            wf_median_pf=1.3
        )
        # 20 trades then 8 consecutive losses
        for i in range(20):
            monitor.record_trade(make_trade(30.0, i))
        for i in range(
            THRESHOLDS['max_consecutive_losses']
        ):
            monitor.record_trade(make_trade(-20.0, 20+i))
        assert monitor.state.status in ('WARNING', 'DEGRADED')

    def test_pf_degradation_triggers_warning(self):
        monitor = DegradationMonitor(
            'ATS-MA-001',
            wf_median_pf=1.5,
            wf_profitable_rate=0.60
        )
        # 30 trades all losing — PF = 0
        for i in range(30):
            monitor.record_trade(make_trade(-15.0, i))
        assert monitor.state.status in ('WARNING', 'DEGRADED')
        assert monitor.state.alert_count > 0

    def test_rolling_metrics_calculated(self):
        monitor = DegradationMonitor(
            'ATS-MA-001',
            wf_median_pf=1.3
        )
        for i in range(10):
            pnl = 30.0 if i % 2 == 0 else -15.0
            monitor.record_trade(make_trade(pnl, i))

        assert monitor.state.rolling_win_rate == pytest.approx(
            0.5, abs=0.01
        )
        assert monitor.state.rolling_profit_factor > 0

    def test_get_summary_returns_dict(self):
        monitor = DegradationMonitor('ATS-MA-001')
        summary = monitor.get_summary()
        assert isinstance(summary, dict)
        assert 'strategy_id'       in summary
        assert 'status'            in summary
        assert 'rolling_pf'        in summary
        assert 'rerun_recommended' in summary

    def test_rerun_recommended_after_multiple_alerts(self):
        monitor = DegradationMonitor(
            'ATS-MA-001',
            wf_median_pf=1.5,
        )
        # Force enough consecutive losses to trigger
        # multiple degradation checks
        threshold = THRESHOLDS['alerts_before_rerun']
        for cycle in range(threshold + 1):
            for i in range(25):
                monitor.record_trade(
                    make_trade(-20.0, cycle*25+i)
                )
        assert monitor.state.rerun_recommended is True

    def test_write_status_file(self, tmp_path):
        monitor  = DegradationMonitor('ATS-MA-001')
        out_path = tmp_path / 'degradation_status.json'
        monitor.write_status_file(str(out_path))
        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert data['strategy_id'] == 'ATS-MA-001'


# ── LIVE LOGGER TESTS ──────────────────────────────────

class TestLiveLogger:

    def test_creates_log_directory(self, tmp_path):
        log_dir = tmp_path / 'logs'
        logger  = LiveLogger(
            log_dir=str(log_dir),
            strategy_id='ATS-MA-001'
        )
        assert log_dir.exists()

    def test_log_file_created(self, tmp_path):
        logger = LiveLogger(
            log_dir=str(tmp_path),
            strategy_id='ATS-MA-001'
        )
        from engine.core.live_logger import TradeLogRecord
        trade = TradeLogRecord(
            schema_version  = '1.0',
            trade_id        = 'trade_001',
            strategy_id     = 'ATS-MA-001',
            run_timestamp   = '20260101_000000',
            candle_time     = '2026-01-01T10:00:00+00:00',
            symbol          = 'EURUSD',
            timeframe       = 'M15',
            direction       = 'BUY',
            entry_price     = 1.10000,
            entry_time      = '2026-01-01T09:45:00+00:00',
            exit_price      = 1.10200,
            exit_time       = '2026-01-01T10:00:00+00:00',
            lots            = 0.02,
            gross_pnl       = 40.0,
            net_pnl         = 33.0,
            commission      = 0.14,
            spread_cost     = 0.20,
            close_reason    = 'TP_HIT',
            entry_session   = 'LONDON',
            entry_regime    = 'TRENDING',
            entry_atr       = 0.00080,
            entry_adx       = 28.5,
            entry_spread_pips=1.1,
            fast_ma_period  = 10,
            slow_ma_period  = 50,
            sl_atr_multiple = 1.5,
            tp_rr_ratio     = 2.0,
            duration_candles= 5,
            duration_minutes= 75.0,
            equity_after    = 10033.0,
            balance_after   = 10033.0,
            drawdown_pct    = 0.0,
        )
        logger.log_trade(trade)
        assert logger.records_written == 1

        # Verify file content
        log_files = list(Path(tmp_path).glob('*.jsonl'))
        assert len(log_files) == 1
        lines = log_files[0].read_text().strip().split('\n')
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record['trade_id']    == 'trade_001'
        assert record['net_pnl']     == 33.0
        assert record['close_reason']== 'TP_HIT'

    def test_get_stats(self, tmp_path):
        logger = LiveLogger(log_dir=str(tmp_path))
        stats  = logger.get_stats()
        assert 'log_dir'         in stats
        assert 'records_written' in stats
        assert stats['records_written'] == 0


# ── TRAINER MANIFEST TESTS ─────────────────────────────

class TestTrainerManifest:

    def test_initialises_empty(self, tmp_path):
        manifest = TrainerManifest(
            manifest_path=str(
                tmp_path / 'trainer_manifest.json'
            )
        )
        assert manifest.data['last_processed_file'] is None
        assert len(manifest.data['processed_files']) == 0

    def test_mark_and_check_processed(self, tmp_path):
        manifest = TrainerManifest(
            manifest_path=str(
                tmp_path / 'trainer_manifest.json'
            )
        )
        manifest.mark_processed('candle_20260101.jsonl', 42)
        assert manifest.is_processed(
            'candle_20260101.jsonl'
        ) is True
        assert manifest.is_processed(
            'candle_20260102.jsonl'
        ) is False

    def test_total_trades_accumulates(self, tmp_path):
        manifest = TrainerManifest(
            manifest_path=str(
                tmp_path / 'trainer_manifest.json'
            )
        )
        manifest.mark_processed('file1.jsonl', 10)
        manifest.mark_processed('file2.jsonl', 25)
        assert manifest.data['total_trades_logged'] == 35

    def test_persists_to_disk(self, tmp_path):
        path = tmp_path / 'trainer_manifest.json'
        m1   = TrainerManifest(manifest_path=str(path))
        m1.mark_processed('candle_20260101.jsonl', 15)

        # Reload from disk
        m2 = TrainerManifest(manifest_path=str(path))
        assert m2.is_processed('candle_20260101.jsonl')
        assert m2.data['total_trades_logged'] == 15


# ── CRASH RECOVERY & CIRCUIT BREAKER TESTS (WAVE 5) ────

class TestEngineCrashRecoveryAndCircuitBreakers:

    def test_atomic_write_json(self, tmp_path):
        from engine.core.engine import atomic_write_json
        target = tmp_path / 'state_snapshot.json'
        data = {'peak_equity': 10500.0, 'circuit_breaker_tripped': False}
        atomic_write_json(target, data)
        assert target.exists()
        with open(target, 'r') as f:
            read_data = json.load(f)
        assert read_data['peak_equity'] == 10500.0

    def test_load_snapshot_corrupt_fails_loudly(self, tmp_path, monkeypatch):
        import engine.core.engine as engine_mod
        snap_file = tmp_path / 'state_snapshot.json'
        snap_file.write_text("corrupted json {{")
        monkeypatch.setattr(engine_mod, 'SNAPSHOT_FILE', snap_file)

        with pytest.raises(RuntimeError, match="corrupted or unparseable"):
            engine_mod.load_snapshot()

    def test_load_snapshot_missing_fields_fails_loudly(self, tmp_path, monkeypatch):
        import engine.core.engine as engine_mod
        snap_file = tmp_path / 'state_snapshot.json'
        snap_file.write_text(json.dumps({'peak_equity': 10000.0})) # Missing daily_start_equity, etc.
        monkeypatch.setattr(engine_mod, 'SNAPSHOT_FILE', snap_file)

        with pytest.raises(RuntimeError, match="missing required fields"):
            engine_mod.load_snapshot()

    def test_outage_exceeding_30_days_fails_loudly(self, tmp_path, monkeypatch):
        import engine.core.engine as engine_mod
        from unittest.mock import patch, MagicMock

        snap_file = tmp_path / 'state_snapshot.json'
        flag_file = tmp_path / 'shutdown_flag.json'
        flag_file.write_text(json.dumps({'clean_shutdown': False}))

        old_dt = (datetime.now(timezone.utc) - timedelta(days=35)).isoformat()
        snapshot_data = {
            'peak_equity': 12000.0,
            'daily_start_equity': 10000.0,
            'circuit_breaker_tripped': False,
            'last_candle_time': old_dt,
            'last_known_tickets': []
        }
        snap_file.write_text(json.dumps(snapshot_data))

        monkeypatch.setattr(engine_mod, 'SNAPSHOT_FILE', snap_file)
        monkeypatch.setattr(engine_mod, 'SHUTDOWN_FLAG_FILE', flag_file)

        mock_loader = MagicMock()
        mock_spec = MagicMock()
        mock_spec.strategy_id = 'ATS-001'
        mock_spec.parameters = {}
        mock_loader.load_strategy.return_value = mock_spec

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        engine.strategy_loader = mock_loader

        with pytest.raises(RuntimeError, match="exceeds maximum automatic reconciliation limit"):
            engine.start()

    def test_engine_start_corrupt_snapshot_halts_loudly(self, tmp_path, monkeypatch):
        import engine.core.engine as engine_mod

        snap_file = tmp_path / 'state_snapshot.json'
        flag_file = tmp_path / 'shutdown_flag.json'
        flag_file.write_text(json.dumps({'clean_shutdown': False}))
        snap_file.write_text("corrupted json {{{")

        monkeypatch.setattr(engine_mod, 'SNAPSHOT_FILE', snap_file)
        monkeypatch.setattr(engine_mod, 'SHUTDOWN_FLAG_FILE', flag_file)

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        with pytest.raises(RuntimeError, match="corrupted or unparseable"):
            engine.start()

    def test_circuit_breaker_blocks_new_entries_only(self, tmp_path, monkeypatch):
        import engine.core.engine as engine_mod
        from unittest.mock import MagicMock
        import pandas as pd

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        engine.daily_start_equity = 10000.0
        engine.peak_equity = 10000.0
        engine.max_daily_loss_pct = 3.0
        engine.last_candle_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        # Mock MT5 candle fetching and account info (-6% loss)
        dates = pd.date_range('2026-01-01 12:15', periods=5, freq='15min', tz='UTC')
        df_mock = pd.DataFrame({
            'open': [1.1000]*5, 'high': [1.1010]*5, 'low': [1.0990]*5, 'close': [1.1000]*5, 'volume': [100.0]*5
        }, index=dates)

        monkeypatch.setattr(engine_mod, 'fetch_candles', lambda sym, tf, count: df_mock)
        monkeypatch.setattr(engine_mod, 'get_account_info', lambda: {'equity': 9400.0, 'balance': 9400.0})
        monkeypatch.setattr(engine_mod, 'get_open_positions', lambda sym: [])

        mock_loader = MagicMock()
        mock_spec = MagicMock()
        mock_spec.strategy_id = 'ATS-001'
        mock_spec.risk_per_trade_pct = 1.0
        mock_spec.pip_size = 0.0001
        mock_spec.parameters = {'exit_on_opposite_crossover': True}
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

        # Run process_candle which evaluates equity loss and trips circuit_breaker_tripped
        engine._process_candle()

        assert engine.circuit_breaker_tripped is True
        # Verify that new entry was blocked
        assert engine.orders_placed == 0

        # Now test that position management STILL runs when circuit_breaker_tripped is True
        class MockPosition:
            ticket = 999
            type = 0 # BUY position
            volume = 0.1
            profit = -50.0

        position_managed = False
        def mock_close(pos, paper_mode):
            nonlocal position_managed
            position_managed = True
            return True

        monkeypatch.setattr(engine_mod, 'close_position', mock_close)
        engine._manage_positions(positions=[MockPosition()], signal={'signal': 'SELL'}, df=df_mock, spec=mock_spec, account={'equity': 9400.0})

        # Confirm position management executed and attempted exit on opposite signal
        assert position_managed is True

    def test_reconcile_closed_positions_detects_broker_sl_close(self, monkeypatch):
        import engine.core.engine as engine_mod
        from unittest.mock import MagicMock

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)
        engine.last_known_tickets = [999]
        engine.last_candle_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        # Mock MT5_AVAILABLE = True
        monkeypatch.setattr(engine_mod, 'MT5_AVAILABLE', True)

        # Mock deal object returned by mt5.history_deals_get
        class MockDeal:
            ticket = 5001
            position = 999
            entry = 1 # DEAL_ENTRY_OUT
            reason = 4 # DEAL_REASON_SL
            type = 1 # SELL deal to close BUY position
            price = 1.0980
            volume = 0.1
            profit = -150.0
            swap = 0.0
            commission = -0.5

        class MockEntryDeal:
            ticket = 5000
            position = 999
            entry = 0 # DEAL_ENTRY_IN
            type = 0 # BUY deal
            price = 1.1000

        mock_deals = [MockEntryDeal(), MockDeal()]
        monkeypatch.setattr(engine_mod.mt5, 'history_deals_get', lambda from_date, to_date, position: mock_deals)
        monkeypatch.setattr(engine_mod.mt5, 'DEAL_ENTRY_OUT', 1)
        monkeypatch.setattr(engine_mod.mt5, 'DEAL_REASON_SL', 4)

        recorded_trades = []
        def mock_record_closed_trade(pos, reason, account):
            recorded_trades.append((pos, reason, account))

        monkeypatch.setattr(engine, '_record_closed_trade', mock_record_closed_trade)

        # Call _reconcile_closed_positions with current_positions = [] (ticket #999 absent)
        engine._reconcile_closed_positions(current_positions=[])

        # Confirm deal reconciliation detected closed ticket #999, identified reason='SL_HIT', and recorded net profit -$150.50
        assert len(recorded_trades) == 1
        pos, reason, acc = recorded_trades[0]
        assert pos.ticket == 999
        assert reason == 'SL_HIT'
        assert pos.profit == -150.5
        assert engine.last_known_tickets == []

    # ── WAVE 16 SAFETY DEFAULT CLEANUP TESTS ──────────────────────

    def test_calculate_lot_size_uses_instrument_spec(self, monkeypatch):
        import engine.core.engine as engine_mod
        from shared.instrument_spec import InstrumentSpec

        # Mock captured spec for EURUSD
        mock_spec = InstrumentSpec(
            symbol="EURUSD", digits=5, point=0.00001, contract_size=100000.0,
            tick_size=0.00001, tick_value=1.0, volume_min=0.01, volume_step=0.01,
            volume_max=100.0, stops_level=10, spread_typical=10, swap_long=-5.0,
            swap_short=-5.0, currency_base="EUR", currency_profit="USD",
            currency_margin="EUR", account_currency="USD", captured_at="2026-01-01T00:00:00Z",
            sanity_ok=True
        )
        monkeypatch.setattr("shared.instrument_spec.get_spec", lambda symbol: mock_spec)

        lots = engine_mod.calculate_lot_size(symbol="EURUSD", sl_pips=20.0, risk_pct=1.0, balance=10000.0)
        # $100 risk / (20 pips * $10/pip) = 0.50 lots
        assert lots == 0.50

    def test_calculate_lot_size_missing_spec_raises_value_error(self, monkeypatch):
        import engine.core.engine as engine_mod

        monkeypatch.setattr("shared.instrument_spec.get_spec", lambda symbol: None)

        with pytest.raises(ValueError, match="INSTRUMENT_SPEC_MISSING"):
            engine_mod.calculate_lot_size(symbol="UNKNOWN_PAIR", sl_pips=20.0, risk_pct=1.0, balance=10000.0)

    def test_calculate_lot_size_invalid_sanity_raises_value_error(self, monkeypatch):
        import engine.core.engine as engine_mod
        from shared.instrument_spec import InstrumentSpec

        invalid_spec = InstrumentSpec(
            symbol="BAD_PAIR", digits=5, point=0.00001, contract_size=100000.0,
            tick_size=0.00001, tick_value=0.0, volume_min=0.01, volume_step=0.01,
            volume_max=100.0, stops_level=10, spread_typical=10, swap_long=-5.0,
            swap_short=-5.0, currency_base="BAD", currency_profit="USD",
            currency_margin="BAD", account_currency="USD", captured_at="2026-01-01T00:00:00Z",
            sanity_ok=False
        )
        monkeypatch.setattr("shared.instrument_spec.get_spec", lambda symbol: invalid_spec)

        with pytest.raises(ValueError, match="INSTRUMENT_SPEC_INVALID"):
            engine_mod.calculate_lot_size(symbol="BAD_PAIR", sl_pips=20.0, risk_pct=1.0, balance=10000.0)

    def test_enter_trade_missing_sl_pips_logs_error_no_order(self, monkeypatch):
        import engine.core.engine as engine_mod

        engine = engine_mod.Engine(symbol='EURUSD', paper_mode=True)

        order_placed = False
        def mock_place_order(*args, **kwargs):
            nonlocal order_placed
            order_placed = True
            return {}

        monkeypatch.setattr(engine_mod, 'place_order', mock_place_order)

        # Call _enter_trade with sl_pips missing/0
        engine._enter_trade(signal={'signal': 'BUY', 'sl_pips': 0}, spec=None, balance=10000.0)

        assert order_placed is False