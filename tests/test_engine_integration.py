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
from datetime import datetime, timezone
from engine.core.strategy_loader import StrategyLoader, StrategySpec
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
        strategy = {
            'strategy_id':    'ATS-MA-001',
            'template':       'ma_crossover',
            'symbol':         'EURUSD',
            'timeframe':      'M15',
            'composite_score': 0.65,
            'promoted_at':    '2026-01-01T00:00:00+00:00',
            'parameters': {
                'fast_ma_period': 10,
                'slow_ma_period': 50,
                'sl_atr_multiple': 1.5,
                'tp_rr_ratio': 2.0,
                'risk_per_trade_pct': 1.0,
                'pip_size': 0.0001,
            },
            'filters': {},
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
        strategy = {
            'strategy_id':    'ATS-MA-001',
            'template':       'ma_crossover',
            'composite_score': 0.65,
            'parameters':     {},
            'filters':        {},
        }
        path = tmp_path / 'active_strategy.json'
        path.write_text(json.dumps(strategy))

        loader = StrategyLoader(strategy_path=str(path))
        loader.load_strategy()

        # No change — should return False
        assert loader.reload_if_changed() is False

        # Modify file
        strategy['strategy_id'] = 'ATS-MA-002'
        path.write_text(json.dumps(strategy))

        # Should detect change and return True
        assert loader.reload_if_changed() is True
        assert loader.get_spec().strategy_id == 'ATS-MA-002'

    def test_strategy_spec_properties(self, tmp_path):
        strategy = {
            'strategy_id':    'test',
            'template':       'ma_crossover',
            'composite_score': 0.60,
            'parameters': {
                'fast_ma_period':     15,
                'slow_ma_period':     80,
                'sl_atr_multiple':    2.0,
                'tp_rr_ratio':        2.5,
                'risk_per_trade_pct': 0.5,
                'pip_size':           0.01,
            },
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