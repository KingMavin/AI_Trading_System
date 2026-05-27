"""
Unit tests for dashboard backend.
Tests data reading, equity curve building, status compilation.
Run: pytest tests/test_dashboard.py -v
No FastAPI installation required for these tests.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import json
import tempfile
from datetime import datetime, timezone


# ── EQUITY CURVE TESTS ─────────────────────────────────

class TestEquityCurve:

    def test_empty_trades_returns_single_point(self):
        from engine.dashboard.app import build_equity_curve
        curve = build_equity_curve([], initial_equity=10000)
        assert len(curve) >= 1
        assert curve[0]['equity'] == 10000

    def test_profitable_trades_increase_equity(self):
        from engine.dashboard.app import build_equity_curve
        trades = [
            {
                'trade_id':   f't{i}',
                'entry_time': '2026-01-01T09:00:00+00:00',
                'exit_time':  f'2026-01-0{i+1}T10:00:00+00:00',
                'net_pnl':    100.0,
            }
            for i in range(5)
        ]
        curve = build_equity_curve(trades, 10000)
        assert curve[-1]['equity'] > 10000
        assert curve[-1]['equity'] == pytest.approx(10500.0)

    def test_losing_trades_decrease_equity(self):
        from engine.dashboard.app import build_equity_curve
        trades = [
            {
                'trade_id':   f't{i}',
                'entry_time': '2026-01-01T09:00:00+00:00',
                'exit_time':  f'2026-01-0{i+1}T10:00:00+00:00',
                'net_pnl':    -50.0,
            }
            for i in range(4)
        ]
        curve = build_equity_curve(trades, 10000)
        assert curve[-1]['equity'] < 10000
        assert curve[-1]['equity'] == pytest.approx(9800.0)

    def test_curve_is_chronological(self):
        from engine.dashboard.app import build_equity_curve
        trades = [
            {
                'trade_id':   't2',
                'entry_time': '2026-01-02T09:00:00+00:00',
                'exit_time':  '2026-01-02T10:00:00+00:00',
                'net_pnl':    200.0,
            },
            {
                'trade_id':   't1',
                'entry_time': '2026-01-01T09:00:00+00:00',
                'exit_time':  '2026-01-01T10:00:00+00:00',
                'net_pnl':    100.0,
            },
        ]
        curve = build_equity_curve(trades, 10000)
        # Should be sorted chronologically
        equities = [p['equity'] for p in curve]
        assert equities[-1] == pytest.approx(10300.0)

    def test_curve_has_time_and_equity_keys(self):
        from engine.dashboard.app import build_equity_curve
        trades = [{
            'trade_id':   't1',
            'entry_time': '2026-01-01T09:00:00+00:00',
            'exit_time':  '2026-01-01T10:00:00+00:00',
            'net_pnl':    50.0,
        }]
        curve = build_equity_curve(trades, 10000)
        for point in curve:
            assert 'time'   in point
            assert 'equity' in point


# ── STATUS COMPILATION TESTS ───────────────────────────

class TestStatusCompilation:

    def test_status_returns_dict(self, tmp_path, monkeypatch):
        import engine.dashboard.app as app_module
        monkeypatch.setattr(
            app_module, 'STRATEGY_PATH',
            tmp_path / 'strategy.json'
        )
        monkeypatch.setattr(
            app_module, 'DEGRADATION_PATH',
            tmp_path / 'degradation.json'
        )
        monkeypatch.setattr(
            app_module, 'STATE_PATH',
            tmp_path / 'state.json'
        )
        monkeypatch.setattr(
            app_module, 'LOGS_PATH',
            tmp_path
        )

        from engine.dashboard.app import get_engine_status
        status = get_engine_status()

        assert isinstance(status, dict)
        assert 'timestamp'    in status
        assert 'strategy'     in status
        assert 'degradation'  in status
        assert 'performance'  in status
        assert 'current_state' in status

    def test_status_with_strategy_file(self,
                                        tmp_path,
                                        monkeypatch):
        import engine.dashboard.app as app_module
        strategy_path = tmp_path / 'strategy.json'
        strategy_path.write_text(json.dumps({
            'strategy_id':    'ATS-MA-001',
            'template':       'ma_crossover',
            'composite_score': 0.65,
            'promoted_at':    '2026-01-01T00:00:00+00:00',
            'parameters':     {'fast_ma_period': 10},
        }))

        monkeypatch.setattr(
            app_module, 'STRATEGY_PATH', strategy_path
        )
        monkeypatch.setattr(
            app_module, 'DEGRADATION_PATH',
            tmp_path / 'missing.json'
        )
        monkeypatch.setattr(
            app_module, 'STATE_PATH',
            tmp_path / 'missing.json'
        )
        monkeypatch.setattr(app_module, 'LOGS_PATH', tmp_path)

        from engine.dashboard.app import get_engine_status
        status = get_engine_status()

        assert status['strategy']['id']    == 'ATS-MA-001'
        assert status['strategy']['score'] == 0.65

    def test_status_without_files_returns_defaults(self,
                                                    tmp_path,
                                                    monkeypatch):
        import engine.dashboard.app as app_module
        monkeypatch.setattr(
            app_module, 'STRATEGY_PATH',
            tmp_path / 'missing.json'
        )
        monkeypatch.setattr(
            app_module, 'DEGRADATION_PATH',
            tmp_path / 'missing.json'
        )
        monkeypatch.setattr(
            app_module, 'STATE_PATH',
            tmp_path / 'missing.json'
        )
        monkeypatch.setattr(app_module, 'LOGS_PATH', tmp_path)

        from engine.dashboard.app import get_engine_status
        status = get_engine_status()

        # Should not crash — return defaults
        assert status['strategy']['id'] is not None
        assert status['performance']['total_trades'] == 0


# ── READ JSON SAFE TESTS ───────────────────────────────

class TestReadJsonSafe:

    def test_missing_file_returns_default(self, tmp_path):
        from engine.dashboard.app import read_json_safe
        result = read_json_safe(
            tmp_path / 'missing.json',
            default={'key': 'default'}
        )
        assert result == {'key': 'default'}

    def test_valid_file_returns_content(self, tmp_path):
        from engine.dashboard.app import read_json_safe
        path = tmp_path / 'test.json'
        path.write_text(json.dumps({'key': 'value'}))
        result = read_json_safe(path)
        assert result == {'key': 'value'}

    def test_invalid_json_returns_default(self, tmp_path):
        from engine.dashboard.app import read_json_safe
        path = tmp_path / 'bad.json'
        path.write_text("not json {{{{")
        result = read_json_safe(path, default={'err': True})
        assert result == {'err': True}


# ── DASHBOARD HTML TESTS ───────────────────────────────

class TestDashboardHTML:

    def test_html_is_non_empty(self):
        from engine.dashboard.app import DASHBOARD_HTML
        assert len(DASHBOARD_HTML) > 1000

    def test_html_has_required_elements(self):
        from engine.dashboard.app import DASHBOARD_HTML
        required_ids = [
            'stat-equity', 'stat-pnl', 'stat-pf',
            'stat-wr', 'stat-session', 'stat-degrade',
            'strategy-info', 'degradation-info',
            'trades-container', 'equity-canvas',
        ]
        for el_id in required_ids:
            assert el_id in DASHBOARD_HTML, \
                f"Missing element id: {el_id}"

    def test_html_polls_api(self):
        from engine.dashboard.app import DASHBOARD_HTML
        assert '/api/status'  in DASHBOARD_HTML
        assert '/api/equity'  in DASHBOARD_HTML
        assert '/api/trades'  in DASHBOARD_HTML

    def test_html_has_canvas_for_equity_chart(self):
        from engine.dashboard.app import DASHBOARD_HTML
        assert 'equity-canvas' in DASHBOARD_HTML
        assert 'getContext'    in DASHBOARD_HTML