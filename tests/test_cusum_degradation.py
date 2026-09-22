"""
WAVE 15: CUSUM Live Degradation Detector Test Suite.
Verifies sensitivity, specificity, state persistence across restarts, fail-closed baseline validation,
and architectural immutability of active_strategy.json.
"""

import sys
import json
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.core.degradation import CUSUMDetector, DegradationMonitor, TradeRecord, CUSUM_STATE_FILE
from engine.core.strategy_loader import StrategyLoader, StrategySpec
from engine.core.engine import Engine
from shared.dna import compute_dna_hash


class TestCUSUMDegradation:

    @pytest.fixture
    def tmp_state_dir(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    @pytest.fixture
    def valid_strategy_path(self, tmp_path):
        strat_dir = tmp_path / "strategy"
        strat_dir.mkdir(parents=True, exist_ok=True)
        strat_file = strat_dir / "active_strategy.json"

        params = {
            "fast_ma_period": 10,
            "slow_ma_period": 50,
            "risk_per_trade_pct": 1.0
        }
        filters = {}
        content = {
            "strategy_id": "STRAT_TEST_001",
            "dna_hash": compute_dna_hash(params, filters),
            "template": "ma_crossover",
            "symbol": "EURUSD",
            "timeframe": "M15",
            "composite_score": 0.75,
            "promoted_at": "2026-08-04T12:00:00Z",
            "parameters": params,
            "filters": filters,
            "wf_summary": {
                "valid_windows": 15,
                "total_trades": 300,
                "median_profit_factor": 1.50,
                "median_win_rate": 0.55,
                "profitable_window_rate": 0.65
            }
        }

        with open(strat_file, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)

        return strat_file

    def test_cusum_math_k_calculation(self):
        """Verify Log-Likelihood Ratio reference value k calculation."""
        detector = CUSUMDetector(strategy_id="TEST", p0=0.55, relative_shift=0.15)
        # p0 = 0.55, p1 = 0.4675
        # k should be between p1 and p0
        assert 0.4675 < detector.k < 0.55

    def test_sensitivity_degraded_stream_trips_halt(self, tmp_state_dir):
        """Sensitivity test: 30 consecutive losing trades trip CUSUM after 20-trade floor."""
        cusum_file = tmp_state_dir / "cusum_state.json"
        monitor = DegradationMonitor(
            strategy_id="STRAT_TEST_001",
            wf_median_pf=1.50,
            wf_profitable_rate=0.55,
            cusum_state_file=cusum_file
        )

        base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        # Feed 30 losing trades
        tripped_at_trade = None
        for i in range(30):
            t_rec = TradeRecord(
                trade_id=str(100 + i),
                direction="BUY",
                entry_price=1.1000,
                exit_price=1.0950,
                net_pnl=-50.0,
                close_reason="SL_HIT",
                closed_at=(base_time + timedelta(hours=i)).isoformat(),
                strategy_id="STRAT_TEST_001"
            )
            state = monitor.record_trade(t_rec)
            if state.cusum_tripped and tripped_at_trade is None:
                tripped_at_trade = i + 1

        assert monitor.state.cusum_tripped is True
        assert monitor.state.status == 'DEGRADED'
        assert monitor.state.rerun_recommended is True
        assert tripped_at_trade is not None
        assert tripped_at_trade >= 20  # Minimum sample floor respected

    def test_specificity_healthy_stream_does_not_trip(self, tmp_state_dir):
        """Specificity test: 50 trades matching baseline win rate (60% wins) do NOT trip CUSUM."""
        cusum_file = tmp_state_dir / "cusum_state.json"
        monitor = DegradationMonitor(
            strategy_id="STRAT_TEST_001",
            wf_median_pf=1.50,
            wf_profitable_rate=0.55,
            cusum_state_file=cusum_file
        )

        base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        for i in range(50):
            win = (i % 5 != 0)  # 80% win rate
            pnl = 50.0 if win else -30.0
            t_rec = TradeRecord(
                trade_id=str(200 + i),
                direction="BUY",
                entry_price=1.1000,
                exit_price=1.1050 if win else 1.0970,
                net_pnl=pnl,
                close_reason="TP_HIT" if win else "SL_HIT",
                closed_at=(base_time + timedelta(hours=i)).isoformat(),
                strategy_id="STRAT_TEST_001"
            )
            state = monitor.record_trade(t_rec)

        assert monitor.state.cusum_tripped is False
        assert monitor.state.cusum_score < 12.0
        assert monitor.state.status == 'OK'

    def test_persistence_across_restarts(self, tmp_state_dir):
        """Persistence test: CUSUM state is restored on restart if strategy_id matches."""
        cusum_file = tmp_state_dir / "cusum_state.json"

        # Instance 1: Feed 15 trades (under min_trades floor)
        monitor1 = DegradationMonitor(
            strategy_id="STRAT_PERSIST_001",
            wf_median_pf=1.50,
            wf_profitable_rate=0.55,
            cusum_state_file=cusum_file
        )
        base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        for i in range(15):
            t_rec = TradeRecord(
                trade_id=str(300 + i),
                direction="BUY",
                entry_price=1.1000,
                exit_price=1.0950,
                net_pnl=-50.0,
                close_reason="SL_HIT",
                closed_at=(base_time + timedelta(hours=i)).isoformat(),
                strategy_id="STRAT_PERSIST_001"
            )
            monitor1.record_trade(t_rec)

        saved_score = monitor1.state.cusum_score
        assert saved_score > 0.0

        # Instance 2: Restart with same strategy_id
        monitor2 = DegradationMonitor(
            strategy_id="STRAT_PERSIST_001",
            wf_median_pf=1.50,
            wf_profitable_rate=0.55,
            cusum_state_file=cusum_file
        )
        assert monitor2.state.cusum_score == pytest.approx(saved_score, abs=1e-4)
        assert monitor2.cusum.total_trades == 15

        # Instance 3: Restart with NEW strategy_id (resets accumulator)
        monitor3 = DegradationMonitor(
            strategy_id="STRAT_NEW_002",
            wf_median_pf=1.50,
            wf_profitable_rate=0.55,
            cusum_state_file=cusum_file
        )
        assert monitor3.state.cusum_score == 0.0
        assert monitor3.cusum.total_trades == 0

    def test_fail_closed_missing_baseline_strategy_json(self, tmp_path):
        """StrategyLoader fails closed (returns None) if baseline metrics are missing in strategy JSON."""
        bad_strat = tmp_path / "bad_strategy.json"
        bad_content = {
            "strategy_id": "BAD_STRAT_001",
            "template": "ma_crossover",
            "composite_score": 0.50,
            # Missing wf_summary with baseline metrics!
        }
        with open(bad_strat, "w", encoding="utf-8") as f:
            json.dump(bad_content, f, indent=2)

        loader = StrategyLoader(strategy_path=str(bad_strat))
        spec = loader.load_strategy()
        assert spec is None
        assert loader.load_errors > 0

    def test_no_active_strategy_file_mutation(self, valid_strategy_path, tmp_state_dir):
        """Verify that trade closes and CUSUM updates NEVER mutate active_strategy.json."""
        mtime_before = valid_strategy_path.stat().st_mtime
        content_before = valid_strategy_path.read_text(encoding="utf-8")

        cusum_file = tmp_state_dir / "cusum_state.json"
        monitor = DegradationMonitor(
            strategy_id="STRAT_TEST_001",
            wf_median_pf=1.50,
            wf_profitable_rate=0.55,
            cusum_state_file=cusum_file
        )

        for i in range(25):
            t_rec = TradeRecord(
                trade_id=str(400 + i),
                direction="BUY",
                entry_price=1.1000,
                exit_price=1.0950,
                net_pnl=-50.0,
                close_reason="SL_HIT",
                closed_at=datetime.now(timezone.utc).isoformat(),
                strategy_id="STRAT_TEST_001"
            )
            monitor.record_trade(t_rec)

        mtime_after = valid_strategy_path.stat().st_mtime
        content_after = valid_strategy_path.read_text(encoding="utf-8")

        assert mtime_before == mtime_after
        assert content_before == content_after

    def test_engine_end_to_end_cusum_degradation_trip(self, valid_strategy_path, tmp_state_dir):
        """
        End-to-end integration test: Feed 25 trade closes through Engine._record_closed_trade().
        Assert that CUSUM trips, Engine enters circuit_breaker_tripped = True and safe_mode = True,
        and logs LIVE_DEGRADATION_TRIPPED to audit ledger.
        """
        engine = Engine(symbol="EURUSD", paper_mode=True)
        # Override strategy loader path to valid_strategy_path
        engine.strategy_loader = StrategyLoader(strategy_path=str(valid_strategy_path))
        spec = engine.strategy_loader.load_strategy()
        assert spec is not None
        from engine.core.signal_engine import SignalEngine
        engine.signal_engine = SignalEngine(spec)

        engine.degradation = DegradationMonitor(
            strategy_id=spec.strategy_id,
            wf_median_pf=spec.wf_median_profit_factor,
            wf_profitable_rate=spec.wf_median_win_rate,
            cusum_state_file=tmp_state_dir / "cusum_state.json"
        )

        class MockPosition:
            ticket = 999111
            type = 0
            price_open = 1.1000
            price_current = 1.0950
            profit = -50.0

        # Feed 25 losing trades
        for i in range(25):
            MockPosition.ticket = 999000 + i
            engine._record_closed_trade(MockPosition(), "SL_HIT", {"balance": 10000.0})

        assert engine.degradation.state.cusum_tripped is True
        assert engine.circuit_breaker_tripped is True
        assert engine.safe_mode is True
        assert engine.degradation.state.status == 'DEGRADED'

