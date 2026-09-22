"""
WAVE 14: Gate 10 Monte Carlo / Slippage Stress Gate Test Suite.

Tests cover:
  - run_monte_carlo_stress_test: lenient/strict mandate, zero trades, missing InstrumentSpec
  - Initial-equity anchor: drawdown from initial balance is caught even when first trade is a loss
  - check_hard_gates Gate 10: absent key fails loudly (GATE_10_DATA_MISSING), empty list fails loudly,
    real trades pass lenient mandate, real trades fail strict mandate
  - Fabrication anti-pattern: confirmed absent — missing all_oos_trades raises GATE_10_DATA_MISSING,
    never invents trade data
  - Wiring: aggregate_wf_results() on real WindowResult objects propagates oos_trades_list correctly
    into all_oos_trades, which check_hard_gates then consumes (end-to-end wiring test)
  - Timing: Gate 10 cost measured with time.perf_counter on 500 permutations of 100 trades
"""

import sys
import time
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.core.mandate import Mandate
from trainer.core.scoring import (
    check_hard_gates,
    run_monte_carlo_stress_test,
    aggregate_wf_results,
    GATES,
)
from trainer.core.walk_forward import WindowResult


# ── HELPERS ────────────────────────────────────────────────────────────────────

def _lenient_mandate() -> Mandate:
    return Mandate(
        mandate_id="MANDATE_LENIENT",
        firm_name="TestFirm",
        account_id="ACC_001",
        max_daily_loss_pct=20.0,
        max_overall_drawdown_pct=20.0,
        drawdown_type="static",
        min_trading_days=1,
        max_daily_risk_pct=5.0,
        symbol_universe=["EURUSD"]
    )


def _strict_mandate() -> Mandate:
    """0.00001% limits — any loss at all triggers a breach."""
    return Mandate(
        mandate_id="MANDATE_STRICT",
        firm_name="TestFirm",
        account_id="ACC_002",
        max_daily_loss_pct=0.00001,
        max_overall_drawdown_pct=0.00001,
        drawdown_type="static",
        min_trading_days=1,
        max_daily_risk_pct=1.0,
        symbol_universe=["EURUSD"]
    )


def _build_trades(n: int, win_pnl: float = 50.0, loss_pnl: float = -30.0) -> list:
    """Build n alternating-win/loss trades, 1 per calendar day."""
    base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    trades = []
    for i in range(n):
        day = base + timedelta(days=i)
        pnl = win_pnl if i % 2 == 0 else loss_pnl
        trades.append({
            'direction': 'BUY',
            'entry_price': 1.1000,
            'entry_time': day,
            'exit_price': 1.1050,
            'exit_time': day + timedelta(minutes=30),
            'lots': 1.0,
            'pnl': pnl,
        })
    return trades


def _build_window_result(window_num: int, trades: list, net_profit: float) -> WindowResult:
    """Construct a minimal valid WindowResult with oos_trades_list populated."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return WindowResult(
        window_num=window_num,
        opt_start=base,
        opt_end=base + timedelta(days=90),
        test_start=base + timedelta(days=91),
        test_end=base + timedelta(days=121),
        oos_trades=len(trades),
        oos_profit_factor=1.5,
        oos_calmar=1.2,
        oos_net_profit=net_profit,
        oos_win_rate=0.60,
        oos_max_drawdown=5.0,
        oos_composite_score=0.65,
        status='OK',
        oos_mandate_compliant=True,
        oos_mandate_breach_reason=None,
        oos_trades_list=trades,
    )


def _good_wf_summary_with_trades(trades: list) -> dict:
    """Return a wf_summary dict that passes Gates 1-9 and includes real trade data."""
    return {
        'valid_windows':             20,
        'total_trades':              len(trades),
        'median_trades_per_window':  22.0,
        'median_profit_factor':      1.35,
        'median_calmar_ratio':       1.20,
        'median_avg_win_loss':       1.80,
        'median_max_drawdown_pct':   12.0,
        'profitable_window_rate':    0.70,
        'max_consecutive_losses':    6,
        'mandate_compliant':         True,
        'all_oos_trades':            trades,
    }


# ── TEST CLASS ─────────────────────────────────────────────────────────────────

class TestMonteCarloGate:

    # ── run_monte_carlo_stress_test ────────────────────────────────────────────

    def test_lenient_mandate_passes(self):
        """Profitable strategy on a lenient mandate should breach 0 permutations."""
        trades = _build_trades(50)
        breach_rate, reason = run_monte_carlo_stress_test(
            trades=trades,
            symbol="EURUSD",
            mandate=_lenient_mandate(),
            initial_equity=10000.0,
            n_permutations=100,
            max_slippage_pips=1.0,
            seed=42,
        )
        assert breach_rate == 0.0
        assert reason is None

    def test_strict_mandate_fails_all_permutations(self):
        """Strict mandate (0.00001% limit) breaches every permutation."""
        trades = _build_trades(50)
        breach_rate, reason = run_monte_carlo_stress_test(
            trades=trades,
            symbol="EURUSD",
            mandate=_strict_mandate(),
            initial_equity=10000.0,
            n_permutations=100,
            max_slippage_pips=1.0,
            seed=42,
        )
        assert breach_rate == 100.0
        assert reason is not None
        assert "MANDATE_" in reason

    def test_zero_trades_fails_loudly(self):
        """Empty trade list returns 100% breach with descriptive reason — no fabrication."""
        breach_rate, reason = run_monte_carlo_stress_test(
            trades=[],
            symbol="EURUSD",
            mandate=_lenient_mandate(),
            n_permutations=50,
        )
        assert breach_rate == 100.0
        assert "Zero out-of-sample trades" in reason

    def test_missing_spec_raises_runtime_error(self):
        """Unknown symbol raises RuntimeError (no InstrumentSpec) — never silently falls back."""
        with pytest.raises(RuntimeError, match="INSTRUMENT_SPEC_MISSING"):
            run_monte_carlo_stress_test(
                trades=_build_trades(10),
                symbol="UNKNOWNSYMBOL_XYZ",
                mandate=_lenient_mandate(),
                n_permutations=5,
            )

    def test_initial_equity_anchor_catches_drawdown_from_start(self):
        """
        All-loss permutation: without anchor, peak_equity starts at first-trade equity (below
        initial_equity), missing the true drawdown. With anchor, peak is initial_equity,
        so a run of immediate losses triggers an overall-DD breach.

        We force this by using a mandate whose overall_DD limit = 0.5% and a trade list
        where the first trade always loses ~1% of initial equity.
        """
        initial_eq = 10000.0
        loss_per_trade = -55.0  # ~0.55% of 10000
        mandate = Mandate(
            mandate_id="ANCHOR_TEST",
            firm_name="TestFirm",
            account_id="ACC_003",
            max_daily_loss_pct=99.0,      # irrelevant — only testing overall DD
            max_overall_drawdown_pct=0.5,  # 0.5% of initial_equity = $50
            drawdown_type="static",
            min_trading_days=1,
            max_daily_risk_pct=99.0,
            symbol_universe=["EURUSD"],
        )
        # All-loss trades; seed=0 → first permutation starts with losses immediately
        trades = [{'direction': 'BUY', 'entry_price': 1.1, 'entry_time': None,
                   'exit_price': 1.09, 'exit_time': None, 'lots': 1.0, 'pnl': loss_per_trade}] * 5

        breach_rate, reason = run_monte_carlo_stress_test(
            trades=trades,
            symbol="EURUSD",
            mandate=mandate,
            initial_equity=initial_eq,
            n_permutations=10,
            max_slippage_pips=0.0,  # no extra slippage so we control exact pnl
            seed=0,
        )
        # Every permutation must breach the 0.5% overall DD limit from initial $10000 anchor
        assert breach_rate == 100.0, (
            f"Expected 100% breach (anchor missing or wrong), got {breach_rate}%"
        )
        assert reason is not None
        assert "MANDATE_OVERALL_DD_BREACH" in reason

    # ── check_hard_gates ───────────────────────────────────────────────────────

    def test_gate_10_passes_with_real_trades(self):
        """Gate 10 passes and populates gate_10_monte_carlo_stress = 0.0."""
        trades = _build_trades(150)
        summary = _good_wf_summary_with_trades(trades)
        result = check_hard_gates(
            wf_summary=summary,
            candidate_id="cand_pass",
            mandate=_lenient_mandate(),
            symbol="EURUSD",
            initial_equity=10000.0,
        )
        assert result.passed is True
        assert 'gate_10_monte_carlo_stress' in result.gate_details
        assert result.gate_details['gate_10_monte_carlo_stress'] == 0.0

    def test_gate_10_fails_with_strict_mandate(self):
        """Gate 10 hard-rejects on strict mandate with descriptive reason."""
        trades = _build_trades(150)
        summary = _good_wf_summary_with_trades(trades)
        result = check_hard_gates(
            wf_summary=summary,
            candidate_id="cand_strict",
            mandate=_strict_mandate(),
            symbol="EURUSD",
            initial_equity=10000.0,
        )
        assert result.passed is False
        assert result.failed_gate == 'gate_10_monte_carlo_stress'
        assert result.gate_details['gate_10_monte_carlo_stress'] == 100.0
        assert "Monte Carlo stress breach" in result.failure_reason

    def test_gate_10_absent_key_fails_loudly_no_fabrication(self):
        """
        When all_oos_trades key is absent, Gate 10 returns GATE_10_DATA_MISSING —
        it does NOT fabricate trade data to produce a result.
        This is the anti-fabrication regression guard.
        """
        summary = _good_wf_summary_with_trades(_build_trades(150))
        del summary['all_oos_trades']   # key entirely absent

        result = check_hard_gates(
            wf_summary=summary,
            candidate_id="cand_no_key",
            mandate=_lenient_mandate(),
            symbol="EURUSD",
            initial_equity=10000.0,
        )
        assert result.passed is False
        assert result.failed_gate == 'gate_10_monte_carlo_stress'
        assert "GATE_10_DATA_MISSING" in result.failure_reason

    def test_gate_10_empty_trades_fails_loudly(self):
        """When all_oos_trades is an empty list, Gate 10 returns GATE_10_ZERO_TRADES."""
        summary = _good_wf_summary_with_trades(_build_trades(150))
        summary['all_oos_trades'] = []  # key present but empty

        result = check_hard_gates(
            wf_summary=summary,
            candidate_id="cand_empty",
            mandate=_lenient_mandate(),
            symbol="EURUSD",
            initial_equity=10000.0,
        )
        assert result.passed is False
        assert result.failed_gate == 'gate_10_monte_carlo_stress'
        assert "GATE_10_ZERO_TRADES" in result.failure_reason

    # ── Wiring test: aggregate_wf_results → check_hard_gates ──────────────────

    def test_wiring_aggregate_wf_results_propagates_trades_to_gate_10(self):
        """
        End-to-end wiring: build real WindowResult objects → call aggregate_wf_results()
        → verify all_oos_trades is populated from WindowResult.oos_trades_list
        → call check_hard_gates() → Gate 10 evaluates real trades, not fabricated data.

        This is the test that validates the actual production path, not just gate logic
        given pre-assembled input.
        """
        # 12 windows × 25 trades each = 300 total OOS trades
        per_window_trades = _build_trades(25)
        windows = [
            _build_window_result(i, per_window_trades, net_profit=500.0)
            for i in range(12)
        ]

        # Step 1: aggregate_wf_results() aggregates WindowResult.oos_trades_list
        wf_summary = aggregate_wf_results(windows)

        # Confirm wiring: all_oos_trades must be present and populated from real windows
        assert 'all_oos_trades' in wf_summary, \
            "aggregate_wf_results() must include 'all_oos_trades' key"
        assert len(wf_summary['all_oos_trades']) == 12 * 25, (
            f"Expected 300 real OOS trades, got {len(wf_summary['all_oos_trades'])}. "
            "WindowResult.oos_trades_list is not being collected."
        )
        # Confirm it is the actual trade objects, not fabricated ones
        first_trade = wf_summary['all_oos_trades'][0]
        assert 'pnl' in first_trade and 'lots' in first_trade, \
            "Trade dicts should have real pnl/lots fields from WindowResult"

        # Step 2: check_hard_gates() with lenient mandate should pass Gate 10
        result = check_hard_gates(
            wf_summary=wf_summary,
            candidate_id="wiring_test_cand",
            mandate=_lenient_mandate(),
            symbol="EURUSD",
            initial_equity=10000.0,
        )
        assert 'gate_10_monte_carlo_stress' in result.gate_details, \
            "Gate 10 must run and appear in gate_details when wf_summary has real trades"
        assert result.gate_details['gate_10_monte_carlo_stress'] == 0.0

    # ── Timing ─────────────────────────────────────────────────────────────────

    def test_gate_10_timing_500_permutations_100_trades(self):
        """
        Measure actual wall-clock time for 500 permutations on 100 trades on AZRAEL
        (i7-6600U, 16GB). Should complete in well under 1 second.
        Report exact measured time — no estimate or assertion on a specific ceiling
        since CI speeds vary; the number is recorded for the changelog.
        """
        trades = _build_trades(100)
        mandate = _lenient_mandate()

        t0 = time.perf_counter()
        breach_rate, _ = run_monte_carlo_stress_test(
            trades=trades,
            symbol="EURUSD",
            mandate=mandate,
            initial_equity=10000.0,
            n_permutations=500,
            max_slippage_pips=1.5,
            seed=42,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # The timing is reported; we assert <30 seconds (very conservative) so the
        # test never false-positives on a loaded CI machine, while surfacing real hangs.
        print(f"\n[TIMING] Gate 10: 500 permutations × 100 trades = {elapsed_ms:.1f} ms")
        assert elapsed_ms < 30_000, (
            f"Gate 10 took {elapsed_ms:.0f} ms — unexpectedly slow. Expected < 30 s."
        )
        assert breach_rate == 0.0  # sanity: lenient mandate should not breach
