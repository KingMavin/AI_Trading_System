"""
Scoring Engine — converts walk-forward results into
a single composite score and ranks candidates.

The composite score is the single number that determines
whether a strategy gets promoted, flagged, or rejected.

Score dimensions (0.0 to 1.0 each):
  40% — Risk-adjusted return (Calmar ratio)
  25% — Profitability quality (Profit Factor)
  15% — Win quality (Avg Win/Loss ratio)
  10% — Regime consistency
  05% — Temporal stability
  05% — Robustness (Tier 2 data sources)

Final score range: 0.0 to 1.0
Good strategy: 0.55 - 0.75
Production-worthy: >= 0.50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ── HARD GATE THRESHOLDS ───────────────────────────────
GATES = {
    'min_total_trades':          100,
    'min_trades_per_window':      20,
    'min_valid_windows':          12,
    'max_median_drawdown_pct':    20.0,
    'min_median_profit_factor':    1.1,
    'min_profitable_window_rate':  0.55,
    'max_consecutive_losses':     10,
    'min_regime_windows':          3,
    'max_monte_carlo_breach_rate': 3.0,
}

# ── SCORING WEIGHTS ────────────────────────────────────
WEIGHTS = {
    'calmar':           0.40,
    'profit_factor':    0.25,
    'win_loss_ratio':   0.15,
    'regime':           0.10,
    'stability':        0.05,
    'robustness':       0.05,
}

# ── SENSITIVITY MODIFIERS ─────────────────────────────
FRAGILE_PENALTY = 0.10   # 10% penalty for fragile strategies
ROBUST_BONUS    = 0.05   # 5% bonus for robust strategies


@dataclass
class GateResult:
    """Result of hard gate filtering for one candidate."""
    candidate_id:   str
    passed:         bool
    failed_gate:    Optional[str]   = None
    failure_reason: Optional[str]   = None
    gate_details:   Dict            = field(default_factory=dict)


@dataclass
class CompositeScore:
    """Full scoring breakdown for one candidate."""
    candidate_id:       str
    composite:          float   = 0.0

    # Dimension scores (each 0.0 to 1.0)
    calmar_score:       float   = 0.0
    pf_score:           float   = 0.0
    wl_score:           float   = 0.0
    regime_score:       float   = 0.0
    stability_score:    float   = 0.0
    robustness_score:   float   = 0.0

    # Weighted contributions
    calmar_contrib:     float   = 0.0
    pf_contrib:         float   = 0.0
    wl_contrib:         float   = 0.0
    regime_contrib:     float   = 0.0
    stability_contrib:  float   = 0.0
    robustness_contrib: float   = 0.0

    # Modifier
    sensitivity_status: str     = 'MODERATE'
    sensitivity_mod:    float   = 0.0

    # Gate result
    gate_result:        Optional[GateResult] = None


# ── HARD GATE FILTER ───────────────────────────────────

def run_monte_carlo_stress_test(
    trades: List[Dict],
    symbol: str = 'EURUSD',
    mandate: Optional[object] = None,
    initial_equity: float = 10000.0,
    n_permutations: int = 500,
    max_slippage_pips: float = 1.5,
    seed: int = 42
) -> Tuple[float, Optional[str]]:
    """
    Executes a vectorized Monte Carlo trade sequence permutation & slippage stress test.
    Applies uniform additive slippage penalty s_i ~ Uniform(0.0, max_slippage_pips) in pips per trade.
    Reuses evaluate_backtest_mandate() for every synthetic equity curve.

    Returns:
        (breach_rate_pct: float, worst_breach_reason: Optional[str])
    """
    if not trades:
        return 100.0, "Zero out-of-sample trades available for Monte Carlo stress test"

    from shared.instrument_spec import get_spec
    from trainer.core.mandate_evaluator import evaluate_backtest_mandate

    spec = get_spec(symbol)
    if spec is None:
        raise RuntimeError(
            f"INSTRUMENT_SPEC_MISSING: Cannot run Monte Carlo stress test for {symbol} — "
            "no cached InstrumentSpec. Run capture_specs.py with MT5 connected."
        )
    # Use the canonical pip_value property — never recompute tick_value*(pip_size/tick_size)
    pip_value_per_lot = spec.pip_value

    rng = np.random.default_rng(seed)
    breach_count = 0
    worst_reason = None
    trade_count = len(trades)

    for k in range(n_permutations):
        # 1. Random shuffle permutation index
        perm_indices = rng.permutation(trade_count)

        # 2. Sample uniform slippage penalty in pips [0.0, max_slippage_pips]
        slippage_pips = rng.uniform(0.0, max_slippage_pips, size=trade_count)

        # 3. Construct synthetic equity curve.
        # Anchor point at initial_equity ensures overall drawdown from the starting
        # balance is correctly detected even if the first trade is a loss.
        synthetic_curve = [{'timestamp': None, 'equity': initial_equity}]
        current_eq = initial_equity

        for i_idx, orig_idx in enumerate(perm_indices):
            tr = trades[orig_idx]
            lots = float(tr.get('lots', 1.0))
            slippage_cost = float(slippage_pips[i_idx] * pip_value_per_lot * lots)
            adjusted_pnl = float(tr.get('pnl', 0.0)) - slippage_cost

            current_eq += adjusted_pnl
            exit_time = tr.get('exit_time')
            synthetic_curve.append({
                'timestamp': exit_time,
                'equity': current_eq
            })

        # 4. Evaluate mandate against synthetic equity curve using single-source-of-truth evaluator
        is_compliant, reason = evaluate_backtest_mandate(synthetic_curve, mandate=mandate)
        if not is_compliant:
            breach_count += 1
            if worst_reason is None:
                worst_reason = reason

    breach_rate = (breach_count / n_permutations) * 100.0
    return breach_rate, worst_reason


def check_hard_gates(wf_summary: Dict,
                     candidate_id: str = 'unknown',
                     mandate: Optional[object] = None,
                     symbol: str = 'EURUSD',
                     initial_equity: float = 10000.0) -> GateResult:
    """
    Run all 10 hard gates against a walk-forward summary.

    Args:
        wf_summary: aggregated walk-forward metrics dict
        candidate_id: identifier for logging
        mandate: prop-firm mandate configuration
        symbol: trading instrument symbol
        initial_equity: starting equity

    Returns:
        GateResult with passed=True if all 10 gates pass
    """
    details = {}

    # Gate 1: Minimum total trades
    total_trades = wf_summary.get('total_trades', 0)
    details['gate_1_total_trades'] = total_trades
    if total_trades < GATES['min_total_trades']:
        return GateResult(
            candidate_id=candidate_id,
            passed=False,
            failed_gate='gate_1_total_trades',
            failure_reason=(
                f"Total trades {total_trades} < "
                f"minimum {GATES['min_total_trades']}"
            ),
            gate_details=details
        )

    # Gate 2: Minimum trades per window
    median_trades = wf_summary.get('median_trades_per_window', 0)
    details['gate_2_trades_per_window'] = median_trades
    if median_trades < GATES['min_trades_per_window']:
        return GateResult(
            candidate_id=candidate_id,
            passed=False,
            failed_gate='gate_2_trades_per_window',
            failure_reason=(
                f"Median trades/window {median_trades:.1f} < "
                f"minimum {GATES['min_trades_per_window']}"
            ),
            gate_details=details
        )

    # Gate 3: Minimum valid windows
    valid_windows = wf_summary.get('valid_windows', 0)
    details['gate_3_valid_windows'] = valid_windows
    if valid_windows < GATES['min_valid_windows']:
        return GateResult(
            candidate_id=candidate_id,
            passed=False,
            failed_gate='gate_3_valid_windows',
            failure_reason=(
                f"Valid windows {valid_windows} < "
                f"minimum {GATES['min_valid_windows']}"
            ),
            gate_details=details
        )

    # Gate 4: Maximum drawdown ceiling
    median_dd = wf_summary.get('median_max_drawdown_pct', 100.0)
    details['gate_4_max_drawdown'] = median_dd
    if median_dd > GATES['max_median_drawdown_pct']:
        return GateResult(
            candidate_id=candidate_id,
            passed=False,
            failed_gate='gate_4_max_drawdown',
            failure_reason=(
                f"Median drawdown {median_dd:.1f}% > "
                f"ceiling {GATES['max_median_drawdown_pct']}%"
            ),
            gate_details=details
        )

    # Gate 5: Minimum profitability
    median_pf = wf_summary.get('median_profit_factor', 0.0)
    details['gate_5_profit_factor'] = median_pf
    if median_pf < GATES['min_median_profit_factor']:
        return GateResult(
            candidate_id=candidate_id,
            passed=False,
            failed_gate='gate_5_profit_factor',
            failure_reason=(
                f"Median PF {median_pf:.3f} < "
                f"minimum {GATES['min_median_profit_factor']}"
            ),
            gate_details=details
        )

    # Gate 6: Profitable window floor
    profitable_rate = wf_summary.get('profitable_window_rate', 0.0)
    details['gate_6_profitable_windows'] = profitable_rate
    if profitable_rate < GATES['min_profitable_window_rate']:
        return GateResult(
            candidate_id=candidate_id,
            passed=False,
            failed_gate='gate_6_profitable_windows',
            failure_reason=(
                f"Profitable window rate "
                f"{profitable_rate*100:.1f}% < "
                f"minimum "
                f"{GATES['min_profitable_window_rate']*100:.0f}%"
            ),
            gate_details=details
        )

    # Gate 7: Consecutive loss limit
    max_consec = wf_summary.get('max_consecutive_losses', 0)
    details['gate_7_consecutive_losses'] = max_consec
    if max_consec > GATES['max_consecutive_losses']:
        return GateResult(
            candidate_id=candidate_id,
            passed=False,
            failed_gate='gate_7_consecutive_losses',
            failure_reason=(
                f"Max consecutive losses {max_consec} > "
                f"limit {GATES['max_consecutive_losses']}"
            ),
            gate_details=details
        )

    # Gate 8: Regime coverage
    # This is a soft gate — produces FLAG not REJECT
    regime_windows = wf_summary.get('regime_windows_min', 0)
    details['gate_8_regime_coverage'] = regime_windows
    # Not a hard failure — flagged in composite score instead

    # Gate 9: Prop-Firm Mandate Compliance (Fix 11.2)
    mandate_compliant = wf_summary.get('mandate_compliant', True)
    breach_reason = wf_summary.get('mandate_breach_reason', 'Mandate compliance failure')
    details['gate_9_mandate_compliant'] = mandate_compliant
    if not mandate_compliant:
        return GateResult(
            candidate_id=candidate_id,
            passed=False,
            failed_gate='gate_9_mandate_compliance',
            failure_reason=f"Mandate compliance breach: {breach_reason}",
            gate_details=details
        )

    # Gate 10: Monte Carlo / Slippage Stress Gate (WAVE 14)
    # all_oos_trades is collected by aggregate_wf_results() from real WindowResult.oos_trades_list
    # data on every production run. When the key is absent (legacy or malformed summary dict) or
    # the list is empty, the gate FAILS LOUDLY — no fabricated trades, ever.
    all_oos_trades = wf_summary.get('all_oos_trades')
    if all_oos_trades is None:
        return GateResult(
            candidate_id=candidate_id,
            passed=False,
            failed_gate='gate_10_monte_carlo_stress',
            failure_reason=(
                "GATE_10_DATA_MISSING: 'all_oos_trades' key absent from wf_summary. "
                "This indicates the summary dict was not produced by aggregate_wf_results() "
                "or WindowResult.oos_trades_list was not populated. Failing closed."
            ),
            gate_details=details
        )
    if len(all_oos_trades) == 0:
        return GateResult(
            candidate_id=candidate_id,
            passed=False,
            failed_gate='gate_10_monte_carlo_stress',
            failure_reason="GATE_10_ZERO_TRADES: Zero out-of-sample trades available for Monte Carlo evaluation.",
            gate_details=details
        )

    breach_rate, mc_reason = run_monte_carlo_stress_test(
        trades=all_oos_trades,
        symbol=symbol,
        mandate=mandate,
        initial_equity=initial_equity,
        n_permutations=500,
        max_slippage_pips=1.5
    )
    details['gate_10_monte_carlo_stress'] = breach_rate

    if breach_rate > GATES['max_monte_carlo_breach_rate']:
        return GateResult(
            candidate_id=candidate_id,
            passed=False,
            failed_gate='gate_10_monte_carlo_stress',
            failure_reason=(
                f"Monte Carlo stress breach: {breach_rate:.1f}% of permutations breached mandate "
                f"(ceiling {GATES['max_monte_carlo_breach_rate']}%) | Detail: {mc_reason}"
            ),
            gate_details=details
        )

    # All 10 hard gates passed
    return GateResult(
        candidate_id=candidate_id,
        passed=True,
        gate_details=details
    )


# ── COMPOSITE SCORING ──────────────────────────────────

def calculate_composite_score(wf_summary: Dict,
                               candidate_id: str = 'unknown',
                               tier2_robustness: float = None,
                               sensitivity_status: str = 'MODERATE'
                               ) -> CompositeScore:
    """
    Calculate composite score for a candidate that passed
    all hard gates.

    Args:
        wf_summary:          aggregated walk-forward metrics
        candidate_id:        identifier for this candidate
        tier2_robustness:    % of Tier 2 sources where PF > 1.0
                             (None = no Tier 2 data available)
        sensitivity_status:  ROBUST / MODERATE / FRAGILE

    Returns:
        CompositeScore with all dimension breakdowns
    """
    result = CompositeScore(candidate_id=candidate_id)

    # ── Dimension 1: Calmar (40%) ──────────────────────
    calmar = wf_summary.get('median_calmar_ratio', 0.0)
    calmar = max(calmar, 0.0)          # floor at 0
    calmar_norm = min(calmar, 3.0) / 3.0
    result.calmar_score   = calmar_norm
    result.calmar_contrib = calmar_norm * WEIGHTS['calmar']

    # ── Dimension 2: Profit Factor (25%) ───────────────
    pf = wf_summary.get('median_profit_factor', 0.0)
    pf = max(pf, 0.0)
    pf_norm = min(pf, 3.0) / 3.0
    result.pf_score   = pf_norm
    result.pf_contrib = pf_norm * WEIGHTS['profit_factor']

    # ── Dimension 3: Win/Loss Ratio (15%) ──────────────
    wl = wf_summary.get('median_avg_win_loss', 0.0)
    wl = max(wl, 0.0)
    wl_norm = min(wl, 3.0) / 3.0
    result.wl_score   = wl_norm
    result.wl_contrib = wl_norm * WEIGHTS['win_loss_ratio']

    # ── Dimension 4: Regime Consistency (10%) ──────────
    regime = wf_summary.get('regime_consistency', 0.0)
    regime = max(min(regime, 1.0), 0.0)
    result.regime_score   = regime
    result.regime_contrib = regime * WEIGHTS['regime']

    # ── Dimension 5: Temporal Stability (5%) ───────────
    stable      = wf_summary.get('temporal_stable', False)
    trend_pct   = wf_summary.get('temporal_trend_pct', 0.0)

    if stable and trend_pct > -10:
        stability = 1.0
    elif stable and trend_pct <= -10:
        stability = 0.7
    elif not stable and trend_pct > 0:
        stability = 0.6
    elif not stable and trend_pct <= -20:
        stability = 0.2
    else:
        stability = 0.4

    result.stability_score   = stability
    result.stability_contrib = stability * WEIGHTS['stability']

    # ── Dimension 6: Robustness (5%) ───────────────────
    if tier2_robustness is None:
        robustness = 0.5    # neutral — no Tier 2 data
    elif tier2_robustness >= 0.70:
        robustness = 1.0
    elif tier2_robustness >= 0.40:
        robustness = 0.6
    else:
        robustness = 0.2

    result.robustness_score   = robustness
    result.robustness_contrib = robustness * WEIGHTS['robustness']

    # ── Raw composite ──────────────────────────────────
    raw = (result.calmar_contrib   +
           result.pf_contrib       +
           result.wl_contrib       +
           result.regime_contrib   +
           result.stability_contrib+
           result.robustness_contrib)

    # ── Sensitivity modifier ───────────────────────────
    result.sensitivity_status = sensitivity_status
    if sensitivity_status == 'FRAGILE':
        result.sensitivity_mod = -FRAGILE_PENALTY * raw
    elif sensitivity_status == 'ROBUST':
        result.sensitivity_mod = ROBUST_BONUS * raw
    else:
        result.sensitivity_mod = 0.0

    # ── Final score ────────────────────────────────────
    final = raw + result.sensitivity_mod
    result.composite = round(min(max(final, 0.0), 1.0), 4)

    return result


def calculate_quick_score(metrics: Dict, min_trades: int = 10) -> float:
    """
    Calculate a fast, simplified score from a single backtest run.
    Used for rapid candidate evaluation in adaptation stages.
    """
    if metrics.get('total_trades', 0) < min_trades:
        return 0.0
    if metrics.get('profit_factor', 0.0) <= 1.0:
        return 0.0

    calmar = max(min(metrics.get('calmar_ratio', 0.0), 3.0), 0) / 3.0
    pf     = max(min(metrics.get('profit_factor', 0.0), 3.0), 0) / 3.0
    
    # 50% Calmar, 50% Profit Factor for rapid adaptation filtering
    return calmar * 0.50 + pf * 0.50


# ── AGGREGATE WALK-FORWARD RESULTS ─────────────────────

def aggregate_wf_results(window_results: list) -> Dict:
    """
    Aggregate a list of WindowResult objects into a
    summary dict for gate checking and scoring.

    Args:
        window_results: list of WindowResult from walk_forward.py

    Returns:
        dict of aggregated metrics
    """
    valid = [r for r in window_results if r.status == 'OK']

    if not valid:
        return {
            'valid_windows':             0,
            'total_trades':              0,
            'median_trades_per_window':  0,
            'median_profit_factor':      0.0,
            'median_calmar_ratio':       0.0,
            'median_avg_win_loss':       0.0,
            'median_max_drawdown_pct':   100.0,
            'profitable_window_rate':    0.0,
            'max_consecutive_losses':    99,
            'regime_consistency':        0.0,
            'temporal_stable':           False,
            'temporal_trend_pct':        -100.0,
        }

    # Trade counts
    total_trades  = sum(r.oos_trades for r in valid)
    trades_list   = [r.oos_trades for r in valid]

    # P&L metrics
    pf_list       = [r.oos_profit_factor for r in valid
                     if r.oos_profit_factor > 0]
    calmar_list   = [r.oos_calmar for r in valid]
    dd_list       = [r.oos_max_drawdown for r in valid]
    net_list      = [r.oos_net_profit for r in valid]

    # Win/loss ratio — approximate from available data
    wl_list       = []
    for r in valid:
        if r.oos_win_rate > 0 and r.oos_win_rate < 1:
            # Approximate avg win/loss from PF and win rate
            # PF = (avg_win * wins) / (avg_loss * losses)
            # wl_ratio = PF * (1 - win_rate) / win_rate
            if r.oos_profit_factor > 0:
                wl = (r.oos_profit_factor *
                      (1 - r.oos_win_rate) /
                      r.oos_win_rate)
                wl_list.append(wl)

    # Profitability
    profitable        = [r for r in valid if r.oos_net_profit > 0]
    profitable_rate   = len(profitable) / len(valid)

    # Consecutive losses — take worst across windows
    max_consec = max(
        (getattr(r, 'oos_max_consec_losses', 0) for r in valid),
        default=0
    )

    # Temporal stability
    n = len(valid)
    if n >= 3:
        third       = n // 3
        early       = [r.oos_composite_score for r in valid[:third]]
        recent      = [r.oos_composite_score for r in valid[-third:]]
        early_avg   = sum(early) / len(early) if early else 0
        recent_avg  = sum(recent) / len(recent) if recent else 0
        trend_pct   = ((recent_avg - early_avg) /
                       early_avg * 100
                       if early_avg > 0 else 0)
        stable      = abs(trend_pct) < 30
    else:
        trend_pct   = 0.0
        stable      = True

    # Regime consistency
    # Count windows where OOS PF > 1.0 vs total valid
    regime_consistency = profitable_rate  # simplified for M2/M3

    def median(lst):
        if not lst:
            return 0.0
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n//2-1] + s[n//2]) / 2

    mandate_compliant = True
    mandate_breach_reason = None
    for r in valid:
        if getattr(r, 'oos_mandate_compliant', True) is False:
            mandate_compliant = False
            mandate_breach_reason = getattr(r, 'oos_mandate_breach_reason', 'Window mandate breach')
            break

    all_oos_trades = []
    for r in valid:
        all_oos_trades.extend(getattr(r, 'oos_trades_list', []))

    win_rate_list = [r.oos_win_rate for r in valid]

    return {
        'valid_windows':             len(valid),
        'total_trades':              total_trades,
        'median_trades_per_window':  median(trades_list),
        'median_profit_factor':      median(pf_list),
        'median_win_rate':          median(win_rate_list),
        'median_calmar_ratio':       median(calmar_list),
        'median_avg_win_loss':       median(wl_list),
        'median_max_drawdown_pct':   median(dd_list),
        'profitable_window_rate':    profitable_rate,
        'max_consecutive_losses':    max_consec,
        'regime_consistency':        regime_consistency,
        'temporal_stable':           stable,
        'temporal_trend_pct':        trend_pct,
        'total_oos_pnl':             sum(net_list),
        'avg_oos_pnl_per_window':    sum(net_list) / len(valid),
        'mandate_compliant':         mandate_compliant,
        'mandate_breach_reason':     mandate_breach_reason,
        'all_oos_trades':            all_oos_trades,
    }


# ── RANK CANDIDATES ────────────────────────────────────

def rank_candidates(scored: List[CompositeScore]
                    ) -> List[CompositeScore]:
    """Sort candidates by composite score descending."""
    return sorted(scored,
                  key=lambda x: x.composite,
                  reverse=True)


# ── PRINT SCORING REPORT ───────────────────────────────

def print_scoring_report(gate_result: GateResult,
                          score: Optional[CompositeScore],
                          wf_summary: Dict) -> None:
    """Print a full scoring report for one candidate."""
    print(f"\n{'='*60}")
    print(f"  SCORING REPORT — {gate_result.candidate_id}")
    print(f"{'='*60}")

    # Gate results
    print(f"\n  HARD GATES:")
    for gate, value in gate_result.gate_details.items():
        print(f"    {gate:<35} {value}")

    if not gate_result.passed:
        print(f"\n  ✗ FAILED: {gate_result.failure_reason}")
        print(f"  Decision: REJECTED")
        print(f"{'='*60}")
        return

    print(f"\n  ✓ All hard gates passed")

    if score is None:
        print(f"  No score computed.")
        return

    # Score breakdown
    print(f"\n  COMPOSITE SCORE BREAKDOWN:")
    print(f"    {'Dimension':<25} {'Raw':>6} "
          f"{'Weight':>7} {'Contrib':>8}")
    print(f"    {'-'*50}")
    print(f"    {'Calmar (risk-adj return)':<25} "
          f"{score.calmar_score:>6.3f} "
          f"{WEIGHTS['calmar']:>7.0%} "
          f"{score.calmar_contrib:>8.4f}")
    print(f"    {'Profit Factor':<25} "
          f"{score.pf_score:>6.3f} "
          f"{WEIGHTS['profit_factor']:>7.0%} "
          f"{score.pf_contrib:>8.4f}")
    print(f"    {'Win/Loss Ratio':<25} "
          f"{score.wl_score:>6.3f} "
          f"{WEIGHTS['win_loss_ratio']:>7.0%} "
          f"{score.wl_contrib:>8.4f}")
    print(f"    {'Regime Consistency':<25} "
          f"{score.regime_score:>6.3f} "
          f"{WEIGHTS['regime']:>7.0%} "
          f"{score.regime_contrib:>8.4f}")
    print(f"    {'Temporal Stability':<25} "
          f"{score.stability_score:>6.3f} "
          f"{WEIGHTS['stability']:>7.0%} "
          f"{score.stability_contrib:>8.4f}")
    print(f"    {'Robustness (Tier 2)':<25} "
          f"{score.robustness_score:>6.3f} "
          f"{WEIGHTS['robustness']:>7.0%} "
          f"{score.robustness_contrib:>8.4f}")
    print(f"    {'-'*50}")

    if score.sensitivity_mod != 0:
        mod_label = (f"Sensitivity ({score.sensitivity_status})")
        print(f"    {mod_label:<25} "
              f"{'':>6} {'':>7} "
              f"{score.sensitivity_mod:>8.4f}")

    print(f"\n  COMPOSITE SCORE:  {score.composite:.4f}")

    # Interpretation
    if score.composite >= 0.70:
        interp = "EXCELLENT — strong candidate for promotion"
    elif score.composite >= 0.55:
        interp = "GOOD — meets promotion threshold"
    elif score.composite >= 0.40:
        interp = "MARGINAL — borderline, needs review"
    else:
        interp = "WEAK — likely to underperform live"

    print(f"  Interpretation:   {interp}")
    print(f"{'='*60}\n")