"""
Walk-Forward Runner — entry point for Milestone 2.
Loads data, runs walk-forward validation, prints full report.

Usage:
  python trainer/core/wf_runner.py
  python trainer/core/wf_runner.py --symbol GBPUSD
  python trainer/core/wf_runner.py --opt_months 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from datetime import datetime
import pandas as pd

from trainer.core.data_loader import load_ohlcv
from trainer.core.walk_forward import WalkForwardValidator
from trainer.core.parameter_grid import get_grid_size
import trainer.signals.ma_crossover as ma_crossover


def print_wf_report(results, symbol: str,
                    opt_months: int, test_months: int) -> None:
    """Print full walk-forward report to console."""

    valid = [r for r in results if r.status == 'OK']
    total = len(results)

    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD REPORT — {symbol} M15")
    print(f"  Opt window: {opt_months}m | "
          f"Test window: {test_months}m | "
          f"Windows: {total} ({len(valid)} valid)")
    print(f"{'='*65}")

    if not valid:
        print("  No valid windows found.")
        return

    # ── Window-by-window table ─────────────────────────
    print(f"\n{'Win':>4} {'Test Period':<22} "
          f"{'Params':^20} "
          f"{'IS':>6} {'IS PF':>6} "
          f"{'OOS':>5} {'OOS PF':>7} "
          f"{'OOS Net':>9} {'Result'}")
    print("-" * 95)

    for r in valid:
        params_str = (
            f"f{r.best_params.get('fast_ma_period','?')} "
            f"s{r.best_params.get('slow_ma_period','?')} "
            f"sl{r.best_params.get('sl_atr_multiple','?')}"
        )
        result_flag = (
            "✓ PROFIT" if r.oos_net_profit > 0
            else "✗ LOSS"
        )
        print(
            f"{r.window_num:>4} "
            f"{str(r.test_start.date()):<12} "
            f"{str(r.test_end.date()):<10} "
            f"{params_str:^20} "
            f"{r.is_trades:>6} "
            f"{r.is_profit_factor:>6.2f} "
            f"{r.oos_trades:>5} "
            f"{r.oos_profit_factor:>7.3f} "
            f"${r.oos_net_profit:>8.2f} "
            f"{result_flag}"
        )

    # ── Skipped windows ────────────────────────────────
    skipped = [r for r in results if r.status != 'OK']
    if skipped:
        print(f"\n  Skipped windows: {len(skipped)}")
        for r in skipped:
            print(f"    Window {r.window_num}: {r.status}")

    # ── Aggregate statistics ───────────────────────────
    profitable = [r for r in valid if r.oos_net_profit > 0]
    pf_values  = [r.oos_profit_factor for r in valid
                  if r.oos_profit_factor > 0]
    net_values = [r.oos_net_profit for r in valid]

    profitable_rate = len(profitable) / len(valid) * 100
    median_pf       = sorted(pf_values)[len(pf_values)//2] \
                      if pf_values else 0
    total_oos_pnl   = sum(net_values)
    avg_oos_pnl     = total_oos_pnl / len(valid)

    # IS vs OOS comparison
    is_scores  = [r.is_composite_score for r in valid]
    oos_scores = [r.oos_composite_score for r in valid]
    avg_is     = sum(is_scores) / len(is_scores)
    avg_oos    = sum(oos_scores) / len(oos_scores) if oos_scores \
                 else 0

    print(f"\n{'='*65}")
    print(f"  AGGREGATE RESULTS ({len(valid)} valid windows)")
    print(f"{'='*65}")
    print(f"  Profitable windows:    "
          f"{len(profitable)}/{len(valid)} "
          f"({profitable_rate:.1f}%)")
    print(f"  Median OOS PF:         {median_pf:.3f}")
    print(f"  Total OOS P&L:        ${total_oos_pnl:,.2f}")
    print(f"  Avg OOS P&L/window:   ${avg_oos_pnl:,.2f}")
    print(f"  Avg IS score:          {avg_is:.3f}")
    print(f"  Avg OOS score:         {avg_oos:.3f}")
    print(f"  IS→OOS degradation:   "
          f"{((avg_oos - avg_is) / avg_is * 100):.1f}% "
          if avg_is > 0 else "  IS→OOS degradation:   N/A")

    # ── Verdict ────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}")

    if profitable_rate >= 60 and median_pf >= 1.1:
        verdict = "PROMISING — strategy shows genuine edge"
        detail  = ("More than 60% of test windows profitable "
                   "with median PF >= 1.1.")
    elif profitable_rate >= 50 and median_pf >= 1.0:
        verdict = "MARGINAL — edge exists but weak"
        detail  = ("Strategy profitable in majority of windows "
                   "but profit factor is low.")
    elif profitable_rate >= 40:
        verdict = "WEAK — inconsistent edge"
        detail  = ("Strategy profitable in less than half of "
                   "test windows. High risk.")
    else:
        verdict = "NO EDGE — strategy fails walk-forward"
        detail  = ("Strategy profitable in fewer than 40% of "
                   "test windows. Do not promote.")

    print(f"  {verdict}")
    print(f"  {detail}")
    print(f"{'='*65}\n")


def run_walk_forward(symbol:      str   = 'EURUSD',
                     start:       str   = '2015-01-01',
                     end:         str   = '2023-12-31',
                     opt_months:  int   = 6,
                     test_months: int   = 1) -> list:
    """Run full walk-forward validation."""

    grid_size = get_grid_size('ma_crossover')
    print(f"\nWalk-Forward Validator")
    print(f"  Symbol:       {symbol} M15")
    print(f"  Period:       {start} → {end}")
    print(f"  Opt window:   {opt_months} months")
    print(f"  Test window:  {test_months} months")
    print(f"  Grid size:    {grid_size} combinations per window")
    print(f"\nLoading data...")

    df = load_ohlcv(
        symbol=symbol,
        timeframe='M15',
        start=datetime.strptime(start, '%Y-%m-%d'),
        end=datetime.strptime(end, '%Y-%m-%d'),
        warmup_candles=300
    )
    print(f"Loaded {len(df):,} candles "
          f"({df.index.min().date()} → {df.index.max().date()})")

    wf = WalkForwardValidator(
        symbol=symbol,
        df=df,
        template='ma_crossover',
        signal_module=ma_crossover,
        opt_months=opt_months,
        test_months=test_months,
        initial_equity=10000.0,
        min_trades_opt=30,
        min_trades_test=10,
    )

    print(f"\nStarting walk-forward validation...")
    print(f"This will take a few minutes...\n")

    results = wf.run()
    print_wf_report(results, symbol, opt_months, test_months)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',
                        default='EURUSD',
                        choices=['EURUSD', 'GBPUSD', 'USDJPY'])
    parser.add_argument('--start',       default='2015-01-01')
    parser.add_argument('--end',         default='2023-12-31')
    parser.add_argument('--opt_months',  default=6,  type=int)
    parser.add_argument('--test_months', default=1,  type=int)
    args = parser.parse_args()

    run_walk_forward(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        opt_months=args.opt_months,
        test_months=args.test_months,
    )