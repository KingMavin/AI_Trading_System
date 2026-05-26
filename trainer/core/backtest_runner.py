"""
Run a complete backtest and display results.
This is the entry point for Milestone 1 testing.

Usage:
  python trainer/core/backtest_runner.py
  python trainer/core/backtest_runner.py --symbol GBPUSD
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import argparse
from datetime import datetime

from trainer.core.data_loader import load_ohlcv
from trainer.core.backtester import Backtester
from shared.metrics import calculate_metrics, print_metrics
import trainer.signals.ma_crossover as ma_crossover


DEFAULT_PARAMS = {
    'fast_ma_period':          20,    # was 10
    'slow_ma_period':          100,   # was 50
    'ma_type':                 'SMA',
    'sl_atr_multiple':         2.0,   # was 1.5
    'tp_rr_ratio':             2.0,
    'risk_per_trade_pct':      1.0,
    'adx_min_threshold':       20,    # was 0 — only trade in trends
    'exit_on_opposite_crossover': False,
    'pip_size':                0.0001,
    'warmup_candles':          250,
}


def run_backtest(symbol:  str = 'EURUSD',
                 start:   str = '2020-01-01',
                 end:     str = '2023-12-31',
                 params:  dict = None,
                 initial_equity: float = 10000.0) -> dict:
    """
    Run a complete backtest.

    Args:
        symbol:         trading instrument
        start:          start date string 'YYYY-MM-DD'
        end:            end date string 'YYYY-MM-DD'
        params:         strategy parameters (uses defaults if None)
        initial_equity: starting account balance

    Returns:
        result dict with trades, equity_curve, metrics
    """
    params = params or DEFAULT_PARAMS.copy()
    params['pip_size'] = (0.01 if symbol == 'USDJPY'
                          else 0.0001)

    print(f"\nLoading {symbol} M15 data ({start} → {end})...")
    df = load_ohlcv(
        symbol=symbol,
        timeframe='M15',
        start=datetime.strptime(start, '%Y-%m-%d'),
        end=datetime.strptime(end, '%Y-%m-%d'),
        warmup_candles=params['warmup_candles']
    )
    print(f"Loaded {len(df):,} candles")

    print(f"Running MA Crossover backtest...")
    print(f"  fast_ma={params['fast_ma_period']} "
          f"slow_ma={params['slow_ma_period']} "
          f"sl={params['sl_atr_multiple']}xATR "
          f"tp={params['tp_rr_ratio']}R")

    bt = Backtester(
        symbol=symbol,
        params=params,
        initial_equity=initial_equity,
        random_seed=42
    )

    result = bt.run(df, signal_module=ma_crossover)

    # Calculate metrics
    metrics = calculate_metrics(
        trades=result['trades'],
        equity_curve=result['equity_curve'],
        initial_equity=initial_equity
    )
    result['metrics'] = metrics

    # Display results
    print_metrics(
        metrics,
        title=f"MA Crossover — {symbol} M15 {start}:{end}"
    )

    # Show sample trades
    trades = result['trades']
    if trades:
        print(f"Sample of last 5 trades:")
        print(f"{'Time':<25} {'Dir':<6} {'Entry':>8} "
              f"{'Exit':>8} {'PnL':>8} {'Reason':<20}")
        print("-" * 80)
        for t in trades[-5:]:
            print(
                f"{str(t['entry_time'])[:19]:<25} "
                f"{t['direction']:<6} "
                f"{t['entry_price']:>8.5f} "
                f"{t['exit_price']:>8.5f} "
                f"{t['net_pnl']:>8.2f} "
                f"{t['close_reason']:<20}"
            )

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',
                        default='EURUSD',
                        choices=['EURUSD', 'GBPUSD', 'USDJPY'])
    parser.add_argument('--start',  default='2020-01-01')
    parser.add_argument('--end',    default='2023-12-31')
    parser.add_argument('--equity', default=10000.0, type=float)
    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_equity=args.equity
    )