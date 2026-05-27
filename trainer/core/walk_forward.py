"""
Walk-Forward Validator.

Rolls a training window + test window across historical data.
For each window:
  1. Find best parameters on training data (in-sample)
  2. Test those parameters on unseen data (out-of-sample)
  3. Record both results

This is the honest test of whether a strategy has real edge.
A strategy that scores well in-sample but poorly out-of-sample
is curve-fitted and will fail in live trading.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging

from trainer.core.backtester import Backtester
from trainer.core.parameter_grid import get_grid
from shared.metrics import calculate_metrics
import trainer.signals.ma_crossover as ma_crossover

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)


# ── WINDOW RESULT ──────────────────────────────────────

@dataclass
class WindowResult:
    """Results for one walk-forward window."""
    window_num:          int
    opt_start:           datetime
    opt_end:             datetime
    test_start:          datetime
    test_end:            datetime

    # Best params found in-sample
    best_params:         Dict = field(default_factory=dict)

    # In-sample performance
    is_trades:           int   = 0
    is_profit_factor:    float = 0.0
    is_calmar:           float = 0.0
    is_net_profit:       float = 0.0
    is_composite_score:  float = 0.0

    # Out-of-sample performance
    oos_trades:          int   = 0
    oos_profit_factor:   float = 0.0
    oos_calmar:          float = 0.0
    oos_net_profit:      float = 0.0
    oos_win_rate:        float = 0.0
    oos_max_drawdown:    float = 0.0
    oos_composite_score: float = 0.0

    # Status
    status:              str   = 'OK'
    # OK / INSUFFICIENT_DATA / INSUFFICIENT_TRADES


def composite_score(metrics: Dict) -> float:
    """
    Score a backtest result.
    Used to rank parameter combinations during optimisation.

    Formula (simplified for Milestone 2):
      40% Calmar ratio (risk-adjusted return)
      35% Profit Factor (profitability quality)
      25% Win rate (consistency)

    Returns 0.0 if insufficient trades or losing strategy.
    """
    if metrics['total_trades'] < 20:
        return 0.0
    if metrics['profit_factor'] <= 1.0:
        return 0.0

    calmar_norm = min(metrics['calmar_ratio'], 3.0) / 3.0
    pf_norm     = min(metrics['profit_factor'], 3.0) / 3.0
    wr_norm     = metrics['win_rate']

    # Clamp negatives
    calmar_norm = max(calmar_norm, 0.0)
    pf_norm     = max(pf_norm, 0.0)

    return (calmar_norm * 0.40 +
            pf_norm    * 0.35 +
            wr_norm    * 0.25)


# ── WALK-FORWARD ENGINE ────────────────────────────────

class WalkForwardValidator:
    """
    Runs walk-forward validation on historical data.

    Parameters:
        symbol:           trading instrument
        df:               full OHLCV DataFrame
        template:         strategy template name
        signal_module:    module with generate_signal + check_exit
        opt_months:       optimisation window length in months
        test_months:      test window length in months
        initial_equity:   starting account equity
        min_trades_opt:   minimum trades required in opt window
        min_trades_test:  minimum trades required in test window
    """

    def __init__(self,
                 symbol:        str,
                 df:            pd.DataFrame,
                 template:      str         = 'ma_crossover',
                 signal_module              = ma_crossover,
                 opt_months:    int         = 6,
                 test_months:   int         = 1,
                 initial_equity:float       = 10000.0,
                 min_trades_opt:int         = 30,
                 min_trades_test:int        = 10):

        self.symbol         = symbol
        self.df             = df
        self.template       = template
        self.signal_module  = signal_module
        self.opt_months     = opt_months
        self.test_months    = test_months
        self.initial_equity = initial_equity
        self.min_trades_opt = min_trades_opt
        self.min_trades_test= min_trades_test

        # All parameter combinations to test
        self.param_grid     = get_grid(template)
        log.info(
            f"Walk-forward initialised: {symbol} | "
            f"template={template} | "
            f"grid={len(self.param_grid)} combinations | "
            f"opt={opt_months}m test={test_months}m"
        )

    def _build_windows(self) -> List[Dict]:
        """Build the sequence of opt+test windows."""
        windows  = []
        idx      = self.df.index

        # Start after enough data for warmup
        start_dt = idx[0] + pd.DateOffset(months=self.opt_months)
        # End with enough room for one test window
        end_dt   = idx[-1] - pd.DateOffset(months=self.test_months)

        current  = start_dt
        win_num  = 1

        while current <= end_dt:
            opt_start  = current - pd.DateOffset(
                months=self.opt_months
            )
            opt_end    = current
            test_start = current
            test_end   = current + pd.DateOffset(
                months=self.test_months
            )

            # Clamp to available data
            if test_end > idx[-1]:
                test_end = idx[-1]

            windows.append({
                'window_num':  win_num,
                'opt_start':   opt_start,
                'opt_end':     opt_end,
                'test_start':  test_start,
                'test_end':    test_end,
            })

            current += pd.DateOffset(months=self.test_months)
            win_num += 1

        return windows

    def _slice(self, start, end, warmup: int = 250) -> pd.DataFrame:
        """
        Slice DataFrame for a window.
        Includes warmup candles before start for indicators.
        """
        # Handle timestamps that may or may not have timezone info
        ts_start = pd.Timestamp(start)
        ts_end   = pd.Timestamp(end)

        # Ensure UTC timezone
        if ts_start.tzinfo is None:
            ts_start = ts_start.tz_localize('UTC')
        else:
            ts_start = ts_start.tz_convert('UTC')

        if ts_end.tzinfo is None:
            ts_end = ts_end.tz_localize('UTC')
        else:
            ts_end = ts_end.tz_convert('UTC')

        # Find start index
        start_idx = self.df.index.searchsorted(ts_start)
        # Include warmup buffer before window
        warmup_idx = max(0, start_idx - warmup)

        return self.df.iloc[warmup_idx:
                            self.df.index.searchsorted(ts_end)]

    def _run_one(self, df_slice: pd.DataFrame,
                 params: Dict,
                 seed: int = 42) -> Dict:
        """Run a single backtest on a data slice."""
        bt = Backtester(
            symbol=self.symbol,
            params=params,
            initial_equity=self.initial_equity,
            random_seed=seed
        )
        result  = bt.run(df_slice, self.signal_module)
        metrics = calculate_metrics(
            trades=result['trades'],
            equity_curve=result['equity_curve'],
            initial_equity=self.initial_equity
        )
        return metrics

    def _optimise(self, df_opt: pd.DataFrame,
                  window_num: int) -> tuple:
        """
        Find the best parameter set on the optimisation window.
        Tests every combination in the parameter grid.
        Returns (best_params, best_score, best_metrics).
        """
        best_score   = -1.0
        best_params  = None
        best_metrics = None

        for i, params in enumerate(self.param_grid):
            params['pip_size'] = (
                0.01 if self.symbol == 'USDJPY' else 0.0001
            )
            metrics = self._run_one(
                df_opt, params,
                seed=window_num * 1000 + i
            )
            score = composite_score(metrics)

            if score > best_score:
                best_score   = score
                best_params  = params.copy()
                best_metrics = metrics

        return best_params, best_score, best_metrics

    def run(self) -> List[WindowResult]:
        """
        Run full walk-forward validation.
        Returns list of WindowResult — one per test window.
        """
        windows = self._build_windows()
        log.info(f"Running {len(windows)} walk-forward windows...")

        results = []

        for w in windows:
            wn = w['window_num']
            log.info(
                f"Window {wn}/{len(windows)}: "
                f"opt {w['opt_start'].date()} → "
                f"{w['opt_end'].date()} | "
                f"test {w['test_start'].date()} → "
                f"{w['test_end'].date()}"
            )

            result = WindowResult(
                window_num  = wn,
                opt_start   = w['opt_start'],
                opt_end     = w['opt_end'],
                test_start  = w['test_start'],
                test_end    = w['test_end'],
            )

            # Slice data
            df_opt  = self._slice(w['opt_start'], w['opt_end'])
            df_test = self._slice(w['test_start'], w['test_end'])

            # Check minimum candle counts
            if len(df_opt) < 1000 or len(df_test) < 100:
                result.status = 'INSUFFICIENT_DATA'
                log.warning(
                    f"Window {wn}: insufficient data "
                    f"(opt={len(df_opt)}, test={len(df_test)})"
                )
                results.append(result)
                continue

            # Step 1: Optimise on training window
            best_params, best_score, is_metrics = \
                self._optimise(df_opt, wn)

            if best_params is None or best_score == 0:
                result.status = 'INSUFFICIENT_TRADES'
                log.warning(
                    f"Window {wn}: no profitable "
                    f"parameter set found in-sample"
                )
                results.append(result)
                continue

            # Record in-sample results
            result.best_params        = best_params
            result.is_trades          = is_metrics['total_trades']
            result.is_profit_factor   = is_metrics['profit_factor']
            result.is_calmar          = is_metrics['calmar_ratio']
            result.is_net_profit      = is_metrics['net_profit']
            result.is_composite_score = best_score

            log.info(
                f"  IS best: fast={best_params['fast_ma_period']} "
                f"slow={best_params['slow_ma_period']} "
                f"sl={best_params['sl_atr_multiple']} "
                f"tp={best_params['tp_rr_ratio']} | "
                f"score={best_score:.3f} "
                f"trades={is_metrics['total_trades']}"
            )

            # Step 2: Test on unseen data
            best_params['pip_size'] = (
                0.01 if self.symbol == 'USDJPY' else 0.0001
            )
            oos_metrics = self._run_one(
                df_test, best_params,
                seed=wn * 9999
            )

            result.oos_trades          = oos_metrics['total_trades']
            result.oos_profit_factor   = oos_metrics['profit_factor']
            result.oos_calmar          = oos_metrics['calmar_ratio']
            result.oos_net_profit      = oos_metrics['net_profit']
            result.oos_win_rate        = oos_metrics['win_rate']
            result.oos_max_drawdown    = oos_metrics['max_drawdown_pct']
            result.oos_composite_score = composite_score(oos_metrics)

            log.info(
                f"  OOS: trades={oos_metrics['total_trades']} "
                f"PF={oos_metrics['profit_factor']:.3f} "
                f"net=${oos_metrics['net_profit']:.2f}"
            )

            results.append(result)

        return results