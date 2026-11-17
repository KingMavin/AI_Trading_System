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

import inspect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

from trainer.core.backtester import Backtester
from trainer.core.parameter_grid import get_grid, TEMPLATE_SPACES
from trainer.core.scoring import (
    calculate_quick_score,
    composite_score as _canonical_composite_score,
)
from shared.metrics import calculate_metrics
from shared.instrument_spec import get_spec
from shared.indicators import calculate_adx, calculate_atr, calculate_regime
import trainer.signals.ma_crossover as ma_crossover

composite_score = _canonical_composite_score


VALID_BASE_TEMPLATES = (
    'ma_crossover',
    'rsi_reversion',
    'donchian_breakout',
    'hybrid_confluence',
)


def parse_combo_template(template: str) -> Optional[Tuple[str, str]]:
    """
    Parse a combination template string into (entry_template, exit_template).
    Supports canonical 'combo_<entry>__<exit>' and legacy 'combo_<entry>_<exit>'.
    Returns None for non-combo strings; raises ValueError for invalid combo strings.
    """
    if not isinstance(template, str) or not template.startswith('combo_'):
        return None
    body = template[len('combo_'):]
    if '__' in body:
        entry_t, exit_t = body.split('__', 1)
        if entry_t in VALID_BASE_TEMPLATES and exit_t in VALID_BASE_TEMPLATES:
            return entry_t, exit_t
        raise ValueError(f"Invalid combination template '{template}': unknown base template(s).")
    for known in VALID_BASE_TEMPLATES:
        prefix = f"{known}_"
        if body.startswith(prefix):
            exit_t = body[len(prefix):]
            if exit_t in VALID_BASE_TEMPLATES:
                return known, exit_t
    raise ValueError(f"Invalid combination template '{template}': could not parse entry/exit templates.")


class ComboSignalAdapter:
    """
    Adapter pairing an entry signal module (generate_signal) with a distinct
    exit signal module (check_exit).
    """

    def __init__(self, entry_mod, exit_mod, template: str = ''):
        self.entry_mod = entry_mod
        self.exit_mod = exit_mod
        self.template = template
        self.__name__ = template or f"combo_{getattr(entry_mod, '__name__', 'entry')}__{getattr(exit_mod, '__name__', 'exit')}"

    def generate_signal(self, cache: Dict, params: Dict) -> Dict:
        return self.entry_mod.generate_signal(cache, params)

    def check_exit(self, cache: Dict, position: Dict, params: Dict) -> Dict:
        return self.exit_mod.check_exit(cache, position, params)


def get_signal_module(template: str):
    """Resolve template name (including combo_<entry>__<exit>) to its signal module or adapter."""
    if not isinstance(template, str) or not template:
        raise ValueError(f"Unknown or empty strategy template: {template!r}")
    parsed = parse_combo_template(template)
    if parsed is not None:
        entry_tmpl, exit_tmpl = parsed
        return ComboSignalAdapter(
            get_signal_module(entry_tmpl),
            get_signal_module(exit_tmpl),
            template=template,
        )
    if template == 'rsi_reversion':
        import trainer.signals.rsi_reversion as rsi_reversion
        return rsi_reversion
    if template == 'donchian_breakout':
        import trainer.signals.donchian_breakout as donchian_breakout
        return donchian_breakout
    if template == 'hybrid_confluence':
        import trainer.signals.hybrid_confluence as hybrid_confluence
        return hybrid_confluence
    if template == 'ma_crossover':
        return ma_crossover
    raise ValueError(f"Unknown strategy template: {template!r}")

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

    # Prop-Firm Mandate Compliance (Fix 11.2)
    oos_mandate_compliant:     bool = True
    oos_mandate_breach_reason: Optional[str] = None

    # Regime Tagging (WAVE 33)
    oos_dominant_regime:       str = 'UNKNOWN'

    # Raw OOS trades for Monte Carlo stress testing (WAVE 14)
    oos_trades_list:           List[Dict] = field(default_factory=list)


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
        mandate:          optional mandate configuration
    """

    def __init__(self,
                 symbol:         str,
                 df:             pd.DataFrame,
                 template:       str         = 'ma_crossover',
                 signal_module               = None,
                 opt_months:     int         = 6,
                 test_months:    int         = 1,
                 initial_equity: float       = 10000.0,
                 min_trades_opt: int         = 30,
                 min_trades_test:int         = 10,
                 mandate                     = None,
                 knowledge_base              = None,
                 anchored:       bool        = False,
                 base_params:    Optional[Dict] = None,
                 seed_params:    Optional[Dict] = None,
                 timeframe:      str         = 'M15',
                 record_window_trials: bool  = False):

        if signal_module is None or (signal_module is ma_crossover and template != 'ma_crossover'):
            signal_module = get_signal_module(template)

        self.symbol         = symbol
        self.df             = df
        self.template       = template
        self.signal_module  = signal_module
        self.opt_months     = opt_months
        self.test_months    = test_months
        self.initial_equity = initial_equity
        self.min_trades_opt = min_trades_opt
        self.min_trades_test= min_trades_test
        self.mandate        = mandate
        self.kb             = knowledge_base
        self.anchored       = bool(anchored)
        self.base_params    = base_params.copy() if base_params else {}
        self.seed_params    = seed_params.copy() if seed_params else None
        self.timeframe      = timeframe
        self.record_window_trials = bool(record_window_trials)

        # Sanity gate — refuse before any WF window starts
        from shared.instrument_spec import get_spec
        _spec = get_spec(symbol)
        if _spec is None:
            raise ValueError(
                f"WalkForwardValidator({symbol}): No InstrumentSpec found. "
                f"Run capture_specs.py with MT5 connected before walk-forward."
            )
        if not _spec.sanity_ok:
            raise ValueError(
                f"WalkForwardValidator({symbol}): InstrumentSpec flagged "
                f"(sanity_ok=False). Re-capture during market hours."
            )

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
            opt_start  = idx[0] if self.anchored else (
                current - pd.DateOffset(
                    months=self.opt_months
                )
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

    def _score_window_metrics(self, metrics: Dict, min_trades: int, allow_partial: bool) -> float:
        """
        Score window metrics via module-level composite_score.
        Supports 1-arg monkeypatches in integration tests cleanly.
        """
        dd_limit = (
            float(getattr(self.mandate, 'max_overall_drawdown_pct', 8.0))
            if self.mandate is not None else 8.0
        )
        sig = inspect.signature(composite_score)
        if 'allow_partial_trades' in sig.parameters:
            return composite_score(
                metrics,
                min_trades=min_trades,
                max_drawdown_limit_pct=dd_limit,
                allow_partial_trades=allow_partial,
            )
        return composite_score(metrics)

    def _run_one(self, df_slice: pd.DataFrame,
                 params: Dict,
                 window_start: Optional[datetime] = None,
                 window_end: Optional[datetime] = None,
                 seed: int = 42) -> Dict:
        """Run a single backtest on a data slice."""
        merged_params = {**self.base_params, **params} if self.base_params else params
        bt = Backtester(
            symbol=self.symbol,
            params=merged_params,
            initial_equity=self.initial_equity,
            random_seed=seed
        )
        result  = bt.run(df_slice, self.signal_module, trade_start_time=window_start)
        metrics = calculate_metrics(
            trades=result['trades'],
            equity_curve=result['equity_curve'],
            initial_equity=self.initial_equity,
            window_start=window_start,
            window_end=window_end
        )
        from trainer.core.mandate_evaluator import evaluate_backtest_mandate
        eq_dicts = [{'timestamp': t, 'equity': e} for t, e in zip(result.get('equity_timestamps', []), result['equity_curve'])]
        is_compliant, breach_reason = evaluate_backtest_mandate(
            eq_dicts,
            mandate=self.mandate,
            trades=result.get('trades', []),
            risk_per_trade_pct=float(merged_params.get('risk_per_trade_pct', 1.0)),
        )
        metrics['mandate_compliant'] = is_compliant
        metrics['mandate_breach_reason'] = breach_reason
        metrics['trades'] = result['trades']
        return metrics

    def _optimise(self, df_opt: pd.DataFrame,
                  window_num: int,
                  opt_start: datetime,
                  opt_end: datetime,
                  n_trials: int = 50,
                  is_dominant_regime: str = 'UNKNOWN'
                  ) -> Tuple[Optional[Dict], float, Optional[Dict]]:
        """
        Find best parameters on training window.
        When grid size <= n_trials, evaluates every parameter combination via GridSampler.
        When grid size > n_trials, samples n_trials parameter indices via TPESampler.
        Note: TPE over a single 1D param_idx integer domain acts as stochastic subsampling
        of the discrete parameter grid rather than multi-dimensional parameter space modeling.
        """
        from trainer.core.coherence_validator import validate_coherence
        from shared.instrument_spec import get_spec

        spec = get_spec(self.symbol)
        if not spec or not spec.sanity_ok:
            raise ValueError(f"INSTRUMENT_SPEC_MISSING: Missing or invalid InstrumentSpec for {self.symbol}")

        if not self.param_grid:
            return None, 0.0, None

        try:
            import optuna
        except ImportError:
            raise ImportError(
                "OPTUNA_NOT_INSTALLED: Optuna package is required for Walk-Forward optimization. "
                "Install via: pip install optuna"
            )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        best_score   = 0.0
        best_params  = None
        best_metrics = None

        opt_seed = 42 + window_num
        space_def = TEMPLATE_SPACES.get(self.template)
        use_multivariate_tpe = (
            len(self.param_grid) > n_trials
            and space_def is not None
        )

        # Decide sampler based on grid size vs n_trials
        if not use_multivariate_tpe and len(self.param_grid) <= n_trials:
            search_space = {'param_idx': range(len(self.param_grid))}
            sampler = optuna.samplers.GridSampler(search_space, seed=opt_seed)
            max_evals = len(self.param_grid)
        else:
            sampler = optuna.samplers.TPESampler(
                seed=opt_seed,
                multivariate=use_multivariate_tpe,
            )
            max_evals = n_trials

        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _enqueue_candidate(cand_params: Optional[Dict]) -> None:
            if not cand_params:
                return
            if use_multivariate_tpe:
                enqueue_dict = {
                    k: cand_params[k]
                    for k in space_def
                    if k in cand_params and cand_params[k] in space_def[k]
                }
                if len(enqueue_dict) == len(space_def):
                    study.enqueue_trial(enqueue_dict)
            else:
                for idx, grid_param in enumerate(self.param_grid):
                    if all(cand_params.get(k) == v for k, v in grid_param.items()):
                        study.enqueue_trial({'param_idx': idx})
                        break

        if self.seed_params:
            _enqueue_candidate(self.seed_params)

        # Enqueue historical candidates matching the current in-sample regime
        if self.kb is not None:
            top_historical = self.kb.get_top_historical_candidates(
                template=self.template,
                symbol=self.symbol,
                top_k=5,
                target_regime=is_dominant_regime,
                timeframe=self.timeframe,
            )
            for hist_params in top_historical:
                _enqueue_candidate(hist_params)

        defaults = (
            {k: v for k, v in self.param_grid[0].items() if space_def and k not in space_def}
            if use_multivariate_tpe and self.param_grid else {}
        )
        eval_cache: Dict[Tuple, float] = {}

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_score, best_params, best_metrics
            if use_multivariate_tpe:
                sampled = {
                    k: trial.suggest_categorical(k, list(choices))
                    for k, choices in space_def.items()
                }
                params = {**defaults, **sampled}
                if 'fast_ma_period' in params and 'slow_ma_period' in params:
                    if params['fast_ma_period'] >= params['slow_ma_period']:
                        return -1.0
                if 'rsi_oversold' in params and 'rsi_overbought' in params:
                    if params['rsi_oversold'] >= params['rsi_overbought']:
                        return -1.0
                if 'exit_period' in params and 'channel_period' in params:
                    if params['exit_period'] >= params['channel_period']:
                        return -1.0
            else:
                param_idx = trial.suggest_int('param_idx', 0, len(self.param_grid) - 1)
                params = self.param_grid[param_idx]

            is_coherent, reason = validate_coherence(params, spec)
            if not is_coherent:
                return -1.0

            cache_key = tuple(sorted(
                (k, tuple(v) if isinstance(v, list) else v)
                for k, v in params.items()
            ))
            if use_multivariate_tpe and cache_key in eval_cache:
                return eval_cache[cache_key]

            params_eval = params.copy()
            params_eval['pip_size'] = spec.pip_size

            metrics = self._run_one(
                df_opt,
                params_eval,
                window_start=opt_start,
                window_end=opt_end,
                seed=opt_seed + trial.number,
            )
            score = self._score_window_metrics(
                metrics,
                min_trades=self.min_trades_opt,
                allow_partial=True,
            )
            eval_cache[cache_key] = score

            if score > best_score:
                best_score   = score
                best_params  = params_eval.copy()
                best_metrics = metrics

            if self.kb is not None and self.record_window_trials:
                self.kb.record_search_trial(
                    template=self.template,
                    symbol=self.symbol,
                    run_id=f"wf_{window_num}",
                    data_hash="wf_slice",
                    trial_number=trial.number,
                    params=params_eval,
                    is_coherent=is_coherent,
                    composite_score=score,
                    in_sample_regime=is_dominant_regime,
                    timeframe=self.timeframe,
                )

            return score

        study.optimize(objective, n_trials=max_evals)

        if best_params is None or best_score <= 0.0:
            return None, 0.0, None

        return best_params, best_score, best_metrics

    def _update_progress(self, current_window: int, total_windows: int) -> None:
        """Update live progress metrics in trainer_status.json for dashboard."""
        try:
            status_file = Path(__file__).parent.parent / 'trainer_data' / 'trainer_status.json'
            if status_file.exists():
                with open(status_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data['current_window'] = current_window
                data['total_windows']   = total_windows
                data['progress_pct']    = round((current_window / max(1, total_windows)) * 100, 1)
                tmp_file = status_file.with_suffix('.tmp')
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                tmp_file.replace(status_file)
        except Exception:
            pass  # Progress reporting is best-effort

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
            self._update_progress(wn, len(windows))
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

            # Calculate dominant regime for the training window (WAVE 34)
            is_dominant_regime = 'UNKNOWN'
            try:
                adx_res_opt = calculate_adx(df_opt)
                atr_s_opt = calculate_atr(df_opt)
                regimes_s_opt = calculate_regime(df_opt, adx_res_opt['adx'], atr_s_opt)
                ts_start_opt = pd.Timestamp(w['opt_start']).tz_localize('UTC') if pd.Timestamp(w['opt_start']).tzinfo is None else pd.Timestamp(w['opt_start']).tz_convert('UTC')
                is_regimes = regimes_s_opt[df_opt.index >= ts_start_opt]
                modes_opt = is_regimes.mode()
                if not modes_opt.empty:
                    is_dominant_regime = modes_opt[0]
            except Exception as e:
                log.warning(f"Window {wn}: IS Regime calculation failed: {e}", exc_info=True)

            # Calculate dominant regime for the test window
            try:
                adx_res = calculate_adx(df_test)
                atr_s = calculate_atr(df_test)
                regimes_s = calculate_regime(df_test, adx_res['adx'], atr_s)
                # Slice to strict OOS period to exclude warmup
                ts_start = pd.Timestamp(w['test_start']).tz_localize('UTC') if pd.Timestamp(w['test_start']).tzinfo is None else pd.Timestamp(w['test_start']).tz_convert('UTC')
                oos_regimes = regimes_s[df_test.index >= ts_start]
                modes = oos_regimes.mode()
                if not modes.empty:
                    result.oos_dominant_regime = modes[0]
            except Exception as e:
                log.warning(f"Window {wn}: Regime calculation failed: {e}", exc_info=True)
                
            # Step 1: Optimise on training window
            best_params, best_score, is_metrics = \
                self._optimise(df_opt, wn, w['opt_start'], w['opt_end'], is_dominant_regime=is_dominant_regime)

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

            if self.template == 'rsi_reversion':
                param_str = f"os={best_params.get('rsi_oversold')} ob={best_params.get('rsi_overbought')} midline={best_params.get('exit_on_midline')} sl={best_params.get('sl_atr_multiple')} tp={best_params.get('tp_rr_ratio')}"
            else:
                param_str = f"fast={best_params.get('fast_ma_period')} slow={best_params.get('slow_ma_period')} sl={best_params.get('sl_atr_multiple')} tp={best_params.get('tp_rr_ratio')}"

            log.info(
                f"  IS best: {param_str} | "
                f"score={best_score:.3f} "
                f"trades={is_metrics['total_trades']}"
            )

            # Step 2: Test on unseen data
            spec = get_spec(self.symbol)
            if not spec or not spec.sanity_ok:
                raise ValueError(f"INSTRUMENT_SPEC_MISSING: Missing or invalid InstrumentSpec for {self.symbol}")
            best_params['pip_size'] = spec.pip_size
            oos_metrics = self._run_one(
                df_test, best_params,
                window_start=w['test_start'],
                window_end=w['test_end'],
                seed=wn * 9999
            )

            result.oos_trades          = oos_metrics['total_trades']
            result.oos_profit_factor   = oos_metrics['profit_factor']
            result.oos_calmar          = oos_metrics['calmar_ratio']
            result.oos_net_profit      = oos_metrics['net_profit']
            result.oos_win_rate        = oos_metrics['win_rate']
            result.oos_max_drawdown    = oos_metrics['max_drawdown_pct']
            result.oos_composite_score = self._score_window_metrics(
                oos_metrics,
                min_trades=self.min_trades_test,
                allow_partial=False,
            )
            result.oos_mandate_compliant = oos_metrics.get('mandate_compliant', True)
            result.oos_mandate_breach_reason = oos_metrics.get('mandate_breach_reason')
            result.oos_trades_list = oos_metrics.get('trades', [])

            log.info(
                f"  OOS: trades={oos_metrics['total_trades']} "
                f"PF={oos_metrics['profit_factor']:.3f} "
                f"net=${oos_metrics['net_profit']:.2f}"
            )

            results.append(result)

        return results