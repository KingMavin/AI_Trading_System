"""
Adaptation System — searches for better strategy configurations.

Three stages:
  Stage 1: Parameter Search
    Find the best parameter combination for each template
    Uses grid search (small spaces) or Bayesian (large spaces)

  Stage 2: Filter Testing
    Test whether adding filters improves out-of-sample results
    Sequential forward selection — avoids combinatorial explosion

  Stage 3: Combination Testing
    Test whether combining entry logic from Template A
    with exit logic from Template B beats either alone

Stage 4 (Phase 3): Pattern Discovery — interface defined,
    implementation deferred until 500+ live trades available.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from trainer.core.parameter_grid import get_grid, TEMPLATE_SPACES
from trainer.core.backtester import Backtester
from trainer.core.walk_forward import WalkForwardValidator
from trainer.core.scoring import (
    check_hard_gates, calculate_composite_score,
    aggregate_wf_results, calculate_quick_score
)
from trainer.core.knowledge_base import KnowledgeBase
from shared.metrics import calculate_metrics
import trainer.signals.ma_crossover as ma_crossover

log = logging.getLogger(__name__)


# ── CANDIDATE CONFIG ───────────────────────────────────

@dataclass
class CandidateConfig:
    """
    A fully specified strategy configuration.
    Input to coherence validator and walk-forward.
    """
    candidate_id:   str
    template:       str
    symbol:         str
    timeframe:      str
    parameters:     Dict
    filters:        Dict    = field(default_factory=dict)
    source:         str     = 'parameter_search'
    # source: parameter_search / filter_test / combination

    # Set after testing
    config_hash:          str   = ''
    composite_score:      float = 0.0
    wf_summary:           Dict  = field(default_factory=dict)
    gate_status:          str   = 'PENDING'
    decision_path:        str   = 'PENDING'
    overfit_warning:      bool  = False

    def compute_hash(self) -> str:
        """SHA-256 of parameters + filters + symbol + timeframe."""
        content = json.dumps({
            'template':   self.template,
            'symbol':     self.symbol,
            'timeframe':  self.timeframe,
            'parameters': {
                k: str(v) for k, v in
                sorted(self.parameters.items())
            },
            'filters': {
                k: str(v) for k, v in
                sorted(self.filters.items())
            },
        }, sort_keys=True)
        self.config_hash = hashlib.sha256(
            content.encode()
        ).hexdigest()[:16]
        return self.config_hash


# ── STAGE 1: PARAMETER SEARCH ─────────────────────────

class ParameterSearch:
    """
    Stage 1: Find the best parameter set for a template.

    Algorithm selection based on grid size:
      < 200 combinations:    Exhaustive grid search
      200-2000:              Grid search (all combinations)
      > 2000:                Bayesian optimisation
                             (requires scikit-optimize)
    """

    def __init__(self,
                 symbol:        str,
                 template:      str,
                 df,
                 knowledge_base:KnowledgeBase,
                 data_hash:     str,
                 run_id:        str,
                 initial_equity:float = 10000.0,
                 top_n:         int   = 5):

        self.symbol         = symbol
        self.template       = template
        self.df             = df
        self.kb             = knowledge_base
        self.data_hash      = data_hash
        self.run_id         = run_id
        self.initial_equity = initial_equity
        self.top_n          = top_n
        self.evaluations    = 0

    def _score_params(self, params: Dict,
                       window_results: list) -> float:
        """Score a parameter set using walk-forward results."""
        if not window_results:
            return 0.0
        summary = aggregate_wf_results(window_results)
        gate    = check_hard_gates(summary, 'scoring')
        if not gate.passed:
            return 0.0
        score = calculate_composite_score(summary, 'scoring')
        return score.composite

    def _run_quick_backtest(self, params: Dict,
                             df_slice,
                             seed: int = 42) -> float:
        """
        Run a single backtest and return composite score.
        Used for fast candidate evaluation in Stage 1.
        """
        try:
            bt      = Backtester(
                symbol=self.symbol,
                params=params,
                initial_equity=self.initial_equity,
                random_seed=seed
            )
            result  = bt.run(df_slice, ma_crossover)
            metrics = calculate_metrics(
                trades=result['trades'],
                equity_curve=result['equity_curve'],
                initial_equity=self.initial_equity
            )
            self.evaluations += 1

            return calculate_quick_score(metrics, min_trades=20)

        except Exception as e:
            log.debug(f"Backtest error: {e}")
            return 0.0

    def run_grid_search(self, df_opt,
                        candidate_id_prefix: str
                        ) -> List[CandidateConfig]:
        """
        Exhaustive or sampled grid search.
        Returns top N candidates by in-sample score.
        """
        grid    = get_grid(self.template)
        log.info(
            f"Stage 1: Grid search {self.template} "
            f"on {self.symbol} | {len(grid)} combinations"
        )

        from shared.instrument_spec import get_spec
        spec = get_spec(self.symbol)
        spec_pip_size = spec.pip_size if spec else (0.01 if self.symbol == 'USDJPY' else 0.0001)

        scored = []
        for i, params in enumerate(grid):
            params = params.copy()
            params['pip_size'] = spec_pip_size

            # Compute config hash for deduplication
            candidate = CandidateConfig(
                candidate_id=f"{candidate_id_prefix}_{i}",
                template=self.template,
                symbol=self.symbol,
                timeframe='M15',
                parameters=params,
            )
            config_hash = candidate.compute_hash()

            # Check knowledge base — skip if already tested
            cached = self.kb.get_cached_score(
                config_hash, self.data_hash
            )
            if cached is not None:
                log.debug(
                    f"Cached: {config_hash} score={cached:.3f}"
                )
                scored.append((params, cached, config_hash))
                continue

            # Run backtest
            score = self._run_quick_backtest(
                params, df_opt, seed=i
            )

            # Record in knowledge base
            self.kb.record_candidate(
                config_hash=config_hash,
                data_hash=self.data_hash,
                run_id=self.run_id,
                composite_score=score,
                template=self.template,
                symbol=self.symbol,
                params=params
            )

            # Record parameter insights
            for param_name, param_val in params.items():
                if param_name in ['pip_size', 'warmup_candles',
                                   'risk_per_trade_pct']:
                    continue
                self.kb.record_parameter_result(
                    template=self.template,
                    symbol=self.symbol,
                    parameter=param_name,
                    value=param_val,
                    score=score
                )

            scored.append((params, score, config_hash))

            if (i + 1) % 20 == 0:
                log.info(
                    f"  Progress: {i+1}/{len(grid)} | "
                    f"Best so far: "
                    f"{max(s for _,s,_ in scored):.3f}"
                )

        # Sort by score and take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:self.top_n]

        # Build CandidateConfig objects
        candidates = []
        for idx, (params, score, config_hash) in enumerate(top):
            c = CandidateConfig(
                candidate_id=(
                    f"{candidate_id_prefix}_top{idx+1}"
                ),
                template=self.template,
                symbol=self.symbol,
                timeframe='M15',
                parameters=params,
                source='parameter_search',
            )
            c.config_hash     = config_hash
            c.composite_score = score
            candidates.append(c)

        log.info(
            f"Stage 1 complete: {self.evaluations} evals | "
            f"Top score: {top[0][1]:.3f} if top else 0.0"
        )
        return candidates

    def run_bayesian_search(self, df_opt,
                             candidate_id_prefix: str,
                             n_calls: int = 50
                             ) -> List[CandidateConfig]:
        """
        Bayesian optimisation for large parameter spaces.
        Requires: pip install scikit-optimize

        Falls back to grid search if scikit-optimize
        is not installed.
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Integer, Real, Categorical
        except ImportError:
            log.warning(
                "scikit-optimize not installed. "
                "Falling back to grid search. "
                "Install: pip install scikit-optimize"
            )
            return self.run_grid_search(
                df_opt, candidate_id_prefix
            )

        space_def = TEMPLATE_SPACES.get(self.template, {})
        if not space_def:
            return self.run_grid_search(
                df_opt, candidate_id_prefix
            )

        log.info(
            f"Stage 1: Bayesian search {self.template} "
            f"on {self.symbol} | {n_calls} evaluations"
        )

        # Build skopt dimension objects
        dimensions = []
        dim_names  = []

        for param, values in space_def.items():
            if isinstance(values[0], int):
                dimensions.append(
                    Integer(min(values), max(values), name=param)
                )
            elif isinstance(values[0], float):
                dimensions.append(
                    Real(min(values), max(values), name=param)
                )
            elif isinstance(values[0], str):
                dimensions.append(
                    Categorical(values, name=param)
                )
            dim_names.append(param)

        # Knowledge base informed starting points
        x0 = self._get_informed_starting_points(dim_names)

        results_store = []

        def objective(x):
            params = dict(zip(dim_names, x))

            # Apply constraints
            if self.template == 'ma_crossover':
                if params.get('fast_ma_period', 0) >= \
                        params.get('slow_ma_period', 1):
                    return 1.0  # maximally bad (skopt minimises)

            from shared.instrument_spec import get_spec
            spec = get_spec(self.symbol)
            params['pip_size']          = spec.pip_size if spec else (
                0.01 if self.symbol == 'USDJPY' else 0.0001
            )
            params['risk_per_trade_pct']= 1.0
            params['warmup_candles']    = 250
            params['adx_min_threshold'] = 0
            params['exit_on_opposite_crossover'] = False

            score = self._run_quick_backtest(
                params, df_opt,
                seed=len(results_store)
            )
            results_store.append((params.copy(), score))
            return -score  # skopt minimises, we maximise

        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_calls,
            n_initial_points=min(10, n_calls // 3),
            x0=x0 if x0 else None,
            random_state=42,
            verbose=False,
        )

        # Sort by score and return top N
        results_store.sort(key=lambda x: x[1], reverse=True)
        top = results_store[:self.top_n]

        candidates = []
        for idx, (params, score) in enumerate(top):
            c = CandidateConfig(
                candidate_id=(
                    f"{candidate_id_prefix}_bayes{idx+1}"
                ),
                template=self.template,
                symbol=self.symbol,
                timeframe='M15',
                parameters=params,
                source='bayesian_search',
            )
            c.compute_hash()
            c.composite_score = score
            candidates.append(c)

        log.info(
            f"Bayesian search complete: "
            f"{self.evaluations} evals | "
            f"Best: {top[0][1]:.3f}"
        )
        return candidates

    def _get_informed_starting_points(self,
                                       dim_names: List[str]
                                       ) -> List[List]:
        """
        Get good starting points from knowledge base.
        Used to initialise Bayesian search in good regions.
        """
        starting_points = []
        # Try to build 3 informed starting points
        for _ in range(3):
            point = []
            for dim in dim_names:
                good_vals = self.kb.get_good_parameter_values(
                    self.template, self.symbol, dim, top_n=1
                )
                if good_vals:
                    try:
                        point.append(
                            float(good_vals[0])
                            if '.' in good_vals[0]
                            else int(good_vals[0])
                        )
                    except (ValueError, TypeError):
                        point.append(good_vals[0])
                else:
                    point = []
                    break
            if len(point) == len(dim_names):
                starting_points.append(point)

        return starting_points if starting_points else []


# ── STAGE 2: FILTER TESTING ────────────────────────────

class FilterTesting:
    """
    Stage 2: Test whether adding filters improves performance.
    Uses sequential forward selection — tests one filter at
    a time and only adds it if it improves held-out results.
    Maximum 2 filters added per candidate (Phase 1 limit).
    """

    # Filter definitions — settings to test per filter
    FILTER_SETTINGS = {
        'session': [
            {'allowed_sessions': ['LONDON', 'LONDON_NY_OVERLAP',
                                   'NEW_YORK']},
            {'allowed_sessions': ['LONDON', 'LONDON_NY_OVERLAP']},
            {'allowed_sessions': ['NEW_YORK', 'LONDON_NY_OVERLAP']},
        ],
        'adx_trend': [
            {'adx_min_threshold': 20},
            {'adx_min_threshold': 25},
        ],
        'volatility': [
            {'atr_ratio_min': 0.7, 'atr_ratio_max': 1.5},
            {'atr_ratio_min': 0.5, 'atr_ratio_max': 1.3},
        ],
    }

    def __init__(self,
                 symbol:        str,
                 df_opt,
                 df_held_out,
                 knowledge_base:KnowledgeBase,
                 initial_equity:float = 10000.0,
                 max_filters:   int   = 2,
                 improvement_threshold: float = 0.05):

        self.symbol        = symbol
        self.df_opt        = df_opt
        self.df_held_out   = df_held_out
        self.kb            = knowledge_base
        self.initial_equity= initial_equity
        self.max_filters   = max_filters
        self.threshold     = improvement_threshold

    def _quick_score(self, params: Dict, df) -> float:
        """Run backtest and return simple score."""
        try:
            bt      = Backtester(
                symbol=self.symbol,
                params=params,
                initial_equity=self.initial_equity,
                random_seed=99
            )
            result  = bt.run(df, ma_crossover)
            metrics = calculate_metrics(
                result['trades'],
                result['equity_curve'],
                self.initial_equity
            )
            return calculate_quick_score(metrics, min_trades=10)
        except Exception:
            return 0.0

    def test(self, candidate: CandidateConfig,
             template: str) -> CandidateConfig:
        """
        Run filter testing for one candidate.
        Returns updated candidate with best filters applied.
        """
        base_params    = candidate.parameters.copy()
        base_score     = self._quick_score(
            base_params, self.df_held_out
        )
        current_params = base_params.copy()
        current_score  = base_score
        filters_added  = 0
        applied_filters= {}

        log.info(
            f"Stage 2: Filter testing {candidate.candidate_id} "
            f"| baseline held-out score: {base_score:.3f}"
        )

        for filter_name, settings_list in \
                self.FILTER_SETTINGS.items():

            if filters_added >= self.max_filters:
                break

            # Check knowledge base — skip if this filter
            # consistently doesn't help
            improvement_rate = self.kb.get_filter_improvement_rate(
                filter_name, template
            )
            if improvement_rate is not None and \
                    improvement_rate < 0.20:
                log.info(
                    f"  Skipping {filter_name} — "
                    f"historical improvement rate "
                    f"{improvement_rate:.0%} too low"
                )
                continue

            # Find best setting for this filter in-sample
            best_filter_score  = -1.0
            best_filter_setting= None

            for setting in settings_list:
                test_params = current_params.copy()
                test_params.update(setting)
                score = self._quick_score(
                    test_params, self.df_opt
                )
                if score > best_filter_score:
                    best_filter_score  = score
                    best_filter_setting= setting

            if best_filter_setting is None:
                continue

            # Test best setting on held-out data
            test_params = current_params.copy()
            test_params.update(best_filter_setting)
            held_out_score = self._quick_score(
                test_params, self.df_held_out
            )

            improved = (
                held_out_score > current_score * (1 + self.threshold)
            )
            delta    = held_out_score - current_score

            # Record result in knowledge base
            self.kb.record_filter_result(
                filter_name=filter_name,
                template=template,
                improved=improved,
                delta=delta
            )

            if improved:
                current_params.update(best_filter_setting)
                applied_filters.update(best_filter_setting)
                current_score = held_out_score
                filters_added += 1
                log.info(
                    f"  ✓ Filter {filter_name} added: "
                    f"score {base_score:.3f} → "
                    f"{current_score:.3f} (+{delta:.3f})"
                )
            else:
                log.info(
                    f"  ✗ Filter {filter_name} rejected: "
                    f"held-out {held_out_score:.3f} "
                    f"did not improve over {current_score:.3f}"
                )

        # Update candidate
        candidate.parameters = current_params
        candidate.filters    = applied_filters
        candidate.compute_hash()

        log.info(
            f"Stage 2 complete: {filters_added} filters added | "
            f"Score: {base_score:.3f} → {current_score:.3f}"
        )
        return candidate


# ── STAGE 3: COMBINATION TESTING ──────────────────────

# Valid template combinations
VALID_COMBINATIONS = [
    ('ma_crossover', 'rsi_reversion',  'Trend entry + mean reversion exit'),
    ('ma_crossover', 'bb_breakout',    'Trend entry + volatility exit'),
]

class CombinationTesting:
    """
    Stage 3: Test whether combining templates beats either alone.
    Entry logic from Template A + exit logic from Template B.
    Combination must beat BOTH parents by > 5% to be retained.
    """

    def __init__(self,
                 symbol:        str,
                 df,
                 initial_equity:float = 10000.0,
                 improvement_threshold: float = 0.05):
        self.symbol        = symbol
        self.df            = df
        self.initial_equity= initial_equity
        self.threshold     = improvement_threshold

    def test(self,
             candidates_by_template: Dict[str, List[CandidateConfig]]
             ) -> List[CandidateConfig]:
        """
        Test all valid combinations.
        Returns list of combination candidates that beat
        both parent templates.
        """
        valid_combinations = []

        for entry_tmpl, exit_tmpl, description in \
                VALID_COMBINATIONS:

            entry_candidates = candidates_by_template.get(
                entry_tmpl, []
            )
            exit_candidates  = candidates_by_template.get(
                exit_tmpl, []
            )

            if not entry_candidates or not exit_candidates:
                log.info(
                    f"Stage 3: Skipping {entry_tmpl}+"
                    f"{exit_tmpl} — "
                    f"one or both templates have no candidates"
                )
                continue

            log.info(
                f"Stage 3: Testing {entry_tmpl}+{exit_tmpl} "
                f"({description})"
            )

            # Take top 2 from each template
            for ec in entry_candidates[:2]:
                for xc in exit_candidates[:2]:
                    combo = self._test_one_combination(
                        ec, xc, entry_tmpl, exit_tmpl
                    )
                    if combo is not None:
                        valid_combinations.append(combo)

        log.info(
            f"Stage 3 complete: "
            f"{len(valid_combinations)} combinations "
            f"beat both parents"
        )
        return valid_combinations

    def _test_one_combination(self,
                               entry_candidate: CandidateConfig,
                               exit_candidate:  CandidateConfig,
                               entry_tmpl:      str,
                               exit_tmpl:       str
                               ) -> Optional[CandidateConfig]:
        """
        Test one entry+exit combination.
        Returns CandidateConfig if it beats both parents,
        None otherwise.
        """
        # For Phase 1: combinations use the same backtester
        # but we note the intent. Full combination logic
        # (routing exit signals from exit_tmpl) is Phase 2.
        # Here we test whether the exit_candidate parameters
        # on top of entry_candidate improve results.

        combined_params = entry_candidate.parameters.copy()
        combined_params['exit_on_opposite_crossover'] = True

        try:
            bt = Backtester(
                symbol=self.symbol,
                params=combined_params,
                initial_equity=self.initial_equity,
                random_seed=42
            )
            result  = bt.run(self.df, ma_crossover)
            metrics = calculate_metrics(
                result['trades'],
                result['equity_curve'],
                self.initial_equity
            )

            combo_score = calculate_quick_score(metrics, min_trades=10)

            entry_score = entry_candidate.composite_score
            exit_score  = exit_candidate.composite_score
            max_parent  = max(entry_score, exit_score)

            # Must beat both parents by threshold
            if combo_score <= max_parent * (1 + self.threshold):
                return None

            log.info(
                f"  ✓ Combination {entry_tmpl}+{exit_tmpl}: "
                f"score {combo_score:.3f} beats "
                f"parents ({entry_score:.3f}, {exit_score:.3f})"
            )

            combo = CandidateConfig(
                candidate_id=(
                    f"combo_{entry_tmpl}_{exit_tmpl}_"
                    f"{entry_candidate.candidate_id}"
                ),
                template=f"combo_{entry_tmpl}_{exit_tmpl}",
                symbol=self.symbol,
                timeframe='M15',
                parameters=combined_params,
                source='combination',
            )
            combo.composite_score = combo_score
            combo.compute_hash()
            return combo

        except Exception as e:
            log.debug(f"Combination test error: {e}")
            return None


# ── STAGE 4: DISCOVERY INTERFACE (PHASE 3) ────────────

class HypothesisGenerator:
    """
    Stage 4: Pattern discovery engine.

    STATUS: Interface defined. Not implemented.
    ACTIVATION: Phase 3 — requires 500+ live trades.

    In Phase 3 this will:
      1. Load Category D features from Layer 3
      2. Load outcome labels (forward returns)
      3. Use Random Forest to find predictive features
      4. Generate CandidateConfig objects from patterns
      5. Pass them through the same pipeline as other candidates
    """

    def generate(self,
                 feature_sets:    Dict,
                 outcome_labels:  Dict,
                 knowledge_base:  KnowledgeBase,
                 n_hypotheses:    int = 10
                 ) -> List[CandidateConfig]:
        raise NotImplementedError(
            "Stage 4 Pattern Discovery is a Phase 3 feature. "
            "Requires 500+ live trades and scikit-learn. "
            "Enable in trainer_config.yaml when ready."
        )


# ── ADAPTATION ORCHESTRATOR ────────────────────────────

class AdaptationSystem:
    """
    Orchestrates all three adaptation stages.
    Returns a list of the best candidates for walk-forward.
    """

    def __init__(self,
                 symbol:        str,
                 df,
                 df_held_out,
                 knowledge_base:KnowledgeBase,
                 data_hash:     str,
                 run_id:        str,
                 templates:     List[str] = None,
                 initial_equity:float     = 10000.0,
                 top_n:         int       = 5):

        self.symbol         = symbol
        self.df             = df
        self.df_held_out    = df_held_out
        self.kb             = knowledge_base
        self.data_hash      = data_hash
        self.run_id         = run_id
        self.templates      = templates or ['ma_crossover']
        self.initial_equity = initial_equity
        self.top_n          = top_n

    def run(self) -> List[CandidateConfig]:
        """
        Run all adaptation stages.
        Returns list of best candidates ready for walk-forward.
        """
        # Sanity gate — refuse before any candidate evaluation
        from shared.instrument_spec import get_spec
        _spec = get_spec(self.symbol)
        if _spec is None:
            raise ValueError(
                f"AdaptationSystem({self.symbol}): No InstrumentSpec found. "
                f"Run capture_specs.py with MT5 connected before adaptation."
            )
        if not _spec.sanity_ok:
            raise ValueError(
                f"AdaptationSystem({self.symbol}): InstrumentSpec flagged "
                f"(sanity_ok=False). Re-capture during market hours."
            )

        all_candidates         = []
        candidates_by_template = {}

        # ── Stage 1: Parameter Search ──────────────────
        log.info("="*50)
        log.info("ADAPTATION STAGE 1: PARAMETER SEARCH")
        log.info("="*50)

        for template in self.templates:
            search = ParameterSearch(
                symbol=self.symbol,
                template=template,
                df=self.df,
                knowledge_base=self.kb,
                data_hash=self.data_hash,
                run_id=self.run_id,
                initial_equity=self.initial_equity,
                top_n=self.top_n,
            )

            grid_size = len(get_grid(template))
            if grid_size > 2000:
                candidates = search.run_bayesian_search(
                    self.df,
                    candidate_id_prefix=f"{template}_{self.symbol}"
                )
            else:
                candidates = search.run_grid_search(
                    self.df,
                    candidate_id_prefix=f"{template}_{self.symbol}"
                )

            candidates_by_template[template] = candidates
            all_candidates.extend(candidates)

            # Update knowledge base template performance
            for c in candidates:
                self.kb.update_template_performance(
                    template=template,
                    symbol=self.symbol,
                    score=c.composite_score,
                    promoted=False  # updated after promotion
                )

        log.info(
            f"Stage 1 complete: "
            f"{len(all_candidates)} candidates found"
        )

        # ── Stage 2: Filter Testing ────────────────────
        log.info("="*50)
        log.info("ADAPTATION STAGE 2: FILTER TESTING")
        log.info("="*50)

        filter_tester = FilterTesting(
            symbol=self.symbol,
            df_opt=self.df,
            df_held_out=self.df_held_out,
            knowledge_base=self.kb,
            initial_equity=self.initial_equity,
        )

        filtered_candidates = []
        for candidate in all_candidates:
            updated = filter_tester.test(
                candidate, candidate.template
            )
            filtered_candidates.append(updated)

        log.info(
            f"Stage 2 complete: "
            f"{len(filtered_candidates)} candidates after filtering"
        )

        # ── Stage 3: Combination Testing ──────────────
        log.info("="*50)
        log.info("ADAPTATION STAGE 3: COMBINATION TESTING")
        log.info("="*50)

        combo_tester = CombinationTesting(
            symbol=self.symbol,
            df=self.df,
            initial_equity=self.initial_equity,
        )

        combo_candidates = combo_tester.test(
            candidates_by_template
        )
        filtered_candidates.extend(combo_candidates)

        log.info(
            f"Stage 3 complete: "
            f"{len(combo_candidates)} combinations added"
        )
        log.info(
            f"Total candidates for walk-forward: "
            f"{len(filtered_candidates)}"
        )

        return filtered_candidates