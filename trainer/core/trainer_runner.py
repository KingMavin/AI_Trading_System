"""
Training Pipeline Orchestrator — Layer 1.

Wires all components into a single training run.
One command runs the complete pipeline end to end.

Usage:
  python trainer/trainer.py run
  python trainer/trainer.py run --symbol EURUSD
  python trainer/trainer.py run --symbol EURUSD --quick
  python trainer/trainer.py check
  python trainer/trainer.py status
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from trainer.core.data_loader import load_ohlcv, get_available_range
from trainer.core.adaptation import AdaptationSystem, CandidateConfig
from trainer.core.walk_forward import WalkForwardValidator
from trainer.core.scoring import (
    check_hard_gates, calculate_composite_score,
    aggregate_wf_results, rank_candidates
)
from trainer.core.promotion import PromotionEngine
from trainer.core.knowledge_base import KnowledgeBase
import trainer.signals.ma_crossover as ma_crossover
import trainer.signals.rsi_reversion as rsi_reversion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)


# ── DEFAULT CONFIGURATION ──────────────────────────────

DEFAULT_TEMPLATE_TEST_MONTHS = {
    'ma_crossover': 1,
    'rsi_reversion': 2
}

DEFAULT_CONFIG = {
    'symbols':          ['EURUSD'],
    'timeframe':        'M15',
    'templates':        ['rsi_reversion'],
    'opt_months':       6,
    'test_months':      None,
    'initial_equity':   10000.0,
    'top_n_candidates': 5,
    'min_trades_opt':   30,
    'min_trades_test':  10,
    'data_start':       '2015-01-01',
    'data_end':         '2023-12-31',
    'held_out_months':  2,    # months reserved for filter testing
}

# Quick mode — smaller grid, shorter period, fewer candidates
QUICK_CONFIG = {
    **DEFAULT_CONFIG,
    'symbols':          ['EURUSD'],
    'data_start':       '2020-01-01',
    'data_end':         '2022-12-31',
    'top_n_candidates': 3,
    'opt_months':       4,
}


# ── RUN RECORD ─────────────────────────────────────────

class RunRecord:
    """Tracks the state and results of one training run."""

    def __init__(self, run_id: str, config: Dict):
        self.run_id      = run_id
        self.config      = config
        self.started_at  = datetime.now(timezone.utc)
        self.completed_at= None
        self.outcome     = 'RUNNING'
        self.layers      = {}
        self.candidates_tested    = 0
        self.candidates_promoted  = 0
        self.candidates_flagged   = 0
        self.candidates_rejected  = 0
        self.best_score           = 0.0
        self.baseline_score       = 0.0
        self.improvement_pct      = None
        self.new_strategy_deployed= False
        self.deployed_strategy_id = None
        self.failure_layer        = None
        self.failure_reason       = None
        self.symbol_status: Dict[str, Dict] = {}

    def layer_start(self, layer: str) -> None:
        self.layers[layer] = {
            'started_at': datetime.now(timezone.utc).isoformat(),
            'status':     'RUNNING',
        }
        log.info(f"{'='*50}")
        log.info(f"LAYER: {layer}")
        log.info(f"{'='*50}")

    def layer_complete(self, layer: str,
                        summary: Dict = None) -> None:
        if layer in self.layers:
            self.layers[layer]['status']       = 'COMPLETE'
            self.layers[layer]['completed_at'] = \
                datetime.now(timezone.utc).isoformat()
            if summary:
                self.layers[layer]['summary'] = summary

    def layer_failed(self, layer: str, reason: str) -> None:
        if layer in self.layers:
            self.layers[layer]['status'] = 'FAILED'
            self.layers[layer]['reason'] = reason
        self.failure_layer  = layer
        self.failure_reason = reason

    def to_dict(self) -> Dict:
        duration = None
        if self.completed_at:
            duration = (
                self.completed_at - self.started_at
            ).total_seconds()
        return {
            'run_id':               self.run_id,
            'started_at':           self.started_at.isoformat(),
            'completed_at':         self.completed_at.isoformat()
                                    if self.completed_at else None,
            'duration_seconds':     duration,
            'outcome':              self.outcome,
            'config':               self.config,
            'layers':               self.layers,
            'candidates_tested':    self.candidates_tested,
            'candidates_promoted':  self.candidates_promoted,
            'candidates_flagged':   self.candidates_flagged,
            'candidates_rejected':  self.candidates_rejected,
            'best_score':           self.best_score,
            'baseline_score':       self.baseline_score,
            'improvement_pct':      self.improvement_pct,
            'new_strategy_deployed':self.new_strategy_deployed,
            'deployed_strategy_id': self.deployed_strategy_id,
            'failure_layer':        self.failure_layer,
            'failure_reason':       self.failure_reason,
        }


# ── DATA HASH ──────────────────────────────────────────

def compute_data_hash(symbol: str, timeframe: str) -> str:
    """
    Compute a hash representing the current state of
    a dataset. Changes when data changes.
    Used for deduplication in knowledge base.
    """
    info = get_available_range(symbol, timeframe)
    if not info.get('exists'):
        return 'no_data'
    content = (
        f"{symbol}_{timeframe}_"
        f"{info['start']}_{info['end']}_"
        f"{info['count']}"
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── PRINT FUNCTIONS ────────────────────────────────────

def print_run_header(config: Dict, run_id: str) -> None:
    print(f"\n{'='*60}")
    print(f"  ATS TRAINER - TRAINING RUN")
    print(f"{'='*60}")
    print(f"  Run ID:    {run_id}")
    print(f"  Symbols:   {', '.join(config['symbols'])}")
    print(f"  Templates: {', '.join(config['templates'])}")
    print(f"  Period:    {config['data_start']} -> "
          f"{config['data_end']}")
    test_m_str = config.get('test_months') or 'auto'
    print(f"  OPT/TEST:  {config['opt_months']}m / "
          f"{test_m_str}m windows")
    print(f"  Equity:    ${config['initial_equity']:,.0f}")
    print(f"{'='*60}\n")


def print_candidate_summary(candidates: List[CandidateConfig],
                             title: str = "CANDIDATES") -> None:
    print(f"\n  {title} ({len(candidates)} total):")
    print(f"  {'ID':<30} {'Template':<15} "
          f"{'Score':>7} {'Fast':>5} {'Slow':>5}")
    print(f"  {'-'*70}")
    for c in candidates[:10]:
        fast = c.parameters.get('fast_ma_period', '-')
        slow = c.parameters.get('slow_ma_period', '-')
        print(
            f"  {c.candidate_id:<30} "
            f"{c.template:<15} "
            f"{c.composite_score:>7.3f} "
            f"{str(fast):>5} "
            f"{str(slow):>5}"
        )


def print_final_report(run_record: RunRecord,
                        decisions: list) -> None:
    """Print the final training run report."""
    duration = 0
    if run_record.completed_at:
        duration = (
            run_record.completed_at - run_record.started_at
        ).total_seconds()

    print(f"\n{'='*60}")
    print(f"  TRAINING RUN COMPLETE")
    print(f"{'='*60}")
    print(f"  Run ID:      {run_record.run_id}")
    print(f"  Duration:    {duration/60:.1f} minutes")
    print(f"  Outcome:     {run_record.outcome}")
    print(f"\n  CANDIDATES:")
    print(f"    Tested:    {run_record.candidates_tested}")
    print(f"    Promoted:  {run_record.candidates_promoted}")
    print(f"    Flagged:   {run_record.candidates_flagged}")
    print(f"    Rejected:  {run_record.candidates_rejected}")
    print(f"\n  SCORES:")
    print(f"    Baseline:  {run_record.baseline_score:.3f}")
    print(f"    Best:      {run_record.best_score:.3f}")
    if run_record.improvement_pct is not None:
        print(f"    Improvement: "
              f"{run_record.improvement_pct:+.1f}%")

    print(f"\n  DECISIONS:")
    for d in decisions:
        tag = {
            'PROMOTED': '[PASS]',
            'FLAGGED':  '[FLAG]',
            'REJECTED': '[FAIL]',
        }.get(d.decision_outcome, '[?]')
        print(
            f"    {tag} {d.candidate_id:<28} "
            f"score={d.composite_score:.3f} "
            f"-> {d.decision_outcome}"
        )
        if d.flags:
            for flag in d.flags[:2]:
                print(f"      [!] {flag}")

    if run_record.new_strategy_deployed:
        print(f"\n  [PASS] NEW STRATEGY DEPLOYED: "
              f"{run_record.deployed_strategy_id}")
    elif run_record.candidates_flagged > 0:
        print(f"\n  [FLAG] {run_record.candidates_flagged} "
              f"candidate(s) require your review.")
    else:
        print(f"\n  Current strategy remains active.")

    print(f"{'='*60}\n")


# ── MAIN TRAINER RUNNER ────────────────────────────────

class TrainerRunner:
    """
    Orchestrates the complete training pipeline.
    Runs all layers in sequence for one training run.
    """

    def __init__(self, config: Dict = None,
                 quick_mode: bool = False):
        self.config   = config or (
            QUICK_CONFIG if quick_mode else DEFAULT_CONFIG
        )
        self.run_id   = datetime.now(timezone.utc).strftime(
            'run_%Y%m%d_%H%M%S'
        )
        self.kb_path  = (
            Path(__file__).parent.parent /
            'trainer_data' / 'history' / 'knowledge_base.json'
        )
        self.log_path = (
            Path(__file__).parent.parent /
            'trainer_data' / 'history' / 'decision_log.jsonl'
        )
        # Verify and load mandate config for prop-rule compliance evaluation (Fix 11.2)
        from engine.core.mandate import load_mandate, MANDATE_FILE
        self.mandate = load_mandate(MANDATE_FILE)

    def run(self) -> RunRecord:
        """Execute the complete training pipeline."""
        record = RunRecord(self.run_id, self.config)
        print_run_header(self.config, self.run_id)

        # Load knowledge base
        kb = KnowledgeBase(base_path=str(self.kb_path))
        log.info(
            f"Knowledge base loaded: "
            f"{kb.get_stats()['candidates_tested']} "
            f"candidates in history"
        )

        all_decisions  = []
        all_candidates = []

        try:
            failed_symbols = 0
            for symbol in self.config['symbols']:
                log.info(
                    f"\nProcessing symbol: {symbol}"
                )
                try:
                    decisions, candidates = self._run_symbol(
                        symbol, record, kb
                    )
                    all_decisions.extend(decisions)
                    all_candidates.extend(candidates)
                    record.symbol_status[symbol] = {'status': 'COMPLETED', 'candidates': len(candidates)}
                except Exception as sym_err:
                    failed_symbols += 1
                    log.error(f"PER_SYMBOL_FAILURE ({symbol}): {sym_err}", exc_info=True)
                    record.symbol_status[symbol] = {'status': 'FAILED', 'error': str(sym_err)}

            # Update knowledge base
            record.layer_start('layer_9_memory')
            kb.record_run_summary({
                'run_id':       self.run_id,
                'best_score':   record.best_score,
                'baseline':     record.baseline_score,
                'improvement':  record.improvement_pct,
                'candidates':   record.candidates_tested,
                'promoted':     record.candidates_promoted,
            })
            kb.save()
            record.layer_complete('layer_9_memory')

            if failed_symbols == len(self.config['symbols']):
                record.outcome = 'FAILED'
            elif failed_symbols > 0:
                record.outcome = 'PARTIAL_FAILURE'
            else:
                record.outcome = 'COMPLETED'
            record.completed_at  = datetime.now(timezone.utc)

        except Exception as e:
            log.error(f"Pipeline failed: {e}", exc_info=True)
            record.outcome      = 'FAILED'
            record.completed_at = datetime.now(timezone.utc)
            if not record.failure_layer:
                record.failure_layer  = 'unknown'
                record.failure_reason = str(e)

        # Always print final report
        print_final_report(record, all_decisions)

        return record

    def _run_symbol(self, symbol: str,
                    record: RunRecord,
                    kb: KnowledgeBase) -> tuple:
        """Run the full pipeline for one symbol."""
        config  = self.config
        decisions  = []
        candidates = []

        from shared.instrument_spec import get_spec
        spec = get_spec(symbol)
        if spec is None:
            raise ValueError(
                f"Gate 1 (Structural) FAILED for {symbol}: "
                f"No InstrumentSpec found. "
                f"Run capture_specs.py with MT5 connected before training."
            )
        if not spec.sanity_ok:
            raise ValueError(
                f"Gate 1 (Structural) FAILED for {symbol}: "
                f"InstrumentSpec flagged (sanity_ok=False). "
                f"Re-capture during market hours before training."
            )

        # ── Layer 2: Data Loading ──────────────────────
        record.layer_start('layer_2_data')
        try:
            from datetime import datetime as dt
            df = load_ohlcv(
                symbol=symbol,
                timeframe=config['timeframe'],
                start=dt.strptime(
                    config['data_start'], '%Y-%m-%d'
                ),
                end=dt.strptime(
                    config['data_end'], '%Y-%m-%d'
                ),
                warmup_candles=300
            )
            log.info(
                f"Loaded {symbol} {config['timeframe']}: "
                f"{len(df):,} candles "
                f"({df.index.min().date()} → "
                f"{df.index.max().date()})"
            )

            # Compute data hash for deduplication
            data_hash = compute_data_hash(
                symbol, config['timeframe']
            )

            # Split: main data + held-out for filter testing
            requested_held = config.get('held_out_months', 2)
            opt_m = config.get('opt_months', 6)
            test_m = config.get('test_months') or max(DEFAULT_TEMPLATE_TEST_MONTHS.values())
            min_main_months = opt_m + test_m
            total_days = (df.index[-1] - df.index[0]).days
            total_months = max(1, int(round(total_days / 30.4375)))
            held_months = max(0, min(requested_held, total_months - min_main_months))

            if held_months > 0:
                split_point = df.index[-1] - __import__('pandas').DateOffset(months=held_months)
                df_main     = df[df.index < split_point]
                df_held_out = df[df.index >= split_point]
            else:
                df_main     = df
                df_held_out = df.iloc[-100:]

            log.info(
                f"Main: {len(df_main):,} candles | "
                f"Held-out: {len(df_held_out):,} candles"
            )
            record.layer_complete('layer_2_data', {
                'candles':    len(df),
                'data_hash':  data_hash,
            })

        except Exception as e:
            record.layer_failed('layer_2_data', str(e))
            log.error(f"Data loading failed: {e}")
            return decisions, candidates

        # ── Layer 6: Adaptation ────────────────────────
        record.layer_start('layer_6_adaptation')
        try:
            adaptation = AdaptationSystem(
                symbol=symbol,
                df=df_main,
                df_held_out=df_held_out,
                knowledge_base=kb,
                data_hash=data_hash,
                run_id=self.run_id,
                templates=config['templates'],
                initial_equity=config['initial_equity'],
                top_n=config['top_n_candidates'],
            )
            candidates = adaptation.run()
            record.candidates_tested = len(candidates)
            record.layer_complete('layer_6_adaptation', {
                'candidates': len(candidates),
            })

        except Exception as e:
            record.layer_failed('layer_6_adaptation', str(e))
            log.error(f"Adaptation failed: {e}")
            return decisions, candidates

        if not candidates:
            log.warning("No candidates produced by adaptation.")
            return decisions, candidates

        # Show top candidates
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.composite_score,
            reverse=True
        )
        print_candidate_summary(
            sorted_candidates,
            f"TOP CANDIDATES — {symbol}"
        )

        # ── Layer 7: Walk-Forward ──────────────────────
        record.layer_start('layer_7_walk_forward')
        try:
            log.info(
                f"Running walk-forward on "
                f"{len(candidates)} candidates..."
            )

            wf_results_by_candidate = {}

            for candidate in sorted_candidates[
                :config['top_n_candidates']
            ]:
                log.info(
                    f"WF: {candidate.candidate_id} "
                    f"(score={candidate.composite_score:.3f})"
                )

                signal_mod = rsi_reversion if candidate.template == 'rsi_reversion' else ma_crossover

                resolved_test_months = config.get('test_months')
                if resolved_test_months is None:
                    resolved_test_months = DEFAULT_TEMPLATE_TEST_MONTHS.get(candidate.template, 1)
                
                wf = WalkForwardValidator(
                    symbol=symbol,
                    df=df_main,
                    template=candidate.template,
                    signal_module=signal_mod,
                    opt_months=config['opt_months'],
                    test_months=resolved_test_months,
                    initial_equity=config['initial_equity'],
                    min_trades_opt=config['min_trades_opt'],
                    min_trades_test=config['min_trades_test'],
                    mandate=self.mandate,
                )
                wf_windows = wf.run()
                wf_summary = aggregate_wf_results(wf_windows)
                wf_results_by_candidate[
                    candidate.candidate_id
                ] = (wf_windows, wf_summary)

                candidate.wf_summary = wf_summary

            record.layer_complete('layer_7_walk_forward')

        except Exception as e:
            record.layer_failed('layer_7_walk_forward', str(e))
            log.error(f"Walk-forward failed: {e}")
            return decisions, candidates

        # ── Layer 8: Scoring & Promotion ───────────────
        record.layer_start('layer_8_scoring')
        try:
            promotion_engine = PromotionEngine(
                run_id=self.run_id,
                log_path=str(self.log_path),
            )

            record.baseline_score = \
                promotion_engine.baseline['composite_score']

            scored_candidates = []

            for candidate in sorted_candidates[
                :config['top_n_candidates']
            ]:
                if candidate.candidate_id not in \
                        wf_results_by_candidate:
                    continue

                _, wf_summary = wf_results_by_candidate[
                    candidate.candidate_id
                ]

                # Hard gate check
                gate = check_hard_gates(
                    wf_summary,
                    candidate.candidate_id,
                    mandate=self.mandate,
                    symbol=symbol,
                    initial_equity=config['initial_equity']
                )

                if not gate.passed:
                    log.info(
                        f"  ✗ {candidate.candidate_id} "
                        f"failed gate: {gate.failed_gate}"
                    )
                    candidate.gate_status = 'FAILED'

                    # Still make a rejection decision
                    decision = promotion_engine.decide(
                        score=__import__(
                            'trainer.core.scoring',
                            fromlist=['CompositeScore']
                        ).CompositeScore(
                            candidate_id=candidate.candidate_id,
                            composite=0.0
                        ),
                        wf_summary=wf_summary,
                        gate_result=gate,
                    )
                    decisions.append(decision)
                    record.candidates_rejected += 1
                    continue

                # Composite score
                try:
                    score = calculate_composite_score(
                        wf_summary,
                        candidate_id=candidate.candidate_id,
                    )
                    candidate.composite_score = score.composite
                    candidate.scoring_status  = 'OK'
                    candidate.gate_status     = 'PASSED'
                    scored_candidates.append((candidate, score))
                    log.info(
                        f"  ✓ {candidate.candidate_id} "
                        f"score={score.composite:.3f}"
                    )
                except Exception as e:
                    log.error(
                        f"CANDIDATE_SCORING_CRASH: Composite scoring crashed for {candidate.candidate_id}: {e}",
                        exc_info=True
                    )
                    candidate.composite_score = None
                    candidate.scoring_status  = 'ERROR'
                    candidate.scoring_error   = str(e)
                    candidate.gate_status     = 'SCORING_ERROR'

                    # Record explicit SCORING_ERROR rejection decision to decision_log.jsonl
                    from trainer.core.scoring import CompositeScore, GateResult
                    decision = promotion_engine.decide(
                        score=CompositeScore(candidate_id=candidate.candidate_id, composite=0.0),
                        wf_summary=wf_summary,
                        gate_result=GateResult(candidate_id=candidate.candidate_id, passed=False, failed_gate='SCORING_ERROR', failure_reason=f'Scoring crashed: {e}')
                    )
                    decisions.append(decision)
                    record.candidates_rejected += 1
                    continue

            # Rank and decide on top candidates
            scored_candidates.sort(
                key=lambda x: x[1].composite, reverse=True
            )

            for candidate, score in scored_candidates:
                _, wf_summary = wf_results_by_candidate[
                    candidate.candidate_id
                ]
                gate = check_hard_gates(
                    wf_summary, candidate.candidate_id
                )

                decision = promotion_engine.decide(
                    score=score,
                    wf_summary=wf_summary,
                    gate_result=gate,
                )
                decisions.append(decision)

                if score.composite > record.best_score:
                    record.best_score = score.composite
                    if decision.improvement_pct:
                        record.improvement_pct = \
                            decision.improvement_pct

                if decision.decision_outcome == 'PROMOTED':
                    record.candidates_promoted  += 1
                    record.new_strategy_deployed = True
                    record.deployed_strategy_id  = \
                        candidate.candidate_id
                elif decision.decision_outcome == 'FLAGGED':
                    record.candidates_flagged += 1
                else:
                    record.candidates_rejected += 1

            record.layer_complete('layer_8_scoring', {
                'scored':   len(scored_candidates),
                'promoted': record.candidates_promoted,
                'flagged':  record.candidates_flagged,
            })

        except Exception as e:
            record.layer_failed('layer_8_scoring', str(e))
            log.error(f"Scoring failed: {e}")

        # Layer 10: Generate persistent Markdown & HTML reports
        try:
            from trainer.core.report import generate_report
            generate_report(record, candidates, decisions, wf_results_by_candidate)
            record.layer_complete('layer_10_reporting', {'candidates': len(candidates), 'decisions': len(decisions)})
        except Exception as e:
            record.layer_failed('layer_10_reporting', str(e))
            log.error(f"Report generation failed: {e}")

        return decisions, candidates