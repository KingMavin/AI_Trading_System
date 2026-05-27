"""
Promotion Engine — makes the final deployment decision.

Three possible outcomes for every training run:
  AUTO PROMOTE — strategy beats baseline by >= 20%
                 AND passes all auto-promote conditions
  FLAG FOR REVIEW — beats baseline but has concerns
                    user must decide manually
  AUTO REJECT  — doesn't beat baseline or fails gates

Every decision is recorded in a promotion log.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field, asdict

from trainer.core.scoring import CompositeScore, GateResult


# ── PROMOTION THRESHOLDS ───────────────────────────────
AUTO_PROMOTE_THRESHOLD    = 0.20   # beat baseline by 20%
FIRST_RUN_AUTO_THRESHOLD  = 0.50   # first run auto floor
FIRST_RUN_FLAG_THRESHOLD  = 0.40   # first run flag floor

# Auto-promote conditions (ALL must be true)
AUTO_PROMOTE_CONDITIONS = {
    'min_calmar':              1.0,
    'min_profit_factor':       1.3,
    'max_drawdown_pct':        18.0,
    'min_profitable_windows':  0.65,
    'min_regime_consistency':  0.60,
}


@dataclass
class PromotionRecord:
    """Complete record of a promotion decision."""
    decision_id:              str
    run_id:                   str
    decided_at:               str

    # Candidate
    candidate_id:             str
    composite_score:          float

    # Baseline
    baseline_strategy_id:     Optional[str]
    baseline_score:           float
    improvement_pct:          Optional[float]

    # Decision
    decision_path:            str   # AUTO_PROMOTE/FLAG/AUTO_REJECT
    decision_outcome:         str   # PROMOTED/FLAGGED/REJECTED
    decided_by:               str   # trainer_auto / user

    # Flags
    flags:                    Dict  = field(default_factory=dict)

    # Gate results
    gate_results:             Dict  = field(default_factory=dict)

    # Outcome
    strategy_deployed:        bool  = False
    user_notes:               str   = ''


@dataclass
class PromotionDecision:
    """The output of the promotion engine for one candidate."""
    candidate_id:       str
    composite_score:    float
    baseline_score:     float
    improvement_pct:    Optional[float]
    decision_path:      str
    decision_outcome:   str
    flags:              List[str]     = field(default_factory=list)
    recommendation:     str          = ''
    record:             Optional[PromotionRecord] = None


class PromotionEngine:
    """
    Makes promotion decisions based on scoring results.
    Records every decision to the promotion log.
    """

    def __init__(self,
                 deploy_path: str = None,
                 log_path:    str = None,
                 run_id:      str = None):

        self.deploy_path = Path(deploy_path) \
            if deploy_path else None
        self.log_path    = Path(log_path) \
            if log_path \
            else Path(__file__).parent.parent / \
                 'history' / 'decision_log.jsonl'
        self.run_id      = run_id or \
            datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

        # Load current baseline
        self.baseline    = self._load_baseline()

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_baseline(self) -> Dict:
        """Load current production strategy as baseline."""
        if self.deploy_path is None:
            return self._default_baseline()

        strategy_file = self.deploy_path / 'active_strategy.json'
        if not strategy_file.exists():
            return self._default_baseline()

        try:
            with open(strategy_file) as f:
                strategy = json.load(f)
            return {
                'strategy_id':       strategy.get('strategy_id',
                                                   'unknown'),
                'composite_score':   strategy.get(
                    'composite_score', 0.40
                ),
                'first_run':         False,
            }
        except Exception:
            return self._default_baseline()

    def _default_baseline(self) -> Dict:
        """Default baseline when no production strategy exists."""
        return {
            'strategy_id':    None,
            'composite_score':0.0,
            'first_run':      True,
        }

    def _calculate_improvement(self,
                                score: float) -> Optional[float]:
        """Calculate % improvement over baseline."""
        baseline = self.baseline['composite_score']
        if baseline == 0:
            return None
        return (score - baseline) / baseline * 100

    def _check_auto_promote_conditions(self,
                                        wf_summary: Dict,
                                        score: CompositeScore
                                        ) -> List[str]:
        """
        Check all auto-promote conditions.
        Returns list of failed conditions (empty = all pass).
        """
        failures = []

        calmar = wf_summary.get('median_calmar_ratio', 0)
        if calmar < AUTO_PROMOTE_CONDITIONS['min_calmar']:
            failures.append(
                f"Calmar {calmar:.2f} < "
                f"{AUTO_PROMOTE_CONDITIONS['min_calmar']}"
            )

        pf = wf_summary.get('median_profit_factor', 0)
        if pf < AUTO_PROMOTE_CONDITIONS['min_profit_factor']:
            failures.append(
                f"PF {pf:.3f} < "
                f"{AUTO_PROMOTE_CONDITIONS['min_profit_factor']}"
            )

        dd = wf_summary.get('median_max_drawdown_pct', 100)
        if dd > AUTO_PROMOTE_CONDITIONS['max_drawdown_pct']:
            failures.append(
                f"Drawdown {dd:.1f}% > "
                f"{AUTO_PROMOTE_CONDITIONS['max_drawdown_pct']}%"
            )

        pw = wf_summary.get('profitable_window_rate', 0)
        if pw < AUTO_PROMOTE_CONDITIONS['min_profitable_windows']:
            failures.append(
                f"Profitable windows "
                f"{pw*100:.1f}% < "
                f"{AUTO_PROMOTE_CONDITIONS['min_profitable_windows']*100:.0f}%"
            )

        rc = wf_summary.get('regime_consistency', 0)
        if rc < AUTO_PROMOTE_CONDITIONS['min_regime_consistency']:
            failures.append(
                f"Regime consistency "
                f"{rc:.2f} < "
                f"{AUTO_PROMOTE_CONDITIONS['min_regime_consistency']}"
            )

        if score.sensitivity_status == 'FRAGILE':
            failures.append("Sensitivity: FRAGILE")

        return failures

    def decide(self,
               score:      CompositeScore,
               wf_summary: Dict,
               gate_result:GateResult) -> PromotionDecision:
        """
        Make a promotion decision for one candidate.

        Args:
            score:       composite score result
            wf_summary:  aggregated walk-forward metrics
            gate_result: result of hard gate filtering

        Returns:
            PromotionDecision with full reasoning
        """
        candidate_id     = score.candidate_id
        composite        = score.composite
        baseline_score   = self.baseline['composite_score']
        improvement_pct  = self._calculate_improvement(composite)
        first_run        = self.baseline['first_run']
        flags            = []

        # ── Gate failure → Auto Reject ─────────────────
        if not gate_result.passed:
            decision = PromotionDecision(
                candidate_id    = candidate_id,
                composite_score = composite,
                baseline_score  = baseline_score,
                improvement_pct = improvement_pct,
                decision_path   = 'AUTO_REJECT',
                decision_outcome= 'REJECTED',
                flags           = [
                    f"Gate failed: {gate_result.failure_reason}"
                ],
                recommendation  = (
                    f"Strategy rejected at hard gate: "
                    f"{gate_result.failed_gate}. "
                    f"{gate_result.failure_reason}."
                )
            )
            self._record(decision, gate_result)
            return decision

        # ── First run logic ────────────────────────────
        if first_run:
            if composite >= FIRST_RUN_AUTO_THRESHOLD:
                path, outcome = 'AUTO_PROMOTE', 'PROMOTED'
                rec = (
                    "First run. Strategy meets auto-promote "
                    "threshold for initial deployment."
                )
            elif composite >= FIRST_RUN_FLAG_THRESHOLD:
                path, outcome = 'FLAG_REVIEW', 'FLAGGED'
                flags.append(
                    "First run — no baseline to compare against"
                )
                rec = (
                    "First run. Score is marginal. "
                    "Review before deploying."
                )
            else:
                path, outcome = 'AUTO_REJECT', 'REJECTED'
                rec = (
                    "First run. Score too low for deployment. "
                    f"Score {composite:.3f} < "
                    f"minimum {FIRST_RUN_FLAG_THRESHOLD}."
                )

            decision = PromotionDecision(
                candidate_id    = candidate_id,
                composite_score = composite,
                baseline_score  = baseline_score,
                improvement_pct = None,
                decision_path   = path,
                decision_outcome= outcome,
                flags           = flags,
                recommendation  = rec
            )
            self._record(decision, gate_result)
            return decision

        # ── Doesn't beat baseline → Auto Reject ────────
        if improvement_pct is None or improvement_pct < 0:
            decision = PromotionDecision(
                candidate_id    = candidate_id,
                composite_score = composite,
                baseline_score  = baseline_score,
                improvement_pct = improvement_pct,
                decision_path   = 'AUTO_REJECT',
                decision_outcome= 'REJECTED',
                flags           = ['Does not beat baseline'],
                recommendation  = (
                    f"Candidate score {composite:.3f} does not "
                    f"beat baseline {baseline_score:.3f}. "
                    f"Current strategy remains active."
                )
            )
            self._record(decision, gate_result)
            return decision

        # ── Check auto-promote conditions ──────────────
        auto_failures = self._check_auto_promote_conditions(
            wf_summary, score
        )

        # ── Auto Promote ───────────────────────────────
        if (improvement_pct >= AUTO_PROMOTE_THRESHOLD * 100
                and not auto_failures):
            decision = PromotionDecision(
                candidate_id    = candidate_id,
                composite_score = composite,
                baseline_score  = baseline_score,
                improvement_pct = improvement_pct,
                decision_path   = 'AUTO_PROMOTE',
                decision_outcome= 'PROMOTED',
                flags           = [],
                recommendation  = (
                    f"Strategy beats baseline by "
                    f"{improvement_pct:.1f}% and passes all "
                    f"auto-promote conditions. Deploying."
                )
            )
            self._record(decision, gate_result)
            if self.deploy_path:
                self._deploy(decision, wf_summary)
            return decision

        # ── Flag for Review ────────────────────────────
        # Beats baseline but doesn't fully qualify for auto
        flags = []
        if improvement_pct < AUTO_PROMOTE_THRESHOLD * 100:
            flags.append(
                f"Improvement {improvement_pct:.1f}% < "
                f"{AUTO_PROMOTE_THRESHOLD*100:.0f}% threshold"
            )
        flags.extend(auto_failures)

        rec_parts = [
            f"Strategy improves on baseline by "
            f"{improvement_pct:.1f}% but has concerns:"
        ]
        rec_parts.extend([f"  • {f}" for f in flags])
        rec_parts.append("Review and decide manually.")

        decision = PromotionDecision(
            candidate_id    = candidate_id,
            composite_score = composite,
            baseline_score  = baseline_score,
            improvement_pct = improvement_pct,
            decision_path   = 'FLAG_REVIEW',
            decision_outcome= 'FLAGGED',
            flags           = flags,
            recommendation  = '\n'.join(rec_parts)
        )
        self._record(decision, gate_result)
        return decision

    def _deploy(self, decision: PromotionDecision,
                wf_summary: Dict) -> None:
        """Write strategy file to deploy path."""
        if self.deploy_path is None:
            return

        self.deploy_path.mkdir(parents=True, exist_ok=True)

        # Archive current strategy
        active = self.deploy_path / 'active_strategy.json'
        if active.exists():
            archive_dir = self.deploy_path / 'archive'
            archive_dir.mkdir(exist_ok=True)
            ts = datetime.now(timezone.utc).strftime(
                '%Y%m%d_%H%M%S'
            )
            archived = archive_dir / \
                f"{self.baseline['strategy_id']}_{ts}.json"
            active.rename(archived)

        # Write new strategy
        strategy = {
            'strategy_id':        decision.candidate_id,
            'composite_score':    decision.composite_score,
            'promoted_at':        datetime.now(
                timezone.utc
            ).isoformat(),
            'promoted_by':        'trainer_auto',
            'improvement_pct':    decision.improvement_pct,
            'baseline_strategy':  self.baseline['strategy_id'],
            'wf_summary':         wf_summary,
        }

        # Atomic write
        tmp = active.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(strategy, f, indent=2)
        tmp.rename(active)

    def _record(self, decision: PromotionDecision,
                gate_result: GateResult) -> None:
        """Append decision to promotion log."""
        record = {
            'decision_id':       hashlib.sha256(
                f"{self.run_id}_{decision.candidate_id}"
                .encode()
            ).hexdigest()[:12],
            'run_id':            self.run_id,
            'decided_at':        datetime.now(
                timezone.utc
            ).isoformat(),
            'candidate_id':      decision.candidate_id,
            'composite_score':   decision.composite_score,
            'baseline_score':    decision.baseline_score,
            'improvement_pct':   decision.improvement_pct,
            'decision_path':     decision.decision_path,
            'decision_outcome':  decision.decision_outcome,
            'flags':             decision.flags,
            'gate_results':      gate_result.gate_details,
            'strategy_deployed': decision.decision_outcome
                                 == 'PROMOTED',
        }

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')