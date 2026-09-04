"""
Knowledge Base — the Trainer's memory across training runs.

Stores what worked, what didn't, and guides future searches
toward promising regions. Grows smarter with every run.

Storage: trainer_data/history/knowledge_base.json
Format:  JSON (structured object, not JSON-Lines)
Updated: after every training run (atomic write)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import shutil
import logging
log = logging.getLogger(__name__)
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


# ── DEFAULT KNOWLEDGE BASE STRUCTURE ──────────────────

def _empty_knowledge_base() -> Dict:
    return {
        'version':     '1.0',
        'created_at':  datetime.now(timezone.utc).isoformat(),
        'updated_at':  datetime.now(timezone.utc).isoformat(),
        'total_runs':  0,

        # Template performance per symbol
        # key: "ma_crossover_EURUSD"
        'template_performance': {},

        # Parameter insights
        # key: "ma_crossover_EURUSD_fast_ma_period"
        'parameter_insights': {},

        # Filter effectiveness
        # key: "session_LONDON_ma_crossover"
        'filter_effectiveness': {},

        # Regime patterns
        # key: "TRENDING"
        'regime_patterns': {},

        # Decision quality tracking
        'decision_quality': {
            'auto_promoted':         [],
            'flagged_promoted':      [],
            'flagged_rejected':      [],
            'auto_rejected':         [],
        },

        # Candidate hashes — for deduplication
        # key: config_hash, value: {score, run_id, data_hash}
        'candidate_results': {},
    }


class KnowledgeBase:
    """
    Persistent memory for the Trainer.
    Loads on startup, updates after each run, saves atomically.
    """

    def __init__(self, base_path: str = None):
        if base_path:
            self.path = Path(base_path)
        else:
            self.path = (
                Path(__file__).parent.parent /
                'trainer_data' / 'history' /
                'knowledge_base.json'
            )
        self.search_history_path = self.path.parent / 'search_history.jsonl'
        self.backup_dir = self.path.parent / 'backups'
        self.data       = self._load()

    # ── LOAD / SAVE ────────────────────────────────────

    def _load(self) -> Dict:
        """Load knowledge base from disk. Init if missing."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            kb = _empty_knowledge_base()
            self._save(kb)
            return kb

        try:
            with open(self.path) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"KB_LOAD_FAILED: Knowledge base load failed ({e}). Initialising fresh.")
            return _empty_knowledge_base()

    def _save(self, data: Dict = None) -> None:
        """Atomic write — temp file + rename."""
        data = data or self.data
        data['updated_at'] = datetime.now(timezone.utc).isoformat()

        # Backup before overwrite (keep last 5)
        if self.path.exists():
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            ts  = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            bak = self.backup_dir / f"kb_{ts}.json"
            shutil.copy2(self.path, bak)

            # Keep only last 5 backups
            backups = sorted(self.backup_dir.glob('kb_*.json'))
            for old in backups[:-5]:
                old.unlink(missing_ok=True)

        tmp = self.path.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        if self.path.exists():
            self.path.unlink()
        tmp.rename(self.path)

    def save(self) -> None:
        """Public save method."""
        self._save()

    # ── DEDUPLICATION & SEARCH HISTORY ─────────────────

    def has_been_tested(self, config_hash: str,
                        data_hash: str) -> bool:
        """
        Check if this exact config was tested on this data.
        Returns True only if BOTH hashes match.
        New data = different data_hash = must retest.
        """
        entry = self.data['candidate_results'].get(config_hash)
        if entry is None:
            return False
        return entry.get('data_hash') == data_hash

    def get_cached_score(self, config_hash: str,
                         data_hash: str) -> Optional[float]:
        """Return cached composite score if available."""
        entry = self.data['candidate_results'].get(config_hash)
        if entry and entry.get('data_hash') == data_hash:
            return entry.get('composite_score')
        return None

    def record_candidate(self, config_hash: str,
                         data_hash:   str,
                         run_id:      str,
                         composite_score: float,
                         template:    str,
                         symbol:      str,
                         params:      Dict) -> None:
        """Record a tested candidate in the knowledge base."""
        ignored_keys = {'pip_size', 'risk_per_trade_pct', 'warmup_candles'}
        clean_params = {k: v for k, v in params.items() if k not in ignored_keys}

        self.data['candidate_results'][config_hash] = {
            'data_hash':       data_hash,
            'run_id':          run_id,
            'tested_at':       datetime.now(
                timezone.utc
            ).isoformat(),
            'composite_score': composite_score,
            'template':        template,
            'symbol':          symbol,
            'params_summary':  clean_params,
        }

    def record_search_trial(self, template: str,
                            symbol: str,
                            run_id: str,
                            data_hash: str,
                            trial_number: int,
                            params: Dict,
                            is_coherent: bool,
                            composite_score: float,
                            in_sample_regime: str = 'UNKNOWN') -> None:
        """Append an Optuna search trial to search_history.jsonl."""
        ignored_keys = {'pip_size', 'risk_per_trade_pct', 'warmup_candles'}
        clean_params = {k: v for k, v in params.items() if k not in ignored_keys}

        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'run_id': run_id,
            'template': template,
            'symbol': symbol,
            'data_hash': data_hash,
            'trial_number': trial_number,
            'parameters': clean_params,
            'is_coherent': is_coherent,
            'composite_score': composite_score,
            'in_sample_regime': in_sample_regime,
        }

        try:
            self.search_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.search_history_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            log.warning(f"SEARCH_HISTORY_WRITE_FAILED: Failed to append search trial ({e})")

    def get_top_historical_candidates(self, template: str,
                                       symbol: str,
                                       top_k: int = 5,
                                       target_regime: str = 'UNKNOWN') -> List[Dict]:
        """
        Query search_history.jsonl and candidate_results to return top_k
        historical parameter combinations for (template, symbol) with score > 0.0.
        Prioritizes candidates matching target_regime, backfilling with others.
        """
        regime_candidates = []
        other_candidates = []

        # 1. Read from search_history.jsonl if present
        if self.search_history_path.exists():
            try:
                with open(self.search_history_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if (rec.get('template') == template and
                                rec.get('symbol') == symbol and
                                rec.get('composite_score', -1.0) > 0.0 and
                                rec.get('is_coherent', True)):
                            score_param = (rec['composite_score'], rec['parameters'])
                            if rec.get('in_sample_regime', 'UNKNOWN') == target_regime:
                                regime_candidates.append(score_param)
                            else:
                                other_candidates.append(score_param)
            except Exception as e:
                log.warning(f"SEARCH_HISTORY_READ_FAILED: ({e})")

        # 2. Read from knowledge_base.json candidate_results
        for entry in self.data.get('candidate_results', {}).values():
            if (entry.get('template') == template and
                    entry.get('symbol') == symbol and
                    entry.get('composite_score', -1.0) > 0.0):
                score_param = (entry['composite_score'], entry.get('params_summary', {}))
                # knowledge_base.json doesn't track regime, put in other
                other_candidates.append(score_param)

        regime_candidates.sort(key=lambda x: x[0], reverse=True)
        other_candidates.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate while preserving order
        final_candidates = []
        seen = set()
        
        for _, p in regime_candidates + other_candidates:
            h = str(sorted(p.items()))
            if h not in seen:
                seen.add(h)
                final_candidates.append(p)
                if len(final_candidates) >= top_k:
                    break

        if not final_candidates:
            log.info(
                f"KB_WARMSTART_SPARSE: 0 historical candidates found for template "
                f"'{template}' on symbol '{symbol}'. Proceeding with standard TPESampler search."
            )
            return []

        return final_candidates

    @staticmethod
    def bucket_parameters(template: str, params: Dict) -> Dict[str, str]:
        """
        Map parameter values to coarse region buckets.
        """
        buckets = {}
        if template == 'ma_crossover':
            fast = params.get('fast_ma_period', 10)
            if fast <= 15:
                buckets['fast_ma_period'] = 'SHORT'
            elif fast <= 30:
                buckets['fast_ma_period'] = 'MEDIUM'
            else:
                buckets['fast_ma_period'] = 'LONG'

            slow = params.get('slow_ma_period', 50)
            if slow <= 60:
                buckets['slow_ma_period'] = 'MEDIUM'
            elif slow <= 120:
                buckets['slow_ma_period'] = 'LONG'
            else:
                buckets['slow_ma_period'] = 'EXTRA_LONG'

            sl = params.get('sl_atr_multiple', 1.5)
            buckets['sl_atr_multiple'] = 'TIGHT' if sl <= 1.5 else 'WIDE'

            tp = params.get('tp_rr_ratio', 2.0)
            buckets['tp_rr_ratio'] = 'LOW' if tp <= 2.0 else 'HIGH'

            adx = params.get('adx_threshold', params.get('adx_min_threshold', 0))
            if adx == 0:
                buckets['adx_threshold'] = 'DISABLED'
            elif adx <= 15:
                buckets['adx_threshold'] = 'WEAK'
            elif adx <= 20:
                buckets['adx_threshold'] = 'MODERATE'
            else:
                buckets['adx_threshold'] = 'STRONG'

        return buckets

    def get_region_statistics(self, template: str,
                              symbol: str,
                              min_samples: int = 30) -> Dict[str, Any]:
        """
        Compute regional score statistics across historical trials.
        Only returns non-empty statistics if total_samples >= min_samples (N >= 30).
        """
        all_trials = []
        if self.search_history_path.exists():
            try:
                with open(self.search_history_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if rec.get('template') == template and rec.get('symbol') == symbol:
                            all_trials.append(rec)
            except Exception as e:
                log.warning(f"REGION_STATS_READ_FAILED: ({e})")

        if len(all_trials) < min_samples:
            return {
                'total_samples': len(all_trials),
                'min_samples_met': False,
                'region_scores': {},
            }

        region_scores: Dict[str, List[float]] = {}
        for trial in all_trials:
            score = trial.get('composite_score', -1.0)
            if score < 0.0:
                continue
            b_dict = self.bucket_parameters(template, trial.get('parameters', {}))
            region_key = "|".join(f"{k}={v}" for k, v in sorted(b_dict.items()))
            if region_key not in region_scores:
                region_scores[region_key] = []
            region_scores[region_key].append(score)

        summary = {}
        for r_key, scores in region_scores.items():
            if scores:
                summary[r_key] = {
                    'count': len(scores),
                    'mean_score': sum(scores) / len(scores),
                    'max_score': max(scores),
                }

        return {
            'total_samples': len(all_trials),
            'min_samples_met': True,
            'region_scores': summary,
        }

    # ── TEMPLATE PERFORMANCE ───────────────────────────

    def update_template_performance(self,
                                     template: str,
                                     symbol:   str,
                                     score:    float,
                                     promoted: bool) -> None:
        """Update running performance stats for a template."""
        key = f"{template}_{symbol}"
        if key not in self.data['template_performance']:
            self.data['template_performance'][key] = {
                'times_tested':          0,
                'times_promoted':        0,
                'best_score_ever':       0.0,
                'all_scores':            [],
                'last_tested':           None,
                'last_promoted':         None,
                'trending_direction':    'FLAT',
            }

        entry = self.data['template_performance'][key]
        entry['times_tested']   += 1
        entry['last_tested']     = datetime.now(
            timezone.utc
        ).isoformat()
        entry['all_scores'].append(score)

        # Keep only last 20 scores
        entry['all_scores'] = entry['all_scores'][-20:]

        if score > entry['best_score_ever']:
            entry['best_score_ever'] = score

        if promoted:
            entry['times_promoted'] += 1
            entry['last_promoted']   = datetime.now(
                timezone.utc
            ).isoformat()

        # Trending direction — compare recent 3 vs previous 3
        scores = entry['all_scores']
        if len(scores) >= 6:
            recent   = sum(scores[-3:]) / 3
            previous = sum(scores[-6:-3]) / 3
            if recent > previous * 1.10:
                entry['trending_direction'] = 'UP'
            elif recent < previous * 0.90:
                entry['trending_direction'] = 'DOWN'
            else:
                entry['trending_direction'] = 'FLAT'

    def get_template_trend(self, template: str,
                           symbol: str) -> Optional[str]:
        """Return trending direction for template+symbol."""
        key   = f"{template}_{symbol}"
        entry = self.data['template_performance'].get(key)
        if entry is None:
            return None
        return entry.get('trending_direction')

    def get_best_score_ever(self, template: str,
                             symbol: str) -> float:
        """Return best composite score ever achieved."""
        key   = f"{template}_{symbol}"
        entry = self.data['template_performance'].get(key)
        if entry is None:
            return 0.0
        return entry.get('best_score_ever', 0.0)

    # ── PARAMETER INSIGHTS ─────────────────────────────

    def record_parameter_result(self,
                                 template:  str,
                                 symbol:    str,
                                 parameter: str,
                                 value:     Any,
                                 score:     float) -> None:
        """
        Record which parameter values produce good scores.
        Builds up insight over many runs.
        """
        key = f"{template}_{symbol}_{parameter}"
        if key not in self.data['parameter_insights']:
            self.data['parameter_insights'][key] = {
                'value_scores': {},
                'best_value':   None,
                'confidence':   'LOW',
            }

        entry = self.data['parameter_insights'][key]
        str_val = str(value)

        if str_val not in entry['value_scores']:
            entry['value_scores'][str_val] = []

        entry['value_scores'][str_val].append(score)

        # Keep last 10 scores per value
        entry['value_scores'][str_val] = \
            entry['value_scores'][str_val][-10:]

        # Find best value
        best_val   = None
        best_score = -1.0
        for val, scores in entry['value_scores'].items():
            if scores:
                avg = sum(scores) / len(scores)
                if avg > best_score:
                    best_score = avg
                    best_val   = val

        entry['best_value'] = best_val

        # Confidence based on test count
        total_tests = sum(
            len(v) for v in entry['value_scores'].values()
        )
        if total_tests >= 15:
            entry['confidence'] = 'HIGH'
        elif total_tests >= 5:
            entry['confidence'] = 'MEDIUM'
        else:
            entry['confidence'] = 'LOW'

    def get_parameter_insight(self,
                               template:  str,
                               symbol:    str,
                               parameter: str) -> Optional[Dict]:
        """
        Return parameter insights if confidence is sufficient.
        Returns None if insufficient history.
        """
        key   = f"{template}_{symbol}_{parameter}"
        entry = self.data['parameter_insights'].get(key)
        if entry is None:
            return None
        if entry['confidence'] == 'LOW':
            return None     # not enough data to guide search
        return entry

    def get_good_parameter_values(self,
                                   template:  str,
                                   symbol:    str,
                                   parameter: str,
                                   top_n: int = 3) -> List:
        """
        Return top N parameter values by historical score.
        Used to bias Bayesian search toward good regions.
        """
        key   = f"{template}_{symbol}_{parameter}"
        entry = self.data['parameter_insights'].get(key)
        if entry is None or entry['confidence'] == 'LOW':
            return []

        # Rank values by average score
        ranked = sorted(
            [
                (val, sum(scores)/len(scores))
                for val, scores in entry['value_scores'].items()
                if scores
            ],
            key=lambda x: x[1],
            reverse=True
        )
        return [val for val, _ in ranked[:top_n]]

    # ── FILTER EFFECTIVENESS ───────────────────────────

    def record_filter_result(self,
                              filter_name: str,
                              template:    str,
                              improved:    bool,
                              delta:       float) -> None:
        """Record whether a filter improved performance."""
        key = f"{filter_name}_{template}"
        if key not in self.data['filter_effectiveness']:
            self.data['filter_effectiveness'][key] = {
                'applied_count':   0,
                'improved_count':  0,
                'improvement_rate':0.0,
                'avg_improvement': 0.0,
                'deltas':          [],
            }

        entry = self.data['filter_effectiveness'][key]
        entry['applied_count'] += 1
        if improved:
            entry['improved_count'] += 1
        entry['deltas'].append(delta)
        entry['deltas'] = entry['deltas'][-20:]

        entry['improvement_rate'] = (
            entry['improved_count'] / entry['applied_count']
        )
        entry['avg_improvement'] = (
            sum(entry['deltas']) / len(entry['deltas'])
        )

    def get_filter_improvement_rate(self,
                                     filter_name: str,
                                     template:    str
                                     ) -> Optional[float]:
        """
        Return how often this filter improves performance.
        Returns None if insufficient history.
        """
        key   = f"{filter_name}_{template}"
        entry = self.data['filter_effectiveness'].get(key)
        if entry is None or entry['applied_count'] < 5:
            return None
        return entry['improvement_rate']

    # ── REGIME PATTERNS ────────────────────────────────

    def update_regime_pattern(self,
                               regime:   str,
                               template: str,
                               pf:       float,
                               win_rate: float) -> None:
        """Update which templates work best per regime."""
        if regime not in self.data['regime_patterns']:
            self.data['regime_patterns'][regime] = {}

        if template not in self.data['regime_patterns'][regime]:
            self.data['regime_patterns'][regime][template] = {
                'pf_values':       [],
                'win_rate_values': [],
                'avg_pf':          0.0,
                'avg_win_rate':    0.0,
            }

        entry = self.data['regime_patterns'][regime][template]
        entry['pf_values'].append(pf)
        entry['win_rate_values'].append(win_rate)
        entry['pf_values']       = entry['pf_values'][-20:]
        entry['win_rate_values'] = entry['win_rate_values'][-20:]
        entry['avg_pf']       = (
            sum(entry['pf_values']) / len(entry['pf_values'])
        )
        entry['avg_win_rate'] = (
            sum(entry['win_rate_values']) /
            len(entry['win_rate_values'])
        )

    def get_best_template_for_regime(self,
                                      regime: str
                                      ) -> Optional[str]:
        """
        Return template that historically performs best
        in the given regime. Returns None if no data.
        """
        patterns = self.data['regime_patterns'].get(regime)
        if not patterns:
            return None

        best_template = max(
            patterns.items(),
            key=lambda x: x[1].get('avg_pf', 0),
            default=(None, {})
        )
        return best_template[0] if best_template[0] else None

    # ── SELF ASSESSMENT ────────────────────────────────

    def get_run_trend(self, metric: str,
                      n_runs: int = 5) -> List[float]:
        """
        Return last N values for a given run metric.
        Used by Trainer self-assessment.
        """
        history = self.data.get('run_history', [])
        values  = [
            r.get(metric, 0)
            for r in history[-n_runs:]
            if metric in r
        ]
        return values

    def record_run_summary(self, run_summary: Dict) -> None:
        """Append a run summary to run history."""
        if 'run_history' not in self.data:
            self.data['run_history'] = []
        self.data['run_history'].append(run_summary)
        # Keep last 50 runs
        self.data['run_history'] = self.data['run_history'][-50:]
        self.data['total_runs'] += 1

    # ── STATISTICS ─────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return summary statistics about the knowledge base."""
        return {
            'total_runs':         self.data['total_runs'],
            'candidates_tested':  len(
                self.data['candidate_results']
            ),
            'templates_tracked':  len(
                self.data['template_performance']
            ),
            'parameters_tracked': len(
                self.data['parameter_insights']
            ),
            'filters_tracked':    len(
                self.data['filter_effectiveness']
            ),
            'regimes_tracked':    len(
                self.data['regime_patterns']
            ),
        }

    def print_summary(self) -> None:
        """Print knowledge base summary to console."""
        stats = self.get_stats()
        print(f"\n{'='*50}")
        print(f"  KNOWLEDGE BASE SUMMARY")
        print(f"{'='*50}")
        print(f"  Total training runs:   {stats['total_runs']}")
        print(f"  Candidates tested:     {stats['candidates_tested']}")
        print(f"  Templates tracked:     {stats['templates_tracked']}")
        print(f"  Parameters tracked:    {stats['parameters_tracked']}")
        print(f"  Filters tracked:       {stats['filters_tracked']}")
        print(f"  Regimes tracked:       {stats['regimes_tracked']}")

        # Template trends
        for key, entry in \
                self.data['template_performance'].items():
            trend = entry.get('trending_direction', 'FLAT')
            best  = entry.get('best_score_ever', 0)
            tests = entry.get('times_tested', 0)
            promo = entry.get('times_promoted', 0)
            arrow = {'UP':'▲','DOWN':'▼','FLAT':'─'}.get(
                trend, '─'
            )
            print(f"\n  {key}:")
            print(f"    Tested: {tests}x | "
                  f"Promoted: {promo}x | "
                  f"Best: {best:.3f} {arrow}")

        print(f"{'='*50}\n")