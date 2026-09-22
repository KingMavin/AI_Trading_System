"""
Strategy Bench Store — WAVE 19.

Manages strategy lifecycle state across:
  - IDEAS:     Un-optimized parameter hypothesis or template definitions
  - TESTING:   Mid-pipeline validation in Trainer
  - APPROVED:  Passed 10-gate evaluation and ready for deployment
  - DEPLOYED:  Currently active strategy deployed to engine/strategy/active_strategy.json
  - ARCHIVED:  Previously deployed strategy retired after replacement or CUSUM trip
  - REJECTED:  Failed hard gate evaluation

Persists state to trainer/bench/strategy_bench.json.
Retains complete strategy history (no auto-pruning).
Tagging for manual stubs (promoted_by: "manual_test_stub") is preserved.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from shared.dna import compute_dna_hash

log = logging.getLogger(__name__)

BENCH_DIR = Path(__file__).parent.parent / 'bench'
BENCH_FILE = BENCH_DIR / 'strategy_bench.json'


class StrategyBenchStore:
    """Manages the strategy bench store and lifecycle transitions."""

    def __init__(self, bench_file: Optional[Path] = None):
        self.bench_file = bench_file or BENCH_FILE
        self.bench_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_init()

    def _ensure_init(self) -> None:
        """Initialise bench file if missing."""
        if not self.bench_file.exists():
            stub_params = {
                'fast_ma_period': 20,
                'slow_ma_period': 100,
                'ma_type': 'SMA',
                'sl_atr_multiple': 2.0,
                'tp_rr_ratio': 2.0,
                'risk_per_trade_pct': 0.5,
                'pip_size': 0.0001,
                'warmup_candles': 250,
                'adx_min_threshold': 20,
                'exit_on_opposite_crossover': False
            }
            stub_filters = {}
            stub_dna = compute_dna_hash(stub_params, stub_filters)

            initial_data = {
                'bench_version': '1.0',
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'active_strategy_id': 'ATS-MA-MANUAL-001',
                'strategies': [
                    {
                        'strategy_id': 'ATS-MA-MANUAL-001',
                        'dna_hash': stub_dna,
                        'status': 'DEPLOYED',
                        'template': 'ma_crossover',
                        'symbol': 'EURUSD',
                        'composite_score': 1.25,
                        'promoted_at': '2026-05-27T00:00:00+00:00',
                        'promoted_by': 'manual_test_stub',
                        'parameters': stub_params,
                        'filters': stub_filters,
                        'wf_summary': {
                            'median_win_rate': 0.55,
                            'median_profit_factor': 1.5,
                            'profitable_window_rate': 0.60
                        },
                        'notes': 'Manual baseline strategy stub for testing.'
                    }
                ]
            }
            self._write_file(initial_data)

    def _read_file(self) -> Dict:
        try:
            with open(self.bench_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to read bench store: {e}")
            raise RuntimeError(
                f"BENCH_STORE_CORRUPT: Failed to parse benchmark file {self.bench_file}: {e}. "
                "Training and promotion halted to prevent unvalidated candidate promotion."
            )

    def _write_file(self, data: Dict) -> None:
        data['updated_at'] = datetime.now(timezone.utc).isoformat()
        tmp = self.bench_file.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.bench_file)

    def get_all(self) -> List[Dict]:
        """Return all strategies on bench."""
        return self._read_file().get('strategies', [])

    def get_by_status(self, status: str) -> List[Dict]:
        """Return strategies matching status."""
        return [s for s in self.get_all() if s.get('status') == status.upper()]

    def get_by_id(self, strategy_id: str) -> Optional[Dict]:
        """Find a strategy by strategy_id."""
        for s in self.get_all():
            if s.get('strategy_id') == strategy_id:
                return s
        return None

    def add_or_update(self, strategy_data: Dict) -> None:
        """Add a new candidate or update an existing candidate."""
        data = self._read_file()
        strategies = data.get('strategies', [])
        strat_id = strategy_data['strategy_id']
        
        params = strategy_data.get('parameters', {})
        filters = strategy_data.get('filters', {})
        if 'dna_hash' not in strategy_data:
            strategy_data['dna_hash'] = compute_dna_hash(params, filters)

        found = False
        for idx, s in enumerate(strategies):
            if s.get('strategy_id') == strat_id:
                strategies[idx] = {**s, **strategy_data}
                found = True
                break

        if not found:
            strategies.append(strategy_data)

        data['strategies'] = strategies
        self._write_file(data)

    def deploy(self, strategy_id: str, deploy_dir: Path) -> bool:
        """
        Deploy candidate to active_strategy.json.
        Updates deployed candidate status to DEPLOYED.
        Updates previous DEPLOYED candidate status to ARCHIVED.
        """
        candidate = self.get_by_id(strategy_id)
        if not candidate:
            log.error(f"Cannot deploy strategy '{strategy_id}': not found in bench store")
            return False

        data = self._read_file()
        strategies = data.get('strategies', [])

        # Update lifecycle statuses (no auto-pruning: all retained)
        for s in strategies:
            if s.get('status') == 'DEPLOYED' and s.get('strategy_id') != strategy_id:
                s['status'] = 'ARCHIVED'
                s['archived_at'] = datetime.now(timezone.utc).isoformat()
            if s.get('strategy_id') == strategy_id:
                s['status'] = 'DEPLOYED'
                s['deployed_at'] = datetime.now(timezone.utc).isoformat()

        data['active_strategy_id'] = strategy_id
        data['strategies'] = strategies
        self._write_file(data)

        # Archive outgoing active_strategy.json file on disk before replacing
        deploy_dir.mkdir(parents=True, exist_ok=True)
        target = deploy_dir / 'active_strategy.json'
        if target.exists():
            archive_dir = deploy_dir / 'archive'
            archive_dir.mkdir(parents=True, exist_ok=True)
            ts_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    old_strat = json.load(f)
                old_id = old_strat.get('strategy_id', 'unknown')
                archive_file = archive_dir / f"{old_id}_{ts_str}.json"
                with open(archive_file, 'w', encoding='utf-8') as f:
                    json.dump(old_strat, f, indent=2)
                log.info(f"Archived outgoing strategy file to {archive_file}")
            except Exception as e:
                log.warning(f"Could not archive outgoing strategy file: {e}")

        # Format payload for StrategyLoader compatibility
        params = candidate.get('parameters', {})
        filters = candidate.get('filters', {})
        dna_hash = candidate.get('dna_hash') or compute_dna_hash(params, filters)

        deploy_payload = {
            'strategy_id': candidate['strategy_id'],
            'dna_hash': dna_hash,
            'template': candidate.get('template', 'ma_crossover'),
            'symbol': candidate.get('symbol', 'EURUSD'),
            'timeframe': candidate.get('timeframe', 'M15'),
            'composite_score': candidate.get('composite_score', 0.0),
            'promoted_at': candidate.get('promoted_at', datetime.now(timezone.utc).isoformat()),
            'promoted_by': candidate.get('promoted_by', 'trainer_auto'),
            'parameters': params,
            'wf_summary': candidate.get('wf_summary', {}),
            'filters': filters,
            'notes': candidate.get('notes', 'Deployed from strategy bench.')
        }

        tmp_target = target.with_suffix('.tmp')
        with open(tmp_target, 'w', encoding='utf-8') as f:
            json.dump(deploy_payload, f, indent=2)
        tmp_target.replace(target)
        log.info(f"Deployed strategy '{strategy_id}' to {target}")
        return True
