"""
Strategy Loader — Engine component.

Reads the active strategy file deployed by the Trainer.
Validates integrity. Reloads when file changes.
Never crashes on bad file — falls back to last known good.

The Engine calls load_strategy() on startup and
reload_if_changed() on every candle close.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Default strategy file location
DEFAULT_STRATEGY_PATH = (
    Path(__file__).parent.parent /
    'strategy' / 'active_strategy.json'
)


@dataclass
class StrategySpec:
    """
    Fully loaded and validated strategy specification.
    The Engine uses this to generate signals.
    """
    strategy_id:        str
    template:           str
    symbol:             str
    timeframe:          str
    parameters:         Dict
    filters:            Dict
    composite_score:    float
    promoted_at:        str
    file_hash:          str

    # Convenience accessors
    @property
    def fast_ma(self) -> int:
        return self.parameters.get('fast_ma_period', 10)

    @property
    def slow_ma(self) -> int:
        return self.parameters.get('slow_ma_period', 50)

    @property
    def sl_atr_multiple(self) -> float:
        return self.parameters.get('sl_atr_multiple', 1.5)

    @property
    def tp_rr_ratio(self) -> float:
        return self.parameters.get('tp_rr_ratio', 2.0)

    @property
    def risk_per_trade_pct(self) -> float:
        return self.parameters.get('risk_per_trade_pct', 1.0)

    @property
    def pip_size(self) -> float:
        return self.parameters.get('pip_size', 0.0001)


class StrategyLoader:
    """
    Loads and monitors the active strategy file.
    Thread-safe reads. Graceful degradation on errors.
    """

    def __init__(self, strategy_path: str = None):
        self.path           = Path(strategy_path) \
                              if strategy_path \
                              else DEFAULT_STRATEGY_PATH
        self.current_spec:  Optional[StrategySpec] = None
        self.last_hash:     str                     = ''
        self.last_checked:  Optional[datetime]      = None
        self.load_errors:   int                     = 0

    def _compute_file_hash(self) -> str:
        """SHA-256 of file contents."""
        try:
            content = self.path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            return ''

    def _load_from_file(self) -> Optional[StrategySpec]:
        """
        Load and validate strategy from JSON file.
        Returns None on any error — never raises.
        """
        if not self.path.exists():
            log.warning(
                f"Strategy file not found: {self.path}"
            )
            return None

        try:
            with open(self.path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            log.error(f"Strategy file is not valid JSON: {e}")
            self.load_errors += 1
            return None

        # Validate required fields
        required = [
            'strategy_id', 'template', 'composite_score'
        ]
        for field in required:
            if field not in data:
                log.error(
                    f"Strategy file missing field: {field}"
                )
                self.load_errors += 1
                return None

        # Extract parameters
        params = data.get('parameters', {})
        if not params:
            # Try to get from wf_summary best_params
            wf = data.get('wf_summary', {})
            params = wf.get('best_params', {})

        file_hash = self._compute_file_hash()

        spec = StrategySpec(
            strategy_id     = data['strategy_id'],
            template        = data.get('template',
                                       'ma_crossover'),
            symbol          = data.get('symbol', 'EURUSD'),
            timeframe       = data.get('timeframe', 'M15'),
            parameters      = params,
            filters         = data.get('filters', {}),
            composite_score = data.get('composite_score', 0.0),
            promoted_at     = data.get('promoted_at', ''),
            file_hash       = file_hash,
        )

        log.info(
            f"Strategy loaded: {spec.strategy_id} | "
            f"score={spec.composite_score:.3f} | "
            f"template={spec.template}"
        )
        return spec

    def load_strategy(self) -> Optional[StrategySpec]:
        """
        Load strategy on startup.
        Called once when Engine starts.
        """
        self.current_spec = self._load_from_file()
        self.last_hash    = self._compute_file_hash()
        self.last_checked = datetime.now(timezone.utc)

        if self.current_spec is None:
            log.warning(
                "No strategy loaded. Engine will run in "
                "monitoring-only mode until strategy deployed."
            )
        return self.current_spec

    def reload_if_changed(self) -> bool:
        """
        Check if strategy file has changed since last load.
        Returns True if a new strategy was loaded.
        Called on every candle close (lightweight — hash only).
        """
        self.last_checked = datetime.now(timezone.utc)
        current_hash      = self._compute_file_hash()

        if current_hash == self.last_hash:
            return False  # no change

        log.info(
            f"Strategy file changed. "
            f"Reloading... "
            f"(was: {self.last_hash}, now: {current_hash})"
        )

        new_spec = self._load_from_file()
        if new_spec is not None:
            old_id             = (
                self.current_spec.strategy_id
                if self.current_spec else 'none'
            )
            self.current_spec  = new_spec
            self.last_hash     = current_hash
            self.load_errors   = 0
            log.info(
                f"Strategy reloaded: "
                f"{old_id} → {new_spec.strategy_id}"
            )
            return True
        else:
            log.error(
                "New strategy file failed validation. "
                "Keeping previous strategy."
            )
            return False

    def get_spec(self) -> Optional[StrategySpec]:
        """Return current loaded strategy spec."""
        return self.current_spec

    def is_loaded(self) -> bool:
        """Return True if a valid strategy is loaded."""
        return self.current_spec is not None

    def get_status(self) -> Dict:
        """Return loader status for dashboard/logging."""
        return {
            'strategy_id':  (
                self.current_spec.strategy_id
                if self.current_spec else None
            ),
            'template':     (
                self.current_spec.template
                if self.current_spec else None
            ),
            'score':        (
                self.current_spec.composite_score
                if self.current_spec else None
            ),
            'promoted_at':  (
                self.current_spec.promoted_at
                if self.current_spec else None
            ),
            'file_hash':    self.last_hash,
            'last_checked': (
                self.last_checked.isoformat()
                if self.last_checked else None
            ),
            'load_errors':  self.load_errors,
            'strategy_path':str(self.path),
        }