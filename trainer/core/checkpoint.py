"""
Atomic Checkpoint Management for Trainer Pipeline — Fix 9.1.
Enables crash-resilient overnight training runs and exact resume.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set

from shared.config import OUTPUT_ROOT

log = logging.getLogger(__name__)

CHECKPOINT_FILE = OUTPUT_ROOT / 'state' / 'trainer_checkpoint.json'


def atomic_write_json(file_path: Path, data: Dict) -> None:
    """Atomic write of JSON dictionary via temporary file + fsync + rename."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix('.json.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    if file_path.exists():
        file_path.unlink()
    tmp_path.rename(file_path)


class CheckpointManager:
    """
    Manages trainer execution state checkpoints.
    Fails loudly on corrupt checkpoint files.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_FILE
        self.run_id: str = ""
        self.completed_symbols: Set[str] = set()
        self.completed_windows: Dict[str, Set[int]] = {}
        self.completed_candidates: Dict[str, Set[str]] = {}
        self.candidate_results: Dict[str, Dict[str, Any]] = {}

    def load(self) -> bool:
        """
        Load checkpoint from disk. Fails loudly on corrupt JSON/structure.
        Returns True if valid checkpoint loaded, False if file does not exist.
        """
        if not self.checkpoint_path.exists():
            return False

        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for key in ['run_id', 'completed_symbols', 'completed_windows', 'completed_candidates']:
                if key not in data:
                    raise ValueError(f"Missing required key '{key}' in checkpoint")

            self.run_id = data['run_id']
            self.completed_symbols = set(data['completed_symbols'])
            self.completed_windows = {
                k: set(v) for k, v in data.get('completed_windows', {}).items()
            }
            self.completed_candidates = {
                k: set(v) for k, v in data.get('completed_candidates', {}).items()
            }
            self.candidate_results = data.get('candidate_results', {})
            log.info(f"Loaded valid checkpoint for run {self.run_id} from {self.checkpoint_path}")
            return True

        except Exception as e:
            log.error(f"CORRUPT_CHECKPOINT_ERROR: Failed to load checkpoint {self.checkpoint_path}: {e}")
            raise RuntimeError(f"Corrupt checkpoint file {self.checkpoint_path}: {e}. Halting run.")

    def save(self, run_id: str) -> None:
        """Save current checkpoint atomically."""
        self.run_id = run_id
        data = {
            'run_id': self.run_id,
            'completed_symbols': sorted(list(self.completed_symbols)),
            'completed_windows': {
                k: sorted(list(v)) for k, v in self.completed_windows.items()
            },
            'completed_candidates': {
                k: sorted(list(v)) for k, v in self.completed_candidates.items()
            },
            'candidate_results': self.candidate_results,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        try:
            atomic_write_json(self.checkpoint_path, data)
        except Exception as e:
            log.error(f"CHECKPOINT_WRITE_FAILED: Failed to save checkpoint to {self.checkpoint_path}: {e}", exc_info=True)
            raise RuntimeError(f"CHECKPOINT_WRITE_FAILED: Failed to save checkpoint to {self.checkpoint_path}: {e}")

    def mark_candidate_completed(self, symbol: str, window_idx: int, candidate_id: str, result: Dict) -> None:
        """Record completed candidate and save checkpoint."""
        key = f"{symbol}_w{window_idx}"
        if key not in self.completed_candidates:
            self.completed_candidates[key] = set()
        self.completed_candidates[key].add(candidate_id)
        self.candidate_results[candidate_id] = result

    def is_candidate_completed(self, symbol: str, window_idx: int, candidate_id: str) -> bool:
        """Check if candidate evaluation was already completed."""
        key = f"{symbol}_w{window_idx}"
        return candidate_id in self.completed_candidates.get(key, set())

    def mark_window_completed(self, symbol: str, window_idx: int) -> None:
        """Record completed walk-forward window."""
        if symbol not in self.completed_windows:
            self.completed_windows[symbol] = set()
        self.completed_windows[symbol].add(window_idx)

    def is_window_completed(self, symbol: str, window_idx: int) -> bool:
        """Check if walk-forward window was completed."""
        return window_idx in self.completed_windows.get(symbol, set())

    def mark_symbol_completed(self, symbol: str) -> None:
        """Record completed symbol."""
        self.completed_symbols.add(symbol)

    def clear(self) -> None:
        """Remove checkpoint file on clean run completion."""
        if self.checkpoint_path.exists():
            try:
                self.checkpoint_path.unlink()
                log.info(f"Cleared completed checkpoint file {self.checkpoint_path}")
            except Exception as e:
                log.warning(f"Could not clear checkpoint file: {e}")
