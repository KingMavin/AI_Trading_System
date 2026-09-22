"""
Concurrency Lock Manager for Trainer Pipeline — Fix 9.5.
Prevents multiple Trainer instances from running concurrently.
Provides PID-based staleness detection for crash recovery.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from shared.config import OUTPUT_ROOT

log = logging.getLogger(__name__)

LOCK_FILE = OUTPUT_ROOT / 'state' / 'trainer.lock'


def is_pid_running(pid: int) -> bool:
    """Check if a process with given PID is currently running."""
    if pid <= 0:
        return False
    if PSUTIL_AVAILABLE:
        return psutil.pid_exists(pid)
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, AttributeError):
            return False


class ConcurrencyLock:
    """
    File-based concurrency lock for Trainer processes.
    Acquires lock file on startup, releases on exit.
    Detects stale locks from crashed processes.
    """

    def __init__(self, lock_path: Optional[Path] = None):
        self.lock_path = lock_path or LOCK_FILE
        self.acquired = False

    def acquire(self) -> None:
        """
        Acquire concurrency lock. Fails loudly if active process holds lock.
        Clears stale locks from dead processes.
        """
        if self.lock_path.exists():
            try:
                with open(self.lock_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                holder_pid = data.get('pid', -1)
                started_at = data.get('started_at', 'UNKNOWN')

                if is_pid_running(holder_pid):
                    log.error(
                        f"CONCURRENCY_LOCK_VIOLATION: Active Trainer process (PID {holder_pid}, started {started_at}) "
                        f"holds lock at {self.lock_path}."
                    )
                    raise RuntimeError(
                        f"TRAINER_CONCURRENCY_LOCK_ACTIVE: Another Trainer instance (PID {holder_pid}) "
                        f"is currently running."
                    )
                else:
                    log.warning(
                        f"STALE_LOCK_DETECTED: Lock holder PID {holder_pid} (started {started_at}) "
                        f"is no longer running. Removing stale lock file {self.lock_path}."
                    )
                    self.lock_path.unlink()

            except (json.JSONDecodeError, KeyError) as parse_err:
                log.warning(f"CORRUPT_LOCK_FILE: Lock file {self.lock_path} corrupt ({parse_err}). Removing.")
                self.lock_path.unlink()

        # Write lock file atomically
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_data = {
            'pid': os.getpid(),
            'started_at': datetime.now(timezone.utc).isoformat(),
        }
        tmp_lock = self.lock_path.with_suffix('.lock.tmp')
        with open(tmp_lock, 'w', encoding='utf-8') as f:
            json.dump(lock_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if self.lock_path.exists():
            self.lock_path.unlink()
        tmp_lock.rename(self.lock_path)

        self.acquired = True
        log.info(f"Acquired concurrency lock for PID {os.getpid()} at {self.lock_path}")

    def release(self) -> None:
        """Release concurrency lock."""
        if self.acquired and self.lock_path.exists():
            try:
                self.lock_path.unlink()
                self.acquired = False
                log.info(f"Released concurrency lock at {self.lock_path}")
            except Exception as e:
                log.warning(f"Error releasing lock file {self.lock_path}: {e}")

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
