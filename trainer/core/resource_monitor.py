"""
Resource Watchdog for Trainer Pipeline — Fix 9.2.
Monitors system RAM and Disk space during long-running training runs.
Enforces GREEN / AMBER / RED / CRITICAL resource thresholds.
"""

import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import json
from datetime import datetime, timezone
from shared.config import LOG_DIR, OUTPUT_ROOT
from trainer.core.checkpoint import CheckpointManager, CHECKPOINT_FILE

log = logging.getLogger(__name__)

PERFORMANCE_LOG_FILE = LOG_DIR / 'performance.jsonl'


def log_performance_record(source_system: str, metric_type: str, data: Dict) -> None:
    r"""
    Appends a performance record to D:\work\files\logs\performance.jsonl.
    Zero-risk: wrapped in try/except so logging errors never disrupt calling logic.
    """
    try:
        PERFORMANCE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source_system': source_system,
            'metric_type': metric_type,
            **data
        }
        with open(PERFORMANCE_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        log.warning(f"Failed to write performance log record: {e}")


# Resource Thresholds (Percentages)
# Baseline hardware: 16 GB RAM total.
# RAM % Used: GREEN < 80%, AMBER 80-90%, RED 90-96%, CRITICAL >= 96% (leaves ~640 MB free RAM)
# Disk % Used: GREEN < 85%, AMBER 85-92%, RED 92-97%, CRITICAL >= 97%
RAM_AMBER = 80.0
RAM_RED = 90.0
RAM_CRITICAL = 96.0

DISK_AMBER = 85.0
DISK_RED = 92.0
DISK_CRITICAL = 97.0


def check_resource_status(path_for_disk: Optional[Path] = None) -> Tuple[str, float, float]:
    """
    Check current RAM % and Disk % usage.
    Returns (status: 'GREEN'|'AMBER'|'RED'|'CRITICAL', ram_pct: float, disk_pct: float).
    """
    ram_pct = 0.0
    if PSUTIL_AVAILABLE:
        ram_pct = psutil.virtual_memory().percent
    else:
        log.warning("psutil module not available; memory monitoring falling back to 0%")

    target_path = path_for_disk or Path.cwd()
    total, used, free = shutil.disk_usage(target_path)
    disk_pct = (used / total) * 100.0 if total > 0 else 0.0

    if ram_pct >= RAM_CRITICAL or disk_pct >= DISK_CRITICAL:
        status = 'CRITICAL'
    elif ram_pct >= RAM_RED or disk_pct >= DISK_RED:
        status = 'RED'
    elif ram_pct >= RAM_AMBER or disk_pct >= DISK_AMBER:
        status = 'AMBER'
    else:
        status = 'GREEN'

    return status, ram_pct, disk_pct


class ResourceWatchdog:
    """
    Enforces resource thresholds during training.
    Triggers atomic checkpoint and clean halt on CRITICAL status.
    Continuous logging to performance.jsonl.
    """

    def __init__(self, checkpoint_manager: CheckpointManager, path_for_disk: Optional[Path] = None):
        self.checkpoint_manager = checkpoint_manager
        self.path_for_disk = path_for_disk or Path.cwd()

    def evaluate(self, run_id: str) -> str:
        """
        Evaluate resource usage.
        - GREEN: Normal.
        - AMBER: Log warning, continue.
        - RED: Log warning, signal caller to pause new candidates.
        - CRITICAL: Save atomic checkpoint immediately and raise RuntimeError.
        """
        status, ram_pct, disk_pct = check_resource_status(self.path_for_disk)

        # Continuous performance logging (Fix 10.3)
        log_performance_record(
            source_system='TRAINER',
            metric_type='RESOURCE_USAGE',
            data={
                'run_id': run_id,
                'ram_pct': ram_pct,
                'disk_pct': disk_pct,
                'status': status,
            }
        )

        if status == 'AMBER':
            log.warning(
                f"RESOURCE_WATCHDOG_AMBER: RAM usage at {ram_pct:.1f}%, Disk usage at {disk_pct:.1f}%."
            )
        elif status == 'RED':
            log.warning(
                f"RESOURCE_WATCHDOG_RED: RAM usage at {ram_pct:.1f}%, Disk usage at {disk_pct:.1f}%. "
                f"Pausing new candidate starts; finishing in-flight work."
            )
        elif status == 'CRITICAL':
            log.error(
                f"RESOURCE_WATCHDOG_CRITICAL: RAM usage at {ram_pct:.1f}%, Disk usage at {disk_pct:.1f}%. "
                f"EMERGENCY ATOMIC CHECKPOINTING AND CLEAN HALT."
            )
            # Save atomic checkpoint before halting
            self.checkpoint_manager.save(run_id)
            raise RuntimeError(
                f"RESOURCE_MONITOR_CRITICAL: System RAM ({ram_pct:.1f}%) or Disk ({disk_pct:.1f}%) "
                f"exceeded critical thresholds. Checkpoint saved. Run halted cleanly."
            )

        return status
