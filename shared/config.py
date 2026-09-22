"""
Central configuration for ATS.
Single source of truth for output locations and account settings.
Everything the system writes goes under OUTPUT_ROOT on the second
drive — never the primary drive.
"""
import os
import shutil
from pathlib import Path

# Override with ATS_OUTPUT_ROOT env var if the drive ever changes.
OUTPUT_ROOT = Path(os.environ.get("ATS_OUTPUT_ROOT", r"D:\work\files"))

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
LOG_DIR        = OUTPUT_ROOT / "logs"
KB_DIR         = OUTPUT_ROOT / "knowledge_base"
REPORT_DIR     = OUTPUT_ROOT / "reports"
DATA_DIR       = OUTPUT_ROOT / "historical"
DECISION_DIR   = OUTPUT_ROOT / "decisions"
STATE_DIR      = OUTPUT_ROOT / "state"

ALL_DIRS = [CHECKPOINT_DIR, LOG_DIR, KB_DIR, REPORT_DIR,
            DATA_DIR, DECISION_DIR, STATE_DIR]

# Demo is EUR; prop account will be USD. Change this one value
# (or set ATS_ACCOUNT_CCY) when you switch — the conversion
# layer reads it.
ACCOUNT_CURRENCY = os.environ.get("ATS_ACCOUNT_CCY", "EUR")

# Trainer stops gracefully if free space drops below this.
MIN_FREE_DISK_GB = 5.0


def ensure_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def free_disk_gb(path=None):
    path = Path(path) if path else OUTPUT_ROOT
    usage = shutil.disk_usage(path.anchor)
    return usage.free / (1024 ** 3)


def disk_ok():
    return free_disk_gb() >= MIN_FREE_DISK_GB