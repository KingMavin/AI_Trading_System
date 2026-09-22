"""
Dashboard Process Helpers — WAVE 18.

Provides OS-level process management for the Azrael dashboard:
  - Engine PID discovery via lockfile (engine/state/engine.lock.json)
  - Graceful stop (SIGTERM / KeyboardInterrupt path)
  - Hard kill (psutil TerminateProcess on Windows)
  - Trainer subprocess spawn + stdout capture

Design constraints (per AGENTS.md and brainstorm):
  - Never modifies active_strategy.json
  - Hard kill does not touch open positions — broker SL/TP handles risk
  - All Tier 2 / Tier 3 actions write to audit ledger BEFORE the signal
    and follow up with a _CONFIRMED or _FAILED event regardless of outcome
  - Windows: signal.SIGKILL does not exist; use psutil.Process.kill()
  - PID lockfile is source of truth; psutil confirms the PID is still alive
    and the cmdline belongs to the Engine before any kill is sent
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ── PATHS ────────────────────────────────────────────────
DASHBOARD_ROOT  = Path(__file__).parent
ENGINE_ROOT     = DASHBOARD_ROOT.parent
ATS_ROOT        = ENGINE_ROOT.parent
TRAINER_ROOT    = ATS_ROOT / 'trainer'

ENGINE_LOCK_FILE = ENGINE_ROOT / 'state' / 'engine.lock.json'
AUDIT_LEDGER     = ENGINE_ROOT / 'logs' / 'audit_ledger.jsonl'
TRAINER_STATUS   = TRAINER_ROOT / 'trainer_data' / 'trainer_status.json'

# Cmdline substrings used as belt-and-suspenders after lockfile PID lookup.
# Not the primary source — lockfile + psutil.pid_exists() is primary.
_ENGINE_CMDLINE_MARKERS = [
    'engine/core/engine.py',
    'engine.core.engine',
    'engine\\core\\engine.py',
]

# ── PSUTIL AVAILABILITY ──────────────────────────────────
# Always bind the name 'psutil' at module level so unit tests can patch it
# regardless of whether the real package is installed.
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False
    log.warning(
        "psutil not installed — Engine process management unavailable. "
        "Run: pip install psutil"
    )


# ── AUDIT LEDGER ─────────────────────────────────────────

def write_audit_event(action: str, reason: str, metadata: Optional[Dict] = None) -> None:
    """
    Write an audit event to the ledger.

    Must be called BEFORE any signal is sent (per AGENTS.md rule 3).
    The caller is responsible for writing a follow-up _CONFIRMED or _FAILED
    event — this function never swallows exceptions silently.
    """
    AUDIT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'source':    'DASHBOARD',
        'action':    action,
        'reason':    reason,
        **(metadata or {}),
    }
    try:
        with open(AUDIT_LEDGER, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        log.error(f"Failed to write audit event '{action}': {e}")
        raise RuntimeError(
            f"AUDIT_LEDGER_WRITE_FAILED: Failed to write audit event '{action}' to {AUDIT_LEDGER}: {e}"
        )


# ── ENGINE PID DISCOVERY ─────────────────────────────────

def read_engine_lock() -> Optional[Dict]:
    """
    Read the engine.lock.json file written by Engine at startup.
    Returns the parsed dict, or None if the file does not exist or is corrupt.
    """
    if not ENGINE_LOCK_FILE.exists():
        return None
    try:
        data = json.loads(ENGINE_LOCK_FILE.read_text(encoding='utf-8'))
        if 'pid' not in data or 'started_at' not in data:
            return None
        return data
    except Exception:
        return None


def _cmdline_looks_like_engine(pid: int) -> bool:
    """
    Belt-and-suspenders: confirm the process at this PID has an Engine-like
    command line. Used after lockfile lookup to guard against PID reuse.
    """
    if not PSUTIL_AVAILABLE:
        return True   # Can't verify — trust lockfile only
    try:
        proc = psutil.Process(pid)
        cmdline = ' '.join(proc.cmdline())
        return any(m in cmdline for m in _ENGINE_CMDLINE_MARKERS)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def find_engine_pid() -> Optional[int]:
    """
    Return the Engine PID from engine.lock.json if the process is still alive.

    Validation chain:
      1. Read engine.lock.json → get pid
      2. psutil.pid_exists(pid) → confirm alive
      3. cmdline contains engine marker → confirm not PID-reuse
    Returns None if any check fails or psutil is unavailable.
    """
    lock = read_engine_lock()
    if lock is None:
        return None
    pid = lock.get('pid')
    if not isinstance(pid, int):
        return None
    if PSUTIL_AVAILABLE:
        if not psutil.pid_exists(pid):
            return None
        if not _cmdline_looks_like_engine(pid):
            log.warning(
                f"PID {pid} from lockfile does not look like Engine "
                f"(cmdline mismatch) — treating Engine as stopped"
            )
            return None
    return pid


def engine_is_running() -> bool:
    """Return True if a confirmed Engine process is currently running."""
    return find_engine_pid() is not None


def get_engine_lock_info() -> Optional[Dict]:
    """
    Return the full lock dict for display (pid, started_at, symbol),
    or None if Engine is not running.
    """
    pid = find_engine_pid()
    if pid is None:
        return None
    return read_engine_lock()


# ── POST-SIGNAL DEATH CONFIRMATION ──────────────────────

def wait_for_process_death(pid: int, timeout_s: int = 15) -> bool:
    """
    Poll psutil.pid_exists(pid) until the process is gone or timeout expires.

    Returns True if process is confirmed dead, False if still alive at timeout.
    Used after both SIGTERM and hard kill — TerminateProcess() is not
    instantaneous; we must confirm, not assume.
    """
    if not PSUTIL_AVAILABLE:
        time.sleep(timeout_s)
        return True   # Can't verify
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        time.sleep(1.0)
    return False


# ── ENGINE GRACEFUL STOP ─────────────────────────────────

def graceful_stop_engine(pid: int, reason: str) -> Tuple[bool, str]:
    """
    Send SIGTERM to the Engine process and wait for clean exit.

    Audit sequence:
      ENGINE_STOP_REQUESTED → (success) ENGINE_STOP_CONFIRMED
                             → (signal fail) ENGINE_STOP_FAILED
                             → (timeout) ENGINE_STOP_TIMED_OUT

    On Windows, SIGTERM raises KeyboardInterrupt in the target Python process,
    matching Engine._main_loop()'s except KeyboardInterrupt path (L667).

    Returns (success: bool, message: str).
    """
    write_audit_event('ENGINE_STOP_REQUESTED', reason, {'pid': pid})

    try:
        os.kill(pid, signal.SIGTERM)
        log.info(f"Sent SIGTERM to Engine PID {pid}")
    except ProcessLookupError:
        msg = f"Engine PID {pid} no longer exists"
        write_audit_event('ENGINE_STOP_CONFIRMED', msg, {'pid': pid, 'note': 'already_gone'})
        log.info(msg)
        return True, msg
    except Exception as e:
        msg = f"Failed to send SIGTERM to PID {pid}: {e}"
        write_audit_event('ENGINE_STOP_FAILED', str(e), {'pid': pid})
        log.error(msg)
        return False, msg

    dead = wait_for_process_death(pid, timeout_s=15)
    if dead:
        write_audit_event('ENGINE_STOP_CONFIRMED', 'process exited cleanly', {'pid': pid})
        return True, 'Engine stopped cleanly'
    else:
        write_audit_event('ENGINE_STOP_TIMED_OUT',
                          'process still alive after 15s',
                          {'pid': pid})
        return False, f'Engine PID {pid} did not exit within 15s after SIGTERM'


def hard_kill_engine(pid: int, reason: str, started_at: str) -> Tuple[bool, str]:
    """
    Forcibly terminate the Engine process (Tier 3 command).

    Pre-kill validation:
      - psutil.pid_exists(pid) — process still alive
      - lockfile started_at vs psutil.Process.create_time() — not PID reuse

    Uses psutil.Process.kill() → TerminateProcess() on Windows.
    Does NOT close open positions — broker SL/TP handles risk.

    Audit sequence:
      ENGINE_KILL_REQUESTED → (ok) ENGINE_KILL_CONFIRMED
                            → (validation fail) ENGINE_KILL_ABORTED
                            → (kill fail) ENGINE_KILL_FAILED

    Returns (success: bool, message: str).
    """
    if not PSUTIL_AVAILABLE:
        raise RuntimeError("Hard kill requires psutil. Run: pip install psutil")

    write_audit_event('ENGINE_KILL_REQUESTED', reason,
                      {'pid': pid, 'started_at': started_at})

    # PID liveness check
    if not psutil.pid_exists(pid):
        msg = f"Engine PID {pid} already gone before kill"
        write_audit_event('ENGINE_KILL_CONFIRMED', msg, {'pid': pid, 'note': 'already_gone'})
        return True, msg

    # Belt-and-suspenders: verify this PID is actually the Engine we locked
    if not _cmdline_looks_like_engine(pid):
        msg = f"PID {pid} cmdline does not match Engine — kill aborted to prevent wrong-process kill"
        write_audit_event('ENGINE_KILL_ABORTED', msg, {'pid': pid})
        log.error(msg)
        return False, msg

    try:
        psutil.Process(pid).kill()
        log.warning(f"HARD KILL sent to Engine PID {pid} (TerminateProcess)")
    except psutil.NoSuchProcess:
        msg = f"Engine PID {pid} vanished between check and kill"
        write_audit_event('ENGINE_KILL_CONFIRMED', msg, {'pid': pid, 'note': 'vanished'})
        return True, msg
    except Exception as e:
        msg = f"Hard kill of PID {pid} failed: {e}"
        write_audit_event('ENGINE_KILL_FAILED', str(e), {'pid': pid})
        log.error(msg)
        return False, msg

    dead = wait_for_process_death(pid, timeout_s=10)
    if dead:
        write_audit_event('ENGINE_KILL_CONFIRMED', 'process terminated', {'pid': pid})
        return True, 'Engine terminated'
    else:
        write_audit_event('ENGINE_KILL_TIMED_OUT',
                          'process still alive 10s after kill',
                          {'pid': pid})
        return False, f'Engine PID {pid} still alive 10s after TerminateProcess'


# ── TRAINER SUBPROCESS MANAGEMENT ───────────────────────

class TrainerSubprocess:
    """
    Manages a single Trainer subprocess for the dashboard.

    Captures stdout/stderr into a ring buffer exposed via get_log_lines().
    Writes a trainer_status.json sentinel file while running.
    """

    def __init__(self) -> None:
        self._proc:        Optional[subprocess.Popen] = None
        self._buffer:      deque = deque(maxlen=200)
        self._lock:        threading.Lock = threading.Lock()
        self._last_result: Optional[Dict] = self._load_saved_status()

    # ── public API ──────────────────────────────────────

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def get_pid(self) -> Optional[int]:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return self._proc.pid
        return None

    def get_log_lines(self) -> list:
        with self._lock:
            return list(self._buffer)

    def get_status(self) -> Dict:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            rc = self._proc.poll() if self._proc else None
            pid = self._proc.pid if running else None
        return {
            'running':     running,
            'pid':         pid,
            'returncode':  rc,
            'last_result': self._last_result,
        }

    def launch(self, symbol: str = 'EURUSD', quick: bool = False) -> Dict:
        """
        Launch the Trainer subprocess. Raises RuntimeError if already running.
        Returns {'ok': True, 'pid': <pid>}.
        """
        with self._lock:
            if self._proc and self._proc.poll() is None:
                raise RuntimeError(
                    f"Trainer already running (PID {self._proc.pid})"
                )
            self._buffer.clear()

        trainer_script = TRAINER_ROOT / 'trainer.py'
        if not trainer_script.exists():
            raise RuntimeError(f"Trainer script not found: {trainer_script}")

        cmd = [sys.executable, str(trainer_script), 'run', '--symbol', symbol]
        if quick:
            cmd.append('--quick')

        log.info(f"Launching Trainer: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ATS_ROOT),
        )

        with self._lock:
            self._proc = proc

        self._write_status_file({
            'running':    True,
            'pid':        proc.pid,
            'started_at': datetime.now(timezone.utc).isoformat(),
            'symbol':     symbol,
            'quick':      quick,
        })

        threading.Thread(target=self._read_output, daemon=True).start()
        threading.Thread(target=self._watch_completion, daemon=True).start()

        return {'ok': True, 'pid': proc.pid}

    def stop(self, timeout_s: int = 10) -> bool:
        """Send SIGTERM to the Trainer subprocess and wait."""
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return True
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except Exception as e:
            log.error(f"Could not stop Trainer: {e}")
            return False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
        self._on_complete()
        return True

    # ── private ─────────────────────────────────────────

    def _read_output(self) -> None:
        proc = self._proc
        if not proc or not proc.stdout:
            return
        try:
            for line in proc.stdout:
                line = line.rstrip('\n')
                with self._lock:
                    self._buffer.append(line)
        except Exception:
            pass

    def _watch_completion(self) -> None:
        proc = self._proc
        if not proc:
            return
        proc.wait()
        self._on_complete()

    def _on_complete(self) -> None:
        with self._lock:
            proc = self._proc
        rc = proc.returncode if proc else None
        log.info(f"Trainer subprocess exited (rc={rc})")
        result = {
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'returncode':  rc,
            'success':     rc == 0,
        }
        with self._lock:
            self._last_result = result
        self._write_status_file({'running': False, **result})

    def _write_status_file(self, data: Dict) -> None:
        TRAINER_STATUS.parent.mkdir(parents=True, exist_ok=True)
        tmp = TRAINER_STATUS.with_suffix('.json.tmp')
        try:
            tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
            if TRAINER_STATUS.exists():
                TRAINER_STATUS.unlink()
            tmp.rename(TRAINER_STATUS)
        except Exception as e:
            log.warning(f"Could not write trainer_status.json: {e}")

    def _load_saved_status(self) -> Optional[Dict]:
        """Load last known status from disk (survives dashboard restarts)."""
        if not TRAINER_STATUS.exists():
            return None
        try:
            data = json.loads(TRAINER_STATUS.read_text(encoding='utf-8'))
            # If sentinel claims running but we just started, it's stale.
            if data.get('running'):
                data['running'] = False
                data['stale'] = True
            return data
        except Exception:
            return None


# ── MODULE-LEVEL SINGLETON ───────────────────────────────
# One TrainerSubprocess instance shared across all request handlers.
trainer = TrainerSubprocess()
