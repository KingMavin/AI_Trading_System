"""
Azrael Local Dashboard — FastAPI backend (WAVE 18).

Bound strictly to 127.0.0.1:8080. Single-user, localhost-only. No auth/CSRF.

Endpoints:
  GET  /                          → dashboard HTML
  GET  /api/status                → full system status JSON
  GET  /api/logs/{source}         → log tail (source: engine|trades|audit|trainer)
  GET  /api/trainer/status        → trainer subprocess state
  GET  /api/trainer/log           → last N lines from trainer stdout
  POST /api/command/engine/stop   → Tier 1: graceful SIGTERM
  POST /api/command/engine/kill   → Tier 3: hard kill (requires confirmation phrase)
  POST /api/command/engine/restart         → Tier 2: graceful stop + relaunch
  POST /api/command/engine/clear_safe_mode → Tier 2: restart to clear SAFE_MODE
  POST /api/command/engine/reload_strategy → existing: kept as-is
  POST /api/command/trainer/run   → Tier 1: launch trainer subprocess
  POST /api/command/trainer/stop  → Tier 2: SIGTERM trainer subprocess
  GET  /api/health                → liveness check

Safety rules (per AGENTS.md):
  - Never writes active_strategy.json
  - Tier 3 hard kill: exact phrase required at API layer (not just UI)
  - Audit event written BEFORE signal, follow-up written after (CONFIRMED/FAILED)
  - Post-kill: polls psutil.pid_exists() to confirm death, not assumed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
import os
import signal
import subprocess
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ── DEPENDENCY CHECKS ──────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ── PATHS ───────────────────────────────────────────────
ENGINE_ROOT      = Path(__file__).parent.parent
ATS_ROOT         = ENGINE_ROOT.parent
TRAINER_ROOT     = ATS_ROOT / 'trainer'

STRATEGY_PATH    = ENGINE_ROOT / 'strategy'  / 'active_strategy.json'
DEGRADATION_PATH = ENGINE_ROOT / 'state'     / 'degradation_status.json'
STATE_PATH       = ENGINE_ROOT / 'state'     / 'engine_state.json'
CUSUM_PATH       = ENGINE_ROOT / 'state'     / 'cusum_state.json'
ENGINE_LOCK_PATH = ENGINE_ROOT / 'state'     / 'engine.lock.json'
SHUTDOWN_FLAG    = ENGINE_ROOT / 'state'     / 'shutdown_flag.json'
LOGS_PATH        = ENGINE_ROOT / 'logs'
AUDIT_LEDGER     = ENGINE_ROOT / 'logs'      / 'audit_ledger.jsonl'
DECISION_LOG     = TRAINER_ROOT / 'trainer_data' / 'history' / 'decision_log.jsonl'
KB_PATH          = TRAINER_ROOT / 'trainer_data' / 'history' / 'knowledge_base.json'

TRAINER_SCRIPT   = TRAINER_ROOT / 'trainer.py'

# Hard-kill confirmation phrase (validated at API layer)
KILL_PHRASE = "KILL AZRAEL ENGINE"

# ── PROCESS HELPERS IMPORT ──────────────────────────────
from engine.dashboard.process_helpers import (
    find_engine_pid,
    engine_is_running,
    get_engine_lock_info,
    graceful_stop_engine,
    hard_kill_engine,
    write_audit_event,
    trainer as _trainer,
)


# ── DATA READERS ────────────────────────────────────────

def read_json_safe(path: Path, default: Optional[Dict] = None) -> Dict:
    """Read JSON file safely. Returns default on any error."""
    default = default or {}
    if not path.exists():
        return default
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not read {path}: {e}")
        return default


def tail_text_file(path: Path, n: int = 100) -> List[str]:
    """Return the last N lines of a text file."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        return lines[-n:]
    except Exception as e:
        log.warning(f"Could not tail {path}: {e}")
        return []


def tail_jsonl_file(path: Path, n: int = 50) -> List[Dict]:
    """Return the last N valid JSON objects from a .jsonl file."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        result = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
                if len(result) >= n:
                    break
            except json.JSONDecodeError:
                continue
        return list(reversed(result))
    except Exception as e:
        log.warning(f"Could not read jsonl {path}: {e}")
        return []


def read_recent_trades(n: int = 50) -> List[Dict]:
    """Read most recent N trade records from *.jsonl log files."""
    trades: List[Dict] = []
    log_files = sorted(LOGS_PATH.glob('*.jsonl'), reverse=True)
    for log_file in log_files[:3]:
        if len(trades) >= n:
            break
        try:
            lines = log_file.read_text(encoding='utf-8', errors='replace').splitlines()
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if 'trade_id' in record:
                        trades.append(record)
                    if len(trades) >= n:
                        break
                except json.JSONDecodeError:
                    continue
        except Exception:
            continue
    return trades


def build_equity_curve(trades: List[Dict], initial_equity: float = 10000.0) -> List[Dict]:
    """Build equity curve from trade list."""
    if not trades:
        return [{'time': datetime.now(timezone.utc).isoformat(), 'equity': initial_equity}]
    sorted_trades = sorted(
        [t for t in trades if 'exit_time' in t],
        key=lambda x: x.get('exit_time', '')
    )
    equity = initial_equity
    curve = [{'time': sorted_trades[0].get('entry_time', ''), 'equity': equity}] if sorted_trades else []
    for trade in sorted_trades:
        equity += trade.get('net_pnl', 0)
        curve.append({'time': trade.get('exit_time', ''), 'equity': round(equity, 2)})
    return curve


def get_last_decision() -> Optional[Dict]:
    """Return the last entry from the Trainer decision log."""
    entries = tail_jsonl_file(DECISION_LOG, n=1)
    return entries[0] if entries else None


def get_engine_status() -> Dict:
    """Compile full system status from all state files."""
    strategy    = read_json_safe(STRATEGY_PATH)
    degradation = read_json_safe(DEGRADATION_PATH)
    state       = read_json_safe(STATE_PATH)
    cusum       = read_json_safe(CUSUM_PATH)
    lock_info   = get_engine_lock_info()
    trainer_st  = _trainer.get_status()
    last_dec    = get_last_decision()

    # Performance from recent trades
    trades      = read_recent_trades(100)
    recent_50   = trades[:50]
    winners     = [t for t in recent_50 if t.get('net_pnl', 0) > 0]
    total_pnl   = sum(t.get('net_pnl', 0) for t in recent_50)
    win_rate    = len(winners) / len(recent_50) if recent_50 else 0.0
    gross_profit = sum(t.get('net_pnl', 0) for t in recent_50 if t.get('net_pnl', 0) > 0)
    gross_loss   = abs(sum(t.get('net_pnl', 0) for t in recent_50 if t.get('net_pnl', 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Engine running state — lockfile is authoritative
    eng_running = lock_info is not None

    return {
        'timestamp':     datetime.now(timezone.utc).isoformat(),
        'engine': {
            'running':           eng_running,
            'pid':               lock_info.get('pid') if lock_info else None,
            'started_at':        lock_info.get('started_at') if lock_info else None,
            'symbol':            lock_info.get('symbol') if lock_info else None,
            'safe_mode':         state.get('safe_mode', False),
            'circuit_breaker':   state.get('circuit_breaker_tripped', False),
            'algo_trading_on':   state.get('algo_trading_enabled', None),
            'open_position':     state.get('open_position'),
            'session':           state.get('session', 'UNKNOWN'),
            'regime':            state.get('regime', 'UNKNOWN'),
            'equity':            state.get('equity', 0),
            'updated_at':        state.get('updated_at'),
            'checklist':         state.get('checklist'),
        },
        'cusum': {
            'tripped':   cusum.get('cusum_tripped', False),
            'reason':    cusum.get('cusum_reason', ''),
            's_i':       cusum.get('s_i', 0),
            'h':         cusum.get('h', 0),
            'p0':        cusum.get('p0', 0),
            'win_rate':  cusum.get('win_rate', 0),
            'n_trades':  cusum.get('n_trades', 0),
        },
        'strategy': {
            'id':          strategy.get('strategy_id', 'No strategy loaded'),
            'dna_hash':    strategy.get('dna_hash', ''),
            'template':    strategy.get('template', 'unknown'),
            'score':       strategy.get('composite_score', 0),
            'promoted_at': strategy.get('promoted_at', ''),
            'parameters':  strategy.get('parameters', {}),
        },
        'degradation': {
            'status':            degradation.get('status', 'OK'),
            'alert_count':       degradation.get('alert_count', 0),
            'rolling_pf':        degradation.get('rolling_pf', 0),
            'rolling_wr':        degradation.get('rolling_win_rate', 0),
            'loss_streak':       degradation.get('current_loss_streak', 0),
            'rerun_recommended': degradation.get('rerun_recommended', False),
        },
        'performance': {
            'total_trades':  len(trades),
            'recent_trades': len(recent_50),
            'win_rate':      round(win_rate, 4),
            'total_pnl':     round(total_pnl, 2),
            'profit_factor': round(profit_factor, 3),
        },
        'trainer': {
            'running':      trainer_st.get('running', False),
            'pid':          trainer_st.get('pid'),
            'last_result':  trainer_st.get('last_result'),
            'last_decision': last_dec,
        },
    }


# ── PYDANTIC REQUEST MODELS ─────────────────────────────

if FASTAPI_AVAILABLE:
    class CommandRequest(BaseModel):
        reason: str = ''
        symbol: str = 'EURUSD'
        quick:  bool = False
        confirmation_phrase: str = ''
        strategy_id: str = ''


# ── FASTAPI APP ─────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Azrael Local Dashboard",
        description="ATS Engine & Trainer control panel — localhost only",
        version="2.0"
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )

    # ── STATUS & DATA ──────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root():
        html_path = Path(__file__).parent / 'static' / 'dashboard.html'
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding='utf-8'))
        return HTMLResponse("<h1>Dashboard HTML not found. Deploy static/dashboard.html</h1>")

    @app.get("/api/status")
    async def api_status():
        return JSONResponse(get_engine_status())

    @app.get("/api/trades")
    async def api_trades(n: int = 50):
        trades = read_recent_trades(n)
        return JSONResponse({'trades': trades})

    @app.get("/api/equity")
    async def api_equity():
        trades = read_recent_trades(200)
        curve  = build_equity_curve(trades)
        return JSONResponse({'equity_curve': curve})

    @app.get("/api/strategy")
    async def api_strategy():
        return JSONResponse(read_json_safe(STRATEGY_PATH))

    @app.get("/api/logs/dates")
    async def api_log_dates():
        """
        List available log dates from engine/logs directory.
        Returns a sorted list of unique log dates (newest first).
        """
        dates_set = set()
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        dates_set.add(today_str)

        if LOGS_PATH.exists():
            for p in LOGS_PATH.glob('engine_*.log'):
                name = p.stem  # engine_YYYYMMDD
                parts = name.split('_')
                if len(parts) >= 2 and len(parts[1]) == 8 and parts[1].isdigit():
                    ds = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}"
                    dates_set.add(ds)

            for p in LOGS_PATH.glob('*.jsonl'):
                name = p.stem
                for token in name.split('_'):
                    if len(token) == 8 and token.isdigit():
                        ds = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
                        dates_set.add(ds)

        sorted_dates = sorted(list(dates_set), reverse=True)
        dates_list = [
            {
                'date': d,
                'display': f"{d} (Today)" if d == today_str else d,
                'is_today': d == today_str
            }
            for d in sorted_dates
        ]
        return JSONResponse({'dates': dates_list})

    @app.get("/api/logs/{source}")
    async def api_logs(source: str, n: int = 150, date: Optional[str] = None):
        """
        Tail log lines.
        source: 'engine' | 'trades' | 'audit' | 'trainer'
        date: optional YYYY-MM-DD string for historical log selection
        """
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        target_date_compact = date.replace('-', '') if (date and len(date) == 10 and date.count('-') == 2) else None

        if source == 'engine':
            if target_date_compact and date != today_str:
                target_file = LOGS_PATH / f"engine_{target_date_compact}.log"
                if target_file.exists():
                    lines = tail_text_file(target_file, n=n)
                    return JSONResponse({'lines': lines, 'source': source, 'file': target_file.name, 'date': date})
                else:
                    return JSONResponse({'lines': [f"No engine log found for date {date}"], 'source': source, 'date': date})
            else:
                log_files = sorted(LOGS_PATH.glob('engine_*.log'), reverse=True)
                if not log_files:
                    return JSONResponse({'lines': [], 'source': source, 'date': today_str})
                lines = tail_text_file(log_files[0], n=n)
                return JSONResponse({'lines': lines, 'source': source, 'file': log_files[0].name, 'date': today_str})

        elif source == 'trades':
            if target_date_compact and date != today_str:
                matching_files = list(LOGS_PATH.glob(f"*{target_date_compact}*.jsonl"))
                if matching_files:
                    lines = tail_text_file(matching_files[0], n=n)
                    return JSONResponse({'lines': lines, 'source': source, 'date': date})
                else:
                    return JSONResponse({'lines': [f"No trade log found for date {date}"], 'source': source, 'date': date})
            else:
                log_files = sorted(LOGS_PATH.glob('*.jsonl'), reverse=True)
                all_lines: List[str] = []
                for lf in log_files[:2]:
                    all_lines = tail_text_file(lf, n=n) + all_lines
                return JSONResponse({'lines': all_lines[-n:], 'source': source, 'date': today_str})

        elif source == 'audit':
            lines = tail_text_file(AUDIT_LEDGER, n=n)
            if date:
                lines = [l for l in lines if date in l]
            return JSONResponse({'lines': lines, 'source': source, 'date': date or today_str})

        elif source == 'trainer':
            lines = _trainer.get_log_lines()[-n:]
            return JSONResponse({'lines': lines, 'source': source, 'date': date or today_str})

        else:
            raise HTTPException(status_code=400,
                                detail=f"Unknown log source '{source}'. "
                                       f"Valid: engine, trades, audit, trainer")

    @app.get("/api/trainer/status")
    async def api_trainer_status():
        return JSONResponse(_trainer.get_status())

    @app.get("/api/trainer/log")
    async def api_trainer_log(n: int = 100):
        return JSONResponse({'lines': _trainer.get_log_lines()[-n:]})

    @app.get("/api/health")
    async def api_health():
        return JSONResponse({
            'status':    'ok',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })

    # ── ADAPTATION & BENCH ENDPOINTS (WAVE 19) ─────────

    @app.get("/api/adaptation/latest")
    async def api_adaptation_latest():
        """
        Returns the latest Trainer decision from decision_log.jsonl,
        along with parameter diffs against current active strategy and 10-gate breakdown.
        """
        active_strat = read_json_safe(STRATEGY_PATH)
        records = tail_jsonl_file(DECISION_LOG, n=1)
        if not records:
            return JSONResponse({
                'has_decision': False,
                'message': 'No trainer decision records found in decision_log.jsonl',
                'active_strategy': active_strat,
            })

        latest = records[-1]
        active_params = active_strat.get('parameters', {})
        candidate_params = latest.get('parameters', {})
        if not candidate_params and 'gate_results' in latest:
            # Fallback parameter extraction if nested
            candidate_params = latest.get('candidate_parameters', {})

        param_diff = []
        all_keys = sorted(set(list(active_params.keys()) + list(candidate_params.keys())))
        for k in all_keys:
            act_val = active_params.get(k, '—')
            cand_val = candidate_params.get(k, '—')
            changed = str(act_val) != str(cand_val)
            param_diff.append({
                'parameter': k,
                'active_value': act_val,
                'candidate_value': cand_val,
                'changed': changed
            })

        # 10-Gate Threshold Mapping
        gate_thresholds = {
            'gate_1_total_trades': {'label': 'Gate 1: Total Trades', 'threshold': '>= 100'},
            'gate_2_trades_per_window': {'label': 'Gate 2: Trades / Window', 'threshold': '>= 10.0'},
            'gate_3_valid_windows': {'label': 'Gate 3: Valid Windows', 'threshold': '>= 5'},
            'gate_4_max_drawdown': {'label': 'Gate 4: Max Drawdown %', 'threshold': '<= 18.0%'},
            'gate_5_profit_factor': {'label': 'Gate 5: Profit Factor', 'threshold': '>= 1.25'},
            'gate_6_profitable_windows': {'label': 'Gate 6: Profitable Window Rate', 'threshold': '>= 60.0%'},
            'gate_7_consecutive_losses': {'label': 'Gate 7: Max Consecutive Losses', 'threshold': '<= 10'},
            'gate_8_regime_coverage': {'label': 'Gate 8: Regime Coverage', 'threshold': '>= 3'},
            'gate_9_mandate_compliant': {'label': 'Gate 9: Mandate Compliance', 'threshold': 'TRUE'},
            'gate_10_monte_carlo_stress': {'label': 'Gate 10: Monte Carlo Stress', 'threshold': '>= 0.0'}
        }

        gate_breakdown = []
        results = latest.get('gate_results', {})
        for g_key, g_meta in gate_thresholds.items():
            val = results.get(g_key, '—')
            # Determine pass/fail based on outcome flags or metric evaluation
            failed_flags = latest.get('flags', [])
            passed = not any(g_key in f for f in failed_flags) if isinstance(failed_flags, list) else True
            gate_breakdown.append({
                'gate': g_meta['label'],
                'metric_value': val,
                'threshold': g_meta['threshold'],
                'passed': passed
            })

        return JSONResponse({
            'has_decision': True,
            'decision': latest,
            'active_strategy': active_strat,
            'parameter_diff': param_diff,
            'gate_breakdown': gate_breakdown
        })

    @app.get("/api/bench")
    async def api_bench_list():
        """Returns all strategies on the strategy bench."""
        from trainer.core.bench import StrategyBenchStore
        store = StrategyBenchStore()
        return JSONResponse({'strategies': store.get_all()})

    @app.post("/api/bench/deploy")
    async def api_bench_deploy(body: CommandRequest):
        """
        Deploy an approved bench candidate to active_strategy.json.
        Enforces Part E Mutual Exclusion.
        """
        if _trainer.get_status().get('running'):
            raise HTTPException(
                status_code=409,
                detail="Cannot deploy bench candidate while Trainer is running. Stop Trainer first."
            )

        strategy_id = body.strategy_id or body.reason
        if not strategy_id:
            raise HTTPException(status_code=400, detail="Missing strategy_id in request body")

        from trainer.core.bench import StrategyBenchStore
        store = StrategyBenchStore()

        ok = store.deploy(strategy_id, STRATEGY_PATH.parent)
        if not ok:
            return JSONResponse({'ok': False, 'message': f"Strategy '{strategy_id}' not found on bench"})

        write_audit_event('BENCH_STRATEGY_DEPLOYED', f"Deployed strategy {strategy_id} from bench",
                          {'strategy_id': strategy_id})

        # If Engine is running, perform graceful restart so it picks up the new strategy immediately
        lock = get_engine_lock_info()
        restarted = False
        if lock is not None:
            pid = lock['pid']
            ok, stop_msg = graceful_stop_engine(pid, f"Redeploying to strategy {strategy_id}")
            if not ok:
                write_audit_event('BENCH_DEPLOY_RESTART_FAILED', stop_msg, {'strategy_id': strategy_id, 'pid': pid})
                return JSONResponse({
                    'ok': True,
                    'message': f"Strategy '{strategy_id}' deployed to bench, but engine stop failed: {stop_msg}. Relaunch aborted to prevent dual process.",
                    'restarted': False
                })
            new_pid = _spawn_engine(lock.get('symbol', 'EURUSD'))
            restarted = new_pid is not None
            if not restarted:
                write_audit_event('BENCH_DEPLOY_RESTART_FAILED', 'Engine spawn failed after stop', {'strategy_id': strategy_id})

        return JSONResponse({
            'ok': True,
            'message': f"Strategy '{strategy_id}' deployed successfully." + (" Engine restarted." if restarted else ""),
            'restarted': restarted
        })

    # ── ENGINE COMMANDS ─────────────────────────────────

    @app.post("/api/command/engine/start")
    async def cmd_engine_start(body: CommandRequest):
        """Tier 1 — 1-click Engine startup when offline. Enforces Mutual Exclusion with Trainer."""
        if _trainer.get_status().get('running'):
            raise HTTPException(
                status_code=409,
                detail="Cannot start Engine while Trainer pipeline is running. Wait for Trainer to finish or stop it first."
            )
        lock = get_engine_lock_info()
        if lock is not None:
            return JSONResponse({'ok': False, 'message': f'Engine is already running (PID {lock["pid"]})'})

        symbol = body.symbol or 'EURUSD'
        new_pid = _spawn_engine(symbol)
        if new_pid is None:
            write_audit_event('ENGINE_START_FAILED', 'Could not spawn engine subprocess', {})
            return JSONResponse({'ok': False, 'message': 'Could not spawn engine subprocess'})

        write_audit_event('ENGINE_START_CONFIRMED', 'Engine launched', {'new_pid': new_pid})
        return JSONResponse({'ok': True, 'message': f'Engine started (PID {new_pid})', 'pid': new_pid})

    @app.post("/api/command/engine/stop")
    async def cmd_engine_stop(body: CommandRequest):
        """Tier 1 — Graceful SIGTERM stop."""
        pid = find_engine_pid()
        if pid is None:
            return JSONResponse({'ok': False, 'message': 'Engine is not running'})
        reason = body.reason or 'Dashboard stop command'
        ok, msg = graceful_stop_engine(pid, reason)
        return JSONResponse({'ok': ok, 'message': msg, 'pid': pid})

    @app.post("/api/command/engine/kill")
    async def cmd_engine_kill(body: CommandRequest):
        """Tier 3 — Hard kill. Confirmation phrase enforced at API layer."""
        if body.confirmation_phrase != KILL_PHRASE:
            raise HTTPException(
                status_code=422,
                detail=f"Hard kill requires exact confirmation phrase: '{KILL_PHRASE}'"
            )
        lock = get_engine_lock_info()
        if lock is None:
            return JSONResponse({'ok': False, 'message': 'Engine is not running'})
        pid = lock['pid']
        started_at = lock.get('started_at', '')
        reason = body.reason or 'Dashboard hard kill command'
        ok, msg = hard_kill_engine(pid, reason, started_at)
        return JSONResponse({'ok': ok, 'message': msg, 'pid': pid})

    @app.post("/api/command/engine/restart")
    async def cmd_engine_restart(body: CommandRequest):
        """
        Tier 2 — Graceful stop then relaunch.
        SAFE_MODE override also routes here (Option A: clear via restart).
        Enforces Mutual Exclusion with Trainer.
        """
        if _trainer.get_status().get('running'):
            raise HTTPException(
                status_code=409,
                detail="Cannot restart Engine while Trainer pipeline is running. Stop the Trainer first."
            )
        reason = body.reason or 'Dashboard restart command'
        lock = get_engine_lock_info()

        if lock is not None:
            pid = lock['pid']
            write_audit_event('ENGINE_RESTART_REQUESTED', reason, {'pid': pid})
            ok, stop_msg = graceful_stop_engine(pid, reason)
            if not ok:
                write_audit_event('ENGINE_RESTART_FAILED', stop_msg, {'pid': pid})
                return JSONResponse({'ok': False, 'message': f'Stop failed: {stop_msg}'})
        else:
            write_audit_event('ENGINE_RESTART_REQUESTED', reason, {'note': 'engine_was_stopped'})

        # Relaunch
        symbol = lock.get('symbol', 'EURUSD') if lock else body.symbol
        new_pid = _spawn_engine(symbol)
        if new_pid is None:
            write_audit_event('ENGINE_RESTART_FAILED', 'Could not spawn engine subprocess', {})
            return JSONResponse({'ok': False, 'message': 'Engine stopped but relaunch failed'})

        write_audit_event('ENGINE_RESTART_CONFIRMED', 'Engine relaunched', {'new_pid': new_pid})
        return JSONResponse({'ok': True, 'message': f'Engine restarted (PID {new_pid})'})

    @app.post("/api/command/engine/clear_safe_mode")
    async def cmd_clear_safe_mode(body: CommandRequest):
        """
        Tier 2 — Clear SAFE_MODE via graceful restart (Option A).
        Enforces Mutual Exclusion with Trainer.
        """
        if _trainer.get_status().get('running'):
            raise HTTPException(
                status_code=409,
                detail="Cannot clear SAFE_MODE while Trainer pipeline is running. Stop the Trainer first."
            )
        reason = body.reason or 'SAFE_MODE override — graceful restart'
        cusum  = read_json_safe(CUSUM_PATH)
        state  = read_json_safe(STATE_PATH)
        meta = {
            'cusum_tripped':   cusum.get('cusum_tripped'),
            'cusum_reason':    cusum.get('cusum_reason'),
            'circuit_breaker': state.get('circuit_breaker_tripped'),
            'override_reason': reason,
        }
        write_audit_event('SAFE_MODE_OVERRIDE_REQUESTED', reason, meta)
        lock = get_engine_lock_info()
        if lock is not None:
            ok, stop_msg = graceful_stop_engine(lock['pid'], reason)
            if not ok:
                write_audit_event('SAFE_MODE_OVERRIDE_FAILED', stop_msg, meta)
                return JSONResponse({'ok': False, 'message': f'Stop failed: {stop_msg}'})
        symbol = lock.get('symbol', 'EURUSD') if lock else body.symbol
        new_pid = _spawn_engine(symbol)
        if new_pid is None:
            write_audit_event('SAFE_MODE_OVERRIDE_FAILED', 'Engine relaunch failed', meta)
            return JSONResponse({'ok': False, 'message': 'Engine stopped but relaunch failed'})
        write_audit_event('SAFE_MODE_OVERRIDE_CONFIRMED',
                          'Engine restarted — SAFE_MODE cleared on init',
                          {**meta, 'new_pid': new_pid})
        return JSONResponse({'ok': True,
                             'message': f'Engine restarted (PID {new_pid}). SAFE_MODE cleared.'})

    @app.post("/api/command/engine/reload_strategy")
    async def cmd_reload_strategy(body: CommandRequest):
        """Existing Tier 1 — Engine picks up strategy file changes on next candle close."""
        return JSONResponse({
            'ok':      True,
            'command': 'reload_strategy',
            'message': 'Engine will reload strategy on next candle close.',
        })

    # ── TRAINER COMMANDS ────────────────────────────────

    @app.post("/api/command/trainer/run")
    async def cmd_trainer_run(body: CommandRequest):
        """Tier 1 — Launch a Trainer run (halts at STAGED, no auto-deploy). Enforces Mutual Exclusion."""
        if get_engine_lock_info() is not None:
            raise HTTPException(
                status_code=409,
                detail="Cannot launch Trainer while Engine is running live. Stop the Engine first."
            )
        try:
            result = _trainer.launch(symbol=body.symbol, quick=body.quick)
            write_audit_event('TRAINER_RUN_LAUNCHED', 'Dashboard trainer launch',
                              {'pid': result['pid'], 'symbol': body.symbol})
            return JSONResponse({'ok': True, **result})
        except RuntimeError as e:
            return JSONResponse({'ok': False, 'message': str(e)})

    @app.post("/api/command/trainer/stop")
    async def cmd_trainer_stop(body: CommandRequest):
        """Tier 2 — SIGTERM the Trainer subprocess."""
        reason = body.reason or 'Dashboard trainer stop'
        pid = _trainer.get_pid()
        write_audit_event('TRAINER_STOP_REQUESTED', reason, {'pid': pid})
        ok = _trainer.stop()
        action = 'TRAINER_STOP_CONFIRMED' if ok else 'TRAINER_STOP_FAILED'
        write_audit_event(action, '' if ok else 'stop failed', {'pid': pid})
        return JSONResponse({'ok': ok, 'message': 'Trainer stopped' if ok else 'Stop failed'})

    # ── LEGACY COMMAND ROUTE ────────────────────────────

    @app.post("/api/command/{command}")
    async def api_command(command: str):
        """Legacy endpoint — only reload_strategy supported for backwards compat."""
        if command == 'reload_strategy':
            return JSONResponse({'status': 'ok', 'command': command,
                                 'message': 'Engine will reload strategy on next candle close.'})
        raise HTTPException(status_code=400,
                            detail=f"Unknown command '{command}'. "
                                   f"Use /api/command/engine/* or /api/command/trainer/*")


# ── ENGINE SPAWNER ──────────────────────────────────────

_engine_proc: Optional[subprocess.Popen] = None
_engine_lock  = threading.Lock()

def _spawn_engine(symbol: str = 'EURUSD') -> Optional[int]:
    """
    Spawn a new Engine subprocess. Returns the PID or None on failure.
    Note: spawning is best-effort; the dashboard does not monitor the child long-term.
    Engine itself writes engine.lock.json on startup.
    """
    global _engine_proc
    try:
        cmd = [sys.executable, '-m', 'engine.core.engine', '--symbol', symbol]
        with _engine_lock:
            _engine_proc = subprocess.Popen(
                cmd,
                cwd=str(ATS_ROOT),
                # Don't capture output — Engine writes its own log files
            )
        log.info(f"Engine spawned: PID {_engine_proc.pid} symbol={symbol}")
        return _engine_proc.pid
    except Exception as e:
        log.error(f"Failed to spawn Engine: {e}")
        return None


# ── ENTRY POINT ──────────────────────────────────────────

if __name__ == '__main__':
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Run: pip install fastapi uvicorn")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    uvicorn.run(
        app,
        host='127.0.0.1',
        port=8080,
        log_level='info',
    )