"""
Engine Dashboard — FastAPI backend.

Serves real-time Engine status to the browser.
Read-only monitoring interface. Never touches MT5 directly.

Run:
  python engine/dashboard/app.py

Access:
  http://localhost:8080

Endpoints:
  GET /              → dashboard HTML
  GET /api/status    → full engine status JSON
  GET /api/trades    → recent trades JSON
  GET /api/equity    → equity curve data JSON
  GET /api/strategy  → current strategy spec JSON
  POST /api/command  → send command to Engine (reload only)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ── CHECK DEPENDENCIES ─────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed.")
    print("Run: pip install fastapi uvicorn")


# ── DATA PATHS ─────────────────────────────────────────
ENGINE_ROOT       = Path(__file__).parent.parent
STRATEGY_PATH     = ENGINE_ROOT / 'strategy' / 'active_strategy.json'
DEGRADATION_PATH  = ENGINE_ROOT / 'state' / 'degradation_status.json'
LOGS_PATH         = ENGINE_ROOT / 'logs'
STATE_PATH        = ENGINE_ROOT / 'state' / 'engine_state.json'


# ── DATA READERS ───────────────────────────────────────

def read_json_safe(path: Path,
                   default: Dict = None) -> Dict:
    """Read JSON file safely. Returns default on any error."""
    default = default or {}
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not read {path}: {e}")
        return default


def read_recent_trades(n: int = 50) -> List[Dict]:
    """
    Read most recent N trades from log files.
    Reads today's log file first, then yesterday's if needed.
    """
    trades = []
    log_files = sorted(
        LOGS_PATH.glob('*.jsonl'),
        reverse=True
    )

    for log_file in log_files[:3]:  # check last 3 days
        if len(trades) >= n:
            break
        try:
            lines = log_file.read_text().strip().split('\n')
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    # Only include trade records, not events
                    if 'trade_id' in record:
                        trades.append(record)
                    if len(trades) >= n:
                        break
                except json.JSONDecodeError:
                    continue
        except Exception:
            continue

    return trades


def build_equity_curve(trades: List[Dict],
                        initial_equity: float = 10000.0
                        ) -> List[Dict]:
    """
    Build equity curve from trade list.
    Returns list of {time, equity} points.
    """
    if not trades:
        return [{'time': datetime.now(
            timezone.utc
        ).isoformat(), 'equity': initial_equity}]

    # Sort trades by exit time
    sorted_trades = sorted(
        [t for t in trades if 'exit_time' in t],
        key=lambda x: x.get('exit_time', '')
    )

    equity  = initial_equity
    curve   = [{'time': sorted_trades[0].get(
        'entry_time', ''
    ), 'equity': equity}]

    for trade in sorted_trades:
        equity += trade.get('net_pnl', 0)
        curve.append({
            'time':   trade.get('exit_time', ''),
            'equity': round(equity, 2),
        })

    return curve


def get_engine_status() -> Dict:
    """
    Compile full engine status from all data sources.
    Returns a comprehensive status dict for the dashboard.
    """
    strategy    = read_json_safe(STRATEGY_PATH)
    degradation = read_json_safe(DEGRADATION_PATH)
    state       = read_json_safe(STATE_PATH)
    trades      = read_recent_trades(100)

    # Compute summary stats from recent trades
    recent_50   = trades[:50]
    winners     = [t for t in recent_50
                   if t.get('net_pnl', 0) > 0]
    total_pnl   = sum(t.get('net_pnl', 0) for t in recent_50)
    win_rate    = (len(winners) / len(recent_50)
                   if recent_50 else 0)

    gross_profit = sum(
        t.get('net_pnl', 0) for t in recent_50
        if t.get('net_pnl', 0) > 0
    )
    gross_loss   = abs(sum(
        t.get('net_pnl', 0) for t in recent_50
        if t.get('net_pnl', 0) < 0
    ))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else 0
    )

    return {
        'timestamp':         datetime.now(
            timezone.utc
        ).isoformat(),
        'engine_running':    state.get('running', False),
        'strategy': {
            'id':            strategy.get(
                'strategy_id', 'No strategy loaded'
            ),
            'template':      strategy.get(
                'template', 'unknown'
            ),
            'score':         strategy.get(
                'composite_score', 0
            ),
            'promoted_at':   strategy.get('promoted_at', ''),
            'parameters':    strategy.get('parameters', {}),
        },
        'degradation': {
            'status':        degradation.get('status', 'OK'),
            'alert_count':   degradation.get('alert_count', 0),
            'rolling_pf':    degradation.get('rolling_pf', 0),
            'rolling_wr':    degradation.get(
                'rolling_win_rate', 0
            ),
            'loss_streak':   degradation.get(
                'current_loss_streak', 0
            ),
            'rerun_recommended': degradation.get(
                'rerun_recommended', False
            ),
        },
        'performance': {
            'total_trades':  len(trades),
            'recent_trades': len(recent_50),
            'win_rate':      round(win_rate, 4),
            'total_pnl':     round(total_pnl, 2),
            'profit_factor': round(profit_factor, 3),
        },
        'current_state': {
            'session':       state.get('session', 'UNKNOWN'),
            'regime':        state.get('regime', 'UNKNOWN'),
            'equity':        state.get('equity', 0),
            'open_position': state.get('open_position'),
        },
    }


# ── FASTAPI APP ────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="ATS Engine Dashboard",
        description="Read-only monitoring for ATS Engine",
        version="1.0"
    )

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the dashboard HTML."""
        html_path = (
            Path(__file__).parent / 'static' / 'dashboard.html'
        )
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/api/status")
    async def api_status():
        """Full engine status."""
        return JSONResponse(get_engine_status())

    @app.get("/api/trades")
    async def api_trades(n: int = 50):
        """Recent trades."""
        trades = read_recent_trades(n)
        return JSONResponse({'trades': trades})

    @app.get("/api/equity")
    async def api_equity():
        """Equity curve data."""
        trades = read_recent_trades(200)
        curve  = build_equity_curve(trades)
        return JSONResponse({'equity_curve': curve})

    @app.get("/api/strategy")
    async def api_strategy():
        """Current strategy specification."""
        strategy = read_json_safe(STRATEGY_PATH)
        return JSONResponse(strategy)

    @app.post("/api/command/{command}")
    async def api_command(command: str):
        """
        Send a command to the Engine.
        Only 'reload_strategy' is supported.
        """
        if command == 'reload_strategy':
            # Engine checks for file changes on next candle
            # This endpoint just confirms the request
            return JSONResponse({
                'status':  'ok',
                'command': command,
                'message': (
                    'Engine will reload strategy on '
                    'next candle close.'
                ),
            })
        raise HTTPException(
            status_code=400,
            detail=f"Unknown command: {command}. "
                   f"Supported: reload_strategy"
        )

    @app.get("/api/health")
    async def api_health():
        """Health check endpoint."""
        return JSONResponse({
            'status':    'ok',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })


# ── EMBEDDED DASHBOARD HTML ────────────────────────────
# Self-contained — works without static files directory.
# Polls /api/status every 5 seconds.

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ATS Engine Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    min-height: 100vh;
  }

  header {
    background: #1a1d2e;
    border-bottom: 1px solid #2a2d3e;
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  header h1 {
    font-size: 1.2rem;
    font-weight: 600;
    color: #fff;
    letter-spacing: 0.05em;
  }

  #connection-status {
    font-size: 0.8rem;
    padding: 4px 10px;
    border-radius: 20px;
    background: #2a2d3e;
  }

  #connection-status.connected    { color: #4caf50; }
  #connection-status.disconnected { color: #f44336; }

  .main { padding: 24px; max-width: 1400px; margin: 0 auto; }

  /* Top stat cards */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }

  .stat-card {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 20px;
  }

  .stat-card .label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
  }

  .stat-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
  }

  .stat-card .sub {
    font-size: 0.8rem;
    color: #666;
    margin-top: 4px;
  }

  .stat-card.positive .value { color: #4caf50; }
  .stat-card.negative .value { color: #f44336; }
  .stat-card.warning  .value { color: #ff9800; }

  /* Content grid */
  .content-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
  }

  @media (max-width: 900px) {
    .content-grid { grid-template-columns: 1fr; }
  }

  .panel {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 20px;
  }

  .panel h2 {
    font-size: 0.85rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
    border-bottom: 1px solid #2a2d3e;
    padding-bottom: 10px;
  }

  /* Status badge */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }

  .badge.ok       { background: #1b3a1f; color: #4caf50; }
  .badge.warning  { background: #3a2c1a; color: #ff9800; }
  .badge.degraded { background: #3a1a1a; color: #f44336; }
  .badge.critical { background: #f44336; color: #fff; }
  .badge.trending { background: #1a2a3a; color: #2196f3; }
  .badge.ranging  { background: #2a1a3a; color: #9c27b0; }
  .badge.volatile { background: #3a2a1a; color: #ff9800; }
  .badge.quiet    { background: #1a3a2a; color: #4caf50; }

  /* Strategy info */
  .strategy-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #2a2d3e;
    font-size: 0.85rem;
  }

  .strategy-row:last-child { border-bottom: none; }
  .strategy-row .key { color: #888; }
  .strategy-row .val { color: #fff; font-weight: 500; }

  /* Equity chart */
  #equity-chart {
    width: 100%;
    height: 200px;
    position: relative;
  }

  canvas {
    width: 100% !important;
    height: 100% !important;
  }

  /* Trades table */
  .trades-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }

  .trades-table th {
    color: #888;
    text-align: left;
    padding: 6px 8px;
    border-bottom: 1px solid #2a2d3e;
    font-weight: 500;
    font-size: 0.72rem;
    text-transform: uppercase;
  }

  .trades-table td {
    padding: 7px 8px;
    border-bottom: 1px solid #1e2130;
    color: #ccc;
  }

  .trades-table tr:hover td { background: #1e2130; }

  .pnl-positive { color: #4caf50; font-weight: 600; }
  .pnl-negative { color: #f44336; font-weight: 600; }

  /* Degradation meter */
  .deg-meter {
    margin: 12px 0;
  }

  .deg-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #888;
    margin-bottom: 6px;
  }

  .deg-bar {
    height: 8px;
    background: #2a2d3e;
    border-radius: 4px;
    overflow: hidden;
  }

  .deg-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
  }

  .deg-fill.ok       { background: #4caf50; }
  .deg-fill.warning  { background: #ff9800; }
  .deg-fill.degraded { background: #f44336; }

  /* Last updated */
  #last-updated {
    text-align: center;
    color: #444;
    font-size: 0.75rem;
    margin-top: 16px;
  }

  /* No data placeholder */
  .no-data {
    text-align: center;
    color: #444;
    padding: 20px;
    font-size: 0.85rem;
  }
</style>
</head>
<body>

<header>
  <h1>⚡ ATS ENGINE DASHBOARD</h1>
  <span id="connection-status" class="disconnected">
    ● CONNECTING...
  </span>
</header>

<div class="main">

  <!-- Stat Cards Row -->
  <div class="stats-row">
    <div class="stat-card" id="card-equity">
      <div class="label">Account Equity</div>
      <div class="value" id="stat-equity">—</div>
      <div class="sub" id="stat-equity-sub">—</div>
    </div>
    <div class="stat-card" id="card-pnl">
      <div class="label">Total P&L (recent 50)</div>
      <div class="value" id="stat-pnl">—</div>
      <div class="sub" id="stat-pnl-sub">—</div>
    </div>
    <div class="stat-card" id="card-pf">
      <div class="label">Profit Factor</div>
      <div class="value" id="stat-pf">—</div>
      <div class="sub">Recent 50 trades</div>
    </div>
    <div class="stat-card" id="card-wr">
      <div class="label">Win Rate</div>
      <div class="value" id="stat-wr">—</div>
      <div class="sub" id="stat-wr-sub">—</div>
    </div>
    <div class="stat-card" id="card-session">
      <div class="label">Session / Regime</div>
      <div class="value" id="stat-session">—</div>
      <div class="sub" id="stat-regime">—</div>
    </div>
    <div class="stat-card" id="card-degrade">
      <div class="label">Strategy Health</div>
      <div class="value" id="stat-degrade">—</div>
      <div class="sub" id="stat-degrade-sub">—</div>
    </div>
  </div>

  <!-- Content Grid -->
  <div class="content-grid">

    <!-- Left: Strategy + Equity -->
    <div>
      <div class="panel" style="margin-bottom:20px">
        <h2>Current Strategy</h2>
        <div id="strategy-info">
          <div class="no-data">Loading...</div>
        </div>
      </div>

      <div class="panel">
        <h2>Equity Curve</h2>
        <div id="equity-chart">
          <canvas id="equity-canvas"></canvas>
        </div>
      </div>
    </div>

    <!-- Right: Degradation + Trades -->
    <div>
      <div class="panel" style="margin-bottom:20px">
        <h2>Degradation Monitor</h2>
        <div id="degradation-info">
          <div class="no-data">Loading...</div>
        </div>
      </div>

      <div class="panel">
        <h2>Recent Trades</h2>
        <div id="trades-container">
          <div class="no-data">Loading...</div>
        </div>
      </div>
    </div>

  </div>

  <div id="last-updated">Last updated: —</div>
</div>

<script>
// ── DATA STATE ───────────────────────────────────────
let equityCurve = [];
let lastStatus  = null;

// ── FETCH STATUS ─────────────────────────────────────
async function fetchStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();
    lastStatus = data;
    updateDashboard(data);
    setConnected(true);
  } catch (e) {
    setConnected(false);
  }
}

async function fetchEquity() {
  try {
    const res  = await fetch('/api/equity');
    const data = await res.json();
    equityCurve = data.equity_curve || [];
    drawEquityCurve();
  } catch (e) {}
}

async function fetchTrades() {
  try {
    const res  = await fetch('/api/trades?n=20');
    const data = await res.json();
    renderTrades(data.trades || []);
  } catch (e) {}
}

// ── CONNECTION STATUS ─────────────────────────────────
function setConnected(connected) {
  const el = document.getElementById('connection-status');
  if (connected) {
    el.textContent  = '● LIVE';
    el.className    = 'connected';
  } else {
    el.textContent  = '● OFFLINE';
    el.className    = 'disconnected';
  }
}

// ── UPDATE DASHBOARD ──────────────────────────────────
function updateDashboard(data) {
  const s = data.strategy    || {};
  const d = data.degradation || {};
  const p = data.performance || {};
  const c = data.current_state || {};

  // Equity card
  const equity = c.equity || 0;
  setText('stat-equity', equity
    ? '$' + equity.toLocaleString('en', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      })
    : '—'
  );
  const pos = c.open_position;
  setText('stat-equity-sub', pos
    ? `Open: ${pos.direction} ${pos.lots}L`
    : 'No open position'
  );

  // P&L card
  const pnl = p.total_pnl || 0;
  setText('stat-pnl', (pnl >= 0 ? '+' : '') +
    '$' + pnl.toFixed(2));
  setCard('card-pnl',
    pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : '');
  setText('stat-pnl-sub',
    `${p.recent_trades || 0} recent trades`);

  // PF card
  const pf = p.profit_factor || 0;
  setText('stat-pf', pf > 0 ? pf.toFixed(3) : '—');
  setCard('card-pf',
    pf > 1.2 ? 'positive' : pf > 0 && pf < 1.0 ? 'negative' : '');

  // Win rate card
  const wr = (p.win_rate || 0) * 100;
  setText('stat-wr', wr > 0 ? wr.toFixed(1) + '%' : '—');
  setText('stat-wr-sub',
    `${p.total_trades || 0} total trades`);

  // Session/Regime card
  setText('stat-session', c.session || '—');
  setText('stat-regime', c.regime || '—');

  // Degradation card
  const status = d.status || 'OK';
  setText('stat-degrade', status);
  setText('stat-degrade-sub',
    `Alerts: ${d.alert_count || 0} | ` +
    `Streak: ${d.loss_streak || 0}`);
  setCard('card-degrade',
    status === 'OK' ? 'positive' :
    status === 'WARNING' ? 'warning' : 'negative');

  // Strategy panel
  renderStrategy(s, d);

  // Degradation panel
  renderDegradation(d);

  // Last updated
  setText('last-updated',
    'Last updated: ' + new Date().toLocaleTimeString());
}

// ── RENDER STRATEGY PANEL ─────────────────────────────
function renderStrategy(s, d) {
  const promoted = s.promoted_at
    ? new Date(s.promoted_at).toLocaleDateString()
    : 'Unknown';

  const params = s.parameters || {};
  const html = `
    <div class="strategy-row">
      <span class="key">Strategy ID</span>
      <span class="val">${s.id || '—'}</span>
    </div>
    <div class="strategy-row">
      <span class="key">Template</span>
      <span class="val">${s.template || '—'}</span>
    </div>
    <div class="strategy-row">
      <span class="key">Score</span>
      <span class="val">${
        s.score ? s.score.toFixed(3) : '—'
      }</span>
    </div>
    <div class="strategy-row">
      <span class="key">Promoted</span>
      <span class="val">${promoted}</span>
    </div>
    <div class="strategy-row">
      <span class="key">Fast MA</span>
      <span class="val">${params.fast_ma_period || '—'}</span>
    </div>
    <div class="strategy-row">
      <span class="key">Slow MA</span>
      <span class="val">${params.slow_ma_period || '—'}</span>
    </div>
    <div class="strategy-row">
      <span class="key">SL ATR</span>
      <span class="val">${params.sl_atr_multiple || '—'}×</span>
    </div>
    <div class="strategy-row">
      <span class="key">TP R:R</span>
      <span class="val">${params.tp_rr_ratio || '—'}R</span>
    </div>
    <div class="strategy-row">
      <span class="key">Risk/Trade</span>
      <span class="val">${
        params.risk_per_trade_pct || '—'
      }%</span>
    </div>
    ${d.rerun_recommended ? `
    <div style="margin-top:12px;padding:10px;
                background:#3a1a1a;border-radius:6px;
                font-size:0.8rem;color:#ff9800;">
      ⚠ Trainer re-run recommended
    </div>` : ''}
  `;
  document.getElementById('strategy-info').innerHTML = html;
}

// ── RENDER DEGRADATION PANEL ─────────────────────────
function renderDegradation(d) {
  const status    = d.status || 'OK';
  const alertPct  = Math.min(
    (d.alert_count || 0) / 3 * 100, 100
  );
  const rollPF    = d.rolling_pf || 0;
  const rollWR    = ((d.rolling_wr || 0) * 100).toFixed(1);
  const streak    = d.loss_streak || 0;

  const statusClass = status.toLowerCase();

  const html = `
    <div class="strategy-row">
      <span class="key">Status</span>
      <span class="badge ${statusClass}">${status}</span>
    </div>
    <div class="strategy-row">
      <span class="key">Rolling PF</span>
      <span class="val" style="color:${
        rollPF >= 1.0 ? '#4caf50' : '#f44336'
      }">${rollPF > 0 ? rollPF.toFixed(3) : '—'}</span>
    </div>
    <div class="strategy-row">
      <span class="key">Rolling Win Rate</span>
      <span class="val">${rollPF > 0 ? rollWR + '%' : '—'}</span>
    </div>
    <div class="strategy-row">
      <span class="key">Loss Streak</span>
      <span class="val" style="color:${
        streak >= 5 ? '#ff9800' : '#ccc'
      }">${streak}</span>
    </div>

    <div class="deg-meter">
      <div class="deg-label">
        <span>Alert Level</span>
        <span>${d.alert_count || 0}/3</span>
      </div>
      <div class="deg-bar">
        <div class="deg-fill ${statusClass}"
             style="width:${alertPct}%"></div>
      </div>
    </div>

    ${d.rerun_recommended ? `
      <div style="font-size:0.8rem;color:#ff9800;
                  margin-top:8px;">
        ${d.rerun_reason || 'Re-run recommended'}
      </div>
    ` : `
      <div style="font-size:0.8rem;color:#4caf50;
                  margin-top:8px;">
        Strategy performing within expected parameters.
      </div>
    `}
  `;
  document.getElementById('degradation-info').innerHTML = html;
}

// ── RENDER TRADES TABLE ───────────────────────────────
function renderTrades(trades) {
  if (!trades || trades.length === 0) {
    document.getElementById('trades-container').innerHTML =
      '<div class="no-data">No trades recorded yet.</div>';
    return;
  }

  const rows = trades.slice(0, 15).map(t => {
    const pnl    = t.net_pnl || 0;
    const cls    = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
    const time   = t.exit_time
      ? new Date(t.exit_time).toLocaleString('en', {
          month:'2-digit', day:'2-digit',
          hour:'2-digit', minute:'2-digit'
        })
      : '—';
    return `
      <tr>
        <td>${time}</td>
        <td>${t.direction || '—'}</td>
        <td>${(t.exit_price||0).toFixed(5)}</td>
        <td class="${cls}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</td>
        <td>${t.close_reason || '—'}</td>
        <td>${t.entry_session || '—'}</td>
      </tr>
    `;
  }).join('');

  document.getElementById('trades-container').innerHTML = `
    <table class="trades-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Dir</th>
          <th>Exit</th>
          <th>P&L</th>
          <th>Reason</th>
          <th>Session</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// ── DRAW EQUITY CURVE ─────────────────────────────────
function drawEquityCurve() {
  const canvas = document.getElementById('equity-canvas');
  if (!canvas) return;
  const ctx    = canvas.getContext('2d');
  const W      = canvas.offsetWidth  || 400;
  const H      = canvas.offsetHeight || 200;
  canvas.width  = W;
  canvas.height = H;

  ctx.clearRect(0, 0, W, H);

  if (!equityCurve || equityCurve.length < 2) {
    ctx.fillStyle = '#444';
    ctx.font      = '13px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(
      'No equity data yet', W/2, H/2
    );
    return;
  }

  const values = equityCurve.map(p => p.equity);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const range  = maxVal - minVal || 1;
  const pad    = { t:10, r:10, b:30, l:60 };
  const chartW = W - pad.l - pad.r;
  const chartH = H - pad.t - pad.b;

  const xScale = i => pad.l + (i / (values.length-1)) * chartW;
  const yScale = v => pad.t + (1 - (v-minVal)/range) * chartH;

  // Grid lines
  ctx.strokeStyle = '#2a2d3e';
  ctx.lineWidth   = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (i/4) * chartH;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(W - pad.r, y);
    ctx.stroke();

    const val = maxVal - (i/4) * range;
    ctx.fillStyle  = '#555';
    ctx.font       = '10px system-ui';
    ctx.textAlign  = 'right';
    ctx.fillText('$' + val.toFixed(0), pad.l - 4, y + 4);
  }

  // Determine colour — green if profitable
  const profitable = values[values.length-1] >= values[0];
  const lineColor  = profitable ? '#4caf50' : '#f44336';
  const fillColor  = profitable
    ? 'rgba(76,175,80,0.12)' : 'rgba(244,67,54,0.12)';

  // Fill area
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(values[0]));
  values.forEach((v,i) => ctx.lineTo(xScale(i), yScale(v)));
  ctx.lineTo(xScale(values.length-1), H - pad.b);
  ctx.lineTo(pad.l, H - pad.b);
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = lineColor;
  ctx.lineWidth   = 2;
  values.forEach((v,i) => {
    if (i === 0) ctx.moveTo(xScale(i), yScale(v));
    else         ctx.lineTo(xScale(i), yScale(v));
  });
  ctx.stroke();

  // Start/end labels
  ctx.fillStyle  = '#888';
  ctx.font       = '10px system-ui';
  ctx.textAlign  = 'left';
  ctx.fillText(
    '$' + values[0].toFixed(0),
    pad.l + 2, yScale(values[0]) - 4
  );
  ctx.textAlign = 'right';
  ctx.fillText(
    '$' + values[values.length-1].toFixed(0),
    W - pad.r,
    yScale(values[values.length-1]) - 4
  );
}

// ── UTILITIES ─────────────────────────────────────────
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setCard(id, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'stat-card ' + cls;
}

// ── STARTUP ───────────────────────────────────────────
fetchStatus();
fetchEquity();
fetchTrades();

// Poll every 5 seconds
setInterval(fetchStatus, 5000);
setInterval(fetchEquity,  30000);  // equity less frequent
setInterval(fetchTrades,  10000);

// Redraw chart on resize
window.addEventListener('resize', drawEquityCurve);
</script>
</body>
</html>"""


# ── ENTRY POINT ────────────────────────────────────────
if __name__ == '__main__':
    if not FASTAPI_AVAILABLE:
        print("Install FastAPI first:")
        print("  pip install fastapi uvicorn")
        sys.exit(1)

    print("\nATS Engine Dashboard")
    print("=" * 40)
    print("URL:  http://localhost:8080")
    print("API:  http://localhost:8080/api/status")
    print("Stop: Ctrl+C")
    print("=" * 40 + "\n")

    uvicorn.run(
        app,
        host='127.0.0.1',  # localhost only — not exposed
        port=8080,
        log_level='warning'
    )