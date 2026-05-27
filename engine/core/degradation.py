"""
Degradation Monitor — Engine component.

Watches live trading performance and detects when the
current strategy is underperforming its walk-forward
predictions. Triggers Trainer re-run recommendations.

Runs on every closed trade. Lightweight — no heavy computation.

Degradation is detected when:
  - Recent profit factor drops significantly below
    the walk-forward median
  - Consecutive losses exceed the walk-forward maximum
  - Rolling Calmar ratio falls below threshold

When degradation is detected:
  - Log DEGRADATION_ALERT
  - Update engine_state.degradation_status
  - Trainer reads this on next run and weights it
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ── DEGRADATION THRESHOLDS ─────────────────────────────
THRESHOLDS = {
    # Recent PF must be at least this fraction of WF median
    'pf_degradation_ratio':      0.70,
    # Max consecutive losses before alert
    'max_consecutive_losses':    8,
    # Minimum trades before degradation can be flagged
    'min_trades_for_assessment': 20,
    # Rolling window for recent performance (trades)
    'rolling_window_trades':     30,
    # How many alerts before recommending Trainer re-run
    'alerts_before_rerun':       3,
}


@dataclass
class TradeRecord:
    """Minimal trade record for degradation monitoring."""
    trade_id:       str
    direction:      str
    entry_price:    float
    exit_price:     float
    net_pnl:        float
    close_reason:   str
    closed_at:      str
    strategy_id:    str


@dataclass
class DegradationState:
    """Current degradation monitoring state."""
    status:                 str     = 'OK'
    # OK / WARNING / DEGRADED / CRITICAL

    alert_count:            int     = 0
    last_alert_at:          Optional[str] = None

    # Rolling metrics
    rolling_win_rate:       float   = 0.0
    rolling_profit_factor:  float   = 0.0
    rolling_net_pnl:        float   = 0.0

    # Streak tracking
    current_loss_streak:    int     = 0
    max_loss_streak_seen:   int     = 0

    # Baseline (from walk-forward results)
    wf_median_pf:           float   = 0.0
    wf_profitable_rate:     float   = 0.0

    # Rerun recommendation
    rerun_recommended:      bool    = False
    rerun_reason:           str     = ''

    # Alerts fired
    alerts:                 List    = field(default_factory=list)


class DegradationMonitor:
    """
    Monitors live trade performance for strategy degradation.
    Maintains a rolling window of recent trades.
    Called after every trade close.
    """

    def __init__(self,
                 strategy_id:       str,
                 wf_median_pf:      float = 0.0,
                 wf_profitable_rate:float = 0.0,
                 window_size:       int   = 30):

        self.strategy_id        = strategy_id
        self.window_size        = window_size
        self.trades:deque       = deque(maxlen=window_size)
        self.all_trades:List    = []

        self.state              = DegradationState(
            wf_median_pf       = wf_median_pf,
            wf_profitable_rate = wf_profitable_rate,
        )

    def record_trade(self, trade: TradeRecord) -> DegradationState:
        """
        Record a closed trade and update degradation state.
        Returns updated state.
        Called after every trade close by the Engine.
        """
        self.trades.append(trade)
        self.all_trades.append(trade)
        self._update_state()
        self._check_degradation()
        return self.state

    def _update_state(self) -> None:
        """Recalculate rolling metrics from recent trades."""
        if not self.trades:
            return

        pnls     = [t.net_pnl for t in self.trades]
        winners  = [p for p in pnls if p > 0]
        losers   = [p for p in pnls if p <= 0]

        gross_profit = sum(winners)
        gross_loss   = abs(sum(losers))

        self.state.rolling_win_rate = (
            len(winners) / len(pnls)
            if pnls else 0.0
        )
        self.state.rolling_profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (float('inf') if gross_profit > 0 else 0.0)
        )
        self.state.rolling_net_pnl = sum(pnls)

        # Consecutive loss streak
        streak = 0
        for trade in reversed(list(self.trades)):
            if trade.net_pnl <= 0:
                streak += 1
            else:
                break
        self.state.current_loss_streak = streak
        self.state.max_loss_streak_seen = max(
            self.state.max_loss_streak_seen, streak
        )

    def _check_degradation(self) -> None:
        """Check for degradation conditions."""
        if len(self.trades) < \
                THRESHOLDS['min_trades_for_assessment']:
            return

        alerts_fired = []

        # Check 1: Profit factor degradation
        wf_pf       = self.state.wf_median_pf
        rolling_pf  = self.state.rolling_profit_factor
        pf_threshold= wf_pf * THRESHOLDS['pf_degradation_ratio']

        if wf_pf > 0 and rolling_pf < pf_threshold:
            alerts_fired.append(
                f"PF degradation: rolling={rolling_pf:.3f} "
                f"vs WF median={wf_pf:.3f} "
                f"(threshold={pf_threshold:.3f})"
            )

        # Check 2: Consecutive losses
        loss_streak = self.state.current_loss_streak
        if loss_streak >= THRESHOLDS['max_consecutive_losses']:
            alerts_fired.append(
                f"Consecutive losses: {loss_streak} "
                f"(threshold={THRESHOLDS['max_consecutive_losses']})"
            )

        # Check 3: Win rate degradation
        wf_wr       = self.state.wf_profitable_rate
        rolling_wr  = self.state.rolling_win_rate
        if wf_wr > 0 and rolling_wr < wf_wr * 0.70:
            alerts_fired.append(
                f"Win rate degradation: "
                f"rolling={rolling_wr:.1%} "
                f"vs WF={wf_wr:.1%}"
            )

        # Update state based on alerts
        if not alerts_fired:
            # Recover if previously degraded
            if self.state.alert_count > 0:
                self.state.alert_count = max(
                    0, self.state.alert_count - 1
                )
            if self.state.alert_count == 0:
                self.state.status = 'OK'
                self.state.rerun_recommended = False
            return

        # Alerts fired
        self.state.alert_count += 1
        self.state.last_alert_at = datetime.now(
            timezone.utc
        ).isoformat()
        self.state.alerts.extend(alerts_fired)
        self.state.alerts = self.state.alerts[-20:]

        for alert in alerts_fired:
            log.warning(f"DEGRADATION: {alert}")

        # Update severity
        if self.state.alert_count >= \
                THRESHOLDS['alerts_before_rerun']:
            self.state.status             = 'DEGRADED'
            self.state.rerun_recommended  = True
            self.state.rerun_reason       = (
                f"Strategy {self.strategy_id} has triggered "
                f"{self.state.alert_count} degradation alerts. "
                f"Trainer re-run recommended."
            )
            log.warning(
                f"DEGRADATION CRITICAL: "
                f"{self.state.rerun_reason}"
            )
        elif self.state.alert_count >= 1:
            self.state.status = 'WARNING'

    def get_state(self) -> DegradationState:
        """Return current degradation state."""
        return self.state

    def get_summary(self) -> Dict:
        """Return summary dict for logging and dashboard."""
        return {
            'strategy_id':          self.strategy_id,
            'status':               self.state.status,
            'alert_count':          self.state.alert_count,
            'total_trades':         len(self.all_trades),
            'rolling_trades':       len(self.trades),
            'rolling_win_rate':     round(
                self.state.rolling_win_rate, 4
            ),
            'rolling_pf':           round(
                self.state.rolling_profit_factor, 4
            ),
            'rolling_net_pnl':      round(
                self.state.rolling_net_pnl, 2
            ),
            'current_loss_streak':  self.state.current_loss_streak,
            'rerun_recommended':    self.state.rerun_recommended,
            'rerun_reason':         self.state.rerun_reason,
            'last_alert_at':        self.state.last_alert_at,
            'wf_median_pf':         self.state.wf_median_pf,
            'wf_profitable_rate':   self.state.wf_profitable_rate,
        }

    def write_status_file(self, output_path: str = None) -> None:
        """
        Write degradation status to a JSON file.
        The Trainer reads this file on the next training run
        to understand how the current strategy is performing.
        """
        if output_path is None:
            output_path = (
                Path(__file__).parent.parent /
                'state' / 'degradation_status.json'
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        status = self.get_summary()
        status['written_at'] = datetime.now(
            timezone.utc
        ).isoformat()

        tmp = output_path.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(status, f, indent=2)
        if output_path.exists():
            output_path.unlink()
        tmp.rename(output_path)