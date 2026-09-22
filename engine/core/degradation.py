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
from typing import Dict, List, Optional, Tuple
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


import math

CUSUM_STATE_FILE = Path(__file__).parent.parent / 'state' / 'cusum_state.json'


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


class CUSUMDetector:
    """
    One-Sided Downward CUSUM Detector for Bernoulli Win Rate Degradation.
    Detects when live win rate shifts below OOS baseline p0.
    Persists state to disk (cusum_state.json) on every update.
    Never modifies active_strategy.json.
    """

    def __init__(self,
                 strategy_id: str,
                 p0: float,
                 relative_shift: float = 0.15,
                 h: float = 12.0,
                 min_trades: int = 20,
                 state_file: Optional[Path] = None):
        self.strategy_id = strategy_id
        # Clamp baseline p0 to [0.05, 0.95] to prevent division by zero in log-likelihood ratio
        self.p0 = max(0.05, min(0.95, float(p0)))

        # Target degraded win rate p1 = p0 * (1 - relative_shift)
        self.p1 = max(0.01, min(self.p0 - 0.01, self.p0 * (1.0 - relative_shift)))

        # Log-likelihood ratio reference value k for Bernoulli trial:
        # k = ln((1 - p1) / (1 - p0)) / ln((p0 * (1 - p1)) / (p1 * (1 - p0)))
        num = math.log((1.0 - self.p1) / (1.0 - self.p0))
        den = math.log((self.p0 * (1.0 - self.p1)) / (self.p1 * (1.0 - self.p0)))
        self.k = num / den if den != 0 else 0.5

        self.h = h
        self.min_trades = min_trades
        if state_file is False:
            self.state_file = None
        else:
            self.state_file = state_file or CUSUM_STATE_FILE

        self.s_i: float = 0.0
        self.total_trades: int = 0
        self.win_count: int = 0
        self.tripped: bool = False
        self.trip_reason: Optional[str] = None

        # Load state from disk if strategy_id matches
        if self.state_file:
            self._load_state()

    def update(self, trade_pnl: float) -> Tuple[bool, float, int]:
        """
        Update CUSUM accumulator with outcome of a closed trade.
        x_i = 1 if pnl > 0 else 0.
        S_i = max(0, S_{i-1} + k - x_i)

        Returns:
            (tripped: bool, s_i: float, total_trades: int)
        """
        x_i = 1.0 if trade_pnl > 0 else 0.0
        self.total_trades += 1
        if x_i == 1.0:
            self.win_count += 1

        self.s_i = max(0.0, self.s_i + (self.k - x_i))

        # Check trip condition (requires min_trades floor)
        if self.total_trades >= self.min_trades and self.s_i >= self.h:
            if not self.tripped:
                live_wr = self.win_count / self.total_trades
                self.tripped = True
                self.trip_reason = (
                    f"CUSUM_LIVE_DEGRADATION_TRIPPED: CUSUM score ({self.s_i:.2f}) "
                    f"exceeded threshold ({self.h}) after {self.total_trades} trades. "
                    f"Live win rate: {live_wr:.1%} vs OOS baseline p0: {self.p0:.1%}"
                )
                log.warning(self.trip_reason)

        self._save_state()
        return self.tripped, self.s_i, self.total_trades

    def _save_state(self) -> None:
        """Atomically persist CUSUM state to disk."""
        if not self.state_file:
            return
        data = {
            'strategy_id': self.strategy_id,
            'p0': self.p0,
            'p1': self.p1,
            'k': self.k,
            'h': self.h,
            'min_trades': self.min_trades,
            's_i': self.s_i,
            'total_trades': self.total_trades,
            'win_count': self.win_count,
            'tripped': self.tripped,
            'trip_reason': self.trip_reason,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix('.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        if self.state_file.exists():
            self.state_file.unlink()
        tmp.rename(self.state_file)

    def _load_state(self) -> None:
        """Load state from disk if matching strategy_id."""
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('strategy_id') == self.strategy_id:
                self.s_i = float(data.get('s_i', 0.0))
                self.total_trades = int(data.get('total_trades', 0))
                self.win_count = int(data.get('win_count', 0))
                self.tripped = bool(data.get('tripped', False))
                self.trip_reason = data.get('trip_reason')
                log.info(
                    f"CUSUM state restored: strategy_id={self.strategy_id} | "
                    f"s_i={self.s_i:.2f} | trades={self.total_trades} | tripped={self.tripped}"
                )
            else:
                log.info(
                    f"CUSUM state reset: disk strategy_id ({data.get('strategy_id')}) "
                    f"!= current ({self.strategy_id})"
                )
        except Exception as e:
            log.error(f"CUSUM_STATE_CORRUPT: Could not load CUSUM state file {self.state_file}: {e}")
            raise RuntimeError(
                f"CUSUM_STATE_CORRUPT: Failed to parse CUSUM status file {self.state_file}: {e}. "
                "Engine startup halted to prevent trading with corrupted degradation state."
            )


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

    # CUSUM tracking (WAVE 15)
    cusum_score:            float   = 0.0
    cusum_tripped:          bool    = False
    cusum_reason:           Optional[str] = None

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
                 window_size:       int   = 30,
                 cusum_state_file:  Optional[Path] = None):

        self.strategy_id        = strategy_id
        self.window_size        = window_size
        self.trades:deque       = deque(maxlen=window_size)
        self.all_trades:List    = []

        self.state              = DegradationState(
            wf_median_pf       = wf_median_pf,
            wf_profitable_rate = wf_profitable_rate,
        )

        self.cusum = CUSUMDetector(
            strategy_id=strategy_id,
            p0=wf_profitable_rate if wf_profitable_rate > 0 else 0.50,
            state_file=cusum_state_file
        )
        self.state.cusum_score = self.cusum.s_i
        self.state.cusum_tripped = self.cusum.tripped
        self.state.cusum_reason = self.cusum.trip_reason
        if self.cusum.tripped:
            self.state.status = 'DEGRADED'
            self.state.rerun_recommended = True

    def record_trade(self, trade: TradeRecord) -> DegradationState:
        """
        Record a closed trade and update degradation state.
        Returns updated state.
        Called after every trade close by the Engine.
        """
        self.trades.append(trade)
        self.all_trades.append(trade)

        # Update CUSUM detector (WAVE 15)
        tripped, s_i, total_trades = self.cusum.update(trade.net_pnl)
        self.state.cusum_score = s_i
        self.state.cusum_tripped = tripped
        self.state.cusum_reason = self.cusum.trip_reason

        self._update_state()
        self._check_degradation()

        if tripped:
            self.state.status = 'DEGRADED'
            self.state.rerun_recommended = True
            self.state.rerun_reason = self.cusum.trip_reason or "CUSUM win rate degradation tripped"

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
            'cusum_score':          round(self.state.cusum_score, 4),
            'cusum_tripped':        self.state.cusum_tripped,
            'cusum_reason':         self.state.cusum_reason,
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