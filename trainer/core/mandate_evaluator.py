"""
Prop-Firm Mandate Evaluator for Trainer Pipeline — Fix 11.2.
Evaluates backtest trade logs and equity curves against prop-firm mandate constraints (max_daily_loss_pct, max_overall_drawdown_pct).
Hard-rejects non-compliant candidate strategies.
Fail-closed: raises RuntimeError if mandate config is missing or invalid.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone

from engine.core.mandate import Mandate, load_mandate, MANDATE_FILE

log = logging.getLogger(__name__)


def evaluate_backtest_mandate(
    equity_curve: List[Dict],
    mandate: Optional[Mandate] = None,
    mandate_path: Optional[Path] = None
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates a candidate's backtest equity curve against prop-firm mandate constraints.

    Args:
        equity_curve: List of dicts with keys 'timestamp' and 'equity' (or 'balance')
        mandate: Loaded Mandate instance (if None, attempts to load from mandate_path/MANDATE_FILE)
        mandate_path: Optional path to mandate.json

    Returns:
        (is_compliant: bool, breach_reason: Optional[str])

    Fail-Closed:
        If mandate is missing or cannot be loaded, raises RuntimeError.
    """
    if mandate is None:
        target_path = mandate_path or MANDATE_FILE
        if not target_path.exists():
            log.error(f"MANDATE_CONFIG_MISSING: Cannot evaluate prop-rule compliance without {target_path}")
            raise RuntimeError(
                f"MANDATE_CONFIG_MISSING: Mandate file {target_path} is required for prop-rule-aware candidate evaluation. "
                "Failing closed per safety rules."
            )
        mandate = load_mandate(target_path)

    if not equity_curve:
        return True, None

    max_daily_loss_pct = mandate.max_daily_loss_pct
    max_overall_dd_pct = mandate.max_overall_drawdown_pct

    peak_equity = 0.0
    daily_start_equity = 0.0
    current_date = None

    for point in equity_curve:
        if isinstance(point, (int, float)):
            eq = float(point)
            ts_str = None
        elif isinstance(point, dict):
            eq = float(point.get('equity', point.get('balance', 0.0)))
            ts_str = point.get('timestamp')
        else:
            continue

        if eq <= 0:
            continue

        # Initialize peak equity
        if peak_equity == 0.0 or eq > peak_equity:
            peak_equity = eq

        # Parse timestamp to track UTC date rollover
        dt = None
        if isinstance(ts_str, datetime):
            dt = ts_str
        elif isinstance(ts_str, str):
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except ValueError:
                pass

        if dt is not None:
            point_date = dt.date()
            if current_date is None or point_date > current_date:
                current_date = point_date
                daily_start_equity = eq

        if daily_start_equity == 0.0:
            daily_start_equity = eq

        # 1. Evaluate Daily Loss Breach
        daily_loss_pct = ((daily_start_equity - eq) / daily_start_equity) * 100.0 if daily_start_equity > 0 else 0.0
        if daily_loss_pct >= max_daily_loss_pct:
            reason = (
                f"MANDATE_DAILY_LOSS_BREACH: Daily loss reached {daily_loss_pct:.2f}% "
                f"(limit: {max_daily_loss_pct:.2f}%) on {current_date or 'backtest date'}"
            )
            log.warning(reason)
            return False, reason

        # 2. Evaluate Overall Drawdown Breach
        overall_dd_pct = ((peak_equity - eq) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
        if overall_dd_pct >= max_overall_dd_pct:
            reason = (
                f"MANDATE_OVERALL_DD_BREACH: Overall drawdown reached {overall_dd_pct:.2f}% "
                f"(limit: {max_overall_dd_pct:.2f}%)"
            )
            log.warning(reason)
            return False, reason

    return True, None
