"""
Prop-Firm Mandate Evaluator for Trainer Pipeline \u2014 Fix 11.2.
Evaluates backtest trade logs and equity curves against prop-firm mandate constraints (max_daily_loss_pct, max_overall_drawdown_pct).
Hard-rejects non-compliant candidate strategies.
Fail-closed: raises RuntimeError if mandate config is missing or invalid.
"""

import logging
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timezone, date
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from engine.core.mandate import Mandate, load_mandate, MANDATE_FILE

log = logging.getLogger(__name__)
PRAGUE_TZ = ZoneInfo("Europe/Prague")


def _parse_prague_datetime(raw_ts: Any, index: int, allow_none: bool = False) -> Optional[datetime]:
    """Parse a timestamp into timezone-aware Europe/Prague datetime, failing loudly on invalid inputs."""
    if raw_ts is None:
        if allow_none:
            return None
        raise ValueError(f"Missing timestamp (None) at equity/trade point index {index}.")

    if isinstance(raw_ts, datetime):
        dt = raw_ts
    elif hasattr(raw_ts, 'to_pydatetime'):
        dt = raw_ts.to_pydatetime()
    elif isinstance(raw_ts, str):
        try:
            dt = datetime.fromisoformat(raw_ts.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError(f"Unparseable ISO timestamp at index {index}: {raw_ts!r}") from exc
    else:
        raise ValueError(
            f"Unsupported timestamp type at index {index}: {type(raw_ts).__name__} ({raw_ts!r})"
        )

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(PRAGUE_TZ)


def count_prague_trading_days(trades: List[Dict]) -> int:
    """Count distinct Europe/Prague calendar dates of trade entries; raises ValueError if timestamps missing."""
    trading_dates: set[date] = set()
    for idx, tr in enumerate(trades):
        if not isinstance(tr, dict) or 'entry_time' not in tr:
            raise ValueError(f"Trade at index {idx} missing 'entry_time' for trading-days evaluation.")
        dt_prague = _parse_prague_datetime(tr['entry_time'], idx, allow_none=False)
        trading_dates.add(dt_prague.date())
    return len(trading_dates)


def evaluate_backtest_mandate(
    equity_curve: List[Dict],
    mandate: Optional[Mandate] = None,
    mandate_path: Optional[Path] = None,
    trades: Optional[List[Dict]] = None,
    risk_per_trade_pct: Optional[float] = None,
    enforce_min_trading_days: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates a candidate's backtest equity curve against prop-firm mandate constraints.

    Args:
        equity_curve: List of dicts with keys 'timestamp' and 'equity' (or 'balance')
        mandate: Loaded Mandate instance (if None, attempts to load from mandate_path/MANDATE_FILE)
        mandate_path: Optional path to mandate.json
        trades: Optional list of trade dicts (required when enforce_min_trading_days=True)
        risk_per_trade_pct: Configured per-trade risk % to check against max_daily_risk_pct
        enforce_min_trading_days: When True, enforces mandate.min_trading_days across trades

    Returns:
        (is_compliant: bool, breach_reason: Optional[str])

    Fail-Closed:
        If mandate is missing or cannot be loaded, raises RuntimeError.
        If equity_curve contains malformed points or missing timestamps (other than point 0 MC anchor), raises ValueError.
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

    max_daily_loss_pct = mandate.max_daily_loss_pct
    max_overall_dd_pct = mandate.max_overall_drawdown_pct
    drawdown_type = mandate.drawdown_type.lower()

    if risk_per_trade_pct is not None:
        risk_val = float(risk_per_trade_pct)
        if math.isnan(risk_val) or math.isinf(risk_val) or risk_val < 0:
            raise ValueError(f"Invalid risk_per_trade_pct: {risk_per_trade_pct!r}")
        if risk_val > mandate.max_daily_risk_pct:
            reason = (
                f"MANDATE_DAILY_RISK_BREACH: Configured risk per trade {risk_val:.2f}% "
                f"exceeds mandate limit {mandate.max_daily_risk_pct:.2f}%"
            )
            log.warning(reason)
            return False, reason

    if trades is not None:
        for idx, tr in enumerate(trades):
            if not isinstance(tr, dict):
                raise ValueError(f"Invalid trade record at index {idx}: expected dict, got {type(tr).__name__}")
            if tr.get('risk_pct') is not None:
                tr_risk = float(tr['risk_pct'])
                if math.isnan(tr_risk) or math.isinf(tr_risk) or tr_risk < 0:
                    raise ValueError(f"Invalid trade risk_pct at index {idx}: {tr['risk_pct']!r}")
                if tr_risk > mandate.max_daily_risk_pct:
                    reason = (
                        f"MANDATE_DAILY_RISK_BREACH: Trade {idx} risk {tr_risk:.2f}% "
                        f"exceeds mandate limit {mandate.max_daily_risk_pct:.2f}%"
                    )
                    log.warning(reason)
                    return False, reason

    if enforce_min_trading_days:
        if trades is None:
            raise ValueError("enforce_min_trading_days=True requires explicit trades list (got None).")
        observed_days = count_prague_trading_days(trades)
        if observed_days < mandate.min_trading_days:
            reason = (
                f"MANDATE_MIN_TRADING_DAYS_BREACH: Observed {observed_days} trading days "
                f"< required minimum {mandate.min_trading_days}"
            )
            log.warning(reason)
            return False, reason

    if not equity_curve:
        return True, None

    # Fail-loud if plain floats are passed without timestamps
    if any(isinstance(p, (int, float)) for p in equity_curve):
        raise ValueError("Missing timestamps in equity curve. Mandate evaluation requires elapsed time.")

    peak_equity = 0.0
    initial_equity = 0.0
    daily_start_equity = 0.0
    current_date = None

    for i, point in enumerate(equity_curve):
        if not isinstance(point, dict):
            raise ValueError(
                f"Invalid equity curve point at index {i}: expected dict with 'timestamp' and 'equity', "
                f"got {type(point).__name__}"
            )

        if 'equity' in point and point['equity'] is not None:
            raw_eq = point['equity']
        elif 'balance' in point and point['balance'] is not None:
            raw_eq = point['balance']
        else:
            raise ValueError(f"Equity point at index {i} missing 'equity'/'balance' value.")

        try:
            eq = float(raw_eq)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Non-numeric equity at index {i}: {raw_eq!r}") from exc
        if math.isnan(eq) or math.isinf(eq):
            raise ValueError(f"Non-finite equity at index {i}: {raw_eq!r}")

        if initial_equity == 0.0:
            if eq <= 0.0:
                raise ValueError(f"Initial equity at index {i} must be strictly positive, got {eq}")
            initial_equity = eq

        if eq > peak_equity:
            peak_equity = eq

        if 'timestamp' not in point:
            raise ValueError(f"Missing 'timestamp' key at equity curve index {i}.")

        dt_prague = _parse_prague_datetime(point['timestamp'], i, allow_none=True)
        if dt_prague is not None:
            point_date = dt_prague.date()
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
        if drawdown_type == 'static':
            overall_dd_pct = ((initial_equity - eq) / initial_equity) * 100.0 if initial_equity > 0 else 0.0
        else:
            # Fallback to trailing peak
            overall_dd_pct = ((peak_equity - eq) / peak_equity) * 100.0 if peak_equity > 0 else 0.0

        if overall_dd_pct >= max_overall_dd_pct:
            reason = (
                f"MANDATE_OVERALL_DD_BREACH: Overall drawdown reached {overall_dd_pct:.2f}% "
                f"(limit: {max_overall_dd_pct:.2f}%, type: {drawdown_type})"
            )
            log.warning(reason)
            return False, reason

    return True, None
