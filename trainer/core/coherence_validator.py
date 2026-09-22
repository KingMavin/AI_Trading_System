"""
Coherence Validator for Trainer Pipeline — Fix 11.1.
Rejects degenerate or invalid candidate parameter combinations prior to walk-forward optimization.
Uses empirical ATR inputs if provided; never invents fake numbers.
"""

import logging
from typing import Dict, Tuple, Optional
from shared.instrument_spec import InstrumentSpec

log = logging.getLogger(__name__)


def validate_coherence(
    params: Dict,
    instrument_spec: Optional[InstrumentSpec] = None,
    historical_atr: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validates a candidate parameter dictionary for structural coherence.

    Args:
        params: Parameter dictionary containing strategy configuration.
        instrument_spec: Optional InstrumentSpec instance for broker constraints.
        historical_atr: Optional empirical ATR statistic in price terms derived from real historical data.
                        If None, ATR-dependent stops_level and lot-size risk checks are safely skipped
                        to avoid inventing fake baseline values.

    Returns:
        (is_coherent: bool, failure_reason: Optional[str])
    """
    if not isinstance(params, dict):
        return False, "DEGENERATE_CONFIG: params must be a dictionary"

    # 1. Moving Average Crossover Order Check
    fast = params.get('fast_ma_period')
    slow = params.get('slow_ma_period')

    if fast is not None and slow is not None:
        try:
            fast_val = int(fast)
            slow_val = int(slow)
            if fast_val <= 0 or slow_val <= 0:
                return False, f"DEGENERATE_CONFIG: Moving average periods must be > 0 (got fast={fast_val}, slow={slow_val})"
            if fast_val >= slow_val:
                return False, f"DEGENERATE_CONFIG: Fast MA period ({fast_val}) must be strictly less than Slow MA period ({slow_val})"
        except (ValueError, TypeError):
            return False, f"DEGENERATE_CONFIG: Invalid MA period type (fast={fast}, slow={slow})"

    # 2. Stop-Loss & Take-Profit Non-Positive Check
    sl_atr = params.get('sl_atr_multiple')
    tp_rr = params.get('tp_rr_ratio')

    if sl_atr is not None:
        try:
            if float(sl_atr) <= 0:
                return False, f"DEGENERATE_CONFIG: sl_atr_multiple must be > 0 (got {sl_atr})"
        except (ValueError, TypeError):
            return False, f"DEGENERATE_CONFIG: Invalid sl_atr_multiple type ({sl_atr})"

    if tp_rr is not None:
        try:
            if float(tp_rr) <= 0:
                return False, f"DEGENERATE_CONFIG: tp_rr_ratio must be > 0 (got {tp_rr})"
        except (ValueError, TypeError):
            return False, f"DEGENERATE_CONFIG: Invalid tp_rr_ratio type ({tp_rr})"

    # 3. Stops Level & Instrument Minimum Lot Verification (ONLY if empirical historical_atr provided)
    if instrument_spec is not None and historical_atr is not None and historical_atr > 0:
        if sl_atr is not None:
            sl_price_distance = float(sl_atr) * float(historical_atr)
            sl_points = sl_price_distance / instrument_spec.point
            if sl_points < instrument_spec.stops_level:
                return False, (
                    f"DEGENERATE_CONFIG: Stop-loss distance ({sl_points:.1f} points) violates "
                    f"broker stops_level ({instrument_spec.stops_level} points) for {instrument_spec.symbol}"
                )

        # Position Sizing / Min Lot Risk Check
        risk_pct = float(params.get('risk_per_trade', 1.0))
        baseline_account = 10000.0
        sl_price = float(sl_atr or 1.5) * float(historical_atr)
        lots = instrument_spec.lot_size_for_risk(baseline_account, risk_pct, sl_price)
        if lots == 0.0:
            return False, (
                f"DEGENERATE_CONFIG: Position size for risk ({risk_pct}%) and SL distance ({sl_price}) "
                f"yields 0.0 lots (below min lot risk guard) for {instrument_spec.symbol}"
            )

    return True, None
