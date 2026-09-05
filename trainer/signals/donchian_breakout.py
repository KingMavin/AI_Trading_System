"""
Donchian Breakout strategy template.
Entry: Price crosses above/below the upper/lower N-period channel.
Exit:  Price crosses opposite M-period channel OR SL/TP hit.

This file is shared between Trainer (backtesting)
and Engine (live signal generation) via the same interface.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from typing import Dict, Any


def generate_signal(cache: Dict[str, Any],
                    params: Dict[str, Any]) -> Dict:
    """
    Generate BUY, SELL, or HOLD signal from indicator cache.

    Args:
        cache:  dict of current indicator values
                Keys: donchian_upper_entry, donchian_lower_entry,
                      donchian_upper_exit, donchian_lower_exit,
                      atr, adx, close
        params: strategy parameters dict

    Returns:
        SignalResult dict with signal, reason, sl_price, tp_price
    """
    upper_entry = cache.get('donchian_upper_entry')
    lower_entry = cache.get('donchian_lower_entry')
    atr         = cache.get('atr')
    close       = cache.get('close')

    # Require all values to be valid
    for name, val in [('upper_entry', upper_entry),
                      ('lower_entry', lower_entry),
                      ('atr', atr),
                      ('close', close)]:
        if val is None or pd.isna(val):
            return _hold(f"NaN indicator: {name}")

    # Optional ADX filter
    adx_threshold = params.get('adx_threshold', params.get('adx_min_threshold', 0))
    if adx_threshold > 0:
        adx = cache.get('adx')
        if adx is None or pd.isna(adx):
            return _hold(
                f"NaN ADX indicator under active ADX filter {adx_threshold}"
            )
        if adx < adx_threshold:
            return _hold(
                f"ADX {adx:.1f} below threshold {adx_threshold}"
            )

    # SL and TP calculation
    sl_distance = atr * params.get('sl_atr_multiple', 1.5)
    rr_ratio    = params.get('tp_rr_ratio', 2.0)
    tp_distance = sl_distance * rr_ratio

    # BUY signal - close > upper entry channel
    if close > upper_entry:
        sl_price = close - sl_distance
        tp_price = close + tp_distance
        return {
            'signal':    'BUY',
            'reason':    (f"DONCHIAN_BREAK_BULL: close={close:.5f} "
                         f"> upper_entry={upper_entry:.5f}"),
            'sl_price':  sl_price,
            'tp_price':  tp_price,
            'sl_pips':   sl_distance / params.get('pip_size', 0.0001),
            'rr_ratio':  rr_ratio,
            'confidence':'HIGH' if adx_threshold == 0 else 'MEDIUM',
        }

    # SELL signal - close < lower entry channel
    if close < lower_entry:
        sl_price = close + sl_distance
        tp_price = close - tp_distance
        return {
            'signal':    'SELL',
            'reason':    (f"DONCHIAN_BREAK_BEAR: close={close:.5f} "
                         f"< lower_entry={lower_entry:.5f}"),
            'sl_price':  sl_price,
            'tp_price':  tp_price,
            'sl_pips':   sl_distance / params.get('pip_size', 0.0001),
            'rr_ratio':  rr_ratio,
            'confidence':'HIGH' if adx_threshold == 0 else 'MEDIUM',
        }

    return _hold("No breakout this candle")


def check_exit(cache: Dict, position: Dict,
               params: Dict) -> Dict:
    """
    Check if an open position should be closed by signal.
    Called on every candle for each open position.

    Returns exit dict with should_exit bool and reason.
    """
    exit_on_opposite = params.get('exit_on_opposite_crossover', False)
    if not exit_on_opposite:
        return {'should_exit': False, 'reason': None}

    lower_exit = cache.get('donchian_lower_exit')
    upper_exit = cache.get('donchian_upper_exit')
    close      = cache.get('close')

    if any(v is None or pd.isna(v) for v in [lower_exit, upper_exit, close]):
        return {'should_exit': False, 'reason': None}

    direction = position.get('direction')

    # Close Buy if price < lower exit channel
    if direction == 'BUY' and close < lower_exit:
        return {
            'should_exit': True,
            'reason': 'EXIT_OPPOSITE_CHANNEL'
        }

    # Close Sell if price > upper exit channel
    if direction == 'SELL' and close > upper_exit:
        return {
            'should_exit': True,
            'reason': 'EXIT_OPPOSITE_CHANNEL'
        }

    return {'should_exit': False, 'reason': None}


def _hold(reason: str) -> Dict:
    return {
        'signal':    'HOLD',
        'reason':    reason,
        'sl_price':  None,
        'tp_price':  None,
        'sl_pips':   None,
        'rr_ratio':  None,
        'confidence':None,
    }
