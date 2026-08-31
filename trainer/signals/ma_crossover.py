"""
MA Crossover strategy template.
Entry: fast MA crosses above/below slow MA.
Exit:  opposite crossover OR SL/TP hit.

This file is shared between Trainer (backtesting)
and Engine (live signal generation) via the same interface.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from shared.indicators import crossed_above, crossed_below
from typing import Dict, Any


def generate_signal(cache: Dict[str, Any],
                    params: Dict[str, Any]) -> Dict:
    """
    Generate BUY, SELL, or HOLD signal from indicator cache.

    Args:
        cache:  dict of current + previous indicator values
                Keys: sma_fast, sma_fast_prev,
                      sma_slow, sma_slow_prev,
                      atr, adx
        params: strategy parameters dict

    Returns:
        SignalResult dict with signal, reason, sl_price, tp_price
    """
    sma_fast      = cache.get('sma_fast')
    sma_fast_prev = cache.get('sma_fast_prev')
    sma_slow      = cache.get('sma_slow')
    sma_slow_prev = cache.get('sma_slow_prev')
    atr           = cache.get('atr')
    close         = cache.get('close')

    # Require all values to be valid
    for name, val in [('sma_fast', sma_fast),
                       ('sma_slow', sma_slow),
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

    # BUY signal — fast crossed above slow
    if crossed_above(sma_fast, sma_fast_prev,
                     sma_slow, sma_slow_prev):
        sl_price = close - sl_distance
        tp_price = close + tp_distance
        return {
            'signal':    'BUY',
            'reason':    (f"MA_CROSS_BULL: fast={sma_fast:.5f} "
                         f"crossed above slow={sma_slow:.5f}"),
            'sl_price':  sl_price,
            'tp_price':  tp_price,
            'sl_pips':   sl_distance / params.get('pip_size', 0.0001),
            'rr_ratio':  rr_ratio,
            'confidence':'HIGH' if adx_threshold == 0
                         else 'MEDIUM',
        }

    # SELL signal — fast crossed below slow
    if crossed_below(sma_fast, sma_fast_prev,
                     sma_slow, sma_slow_prev):
        sl_price = close + sl_distance
        tp_price = close - tp_distance
        return {
            'signal':    'SELL',
            'reason':    (f"MA_CROSS_BEAR: fast={sma_fast:.5f} "
                         f"crossed below slow={sma_slow:.5f}"),
            'sl_price':  sl_price,
            'tp_price':  tp_price,
            'sl_pips':   sl_distance / params.get('pip_size', 0.0001),
            'rr_ratio':  rr_ratio,
            'confidence':'HIGH' if adx_threshold == 0
                         else 'MEDIUM',
        }

    return _hold("No crossover this candle")


def check_exit(cache: Dict, position: Dict,
               params: Dict) -> Dict:
    """
    Check if an open position should be closed by signal.
    Called on every candle for each open position.

    Returns exit dict with should_exit bool and reason.
    """
    exit_on_opposite = params.get('exit_on_opposite_crossover',
                                   False)
    if not exit_on_opposite:
        return {'should_exit': False, 'reason': None}

    sma_fast      = cache.get('sma_fast')
    sma_fast_prev = cache.get('sma_fast_prev')
    sma_slow      = cache.get('sma_slow')
    sma_slow_prev = cache.get('sma_slow_prev')

    if any(v is None or pd.isna(v) for v in
           [sma_fast, sma_fast_prev, sma_slow, sma_slow_prev]):
        return {'should_exit': False, 'reason': None}

    direction = position.get('direction')

    if direction == 'BUY' and crossed_below(
            sma_fast, sma_fast_prev, sma_slow, sma_slow_prev):
        return {
            'should_exit': True,
            'reason': 'EXIT_OPPOSITE_CROSSOVER'
        }

    if direction == 'SELL' and crossed_above(
            sma_fast, sma_fast_prev, sma_slow, sma_slow_prev):
        return {
            'should_exit': True,
            'reason': 'EXIT_OPPOSITE_CROSSOVER'
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