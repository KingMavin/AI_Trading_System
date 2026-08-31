"""
RSI Mean-Reversion strategy template.
Entry: RSI crosses back into normal range from oversold (< oversold) or overbought (> overbought).
Exit:  RSI crosses midline (50.0) if exit_on_midline is True, OR SL/TP hit.

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
        cache:  dict of current + previous indicator values
                Keys: rsi, rsi_prev, atr, close, adx
        params: strategy parameters dict

    Returns:
        SignalResult dict with signal, reason, sl_price, tp_price
    """
    rsi       = cache.get('rsi')
    rsi_prev  = cache.get('rsi_prev')
    atr       = cache.get('atr')
    close     = cache.get('close')

    # Require all values to be valid
    for name, val in [('rsi', rsi),
                       ('rsi_prev', rsi_prev),
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

    # Threshold parameters
    rsi_oversold  = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    # Same-direction re-entry cooldown (WAVE 38)
    strategy_state = cache.get('strategy_state', {})
    last_trade = cache.get('last_closed_trade')
    
    # Track midline crossing (resets on position close)
    if (rsi_prev < 50 and rsi >= 50) or (rsi_prev > 50 and rsi <= 50):
        strategy_state['midline_crossed'] = True

    cooldown_active_for_dir = None
    if params.get('cooldown_requires_midline', False):
        if last_trade and last_trade.get('net_pnl', 0) < 0:
            if not strategy_state.get('midline_crossed', False):
                cooldown_active_for_dir = last_trade.get('direction')

    # SL and TP calculation
    sl_distance = atr * params.get('sl_atr_multiple', 1.5)
    rr_ratio    = params.get('tp_rr_ratio', 2.0)
    tp_distance = sl_distance * rr_ratio

    # BUY signal — RSI crossed back above oversold threshold
    if rsi_prev < rsi_oversold and rsi >= rsi_oversold:
        if cooldown_active_for_dir == 'BUY':
            return _hold("COOLDOWN_ACTIVE: Waiting for midline cross after BUY loss")
            
        sl_price = close - sl_distance
        tp_price = close + tp_distance
        return {
            'signal':    'BUY',
            'reason':    (f"RSI_OVERSOLD_REVERSION: rsi_prev={rsi_prev:.2f} < {rsi_oversold} "
                          f"-> rsi={rsi:.2f} >= {rsi_oversold}"),
            'sl_price':  sl_price,
            'tp_price':  tp_price,
            'sl_pips':   sl_distance / params.get('pip_size', 0.0001),
            'rr_ratio':  rr_ratio,
            'confidence':'HIGH' if adx_threshold == 0 else 'MEDIUM',
        }

    # SELL signal — RSI crossed back below overbought threshold
    if rsi_prev > rsi_overbought and rsi <= rsi_overbought:
        if cooldown_active_for_dir == 'SELL':
            return _hold("COOLDOWN_ACTIVE: Waiting for midline cross after SELL loss")
            
        sl_price = close + sl_distance
        tp_price = close - tp_distance
        return {
            'signal':    'SELL',
            'reason':    (f"RSI_OVERBOUGHT_REVERSION: rsi_prev={rsi_prev:.2f} > {rsi_overbought} "
                          f"-> rsi={rsi:.2f} <= {rsi_overbought}"),
            'sl_price':  sl_price,
            'tp_price':  tp_price,
            'sl_pips':   sl_distance / params.get('pip_size', 0.0001),
            'rr_ratio':  rr_ratio,
            'confidence':'HIGH' if adx_threshold == 0 else 'MEDIUM',
        }

    return _hold("No RSI threshold reversion this candle")


def check_exit(cache: Dict, position: Dict,
               params: Dict) -> Dict:
    """
    Check if an open position should be closed by signal (midline reversion exit).
    Called on every candle for each open position.

    Returns exit dict with should_exit bool and reason.
    """
    exit_on_midline = params.get('exit_on_midline', True)
    if not exit_on_midline:
        return {'should_exit': False, 'reason': None}

    rsi = cache.get('rsi')
    if rsi is None or pd.isna(rsi):
        return {'should_exit': False, 'reason': None}

    direction = position.get('direction')

    if direction == 'BUY' and rsi >= 50.0:
        return {
            'should_exit': True,
            'reason': 'EXIT_RSI_MIDLINE'
        }

    if direction == 'SELL' and rsi <= 50.0:
        return {
            'should_exit': True,
            'reason': 'EXIT_RSI_MIDLINE'
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
