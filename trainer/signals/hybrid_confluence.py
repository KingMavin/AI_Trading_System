"""
Hybrid Confluence strategy template.

Entry logic (deterministic, auditable):
  BUY:  close > slow_ma                         [MA trend filter: uptrend]
        AND adx >= adx_threshold                [trend strength confirmation]
        AND rsi_prev < rsi_oversold             [RSI was in oversold territory]
        AND rsi >= rsi_oversold                 [RSI has now bounced back]

  SELL: close < slow_ma                         [MA trend filter: downtrend]
        AND adx >= adx_threshold                [trend strength confirmation]
        AND rsi_prev > rsi_overbought           [RSI was in overbought territory]
        AND rsi <= rsi_overbought               [RSI has now crossed back]

Exit logic:
  Signal exit: RSI crosses the 50 midline in the opposing direction.
  Hard exit:   SL / TP (handled by Backtester intracandle logic).
"""

import pandas as pd
from typing import Dict, Any


def generate_signal(cache: Dict[str, Any],
                    params: Dict[str, Any]) -> Dict:
    """
    Generate BUY, SELL, or HOLD signal.

    Required cache keys:
        close, sma_slow, rsi, rsi_prev, adx, atr
    """
    close    = cache.get('close')
    sma_slow = cache.get('sma_slow')
    rsi      = cache.get('rsi')
    rsi_prev = cache.get('rsi_prev')
    adx      = cache.get('adx')
    atr      = cache.get('atr')

    for name, val in [('close',    close),
                      ('sma_slow', sma_slow),
                      ('rsi',      rsi),
                      ('rsi_prev', rsi_prev),
                      ('adx',      adx),
                      ('atr',      atr)]:
        if val is None or pd.isna(val):
            return _hold(f'NaN indicator: {name}')

    rsi_oversold   = params.get('rsi_oversold',   30)
    rsi_overbought = params.get('rsi_overbought', 70)
    adx_threshold  = params.get('adx_threshold',  20)
    sl_multiple    = params.get('sl_atr_multiple', 2.0)
    tp_rr          = params.get('tp_rr_ratio',     2.0)

    sl_dist = atr * sl_multiple
    tp_dist = sl_dist * tp_rr

    # BUY: uptrend + ADX confirm + RSI bounce from oversold
    if (close > sma_slow
            and adx >= adx_threshold
            and rsi_prev < rsi_oversold
            and rsi >= rsi_oversold):
        return {
            'signal':     'BUY',
            'reason':     (
                f'HYBRID_BULL: close={close:.5f}>sma_slow={sma_slow:.5f}, '
                f'adx={adx:.1f}>={adx_threshold}, '
                f'rsi_bounce {rsi_prev:.1f}->{rsi:.1f}(thr={rsi_oversold})'
            ),
            'sl_price':   close - sl_dist,
            'tp_price':   close + tp_dist,
            'sl_pips':    sl_dist / params.get('pip_size', 0.0001),
            'rr_ratio':   tp_rr,
            'confidence': 'HIGH',
        }

    # SELL: downtrend + ADX confirm + RSI cross back from overbought
    if (close < sma_slow
            and adx >= adx_threshold
            and rsi_prev > rsi_overbought
            and rsi <= rsi_overbought):
        return {
            'signal':     'SELL',
            'reason':     (
                f'HYBRID_BEAR: close={close:.5f}<sma_slow={sma_slow:.5f}, '
                f'adx={adx:.1f}>={adx_threshold}, '
                f'rsi_bounce {rsi_prev:.1f}->{rsi:.1f}(thr={rsi_overbought})'
            ),
            'sl_price':   close + sl_dist,
            'tp_price':   close - tp_dist,
            'sl_pips':    sl_dist / params.get('pip_size', 0.0001),
            'rr_ratio':   tp_rr,
            'confidence': 'HIGH',
        }

    return _hold('No confluence signal this candle')


def check_exit(cache: Dict, position: Dict,
               params: Dict) -> Dict:
    """
    Signal exit on RSI midline cross (50).
      BUY  closes when RSI crosses below 50.
      SELL closes when RSI crosses above 50.
    Hard SL/TP is handled by the Backtester intracandle loop.
    """
    rsi      = cache.get('rsi')
    rsi_prev = cache.get('rsi_prev')

    if any(v is None or pd.isna(v) for v in [rsi, rsi_prev]):
        return {'should_exit': False, 'reason': None}

    direction = position.get('direction')

    if direction == 'BUY' and rsi_prev >= 50 and rsi < 50:
        return {'should_exit': True,  'reason': 'EXIT_RSI_MIDLINE_CROSS_DOWN'}

    if direction == 'SELL' and rsi_prev <= 50 and rsi > 50:
        return {'should_exit': True,  'reason': 'EXIT_RSI_MIDLINE_CROSS_UP'}

    return {'should_exit': False, 'reason': None}


def _hold(reason: str) -> Dict:
    return {
        'signal':     'HOLD',
        'reason':     reason,
        'sl_price':   None,
        'tp_price':   None,
        'sl_pips':    None,
        'rr_ratio':   None,
        'confidence': None,
    }
