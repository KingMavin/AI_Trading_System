"""
Parameter grid generation for strategy optimisation.
Generates all valid combinations for a given template.
Respects parameter constraints (e.g. fast_ma < slow_ma).

Usage:
  from trainer.core.parameter_grid import get_grid
  combinations = get_grid('ma_crossover')
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from itertools import product
from typing import List, Dict


# ── PARAMETER SPACES ──────────────────────────────────
# Each template defines its own search space.
# Kept small for Milestone 2 — Milestone 3 expands these.

MA_CROSSOVER_SPACE = {
    'fast_ma_period':   [5, 10, 15, 20],
    'slow_ma_period':   [30, 50, 80, 100],
    'ma_type':          ['SMA'],
    'sl_atr_multiple':  [1.0, 1.5, 2.0],
    'tp_rr_ratio':      [1.5, 2.0, 2.5],
    'adx_threshold':    [0, 15, 20, 25, 30],
}

RSI_REVERSION_SPACE = {
    'rsi_period':       [14],
    'rsi_oversold':     [20, 25, 30, 35],
    'rsi_overbought':   [65, 70, 75, 80],
    'exit_on_midline':  [True, False],
    'sl_atr_multiple':  [1.0, 1.5, 2.0],
    'tp_rr_ratio':      [1.5, 2.0, 2.5],
    'adx_threshold':    [0, 15, 20, 25, 30],
    'cooldown_requires_midline': [True, False],
    'daily_loss_halt_pct': [0.0, 1.5, 2.0, 2.5, 3.0],
}

DONCHIAN_BREAKOUT_SPACE = {
    'channel_period':   [20, 40, 60, 80, 120],
    'exit_period':      [5, 10, 20, 40],
    'sl_atr_multiple':  [1.0, 1.5, 2.0],
    'tp_rr_ratio':      [1.5, 2.0, 2.5, 3.0],
    'adx_threshold':    [0, 15, 20, 25, 30],
}

HYBRID_CONFLUENCE_SPACE = {
    # Trend direction filter — slow MA only (no fast MA; entry is RSI-driven)
    'slow_ma_period':  [30, 50, 80, 100],   # empirically dominant slow values
    'ma_type':         ['SMA'],              # 100% of top-decile was SMA
    # RSI pullback entry trigger
    'rsi_period':      [14],                 # only value historically tested
    'rsi_oversold':    [25, 30, 35],         # bounce-from threshold for BUY
    'rsi_overbought':  [65, 70, 75],         # bounce-from threshold for SELL
    # Trend strength confirmation
    'adx_threshold':   [0, 20, 25],          # 20/25 dominated top-decile; 0 = control arm
    # Risk parameters — narrow range to concentrate search on signal params
    'sl_atr_multiple': [1.5, 2.0],           # 2.0 dominated 7/12 top sets
    'tp_rr_ratio':     [2.0, 2.5],           # 2.0 dominated 7/12 top sets
    # → Total: 4 × 1 × 1 × 3 × 3 × 3 × 2 × 2 = 432 combinations
    # Constraint: rsi_oversold < rsi_overbought always satisfied (max 35 < min 65)
}

TEMPLATE_SPACES = {
    'ma_crossover':      MA_CROSSOVER_SPACE,
    'rsi_reversion':     RSI_REVERSION_SPACE,
    'donchian_breakout': DONCHIAN_BREAKOUT_SPACE,
    'hybrid_confluence': HYBRID_CONFLUENCE_SPACE,
}


def get_grid(template: str,
             custom_space: Dict = None) -> List[Dict]:
    """
    Generate all valid parameter combinations for a template.

    Args:
        template:     template name e.g. 'ma_crossover', 'rsi_reversion'
        custom_space: override default space if provided

    Returns:
        list of parameter dicts — one per combination
    """
    space = custom_space or TEMPLATE_SPACES.get(template)
    if space is None:
        raise ValueError(
            f"Unknown template: {template}. "
            f"Known: {list(TEMPLATE_SPACES.keys())}"
        )

    keys   = list(space.keys())
    values = list(space.values())
    combos = list(product(*values))

    all_params = []
    for combo in combos:
        params = dict(zip(keys, combo))

        # Apply template-specific constraints
        if template == 'ma_crossover':
            # fast_ma must be strictly less than slow_ma
            if params.get('fast_ma_period', 0) >= params.get('slow_ma_period', 0):
                continue
        elif template == 'rsi_reversion':
            # rsi_oversold must be strictly less than rsi_overbought
            if params.get('rsi_oversold', 0) >= params.get('rsi_overbought', 100):
                continue
        elif template == 'donchian_breakout':
            # exit_period must be less than channel_period
            if params.get('exit_period', 0) >= params.get('channel_period', 0):
                continue

        # Add fixed params not in search space
        params['risk_per_trade_pct']       = 1.0
        params['adx_threshold']            = params.get('adx_threshold', 0)
        params['adx_min_threshold']        = params['adx_threshold']
        params['exit_on_opposite_crossover'] = False
        params['warmup_candles']           = 250

        all_params.append(params)

    return all_params


def get_grid_size(template: str) -> int:
    """Return number of valid combinations for a template."""
    return len(get_grid(template))


if __name__ == '__main__':
    combos = get_grid('ma_crossover')
    print(f"MA Crossover combinations: {len(combos)}")
    print(f"\nSample (first 5):")
    for c in combos[:5]:
        print(f"  fast={c['fast_ma_period']} "
              f"slow={c['slow_ma_period']} "
              f"sl={c['sl_atr_multiple']} "
              f"tp={c['tp_rr_ratio']}")
    print(f"\nSample (last 5):")
    for c in combos[-5:]:
        print(f"  fast={c['fast_ma_period']} "
              f"slow={c['slow_ma_period']} "
              f"sl={c['sl_atr_multiple']} "
              f"tp={c['tp_rr_ratio']}")