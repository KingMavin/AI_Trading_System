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
}

TEMPLATE_SPACES = {
    'ma_crossover': MA_CROSSOVER_SPACE,
}


def get_grid(template: str,
             custom_space: Dict = None) -> List[Dict]:
    """
    Generate all valid parameter combinations for a template.

    Args:
        template:     template name e.g. 'ma_crossover'
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
            if params['fast_ma_period'] >= params['slow_ma_period']:
                continue

        # Add fixed params not in search space
        params['risk_per_trade_pct']       = 1.0
        params['adx_min_threshold']        = 0
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