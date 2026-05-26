"""
Simulation engine — runs a strategy on historical OHLCV data.
Evaluates on candle close only. Applies spread, slippage,
and commission costs. Uses conservative intracandle fill model.

This is the core of the Trainer. The Engine uses identical
logic for live trading via the shared indicators module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import importlib

from shared.indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_bbands, calculate_atr, calculate_adx,
    calculate_volume_sma, calculate_session, calculate_regime
)


# ── SYMBOL SPECIFICATIONS ──────────────────────────────
SYMBOL_SPECS = {
    'EURUSD': {
        'pip_size':           0.0001,
        'lot_step':           0.01,
        'lot_min':            0.01,
        'lot_max':            100.0,
        'stops_level_pips':   1.0,
        'typical_spread':     1.0,    # pips
        'typical_slippage':   0.3,    # pips
        'commission_per_lot': 7.0,    # USD round trip
    },
    'GBPUSD': {
        'pip_size':           0.0001,
        'lot_step':           0.01,
        'lot_min':            0.01,
        'lot_max':            100.0,
        'stops_level_pips':   1.0,
        'typical_spread':     1.2,
        'typical_slippage':   0.4,
        'commission_per_lot': 7.0,
    },
    'USDJPY': {
        'pip_size':           0.01,
        'lot_step':           0.01,
        'lot_min':            0.01,
        'lot_max':            100.0,
        'stops_level_pips':   1.0,
        'typical_spread':     0.9,
        'typical_slippage':   0.3,
        'commission_per_lot': 7.0,
    },
}

# Session multipliers for spread model
SESSION_SPREAD_MULT = {
    'LONDON':           1.0,
    'LONDON_NY_OVERLAP':1.0,
    'NEW_YORK':         1.1,
    'LONDON_OPEN':      1.3,
    'NY_OPEN':          1.2,
    'ASIAN':            1.5,
    'DEAD_ZONE':        2.5,
}

# Volatility multipliers for spread model
def _vol_mult(atr_ratio: float) -> float:
    if atr_ratio < 0.7:  return 0.9
    if atr_ratio < 1.3:  return 1.0
    if atr_ratio < 2.0:  return 1.4
    return 2.0


class Backtester:
    """
    Runs a strategy on historical OHLCV data.
    One Backtester instance per backtest run.
    Reset between runs.
    """

    def __init__(self,
                 symbol: str,
                 params: Dict,
                 initial_equity: float = 10000.0,
                 random_seed: int = 42):
        self.symbol         = symbol
        self.params         = params
        self.initial_equity = initial_equity
        self.specs          = SYMBOL_SPECS[symbol]
        self.pip_size       = self.specs['pip_size']
        self.rng            = np.random.default_rng(random_seed)

        # Account state
        self.equity         = initial_equity
        self.balance        = initial_equity
        self.peak_equity    = initial_equity

        # Position tracking
        self.open_position: Optional[Dict] = None
        self.closed_trades: List[Dict]     = []
        self.equity_curve:  List[float]    = [initial_equity]

    def run(self, df: pd.DataFrame,
            signal_module) -> Dict:
        """
        Run backtest on DataFrame.

        Args:
            df:            OHLCV DataFrame with DatetimeIndex
            signal_module: module with generate_signal() and
                           check_exit() functions

        Returns:
            dict with trades, equity_curve, metrics
        """
        # Pre-compute all indicators
        features = self._compute_features(df)

        # Main candle loop
        warmup = self.params.get('warmup_candles', 250)

        for i in range(warmup, len(features)):
            row     = features.iloc[i]
            row_prev= features.iloc[i - 1]

            # Build indicator cache
            cache = self._build_cache(row, row_prev)

            # Skip Dead Zone
            if cache['session'] == 'DEAD_ZONE':
                continue

            # Step 1: Check SL/TP on open position
            if self.open_position:
                self._check_sl_tp(row, cache)

            # Step 2: Check signal-based exit
            if self.open_position:
                exit_result = signal_module.check_exit(
                    cache, self.open_position, self.params
                )
                if exit_result['should_exit']:
                    self._close_position(
                        price=cache['close'],
                        candle_time=row.name,
                        reason=exit_result['reason'],
                        cache=cache
                    )

            # Step 3: Generate entry signal
            if self.open_position is None:
                signal = signal_module.generate_signal(
                    cache, self.params
                )
                if signal['signal'] in ('BUY', 'SELL'):
                    self._open_position(
                        signal=signal,
                        candle_time=row.name,
                        cache=cache
                    )

            # Record equity
            unrealised = self._unrealised_pnl(cache['close'])
            self.equity_curve.append(self.balance + unrealised)

        # Close any open position at end
        if self.open_position:
            last_row = features.iloc[-1]
            last_cache = self._build_cache(
                last_row, features.iloc[-2]
            )
            self._close_position(
                price=last_cache['close'],
                candle_time=last_row.name,
                reason='END_OF_DATA',
                cache=last_cache
            )

        return {
            'trades':      self.closed_trades,
            'equity_curve':self.equity_curve,
            'symbol':      self.symbol,
            'params':      self.params,
        }

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute all indicators on the full dataset."""
        f = df.copy()

        fast = self.params.get('fast_ma_period', 10)
        slow = self.params.get('slow_ma_period', 50)
        ma_type = self.params.get('ma_type', 'SMA')

        if ma_type == 'EMA':
            f['sma_fast'] = calculate_ema(df, fast)
            f['sma_slow'] = calculate_ema(df, slow)
        else:
            f['sma_fast'] = calculate_sma(df, fast)
            f['sma_slow'] = calculate_sma(df, slow)

        f['atr']        = calculate_atr(df, 14)
        adx_result      = calculate_adx(df, 14)
        f['adx']        = adx_result['adx']
        f['volume_sma'] = calculate_volume_sma(df, 20)
        f['atr_sma20']  = f['atr'].rolling(20).mean()
        f['atr_ratio']  = f['atr'] / f['atr_sma20']
        f['session']    = calculate_session(df)
        f['regime']     = calculate_regime(
            df, f['adx'], f['atr']
        )
        return f

    def _build_cache(self, row, row_prev) -> Dict:
        """Build indicator cache for current candle."""
        return {
            'close':         row['close'],
            'high':          row['high'],
            'low':           row['low'],
            'open':          row['open'],
            'volume':        row['volume'],
            'sma_fast':      row['sma_fast'],
            'sma_fast_prev': row_prev['sma_fast'],
            'sma_slow':      row['sma_slow'],
            'sma_slow_prev': row_prev['sma_slow'],
            'atr':           row['atr'],
            'adx':           row['adx'],
            'atr_ratio':     row.get('atr_ratio', 1.0),
            'volume_sma':    row['volume_sma'],
            'session':       row['session'],
            'regime':        row['regime'],
        }

    def _simulated_spread(self, cache: Dict) -> float:
        """Dynamic spread based on session and volatility."""
        base     = self.specs['typical_spread']
        sess_m   = SESSION_SPREAD_MULT.get(
            cache['session'], 1.0
        )
        vol_m    = _vol_mult(
            cache.get('atr_ratio', 1.0) or 1.0
        )
        noise    = self.rng.uniform(0.85, 1.15)
        return base * sess_m * vol_m * noise

    def _simulated_slippage(self, cache: Dict) -> float:
        """Dynamic slippage based on volatility."""
        base  = self.specs['typical_slippage']
        vol_m = _vol_mult(cache.get('atr_ratio', 1.0) or 1.0)
        noise = self.rng.uniform(0.85, 1.15)
        return base * vol_m * noise

    def _calculate_lot_size(self, sl_price: float,
                             entry_price: float) -> float:
        """Calculate position size based on risk percentage."""
        risk_pct    = self.params.get('risk_per_trade_pct', 1.0)
        risk_amount = self.balance * (risk_pct / 100)

        sl_distance = abs(entry_price - sl_price)
        sl_pips     = sl_distance / self.pip_size

        # Pip value: $10 per pip per lot for USD pairs
        pip_value_per_lot = 10.0  # approximate for major pairs

        if sl_pips <= 0:
            return self.specs['lot_min']

        raw_lots = risk_amount / (sl_pips * pip_value_per_lot)

        # Round down to lot step
        step      = self.specs['lot_step']
        adj_lots  = int(raw_lots / step) * step
        adj_lots  = max(adj_lots, self.specs['lot_min'])
        adj_lots  = min(adj_lots, self.specs['lot_max'])

        return round(adj_lots, 2)

    def _open_position(self, signal: Dict,
                        candle_time,
                        cache: Dict) -> None:
        """Open a new position."""
        direction   = signal['signal']
        spread_pips = self._simulated_spread(cache)
        slip_pips   = self._simulated_slippage(cache)
        spread      = spread_pips * self.pip_size
        slip        = slip_pips * self.pip_size

        if direction == 'BUY':
            entry_price = cache['close'] + (spread / 2) + slip
        else:
            entry_price = cache['close'] - (spread / 2) - slip

        sl_price = signal['sl_price']
        tp_price = signal['tp_price']

        lots        = self._calculate_lot_size(sl_price, entry_price)
        commission  = lots * self.specs['commission_per_lot'] / 2

        self.balance -= commission

        self.open_position = {
            'direction':     direction,
            'entry_price':   entry_price,
            'entry_time':    candle_time,
            'sl_price':      sl_price,
            'tp_price':      tp_price,
            'lots':          lots,
            'commission_open':commission,
            'sl_distance':   abs(entry_price - sl_price),
            'entry_session': cache['session'],
            'entry_regime':  cache['regime'],
            'candles_open':  0,
            'highest_price': entry_price,
            'lowest_price':  entry_price,
        }

    def _check_sl_tp(self, row, cache: Dict) -> None:
        """
        Check if SL or TP was hit during this candle.
        Uses conservative intracandle model:
          BUY:  check low (SL) first, then high (TP)
          SELL: check high (SL) first, then low (TP)
        """
        pos = self.open_position
        if pos is None:
            return

        candle_high = row['high']
        candle_low  = row['low']
        candle_open = row['open']

        if pos['direction'] == 'BUY':
            # Check SL first (worst case — low comes first)
            if candle_low <= pos['sl_price']:
                fill = min(pos['sl_price'], candle_open)
                self._close_position(
                    price=fill,
                    candle_time=row.name,
                    reason='SL_HIT',
                    cache=cache
                )
                return
            # Check TP
            if candle_high >= pos['tp_price']:
                self._close_position(
                    price=pos['tp_price'],
                    candle_time=row.name,
                    reason='TP_HIT',
                    cache=cache
                )
                return

        else:  # SELL
            # Check SL first (worst case — high comes first)
            if candle_high >= pos['sl_price']:
                fill = max(pos['sl_price'], candle_open)
                self._close_position(
                    price=fill,
                    candle_time=row.name,
                    reason='SL_HIT',
                    cache=cache
                )
                return
            # Check TP
            if candle_low <= pos['tp_price']:
                self._close_position(
                    price=pos['tp_price'],
                    candle_time=row.name,
                    reason='TP_HIT',
                    cache=cache
                )
                return

        # Update tracking
        pos['candles_open']  += 1
        pos['highest_price']  = max(
            pos['highest_price'], candle_high
        )
        pos['lowest_price']   = min(
            pos['lowest_price'], candle_low
        )

    def _close_position(self, price: float,
                         candle_time,
                         reason: str,
                         cache: Dict) -> None:
        """Close the current open position."""
        pos = self.open_position
        if pos is None:
            return

        commission_close = (
            pos['lots'] * self.specs['commission_per_lot'] / 2
        )

        if pos['direction'] == 'BUY':
            gross_pnl = (price - pos['entry_price']) * \
                        pos['lots'] * 100000
        else:
            gross_pnl = (pos['entry_price'] - price) * \
                        pos['lots'] * 100000

        net_pnl = gross_pnl - pos['commission_open'] \
                  - commission_close

        self.balance    += net_pnl
        self.equity      = self.balance
        self.peak_equity = max(self.peak_equity, self.equity)

        self.closed_trades.append({
            'direction':      pos['direction'],
            'entry_price':    pos['entry_price'],
            'entry_time':     pos['entry_time'],
            'exit_price':     price,
            'exit_time':      candle_time,
            'lots':           pos['lots'],
            'gross_pnl':      round(gross_pnl, 2),
            'net_pnl':        round(net_pnl, 2),
            'commission':     round(
                pos['commission_open'] + commission_close, 2
            ),
            'close_reason':   reason,
            'duration_candles':pos['candles_open'],
            'entry_session':  pos['entry_session'],
            'entry_regime':   pos['entry_regime'],
            'exit_session':   cache['session'],
        })

        self.open_position = None

    def _unrealised_pnl(self, current_price: float) -> float:
        """Calculate unrealised P&L on open position."""
        if self.open_position is None:
            return 0.0
        pos = self.open_position
        if pos['direction'] == 'BUY':
            return (current_price - pos['entry_price']) * \
                   pos['lots'] * 100000
        else:
            return (pos['entry_price'] - current_price) * \
                   pos['lots'] * 100000