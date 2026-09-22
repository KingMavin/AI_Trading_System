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
from datetime import datetime

from shared.indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_bbands, calculate_atr, calculate_adx,
    calculate_volume_sma, calculate_session, calculate_regime,
    calculate_donchian_channel
)


import logging
log = logging.getLogger(__name__)

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
                 random_seed: int = 42,
                 allow_simulation_defaults: bool = False):
        self.symbol         = symbol
        self.params         = params
        self.initial_equity = initial_equity
        
        from shared.instrument_spec import get_spec, InstrumentSpec
        self.instrument_spec = get_spec(symbol)
        
        if self.instrument_spec is None:
            if not allow_simulation_defaults:
                raise ValueError(f"{symbol}: No captured InstrumentSpec found and allow_simulation_defaults=False.")
            log.warning(
                f"{symbol}: no captured InstrumentSpec found. "
                f"Falling back to SIMULATION_DEFAULT. "
                f"Any generated score/report must be tagged spec_source: 'SIMULATION_DEFAULT'."
            )
            self.instrument_spec = InstrumentSpec.build_simulation_default(symbol)
            
        if not self.instrument_spec.sanity_ok and not allow_simulation_defaults:
            raise ValueError(f"{symbol}: InstrumentSpec sanity_ok is False. Refusing to backtest.")

        self.pip_size       = self.instrument_spec.pip_size
        self.rng            = np.random.default_rng(random_seed)

        # Account state
        self.equity         = initial_equity
        self.balance        = initial_equity
        self.peak_equity    = initial_equity

        # State & Halts (WAVE 38)
        self.strategy_state = {}
        self.last_closed_trade = None
        self.daily_halt_active = False

        # Position tracking
        self.open_position: Optional[Dict] = None
        self.closed_trades: List[Dict]     = []
        self.equity_curve:  List[float]    = [initial_equity]

    def run(self, df: pd.DataFrame,
            signal_module) -> Dict:
        """
        Run backtest on DataFrame.
        Evaluates signals on candle close, executes fills on next candle open (zero look-ahead).

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
        pending_entry: Optional[Dict] = None
        pending_exit: Optional[Dict] = None

        # Date rollover tracking
        current_date = None
        daily_start_equity = self.equity
        daily_loss_halt_pct = self.params.get('daily_loss_halt_pct', 0.0)
        dates = features.index.date

        for i in range(warmup, len(features)):
            row     = features.iloc[i]
            row_prev= features.iloc[i - 1]

            # Rollover check
            if current_date is None or dates[i] > current_date:
                current_date = dates[i]
                daily_start_equity = self.balance + self._unrealised_pnl(row['close'])
                self.daily_halt_active = False

            # Daily halt evaluation
            if daily_loss_halt_pct > 0.0 and not self.daily_halt_active:
                current_equity = self.balance + self._unrealised_pnl(row['close'])
                daily_pnl_pct = ((current_equity - daily_start_equity) / daily_start_equity) * 100.0 if daily_start_equity > 0 else -100.0
                if daily_pnl_pct <= -daily_loss_halt_pct:
                    self.daily_halt_active = True

            # Build indicator cache
            cache = self._build_cache(row, row_prev)
            
            # Inject state (reference semantics, WAVE 38)
            cache['strategy_state'] = self.strategy_state
            cache['last_closed_trade'] = self.last_closed_trade

            # Process pending signal-based exit from candle i-1 at candle i's open
            if pending_exit and self.open_position:
                self._close_position(
                    price=row['open'],
                    candle_time=row.name,
                    reason=pending_exit['reason'],
                    cache=cache
                )
                pending_exit = None

            # Process pending entry signal from candle i-1 at candle i's open
            if pending_entry and self.open_position is None:
                self._open_position(
                    signal=pending_entry['signal'],
                    base_price=row['open'],
                    candle_time=row.name,
                    cache=cache
                )
                pending_entry = None

            # Skip Dead Zone for new evaluations/checks. Note: pending entry/exit execution above
            # is NOT gated by session because orders queued prior to Dead Zone execute at market open tick.
            if cache['session'] == 'DEAD_ZONE':
                continue

            # Step 1: Check SL/TP on open position during candle i
            if self.open_position:
                self._check_sl_tp(row, cache)

            # Step 2: Check signal-based exit (evaluated at candle i close, queued for candle i+1 open)
            if self.open_position and pending_exit is None:
                exit_result = signal_module.check_exit(
                    cache, self.open_position, self.params
                )
                if exit_result['should_exit']:
                    pending_exit = {'reason': exit_result['reason']}

            # Daily halt re-evaluation in case a position closed intracandle
            if daily_loss_halt_pct > 0.0 and not self.daily_halt_active:
                current_equity = self.balance + self._unrealised_pnl(row['close'])
                daily_pnl = ((current_equity - daily_start_equity) / daily_start_equity) * 100.0 if daily_start_equity > 0 else -100.0
                if daily_pnl <= -daily_loss_halt_pct:
                    self.daily_halt_active = True

            # Step 3: Generate entry signal (evaluated at candle i close, queued for candle i+1 open)
            if self.open_position is None and pending_entry is None and not self.daily_halt_active:
                signal = signal_module.generate_signal(
                    cache, self.params
                )
                if signal['signal'] in ('BUY', 'SELL'):
                    pending_entry = {'signal': signal}

            # Record equity
            unrealised = self._unrealised_pnl(cache['close'])
            self.equity_curve.append(self.balance + unrealised)

        # Close any open position at end of data
        if self.open_position:
            last_row = features.iloc[-1]
            last_cache = self._build_cache(
                last_row, features.iloc[-2]
            )
            self._close_position(
                price=last_row['close'],
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

        f['rsi']        = calculate_rsi(df, 14)
        bb              = calculate_bbands(df, 20, 2.0)
        f['bb_pct']     = bb['bb_pct']
        f['atr']        = calculate_atr(df, 14)
        adx_result      = calculate_adx(df, 14)
        f['adx']        = adx_result['adx']
        f['volume_sma'] = calculate_volume_sma(df, 20)
        f['volume_rel'] = f['volume'] / f['volume_sma']
        f['atr_sma20']  = f['atr'].rolling(20).mean()
        f['atr_ratio']  = f['atr'] / f['atr_sma20']
        f['session']    = calculate_session(df)
        f['regime']     = calculate_regime(
            df, f['adx'], f['atr']
        )
        
        if 'channel_period' in self.params:
            d_entry = calculate_donchian_channel(df, self.params['channel_period'])
            f['donchian_upper_entry'] = d_entry['upper']
            f['donchian_lower_entry'] = d_entry['lower']
            
            d_exit = calculate_donchian_channel(df, self.params['exit_period'])
            f['donchian_upper_exit'] = d_exit['upper']
            f['donchian_lower_exit'] = d_exit['lower']

        return f

    @staticmethod
    def _clean_float(val, default: float) -> float:
        """Sanitize float values, guarding against None, NaN, or inf during warmup."""
        if val is None or pd.isna(val) or np.isinf(val):
            return float(default)
        return float(val)

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
            'rsi':           self._clean_float(row.get('rsi'), 50.0),
            'rsi_prev':      self._clean_float(row_prev.get('rsi'), 50.0),
            'bb_pct':        self._clean_float(row.get('bb_pct'), 0.5),
            'atr':           self._clean_float(row.get('atr'), 0.0),
            'adx':           self._clean_float(row.get('adx'), 0.0),
            'atr_ratio':     self._clean_float(row.get('atr_ratio'), 1.0),
            'volume_sma':    row.get('volume_sma', row['volume']),
            'volume_rel':    self._clean_float(row.get('volume_rel'), 1.0),
            'session':       row['session'],
            'regime':        row['regime'],
            'donchian_upper_entry': row.get('donchian_upper_entry'),
            'donchian_lower_entry': row.get('donchian_lower_entry'),
            'donchian_upper_exit':  row.get('donchian_upper_exit'),
            'donchian_lower_exit':  row.get('donchian_lower_exit'),
        }

    def _simulated_spread(self, cache: Dict) -> float:
        """Dynamic spread based on session and volatility."""
        # Convert typical spread in points to pips
        point_value = self.instrument_spec.spread_typical * self.instrument_spec.point
        base = self.instrument_spec.price_to_pips(point_value) if self.instrument_spec.pip_size > 0 else 1.0
        
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
        base  = 0.3  # typical slippage in pips
        vol_m = _vol_mult(cache.get('atr_ratio', 1.0) or 1.0)
        noise = self.rng.uniform(0.85, 1.15)
        return base * vol_m * noise

    def _calculate_lot_size(self, sl_price: float,
                             entry_price: float) -> float:
        """Calculate position size based on risk percentage."""
        risk_pct = self.params.get('risk_per_trade_pct', 1.0)
        distance = abs(entry_price - sl_price)
        return self.instrument_spec.lot_size_for_risk(self.balance, risk_pct, distance)

    def _open_position(self, signal: Dict,
                        base_price: float,
                        candle_time,
                        cache: Dict) -> None:
        """Open a new position at base_price (next candle open) + spread / slip."""
        direction   = signal['signal']
        spread_pips = self._simulated_spread(cache)
        slip_pips   = self._simulated_slippage(cache)
        spread      = spread_pips * self.pip_size
        slip        = slip_pips * self.pip_size

        if direction == 'BUY':
            entry_price = base_price + (spread / 2) + slip
        else:
            entry_price = base_price - (spread / 2) - slip

        sl_price = signal['sl_price']
        tp_price = signal['tp_price']

        lots        = self._calculate_lot_size(sl_price, entry_price)
        commission  = lots * 7.0 / 2  # Hardcoded $7/lot RT

        self.balance -= commission

        # Distance from fast MA specifically (sma_fast), in price ticks
        fast_ma = cache.get('sma_fast')
        if fast_ma is not None and not pd.isna(fast_ma) and self.instrument_spec and self.instrument_spec.tick_size > 0:
            ma_dist_ticks = abs(entry_price - fast_ma) / self.instrument_spec.tick_size
        else:
            ma_dist_ticks = 0.0

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
            # Category A/B Feature Snapshot (captured at entry-open time)
            'rsi':              self._clean_float(cache.get('rsi'), 50.0),
            'bb_pct':           self._clean_float(cache.get('bb_pct'), 0.5),
            'volatility_atr':   self._clean_float(cache.get('atr'), 0.0),
            'volatility_ratio': self._clean_float(cache.get('atr_ratio'), 1.0),
            'volume_relative':  self._clean_float(cache.get('volume_rel'), 1.0),
            'ma_distance_ticks':round(ma_dist_ticks, 2),
            'hour_utc':         candle_time.hour,
            'day_of_week':      candle_time.weekday(),
            'accrued_swap':     0.0,  # Snapshotting at entry-open time before rollover
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

        commission_close = pos['lots'] * 7.0 / 2  # Hardcoded $7/lot RT
        
        ticks_won = (price - pos['entry_price']) / self.instrument_spec.tick_size
        if pos['direction'] == 'SELL':
            ticks_won = -ticks_won
        gross_pnl = ticks_won * pos['lots'] * self.instrument_spec.tick_value

        swap_cost = self._calculate_swap_cost(
            direction=pos['direction'],
            lots=pos['lots'],
            entry_time=pos['entry_time'],
            exit_time=candle_time
        )

        net_pnl = gross_pnl - pos['commission_open'] - commission_close + swap_cost

        self.balance    += net_pnl
        self.equity      = self.balance
        self.peak_equity = max(self.peak_equity, self.equity)

        self.closed_trades.append({
            'direction':         pos['direction'],
            'entry_price':       pos['entry_price'],
            'entry_time':        pos['entry_time'],
            'exit_price':        price,
            'exit_time':         candle_time,
            'lots':              pos['lots'],
            'gross_pnl':         round(gross_pnl, 2),
            'net_pnl':           round(net_pnl, 2),
            'commission':        round(pos['commission_open'] + commission_close, 2),
            'swap_cost':         round(swap_cost, 2),
            'close_reason':      reason,
            'duration_candles':  pos['candles_open'],
            'entry_session':     pos['entry_session'],
            'entry_regime':      pos['entry_regime'],
            'exit_session':      cache['session'],
            # Feature Snapshot fields
            'rsi':               pos.get('rsi'),
            'bb_pct':            pos.get('bb_pct'),
            'volatility_atr':    pos.get('volatility_atr'),
            'volatility_ratio':  pos.get('volatility_ratio'),
            'volume_relative':   pos.get('volume_relative'),
            'ma_distance_ticks': pos.get('ma_distance_ticks'),
            'hour_utc':          pos.get('hour_utc'),
            'day_of_week':       pos.get('day_of_week'),
            'accrued_swap':      pos.get('accrued_swap', 0.0),
        })

        self.open_position = None
        
        # Clear per-trade strategy state (WAVE 38)
        self.strategy_state.clear()
        self.last_closed_trade = {
            'net_pnl': net_pnl,
            'direction': pos['direction'],
            'reason': reason
        }

    def _unrealised_pnl(self, current_price: float) -> float:
        """Calculate unrealised P&L on open position."""
        if self.open_position is None:
            return 0.0
        pos = self.open_position
        ticks_won = (current_price - pos['entry_price']) / self.instrument_spec.tick_size
        if pos['direction'] == 'SELL':
            ticks_won = -ticks_won
        return ticks_won * pos['lots'] * self.instrument_spec.tick_value

    def _calculate_swap_cost(self, direction: str, lots: float, entry_time: datetime, exit_time: datetime) -> float:
        """
        Calculate total swap cost between entry and exit.
        Swap is applied at 21:00 UTC (Rollover).
        Wednesday at 21:00 UTC charges 3x swap.
        Saturday and Sunday at 21:00 UTC charge 0x swap.
        """
        if self.instrument_spec is None:
            raise ValueError("InstrumentSpec is required for swap calculation")
            
        rate = self.instrument_spec.swap_long if direction == 'BUY' else self.instrument_spec.swap_short
        total_cost = 0.0
        
        from datetime import timedelta
        
        curr_day = entry_time.replace(hour=21, minute=0, second=0, microsecond=0)
        if curr_day <= entry_time:
            curr_day += timedelta(days=1)
            
        while curr_day < exit_time:
            weekday = curr_day.weekday()
            if weekday == 2:  # Wednesday
                total_cost += rate * lots * 3.0
            elif weekday in (5, 6):  # Saturday, Sunday
                pass # 0 swap
            else:
                total_cost += rate * lots * 1.0
            curr_day += timedelta(days=1)
            
        return total_cost
