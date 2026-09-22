"""
Signal Engine — generates trading signals from live MT5 data.

Called on every candle close. Reads live OHLCV data from MT5,
computes indicators, runs the strategy template, applies
pre-trade checklist, and returns a signal decision.

This is the Engine's brain. It never places orders —
it only decides what to do. Order execution is separate.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional

from shared.indicators import (
    calculate_sma, calculate_ema, calculate_atr,
    calculate_adx, calculate_rsi, calculate_volume_sma,
    calculate_session, calculate_regime,
    crossed_above, crossed_below
)
from engine.core.strategy_loader import StrategySpec

log = logging.getLogger(__name__)

# How many candles to fetch from MT5 for indicator warmup
FETCH_CANDLES = 300


class SignalEngine:
    """
    Computes indicators and generates trading signals
    from live MT5 OHLCV data.

    One instance per Engine run. Maintains indicator
    cache between candles for crossover detection.
    """

    def __init__(self, spec: StrategySpec):
        self.spec           = spec
        self.indicator_cache: Dict = {}
        self.candles_processed = 0
        log.info(
            f"SignalEngine initialised: "
            f"{spec.strategy_id} | "
            f"{spec.template} | "
            f"fast={spec.fast_ma} slow={spec.slow_ma}"
        )

    def update_spec(self, spec: StrategySpec) -> None:
        """Update strategy spec when new strategy deployed."""
        log.info(
            f"SignalEngine: strategy updated "
            f"{self.spec.strategy_id} → {spec.strategy_id}"
        )
        self.spec             = spec
        self.indicator_cache  = {}  # reset cache

    def process_candle(self, df: pd.DataFrame, strategy_state: Dict = None, last_closed_trade: Dict = None) -> Dict:
        """
        Process one candle of live data.
        Called on every M15 candle close.

        Args:
            df: DataFrame containing the latest candles
            strategy_state: Per-trade strategy state dict
            last_closed_trade: Last closed trade dict

        Returns:
            SignalResult dict with signal, reason, prices
        """
        if df is None or len(df) < 50:
            return self._no_signal('Insufficient data')

        try:
            # Compute all indicators on full DataFrame
            cache = self._compute_indicators(df)

            # Update instance cache for next candle
            prev_cache          = self.indicator_cache.copy()
            self.indicator_cache= cache
            self.candles_processed += 1

            # Need at least 2 candles for crossover detection
            if self.candles_processed < 2:
                return self._no_signal('Warming up indicators')

            # Add previous values for crossover detection
            cache['sma_fast_prev'] = prev_cache.get('sma_fast')
            cache['sma_slow_prev'] = prev_cache.get('sma_slow')

            # Inject state (WAVE 38)
            if strategy_state is not None:
                cache['strategy_state'] = strategy_state
            if last_closed_trade is not None:
                cache['last_closed_trade'] = last_closed_trade

            # Run strategy signal generation
            return self._generate_signal(cache)

        except Exception as e:
            log.error(f"Signal generation error: {e}",
                      exc_info=True)
            return {'signal': 'ERROR', 'reason': f'SIGNAL_EVALUATION_FAILED: {e}'}

    def _compute_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Compute all indicators on the DataFrame.
        Returns dict of current candle values.
        """
        spec = self.spec

        # Moving averages
        if spec.parameters.get('ma_type') == 'EMA':
            fast_series = calculate_ema(df, spec.fast_ma)
            slow_series = calculate_ema(df, spec.slow_ma)
        else:
            fast_series = calculate_sma(df, spec.fast_ma)
            slow_series = calculate_sma(df, spec.slow_ma)

        # Volatility and trend
        atr_series = calculate_atr(df, 14)
        adx_result = calculate_adx(df, 14)
        adx_series = adx_result['adx']

        # Volume
        vol_sma = calculate_volume_sma(df, 20)

        # RSI
        from shared.indicators import calculate_rsi, calculate_donchian_channel
        rsi_series = calculate_rsi(df, period=spec.parameters.get('rsi_period', 14))

        # Donchian Channels
        d_entry = calculate_donchian_channel(df, spec.parameters.get('channel_period', 20))
        d_exit  = calculate_donchian_channel(df, spec.parameters.get('exit_period', 10))

        # ATR ratio for regime detection
        atr_sma20 = atr_series.rolling(20).mean()
        atr_ratio = atr_series / atr_sma20

        # Session and regime on last candle
        session = calculate_session(df).iloc[-1]
        regime  = calculate_regime(
            df, adx_series, atr_series
        ).iloc[-1]

        # Extract current (last) candle values
        return {
            'close':        float(df['close'].iloc[-1]),
            'high':         float(df['high'].iloc[-1]),
            'low':          float(df['low'].iloc[-1]),
            'open':         float(df['open'].iloc[-1]),
            'volume':       float(df['volume'].iloc[-1]),
            'sma_fast':     float(fast_series.iloc[-1])
                            if not pd.isna(fast_series.iloc[-1])
                            else None,
            'sma_fast_prev':float(fast_series.iloc[-2])
                            if len(fast_series) > 1 and not pd.isna(fast_series.iloc[-2])
                            else None,
            'sma_slow':     float(slow_series.iloc[-1])
                            if not pd.isna(slow_series.iloc[-1])
                            else None,
            'sma_slow_prev':float(slow_series.iloc[-2])
                            if len(slow_series) > 1 and not pd.isna(slow_series.iloc[-2])
                            else None,
            'rsi':          float(rsi_series.iloc[-1])
                            if not pd.isna(rsi_series.iloc[-1])
                            else None,
            'rsi_prev':     float(rsi_series.iloc[-2])
                            if len(rsi_series) > 1 and not pd.isna(rsi_series.iloc[-2])
                            else None,
            'atr':          float(atr_series.iloc[-1])
                            if not pd.isna(atr_series.iloc[-1])
                            else None,
            'adx':          float(adx_series.iloc[-1])
                            if not pd.isna(adx_series.iloc[-1])
                            else None,
            'donchian_upper_entry': float(d_entry['upper'].iloc[-1]) if not pd.isna(d_entry['upper'].iloc[-1]) else None,
            'donchian_lower_entry': float(d_entry['lower'].iloc[-1]) if not pd.isna(d_entry['lower'].iloc[-1]) else None,
            'donchian_upper_exit':  float(d_exit['upper'].iloc[-1])  if not pd.isna(d_exit['upper'].iloc[-1])  else None,
            'donchian_lower_exit':  float(d_exit['lower'].iloc[-1])  if not pd.isna(d_exit['lower'].iloc[-1])  else None,
            'atr_ratio':    float(atr_ratio.iloc[-1])
                            if not pd.isna(atr_ratio.iloc[-1])
                            else 1.0,
            'volume_sma':   float(vol_sma.iloc[-1])
                            if not pd.isna(vol_sma.iloc[-1])
                            else None,
            'session':      session,
            'regime':       regime,
            'candle_time':  df.index[-1],
        }

    def _generate_signal(self, cache: Dict) -> Dict:
        spec = self.spec

        # Template delegation for templates
        if spec.template == 'rsi_reversion':
            import trainer.signals.rsi_reversion as rsi_reversion
            res = rsi_reversion.generate_signal(cache, spec.parameters)
            res['session'] = cache.get('session')
            res['regime'] = cache.get('regime')
            res['candle_time'] = str(cache.get('candle_time', ''))
            return res
        elif spec.template == 'donchian_breakout':
            import trainer.signals.donchian_breakout as donchian_breakout
            res = donchian_breakout.generate_signal(cache, spec.parameters)
            res['session'] = cache.get('session')
            res['regime'] = cache.get('regime')
            res['candle_time'] = str(cache.get('candle_time', ''))
            return res
        elif spec.template == 'hybrid_confluence':
            import trainer.signals.hybrid_confluence as hybrid_confluence
            res = hybrid_confluence.generate_signal(cache, spec.parameters)
            res['session'] = cache.get('session')
            res['regime'] = cache.get('regime')
            res['candle_time'] = str(cache.get('candle_time', ''))
            return res

        # Validate required values
        required = ['sma_fast', 'sma_slow', 'atr', 'close']
        for key in required:
            if cache.get(key) is None:
                return self._no_signal(
                    f'NaN indicator: {key}'
                )

        # Dead zone — never trade
        if cache['session'] == 'DEAD_ZONE':
            return self._no_signal('Dead zone — no trading')

        # ADX filter
        adx_threshold = spec.parameters.get(
            'adx_threshold', spec.parameters.get('adx_min_threshold', 0)
        )
        if adx_threshold > 0:
            adx = cache.get('adx')
            if adx is not None and adx < adx_threshold:
                return self._no_signal(
                    f"ADX {adx:.1f} below "
                    f"threshold {adx_threshold}"
                )

        # Session filter
        allowed = spec.filters.get('allowed_sessions')
        if allowed and cache['session'] not in allowed:
            return self._no_signal(
                f"Session {cache['session']} "
                f"not in allowed list"
            )

        sma_fast      = cache['sma_fast']
        sma_fast_prev = cache.get('sma_fast_prev')
        sma_slow      = cache['sma_slow']
        sma_slow_prev = cache.get('sma_slow_prev')
        close         = cache['close']
        atr           = cache['atr']

        # Need previous values for crossover
        if sma_fast_prev is None or sma_slow_prev is None:
            return self._no_signal('No previous candle data')

        sl_distance = atr * spec.sl_atr_multiple
        tp_distance = sl_distance * spec.tp_rr_ratio

        # BUY signal
        if crossed_above(sma_fast, sma_fast_prev,
                         sma_slow, sma_slow_prev):
            return {
                'signal':    'BUY',
                'reason':    (
                    f"MA_CROSS_BULL: fast={sma_fast:.5f} "
                    f"crossed above slow={sma_slow:.5f}"
                ),
                'sl_price':  close - sl_distance,
                'tp_price':  close + tp_distance,
                'sl_pips':   sl_distance / spec.pip_size,
                'tp_pips':   tp_distance / spec.pip_size,
                'atr':       atr,
                'adx':       cache.get('adx'),
                'session':   cache['session'],
                'regime':    cache['regime'],
                'candle_time': str(cache['candle_time']),
            }

        # SELL signal
        if crossed_below(sma_fast, sma_fast_prev,
                         sma_slow, sma_slow_prev):
            return {
                'signal':    'SELL',
                'reason':    (
                    f"MA_CROSS_BEAR: fast={sma_fast:.5f} "
                    f"crossed below slow={sma_slow:.5f}"
                ),
                'sl_price':  close + sl_distance,
                'tp_price':  close - tp_distance,
                'sl_pips':   sl_distance / spec.pip_size,
                'tp_pips':   tp_distance / spec.pip_size,
                'atr':       atr,
                'adx':       cache.get('adx'),
                'session':   cache['session'],
                'regime':    cache['regime'],
                'candle_time': str(cache['candle_time']),
            }

        return self._no_signal('No crossover this candle')

    def _no_signal(self, reason: str) -> Dict:
        return {
            'signal':   'HOLD',
            'reason':   reason,
            'sl_price': None,
            'tp_price': None,
            'sl_pips':  None,
            'tp_pips':  None,
            'atr':      self.indicator_cache.get('atr'),
            'session':  self.indicator_cache.get('session'),
            'regime':   self.indicator_cache.get('regime'),
            'candle_time': str(
                self.indicator_cache.get('candle_time', '')
            ),
        }

    def get_cache(self) -> Dict:
        """Return current indicator cache."""
        return self.indicator_cache.copy()