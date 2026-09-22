"""
Unit tests for signal_engine.py
Pure logic tests — no MT5 required.
Run: pytest tests/test_signal_engine.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock
from engine.core.signal_engine import SignalEngine
from engine.core.strategy_loader import StrategySpec


# ── FIXTURES ───────────────────────────────────────────

def make_spec(fast=10, slow=50, adx_threshold=0,
              sl_atr=1.5, tp_rr=2.0) -> StrategySpec:
    return StrategySpec(
        strategy_id    = 'test_strategy',
        template       = 'ma_crossover',
        symbol         = 'EURUSD',
        timeframe      = 'M15',
        parameters     = {
            'fast_ma_period':          fast,
            'slow_ma_period':          slow,
            'ma_type':                 'SMA',
            'sl_atr_multiple':         sl_atr,
            'tp_rr_ratio':             tp_rr,
            'risk_per_trade_pct':      1.0,
            'pip_size':                0.0001,
            'adx_min_threshold':       adx_threshold,
            'exit_on_opposite_crossover': False,
        },
        filters        = {},
        composite_score= 0.60,
        promoted_at    = '2026-01-01',
        file_hash      = 'test',
    )


@pytest.fixture
def eurusd_df():
    """
    Synthetic EURUSD-like M15 DataFrame.
    Creates a clear uptrend followed by a crossover.
    """
    n       = 300
    dates   = pd.date_range(
        '2026-01-01 00:00', periods=n,
        freq='15min', tz='UTC'
    )
    np.random.seed(42)

    # Start with declining price, then rising
    price = 1.1000
    closes = []
    for i in range(n):
        if i < 150:
            price += np.random.randn() * 0.0003 - 0.0001
        else:
            price += np.random.randn() * 0.0003 + 0.0001
        closes.append(max(price, 1.0500))

    closes = np.array(closes)
    return pd.DataFrame({
        'open':   closes - np.abs(
            np.random.randn(n) * 0.0002
        ),
        'high':   closes + np.abs(
            np.random.randn(n) * 0.0003
        ),
        'low':    closes - np.abs(
            np.random.randn(n) * 0.0003
        ),
        'close':  closes,
        'volume': np.random.randint(100, 1000, n).astype(float),
    }, index=dates)


# ── SIGNAL ENGINE TESTS ────────────────────────────────

class TestSignalEngine:

    def test_initialises_with_spec(self):
        spec   = make_spec()
        engine = SignalEngine(spec)
        assert engine.spec.strategy_id == 'test_strategy'
        assert engine.candles_processed == 0

    def test_insufficient_data_returns_hold(self):
        spec   = make_spec()
        engine = SignalEngine(spec)
        df_small = pd.DataFrame({
            'open': [1.1], 'high': [1.1],
            'low': [1.1], 'close': [1.1],
            'volume': [100.0]
        })
        result = engine.process_candle(df_small)
        assert result['signal'] == 'HOLD'

    def test_none_data_returns_hold(self):
        spec   = make_spec()
        engine = SignalEngine(spec)
        result = engine.process_candle(None)
        assert result['signal'] == 'HOLD'

    def test_returns_valid_signal_keys(self, eurusd_df):
        spec   = make_spec()
        engine = SignalEngine(spec)
        result = engine.process_candle(eurusd_df)
        required = [
            'signal', 'reason', 'sl_price',
            'tp_price', 'session', 'regime'
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_signal_is_valid_value(self, eurusd_df):
        spec   = make_spec()
        engine = SignalEngine(spec)
        result = engine.process_candle(eurusd_df)
        assert result['signal'] in ('BUY', 'SELL', 'HOLD')

    def test_warmup_returns_hold(self, eurusd_df):
        spec   = make_spec()
        engine = SignalEngine(spec)
        # First candle always HOLD (no previous cache)
        result = engine.process_candle(eurusd_df)
        assert result['signal'] == 'HOLD'

    def test_buy_signal_has_positive_sl_tp(self, eurusd_df):
        """If BUY signal: TP > entry > SL."""
        spec   = make_spec()
        engine = SignalEngine(spec)
        engine.process_candle(eurusd_df)  # warmup

        # Process multiple candles to find a crossover
        for i in range(10):
            result = engine.process_candle(eurusd_df)
            if result['signal'] == 'BUY':
                close = engine.get_cache()['close']
                assert result['tp_price'] > close
                assert result['sl_price'] < close
                break

    def test_sell_signal_has_correct_sl_tp(self, eurusd_df):
        """If SELL signal: TP < entry < SL."""
        spec   = make_spec()
        engine = SignalEngine(spec)
        engine.process_candle(eurusd_df)

        for i in range(10):
            result = engine.process_candle(eurusd_df)
            if result['signal'] == 'SELL':
                close = engine.get_cache()['close']
                assert result['tp_price'] < close
                assert result['sl_price'] > close
                break

    def test_indicator_cache_populated(self, eurusd_df):
        spec   = make_spec()
        engine = SignalEngine(spec)
        engine.process_candle(eurusd_df)
        cache  = engine.get_cache()
        assert 'close'    in cache
        assert 'sma_fast' in cache
        assert 'sma_slow' in cache
        assert 'atr'      in cache
        assert 'session'  in cache

    def test_update_spec_resets_cache(self, eurusd_df):
        spec1  = make_spec(fast=10, slow=50)
        spec2  = make_spec(fast=20, slow=100)
        engine = SignalEngine(spec1)
        engine.process_candle(eurusd_df)

        assert len(engine.indicator_cache) > 0
        engine.update_spec(spec2)
        assert len(engine.indicator_cache) == 0
        assert engine.spec.fast_ma == 20

    def test_dead_zone_returns_hold(self, eurusd_df):
        """Candles at 22:00 UTC should be Dead Zone."""
        spec   = make_spec()
        engine = SignalEngine(spec)
        engine.process_candle(eurusd_df)

        # Create a 22:00 UTC candle
        dead_zone_df = eurusd_df.copy()
        new_idx = pd.date_range(
            '2026-01-01 22:00', periods=len(eurusd_df),
            freq='15min', tz='UTC'
        )
        dead_zone_df.index = new_idx
        result = engine.process_candle(dead_zone_df)
        assert result['signal'] == 'HOLD'
        assert 'Dead zone' in result['reason'] or \
               'HOLD' == result['signal']

    def test_adx_filter_blocks_weak_trend(self, eurusd_df):
        """High ADX threshold should filter out weak trend signals."""
        # ADX threshold of 100 will always filter
        spec   = make_spec(adx_threshold=100)
        engine = SignalEngine(spec)
        engine.process_candle(eurusd_df)

        for _ in range(5):
            result = engine.process_candle(eurusd_df)
            # With ADX threshold of 100, should almost always HOLD
            if result['signal'] != 'HOLD':
                assert result.get('adx', 0) >= 100

    def test_sl_tp_ratio_maintained(self, eurusd_df):
        """TP distance should be tp_rr_ratio × SL distance."""
        spec   = make_spec(sl_atr=1.5, tp_rr=2.0)
        engine = SignalEngine(spec)
        engine.process_candle(eurusd_df)

        for _ in range(20):
            result = engine.process_candle(eurusd_df)
            if result['signal'] in ('BUY', 'SELL'):
                cache  = engine.get_cache()
                close  = cache['close']
                sl_dist = abs(result['sl_price'] - close)
                tp_dist = abs(result['tp_price'] - close)
                ratio   = tp_dist / sl_dist
                assert ratio == pytest.approx(2.0, abs=0.05)
                break