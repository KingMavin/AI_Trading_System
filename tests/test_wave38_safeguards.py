import pytest
import pandas as pd
from datetime import datetime, timezone
from trainer.signals.rsi_reversion import generate_signal
from trainer.core.backtester import Backtester

def test_rsi_same_direction_cooldown():
    """
    Test that after a losing SELL trade, another SELL signal is blocked
    until RSI crosses the midline (50).
    Uses the exact entry condition: rsi_prev > 70 AND rsi <= 70.
    """
    params = {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'sl_atr_multiple': 1.5,
        'tp_rr_ratio': 2.0,
        'pip_size': 0.0001,
        'cooldown_requires_midline': True
    }

    # Simulate losing SELL trade just closed
    strategy_state = {}
    last_trade = {
        'direction': 'SELL',
        'net_pnl': -15.0,  # A loss
        'reason': 'SL_HIT'
    }

    # Sequence 1: RSI dips back below 70 from overbought (SELL Signal)
    # But cooldown is active because midline hasn't been crossed
    cache1 = {
        'rsi_prev': 72.0,
        'rsi': 68.0,
        'atr': 0.0010,
        'close': 1.1000,
        'adx': 25.0,
        'strategy_state': strategy_state,
        'last_closed_trade': last_trade
    }
    sig1 = generate_signal(cache1, params)
    assert sig1['signal'] == 'HOLD'
    assert 'COOLDOWN_ACTIVE' in sig1['reason']

    # Sequence 2: RSI crosses the midline (50) down
    cache2 = {
        'rsi_prev': 52.0,
        'rsi': 48.0,
        'atr': 0.0010,
        'close': 1.0980,
        'adx': 25.0,
        'strategy_state': strategy_state,
        'last_closed_trade': last_trade
    }
    sig2 = generate_signal(cache2, params)
    assert sig2['signal'] == 'HOLD'
    assert strategy_state.get('midline_crossed') is True

    # Sequence 3: RSI goes back to overbought and dips below 70 again (SELL Signal)
    # Cooldown should be lifted now
    cache3 = {
        'rsi_prev': 75.0,
        'rsi': 69.0,
        'atr': 0.0010,
        'close': 1.1020,
        'adx': 25.0,
        'strategy_state': strategy_state,
        'last_closed_trade': last_trade
    }
    sig3 = generate_signal(cache3, params)
    assert sig3['signal'] == 'SELL'
    assert 'RSI_OVERBOUGHT_REVERSION' in sig3['reason']


class MockSignalModule:
    """A mock signal module that always issues BUY signals and immediately exits for a loss."""
    def __init__(self):
        self.entry_count = 0
    
    def generate_signal(self, cache, params):
        self.entry_count += 1
        return {
            'signal': 'BUY',
            'reason': 'TEST_BUY',
            'sl_price': cache['close'] - 0.0010,
            'tp_price': cache['close'] + 0.0020,
            'sl_pips': 10,
            'rr_ratio': 2.0
        }

    def check_exit(self, cache, position, params):
        # Force a loss exit on every candle
        return {'should_exit': True, 'reason': 'MOCK_LOSS'}

def test_daily_loss_circuit_breaker():
    """
    Test that the backtester enforces the daily_loss_halt_pct logic,
    halting new entries on the same day once tripped, and resetting the next day.
    """
    dates = [
        datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc)
    ]
    for i in range(1, 26):
        dates.append(dates[-1] + pd.Timedelta(hours=1))

    # The last two rows are what we actually care about for the day rollover.
    # Day 1: 24 candles
    # Day 2: 2 candles
    dates[-2] = datetime(2020, 1, 2, 10, 0, tzinfo=timezone.utc)
    dates[-1] = datetime(2020, 1, 2, 11, 0, tzinfo=timezone.utc)
    
    df = pd.DataFrame({
        'open': [1.0990] * 26,
        'high': [1.1010] * 26,
        'low':  [1.0980] * 26,
        'close':[1.0995] * 26,
        'tick_volume': [100] * 26,
        'volume': [1000] * 26
    }, index=dates)

    backtester = Backtester(symbol='EURUSD', params={}, initial_equity=10000.0, allow_simulation_defaults=True)
    
    # Configure backtester
    backtester.params = {
        'warmup_candles': 1,
        'fast_ma_period': 2,
        'slow_ma_period': 3,
        'ma_type': 'SMA',
        'daily_loss_halt_pct': 1.0, # 1% halt limit
        'risk_per_trade_pct': 2.0,  # 2% risk, so 1 trade loss will trip the 1% limit
    }
    
    # We will override position sizing so we take significant losses
    def mock_open_position(signal, candle_time, cache, base_price=None):
        backtester.open_position = {
            'direction': signal['signal'],
            'entry_price': cache['close'],
            'entry_time': candle_time,
            'sl_price': signal['sl_price'],
            'tp_price': signal['tp_price'],
            'lots': 50.0, # Large size to guarantee >1% loss
            'commission_open': 0.0,
            'sl_distance': 0.0010,
            'entry_session': 'TEST',
            'entry_regime': 'TEST',
            'candles_open': 0,
            'highest_price': cache['close'],
            'lowest_price': cache['close']
        }
    
    backtester._open_position = mock_open_position

    module = MockSignalModule()
    res = backtester.run(df, module)

    trades = res['trades']

    # We expect exactly 2 trades
    assert len(trades) == 2
    
    trade1 = trades[0]
    assert trade1['entry_time'].date() == datetime(2020, 1, 1).date()
    assert trade1['net_pnl'] < -100.0 # Verify we hit the 1% loss
    
    trade2 = trades[1]
    assert trade2['entry_time'].date() == datetime(2020, 1, 2).date()
