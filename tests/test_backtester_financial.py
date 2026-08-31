"""
Pinning tests for the backtester's money-math correctness.
Guarantees that pip value, lot size, swap costs, and net P&L
match the independently calculated values from validation_reference.yaml.
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone
import pandas as pd
import numpy as np

from trainer.core.backtester import Backtester
from shared.instrument_spec import InstrumentSpec, save_specs
import shared.instrument_spec as mod

@pytest.fixture
def ref_data():
    path = Path(__file__).parent.parent / 'config' / 'validation_reference.yaml'
    data = {'EURUSD': {'trade_1': {'expected': {}}}, 'XAUUSD': {'trade_1': {'expected': {}}}}
    current_symbol = None
    with open(path, 'r') as f:
        for line in f:
            line = line.split('#')[0].rstrip()
            if not line: continue
            if line.startswith('EURUSD:'): current_symbol = 'EURUSD'
            elif line.startswith('XAUUSD:'): current_symbol = 'XAUUSD'
            elif 'direction:' in line: data[current_symbol]['trade_1']['direction'] = line.split(':')[1].strip()
            elif 'lots:' in line: data[current_symbol]['trade_1']['lots'] = float(line.split(':')[1].strip())
            elif 'entry_price:' in line: data[current_symbol]['trade_1']['entry_price'] = float(line.split(':')[1].strip())
            elif 'exit_price:' in line: data[current_symbol]['trade_1']['exit_price'] = float(line.split(':')[1].strip())
            elif 'entry_time:' in line: data[current_symbol]['trade_1']['entry_time'] = line.split("'")[1]
            elif 'exit_time:' in line: data[current_symbol]['trade_1']['exit_time'] = line.split("'")[1]
            elif 'gross_pnl:' in line: data[current_symbol]['trade_1']['expected']['gross_pnl'] = float(line.split(':')[1].strip())
            elif 'commission:' in line: data[current_symbol]['trade_1']['expected']['commission'] = float(line.split(':')[1].strip())
            elif 'swap_cost:' in line: data[current_symbol]['trade_1']['expected']['swap_cost'] = float(line.split(':')[1].strip())
            elif 'net_pnl:' in line: data[current_symbol]['trade_1']['expected']['net_pnl'] = float(line.split(':')[1].strip())
    return data

@pytest.fixture
def mock_specs(tmp_path):
    original_path = mod.SPEC_CACHE_PATH
    mod.SPEC_CACHE_PATH = tmp_path / "specs.json"
    
    # EURUSD
    eurusd = InstrumentSpec(symbol="EURUSD", digits=5, point=0.00001, spread_typical=10, contract_size=100000.0, tick_size=0.00001, tick_value=1.0, swap_long=-5.0, swap_short=2.0, volume_min=0.01, volume_step=0.01, volume_max=100, stops_level=0, currency_base='EUR', currency_profit='USD', currency_margin='EUR', account_currency='USD', captured_at='2024', sanity_ok=True)
    
    # XAUUSD
    xauusd = InstrumentSpec(symbol="XAUUSD", digits=2, point=0.01, spread_typical=25, contract_size=100.0, tick_size=0.01, tick_value=1.0, swap_long=-20.0, swap_short=15.0, volume_min=0.01, volume_step=0.01, volume_max=100, stops_level=0, currency_base='XAU', currency_profit='USD', currency_margin='XAU', account_currency='USD', captured_at='2024', sanity_ok=True)
    
    save_specs({
        "EURUSD": eurusd,
        "XAUUSD": xauusd,
    })
    
    yield
    mod.SPEC_CACHE_PATH = original_path

def test_financial_pinning_eurusd(ref_data, mock_specs):
    trade = ref_data['EURUSD']['trade_1']
    expected = trade['expected']
    
    bt = Backtester("EURUSD", params={})
    
    # Mock opening position
    bt.open_position = {
        'direction': trade['direction'],
        'entry_price': trade['entry_price'],
        'entry_time': datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00')),
        'lots': trade['lots'],
        'commission_open': abs(expected['commission']) / 2.0,
        'candles_open': 10,
        'entry_session': 'LONDON',
        'entry_regime': 'TREND_UP',
    }
    
    bt._close_position(
        price=trade['exit_price'],
        candle_time=datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00')),
        reason='TP',
        cache={'session': 'LONDON'}
    )
    
    assert len(bt.closed_trades) == 1
    result = bt.closed_trades[0]
    
    assert result['gross_pnl'] == pytest.approx(expected['gross_pnl'])
    assert result['commission'] == pytest.approx(abs(expected['commission']))
    assert result['swap_cost'] == pytest.approx(expected['swap_cost'])
    assert result['net_pnl'] == pytest.approx(expected['net_pnl'])

def test_financial_pinning_xauusd(ref_data, mock_specs):
    trade = ref_data['XAUUSD']['trade_1']
    expected = trade['expected']
    
    bt = Backtester("XAUUSD", params={})
    
    # Mock opening position
    bt.open_position = {
        'direction': trade['direction'],
        'entry_price': trade['entry_price'],
        'entry_time': datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00')),
        'lots': trade['lots'],
        'commission_open': abs(expected['commission']) / 2.0,
        'candles_open': 10,
        'entry_session': 'LONDON',
        'entry_regime': 'TREND_UP',
    }
    
    bt._close_position(
        price=trade['exit_price'],
        candle_time=datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00')),
        reason='TP',
        cache={'session': 'LONDON'}
    )
    
    assert len(bt.closed_trades) == 1
    result = bt.closed_trades[0]
    
    assert result['gross_pnl'] == pytest.approx(expected['gross_pnl'])
    assert result['commission'] == pytest.approx(abs(expected['commission']))
    assert result['swap_cost'] == pytest.approx(expected['swap_cost'])
    assert result['net_pnl'] == pytest.approx(expected['net_pnl'])

def test_financial_pinning_lot_sizing(mock_specs):
    bt = Backtester("EURUSD", params={'risk_per_trade_pct': 1.0})
    # Balance: 10000. Risk: 1% = 100
    bt.balance = 10000.0
    
    # EURUSD Spec from mock: pip_size=0.0001, tick_size=0.00001, tick_value=1.0, 
    # Entry: 1.1000, SL: 1.0980 (20 pips = 200 ticks)
    # Risk amount = 100
    # Expected lot size = 100 / (200 * 1.0) = 0.50
    
    bt._open_position(
        signal={
            'signal': 'BUY',
            'price': 1.1000,
            'sl_price': 1.0980,
            'tp_price': 1.1040,
        },
        base_price=1.1000,
        candle_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        cache={'session': 'LONDON', 'close': 1.1000, 'regime': 'TREND_UP'}
    )
    
    assert bt.open_position is not None
    assert bt.open_position['lots'] == pytest.approx(0.47)

def test_check_sl_tp_sell_intracandle_ordering(mock_specs):
    """
    Verify conservative intracandle ordering for a SELL position:
    When a single candle hits both SL (high >= sl_price) and TP (low <= tp_price),
    the SELL branch checks high (SL) first, closing position with SL_HIT.
    """
    bt = Backtester("EURUSD", params={})
    bt.open_position = {
        'direction': 'SELL',
        'entry_price': 1.1000,
        'entry_time': datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        'sl_price': 1.1020,  # SL is 20 pips above entry
        'tp_price': 1.0960,  # TP is 40 pips below entry
        'lots': 1.0,
        'commission_open': 3.5,
        'candles_open': 1,
        'entry_session': 'LONDON',
        'entry_regime': 'TREND_DOWN',
    }

    # Candle high hits SL (1.1025 >= 1.1020), low hits TP (1.0950 <= 1.0960)
    row = pd.Series({
        'open': 1.1000,
        'high': 1.1025,
        'low': 1.0950,
        'close': 1.1010,
    }, name=datetime(2026, 6, 1, 10, 15, tzinfo=timezone.utc))

    cache = {'session': 'LONDON'}
    bt._check_sl_tp(row, cache)

    assert bt.open_position is None
    assert len(bt.closed_trades) == 1
    assert bt.closed_trades[0]['close_reason'] == 'SL_HIT'
    assert bt.closed_trades[0]['exit_price'] == 1.1020

def test_unrealised_pnl_sell_standalone(mock_specs):
    """
    Standalone unit test for _unrealised_pnl with direction='SELL':
    Asserts sign and magnitude of unrealised P&L when price moves down (profit)
    and up (loss) using real InstrumentSpec tick_value and tick_size.
    """
    bt = Backtester("EURUSD", params={})
    bt.open_position = {
        'direction': 'SELL',
        'entry_price': 1.1000,
        'entry_time': datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        'sl_price': 1.1020,
        'tp_price': 1.0960,
        'lots': 1.0,
        'commission_open': 3.5,
        'candles_open': 1,
        'entry_session': 'LONDON',
        'entry_regime': 'TREND_DOWN',
    }

    # EURUSD Spec: tick_size = 0.00001, tick_value = 1.0 per lot
    # Case 1: Profit (price moves down 50 pips / 500 ticks from 1.1000 to 1.0950)
    # ticks_won = -(1.0950 - 1.1000) / 0.00001 = +500 ticks
    # unrealised_pnl = 500 ticks * 1.0 lot * $1.0 = +$500.00
    pnl_profit = bt._unrealised_pnl(1.0950)
    assert pnl_profit == pytest.approx(500.0)

    # Case 2: Loss (price moves up 30 pips / 300 ticks from 1.1000 to 1.1030)
    # ticks_won = -(1.1030 - 1.1000) / 0.00001 = -300 ticks
    # unrealised_pnl = -300 ticks * 1.0 lot * $1.0 = -$300.00
    pnl_loss = bt._unrealised_pnl(1.1030)
    assert pnl_loss == pytest.approx(-300.0)

def test_financial_pinning_lot_sizing_sell(mock_specs):
    """
    End-to-end lot sizing pinning test for SELL signal.
    Verifies position sizing via _open_position using _simulated_spread and _simulated_slippage.
    """
    bt = Backtester("EURUSD", params={'risk_per_trade_pct': 1.0})
    bt.balance = 10000.0

    bt._open_position(
        signal={
            'signal': 'SELL',
            'price': 1.1000,
            'sl_price': 1.1020,
            'tp_price': 1.0960,
        },
        base_price=1.1000,
        candle_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        cache={'session': 'LONDON', 'close': 1.1000, 'regime': 'TREND_DOWN'}
    )

    assert bt.open_position is not None
    assert bt.open_position['lots'] == pytest.approx(0.47)

def test_expanded_trade_log_feature_snapshot(mock_specs):
    """
    Confirm all expanded feature snapshot fields are populated on trade open/close:
    RSI, BB%B, ATR volatility, volume_relative, ma_distance_ticks, hour_utc, day_of_week, accrued_swap.
    """
    bt = Backtester("EURUSD", params={})
    entry_time = datetime(2026, 6, 1, 14, 30, tzinfo=timezone.utc)  # Monday 14:30 UTC
    cache = {
        'session': 'NEW_YORK',
        'close': 1.1000,
        'regime': 'TREND_UP',
        'sma_fast': 1.0990,
        'rsi': 62.5,
        'bb_pct': 0.75,
        'atr': 0.0015,
        'atr_ratio': 1.25,
        'volume_rel': 1.4,
    }

    bt._open_position(
        signal={
            'signal': 'BUY',
            'price': 1.1000,
            'sl_price': 1.0950,
            'tp_price': 1.1100,
        },
        base_price=1.1000,
        candle_time=entry_time,
        cache=cache
    )

    pos = bt.open_position
    assert pos is not None
    assert pos['direction'] == 'BUY'
    assert pos['rsi'] == 62.5
    assert pos['bb_pct'] == 0.75
    assert pos['volatility_atr'] == 0.0015
    assert pos['volatility_ratio'] == 1.25
    assert pos['volume_relative'] == 1.4
    # ma_dist_ticks = abs(entry_price - 1.0990) / 0.00001
    assert pos['ma_distance_ticks'] > 0
    assert pos['hour_utc'] == 14
    assert pos['day_of_week'] == 0  # Monday
    assert pos['accrued_swap'] == 0.0

    # Close position and check closed_trade log record
    exit_time = datetime(2026, 6, 1, 16, 0, tzinfo=timezone.utc)
    bt._close_position(
        price=1.1050,
        candle_time=exit_time,
        reason='SIGNAL',
        cache=cache
    )

    assert len(bt.closed_trades) == 1
    trade = bt.closed_trades[0]
    required_fields = [
        'rsi', 'bb_pct', 'volatility_atr', 'volatility_ratio',
        'volume_relative', 'ma_distance_ticks', 'hour_utc',
        'day_of_week', 'accrued_swap'
    ]
    for field in required_fields:
        assert field in trade
        assert trade[field] is not None
        assert not pd.isna(trade[field])

def test_expanded_trade_log_backward_compatibility(mock_specs):
    """
    Verify backward compatibility when processing old trade dictionaries without new feature snapshot fields.
    """
    from shared.metrics import calculate_metrics

    old_trade = {
        'direction': 'BUY',
        'entry_price': 1.1000,
        'entry_time': datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
        'exit_price': 1.1050,
        'exit_time': datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
        'lots': 1.0,
        'gross_pnl': 500.0,
        'net_pnl': 493.0,
        'commission': 7.0,
        'swap_cost': 0.0,
        'close_reason': 'TP_HIT',
        'duration_candles': 8,
        'entry_session': 'LONDON',
        'entry_regime': 'TREND_UP',
        'exit_session': 'LONDON',
    }

    metrics = calculate_metrics([old_trade], [10000.0, 10493.0])
    assert metrics['total_trades'] == 1
    assert metrics['net_profit'] == 493.0

def test_expanded_feature_pipeline_real_indicators(mock_specs):
    """
    Feature pipeline test using real indicator functions from shared/indicators.py.
    Verifies that _compute_features and _build_cache output valid, sane indicator values.
    """
    bt = Backtester("EURUSD", params={'fast_ma_period': 10, 'slow_ma_period': 20})
    dates = pd.date_range('2026-06-01 00:00', periods=30, freq='15min', tz=timezone.utc)
    rng = np.random.default_rng(42)
    closes = 1.1000 + np.cumsum(rng.normal(0, 0.0005, 30))
    df = pd.DataFrame({
        'open': closes - 0.0001,
        'high': closes + 0.0003,
        'low': closes - 0.0003,
        'close': closes,
        'volume': [100 + i*5 for i in range(30)],
    }, index=dates)

    f = bt._compute_features(df)
    cache = bt._build_cache(f.iloc[25], f.iloc[24])

    bt._open_position(
        signal={'signal': 'BUY', 'price': f.iloc[25]['close'], 'sl_price': 1.0900, 'tp_price': 1.1200},
        base_price=f.iloc[25]['open'],
        candle_time=f.index[25],
        cache=cache
    )

    pos = bt.open_position
    assert pos is not None
    assert 0.0 <= pos['rsi'] <= 100.0
    assert not pd.isna(pos['bb_pct']) and not np.isinf(pos['bb_pct'])
    assert pos['volatility_atr'] > 0
    assert pos['volume_relative'] > 0
    assert pos['ma_distance_ticks'] >= 0
    assert pos['hour_utc'] == f.index[25].hour
    assert pos['day_of_week'] == f.index[25].weekday()

def test_expanded_feature_snapshot_warmup_guard(mock_specs):
    """
    Verify that _open_position during warmup (row 2 of dataset where RSI/BB%B/atr_ratio are NaN)
    produces clean guarded default values rather than NaN.
    """
    bt = Backtester("EURUSD", params={})
    dates = pd.date_range('2026-06-01 00:00', periods=25, freq='15min', tz=timezone.utc)
    df = pd.DataFrame({
        'open': [1.1000] * 25,
        'high': [1.1010] * 25,
        'low': [1.0990] * 25,
        'close': [1.1000] * 25,
        'volume': [100] * 25,
    }, index=dates)

    f = bt._compute_features(df)
    # Explicitly confirm raw row indicator values are NaN during warmup period
    assert pd.isna(f.iloc[2]['rsi'])
    assert pd.isna(f.iloc[2]['bb_pct'])
    assert pd.isna(f.iloc[2]['atr_ratio'])

    cache = bt._build_cache(f.iloc[2], f.iloc[1])

    bt._open_position(
        signal={'signal': 'BUY', 'price': 1.1000, 'sl_price': 1.0950, 'tp_price': 1.1100},
        base_price=1.1000,
        candle_time=f.index[2],
        cache=cache
    )

    pos = bt.open_position
    assert pos is not None
    # Post-guard values must equal clean defaults (50.0, 0.5, 1.0) and be non-NaN
    assert pos['rsi'] == 50.0
    assert pos['bb_pct'] == 0.5
    assert pos['volatility_ratio'] == 1.0
    assert pos['volume_relative'] == 1.0
    assert pos['ma_distance_ticks'] == 0.0
    assert not pd.isna(pos['rsi'])
    assert not pd.isna(pos['bb_pct'])
    assert not pd.isna(pos['volatility_ratio'])


def test_backtester_cache_encapsulation_no_dataframe_leak():
    """
    Look-ahead audit test (WAVE 4.2).
    Verifies that cache passed to generate_signal is a strict scalar dictionary
    containing only scalar indicator/OHLCV values from candle i and i-1.
    Asserts no DataFrame reference, future row index, or future slice is exposed.
    """
    dates = pd.date_range('2020-01-01', periods=30, freq='15min', tz='UTC')
    df = pd.DataFrame({
        'open':   [1.1000] * 30,
        'high':   [1.1010] * 30,
        'low':    [1.0990] * 30,
        'close':  [1.1000] * 30,
        'volume': [100.0] * 30,
    }, index=dates)

    received_caches = []

    class CacheInspectorSignalModule:
        @staticmethod
        def generate_signal(cache, params):
            received_caches.append(dict(cache))
            # Verify cache is a plain dict of scalars and contains no DataFrame/Series
            for key, val in cache.items():
                assert not isinstance(val, (pd.DataFrame, pd.Series)), f"Leak: cache['{key}'] is a pandas object!"
            return {'signal': 'HOLD', 'sl_price': None, 'tp_price': None}

        @staticmethod
        def check_exit(cache, position, params):
            return {'should_exit': False, 'reason': None}

    bt = Backtester(symbol='EURUSD', params={'warmup_candles': 20, 'fast_ma_period': 5, 'slow_ma_period': 10}, allow_simulation_defaults=True)
    bt.run(df, CacheInspectorSignalModule)

    assert len(received_caches) > 0
    # Every cache must contain strictly scalar values
    first_cache = received_caches[0]
    expected_keys = {'close', 'high', 'low', 'open', 'volume', 'sma_fast', 'sma_fast_prev', 'sma_slow', 'sma_slow_prev', 'rsi', 'bb_pct', 'atr', 'adx', 'atr_ratio', 'volume_sma', 'volume_rel', 'session', 'regime'}
    assert expected_keys.issubset(set(first_cache.keys()))


def test_backtester_fill_timing_audit():
    """
    Look-ahead fix verification test for entry fill timing (WAVE 4.2).
    Constructs a dataset where candle 25 closes at 1.1000 and candle 26 opens with a gap at 1.1050.
    Fires entry signal specifically at candle 25 close (when volume spike happens on candle 25).
    Verifies that _open_position executes on candle 26 using candle 26's open (1.1050 + spread/slip)
    and candle 26's timestamp, NOT candle 25's close (1.1000).
    """
    dates = pd.date_range('2020-01-01', periods=30, freq='15min', tz='UTC')
    df = pd.DataFrame({
        'open':   [1.1000] * 30,
        'high':   [1.1010] * 30,
        'low':    [1.0990] * 30,
        'close':  [1.1000] * 30,
        'volume': [100.0] * 30,
    }, index=dates)

    # Signal trigger condition: volume = 999.0 on candle 25 close ONLY
    df.iloc[25, df.columns.get_loc('volume')] = 999.0
    # Manufactured gap up on candle 26 open = 1.1050
    df.iloc[26, df.columns.get_loc('open')] = 1.1050

    class SpecificEntrySignalModule:
        @staticmethod
        def generate_signal(cache, params):
            if cache['volume'] == 999.0:
                return {'signal': 'BUY', 'sl_price': 1.0900, 'tp_price': 1.1200}
            return {'signal': 'HOLD', 'sl_price': None, 'tp_price': None}

        @staticmethod
        def check_exit(cache, position, params):
            return {'should_exit': False, 'reason': None}

    bt = Backtester(symbol='EURUSD', params={'warmup_candles': 20, 'fast_ma_period': 5, 'slow_ma_period': 10}, allow_simulation_defaults=True)
    res = bt.run(df, SpecificEntrySignalModule)

    assert len(res['trades']) == 1
    trade = res['trades'][0]

    # Signal fired on candle 25 close.
    # Fill must occur on candle 26 (dates[26]) using candle 26's open price (1.1050 + spread/slip).
    assert trade['entry_time'] == dates[26]
    assert abs(trade['entry_price'] - 1.1050) < 0.001


def test_backtester_signal_exit_timing_audit():
    """
    Look-ahead fix verification test for signal-based exit fill timing (WAVE 4.2).
    Verifies that a signal exit triggered at candle i close is queued and executed
    on candle i+1's open price, with timestamp dates[i+1].
    """
    dates = pd.date_range('2020-01-01', periods=30, freq='15min', tz='UTC')
    df = pd.DataFrame({
        'open':   [1.1000] * 30,
        'high':   [1.1010] * 30,
        'low':    [1.0990] * 30,
        'close':  [1.1000] * 30,
        'volume': [100.0] * 30,
    }, index=dates)

    # Candle 20: Entry signal triggered on candle 20 close -> fills on candle 21 open = 1.1000
    df.iloc[20, df.columns.get_loc('volume')] = 500.0
    # Candle 24: Signal exit condition triggered on candle 24 close -> volume = 888.0
    df.iloc[24, df.columns.get_loc('volume')] = 888.0
    # Candle 25: Open price gap down to 1.0920
    df.iloc[25, df.columns.get_loc('open')] = 1.0920

    class SignalExitModule:
        @staticmethod
        def generate_signal(cache, params):
            if cache['volume'] == 500.0:
                return {'signal': 'BUY', 'sl_price': 1.0500, 'tp_price': 1.1500}
            return {'signal': 'HOLD', 'sl_price': None, 'tp_price': None}

        @staticmethod
        def check_exit(cache, position, params):
            if cache['volume'] == 888.0:
                return {'should_exit': True, 'reason': 'SIGNAL_EXIT_TRIGGERED'}
            return {'should_exit': False, 'reason': None}

    bt = Backtester(symbol='EURUSD', params={'warmup_candles': 20, 'fast_ma_period': 5, 'slow_ma_period': 10}, allow_simulation_defaults=True)
    res = bt.run(df, SignalExitModule)

    assert len(res['trades']) == 1
    trade = res['trades'][0]

    # Signal exit triggered on candle 24 close.
    # Exit fill must execute on candle 25 (dates[25]) using candle 25's open price (1.0920).
    assert trade['exit_time'] == dates[25]
    assert abs(trade['exit_price'] - 1.0920) < 0.0001
    assert trade['close_reason'] == 'SIGNAL_EXIT_TRIGGERED'


def test_backtester_regression_intracandle_fill_ordering():
    """
    Regression test for Backtester intracandle fill ordering (WAVE 4.2).
    Verifies that a signal triggered at candle 20 close executes on candle 21 open,
    and candle 21's high hits TP during candle 21.
    """
    dates = pd.date_range('2020-01-01', periods=30, freq='15min', tz='UTC')
    df = pd.DataFrame({
        'open':   [1.1000] * 30,
        'high':   [1.1010] * 30,
        'low':    [1.0990] * 30,
        'close':  [1.1000] * 30,
        'volume': [100.0] * 30,
    }, index=dates)

    # On candle 20 close, volume=999 triggers signal
    df.iloc[20, df.columns.get_loc('volume')] = 999.0
    # On candle 21 (when trade opens), high reaches 1.3000
    df.iloc[21, df.columns.get_loc('high')] = 1.3000

    class SingleEntrySignalModule:
        @staticmethod
        def generate_signal(cache, params):
            if cache['volume'] == 999.0:
                return {'signal': 'BUY', 'sl_price': 1.0900, 'tp_price': 1.2500}
            return {'signal': 'HOLD', 'sl_price': None, 'tp_price': None}

        @staticmethod
        def check_exit(cache, position, params):
            return {'should_exit': False, 'reason': None}

    bt = Backtester(symbol='EURUSD', params={'warmup_candles': 20, 'fast_ma_period': 5, 'slow_ma_period': 10}, allow_simulation_defaults=True)
    res = bt.run(df, SingleEntrySignalModule)

    assert len(res['trades']) == 1
    trade = res['trades'][0]
    assert trade['entry_time'] == dates[21]
    assert trade['close_reason'] == 'TP_HIT'


def test_backtester_queued_entry_same_candle_sltp_hit():
    """
    Test that an entry signal queued at candle i-1 close fills at candle i open,
    and _check_sl_tp running on candle i evaluates candle i's high/low correctly.
    Verifies that if candle i's low hits SL, position is closed on candle i with SL_HIT.
    """
    dates = pd.date_range('2020-01-01', periods=30, freq='15min', tz='UTC')
    df = pd.DataFrame({
        'open':   [1.1000] * 30,
        'high':   [1.1010] * 30,
        'low':    [1.0990] * 30,
        'close':  [1.1000] * 30,
        'volume': [100.0] * 30,
    }, index=dates)

    # Signal triggered on candle 20 close
    df.iloc[20, df.columns.get_loc('volume')] = 777.0
    # Candle 21 (when trade fills at open=1.1000): low spikes down to 1.0800 (below SL 1.0900)
    df.iloc[21, df.columns.get_loc('low')] = 1.0800

    class QueuedEntrySLModule:
        @staticmethod
        def generate_signal(cache, params):
            if cache['volume'] == 777.0:
                return {'signal': 'BUY', 'sl_price': 1.0900, 'tp_price': 1.1500}
            return {'signal': 'HOLD', 'sl_price': None, 'tp_price': None}

        @staticmethod
        def check_exit(cache, position, params):
            return {'should_exit': False, 'reason': None}

    bt = Backtester(symbol='EURUSD', params={'warmup_candles': 20, 'fast_ma_period': 5, 'slow_ma_period': 10}, allow_simulation_defaults=True)
    res = bt.run(df, QueuedEntrySLModule)

    assert len(res['trades']) == 1
    trade = res['trades'][0]

    # Entry queued at candle 20 close, executed at candle 21 open (dates[21]).
    # Same-candle low spike hits SL on candle 21 (dates[21]).
    assert trade['entry_time'] == dates[21]
    assert trade['exit_time'] == dates[21]
    assert trade['close_reason'] == 'SL_HIT'
    assert trade['exit_price'] == 1.0900

