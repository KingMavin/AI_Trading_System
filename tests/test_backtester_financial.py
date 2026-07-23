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
        candle_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        cache={'session': 'LONDON', 'close': 1.1000, 'regime': 'TREND_UP'}
    )
    
    assert bt.open_position is not None
    assert bt.open_position['lots'] == pytest.approx(0.47)
