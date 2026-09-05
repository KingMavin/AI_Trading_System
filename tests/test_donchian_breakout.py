import pytest
import pandas as pd
from trainer.signals.donchian_breakout import generate_signal, check_exit
from shared.indicators import calculate_donchian_channel

def test_calculate_donchian_channel():
    df = pd.DataFrame({
        'high': [10, 12, 15, 14, 13],
        'low':  [8, 9, 11, 10, 9]
    })
    result = calculate_donchian_channel(df, period=2)
    
    # row 0: NaN
    # row 1: NaN (shifted rolling window of 2 starts at index 2)
    # wait, rolling(2).max().shift(1) on [10, 12, 15] -> 
    # max([10,12]) = 12, shifted by 1 means index 2 has 12.
    assert pd.isna(result['upper'].iloc[0])
    assert pd.isna(result['upper'].iloc[1])
    assert result['upper'].iloc[2] == 12.0
    assert result['lower'].iloc[2] == 8.0
    assert result['mid'].iloc[2] == 10.0

def test_donchian_breakout_generate_signal():
    params = {'sl_atr_multiple': 1.0, 'tp_rr_ratio': 2.0, 'adx_threshold': 0}
    cache_buy = {
        'donchian_upper_entry': 1.1000,
        'donchian_lower_entry': 1.0900,
        'atr': 0.0020,
        'close': 1.1010
    }
    res_buy = generate_signal(cache_buy, params)
    assert res_buy['signal'] == 'BUY'
    
    cache_sell = {
        'donchian_upper_entry': 1.1000,
        'donchian_lower_entry': 1.0900,
        'atr': 0.0020,
        'close': 1.0890
    }
    res_sell = generate_signal(cache_sell, params)
    assert res_sell['signal'] == 'SELL'
    
    cache_hold = {
        'donchian_upper_entry': 1.1000,
        'donchian_lower_entry': 1.0900,
        'atr': 0.0020,
        'close': 1.0950
    }
    res_hold = generate_signal(cache_hold, params)
    assert res_hold['signal'] == 'HOLD'

def test_donchian_breakout_check_exit():
    params = {'exit_on_opposite_crossover': True}
    position = {'direction': 'BUY'}
    cache_exit = {
        'donchian_lower_exit': 1.0900,
        'donchian_upper_exit': 1.1000,
        'close': 1.0890
    }
    res_exit = check_exit(cache_exit, position, params)
    assert res_exit['should_exit'] == True

    cache_no_exit = {
        'donchian_lower_exit': 1.0900,
        'donchian_upper_exit': 1.1000,
        'close': 1.0950
    }
    res_no_exit = check_exit(cache_no_exit, position, params)
    assert res_no_exit['should_exit'] == False
