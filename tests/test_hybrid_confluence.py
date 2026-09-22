"""
Comprehensive tests for the hybrid_confluence strategy template.
Tests cover:
  1. Trend filter (MA gate)
  2. ADX filter
  3. RSI bounce entry (BUY and SELL)
  4. NaN / missing indicator handling
  5. RSI midline exit
  6. Parameter grid integrity (432 combinations, no KeyError on fast_ma absence)
"""
import pytest
import pandas as pd
from trainer.signals.hybrid_confluence import generate_signal, check_exit


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _base_params():
    return {
        'slow_ma_period':  50,
        'ma_type':         'SMA',
        'rsi_period':      14,
        'rsi_oversold':    30,
        'rsi_overbought':  70,
        'adx_threshold':   20,
        'sl_atr_multiple': 2.0,
        'tp_rr_ratio':     2.0,
        'pip_size':        0.0001,
    }

def _buy_cache(close=1.1100, sma_slow=1.1000, adx=25,
               rsi=31.0, rsi_prev=28.0, atr=0.0010):
    """Cache that satisfies all BUY conditions."""
    return {
        'close':    close,
        'sma_slow': sma_slow,
        'adx':      adx,
        'rsi':      rsi,
        'rsi_prev': rsi_prev,
        'atr':      atr,
    }

def _sell_cache(close=1.0900, sma_slow=1.1000, adx=25,
                rsi=69.0, rsi_prev=72.0, atr=0.0010):
    """Cache that satisfies all SELL conditions."""
    return {
        'close':    close,
        'sma_slow': sma_slow,
        'adx':      adx,
        'rsi':      rsi,
        'rsi_prev': rsi_prev,
        'atr':      atr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BUY signal tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuySignal:
    def test_all_conditions_met_produces_buy(self):
        result = generate_signal(_buy_cache(), _base_params())
        assert result['signal'] == 'BUY'
        assert 'HYBRID_BULL' in result['reason']

    def test_buy_sl_below_entry(self):
        cache = _buy_cache(close=1.1100, atr=0.0010)
        params = _base_params()
        result = generate_signal(cache, params)
        assert result['signal'] == 'BUY'
        expected_sl = 1.1100 - (0.0010 * 2.0)
        assert abs(result['sl_price'] - expected_sl) < 1e-9

    def test_buy_tp_above_entry(self):
        cache = _buy_cache(close=1.1100, atr=0.0010)
        params = _base_params()
        result = generate_signal(cache, params)
        expected_tp = 1.1100 + (0.0010 * 2.0 * 2.0)
        assert abs(result['tp_price'] - expected_tp) < 1e-9

    def test_trend_filter_blocks_buy_when_close_below_slow_ma(self):
        """close < sma_slow → HOLD even if all other BUY conditions met."""
        cache = _buy_cache(close=1.0990, sma_slow=1.1000)
        result = generate_signal(cache, _base_params())
        assert result['signal'] == 'HOLD'

    def test_adx_filter_blocks_buy_when_adx_below_threshold(self):
        """adx < adx_threshold → HOLD."""
        cache = _buy_cache(adx=15)
        result = generate_signal(cache, _base_params())  # threshold = 20
        assert result['signal'] == 'HOLD'

    def test_adx_threshold_zero_disables_adx_filter(self):
        """adx_threshold=0 means ADX gate is disabled; BUY should fire."""
        params = _base_params()
        params['adx_threshold'] = 0
        cache = _buy_cache(adx=5)   # ADX is weak — would fail threshold=20
        result = generate_signal(cache, params)
        assert result['signal'] == 'BUY'

    def test_no_signal_when_rsi_already_above_oversold_on_prev(self):
        """RSI was NOT in oversold territory last candle → no bounce → HOLD."""
        cache = _buy_cache(rsi_prev=35.0, rsi=36.0)  # rsi_prev >= oversold(30)
        result = generate_signal(cache, _base_params())
        assert result['signal'] == 'HOLD'

    def test_no_signal_when_rsi_still_below_oversold_this_candle(self):
        """RSI has not yet crossed back above oversold threshold → HOLD (knife-catching guard)."""
        cache = _buy_cache(rsi_prev=25.0, rsi=28.0)  # rsi < oversold(30)
        result = generate_signal(cache, _base_params())
        assert result['signal'] == 'HOLD'

    def test_rsi_exactly_at_threshold_counts_as_bounce(self):
        """rsi == rsi_oversold (i.e., >= threshold) should trigger BUY."""
        cache = _buy_cache(rsi_prev=29.0, rsi=30.0)
        result = generate_signal(cache, _base_params())
        assert result['signal'] == 'BUY'


# ─────────────────────────────────────────────────────────────────────────────
# SELL signal tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSellSignal:
    def test_all_conditions_met_produces_sell(self):
        result = generate_signal(_sell_cache(), _base_params())
        assert result['signal'] == 'SELL'
        assert 'HYBRID_BEAR' in result['reason']

    def test_sell_sl_above_entry(self):
        cache = _sell_cache(close=1.0900, atr=0.0010)
        result = generate_signal(cache, _base_params())
        expected_sl = 1.0900 + (0.0010 * 2.0)
        assert abs(result['sl_price'] - expected_sl) < 1e-9

    def test_sell_tp_below_entry(self):
        cache = _sell_cache(close=1.0900, atr=0.0010)
        result = generate_signal(cache, _base_params())
        expected_tp = 1.0900 - (0.0010 * 2.0 * 2.0)
        assert abs(result['tp_price'] - expected_tp) < 1e-9

    def test_trend_filter_blocks_sell_when_close_above_slow_ma(self):
        cache = _sell_cache(close=1.1100, sma_slow=1.1000)
        result = generate_signal(cache, _base_params())
        assert result['signal'] == 'HOLD'

    def test_adx_filter_blocks_sell(self):
        cache = _sell_cache(adx=10)
        result = generate_signal(cache, _base_params())
        assert result['signal'] == 'HOLD'

    def test_no_signal_when_rsi_prev_not_overbought(self):
        cache = _sell_cache(rsi_prev=65.0, rsi=64.0)  # rsi_prev <= overbought(70)
        result = generate_signal(cache, _base_params())
        assert result['signal'] == 'HOLD'

    def test_no_signal_when_rsi_still_above_overbought(self):
        """RSI has not yet crossed back below overbought threshold → HOLD."""
        cache = _sell_cache(rsi_prev=75.0, rsi=72.0)  # rsi > overbought(70)
        result = generate_signal(cache, _base_params())
        assert result['signal'] == 'HOLD'

    def test_rsi_exactly_at_overbought_counts_as_bounce(self):
        cache = _sell_cache(rsi_prev=71.0, rsi=70.0)
        result = generate_signal(cache, _base_params())
        assert result['signal'] == 'SELL'


# ─────────────────────────────────────────────────────────────────────────────
# NaN / missing indicator safety
# ─────────────────────────────────────────────────────────────────────────────

class TestNaNHandling:
    @pytest.mark.parametrize("missing_key", [
        'close', 'sma_slow', 'rsi', 'rsi_prev', 'adx', 'atr'
    ])
    def test_holds_on_none_indicator(self, missing_key):
        cache = _buy_cache()
        cache[missing_key] = None
        result = generate_signal(cache, _base_params())
        assert result['signal'] == 'HOLD'
        assert missing_key in result['reason']

    @pytest.mark.parametrize("nan_key", [
        'close', 'sma_slow', 'rsi', 'rsi_prev', 'adx', 'atr'
    ])
    def test_holds_on_nan_indicator(self, nan_key):
        cache = _buy_cache()
        cache[nan_key] = float('nan')
        result = generate_signal(cache, _base_params())
        assert result['signal'] == 'HOLD'

    def test_hold_result_keys_present(self):
        """HOLD result must include all expected keys for downstream safety."""
        cache = _buy_cache()
        cache['close'] = None
        result = generate_signal(cache, _base_params())
        for key in ('signal', 'reason', 'sl_price', 'tp_price', 'sl_pips', 'rr_ratio'):
            assert key in result


# ─────────────────────────────────────────────────────────────────────────────
# RSI midline exit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckExit:
    def test_buy_exits_when_rsi_crosses_below_50(self):
        cache    = {'rsi': 49.5, 'rsi_prev': 51.0}
        position = {'direction': 'BUY'}
        result   = check_exit(cache, position, _base_params())
        assert result['should_exit'] is True
        assert result['reason'] == 'EXIT_RSI_MIDLINE_CROSS_DOWN'

    def test_buy_does_not_exit_while_rsi_above_50(self):
        cache    = {'rsi': 55.0, 'rsi_prev': 52.0}
        position = {'direction': 'BUY'}
        result   = check_exit(cache, position, _base_params())
        assert result['should_exit'] is False

    def test_buy_does_not_exit_when_rsi_already_below_50_prev(self):
        """No cross — RSI was already below 50 last candle."""
        cache    = {'rsi': 45.0, 'rsi_prev': 48.0}
        position = {'direction': 'BUY'}
        result   = check_exit(cache, position, _base_params())
        assert result['should_exit'] is False

    def test_sell_exits_when_rsi_crosses_above_50(self):
        cache    = {'rsi': 50.5, 'rsi_prev': 49.0}
        position = {'direction': 'SELL'}
        result   = check_exit(cache, position, _base_params())
        assert result['should_exit'] is True
        assert result['reason'] == 'EXIT_RSI_MIDLINE_CROSS_UP'

    def test_sell_does_not_exit_while_rsi_below_50(self):
        cache    = {'rsi': 45.0, 'rsi_prev': 48.0}
        position = {'direction': 'SELL'}
        result   = check_exit(cache, position, _base_params())
        assert result['should_exit'] is False

    def test_exit_holds_on_nan_rsi(self):
        cache    = {'rsi': float('nan'), 'rsi_prev': 51.0}
        position = {'direction': 'BUY'}
        result   = check_exit(cache, position, _base_params())
        assert result['should_exit'] is False

    def test_exit_holds_on_none_rsi_prev(self):
        cache    = {'rsi': 49.0, 'rsi_prev': None}
        position = {'direction': 'BUY'}
        result   = check_exit(cache, position, _base_params())
        assert result['should_exit'] is False


# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterGrid:
    def test_grid_produces_432_combinations(self):
        from trainer.core.parameter_grid import get_grid
        combos = get_grid('hybrid_confluence')
        assert len(combos) == 432, (
            f"Expected 432 combinations, got {len(combos)}"
        )

    def test_no_fast_ma_in_hybrid_grid(self):
        """The hybrid grid must NOT include fast_ma_period — no KeyError expected."""
        from trainer.core.parameter_grid import get_grid
        combos = get_grid('hybrid_confluence')
        for combo in combos:
            assert 'fast_ma_period' not in combo

    def test_rsi_oversold_always_lt_overbought(self):
        from trainer.core.parameter_grid import get_grid
        combos = get_grid('hybrid_confluence')
        for combo in combos:
            assert combo['rsi_oversold'] < combo['rsi_overbought'], (
                f"Constraint violated: {combo}"
            )

    def test_generate_signal_no_crash_on_hybrid_grid_params(self):
        """No KeyError when generate_signal receives any hybrid grid param set."""
        from trainer.core.parameter_grid import get_grid
        combos = get_grid('hybrid_confluence')
        cache = _buy_cache()
        errors = []
        for params in combos:
            try:
                generate_signal(cache, params)
            except Exception as e:
                errors.append((params, str(e)))
        assert not errors, f"generate_signal crashed on {len(errors)} param sets: {errors[:3]}"
