"""
Unit tests for InstrumentSpec.
Pure logic — no MT5 required.
Run: pytest tests/test_instrument_spec.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pytest
import json
import tempfile
from datetime import datetime, timezone
from shared.instrument_spec import InstrumentSpec, SANITY_RANGES, load_specs, save_specs
from trainer.core.backtester import Backtester


def make_spec(symbol="EURUSD", digits=5, point=1e-5,
              contract_size=100000, tick_size=1e-5,
              tick_value=0.858, volume_min=0.01,
              volume_step=0.01, volume_max=500,
              stops_level=0, spread=1,
              swap_long=-0.7, swap_short=-1.0,
              currency_base="EUR", currency_profit="USD",
              currency_margin="EUR", account_currency="EUR",
              sanity_ok=True):
    return InstrumentSpec(
        symbol=symbol, digits=digits, point=point,
        contract_size=contract_size, tick_size=tick_size,
        tick_value=tick_value, volume_min=volume_min,
        volume_step=volume_step, volume_max=volume_max,
        stops_level=stops_level, spread_typical=spread,
        swap_long=swap_long, swap_short=swap_short,
        currency_base=currency_base,
        currency_profit=currency_profit,
        currency_margin=currency_margin,
        account_currency=account_currency,
        captured_at="2026-06-01T00:00:00+00:00",
        sanity_ok=sanity_ok,
    )


class TestPipSize:

    def test_eurusd_pip_size(self):
        spec = make_spec("EURUSD", digits=5, point=1e-5)
        assert spec.pip_size == pytest.approx(0.0001)

    def test_usdjpy_pip_size(self):
        spec = make_spec("USDJPY", digits=3, point=0.001)
        assert spec.pip_size == pytest.approx(0.01)

    def test_xauusd_pip_size(self):
        # Gold uses point directly — no 10x multiplier
        spec = make_spec("XAUUSD", digits=2, point=0.01,
                         contract_size=100, tick_size=0.01)
        assert spec.pip_size == pytest.approx(0.01)


class TestPipValue:

    def test_eurusd_pip_value(self):
        # tick_value 0.858 per 0.00001 tick
        # pip = 10 ticks → pip_value = 0.858 × 10 = 8.58
        spec = make_spec("EURUSD", digits=5, point=1e-5,
                         tick_size=1e-5, tick_value=0.858)
        assert spec.pip_value == pytest.approx(8.58, rel=0.01)

    def test_usdjpy_pip_value(self):
        # tick_value 0.538 per 0.001 tick
        # pip = 10 ticks → pip_value = 0.538 × 10 = 5.38
        spec = make_spec("USDJPY", digits=3, point=0.001,
                         tick_size=0.001, tick_value=0.538)
        assert spec.pip_value == pytest.approx(5.38, rel=0.01)


class TestLotSizing:

    def test_basic_sizing(self):
        spec = make_spec()
        # 1% of 10000 EUR = 100 EUR risk
        # SL = 20 pips = 0.0020 price = 200 ticks
        # lots = 100 / (200 × 0.858) = 0.58 → rounds down to 0.57
        lots = spec.lot_size_for_risk(
            account_balance=10000,
            risk_pct=1.0,
            sl_distance_price=0.0020
        )
        assert lots >= spec.volume_min
        assert lots <= spec.volume_max
        # Check it's a multiple of volume_step
        assert round(lots / spec.volume_step, 5) == \
               round(round(lots / spec.volume_step), 5)

    def test_zero_sl_returns_minimum(self):
        spec = make_spec()
        lots = spec.lot_size_for_risk(10000, 1.0, 0.0)
        assert lots == spec.volume_min

    def test_over_risk_returns_zero(self):
        # With a tiny balance and a large enough SL, even the minimum
        # lot would exceed the 1.5x risk guard — should return 0
        spec = make_spec(volume_min=0.01, tick_value=0.858)
        lots = spec.lot_size_for_risk(
            account_balance=10,     # tiny balance
            risk_pct=1.0,
            sl_distance_price=0.0002
        )
        # 1% of 10 = 0.10; guard threshold = 0.15
        # min_risk = 0.01 lots × 20 ticks × 0.858 = 0.1716
        assert lots == 0.0

    def test_lot_capped_at_maximum(self):
        spec = make_spec()
        lots = spec.lot_size_for_risk(
            account_balance=999999999,
            risk_pct=100.0,
            sl_distance_price=0.0001
        )
        assert lots <= spec.volume_max


class TestStopsLevel:

    def test_sl_above_stops_level_passes(self):
        # stops_level=50 points, point=0.00001
        spec = make_spec(stops_level=50, point=1e-5)
        # Distance of 100 points — should pass
        assert spec.sl_above_stops_level(1.1000, 1.0990) is True

    def test_sl_below_stops_level_fails(self):
        spec = make_spec(stops_level=50, point=1e-5)
        # Distance of 2 points — below minimum
        assert spec.sl_above_stops_level(1.1000, 1.09998) is False

    def test_stops_level_zero_always_passes(self):
        spec = make_spec(stops_level=0)
        assert spec.sl_above_stops_level(1.1000, 1.0999) is True


class TestSwapCost:

    def test_long_swap(self):
        spec = make_spec(swap_long=-0.7)
        cost = spec.swap_cost(lots=1.0, direction="BUY", nights=1)
        assert cost == pytest.approx(-0.7)

    def test_short_swap(self):
        spec = make_spec(swap_short=-1.0)
        cost = spec.swap_cost(lots=1.0, direction="SELL", nights=1)
        assert cost == pytest.approx(-1.0)

    def test_wednesday_triple_swap(self):
        spec = make_spec(swap_long=-0.7)
        cost = spec.swap_cost(lots=1.0, direction="BUY", nights=3)
        assert cost == pytest.approx(-2.1)

    def test_gold_swap_is_large(self):
        spec = make_spec("XAUUSD", swap_long=-12.6, swap_short=-4.6)
        cost = spec.swap_cost(lots=1.0, direction="BUY", nights=1)
        assert cost == pytest.approx(-12.6)


class TestPriceConversions:

    def test_price_to_pips_eurusd(self):
        spec = make_spec("EURUSD", digits=5, point=1e-5)
        assert spec.price_to_pips(0.0010) == pytest.approx(10.0)

    def test_pips_to_price_eurusd(self):
        spec = make_spec("EURUSD", digits=5, point=1e-5)
        assert spec.pips_to_price(10.0) == pytest.approx(0.0010)


class TestSanityFlag:

    def test_flagged_spec_is_readable(self):
        spec = make_spec(sanity_ok=False)
        assert spec.sanity_ok is False
        # Should still be usable for inspection — just not for trading

    def test_backtester_raises_on_sanity_not_ok(self, tmp_path):
        """
        Backtester.__init__ must raise ValueError when the cached spec
        has sanity_ok=False and allow_simulation_defaults=False.
        No backtest may proceed with a flagged instrument.
        """
        import shared.instrument_spec as mod
        original_path = mod.SPEC_CACHE_PATH
        mod.SPEC_CACHE_PATH = tmp_path / "specs.json"

        flagged_spec = make_spec(symbol="EURUSD", sanity_ok=False)
        save_specs({"EURUSD": flagged_spec})

        with pytest.raises(ValueError, match="sanity_ok is False"):
            Backtester(
                symbol="EURUSD",
                params={},
                allow_simulation_defaults=False,
            )

        mod.SPEC_CACHE_PATH = original_path


class TestPersistence:

    def test_save_and_reload(self, tmp_path):
        import shared.instrument_spec as mod
        original_path = mod.SPEC_CACHE_PATH
        mod.SPEC_CACHE_PATH = tmp_path / "specs.json"

        specs = {"EURUSD": make_spec("EURUSD")}
        save_specs(specs)

        loaded = load_specs()
        assert "EURUSD" in loaded
        assert loaded["EURUSD"].pip_size == \
               pytest.approx(specs["EURUSD"].pip_size)
        assert loaded["EURUSD"].pip_value == \
               pytest.approx(specs["EURUSD"].pip_value)

        mod.SPEC_CACHE_PATH = original_path

    def test_load_missing_returns_empty(self, tmp_path):
        import shared.instrument_spec as mod
        original_path = mod.SPEC_CACHE_PATH
        mod.SPEC_CACHE_PATH = tmp_path / "nonexistent.json"
        result = load_specs()
        assert result == {}
        mod.SPEC_CACHE_PATH = original_path


class TestBacktesterIntegration:

    def test_backtester_uses_captured_spread_from_instrument_spec(
            self, tmp_path):
        import shared.instrument_spec as mod
        original_path = mod.SPEC_CACHE_PATH
        mod.SPEC_CACHE_PATH = tmp_path / "specs.json"

        spec = make_spec(
            symbol="XAUUSD",
            digits=2,
            point=0.01,
            spread=25,
        )
        save_specs({"XAUUSD": spec})

        bt = Backtester(
            symbol="XAUUSD",
            params={},
        )

        assert bt.instrument_spec.spread_typical == 25
        mod.SPEC_CACHE_PATH = original_path

    def test_backtester_warns_when_instrument_spec_missing(
            self, tmp_path, caplog):
        import shared.instrument_spec as mod
        original_path = mod.SPEC_CACHE_PATH
        mod.SPEC_CACHE_PATH = tmp_path / "missing.json"

        caplog.set_level(logging.WARNING)
        bt = Backtester(
            symbol="EURUSD",
            params={},
            allow_simulation_defaults=True
        )

        assert any(
            "no captured InstrumentSpec found" in rec.message
            for rec in caplog.records
        )
        assert bt.instrument_spec.spread_typical == 10
        assert bt.instrument_spec.sanity_ok is False
        mod.SPEC_CACHE_PATH = original_path

    def test_backtester_charges_swap_cost_for_overnight_trade(
            self, tmp_path):
        import shared.instrument_spec as mod
        original_path = mod.SPEC_CACHE_PATH
        mod.SPEC_CACHE_PATH = tmp_path / "specs.json"

        spec = make_spec(
            symbol="EURUSD",
            swap_long=-0.7,
            swap_short=-1.0,
        )
        save_specs({"EURUSD": spec})

        bt = Backtester(
            symbol="EURUSD",
            params={},
        )

        entry = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
        exit = datetime(2026, 6, 2, 10, 0, tzinfo=timezone.utc)
        cost = bt._calculate_swap_cost(
            direction='BUY',
            lots=1.0,
            entry_time=entry,
            exit_time=exit,
        )

        assert cost == pytest.approx(-0.7)
        mod.SPEC_CACHE_PATH = original_path

    def test_backtester_charges_swap_cost_for_overnight_sell_trade(
            self, tmp_path):
        import shared.instrument_spec as mod
        original_path = mod.SPEC_CACHE_PATH
        mod.SPEC_CACHE_PATH = tmp_path / "specs.json"

        spec = make_spec(
            symbol="EURUSD",
            swap_long=-0.7,
            swap_short=-1.0,
        )
        save_specs({"EURUSD": spec})

        bt = Backtester(
            symbol="EURUSD",
            params={},
        )

        entry = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
        exit = datetime(2026, 6, 2, 10, 0, tzinfo=timezone.utc)
        cost = bt._calculate_swap_cost(
            direction='SELL',
            lots=1.0,
            entry_time=entry,
            exit_time=exit,
        )

        assert cost == pytest.approx(-1.0)
        mod.SPEC_CACHE_PATH = original_path


class TestXauusdRealCapturedBrokerData:

    def test_xauusd_real_captured_broker_data_sanity_ok(self):
        """
        Regression test using real broker-captured MT5 data for XAUUSD from 2026-07-27.
        Attributes: tick_value=0.1, digits=2, point=0.01, contract_size=100.0, tick_size=0.01.
        Verifies that sanity_ok evaluates to True with the updated SANITY_RANGES bounds [0.05, 0.20].
        """
        # Real broker-reported MT5 values for XAUUSD (captured 2026-07-27)
        tv = 0.10
        sanity = SANITY_RANGES.get("XAUUSD", {})
        tv_min = sanity.get("tick_value_min", 0.01)
        tv_max = sanity.get("tick_value_max", 10.0)
        san_ok = tv_min <= tv <= tv_max

        spec = InstrumentSpec(
            symbol           = "XAUUSD",
            digits           = 2,
            point            = 0.01,
            contract_size    = 100.0,
            tick_size        = 0.01,
            tick_value       = tv,
            volume_min       = 0.01,
            volume_step      = 0.01,
            volume_max       = 100.0,
            stops_level      = 0,
            spread_typical   = 13,
            swap_long        = -12.6,
            swap_short       = -4.6,
            currency_base    = "XAU",
            currency_profit  = "USD",
            currency_margin  = "USD",
            account_currency = "USD",
            captured_at      = "2026-07-27T20:25:25.234683+00:00",
            sanity_ok        = san_ok,
        )

        assert spec.sanity_ok is True
        assert spec.pip_size == pytest.approx(0.01)
        assert spec.pip_value == pytest.approx(0.10)
        lots = spec.lot_size_for_risk(account_balance=10000.0, risk_pct=1.0, sl_distance_price=5.0)
        assert lots > 0.0

