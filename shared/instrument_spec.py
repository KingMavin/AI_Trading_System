"""
InstrumentSpec — broker-derived contract specification.

Single source of truth for all per-instrument constants.
Built from live MT5 data at Engine startup and cached.
Used by both Engine (live sizing) and Trainer (backtest costs).

NEVER hardcode pip values, contract sizes, or tick values.
All money math flows through this module.

Supported instruments (Phase 1):
  EURUSD, GBPUSD, USDJPY, USDCAD, XAUUSD
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict
from shared.config import OUTPUT_ROOT

log = logging.getLogger(__name__)

# Where we cache the captured spec so the Trainer can use
# it offline without a live MT5 connection
SPEC_CACHE_PATH = OUTPUT_ROOT / "state" / "instrument_specs.json"

# ── SANITY RANGES ──────────────────────────────────────
# Used to detect anomalous broker-reported values.
# XAUUSD: tick_value is 0.10 for 100-oz contracts under standard MT5 tick reporting.
# Range [0.05, 0.20] provides headroom for account currency variations (e.g. EUR/USD conversion).
SANITY_RANGES = {
    "EURUSD": {"tick_value_min": 0.50, "tick_value_max": 1.50},
    "GBPUSD": {"tick_value_min": 0.50, "tick_value_max": 1.50},
    "USDJPY": {"tick_value_min": 0.30, "tick_value_max": 1.20},
    "USDCAD": {"tick_value_min": 0.30, "tick_value_max": 1.20},
    "XAUUSD": {"tick_value_min": 0.05, "tick_value_max": 0.20},
}


@dataclass
class InstrumentSpec:
    """
    Complete broker-derived specification for one instrument.
    Every field sourced from MT5 symbol_info at runtime.
    """
    symbol:             str
    digits:             int       # decimal places in price
    point:              float     # smallest price increment
    contract_size:      float     # units per lot (100000 for FX, 100 for gold)
    tick_size:          float     # smallest measurable price change
    tick_value:         float     # value of one tick move, in account currency
    volume_min:         float
    volume_step:        float
    volume_max:         float
    stops_level:        int       # minimum SL/TP distance in points
    spread_typical:     int       # typical spread in points
    swap_long:          float     # overnight financing per lot (long)
    swap_short:         float     # overnight financing per lot (short)
    currency_base:      str
    currency_profit:    str
    currency_margin:    str
    account_currency:   str       # from account_info, not assumed
    captured_at:        str       # ISO timestamp of last live capture
    sanity_ok:          bool      # False = value is flagged, do not trade

    # ── COMPUTED PROPERTIES ────────────────────────────

    @property
    def pip_size(self) -> float:
        """
        One pip in price terms.
        FX 5-digit: 0.0001 (10 points)
        FX 3-digit (JPY): 0.01 (10 points)
        Gold 2-digit: 0.01 (1 point — no 'pip' concept)
        """
        if self.digits == 5:
            return self.point * 10    # 0.00010
        elif self.digits == 3:
            return self.point * 10    # 0.010
        else:
            return self.point         # gold uses points directly

    @property
    def pip_value(self) -> float:
        """
        Value of one pip move per lot, in account currency.
        Derived from tick_value — the broker gives us this directly
        so no manual currency conversion is needed.
        pip_value = tick_value × (pip_size / tick_size)
        """
        if self.tick_size == 0:
            return 0.0
        return self.tick_value * (self.pip_size / self.tick_size)

    def lot_size_for_risk(self,
                           account_balance: float,
                           risk_pct: float,
                           sl_distance_price: float) -> float:
        """
        Calculate position size that risks exactly risk_pct of balance
        for a given stop-loss distance in price terms.

        Args:
            account_balance:   current account balance in account_currency
            risk_pct:          e.g. 1.0 = 1% of balance
            sl_distance_price: abs(entry_price - sl_price)

        Returns:
            lot size, rounded to volume_step, clamped to [min, max]
        """
        if sl_distance_price <= 0 or self.tick_value == 0:
            return self.volume_min

        risk_amount      = account_balance * (risk_pct / 100)
        sl_distance_ticks= sl_distance_price / self.tick_size
        raw_lots         = risk_amount / (sl_distance_ticks * self.tick_value)

        # Round down to volume_step
        step     = self.volume_step
        adj_lots = int(raw_lots / step) * step

        # Safety: if minimum lot would risk more than configured,
        # log and return 0 so the caller can skip the trade
        if adj_lots <= self.volume_min:
            min_risk = (
                self.volume_min *
                sl_distance_ticks *
                self.tick_value
            )
            if min_risk > risk_amount * 1.5:
                log.warning(
                    f"{self.symbol}: minimum lot ({self.volume_min}) "
                    f"would risk {min_risk:.2f} vs allowed "
                    f"{risk_amount:.2f}. Returning 0 — skip trade."
                )
                return 0.0

        adj_lots = max(self.volume_min, min(self.volume_max, adj_lots))
        adj_lots = round(adj_lots, 2)

        return adj_lots

    def price_to_pips(self, price_distance: float) -> float:
        """Convert a price distance to pips."""
        if self.pip_size == 0:
            return 0.0
        return price_distance / self.pip_size

    def pips_to_price(self, pips: float) -> float:
        """Convert pips to a price distance."""
        return pips * self.pip_size

    def sl_above_stops_level(self, entry: float,
                              sl: float) -> bool:
        """
        Returns True if the SL distance respects the broker's
        minimum stop distance (stops_level in points).
        """
        distance_points = abs(entry - sl) / self.point
        return distance_points >= self.stops_level

    def swap_cost(self, lots: float,
                  direction: str,
                  nights: int = 1) -> float:
        """
        Financing cost for holding a position overnight.
        swap values are per-lot-per-night in account currency.

        Args:
            lots:      position size
            direction: 'BUY' or 'SELL'
            nights:    number of rollovers (usually 1, or 3 on Wed)

        Returns:
            cost in account currency (negative = cost to you)
        """
        rate = self.swap_long if direction == 'BUY' \
               else self.swap_short
        return rate * lots * nights

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def build_simulation_default(cls, symbol: str) -> 'InstrumentSpec':
        """
        Fallback simulation defaults for testing.
        MUST ONLY BE USED when allow_simulation_defaults=True.
        """
        is_jpy = 'JPY' in symbol
        is_gold = 'XAU' in symbol
        
        digits = 3 if is_jpy else (2 if is_gold else 5)
        point = 0.001 if is_jpy else (0.01 if is_gold else 0.00001)
        contract = 100.0 if is_gold else 100000.0
        
        return cls(
            symbol=symbol,
            digits=digits,
            point=point,
            contract_size=contract,
            tick_size=point,
            tick_value=1.0,  # Highly inaccurate fallback
            volume_min=0.01,
            volume_step=0.01,
            volume_max=100.0,
            stops_level=10,
            spread_typical=10,
            swap_long=-5.0,
            swap_short=-5.0,
            currency_base=symbol[:3],
            currency_profit=symbol[3:],
            currency_margin=symbol[:3],
            account_currency='USD',
            captured_at=datetime.now(timezone.utc).isoformat(),
            sanity_ok=False  # Always flag simulation defaults
        )


# ── BUILDER ────────────────────────────────────────────

def build_from_mt5(symbol: str,
                    account_currency: str = None
                    ) -> Optional[InstrumentSpec]:
    """
    Build an InstrumentSpec from a live MT5 connection.
    Performs sanity check on tick_value.

    Args:
        symbol:           MT5 symbol name e.g. 'EURUSD'
        account_currency: from mt5.account_info().currency
                          if None, reads it from MT5 directly

    Returns:
        InstrumentSpec, or None if MT5 unavailable or symbol invalid
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        log.error("MetaTrader5 not installed.")
        return None

    # Ensure symbol is selected in Market Watch
    if not mt5.symbol_select(symbol, True):
        log.error(f"Cannot select {symbol} in MT5.")
        return None

    info = mt5.symbol_info(symbol)
    if info is None:
        log.error(f"symbol_info returned None for {symbol}.")
        return None

    # Read account currency if not provided
    if account_currency is None:
        acct = mt5.account_info()
        account_currency = acct.currency if acct else "EUR"

    # Sanity check tick_value
    sanity  = SANITY_RANGES.get(symbol, {})
    tv_min  = sanity.get("tick_value_min", 0.01)
    tv_max  = sanity.get("tick_value_max", 10.0)
    tv      = info.trade_tick_value
    san_ok  = tv_min <= tv <= tv_max

    if not san_ok:
        log.warning(
            f"{symbol}: tick_value {tv:.6f} is OUTSIDE expected "
            f"range [{tv_min}, {tv_max}]. "
            f"This instrument is FLAGGED — do not trade until "
            f"re-verified during market hours."
        )

    spec = InstrumentSpec(
        symbol           = symbol,
        digits           = info.digits,
        point            = info.point,
        contract_size    = info.trade_contract_size,
        tick_size        = info.trade_tick_size,
        tick_value       = tv,
        volume_min       = info.volume_min,
        volume_step      = info.volume_step,
        volume_max       = info.volume_max,
        stops_level      = info.trade_stops_level,
        spread_typical   = info.spread,
        swap_long        = info.swap_long,
        swap_short       = info.swap_short,
        currency_base    = info.currency_base,
        currency_profit  = info.currency_profit,
        currency_margin  = info.currency_margin,
        account_currency = account_currency,
        captured_at      = datetime.now(timezone.utc).isoformat(),
        sanity_ok        = san_ok,
    )

    log.info(
        f"InstrumentSpec built: {symbol} | "
        f"digits={spec.digits} | "
        f"pip_size={spec.pip_size:.5f} | "
        f"pip_value={spec.pip_value:.4f} {account_currency} | "
        f"contract={spec.contract_size} | "
        f"swap_long={spec.swap_long} | "
        f"sanity={'OK' if san_ok else 'FLAGGED'}"
    )
    return spec


def build_all_from_mt5(symbols: list = None,
                        account_currency: str = None
                        ) -> Dict[str, InstrumentSpec]:
    """
    Build InstrumentSpec for all configured symbols.
    Returns dict keyed by symbol name.
    Failed symbols are excluded with a warning, not an exception.
    """
    symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY",
                          "USDCAD", "XAUUSD"]
    specs   = {}
    for sym in symbols:
        spec = build_from_mt5(sym, account_currency)
        if spec is not None:
            specs[sym] = spec
        else:
            log.warning(f"Could not build spec for {sym} — skipped.")
    return specs


# ── PERSISTENCE ────────────────────────────────────────

def save_specs(specs: Dict[str, InstrumentSpec]) -> None:
    """Save captured specs to disk for offline use."""
    SPEC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {sym: spec.to_dict() for sym, spec in specs.items()}
    tmp  = SPEC_CACHE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    if SPEC_CACHE_PATH.exists():
        SPEC_CACHE_PATH.unlink()
    tmp.rename(SPEC_CACHE_PATH)
    log.info(f"Specs saved to {SPEC_CACHE_PATH}")


def load_specs() -> Dict[str, InstrumentSpec]:
    """
    Load cached specs from disk.
    Returns empty dict if no cache exists.
    Flags any spec whose sanity_ok is False.
    """
    if not SPEC_CACHE_PATH.exists():
        log.warning(
            f"No spec cache at {SPEC_CACHE_PATH}. "
            f"Run capture_specs.py with MT5 connected."
        )
        return {}

    with open(SPEC_CACHE_PATH) as f:
        data = json.load(f)

    specs = {}
    for sym, d in data.items():
        spec = InstrumentSpec(**d)
        if not spec.sanity_ok:
            log.warning(
                f"{sym}: loaded spec is FLAGGED (sanity_ok=False). "
                f"Re-capture during market hours before trading."
            )
        specs[sym] = spec
    return specs


def get_spec(symbol: str,
             specs: Dict[str, InstrumentSpec] = None
             ) -> Optional[InstrumentSpec]:
    """Get spec for one symbol, loading from cache if needed."""
    if specs and symbol in specs:
        return specs[symbol]
    cached = load_specs()
    return cached.get(symbol)
