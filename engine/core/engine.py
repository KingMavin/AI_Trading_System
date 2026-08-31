"""
ATS Engine — live trading loop.

The Engine runs continuously, waking up on each M15 candle
close to check signals and manage positions. It never
self-modifies — all strategy decisions come from the
Trainer-deployed strategy file.

Run:
  python engine/engine.py
  python engine/engine.py --symbol EURUSD --paper

Paper mode:
  Signals are generated and logged but NO orders placed.
  Use this for 1-month validation before live capital.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import logging
import argparse
import json
import os
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple

# MT5 import — graceful degradation if not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

import pandas as pd

from engine.core.strategy_loader import StrategyLoader, StrategySpec
from engine.core.checklist import PreTradeChecklistEvaluator, ChecklistResult
from engine.core.signal_engine import SignalEngine, FETCH_CANDLES
from engine.core.degradation import (
    DegradationMonitor, TradeRecord
)
from engine.core.live_logger import LiveLogger, TradeLogRecord
from engine.core.mandate import Mandate, load_mandate, log_audit_event

LOG_DIR = Path(__file__).parent.parent / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / f"engine_{datetime.now().strftime('%Y%m%d')}.log"
        )
    ]
)
log = logging.getLogger(__name__)

# ── ENGINE STATE & SNAPSHOT PERSISTENCE ────────────────
ENGINE_ROOT        = Path(__file__).parent.parent
STATE_FILE         = ENGINE_ROOT / 'state' / 'engine_state.json'
SNAPSHOT_FILE      = ENGINE_ROOT / 'state' / 'state_snapshot.json'
SHUTDOWN_FLAG_FILE = ENGINE_ROOT / 'state' / 'shutdown_flag.json'
KILL_SWITCH_FILE   = ENGINE_ROOT / 'state' / 'KILL_SWITCH'
ENGINE_LOCK_FILE   = ENGINE_ROOT / 'state' / 'engine.lock.json'  # Dashboard PID discovery
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

def atomic_write_json(file_path: Path, data: Dict) -> None:
    """Atomic write of JSON dictionary via temporary file + fsync + rename."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix('.json.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.flush()
        import os
        os.fsync(f.fileno())
    if file_path.exists():
        file_path.unlink()
    tmp_path.rename(file_path)

def write_state(state: Dict) -> None:
    """Write engine state to disk for dashboard."""
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(STATE_FILE, state)

def write_shutdown_flag(clean: bool) -> None:
    """Write clean or unclean shutdown flag."""
    data = {
        'clean_shutdown': clean,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    atomic_write_json(SHUTDOWN_FLAG_FILE, data)

def load_snapshot() -> Dict:
    """
    Load state snapshot during crash recovery.
    Raises RuntimeError if snapshot is missing, corrupted, or unparseable.
    """
    if not SNAPSHOT_FILE.exists():
        raise RuntimeError(
            f"Crash recovery failed: snapshot file '{SNAPSHOT_FILE}' does not exist. "
            f"Manual investigation required before starting Engine."
        )
    try:
        with open(SNAPSHOT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Crash recovery failed: snapshot file '{SNAPSHOT_FILE}' is corrupted or unparseable ({e}). "
            f"Manual investigation required."
        ) from e

    required_keys = ['peak_equity', 'daily_start_equity', 'last_candle_time', 'circuit_breaker_tripped']
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise RuntimeError(
            f"Crash recovery failed: snapshot file '{SNAPSHOT_FILE}' missing required fields: {missing}. "
            f"Manual investigation required."
        )

    return data

# ── MT5 HELPERS ────────────────────────────────────────

def connect_mt5(login: int = None,
                password: str = None,
                server: str = None) -> bool:
    """Connect to MT5 terminal."""
    if not MT5_AVAILABLE:
        log.error("MetaTrader5 package not available.")
        return False

    if not mt5.initialize():
        log.error(f"MT5 init failed: {mt5.last_error()}")
        return False

    if login and password and server:
        if not mt5.login(login, password=password,
                         server=server):
            log.error(
                f"MT5 login failed: {mt5.last_error()}"
            )
            return False

    info = mt5.account_info()
    if info is None:
        log.error("Cannot get account info.")
        return False

    term_info = mt5.terminal_info()
    if term_info is None or not getattr(term_info, 'trade_allowed', False):
        log.error(
            "MT5_ALGO_TRADING_DISABLED: MT5 Algo Trading is disabled in terminal. "
            "Enable the 'Algo Trading' button in the MT5 toolbar before starting the Engine."
        )
        return False

    log.info(
        f"MT5 connected: account={info.login} | "
        f"server={info.server} | "
        f"balance=${info.balance:,.2f} | "
        f"currency={info.currency}"
    )
    return True

def fetch_candles(symbol: str,
                  timeframe,
                  count: int = FETCH_CANDLES
                  ) -> Optional[pd.DataFrame]:
    """Fetch recent OHLCV candles from MT5."""
    if not MT5_AVAILABLE:
        return None

    rates = mt5.copy_rates_from_pos(
        symbol, timeframe, 0, count
    )
    if rates is None or len(rates) == 0:
        log.warning(
            f"No data returned from MT5 for {symbol}"
        )
        return None

    df = pd.DataFrame(rates)
    df['timestamp'] = pd.to_datetime(
        df['time'], unit='s', utc=True
    )
    df = df.rename(columns={'tick_volume': 'volume'})
    df = df[['timestamp', 'open', 'high',
              'low', 'close', 'volume']]
    df = df.set_index('timestamp').sort_index()
    return df

def get_account_info() -> Dict:
    """Get current account state from MT5."""
    if not MT5_AVAILABLE:
        return {}
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        'balance': info.balance,
        'equity':  info.equity,
        'margin':  info.margin,
        'profit':  info.profit,
    }

def get_open_positions(symbol: str) -> list:
    """Get open positions for symbol."""
    if not MT5_AVAILABLE:
        return []
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    return list(positions)

def calculate_lot_size(symbol: str,
                        sl_pips: float,
                        risk_pct: float,
                        balance: float,
                        pip_size: float = None) -> float:
    """
    Calculate position size using InstrumentSpec money math.
    Fails loudly with ValueError if InstrumentSpec is missing or sanity_ok is False.
    """
    if sl_pips <= 0:
        log.error(f"CALCULATE_LOT_SIZE_FAILED: Invalid sl_pips={sl_pips} for {symbol}. Must be > 0.")
        raise ValueError(f"CALCULATE_LOT_SIZE_FAILED: sl_pips ({sl_pips}) must be > 0.")

    from shared.instrument_spec import get_spec
    spec = get_spec(symbol)
    if spec is None:
        log.error(f"CALCULATE_LOT_SIZE_FAILED: No InstrumentSpec cached for {symbol}.")
        raise ValueError(f"INSTRUMENT_SPEC_MISSING: Cannot calculate lot size for {symbol} — missing InstrumentSpec.")

    if not spec.sanity_ok:
        log.error(f"CALCULATE_LOT_SIZE_FAILED: InstrumentSpec for {symbol} flagged sanity_ok=False.")
        raise ValueError(f"INSTRUMENT_SPEC_INVALID: InstrumentSpec for {symbol} is flagged sanity_ok=False.")

    sl_distance_price = sl_pips * spec.pip_size
    lots = spec.lot_size_for_risk(account_balance=balance, risk_pct=risk_pct, sl_distance_price=sl_distance_price)
    return lots

def place_order(symbol: str,
                signal: Dict,
                lots: float,
                magic: int = 123456,
                paper_mode: bool = True) -> Optional[Dict]:
    """
    Place a market order via MT5.
    In paper mode: log only, no real order.
    """
    direction = signal['signal']
    sl_price  = signal['sl_price']
    tp_price  = signal['tp_price']

    if paper_mode:
        log.info(
            f"[PAPER] {direction} {lots}L {symbol} | "
            f"SL={sl_price:.5f} TP={tp_price:.5f}"
        )
        return {
            'ticket':     -1,
            'symbol':     symbol,
            'direction':  direction,
            'lots':       lots,
            'price':      signal.get('close', 0),
            'sl':         sl_price,
            'tp':         tp_price,
            'paper':      True,
        }

    if not MT5_AVAILABLE:
        return None

    info = mt5.symbol_info(symbol)
    if info is None:
        log.error(f"Cannot get symbol_info for {symbol}. Order placement aborted.")
        return None

    order_type = (mt5.ORDER_TYPE_BUY
                  if direction == 'BUY'
                  else mt5.ORDER_TYPE_SELL)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error(f"Cannot get tick info for {symbol}. Order placement aborted.")
        return None
    price = tick.ask if direction == 'BUY' else tick.bid

    # Negotiate filling mode (Fix 8.2)
    filling_mode_mask = getattr(info, 'filling_mode', 0)
    # Bit 0 (1): FOK, Bit 1 (2): IOC, Bit 2 (4): RETURN
    if filling_mode_mask & 1 or filling_mode_mask == 0:
        filling_type = mt5.ORDER_FILLING_FOK
    elif filling_mode_mask & 2:
        filling_type = mt5.ORDER_FILLING_IOC
    elif filling_mode_mask & 4:
        filling_type = mt5.ORDER_FILLING_RETURN
    else:
        log.error(f"{symbol}: Unsupported filling mode mask {filling_mode_mask}. Order aborted.")
        return None

    # Stops level validation (Fix 8.2)
    stops_level_raw = getattr(info, 'trade_stops_level', 0) or 0
    point_raw = getattr(info, 'point', 0.00001) or 0.00001
    try:
        stops_level = float(stops_level_raw)
    except (ValueError, TypeError):
        stops_level = 0.0
    try:
        point = float(point_raw)
    except (ValueError, TypeError):
        point = 0.00001

    min_stop_dist = stops_level * point

    if min_stop_dist > 0:
        if sl_price > 0 and abs(price - sl_price) < min_stop_dist:
            log.error(
                f"ORDER_REJECTED_STOPS_LEVEL: {symbol} SL distance "
                f"({abs(price - sl_price):.5f}) < min stops level ({min_stop_dist:.5f})"
            )
            return None
        if tp_price > 0 and abs(price - tp_price) < min_stop_dist:
            log.error(
                f"ORDER_REJECTED_STOPS_LEVEL: {symbol} TP distance "
                f"({abs(price - tp_price):.5f}) < min stops level ({min_stop_dist:.5f})"
            )
            return None

    # Deviation cap negotiation (Fix 8.2)
    # Default 10 points for FX majors, 50 points for XAUUSD (Gold)
    DEVIATION_CAPS = {'XAUUSD': 50}
    deviation = DEVIATION_CAPS.get(symbol.upper(), 10)

    request = {
        'action':       mt5.TRADE_ACTION_DEAL,
        'symbol':       symbol,
        'volume':       lots,
        'type':         order_type,
        'price':        price,
        'sl':           sl_price,
        'tp':           tp_price,
        'deviation':    deviation,
        'magic':        magic,
        'comment':      'ATS Engine',
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': filling_type,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(
            f"Order failed: {result.retcode if result else 'None'} "
            f"| {result.comment if result else ''}"
        )
        return None

    log.info(
        f"Order placed: ticket={result.order} | "
        f"{direction} {lots}L {symbol} @ {result.price:.5f} | "
        f"SL={sl_price:.5f} TP={tp_price:.5f}"
    )
    return {
        'ticket':    result.order,
        'symbol':    symbol,
        'direction': direction,
        'lots':      lots,
        'price':     result.price,
        'sl':        sl_price,
        'tp':        tp_price,
        'paper':     False,
    }

def close_position(position,
                   paper_mode: bool = True) -> bool:
    """Close an open position."""
    if paper_mode:
        log.info(
            f"[PAPER] CLOSE ticket={position.ticket} | "
            f"{position.type} {position.volume}L"
        )
        return True

    if not MT5_AVAILABLE:
        return False

    direction = (mt5.ORDER_TYPE_SELL
                 if position.type == 0  # BUY position
                 else mt5.ORDER_TYPE_BUY)

    tick  = mt5.symbol_info_tick(position.symbol)
    price = (tick.bid if direction == mt5.ORDER_TYPE_SELL
             else tick.ask)

    request = {
        'action':       mt5.TRADE_ACTION_DEAL,
        'symbol':       position.symbol,
        'volume':       position.volume,
        'type':         direction,
        'position':     position.ticket,
        'price':        price,
        'magic':        position.magic,
        'comment':      'ATS Engine Close',
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(
            f"Position closed: ticket={position.ticket} | "
            f"price={result.price:.5f}"
        )
        return True

    log.error(
        f"Close failed: {result.retcode if result else 'None'}"
    )
    return False

def wait_for_next_candle(symbol: str,
                          timeframe,
                          check_interval: int = 10) -> None:
    """
    Sleep until the next M15 candle closes.
    Wakes up every check_interval seconds to verify.
    """
    if not MT5_AVAILABLE:
        log.info("MT5 not available. Sleeping 60s.")
        time.sleep(60)
        return

    # Get current candle open time
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
    if rates is None or len(rates) == 0:
        time.sleep(check_interval)
        return

    current_candle_time = rates[0]['time']

    while True:
        time.sleep(check_interval)
        rates = mt5.copy_rates_from_pos(
            symbol, timeframe, 0, 1
        )
        if rates is None:
            continue
        if rates[0]['time'] > current_candle_time:
            return  # new candle has opened

# ── MAIN ENGINE LOOP ───────────────────────────────────

class Engine:
    """
    The live trading Engine.
    Runs continuously. Wakes on each candle close.
    Never self-modifies. All logic from strategy file.
    """

    def __init__(self,
                 symbol:     str   = 'EURUSD',
                 paper_mode: bool  = True,
                 magic:      int   = 123456):

        self.symbol      = symbol
        self.paper_mode  = paper_mode
        self.magic       = magic
        self.running     = False
        self.start_time  = None

        # MT5 timeframe
        self.timeframe   = (mt5.TIMEFRAME_M15
                            if MT5_AVAILABLE
                            else None)

        # Components
        self.strategy_loader = StrategyLoader()
        self.signal_engine:  Optional[SignalEngine]   = None
        self.degradation:    Optional[DegradationMonitor] = None
        self.live_logger:    Optional[LiveLogger]     = None

        # Open position tracking
        self.open_order: Optional[Dict] = None
        self.last_known_tickets: list = []

        # Mandate configuration (WAVE 6)
        self.mandate: Optional[Mandate] = None
        self.mandate_daily_start_equity: float = 0.0

        # Risk state variables (Fix 5.1 & Fix 5.3)
        self.peak_equity: float = 0.0
        self.daily_start_equity: float = 0.0
        self.last_candle_time: Optional[datetime] = None
        self.circuit_breaker_tripped: bool = False
        self.max_daily_loss_pct: float = 3.0
        self.max_trailing_dd_pct: float = 6.0
        self.daily_halt_active: bool = False
        
        # Strategy State persistence
        self.strategy_state: Dict = {}
        self.last_closed_trade: Optional[Dict] = None

        # Connection resilience & SAFE_MODE (Fix 8.1)
        self.safe_mode: bool = False
        self.failed_reconnect_attempts: int = 0
        self.failed_reconciliation_tickets: set = set()
        self.max_reconnect_attempts: int = 3
        self.stale_data_threshold_minutes: int = 45

        # Pre-Trade Checklist Evaluator (Main PRD §3.2 & Panel 6)
        self.checklist_evaluator = PreTradeChecklistEvaluator()
        self.last_checklist_result: Optional[Dict] = None

        # Run stats
        self.candles_processed = 0
        self.signals_generated = 0
        self.orders_placed     = 0

        log.info(
            f"Engine initialised: symbol={symbol} | "
            f"paper={'YES' if paper_mode else 'NO'}"
        )

    def start(self) -> None:
        """Start the Engine. Runs until stopped."""
        self.running    = True
        self.start_time = datetime.now(timezone.utc)

        # Write PID lockfile for dashboard process discovery (WAVE 18).
        # Deleted in _shutdown(). If stale (Engine died without clean shutdown),
        # the dashboard detects staleness via psutil.pid_exists().
        atomic_write_json(ENGINE_LOCK_FILE, {
            'pid':        os.getpid(),
            'started_at': self.start_time.isoformat(),
            'symbol':     self.symbol,
        })

        log.info("="*50)
        log.info("ATS ENGINE STARTING")
        log.info(f"Symbol:     {self.symbol}")
        log.info(f"Mode:       "
                 f"{'PAPER TRADING' if self.paper_mode else 'LIVE TRADING'}")
        log.info(f"Started at: {self.start_time.isoformat()}")
        log.info("="*50)

        # Load Mandate Configuration — Fails loudly if missing/malformed (Fix 6.1)
        self.mandate = load_mandate()

        # Check for filesystem Kill Switch on startup — Halts startup immediately (Fix 6.3)
        if KILL_SWITCH_FILE.exists():
            log.warning("KILL_SWITCH_DETECTED_ON_STARTUP: Sentinel file exists. Trading is HALTED.")
            log_audit_event(
                'KILL_SWITCH_ENGAGED',
                self.mandate.mandate_id if self.mandate else 'UNKNOWN',
                {'reason': 'Kill switch file present on engine startup', 'file': str(KILL_SWITCH_FILE)}
            )
            self.circuit_breaker_tripped = True
            self.running = False
            return

        # Check startup shutdown flag / crash detection (Fix 5.1)
        clean_startup = False
        if SHUTDOWN_FLAG_FILE.exists():
            try:
                with open(SHUTDOWN_FLAG_FILE, 'r', encoding='utf-8') as f:
                    flag_data = json.load(f)
                    clean_startup = flag_data.get('clean_shutdown', False)
            except Exception:
                clean_startup = False

        if not clean_startup:
            log.warning("CRASH_DETECTED_RECOVERING: Unclean shutdown detected. Loading state snapshot...")
            snapshot = load_snapshot()
            self.peak_equity = float(snapshot['peak_equity'])
            self.daily_start_equity = float(snapshot['daily_start_equity'])
            self.circuit_breaker_tripped = bool(snapshot['circuit_breaker_tripped'])
            self.last_known_tickets = list(snapshot.get('last_known_tickets', []))
            
            last_candle_iso = snapshot['last_candle_time']
            self.last_candle_time = datetime.fromisoformat(last_candle_iso)

            # Check outage duration
            outage_duration = datetime.now(timezone.utc) - self.last_candle_time
            if outage_duration > timedelta(days=30):
                log.error(
                    f"OUTAGE_EXCEEDS_MAX_RECONCILIATION_WINDOW: Outage duration of "
                    f"{outage_duration.days} days exceeds maximum limit (30 days)."
                )
                raise RuntimeError(
                    f"Engine outage of {outage_duration.days} days exceeds maximum automatic "
                    f"reconciliation limit (30 days). Manual audit of account state required."
                )

            log.info(
                f"Crash state restored: peak_equity=${self.peak_equity:,.2f} | "
                f"daily_start_equity=${self.daily_start_equity:,.2f} | "
                f"circuit_breaker_tripped={self.circuit_breaker_tripped} | "
                f"last_candle_time={self.last_candle_time.isoformat()}"
            )
        else:
            log.info("CLEAN_STARTUP: Clean shutdown flag present.")

        # Set unclean shutdown flag until clean exit
        write_shutdown_flag(clean=False)

        # Load strategy
        spec = self.strategy_loader.load_strategy()
        if spec is None:
            log.error(
                "No strategy file found. "
                "Deploy a strategy first:\n"
                "  engine/strategy/active_strategy.json"
            )
            return

        # Initialise components
        self.signal_engine = SignalEngine(spec)
        self.degradation   = DegradationMonitor(
            strategy_id=spec.strategy_id,
            wf_median_pf=spec.wf_median_profit_factor,
            wf_profitable_rate=spec.wf_median_win_rate,
        )
        self.live_logger   = LiveLogger(
            strategy_id=spec.strategy_id
        )

        if self.paper_mode:
            log.info(
                "PAPER MODE: signals logged, "
                "no real orders placed."
            )

        # Main loop
        self._main_loop()

    def save_snapshot(self) -> None:
        """Atomically persist state snapshot for crash recovery (Fix 5.1)."""
        snapshot = {
            'peak_equity':            self.peak_equity,
            'daily_start_equity':     self.daily_start_equity,
            'circuit_breaker_tripped': self.circuit_breaker_tripped,
            'last_known_tickets':     self.last_known_tickets,
            'last_candle_time':       self.last_candle_time.isoformat() if self.last_candle_time else datetime.now(timezone.utc).isoformat(),
            'updated_at':             datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(SNAPSHOT_FILE, snapshot)

    def _main_loop(self) -> None:
        """
        Main candle processing loop.
        Runs forever until stopped or error.
        """
        consecutive_errors = 0

        while self.running:
            try:
                self._process_candle()
                consecutive_errors = 0

                if not self.running:
                    log.info("Engine requested stop. Exiting main loop.")
                    break

                # Wait for next candle
                wait_for_next_candle(
                    self.symbol, self.timeframe
                )

            except KeyboardInterrupt:
                log.info("Engine stopped by user.")
                self.running = False

            except Exception as e:
                consecutive_errors += 1
                log.error(
                    f"Candle error (#{consecutive_errors}): {e}",
                    exc_info=True
                )
                if consecutive_errors >= 5:
                    log.error(
                        "5 consecutive errors. "
                        "Pausing 5 minutes before retry."
                    )
                    time.sleep(300)
                    consecutive_errors = 0
                else:
                    time.sleep(30)

        self._shutdown()

    def _process_candle(self) -> None:
        """Process one candle close."""
        t_start = time.perf_counter()
        self.candles_processed += 1
        now = datetime.now(timezone.utc)

        # Check for filesystem Kill Switch (Fix 6.3)
        if KILL_SWITCH_FILE.exists():
            if not self.circuit_breaker_tripped:
                log.warning("KILL_SWITCH_ENGAGED: Sentinel file detected. Halting engine.")
                log_audit_event(
                    'KILL_SWITCH_ENGAGED',
                    self.mandate.mandate_id if self.mandate else 'UNKNOWN',
                    {'reason': 'Kill switch file detected during candle processing', 'file': str(KILL_SWITCH_FILE)}
                )
            self.circuit_breaker_tripped = True
            self.running = False
            return

        # Check MT5 connection health & stale data guard (Fix 8.1)
        healthy = True if self.paper_mode else self._check_connection_health()
        if not healthy:
            log.warning("Connection health check failed. Skipping candle fetch, signal generation, and new trade entries.")
            spec = self.strategy_loader.get_spec()
            positions = get_open_positions(self.symbol)
            if positions and spec:
                account = get_account_info()
                signal = {'signal': 'NONE', 'reason': 'Connection unhealthy — exit monitoring only'}
                self._manage_positions(positions, signal, None, spec, account)
            return

        # Check for strategy reload
        reloaded = self.strategy_loader.reload_if_changed()
        if reloaded:
            new_spec = self.strategy_loader.get_spec()
            if new_spec:
                self.signal_engine.update_spec(new_spec)
                self.degradation = DegradationMonitor(
                    strategy_id=new_spec.strategy_id,
                    wf_median_pf=new_spec.wf_median_profit_factor,
                    wf_profitable_rate=new_spec.wf_median_win_rate,
                )

        spec = self.strategy_loader.get_spec()
        if spec is None:
            log.warning("No strategy loaded. Skipping candle.")
            return

        # Fetch live candles from MT5 (timed for performance observation)
        t_mt5_start = time.perf_counter()
        df = fetch_candles(
            self.symbol, self.timeframe, FETCH_CANDLES
        )
        if df is None:
            log.warning("Could not fetch candles from MT5.")
            return

        # Track candle timestamp & midnight date rollover (Fix 5.1 & Fix 5.3)
        candle_timestamp = df.index[-1]
        candle_dt = candle_timestamp.to_pydatetime() if hasattr(candle_timestamp, 'to_pydatetime') else candle_timestamp

        # Get account state
        account = get_account_info()
        mt5_duration_ms = (time.perf_counter() - t_mt5_start) * 1000.0
        equity  = account.get('equity', 10000.0)
        balance = account.get('balance', 10000.0)

        # Initialise peak_equity and daily_start_equity on first candle if 0.0
        if self.peak_equity == 0.0 or equity > self.peak_equity:
            self.peak_equity = equity

        if self.daily_start_equity == 0.0:
            self.daily_start_equity = equity

        if self.mandate_daily_start_equity == 0.0:
            self.mandate_daily_start_equity = equity

        # Check for UTC midnight date rollover
        if self.last_candle_time is not None:
            if candle_dt.date() > self.last_candle_time.date():
                log.info(
                    f"UTC DATE ROLLOVER ({self.last_candle_time.date()} -> {candle_dt.date()}): "
                    f"Resetting daily_start_equity from ${self.daily_start_equity:,.2f} to ${equity:,.2f}"
                )
                self.daily_start_equity = equity
                self.mandate_daily_start_equity = equity
                self.daily_halt_active = False

        self.last_candle_time = candle_dt

        # Circuit Breaker Evaluation (Fix 5.3 & WAVE 38)
        daily_pnl_pct = ((equity - self.daily_start_equity) / self.daily_start_equity) * 100.0 if self.daily_start_equity > 0 else 0.0
        trailing_dd_pct = ((self.peak_equity - equity) / self.peak_equity) * 100.0 if self.peak_equity > 0 else 0.0

        if daily_pnl_pct <= -self.max_daily_loss_pct:
            if not self.circuit_breaker_tripped:
                log.warning(
                    f"DAILY_LOSS_LIMIT_HIT: Current daily loss ({daily_pnl_pct:.2f}%) "
                    f"exceeds max threshold (-{self.max_daily_loss_pct}%). Circuit breaker TRIPPED."
                )
            self.circuit_breaker_tripped = True

        if trailing_dd_pct >= self.max_trailing_dd_pct:
            if not self.circuit_breaker_tripped:
                log.warning(
                    f"TRAILING_DD_LIMIT_HIT: Trailing drawdown ({trailing_dd_pct:.2f}%) "
                    f"exceeds max threshold ({self.max_trailing_dd_pct}%). Circuit breaker TRIPPED."
                )
            self.circuit_breaker_tripped = True

        # Intraday daily loss halt evaluation
        daily_loss_halt_pct = spec.parameters.get('daily_loss_halt_pct', 0.0)
        if daily_loss_halt_pct > 0.0 and daily_pnl_pct <= -daily_loss_halt_pct:
            if not self.daily_halt_active:
                log.warning(
                    f"STRATEGY_DAILY_HALT: Current daily loss ({daily_pnl_pct:.2f}%) "
                    f"exceeds template intraday halt limit (-{daily_loss_halt_pct}%). Halting new entries for the day."
                )
            self.daily_halt_active = True

        # Real-Time Mandate Evaluation & Position Flattening (Fix 6.3)
        if self.mandate:
            mandate_daily_loss = ((equity - self.mandate_daily_start_equity) / self.mandate_daily_start_equity) * 100.0 if self.mandate_daily_start_equity > 0 else 0.0
            if mandate_daily_loss <= -self.mandate.max_daily_loss_pct:
                if not self.circuit_breaker_tripped:
                    log.warning(
                        f"MANDATE_DAILY_LOSS_HIT: Daily loss ({mandate_daily_loss:.2f}%) "
                        f"breached mandate limit (-{self.mandate.max_daily_loss_pct}%)."
                    )
                    log_audit_event(
                        'MANDATE_DAILY_LOSS_BREACH',
                        self.mandate.mandate_id,
                        {'daily_loss_pct': mandate_daily_loss, 'limit_pct': self.mandate.max_daily_loss_pct, 'equity': equity}
                    )
                self.circuit_breaker_tripped = True

            mandate_overall_dd = ((self.peak_equity - equity) / self.peak_equity) * 100.0 if self.peak_equity > 0 else 0.0
            if mandate_overall_dd >= self.mandate.max_overall_drawdown_pct:
                log.error(
                    f"MANDATE_OVERALL_DD_BREACH: Drawdown ({mandate_overall_dd:.2f}%) "
                    f"breached mandate threshold ({self.mandate.max_overall_drawdown_pct}%). FLATTENING ALL POSITIONS."
                )
                log_audit_event(
                    'MANDATE_OVERALL_DD_BREACH',
                    self.mandate.mandate_id,
                    {
                        'overall_dd_pct': mandate_overall_dd,
                        'limit_pct': self.mandate.max_overall_drawdown_pct,
                        'drawdown_type': self.mandate.drawdown_type,
                        'peak_equity': self.peak_equity,
                        'current_equity': equity,
                    }
                )
                self.circuit_breaker_tripped = True

                # Flatten all open positions immediately (Fix 6.3)
                open_pos = get_open_positions(self.symbol)
                if open_pos:
                    log.info(f"FLATTENING {len(open_pos)} OPEN POSITIONS DUE TO MANDATE DD BREACH.")
                    for pos in open_pos:
                        close_position(pos, self.paper_mode)
                    log_audit_event(
                        'POSITIONS_FLATTENED',
                        self.mandate.mandate_id,
                        {'positions_closed': [getattr(p, 'ticket', -1) for p in open_pos]}
                    )

        # Generate signal
        signal = self.signal_engine.process_candle(
            df,
            strategy_state=self.strategy_state,
            last_closed_trade=self.last_closed_trade
        )
        cache  = self.signal_engine.get_cache()

        if signal.get('signal') == 'ERROR':
            log.error(f"SIGNAL_EVALUATION_CRASH: Signal generation crashed: {signal.get('reason')}. Tripping circuit breaker.")
            self.circuit_breaker_tripped = True
            return

        log.info(
            f"Candle #{self.candles_processed} | "
            f"{cache.get('session','?')} | "
            f"{cache.get('regime','?')} | "
            f"Signal: {signal['signal']} | "
            f"{signal['reason'][:60]}"
        )

        # 1. Position Management & Reconciliation (Closes run before Entries) (Fix 8.3)
        positions = get_open_positions(self.symbol)
        self._manage_positions(positions, signal,
                                df, spec, account)

        # Re-query remaining positions post-management to reflect closes before entry evaluation (Fix 8.3)
        remaining_positions = get_open_positions(self.symbol)

        # Evaluate 9-Item Pre-Trade Checklist (Main PRD §3.2 & Panel 6)
        max_spread = (spec.filters.get('max_spread_pips', 3.0) if (spec and spec.filters) else 3.0)
        allowed_sess = (spec.filters.get('allowed_sessions') if (spec and spec.filters) else None)
        regime_tgt = (getattr(spec, 'regime_target', None) if spec else None)
        mandate_d_loss = ((equity - self.mandate_daily_start_equity) / self.mandate_daily_start_equity) * 100.0 if (self.mandate and self.mandate_daily_start_equity > 0) else 0.0
        d_loss_ok = not (self.mandate and mandate_d_loss <= -self.mandate.max_daily_loss_pct)
        mandate_ov_dd = ((self.peak_equity - equity) / self.peak_equity) * 100.0 if (self.mandate and self.peak_equity > 0) else 0.0
        dd_ok = not (self.mandate and mandate_ov_dd >= self.mandate.max_overall_drawdown_pct)
        f_margin = float(getattr(account, 'margin_free', balance)) if account else float(balance)

        mt5_conn = (not self.safe_mode)
        if MT5_AVAILABLE and hasattr(mt5, 'terminal_info'):
            info = mt5.terminal_info()
            if info is not None:
                mt5_conn = getattr(info, 'connected', True)

        try:
            stale_secs = float((now - self.last_candle_time).total_seconds()) if self.last_candle_time else 0.0
        except (TypeError, ValueError, AttributeError):
            stale_secs = 0.0

        stale_limit = (self.stale_data_threshold_minutes * 60) if isinstance(self.stale_data_threshold_minutes, (int, float)) else 2700
        data_is_fresh = stale_secs < stale_limit

        checklist_res = self.checklist_evaluator.evaluate(
            mt5_connected=mt5_conn,
            strategy_spec=spec,
            data_fresh=data_is_fresh,
            current_spread_pips=1.0,  # normalized live/backtest spread
            max_spread_pips=max_spread,
            current_session=str(cache.get('session', 'UNKNOWN')),
            allowed_sessions=allowed_sess,
            current_regime=str(cache.get('regime', 'UNKNOWN')),
            regime_target=regime_tgt,
            safe_mode=self.safe_mode or self.circuit_breaker_tripped,
            daily_loss_ok=d_loss_ok,
            drawdown_ok=dd_ok,
            free_margin=f_margin,
            required_margin=100.0,
            open_positions_count=len(remaining_positions),
            max_open_positions=getattr(self.mandate, 'max_open_positions', 3) if self.mandate else 3,
            has_conflicting_position=False,
            stale_seconds=stale_secs,
        )

        self.last_checklist_result = checklist_res.to_dict()

        # Log CANDLE_EVALUATION event with full 9-item checklist status (Main PRD §3.2)
        logger_obj = getattr(self, 'live_logger', None) or getattr(self, 'logger', None)
        if logger_obj:
            logger_obj.log_candle_event({
                'event_type': 'CANDLE_EVALUATION',
                'candle_index': self.candles_processed,
                'signal': signal['signal'],
                'checklist': self.last_checklist_result
            })

        # 2. Entry Logic (Evaluated after position management & closes complete) (Fix 8.3)
        if not remaining_positions and signal['signal'] in ('BUY', 'SELL'):
            allowed, gate_reason = self._check_mandate_pre_trade_gate(signal, spec)
            if not allowed:
                log.warning(f"PRE_TRADE_MANDATE_BLOCKED: {gate_reason}")
                log_audit_event(
                    'PRE_TRADE_BLOCKED',
                    self.mandate.mandate_id if self.mandate else 'UNKNOWN',
                    {'symbol': self.symbol, 'signal': signal['signal'], 'reason': gate_reason}
                )
            else:
                self._enter_trade(signal, spec, balance)

        # Write engine state for dashboard
        write_state({
            'running':        True,
            'candle':         self.candles_processed,
            'symbol':         self.symbol,
            'last_candle':    now.isoformat(),
            'session':        cache.get('session'),
            'regime':         str(cache.get('regime')),
            'equity':         equity,
            'balance':        balance,
            'open_position':  (
                {
                    'direction': positions[0].type,
                    'lots':      positions[0].volume,
                    'profit':    positions[0].profit,
                }
                if positions else None
            ),
            'last_signal':    signal['signal'],
            'paper_mode':     self.paper_mode,
            'degradation':    self.degradation.get_summary(),
            'checklist':      self.last_checklist_result,
        })

        # Save atomic state snapshot (Fix 5.1)
        self.save_snapshot()

        # Non-blocking engine performance observation (Fix 10.4)
        try:
            total_duration_ms = (time.perf_counter() - t_start) * 1000.0
            eval_duration_ms = max(0.0, total_duration_ms - mt5_duration_ms)
            from trainer.core.resource_monitor import log_performance_record
            log_performance_record(
                source_system='ENGINE',
                metric_type='CANDLE_TIMING',
                data={
                    'symbol': self.symbol,
                    'candles_processed': self.candles_processed,
                    'total_duration_ms': round(total_duration_ms, 2),
                    'mt5_duration_ms': round(mt5_duration_ms, 2),
                    'eval_duration_ms': round(eval_duration_ms, 2),
                }
            )
        except Exception as perf_err:
            log.warning(f"Engine performance logging failed: {perf_err}")

        # Write degradation status file for Trainer
        self.degradation.write_status_file()

    def _check_connection_health(self) -> bool:
        """
        Check MT5 connection status and candle data freshness (Fix 8.1).
        On disconnection or stale data, attempts exponential backoff reconnection.
        After 3 consecutive failures, enters SAFE_MODE (blocks new entries).
        Returns True if connection and data are healthy, False otherwise.
        """
        now = datetime.now(timezone.utc)
        is_connected = True
        disconnect_reason = ""

        # 1. MT5 Terminal Info Check
        if MT5_AVAILABLE:
            info = mt5.terminal_info()
            if info is None or not getattr(info, 'connected', False):
                is_connected = False
                disconnect_reason = "MT5 terminal_info reports disconnected or None"
            elif not getattr(info, 'trade_allowed', False):
                is_connected = False
                disconnect_reason = (
                    "MT5_ALGO_TRADING_DISABLED: MT5 terminal_info reports trade_allowed is False "
                    "(Algo Trading toolbar button is turned OFF)"
                )

        # 2. Stale-Data Guard Check
        if is_connected and self.last_candle_time is not None:
            candle_age_minutes = (now - self.last_candle_time).total_seconds() / 60.0
            if candle_age_minutes > self.stale_data_threshold_minutes:
                is_connected = False
                disconnect_reason = (
                    f"Stale data guard: last candle ({self.last_candle_time.isoformat()}) "
                    f"is {candle_age_minutes:.1f}m old (threshold: {self.stale_data_threshold_minutes}m)"
                )

        if is_connected:
            if self.failed_reconnect_attempts > 0:
                log.info(f"MT5 Connection RESTORED after {self.failed_reconnect_attempts} failed attempt(s).")
                self.failed_reconnect_attempts = 0
                self.safe_mode = False
            return True

        # Disconnection or Stale Data Detected
        self.failed_reconnect_attempts += 1
        log.warning(
            f"CONNECTION_HEALTH_FAILURE (#{self.failed_reconnect_attempts}): {disconnect_reason}"
        )

        # Exponential backoff schedule: 5s, 10s, 20s, 40s, capped at 60s
        backoff_sec = min(5 * (2 ** (self.failed_reconnect_attempts - 1)), 60)
        log.info(f"Retrying connection in {backoff_sec}s...")
        time.sleep(backoff_sec)

        # Attempt reconnection
        reconnected = False
        if MT5_AVAILABLE:
            reconnected = mt5.initialize()

        if reconnected:
            log.info("Reconnection attempt succeeded.")
            self.failed_reconnect_attempts = 0
            self.safe_mode = False
            return True

        # Check if max reconnect attempts exceeded -> Enter SAFE_MODE
        if self.failed_reconnect_attempts >= self.max_reconnect_attempts:
            if not self.safe_mode:
                log.error(
                    f"MAX_RECONNECT_ATTEMPTS_EXCEEDED ({self.failed_reconnect_attempts}/{self.max_reconnect_attempts}). "
                    f"Entering SAFE_MODE: Blocking all new trade entries."
                )
                log_audit_event(
                    'SAFE_MODE_ENGAGED',
                    self.mandate.mandate_id if self.mandate else 'UNKNOWN',
                    {
                        'reason': disconnect_reason,
                        'failed_attempts': self.failed_reconnect_attempts,
                        'max_attempts': self.max_reconnect_attempts,
                    }
                )
            self.safe_mode = True
            self.circuit_breaker_tripped = True

        return False

    def _check_mandate_pre_trade_gate(self, signal: Dict, spec: StrategySpec) -> Tuple[bool, str]:
        """
        Evaluate Pre-Trade Mandate Gate (Fix 6.2).
        Returns (allowed: bool, reason: str).
        """
        if not self.mandate:
            return False, "No mandate loaded — engine fail-closed"

        # Check SAFE_MODE state (Fix 8.1)
        if self.safe_mode:
            return False, "SAFE_MODE is active due to MT5 connection failure or stale data"

        # 1. Symbol Universe Check
        if self.symbol.upper() not in self.mandate.symbol_universe:
            return False, f"Symbol '{self.symbol}' is not in mandate symbol universe {self.mandate.symbol_universe}"

        # 2. Per-trade risk sizing cap check
        if spec.risk_per_trade_pct > self.mandate.max_daily_risk_pct:
            return False, f"Strategy risk per trade ({spec.risk_per_trade_pct}%) exceeds mandate max risk ({self.mandate.max_daily_risk_pct}%)"

        # 3. Circuit breaker or mandate violation status check
        if getattr(self, 'circuit_breaker_tripped', False):
            return False, "Circuit breaker or mandate violation is currently active"
            
        if getattr(self, 'daily_halt_active', False):
            return False, "STRATEGY_DAILY_HALT is currently active"

        return True, "OK"

    def _enter_trade(self, signal: Dict,
                      spec: StrategySpec,
                      balance: float) -> None:
        """Enter a new trade."""
        sl_pips = signal.get('sl_pips', 0)
        if sl_pips <= 0:
            log.error(f"ORDER_REJECTED_MISSING_SL_PIPS: Signal for {self.symbol} missing or invalid sl_pips ({sl_pips}).")
            return

        lots = calculate_lot_size(
            symbol=self.symbol,
            sl_pips=sl_pips,
            risk_pct=spec.risk_per_trade_pct,
            balance=balance,
            pip_size=spec.pip_size,
        )

        order = place_order(
            symbol=self.symbol,
            signal={
                **signal,
                'close': self.signal_engine.get_cache().get(
                    'close', 0
                ),
            },
            lots=lots,
            magic=self.magic,
            paper_mode=self.paper_mode,
        )

        if order:
            self.open_order = order
            self.orders_placed += 1
            self.signals_generated += 1
            log.info(
                f"Trade entered: {signal['signal']} "
                f"{lots}L | "
                f"SL={signal['sl_price']:.5f} "
                f"TP={signal['tp_price']:.5f}"
            )

    def _reconcile_closed_positions(self, current_positions: list) -> None:
        """
        Reconcile open positions against MT5 deal history to detect broker-side SL/TP closes (Fix 5.2).
        """
        current_tickets = [p.ticket for p in current_positions] if current_positions else []
        closed_tickets = [t for t in self.last_known_tickets if t not in current_tickets]

        if closed_tickets and MT5_AVAILABLE:
            for ticket in closed_tickets:
                try:
                    # Query history deals from 30 days prior to handle any outage duration
                    from_date = (self.last_candle_time - timedelta(days=30)) if self.last_candle_time else (datetime.now(timezone.utc) - timedelta(days=30))
                    to_date = datetime.now(timezone.utc)
                    deals = mt5.history_deals_get(from_date, to_date, position=ticket)
                    if deals:
                        out_deals = [d for d in deals if getattr(d, 'entry', -1) == 1 or getattr(d, 'entry', -1) == getattr(mt5, 'DEAL_ENTRY_OUT', 1)]
                        if not out_deals:
                            out_deals = list(deals)
                        exit_deal = out_deals[-1]

                        # Determine close reason
                        reason_code = getattr(exit_deal, 'reason', -1)
                        if reason_code == getattr(mt5, 'DEAL_REASON_SL', 4):
                            reason_str = 'SL_HIT'
                        elif reason_code == getattr(mt5, 'DEAL_REASON_TP', 5):
                            reason_str = 'TP_HIT'
                        elif reason_code == getattr(mt5, 'DEAL_REASON_CLIENT', 0):
                            reason_str = 'MANUAL_CLOSE'
                        elif reason_code == getattr(mt5, 'DEAL_REASON_SO', 3):
                            reason_str = 'MARGIN_STOP_OUT'
                        else:
                            reason_str = getattr(exit_deal, 'comment', 'BROKER_CLOSE') or 'BROKER_CLOSE'

                        log.info(
                            f"RECONCILED BROKER CLOSE: ticket={ticket} | "
                            f"reason={reason_str} | profit=${exit_deal.profit:,.2f}"
                        )

                        class ReconstructedPosition:
                            pass
                        dummy_pos = ReconstructedPosition()
                        dummy_pos.ticket = ticket
                        dummy_pos.type = 0 if getattr(exit_deal, 'type', 0) == 1 else 1
                        dummy_pos.price_open = deals[0].price if deals else exit_deal.price
                        dummy_pos.price_current = exit_deal.price
                        dummy_pos.volume = exit_deal.volume
                        dummy_pos.profit = exit_deal.profit + getattr(exit_deal, 'swap', 0.0) + getattr(exit_deal, 'commission', 0.0)

                        account = get_account_info()
                        self._record_closed_trade(dummy_pos, reason_str, account)

                except Exception as e:
                    log.error(f"Error reconciling closed position ticket={ticket}: {e}")
                    self.safe_mode = True
                    if ticket not in self.failed_reconciliation_tickets:
                        self.failed_reconciliation_tickets.add(ticket)
                        log_audit_event('SAFE_MODE_ENGAGED', self.symbol, {'reason': f'Closed position reconciliation failed for ticket={ticket}: {e}'})
                    current_tickets.append(ticket)  # Retain ticket so tracking is not dropped

        self.last_known_tickets = current_tickets

    def _manage_positions(self, positions: list,
                           signal: Dict,
                           df: pd.DataFrame,
                           spec: StrategySpec,
                           account: Dict) -> None:
        """
        Manage open positions and reconcile broker-side SL/TP closes (Fix 5.2).
        Check for signal-based exits.
        """
        # Reconcile closed positions (Fix 5.2)
        self._reconcile_closed_positions(positions)

        if not positions:
            return

        for position in positions:
            # Signal-based exit
            if spec.parameters.get(
                'exit_on_opposite_crossover', False
            ):
                pos_dir = 'BUY' if position.type == 0 \
                          else 'SELL'
                opposite = (
                    signal['signal'] == 'SELL'
                    and pos_dir == 'BUY'
                ) or (
                    signal['signal'] == 'BUY'
                    and pos_dir == 'SELL'
                )
                if opposite:
                    log.info(
                        f"Opposite signal — closing "
                        f"position {position.ticket}"
                    )
                    closed = close_position(
                        position, self.paper_mode
                    )
                    if closed:
                        self._record_closed_trade(
                            position,
                            'EXIT_OPPOSITE_CROSSOVER',
                            account
                        )

    def _record_closed_trade(self, position,
                              reason: str,
                              account: Dict) -> None:
        """
        Record a closed trade to live logs and
        degradation monitor.
        """
        cache   = self.signal_engine.get_cache() if self.signal_engine else {}
        net_pnl = getattr(position, 'profit', 0)
        now     = datetime.now(timezone.utc).isoformat()

        # Degradation monitor
        trade_record = TradeRecord(
            trade_id   = str(getattr(position, 'ticket', -1)),
            direction  = 'BUY' if position.type == 0
                         else 'SELL',
            entry_price= getattr(position, 'price_open', 0),
            exit_price = getattr(position, 'price_current', 0),
            net_pnl    = net_pnl,
            close_reason=reason,
            closed_at  = now,
            strategy_id= (
                self.strategy_loader.get_spec().strategy_id
                if self.strategy_loader.get_spec() else ''
            ),
        )
        self.degradation.record_trade(trade_record)

        # Update strategy state (WAVE 38)
        self.strategy_state.clear()
        self.last_closed_trade = {
            'net_pnl': net_pnl,
            'direction': trade_record.direction,
            'reason': reason
        }

        # Check for CUSUM Degradation Trip (WAVE 15)
        cusum_tripped = getattr(self.degradation.state, 'cusum_tripped', False) is True
        if cusum_tripped and not self.circuit_breaker_tripped:
            reason_str = str(getattr(self.degradation.state, 'cusum_reason', 'CUSUM degradation tripped'))
            try:
                score_val = float(getattr(self.degradation.state, 'cusum_score', 0.0))
            except (TypeError, ValueError):
                score_val = 0.0
            log.warning(f"LIVE_DEGRADATION_TRIPPED: {reason_str}")
            self.circuit_breaker_tripped = True
            self.safe_mode = True
            log_audit_event(
                'LIVE_DEGRADATION_TRIPPED',
                self.mandate.mandate_id if self.mandate else 'UNKNOWN',
                {
                    'strategy_id': self.strategy_loader.get_spec().strategy_id if self.strategy_loader.get_spec() else 'UNKNOWN',
                    'cusum_score': round(score_val, 4),
                    'threshold_h': float(getattr(self.degradation.cusum, 'h', 12.0)) if hasattr(self.degradation, 'cusum') else 12.0,
                    'reason': reason_str,
                    'total_trades': len(self.degradation.all_trades) if hasattr(self.degradation, 'all_trades') and isinstance(self.degradation.all_trades, (list, deque)) else 0,
                }
            )

        # Live log
        spec = self.strategy_loader.get_spec()
        if spec and self.live_logger:
            log_record = TradeLogRecord(
                schema_version   = '1.0',
                trade_id         = str(
                    getattr(position, 'ticket', -1)
                ),
                strategy_id      = spec.strategy_id,
                run_timestamp    = self.start_time.strftime(
                    '%Y%m%d_%H%M%S'
                ) if self.start_time else '',
                candle_time      = now,
                symbol           = self.symbol,
                timeframe        = 'M15',
                direction        = (
                    'BUY' if position.type == 0 else 'SELL'
                ),
                entry_price      = getattr(
                    position, 'price_open', 0
                ),
                entry_time       = now,
                exit_price       = getattr(
                    position, 'price_current', 0
                ),
                exit_time        = now,
                lots             = getattr(position, 'volume', 0),
                gross_pnl        = net_pnl,
                net_pnl          = net_pnl,
                commission       = 0.0,
                spread_cost      = 0.0,
                close_reason     = reason,
                entry_session    = cache.get('session', ''),
                entry_regime     = str(cache.get('regime', '')),
                entry_atr        = cache.get('atr', 0) or 0,
                entry_adx        = cache.get('adx', 0) or 0,
                entry_spread_pips= 0.0,
                fast_ma_period   = spec.fast_ma,
                slow_ma_period   = spec.slow_ma,
                sl_atr_multiple  = spec.sl_atr_multiple,
                tp_rr_ratio      = spec.tp_rr_ratio,
                duration_candles = 0,
                duration_minutes = 0.0,
                equity_after     = account.get('equity', 0),
                balance_after    = account.get('balance', 0),
                drawdown_pct     = 0.0,
            )
            self.live_logger.log_trade(log_record)

    def _shutdown(self) -> None:
        """Clean shutdown."""
        log.info("Engine shutting down...")
        if MT5_AVAILABLE:
            mt5.shutdown()
        write_state({'running': False})
        write_shutdown_flag(clean=True)
        # Remove PID lockfile so dashboard sees Engine as stopped (WAVE 18).
        try:
            if ENGINE_LOCK_FILE.exists():
                ENGINE_LOCK_FILE.unlink()
        except Exception as e:
            log.warning(f"Could not remove engine lock file: {e}")
        log.info(
            f"Engine stopped. "
            f"Candles: {self.candles_processed} | "
            f"Orders: {self.orders_placed}"
        )

# ── ENTRY POINT ────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ATS Engine — live trading'
    )
    parser.add_argument(
        '--symbol', default='EURUSD',
        choices=['EURUSD', 'GBPUSD', 'USDJPY']
    )
    parser.add_argument(
        '--paper', action='store_true',
        help='Paper trading mode (no real orders)'
    )
    parser.add_argument('--login',    type=int)
    parser.add_argument('--password', type=str)
    parser.add_argument('--server',   type=str)
    args = parser.parse_args()

    # Connect to MT5
    if MT5_AVAILABLE:
        connected = connect_mt5(
            login=args.login,
            password=args.password,
            server=args.server,
        )
        if not connected:
            log.error("MT5_CONNECTION_FAILED: Could not connect to MT5. Make sure MT5 is open and logged in.")
            sys.exit(1)

    # HARD SAFETY GATE: paper_mode is enforced to True by default to prevent accidental
    # live order execution unless explicitly set to False via code modification after verification.
    paper_mode_enforced = True if not hasattr(args, 'live') else not args.live

    engine = Engine(
        symbol=args.symbol,
        paper_mode=paper_mode_enforced,
        magic=123456,
    )

    try:
        engine.start()
    except KeyboardInterrupt:
        print("\nEngine stopped.")
