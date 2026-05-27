"""
Live Logger — Engine component.

Writes structured trade logs that the Trainer reads
on the next training run to supplement historical data
with actual live performance.

Log format: JSON-Lines (.jsonl)
One line per closed trade.
Location: engine/logs/candle_{YYYYMMDD}.jsonl

The Trainer's Layer 2 (Data Ingestion) reads these files
and merges them with historical data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import gzip
import logging
from datetime import datetime, timezone, date
from typing import Dict, Optional
from dataclasses import dataclass, asdict

log = logging.getLogger(__name__)

# Log directory
DEFAULT_LOG_DIR = Path(__file__).parent.parent / 'logs'

# Schema version — increment when log format changes
SCHEMA_VERSION = '1.0'


@dataclass
class TradeLogRecord:
    """
    Complete record of one closed trade.
    Written to log file. Read by Trainer.
    """
    schema_version:     str

    # Identity
    trade_id:           str
    strategy_id:        str
    run_timestamp:      str     # engine startup time
    candle_time:        str     # candle that triggered close

    # Trade details
    symbol:             str
    timeframe:          str
    direction:          str     # BUY / SELL
    entry_price:        float
    entry_time:         str
    exit_price:         float
    exit_time:          str
    lots:               float

    # P&L
    gross_pnl:          float
    net_pnl:            float
    commission:         float
    spread_cost:        float

    # Close reason
    close_reason:       str     # SL_HIT / TP_HIT / EXIT_SIGNAL
                                # / MANUAL / END_OF_SESSION

    # Context at entry
    entry_session:      str
    entry_regime:       str
    entry_atr:          float
    entry_adx:          float
    entry_spread_pips:  float

    # Strategy parameters at time of trade
    fast_ma_period:     int
    slow_ma_period:     int
    sl_atr_multiple:    float
    tp_rr_ratio:        float

    # Duration
    duration_candles:   int
    duration_minutes:   float

    # Account state at close
    equity_after:       float
    balance_after:      float
    drawdown_pct:       float


class LiveLogger:
    """
    Writes trade logs for the Trainer to consume.
    Handles daily file rotation and compression.
    """

    def __init__(self, log_dir: str = None,
                 strategy_id: str = 'unknown',
                 run_timestamp: str = None):
        self.log_dir       = Path(log_dir) \
                             if log_dir \
                             else DEFAULT_LOG_DIR
        self.strategy_id   = strategy_id
        self.run_timestamp = run_timestamp or \
            datetime.now(timezone.utc).strftime(
                '%Y%m%d_%H%M%S'
            )
        self.current_date  = date.today()
        self.current_file  = None
        self.records_written = 0

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._open_file()

    def _get_filename(self, log_date: date = None) -> Path:
        """Get log filename for a given date."""
        d = log_date or date.today()
        return self.log_dir / f"candle_{d.strftime('%Y%m%d')}.jsonl"

    def _open_file(self) -> None:
        """Open today's log file in append mode."""
        self.current_file = self._get_filename()
        log.debug(f"Log file: {self.current_file}")

    def _rotate_if_needed(self) -> None:
        """Rotate to new file if date has changed."""
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self._open_file()

    def log_trade(self, trade: TradeLogRecord) -> None:
        """
        Write one trade record to the log file.
        Append-only. Each record is one JSON line.
        """
        self._rotate_if_needed()

        record_dict = asdict(trade)
        line        = json.dumps(record_dict) + '\n'

        with open(self.current_file, 'a') as f:
            f.write(line)

        self.records_written += 1
        log.debug(
            f"Logged trade {trade.trade_id}: "
            f"{trade.direction} {trade.symbol} "
            f"net_pnl={trade.net_pnl:.2f}"
        )

    def log_candle_event(self, event: Dict) -> None:
        """
        Log a non-trade event (session change, regime change,
        checklist skip, etc.) for Trainer analysis.
        """
        self._rotate_if_needed()
        event['schema_version'] = SCHEMA_VERSION
        event['event_time']     = datetime.now(
            timezone.utc
        ).isoformat()
        event['record_type']    = 'EVENT'

        with open(self.current_file, 'a') as f:
            f.write(json.dumps(event) + '\n')

    def compress_old_logs(self, days_old: int = 7) -> int:
        """
        Compress log files older than N days.
        Returns count of files compressed.
        """
        compressed = 0
        cutoff     = date.today()

        for log_file in sorted(self.log_dir.glob('*.jsonl')):
            # Parse date from filename
            try:
                file_date = datetime.strptime(
                    log_file.stem.replace('candle_', ''),
                    '%Y%m%d'
                ).date()
            except ValueError:
                continue

            age_days = (cutoff - file_date).days
            if age_days >= days_old:
                gz_path = log_file.with_suffix('.jsonl.gz')
                with open(log_file, 'rb') as f_in:
                    with gzip.open(gz_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                log_file.unlink()
                compressed += 1
                log.info(
                    f"Compressed: {log_file.name} → "
                    f"{gz_path.name}"
                )

        return compressed

    def get_stats(self) -> Dict:
        """Return logger statistics."""
        log_files = list(self.log_dir.glob('*.jsonl'))
        gz_files  = list(self.log_dir.glob('*.jsonl.gz'))

        return {
            'log_dir':        str(self.log_dir),
            'strategy_id':    self.strategy_id,
            'current_file':   str(self.current_file),
            'records_written':self.records_written,
            'log_files':      len(log_files),
            'gz_files':       len(gz_files),
        }


# ── TRAINER MANIFEST ────────────────────────────────────

class TrainerManifest:
    """
    Tracks which log files have been processed by the Trainer.
    Prevents re-processing the same log files on every run.
    Written by Engine. Read by Trainer Layer 2.
    """

    def __init__(self, manifest_path: str = None):
        self.path = Path(manifest_path) \
                    if manifest_path \
                    else (
                        DEFAULT_LOG_DIR /
                        'trainer_manifest.json'
                    )
        self.data = self._load()

    def _load(self) -> Dict:
        if not self.path.exists():
            return {
                'last_processed_file': None,
                'last_processed_at':   None,
                'processed_files':     [],
                'total_trades_logged': 0,
            }
        try:
            with open(self.path) as f:
                return json.load(f)
        except Exception:
            return {
                'last_processed_file': None,
                'last_processed_at':   None,
                'processed_files':     [],
                'total_trades_logged': 0,
            }

    def mark_processed(self, filename: str,
                        trade_count: int) -> None:
        """Mark a log file as processed by the Trainer."""
        if filename not in self.data['processed_files']:
            self.data['processed_files'].append(filename)
        self.data['last_processed_file'] = filename
        self.data['last_processed_at']   = datetime.now(
            timezone.utc
        ).isoformat()
        self.data['total_trades_logged'] += trade_count
        self._save()

    def is_processed(self, filename: str) -> bool:
        """Return True if this file has been processed."""
        return filename in self.data['processed_files']

    def _save(self) -> None:
        tmp = self.path.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.data, f, indent=2)
        if self.path.exists():
            self.path.unlink()
        tmp.rename(self.path)