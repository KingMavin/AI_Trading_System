"""
ATS Engine — Mandate Schema, Configuration, and Audit Ledger (WAVE 6)

Captures prop-firm compliance constraints and provides append-only audit logging.
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timezone

log = logging.getLogger(__name__)

ENGINE_ROOT     = Path(__file__).parent.parent
MANDATE_FILE    = ENGINE_ROOT / 'config' / 'mandate.json'
AUDIT_LOG_FILE  = ENGINE_ROOT / 'logs' / 'audit_ledger.jsonl'


@dataclass
class Mandate:
    mandate_id:               str
    firm_name:                str
    account_id:               str
    max_daily_loss_pct:       float
    max_overall_drawdown_pct: float
    drawdown_type:            str  # "static" or "trailing"
    min_trading_days:         int
    # Per-trade risk sizing cap enforced at pre-trade mandate gate. Ensures no single order
    # risks more than max_daily_risk_pct of account balance. (Note: System operates one-position-at-a-time).
    max_daily_risk_pct:       float
    symbol_universe:          List[str]
    prohibited_patterns:      List[str] = field(default_factory=list)


def load_mandate(config_path: Optional[Path] = None) -> Mandate:
    """
    Load mandate configuration from JSON file.
    Fails loudly with RuntimeError if missing, corrupted, or invalid (Fix 6.1).
    """
    path = config_path or MANDATE_FILE

    if not path.exists():
        log.error(f"MANDATE_LOAD_FAILED: Configuration file {path} does not exist.")
        raise RuntimeError(
            f"Mandate configuration file {path} does not exist. "
            "Engine startup halted to prevent trading without compliance rules."
        )

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        log.error(f"MANDATE_LOAD_FAILED: Failed to parse {path}: {e}")
        raise RuntimeError(
            f"Mandate configuration file {path} is corrupted or unparseable. "
            f"Error: {e}. Engine startup halted."
        )

    required_fields = [
        'mandate_id', 'firm_name', 'account_id',
        'max_daily_loss_pct', 'max_overall_drawdown_pct', 'drawdown_type',
        'min_trading_days', 'max_daily_risk_pct', 'symbol_universe',
        'prohibited_patterns'
    ]

    missing = [f for f in required_fields if f not in data]
    if missing:
        log.error(f"MANDATE_LOAD_FAILED: Missing required fields in {path}: {missing}")
        raise RuntimeError(
            f"Mandate configuration file {path} missing required fields: {missing}. "
            "Engine startup halted."
        )

    # Validate drawdown_type
    drawdown_type = str(data['drawdown_type']).lower()
    if drawdown_type not in ('static', 'trailing'):
        log.error(f"MANDATE_LOAD_FAILED: Invalid drawdown_type '{drawdown_type}'. Must be 'static' or 'trailing'.")
        raise RuntimeError(
            f"Mandate configuration error: drawdown_type '{drawdown_type}' invalid. "
            "Must be 'static' or 'trailing'."
        )

    # Validate numeric bounds
    try:
        max_daily_loss_pct       = float(data['max_daily_loss_pct'])
        max_overall_drawdown_pct = float(data['max_overall_drawdown_pct'])
        max_daily_risk_pct       = float(data['max_daily_risk_pct'])
        min_trading_days         = int(data['min_trading_days'])
    except (ValueError, TypeError) as e:
        log.error(f"MANDATE_LOAD_FAILED: Invalid numeric field in {path}: {e}")
        raise RuntimeError(f"Mandate configuration invalid numeric types in {path}: {e}")

    if max_daily_loss_pct <= 0 or max_overall_drawdown_pct <= 0 or max_daily_risk_pct <= 0:
        log.error("MANDATE_LOAD_FAILED: Percentage thresholds must be strictly positive.")
        raise RuntimeError("Mandate configuration error: percentage thresholds must be > 0.")

    symbol_universe     = [str(s).strip().upper() for s in data['symbol_universe']]
    prohibited_patterns = [str(p) for p in data['prohibited_patterns']]

    mandate = Mandate(
        mandate_id               = str(data['mandate_id']),
        firm_name                = str(data['firm_name']),
        account_id               = str(data['account_id']),
        max_daily_loss_pct       = max_daily_loss_pct,
        max_overall_drawdown_pct = max_overall_drawdown_pct,
        drawdown_type            = drawdown_type,
        min_trading_days         = min_trading_days,
        max_daily_risk_pct       = max_daily_risk_pct,
        symbol_universe          = symbol_universe,
        prohibited_patterns      = prohibited_patterns,
    )

    log.info(
        f"MANDATE_LOADED: ID={mandate.mandate_id} | Firm={mandate.firm_name} | "
        f"MaxDailyLoss={mandate.max_daily_loss_pct}% | MaxOverallDD={mandate.max_overall_drawdown_pct}% "
        f"({mandate.drawdown_type}) | Universe={mandate.symbol_universe}"
    )

    return mandate


def log_audit_event(event_type: str, mandate_id: str, details: Dict, log_file: Optional[Path] = None) -> None:
    """
    Append durable compliance record to audit ledger (Fix 6.4).
    Uses fsync for non-volatile storage.
    """
    target_file = log_file or AUDIT_LOG_FILE
    target_file.parent.mkdir(parents=True, exist_ok=True)

    record = {
        'timestamp':  datetime.now(timezone.utc).isoformat(),
        'event_type': event_type,
        'mandate_id': mandate_id,
        'details':    details,
    }

    line = json.dumps(record) + '\n'
    with open(target_file, 'a', encoding='utf-8') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
