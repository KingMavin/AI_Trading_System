# ATS Manual Verification Master Run Sheet

> [!IMPORTANT]
> This master run sheet consolidates all manual verification procedures for WAVE 2 through WAVE 21 in run order.
>
> **Usage:** Copy and paste the command blocks directly into your PowerShell or Command Prompt terminal in `d:\work\sy\ats`. Every code block includes directory navigation (`cd d:\work\sy\ats`) and virtual environment activation (`.\venv\Scripts\Activate.ps1`) so you can execute them without setup errors.

---

## 💡 Quick Legend: Online vs. Offline Verification

- 🟢 **`[OFFLINE]`**: Does **NOT** require MetaTrader 5 (MT5) terminal. Runs standalone using historical Parquet data, local strategy specs, or python test scripts.
- 🔴 **`[ONLINE]`**: **REQUIRES** live MetaTrader 5 (MT5) terminal open, logged into your trading account, and connected to the broker server.

---

## ⚡ Global Environment Setup (Run First if opening a new terminal)

```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
```

*Note: Alternatively, you can use `.\venv\Scripts\python.exe` directly in any command without needing script activation permissions.*

---

## 0. Baseline Automated Test Suite Check
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Runs the full project automated unit test suite (376 tests) to verify that all money-math, engine, trainer, DNA hash, and checklist modules are 100% passing.
- **Estimated Time:** ~5–7 minutes.

### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
pytest tests/ --tb=short
```

### Pass / Fail Criteria:
- **Pass:** Terminal output concludes with `376 passed`. Zero failures or errors.

---

## WAVE 2 — Money Math, Swap Rollover & Instrument Spec Pipeline

### Overview & Purpose
Validates runtime swap cost calculations (`_calculate_swap_cost`), queries MT5 directly to confirm broker 3-day rollover properties, and executes a strategy walk-forward validation run.

---

### Step 1: Verify Overnight SELL Swap Cost Calculation
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Verifies that the backtester accurately calculates overnight swap charges for SELL positions using naive UTC timestamps.
- **Estimated Time:** ~5 seconds.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -c "from trainer.core.backtester import Backtester; from datetime import datetime, timezone; bt = Backtester('EURUSD', params={}, allow_simulation_defaults=True); print('SELL Swap Cost:', bt._calculate_swap_cost('SELL', 1.0, datetime(2026,6,1,10,0,tzinfo=timezone.utc), datetime(2026,6,2,10,0,tzinfo=timezone.utc)))"
```

#### Pass / Fail Criteria:
- **Pass:** Returns a float representing SELL swap cost without raising an exception.

---

### Step 2: Verify MT5 Broker 3-Day Rollover Property
- **Mode:** 🔴 `[ONLINE]` *(Requires MT5 Terminal Open & Connected)*
- **Simple Explanation:** Connects directly to MT5 terminal to verify the broker's 3-day swap multiplier day (Wednesday 3x swap).
- **Estimated Time:** ~5 seconds.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -c "import MetaTrader5 as mt5; mt5.initialize(); print('EURUSD Rollover 3-Day:', mt5.symbol_info('EURUSD').swap_rollover3days); mt5.shutdown()"
```

#### Pass / Fail Criteria:
- **Pass:** Returns `3` (Wednesday 3x swap multiplier) matching the MT5 broker specification.

---

### Step 3: Execute Strategy Walk-Forward Validation Run
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Runs a quick walk-forward optimization pass on EURUSD to verify lot sizing, money-math metrics, and walk-forward execution.
- **Estimated Time:** ~1–2 minutes.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -m trainer.trainer run --quick --symbol EURUSD
```

#### Pass / Fail Criteria:
- **Pass:** Training run completes cleanly with `outcome: COMPLETED`.

---

## WAVE 4 — Real-Data Fill Timing Verification (Fix 4.2)

### Overview & Purpose
Verifies that backtest trade entries execute at the **next-candle open** (`base_price = row['open']`) rather than the signal candle's close price across real market data and price gaps.

---

### Step 1: Execute Quick Backtest & Inspect Trade Entry Logs
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Runs a backtest on EURUSD M15 historical data and prints detailed trade entry timestamps and fill prices to spot-check fill timing.
- **Estimated Time:** ~10 seconds.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -c "from trainer.core.backtester import Backtester; from trainer.core.data_loader import load_ohlcv; from datetime import datetime; import trainer.signals.ma_crossover as ma; df = load_ohlcv('EURUSD', 'M15', datetime(2020,1,1), datetime(2022,12,31)); bt = Backtester('EURUSD', params={'fast_ma_period': 10, 'slow_ma_period': 50, 'sl_atr_multiple': 1.5, 'tp_rr_ratio': 2.0}, allow_simulation_defaults=True); res = bt.run(df, ma); import json; print('Total trades:', len(res['trades'])); print(json.dumps(res['trades'][:15], indent=2, default=str))"
```

#### Pass / Fail Criteria:
- **Next-Candle Open Verification:** For each trade, `trade['entry_time']` corresponds to the candle open following signal generation at candle $i-1$ close. `trade['entry_price']` matches the open price of the candle at `entry_time` (adjusted for spread/slippage) in raw M15 historical data.
- **No Same-Candle Close Reappearance:** Zero trades exhibit an `entry_price` identical to the signal candle's `close` price with zero time offset.
- **Price Gap Fill Verification:** Trades near market price gaps (e.g. Sunday market open) fill at the actual Sunday next-candle open (including gap distance), not the pre-gap Friday close.

---

## WAVE 7 — Live Broker Contract Spec & Spread Capture

### Overview & Purpose
Queries live MT5 quotes for all 5 universe instruments (`EURUSD`, `GBPUSD`, `USDJPY`, `USDCAD`, `XAUUSD`) during active market hours to capture contract specs, tick values, pip values, and live spreads.

---

### Step 1: Capture Live Instrument Specs
- **Mode:** 🔴 `[ONLINE]` *(Requires MT5 Terminal Open & Connected during active market hours)*
- **Simple Explanation:** Connects to MT5 terminal and writes validated contract specs to `D:\work\files\state\instrument_specs.json`.
- **Estimated Time:** ~10 seconds.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python capture_specs.py
```

#### Pass / Fail Criteria:
- **Output Files:** Specs written to `D:\work\files\state\instrument_specs.json` and raw Gold info to `scratch\xauusd_raw_symbol_info.json`.
- **Sanity Verification:** All 5 universe instruments (`EURUSD`, `GBPUSD`, `USDJPY`, `USDCAD`, `XAUUSD`) output `sanity_ok = True` ("OK"). `XAUUSD` tick_value of `0.10` falls cleanly within `SANITY_RANGES["XAUUSD"]` (`[0.05, 0.20]`).

---

## WAVE 8 — Live Order Robustness & Connection Resilience

### Overview & Purpose
Tests the Engine's connection watchdog, heartbeat retries, exponential backoff, and automatic `SAFE_MODE` entry-blocking during network disconnections.

---

### Step 1: Launch Engine in Paper Trading Mode
- **Mode:** 🔴 `[ONLINE]` *(Requires MT5 Terminal Open & Connected initially)*
- **Simple Explanation:** Starts the Engine in paper trading mode. You can test disconnection by manually turning off Wi-Fi or disconnecting Ethernet while it runs.
- **Estimated Time:** ~15 minutes.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -m engine.core.engine --symbol EURUSD --paper
```

#### Test Steps & Pass / Fail Criteria:
1. Launch command above and confirm log displays `ATS ENGINE STARTING`.
2. Disconnect Wi-Fi / Ethernet adapter.
3. Verify log records `CONNECTION_HEALTH_FAILURE (#1)` and retries at 5s, 10s, 20s backoff intervals.
4. After 3 failed reconnect attempts, verify log outputs `MAX_RECONNECT_ATTEMPTS_EXCEEDED (3/3). Entering SAFE_MODE: Blocking all new trade entries.`.
5. Re-enable Wi-Fi / Ethernet. Verify log displays `MT5 Connection RESTORED`, `SAFE_MODE` clears, and candle processing resumes.

---

## WAVE 9 — Trainer Long-Run Robustness & Overnight Execution

### Overview & Purpose
Executes multi-symbol training runs with atomic state checkpointing, resource watchdog monitoring, per-symbol failure isolation, and process locking.

---

### Step 1: Launch Multi-Symbol Overnight Trainer Run
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Runs the multi-symbol trainer across configured instruments (`EURUSD`, `GBPUSD`, `USDJPY`). Can run overnight.
- **Estimated Time:** ~4–8 hours (unattended) or run with `--quick` for a fast 2-minute test.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -m trainer.trainer run
```

---

### Step 2: Verify Concurrency Lock (Run in a 2nd Terminal Window)
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Attempts to start a second trainer instance while the first is running to verify that process locking prevents duplicate training runs.
- **Estimated Time:** ~5 seconds.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -m trainer.trainer run --symbol EURUSD
```

#### Pass / Fail Criteria:
- **Concurrency Lock:** Second instance fails immediately with `TRAINER_CONCURRENCY_LOCK_ACTIVE: Another Trainer instance (PID ...) is currently running.`.
- **Checkpoint Resume:** On crash/restart, log displays `Loaded valid checkpoint for run ...`. Training resumes from the exact boundary of the crash.
- **Clean Completion:** Upon completion, `trainer_checkpoint.json` is cleared and `trainer.lock` at `D:\work\files\state\trainer.lock` is released.

---

## WAVE 10 — Central Log Aggregation & Performance Instrumentation

### Overview & Purpose
Verifies log aggregation across all engine and trainer log files and confirms continuous performance metric logging (`performance.jsonl`).

---

### Step 1: Query Aggregated Log Records
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Queries system log files across the past 24 hours and displays ERROR-level events with absolute file paths.
- **Estimated Time:** ~5 seconds.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -m shared.log_aggregator --since-hours 24
python -m shared.log_aggregator --level ERROR
```

---

### Step 2: Verify Performance Metric Logging
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Runs a quick training pass and verifies that CPU/RAM resource metrics and candle timing are recorded in `performance.jsonl`.
- **Estimated Time:** ~1 minute.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -m trainer.trainer run --symbol EURUSD --quick
```

#### Pass / Fail Criteria:
- Returned log entries display absolute file paths (`Source File: D:\work\files\...`).
- `D:\work\files\logs\performance.jsonl` contains structured JSON records for resource usage and timing.

---

## WAVE 11 — Coherence Validator & Prop-Rule-Aware Mandate Gating

### Overview & Purpose
Verifies that degenerate parameter configurations are rejected prior to walk-forward optimization, and candidate strategies breaching daily loss or drawdown limits are rejected at Gate 9.

---

### Step 1: Execute Training Pass with Coherence Pre-Checks & Gate 9
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Runs trainer to verify that invalid parameter grids are rejected early and non-compliant drawdown strategies are blocked at Gate 9.
- **Estimated Time:** ~1–2 minutes.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -m trainer.trainer run --symbol EURUSD --quick
```

#### Pass / Fail Criteria:
- Log displays `COHERENCE_REJECTED` warning messages for degenerate parameter grid entries.
- Candidates breaching drawdown limits log `failed_gate: "gate_9_mandate_compliance"` in `decision_log.jsonl`. Zero non-compliant strategies receive promotion.

---

## WAVE 12 — Data Foundation & Multi-Symbol Pipeline Verification

### Overview & Purpose
Validates Parquet data completeness and walk-forward training across non-EURUSD universe instruments (`XAUUSD`, `USDJPY`, `USDCAD`).

---

### Step 1: Run Multi-Symbol Quick Verification
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Tests historical training individually on Gold (`XAUUSD`), Yen (`USDJPY`), and Canadian Dollar (`USDCAD`).
- **Estimated Time:** ~2 minutes per symbol.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -m trainer.trainer run --quick --symbol XAUUSD
python -m trainer.trainer run --quick --symbol USDJPY
python -m trainer.trainer run --quick --symbol USDCAD
```

#### Pass / Fail Criteria:
- All 3 non-EURUSD instruments execute without missing historical data or `INSTRUMENT_SPEC_MISSING` errors.
- Pip size calculations resolve correctly (`pip_size = 0.01` for `XAUUSD`/`USDJPY`, `pip_size = 0.0001` for `USDCAD`).

---

## WAVE 13 — Production Full-Pipeline Integration (Capstone Run)

### Overview & Purpose
Executes a multi-symbol production training pass across all 5 active trading instruments (`EURUSD`, `GBPUSD`, `USDJPY`, `USDCAD`, `XAUUSD`) over complete historical Parquet datasets with full HTML/Markdown report generation.

---

### Step 1: Launch Full Production Capstone Training Pass
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Runs the complete production training pipeline across all 5 universe instruments, generating full strategy ranking reports.
- **Estimated Time:** ~15–30 minutes (depending on CPU).

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -m trainer.cli --symbols EURUSD,GBPUSD,USDJPY,USDCAD,XAUUSD --timeframe M15 --data-start 2025-01-01 --data-end 2026-05-19 --opt-months 6 --test-months 2 --top-n 3
```

#### Pass / Fail Criteria:
- Terminal output concludes with `TRAINING RUN COMPLETE` (`Outcome: COMPLETED`).
- Persistent HTML & Markdown reports generated in `D:\work\files\reports\` (`report_run_*.md` and `report_run_*.html`).

---

## WAVE 19–21 — Canonical Strategy DNA Hash Utility

### Overview & Purpose
Verifies the SHA-256 canonical DNA hash utility (`rehash_strategy.py`). Enforces explicit confirmation before updating strategy files and emits `STRATEGY_REHASHED` audit log events.

---

### Step 1: Test Unconfirmed / Prompt Mode (Interactive Prompt)
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Runs rehash without `--confirm` flag to verify that an interactive prompt appears and prevents accidental file modification.
- **Estimated Time:** ~5 seconds.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python rehash_strategy.py engine/strategy/active_strategy.json
```

---

### Step 2: Test Confirmed Rehash Execution (`--confirm` Flag)
- **Mode:** 🟢 `[OFFLINE]`
- **Simple Explanation:** Recomputes canonical DNA hash and updates `active_strategy.json` automatically while logging an audit event.
- **Estimated Time:** ~5 seconds.

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python rehash_strategy.py engine/strategy/active_strategy.json --confirm
```

#### Pass / Fail Criteria:
- Interactive mode requires typing `yes` to update.
- Confirmed execution updates `"dna_hash"` in `active_strategy.json` and writes a `STRATEGY_REHASHED` record to `D:\work\files\logs\audit_ledger.jsonl`.

---

## WAVE 21 — Engine Dashboard Web Application & 9-Item Pre-Trade Checklist

### Overview & Purpose
Launches the Flask web dashboard server to monitor real-time engine metrics, Panel 1 Strategy DNA badge (`DNA: 7d1dc3c7`), and Panel 6 9-Item Pre-Trade Checklist verification grid.

---

### Step 1: Launch Web Dashboard Application Server
- **Mode:** 🟢 `[OFFLINE]` (Runs offline or live)
- **Simple Explanation:** Starts the web server on port 5000 to display live system state, strategy DNA badge, and pre-trade checklist grid in your web browser.
- **Estimated Time:** Runs continuously until stopped (`Ctrl+C`).

#### Copy-Pasteable Command:
```powershell
cd d:\work\sy\ats
.\venv\Scripts\Activate.ps1
python -m engine.dashboard.app
```

#### Access Web Dashboard UI:
Open your browser and navigate to: `http://localhost:5000`

#### Pass / Fail Criteria:
- Server launches cleanly displaying `Running on http://127.0.0.1:5000`.
- Browser shows Panel 1 with Strategy DNA badge (`DNA: 7d1dc3c7`).
- Browser shows Panel 6 with interactive 9-item pre-trade checklist grid (all green checkmarks when system is healthy).
