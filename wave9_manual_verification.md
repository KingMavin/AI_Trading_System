# WAVE 9 — Manual Verification Protocol

This document outlines manual testing instructions for running overnight multi-symbol Trainer runs with checkpointing, resource watchdog monitoring, per-symbol failure isolation, and process locking.

---

## Overnight Multi-Symbol Training Run Procedure

### Objective
Execute a multi-symbol training run overnight while verifying that state checkpointing, resource watchdog monitoring, per-symbol failure isolation, and process locking operate reliably.

### Step-by-Step Instructions

1. **Verify MT5 Connection & Active Specs:**
   - Ensure MT5 terminal is connected during market hours or that `D:\work\files\state\instrument_specs.json` contains valid live-captured specs (`sanity_ok = True`).

2. **Launch Overnight Trainer Run:**
   - Execute the Trainer CLI (runs across all configured symbols: `EURUSD`, `GBPUSD`, `USDJPY`):
     ```powershell
     python trainer/trainer.py run
     ```
   - *Note:* To run on a specific single symbol, use `python trainer/trainer.py run --symbol EURUSD`.

3. **Verify Concurrency Lock:**
   - While the run is active, attempt to launch a second instance in a separate terminal:
     ```powershell
     python trainer/trainer.py run --symbol EURUSD
     ```
   - Confirm the second instance fails immediately with:
     `TRAINER_CONCURRENCY_LOCK_ACTIVE: Another Trainer instance (PID ...) is currently running.`

4. **Verify Checkpointing & Resume:**
   - To test mid-run crash recovery, terminate the active Trainer process (`Ctrl+C` or kill process) after 10+ minutes.
   - Confirm `D:\work\files\state\trainer_checkpoint.json` exists and contains JSON state with `completed_candidates` and `completed_windows`.
   - Re-launch the Trainer command:
     ```powershell
     python trainer/trainer.py run
     ```
   - Confirm log displays:
     `Loaded valid checkpoint for run ... from D:\work\files\state\trainer_checkpoint.json`
   - Observe that completed candidates/windows are skipped, and training resumes from the exact boundary of the crash.

5. **Verify Resource Watchdog Thresholds:**
   - Resource Watchdog active thresholds (16 GB baseline hardware):
     - **GREEN** (< 80% RAM, < 85% Disk): Normal execution.
     - **AMBER** (80%–90% RAM, 85%–92% Disk): Log warning `RESOURCE_WATCHDOG_AMBER`, continue.
     - **RED** (90%–96% RAM, 92%–97% Disk): Log warning `RESOURCE_WATCHDOG_RED`, pause new candidate starts, finish in-flight work.
     - **CRITICAL** ($\ge$ 96% RAM, $\ge$ 97% Disk): Log error `RESOURCE_WATCHDOG_CRITICAL`, execute immediate atomic checkpoint to `D:\work\files\state\trainer_checkpoint.json`, and halt cleanly with `RuntimeError`.

6. **Clean Completion Verification:**
   - Upon successful completion of all symbols, verify `trainer_checkpoint.json` is cleared and `trainer.lock` at `D:\work\files\state\trainer.lock` is released.
