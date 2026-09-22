# WAVE 10 — Manual Verification Protocol

This document outlines manual testing procedures for central log aggregation and multi-day continuous performance observation.

---

## Central Log Aggregation & Performance Log Verification

### Objective
Verify that `LogAggregator` queries all system log/state sources while retaining exact `source_file` absolute paths, and confirm `D:\work\files\logs\performance.jsonl` continuously records resource usage and Engine timing metrics without performance degradation.

---

## Step-by-Step Execution Protocol

1. **Verify Log Aggregator CLI Querying:**
   - Execute the Log Aggregator CLI to query all system logs from the past 24 hours:
     ```powershell
     python -m shared.log_aggregator --since-hours 24
     ```
   - Filter for errors:
     ```powershell
     python -m shared.log_aggregator --level ERROR
     ```
   - Verify every returned entry displays `Source File: D:\work\files\...` with absolute path.

2. **Verify Performance Log Generation:**
   - Run the Trainer or Engine for several candle cycles:
     ```powershell
     python trainer/trainer.py run --symbol EURUSD --quick
     ```
   - Check `D:\work\files\logs\performance.jsonl`.
   - Confirm records contain structured JSON lines for both `TRAINER` (`metric_type: RESOURCE_USAGE`) and `ENGINE` (`metric_type: CANDLE_TIMING`).

3. **Verify Log Rotation & Active Write Safety:**
   - Run a query while the Engine or Trainer is actively writing logs.
   - Confirm no file access crashes, locks, or IO errors occur.
