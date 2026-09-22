# WAVE 11 — Manual Verification Protocol

This document outlines manual testing procedures for verifying Coherence Validation pre-checks and Prop-Rule-Aware Mandate Gating during strategy training.

---

## 1. Verify Coherence Validator Pre-Checks

### Objective
Confirm that degenerate candidate parameter configurations (e.g. `fast_ma_period >= slow_ma_period`, `sl_atr_multiple <= 0`, or SL/TP below `stops_level`) are rejected before walk-forward optimization runs, avoiding wasted backtesting compute.

### Instructions
1. Run a quick training pass on `EURUSD`:
   ```powershell
   python trainer/trainer.py run --symbol EURUSD --quick
   ```
2. Inspect terminal output or `D:\work\files\logs\trainer_YYYYMMDD.log`.
3. Verify log displays warning messages for degenerate parameter grid entries:
   `COHERENCE_REJECTED [EURUSD]: DEGENERATE_CONFIG: Fast MA period (50) must be strictly less than Slow MA period (20)`
4. Confirm backtesting execution (`_run_one`) was skipped for all rejected combinations.

---

## 2. Verify Prop-Rule-Aware Mandate Gating (Gate 9)

### Objective
Verify that candidate strategies whose backtest equity curve breaches `max_daily_loss_pct` or `max_overall_drawdown_pct` are hard-rejected at Gate 9 during candidate ranking and promotion evaluation.

### Instructions
1. Confirm `D:\work\files\config\mandate.json` exists with active prop-firm limits (e.g. `max_daily_loss_pct: 5.0`, `max_overall_drawdown_pct: 10.0`).
2. Run a full strategy training pass:
   ```powershell
   python trainer/trainer.py run
   ```
3. Inspect `D:\work\files\decisions\decision_log.jsonl`.
4. Confirm any candidate exhibiting backtest daily drawdown breaches is logged with:
   `failed_gate: "gate_9_mandate_compliance"`
   `failure_reason: "Mandate compliance breach: MANDATE_DAILY_LOSS_BREACH..."`
5. Verify zero non-compliant candidate strategies receive composite score promotion.
