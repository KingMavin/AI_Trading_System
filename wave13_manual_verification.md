# WAVE 13 Manual Verification Protocol — Production Full-Pipeline Training Run

## Description & Purpose
This run sheet executes a full-scale, multi-symbol production training pass across all 5 active trading instruments (`EURUSD`, `GBPUSD`, `USDJPY`, `USDCAD`, `XAUUSD`) over complete historical Parquet datasets. It exercises data loading, parameter grid search, walk-forward validation, prop-mandate evaluation, gate checking, scoring, and report generation end-to-end.

Estimated time: **Hands-off, ~1.5 to 2.5 hours**.

---

## 1. Full Production Training Command

Execute from project root in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python -m trainer.cli --symbols EURUSD,GBPUSD,USDJPY,USDCAD,XAUUSD --timeframe M15 --data-start 2025-01-01 --data-end 2026-05-19 --opt-months 6 --test-months 2 --top-n 3
```

---

## 2. Pass / Fail Verification Checklist

Check the following output artifacts upon completion:

1. **Terminal Console Header & Summary:**
   - Confirm output concludes with `TRAINING RUN COMPLETE`.
   - Confirm `Outcome: COMPLETED` or `PARTIAL_FAILURE` (0 unhandled exceptions).
2. **Persistent Training Reports:**
   - Check `D:\work\files\reports\` for `report_run_*.md` and `report_run_*.html`.
   - Verify `Pipeline Outcome:` in the Markdown report header displays `COMPLETED` (or `PARTIAL_FAILURE`), matching terminal summary.
3. **Knowledge Base Memory Update:**
   - Check `trainer/trainer_data/history/decision_log.jsonl`.
   - Confirm new decision records appended with 9-gate details present for all evaluated candidates.
