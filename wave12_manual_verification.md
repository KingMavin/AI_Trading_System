# WAVE 12 — Manual Verification Protocol & Data Audit Sheet

This document outlines manual verification procedures and data foundation inventory for WAVE 12.

---

## 1. Verified Historical Parquet Datasets Inventory

All 5 instruments in the system universe have complete, valid backtestable Parquet datasets stored in `trainer/trainer_data/historical/`.

| Instrument | Timeframe | Candle Count | Start Timestamp | End Timestamp | File Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EURUSD** | M15 | 681,909 | 1985-05-19 21:00:00+00:00 | 2026-05-19 05:30:00+00:00 | 12.58 MB |
| **EURUSD** | H1  | 173,445 | 1985-05-19 21:00:00+00:00 | 2026-05-19 05:00:00+00:00 | 4.42 MB |
| **EURUSD** | H4  | 47,369  | 1985-05-19 20:00:00+00:00 | 2026-05-19 04:00:00+00:00 | 1.64 MB |
| **GBPUSD** | M15 | 529,279 | 2005-01-02 22:00:00+00:00 | 2026-05-19 20:45:00+00:00 | 10.66 MB |
| **GBPUSD** | H1  | 132,487 | 2005-01-02 22:00:00+00:00 | 2026-05-19 20:00:00+00:00 | 3.95 MB |
| **GBPUSD** | H4  | 34,163  | 2005-01-02 20:00:00+00:00 | 2026-05-19 20:00:00+00:00 | 1.47 MB |
| **USDJPY** | M15 | 257,903 | 2016-01-03 22:00:00+00:00 | 2026-05-19 21:15:00+00:00 | 5.43 MB |
| **USDJPY** | H1  | 64,500  | 2016-01-03 22:00:00+00:00 | 2026-05-19 21:00:00+00:00 | 1.85 MB |
| **USDJPY** | H4  | 16,686  | 2016-01-03 20:00:00+00:00 | 2026-05-19 20:00:00+00:00 | 0.64 MB |
| **USDCAD** | M15 | 680,213 | 1993-04-27 21:00:00+00:00 | 2026-07-09 18:45:00+00:00 | 12.38 MB |
| **USDCAD** | H1  | 172,198 | 1993-04-27 21:00:00+00:00 | 2026-07-09 18:00:00+00:00 | 4.32 MB |
| **USDCAD** | H4  | 45,537  | 1993-04-27 20:00:00+00:00 | 2026-07-09 16:00:00+00:00 | 1.64 MB |
| **XAUUSD** | M15 | 506,985 | 2004-06-11 04:15:00+00:00 | 2026-07-09 18:45:00+00:00 | 11.02 MB |
| **XAUUSD** | H1  | 128,408 | 2004-06-11 04:00:00+00:00 | 2026-07-09 18:00:00+00:00 | 3.77 MB |
| **XAUUSD** | H4  | 34,759  | 2004-06-11 04:00:00+00:00 | 2026-07-09 16:00:00+00:00 | 1.28 MB |

---

## 2. Verification of Multi-Symbol Strategy Training

### Instructions
1. Run a quick multi-symbol training pass covering `XAUUSD`, `USDJPY`, and `USDCAD`:
   ```powershell
   python trainer/trainer.py run --quick --symbol XAUUSD
   python trainer/trainer.py run --quick --symbol USDJPY
   python trainer/trainer.py run --quick --symbol USDCAD
   ```
2. Verify log output confirms `pip_size = 0.01` for `XAUUSD` and `USDJPY`, and `pip_size = 0.0001` for `USDCAD`.
3. Confirm zero `INSTRUMENT_SPEC_MISSING` errors occur during execution.
