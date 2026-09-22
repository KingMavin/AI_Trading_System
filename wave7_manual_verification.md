# WAVE 7 — Fix 7.1 Manual Verification Instructions

## Purpose
Capture live broker contract specifications and raw MT5 symbol information for all 5 trading universe instruments (`EURUSD`, `GBPUSD`, `USDJPY`, `USDCAD`, `XAUUSD`) during active market hours to verify contract specifications and live spread integrity.

---

## Market Hours & Session Timing

> [!NOTE]
> Gold (`XAUUSD`) trades on a distinct schedule compared to standard FX majors:
> - **Trading Hours:** Sunday 23:00 UTC through Friday 21:59 UTC.
> - **Daily Market Break:** Monday–Thursday 22:00 UTC – 23:00 UTC (market closed).
> - **Recommended Session Windows:**
>   - **London / New York Overlap (Optimal):** 13:00 UTC – 16:00 UTC.
>   - **Asian Session Window:** 01:00 UTC – 06:00 UTC.

---

## Step-by-Step Execution Protocol

1. Open MetaTrader 5 (MT5), log into your trading account, and verify that live market quotes are updating in Market Watch.
2. Open PowerShell / Terminal in `d:\work\sy\ats`.
3. Run the live capture script:
   ```powershell
   .\venv\Scripts\Activate.ps1
   python capture_specs.py
   ```
4. Verify the output:
   - Captured spec file written to: `D:\work\files\state\instrument_specs.json`
   - Raw XAUUSD symbol dictionary written to: `scratch\xauusd_raw_symbol_info.json`
   - Verify `XAUUSD` outputs `sanity_ok = True` ("OK") with captured `tick_value = 0.10` falling within updated `SANITY_RANGES["XAUUSD"]` (`[0.05, 0.20]`).
5. Repeat at a second session window (e.g., during London/NY overlap vs Asian session) to capture potential spread/tick_value variance across sessions.

---

## Data Hand-Off
Once captured, verify the contents of:
- `D:\work\files\state\instrument_specs.json`
- `scratch\xauusd_raw_symbol_info.json`

All 5 universe instruments (`EURUSD`, `GBPUSD`, `USDJPY`, `USDCAD`, `XAUUSD`) must show `sanity_ok = True` ("OK") before proceeding to training or live trading.
