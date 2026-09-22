# WAVE 4 — Manual Verification Protocol (Fix 4.2 Fill-Timing Verification)

## Purpose
Verify that the WAVE 4.2 fill-timing fix (where trade entries execute at the **next-candle open** instead of the **same-candle close**) operates correctly on **real historical market data** across real-world market conditions, price gaps, and rollover boundaries.

---

## Background & Fill-Timing Specification

In WAVE 4.2, a critical fill-timing bug was resolved:
- **Old (Broken) Behavior:** Signal generated at candle $i$ close filled immediately at candle $i$ close.
- **Current (Correct) Behavior:** Signal generated at candle $i$ close is queued and filled at candle $i+1$ **open** (`base_price = row['open']`).

This protocol validates that backtest trade logs produced from real historical OHLCV market data strictly obey next-candle open execution.

---

## Step-by-Step Execution Protocol

### 1. Run Real Backtest on Historical Data
Run a backtest on `EURUSD` using the Trainer CLI:
```powershell
.\venv\Scripts\Activate.ps1
python trainer/trainer.py run --symbol EURUSD --quick
```

Or extract detailed trade logs directly via Python using real historical dataset loading (`load_ohlcv`) and standard strategy signal modules (`trainer.signals.ma_crossover`):
```powershell
python -c "from trainer.core.backtester import Backtester; from trainer.core.data_loader import load_ohlcv; from datetime import datetime; import trainer.signals.ma_crossover as ma; df = load_ohlcv('EURUSD', 'M15', datetime(2020,1,1), datetime(2022,12,31)); bt = Backtester('EURUSD', params={'fast_ma_period': 10, 'slow_ma_period': 50, 'sl_atr_multiple': 1.5, 'tp_rr_ratio': 2.0}, allow_simulation_defaults=True); res = bt.run(df, ma); import json; print('Total trades:', len(res['trades'])); print(json.dumps(res['trades'][:15], indent=2, default=str))"
```

---

## Verification Criteria (Spot-Check 10–20 Trades)

Inspect a sample of 10 to 20 executed trades from the backtest output:

1. **Next-Candle Open Verification:**
   - Note the `entry_time` for each trade.
   - Cross-reference `entry_time` against the raw M15 historical OHLCV data (`load_ohlcv('EURUSD', 'M15')` or `D:\work\files\historical\EURUSD_M15.parquet`).
   - Confirm `trade['entry_price']` matches the **open** price of the candle at `entry_time` (plus/minus spread and slippage cost), **not** the close price of the preceding candle.

2. **No Same-Candle Close Reappearance:**
   - Confirm that zero trades exhibit an `entry_price` identical to the signal candle's `close` price with zero time offset.

3. **Weekend / News Price Gap Spot-Check:**
   - Locate at least one trade that entered near a market price gap (e.g. Sunday market open following a weekend gap).
   - Verify that the trade fill price reflects the **actual Sunday next-candle open** (including the price gap), rather than the pre-gap Friday close price.

---

## Disclaimer & Escalation

> [!IMPORTANT]
> This is a sanity-check against real market data to confirm the fix generalizes to actual market conditions. If any trade is observed filling at a signal-candle close price or failing to reflect next-candle open gaps, **halt training immediately** and flag the anomaly.
