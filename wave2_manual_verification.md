# WAVE 2 Manual Verification Steps

Run the following commands to complete full end-to-end verification of WAVE 2 changes.

### 1. Verification of `_calculate_swap_cost` for Overnight SELL Trades
Run a manual minimal backtest execution checking swap costs applied to SELL trades:
```bash
python -c "from trainer.core.backtester import Backtester; from datetime import datetime, timezone; bt = Backtester('EURUSD', params={}, allow_simulation_defaults=True); print('SELL Swap Cost:', bt._calculate_swap_cost('SELL', 1.0, datetime(2026,6,1,10,0,tzinfo=timezone.utc), datetime(2026,6,2,10,0,tzinfo=timezone.utc)))"
```
*Explanation:* Executes `_calculate_swap_cost` for a SELL trade across an overnight rollover boundary to verify runtime execution without error.

### 2. Verification of MT5 Broker 3-Day Rollover Property
Verify MT5 broker symbol contract rollover property against expected Wednesday 3-day swap convention:
```bash
python -c "import MetaTrader5 as mt5; mt5.initialize(); print('EURUSD Rollover 3-Day:', mt5.symbol_info('EURUSD').swap_rollover3days); mt5.shutdown()"
```
*Explanation:* Queries MT5 directly to confirm `swap_rollover3days` matches the assumed Wednesday 3x swap multiplier rule.

### 3. Full Strategy Walk-Forward Validation Run
Run full walk-forward validation across historical dataset:
```bash
python trainer/trainer.py run
```
*Explanation:* Executes full walk-forward training pipeline verifying that all instrument specs read from `D:\work\files\state\instrument_specs.json`, lot sizing, and swap cost calculations function under complete dataset evaluation.
