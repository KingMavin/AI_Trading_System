# Adaptive Trading System (ATS) v3.0

An enterprise-grade, autonomous quantitative algorithmic trading system engineered in Python for MetaTrader 5 (MT5). ATS v3.0 is built from first principles for prop firm evaluation rules (e.g., FTMO) and live capital protection, combining walk-forward optimization, multi-gate statistical validation, look-ahead bias elimination, and execution safety guards.

---

## 1. System Overview & Core Philosophy

Wrong code in financial systems costs real money. ATS v3.0 separates strategy discovery from real-time trade execution through an immutable interface:

```
┌─────────────────────────────────────────────────────────────┐
│                       ATS TRAINER                           │
│  - Historical Data Ingestion & Indicator Caching            │
│  - Multivariate Optuna Optimization (TPE Sampler)           │
│  - Walk-Forward Out-Of-Sample (OOS) Validation              │
│  - 10-Gate Statistical & Prop Mandate Audits                │
│  - Monte Carlo Permutation Stress Testing                   │
│  - Multiple-Testing Deflation & Prop-Firm Buffer Penalties  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Writes deployed strategy
                               ▼
              engine/strategy/active_strategy.json
                               ▲
                               │ Reads deployed strategy
┌──────────────────────────────┴──────────────────────────────┐
│                       ATS ENGINE                            │
│  - Real-Time MT5 Tick & Candle Ingestion                    │
│  - Closed-Candle Signal Evaluation & Exit Delegation        │
│  - Strict Risk Math & Lot Sizing (1% equity risk / trade)   │
│  - Static Drawdown & Europe/Prague Daily Loss Tracking      │
│  - Magic Number Position Isolation (123456)                 │
│  - Live CUSUM & Profit Factor Degradation Monitoring        │
└─────────────────────────────────────────────────────────────┘
```

- **Architectural Boundary:** The **Trainer** discovers, evaluates, and writes `active_strategy.json`. The **Engine** reads `active_strategy.json` and executes orders. The Engine never modifies strategy definitions.
- **Strict Look-Ahead Bias Prevention:** Signals are computed strictly using closed bar features (`row`, `row_prev`). Simulated orders execute at the **open of the next candle**. Forward-return labels or MT5 runtime states are never leaked into simulation.
- **Fail-Loud Convention:** Missing configurations, corrupted datasets, unanchored timestamps, or missing instrument specifications raise explicit exceptions and halt processing immediately rather than degrading silently.

---

## 2. Hard Mandates & Risk Governance

ATS v3.0 implements prop firm rules natively in both offline backtesting and the live execution engine:

| Mandate Rule | Specification | Implementation Detail |
| :--- | :--- | :--- |
| **Overall Drawdown** | **Static 8.0%** ceiling from initial equity | Measured against `initial_equity` (`equity < initial_equity * 0.92`), NOT high-water mark trailing. |
| **Max Daily Loss** | **4.0%** daily loss limit | Equity measured against balance established at **00:00 Europe/Prague** midnight. |
| **Risk per Trade** | **1.0%** equity risk cap | Dynamic lot sizing derived strictly from `volume_min × sl_ticks × tick_value`. |
| **Min Trading Days** | **5** distinct trading days | Counted as unique calendar dates in `Europe/Prague` across out-of-sample trades. |
| **Magic Number Isolation**| `DEFAULT_MAGIC = 123456` | Engine only monitors, manages, or closes positions matching its assigned magic number. |
| **Trading Dead-Zone** | 21:00 – 23:00 UTC Blocked | Blocks new entries during rollover widened spreads and low-liquidity periods. |

---

## 3. Current System Capabilities

### 3.1 Strategy Templates & Signal Modules
Located in `trainer/signals/` and executed via `engine/core/signal_engine.py`:
- **Hybrid Confluence (`hybrid_confluence`)**: Combines Directional Moving Average (DMA) trend filter, ADX trend strength threshold, and RSI pullback detection with ATR-based Stop Loss and Take Profit.
- **Donchian Breakout (`donchian_breakout`)**: Volatility breakout system entering on channel boundary penetrations with trailing ATR channels or opposite-band exits.
- **RSI Mean Reversion (`rsi_reversion`)**: Overbought/oversold oscillator entries paired with dynamic RSI midline or reversal exit checks.
- **Moving Average Crossover (`ma_crossover`)**: Fast/Slow SMA crossover with directional ADX filter.
- **Combination Strategies (`combo_<entry>__<exit>`)**: Modular pairing of an entry signal module with a distinct exit signal module (e.g. `combo_hybrid_confluence__rsi_reversion` routes entries from confluence and exits from RSI midline crosses).

### 3.2 10-Gate Validation & Scoring Engine
Located in `trainer/core/scoring.py` and `trainer/core/walk_forward.py`:
1. **Minimum Total Trades:** Ensures statistical sample size (`>= 100` total OOS trades).
2. **Trades per Window:** Rejects under-active parameter sets (`>= 20` trades/window).
3. **Valid Windows:** Requires consistency across market regimes (`>= 12` valid windows).
4. **Drawdown Headroom:** Checks median OOS drawdown against mandate limits.
5. **Profit Factor:** Enforces minimum OOS profitability (`PF >= 1.10`).
6. **Profitable Window Rate:** Requires positive returns across `>= 55%` of test periods.
7. **Consecutive Losses:** Caps maximum losing runs (`<= 10` consecutive losses).
8. **Regime Consistency:** Evaluates performance across trending and ranging market states.
9. **Mandate Compliance Gate:** Fail-loud gate enforcing zero static 8% breaches, zero 4% daily loss breaches, max 1.0% trade risk, and minimum 5 Prague trading days across all aggregated OOS trades.
10. **Monte Carlo Stress Test:** Permutes OOS trade sequences 1,000 times under varying execution paths; rejects candidates with a breach rate exceeding `3.0%`.

**Composite Scoring Formula:**
- Weighted evaluation: Calmar ratio (40%), Profit Factor (25%), Win/Loss ratio (15%), Regime consistency (10%), Stability (5%), Robustness (5%).
- **Multiple-Testing Deflation Penalty:** Monotonically deflates scores as candidate search trials increase (`trial_count: int` required keyword argument, zero-bypass prohibited).
- **Prop-Firm Penalty:** Applies incremental penalties when maximum drawdown approaches `> 85%` of the static drawdown limit.

### 3.3 Realistic Execution & Backtester Model
Located in `trainer/core/backtester.py`:
- **Spread Modeling:** Scaled dynamically by session (`LONDON`: 1.0x, `NEW_YORK`: 1.1x, `ASIAN`: 1.5x, `DEAD_ZONE`: 2.5x) and volatility (`ATR / baseline`).
- **Slippage Modeling:** Variable execution slippage (`0.3` pips base scaled by ATR ratio and uniform randomization).
- **Transaction Costs:** Full commission deduction (`$7.00` per lot round-trip).
- **Financing / Swaps:** Simulates daily swap rollover at `21:00 UTC` with 3x triple swaps applied on Wednesdays.

### 3.4 Live Engine & Runtime Safety
Located in `engine/core/engine.py`:
- **Real-Time Data Streaming:** Synchronizes live MT5 M15/H1 candle data with automatic connection health monitoring.
- **Dynamic Lot Sizing:** Automatically queries account equity, converts pip distance to ticks, and computes exact volume using broker-reported `trade_tick_value`.
- **Dual Circuit Breakers:** Evaluates daily and overall drawdown after every tick and candle; halts new trade placement upon reaching mandate thresholds.
- **Degradation Detection:** Implements rolling Profit Factor monitoring and a cumulative sum (**CUSUM**) detector (`h = 12.0`, `min_trades = 20`) to identify structural edge decay in live conditions.

---

## 4. Repository Structure

```
ats/
├── docs/
│   └── archive/              # Archived changelogs and verification reports
├── engine/
│   ├── config/
│   │   └── mandate.json      # Prop firm mandate parameters (drawdown, universe, daily loss)
│   ├── core/
│   │   ├── engine.py         # Live MT5 execution loop, circuit breakers, lot sizing
│   │   ├── degradation.py    # CUSUM drift detection & PF degradation monitors
│   │   ├── signal_engine.py  # Live signal generation and check_exit delegation
│   │   └── strategy_loader.py# Validates and loads deployed strategy JSON files
│   ├── dashboard/            # Web dashboard interface (app.py)
│   └── strategy/
│       └── active_strategy.json # Deployed production strategy configuration
├── shared/
│   ├── broker_time.py        # Naive datetime handling & timezone converters
│   ├── config.py             # System paths, account currency defaults, storage config
│   ├── dna.py                # Deterministic SHA-256 strategy parameter hashing
│   ├── indicators.py         # Canonical TA indicators (SMA, EMA, ATR, ADX, RSI, Donchian)
│   ├── instrument_spec.py    # Symbol specifications, tick value, and pip size validation
│   └── metrics.py            # Wall-clock annualization, Sharpe, Calmar, profit factor
├── tests/                    # 490+ unit, integration, and financial test suites
├── trainer/
│   ├── core/
│   │   ├── adaptation.py     # Parameter search & combination strategy routing
│   │   ├── backtester.py     # Cost-aware event-driven candle backtesting engine
│   │   ├── data_loader.py    # Parquet & CSV historical data management
│   │   ├── knowledge_base.py # Timeframe-isolated strategy memory & warm-starts
│   │   ├── mandate_evaluator.py # Europe/Prague daily loss & static drawdown evaluator
│   │   ├── parameter_grid.py # Parameter spaces for all supported templates
│   │   ├── promotion.py      # Automated promotion & deployment logic
│   │   ├── scoring.py        # 10-gate audit, composite score, deflation penalty
│   │   ├── trainer_runner.py # Multi-symbol orchestration pipeline
│   │   └── walk_forward.py   # Walk-Forward Optimizer with multivariate Optuna TPE
│   ├── signals/              # Strategy signal definitions (Hybrid, Donchian, RSI, MA)
│   └── trainer.py            # CLI entry point for training runs and readiness checks
└── capture_specs.py          # Script to fetch live broker instrument specs via MT5
```

---

## 5. Quickstart & Operational Commands

### 5.1 Environment Setup
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

### 5.2 Capture Live Instrument Specs
Run while MetaTrader 5 is connected during active market hours to capture live spreads, tick sizes, and tick values:
```powershell
python capture_specs.py
```
Outputs are verified and saved to `D:\work\files\state\instrument_specs.json`.

### 5.3 System Readiness & Health Checks
Verify that historical data, indicators, packages, and strategy dependencies are operational:
```powershell
python -m trainer.trainer check
```

### 5.4 Execute Strategy Training Pipeline
Run a walk-forward training campaign on XAUUSD (H1 timeframe):
```powershell
python trainer/trainer.py run --symbol XAUUSD --start 2015-01-01 --end 2023-12-31
```
For rapid development iteration:
```powershell
python trainer/trainer.py run --quick --symbol EURUSD
```

### 5.5 Start Live Execution Engine
To run paper trading mode with safety circuit breakers:
```powershell
python engine/core/engine.py --symbol XAUUSD --paper
```

---

## 6. Testing & Verification

The test suite validates financial correctness, boundary conditions, and mandate enforcement across **491 tests** with zero failures:

```powershell
# Run the entire test suite
pytest tests/ --tb=short

# Run specific functional areas
pytest tests/test_mandate.py tests/test_mandate_evaluator.py --tb=short
pytest tests/test_scoring.py tests/test_monte_carlo_gate.py --tb=short
pytest tests/test_backtester_financial.py tests/test_instrument_spec.py --tb=short
```

---

## 7. Developer & Safety Rules

All contributors must adhere strictly to `AGENTS.md`:
1. **Never bypass a hard mandate rule** to make a test pass.
2. **Money-math formulas are canonical**:
   - `tick_value` = cost per 1.0 standard lot per tick.
   - `pip_value = tick_value × (pip_size / tick_size)`.
   - `lot_size_for_risk` uses `volume_min × sl_distance_ticks × tick_value`.
3. **No silent fallbacks on safety-critical paths**: Missing specs, incomplete data partitions, or unanchored timestamps must fail loudly.
4. **Mandatory parameter validation**: Functions implementing statistical or risk guards must require explicit inputs without defaults that disable the guard.
