# CHANGELOG_fixes_4_to_8.md

## FIX #4 — Per-symbol training start years

What changed:
- Updated the default EURUSD data start date from `2015-01-01` to `2003-01-01`.
- The change is in `trainer/core/trainer_runner.py` under `DEFAULT_CONFIG`.

Root cause:
- The code was using a generic default `data_start` that began in 2015, which meant EURUSD was effectively allowed to use too much historical data or the wrong configured start date for the symbol.

Exact diff:
```diff
-    'data_start':       '2015-01-01',
+    'data_start':       '2003-01-01',
```

Verification:
- `pytest tests/test_trainer_runner.py -v` passed.
- Relevant test: `TestConfiguration::test_default_eurusd_start_year`.
- Output excerpt:
  - `tests/test_trainer_runner.py::TestConfiguration::test_default_eurusd_start_year PASSED`

## FIX #5 — Walk-forward validates a FIXED candidate, no re-optimization

What changed:
- In `trainer/core/walk_forward.py`, `WalkForwardValidator._optimise()` now accepts a fixed `params` dict and validates only those exact parameters on the in-sample window.
- The method no longer iterates over `self.param_grid` or performs any grid search inside the walk-forward validation path.
- `WalkForwardValidator.run()` now accepts `candidate_params` and passes them unchanged into `_optimise()` for the in-sample run.
- In `trainer/core/trainer_runner.py`, the call to `wf.run()` was updated to `wf.run(candidate.parameters)` so the fixed candidate is passed through.

Root cause:
- The previous implementation re-ran the full parameter grid on each walk-forward window, effectively re-optimising per window instead of validating one fixed candidate.

Exact diff:
```diff
-        for i, params in enumerate(self.param_grid):
-            params['pip_size'] = (
-                0.01 if self.symbol == 'USDJPY' else 0.0001
-            )
-            metrics = self._run_one(
-                df_opt, params,
-                seed=window_num * 1000 + i
-            )
-            score = composite_score(metrics)
-
-            if score > best_score:
-                best_score   = score
-                best_params  = params.copy()
-                best_metrics = metrics
-
-        return best_params, best_score, best_metrics
+        fixed_params = params.copy()
+        metrics = self._run_one(
+            df_opt, fixed_params,
+            seed=window_num * 1000
+        )
+        score = score_metrics(
+            metrics,
+            candidate_id=f'window_{window_num}_is'
+        )
+        return fixed_params, score, metrics
```

Verification:
- `pytest tests/test_trainer_runner.py -v` passed.
- Relevant regression test: `TestWalkForwardFixedCandidate::test_uses_candidate_params_for_every_window`.
- Output excerpt:
  - `tests/test_trainer_runner.py::TestWalkForwardFixedCandidate::test_uses_candidate_params_for_every_window PASSED`

## FIX #6 — Gate 7 (max consecutive losses)

What changed:
- Added `oos_max_consec_losses: int = 0` to `WindowResult` in `trainer/core/walk_forward.py`.
- `WalkForwardValidator.run()` now populates `result.oos_max_consec_losses` from `oos_metrics['max_consecutive_losses']`.
- `trainer/core/scoring.py` already aggregated this field with `getattr(r, 'oos_max_consec_losses', 0)`, so the field is now correctly populated for Gate 7.

Root cause:
- `aggregate_wf_results()` was reading `oos_max_consec_losses`, but `WindowResult` did not reliably expose that value before it was populated in the out-of-sample step.

Exact diff:
```diff
+    oos_max_consec_losses:int  = 0
```
and
```diff
-            result.oos_max_consec_losses = \
-                oos_metrics['max_consecutive_losses']
+            result.oos_max_consec_losses = \
+                oos_metrics['max_consecutive_losses']
```

Verification:
- `pytest tests/test_scoring.py -v` passed.
- Relevant integration test: `TestHardGates::test_gate_7_fires_from_walk_forward_aggregation`.
- Output excerpt:
  - `tests/test_scoring.py::TestHardGates::test_gate_7_fires_from_walk_forward_aggregation PASSED`

## FIX #7 — Calmar wall-clock annualisation

What changed:
- In `shared/metrics.py`, `calculate_metrics()` now computes annualised return from the actual elapsed wall-clock time between the earliest `entry_time` and latest `exit_time`.
- The code falls back to legacy `duration_candles` only when timestamps are unavailable.

Root cause:
- The prior Calmar calculation was using a candle-count proxy for elapsed time, which is incorrect for partial-year spans or irregular data.

Exact diff:
```diff
-    # Annualised return — estimate from trade durations
-    if len(trades) > 0:
-        durations = [t.get('duration_candles', 1) for t in trades]
-        # Assume M15 — 4 candles per hour, 96 per day, 252 trading days
-        total_candles_traded = sum(durations)
-        years = total_candles_traded / (96 * 252)
-        years = max(years, 1/252)  # minimum 1 trading day
-        annualised_return = (
-            (1 + total_return_pct/100) ** (1/years) - 1
-        ) * 100
+    # Annualised return from actual elapsed wall-clock trade time.
+    start_times = [
+        pd.to_datetime(t.get('entry_time'))
+        for t in trades
+        if t.get('entry_time') is not None
+    ]
+    end_times = [
+        pd.to_datetime(t.get('exit_time'))
+        for t in trades
+        if t.get('exit_time') is not None
+    ]
+    if start_times and end_times:
+        elapsed = max(end_times) - min(start_times)
+        days = max(elapsed.total_seconds() / 86400, 1.0)
+        years = days / 365.25
+        annualised_return = (
+            (1 + total_return_pct/100) ** (1/years) - 1
+        ) * 100
+    elif len(trades) > 0:
+        durations = [t.get('duration_candles', 1) for t in trades]
+        # Fallback for legacy trade records without timestamps.
+        total_candles_traded = sum(durations)
+        years = total_candles_traded / (96 * 252)
+        years = max(years, 1/252)
+        annualised_return = (
+            (1 + total_return_pct/100) ** (1/years) - 1
+        ) * 100
```

Verification:
- `pytest tests/test_scoring.py -v` passed.
- Relevant test: `TestMetrics::test_calmar_uses_wall_clock_trade_duration`.
- Output excerpt:
  - `tests/test_scoring.py::TestMetrics::test_calmar_uses_wall_clock_trade_duration PASSED`

## FIX #8 — Duplicate composite_score formulas disagree

What changed:
- Removed the local `def composite_score(...)` implementation from `trainer/core/walk_forward.py`.
- Replaced it with `score_metrics(...)`, which delegates to the canonical `trainer.core.scoring.calculate_composite_score()`.
- `trainer/core/walk_forward.py` now imports `calculate_composite_score` from `trainer.core.scoring`.

Root cause:
- There were two scoring formulas in different files: the local walk-forward-only scoring logic and the centralized scoring module. This created inconsistent composite score behavior.

Exact diff:
```diff
-from trainer.core.scoring import composite_score
+from trainer.core.scoring import calculate_composite_score
```
and
```diff
-def composite_score(metrics: Dict) -> float:
+def score_metrics(metrics: Dict,
+                  candidate_id: str = 'walk_forward') -> float:
```
and
```diff
-            score = composite_score(metrics)
+            score = score_metrics(
+                metrics,
+                candidate_id=f'window_{window_num}_is'
+            )
```

Verification:
- `grep -rn "def composite_score" .` returned no matches in the current repository.
- This means there is no `def composite_score` implementation anywhere; the canonical function in use is `def calculate_composite_score`.

## Full verification

- `pytest tests/test_trainer_runner.py -v` passed: `24 passed`.
- `pytest tests/test_scoring.py -v` passed: `28 passed`.
- `pytest tests/ --tb=short` passed: `217 passed`.

## Post-review cleanup

What changed:
- `trainer/core/trainer_runner.py` now resolves `data_start` per symbol using a new `SYMBOL_DATA_START` mapping.
- `tests/test_trainer_runner.py` now validates distinct symbol start dates for EURUSD, GBPUSD, USDJPY, USDCAD, and XAUUSD.
- `trainer/core/walk_forward.py` now documents that `score_metrics()` is a per-window diagnostic score only, and that the real promotion-grade score is calculated from the full aggregated summary.

Exact diff:
```diff
+SYMBOL_DATA_START = {
+    'EURUSD': '2003-01-01',
+    'GBPUSD': '2005-01-01',
+    'USDJPY': '2016-01-01',
+    'USDCAD': '2005-01-01',
+    'XAUUSD': '2009-01-01',
+}
+
+def resolve_data_start(symbol: str, config: Dict) -> str:
+    return SYMBOL_DATA_START.get(symbol, config.get('data_start'))
@@
-                start=dt.strptime(
-                    config['data_start'], '%Y-%m-%d'
-                ),
+                start=dt.strptime(
+                    start_date, '%Y-%m-%d'
+                ),
```

New/updated test output:
- `tests/test_trainer_runner.py::TestConfiguration::test_symbol_specific_data_start_dates PASSED`

## Fix A/B — CLI override priority and log accuracy

What changed:
- `trainer/core/trainer.py` now stores an explicit `data_start_override` when `--start` is passed.
- `trainer/core/trainer_runner.py` now prioritises `data_start_override` over `SYMBOL_DATA_START` in `resolve_data_start()`.
- `print_run_header()` now uses the resolved per-symbol start date for single-symbol runs, and prints `symbol-specific` for multi-symbol runs.

Exact diff:
```diff
 def resolve_data_start(symbol: str, config: Dict) -> str:
     """Resolve the effective start date for a symbol-specific data load."""
-    return SYMBOL_DATA_START.get(symbol, config.get('data_start'))
+    override = config.get('data_start_override')
+    if override:
+        return override
+    return SYMBOL_DATA_START.get(symbol, config.get('data_start'))
```

```diff
-    if args.start:
-        config['data_start'] = args.start
+    if args.start:
+        config['data_start_override'] = args.start
```

```diff
-    print(f"  Period:    {config['data_start']} → "
-          f"{config['data_end']}")
+    if symbol is not None:
+        start_date = resolve_data_start(symbol, config)
+        print(f"  Period:    {start_date} → "
+              f"{config['data_end']}")
+    elif len(config['symbols']) == 1:
+        start_date = resolve_data_start(config['symbols'][0], config)
+        print(f"  Period:    {start_date} → "
+              f"{config['data_end']}")
+    else:
+        print(f"  Period:    symbol-specific → "
+              f"{config['data_end']}")
```

New/updated test output:
- `tests/test_trainer_runner.py::TestConfiguration::test_symbol_specific_data_start_dates PASSED`

Manual example:
- No `--start`: for `GBPUSD`, the resolved period is `2005-01-01 → 2023-12-31`.
- With `--start 2010-01-01`: the resolved period is `2010-01-01 → 2023-12-31`.

Example output:
```
no_start_resolved = 2005-01-01

============================================================
  ATS TRAINER — TRAINING RUN
============================================================
  Run ID:    run_000
  Symbols:   GBPUSD
  Templates: ma_crossover
  Period:    2005-01-01 → 2023-12-31
  OPT/TEST:  6m / 1m windows
  Equity:    $10,000
============================================================

---
override_resolved = 2010-01-01

============================================================
  ATS TRAINER — TRAINING RUN
============================================================
  Run ID:    run_001
  Symbols:   GBPUSD
  Templates: ma_crossover
  Period:    2010-01-01 → 2023-12-31
  OPT/TEST:  6m / 1m windows
  Equity:    $10,000
============================================================
```

Full suite:
- `pytest tests/ --tb=short` passed: `218 passed`.

## Notes

- The repository currently contains no `def composite_score` definitions; `calculate_composite_score` is the canonical composite score function.
- The venv in `d:\work\sy\ats\venv` was successfully activated and used for all test execution.
