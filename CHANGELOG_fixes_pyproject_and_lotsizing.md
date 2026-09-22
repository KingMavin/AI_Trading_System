# Fixes: pytest pythonpath and lot sizing over-risk guard

## Issue 1: isolated pytest could not import `shared`

Root cause: `pyproject.toml` had no `[tool.pytest.ini_options]` block, so when `tests/test_broker_time.py` was run directly from a context where pytest did not place the project root on `sys.path`, `from shared...` imports failed. Other tests masked this because some of them insert the project root manually.

Diff:

```diff
diff --git a/pyproject.toml b/pyproject.toml
index 889ef68..6ae9392 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -7,4 +7,7 @@ name = "ats"
 version = "0.1.0"
 
 [tool.setuptools.packages.find]
-where = ["."]
\ No newline at end of file
+where = ["."]
+
+[tool.pytest.ini_options]
+pythonpath = ["."]
```

## Issue 2: `lot_size_for_risk()` minimum-lot over-risk guard returned `volume_min`

Root cause: the guard calculated minimum-lot risk only after `adj_lots` had already been clamped to `[volume_min, volume_max]`, and its `min_risk` calculation multiplied by `volume_min`. For the failing tiny-account case, that made the minimum trade appear to risk `0.0858`, below the `0.15` threshold, so the method silently returned `0.01`. The check now runs before clamping and evaluates the minimum tradable step risk when the rounded-down size would otherwise fall to the broker minimum.

Diff:

```diff
diff --git a/shared/instrument_spec.py b/shared/instrument_spec.py
--- a/shared/instrument_spec.py
+++ b/shared/instrument_spec.py
@@ -122,25 +122,28 @@ class InstrumentSpec:
         # Round down to volume_step
         step     = self.volume_step
         adj_lots = int(raw_lots / step) * step
-        adj_lots = max(self.volume_min, min(self.volume_max, adj_lots))
-        adj_lots = round(adj_lots, 2)
 
         # Safety: if minimum lot would risk more than configured,
         # log and return 0 so the caller can skip the trade
-        min_risk = (
-            self.volume_min *
-            (sl_distance_price / self.tick_size) *
-            self.tick_value
-        )
-        if min_risk > risk_amount * 1.5:
-            log.warning(
-                f"{self.symbol}: minimum lot ({self.volume_min}) "
-                f"would risk {min_risk:.2f} vs allowed "
-                f"{risk_amount:.2f}. Returning 0 — skip trade."
+        if adj_lots <= self.volume_min:
+            min_risk = (
+                (self.volume_min / self.volume_step) *
+                sl_distance_ticks *
+                self.tick_value
             )
-            return 0.0
+            if min_risk > risk_amount * 1.5:
+                log.warning(
+                    f"{self.symbol}: minimum lot ({self.volume_min}) "
+                    f"would risk {min_risk:.2f} vs allowed "
+                    f"{risk_amount:.2f}. Returning 0 — skip trade."
+                )
+                return 0.0
+
+        adj_lots = max(self.volume_min, min(self.volume_max, adj_lots))
+        adj_lots = round(adj_lots, 2)
 
         return adj_lots
```

## Verification

Final full-suite command used the project virtualenv pytest because bare `pytest` on PATH is pytest 6.2.4, which does not support the `pythonpath` config option. The project virtualenv pytest is 9.0.3.

Command:

```powershell
.\venv\Scripts\pytest.exe tests/ --tb=short
```

Full output:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-9.0.3, pluggy-1.6.0
rootdir: D:\work\sy\ats
configfile: pyproject.toml
plugins: anyio-4.13.0
collected 213 items

tests\test_adaptation.py .......................                         [ 10%]
tests\test_broker_time.py .....................                          [ 20%]
tests\test_dashboard.py ...............                                  [ 27%]
tests\test_engine_integration.py .......................                 [ 38%]
tests\test_indicators.py ............................................... [ 60%]
..                                                                       [ 61%]
tests\test_instrument_spec.py .....................                      [ 71%]
tests\test_scoring.py ..........................                         [ 83%]
tests\test_signal_engine.py .............                                [ 89%]
tests\test_trainer_runner.py ......................                      [100%]

============================== warnings summary ===============================
venv\Lib\site-packages\pandas_ta\__init__.py:37
  D:\work\sy\ats\venv\Lib\site-packages\pandas_ta\__init__.py:37: Pandas4Warning: The 'mode.copy_on_write' option is deprecated. Copy-on-Write can no longer be disabled (it is always enabled with pandas >= 3.0), and setting the option has no impact. This option will be removed in pandas 4.0.
    from pandas_ta.core import AnalysisIndicators

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 213 passed, 1 warning in 20.24s =======================
```

Targeted checks also passed:

```text
.\venv\Scripts\pytest.exe tests/test_broker_time.py -v
21 passed in 5.89s

.\venv\Scripts\pytest.exe tests/test_instrument_spec.py -v
21 passed in 0.24s
```

## Scope confirmation

Only these files were intentionally changed:

- `pyproject.toml`
- `shared/instrument_spec.py`
- `CHANGELOG_fixes_pyproject_and_lotsizing.md`

No file outside `pyproject.toml` and `shared/instrument_spec.py` was modified for the code/config fix. The changelog file was created as requested by the deliverable.

## Correction: over-risk guard formula

The previous guard placement was correct, but the formula inside the guard was wrong. `tick_value` is the cost of a one-tick move for a full 1.0 lot, so the minimum-lot risk must be:

```text
min_risk = volume_min * sl_distance_ticks * tick_value
```

### Corrected diff for `shared/instrument_spec.py`

```diff
diff --git a/shared/instrument_spec.py b/shared/instrument_spec.py
--- a/shared/instrument_spec.py
+++ b/shared/instrument_spec.py
@@ -127,7 +127,7 @@ class InstrumentSpec:
         # log and return 0 so the caller can skip the trade
         if adj_lots <= self.volume_min:
             min_risk = (
-                (self.volume_min / self.volume_step) *
+                self.volume_min *
                 sl_distance_ticks *
                 self.tick_value
             )
```

### Corrected diff for `tests/test_instrument_spec.py`

```diff
diff --git a/tests/test_instrument_spec.py b/tests/test_instrument_spec.py
--- a/tests/test_instrument_spec.py
+++ b/tests/test_instrument_spec.py
@@ -96,17 +96,17 @@ class TestLotSizing:
         assert lots == spec.volume_min
 
     def test_over_risk_returns_zero(self):
-        # With a tiny balance and a small SL, minimum lot
-        # would over-risk — should return 0
+        # With a tiny balance and a large enough SL, even the minimum
+        # lot would exceed the 1.5x risk guard — should return 0
         spec = make_spec(volume_min=0.01, tick_value=0.858)
         lots = spec.lot_size_for_risk(
             account_balance=10,     # tiny balance
             risk_pct=1.0,
-            sl_distance_price=0.0001
+            sl_distance_price=0.0002
         )
-        # 1% of 10 = 0.10; min lot (0.01) costs 0.858 per tick
-        # that's way more than 0.10 per tick — should return 0
+        # 1% of 10 = 0.10; guard threshold = 0.15
+        # min_risk = 0.01 lots × 20 ticks × 0.858 = 0.1716
         assert lots == 0.0
```

### Full `pytest tests/ --tb=short` output

Command:

```powershell
.\venv\Scripts\pytest.exe tests/ --tb=short
```

Output:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-9.0.3, pluggy-1.6.0
rootdir: D:\work\sy\ats
configfile: pyproject.toml
plugins: anyio-4.13.0
collected 213 items

tests\test_adaptation.py .......................                         [ 10%]
tests\test_broker_time.py .....................                          [ 20%]
tests\test_dashboard.py ...............                                  [ 27%]
tests\test_engine_integration.py .......................                 [ 38%]
tests\test_indicators.py ............................................... [ 60%]
..                                                                       [ 61%]
tests\test_instrument_spec.py .....................                      [ 71%]
tests\test_scoring.py ..........................                         [ 83%]
tests\test_signal_engine.py .............                                [ 89%]
tests\test_trainer_runner.py ......................                      [100%]

============================== warnings summary ===============================
venv\Lib\site-packages\pandas_ta\__init__.py:37
  D:\work\sy\ats\venv\Lib\site-packages\pandas_ta\__init__.py:37: Pandas4Warning: The 'mode.copy_on_write' option is deprecated. Copy-on-Write can no longer be disabled (it is always enabled with pandas >= 3.0), and setting the option has no impact. This option will be removed in pandas 4.0.
    from pandas_ta.core import AnalysisIndicators

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 213 passed, 1 warning in 13.84s =======================
```

### Worked examples

Safe minimum-lot case:

```text
account_balance = 10
risk_pct = 1.0
risk_amount = 10 * 0.01 = 0.10
guard_threshold = risk_amount * 1.5 = 0.15
volume_min = 0.01
sl_distance_ticks = 10
tick_value = 0.858
min_risk = 0.01 * 10 * 0.858 = 0.0858
```

Because `0.0858 <= 0.15`, the guard does not trigger. Manual check: `lot_size_for_risk(account_balance=10, risk_pct=1.0, sl_distance_price=0.0001)` returns `0.01`, not `0.0`.

Over-risk minimum-lot case:

```text
account_balance = 10
risk_pct = 1.0
risk_amount = 10 * 0.01 = 0.10
guard_threshold = risk_amount * 1.5 = 0.15
volume_min = 0.01
sl_distance_ticks = 20
tick_value = 0.858
min_risk = 0.01 * 20 * 0.858 = 0.1716
```

Because `0.1716 > 0.15`, the guard triggers. Manual check: `lot_size_for_risk(account_balance=10, risk_pct=1.0, sl_distance_price=0.0002)` returns `0.0`.
