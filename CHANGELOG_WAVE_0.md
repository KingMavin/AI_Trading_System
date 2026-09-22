# CHANGELOG: WAVE 0

## Root Cause / What was done
1. **Part 1 & 3 Updates:** Updated the execution plan (`implementation_plan.md`) to explicitly remove the stale duplicate step 8.26, clarified step 6.5 as a manual operational setup step rather than hardcoded configuration, and replaced manual CSV download steps with an automated MT5 script (`fetch_mt5_data.py`).
2. **Part 2 Updates:** Added `USDCAD` and `XAUUSD` to the valid symbol choices in `trainer/trainer.py`, refactored the CLI parsing logic so tests can import the parser, and added the CLI test `tests/test_trainer_cli.py`.
3. **WAVE 0 Execution:** Created `fetch_mt5_data.py` to pull M1 history from MT5. Attempted to execute WAVE 0 steps (0.1 through 0.5) against the live MT5 terminal, but encountered terminal authorization failures. As per AGENTS.md rule 3, rather than failing silently, the exact output of the failure is captured and reported below.

## Files Touched
```diff
--- d:\work\sy\ats\trainer\trainer.py
+++ d:\work\sy\ats\trainer\trainer.py
@@ -60,7 +60,7 @@
 
     # Check 1: Data files exist
     from trainer.core.data_loader import get_available_range
-    symbols    = ['EURUSD', 'GBPUSD', 'USDJPY']
+    symbols    = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'XAUUSD']
     timeframes = ['M15', 'H1', 'H4']
 
     data_ok = True
@@ -155,7 +155,7 @@
           "engine/strategy/active_strategy.json")
 
 
-if __name__ == '__main__':
+def build_parser():
     parser = argparse.ArgumentParser(
         description='ATS Trainer — adaptive strategy training'
     )
@@ -166,7 +166,7 @@
     )
     run_parser.add_argument(
         '--symbol',
-        choices=['EURUSD', 'GBPUSD', 'USDJPY'],
+        choices=['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'XAUUSD'],
         help='Single symbol to train on'
     )
     run_parser.add_argument(
@@ -201,7 +201,11 @@
         'rollback', help='Roll back to previous strategy'
     )
     rollback_parser.set_defaults(func=cmd_rollback)
-
+    
+    return parser
+
+if __name__ == '__main__':
+    parser = build_parser()
     args = parser.parse_args()
 
     if not args.command:
```

## New or Changed Tests
**`tests/test_trainer_cli.py` (NEW):**
```python
import pytest
import argparse
from trainer.trainer import build_parser

def test_trainer_cli_valid_symbols():
    parser = build_parser()
    # Test all 5 valid symbols
    valid_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'XAUUSD']
    for sym in valid_symbols:
        args = parser.parse_args(['run', '--symbol', sym])
        assert args.symbol == sym
        assert args.command == 'run'

def test_trainer_cli_invalid_symbol(capsys):
    parser = build_parser()
    # Test invalid symbol rejection
    with pytest.raises(SystemExit):
        parser.parse_args(['run', '--symbol', 'INVALID'])

    
    captured = capsys.readouterr()
    assert 'invalid choice' in captured.err
```

## Command Output
**pytest tests/test_trainer_cli.py:**
```text
============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-9.0.3, pluggy-1.6.0 -- D:\work\sy\ats\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\work\sy\ats
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 2 items

tests/test_trainer_cli.py::test_trainer_cli_valid_symbols PASSED         [ 50%]
tests/test_trainer_cli.py::test_trainer_cli_invalid_symbol PASSED        [100%]

============================== warnings summary ===============================
venv\Lib\site-packages\pandas_ta\__init__.py:37
  D:\work\sy\ats\venv\Lib\site-packages\pandas_ta\__init__.py:37: Pandas4Warning: The 'mode.copy_on_write' option is deprecated. Copy-on-Write can no longer be disabled (it is always enabled with pandas >= 3.0), and setting the option has no impact. This option will be removed in pandas 4.0.
    from pandas_ta.core import AnalysisIndicators

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 2 passed, 1 warning in 3.70s =========================
```

**capture_specs.py (WAVE 0 Step 0.2):**
```text
D:\work\sy\ats\capture_specs.py:1: SyntaxWarning: invalid escape sequence '\w'
  """

ATS — InstrumentSpec Capture
=======================================================
MT5 init failed: (-6, 'Terminal: Authorization failed')
```

## Observed but out of scope
- The `capture_specs.py` script and the new `fetch_mt5_data.py` script both encounter the MT5 authorization error `(-6, 'Terminal: Authorization failed')`. As a result, the live data dependencies for WAVE 0 (steps 0.1 through 0.5) could not be satisfied. The MetaTrader 5 terminal on this machine appears to be logged out or lacks the necessary credentials/permissions.
- There is a `SyntaxWarning: invalid escape sequence '\w'` at the top of `capture_specs.py` due to a backslash in the module docstring (e.g. `D:\work\files`). I left it as is.
