# CHANGELOG — fixes 9 to 12

## Overview
Implemented fixes for the trainer/backtester pipeline and scoring logic, including explicit tests for each change. The focus was on ensuring disjoint data partitioning for Stage 2 filter selection, correct swap cost accounting, captured spread lookup from instrument specs, and trial-count-aware score deflation.

## Files changed
- `trainer/core/trainer_runner.py`
- `trainer/core/adaptation.py`
- `trainer/core/backtester.py`
- `trainer/core/scoring.py`
- `trainer/core/walk_forward.py`
- `tests/test_trainer_runner.py`
- `tests/test_scoring.py`
- `tests/test_instrument_spec.py`

## Fixes applied

### Fix #9 — Disjoint filter selection & held-out partitioning
- Updated `trainer/core/trainer_runner.py` to reserve the final walk-forward `OPT+TEST` period from the main dataset.
- Ensured Stage 2 adaptation uses only historical filter-selection data.
- Added explicit test coverage in `tests/test_trainer_runner.py` verifying no overlap between:
  - filter-selection data
  - walk-forward data
  - held-out data

### Fix #10 — Swap charge accounting in backtester
- Updated `trainer/core/backtester.py` to apply overnight swap costs on round-trip trades.
- Swap cost is computed via `InstrumentSpec.swap_cost()` and supports the triple-swap Wednesday rule.
- Trade records now include `swap_cost`.
- Added `tests/test_instrument_spec.py` coverage for swap cost calculation and backtester swap integration.

### Fix #11 — Real captured spread from `InstrumentSpec`
- Updated backtester initialization to override placeholder spread using captured `InstrumentSpec.spread_typical`.
- Added support for symbol fallback when a spec cache exists but symbol is not in hardcoded `SYMBOL_SPECS`.
- Added warning behavior when no cached spec exists and a placeholder spread fallback is used.
- Verified via `tests/test_instrument_spec.py`.

### Fix #12 — Candidate score deflation for trial count
- Updated `trainer/core/scoring.py` to add a trial-count deflation penalty to composite scores.
- The penalty is applied only when trial count exceeds the baseline threshold, avoiding unfair penalization of small search spaces.
- Wired `adaptation.trial_count` into the score calculation path in `trainer/core/trainer_runner.py`.
- Added tests in `tests/test_trainer_runner.py` to validate penalty behavior and score propagation.

## Validation
Ran the full test suite using the local virtual environment:

```bash
cd d:\work\sy\ats
.\venv\Scripts\python.exe -m pytest tests -q
```

Result: `226 passed, 1 warning`

## Fix #9 — hard failure on insufficient history
- Updated `trainer/core/trainer_runner.py` to raise `InsufficientHistoryError` when the available main dataset cannot reserve the required `held_out_months + opt_months + test_months` period for disjoint walk-forward validation.
- This prevents the prior fallback from continuing with identical `df_filter_selection` and `df_walk_forward` ranges.
- Added regression coverage in `tests/test_trainer_runner.py` with a short-history dataset that now raises instead of silently passing.

```python
    def test_short_history_raises_insufficient_history_error(
            self, monkeypatch):
        idx = pd.date_range(
            '2020-01-01',
            '2020-08-31',
            freq='D',
            tz='UTC'
        )
        df = pd.DataFrame({
            'open': 1.0,
            'high': 1.0,
            'low': 1.0,
            'close': 1.0,
            'volume': 100,
        }, index=idx)

        def fake_load_ohlcv(symbol, timeframe, start, end, warmup_candles):
            return df

        import trainer.core.trainer_runner as runner_mod

        monkeypatch.setattr(runner_mod, 'load_ohlcv', fake_load_ohlcv)

        runner = TrainerRunner(config={
            **DEFAULT_CONFIG,
            'symbols': ['EURUSD'],
            'held_out_months': 2,
            'opt_months': 6,
            'test_months': 1,
        })

        record = RunRecord('run_test', runner.config)
        with pytest.raises(runner_mod.InsufficientHistoryError) as excinfo:
            runner._run_symbol('EURUSD', record, None)

        assert 'Insufficient history' in str(excinfo.value)
```

## Fix #12 — corrected deflation curve and required parameter
- Updated `trainer/core/scoring.py` to require `trial_count` as an explicit keyword-only argument when computing a composite score.
- Changed the penalty formula from `0.02 * (log10(trial_count) - 1.0)` to `0.04 * log10(trial_count)` with a `0.20` cap.
- This produces the following benchmark values:
  - `trial_count=1` → `0.000`
  - `trial_count=2` → `0.012`
  - `trial_count=10` → `0.040`
  - `trial_count=50` → `0.068`
  - `trial_count=144` → `0.086`
- Updated `trainer/core/walk_forward.py` to pass `trial_count=1` explicitly for per-window diagnostics.
- Updated `trainer/core/adaptation.py` to pass `trial_count=1` explicitly for adaptation-stage summary scoring.
- Updated `trainer/core/trainer_runner.py` to pass `trial_count=adaptation.trial_count` into final candidate scoring.
- Added regression coverage for the new curve and required parameter behavior.

```python
    def test_trial_count_is_required(self):
        with pytest.raises(TypeError):
            calculate_composite_score(good_wf_summary(), 'test')
```

```python
    def test_scoring_uses_trial_count_deflation(self):
        wf_summary = {
            'median_calmar_ratio': 1.8,
            'median_profit_factor': 2.0,
            'median_avg_win_loss': 1.5,
            'regime_consistency': 0.8,
            'temporal_stable': True,
            'temporal_trend_pct': 2.0,
        }

        score_small = calculate_composite_score(
            wf_summary,
            'cand_x',
            trial_count=2,
        )
        score_large = calculate_composite_score(
            wf_summary,
            'cand_x',
            trial_count=100,
        )

        assert score_large.raw_composite == score_small.raw_composite
        assert score_small.deflation_penalty == 0.0
        assert score_large.deflation_penalty > 0.0
        assert score_large.composite < score_small.composite
```

## Notes
- `trainer/core/backtester.py` now imports `logging` and defines a logger for warning messages.
- `trainer/core/adaptation.py` was updated to expose `trial_count` from parameter search, so walk-forward scoring receives correct candidate trial information.
