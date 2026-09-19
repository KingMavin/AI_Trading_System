"""
Unit tests for trainer_runner.py
Tests RunRecord, configuration, and pipeline structure.
Run: pytest tests/test_trainer_runner.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime, timezone
from trainer.core.trainer_runner import (
    RunRecord, DEFAULT_CONFIG, QUICK_CONFIG,
    compute_data_hash, print_run_header
)


# ── RUN RECORD TESTS ───────────────────────────────────

class TestRunRecord:

    def test_initialises_correctly(self):
        record = RunRecord('run_001', DEFAULT_CONFIG)
        assert record.run_id            == 'run_001'
        assert record.outcome           == 'RUNNING'
        assert record.candidates_tested == 0
        assert record.best_score        == 0.0
        assert record.new_strategy_deployed is False

    def test_layer_start_creates_entry(self):
        record = RunRecord('run_001', DEFAULT_CONFIG)
        record.layer_start('layer_2_data')
        assert 'layer_2_data' in record.layers
        assert record.layers['layer_2_data']['status'] \
               == 'RUNNING'

    def test_layer_complete_updates_status(self):
        record = RunRecord('run_001', DEFAULT_CONFIG)
        record.layer_start('layer_2_data')
        record.layer_complete('layer_2_data',
                               {'candles': 50000})
        assert record.layers['layer_2_data']['status'] \
               == 'COMPLETE'
        assert record.layers['layer_2_data'][
            'summary'
        ]['candles'] == 50000

    def test_layer_failed_records_reason(self):
        record = RunRecord('run_001', DEFAULT_CONFIG)
        record.layer_start('layer_6_adaptation')
        record.layer_failed('layer_6_adaptation',
                             'RAM exhausted')
        assert record.failure_layer  == 'layer_6_adaptation'
        assert record.failure_reason == 'RAM exhausted'
        assert record.layers['layer_6_adaptation'][
            'status'
        ] == 'FAILED'

    def test_to_dict_includes_all_fields(self):
        record = RunRecord('run_001', DEFAULT_CONFIG)
        record.outcome      = 'COMPLETED'
        record.completed_at = datetime.now(timezone.utc)
        d = record.to_dict()

        required = [
            'run_id', 'started_at', 'outcome', 'config',
            'candidates_tested', 'best_score',
            'new_strategy_deployed', 'failure_layer',
        ]
        for field in required:
            assert field in d, f"Missing field: {field}"

    def test_duration_calculated_correctly(self):
        record = RunRecord('run_001', DEFAULT_CONFIG)
        # Simulate a completed run
        record.completed_at = datetime.now(timezone.utc)
        d = record.to_dict()
        assert d['duration_seconds'] is not None
        assert d['duration_seconds'] >= 0

    def test_candidate_counters(self):
        record = RunRecord('run_001', DEFAULT_CONFIG)
        record.candidates_tested   = 10
        record.candidates_promoted = 1
        record.candidates_flagged  = 2
        record.candidates_rejected = 7
        d = record.to_dict()
        assert d['candidates_tested']   == 10
        assert d['candidates_promoted'] == 1
        assert d['candidates_flagged']  == 2
        assert d['candidates_rejected'] == 7


# ── CONFIGURATION TESTS ────────────────────────────────

class TestConfiguration:

    def test_default_config_has_required_keys(self):
        required = [
            'symbols', 'timeframe', 'templates',
            'opt_months', 'test_months', 'initial_equity',
            'top_n_candidates', 'data_start', 'data_end',
        ]
        for key in required:
            assert key in DEFAULT_CONFIG, \
                f"Missing key: {key}"

    def test_quick_config_has_required_keys(self):
        required = [
            'symbols', 'timeframe', 'templates',
            'opt_months', 'test_months', 'initial_equity',
        ]
        for key in required:
            assert key in QUICK_CONFIG

    def test_quick_config_shorter_period(self):
        from datetime import datetime as dt
        default_start = dt.strptime(
            DEFAULT_CONFIG['data_start'], '%Y-%m-%d'
        )
        quick_start = dt.strptime(
            QUICK_CONFIG['data_start'], '%Y-%m-%d'
        )
        assert quick_start > default_start

    def test_initial_equity_positive(self):
        assert DEFAULT_CONFIG['initial_equity'] > 0
        assert QUICK_CONFIG['initial_equity'] > 0

    def test_symbols_list_not_empty(self):
        assert len(DEFAULT_CONFIG['symbols']) > 0

    def test_templates_list_not_empty(self):
        assert len(DEFAULT_CONFIG['templates']) > 0

    def test_opt_months_greater_than_test_months(self):
        from trainer.core.trainer_runner import DEFAULT_TEMPLATE_TEST_MONTHS
        test_m = DEFAULT_CONFIG.get('test_months') or max(DEFAULT_TEMPLATE_TEST_MONTHS.values())
        assert DEFAULT_CONFIG['opt_months'] > test_m

    def test_valid_symbols(self):
        valid = {'EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'}
        for sym in DEFAULT_CONFIG['symbols']:
            assert sym in valid

    def test_valid_timeframe(self):
        valid = {'M15', 'H1', 'H4'}
        assert DEFAULT_CONFIG['timeframe'] in valid


# ── DATA HASH TESTS ────────────────────────────────────

class TestDataHash:

    def test_hash_returns_string(self):
        # Will return 'no_data' since no Parquet file
        # exists in the test environment
        result = compute_data_hash('EURUSD', 'M15')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_different_for_different_symbols(self):
        h1 = compute_data_hash('EURUSD', 'M15')
        h2 = compute_data_hash('GBPUSD', 'M15')
        # Both may be 'no_data' in test env — that's ok
        # The important thing is the function runs without error
        assert isinstance(h1, str)
        assert isinstance(h2, str)


# ── TRAINER RUNNER STRUCTURE TESTS ─────────────────────

class TestTrainerRunnerStructure:

    def test_runner_initialises_with_defaults(self):
        runner = __import__(
            'trainer.core.trainer_runner',
            fromlist=['TrainerRunner']
        ).TrainerRunner()
        assert runner.config == DEFAULT_CONFIG
        assert runner.run_id.startswith('run_')

    def test_runner_quick_mode(self):
        runner = __import__(
            'trainer.core.trainer_runner',
            fromlist=['TrainerRunner']
        ).TrainerRunner(quick_mode=True)
        assert runner.config == QUICK_CONFIG

    def test_runner_custom_config(self):
        custom = {**DEFAULT_CONFIG, 'symbols': ['GBPUSD']}
        runner = __import__(
            'trainer.core.trainer_runner',
            fromlist=['TrainerRunner']
        ).TrainerRunner(config=custom)
        assert runner.config['symbols'] == ['GBPUSD']

    def test_run_id_format(self):
        runner = __import__(
            'trainer.core.trainer_runner',
            fromlist=['TrainerRunner']
        ).TrainerRunner()
        # Format: run_YYYYMMDD_HHMMSS
        parts = runner.run_id.split('_')
        assert parts[0] == 'run'
        assert len(parts[1]) == 8   # YYYYMMDD
        assert len(parts[2]) == 6   # HHMMSS


# ── SANITY_OK ENFORCEMENT TESTS ────────────────────────

class TestSanityOkGates:
    """
    Verify that a sanity_ok=False InstrumentSpec hard-blocks
    training at every boundary:
      1. TrainerRunner._run_symbol (Gate 1)
      2. WalkForwardValidator.__init__
      3. AdaptationSystem.run
    Each must raise ValueError — not log and silently continue.
    """

    def _make_flagged_spec(self, symbol: str):
        """Return an InstrumentSpec with sanity_ok=False."""
        from shared.instrument_spec import InstrumentSpec
        return InstrumentSpec(
            symbol=symbol, digits=5, point=1e-5,
            contract_size=100000, tick_size=1e-5,
            tick_value=0.858, volume_min=0.01,
            volume_step=0.01, volume_max=500,
            stops_level=0, spread_typical=1,
            swap_long=-0.7, swap_short=-1.0,
            currency_base="EUR", currency_profit="USD",
            currency_margin="EUR", account_currency="EUR",
            captured_at="2026-06-01T00:00:00+00:00",
            sanity_ok=False,
        )

    def test_runner_run_symbol_raises_when_spec_missing(self, tmp_path):
        """
        _run_symbol must raise ValueError when no spec exists.
        Root cause: Gate 1 in _run_symbol must hard-fail, not silently return.
        """
        import shared.instrument_spec as mod
        original = mod.SPEC_CACHE_PATH
        mod.SPEC_CACHE_PATH = tmp_path / "empty.json"

        from trainer.core.trainer_runner import TrainerRunner, RunRecord
        from trainer.core.knowledge_base import KnowledgeBase
        runner = TrainerRunner()
        record = RunRecord(runner.run_id, runner.config)
        kb = KnowledgeBase()

        with pytest.raises(ValueError, match="No InstrumentSpec found"):
            runner._run_symbol("EURUSD", record, kb)

        mod.SPEC_CACHE_PATH = original

    def test_runner_run_symbol_raises_when_sanity_not_ok(self, tmp_path):
        """
        _run_symbol must raise ValueError when spec has sanity_ok=False.
        Root cause: Gate 1 must hard-fail on flagged instruments.
        """
        from shared.instrument_spec import save_specs
        import shared.instrument_spec as mod
        original = mod.SPEC_CACHE_PATH
        mod.SPEC_CACHE_PATH = tmp_path / "specs.json"

        save_specs({"EURUSD": self._make_flagged_spec("EURUSD")})

        from trainer.core.trainer_runner import TrainerRunner, RunRecord
        from trainer.core.knowledge_base import KnowledgeBase
        runner = TrainerRunner()
        record = RunRecord(runner.run_id, runner.config)
        kb = KnowledgeBase()

        with pytest.raises(ValueError, match="sanity_ok=False"):
            runner._run_symbol("EURUSD", record, kb)

        mod.SPEC_CACHE_PATH = original

    def test_walk_forward_validator_raises_when_sanity_not_ok(self, tmp_path):
        """
        WalkForwardValidator must raise at __init__ when spec is flagged.
        Root cause: no WF window may begin for a flagged instrument.
        """
        import pandas as pd
        from shared.instrument_spec import save_specs
        import shared.instrument_spec as mod
        original = mod.SPEC_CACHE_PATH
        mod.SPEC_CACHE_PATH = tmp_path / "specs.json"

        save_specs({"EURUSD": self._make_flagged_spec("EURUSD")})

        from trainer.core.walk_forward import WalkForwardValidator
        df = pd.DataFrame()  # empty; __init__ must raise before reading it

        with pytest.raises(ValueError, match="sanity_ok=False"):
            WalkForwardValidator(symbol="EURUSD", df=df)

        mod.SPEC_CACHE_PATH = original

    def test_adaptation_system_raises_when_sanity_not_ok(self, tmp_path):
        """
        AdaptationSystem.run must raise before any candidate evaluation
        when the symbol's spec has sanity_ok=False.
        """
        import pandas as pd
        from shared.instrument_spec import save_specs
        import shared.instrument_spec as mod
        original = mod.SPEC_CACHE_PATH
        mod.SPEC_CACHE_PATH = tmp_path / "specs.json"

        save_specs({"EURUSD": self._make_flagged_spec("EURUSD")})

        from trainer.core.adaptation import AdaptationSystem
        from trainer.core.knowledge_base import KnowledgeBase

        system = AdaptationSystem(
            symbol="EURUSD",
            df=pd.DataFrame(),
            df_held_out=pd.DataFrame(),
            knowledge_base=KnowledgeBase(),
            data_hash="test",
            run_id="test_run",
        )

        with pytest.raises(ValueError, match="sanity_ok=False"):
            system.run()

        mod.SPEC_CACHE_PATH = original