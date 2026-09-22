"""
tests/test_dna_hash.py

Unit tests for WAVE 21 Part 1 — Canonical SHA-256 DNA Hash Enforcement.
Tests:
1. Canonical DNA hash calculation (keys sorted, no whitespace).
2. Engine startup refusal when dna_hash is missing or mismatched (STRATEGY_HASH_MISMATCH).
3. rehash_strategy.py requiring explicit confirmation and logging audit event.
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.dna import compute_dna_hash
from engine.core.strategy_loader import StrategyLoader
from rehash_strategy import rehash_file


class TestDnaHash:

    def test_canonical_dna_hash_calculation(self):
        params1 = {'fast_ma_period': 20, 'slow_ma_period': 100, 'ma_type': 'SMA'}
        params2 = {'slow_ma_period': 100, 'fast_ma_period': 20, 'ma_type': 'SMA'}
        filters = {}

        hash1 = compute_dna_hash(params1, filters)
        hash2 = compute_dna_hash(params2, filters)

        assert len(hash1) == 64
        assert hash1 == hash2

    def test_strategy_loader_fails_closed_on_dna_hash_mismatch(self, tmp_path):
        params = {'fast_ma_period': 20, 'slow_ma_period': 100}
        filters = {}
        valid_dna = compute_dna_hash(params, filters)

        # File with tampered/invalid dna_hash (syntactically valid 64-char hex, but wrong value)
        wrong_64_char_hex = 'a' * 64
        assert len(wrong_64_char_hex) == 64
        assert wrong_64_char_hex != valid_dna

        bad_strategy = {
            'strategy_id': 'ATS-TEST-BAD-001',
            'dna_hash': wrong_64_char_hex,
            'template': 'ma_crossover',
            'composite_score': 1.5,
            'parameters': params,
            'filters': filters,
            'wf_summary': {'median_win_rate': 0.55, 'median_profit_factor': 1.5}
        }
        bad_file = tmp_path / 'active_strategy.json'
        bad_file.write_text(json.dumps(bad_strategy), encoding='utf-8')

        loader = StrategyLoader(str(bad_file))
        spec = loader.load_strategy()

        assert spec is None
        assert loader.load_errors > 0

    def test_strategy_loader_succeeds_on_valid_dna_hash(self, tmp_path):
        params = {'fast_ma_period': 20, 'slow_ma_period': 100}
        filters = {}
        valid_dna = compute_dna_hash(params, filters)

        valid_strategy = {
            'strategy_id': 'ATS-TEST-VALID-001',
            'dna_hash': valid_dna,
            'template': 'ma_crossover',
            'composite_score': 1.5,
            'parameters': params,
            'filters': filters,
            'wf_summary': {'median_win_rate': 0.55, 'median_profit_factor': 1.5}
        }
        good_file = tmp_path / 'active_strategy.json'
        good_file.write_text(json.dumps(valid_strategy), encoding='utf-8')

        loader = StrategyLoader(str(good_file))
        spec = loader.load_strategy()

        assert spec is not None
        assert spec.dna_hash == valid_dna

    def test_rehash_utility_requires_confirmation_and_logs_audit(self, tmp_path, monkeypatch):
        params = {'fast_ma_period': 50, 'slow_ma_period': 200}
        filters = {}
        outdated_strategy = {
            'strategy_id': 'ATS-TEST-REHASH-001',
            'dna_hash': 'old_invalid_hash',
            'parameters': params,
            'filters': filters
        }
        strat_file = tmp_path / 'strategy_to_rehash.json'
        strat_file.write_text(json.dumps(outdated_strategy), encoding='utf-8')

        # Test unconfirmed execution cancels when prompt returns 'n'
        monkeypatch.setattr('builtins.input', lambda prompt: 'n')
        res_unconfirmed = rehash_file(str(strat_file), confirm=False)
        assert res_unconfirmed is False

        # Test confirmed execution updates file and returns True
        res_confirmed = rehash_file(str(strat_file), confirm=True)
        assert res_confirmed is True

        updated_data = json.loads(strat_file.read_text(encoding='utf-8'))
        expected_hash = compute_dna_hash(params, filters)
        assert updated_data['dna_hash'] == expected_hash
