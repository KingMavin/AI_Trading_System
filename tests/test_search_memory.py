import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import pytest
import numpy as np
import optuna
from trainer.core.knowledge_base import KnowledgeBase
from trainer.core.parameter_grid import TEMPLATE_SPACES


def test_search_history_logging(tmp_path):
    kb_file = tmp_path / "knowledge_base.json"
    kb = KnowledgeBase(base_path=str(kb_file))

    params = {
        'fast_ma_period': 10,
        'slow_ma_period': 50,
        'sl_atr_multiple': 1.5,
        'tp_rr_ratio': 2.0,
        'adx_threshold': 20,
        'pip_size': 0.0001,
        'risk_per_trade_pct': 1.0,
    }

    kb.record_search_trial(
        template="ma_crossover",
        symbol="EURUSD",
        run_id="run_test_001",
        data_hash="hash12345",
        trial_number=0,
        params=params,
        is_coherent=True,
        composite_score=0.45,
    )

    search_log = tmp_path / "search_history.jsonl"
    assert search_log.exists()

    with open(search_log) as f:
        lines = f.readlines()

    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec['template'] == "ma_crossover"
    assert rec['symbol'] == "EURUSD"
    assert rec['composite_score'] == 0.45
    assert 'adx_threshold' in rec['parameters']
    assert rec['parameters']['adx_threshold'] == 20
    assert 'pip_size' not in rec['parameters']


def test_region_bucketing_min_samples(tmp_path):
    kb_file = tmp_path / "knowledge_base.json"
    kb = KnowledgeBase(base_path=str(kb_file))

    params = {
        'fast_ma_period': 10,
        'slow_ma_period': 50,
        'sl_atr_multiple': 1.5,
        'tp_rr_ratio': 2.0,
        'adx_threshold': 20,
    }

    # 1. Less than 30 samples -> min_samples_met should be False
    for i in range(15):
        kb.record_search_trial(
            template="ma_crossover",
            symbol="EURUSD",
            run_id="run_test",
            data_hash="hash123",
            trial_number=i,
            params=params,
            is_coherent=True,
            composite_score=0.3 + i * 0.01,
        )

    stats_sparse = kb.get_region_statistics("ma_crossover", "EURUSD", min_samples=30)
    assert stats_sparse['total_samples'] == 15
    assert stats_sparse['min_samples_met'] is False
    assert stats_sparse['region_scores'] == {}

    # 2. Add 15 more samples (total 30) -> min_samples_met should be True
    for i in range(15, 30):
        kb.record_search_trial(
            template="ma_crossover",
            symbol="EURUSD",
            run_id="run_test",
            data_hash="hash123",
            trial_number=i,
            params=params,
            is_coherent=True,
            composite_score=0.3 + i * 0.01,
        )

    stats_sufficient = kb.get_region_statistics("ma_crossover", "EURUSD", min_samples=30)
    assert stats_sufficient['total_samples'] == 30
    assert stats_sufficient['min_samples_met'] is True
    assert len(stats_sufficient['region_scores']) > 0


def test_cold_start_empty_history(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO)
    kb_file = tmp_path / "knowledge_base.json"
    kb = KnowledgeBase(base_path=str(kb_file))

    top = kb.get_top_historical_candidates("ma_crossover", "EURUSD", top_k=5)
    assert top == []
    assert "KB_WARMSTART_SPARSE" in caplog.text


def test_deprioritize_never_exclude():
    """
    Verify that calling study.enqueue_trial with 5 top historical candidates
    in Optuna's TPESampler does not starve unexplored parameter regions over a 35-trial run.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = TEMPLATE_SPACES['ma_crossover']
    enqueued_candidates = [
        {'fast_ma_period': 5, 'slow_ma_period': 30, 'sl_atr_multiple': 1.0, 'tp_rr_ratio': 1.5, 'adx_threshold': 0},
        {'fast_ma_period': 10, 'slow_ma_period': 50, 'sl_atr_multiple': 1.5, 'tp_rr_ratio': 2.0, 'adx_threshold': 15},
        {'fast_ma_period': 10, 'slow_ma_period': 80, 'sl_atr_multiple': 2.0, 'tp_rr_ratio': 2.5, 'adx_threshold': 20},
        {'fast_ma_period': 15, 'slow_ma_period': 50, 'sl_atr_multiple': 1.5, 'tp_rr_ratio': 2.0, 'adx_threshold': 25},
        {'fast_ma_period': 15, 'slow_ma_period': 100, 'sl_atr_multiple': 2.0, 'tp_rr_ratio': 2.5, 'adx_threshold': 30},
    ]

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    for c in enqueued_candidates:
        study.enqueue_trial(c)

    sampled_params = []

    def dummy_objective(trial):
        p = {}
        for param, values in space.items():
            if isinstance(values, (list, tuple)):
                p[param] = trial.suggest_categorical(param, values)
            else:
                p[param] = values
        sampled_params.append(p)
        if p['fast_ma_period'] in [5, 10, 15]:
            return 0.1
        return 0.8

    study.optimize(dummy_objective, n_trials=35)

    assert len(sampled_params) == 35

    # Check trials landing on fast_ma_period == 20 (which is outside the enqueued fast_ma_period set of [5, 10, 15])
    unexplored_fast = [p['fast_ma_period'] for p in sampled_params if p['fast_ma_period'] == 20]
    assert len(unexplored_fast) >= 1, "TPESampler failed to explore parameters outside enqueued set"
