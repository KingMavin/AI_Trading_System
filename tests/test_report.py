"""
Unit tests for Report Generator (Layer 10).
"""

import pytest
from pathlib import Path
from types import SimpleNamespace

from trainer.core.report import ReportGenerator, generate_report
from trainer.core.trainer_runner import RunRecord
from trainer.core.adaptation import CandidateConfig
from trainer.core.promotion import PromotionDecision


@pytest.fixture
def sample_run_data():
    run_record = SimpleNamespace(
        run_id="test_run_12345",
        symbol="EURUSD",
        outcome="COMPLETED",
        duration_seconds=12.34,
        candidates_promoted=1,
        candidates_flagged=0,
        candidates_rejected=1,
        deployed_strategy_id="EURUSD_ma_crossover_c1",
        layers={
            "layer_1_data": {"status": "COMPLETED", "details": "Loaded 1000 rows"},
            "layer_8_scoring": {"status": "COMPLETED", "details": "Scored 2 candidates"}
        }
    )

    candidate_1 = SimpleNamespace(
        candidate_id="EURUSD_ma_crossover_c1",
        template="ma_crossover",
        gate_status="PASSED",
        composite_score=0.8542,
        deflated_score=0.8210
    )
    candidate_2 = SimpleNamespace(
        candidate_id="EURUSD_ma_crossover_c2",
        template="ma_crossover",
        gate_status="FAILED",
        composite_score=0.3120,
        deflated_score=0.2900
    )
    candidates = [candidate_1, candidate_2]

    decision_1 = SimpleNamespace(
        candidate_id="EURUSD_ma_crossover_c1",
        decision_outcome="AUTO_PROMOTE",
        improvement_pct=15.4,
        recommendation="Passed all hard gates and exceeded baseline score by 15.4%."
    )
    decision_2 = SimpleNamespace(
        candidate_id="EURUSD_ma_crossover_c2",
        decision_outcome="AUTO_REJECT",
        improvement_pct=-20.1,
        recommendation="Failed Gate 2 (Max Drawdown)."
    )
    decisions = [decision_1, decision_2]

    return run_record, candidates, decisions


def test_generate_report_files_created(tmp_path, sample_run_data):
    run_record, candidates, decisions = sample_run_data
    md_path, html_path = generate_report(run_record, candidates, decisions, output_dir=str(tmp_path))

    assert md_path.exists()
    assert html_path.exists()
    assert md_path.name == "report_test_run_12345.md"
    assert html_path.name == "report_test_run_12345.html"
    assert md_path.stat().st_size > 0
    assert html_path.stat().st_size > 0


def test_report_content_structure(tmp_path, sample_run_data):
    run_record, candidates, decisions = sample_run_data
    md_path, html_path = generate_report(run_record, candidates, decisions, output_dir=str(tmp_path))

    md_content = md_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")

    # Assert Markdown structure and content
    assert "# ATS Training Run Report: `test_run_12345`" in md_content
    assert "EURUSD_ma_crossover_c1" in md_content
    assert "AUTO_PROMOTE" in md_content
    assert "0.8542" in md_content
    assert "Passed all hard gates" in md_content

    # Assert HTML structure and content
    assert "<title>ATS Training Report - test_run_12345</title>" in html_content
    assert "EURUSD_ma_crossover_c1" in html_content
    assert "AUTO_PROMOTE" in html_content
    assert "badge-promoted" in html_content
    assert "0.8542" in html_content


def test_report_empty_candidates_handling(tmp_path):
    run_record = SimpleNamespace(
        run_id="empty_run_999",
        symbol="GBPUSD",
        status="COMPLETED",
        duration_seconds=1.2,
        candidates_promoted=0,
        candidates_flagged=0,
        candidates_rejected=0,
        layers={}
    )

    md_path, html_path = generate_report(run_record, [], [], output_dir=str(tmp_path))
    assert md_path.exists()
    assert html_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")

    assert "_No candidates evaluated in this run._" in md_content
    assert "No candidates evaluated in this run" in html_content


def test_report_directory_creation(tmp_path):
    sub_dir = tmp_path / "nested" / "reports"
    assert not sub_dir.exists()

    gen = ReportGenerator(output_dir=str(sub_dir))
    assert sub_dir.exists()


def test_report_integration_real_objects(tmp_path):
    """
    Integration test passing REAL production dataclass & class instances:
    RunRecord, CandidateConfig, and PromotionDecision.
    Verifies that real attribute names (template, candidate_id, recommendation, outcome)
    are correctly read without falling back to defaults.
    """
    record = RunRecord(
        run_id="real_run_888",
        config={'symbol': 'USDJPY', 'timeframe': 'M15'}
    )
    record.outcome = 'COMPLETED'
    record.candidates_promoted = 1

    candidate = CandidateConfig(
        candidate_id="USDJPY_ma_crossover_c1",
        template="ma_crossover",
        symbol="USDJPY",
        timeframe="M15",
        parameters={'fast_period': 10, 'slow_period': 20},
        composite_score=0.7891,
        gate_status="PASSED",
        spec_source="CAPTURED"
    )

    decision = PromotionDecision(
        candidate_id="USDJPY_ma_crossover_c1",
        composite_score=0.7891,
        baseline_score=0.6000,
        improvement_pct=31.5,
        decision_path="AUTO_PROMOTE",
        decision_outcome="PROMOTED",
        recommendation="Passed all conditions with 31.5% improvement over baseline."
    )

    md_path, html_path = generate_report(record, [candidate], [decision], output_dir=str(tmp_path))

    md_content = md_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")

    # Assert real attributes were read properly
    assert "real_run_888" in md_content
    assert "USDJPY" in md_content
    assert "COMPLETED" in md_content
    assert "USDJPY_ma_crossover_c1" in md_content
    assert "ma_crossover" in md_content  # Reads candidate.template
    assert "PROMOTED" in md_content       # Reads decision.decision_outcome
    assert "+31.50%" in md_content
    assert "Passed all conditions with 31.5% improvement" in md_content  # Reads decision.recommendation

    assert "real_run_888" in html_content
    assert "USDJPY_ma_crossover_c1" in html_content
    assert "ma_crossover" in html_content
    assert "PROMOTED" in html_content


def test_report_simulation_default_badge_rendering(tmp_path):
    """
    Verify that candidates using SIMULATION_DEFAULT render with clear visual emphasis
    (**SIMULATION_DEFAULT** in Markdown, badge-sim-default in HTML).
    """
    record = RunRecord(
        run_id="sim_def_run_777",
        config={'symbol': 'AUDUSD'}
    )
    candidate = CandidateConfig(
        candidate_id="AUDUSD_ma_crossover_c1",
        template="ma_crossover",
        symbol="AUDUSD",
        timeframe="M15",
        parameters={'fast_period': 10, 'slow_period': 20},
        composite_score=0.6500,
        gate_status="PASSED",
        spec_source="SIMULATION_DEFAULT"
    )

    md_path, html_path = generate_report(record, [candidate], [], output_dir=str(tmp_path))

    md_content = md_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")

    assert "**SIMULATION_DEFAULT**" in md_content
    assert '<span class="badge-sim-default">SIMULATION_DEFAULT</span>' in html_content
