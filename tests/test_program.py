"""Unit tests for the risk program validator and deterministic executor."""

from __future__ import annotations

import copy

import pytest

from risk.program import (
    FORBIDDEN_ANSWER_KEYS,
    PROGRAM_TYPE,
    compile_risk_program,
    execute_risk_program,
    validate_risk_program,
)

CASE = {
    "loan_amnt": 25000.0,
    "dti": 35.78,
    "revol_util": 62.0,
    "annual_inc": 64000.0,
    "installment": 780.0,
}


def fixed_probability(case):
    """Deterministic stand-in for a calibrated model, used only in tests."""
    return 0.5


def test_compiled_program_is_valid():
    program = compile_risk_program(CASE, case_id="t-1")
    result = validate_risk_program(program)
    assert result.valid, result.errors
    assert program["program_type"] == PROGRAM_TYPE


def test_rejects_cyclic_graph():
    program = compile_risk_program(CASE, case_id="t-2")
    broken = copy.deepcopy(program)
    broken["graph_program"]["edges"].append(
        {"source": "default_within_term", "target": "annual_inc", "mechanism": "backedge"}
    )
    result = validate_risk_program(broken)
    assert not result.valid
    assert any("acyclic" in e for e in result.errors)


def test_rejects_unknown_edge_endpoint():
    program = compile_risk_program(CASE, case_id="t-3")
    broken = copy.deepcopy(program)
    broken["graph_program"]["edges"].append(
        {"source": "not_a_node", "target": "annual_inc", "mechanism": "dangling"}
    )
    result = validate_risk_program(broken)
    assert not result.valid
    assert any("unknown node" in e for e in result.errors)


def test_rejects_answer_bearing_key():
    program = compile_risk_program(CASE, case_id="t-4")
    broken = copy.deepcopy(program)
    broken["query_program"]["predicted_label"] = 1
    result = validate_risk_program(broken)
    assert not result.valid
    assert any("answer-bearing key" in e for e in result.errors)


def test_rejects_future_information_field():
    program = compile_risk_program(CASE, case_id="t-5")
    broken = copy.deepcopy(program)
    broken["leaked"] = {"loan_status": "Charged Off"}
    result = validate_risk_program(broken)
    assert not result.valid
    assert any("future-information" in e for e in result.errors)


def test_forbidden_sets_are_non_trivial():
    # guard against accidental gutting of the policy
    assert len(FORBIDDEN_ANSWER_KEYS) >= 5
    program = compile_risk_program(CASE, case_id="t-6")
    forbidden = program["audit_policy"]["forbidden_future_fields"]
    # the shipped policy forbids the full post-origination field set
    assert len(forbidden) >= 35


def test_invalid_program_returns_manual_review_status():
    program = compile_risk_program(CASE, case_id="t-7")
    broken = copy.deepcopy(program)
    broken["schema_version"] = 999
    out = execute_risk_program(broken, CASE, predict_probability=fixed_probability)
    assert out["status"] == "manual_review_invalid_program"
    assert out["observational_result"] is None
    assert out["scenario_results"] == []


def test_executor_never_claims_causal_effect():
    program = compile_risk_program(CASE, case_id="t-8")
    out = execute_risk_program(program, CASE, predict_probability=fixed_probability)
    assert out["status"] == "executed_predictive_scenarios"
    assert out["causal_status"]["identified"] is False
    for scenario in out["scenario_results"]:
        assert scenario["causal_effect_claimed"] is False


def test_executor_is_deterministic():
    program = compile_risk_program(CASE, case_id="t-9")
    a = execute_risk_program(program, CASE, predict_probability=fixed_probability)
    b = execute_risk_program(program, CASE, predict_probability=fixed_probability)
    assert a == b


def test_scenario_delta_is_reported_with_direction():
    program = compile_risk_program(CASE, case_id="t-10")
    calls = {"n": 0}

    def decreasing_probability(case):
        # first call is the baseline, later calls are scenarios
        calls["n"] += 1
        return 0.5 if calls["n"] == 1 else 0.4

    out = execute_risk_program(program, CASE, predict_probability=decreasing_probability)
    assert out["scenario_results"], "expected at least one scenario"
    first = out["scenario_results"][0]
    assert first["baseline_probability"] == pytest.approx(0.5)
    assert first["scenario_probability"] == pytest.approx(0.4)
    assert first["risk_reduction"] == pytest.approx(0.1)
    assert first["direction_check"] == "lower_risk"


def test_rejects_non_mutable_treatment():
    program = compile_risk_program(CASE, case_id="t-11")
    broken = copy.deepcopy(program)
    broken["query_program"]["scenario_queries"][0]["treatment"] = "fico_score"
    result = validate_risk_program(broken)
    assert not result.valid
    assert any("not mutable" in e for e in result.errors)
