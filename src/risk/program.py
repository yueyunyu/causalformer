"""Answer-free risk-program compiler, validator, and deterministic executor.

The program describes a graph and typed scenario queries. It intentionally
contains no observed repayment label, model probability, or final decision.
Those values are produced only by an executor after validation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence


PROGRAM_TYPE = "answer_free_credit_risk_dual_program_v1"

FORBIDDEN_FUTURE_FIELDS = frozenset(
    {
        "loan_status",
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        "hardship_flag",
        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "deferral_term",
        "hardship_amount",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "hardship_length",
        "hardship_dpd",
        "hardship_loan_status",
        "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
        "debt_settlement_flag",
        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
    }
)

FORBIDDEN_ANSWER_KEYS = frozenset(
    {
        "answer",
        "final_answer",
        "actual_outcome",
        "default_probability",
        "predicted_probability",
        "predicted_label",
        "repayment_label",
    }
)

MUTABLE_SCENARIO_FIELDS = frozenset({"loan_amnt", "dti", "revol_util"})

GRAPH_NODES = (
    ("annual_inc", "observed", "Annual income at application"),
    ("loan_amnt", "decision_input", "Requested principal"),
    ("term", "decision_input", "Requested repayment term"),
    ("int_rate", "platform_decision", "Offered annual interest rate"),
    ("installment", "derived", "Contractual monthly installment"),
    ("dti", "observed", "Debt-to-income ratio at application"),
    ("revol_util", "observed", "Revolving credit utilization"),
    ("fico_score", "observed", "Application-time FICO midpoint"),
    ("delinq_2yrs", "observed", "Recent delinquency count"),
    ("pub_rec", "observed", "Public derogatory record count"),
    ("credit_history_months", "derived", "Observed credit-history length"),
    ("payment_burden", "mechanism", "Monthly contractual burden"),
    ("credit_profile", "mechanism", "Repayment-history and bureau profile"),
    ("liquidity_pressure", "mechanism", "Current debt and revolving pressure"),
    ("platform_pricing", "mechanism", "Lender underwriting and pricing"),
    ("default_within_term", "outcome", "Terminal charged-off/default outcome"),
)

GRAPH_EDGES = (
    ("loan_amnt", "installment", "contract"),
    ("term", "installment", "contract"),
    ("int_rate", "installment", "contract"),
    ("installment", "payment_burden", "burden"),
    ("annual_inc", "payment_burden", "capacity"),
    ("dti", "liquidity_pressure", "debt_load"),
    ("revol_util", "liquidity_pressure", "utilization"),
    ("fico_score", "credit_profile", "bureau_signal"),
    ("delinq_2yrs", "credit_profile", "repayment_history"),
    ("pub_rec", "credit_profile", "repayment_history"),
    ("credit_history_months", "credit_profile", "history_depth"),
    ("credit_profile", "platform_pricing", "underwriting"),
    ("liquidity_pressure", "platform_pricing", "underwriting"),
    ("platform_pricing", "int_rate", "price_assignment"),
    ("payment_burden", "default_within_term", "predictive_path"),
    ("credit_profile", "default_within_term", "predictive_path"),
    ("liquidity_pressure", "default_within_term", "predictive_path"),
    ("platform_pricing", "default_within_term", "predictive_path"),
)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": list(self.errors), "warnings": list(self.warnings)}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _scenario_queries(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    loan_amnt = _finite(case.get("loan_amnt"))
    dti = _finite(case.get("dti"))
    revol_util = _finite(case.get("revol_util"))
    if loan_amnt and loan_amnt > 1000:
        queries.append(
            {
                "query_id": "requested_amount_minus_10pct",
                "treatment": "loan_amnt",
                "operation": "set",
                "proposed_value": round(max(1000.0, loan_amnt * 0.90), 2),
                "outcome": "default_within_term",
                "estimand": "individual_model_scenario_delta",
                "business_action": "Re-underwrite a 10% lower requested principal",
            }
        )
    if dti is not None and dti > 5:
        queries.append(
            {
                "query_id": "dti_minus_5pp",
                "treatment": "dti",
                "operation": "set",
                "proposed_value": round(max(0.0, dti - 5.0), 2),
                "outcome": "default_within_term",
                "estimand": "individual_model_scenario_delta",
                "business_action": "Reassess after documented debt reduction",
            }
        )
    if revol_util is not None and revol_util > 10:
        queries.append(
            {
                "query_id": "revolving_utilization_minus_10pp",
                "treatment": "revol_util",
                "operation": "set",
                "proposed_value": round(max(0.0, revol_util - 10.0), 2),
                "outcome": "default_within_term",
                "estimand": "individual_model_scenario_delta",
                "business_action": "Reassess after verified revolving-balance reduction",
            }
        )
    return queries


def compile_risk_program(case: Mapping[str, Any], *, case_id: str) -> dict[str, Any]:
    """Compile one application into an answer-free graph/query program."""
    return {
        "schema_version": 1,
        "program_type": PROGRAM_TYPE,
        "case_id": str(case_id),
        "graph_program": {
            "nodes": [
                {"id": node_id, "kind": kind, "description": description}
                for node_id, kind, description in GRAPH_NODES
            ],
            "edges": [
                {"source": source, "target": target, "mechanism": mechanism}
                for source, target, mechanism in GRAPH_EDGES
            ],
            "mechanisms": [
                {
                    "id": "payment_burden_ratio",
                    "inputs": ["installment", "annual_inc"],
                    "output": "payment_burden",
                    "executor": "installment * 12 / annual_inc",
                },
                {
                    "id": "validated_risk_model",
                    "inputs": ["application_time_features"],
                    "output": "default_within_term",
                    "executor": "frozen_calibrated_catboost",
                },
            ],
        },
        "query_program": {
            "observational_query": {
                "outcome": "default_within_term",
                "estimand": "calibrated_default_probability",
                "execution_policy": "frozen_calibrated_catboost",
            },
            "scenario_queries": _scenario_queries(case),
            "causal_status": {
                "identified": False,
                "status": "predictive_scenario_only",
                "reason": (
                    "Historical accepted-loan data contain no randomized policy action or sufficient "
                    "treatment-assignment record. Scenario deltas are model responses, not causal effects."
                ),
            },
        },
        "audit_policy": {
            "allowed_information_time": "application_or_pricing_time",
            "forbidden_future_fields": sorted(FORBIDDEN_FUTURE_FIELDS),
            "manual_review_on_invalid_program": True,
            "adverse_action_notice_requires_human_review": True,
        },
    }


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield child_path, str(key), child
            yield from _walk(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _has_cycle(nodes: set[str], edges: Sequence[Mapping[str, Any]]) -> bool:
    adjacency = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source in nodes and target in nodes:
            adjacency[source].append(target)
            indegree[target] += 1
    queue = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited != len(nodes)


def validate_risk_program(program: Mapping[str, Any]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if program.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if program.get("program_type") != PROGRAM_TYPE:
        errors.append(f"program_type must equal {PROGRAM_TYPE}")
    if not str(program.get("case_id", "")).strip():
        errors.append("case_id is required")

    for path, key, value in _walk(program):
        lowered = key.lower()
        if lowered in FORBIDDEN_ANSWER_KEYS:
            errors.append(f"answer-bearing key is forbidden at {path}")
        if lowered in FORBIDDEN_FUTURE_FIELDS and ".forbidden_future_fields[" not in path:
            errors.append(f"future-information field is forbidden at {path}")
        if (
            isinstance(value, str)
            and value.lower() in FORBIDDEN_FUTURE_FIELDS
            and ".forbidden_future_fields[" not in path
        ):
            errors.append(f"future-information field reference is forbidden at {path}")

    graph = program.get("graph_program")
    if not isinstance(graph, Mapping):
        errors.append("graph_program must be an object")
        graph = {}
    nodes_raw = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not isinstance(nodes_raw, list) or not nodes_raw:
        errors.append("graph_program.nodes must be a non-empty list")
        nodes_raw = []
    if not isinstance(edges, list) or not edges:
        errors.append("graph_program.edges must be a non-empty list")
        edges = []
    node_ids = [item.get("id") for item in nodes_raw if isinstance(item, Mapping)]
    node_set = {str(item) for item in node_ids if item}
    if len(node_set) != len(node_ids):
        errors.append("graph node ids must be unique and non-empty")
    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            errors.append(f"edge {index} must be an object")
            continue
        if edge.get("source") not in node_set or edge.get("target") not in node_set:
            errors.append(f"edge {index} references an unknown node")
    if node_set and _has_cycle(node_set, edges):
        errors.append("graph_program must be acyclic")

    query = program.get("query_program")
    if not isinstance(query, Mapping):
        errors.append("query_program must be an object")
        query = {}
    scenarios = query.get("scenario_queries", [])
    if not isinstance(scenarios, list):
        errors.append("scenario_queries must be a list")
        scenarios = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            errors.append(f"scenario {index} must be an object")
            continue
        treatment = scenario.get("treatment")
        if treatment not in MUTABLE_SCENARIO_FIELDS:
            errors.append(f"scenario {index} treatment is not mutable: {treatment}")
        if scenario.get("operation") != "set":
            errors.append(f"scenario {index} operation must be set")
        if _finite(scenario.get("proposed_value")) is None:
            errors.append(f"scenario {index} proposed_value must be finite")
        if scenario.get("outcome") != "default_within_term":
            errors.append(f"scenario {index} has an unsupported outcome")
    causal = query.get("causal_status", {})
    if isinstance(causal, Mapping) and causal.get("identified") is not False:
        warnings.append("This observational protocol should not claim identified causal effects")
    return ValidationResult(not errors, tuple(errors), tuple(warnings))


def _update_derived_features(case: dict[str, Any], changed_field: str) -> None:
    income = _finite(case.get("annual_inc"))
    loan = _finite(case.get("loan_amnt"))
    installment = _finite(case.get("installment"))
    if changed_field == "loan_amnt" and loan is not None:
        original_loan = _finite(case.get("_original_loan_amnt"))
        original_installment = _finite(case.get("_original_installment"))
        if original_loan and original_installment is not None:
            case["installment"] = original_installment * loan / original_loan
            installment = _finite(case["installment"])
    if income and income > 0:
        case["loan_to_income"] = (loan / income) if loan is not None else None
        case["installment_to_income"] = (
            installment * 12.0 / income if installment is not None else None
        )


def execute_risk_program(
    program: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    predict_probability: Callable[[Mapping[str, Any]], float],
) -> dict[str, Any]:
    """Validate and execute observational and model-scenario queries."""
    validation = validate_risk_program(program)
    if not validation.valid:
        return {
            "schema_version": 1,
            "case_id": str(program.get("case_id", "")),
            "status": "manual_review_invalid_program",
            "validation": validation.as_dict(),
            "observational_result": None,
            "scenario_results": [],
        }

    baseline_case = dict(case)
    baseline_case["_original_loan_amnt"] = baseline_case.get("loan_amnt")
    baseline_case["_original_installment"] = baseline_case.get("installment")
    baseline = float(predict_probability(baseline_case))
    scenarios: list[dict[str, Any]] = []
    for query in program["query_program"]["scenario_queries"]:
        changed = deepcopy(baseline_case)
        treatment = str(query["treatment"])
        old_value = _finite(changed.get(treatment))
        changed[treatment] = float(query["proposed_value"])
        _update_derived_features(changed, treatment)
        scenario_probability = float(predict_probability(changed))
        scenarios.append(
            {
                "query_id": query["query_id"],
                "treatment": treatment,
                "observed_value": old_value,
                "proposed_value": float(query["proposed_value"]),
                "baseline_probability": baseline,
                "scenario_probability": scenario_probability,
                "model_risk_delta": scenario_probability - baseline,
                "risk_reduction": baseline - scenario_probability,
                "direction_check": "lower_risk" if scenario_probability < baseline else "no_improvement",
                "business_action": query["business_action"],
                "causal_effect_claimed": False,
            }
        )
    return {
        "schema_version": 1,
        "case_id": str(program["case_id"]),
        "status": "executed_predictive_scenarios",
        "validation": validation.as_dict(),
        "observational_result": {"calibrated_default_probability": baseline},
        "scenario_results": scenarios,
        "causal_status": program["query_program"]["causal_status"],
    }
