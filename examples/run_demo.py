"""End-to-end walkthrough: compile -> validate -> execute -> audit summary.

Runs offline with no model and no data file. Prints an audit-style report so you
can see exactly what the framework produces for a single application.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from risk.program import (  # noqa: E402
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
    "fico_score": 692.0,
}


def toy_probability(case) -> float:
    """A transparent stand-in 'model'.

    Deliberately simple so the demo needs no trained artifact. In a real
    deployment you pass your calibrated model's predict_proba here.

    Reads the raw application fields only (dti, revol_util, loan_amnt,
    annual_inc), so that perturbing a lever moves the score in the expected
    direction: lower principal / lower debt / lower utilization -> lower risk.
    """
    dti = float(case.get("dti") or 0.0)
    util = float(case.get("revol_util") or 0.0)
    loan = float(case.get("loan_amnt") or 0.0)
    income = float(case.get("annual_inc") or 0.0)

    lti = (loan / income) if income > 0 else 0.0  # ~0.39 for the demo case
    score = 0.010 * dti + 0.004 * util + 0.90 * lti
    # squash into (0, 1); score ~1.15 for the demo case
    return 1.0 / (1.0 + 2.718281828 ** (-(score - 1.15)))


def main() -> None:
    program = compile_risk_program(CASE, case_id="demo-0001")

    print("=" * 72)
    print("STEP 1  compiled program (audit policy excerpt)")
    print("=" * 72)
    policy = program["audit_policy"]
    print(f"  allowed_information_time : {policy['allowed_information_time']}")
    print(f"  forbidden future fields  : {len(policy['forbidden_future_fields'])}")
    print(f"  graph nodes / edges      : "
          f"{len(program['graph_program']['nodes'])} / {len(program['graph_program']['edges'])}")

    print()
    print("=" * 72)
    print("STEP 2  validation")
    print("=" * 72)
    result = validate_risk_program(program)
    print(f"  valid    : {result.valid}")
    print(f"  errors   : {list(result.errors)}")
    print(f"  warnings : {list(result.warnings)}")

    if not result.valid:
        print("\nProgram rejected. No answer is produced. This is by design.")
        return

    print()
    print("=" * 72)
    print("STEP 3  deterministic execution")
    print("=" * 72)
    out = execute_risk_program(program, CASE, predict_probability=toy_probability)
    print(f"  status      : {out['status']}")
    print(f"  probability : {out['observational_result']['calibrated_default_probability']:.6f}")
    print()
    print("  scenario results")
    for scenario in out["scenario_results"]:
        print(f"    [{scenario['query_id']}]")
        print(f"      {scenario['treatment']}: "
              f"{scenario['observed_value']} -> {scenario['proposed_value']}")
        print(f"      probability: {scenario['baseline_probability']:.6f} "
              f"-> {scenario['scenario_probability']:.6f} "
              f"(delta {scenario['model_risk_delta']:+.6f})")
        print(f"      business action: {scenario['business_action']}")
        print(f"      causal effect claimed: {scenario['causal_effect_claimed']}")

    print()
    print("=" * 72)
    print("STEP 4  causal boundary (written into the artifact)")
    print("=" * 72)
    causal = out["causal_status"]
    print(f"  identified : {causal['identified']}")
    print(f"  status     : {causal['status']}")
    print(f"  reason     : {causal['reason']}")

    print()
    print("=" * 72)
    print("STEP 5  reproducibility check")
    print("=" * 72)
    again = execute_risk_program(program, CASE, predict_probability=toy_probability)
    print(f"  re-run identical: {again == out}")

    demo_dir = ROOT / "demos"
    if (demo_dir / "execution_results.jsonl").exists():
        print()
        print("=" * 72)
        print("shipped demo artifacts")
        print("=" * 72)
        rows = [json.loads(line) for line in
                (demo_dir / "execution_results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        valid = sum(1 for r in rows if r.get("validation", {}).get("valid"))
        print(f"  cases         : {len(rows)}")
        print(f"  all valid     : {valid == len(rows)}")
        print(f"  causal claims : "
              f"{sum(1 for r in rows if r.get('causal_status', {}).get('identified'))}")


if __name__ == "__main__":
    main()
