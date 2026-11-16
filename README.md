# Auditable Risk Programs

**Turn a credit-risk decision from "a probability number" into "a checkable, executable, auditable program."**

Most risk models cannot answer the question every regulator, auditor, and business owner eventually asks:

> *"Why was this application declined?"*

"Score 0.72 exceeded the threshold of 0.65" is not an explanation — it is a restatement.
"FICO is low and DTI is high" is feature importance, not decision logic.
And if you re-run the same application twice, you may not get the same answer.

This project separates **decision generation** from **decision computation**:

```
application data
      │
      ▼
 program compiler  ──►  typed risk program (JSON)
                              │
                              ▼
                        validator  ──failure──► ⊥  (manual review, counted, never silently dropped)
                              │ pass
                              ▼
                  deterministic executor
                              │
                              ▼
        calibrated probability + scenario deltas + audit trail
```

The model (or the compiler) only *writes the program*. The number is computed by a fixed executor. That is what makes the result reproducible and auditable.

---

## What this is — and what it is not

| This is | This is not |
|---|---|
| An **auditability** framework: validation, deterministic execution, explicit failure | A better classifier |
| A way to show **per-case scenario deltas** with an explicit causal disclaimer | A causal-inference engine |
| A prototype with **real data, real metrics, real artifacts** | A production lending system |

**We do not claim a higher AUC.** We claim that a decision of equal accuracy becomes inspectable.

---

## Three design rules that produce the audit trail

### 1. Future information is structurally forbidden

The validator rejects more than 40 fields that are only observable *after* origination — `loan_status`, `total_pymnt`, `out_prncp`, `recoveries`, `debt_settlement_flag`, `hardship_*`, and so on. See `FORBIDDEN_FUTURE_FIELDS` in [`src/risk/program.py`](src/risk/program.py).

A program that references them is rejected before execution. There is no "best effort" fallback.

### 2. Answer-bearing keys are forbidden

`answer`, `final_answer`, `predicted_label`, `default_probability`, `actual_outcome` — the program cannot carry its own conclusion. The conclusion must be produced by the executor.

### 3. Failures stay in the denominator

An invalid program is not repaired, not retried silently, and not dropped. It returns `status: "manual_review_invalid_program"` and stays in the accounting. Errors must be visible, not filtered.

### Bonus: the causal boundary is written into the artifact

Every executed result carries:

```json
"causal_status": {
  "identified": false,
  "status": "predictive_scenario_only",
  "reason": "Historical accepted-loan data contain no randomized policy action or sufficient treatment-assignment record. Scenario deltas are model responses, not causal effects."
}
```

The system states what it does **not** know. That is the single most important property for regulated use.

---

## A real case

From [`demos/execution_results.jsonl`](demos/execution_results.jsonl) (`case_id: 5754ebf05516b48a`):

```json
{
  "status": "executed_predictive_scenarios",
  "validation": { "valid": true, "errors": [], "warnings": [] },
  "observational_result": { "calibrated_default_probability": 0.8932 },

  "scenario_results": [
    {
      "query_id": "requested_amount_minus_10pct",
      "treatment": "loan_amnt",
      "observed_value": 25000.0,
      "proposed_value": 22500.0,
      "baseline_probability": 0.8932,
      "scenario_probability": 0.8806,
      "model_risk_delta": -0.0126,
      "business_action": "Re-underwrite a 10% lower requested principal",
      "causal_effect_claimed": false
    },
    {
      "query_id": "dti_minus_5pp",
      "treatment": "dti",
      "observed_value": 35.78,
      "proposed_value": 30.78,
      "scenario_probability": 0.8688,
      "business_action": "Reassess after documented debt reduction",
      "causal_effect_claimed": false
    }
  ]
}
```

What this buys you over the bare number `0.8932`:

- **Business** sees what levers do: −10% principal, −5pp DTI → how much risk moves
- **Compliance** can re-run it: the executor is deterministic
- **Audit** gets a full trace: treatment, baseline, proposed value, delta, business action
- **The boundary is hard-coded**: `causal_effect_claimed: false` on every scenario

Open [`demos/causalrisk_interview_demo.html`](demos/causalrisk_interview_demo.html) in a browser for the visual version.

---

## Baseline performance (honest)

Test set: 150,000 applications, **chronological split** (train 400k / valid 100k / test 150k). All models calibrated.

| Model | ROC-AUC | PR-AUC | Brier | ECE₁₀ | KS | Top-decile lift |
|---|---|---|---|---|---|---|
| Logistic (application-only) | 0.6820 | 0.3585 | 0.1556 | 0.0341 | 0.2710 | 2.05 |
| CatBoost (application-only) | 0.7064 | 0.3759 | 0.1525 | 0.0312 | 0.2976 | 2.15 |
| CatBoost (post-pricing) | **0.7159** | **0.3838** | 0.1513 | 0.0312 | 0.3128 | **2.18** |

Validation→test **PSI = 0.0099**, well below the 0.1 stability warning line.

**AUC 0.72 is unremarkable — that is the point.** What matters for risk pricing is that the probability is trustworthy (ECE 0.031) and stable over time (PSI 0.0099). The contribution of this project is not the score; it is that the same score now comes with a readable decision.

Source: [`examples/model_comparison.csv`](examples/model_comparison.csv)

---

## Quick start

```bash
pip install -r requirements.txt

# validate the shipped programs and print an audit summary
python examples/run_demo.py

# run the test suite
pytest -q

# open the visual demo
#   demos/causalrisk_interview_demo.html
```

### Use the library

```python
from risk import compile_risk_program, validate_risk_program, execute_risk_program

case = {"loan_amnt": 25000, "dti": 35.78, "revol_util": 62.0, "annual_inc": 64000}
program = compile_risk_program(case, case_id="demo-001")

result = validate_risk_program(program)
assert result.valid, result.errors

def predict_probability(c):        # plug in your calibrated model here
    return 0.42

out = execute_risk_program(program, case, predict_probability=predict_probability)
print(out["observational_result"], out["causal_status"]["identified"])
```

---

## Repository layout

```
src/risk/program.py        core: compiler, validator, deterministic executor
demos/                     real artifacts from the reference run (10 cases)
examples/model_comparison.csv   baseline metrics
examples/run_demo.py       minimal end-to-end walkthrough
tests/test_program.py      validator / executor unit tests
```

---

## Data

The reference run uses the public **LendingClub** loan dataset with a chronological
(not random) split. Data files are **not** redistributed here. See [`DATA.md`](DATA.md)
for sources, split policy, and field definitions.

---

## Limitations

Stated plainly, because an auditability project that hides its limits is self-refuting:

1. **The graph is expert-specified, not learned.** The 16-node / 16-edge risk graph is hand-authored from domain knowledge. The system does not discover causal structure.
2. **No causal effect is identified.** Scenario deltas are *model responses* to perturbed inputs, not causal effects. Every artifact says so.
3. **Offline prototype.** Not approved for autonomous lending decisions. Adverse-action workflows require human review by design.
4. **One dataset, one model family.** Broader deployment requires re-validation on the target portfolio.

---

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

```bibtex
@software{auditable_risk_programs,
  title  = {Auditable Risk Programs: separating decision generation from decision computation},
  author = {Yue, Yunyu},
  year   = {2026},
  url    = {https://github.com/<your-handle>/auditable-risk-programs}
}
```
