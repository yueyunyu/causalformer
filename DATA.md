# Data

## Source

The reference run uses the public **LendingClub** loan dataset (2007–2018 loans,
accepted applications with terminal outcomes).

This repository does **not** redistribute the data. Download it yourself from
LendingClub's public release or from a mirrored copy on Kaggle, then place the
raw file under `data/` (git-ignored).

## Split policy

**Chronological split** — not random. Random splits leak future information into
training and inflate AUC.

| Split | Rows | Purpose |
|---|---|---|
| train | 400,000 | fit |
| validation | 100,000 | calibration, threshold freezing, checkpoint selection |
| test | 150,000 | final reporting only |

Distribution stability between validation and test is monitored with **PSI**
(population stability index); the reference run reports **PSI = 0.0099**.

## Information boundary

A model built on this data must only use fields observable **at or before
application/pricing time**. Anything that reveals the outcome after origination
is forbidden — the validator enforces this at the program level, not by
convention.

Forbidden post-origination fields include (excerpt):

```
loan_status, out_prncp, out_prncp_inv, total_pymnt, total_pymnt_inv,
total_rec_prncp, total_rec_int, total_rec_late_fee, recoveries,
collection_recovery_fee, last_pymnt_d, last_pymnt_amnt, next_pymnt_d,
last_credit_pull_d, last_fico_range_high, last_fico_range_low,
hardship_flag, hardship_type, hardship_reason, hardship_status,
deferral_term, hardship_amount, hardship_start_date, hardship_end_date,
payment_plan_start_date, hardship_length, hardship_dpd, hardship_loan_status,
orig_projected_additional_accrued_interest, hardship_payoff_balance_amount,
hardship_last_payment_amount, debt_settlement_flag, debt_settlement_flag_date,
settlement_status, settlement_date, settlement_amount, settlement_percentage,
settlement_term
```

The authoritative list lives in `FORBIDDEN_FUTURE_FIELDS` inside
[`src/risk/program.py`](../src/risk/program.py).

## Mutable scenario fields

Only these inputs may be perturbed in a scenario query:

```
loan_amnt, dti, revol_util
```

They correspond to levers a lender can actually discuss with an applicant
(requested principal, documented debt reduction, verified revolving-balance
reduction). Perturbing protected or immutable attributes is rejected.

## Metrics produced

`examples/model_comparison.csv` reports, per model:

ROC-AUC, PR-AUC, Brier, log loss, ECE (10 bins), KS, recall at 5% FPR,
threshold at 5% FPR, top-decile default capture and lift, prevalence, count.
