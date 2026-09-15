---
name: rwe-study
description: Design, preregister, validate, execute, and interpret traceable real-world-evidence studies with the pheno-rwe CLI. Use for comparative effectiveness, case-control, survival, incidence, longitudinal, treatment-pathway, patient-signature, or causal-association questions over local FHIR/OMOP data where a researcher must approve code sets and deterministic guardrails must own every statistic.
---

# Run an RWE study

Use `pheno-rwe` as the deterministic engine. You design the study with the researcher, write a
declarative plan, and interpret engine artifacts. Never replace an engine step with your own SQL,
Python, arithmetic, or statistical judgment.

Read [references/interview.md](references/interview.md) before designing the study and
[references/prohibitions.md](references/prohibitions.md) before touching study artifacts.

## 1. Locate and inspect the study

Run `pheno-rwe status --json`. If no study exists, ask for a study name and run
`pheno-rwe init <directory> --name <name>`. Treat `raw/`, `enriched/`, and
`omop/omop.duckdb` as identified zones. Never query or quote patient-level records.

## 2. Interview and reflect the design

Ask only the unanswered questions in the interview reference. Establish the research question,
target population, exposure, comparator, outcomes, index-date rule, washout, follow-up and
censoring, covariates with a rationale for each, subgroups, and de-identification tier.

State the proposed design back in plain language. Flag likely small-sample limitations early, but
do not invent counts. For causal work, say that the supported claim is an adjusted association.

## 3. Resolve and review code sets

Run one `pheno-rwe resolve-codes` command per clinical concept. Present every returned coding and
the `ALREADY_STANDARD`, `MAPPED`, `UNCHECKED`, and `UNMAPPED` counts. Ask the researcher to accept,
remove, or add codings, and reconcile the resolved artifact with the same set in `plan.json`.
Do not proceed until they approve the sets. Run `pheno-rwe review-codes --name <name> --decision
approved --reviewed-by <researcher> --reviewed-at <ISO-8601-with-offset>`; it records the same
approval atomically in the artifact and plan and refuses a coding/domain mismatch. Never silently
discard concept_id `0` codings because source-value matching is supported. Validation
deterministically refuses every `pending` or `rejected` code set.

## 4. Write and preregister the plan

Obtain the current schema with `pheno-rwe validate --schema`. Write `plan.json` using only that
schema. Declare the seed and every analysis before execution. Include a rationale for every causal
adjustment covariate. Include a censoring rule for survival. Put any permitted provenance override
and its justification in the plan so it is hashed and auditable.

For every `cohort_compare` or `causal_effect` outcome, use the schema's structured outcome object:
`code_set` plus `risk_window.start_day`, `end_day`, and `washout_days`. Confirm these choices with
the researcher; do not carry forward a legacy bare code-set string or invent a default horizon.
Both risk boundaries are inclusive, washout covers `[index - washout_days, index)`, and
`observation_censor` is `end_of_observation`. Explain that an event-free patient lost before the
fixed horizon is censored/missing, not a non-case.

Run `pheno-rwe validate --json`. Relay every refusal, warning, and minimum-detectable-effect
sentence verbatim. If validation refuses, redesign with the researcher and validate an amendment.
Never use flags or edits to bypass a refusal. Do not execute a plan whose hash differs from the
last validated manifest entry.

## 5. Execute serially

Run the required deterministic steps in order:

```text
pull|ingest -> enrich? -> materialize -> resolve-cohort -> deid -> validate -> analyze --all -> plot --all
```

Stop on exit code `3` (guardrail refusal) or `4` (stale plan). Exit code `2` means some independent
items failed; show the item ledger and ask whether to resume. Do not manually patch derived data.

## 6. Interpret only recorded outputs

Read `results/<analysis_id>/result.json` and the tidy CSV files it names. Report sample sizes,
effect estimates with confidence intervals, q-values where emitted, assumptions checked,
guardrail outcomes, mapping/enrichment provenance fractions, and the exact `power_note`.

Separate association from causation and exploratory from confirmatory work. Describe patient
signatures as hypothesis-generating. Do not calculate a missing statistic or infer one from a plot.
If the engine did not emit a requested number, say so and propose a preregistered plan amendment.

## 7. Close the trace

Run `pheno-rwe trace --verify`. Report whether the hash chain and artifacts verify, the plan hash,
any amendments, partial steps, and the paths to result/plot artifacts. Recommend sharing only the
de-identified export created by `pheno-rwe export`.
