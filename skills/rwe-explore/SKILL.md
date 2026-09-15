---
name: rwe-explore
description: Run guarded, descriptive exploration of a local real-world-evidence cohort with pheno-rwe. Use for cohort summaries, attrition, mapping coverage, treatment pathways, trajectories, or hypothesis-generating patient signatures when no confirmatory comparative or causal claim is requested; graduate comparative questions to rwe-study.
---

# Explore an RWE cohort

Use `pheno-rwe` for every data operation and number. Keep the exploration descriptive and label all
findings exploratory.

Read [references/prohibitions.md](references/prohibitions.md) before working with study artifacts.

## 1. Confirm exploratory scope

Run `pheno-rwe status --json`. Ask for the target population, descriptive domains, time window,
and de-identification tier. If the request compares treatments, estimates an effect, tests a
confirmatory hypothesis, or seeks a causal conclusion, stop and use `rwe-study` instead.

## 2. Resolve and review concepts

Run `pheno-rwe resolve-codes` for each required concept. Show each coding and all mapping-status
counts. Require researcher approval before materialization or cohort labeling. Retain concept_id
`0` source values so unverified mappings remain visible rather than silently disappearing. In
each plan code set, use `pheno-rwe review-codes --decision approved --reviewed-by ...
--reviewed-at ...` after reconciling the exact artifact/plan codings; validation refuses pending
or rejected sets.

## 3. Emit the minimal plan

Read the current schema with `pheno-rwe validate --schema`. Create a plan containing only the
required cohort, index-date and de-identification rules, a fixed seed, and descriptive analyses:
`table_one`, `patient_signature`, treatment pathways or trajectories as requested. Add
`mapping_coverage` and `attrition` plots when available. Mark the plan and signature results
exploratory.

Run `pheno-rwe validate --json`. Relay refusals, warnings, and precision notes verbatim. A UMAP
minimum-n refusal is an invitation to use the engine's PCA downgrade, not to weaken the threshold.

## 4. Execute through the engine

Run the applicable pipeline steps, then `pheno-rwe analyze --all` and `pheno-rwe plot --all`.
Never query identified OMOP data. Stop on a refusal or stale-plan exit. Resume partial item-level
work through the same command rather than reimplementing a step.

## 5. Interpret conservatively

Use only `result.json`, tidy CSV, validation, coverage, and attrition artifacts. Report counts,
missingness, mapping provenance, stability diagnostics, and confidence intervals exactly as
emitted. Call clusters and signatures hypothesis-generating; do not assign disease mechanisms or
causal meaning to them.

Offer to graduate a promising comparative question into a preregistered `rwe-study` rather than
tweaking this exploration to chase significance.

## 6. Verify

Run `pheno-rwe trace --verify` and recommend sharing only the output of `pheno-rwe export`.
