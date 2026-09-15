---
name: rwe-discover
description: Build a precise patient retrieval set for `pheno-rwe pull` by orchestrating stateless discovery primitives (Construe extract-codes, crosswalk, and FHIR-path search) in a count-first loop. Use before pulling live FHIR data when a one-shot natural-language cohort is too broad or too narrow, or needs cross-resource criteria the FHIR server will not join server-side (for example Patient.gender ∩ MedicationRequest.code). Discovery decides only what to pull; graduate the pulled data to rwe-study or rwe-explore.
---

# Discover a patient retrieval set

Use `pheno-rwe` for every concept, count, and patient ID. The discovery commands are stateless query
primitives: they print JSON, record no manifest step, and never write patient rows to disk.

**Stay on the acquisition side of the determinism wall.** Discovery decides only *what to pull* — a
deliberate superset of candidate patients. The deterministic `resolve-cohort` step on the
materialized OMOP still owns the *final analytic cohort*. Never present a discovery count or ID list
as a result, and never narrow the retrieval set to hit a target N. Build the superset and let the
engine's guardrails own inclusion downstream. Every number and every ID must come from a CLI output,
never from hand-editing or arithmetic.

## 1. Locate the study and confirm access

Run `pheno-rwe status --check-env --json`. Discovery reads PhenoML credentials and the FHIR provider
from the study's `.env` (`pheno-rwe init <directory> --name <name>` first if no study exists).
`extract-codes` and `crosswalk` need PhenoML credentials only; `fhir-search` also needs a provider
(`FHIR_PROVIDER_ID` in `.env`, or `--provider <id>`).

## 2. Resolve each concept to source codes

Run `pheno-rwe extract-codes "<concept>" [--domain <omop-domain>] --json` for every clinical
criterion. Read the returned `items` (system/code/display). These are candidate source codes for
retrieval, not an approved code set — code-set approval happens later, in the deterministic pipeline.

## 3. Crosswalk to the vocabularies in the provider's data

Provider data is rarely coded in the source vocabulary. For each source coding, run
`pheno-rwe crosswalk --system <s> --code <c> --to <target-system> [--to …] --json` to get the target
codings that share a UMLS concept. Collect the target `code`s you will actually search on.

## 4. Iterate count-first, then extract patients

Tune each criterion cheaply with counts before extracting patients:

```bash
pheno-rwe fhir-search MedicationRequest --param code=<rxnorm,set> --count --json   # too broad / narrow?
```

Adjust the code set until the count is defensible. Then extract the distinct patient IDs for each
criterion. IDs are returned in the JSON `items`; capture them one per line with `jq`:

```bash
pheno-rwe fhir-search MedicationRequest --param code=<rxnorm,set> --patients --json \
  | jq -r '.items[].id' > on_glp1.txt
pheno-rwe fhir-search Patient --param gender=male --patients --json \
  | jq -r '.items[].id' > male.txt
```

`--param` is repeatable and parsed as `key=value`. `--patients` follows FHIR `next` links and
de-duplicates across pages, reading `Patient.id` and the `subject`/`patient` references on other
resources.

## 5. Compose the retrieval set

Combine criteria with ordinary set tools to express cross-resource logic the FHIR server will not
join server-side. Intersect for AND, concatenate for OR:

```bash
comm -12 <(sort -u on_glp1.txt) <(sort -u male.txt) > cohort.txt   # male ∩ on a GLP-1
```

Keep the set a deliberate superset — prefer over-inclusion; the deterministic `resolve-cohort`
narrows to the final cohort.

## 6. Pull the retrieval set (provenance begins here)

Hand the file to `pull`. This is the first provenanced step: it records a manifest entry and fetches
`$everything` per patient.

```bash
pheno-rwe pull --patients cohort.txt --yes
```

`pull` requires exactly one of `--cohort` or `--patients`; `--queries-only` applies only to
`--cohort`. IDs are read whitespace/newline-delimited and de-duplicated.

## 7. Hand off to the deterministic pipeline

Continue with the normal engine pipeline and graduate to the right skill:

```text
materialize -> resolve-cohort -> deid -> validate -> analyze -> plot -> export
```

Use `rwe-study` for comparative or causal questions and `rwe-explore` for descriptive exploration.
The final analytic cohort and every statistic come from the engine, never from discovery output.
