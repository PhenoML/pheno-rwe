# Architecture

## Boundaries

The Typer CLI is an adapter over `pheno_rwe.steps`. Long-running steps accept a progress callback
and cancellation token. They isolate work per patient or document and append one manifest entry
after the durable outputs are complete.

```text
Claude/Codex skill ─┐
Terminal CLI ───────┼─> step library ─> validation gate ─> analysis registry
                                                            v
raw/enriched ─> omop.duckdb ─> deid.duckdb ─> result JSON/tidy CSV/plots
                          manifest.jsonl spans every transition
```

The generated PhenoML SDK is wrapped behind `PhenoTransport`. Tests use a fake implementation.
Live FHIR paths always call `fhir.search(provider_id, fhir_path=...)`; the package contains no
SMART-on-FHIR client.

## Storage

One study is discovered by walking upward to the nearest `study.json`:

```text
study.json                 stable study UUID and display metadata
plan.json                  validated declarative study design
manifest.jsonl             append-only prev_hash/entry_hash chain
raw/patients/              one FHIR Bundle per patient (identified)
enriched/                  enriched bundles and document-response cache (identified)
codesets/                  reviewed coding/concept mappings
omop/omop.duckdb           identified OMOP zone
omop/deid.duckdb           de-identified analysis zone
reports/                   validation, mapping, attrition and de-id reports
results/<analysis>/        result envelope and tidy tables
plots/<plot>/              PNG, SVG, input CSV and plot spec
exports/                   shareable de-identified bundles
```

fhir2omop row IDs are local to a response. The loader assigns each source patient a persisted
million-row offset block and applies explicit row/FK maps; concept IDs are never re-keyed.

### De-identified column policy

`pheno_rwe.deid.policy` is the fail-closed sharing boundary. Every column in all 22 OMOP,
metadata, and study tables is classified as retained, nulled, study-keyed HMAC, normalized, or
dropped with its table. Any unregistered table or column makes de-identification and export fail.

- Retained: generated surrogate keys and joins, controlled coding/unit source values needed for
  concept-ID fallback, clinical dates (date-shifted when configured), numeric measurements,
  mapping status, cohort/attrition fields, and age buckets.
- Nulled: drug sig/stop reason, observation and measurement text values, provider/care-site
  references, mapping display/notes, item errors/drop reasons, source pages, and unparseable
  original temporal values.
- Study-keyed HMAC: source-patient/resource/document-reference identifiers and bundle,
  fhir2omop-response, and document-content fingerprints. Raw SHA-256 values are not retained.
- Removed: all location, care-site, and provider rows. Operational ledger timestamps are
  normalized to a constant epoch.

The engine first scrubs an unpublished working copy, then copies only policy-declared logical
columns into a newly created DuckDB file. It never publishes the in-place copy, whose freed pages
could retain bytes from the identified database. Export re-audits the database and writes Parquet
with explicit table and column selections from the same registry.

## Reproducibility

Manifest entries include parameters, an input signature, logical/file hashes, item ledgers, API
call timing/credits, runtime versions, and a chain pointer. Plans are normalized with defaults
before canonical SHA-256 hashing. Analysis refuses a plan whose hash is not the latest successful
validation event. Re-validation records amendments rather than hiding them.

Code-set approval is a first-class plan field: only `approved` sets with a reviewer and review
timestamp pass the static gate. For `cohort_compare` and `causal_effect`, each outcome binds a code
set to an inclusive finite risk window, an incident-outcome washout, and mandatory
end-of-observation censoring. Preparation gives early-censored non-cases a missing outcome and
uses the same contract for aggregate DATA-guardrail counts; it never turns unequal follow-up into
silent non-events.

Shareable exports use a separately rebuilt `shareable-v1` manifest. It retains an allowlisted
step/status/version timeline and aggregate API/item counts, but deliberately drops the source
chain fingerprint, queries, external paths, per-item identifiers, input signatures, and every
raw/identified artifact declaration. The final export entry declares all included payload files
as inputs and `bundle.json` as its output. This keeps `trace --verify` useful on a raw-free bundle
without making its verification depend on artifacts that must remain at the source site.

## Adding an analysis

Implement the analysis protocol, define validated parameters, register its `kind`, declare data
requirements and guardrail rules, and return an `AnalysisResult` envelope plus tidy tables. Never
write to `deid.duckdb`. Add crafted threshold tests and a deterministic statistical smoke test.

## Adding a plot

Register a renderer that accepts only a DataFrame/CSV and a validated plot spec. Emit both PNG and
SVG, copy the exact tidy input to `data.csv`, and record package/style versions. Plots must remain
readable without color and use the bundled Matplotlib style.
