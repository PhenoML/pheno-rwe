# Absolute prohibitions

- Never compute a statistic yourself. Do not use ad-hoc Python, pandas, SQL, spreadsheets, mental
  arithmetic, or plot measurements to supply a number. Every reported number must come from a
  `pheno-rwe` output artifact.
- Never query, inspect, summarize, or quote patient-level identified data. Identified paths include
  `raw/`, `enriched/`, and `omop/omop.duckdb`.
- Never soften, paraphrase away, suppress, or omit a guardrail refusal, warning, assumption, or
  minimum-detectable-effect statement. Relay it verbatim.
- Never work around a refusal. CLI flags cannot override one; a permitted override must be an
  explicit, justified, hashed plan field accepted by the validator.
- Never edit a validated plan and run against the stale hash. Validate a recorded amendment first.
- Never tweak inclusion, outcomes, windows, covariates, tests, seeds, or plots in response to
  significance. Changes require a scientific reason and a plan amendment.
- Never treat patient signatures, UMAP positions, or clusters as confirmed phenotypes, mechanisms,
  diagnoses, or causal subtypes. They are hypothesis-generating.
- Never describe `causal_effect` output as a causal effect. Use “adjusted association” and report
  balance, positivity/weight diagnostics, sensitivity analysis, and the E-value the engine emits.
- Never share an identified database or raw/enriched data. Use `pheno-rwe export` and verify its
  bundle says `identified_data_included: false`.
