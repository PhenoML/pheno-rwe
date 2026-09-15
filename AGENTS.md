# pheno-rwe agent protocol

Use `pheno-rwe` as the only engine for study data movement, cohort construction, statistics, and
plots. The agent's job is to converge with the researcher on a declarative `plan.json`, relay
validation exactly, drive commands serially, and interpret recorded artifacts.

Before a study, establish the question, population, exposure/comparator, outcomes, index date,
washout/follow-up/censoring, covariates with rationales, subgroups, and de-identification tier.
Resolve each code set, show every mapping-status count, and require researcher approval.

Never:

- compute a statistic with ad-hoc Python, SQL, pandas, a spreadsheet, or mental arithmetic;
- query or expose `raw/`, `enriched/`, or `omop/omop.duckdb` patient-level content;
- soften, omit, override, or route around a guardrail outcome;
- execute after changing a validated plan without validating the amendment;
- tweak a design to chase significance;
- oversell patient signatures or describe adjusted associations as causal effects;
- share anything except the de-identified bundle created by `pheno-rwe export`.

Use exit codes as contracts: `0` success, `1` fatal, `2` partial item failure, `3` deterministic
guardrail refusal, and `4` plan hash mismatch. Interpret only `result.json`, its tidy CSVs,
validation reports, mapping/attrition reports, and `trace --verify`.
