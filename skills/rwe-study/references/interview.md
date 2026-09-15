# Study design interview

Ask only questions not already answered. Prefer one compact batch, then reflect the design back.

1. What decision will this study inform, and what is the target population?
2. What defines exposure and comparator? Are new-user, active-comparator, or washout rules needed?
3. What outcomes matter, how are they operationalized, and which is primary?
4. What event anchors time zero: first occurrence, last occurrence, or a fixed date?
5. What outcome washout, inclusive risk-window start/end, continuous-observation requirement,
   follow-up horizon, and censoring events apply?
6. Which inclusion/exclusion criteria and subgroups are prespecified?
7. Which baseline covariates are needed? For causal work, require a scientific rationale for each.
8. Is baseline de-identification enough, or should consistent per-patient date shifting be enabled?
9. Which sensitivity analyses are genuinely prespecified rather than prompted by results?

State the estimand for comparative work: population, treatment strategies, outcome, time horizon,
and summary measure. If any component remains ambiguous, do not invent it in `plan.json`.
