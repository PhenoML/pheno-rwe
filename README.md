# pheno-rwe

`pheno-rwe` is a local-first, deterministic engine for traceable real-world-evidence studies.
It turns bulk FHIR or live FHIR data (through a PhenoML provider) into OMOP tables in DuckDB,
derives cohorts and index dates, creates a physically separate de-identified database, runs
guarded analyses, and records every step in a hash-chained manifest.

The coding agent is the research-design collaborator. It interviews the researcher, resolves and
reviews concepts, writes `plan.json`, drives the CLI, and interprets result envelopes. It never
touches patient rows or calculates statistics itself.

## Install

```bash
pip install "pheno-rwe[analysis,phenoml]"
pheno-rwe --help
```

For development, install with [uv](https://docs.astral.sh/uv/). The repository pins Python 3.11
via `.python-version` (matching CI), so uv provisions and reuses a single 3.11 interpreter instead
of recreating the environment:

```bash
uv sync --all-extras --dev
uv run pheno-rwe --help
uv run pytest
```

`uv run` re-checks the environment on every call. After the first `uv sync` that check is a fast
no-op, but you can skip it entirely by activating the virtualenv once and calling the console
script directly:

```bash
source .venv/bin/activate
pheno-rwe --help
pytest
```

Equivalently, pass `uv run --no-sync …` (or set `UV_NO_SYNC=1`) to run without touching
dependencies. For a reproducible install straight from the lockfile, use
`uv sync --frozen --all-extras --dev` as CI does.

Common tasks have `make` shortcuts: `make install`, `make format`, `make test`, and `make check`
(lint, format check, type-check, and tests — the same gate CI runs).

## Study pipeline

```text
init → pull|ingest → enrich? → resolve-codes → materialize
     → resolve-cohort → deid → validate → analyze → plot → export
```

Create and inspect a local study. `init`, `ingest`, and `status` run offline, and the repository
ships a small synthetic FHIR fixture you can ingest right away:

```bash
pheno-rwe init studies/diabetes --name "Diabetes treatment study"
pheno-rwe --study studies/diabetes ingest tests/fixtures/testdata/synthea_mini
pheno-rwe --study studies/diabetes status
```

The remaining pipeline steps call PhenoML and require credentials (see
[Authentication](#authentication)).

Live FHIR is always mediated by PhenoML. The CLI displays the derived FHIR queries and requires
confirmation before fetching per-patient `$everything` bundles:

```bash
pheno-rwe --study studies/diabetes pull \
  --cohort "adults with type 2 diabetes starting metformin" \
  --provider "$FHIR_PROVIDER_ID"
```

Resolve and review code sets before writing the plan:

```bash
pheno-rwe --study studies/diabetes resolve-codes \
  --name type-2-diabetes --domain condition --text "type 2 diabetes"
pheno-rwe --study studies/diabetes review-codes \
  --name type-2-diabetes --decision approved \
  --reviewed-by "Researcher Name" --reviewed-at "2026-08-12T16:00:00Z"
pheno-rwe --study studies/diabetes validate --schema
pheno-rwe --study studies/diabetes validate
pheno-rwe --study studies/diabetes analyze --all
pheno-rwe --study studies/diabetes plot --all
pheno-rwe --study studies/diabetes trace --verify
```

## Physical PHI boundary

- `raw/`, `enriched/`, and `omop/omop.duckdb` are identified and never shareable.
- `omop/deid.duckdb` is rebuilt by the de-identification step and is the only database analyses
  may open.
- Baseline de-identification removes direct identifiers and buckets age. Optional date shifting
  uses one HMAC-derived offset per patient across every registered date column.
- A fail-closed column policy classifies every OMOP, metadata, and study column. Analyses retain
  surrogate joins, controlled codes, dates, numeric values, and cohort fields; provider/care-site
  linkage and unsafe narrative/value/error/original-date text are removed. Source/resource IDs
  and content fingerprints needed for provenance are study-keyed HMACs, never raw hashes.
- `pheno-rwe export` creates a de-identified DuckDB/Parquet bundle and explicitly records that no
  identified data is included. Its manifest is rebuilt with the `shareable-v1` profile: source
  paths, queries, item identifiers, raw/identified hashes, and the original chain fingerprint are
  omitted, while safe step metadata and a re-hashable inventory are retained.

## Plan and execution contracts

`plan.json` is validated with Pydantic and hashed as canonical JSON. Static guardrails run at plan
time; data guardrails run immediately before each analysis. A refusal is not flag-overridable.
After validation, changing the plan produces exit code `4` until the amendment is validated.

Every code set has plan-hashed approval metadata. Validation refuses `pending` or `rejected` sets;
an approved set records both `reviewed_by` and `reviewed_at` after the researcher has inspected all
codings and mapping-status counts. `cohort_compare` and `causal_effect` use structured binary-risk
outcomes rather than an unbounded code-set string:

```json
{
  "code_set": "stroke",
  "risk_window": {
    "start_day": 0,
    "end_day": 90,
    "washout_days": 365,
    "observation_censor": "end_of_observation"
  }
}
```

Risk boundaries are inclusive; washout is `[index - washout_days, index)`. A patient must have
continuous baseline observation for the washout. An observed event before the patient's
observation end is a case, while an event-free patient lost before the fixed horizon is censored
and remains missing rather than being treated as a non-case. These choices, including the default
observation-censor strategy, are part of the canonical plan hash.

`trace --verify` checks the chain plus the latest declared state of both inputs and outputs. In a
shareable bundle it verifies only in-bundle artifacts and never follows a manifest path outside
the bundle root.

Exit codes are `0` success, `1` fatal, `2` partial item failure, `3` guardrail refusal, and `4`
stale plan. Typer usage errors also use `2` before a study step starts.

Each analysis writes `results/<id>/result.json` and tidy CSVs. Each plot reads only a tidy CSV and
writes `plot.png`, `plot.svg`, `data.csv`, and `spec.json`, so publication graphics can be
regenerated without retaining a model object.

## Agent skills

Portable skills live in [`skills/rwe-study`](skills/rwe-study/SKILL.md) and
[`skills/rwe-explore`](skills/rwe-explore/SKILL.md). Copy or symlink a skill folder into the skill
directory used by Claude Code, Codex, or another compatible coding agent. Agents that consume
repository instructions can use [`AGENTS.md`](AGENTS.md) directly.

For project-local OpenCode discovery and the synthetic networked smoke-test workflow, see
[`docs/testing-with-opencode.md`](docs/testing-with-opencode.md).

## Authentication

Copy `.env.example` to a local `.env`. Prefer `PHENOML_CLIENT_ID` and
`PHENOML_CLIENT_SECRET`; username/password is supported only as a compatibility fallback. All
FHIR access uses the configured PhenoML provider ID. Do not put credentials or study workspaces in
Git. Live API calls use a deterministic token-bucket limiter and bounded retries for timeouts,
HTTP 429, and HTTP 5xx responses; `Retry-After` is honored up to the configured maximum wait.
Set `PHENOML_MAX_RPS` or pass `--max-rps` to a live command to lower the request rate.

Normal tests never call PhenoML. The developer-only `scripts/record_fixtures.py` command requires
two explicit opt-ins and accepts synthetic inputs only; review every generated fixture before
adding it to version control.

See [`docs/architecture.md`](docs/architecture.md) for contracts and extension points.
