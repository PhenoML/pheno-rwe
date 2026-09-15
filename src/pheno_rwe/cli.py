"""Thin Typer adapter over the deterministic step library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, overload

import typer
from rich.console import Console
from rich.table import Table
from typer.core import TyperGroup

from pheno_rwe import __version__
from pheno_rwe.config import Settings
from pheno_rwe.errors import (
    GuardrailRefusal,
    PhenoRWEError,
    StalePlanError,
)
from pheno_rwe.serialization import to_data
from pheno_rwe.steps.common import StepResult
from pheno_rwe.workspace import StudyWorkspace, find_study


class FlexibleGlobalOptionsGroup(TyperGroup):
    """Accept pheno-rwe's global options before or after its subcommand.

    Click normally stops parsing group options once it reaches a subcommand.  The
    public CLI examples historically used both ``pheno-rwe --json status`` and
    ``pheno-rwe status --json``; normalizing only this small, explicit option set
    keeps those forms equivalent without duplicating global flags on every command.
    """

    _global_flags = frozenset({"--force", "--dry-run", "--json", "--version"})
    _global_values = frozenset({"--study"})

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:
        global_args: list[str] = []
        command_args: list[str] = []
        index = 0
        while index < len(args):
            argument = args[index]
            if argument == "--":
                command_args.extend(args[index:])
                break
            if argument in self._global_flags:
                global_args.append(argument)
            elif argument in self._global_values:
                global_args.append(argument)
                index += 1
                if index >= len(args):
                    # Let Click produce its standard "requires an argument" error.
                    break
                global_args.append(args[index])
            elif any(argument.startswith(f"{option}=") for option in self._global_values):
                global_args.append(argument)
            else:
                command_args.append(argument)
            index += 1
        return super().parse_args(ctx, [*global_args, *command_args])


app = typer.Typer(
    name="pheno-rwe",
    cls=FlexibleGlobalOptionsGroup,
    help="Local-first, traceable real-world-evidence studies over FHIR and OMOP.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


class State:
    study: Path | None = None
    force: bool = False
    dry_run: bool = False
    json_output: bool = False


@app.callback()
def main(
    ctx: typer.Context,
    study: Annotated[
        Path | None,
        typer.Option("--study", help="Study directory; defaults to nearest ancestor."),
    ] = None,
    force: bool = typer.Option(
        False, "--force", help="Re-run or replace otherwise current outputs."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Describe writes without committing where supported."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the package version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    state = State()
    state.study = study
    state.force = force
    state.dry_run = dry_run
    state.json_output = json_output
    ctx.obj = state


def _state(ctx: typer.Context) -> State:
    return ctx.ensure_object(State)


@overload
def _workspace(ctx: typer.Context, required: Literal[True] = True) -> StudyWorkspace: ...


@overload
def _workspace(ctx: typer.Context, required: Literal[False]) -> StudyWorkspace | None: ...


def _workspace(ctx: typer.Context, required: bool = True) -> StudyWorkspace | None:
    state = _state(ctx)
    if not required and state.study is None:
        try:
            return find_study()
        except Exception:
            return None
    return find_study(state.study)


def _render(ctx: typer.Context, value: Any, *, exit_code: int | None = None) -> None:
    payload = value.model_dump() if hasattr(value, "model_dump") else to_data(value)
    if _state(ctx).json_output:
        console.print_json(json.dumps(payload, default=str))
    elif isinstance(value, StepResult):
        color = (
            "yellow"
            if value.status == "partial"
            else ("red" if value.status in {"failed", "refused"} else "green")
        )
        console.print(f"[{color}]{value.status}[/{color}] {value.message}")
        for warning in value.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")
        for path in value.outputs:
            console.print(f"  {path}")
    else:
        console.print_json(json.dumps(payload, default=str))
    code = exit_code if exit_code is not None else getattr(value, "exit_code", 0)
    if code:
        raise typer.Exit(code=int(code))


def _run(ctx: typer.Context, function: Any, *args: Any, **kwargs: Any) -> None:
    try:
        result = function(*args, **kwargs)
    except GuardrailRefusal as exc:
        _failure(ctx, str(exc), code=3, details=[to_data(item) for item in exc.outcomes])
    except StalePlanError as exc:
        _failure(ctx, str(exc), code=4)
    except (PhenoRWEError, FileNotFoundError, ValueError, RuntimeError) as exc:
        _failure(ctx, str(exc), code=1)
    _render(ctx, result)


def _failure(ctx: typer.Context, message: str, *, code: int, details: Any = None) -> None:
    if _state(ctx).json_output:
        console.print_json(
            json.dumps(
                {"status": "error", "message": message, "details": details, "exit_code": code},
                default=str,
            )
        )
    else:
        err_console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=code)


def _parse_key_values(pairs: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ValueError(f"Invalid --param {pair!r}; expected key=value.")
        params[key] = value
    return params


@app.command("init")
def init_command(
    ctx: typer.Context,
    directory: Annotated[Path, typer.Argument(help="Directory for the new study.")],
    name: str = typer.Option(..., "--name", help="Researcher-facing study name."),
) -> None:
    from pheno_rwe.steps.init import initialize_study

    _run(ctx, initialize_study, directory, name, force=_state(ctx).force)


@app.command()
def ingest(
    ctx: typer.Context,
    sources: Annotated[list[Path], typer.Argument(help="NDJSON files or directories.")],
) -> None:
    from pheno_rwe.steps.ingest import ingest_ndjson

    _run(
        ctx,
        ingest_ndjson,
        _workspace(ctx),
        sources,
        force=_state(ctx).force,
        dry_run=_state(ctx).dry_run,
    )


@app.command()
def pull(
    ctx: typer.Context,
    cohort: str | None = typer.Option(None, "--cohort", help="Natural-language cohort definition."),
    patients: Annotated[
        Path | None,
        typer.Option(
            "--patients",
            help="File of newline/whitespace-delimited patient IDs to fetch directly.",
        ),
    ] = None,
    provider: str | None = typer.Option(None, "--provider", help="PhenoML FHIR provider ID."),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Approve displayed FHIR queries non-interactively."
    ),
    queries_only: bool = typer.Option(
        False, "--queries-only", help="Derive queries without executing them."
    ),
    max_rps: float | None = typer.Option(
        None,
        "--max-rps",
        min=0.01,
        help="Override the configured PhenoML request rate.",
    ),
) -> None:
    from pheno_rwe.client import build_transport
    from pheno_rwe.steps.pull import preview_from_ids, preview_live_cohort, pull_live_cohort

    workspace = _workspace(ctx)
    study_env = workspace.root / ".env"
    settings = Settings.from_env(study_env)
    provider_id = provider or settings.require_fhir_provider()
    transport = build_transport(settings, max_rps=max_rps)
    try:
        if (cohort is None) == (patients is None):
            raise ValueError("Provide exactly one of --cohort or --patients.")
        if queries_only and patients is not None:
            raise ValueError("--queries-only is only valid with --cohort.")
        if patients is not None:
            try:
                raw_ids = patients.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Could not read patient IDs from {patients}: {exc}") from exc
            ids = list(dict.fromkeys(raw_ids.split()))
            if not ids:
                raise ValueError(f"No patient IDs found in {patients}.")
            preview = preview_from_ids(ids)
            text = f"explicit:{patients}"
        else:
            assert cohort is not None
            preview = preview_live_cohort(transport, cohort, provider_id, queries_only=queries_only)
            text = cohort
            if queries_only:
                _render(
                    ctx,
                    {
                        "queries": preview.queries,
                        "patient_count": len(preview.patient_ids),
                        "patient_ids": [],
                    },
                )
                return
        if _state(ctx).json_output and not yes:
            raise ValueError(
                "JSON pull execution requires --yes; use --queries-only to preview queries."
            )
        if not _state(ctx).json_output:
            if patients is None:
                console.print("[bold]Derived FHIR queries[/bold]")
                console.print_json(json.dumps(preview.queries, default=str))
            else:
                console.print(
                    f"[bold]Explicit patient set[/bold] "
                    f"({len(preview.patient_ids)} patients from {patients})"
                )
        approved = yes or typer.confirm(
            f"Fetch $everything for {len(preview.patient_ids)} patients?"
        )
        if not approved:
            console.print("Cancelled before any FHIR patient data was fetched.")
            return
        _run(
            ctx,
            pull_live_cohort,
            workspace,
            transport,
            text=text,
            provider_id=provider_id,
            approved=True,
            preview=preview,
            force=_state(ctx).force,
        )
    except (PhenoRWEError, ValueError, RuntimeError) as exc:
        _failure(ctx, str(exc), code=1)


@app.command("extract-codes")
def extract_codes_command(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument(help="Clinical concept to resolve to source codes.")],
    domain: str | None = typer.Option(None, "--domain", help="Optional OMOP domain hint."),
    max_rps: float | None = typer.Option(
        None,
        "--max-rps",
        min=0.01,
        help="Override the configured PhenoML request rate.",
    ),
) -> None:
    from pheno_rwe.client import build_transport
    from pheno_rwe.steps.discover import extract_codes

    workspace = _workspace(ctx)
    settings = Settings.from_env(workspace.root / ".env")
    transport = build_transport(settings, max_rps=max_rps)
    _run(ctx, extract_codes, transport, text, domain)


@app.command()
def crosswalk(
    ctx: typer.Context,
    system: str = typer.Option(..., "--system", help="Source code system URI."),
    code: str = typer.Option(..., "--code", help="Source code value."),
    to: list[str] = typer.Option(
        ..., "--to", help="Target system URI; repeat for multiple targets."
    ),
    max_rps: float | None = typer.Option(
        None,
        "--max-rps",
        min=0.01,
        help="Override the configured PhenoML request rate.",
    ),
) -> None:
    from pheno_rwe.client import build_transport
    from pheno_rwe.steps.discover import crosswalk_code

    workspace = _workspace(ctx)
    settings = Settings.from_env(workspace.root / ".env")
    transport = build_transport(settings, max_rps=max_rps)
    _run(ctx, crosswalk_code, transport, system=system, code=code, targets=to)


@app.command("fhir-search")
def fhir_search_command(
    ctx: typer.Context,
    path: Annotated[str, typer.Argument(help="FHIR resource type or search path, e.g. Patient.")],
    param: list[str] = typer.Option(
        [], "--param", help="Repeatable FHIR search parameter as key=value."
    ),
    count: bool = typer.Option(False, "--count", help="Return only the match count."),
    patients: bool = typer.Option(
        False, "--patients", help="Collect distinct patient IDs across all pages."
    ),
    provider: str | None = typer.Option(None, "--provider", help="PhenoML FHIR provider ID."),
    max_rps: float | None = typer.Option(
        None,
        "--max-rps",
        min=0.01,
        help="Override the configured PhenoML request rate.",
    ),
) -> None:
    from pheno_rwe.client import build_transport
    from pheno_rwe.steps.discover import search_fhir

    workspace = _workspace(ctx)
    settings = Settings.from_env(workspace.root / ".env")
    provider_id = provider or settings.require_fhir_provider()
    transport = build_transport(settings, max_rps=max_rps)
    try:
        params = _parse_key_values(param)
    except ValueError as exc:
        _failure(ctx, str(exc), code=1)
        return
    _run(ctx, search_fhir, transport, provider_id, path, params, count=count, patients=patients)


@app.command()
def enrich(
    ctx: typer.Context,
    provider: str | None = typer.Option(None, "--provider"),
    detection_effort: str = typer.Option("standard", "--detection-effort"),
    validation_method: str = typer.Option("check", "--validation-method"),
    max_rps: float | None = typer.Option(
        None,
        "--max-rps",
        min=0.01,
        help="Override the configured PhenoML request rate.",
    ),
) -> None:
    from pheno_rwe.client import build_transport
    from pheno_rwe.steps.enrich import enrich_documents

    workspace = _workspace(ctx)
    study_env = workspace.root / ".env"
    settings = Settings.from_env(study_env)
    _run(
        ctx,
        enrich_documents,
        workspace,
        build_transport(settings, max_rps=max_rps),
        provider_id=provider or settings.fhir_provider_id,
        detection_effort=detection_effort,
        validation_method=validation_method,
        force=_state(ctx).force,
    )


@app.command("resolve-codes")
def resolve_codes(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name"),
    text: str = typer.Option(..., "--text"),
    domain: str | None = typer.Option(None, "--domain"),
    max_rps: float | None = typer.Option(
        None,
        "--max-rps",
        min=0.01,
        help="Override the configured PhenoML request rate.",
    ),
) -> None:
    from pheno_rwe.client import build_transport
    from pheno_rwe.steps.resolve_codes import resolve_code_set

    workspace = _workspace(ctx)
    study_env = workspace.root / ".env"
    settings = Settings.from_env(study_env)
    _run(
        ctx,
        resolve_code_set,
        workspace,
        build_transport(settings, max_rps=max_rps),
        name=name,
        text=text,
        domain=domain,
        force=_state(ctx).force,
    )


@app.command("review-codes")
def review_codes(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Existing resolved and planned code-set name."),
    decision: str = typer.Option(..., "--decision", help="Researcher decision: approved|rejected."),
    reviewed_by: str = typer.Option(..., "--reviewed-by", help="Researcher recording the review."),
    reviewed_at: str = typer.Option(
        ...,
        "--reviewed-at",
        help="ISO-8601 review timestamp including a UTC offset.",
    ),
    notes: str | None = typer.Option(None, "--notes", help="Optional review notes."),
) -> None:
    from pheno_rwe.steps.review_codes import review_code_set

    _run(
        ctx,
        review_code_set,
        _workspace(ctx),
        name=name,
        decision=decision,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
        notes=notes,
    )


@app.command()
def materialize(
    ctx: typer.Context,
    max_rps: float | None = typer.Option(
        None,
        "--max-rps",
        min=0.01,
        help="Override the configured PhenoML request rate.",
    ),
) -> None:
    from pheno_rwe.client import build_transport
    from pheno_rwe.steps.materialize import materialize_bundles

    _run(
        ctx,
        materialize_bundles,
        _workspace(ctx),
        build_transport(max_rps=max_rps),
        force=_state(ctx).force,
    )


@app.command("resolve-cohort")
def resolve_cohort_command(ctx: typer.Context) -> None:
    from pheno_rwe.steps.resolve_cohort import resolve_cohort

    _run(ctx, resolve_cohort, _workspace(ctx))


@app.command()
def deid(ctx: typer.Context, k: int = typer.Option(5, "--k", min=2)) -> None:
    from pheno_rwe.steps.deid import deidentify

    _run(ctx, deidentify, _workspace(ctx), k=k)


@app.command()
def validate(
    ctx: typer.Context,
    schema: bool = typer.Option(False, "--schema", help="Print the current plan JSON Schema."),
    static_only: bool = typer.Option(False, "--static-only", help="Skip DATA guardrails."),
) -> None:
    if schema:
        from pheno_rwe.plan import plan_json_schema

        console.print_json(json.dumps(plan_json_schema()))
        return
    from pheno_rwe.steps.validate import validate_study

    _run(ctx, validate_study, _workspace(ctx), include_data=not static_only)


@app.command()
def analyze(
    ctx: typer.Context,
    analysis_id: str | None = typer.Option(None, "--id"),
    run_all: bool = typer.Option(False, "--all"),
) -> None:
    from pheno_rwe.steps.analyze import analyze_study

    _run(
        ctx,
        analyze_study,
        _workspace(ctx),
        analysis_id=analysis_id,
        all=run_all,
        force=_state(ctx).force,
    )


@app.command()
def plot(
    ctx: typer.Context,
    plot_id: str | None = typer.Option(None, "--id"),
    run_all: bool = typer.Option(False, "--all"),
    from_csv: Annotated[Path | None, typer.Option("--from-csv")] = None,
    kind: str | None = typer.Option(None, "--kind"),
) -> None:
    from pheno_rwe.steps.plot import plot_study

    _run(
        ctx,
        plot_study,
        _workspace(ctx),
        plot_id=plot_id,
        all=run_all,
        from_csv=from_csv,
        kind=kind,
        force=_state(ctx).force,
    )


@app.command()
def export(
    ctx: typer.Context,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    no_database: bool = typer.Option(False, "--no-database"),
) -> None:
    from pheno_rwe.steps.export import export_study

    _run(ctx, export_study, _workspace(ctx), output=output, include_database=not no_database)


@app.command()
def trace(
    ctx: typer.Context,
    verify: bool = typer.Option(False, "--verify"),
    graph: bool = typer.Option(False, "--graph"),
) -> None:
    from pheno_rwe.steps.trace import trace_study

    try:
        workspace = _workspace(ctx)
        assert workspace is not None
        result = trace_study(workspace, verify=verify, graph=graph)
        if _state(ctx).json_output:
            _render(
                ctx,
                result,
                exit_code=1 if result.verification and not result.verification.valid else 0,
            )
            return
        table = Table("#", "Step", "Status", "Finished", "Hash")
        for index, entry in enumerate(result.entries, 1):
            table.add_row(
                str(index),
                str(entry.get("step")),
                str(entry.get("status")),
                str(entry.get("finished_at", "")),
                str(entry.get("entry_hash", ""))[:12],
            )
        console.print(table)
        if result.verification:
            color = "green" if result.verification.valid else "red"
            status_label = "valid" if result.verification.valid else "failed"
            console.print(f"[{color}]Manifest verification: {status_label}[/{color}]")
            for error in result.verification.errors:
                console.print(f"[red]{error}[/red]")
            if not result.verification.valid:
                raise typer.Exit(code=1)
        if result.graph:
            for parent, child in result.graph:
                console.print(f"{parent} -> {child}")
    except (PhenoRWEError, ValueError) as exc:
        _failure(ctx, str(exc), code=1)


@app.command()
def status(
    ctx: typer.Context,
    check_env: bool = typer.Option(False, "--check-env"),
) -> None:
    from pheno_rwe.steps.status import study_status

    _run(ctx, study_status, _workspace(ctx), check_env=check_env)


if __name__ == "__main__":
    app()
