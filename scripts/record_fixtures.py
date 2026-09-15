#!/usr/bin/env python3
"""Explicitly record synthetic PhenoML API fixtures for developer use.

This script is intentionally absent from normal tests and refuses to make a
network call unless both a command-line acknowledgement and an environment
opt-in are present. It never pulls patient data from a FHIR provider.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from pheno_rwe.client import build_transport
from pheno_rwe.config import Settings
from pheno_rwe.hashing import canonical_json
from pheno_rwe.serialization import to_data

OPT_IN_ENV = "PHENO_RWE_RECORD_FIXTURES"
OPT_IN_VALUE = "synthetic-only"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record selected PhenoML responses from researcher-supplied synthetic inputs. "
            "This command makes live API calls and is never run by the test suite."
        )
    )
    parser.add_argument(
        "--confirm-live-phenoml-call",
        action="store_true",
        required=True,
        help="Acknowledge that this command sends inputs to the configured live API.",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        required=True,
        help="Confirm that every supplied input is synthetic and contains no PHI.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rps", type=float, default=None)
    parser.add_argument(
        "--bundle",
        type=Path,
        help="Synthetic Patient-containing FHIR Bundle for fhir2omop.",
    )
    parser.add_argument(
        "--document",
        type=Path,
        help="Synthetic text, PDF, PNG, JPEG, or TIFF for lang2fhir.",
    )
    parser.add_argument(
        "--mime-type",
        help="MIME type for --document (for example application/pdf).",
    )
    parser.add_argument(
        "--coding-text",
        help="Synthetic clinical phrase for Construe code extraction.",
    )
    parser.add_argument(
        "--coding-domain",
        help="Optional OMOP domain recorded with the code extraction request.",
    )
    parser.add_argument(
        "--cohort-text",
        help="Synthetic cohort phrase for query derivation only; no query is executed.",
    )
    return parser


def _targets(args: argparse.Namespace) -> dict[str, Path]:
    names: list[str] = []
    if args.bundle is not None:
        names.append("fhir2omop.json")
    if args.document is not None:
        names.append("document.json")
    if args.coding_text is not None:
        names.append("codings.json")
    if args.cohort_text is not None:
        names.append("cohort_queries.json")
    return {name: args.output_dir / name for name in names}


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Path]:
    if os.getenv(OPT_IN_ENV) != OPT_IN_VALUE:
        parser.error(
            f"set {OPT_IN_ENV}={OPT_IN_VALUE!r} to explicitly enable live fixture recording"
        )
    if not any(
        value is not None
        for value in (args.bundle, args.document, args.coding_text, args.cohort_text)
    ):
        parser.error("select at least one synthetic input to record")
    if args.document is not None and not args.mime_type:
        parser.error("--mime-type is required with --document")
    for path in (args.bundle, args.document):
        if path is not None and not path.is_file():
            parser.error(f"input file does not exist: {path}")
    targets = _targets(args)
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "refusing to overwrite existing fixtures without --overwrite: " + ", ".join(existing)
        )
    return targets


def _record(args: argparse.Namespace) -> dict[str, Any]:
    transport = build_transport(
        Settings.from_env(args.env_file),
        max_rps=args.max_rps,
    )
    responses: dict[str, Any] = {}
    if args.bundle is not None:
        bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
        if not isinstance(bundle, dict):
            raise ValueError("--bundle must contain one JSON object")
        responses["fhir2omop.json"] = to_data(transport.fhir2omop(bundle))
    if args.document is not None:
        responses["document.json"] = to_data(
            transport.document(args.document.read_bytes(), args.mime_type)
        )
    if args.coding_text is not None:
        responses["codings.json"] = to_data(
            transport.resolve_codings(args.coding_text, args.coding_domain)
        )
    if args.cohort_text is not None:
        responses["cohort_queries.json"] = to_data(
            transport.cohort_queries(args.cohort_text, "queries-only")
        )
    return responses


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    targets = _validate(args, parser)
    try:
        responses = _record(args)
    except Exception as exc:
        # Generated SDK exceptions can include response bodies. Do not echo
        # them into a terminal or CI log where synthetic-only discipline may
        # have failed.
        print(
            f"fixture recording failed ({type(exc).__name__}); no response body was printed",
            file=sys.stderr,
        )
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, response in responses.items():
        targets[name].write_text(canonical_json(response) + "\n", encoding="utf-8")
        print(f"recorded {targets[name]}")
    print("Review generated files for identifiers before adding them to version control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
