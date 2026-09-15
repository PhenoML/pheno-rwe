from __future__ import annotations

import base64
import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from pheno_rwe.client import (
    AuditedTransport,
    RetryPolicy,
    SDKTransport,
    TokenBucket,
)
from pheno_rwe.config import DEFAULT_PHENOML_BASE_URL, Settings
from pheno_rwe.errors import ConceptResolverUnavailable, ConfigurationError


class FakeApiError(Exception):
    def __init__(
        self,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        self.headers = headers or {}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class SequenceTransport:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.attempts = 0

    def analyze_cohort(self, text: str, provider: str) -> Any:
        self.attempts += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeRawResponse:
    def __init__(
        self,
        data: Any,
        *,
        headers: dict[str, str] | None = None,
        status_code: int = 200,
    ) -> None:
        self.data = data
        self.headers = headers or {}
        self.status_code = status_code


def test_retries_timeout_and_5xx_with_bounded_exponential_waits() -> None:
    clock = FakeClock()
    inner = SequenceTransport([TimeoutError("slow"), FakeApiError(500), {"patientIds": []}])
    transport = AuditedTransport(
        inner,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.25,
            max_delay_seconds=1.0,
        ),
        sleep=clock.sleep,
        timer=clock.monotonic,
    )

    assert transport.analyze_cohort("synthetic cohort", "provider") == {"patientIds": []}
    assert inner.attempts == 3
    assert clock.sleeps == [0.25, 0.5]
    assert [call.status for call in transport.calls] == ["retry", "retry", "success"]
    assert [call.attempt for call in transport.calls] == [1, 2, 3]


def test_429_honors_retry_after_and_records_each_attempt() -> None:
    clock = FakeClock()
    inner = SequenceTransport([FakeApiError(429, {"rEtRy-AfTeR": "2"}), {"queries": []}])
    transport = AuditedTransport(
        inner,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=2, max_delay_seconds=5),
        sleep=clock.sleep,
        timer=clock.monotonic,
    )

    assert transport.analyze_cohort("synthetic cohort", "provider") == {"queries": []}
    assert clock.sleeps == [2.0]
    calls = transport.manifest_calls()
    assert [(call["status"], call["status_code"]) for call in calls] == [
        ("retry", 429),
        ("success", None),
    ]


def test_audit_records_credits_from_success_and_error_response_headers() -> None:
    class CreditTransport:
        def __init__(self) -> None:
            self.calls = 0

        def analyze_cohort(self, text: str, provider: str) -> Any:
            del text, provider
            self.calls += 1
            if self.calls == 1:
                raise FakeApiError(429, {"X-PhenoML-Credits-Used": "0.25"})
            from pheno_rwe.client import _SDKResponse

            return _SDKResponse(
                data={"patientIds": []},
                headers={"x-credits-used": "1.75"},
                status_code=200,
            )

    clock = FakeClock()
    transport = AuditedTransport(
        CreditTransport(),  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0),
        sleep=clock.sleep,
        timer=clock.monotonic,
    )

    assert transport.analyze_cohort("synthetic", "provider") == {"patientIds": []}
    calls = transport.manifest_calls()
    assert [call["credits"] for call in calls] == [0.25, 1.75]
    assert [call["status_code"] for call in calls] == [429, 200]


def test_retry_after_http_date_is_supported_and_capped() -> None:
    policy = RetryPolicy(max_attempts=2, max_delay_seconds=2)
    error = FakeApiError(429, {"Retry-After": "Tue, 11 Aug 2026 12:00:03 GMT"})

    assert policy.delay_for(error, 1, datetime(2026, 8, 11, 12, tzinfo=UTC)) == 2


def test_fhir2omop_503_is_typed_after_bounded_retries() -> None:
    class ResolverUnavailable:
        def __init__(self) -> None:
            self.attempts = 0

        def fhir2omop(self, bundle: dict[str, Any]) -> Any:
            self.attempts += 1
            raise FakeApiError(503)

    clock = FakeClock()
    inner = ResolverUnavailable()
    transport = AuditedTransport(
        inner,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0,
            max_delay_seconds=0,
        ),
        sleep=clock.sleep,
        timer=clock.monotonic,
    )

    with pytest.raises(ConceptResolverUnavailable) as raised:
        transport.fhir2omop({"resourceType": "Bundle"})

    assert inner.attempts == 3
    assert isinstance(raised.value.__cause__, FakeApiError)
    assert [call.status for call in transport.calls] == ["retry", "retry", "failed"]


def test_token_bucket_is_deterministic_and_thread_safe() -> None:
    clock = FakeClock()
    bucket = TokenBucket(
        4,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: bucket.acquire(), range(8)))

    assert clock.now == pytest.approx(1.75)
    assert clock.sleeps == pytest.approx([0.25] * 7)


def test_settings_use_real_defaults_and_parse_live_client_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    names = (
        "PHENOML_BASE_URL",
        "PHENOML_TIMEOUT_SECONDS",
        "PHENOML_MAX_RPS",
        "PHENOML_RETRY_MAX_ATTEMPTS",
        "PHENOML_RETRY_BASE_SECONDS",
        "PHENOML_RETRY_MAX_WAIT_SECONDS",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    defaults = Settings.from_env(tmp_path / "missing.env")
    assert defaults.phenoml_base_url == DEFAULT_PHENOML_BASE_URL
    assert isinstance(defaults.phenoml_base_url, str)

    monkeypatch.setenv("PHENOML_MAX_RPS", "1.5")
    monkeypatch.setenv("PHENOML_RETRY_MAX_ATTEMPTS", "3")
    configured = Settings.from_env(tmp_path / "missing.env")
    assert configured.phenoml_max_rps == 1.5
    assert configured.phenoml_retry_max_attempts == 3

    monkeypatch.setenv("PHENOML_MAX_RPS", "0")
    with pytest.raises(ConfigurationError, match="greater than zero"):
        Settings.from_env(tmp_path / "missing.env")


def test_study_env_files_are_isolated_without_process_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    names = ("PHENOML_CLIENT_ID", "PHENOML_CLIENT_SECRET", "FHIR_PROVIDER_ID")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    first.write_text(
        "PHENOML_CLIENT_ID=first-client\n"
        "PHENOML_CLIENT_SECRET=first-secret\n"
        "FHIR_PROVIDER_ID=first-provider\n",
        encoding="utf-8",
    )
    second.write_text(
        "PHENOML_CLIENT_ID=second-client\n"
        "PHENOML_CLIENT_SECRET=second-secret\n"
        "FHIR_PROVIDER_ID=second-provider\n",
        encoding="utf-8",
    )
    first.chmod(0o600)
    second.chmod(0o600)

    first_settings = Settings.from_env(first)
    second_settings = Settings.from_env(second)

    assert first_settings.phenoml_client_id == "first-client"
    assert first_settings.fhir_provider_id == "first-provider"
    assert second_settings.phenoml_client_id == "second-client"
    assert second_settings.fhir_provider_id == "second-provider"
    assert all(name not in os.environ for name in names)


def test_sdk_adapter_uses_inspected_fern_signatures_and_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor: dict[str, Any] = {}
    calls: dict[str, Any] = {}

    class FakeGeneratedClient:
        def __init__(self, **kwargs: Any) -> None:
            constructor.update(kwargs)
            self.tools = SimpleNamespace(
                with_raw_response=SimpleNamespace(analyze_cohort=self._analyze)
            )
            self.cohort = SimpleNamespace(with_raw_response=SimpleNamespace(analyze=self._cohort))
            self.fhir = SimpleNamespace(with_raw_response=SimpleNamespace(search=self._fhir))
            self.fhir2omop = SimpleNamespace(
                with_raw_response=SimpleNamespace(create=self._fhir2omop)
            )
            self.lang2fhir = SimpleNamespace(
                with_raw_response=SimpleNamespace(
                    document=self._document,
                    document_multi=self._document_multi,
                )
            )
            self.construe = SimpleNamespace(
                codes=SimpleNamespace(with_raw_response=SimpleNamespace(extract=self._extract))
            )

        @staticmethod
        def _raw(data: dict[str, Any] | None = None) -> FakeRawResponse:
            return FakeRawResponse(data or {}, headers={"X-PhenoML-Credits-Used": "2"})

        def _analyze(self, **kwargs: Any) -> dict[str, Any]:
            calls["analyze"] = kwargs
            return self._raw()

        def _cohort(self, **kwargs: Any) -> dict[str, Any]:
            calls["cohort"] = kwargs
            return self._raw()

        def _fhir(self, provider: str, path: str, **kwargs: Any) -> dict[str, Any]:
            calls["fhir"] = (provider, path, kwargs)
            return self._raw()

        def _fhir2omop(self, **kwargs: Any) -> dict[str, Any]:
            calls["fhir2omop"] = kwargs
            return self._raw()

        def _document(self, **kwargs: Any) -> dict[str, Any]:
            calls["document"] = kwargs
            return self._raw()

        def _document_multi(self, **kwargs: Any) -> dict[str, Any]:
            calls["document_multi"] = kwargs
            return self._raw()

        def _extract(self, **kwargs: Any) -> dict[str, Any]:
            calls["extract"] = kwargs
            return self._raw()

    module = ModuleType("phenoml")
    module.PhenomlClient = FakeGeneratedClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "phenoml", module)
    settings = Settings(
        phenoml_client_id="synthetic-client",
        phenoml_client_secret="synthetic-secret",
    )
    transport = SDKTransport(settings)

    assert transport.analyze_cohort("synthetic cohort", "provider").headers
    assert transport.cohort_queries("synthetic cohort", "ignored-provider").headers
    assert transport.fhir_search("provider", "Patient", _count=10).headers
    assert transport.fhir2omop({"resourceType": "Bundle"}).headers
    assert transport.document(b"synthetic note", "text/plain").headers
    assert transport.document(b"synthetic image", "image/png", detection_effort="standard").headers
    assert transport.resolve_codings("synthetic diagnosis", "condition").headers

    assert constructor["max_retries"] == 0
    assert constructor["timeout"] == settings.phenoml_timeout_seconds
    assert constructor["follow_redirects"] is False
    assert calls["analyze"] == {"text": "synthetic cohort", "provider": "provider"}
    assert calls["cohort"] == {"text": "synthetic cohort"}
    assert calls["fhir"] == (
        "provider",
        "Patient",
        {"request_options": {"additional_query_parameters": {"_count": 10}}},
    )
    assert calls["fhir2omop"] == {"fhir_resources": {"resourceType": "Bundle"}}
    assert calls["document"]["content"] == base64.b64encode(b"synthetic note").decode("ascii")
    assert calls["document_multi"]["detection_effort"] == "standard"
    assert calls["extract"] == {"text": "synthetic diagnosis"}


def test_fixture_recorder_refuses_without_environment_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_path = Path(__file__).parents[2] / "scripts" / "record_fixtures.py"
    spec = importlib.util.spec_from_file_location("record_fixtures", script_path)
    assert spec is not None and spec.loader is not None
    record_fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(record_fixtures)

    monkeypatch.delenv(record_fixtures.OPT_IN_ENV, raising=False)
    monkeypatch.setattr(
        record_fixtures,
        "_record",
        lambda args: pytest.fail("live recorder must not run"),
    )

    with pytest.raises(SystemExit) as raised:
        record_fixtures.main(
            [
                "--confirm-live-phenoml-call",
                "--synthetic-only",
                "--output-dir",
                str(tmp_path),
                "--coding-text",
                "synthetic diagnosis",
            ]
        )

    assert raised.value.code == 2
