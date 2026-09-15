"""One audited choke point for every PhenoML API operation."""

from __future__ import annotations

import base64
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol, runtime_checkable

import httpx

from pheno_rwe.config import Settings
from pheno_rwe.errors import ConceptResolverUnavailable, ConfigurationError


@runtime_checkable
class PhenoTransport(Protocol):
    def analyze_cohort(self, text: str, provider: str) -> Any: ...

    def cohort_queries(self, text: str, provider: str) -> Any: ...

    def fhir_search(self, provider: str, fhir_path: str, **params: Any) -> Any: ...

    def fhir2omop(self, bundle: dict[str, Any]) -> Any: ...

    def document(self, content: bytes, mime_type: str, **options: Any) -> Any: ...

    def resolve_codings(self, text: str, domain: str | None = None) -> Any: ...


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _response_headers(exc: Exception) -> Mapping[str, Any]:
    headers = getattr(exc, "headers", None)
    response = getattr(exc, "response", None)
    if headers is None and response is not None:
        headers = getattr(response, "headers", None)
    return headers if isinstance(headers, Mapping) else {}


_CREDIT_HEADERS = (
    "x-phenoml-credits-used",
    "x-phenoml-credits",
    "x-credits-used",
    "x-credit-usage",
    "x-usage-credits",
)


def _credits_from_headers(headers: Mapping[str, Any]) -> float | None:
    """Extract a finite, non-negative credit charge from response metadata."""

    normalized = {str(name).lower(): value for name, value in headers.items()}
    for name in _CREDIT_HEADERS:
        value = normalized.get(name)
        if value is None:
            continue
        try:
            credits = float(str(value).strip())
        except (TypeError, ValueError):
            continue
        if math.isfinite(credits) and credits >= 0:
            return credits
    return None


def _retry_after_seconds(exc: Exception, now: datetime) -> float | None:
    value: Any = None
    for name, candidate in _response_headers(exc).items():
        if str(name).lower() == "retry-after":
            value = candidate
            break
    if value is None:
        return None
    raw = str(value).strip()
    try:
        seconds = float(raw)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - now).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(
        exc,
        (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.TransportError),
    ):
        return True
    status = _status_code(exc)
    return status == 429 or (status is not None and 500 <= status <= 599)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Deterministic, bounded retry settings for one logical API call."""

    max_attempts: int = 5
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("retry max_attempts must be at least 1")
        if not math.isfinite(self.base_delay_seconds) or self.base_delay_seconds < 0:
            raise ValueError("retry base_delay_seconds must be finite and non-negative")
        if not math.isfinite(self.max_delay_seconds) or self.max_delay_seconds < 0:
            raise ValueError("retry max_delay_seconds must be finite and non-negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("retry max_delay_seconds must be at least base_delay_seconds")

    def delay_for(self, exc: Exception, failed_attempt: int, now: datetime) -> float:
        retry_after = _retry_after_seconds(exc, now)
        if retry_after is not None:
            return min(retry_after, self.max_delay_seconds)
        exponential = self.base_delay_seconds * (2 ** max(0, failed_attempt - 1))
        return min(exponential, self.max_delay_seconds)


class TokenBucket:
    """Thread-safe token bucket with deterministic, jitter-free waits.

    A capacity of one deliberately prevents bursts: ``max_rps`` is both the
    sustained rate and the strict request-start rate used by the live client.
    """

    def __init__(
        self,
        max_rps: float,
        *,
        capacity: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(max_rps) or max_rps <= 0:
            raise ValueError("max_rps must be finite and greater than zero")
        if not math.isfinite(capacity) or capacity < 1:
            raise ValueError("token-bucket capacity must be finite and at least one")
        self.max_rps = float(max_rps)
        self.capacity = float(capacity)
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._updated_at = self._clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        # Keeping the lock while waiting ensures concurrent callers cannot all
        # reserve the same future token. Live pipeline loops are serial, but the
        # Concurrent CLI work may invoke transports from worker threads.
        with self._lock:
            while True:
                now = self._clock()
                elapsed = max(0.0, now - self._updated_at)
                self._tokens = min(
                    self.capacity,
                    self._tokens + elapsed * self.max_rps,
                )
                self._updated_at = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                self._sleep((1.0 - self._tokens) / self.max_rps)


@dataclass(slots=True)
class ApiCall:
    endpoint: str
    milliseconds: float
    credits: float | None = None
    status: str = "success"
    status_code: int | None = None
    attempt: int = 1


@dataclass(frozen=True, slots=True)
class _SDKResponse:
    """Private raw-response carrier between the SDK and audit layers."""

    data: Any
    headers: Mapping[str, Any]
    status_code: int


@dataclass(slots=True)
class AuditedTransport:
    """Apply rate limiting, retries, typed errors, and audit recording."""

    inner: PhenoTransport
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    limiter: TokenBucket | None = None
    calls: list[ApiCall] = field(default_factory=list)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    timer: Callable[[], float] = field(default=time.perf_counter, repr=False)
    wall_clock: Callable[[], datetime] = field(default=_utcnow, repr=False)
    _calls_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def _record_call(
        self,
        endpoint: str,
        started: float,
        *,
        status: str,
        status_code: int | None,
        attempt: int,
        credits: float | None = None,
    ) -> None:
        call = ApiCall(
            endpoint=endpoint,
            milliseconds=(self.timer() - started) * 1000,
            status=status,
            status_code=status_code,
            attempt=attempt,
            credits=credits,
        )
        with self._calls_lock:
            self.calls.append(call)

    def _invoke(self, endpoint: str, method: str, *args: Any, **kwargs: Any) -> Any:
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            if self.limiter is not None:
                self.limiter.acquire()
            started = self.timer()
            try:
                result = getattr(self.inner, method)(*args, **kwargs)
            except Exception as exc:
                retry = _is_retryable(exc) and attempt < self.retry_policy.max_attempts
                self._record_call(
                    endpoint,
                    started,
                    status="retry" if retry else "failed",
                    status_code=_status_code(exc),
                    attempt=attempt,
                    credits=_credits_from_headers(_response_headers(exc)),
                )
                if not retry:
                    raise
                delay = self.retry_policy.delay_for(exc, attempt, self.wall_clock())
                self.sleep(delay)
            else:
                if isinstance(result, _SDKResponse):
                    data = result.data
                    headers = result.headers
                    status_code = result.status_code
                else:
                    data = result
                    headers = {}
                    status_code = None
                self._record_call(
                    endpoint,
                    started,
                    status="success",
                    status_code=status_code,
                    attempt=attempt,
                    credits=_credits_from_headers(headers),
                )
                return data
        raise AssertionError("bounded retry loop exited without returning or raising")

    def analyze_cohort(self, text: str, provider: str) -> Any:
        return self._invoke("/tools/cohort", "analyze_cohort", text, provider)

    def cohort_queries(self, text: str, provider: str) -> Any:
        return self._invoke("/cohort", "cohort_queries", text, provider)

    def fhir_search(self, provider: str, fhir_path: str, **params: Any) -> Any:
        return self._invoke(
            "/fhir/search",
            "fhir_search",
            provider,
            fhir_path,
            **params,
        )

    def fhir2omop(self, bundle: dict[str, Any]) -> Any:
        try:
            return self._invoke("/fhir2omop/create", "fhir2omop", bundle)
        except Exception as exc:
            if _status_code(exc) == 503:
                raise ConceptResolverUnavailable(
                    "PhenoML's concept resolver is temporarily unavailable; "
                    "completed patients were kept."
                ) from exc
            raise

    def document(self, content: bytes, mime_type: str, **options: Any) -> Any:
        endpoint = (
            "/lang2fhir/document" if mime_type == "text/plain" else "/lang2fhir/document/multi"
        )
        return self._invoke(endpoint, "document", content, mime_type, **options)

    def resolve_codings(self, text: str, domain: str | None = None) -> Any:
        return self._invoke("/construe/codes/extract", "resolve_codings", text, domain)

    def manifest_calls(self) -> list[dict[str, Any]]:
        with self._calls_lock:
            calls = list(self.calls)
        return [
            {
                "endpoint": call.endpoint,
                "count": 1,
                "ms": round(call.milliseconds, 3),
                "credits": call.credits,
                "status": call.status,
                "status_code": call.status_code,
                "attempt": call.attempt,
            }
            for call in calls
        ]


class SDKTransport:
    """Adapter around the generated PhenoML SDK; imported only when requested."""

    def __init__(self, settings: Settings) -> None:
        settings.require_phenoml()
        try:
            from phenoml import PhenomlClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigurationError(
                "Install the PhenoML integration with 'pip install pheno-rwe[phenoml]'."
            ) from exc
        kwargs: dict[str, Any] = {
            "base_url": settings.phenoml_base_url,
            # Credentials and clinical requests must never follow an
            # unreviewed redirect to a different destination.
            "follow_redirects": False,
            # The audited outer layer owns the one bounded retry policy. Keeping
            # Fern retries disabled prevents multiplicative retry counts.
            "max_retries": 0,
            "timeout": settings.phenoml_timeout_seconds,
        }
        if settings.auth_mode == "client_credentials":
            kwargs.update(
                client_id=settings.phenoml_client_id,
                client_secret=settings.phenoml_client_secret,
            )
        else:
            # Modern generated clients use OAuth only. Support the documented
            # compatibility fallback by obtaining a bearer token lazily.
            username = settings.phenoml_username
            password = settings.phenoml_password

            def password_token() -> str:
                response = httpx.post(
                    settings.phenoml_base_url.rstrip("/") + "/v2/auth/token",
                    json={"username": username, "password": password},
                    timeout=settings.phenoml_timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                token = data.get("access_token") or data.get("token")
                if not token:
                    raise ConfigurationError(
                        "PhenoML password authentication returned no bearer token."
                    )
                return str(token)

            kwargs["token"] = password_token
        self.client = PhenomlClient(**kwargs)

    @staticmethod
    def _response(value: Any) -> _SDKResponse:
        return _SDKResponse(
            data=value.data,
            headers=value.headers,
            status_code=int(value.status_code),
        )

    def analyze_cohort(self, text: str, provider: str) -> Any:
        return self._response(
            self.client.tools.with_raw_response.analyze_cohort(text=text, provider=provider)
        )

    def cohort_queries(self, text: str, provider: str) -> Any:
        del provider  # The query-only generated endpoint does not accept a provider.
        return self._response(self.client.cohort.with_raw_response.analyze(text=text))

    def fhir_search(self, provider: str, fhir_path: str, **params: Any) -> Any:
        # Fern exposes arbitrary FHIR query parameters through RequestOptions.
        if not params:
            return self._response(self.client.fhir.with_raw_response.search(provider, fhir_path))
        return self._response(
            self.client.fhir.with_raw_response.search(
                provider,
                fhir_path,
                request_options={"additional_query_parameters": params},
            )
        )

    def fhir2omop(self, bundle: dict[str, Any]) -> Any:
        return self._response(self.client.fhir2omop.with_raw_response.create(fhir_resources=bundle))

    def document(self, content: bytes, mime_type: str, **options: Any) -> Any:
        encoded = base64.b64encode(content).decode("ascii")
        if mime_type == "text/plain":
            return self._response(
                self.client.lang2fhir.with_raw_response.document(
                    version="R4",
                    resource="auto",
                    content=encoded,
                )
            )
        return self._response(
            self.client.lang2fhir.with_raw_response.document_multi(
                version="R4",
                content=encoded,
                **options,
            )
        )

    def resolve_codings(self, text: str, domain: str | None = None) -> Any:
        # The domain is an OMOP domain, not a terminology system accepted by
        # Construe. The fhir2omop probe remains authoritative for concept IDs.
        del domain
        return self._response(self.client.construe.codes.with_raw_response.extract(text=text))


def build_transport(
    settings: Settings | None = None,
    *,
    max_rps: float | None = None,
) -> AuditedTransport:
    resolved = settings or Settings.from_env()
    try:
        retry_policy = RetryPolicy(
            max_attempts=resolved.phenoml_retry_max_attempts,
            base_delay_seconds=resolved.phenoml_retry_base_seconds,
            max_delay_seconds=resolved.phenoml_retry_max_wait_seconds,
        )
        limiter = TokenBucket(resolved.phenoml_max_rps if max_rps is None else max_rps)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    return AuditedTransport(
        SDKTransport(resolved),
        retry_policy=retry_policy,
        limiter=limiter,
    )
