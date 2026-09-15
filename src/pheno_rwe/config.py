"""Environment configuration with explicit local/cloud boundaries."""

from __future__ import annotations

import ipaddress
import math
import os
import re
import stat
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values, find_dotenv

from pheno_rwe.errors import ConfigurationError

DEFAULT_PHENOML_BASE_URL = "https://api.phenoml.com"
DEFAULT_PHENOML_TIMEOUT_SECONDS = 300.0
DEFAULT_PHENOML_MAX_RPS = 2.0
DEFAULT_PHENOML_RETRY_MAX_ATTEMPTS = 5
DEFAULT_PHENOML_RETRY_BASE_SECONDS = 0.5
DEFAULT_PHENOML_RETRY_MAX_WAIT_SECONDS = 30.0
_MAX_ENV_BYTES = 1024 * 1024


def normalize_phenoml_base_url(value: str) -> str:
    """Validate and conservatively normalize a PhenoML API destination."""

    if not value:
        raise ConfigurationError("PHENOML_BASE_URL must be a non-empty HTTPS URL.")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ConfigurationError("PHENOML_BASE_URL must not contain control characters.")
    if any(character.isspace() for character in value) or "\\" in value:
        raise ConfigurationError("PHENOML_BASE_URL must not contain whitespace or backslashes.")
    try:
        parsed = urlsplit(value)
        # Accessing port performs validation that urlsplit otherwise defers.
        _ = parsed.port
    except ValueError as exc:
        raise ConfigurationError("PHENOML_BASE_URL must be a valid HTTPS URL.") from exc
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ConfigurationError("PHENOML_BASE_URL must use HTTPS and include a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError("PHENOML_BASE_URL must not contain user information.")
    if parsed.query or parsed.fragment:
        raise ConfigurationError("PHENOML_BASE_URL must not contain a query or fragment.")
    if parsed.path not in ("", "/"):
        raise ConfigurationError("PHENOML_BASE_URL must be an origin without a path.")

    hostname = parsed.hostname
    try:
        normalized_hostname = hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise ConfigurationError("PHENOML_BASE_URL must contain a valid hostname.") from exc
    if not normalized_hostname:
        raise ConfigurationError("PHENOML_BASE_URL must contain a valid hostname.")
    literal_address: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    if ":" in normalized_hostname:
        try:
            literal_address = ipaddress.IPv6Address(normalized_hostname)
        except ValueError as exc:
            raise ConfigurationError("PHENOML_BASE_URL must contain a valid hostname.") from exc
        normalized_host_port = f"[{normalized_hostname}]"
    else:
        labels = normalized_hostname.split(".")
        if len(normalized_hostname) > 253 or any(
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or re.fullmatch(r"[a-z0-9-]+", label) is None
            for label in labels
        ):
            raise ConfigurationError("PHENOML_BASE_URL must contain a valid hostname.")
        try:
            literal_address = ipaddress.IPv4Address(normalized_hostname)
        except ValueError:
            # Reject ambiguous all-numeric spellings that URL clients may
            # interpret as an IPv4 literal (for example 2130706433).
            if re.fullmatch(r"[0-9.]+", normalized_hostname):
                raise ConfigurationError(
                    "PHENOML_BASE_URL must contain an unambiguous hostname."
                ) from None
        normalized_host_port = normalized_hostname
    if literal_address is not None and not literal_address.is_global:
        raise ConfigurationError(
            "PHENOML_BASE_URL must not use a private, loopback, link-local, reserved, "
            "or multicast IP address."
        )
    if parsed.port is not None and parsed.port != 443:
        normalized_host_port = f"{normalized_host_port}:{parsed.port}"

    return urlunsplit(("https", normalized_host_port, "", "", ""))


def _dotenv_values_from_file(path: str | Path) -> Mapping[str, str | None]:
    """Read credentials without following links or mutating the process environment."""

    selected = Path(path)
    try:
        expected = selected.lstat()
    except FileNotFoundError:
        return {}
    if stat.S_ISLNK(expected.st_mode):
        raise ConfigurationError("Refusing to use a symlinked environment file.")
    if not stat.S_ISREG(expected.st_mode):
        raise ConfigurationError("The environment file must be a regular file.")
    if expected.st_nlink != 1:
        raise ConfigurationError("Refusing to use a hard-linked environment file.")
    if expected.st_uid != os.getuid():
        raise ConfigurationError("The environment file must be owned by the current user.")
    if stat.S_IMODE(expected.st_mode) != 0o600:
        raise ConfigurationError("The environment file permissions must be 0600.")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(selected, flags)
    except OSError as exc:
        raise ConfigurationError("The environment file could not be opened safely.") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            raise ConfigurationError("The environment file changed while it was being opened.")
        if opened.st_size > _MAX_ENV_BYTES:
            raise ConfigurationError("The environment file is too large.")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, _MAX_ENV_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_ENV_BYTES:
                raise ConfigurationError("The environment file is too large.")
    finally:
        os.close(descriptor)
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError("The environment file must be valid UTF-8.") from exc
    return dotenv_values(stream=StringIO(text), interpolate=False)


def _positive_float(name: str, default: float, getenv: Callable[[str], str | None]) -> float:
    raw = getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be finite and greater than zero.")
    return value


def _positive_int(name: str, default: int, getenv: Callable[[str], str | None]) -> int:
    raw = getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if value < 1:
        raise ConfigurationError(f"{name} must be at least 1.")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    phenoml_base_url: str = DEFAULT_PHENOML_BASE_URL
    phenoml_client_id: str | None = None
    phenoml_client_secret: str | None = field(default=None, repr=False)
    phenoml_username: str | None = None
    phenoml_password: str | None = field(default=None, repr=False)
    phenoml_timeout_seconds: float = DEFAULT_PHENOML_TIMEOUT_SECONDS
    phenoml_max_rps: float = DEFAULT_PHENOML_MAX_RPS
    phenoml_retry_max_attempts: int = DEFAULT_PHENOML_RETRY_MAX_ATTEMPTS
    phenoml_retry_base_seconds: float = DEFAULT_PHENOML_RETRY_BASE_SECONDS
    phenoml_retry_max_wait_seconds: float = DEFAULT_PHENOML_RETRY_MAX_WAIT_SECONDS
    fhir_provider_id: str | None = None

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> Settings:
        # Read a study-local .env without mutating os.environ so credentials
        # cannot bleed into a later command's configuration.
        selected_file: str | Path | None = env_file
        if selected_file is None:
            discovered = find_dotenv(usecwd=True)
            selected_file = discovered or None
        file_values: Mapping[str, str | None] = (
            _dotenv_values_from_file(selected_file) if selected_file is not None else {}
        )

        def getenv(name: str) -> str | None:
            return os.environ[name] if name in os.environ else file_values.get(name)

        retry_base = _positive_float(
            "PHENOML_RETRY_BASE_SECONDS",
            DEFAULT_PHENOML_RETRY_BASE_SECONDS,
            getenv,
        )
        retry_max_wait = _positive_float(
            "PHENOML_RETRY_MAX_WAIT_SECONDS",
            DEFAULT_PHENOML_RETRY_MAX_WAIT_SECONDS,
            getenv,
        )
        if retry_max_wait < retry_base:
            raise ConfigurationError(
                "PHENOML_RETRY_MAX_WAIT_SECONDS must be at least PHENOML_RETRY_BASE_SECONDS."
            )
        return cls(
            phenoml_base_url=normalize_phenoml_base_url(
                getenv("PHENOML_BASE_URL") or DEFAULT_PHENOML_BASE_URL
            ),
            phenoml_client_id=getenv("PHENOML_CLIENT_ID"),
            phenoml_client_secret=getenv("PHENOML_CLIENT_SECRET"),
            phenoml_username=getenv("PHENOML_USERNAME"),
            phenoml_password=getenv("PHENOML_PASSWORD"),
            phenoml_timeout_seconds=_positive_float(
                "PHENOML_TIMEOUT_SECONDS",
                DEFAULT_PHENOML_TIMEOUT_SECONDS,
                getenv,
            ),
            phenoml_max_rps=_positive_float(
                "PHENOML_MAX_RPS",
                DEFAULT_PHENOML_MAX_RPS,
                getenv,
            ),
            phenoml_retry_max_attempts=_positive_int(
                "PHENOML_RETRY_MAX_ATTEMPTS",
                DEFAULT_PHENOML_RETRY_MAX_ATTEMPTS,
                getenv,
            ),
            phenoml_retry_base_seconds=retry_base,
            phenoml_retry_max_wait_seconds=retry_max_wait,
            fhir_provider_id=getenv("FHIR_PROVIDER_ID"),
        )

    @property
    def auth_mode(self) -> str | None:
        if self.phenoml_client_id and self.phenoml_client_secret:
            return "client_credentials"
        if self.phenoml_username and self.phenoml_password:
            return "password"
        return None

    def require_phenoml(self) -> None:
        if self.auth_mode is None:
            raise ConfigurationError(
                "Set PHENOML_CLIENT_ID/PHENOML_CLIENT_SECRET (preferred) or "
                "PHENOML_USERNAME/PHENOML_PASSWORD."
            )

    def require_fhir_provider(self) -> str:
        if not self.fhir_provider_id:
            raise ConfigurationError("Set FHIR_PROVIDER_ID for live FHIR operations.")
        return self.fhir_provider_id
