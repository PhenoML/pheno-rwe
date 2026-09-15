.PHONY: install format format-check lint typecheck test check clean

install:
	uv sync --all-extras --dev

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run pyright

test:
	uv run pytest

# Mirrors the CI gate (.github/workflows/ci.yml).
check: lint format-check typecheck test

clean:
	rm -rf build dist .pytest_cache .ruff_cache
