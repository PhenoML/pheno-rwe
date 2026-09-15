.PHONY: install test lint typecheck check clean

install:
	uv sync --all-extras --dev

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run pyright

check: lint typecheck test

clean:
	rm -rf build dist .pytest_cache .ruff_cache
