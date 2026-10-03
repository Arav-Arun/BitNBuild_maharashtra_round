UV ?= uv
RUN = $(UV) run --locked --extra dev

.PHONY: help setup check lint format test build

help:
	@echo "setup   Install the locked development environment"
	@echo "check   Run lint, formatting checks, and tests"
	@echo "format  Format Python and sort imports"
	@echo "build   Build wheel and source distribution in dist/"

setup:
	$(UV) sync --locked --extra dev

check: lint test

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff check --select I --fix .
	$(RUN) ruff format .

test:
	$(RUN) python -m unittest discover -s tests -v

build:
	$(UV) build
