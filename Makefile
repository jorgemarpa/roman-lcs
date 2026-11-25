.PHONY: all clean pytest mypy ruff-check ruff-check-fix ruff-format build serve deploy

CMD:=uv run
PYMODULE:=src
TESTS:=tests

# Run all the checks which do not change files
all: ruff-check mypy pytest build

# Run unit tests using pytest
pytest:
	$(CMD) pytest $(PYMODULE) $(TESTS)

# Type check using mypy
mypy:
	$(CMD) mypy $(PYMODULE)

# Ruff check
ruff-check:
	$(CMD) ruff check $(PYMODULE) $(TESTS)

# Ruff check with fix
ruff-check-fix:
	$(CMD) ruff check --fix $(PYMODULE) $(TESTS)

# Ruff format
ruff-format:
	$(CMD) ruff format $(PYMODULE) $(TESTS)

# Build docs
build:
	$(CMD) mkdocs build --strict

serve:
	$(CMD) mkdocs serve

deploy:
	$(CMD) mkdocs gh-deploy --force

clean:
	rm -rf dist site .pytest_cache .mypy_cache .ruff_cache
