.PHONY: all clean pytest coverage flake8 black mypy isort

CMD:=poetry run
PYMODULE:=src
TESTS:=tests

# Run all the checks which do not change files
all: pytest mypy ruff-check ruff-check-fix ruff-format build serve deploy

# Run the unit tests using `pytest`
pytest:
	$(CMD) pytest $(PYMODULE) $(TESTS)

# Generate a unit test coverage report using `pytest-cov`
coverage:
	$(CMD) pytest --cov=$(PYMODULE) $(TESTS) --cov-report html

# Run the unit tests using `pytest`
mypy:
	$(CMD) mypy $(PYMODULE)

# Ruff 
ruff-check:
	$(CMD) ruff check $(PYMODULE) $(TESTS)

# Ruff 
ruff-check-fix:
	$(CMD) ruff check --fix $(PYMODULE) $(TESTS)

# Ruff 
ruff-format:
	$(CMD) ruff format $(PYMODULE) $(TESTS)

# build docs
build:
	$(CMD) mkdocs build

serve:
	$(CMD) mkdocs serve

deploy:
	$(CMD) mkdocs gh-deploy --force