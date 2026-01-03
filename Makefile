.PHONY: help install install-dev test test-unit test-integration test-gpu coverage lint format type-check security profile clean all

# Default target
help:
	@echo "Lambda Synthesis Experiments - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  install           - Install production dependencies"
	@echo "  install-dev       - Install development dependencies"
	@echo "  test              - Run all tests"
	@echo "  test-unit         - Run unit tests only"
	@echo "  test-integration  - Run integration tests only"
	@echo "  test-gpu          - Run GPU-specific tests"
	@echo "  coverage          - Run tests with coverage report"
	@echo "  lint              - Run all linters (ruff, pylint)"
	@echo "  format            - Format code (black, isort)"
	@echo "  type-check        - Run type checking (mypy)"
	@echo "  security          - Run security analysis (bandit, safety)"
	@echo "  profile           - Run performance profiling"
	@echo "  clean             - Clean build artifacts"
	@echo "  all               - Run format, lint, type-check, security, and test"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,formal,analysis]"

# Testing
test:
	pytest -v

test-unit:
	pytest -v -m "unit"

test-integration:
	pytest -v -m "integration"

test-gpu:
	pytest -v -m "gpu"

coverage:
	pytest --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml

# Code Quality
lint:
	@echo "Running ruff..."
	ruff check src/ tests/
	@echo "Running pylint..."
	pylint src/

format:
	@echo "Running black..."
	black src/ tests/
	@echo "Running isort..."
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

type-check:
	mypy src/

# Security
security:
	@echo "Running bandit security scanner..."
	bandit -r src/ -f json -o bandit-report.json || true
	bandit -r src/ -f screen
	@echo "Running safety dependency checker..."
	safety check --json || true

# Profiling and Analysis
profile:
	@echo "Profiling USS pipeline..."
	python -m cProfile -o profile.stats src/experiments/uss_pipeline.py
	@echo "Profile saved to profile.stats"

profile-memory:
	@echo "Memory profiling USS pipeline..."
	python -m memory_profiler src/experiments/uss_pipeline.py

flamegraph:
	@echo "Generating flamegraph..."
	py-spy record -o flamegraph.svg --duration 30 -- python src/experiments/uss_pipeline.py
	@echo "Flamegraph saved to flamegraph.svg"

# Data Generation
generate-data:
	python src/data/generator.py

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf profile.stats
	rm -rf flamegraph.svg
	rm -rf bandit-report.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Full validation pipeline
all: format lint type-check security test
	@echo "✓ All checks passed!"

# Continuous Integration target
ci: format-check lint type-check security coverage
	@echo "✓ CI validation complete!"
