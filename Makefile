.PHONY: help install dev-install test lint format clean run-monitor run-tests

help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make dev-install  - Install development dependencies"
	@echo "  make test        - Run tests with coverage"
	@echo "  make lint        - Run linters (flake8, mypy, bandit)"
	@echo "  make format      - Format code with black and isort"
	@echo "  make clean       - Clean up cache and build files"
	@echo "  make run-monitor - Run kimchi premium monitor"
	@echo "  make run-tests   - Run all test suites"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev-install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=term --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports
	bandit -r src/ -ll

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true

run-monitor:
	python scripts/monitor_kimchi_premium.py

run-tests:
	python scripts/test_exchange_rate.py
	python scripts/test_kimchi_integration.py
	python tests/test_websocket_manager.py
	python tests/test_kimchi_premium.py

check-security:
	safety check
	bandit -r src/ -f json -o security-report.json

pre-commit-all:
	pre-commit run --all-files