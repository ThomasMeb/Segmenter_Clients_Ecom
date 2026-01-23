# Makefile for Olist Customer Segmentation
# ==========================================

.PHONY: help install install-dev test lint format train serve clean check all

# Default target
help:
	@echo "Olist Customer Segmentation - Available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install      Install production dependencies"
	@echo "    make install-dev  Install with development dependencies"
	@echo ""
	@echo "  Development:"
	@echo "    make test         Run tests with coverage"
	@echo "    make lint         Run linters (ruff, mypy)"
	@echo "    make format       Auto-format code with ruff"
	@echo "    make check        Run all checks (lint + test)"
	@echo ""
	@echo "  ML Pipeline:"
	@echo "    make train        Train the segmentation model"
	@echo "    make evaluate     Evaluate model metrics"
	@echo "    make serve        Launch Streamlit dashboard"
	@echo ""
	@echo "  Utilities:"
	@echo "    make clean        Remove build artifacts and caches"
	@echo "    make info         Show project information"
	@echo "    make all          Run install-dev, check, and train"

# =============================================================================
# SETUP
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install
	@echo ""
	@echo "Development environment ready!"

# =============================================================================
# DEVELOPMENT
# =============================================================================

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report: open htmlcov/index.html"

lint:
	@echo "Running ruff..."
	ruff check src/ tests/ app/
	@echo ""
	@echo "Running mypy..."
	mypy src/ --ignore-missing-imports

format:
	@echo "Formatting with ruff..."
	ruff format src/ tests/ app/
	ruff check --fix src/ tests/ app/

check: lint test
	@echo ""
	@echo "All checks passed!"

# =============================================================================
# ML PIPELINE
# =============================================================================

train:
	@echo "Training customer segmentation model..."
	python -m src.cli train --verbose

evaluate:
	@echo "Evaluating model..."
	python -m src.cli evaluate --verbose

serve:
	@echo "Starting Streamlit dashboard..."
	python -m src.cli serve

# Alternative: direct streamlit command
serve-direct:
	streamlit run app/app.py

# =============================================================================
# UTILITIES
# =============================================================================

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done!"

info:
	python -m src.cli info --verbose

# =============================================================================
# COMBINED TARGETS
# =============================================================================

all: install-dev check train
	@echo ""
	@echo "Setup complete! Run 'make serve' to start the dashboard."
