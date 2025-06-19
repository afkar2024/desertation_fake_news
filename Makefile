# Makefile for Adaptive Fake News Detector

.PHONY: help install install-dev install-ml run test format lint clean

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install basic dependencies"
	@echo "  install-dev - Install with development dependencies"
	@echo "  install-ml  - Install with ML dependencies"
	@echo "  run         - Start the FastAPI server"
	@echo "  test        - Run tests (when available)"
	@echo "  format      - Format code with black"
	@echo "  lint        - Run linting with ruff"
	@echo "  clean       - Clean up temporary files"

# Install dependencies
install:
	uv pip install -e .

# Install with development dependencies
install-dev:
	uv pip install -e ".[dev]"

# Install with ML dependencies
install-ml:
	uv pip install -e ".[ml]"

# Run the server
run:
	python start_server.py

# Run tests
test:
	pytest tests/

# Format code
format:
	black .
	ruff check --fix .

# Lint code
lint:
	ruff check .
	black --check .

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/ 