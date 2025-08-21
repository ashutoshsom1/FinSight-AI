# FinSight AI - Development Makefile

.PHONY: help install test clean run ingest setup lint format

# Default target
help:
	@echo "FinSight AI - Available commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup     - Run complete setup (install deps, create dirs, etc.)"
	@echo "  install   - Install Python dependencies"
	@echo "  clean     - Clean up generated files and caches"
	@echo ""
	@echo "Development:"
	@echo "  test      - Run all tests"
	@echo "  lint      - Run code linting"
	@echo "  format    - Format code with black"
	@echo "  run       - Start the API server"
	@echo ""
	@echo "Data Management:"
	@echo "  ingest    - Ingest sample documents"
	@echo "  test-sys  - Test the complete system"
	@echo ""
	@echo "Examples:"
	@echo "  examples  - Run usage examples"

# Setup and Installation
setup:
	@echo "Setting up FinSight AI development environment..."
	python scripts/setup.py

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf src/**/__pycache__ tests/__pycache__
	rm -rf data/chroma_db/* data/feedback/* logs/*
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Development
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

lint:
	@echo "Running linting..."
	flake8 src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	black src/ tests/ scripts/ examples/

run:
	@echo "Starting FinSight AI API server..."
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Data Management
ingest:
	@echo "Ingesting sample documents..."
	python scripts/ingest_reports.py --directory ./data/sample_reports

test-sys:
	@echo "Testing complete system..."
	python scripts/test_system.py

# Examples
examples:
	@echo "Running usage examples..."
	python examples/sample_usage.py

# Development helpers
dev-setup: clean install
	@echo "Development environment ready!"

check-env:
	@echo "Checking environment configuration..."
	@python -c "from config.settings import settings; print('✓ Configuration loaded successfully')"
	@python -c "import openai; print('✓ OpenAI library available')"
	@python -c "import chromadb; print('✓ ChromaDB available')"

# Production helpers
prod-test:
	@echo "Running production readiness tests..."
	python -m pytest tests/ -v --tb=short
	python scripts/test_system.py

build-docker:
	@echo "Building Docker image..."
	docker build -t finsight-ai:latest .

run-docker:
	@echo "Running Docker container..."
	docker run -p 8000:8000 --env-file .env finsight-ai:latest

# Documentation
docs:
	@echo "Available documentation:"
	@echo "  README.md - Project overview"
	@echo "  docs/GETTING_STARTED.md - Setup guide"
	@echo "  docs/API_REFERENCE.md - API documentation"
	@echo "  docs/DEPLOYMENT.md - Deployment guide"
	@echo "  data/sample_reports/README.md - Sample data guide"

# Quick start for new developers
quickstart: setup ingest test-sys
	@echo ""
	@echo "🎉 FinSight AI is ready!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Add your OpenAI API key to .env file"
	@echo "2. Add sample reports to data/sample_reports/"
	@echo "3. Run: make ingest"
	@echo "4. Run: make run"
	@echo "5. Visit: http://localhost:8000/docs"
