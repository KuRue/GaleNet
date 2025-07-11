# GaleNet - Makefile

.PHONY: help setup install clean test lint format docker-build docker-run notebook docs

# Default target
help:
	@echo "GaleNet - AI-Powered Hurricane Forecasting"
	@echo "=========================================="
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make setup          Create conda environment and install dependencies"
	@echo "  make install        Install package in development mode"
	@echo "  make download-data  Download hurricane and model data"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run unit tests"
	@echo "  make lint          Run code linting"
	@echo "  make format        Format code with black"
	@echo "  make clean         Clean build artifacts and caches"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make docker-shell  Start interactive Docker shell"
	@echo ""
	@echo "Services:"
	@echo "  make notebook      Start Jupyter notebook server"
	@echo "  make api          Start FastAPI server"
	@echo "  make mlflow       Start MLflow tracking server"
	@echo "  make tensorboard  Start TensorBoard"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         Build documentation"
	@echo "  make docs-serve   Serve documentation locally"

# Variables
PYTHON := python
PIP := pip
CONDA := conda
DOCKER := docker
DOCKER_COMPOSE := docker-compose

PROJECT_NAME := galenet
DOCKER_IMAGE := $(PROJECT_NAME):dev
DATA_DIR := ~/data/galenet

# Setup and Installation
setup:
	@echo "Creating conda environment..."
	$(CONDA) env create -f environment.yml
	@echo ""
	@echo "Environment created! Activate with:"
	@echo "  conda activate galenet"

install:
	@echo "Installing package in development mode..."
	$(PIP) install -e .
	@echo "Installing pre-commit hooks..."
	pre-commit install

download-data:
	@echo "Downloading hurricane data..."
	$(PYTHON) scripts/setup_data.py --all

# Development
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src/galenet --cov-report=html --cov-report=term

test-quick:
	@echo "Running quick tests..."
	pytest tests/ -v -k "not slow"

test-data:
	@echo "Testing data loading..."
	$(PYTHON) scripts/test_data_loading.py

lint:
	@echo "Running linters..."
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203
	mypy src/galenet --ignore-missing-imports
	@echo "Checking imports..."
	isort --check-only src/ tests/

format:
	@echo "Formatting code..."
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Docker
docker-build:
	@echo "Building Docker image..."
	$(DOCKER) build -t $(DOCKER_IMAGE) -f docker/Dockerfile .

docker-run:
	@echo "Running Docker container..."
	$(DOCKER) run --gpus all -it --rm \
		-v $(PWD):/app \
		-v $(DATA_DIR):/data/galenet \
		-p 8000:8000 -p 8888:8888 -p 6006:6006 \
		$(DOCKER_IMAGE)

docker-shell:
	@echo "Starting Docker shell..."
	$(DOCKER) run --gpus all -it --rm \
		-v $(PWD):/app \
		-v $(DATA_DIR):/data/galenet \
		$(DOCKER_IMAGE) bash

# Services
notebook:
	@echo "Starting Jupyter Lab..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

api:
	@echo "Starting FastAPI server..."
	@echo "API not yet implemented"
	# uvicorn src.galenet.inference.api:app --reload --host 0.0.0.0 --port 8000

mlflow:
	@echo "Starting MLflow server..."
	mlflow server \
		--backend-store-uri file://$(DATA_DIR)/mlruns \
		--default-artifact-root file://$(DATA_DIR)/mlruns \
		--host 0.0.0.0 \
		--port 5000

tensorboard:
	@echo "Starting TensorBoard..."
	tensorboard --logdir=$(DATA_DIR)/logs --host=0.0.0.0 --port=6006

# Documentation
docs:
	@echo "Building documentation..."
	@echo "Documentation not yet implemented"
	# cd docs && make html

docs-serve:
	@echo "Serving documentation..."
	@echo "Documentation not yet implemented"
	# cd docs && python -m http.server --directory _build/html 8080

# Data validation
validate-data:
	@echo "Validating data setup..."
	$(PYTHON) -c "from galenet.data.loaders import HURDAT2Loader; \
	              loader = HURDAT2Loader(); \
	              print('Data validation:', 'PASSED' if loader.data_path.exists() else 'FAILED')"

# Training shortcuts
train-quick:
	@echo "Running quick training test..."
	@echo "Training not yet implemented"
	# $(PYTHON) scripts/train_model.py --config configs/quick_test.yaml

train-full:
	@echo "Running full training..."
	@echo "Training not yet implemented"
	# $(PYTHON) scripts/train_model.py --config configs/default_config.yaml

# Evaluation
evaluate:
	@echo "Running model evaluation..."
	@echo "Evaluation not yet implemented"
	# $(PYTHON) scripts/evaluate.py

# GPU check
check-gpu:
	@echo "Checking GPU availability..."
	@$(PYTHON) -c "import torch; \
	              print(f'CUDA available: {torch.cuda.is_available()}'); \
	              print(f'GPU count: {torch.cuda.device_count()}'); \
	              [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Environment export
export-env:
	@echo "Exporting conda environment..."
	$(CONDA) env export --no-builds > environment-lock.yml
	$(PIP) freeze > requirements-lock.txt

# Pre-commit
pre-commit:
	@echo "Running pre-commit checks..."
	pre-commit run --all-files

# CI/CD simulation
ci:
	@echo "Running CI pipeline..."
	make lint
	make test
	make docker-build
	@echo "CI pipeline complete!"

# Release
release-patch:
	@echo "Creating patch release..."
	bumpversion patch

release-minor:
	@echo "Creating minor release..."
	bumpversion minor

release-major:
	@echo "Creating major release..."
	bumpversion major

# Quick start
quickstart: setup install download-data test-data
	@echo ""
	@echo "GaleNet setup complete! 🌀"
	@echo ""
	@echo "Next steps:"
	@echo "1. Activate environment: conda activate galenet"
	@echo "2. Start Jupyter: make notebook"
	@echo "3. Run tests: make test"
	@echo ""
	@echo "Happy forecasting!"
