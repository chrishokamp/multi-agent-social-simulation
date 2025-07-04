# Multi-Agent Social Simulation Framework Makefile

# Variables
PYTHON := python3
PIP := pip3
UV := uv
NPM := npm
VENV := venv
UV_VENV := .venv
BACKEND_DIR := src/backend
FRONTEND_DIR := src/frontend
DB_CONNECTION_STRING ?= mongodb://localhost:27017
SIMULATION_CONFIG ?= src/configs/bike_negotiation_config.json
CONFIG ?= $(SIMULATION_CONFIG)

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: help
help: ## Display this help message
	@echo "Multi-Agent Social Simulation Framework"
	@echo "======================================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start:"
	@echo "  make install    - Install all dependencies"
	@echo "  make dev        - Start development servers"

# Installation targets
.PHONY: install
install: install-backend install-frontend ## Install all dependencies

.PHONY: install-backend
install-backend: ## Install backend dependencies
	@echo "$(YELLOW)Installing backend dependencies...$(NC)"
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt
	@echo "$(GREEN)Backend dependencies installed!$(NC)"

.PHONY: install-backend-editable
install-backend-editable: ## Install backend package in editable mode
	@echo "$(YELLOW)Installing backend package in editable mode...$(NC)"
	$(PIP) install -e $(BACKEND_DIR)
	@echo "$(GREEN)Backend package installed!$(NC)"

.PHONY: install-frontend
install-frontend: ## Install frontend dependencies
	@echo "$(YELLOW)Installing frontend dependencies...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) install
	@echo "$(GREEN)Frontend dependencies installed!$(NC)"

# Development targets
.PHONY: dev
dev: ## Start all development servers (requires two terminals)
	@echo "$(YELLOW)Starting development servers...$(NC)"
	@echo "Run 'make dev-backend' in one terminal"
	@echo "Run 'make dev-frontend' in another terminal"

.PHONY: dev-backend
dev-backend: check-env ## Start backend development server
	@echo "$(YELLOW)Starting backend server...$(NC)"
	cd $(BACKEND_DIR) && $(PYTHON) main.py

.PHONY: dev-frontend
dev-frontend: ## Start frontend development server
	@echo "$(YELLOW)Starting frontend development server...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) run dev

.PHONY: frontend-setup
frontend-setup: install-frontend dev-frontend ## Install dependencies and start frontend dev server

# Build targets
.PHONY: build
build: build-frontend ## Build production assets

.PHONY: build-frontend
build-frontend: ## Build frontend for production
	@echo "$(YELLOW)Building frontend...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) run build
	@echo "$(GREEN)Frontend built successfully!$(NC)"

# Database targets
.PHONY: db-start
db-start: ## Start MongoDB (macOS with Homebrew)
	@echo "$(YELLOW)Starting MongoDB...$(NC)"
	brew services start mongodb/brew/mongodb-community@6.0
	@echo "$(GREEN)MongoDB started!$(NC)"

.PHONY: db-stop
db-stop: ## Stop MongoDB (macOS with Homebrew)
	@echo "$(YELLOW)Stopping MongoDB...$(NC)"
	brew services stop mongodb/brew/mongodb-community@6.0
	@echo "$(GREEN)MongoDB stopped!$(NC)"

.PHONY: db-status
db-status: ## Check MongoDB status
	@echo "$(YELLOW)MongoDB status:$(NC)"
	brew services list | grep mongodb

# Testing and linting
.PHONY: test
test: test-unit ## Run unit tests (default)

.PHONY: test-backend
test-backend: test-unit ## Run backend unit tests

.PHONY: test-all
test-all: ## Run all tests (unit + integration)
	@echo "$(YELLOW)Running all backend tests...$(NC)"
	@if [ ! -f $(UV_VENV)/bin/python ]; then \
		echo "$(RED)Error: UV environment not found. Run 'make uv-setup' first.$(NC)"; \
		exit 1; \
	fi
	cd $(BACKEND_DIR) && ../../$(UV_VENV)/bin/python -m pytest tests/ -v
	@echo "$(GREEN)All backend tests completed!$(NC)"

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(NC)"
	@if [ ! -f $(UV_VENV)/bin/python ]; then \
		echo "$(RED)Error: UV environment not found. Run 'make uv-setup' first.$(NC)"; \
		exit 1; \
	fi
	cd $(BACKEND_DIR) && ../../$(UV_VENV)/bin/python -m pytest tests/unit/ -v
	@echo "$(GREEN)Unit tests completed!$(NC)"

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "$(YELLOW)Running integration tests...$(NC)"
	@if [ ! -f $(UV_VENV)/bin/python ]; then \
		echo "$(RED)Error: UV environment not found. Run 'make uv-setup' first.$(NC)"; \
		exit 1; \
	fi
	cd $(BACKEND_DIR) && ../../$(UV_VENV)/bin/python -m pytest tests/integration/ -v
	@echo "$(GREEN)Integration tests completed!$(NC)"

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	cd $(BACKEND_DIR) && ../../$(UV_VENV)/bin/python -m pytest tests/ --cov=. --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

.PHONY: lint
lint: lint-backend lint-frontend ## Run all linters

.PHONY: lint-backend
lint-backend: ## Run backend linting
	@echo "$(YELLOW)Linting backend...$(NC)"
	cd $(BACKEND_DIR) && $(PYTHON) -m flake8 . --exclude=venv,__pycache__
	@echo "$(GREEN)Backend linting complete!$(NC)"

.PHONY: lint-frontend
lint-frontend: ## Run frontend linting
	@echo "$(YELLOW)Linting frontend...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) run lint
	@echo "$(GREEN)Frontend linting complete!$(NC)"

.PHONY: format
format: format-backend format-frontend ## Format all code

.PHONY: format-backend
format-backend: ## Format backend code
	@echo "$(YELLOW)Formatting backend...$(NC)"
	cd $(BACKEND_DIR) && $(PYTHON) -m black .
	@echo "$(GREEN)Backend formatting complete!$(NC)"

.PHONY: format-frontend
format-frontend: ## Format frontend code
	@echo "$(YELLOW)Formatting frontend...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) run format
	@echo "$(GREEN)Frontend formatting complete!$(NC)"

# Environment setup
.PHONY: setup-env
setup-env: ## Create .env file from example
	@echo "$(YELLOW)Setting up environment...$(NC)"
	@if [ ! -f $(BACKEND_DIR)/.env ]; then \
		cp $(BACKEND_DIR)/.env.example $(BACKEND_DIR)/.env; \
		echo "$(GREEN)Created $(BACKEND_DIR)/.env - Please update with your API keys!$(NC)"; \
	else \
		echo "$(YELLOW)$(BACKEND_DIR)/.env already exists$(NC)"; \
	fi

.PHONY: check-env
check-env: ## Check if environment variables are set
	@if [ ! -f $(BACKEND_DIR)/.env ]; then \
		echo "$(RED)Error: $(BACKEND_DIR)/.env not found!$(NC)"; \
		echo "Run 'make setup-env' to create it"; \
		exit 1; \
	fi
	@echo "$(GREEN)Environment file found$(NC)"

# Cleaning
.PHONY: clean
clean: clean-backend clean-frontend ## Clean all build artifacts

.PHONY: clean-backend
clean-backend: ## Clean backend artifacts
	@echo "$(YELLOW)Cleaning backend...$(NC)"
	find $(BACKEND_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find $(BACKEND_DIR) -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf $(BACKEND_DIR)/.pytest_cache
	@echo "$(GREEN)Backend cleaned!$(NC)"

.PHONY: clean-frontend
clean-frontend: ## Clean frontend artifacts
	@echo "$(YELLOW)Cleaning frontend...$(NC)"
	rm -rf $(FRONTEND_DIR)/dist
	rm -rf $(FRONTEND_DIR)/node_modules
	@echo "$(GREEN)Frontend cleaned!$(NC)"

# Virtual environment
.PHONY: venv
venv: ## Create Python virtual environment
	@echo "$(YELLOW)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV) --python=3.10
	@echo "$(GREEN)Virtual environment created!$(NC)"
	@echo "Activate with: source $(VENV)/bin/activate"

.PHONY: venv-install
venv-install: venv ## Create venv and install backend dependencies
	@echo "$(YELLOW)Installing in virtual environment...$(NC)"
	. $(VENV)/bin/activate && cd $(BACKEND_DIR) && pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed in venv!$(NC)"

# UV environment management
.PHONY: uv-venv
uv-venv: ## Create Python virtual environment with uv
	@echo "$(YELLOW)Creating virtual environment with uv...$(NC)"
	$(UV) venv $(UV_VENV)
	@echo "$(GREEN)UV virtual environment created!$(NC)"
	@echo "Activate with: source $(UV_VENV)/bin/activate"

.PHONY: uv-install
uv-install: ## Install backend dependencies with uv
	@echo "$(YELLOW)Installing backend dependencies with uv...$(NC)"
	$(UV) pip install -r $(BACKEND_DIR)/requirements.txt
	@echo "$(GREEN)Dependencies installed with uv!$(NC)"

.PHONY: uv-setup
uv-setup: uv-venv uv-install ## Create uv environment and install dependencies
	@echo "$(GREEN)UV environment setup complete!$(NC)"

# Example simulation
.PHONY: run-example
run-example: ## Run the car negotiation example simulation
	@echo "$(YELLOW)Running car negotiation example simulation...$(NC)"
	@if [ ! -f $(UV_VENV)/bin/python ]; then \
		echo "$(RED)Error: UV environment not found. Run 'make uv-setup' first.$(NC)"; \
		exit 1; \
	fi
	cd $(BACKEND_DIR) && ../../$(UV_VENV)/bin/python -c "\
import sys, os, json, asyncio; \
sys.path.insert(0, '.'); \
from engine.simulation import SelectorGCSimulation; \
config_path = '../../src/configs/car-sale-negotiation.json'; \
with open(config_path) as f: config = json.load(f); \
sim = SelectorGCSimulation(config['config'], environment=None); \
result = asyncio.run(sim.run()); \
print('Simulation completed!'); \
print('Result:', json.dumps(result, indent=2) if result else 'No result')"
	@echo "$(GREEN)Example simulation completed!$(NC)"

.PHONY: run-simulation
run-simulation: ## Run simulation with enhanced self-optimization and rich visualizations
	@echo "$(YELLOW)🚀 Running simulation, configuration: $(CONFIG)...$(NC)"
	@if [ ! -f $(UV_VENV)/bin/python ]; then \
		echo "$(RED)Error: UV environment not found. Run 'make uv-setup' first.$(NC)"; \
		exit 1; \
	fi
	$(UV_VENV)/bin/python scripts/self_optimize_negotiation_with_logging.py --config $(CONFIG)
	@echo "$(GREEN)✨ Simulation completed $(NC)"

.PHONY: run-simulation-simple
run-simulation-simple: ## Run basic simulation without enhanced logging
	@echo "$(YELLOW)Running basic simulation...$(NC)"
	@if [ ! -f $(UV_VENV)/bin/python ]; then \
	echo "$(RED)Error: UV environment not found. Run 'make uv-setup' first.$(NC)"; \
	exit 1; \
	fi
	$(UV_VENV)/bin/python scripts/run_simulation.py --config $(SIMULATION_CONFIG)
	@echo "$(GREEN)Basic simulation completed!$(NC)"

.PHONY: run-simulation-no-optimization
run-simulation-no-optimization: ## Run simulation with rich logging but without optimization
	@echo "$(YELLOW)Running simulation with logging only (no optimization)...$(NC)"
	@if [ ! -f $(UV_VENV)/bin/python ]; then \
	echo "$(RED)Error: UV environment not found. Run 'make uv-setup' first.$(NC)"; \
	exit 1; \
	fi
	$(UV_VENV)/bin/python scripts/self_optimize_negotiation_with_logging.py --config $(SIMULATION_CONFIG) --no-pdf
	@echo "$(GREEN)Simulation completed!$(NC)"


.PHONY: demo-logging
demo-logging: ## Demo the logging framework without API calls
	@echo "$(YELLOW)🎬 Running logging framework demo...$(NC)"
	@if [ ! -f $(UV_VENV)/bin/python ]; then \
		echo "$(RED)Error: UV environment not found. Run 'make uv-setup' first.$(NC)"; \
		exit 1; \
	fi
	$(UV_VENV)/bin/python scripts/demo_logging_framework.py
	@echo "$(GREEN)Logging framework demo completed!$(NC)"

.PHONY: test-logging
test-logging: ## Test the logging framework functionality  
	@echo "$(YELLOW)Testing logging framework...$(NC)"
	@if [ ! -f $(UV_VENV)/bin/python ]; then \
		echo "$(RED)Error: UV environment not found. Run 'make uv-setup' first.$(NC)"; \
		exit 1; \
	fi
	$(UV_VENV)/bin/python scripts/test_logging_framework.py
	@echo "$(GREEN)Logging framework test completed!$(NC)"

.PHONY: test-logging-unit
test-logging-unit: ## Run logging framework unit tests
	@echo "$(YELLOW)Running logging framework unit tests...$(NC)"
	@if [ ! -f $(UV_VENV)/bin/python ]; then \
		echo "$(RED)Error: UV environment not found. Run 'make uv-setup' first.$(NC)"; \
		exit 1; \
	fi
	$(UV_VENV)/bin/python -m pytest src/backend/tests/unit/test_logging_framework.py -v
	$(UV_VENV)/bin/python -m pytest src/backend/tests/integration/test_logged_simulation.py -v
	@echo "$(GREEN)Logging framework unit tests completed!$(NC)"

# UI Testing with Playwright
.PHONY: install-playwright
install-playwright: ## Install Playwright and browsers for UI testing
	@echo "$(YELLOW)Installing Playwright...$(NC)"
	@if [ ! -f $(UV_VENV)/bin/python ]; then \
		echo "$(RED)Error: UV environment not found. Run 'make uv-setup' first.$(NC)"; \
		exit 1; \
	fi
	$(UV_VENV)/bin/pip install playwright
	$(UV_VENV)/bin/playwright install chromium
	@echo "$(GREEN)Playwright installed!$(NC)"

.PHONY: test-ui
test-ui: ## Run UI integration tests with Playwright
	@echo "$(YELLOW)Running UI tests...$(NC)"
	@if ! command -v $(UV_VENV)/bin/playwright &> /dev/null; then \
		echo "$(RED)Playwright not installed. Run 'make install-playwright' first.$(NC)"; \
		exit 1; \
	fi
	@if ! curl -s http://localhost:5173 > /dev/null; then \
		echo "$(RED)Frontend not running. Start it with 'make dev-frontend'$(NC)"; \
		exit 1; \
	fi
	@if ! curl -s http://localhost:5000 > /dev/null; then \
		echo "$(RED)Backend not running. Start it with 'make dev-backend'$(NC)"; \
		exit 1; \
	fi
	$(UV_VENV)/bin/python test_ui.py
	@echo "$(GREEN)UI tests completed!$(NC)"

.PHONY: test-ui-simple
test-ui-simple: ## Run simple UI-backend integration test (no browser required)
	@echo "$(YELLOW)Running simple UI integration test...$(NC)"
	@if ! curl -s http://localhost:5173 > /dev/null; then \
		echo "$(RED)Frontend not running. Start it with 'make dev-frontend'$(NC)"; \
		exit 1; \
	fi
	@if ! curl -s http://localhost:5000 > /dev/null; then \
		echo "$(RED)Backend not running. Start it with 'make dev-backend'$(NC)"; \
		exit 1; \
	fi
	$(UV_VENV)/bin/python test_ui_simple.py
	@echo "$(GREEN)Simple UI test completed!$(NC)"

# Docker support (future)
.PHONY: docker-build
docker-build: ## Build Docker images
	@echo "$(YELLOW)Docker support coming soon...$(NC)"

.PHONY: docker-up
docker-up: ## Start services with Docker Compose
	@echo "$(YELLOW)Docker support coming soon...$(NC)"

# Shortcuts
.PHONY: i
i: install ## Shortcut for install

.PHONY: d
d: dev ## Shortcut for dev

.PHONY: b
b: build ## Shortcut for build

.PHONY: l
l: lint ## Shortcut for lint

.PHONY: f
f: format ## Shortcut for format

.PHONY: c
c: clean ## Shortcut for clean

# Default target
.DEFAULT_GOAL := help
