.PHONY: help dev dev-backend dev-frontend test build docker-up docker-down clean format lint security seed

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

dev: ## Run backend + frontend dev servers concurrently
	@echo "🚀 Starting ASR-Pro development stack..."
	@echo "   Backend API  → http://localhost:8000/api/docs"
	@echo "   Frontend UI  → http://localhost:5173"
	@echo "   Streamlit UI → http://localhost:8501"
	@(trap 'kill 0' SIGINT; \
		uvicorn asr_pro.api.main:app --reload --host 0.0.0.0 --port 8000 --reload-dir asr_pro & \
		npm run dev & \
		python3 -c "import sys; sys.modules['uvloop'] = None; import runpy; sys.argv=['streamlit', 'run', 'tools/legacy_streamlit/ASR/ASR.py', '--server.address=0.0.0.0', '--server.port=8501']; runpy.run_module('streamlit', run_name='__main__')" & \
		wait)

dev-backend: ## Run only the FastAPI backend
	uvicorn asr_pro.api.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend: ## Run only the Vite frontend
	npm run dev

seed: ## Initialize database and seed default data
	python -m asr_pro.db.seed

test: ## Run pytest suite with coverage report
	pytest tests/ --cov=asr_pro --cov-report=term-missing -v

test-ci: ## Run tests with XML coverage (for CI)
	pytest tests/ --cov=asr_pro --cov-report=xml --cov-fail-under=85

format: ## Format Python code with ruff
	ruff format asr_pro/ tests/
	isort asr_pro/ tests/

lint: ## Lint with ruff, mypy, and bandit
	ruff check asr_pro/ tests/
	mypy asr_pro/ --ignore-missing-imports
	bandit -r asr_pro/ -ll

security: ## Full security audit (bandit + pip-audit)
	bandit -r asr_pro/ tools/legacy_streamlit/ASR/ -ll
	pip-audit -r requirements.txt

build: ## Build all Docker images
	docker-compose build

docker-up: ## Start the full stack via Docker Compose
	docker-compose up -d
	@echo "✅ Stack running:"
	@echo "   API       → http://localhost:8000/api/docs"
	@echo "   Dashboard → http://localhost:5173"

docker-down: ## Stop all Docker Compose services
	docker-compose down

clean: ## Remove Python and build cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache/ .coverage .coverage.* coverage.xml htmlcov/ dist/ .vite/ data/benchmark_results.json
	@echo "🧹 Clean complete."
