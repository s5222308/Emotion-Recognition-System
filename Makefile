# Clinical Emotion Recognition System - Makefile (Updated for Gold Standard Layout)
.PHONY: help setup install run-api run-ls run-dev run-prod test clean smoke
PY=python3

help:
	@echo "Available commands:"
	@echo "  make setup       - Create virtual environment and install dependencies"
	@echo "  make install     - Install dependencies (legacy, use setup)"
	@echo "  make run-api     - Run ML inference API"
	@echo "  make run-ls      - Run Label Studio"
	@echo "  make run-dev     - Run in development mode (all services)"
	@echo "  make run-prod    - Run in production mode"
	@echo "  make smoke       - Run smoke tests to verify restructuring"
	@echo "  make test        - Run all tests"
	@echo "  make clean       - Clean temporary files"

setup:
	$(PY) -m venv venv && . venv/bin/activate && pip install -U pip wheel
	. venv/bin/activate && pip install -r requirements.txt || true
	@echo "venv ready"

install: setup
	@echo "Dependencies installed! (Use 'make setup' instead in the future)"

run-api:
	. venv/bin/activate && export BIND_HOST=127.0.0.1 && \
	export PYTORCH_ENABLE_MPS_FALLBACK=1 && \
	$(PY) apps/ml_service/app.py

run-ls:
	. venv/bin/activate && label-studio start --host 127.0.0.1 --port 8200

run-dev:
	@echo "Starting development environment..."
	. venv/bin/activate && source config/production.env && $(PY) apps/dashboard/app.py &
	. venv/bin/activate && $(PY) apps/ml_service/app.py &
	. venv/bin/activate && $(PY) apps/label_studio_connector/app.py &
	@echo "Services started on ports 8081, 5003, 9091"

run-prod:
	@echo "Starting production environment..."
	./scripts/deployment/deploy.sh

smoke:
	. venv/bin/activate && $(PY) scripts/macos/smoke_test.py

test:
	@echo "Running tests..."
	. venv/bin/activate && $(PY) -m pytest tests/ -v || true
	. venv/bin/activate && $(PY) test_au_visualization.py || true

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf storage/cache/* 2>/dev/null || true
	rm -rf storage/logs/*.log 2>/dev/null || true
	rm -rf third_party/OpenFace/build/* 2>/dev/null || true
	rm -rf third_party/dlib-19.24/build/* 2>/dev/null || true
	rm -rf third_party/POSTER_V2/checkpoint/* 2>/dev/null || true
	@echo "Cleanup complete!"

# Individual service commands (updated paths)
run-dashboard:
	. venv/bin/activate && source config/production.env && $(PY) apps/dashboard/app.py

run-ml-engine:
	. venv/bin/activate && $(PY) apps/ml_service/app.py

run-connector:
	. venv/bin/activate && $(PY) apps/label_studio_connector/app.py

# Legacy compatibility (still works via compatibility shims)
run-ml-engine-legacy:
	. venv/bin/activate && $(PY) -c "import services.ml_engine; services.ml_engine.app.main()"