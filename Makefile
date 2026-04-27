# Pharma Voice Assistant
# Author: Gourav Pandey

.PHONY: install run-api run-websocket test-voice test-avatar clean help

PYTHON := python
PIP := pip

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	$(PIP) install -r requirements.txt
	@echo "Installation complete."
	@echo "For voice input, ensure ffmpeg is installed: sudo apt install ffmpeg"

setup-env: ## Create .env from example
	@if [ ! -f .env ]; then cp .env.example .env && echo ".env created"; else echo ".env already exists"; fi

run-api: ## Start the REST API server
	$(PYTHON) -m src.api

run-websocket: ## Start the WebSocket server for real-time streaming
	$(PYTHON) -m src.websocket_server

run-both: ## Start both API and WebSocket servers
	$(PYTHON) -m src.api &
	$(PYTHON) -m src.websocket_server &
	@echo "API running on port 8000, WebSocket on port 8001"
	wait

test-voice: ## Test voice pipeline with a sample audio file
	@echo "Testing voice pipeline..."
	$(PYTHON) -c "\
from src.tts import TTSFactory; \
tts = TTSFactory.create('google'); \
result = tts.synthesize('This is a test of the pharmaceutical voice assistant.'); \
print(f'TTS OK: {len(result.audio_bytes)} bytes, {result.processing_time_s:.2f}s'); \
result.save('/tmp/test_tts.mp3'); \
print('Saved to /tmp/test_tts.mp3')"

test-avatar: ## Test HeyGen avatar integration
	@echo "Testing avatar integration..."
	$(PYTHON) -c "\
from src.avatar import HeyGenAvatar; \
client = HeyGenAvatar(); \
print(f'Avatar available: {client.is_available()}')"

test-llm: ## Test LLM engine connectivity
	@echo "Testing LLM providers..."
	$(PYTHON) -c "\
from src.llm_engine import LLMEngine; \
engine = LLMEngine(); \
available = engine.list_available(); \
print(f'Available providers: {available}'); \
if available: \
    resp = engine.generate('What is GMP?'); \
    print(f'Response from {resp.provider}/{resp.model}: {resp.text[:200]}...')"

lint: ## Run linting
	$(PYTHON) -m flake8 src/ --max-line-length=120
	$(PYTHON) -m mypy src/ --ignore-missing-imports

clean: ## Remove generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf audio_cache/ video_cache/ logs/ 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true
