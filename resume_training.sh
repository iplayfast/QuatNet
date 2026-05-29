#!/bin/bash
# Resume training from existing quaternary_trained.gguf
# Kills library_populator.py and Ollama to free VRAM for BATCH=512
set -e

echo "[RESUME] Killing library_populator.py..."
pkill -f library_populator 2>/dev/null || true
echo "[RESUME] Stopping Ollama to free ~30 GiB VRAM..."
ollama stop llama3.2 2>/dev/null || true
ollama stop gemma4 2>/dev/null || true
ollama stop granite4 2>/dev/null || true
ollama stop qwen3-coder-next 2>/dev/null || true

echo "[RESUME] Resuming teacher_pipeline.py (BATCH=16)..."
source .venv/bin/activate
exec python teacher_pipeline.py
