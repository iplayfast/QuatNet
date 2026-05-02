#!/bin/bash
# Kill any running training, clean all artifacts, restart fresh
set -e

echo "[RESTART] Killing any running teacher_pipeline..."
pkill -f teacher_pipeline 2>/dev/null || true
sleep 1

echo "[RESTART] Cleaning artifacts..."
rm -f quaternary_trained.gguf training_log.csv images/training_log.png

echo "[RESTART] Restarting teacher_pipeline.py..."
source .venv/bin/activate
exec python teacher_pipeline.py
