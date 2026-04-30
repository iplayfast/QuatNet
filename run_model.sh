#!/bin/bash
# Run the quaternary model built by teacher_pipeline.py
# Usage: run_model.sh [prompt text]
PROMPT="${*:-<|Q|>}"
./llama.cpp/build/bin/llama-simple -m quaternary_trained.gguf --no-warmup -ngl 0 -n 50 -p "$PROMPT" 2>/dev/null
