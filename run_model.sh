#!/bin/bash
# Run the quaternary model
PROMPT="${1:-<|Q|>}"
./llama.cpp/build/bin/llama-simple -m quaternary_trained.gguf --no-warmup -ngl 0 -n 50 -p "$PROMPT" 2>/dev/null
