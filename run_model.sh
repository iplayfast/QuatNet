#!/bin/bash
# Run the quaternary model
export LD_LIBRARY_PATH="./llama.cpp/build/bin:${LD_LIBRARY_PATH}"
PROMPT="${1:-<|Q|>}"
./llama.cpp/build/bin/llama-simple -m quaternary_trained.gguf --no-warmup -ngl 0 -n 50 "$PROMPT" 2>/dev/null
