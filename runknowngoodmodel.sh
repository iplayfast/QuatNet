#!/bin/bash
# Verify llama.cpp can still run a known-good standard model
./llama.cpp/build/bin/llama-cli --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --no-warmup "$@"
