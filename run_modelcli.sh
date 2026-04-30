#!/bin/bash
# Run the quaternary model built by teacher_pipeline.py
# Use llama-simple to avoid chat parser issues with byte-level tokenizer
./llama.cpp/build/bin/llama-cli -m quaternary_trained.gguf --no-warmup -ngl 0

