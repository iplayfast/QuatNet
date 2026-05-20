#!/bin/bash
# Run the quaternary model built by teacher_pipeline.py
# Use llama-simple to avoid chat parser issues with byte-level tokenizer
export LD_LIBRARY_PATH="./llama.cpp/build/bin:${LD_LIBRARY_PATH}"
./llama.cpp/build/bin/llama-cli -m quaternary_trained.gguf --no-warmup -ngl 0 --temp 0.9 --repeat-penalty 1.3

