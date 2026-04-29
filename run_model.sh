#!/bin/bash
# Run the quaternary model built by teacher_pipeline.py
./llama.cpp/build/bin/llama-simple -m quaternary_trained.gguf --no-warmup -ngl 0 -n 10 "$@"
