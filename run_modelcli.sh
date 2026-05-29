#!/bin/bash
# Run the quaternary model in interactive llama-cli mode.
# -c 2048 overrides the model's 256-token trained context.
export LD_LIBRARY_PATH="./llama.cpp/build/bin:${LD_LIBRARY_PATH}"
./llama.cpp/build/bin/llama-cli -m quaternary_trained.gguf -c 2048 -ngl 0 --temp 0.5 --repeat-penalty 1.1
