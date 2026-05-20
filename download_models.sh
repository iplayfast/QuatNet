#!/bin/bash
# Download GGUF models from Hugging Face
# Usage:
#   ./download_models.sh                          # download all default models
#   ./download_models.sh --repo User/Repo --file model.gguf  # download a specific model
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"
HF="$SCRIPT_DIR/llama.cpp/scripts/hf.sh"

mkdir -p "$MODEL_DIR"

# Default models needed by the project
MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

download_model() {
    local repo="$1"
    local file="$2"
    local dest="$MODEL_DIR/$file"

    if [ -f "$dest" ]; then
        echo "[OK] $file already exists"
        return 0
    fi

    echo "[DL] Downloading $file from $repo..."
    if "$HF" --repo "$repo" --file "$file" --outdir "$MODEL_DIR"; then
        echo "[OK] $file downloaded"
    else
        echo "[FAIL] Failed to download $file"
        return 1
    fi
}

if [ "$1" = "--repo" ] && [ -n "$2" ] && [ "$3" = "--file" ] && [ -n "$4" ]; then
    # Download a specific model
    download_model "$2" "$4"
else
    # Download all default models
    echo "=== Downloading default models to $MODEL_DIR ==="
    for entry in "${MODELS[@]}"; do
        read -r repo file <<< "$entry"
        download_model "$repo" "$file"
    done
    echo ""
    echo "To add more models:"
    echo "  ./download_models.sh --repo Owner/RepoName-GGUF --file model-name.gguf"
fi
