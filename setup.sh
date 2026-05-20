#!/bin/bash
# QuatNet setup — run this after cloning on a fresh machine
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }

echo "=== QuatNet Setup ==="
echo ""

# ── Check system dependencies ────────────────────────────────────
echo "--- Checking system dependencies ---"

MISSING=()

if command -v uv &>/dev/null; then
    ok "uv: $(uv --version)"
else
    fail "uv not found — install from https://docs.astral.sh/uv/"
    MISSING+=("uv")
fi

if command -v python3 &>/dev/null; then
    PYVER=$(python3 --version 2>&1)
    ok "Python: $PYVER"
else
    fail "python3 not found"
    MISSING+=("python3")
fi

if command -v cmake &>/dev/null; then
    ok "cmake: $(cmake --version | head -1)"
else
    fail "cmake not found (needed to build llama.cpp)"
    MISSING+=("cmake")
fi

if command -v g++ &>/dev/null; then
    ok "g++: $(g++ --version | head -1)"
else
    fail "g++ not found (needed to build llama.cpp)"
    MISSING+=("g++")
fi

if command -v nvcc &>/dev/null; then
    ok "CUDA: $(nvcc --version | grep release)"
else
    warn "nvcc not found — will build llama.cpp CPU-only (slower inference)"
fi

if python3 -c "import tkinter" 2>/dev/null; then
    ok "tkinter available"
else
    warn "tkinter not available — install python3-tk for plot_training.sh"
fi

if command -v convert &>/dev/null; then
    ok "ImageMagick convert available (PNG export)"
else
    warn "ImageMagick not found — plot PNG export will be skipped"
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    fail "Missing required packages: ${MISSING[*]}"
    echo "  On Ubuntu/Debian: sudo apt install ${MISSING[*]}"
    exit 1
fi

# ── Python virtualenv ────────────────────────────────────────────
echo ""
echo "--- Setting up Python virtualenv ---"

if [ ! -d ".venv" ]; then
    uv venv
    ok "Created .venv"
else
    ok ".venv already exists"
fi

source .venv/bin/activate
uv pip install -r requirements.txt
ok "Python dependencies installed"

# ── Build llama.cpp ──────────────────────────────────────────────
echo ""
echo "--- Building llama.cpp ---"

if [ -f "llama.cpp/build/bin/llama-cli" ]; then
    ok "llama.cpp already built (run ./build_llama.sh to rebuild)"
else
    ./build_llama.sh
    ok "llama.cpp built"
fi

# ── Create directories ───────────────────────────────────────────
echo ""
echo "--- Creating directories ---"
mkdir -p models images
ok "models/ and images/ ready"

# ── Ollama check ─────────────────────────────────────────────────
echo ""
echo "--- Checking Ollama ---"

if command -v ollama &>/dev/null; then
    ok "Ollama installed"
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        ok "Ollama service running"
        # Check for enabled models from servers.json
        for model in llama3.2 dolphin-llama3; do
            if ollama list 2>/dev/null | grep -q "$model"; then
                ok "Model '$model' available"
            else
                warn "Model '$model' not pulled — run: ollama pull $model"
            fi
        done
    else
        warn "Ollama installed but not running — start with: ollama serve"
    fi
else
    warn "Ollama not installed — needed for training data generation"
    echo "  Install: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Then pull models: ollama pull llama3.2 && ollama pull dolphin-llama3"
fi

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo ""
echo "  # Generate training data (needs Ollama running):"
echo "  python library_populator.py"
echo ""
echo "  # Train the model:"
echo "  ./restart_training.sh"
echo ""
echo "  # Plot training metrics:"
echo "  python plot_training.sh"
echo ""
echo "  # Verify llama.cpp with a known model (optional, needs models/qwen2.5-0.5b-instruct-q4_k_m.gguf):"
echo "  ./runknowngoodmodel.sh -p 'Hello' -n 20"
