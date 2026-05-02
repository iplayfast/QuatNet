# QuatNet: 2-Bit Quaternary Neural Network with Self-Growing Architecture

## Abstract

QuatNet is a framework for training transformer language models with **2-bit quaternary weights** using straight-through estimation (STE). It employs a quaternary value set {1.0, 0.5, -0.5, -1.0} for attention projection matrices, achieving extreme compression while maintaining trainability via gradient bypass. The system introduces several novel techniques beyond standard quantization-aware training:

- **Dynamic architecture growth**: model starts at 2 dimensions and automatically widens/deepens at plateaus
- **Dual-objective loss**: byte-level CE + Q2_Q quantization regularization + KL distillation from heterogeneous teacher pools
- **Self-pacing via plateau detection**: training autoregulates LR decay and model growth
- **Native GGUF 2-bit type**: custom GGML quantization type 99 with CPU kernel and full llama.cpp integration

## Novel Techniques

### 1. Dynamic Architecture Growth with Weight Preservation

The model starts at **2d × 4 layers** (~7K params) and automatically grows when training plateaus:

| Growth Stage | Schedule |
|---|---|
| d < 16 | Square (2→4→16) |
| d < 64 | ×4 (16→64) |
| d < 16384 | Double (64→128→256→...→16384) |
| d ≥ 16384 | Add 2 layers per growth |

**Key**: learned weights are copied into the expanded structure. The model never starts from scratch — it widens existing weights and randomly initializes new dimensions. This is unlike standard practice where model size is fixed at init and you discard smaller models when scaling up.

### 2. Quaternary 2-Bit STE Training with Triple-Objective Loss

Three losses combined dynamically:

```
L = L_ce + λ_q2q * L_q2q + λ_distill * KL(student || teacher)
```

- **L_ce**: Standard next-byte prediction (256-class cross-entropy)
- **L_q2q**: MSE pulling attention weights toward {1.0, 0.5, -0.5, -1.0} (λ_q2q = 0.5)
- **KL divergence**: Student matches teacher output distribution

The distillation weight is **dynamic**: λ_distill scales from 0 (model still learning basics) to 1 (model refined enough to benefit from teacher guidance), based on `min(1.0, max(0.0, 1.0 - L_ce / 5.0))`.

### 3. Distributed Heterogeneous Teacher Pool

Multiple Ollama servers contribute to distillation simultaneously:
- Local models (llama3.2, gemma4)
- Remote servers (mac-mini with qwen3-coder, glm-4.7-flash, gpt-oss)
- Different architectures, sizes, and quantization levels

The student learns from a diverse teacher ensemble, not a single fixed teacher. Data generation (library_populator.py) runs as a separate process, decoupled from training.

### 4. Self-Pacing via Plateau Detection

No manual LR scheduling or early stopping:

1. If Q2_Q convergence improves → keep LR
2. If Q2_Q plateaus for 2000 steps → halve LR
3. If LR exhausts (≤1e-8) OR loss plateaus for 10K steps → **grow the model**
4. After growth → reset optimizer and continue

This lets the model auto-regulate its capacity: small when the task is easy, larger when it needs more representational power.

### 5. Native GGUF 2-Bit Type (Q2_Q = 99)

A custom GGML quantization type registered at type ID 99:
- **32 weights per block**, 4 weights per byte (2 bits each)
- **No scale/min** — pure packed 2-bit codes for {1.0, 0.5, -0.5, -1.0}
- CPU kernel in `ggml-cpu.c` with thread-parallel matmul
- Full llama.cpp architecture: `quaternary_nn` with `LLM_ARCH_QUATERNARY_NN` enum, load_tensors, build_graph, and model handler
- Python quantize/dequantize in `gguf-py/gguf/quants.py` with GGUF writer fix for `raw_shape` passthrough

### 6. Live-Plot Feedback Loop

`plot_training.sh` with `q`-key refresh shows real-time:
- Training loss
- Learning rate (log scale)
- Q2_Q convergence percentage
- Data size
- Model architecture (d_model + n_layers with green/blue overlay and growth annotations)

## Pipeline Components

| Script | Role |
|--------|------|
| `teacher_pipeline.py` | Core training with auto-growth, triple loss, GGUF export |
| `library_populator.py` | Independent data collector — queries Ollama servers continuously |
| `convert_to_quaternary.py` | Convert any GGUF model to quaternary_nn architecture |
| `buildLibrary.py` | One-shot library builder from programmingquestions.txt |
| `plot_training.sh` | Real-time training visualization (press 'q' to refresh) |
| `validate_servers.py` | Test all Ollama workers in servers.json |
| `restart_training.sh` | Clean reset: kill, clean artifacts, restart |

## Project Structure

```
├── teacher_pipeline.py       # Main training loop with auto-growth
├── library_populator.py      # Independent data generation (runs alongside training)
├── convert_to_quaternary.py  # GGUF → quaternary_nn converter
├── plot_training.sh          # Real-time matplotlib visualization
├── restart_training.sh       # Clean restart script
├── library/                  # Q&A training data
│   ├── questions/            # Prompts
│   ├── answers/              # Solutions
│   └── questions/bestquestions/  # Curated seed questions
├── llama.cpp/                # Fork with Q2_Q type 99, quaternary_nn arch
│   ├── src/models/quaternary_nn.cpp
│   ├── ggml/src/ggml.c       # Q2_Q type traits
│   ├── ggml/src/ggml-cpu/ggml-cpu.c  # MUL_MAT_Q2_Q kernel
│   └── gguf-py/gguf/quants.py        # Python Q2_Q class
├── models/                   # Converted models
├── images/                   # Training plots
└── servers.json              # Ollama worker pool config
```

## Running

```bash
# Terminal 1: Data generation
source .venv/bin/activate && python library_populator.py

# Terminal 2: Training
./restart_training.sh

# Terminal 3: Monitoring
python3 plot_training.sh
```

## Dependencies

- Python 3.10+, PyTorch (CUDA recommended)
- llama.cpp (via git subtree — included)
- Access to Ollama instances for data generation
