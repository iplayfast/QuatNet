# Quaternary Neural Network Integration with llama.cpp

## Project Vision

Create a neural network architecture that uses **2-bit quaternary weights** storing exactly 4 values:

| Bits | Symbol | Value | Role |
|------|--------|-------|------|
| 00 | +1.0 | 1.0 | Strong positive |
| 01 | +0.5 | 0.5 | Small positive |
| 10 | -0.5 | -0.5 | Small negative |
| 11 | -1.0 | -1.0 | Strong negative |

The architecture integrates into llama.cpp as a first-class model that can participate in attention layers.

**Core insight**: 2-bit ternary (1, 0, -1) is standard. 2-bit quaternary (1, +0.5, -0.5, -1) adds a 4th state that gives "a bit more discretion" around zero - a small nudge either way without needing full floating point.

**First attempt** used {1.0, 0.01, -0.01, -1.0} aiming for integer-only shift operations. Switched to {1.0, 0.5, -0.5, -1.0} because 0.01 was too small to carry meaningful gradient signal through the straight-through estimator.

## Project Structure

```
fourBitState/
├── quaternary_nn/          # Standalone C++ training/inference engine
│   ├── src/QuaternaryNN.cpp   # Forward pass, training, weight quantization
│   ├── src/GGUFHelper.cpp    # GGUF save/load with Q2_Q type 99
│   ├── src/main.cpp         # CLI for train/infer modes
│   └── build/quaternary_nn  # Compiled binary
│
├── llama.cpp/               # Modified for Q2_Q quantization support
│   ├── src/llama-arch.h     # + LLM_ARCH_QUATERNARY_NN (added)
│   ├── src/llama-arch.cpp   # + name mapping (added)
│   ├── ggml/src/ggml-quants.c # Q2_Q quant/dequant (updated values)
│   └── gguf-py/gguf/quants.py # Python Q2_Q (updated values)
│
├── teacher_pipeline.py     # Training orchestrator
├── buildLibrary.py         # Q&A library builder (queries Ollama)
├── library/                 # Cached Q&A pairs
│   ├── questions/N.txt      # Question text
│   └── answers/N.txt       # Answer text
├── servers.json            # Ollama server configuration
└── programmingquestions.txt # Source questions
```

## Current State

### What's Working

1. **quaternary_nn standalone**: Full training and inference cycle works
   - Binary compiled, trains via stdin, saves GGUF
    - Current values: +1.0, +0.5, -0.5, -1.0 (second attempt; first attempt used ±0.01)
   - Gradient-based training with STE (straight-through estimator)
   - Dynamic growth: widening vectors + adding heads at plateaus

2. **teacher_pipeline.py**: Orchestrates training with semantic critic
   - Loads Q&A from library (random sampling)
   - Hashes question to 1D coordinate (-1.0 to 1.0)
   - Queries Ollama "judge" server for semantic depth score (0.0-1.0)
   - Feeds (x, truth) to quaternary_nn student process
   - Reads PRED: from stdout for live error tracking
   - Saves model every SAVE_INTERVAL=1000 steps (subprocess restart)
   - Graceful shutdown with SIGINT handler

3. **buildLibrary.py**: Populates Q&A library from Ollama servers
   - Parallel queries with timeout handling
   - Streaming support for larger responses
   - Server failover and retry logic

4. **Q2_Q quantization integration**: llama.cpp can handle the quantized weights
   - GGML_TYPE_Q2_Q = 99 added
   - C++ quantize/dequantize: ggml-quants.c
   - Python: gguf-py/gguf/quants.py
    - Both updated to use {1.0, 0.5, -0.5, -1.0} (second attempt)

5. **Full llama.cpp integration (Phase 1 complete)**: Model loads, builds graph, runs forward pass, produces output through llama.cpp
   - LLM_ARCH_QUATERNARY_NN enum, name mapping, tensor enums all wired
   - Full forward pass in models/quaternary_nn.cpp (GaLA attention + SSM)
   - Tensor loader with `create_tensor_actual` for shape-adaptive loading
   - Verified: minimal attention model loads and runs inference (outputs garbage with random weights, which is expected)

### What's NOT Done Yet

6. **Attention integration**: Quaternary Q2_Q weights don't participate in attention yet (type 99 tensors exist but the quaternary_nn.cpp uses float32 attention)
7. **Token generation with trained model**: No trained quaternary model exists for llama.cpp format
8. **GGUFHelper fix**: quaternary_nn standalone's GGUFHelper still writes non-standard GGUF (use gguf-py for production GGUF files)
9. **Training pipeline integration**: No way to train quaternary_nn models from within llama.cpp

## Decisions Made

### 1. Weight values: +1.0, +0.5, -0.5, -1.0
**First attempt**: {+1.0, +0.01, -0.01, -1.0}. Rejected {+0.8, +0.2, -0.2, -0.8} because it requires floating point even for "identity" case. The ±0.01 values allow ±1.0 to be computed with bit operations.

**Second attempt (current)**: {+1.0, +0.5, -0.5, -1.0}. Switched because 0.01 was too small for meaningful weight contribution — gradients through the STE could not reliably push weights across the 0.75 quantization boundary. The 0.5 values provide a genuine middle ground that carries usable gradient signal while still allowing efficient packed 2-bit storage.

### 2. Q2_Q quantization type (99)
Uses existing ggml quantization infrastructure rather than creating a new type. 2 bits per weight, 32 weights per block, no scale factor.

### 3. SAVE_INTERVAL approach: subprocess restart
quaternary_nn saves on stdin close. Python restarts subprocess every N steps to force checkpoint. Alternative (adding explicit save command to C++) was considered but deferred.

### 4. Semantic critic training
Instead of naive `target = len(answer)/100`, we query an Ollama "judge" to rate Q&A semantic depth. This is slower (network call per step) but produces meaningful training signals.

### 5. Library-based training
BuildLibrary pre-collects Q&A pairs. teacher_pipeline trains from cached library files. This decouples data collection from training speed.

## Dequantization Strategy (Decision)

**Option A: Full dequantize (current plan)**
```
Q2_Q → [float32] → matrix multiply → softmax
```
- Pro: Standard attention kernel works
- Con: Still needs floating point multiplies

**Option B: Native quaternary matmul**
```
Q2_Q @ Q2_Q → custom kernel → int8 → softmax
```
- Pro: Maximum efficiency
- Con: Need to implement custom ggml operation

**Decision**: Start with Option A for correctness, optimize to B later.

## Implementation Plan (Updated)

### Phase 1: Minimal Integration ✅ COMPLETE

- [x] 1. Define `LLM_ARCH_QUATERNARY_NN` enum
- [x] 2. Add name mapping to llama-arch.cpp
- [x] 3. Add tensor enums (LLM_TENSOR_QUATERNARY_HEAD, LLM_TENSOR_QUATERNARY_BIAS)
- [x] 4. Add tensor names/infos to llama-arch.cpp
- [x] 5. Create model loading in llama-model-loader.cpp
- [x] 6. Add case in llama-model.cpp build_graph()
- [x] 7. Create models/quaternary_nn.cpp with forward pass
- [x] 8. llama.cpp compiles successfully with quaternary_nn support
- [x] 9. GGUF format: use gguf-py for writing (standard format), not custom GGUFHelper
- [x] 10. Test: Model loads, builds graph, runs inference (verified with `minimal_test.gguf`)
- [x] Fixes applied: rope sections optional, s_copy null-buffer guard, rs input pointer init

### Phase 2: Attention Integration

- [ ] 1. Wire up Q, K, V quaternary weights (Q2_Q type 99) into GaLA attention
- [ ] 2. Fix quaternary_nn.cpp to use Q2_Q quantized heads for attention projection
- [ ] 3. Train a real quaternary model (not scalar regression)
- [ ] 4. Test: Generate coherent tokens through llama.cpp

### Phase 3: Training / Fine-tuning

- [ ] 1. Train with real text, not just scalar regression
- [ ] 2. Measure perplexity
- [ ] 3. Iterate on weight values if needed

### Phase 4: Optimization

- [ ] 1. Native quaternary matmul in ggml
- [ ] 2. GPU/CUDA kernels
- [ ] 3. Optimize quantization boundaries

## Key Files to Modify

```
llama.cpp/src/
  llama-arch.h           - [x] Add LLM_ARCH_QUATERNARY_NN, [ ] tensor enums
  llama-arch.cpp        - [x] Add name mapping, [ ] tensor info
  llama-model-loader.cpp - [ ] Load quaternary weights
  llama-model.cpp      - [ ] Add case in build_graph()
  models/
    quaternary_nn.cpp  - [ ] NEW: Forward pass implementation

fourBitState/
  quaternary_nn/src/QuaternaryNN.cpp - [x] Updated values
  quaternary_nn/src/GGUFHelper.cpp   - [x] Uses GGML_TYPE_Q2_Q (99)
  teacher_pipeline.py                - [x] Working with semantic critic
  buildLibrary.py                   - [x] Library builder
```

## Open Questions / Worries

### 1. Attention Quality
**Worry**: {±1.0, ±0.5} may be too coarse for fine-grained attention patterns. 
**To test**: Train on text corpus, measure perplexity vs float baseline.

### 2. Error Accumulation  
4096 hidden × 32 heads × 4096^2 multiplications = 16M ops per layer. Small errors compound.
**Mitigation**: Start with small models, measure degradation empirically.

### 3. Token Coherence
Will binary-ish attention produce coherent text?
**Requirement**: Must train on actual text, not just scalar regression.

### 4. GGUF Format Compatibility
Current quaternary_nn GGUF format may not match llama.cpp expectations. We may need to adjust or create a proper GGUF with an appropriate architecture and tensors.

### 5. Training Infrastructure Gap
Current training (teacher_pipeline.py) only does scalar regression (input f64 → output f64). To do attention-based token generation, we need a completely different training approach.

### 6. Bootstrapping Problem
How to start training? Options:
- Train base model with standard weights, then quantize → requires HF model
- Train directly with quaternary weights → harder convergence

### 7. Value selection: ±0.01 vs ±0.5
**RESOLVED**: First attempt with ±0.01 was too small — weights could not traverse the quantization boundaries via STE gradients. Second attempt with ±0.5 works: the Q2_Q convergence metric climbs steadily as training progresses (from ~5% to ~70% on d=64 over 50K steps in the training log). The ±0.5 values provide enough room for meaningful gradient updates while maintaining the 2-bit packing benefit.

## Timeline Estimate

- Phase 1: 1 day (minimal working - C++ code compiles, loads model)
- Phase 2: 2 days (attention working - text generation possible)
- Phase 3: 1+ week (quality acceptable)
- Phase 4: Ongoing

### 4. GGUF Format Compatibility (RESOLVED)

Use **gguf-py** exclusively to write production GGUF files. The standalone `GGUFHelper.cpp` writes
a simplified format that llama.cpp cannot parse. The `create_minimal_test.py` script demonstrates
the correct approach: use `GGUFWriter` from gguf-py with proper metadata keys and tensor shapes.

Key: always pass numpy arrays with dimensions in the order (`n_vocab, n_embd`) for a tensor
with ggml shape `[n_embd, n_vocab]`.

## Tensor Shape Conventions (Important)

When creating GGUF models with gguf-py:
- **numpy shape (M, N)** → **GGUF reader shape [N, M]** → **C++ ggml ne[0]=N, ne[1]=M**
- For `create_tensor(..., {n_embd, n_vocab})` → needs numpy shape `(n_vocab, n_embd)`
- The gguf-py `add_tensor` stores `tensor.shape` directly (no reversal), so you must pass
  the numpy array with dimensions in the order ggml expects reversed.

## Session Log

### [2025-04-25] Phase 1 Implementation
- Added LLM_ARCH_QUATERNARY_NN enum and name mapping
- Added tensor enums (QUATERNARY_HEAD, QUATERNARY_BIAS)  
- Added tensor names and infos to llama-arch.cpp
- Added hparams loading for quaternary_nn in llama-model.cpp
- Added tensor creation for quaternary head tensors
- Created models/quaternary_nn.cpp with forward pass
- Fixed compilation errors (hparam arrays, layer field names)
- Fixed struct declaration to include `const llama_model & model` field
- llama.cpp compiles with quaternary_nn support
- **BLOCKER**: GGUF format incompatibility - model file won't load
- Updated quaternary_nn to write tensor names with ".weight" suffix

### [2025-04-26] Phase 1 Completion + vec_dot

## Converter State (PAUSED)

`convert_to_quaternary.py` needs rewriting to:
1. Read source GGUF (e.g. Qwen2.5 0.5B in `models/`)
2. Dequantize attention weights (attn_q/k/v/o) to f32 via `gguf.quants.Q6_K/Q5_0.dequantize()`
3. Re-quantize to Q2_Q via `Q2_Q.quantize_blocks()`, add with `raw_dtype=GGMLQuantizationType.Q2_Q`
4. Add required metadata: `full_attention_interval=1`, `rope.dimension_sections`
5. Keep everything else (tokenizer, FFN weights) as f32
6. Rename metadata keys from source arch → `quaternary_nn.*`

Source model (`models/qwen2.5-0.5b-instruct-q4_k_m.gguf`):
- `qwen2` arch, 24L, 896 embd, 14H (2 KV), 4864 FF
- Tensor names already match quaternary_nn convention (`blk.{i}.attn_q.weight`)
- attn_q ne=[896,896] (Q5_0), attn_k/v ne=[896,128] (Q5_0/Q6_K)
- attn_output ne=[896,896] (Q5_0/Q6_K)

## Session Log

### [2025-04-25] Phase 1 Implementation
- Added LLM_ARCH_QUATERNARY_NN enum and name mapping
- Added tensor enums (QUATERNARY_HEAD, QUATERNARY_BIAS)  
- Added tensor names and infos to llama-arch.cpp
- Added hparams loading for quaternary_nn in llama-model.cpp
- Added tensor creation for quaternary head tensors
- Created models/quaternary_nn.cpp with forward pass
- Fixed compilation errors (hparam arrays, layer field names)
- Fixed struct declaration to include `const llama_model & model` field
- llama.cpp compiles with quaternary_nn support
- **BLOCKER**: GGUF format incompatibility - model file won't load
- Updated quaternary_nn to write tensor names with ".weight" suffix

### [2025-04-26] Phase 1 Completion
- Fixed GGUF format: gguf-py instead of custom GGUFHelper
- Made rope.dimension_sections optional for quaternary_nn
- Fixed rs cache crash: s_copy init, null-buffer guards, n_rs guard
- Created create_minimal_test.py for test GGUF generation

### [2025-04-27] vec_dot + Standard Attention + Cleanup
- **CPU vec_dot**: Implemented `ggml_vec_dot_q2_Q_q8_0` (strong/weak accumulator split)
- **Wired up**: Registered in `ggml-cpu.c` type_traits_cpu
- **Standard attention**: Added to `quaternary_nn.cpp` when `layer.wqkv_gate` is NULL
- **FFN fallback**: Uses `ffn_norm` when `attn_post_norm` absent
- **Verified**: Teacher-style model (Q2_Q attn + F32 FFN) loads & infers on CPU + GPU
- **CUDA**: No custom Q2_Q kernel yet (falls back to dequantize+f16 matmul)
- **Cleanup**: Removed 22 garbage files (old ggufs, one-off scripts, stale docs)
