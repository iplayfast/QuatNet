# TODO — Issues found during documentation review

Status: ✅ Fixed   🔄 In Progress   ❌ Not Started   ⏸️ Deferred

## Critical

### 1. GGUF Export Never Uses Q2_Q Packing
**Status: ✅ Fixed**

Removed dead `q2q()` / `Q2_Q` import code. Attention weights are exported as F32 with values quantized to {1.0, 0.5, -0.5, -1.0}. Comment updated to explain the deferred status.

### 2. llama.cpp CPU Backend Will Crash on Q2_Q Tensors
**Status: ⏸️ Deferred — requires substantial C++ `type_traits_cpu` work**

`ggml-cpu.c` is missing a `type_traits_cpu[GGML_TYPE_Q2_Q]` entry. Not blocking because the export uses F32 (see #1). When Q2_Q packing is needed, this requires implementing `quantize_row_q2_Q`, `ggml_vec_dot_q2_Q_q8_0`, and wiring the `build_lora_mm` graph builder to use the dedicated `ggml_mul_mat_q2_Q` op instead of standard `ggml_mul_mat`.

### 3. Distillation Signal is Effectively 1-Hot
**Status: ✅ Fixed**

Removed the broken distillation entirely. The KL divergence against a near-1-hot teacher distribution was adding noise, not useful signal. Ollama's `/api/generate` doesn't expose logits, so proper distillation isn't feasible without a different server setup.

## High

### 4. Training Data Gets Silently Corrupted
**Status: ✅ Fixed**

`library_populator.py` now uses an incrementing counter (`len(existing) + 1`) instead of overwriting random existing files.

### 5. No Layer Growth — Stuck at 4 Layers Forever
**Status: ✅ Fixed**

`grow()` now tracks a `_growth_count` and adds a layer every other growth event, clamped at `d//32` max layers. The formula `new_layers = max(self.n_layers, d//64)` was replaced with `self.n_layers + (1 if growth_count % 2 == 0 else 0)`.

### 6. Q2_Q Convergence Metric is Misleading at Small Dimensions
**Status: ✅ Fixed**

Growth via Q2_Q convergence is now blocked until `d_model >= 64`. Below that, growth is only triggered by loss plateaus.

### 7. No Held-Out Validation
**Status: ✅ Fixed**

Training data is now split 90/10 train/val. Validation loss is computed and logged at every save interval. Log format expanded to include `val_loss` column.

## Medium

### 8. GGUF Export Comment is Misleading
**Status: ✅ Fixed**

Updated comment to accurately describe the F32 export with deferred Q2_Q packing.

### 9. create_minimal_test.py Adds Unused SSM Hparams
**Status: ✅ Fixed**

Removed all SSM-related hparams (`ssm.conv_kernel`, `ssm.state_size`, `ssm.group_count`, `ssm.time_step_rank`, `ssm.inner_size`) and unused imports.

### 10. Byte-Level Modeling on Code is Inefficient
**Status: ✅ Fixed (partial — increased MAX_SEQ)**

`MAX_SEQ` increased from 128 → 512. Full BPE tokenizer is a larger project deferred for later.

## Low

### 11. Batch Sampling Has Heavy Overlap
**Status: ✅ Fixed**

Replaced independent `randint()` sampling with `torch.randperm()` — non-overlapping shuffled indices per batch.

### 12. Python Q2_Q Quantize is Slow
**Status: ✅ Fixed**

Vectorized both `quantize_blocks()` and `dequantize_blocks()` using NumPy vector operations (`np.select`, broadcasting shifts) instead of element-wise Python loops.

### 13. Library Populator Spams bestquestions on Every Write
**Status: ✅ Fixed**

Removed the bestquestions seeding loop entirely. The library now only gets freshly-generated Q&A pairs.

### 14. Output Weight Decay / Regularization Not Used
**Status: ✅ Fixed**

`AdamW` now uses `weight_decay=0.01` on initial optimizer creation and after every growth reset.
