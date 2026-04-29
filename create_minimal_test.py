#!/usr/bin/env python3
"""Create a minimal quaternary_nn GGUF model with a real attention layer.

All layers are attention (not recurrent/SSM) to avoid rs cache issues.
Uses proper tensor shapes matching the C++ forward pass.

Shape conventions:
  numpy shape (A, B) → GGUF ne = [B, A]
  C++ create_tensor expects {ne0, ne1, ...}
"""

import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'llama.cpp/gguf-py'))

from gguf import GGUFWriter, GGMLQuantizationType
from gguf.quants import Q2_Q

ARCH = "quaternary_nn"
OUTPUT = "minimal_test.gguf"

# Hyperparameters
n_embd = 32
n_head = 8
n_head_kv = 8
n_layer = 1
n_embd_head_k = n_embd // n_head  # 4
n_embd_head_v = n_embd // n_head  # 4
n_embd_k_gqa = n_embd_head_k * n_head_kv  # 32
n_embd_v_gqa = n_embd_head_v * n_head_kv  # 32
n_ff = n_embd * 8  # 256
vocab_size = 256

# SSM params (non-zero to avoid divide-by-zero, but no SSM tensors)
ssm_d_conv = 4
ssm_d_state = 4
ssm_n_group = 4
ssm_dt_rank = 4
ssm_d_inner = 48

print(f"Creating test model: n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}")

w = GGUFWriter(OUTPUT, ARCH)

# ── Metadata ──
w.add_uint32(f"{ARCH}.context_length", 256)
w.add_uint32(f"{ARCH}.embedding_length", n_embd)
w.add_uint32(f"{ARCH}.block_count", n_layer)
w.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-5)
w.add_uint32(f"{ARCH}.feed_forward_length", n_ff)
w.add_uint32(f"{ARCH}.attention.head_count", n_head)
w.add_uint32(f"{ARCH}.attention.head_count_kv", n_head_kv)
w.add_uint32(f"{ARCH}.attention.key_length", n_embd_head_k)
w.add_uint32(f"{ARCH}.attention.value_length", n_embd_head_v)
w.add_uint32(f"{ARCH}.vocab_size", vocab_size)

# SSM params (to appease n_embd_r/s calc)
w.add_uint32(f"{ARCH}.ssm.conv_kernel", ssm_d_conv)
w.add_uint32(f"{ARCH}.ssm.state_size", ssm_d_state)
w.add_uint32(f"{ARCH}.ssm.group_count", ssm_n_group)
w.add_uint32(f"{ARCH}.ssm.time_step_rank", ssm_dt_rank)
w.add_uint32(f"{ARCH}.ssm.inner_size", ssm_d_inner)

# full_attention_interval=1: all layers are attention (not recurrent)
w.add_uint32(f"{ARCH}.full_attention_interval", 1)

w.add_array(f"{ARCH}.rope.dimension_sections", [n_embd_head_k, 0, 0, 0])
w.add_uint32(f"{ARCH}.rope.dimension_count", n_embd_head_k)

w.add_string("tokenizer.ggml.model", "none")
w.add_uint32("tokenizer.ggml.tokens", vocab_size)

# ── Helper: create weight tensor with correct shape ──
def add_weight(name, numpy_shape):
    """numpy_shape should be (ne1, ne0) to produce GGUF ne=(ne0, ne1)."""
    data = np.random.randn(*numpy_shape).astype(np.float32) * 0.02
    w.add_tensor(name, data)

# ── Global tensors ──
# token_embd: ne=[n_embd, n_vocab] = [32, 256] → numpy (256, 32)
add_weight("token_embd.weight", (vocab_size, n_embd))

# output_norm: ne=[n_embd] → numpy (n_embd,)
w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))

# output: ne=[n_embd, n_vocab] → numpy (n_vocab, n_embd)
add_weight("output.weight", (vocab_size, n_embd))

# ── Layer 0 (attention) ──
# attn_norm: ne=[n_embd]
w.add_tensor("blk.0.attn_norm.weight", np.ones(n_embd, dtype=np.float32))

# attn_q: standard attention
# ne=[n_embd, n_embd_head_k * n_head] = [32, 32]
add_weight("blk.0.attn_q.weight", (n_embd_k_gqa, n_embd))

# attn_k: ne=[n_embd, n_embd_k_gqa] → numpy (n_embd_k_gqa, n_embd)
add_weight("blk.0.attn_k.weight", (n_embd_k_gqa, n_embd))

# attn_v: ne=[n_embd, n_embd_v_gqa] → numpy (n_embd_v_gqa, n_embd)
add_weight("blk.0.attn_v.weight", (n_embd_v_gqa, n_embd))

# attn_output: ne=[n_embd_k_gqa, n_embd] → numpy (n_embd, n_embd_k_gqa)
add_weight("blk.0.attn_output.weight", (n_embd, n_embd_k_gqa))

# FFN norm
w.add_tensor("blk.0.ffn_norm.weight", np.ones(n_embd, dtype=np.float32))

# FFN gate: ne=[n_embd, n_ff] → numpy (n_ff, n_embd)
add_weight("blk.0.ffn_gate.weight", (n_ff, n_embd))

# FFN down: ne=[n_ff, n_embd] → numpy (n_embd, n_ff)
add_weight("blk.0.ffn_down.weight", (n_embd, n_ff))

# FFN up: ne=[n_embd, n_ff] → numpy (n_ff, n_embd)
add_weight("blk.0.ffn_up.weight", (n_ff, n_embd))

# ── Write ──
print("Writing GGUF...")
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()

sz = os.path.getsize(OUTPUT)
print(f"Done: {sz} bytes → {OUTPUT}")

# ── Verify ──
from gguf import GGUFReader
r = GGUFReader(OUTPUT)
print(f"\nVerification: {len(r.tensors)} tensors")
for t in r.tensors:
    print(f"  {t.name}: ne={list(t.shape)} type={t.tensor_type}")
