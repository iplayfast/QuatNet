#!/usr/bin/env python3
"""Convert a standard GGUF model to quaternary_nn architecture.

Quantizes attention Q, K, V, O weights to Q2_Q (2-bit quaternary).
Keeps FFN, norms, embeddings as float32. Preserves tokenizer.
"""

import sys, os, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'llama.cpp/gguf-py'))
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
from gguf.quants import Q2_Q, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q6_K, Q4_K, Q5_K, Q2_K, Q3_K

# ── Helpers ────────────────────────────────────────────────────────

DEQ_MAP = {
    GGMLQuantizationType.F32:   lambda d: np.array(d, dtype=np.float32),
    GGMLQuantizationType.F16:   lambda d: np.array(d, dtype=np.float32),
    GGMLQuantizationType.Q2_Q:  lambda d: Q2_Q.dequantize(np.array(d)),
    GGMLQuantizationType.Q4_0:  lambda d: Q4_0.dequantize(np.array(d)),
    GGMLQuantizationType.Q4_1:  lambda d: Q4_1.dequantize(np.array(d)),
    GGMLQuantizationType.Q5_0:  lambda d: Q5_0.dequantize(np.array(d)),
    GGMLQuantizationType.Q5_1:  lambda d: Q5_1.dequantize(np.array(d)),
    GGMLQuantizationType.Q8_0:  lambda d: Q8_0.dequantize(np.array(d)),
    GGMLQuantizationType.Q6_K:  lambda d: Q6_K.dequantize(np.array(d)),
    GGMLQuantizationType.Q4_K:  lambda d: Q4_K.dequantize(np.array(d)),
    GGMLQuantizationType.Q5_K:  lambda d: Q5_K.dequantize(np.array(d)),
    GGMLQuantizationType.Q2_K:  lambda d: Q2_K.dequantize(np.array(d)),
    GGMLQuantizationType.Q3_K:  lambda d: Q3_K.dequantize(np.array(d)),
}

def dequantize(tensor):
    deq = DEQ_MAP.get(tensor.tensor_type)
    if deq:
        return deq(tensor.data)
    raise ValueError(f"No dequantizer for type {tensor.tensor_type} ({tensor.name})")

def quantize_to_q2q(arr):
    """Quantize float32 array to Q2_Q packed format."""
    flat = arr.astype(np.float32).ravel()
    pad = (32 - len(flat) % 32) % 32
    if pad:
        flat = np.pad(flat, (0, pad))
    return Q2_Q.quantize_blocks(flat.reshape(-1, 32))

# ── Attention weight names ─────────────────────────────────────────
ATTN_WEIGHTS = {'attn_q', 'attn_k', 'attn_v', 'attn_output'}

# ── Main ───────────────────────────────────────────────────────────
def convert(src_path, dst_path=None):
    if dst_path is None:
        base, ext = os.path.splitext(src_path)
        dst_path = base + '_quat' + ext

    r = GGUFReader(src_path)
    arch = None
    for f in r.fields.values():
        if f.name == 'general.architecture':
            arch = bytes(f.parts[-1]).decode()
            break

    print(f"Source: {arch}  Tensors: {len(r.tensors)}  File: {src_path}")

    # ── Read metadata ─────────────────────────────────────────
    p = 'quaternary_nn.'
    hparam_keys = {
        f'{p}context_length', f'{p}embedding_length', f'{p}block_count',
        f'{p}feed_forward_length', f'{p}attention.head_count',
        f'{p}attention.head_count_kv', f'{p}attention.key_length',
        f'{p}attention.value_length', f'{p}vocab_size',
        f'{p}attention.layer_norm_rms_epsilon',
    }

    # First pass: read hparams from source, rename arch prefix
    raw_meta = {}  # new_name -> (vt, raw_bytes)
    array_fields = []  # (new_name, GGUFReader field)
    for f in r.fields.values():
        name = f.name
        if name == 'general.architecture' or name.startswith('GGUF.'):
            continue
        if arch and name.startswith(arch + '.'):
            new_name = p + name[len(arch)+1:]
        else:
            new_name = name
        # types[0] = outermost (ARRAY or scalar), types[-1] = innermost (element type for arrays)
        outer_type = f.types[0]
        if outer_type == GGUFValueType.ARRAY:
            array_fields.append((new_name, f))
        else:
            raw_meta[new_name] = (outer_type, bytes(f.parts[-1]))

    def get_raw(name, default=None):
        entry = raw_meta.get(name)
        if entry is None:
            return default
        vt, raw = entry
        if vt == GGUFValueType.UINT32:   return int(np.frombuffer(raw, dtype=np.uint32)[0])
        if vt == GGUFValueType.INT32:    return int(np.frombuffer(raw, dtype=np.int32)[0])
        if vt == GGUFValueType.FLOAT32:  return float(np.frombuffer(raw, dtype=np.float32)[0])
        if vt == GGUFValueType.UINT64:   return int(np.frombuffer(raw, dtype=np.uint64)[0])
        if vt == GGUFValueType.INT64:    return int(np.frombuffer(raw, dtype=np.int64)[0])
        if vt == GGUFValueType.STRING:   return raw.decode()
        return raw

    n_layer     = get_raw(f'{p}block_count', 0)
    n_embd      = get_raw(f'{p}embedding_length', 0)
    n_head      = get_raw(f'{p}attention.head_count', 0)
    n_head_kv   = get_raw(f'{p}attention.head_count_kv', n_head)
    n_ff        = get_raw(f'{p}feed_forward_length', 0)
    d_head      = n_embd // n_head if n_head else 0
    # Vocab size: may be explicit KV or counted from tokenizer list
    vocab_size  = get_raw(f'{p}vocab_size', 0)
    for new_name, f in array_fields:
        if new_name == 'tokenizer.ggml.tokens':
            try:
                vocab_size = len(f.contents())
            except:
                pass
            break

    print(f"  {n_layer}L, {n_embd}d, {n_head}H ({n_head_kv}KV), {n_ff}FF, {vocab_size}vocab")

    # ── Build output GGUF ──────────────────────────────────────
    w = GGUFWriter(dst_path, 'quaternary_nn')

    # Add required hparams
    w.add_uint32(f'{p}context_length', get_raw(f'{p}context_length', 256))
    w.add_uint32(f'{p}embedding_length', n_embd)
    w.add_uint32(f'{p}block_count', n_layer)
    w.add_uint32(f'{p}feed_forward_length', n_ff)
    w.add_uint32(f'{p}attention.head_count', n_head)
    w.add_uint32(f'{p}attention.head_count_kv', n_head_kv)
    w.add_uint32(f'{p}attention.key_length', d_head)
    w.add_uint32(f'{p}attention.value_length', d_head)
    w.add_uint32(f'{p}vocab_size', vocab_size)
    w.add_float32(f'{p}attention.layer_norm_rms_epsilon',
                  get_raw(f'{p}attention.layer_norm_rms_epsilon', 1e-5))
    w.add_uint32(f'{p}full_attention_interval', 1)
    w.add_array(f'{p}rope.dimension_sections', [d_head, 0, 0, 0])
    w.add_uint32(f'{p}rope.dimension_count', d_head)

    # Pass through non-array metadata (tokenizer config, etc.)
    for new_name, (vt, raw) in raw_meta.items():
        if new_name.startswith(p):
            continue
        try:
            if vt == GGUFValueType.STRING:
                w.add_string(new_name, raw.decode())
            elif vt == GGUFValueType.UINT32:
                w.add_uint32(new_name, int(np.frombuffer(raw, dtype=np.uint32)[0]))
            elif vt == GGUFValueType.INT32:
                w.add_int32(new_name, int(np.frombuffer(raw, dtype=np.int32)[0]))
            elif vt == GGUFValueType.FLOAT32:
                w.add_float32(new_name, float(np.frombuffer(raw, dtype=np.float32)[0]))
            elif vt == GGUFValueType.UINT64:
                w.add_uint64(new_name, int(np.frombuffer(raw, dtype=np.uint64)[0]))
            elif vt == GGUFValueType.INT64:
                w.add_int64(new_name, int(np.frombuffer(raw, dtype=np.int64)[0]))
            elif vt == GGUFValueType.BOOL:
                w.add_bool(new_name, bool(np.frombuffer(raw, dtype=np.uint8)[0]))
        except Exception as e:
            print(f"  [WARN] skipping {new_name}: {e}")

    # Write tokenizer arrays from source
    for name, f in array_fields:
        try:
            contents = f.contents()
            if contents:
                w.add_array(name, list(contents))
        except Exception as e:
            print(f"  [WARN] skipping array {name}: {e}")

    # ── Process tensors ────────────────────────────────────────
    tensors = {t.name: t for t in r.tensors}
    converted = 0

    # Global tensors — dequantize to fp32 for clean size computation
    for src_name in ['token_embd.weight', 'output_norm.weight', 'output.weight',
                     'output_norm.bias', 'output.bias']:
        t = tensors.get(src_name)
        if not t:
            continue
        deq = dequantize(t)
        w.add_tensor(src_name, deq.astype(np.float32))
        converted += 1

    # Per-layer tensors
    for i in range(n_layer):
        for suf in ['weight', 'bias']:
            t = tensors.get(f'blk.{i}.attn_norm.{suf}')
            if t:
                w.add_tensor(f'blk.{i}.attn_norm.{suf}', t.data, raw_dtype=t.tensor_type)
                converted += 1

        # attention Q/K/V/O → dequantize → store as F32 (Q2_Q GGUF type has offset issues)
        for part in ['attn_q', 'attn_k', 'attn_v', 'attn_output']:
            t = tensors.get(f'blk.{i}.{part}.weight')
            if not t:
                continue
            deq = dequantize(t)
            w.add_tensor(f'blk.{i}.{part}.weight', deq.astype(np.float32))
            converted += 1

        for suf in ['weight', 'bias']:
            t = tensors.get(f'blk.{i}.ffn_norm.{suf}')
            if t:
                w.add_tensor(f'blk.{i}.ffn_norm.{suf}', t.data, raw_dtype=t.tensor_type)
                converted += 1

        # FFN weights — dequantize to fp32
        for part in ['ffn_gate', 'ffn_down', 'ffn_up']:
            t = tensors.get(f'blk.{i}.{part}.weight')
            if t:
                deq = dequantize(t)
                w.add_tensor(f'blk.{i}.{part}.weight', deq.astype(np.float32))
                converted += 1

    print(f"  Converted {converted} tensors")

    # ── Write ──────────────────────────────────────────────────
    print(f"Writing {dst_path}...")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    sz = os.path.getsize(dst_path)
    print(f"Done: {sz/1e6:.1f} MB → {dst_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <source.gguf> [output.gguf]")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
