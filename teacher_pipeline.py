#!/usr/bin/env python3
"""
teacher_pipeline.py - Quaternary Transformer Training Pipeline
Trains a tiny Q2_Q transformer on Q&A library text, exports GGUF, verifies with llama.cpp.
"""
import os, sys, json, time, signal, random, math, subprocess

# Reduce CUDA memory fragmentation before any torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llama.cpp", "gguf-py"))

import torch
import torch.nn as nn
import torch.nn.functional as F



# ── Configuration ─────────────────────────────────────────────────
LIBRARY_DIR    = "library"
LLAMA_MODEL    = "quaternary_trained.gguf"
SAVE_INTERVAL  = 500   # save every N training steps
VERIFY_EVERY   = 5000  # run llama-cli verification
LOG_FILE       = "training_log.csv"  # metrics CSV for plotting

# Model hyperparams (same as train_quaternary.py)
D_MODEL  = 2
N_HEADS  = 8
N_LAYERS = 4
D_FF     = 256
MAX_SEQ  = 512
BATCH    = 16
LR       = 5e-4
VOCAB    = 256
WEIGHT_DECAY = 0.01

# Global for signal handler
_model = None
_device = None
_all_text = ""

# ── Model Definition ──────────────────────────────────────────────
class QuaternaryLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        w = self.weight
        w_q = torch.where(w > 0.75, torch.tensor(1.0, device=w.device),
               torch.where(w > 0.0,  torch.tensor(0.5, device=w.device),
               torch.where(w > -0.75,torch.tensor(-0.5, device=w.device),
                                      torch.tensor(-1.0, device=w.device))))
        w_ste = w_q.detach() + w - w.detach()
        return F.linear(x, w_ste, self.bias)

    def get_quantized_weight(self):
        w = self.weight.detach()
        return torch.where(w > 0.75, torch.tensor(1.0),
               torch.where(w > 0.0,  torch.tensor(0.5),
               torch.where(w > -0.75,torch.tensor(-0.5),
                                      torch.tensor(-1.0)))).cpu().numpy()


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class QuaternaryAttention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.d_head = d // n_heads
        self.d = d
        self.q = QuaternaryLinear(d, self.n_heads * self.d_head)
        self.k = QuaternaryLinear(d, self.n_kv_heads * self.d_head)
        self.v = QuaternaryLinear(d, self.n_kv_heads * self.d_head)
        self.o = QuaternaryLinear(self.n_heads * self.d_head, d)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        # Expand KV heads to match Q heads for GQA
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        return self.o(out)


class FeedForward(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up   = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class QuaternaryBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff, n_kv_heads=None, dropout=0.1):
        super().__init__()
        self.attn_norm = RMSNorm(d)
        self.attn = QuaternaryAttention(d, n_heads, n_kv_heads)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_norm = RMSNorm(d)
        self.ffn = FeedForward(d, d_ff)
        self.ffn_drop = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.attn_drop(self.attn(self.attn_norm(x)))
        x = x + self.ffn_drop(self.ffn(self.ffn_norm(x)))
        return x


class QuaternaryLLM(nn.Module):
    def __init__(self, vocab, d, n_heads, n_layers, d_ff, max_seq, n_kv_heads=None, growth_count=0, dropout=0.1):
        super().__init__()
        n_heads = max(1, min(n_heads, d))
        n_kv_heads = max(1, min(n_kv_heads if n_kv_heads is not None else n_heads, d))
        self.vocab_size, self.d_model = vocab, d
        self.n_heads, self.n_layers, self.max_seq = n_heads, n_layers, max_seq
        self.n_kv_heads = n_kv_heads
        self._growth_count = growth_count
        self._dropout = dropout
        self.tok_embd = nn.Embedding(vocab, d)
        self.pos_embd = nn.Embedding(max_seq, d)
        self.layers = nn.ModuleList([QuaternaryBlock(d, n_heads, d_ff, self.n_kv_heads, dropout) for _ in range(n_layers)])
        self.output_norm = RMSNorm(d)
        self.output = nn.Linear(d, vocab, bias=False)

    def grow(self):
        """Increase model capacity: widen d_model, add layers every other growth."""
        MAX_D = 16384
        old_d = self.d_model
        growth_count = getattr(self, '_growth_count', 0) + 1
        self._growth_count = growth_count

        if old_d < 16:
            new_d = old_d * old_d  # square: 2→4, 4→16
        elif old_d < 64:
            new_d = old_d * 4      # 16→64
        elif old_d < MAX_D:
            new_d = old_d * 2      # double: 64→128→256→512→1024→...→16384
        else:
            new_d = old_d

        new_heads = min(new_d, max(1, new_d // 16))
        new_kv = min(new_heads, max(1, new_d // 64))
        new_ff = new_d * 4

        # Add a layer every 2 growths, clamped at d//32
        max_layers = max(4, new_d // 32)
        new_layers = min(max_layers, self.n_layers + (1 if growth_count % 2 == 0 else 0))
        new_layers = max(new_layers, self.n_layers)  # never shrink
        desc = f"Widening {old_d}d→{new_d}d, layers {self.n_layers}→{new_layers}"
        old_vocab = self.vocab_size

        new_model = QuaternaryLLM(old_vocab, new_d, new_heads, new_layers, new_ff, self.max_seq, new_kv, growth_count, self._dropout)
        new_model = new_model.to(next(self.parameters()).device)

        # Widen embedding
        with torch.no_grad():
            new_model.tok_embd.weight[:old_vocab, :old_d] = self.tok_embd.weight
            new_model.pos_embd.weight[:, :old_d] = self.pos_embd.weight

        # Copy existing layers, pad new ones (handles changing head counts)
        for i in range(min(self.n_layers, len(new_model.layers))):
            old_layer = self.layers[i]
            new_layer = new_model.layers[i]
            with torch.no_grad():
                for name in ['attn_norm', 'ffn_norm']:
                    old_w = getattr(old_layer, name).weight
                    new_w = getattr(new_layer, name).weight
                    d_min = min(old_w.shape[0], new_w.shape[0])
                    new_w[:d_min] = old_w[:d_min]
                for name in ['q', 'k', 'v', 'o']:
                    old_w = getattr(old_layer.attn, name).weight
                    new_w = getattr(new_layer.attn, name).weight
                    d0 = min(old_w.shape[0], new_w.shape[0])
                    d1 = min(old_w.shape[1], new_w.shape[1])
                    new_w[:d0, :d1] = old_w[:d0, :d1]
                for name in ['gate', 'up', 'down']:
                    old_w = getattr(old_layer.ffn, name).weight
                    new_w = getattr(new_layer.ffn, name).weight
                    d0 = min(old_w.shape[0], new_w.shape[0])
                    d1 = min(old_w.shape[1], new_w.shape[1])
                    new_w[:d0, :d1] = old_w[:d0, :d1]

        # Copy output norm and projection
        with torch.no_grad():
            d_min = min(self.output_norm.weight.shape[0], new_model.output_norm.weight.shape[0])
            new_model.output_norm.weight[:d_min] = self.output_norm.weight[:d_min]
            d0 = min(self.output.weight.shape[0], new_model.output.weight.shape[0])
            d1 = min(self.output.weight.shape[1], new_model.output.weight.shape[1])
            new_model.output.weight[:d0, :d1] = self.output.weight[:d0, :d1]

        print(f"  [GROW] {old_d}d×{self.n_layers}L→{new_d}d×{new_layers}L ({sum(p.numel() for p in new_model.parameters()):,} params)")
        return new_model

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_embd(idx) + self.pos_embd(pos)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.output_norm(x))

    def export_gguf(self, path):
        from gguf import GGUFWriter

        d, vocab = self.d_model, self.vocab_size
        n_layer, n_head, d_ff = self.n_layers, self.n_heads, self.layers[0].ffn.gate.out_features
        d_head = d // n_head

        w = GGUFWriter(path, "quaternary_nn")
        w.add_uint32("quaternary_nn.context_length", self.max_seq)
        w.add_uint32("quaternary_nn.embedding_length", d)
        w.add_uint32("quaternary_nn.block_count", n_layer)
        w.add_uint32("quaternary_nn.feed_forward_length", d_ff)
        w.add_uint32("quaternary_nn.attention.head_count", n_head)
        w.add_uint32("quaternary_nn.attention.head_count_kv", self.n_kv_heads)
        w.add_uint32("quaternary_nn.attention.key_length", d_head)
        w.add_uint32("quaternary_nn.attention.value_length", d_head)
        w.add_uint32("quaternary_nn.vocab_size", vocab)
        w.add_float32("quaternary_nn.attention.layer_norm_rms_epsilon", 1e-5)
        w.add_float32("quaternary_nn.dropout", self._dropout)
        w.add_uint32("quaternary_nn.full_attention_interval", 1)
        w.add_uint32("quaternary_nn.growth_count", getattr(self, '_growth_count', 0))
        w.add_array("quaternary_nn.rope.dimension_sections", [d_head, 0, 0, 0])
        w.add_uint32("quaternary_nn.rope.dimension_count", d_head)
        w.add_string("tokenizer.ggml.model", "gpt2")
        # GPT-2 byte encoding: self-mapped bytes (0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF)
        # are stored as their UTF-8 codepoint encoding (e.g. 0xA1 → b'\xc2\xa1' for '¡').
        # All other bytes map to U+0100+ so llama.cpp's byte decoder can round-trip.
        byte_tokens = []
        n = 0
        for i in range(256):
            if 0x21 <= i <= 0x7E or 0xA1 <= i <= 0xAC or 0xAE <= i <= 0xFF:
                byte_tokens.append(chr(i).encode('utf-8'))
            else:
                byte_tokens.append(chr(256 + n).encode('utf-8'))
                n += 1
        w.add_token_list(byte_tokens)
        w.add_token_merges([b""])

        # All weights exported as F32. Attention weights have values quantized to
        # {1.0, 0.5, -0.5, -1.0} via get_quantized_weight(). True Q2_Q packing
        # (type 99) is deferred until the C++ backend has full type_traits_cpu support.
        w.add_tensor("token_embd.weight", self.tok_embd.weight.detach().cpu().numpy().astype(np.float32))
        w.add_tensor("output_norm.weight", self.output_norm.weight.detach().cpu().numpy().astype(np.float32))
        w.add_tensor("output.weight", self.output.weight.detach().cpu().numpy().astype(np.float32))

        for i, layer in enumerate(self.layers):
            w.add_tensor(f"blk.{i}.attn_norm.weight", layer.attn_norm.weight.detach().cpu().numpy().astype(np.float32))
            w.add_tensor(f"blk.{i}.attn_q.weight", layer.attn.q.get_quantized_weight().astype(np.float32))
            w.add_tensor(f"blk.{i}.attn_k.weight", layer.attn.k.get_quantized_weight().astype(np.float32))
            w.add_tensor(f"blk.{i}.attn_v.weight", layer.attn.v.get_quantized_weight().astype(np.float32))
            w.add_tensor(f"blk.{i}.attn_output.weight", layer.attn.o.get_quantized_weight().astype(np.float32))
            w.add_tensor(f"blk.{i}.ffn_norm.weight", layer.ffn_norm.weight.detach().cpu().numpy().astype(np.float32))
            w.add_tensor(f"blk.{i}.ffn_gate.weight", layer.ffn.gate.weight.detach().cpu().numpy().astype(np.float32))
            w.add_tensor(f"blk.{i}.ffn_down.weight", layer.ffn.down.weight.detach().cpu().numpy().astype(np.float32))
            w.add_tensor(f"blk.{i}.ffn_up.weight", layer.ffn.up.weight.detach().cpu().numpy().astype(np.float32))

        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        sz = os.path.getsize(path)
        print(f"[GGUF] Saved {path} ({sz} bytes)")

    @classmethod
    def from_gguf(cls, path, device):
        """Load model weights from any GGUF file, auto-detecting architecture."""
        from gguf import GGUFReader, GGMLQuantizationType, GGUFValueType
        from gguf.quants import Q2_Q

        r = GGUFReader(path)
        meta = {}
        for f in r.fields.values():
            if f.name.startswith('GGUF.'):
                continue
            raw = bytes(f.parts[-1])
            vt = f.types[-1]
            if vt == GGUFValueType.STRING:
                meta[f.name] = raw.decode('utf-8', errors='replace')
            elif vt == GGUFValueType.UINT32:
                meta[f.name] = int(np.frombuffer(raw, dtype=np.uint32)[0])
            elif vt == GGUFValueType.FLOAT32:
                meta[f.name] = float(np.frombuffer(raw, dtype=np.float32)[0])
            elif vt == GGUFValueType.INT32:
                meta[f.name] = int(np.frombuffer(raw, dtype=np.int32)[0])
            elif vt == GGUFValueType.FLOAT64:
                meta[f.name] = float(np.frombuffer(raw, dtype=np.float64)[0])

        # Detect architecture — try quaternary_nn keys first, then standard llama keys
        d       = int(meta.get('quaternary_nn.embedding_length',
                       meta.get('llama.embedding_length',
                       meta.get('bert.embedding_length',
                       meta.get('general.embedding_length', 128)))))
        n_layer = int(meta.get('quaternary_nn.block_count',
                       meta.get('llama.block_count',
                       meta.get('bert.block_count', 4))))
        n_head  = int(meta.get('quaternary_nn.attention.head_count',
                       meta.get('llama.attention.head_count',
                       meta.get('bert.attention.head_count', 8))))
        d_ff    = int(meta.get('quaternary_nn.feed_forward_length',
                       meta.get('llama.feed_forward_length',
                       meta.get('bert.feed_forward_length', d * 4))))
        vocab   = int(meta.get('quaternary_nn.vocab_size',
                       meta.get('llama.vocab_size',
                       meta.get('bert.vocab_size', 256))))

        n_kv_head = int(meta.get('quaternary_nn.attention.head_count_kv',
                        meta.get('llama.attention.head_count_kv', n_head)))
        d_head  = int(meta.get('quaternary_nn.attention.key_length',
                       meta.get('llama.attention.key_length',
                       meta.get('bert.attention.key_length', d // n_head))))

        print(f"  [LOAD] Detected architecture: d={d}, layers={n_layer}, heads={n_head}, kv_heads={n_kv_head}, d_ff={d_ff}, vocab={vocab}")

        # Default dropout=0 / growth_count=0 for backward compat with old checkpoints
        dropout = meta.get('quaternary_nn.dropout', 0.0)
        growth_count = int(meta.get('quaternary_nn.growth_count', 0))
        model = cls(vocab, d, n_head, n_layer, d_ff, MAX_SEQ, n_kv_head, growth_count=growth_count, dropout=dropout)

        # Fall back to CPU if model is too large for GPU
        param_count = sum(p.numel() for p in model.parameters())
        if 'cuda' in str(device):
            free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            # fp32 weights + optimizer states (2x) + activations (~1x) = 4x
            est_mem = param_count * 4 * 4
            print(f"  [LOAD] Model: {param_count/1e6:.0f}M params, est ~{est_mem//(1024**3)} GiB needed, ~{free//(1024**3)} GiB free")
            if est_mem > free * 0.8:
                print(f"  [LOAD] Model too large for GPU, falling back to CPU")
                device = torch.device("cpu")

        model = model.to(device)
        tensors = {t.name: t for t in r.tensors}

        # Build mapping from gguf tensor names to quaternary param names
        # Standard llama.cpp names: blk.{i}.{attn_norm,ffn_norm,attn_{q,k,v,output},ffn_{gate,down,up}}
        gguf_names = list(tensors.keys())
        print(f"  [LOAD] Found {len(gguf_names)} tensors: {gguf_names[0] if gguf_names else 'none'}...")

        def dequantize(t):
            if t.tensor_type == GGMLQuantizationType.F32:
                return np.array(t.data, dtype=np.float32)
            from gguf.quants import Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q6_K, Q4_K, Q5_K, Q2_K, Q3_K
            deq_map = {
                GGMLQuantizationType.Q4_0: Q4_0, GGMLQuantizationType.Q4_1: Q4_1,
                GGMLQuantizationType.Q5_0: Q5_0, GGMLQuantizationType.Q5_1: Q5_1,
                GGMLQuantizationType.Q8_0: Q8_0,
                GGMLQuantizationType.Q6_K: Q6_K, GGMLQuantizationType.Q4_K: Q4_K,
                GGMLQuantizationType.Q5_K: Q5_K, GGMLQuantizationType.Q2_K: Q2_K,
                GGMLQuantizationType.Q3_K: Q3_K,
                GGMLQuantizationType.Q2_Q: Q2_Q,
            }
            deq = deq_map.get(t.tensor_type)
            if deq:
                return deq.dequantize(np.array(t.data))
            print(f"  [WARN] No dequantize for {t.name} type {t.tensor_type}, skipping")
            return None

        def try_load(param_name, expected_shape, gguf_candidates):
            for gname in gguf_candidates:
                t = tensors.get(gname)
                if t is None:
                    continue
                data = dequantize(t)
                if data is None:
                    continue
                # Standard llama gguf stores weight.T, so we may need to transpose
                if data.shape == expected_shape:
                    pass
                elif data.shape == expected_shape[::-1]:
                    data = data.T
                elif len(data.shape) == 1 and len(expected_shape) == 1:
                    pass
                else:
                    print(f"  [WARN] Shape mismatch {gname}: got {data.shape}, expected {expected_shape}, skipping")
                    continue
                param = dict(model.named_parameters()).get(param_name)
                if param is None:
                    continue
                noise = torch.randn_like(param) * 0.001 if 'attn.' in param_name and 'norm' not in param_name else 0
                param.data.copy_(torch.from_numpy(data.astype(np.float32).reshape(param.shape)).to(device) + noise)
                return True
            return False

        print(f"[LOAD] Loading model from {path}")

        # Token embeddings
        try_load('tok_embd.weight', (vocab, d),
                 ['token_embd.weight', 'bert.embeddings.word_embeddings.weight'])

        # Output norm
        try_load('output_norm.weight', (d,),
                 ['output_norm.weight', 'bert.encoder.layer.0.output.LayerNorm.weight',
                  'model.norm.weight'])

        # Output projection (lm_head)
        try_load('output.weight', (vocab, d),
                 ['output.weight', 'lm_head.weight', 'bert.pooler.dense.weight'])

        for i in range(n_layer):
            prefix = f'blk.{i}.'
            # Norms
            try_load(f'layers.{i}.attn_norm.weight', (d,),
                     [f'{prefix}attn_norm.weight', f'model.layers.{i}.input_layernorm.weight'])
            try_load(f'layers.{i}.ffn_norm.weight', (d,),
                     [f'{prefix}ffn_norm.weight', f'model.layers.{i}.post_attention_layernorm.weight'])

            # Attention Q/K/V/O — try q2q first, fallback to f32
            for gate, tname in [('q', 'attn_q'), ('k', 'attn_k'), ('v', 'attn_v'), ('o', 'attn_output')]:
                param_name = f'layers.{i}.attn.{gate}.weight'
                # Q2_Q format
                q2q_t = tensors.get(f'{prefix}{tname}.weight')
                if q2q_t is not None and q2q_t.tensor_type == GGMLQuantizationType.Q2_Q:
                    data = Q2_Q.dequantize(np.array(q2q_t.data))
                    data = data.reshape(q2q_t.shape[1], q2q_t.shape[0]).T
                    param = dict(model.named_parameters())[param_name]
                    expected_cols = d if gate == 'q' or gate == 'o' else n_kv_head * d_head
                    expected_shape = (d, expected_cols)
                    # QuaternaryLinear stores weight as (out_features, in_features) = (cols, d)
                    param_expected = (expected_cols, d)
                    if data.shape != expected_shape:
                        print(f"  [WARN] Shape mismatch Q2_Q {param_name}: data {data.shape} != expected {expected_shape}, skipping")
                        continue
                    if param.shape != param_expected:
                        print(f"  [WARN] Param shape mismatch {param_name}: param {param.shape} != expected {param_expected}, skipping")
                        continue
                    noise = torch.randn_like(param) * 0.001
                    param.data.copy_(torch.from_numpy(data.astype(np.float32)).to(device).T + noise)
                    continue
                # F32 format (standard llama)
                exp_cols = d if gate in ('q', 'o') else n_kv_head * d_head
                try_load(param_name, (d, exp_cols),
                         [f'{prefix}{tname}.weight',
                          f'model.layers.{i}.self_attn.{gate}_proj.weight'])

            # FFN gate/up/down
            try_load(f'layers.{i}.ffn.gate.weight', (d_ff, d),
                     [f'{prefix}ffn_gate.weight', f'model.layers.{i}.mlp.gate_proj.weight'])
            try_load(f'layers.{i}.ffn.down.weight', (d, d_ff),
                     [f'{prefix}ffn_down.weight', f'model.layers.{i}.mlp.down_proj.weight'])
            try_load(f'layers.{i}.ffn.up.weight', (d_ff, d),
                     [f'{prefix}ffn_up.weight', f'model.layers.{i}.mlp.up_proj.weight'])

        n_loaded = sum(p.numel() for p in model.parameters())
        print(f"[LOAD] Loaded {n_loaded:,} params")
        return model

    def generate_sample(self, prompt, max_tokens=30, temperature=0.5):
        dev = next(self.parameters()).device
        if self.vocab_size <= 256:
            data_t = torch.tensor([[b for b in prompt.encode('utf-8', errors='replace')[:self.max_seq]]],
                                  dtype=torch.long, device=dev)
            data_t = torch.clamp(data_t, 0, self.vocab_size - 1)
            output = []
            for _ in range(max_tokens):
                ctx = data_t[:, -self.max_seq:] if data_t.size(1) > self.max_seq else data_t
                with torch.no_grad():
                    logits = self.forward(ctx)[0, -1]
                    if output:
                        logits[output[-1]] -= 0.5
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                output.append(next_token)
                data_t = torch.cat([data_t, torch.tensor([[next_token]], device=dev)], dim=1)
            return bytes(output).decode('utf-8', errors='replace')
        else:
            return f"[generated {max_tokens} tokens on {self.vocab_size}-vocab model]"


# ── Data Loading ──────────────────────────────────────────────────
def load_all_text():
    text = ""
    q_root = os.path.join(LIBRARY_DIR, "questions")
    a_root = os.path.join(LIBRARY_DIR, "answers")
    for dirpath, _, filenames in os.walk(q_root):
        rel = os.path.relpath(dirpath, q_root)
        a_dir = os.path.join(a_root, rel)
        for f in sorted(filenames):
            if not f.endswith(".txt"):
                continue
            with open(os.path.join(dirpath, f)) as fh:
                q = fh.read().strip()
            af = os.path.join(a_dir, f)
            if os.path.exists(af):
                with open(af) as fh:
                    a = fh.read().strip()
                text += "<|Q|>" + q + "<|A|>" + a + "<|END|>\n"
    if len(text) < 1000:
        text += "The quick brown fox jumps over the lazy dog. " * 100
    return text


# ── Verification ──────────────────────────────────────────────────
def verify_with_llama(path, prompt="Hello"):
    """Load model and check it loads + runs."""
    binary = "./llama.cpp/build/bin/llama-simple"
    if not os.path.exists(binary):
        subprocess.run(["make", "-C", "llama.cpp/build", "llama-simple"], capture_output=True)
    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "./llama.cpp/build/bin:" + env.get("LD_LIBRARY_PATH", "")
        result = subprocess.run([binary, "-m", path, "--no-warmup", "-p", "a", "-n", "1",
                                 "-ngl", "0"],
                                capture_output=True, timeout=15, env=env)
        if result.returncode == 0:
            print(f"[VERIFY] llama.cpp runs OK")
        elif b"CUDA error" in result.stderr or b"out of memory" in result.stderr:
            print(f"[VERIFY] CUDA OOM (GPU busy), model file valid")
        else:
            stderr_str = result.stderr.decode('utf-8', errors='replace')
            print(f"[VERIFY] llama.cpp exit={result.returncode}: {stderr_str[-200:]}")
    except subprocess.TimeoutExpired:
        print(f"[VERIFY] timed out")
    except Exception as e:
        print(f"[VERIFY] FAILED: {e}")


# ── Library reload (data populated by library_populator.py) ───────
DATA_RELOAD_EVERY = 3000





# ── Signal Handler ────────────────────────────────────────────────
def signal_handler(sig, frame):
    global _model
    print(f"\n[STOP] Saving model...")
    if _model:
        _model.export_gguf(LLAMA_MODEL)
        verify_with_llama(LLAMA_MODEL)
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _device.type == "cuda":
        # Reserve 95% of available memory (Ollama stopped, no other GPU consumers)
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        fraction = min(0.95, free * 0.95 / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction)
        print(f"[INIT] CUDA memory budget: {fraction*100:.0f}% ({int(fraction * torch.cuda.get_device_properties(0).total_memory / 1e9)} GiB)")
    print(f"[INIT] Device: {_device}")

    _all_text = load_all_text()
    data = torch.tensor([b for b in _all_text.encode('utf-8', errors='replace')], dtype=torch.long)

    # Hold out 10% for validation
    val_size = max(1, len(data) // 10)
    train_data = data[:-val_size]
    val_data = data[-val_size:]
    print(f"[INIT] Data: {len(train_data)} train + {len(val_data)} val bytes")

    # Resume from existing GGUF if available
    if os.path.exists(LLAMA_MODEL):
        _model = QuaternaryLLM.from_gguf(LLAMA_MODEL, _device)
        _device = next(_model.parameters()).device
        n_params = sum(p.numel() for p in _model.parameters())
        print(f"[INIT] Resumed model: {n_params:,} params on {_device}")
    else:
        _model = QuaternaryLLM(VOCAB, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ).to(_device)
        n_params = sum(p.numel() for p in _model.parameters())
        print(f"[INIT] New model: {n_params:,} params, {D_MODEL}d, {N_LAYERS}L, {N_HEADS}H")
    print(f"[INIT] Q2_Q attention: {N_LAYERS} layers x 4 matrices = {N_LAYERS*4} quantized tensors")

    opt = torch.optim.AdamW(_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    step = 0
    # LR scheduler: halve LR when Q2_Q convergence plateaus
    best_q_ratio = 0.0
    q_ratio_steps = 0
    best_loss = float('inf')
    loss_plateau_steps = 0

    # Initialize log file (rewrite header if old format detected)
    log_header = "step,loss,val_loss,lr,q2q_pct,data_bytes,elapsed_sec,d_model,n_layers\n"
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w") as f: f.write(log_header)
    else:
        with open(LOG_FILE) as f:
            existing_header = f.readline().strip()
        if existing_header != log_header.strip():
            os.rename(LOG_FILE, LOG_FILE + ".bak")
            with open(LOG_FILE, "w") as f: f.write(log_header)
            print(f"[INIT] Backup old log → {LOG_FILE}.bak, fresh header written")
    start_time = time.time()

    print("─── Training ───")

    try:
        while True:
            # Sample random non-overlapping batches via shuffled permutation
            n = len(train_data) - MAX_SEQ - 1
            perm = torch.randperm(n, device='cpu')[:BATCH]
            x = torch.stack([train_data[i:i + MAX_SEQ] for i in perm]).to(_device)
            y = torch.stack([train_data[i + 1:i + MAX_SEQ + 1] for i in perm]).to(_device)

            logits = _model(x)
            loss = F.cross_entropy(logits.view(-1, _model.vocab_size), y.view(-1))

            # Q2_Q quantization regularization — pull attention weights toward target values
            q2q_loss = 0.0
            for name, p in _model.named_parameters():
                if 'attn' in name and 'weight' in name:
                    for val in (1.0, 0.5, -0.5, -1.0):
                        mask = torch.abs(p - val) < 0.4
                        if mask.any():
                            q2q_loss = q2q_loss + F.mse_loss(p[mask], torch.full_like(p[mask], val))
            loss = loss + q2q_loss * 0.5

            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            # Periodic save
            if step % SAVE_INTERVAL == 0:
                # Count attention weights that have converged to Q2_Q values
                # NOTE: metric is unreliable below d=64 (too few weights for statistics)
                attn_counts = 0
                attn_total = 0
                for name, p in _model.named_parameters():
                    if 'attn' in name and 'weight' in name:
                        w = p.detach()
                        n = w.numel()
                        c = ((torch.abs(w - 1.0) < 0.2) | (torch.abs(w - 0.5) < 0.15) |
                             (torch.abs(w + 0.5) < 0.15) | (torch.abs(w + 1.0) < 0.2)).sum().item()
                        attn_counts += c; attn_total += n
                q_ratio = attn_counts / max(attn_total, 1) * 100

                # LR scheduling: if Q2_Q convergence stalls, halve LR
                if q_ratio > best_q_ratio + 1.0:
                    best_q_ratio = q_ratio
                    q_ratio_steps = 0
                else:
                    q_ratio_steps += SAVE_INTERVAL
                # Loss plateau tracking (independent of Q2_Q)
                if loss.item() < best_loss - 0.05:
                    best_loss = min(best_loss, loss.item())
                    loss_plateau_steps = 0
                else:
                    loss_plateau_steps += SAVE_INTERVAL
                if q_ratio_steps > 0 and q_ratio_steps % 2000 == 0:
                    old_lr = opt.param_groups[0]['lr']
                    should_grow = False
                    # Don't trust Q2_Q convergence for growth signal until d >= 64
                    if _model.d_model >= 64 and q_ratio > 95 and (old_lr < 1e-8 or q_ratio > 98):
                        should_grow = True  # Q2_Q converged
                    elif loss_plateau_steps >= 10000 and loss.item() < 5.0:
                        should_grow = True  # Loss plateaued for 10K steps
                    if should_grow:
                        _model = _model.grow()
                        opt = torch.optim.AdamW(_model.parameters(), lr=LR, weight_decay=0.01)
                        best_q_ratio = 0.0
                        q_ratio_steps = 0
                        print(f"  [GROW] Reset optimizer, LR={LR:.1e}")
                    elif old_lr < 1e-8:
                        for pg in opt.param_groups:
                            pg['lr'] = 5e-4
                        print(f"  [LR] Reset: {old_lr:.2e} → {pg['lr']:.2e}")
                        q_ratio_steps = 0
                    else:
                        for pg in opt.param_groups:
                            pg['lr'] = old_lr / 2
                        print(f"  [LR] Halved: {old_lr:.2e} → {pg['lr']:.2e}")
                        q_ratio_steps = 0

                # Validation loss (on first BATCH of val_data)
                with torch.no_grad():
                    vn = len(val_data) - MAX_SEQ - 1
                    if vn > 0:
                        vp = torch.randperm(vn, device='cpu')[:min(BATCH, vn)]
                        vx = torch.stack([val_data[i:i + MAX_SEQ] for i in vp]).to(_device)
                        vy = torch.stack([val_data[i + 1:i + MAX_SEQ + 1] for i in vp]).to(_device)
                        v_logits = _model(vx)
                        val_loss = F.cross_entropy(v_logits.view(-1, _model.vocab_size), vy.view(-1)).item()
                    else:
                        val_loss = 0.0

                elapsed = time.time() - start_time
                sample = _model.generate_sample("<|Q|>", 30)
                print(f"  step {step:5d} | LR {opt.param_groups[0]['lr']:.1e} | loss {loss.item():.4f} val {val_loss:.4f} | Q2_Q {q_ratio:.0f}% | gen: {sample[:40]}")
                with open(LOG_FILE, "a") as f:
                    f.write(f"{step},{loss.item():.6f},{val_loss:.6f},{opt.param_groups[0]['lr']:.1e},{q_ratio:.1f},{len(train_data)},{elapsed:.1f},{_model.d_model},{_model.n_layers}\n")

                _model.export_gguf(LLAMA_MODEL)

                if step % VERIFY_EVERY == 0:
                    verify_with_llama(LLAMA_MODEL)

                if step > 0 and step % DATA_RELOAD_EVERY == 0:
                    _all_text = load_all_text()
                    old_len = len(train_data) + len(val_data)
                    new_data = torch.tensor([b for b in _all_text.encode('utf-8', errors='replace')], dtype=torch.long)
                    if len(new_data) != old_len:
                        val_size = max(1, len(new_data) // 10)
                        train_data = new_data[:-val_size]
                        val_data = new_data[-val_size:]
                        print(f"  [DATA] Reloaded: {len(train_data) + len(val_data)} bytes ({len(new_data) - old_len:+d})")

    except KeyboardInterrupt:
        pass
    finally:
        if _model:
            _model.export_gguf(LLAMA_MODEL)
            verify_with_llama(LLAMA_MODEL)
        print("[DONE]")
