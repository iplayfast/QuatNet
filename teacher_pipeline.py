#!/usr/bin/env python3
"""
teacher_pipeline.py - Quaternary Transformer Training Pipeline
Trains a tiny Q2_Q transformer on Q&A library text, exports GGUF, verifies with llama.cpp.
"""
import os, sys, json, time, signal, random, math, subprocess, urllib.request, urllib.error
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
D_MODEL  = 128
N_HEADS  = 8
N_LAYERS = 4
D_FF     = 256
MAX_SEQ  = 128
BATCH    = 128
LR       = 5e-4
VOCAB    = 256

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
    def __init__(self, d, n_heads, d_ff, n_kv_heads=None):
        super().__init__()
        self.attn_norm = RMSNorm(d)
        self.attn = QuaternaryAttention(d, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d)
        self.ffn = FeedForward(d, d_ff)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class QuaternaryLLM(nn.Module):
    def __init__(self, vocab, d, n_heads, n_layers, d_ff, max_seq, n_kv_heads=None):
        super().__init__()
        self.vocab_size, self.d_model = vocab, d
        self.n_heads, self.n_layers, self.max_seq = n_heads, n_layers, max_seq
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.tok_embd = nn.Embedding(vocab, d)
        self.pos_embd = nn.Embedding(max_seq, d)
        self.layers = nn.ModuleList([QuaternaryBlock(d, n_heads, d_ff, self.n_kv_heads) for _ in range(n_layers)])
        self.output_norm = RMSNorm(d)
        self.output = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_embd(idx) + self.pos_embd(pos)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.output_norm(x))

    def export_gguf(self, path):
        from gguf import GGUFWriter, GGMLQuantizationType
        from gguf.quants import Q2_Q

        d, vocab = self.d_model, self.vocab_size
        n_layer, n_head, d_ff = self.n_layers, self.n_heads, self.layers[0].ffn.gate.out_features
        d_head = d // n_head

        def q2q(w):
            arr = w.astype(np.float32).reshape(-1)
            pad = (32 - len(arr) % 32) % 32
            if pad: arr = np.pad(arr, (0, pad))
            return Q2_Q.quantize_blocks(arr.reshape(len(arr) // 32, 32))

        w = GGUFWriter(path, "quaternary_nn")
        w.add_uint32("quaternary_nn.context_length", 256)
        w.add_uint32("quaternary_nn.embedding_length", d)
        w.add_uint32("quaternary_nn.block_count", n_layer)
        w.add_uint32("quaternary_nn.feed_forward_length", d_ff)
        w.add_uint32("quaternary_nn.attention.head_count", n_head)
        w.add_uint32("quaternary_nn.attention.head_count_kv", n_head)
        w.add_uint32("quaternary_nn.attention.key_length", d_head)
        w.add_uint32("quaternary_nn.attention.value_length", d_head)
        w.add_uint32("quaternary_nn.vocab_size", vocab)
        w.add_float32("quaternary_nn.attention.layer_norm_rms_epsilon", 1e-5)
        w.add_uint32("quaternary_nn.full_attention_interval", 1)
        w.add_array("quaternary_nn.rope.dimension_sections", [d_head, 0, 0, 0])
        w.add_uint32("quaternary_nn.rope.dimension_count", d_head)
        w.add_string("tokenizer.ggml.model", "byte")
        w.add_uint32("tokenizer.ggml.tokens", vocab)
        w.add_token_list([bytes([i]) for i in range(vocab)])
        w.add_string("tokenizer.ggml.pre", "byte")

        # Store all tensors as F32 for llama.cpp compatibility (Q2_Q GGUF type 99
        # lacks full backend support for KV cache and get_rows operations)
        w.add_tensor("token_embd.weight", self.tok_embd.weight.detach().cpu().numpy().astype(np.float32))
        w.add_tensor("output_norm.weight", self.output_norm.weight.detach().cpu().numpy().astype(np.float32))
        w.add_tensor("output.weight", self.output.weight.detach().cpu().numpy().astype(np.float32))

        for i, layer in enumerate(self.layers):
            w.add_tensor(f"blk.{i}.attn_norm.weight", layer.attn_norm.weight.detach().cpu().numpy().astype(np.float32))
            w.add_tensor(f"blk.{i}.attn_q.weight", layer.attn.q.get_quantized_weight().T.astype(np.float32))
            w.add_tensor(f"blk.{i}.attn_k.weight", layer.attn.k.get_quantized_weight().T.astype(np.float32))
            w.add_tensor(f"blk.{i}.attn_v.weight", layer.attn.v.get_quantized_weight().T.astype(np.float32))
            w.add_tensor(f"blk.{i}.attn_output.weight", layer.attn.o.get_quantized_weight().T.astype(np.float32))
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
                meta[f.name] = raw.decode()
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

        model = cls(vocab, d, n_head, n_layer, d_ff, MAX_SEQ, n_kv_head)

        # Fall back to CPU if model is too large for GPU
        param_count = sum(p.numel() for p in model.parameters())
        if 'cuda' in str(device):
            free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            # fp32 weights + optimizer states (2x) + activations (~1x) = 4x
            est_mem = param_count * 4 * 4
            print(f"  [LOAD] Model: {param_count/1e6:.0f}M params, est ~{est_mem//(1024**3)} GiB needed, ~{free//(1024**3)} GiB free")
            if param_count > 100_000_000:
                print(f"  [LOAD] Large model detected, forcing CPU to avoid OOM")
                device = torch.device("cpu")
            elif est_mem > free * 0.8:
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

    def generate_sample(self, prompt, max_tokens=30, temperature=0.8):
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
                        logits[output[-1]] -= 2.0
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
    q_dir = os.path.join(LIBRARY_DIR, "questions")
    a_dir = os.path.join(LIBRARY_DIR, "answers")
    if os.path.exists(q_dir):
        for f in sorted(os.listdir(q_dir)):
            if f.endswith(".txt"):
                with open(os.path.join(q_dir, f)) as fh:
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
    """Load model in llama-cli and check it loads + runs."""
    binary = "./llama.cpp/build/bin/llama-simple"
    if not os.path.exists(binary):
        subprocess.run(["make", "-C", "llama.cpp/build", "llama-simple"], capture_output=True)
    try:
        result = subprocess.run([binary, "-m", path, "--no-warmup", "-p", "a", "-n", "1",
                                 "-ngl", "0"],
                                capture_output=True, timeout=15)
        if result.returncode == 0:
            print(f"[VERIFY] llama.cpp runs OK")
        else:
            stderr_str = result.stderr.decode('utf-8', errors='replace')
            print(f"[VERIFY] llama.cpp exit={result.returncode}: {stderr_str[-200:]}")
    except subprocess.TimeoutExpired:
        print(f"[VERIFY] timed out")
    except Exception as e:
        print(f"[VERIFY] FAILED: {e}")


# ── Ollama Q&A Generator ──────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
GENERATE_EVERY = 1000  # generate new Q&A every N steps

def generate_qa():
    """Query Ollama to answer a random question from programmingquestions.txt."""
    # Load questions
    try:
        with open(QUESTIONS_FILE) as f:
            raw = f.read()
        questions = [q.strip() for q in raw.strip().split("\n") if q.strip() and q.strip() != "Advanced"]
    except:
        questions = ["Write a Python function."]
    question = random.choice(questions)

    # Pick a random Ollama model for variety
    try:
        with open("servers.json") as f:
            servers = json.load(f)
        models = [s["model"] for s in servers if s.get("enabled", False) and s.get("role", "worker") == "worker"]
    except:
        models = [OLLAMA_MODEL]
    model = random.choice(models)

    prompt = (f"Write a complete working Python solution for this problem. "
              f"Show only the code, no explanation.\n\nProblem: {question}")
    try:
        req = urllib.request.Request(OLLAMA_URL, data=json.dumps({
            "model": model, "prompt": prompt, "stream": False,
        }).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            answer = json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        print(f"  [OLLAMA] Failed ({model}): {e}")
        return None

    if not answer:
        return None

    # Replace an existing library file with the Ollama answer
    q_dir = os.path.join(LIBRARY_DIR, "questions")
    a_dir = os.path.join(LIBRARY_DIR, "answers")
    os.makedirs(q_dir, exist_ok=True)
    os.makedirs(a_dir, exist_ok=True)

    existing = sorted([f for f in os.listdir(q_dir) if f.endswith(".txt")])
    if existing:
        n = random.choice(existing).replace(".txt", "")
    else:
        n = str(len(os.listdir(q_dir)) + 1)

    with open(os.path.join(q_dir, f"{n}.txt"), "w") as f:
        f.write(question + "\n")
    with open(os.path.join(a_dir, f"{n}.txt"), "w") as f:
        f.write(answer + "\n")
    print(f"  [OLLAMA] Generated Q&A #{n}: {question[:60]}...")

    # Also copy a random "best" Q&A pair into the main library
    best_q_dir = os.path.join(LIBRARY_DIR, "questions", "bestquestions")
    best_a_dir = os.path.join(LIBRARY_DIR, "answers", "bestanswers")
    best_files = sorted([f for f in os.listdir(best_q_dir) if f.endswith(".txt")])
    if best_files:
        bf = random.choice(best_files)
        bq_path = os.path.join(best_q_dir, bf)
        ba_path = os.path.join(best_a_dir, bf)
        if os.path.exists(bq_path) and os.path.exists(ba_path):
            dst_n = random.choice(existing).replace(".txt", "") if existing else str(len(os.listdir(q_dir)) + 1)
            with open(bq_path) as f:
                best_q = f.read()
            with open(ba_path) as f:
                best_a = f.read()
            with open(os.path.join(q_dir, f"{dst_n}.txt"), "w") as f:
                f.write(best_q)
            with open(os.path.join(a_dir, f"{dst_n}.txt"), "w") as f:
                f.write(best_a)
            print(f"  [BEST]  Seeded Q&A #{dst_n}: {best_q[:60].strip()}...")

    return True


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
    print(f"[INIT] Device: {_device}")

    _all_text = load_all_text()
    data = torch.tensor([b for b in _all_text.encode('utf-8', errors='replace')], dtype=torch.long)

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

    opt = torch.optim.AdamW(_model.parameters(), lr=LR)
    step = 0
    # LR scheduler: halve LR when Q2_Q convergence plateaus
    best_q_ratio = 0.0
    q_ratio_steps = 0

    # Initialize log file
    log_header = "step,loss,lr,q2q_pct,data_bytes,elapsed_sec\n"
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w") as f: f.write(log_header)
    start_time = time.time()

    print("─── Training ───")

    try:
        while True:
            # Sample random batches
            idx = torch.randint(0, len(data) - MAX_SEQ - 1, (BATCH,))
            x = torch.stack([data[i:i + MAX_SEQ] for i in idx]).to(_device)
            y = torch.stack([data[i + 1:i + MAX_SEQ + 1] for i in idx]).to(_device)

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
                if q_ratio_steps > 0 and q_ratio_steps % 2000 == 0:
                    for pg in opt.param_groups:
                        old_lr = pg['lr']
                        if old_lr < 1e-8:
                            pg['lr'] = 5e-4
                            print(f"  [LR] Reset: {old_lr:.2e} → {pg['lr']:.2e}")
                        else:
                            pg['lr'] = old_lr / 2
                            print(f"  [LR] Halved: {old_lr:.2e} → {pg['lr']:.2e}")
                    q_ratio_steps = 0

                elapsed = time.time() - start_time
                sample = _model.generate_sample("<|Q|>", 30)
                print(f"  step {step:5d} | LR {opt.param_groups[0]['lr']:.1e} | loss {loss.item():.4f} | Q2_Q converged: {q_ratio:.0f}% | gen: {sample[:40]}")
                with open(LOG_FILE, "a") as f:
                    f.write(f"{step},{loss.item():.6f},{opt.param_groups[0]['lr']:.1e},{q_ratio:.1f},{len(data)},{elapsed:.1f}\n")

                _model.export_gguf(LLAMA_MODEL)

                if step % VERIFY_EVERY == 0:
                    verify_with_llama(LLAMA_MODEL)

                if step > 0 and step % GENERATE_EVERY == 0:
                    if generate_qa():
                        _all_text = load_all_text()
                        data = torch.tensor([b for b in _all_text.encode('utf-8', errors='replace')], dtype=torch.long)
                        print(f"  [DATA] Reloaded: {len(data)} bytes")

    except KeyboardInterrupt:
        pass
    finally:
        if _model:
            _model.export_gguf(LLAMA_MODEL)
            verify_with_llama(LLAMA_MODEL)
        print("[DONE]")
