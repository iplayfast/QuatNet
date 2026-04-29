#include "models.h"
#include "llama-model.h"
#include "llama-graph.h"

static ggml_tensor * ensure_f32(ggml_context * ctx, ggml_tensor * t) {
    if (t && t->type == GGML_TYPE_Q2_Q)
        return ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

llm_build_quaternary_nn::llm_build_quaternary_nn(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {

    const auto & layer0 = model.layers[0];
    bool has_attention = (layer0.wq && layer0.wq->ne[1] > 1) || layer0.wqkv != nullptr;

    auto * inpL = build_inp_embd(model.tok_embd);
    auto * inp_out_ids = build_inp_out_ids();

    if (has_attention) {
        auto * inp_attn = build_attn_inp_kv();
        const int64_t n_embd_head = hparams.n_embd_head_v(0);

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];
            const int64_t n_head_i    = hparams.n_head(il);
            const int64_t n_head_kv_i = hparams.n_head_kv(il);

            auto cur_layer = layer;
            cur_layer.wq = ensure_f32(ctx0, cur_layer.wq);
            cur_layer.wk = ensure_f32(ctx0, cur_layer.wk);
            cur_layer.wv = ensure_f32(ctx0, cur_layer.wv);
            cur_layer.wo = ensure_f32(ctx0, cur_layer.wo);

            ggml_tensor * cur = inpL;

            if (layer.attn_norm) {
                cur = build_norm(cur, layer.attn_norm, layer.attn_norm_b, LLM_NORM_RMS, il);
                cb(cur, "attn_norm", il);
            }

            if (cur_layer.wqkv) {
                auto [Qcur, Kcur, Vcur] = build_qkv(cur_layer, cur, n_embd_head, n_head_i, n_head_kv_i, il);
                cur = build_attn(inp_attn, cur_layer.wo, cur_layer.wo_b, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                        1.0f / sqrtf((float) n_embd_head), il);
                cb(cur, "attn_out", il);
            } else if (cur_layer.wq && cur_layer.wo) {
                auto [Qcur, Kcur, Vcur] = build_qkv(cur_layer, cur, n_embd_head, n_head_i, n_head_kv_i, il);
                cur = build_attn(inp_attn, cur_layer.wo, cur_layer.wo_b, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                        1.0f / sqrtf((float) n_embd_head), il);
                cb(cur, "attn_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            if (layer.ffn_norm) {
                cur = build_norm(ffn_inp, layer.ffn_norm, layer.ffn_norm_b, LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);
                if (layer.ffn_gate && layer.ffn_down && layer.ffn_up) {
                    cur = build_ffn(cur,
                            layer.ffn_up,   nullptr, nullptr,
                            layer.ffn_gate, nullptr, nullptr,
                            layer.ffn_down, nullptr, nullptr,
                            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(cur, "ffn_out", il);
                    cur = ggml_add(ctx0, cur, ffn_inp);
                }
            }

            cb(cur, "l_out", il);
            inpL = cur;
        }
    } else {
        ggml_tensor * cur = inpL;
        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];
            if (!layer.wq) continue;
            ggml_tensor * head_sum = ggml_sum(ctx0, layer.wq);
            cb(head_sum, "head_sum", il);
            cur = ggml_add(ctx0, cur, head_sum);
        }
        if (inp_out_ids) cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        inpL = cur;
    }

    ggml_tensor * cur = inpL;
    if (model.output_norm) {
        cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
    }
    res->t_embd = cur;
    if (model.output) {
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);
    }
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
