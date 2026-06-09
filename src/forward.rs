// ============================================================
// forward.rs — the transformer forward pass
// ============================================================
// This is the heart of inference.
// Given ONE token and its position, produces logits over the vocab.
// Called once per generated token.
// ============================================================

use crate::math::{rmsnorm, softmax, matmul, accum, copy_into};
use crate::model::{Config, Weights};

// ── RunState ─────────────────────────────────────────────────
// All the temporary buffers needed during the forward pass.
// Allocated ONCE, reused on every forward call.
pub struct RunState {
    pub x:           Vec<f32>,  // current token embedding + residual stream [dim]
    pub xb:          Vec<f32>,  // scratch buffer inside residual branch      [dim]
    pub xb2:         Vec<f32>,  // second scratch buffer                      [dim]
    pub hb:          Vec<f32>,  // ffn hidden buffer                          [hidden_dim]
    pub hb2:         Vec<f32>,  // ffn hidden buffer 2                        [hidden_dim]
    pub q:           Vec<f32>,  // query vector                               [dim]
    pub att:         Vec<f32>,  // attention scores                           [n_heads * seq_len]
    pub logits:      Vec<f32>,  // final output: probability for each token   [vocab_size]
    pub key_cache:   Vec<f32>,  // KV cache keys   [n_layers * seq_len * kv_dim]
    pub value_cache: Vec<f32>,  // KV cache values [n_layers * seq_len * kv_dim]
}

impl RunState {
    pub fn new(cfg: &Config) -> Self {
        let kv_dim = cfg.kv_dim();
        RunState {
            x:           vec![0.0; cfg.dim],
            xb:          vec![0.0; cfg.dim],
            xb2:         vec![0.0; cfg.dim],
            hb:          vec![0.0; cfg.hidden_dim],
            hb2:         vec![0.0; cfg.hidden_dim],
            q:           vec![0.0; cfg.dim],
            att:         vec![0.0; cfg.n_heads * cfg.seq_len],
            logits:      vec![0.0; cfg.vocab_size],
            key_cache:   vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim],
            value_cache: vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim],
        }
    }
}

// ── forward ──────────────────────────────────────────────────
// The full transformer forward pass for ONE token at position pos.
// Returns a slice into state.logits — one float per vocab token.
//
// The caller loops over positions, calling this once per token.
// pos tells the model WHERE in the sequence this token sits.
pub fn forward<'a>(
    cfg:     &Config,
    weights: &Weights,
    state:   &'a mut RunState,
    token:   u32,
    pos:     usize,
) -> &'a [f32] {

    let dim      = cfg.dim;
    let kv_dim   = cfg.kv_dim();
    let kv_mul   = cfg.n_heads / cfg.n_kv_heads; // for grouped query attention
    let hidden   = cfg.hidden_dim;
    let head_sz  = cfg.head_size();
    let n_heads  = cfg.n_heads;
    let n_layers = cfg.n_layers;
    let seq_len  = cfg.seq_len;

    // ── Step 1: embedding lookup ─────────────────────────────
    // Copy the token's embedding row into x.
    let emb_start = weights.token_embedding_offset + token as usize * dim;
    copy_into(&mut state.x, &weights.data[emb_start..emb_start + dim]);

    // ── Step 2: loop over transformer layers ─────────────────
    // Each layer: attention → residual → ffn → residual
    for l in 0..n_layers {

        // ── 2a: attention rmsnorm ─────────────────────────────
        // Normalize x before feeding into attention.
        let rms_att = weights.rms_att_weight(l, dim);
        // need a tmp copy because rmsnorm reads x and writes xb
        let x_copy: Vec<f32> = state.x.clone();
        rmsnorm(&mut state.xb, &x_copy, rms_att);

        // ── 2b: KV cache slot for this layer + position ───────
        // The KV cache stores keys and values for ALL past positions
        // so we don't recompute them every step.
        let loff = l * seq_len * kv_dim;           // layer offset
        let k_start = loff + pos * kv_dim;         // where to write k for this pos
        let v_start = loff + pos * kv_dim;         // where to write v for this pos

        // ── 2c: QKV projections ───────────────────────────────
        // Project xb into query, key, value vectors via learned weight matrices.
        // q has shape [dim], k and v have shape [kv_dim]
        // (kv_dim can be smaller than dim for grouped query attention)
        {
            let xb = state.xb.clone();
            matmul(&mut state.q,
                   &xb, weights.wq(l, dim), dim, dim);
            matmul(&mut state.key_cache[k_start..k_start + kv_dim],
                   &xb, weights.wk(l, dim, kv_dim), dim, kv_dim);
            matmul(&mut state.value_cache[v_start..v_start + kv_dim],
                   &xb, weights.wv(l, dim, kv_dim), dim, kv_dim);
        }

        // ── 2d: RoPE — Rotary Position Encoding ──────────────
        // Rotates q and k vectors to encode position information.
        // Instead of adding a position vector (like original transformer),
        // RoPE ROTATES pairs of dimensions by an angle proportional to pos.
        for i in (0..dim).step_by(2) {
            let head_dim = i % head_sz;
            let freq = 1.0_f32 / 10000_f32.powf(head_dim as f32 / head_sz as f32);
            let val  = pos as f32 * freq;
            let fcr  = val.cos();
            let fci  = val.sin();

            // rotate q always
            let q0 = state.q[i];
            let q1 = state.q[i + 1];
            state.q[i]     = q0 * fcr - q1 * fci;
            state.q[i + 1] = q0 * fci + q1 * fcr;

            // rotate k only for dimensions within kv_dim
            if i < kv_dim {
                let k0 = state.key_cache[k_start + i];
                let k1 = state.key_cache[k_start + i + 1];
                state.key_cache[k_start + i]     = k0 * fcr - k1 * fci;
                state.key_cache[k_start + i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // ── 2e: Multi-head attention ──────────────────────────
        // For each head: compute attention scores against ALL past positions,
        // softmax them, then take weighted sum of value vectors.
        //
        // This is the "memory" mechanism — each head can attend to
        // any previous token. The scores tell it HOW MUCH to attend.
        for h in 0..n_heads {
            let q_head  = &state.q[h * head_sz..(h + 1) * head_sz];
            let att_buf = &mut state.att[h * seq_len..h * seq_len + seq_len];

            // compute attention score for each past position t
            for t in 0..=pos {
                let kv_head = (h / kv_mul) * head_sz; // grouped query: multiple q heads share one k head
                let k_pos   = &state.key_cache[loff + t * kv_dim + kv_head
                                              ..loff + t * kv_dim + kv_head + head_sz];
                // dot product q · k, scaled by sqrt(head_size)
                let score: f32 = q_head.iter().zip(k_pos.iter())
                                        .map(|(a, b)| a * b)
                                        .sum::<f32>()
                                 / (head_sz as f32).sqrt();
                att_buf[t] = score;
            }

            // softmax over positions 0..=pos
            softmax(&mut att_buf[..=pos]);

            // weighted sum of value vectors → write into xb
            let xb_head = &mut state.xb[h * head_sz..(h + 1) * head_sz];
            xb_head.fill(0.0); // zero before accumulating
            // need att_buf values — copy them out to avoid borrow conflict
            let att_scores: Vec<f32> = att_buf[..=pos].to_vec();
            for t in 0..=pos {
                let kv_head = (h / kv_mul) * head_sz;
                let v_pos = &state.value_cache[loff + t * kv_dim + kv_head
                                              ..loff + t * kv_dim + kv_head + head_sz];
                let a = att_scores[t];
                for i in 0..head_sz {
                    xb_head[i] += a * v_pos[i];
                }
            }
        }

        // ── 2f: attention output projection ──────────────────
        // Project concatenated head outputs back to dim.
        {
            let xb = state.xb.clone();
            matmul(&mut state.xb2, &xb, weights.wo(l, dim), dim, dim);
        }

        // ── 2g: residual connection ───────────────────────────
        // Add attention output back to x (the residual stream).
        // This is why transformers can be deep — the original signal
        // is always preserved and just modified by each layer.
        let xb2 = state.xb2.clone();
        accum(&mut state.x, &xb2);

        // ── 2h: ffn rmsnorm ───────────────────────────────────
        // Normalize again before the feed-forward network.
        let rms_ffn = weights.rms_ffn_weight(l, dim);
        let x_copy: Vec<f32> = state.x.clone();
        rmsnorm(&mut state.xb, &x_copy, rms_ffn);

        // ── 2i: feed-forward network (SwiGLU) ─────────────────
        // FFN formula: output = W2 * (SiLU(W1 * x) ⊙ W3 * x)
        // This is the SwiGLU variant used by LLaMA.
        {
            let xb = state.xb.clone();
            matmul(&mut state.hb,  &xb, weights.w1(l, dim, hidden), dim, hidden);
            matmul(&mut state.hb2, &xb, weights.w3(l, dim, hidden), dim, hidden);
        }

        // SwiGLU activation: silu(hb) * hb2
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        for i in 0..hidden {
            let val = state.hb[i];
            let silu = val * (1.0 / (1.0 + (-val).exp())); // silu(x)
            state.hb[i] = silu * state.hb2[i];             // elementwise * hb2
        }

        // project back from hidden_dim to dim
        {
            let hb = state.hb.clone();
            matmul(&mut state.xb, &hb, weights.w2(l, hidden, dim), hidden, dim);
        }

        // ── 2j: second residual connection ────────────────────
        // Add FFN output back to residual stream.
        let xb = state.xb.clone();
        accum(&mut state.x, &xb);
    }

    // ── Step 3: final rmsnorm ─────────────────────────────────
    // Normalize the final residual stream output.
    let rms_final = weights.rms_final_weight(dim).to_vec();
    let x_copy: Vec<f32> = state.x.clone();
    rmsnorm(&mut state.x, &x_copy, &rms_final);

    // ── Step 4: classifier → logits ───────────────────────────
    // Project from dim → vocab_size to get one score per token.
    // The highest score = the most likely next token.
    {
        let x = state.x.clone();
        let wcls = weights.wcls(dim, cfg.vocab_size).to_vec();
        matmul(&mut state.logits, &x, &wcls, dim, cfg.vocab_size);
    }

    &state.logits
}
