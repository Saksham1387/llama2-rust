// Config, Weights, and Embedding lookup

use std::fs::{File};
use std::io::{self, Read, Cursor};

use memmap2::Mmap;

// ── Config ───────────────────────────────────────────────────
// Exactly 7 signed integers — the first 28 bytes of the .bin file.
#[derive(Debug)]
pub struct Config {
    pub dim:        usize,   // embedding dimension          e.g. 288
    pub hidden_dim: usize,   // ffn hidden dimension         e.g. 768
    pub n_layers:   usize,   // number of transformer layers e.g. 6
    pub n_heads:    usize,   // number of attention heads    e.g. 6
    pub n_kv_heads: usize,   // key/value heads              e.g. 6
    pub vocab_size: usize,   // vocabulary size              e.g. 32000
    pub seq_len:    usize,   // max sequence length          e.g. 256
}

impl Config {
    // head_size is derived from dim / n_heads
    pub fn head_size(&self) -> usize { self.dim / self.n_heads }

    // kv_dim is also derived
    pub fn kv_dim(&self) -> usize { (self.dim * self.n_kv_heads) / self.n_heads }
}

// ── Weights ──────────────────────────────────────────────────
// One big Vec<f32> for the whole file, plus usize offsets telling us
// where each matrix starts.
//
// We slice into data[offset..offset+size] whenever we need a matrix.
pub struct Weights {
    pub data: Vec<f32>,                    // entire file as f32s

    // offsets (in number of f32s, not bytes) into data
    pub token_embedding_offset: usize,     // [vocab_size × dim]
    pub rms_att_offset:         usize,     // [n_layers × dim]
    pub wq_offset:              usize,     // [n_layers × dim × dim]
    pub wk_offset:              usize,     // [n_layers × dim × kv_dim]
    pub wv_offset:              usize,     // [n_layers × dim × kv_dim]
    pub wo_offset:              usize,     // [n_layers × dim × dim]
    pub rms_ffn_offset:         usize,     // [n_layers × dim]
    pub w1_offset:              usize,     // [n_layers × hidden_dim × dim]
    pub w2_offset:              usize,     // [n_layers × dim × hidden_dim]
    pub w3_offset:              usize,     // [n_layers × hidden_dim × dim]
    pub rms_final_offset:       usize,     // [dim]
    pub wcls_offset:            usize,     // [vocab_size × dim]
    pub shared_weights:         bool,
}

impl Weights {
    // ── slice helpers ─────────────────────────────────────────
    // Each method returns a &[f32] slice into the big data Vec.

    #[allow(dead_code)]
    pub fn token_embedding(&self, dim: usize, vocab_size: usize) -> &[f32] {
        &self.data[self.token_embedding_offset..self.token_embedding_offset + vocab_size * dim]
    }

    pub fn rms_att_weight(&self, layer: usize, dim: usize) -> &[f32] {
        let start = self.rms_att_offset + layer * dim;
        &self.data[start..start + dim]
    }

    pub fn wq(&self, layer: usize, dim: usize) -> &[f32] {
        let size = dim * dim;
        let start = self.wq_offset + layer * size;
        &self.data[start..start + size]
    }

    pub fn wk(&self, layer: usize, dim: usize, kv_dim: usize) -> &[f32] {
        let size = dim * kv_dim;
        let start = self.wk_offset + layer * size;
        &self.data[start..start + size]
    }

    pub fn wv(&self, layer: usize, dim: usize, kv_dim: usize) -> &[f32] {
        let size = dim * kv_dim;
        let start = self.wv_offset + layer * size;
        &self.data[start..start + size]
    }

    pub fn wo(&self, layer: usize, dim: usize) -> &[f32] {
        let size = dim * dim;
        let start = self.wo_offset + layer * size;
        &self.data[start..start + size]
    }

    pub fn rms_ffn_weight(&self, layer: usize, dim: usize) -> &[f32] {
        let start = self.rms_ffn_offset + layer * dim;
        &self.data[start..start + dim]
    }

    pub fn w1(&self, layer: usize, dim: usize, hidden_dim: usize) -> &[f32] {
        let size = dim * hidden_dim;
        let start = self.w1_offset + layer * size;
        &self.data[start..start + size]
    }

    pub fn w2(&self, layer: usize, hidden_dim: usize, dim: usize) -> &[f32] {
        let size = hidden_dim * dim;
        let start = self.w2_offset + layer * size;
        &self.data[start..start + size]
    }

    pub fn w3(&self, layer: usize, dim: usize, hidden_dim: usize) -> &[f32] {
        let size = dim * hidden_dim;
        let start = self.w3_offset + layer * size;
        &self.data[start..start + size]
    }

    pub fn rms_final_weight(&self, dim: usize) -> &[f32] {
        &self.data[self.rms_final_offset..self.rms_final_offset + dim]
    }

    pub fn wcls(&self, dim: usize, vocab_size: usize) -> &[f32] {
        // if shared_weights → wcls IS token_embedding (same data, same offset)
        let offset = if self.shared_weights {
            self.token_embedding_offset
        } else {
            self.wcls_offset
        };
        &self.data[offset..offset + vocab_size * dim]
    }
}

// ── Transformer ──────────────────────────────────────────────
// Top level struct for the model config and weights.
pub struct Transformer {
    pub config:  Config,
    pub weights: Weights,
}

impl Transformer {
    // ── from_file ─────────────────────────────────────────────
    // Reads a model checkpoint.
    //
    // File layout:
    //   [Config : 7 × i32  =  28 bytes]  ← read first
    //   [Weights: N × f32             ]  ← read the rest
    pub fn from_file(path: &str) -> io::Result<Self> {

        // read the ENTIRE file into memory as raw bytes
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = Cursor::new(&mmap);

        // ── Step 1: read Config ───────────────────────────────
        let dim            = read_i32(&mut cursor)?;
        let hidden_dim     = read_i32(&mut cursor)?;
        let n_layers       = read_i32(&mut cursor)?;
        let n_heads        = read_i32(&mut cursor)?;
        let n_kv_heads     = read_i32(&mut cursor)?;
        let vocab_size_raw = read_i32(&mut cursor)?;  // may be negative!
        let seq_len        = read_i32(&mut cursor)?;

        // ── Step 2: shared weights flag ───────────────────────
        //   positive vocab_size → wcls SHARES weights with token_embedding
        //   negative vocab_size → wcls has its OWN separate weights
        let shared_weights = vocab_size_raw > 0;
        let vocab_size = vocab_size_raw.unsigned_abs() as usize;

        let config = Config {
            dim:        dim        as usize,
            hidden_dim: hidden_dim as usize,
            n_layers:   n_layers   as usize,
            n_heads:    n_heads    as usize,
            n_kv_heads: n_kv_heads as usize,
            vocab_size,
            seq_len:    seq_len    as usize,
        };

        // ── Step 3: convert remaining bytes to f32s ───────────
        // Everything after the 28-byte config header is float32 weights.
        // We convert the raw bytes to f32s: every 4 bytes = one f32.
        let weights_bytes = &mmap[28..]; // skip the 28-byte config header
        let data: Vec<f32> = weights_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // ── Step 4: compute offsets ───────────────────────────
        // The order here must exactly match the checkpoint file format.
        let d    = config.dim;
        let h    = config.hidden_dim;
        let l    = config.n_layers;
        let nh   = config.n_heads;
        let v    = config.vocab_size;
        let s    = config.seq_len;
        let hdsz = d / nh;          // head_size
        let kvd  = config.kv_dim(); // kv_dim

        let mut ptr: usize = 0;

        // each line: save current position as offset, then advance ptr
        let token_embedding_offset = ptr; ptr += v * d;
        let rms_att_offset         = ptr; ptr += l * d;
        let wq_offset              = ptr; ptr += l * d * d;
        let wk_offset              = ptr; ptr += l * d * kvd;
        let wv_offset              = ptr; ptr += l * d * kvd;
        let wo_offset              = ptr; ptr += l * d * d;
        let rms_ffn_offset         = ptr; ptr += l * d;
        let w1_offset              = ptr; ptr += l * d * h;
        let w2_offset              = ptr; ptr += l * h * d;
        let w3_offset              = ptr; ptr += l * d * h;
        let rms_final_offset       = ptr; ptr += d;

        // skip old RoPE frequency buffers (no longer used, kept for file compat)
        ptr += s * hdsz / 2;
        ptr += s * hdsz / 2;

        // wcls: if shared, same offset as token_embedding; else its own block
        let wcls_offset = if shared_weights {
            token_embedding_offset
        } else {
            ptr // uses its own weights right here
        };

        let weights = Weights {
            data,
            token_embedding_offset,
            rms_att_offset,
            wq_offset,
            wk_offset,
            wv_offset,
            wo_offset,
            rms_ffn_offset,
            w1_offset,
            w2_offset,
            w3_offset,
            rms_final_offset,
            wcls_offset,
            shared_weights,
        };

        Ok(Transformer { config, weights })
    }
}

// ── embedding_lookup ─────────────────────────────────────────
// The FIRST operation in the forward pass.
// Token id → row in embedding table → &[f32] of length dim.
// The table is a flat 2D matrix [vocab_size × dim].
// Each token id selects a ROW.
// That row is a vector of `dim` floats = the token's embedding.
//
//   token_id=0  → data[0    .. dim]
//   token_id=1  → data[dim  .. 2*dim]
//   token_id=2  → data[2dim .. 3*dim]
//   token_id=N  → data[N*dim.. N*dim + dim]
#[allow(dead_code)]
pub fn embedding_lookup<'a>(weights: &'a Weights, token_id: u32, dim: usize) -> &'a [f32] {
    let start = weights.token_embedding_offset + token_id as usize * dim;
    &weights.data[start..start + dim]
}

// ── helpers ───────────────────────────────────────────────────
fn read_i32(r: &mut impl Read) -> io::Result<i32>  {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}
