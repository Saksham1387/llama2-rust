// ============================================================
// sampler.rs — turns logits into a token id
// ============================================================
// The model outputs a float score for every vocab token (logits).
// The sampler picks ONE token from those scores.
//
// Three strategies:
//   1. Greedy (temperature=0) — always pick the highest score
//   2. Multinomial (0 < temperature, topp=1) — sample from distribution
//   3. Top-p / nucleus (0 < topp < 1) — sample from top tokens only
// ============================================================

// ── Sampler ──────────────────────────────────────────────────
pub struct Sampler {
    vocab_size:  usize,
    temperature: f32,
    topp:        f32,
    rng_state:   u64,
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, seed: u64) -> Self {
        Sampler { vocab_size, temperature, topp, rng_state: seed }
    }

    // ── sample ───────────────────────────────────────────────
    // Main entry point — takes logits, returns a token id.
    pub fn sample(&mut self, logits: &mut Vec<f32>) -> u32 {
        if self.temperature == 0.0 {
            // ── greedy: just take the highest scoring token ───
            sample_argmax(logits) as u32
        } else {
            // ── apply temperature ─────────────────────────────
            // Dividing by temperature makes the distribution sharper (low temp)
            // or flatter (high temp).
            for v in logits.iter_mut() {
                *v /= self.temperature;
            }

            // ── convert logits → probabilities via softmax ────
            softmax_inplace(logits);

            // ── sample from distribution ──────────────────────
            let coin = self.random_f32();

            if self.topp <= 0.0 || self.topp >= 1.0 {
                // multinomial: sample from full distribution
                sample_mult(logits, coin) as u32
            } else {
                // top-p: sample from smallest set of tokens
                // whose cumulative probability exceeds topp
                sample_topp(logits, self.topp, coin) as u32
            }
        }
    }

    // ── xorshift RNG ─────────────────────────────────────────
    // Simple, fast, reproducible random number generator.
    fn random_u32(&mut self) -> u32 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        ((self.rng_state.wrapping_mul(0x2545F4914F6CDD1D)) >> 32) as u32
    }

    fn random_f32(&mut self) -> f32 {
        (self.random_u32() >> 8) as f32 / 16777216.0
    }
}

// ── sample_argmax ────────────────────────────────────────────
// Return index of the highest value — deterministic, temperature=0.
fn sample_argmax(probs: &[f32]) -> usize {
    probs.iter()
         .enumerate()
         .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
         .map(|(i, _)| i)
         .unwrap_or(0)
}

// ── sample_mult ──────────────────────────────────────────────
// Multinomial sampling — walk the CDF until coin lands.
fn sample_mult(probs: &[f32], coin: f32) -> usize {
    let mut cdf = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if coin < cdf { return i; }
    }
    probs.len() - 1 // rounding fallback
}

// ── sample_topp ──────────────────────────────────────────────
// Top-p (nucleus) sampling — only sample from the most likely tokens
// whose cumulative probability reaches topp.
// This prevents the model from ever producing very unlikely tokens.
fn sample_topp(probs: &[f32], topp: f32, coin: f32) -> usize {
    // collect (probability, original_index) pairs
    // filter out tokens too unlikely to ever be sampled
    let cutoff = (1.0 - topp) / (probs.len() - 1) as f32;
    let mut candidates: Vec<(f32, usize)> = probs.iter()
        .enumerate()
        .filter(|&(_, &p)| p >= cutoff)
        .map(|(i, &p)| (p, i))
        .collect();

    // sort descending by probability
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // find cutoff where cumulative prob exceeds topp
    let mut cumulative = 0.0_f32;
    let mut last_idx = candidates.len() - 1;
    for (i, &(p, _)) in candidates.iter().enumerate() {
        cumulative += p;
        if cumulative > topp {
            last_idx = i;
            break;
        }
    }

    // sample from the truncated list
    let r = coin * cumulative;
    let mut cdf = 0.0_f32;
    for &(p, idx) in &candidates[..=last_idx] {
        cdf += p;
        if r < cdf { return idx; }
    }
    candidates[last_idx].1 // rounding fallback
}

// ── softmax_inplace ──────────────────────────────────────────
// Same as math.rs softmax but operates on a Vec in place.
fn softmax_inplace(x: &mut Vec<f32>) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    for v in x.iter_mut() { *v /= sum; }
}
