// ── rmsnorm ──────────────────────────────────────────────────
// Root Mean Square Layer Normalization.
// Normalizes the vector x, then scales it by the learned weight.
// What it does mathematically:
//   1. compute RMS = sqrt(mean(x^2))
//   2. normalize: x_norm = x / RMS
//   3. scale:     output  = weight * x_norm
//
// Why we need it: transformer activations can grow very large.
// Normalizing keeps numbers stable so gradients don't explode.
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let size = x.len();

    // step 1: sum of squares
    let mut ss: f32 = x.iter().map(|v| v * v).sum();

    // step 2: mean, add epsilon for numerical stability, take inverse sqrt
    ss /= size as f32;
    ss += 1e-5_f32;
    ss = 1.0 / ss.sqrt();

    // step 3: normalize and scale by learned weight
    for j in 0..size {
        out[j] = weight[j] * (ss * x[j]);
    }
}

// ── softmax ──────────────────────────────────────────────────
// Converts a vector of raw scores into probabilities (sum to 1).
// The max subtraction trick: prevents exp() overflow.
//   softmax([1000, 1001]) == softmax([0, 1]) mathematically,
//   but exp(1000) overflows float32 while exp(0) doesn't.
pub fn softmax(x: &mut [f32]) {
    // find max for numerical stability
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // exp(x - max) and sum
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    // normalize
    for v in x.iter_mut() {
        *v /= sum;
    }
}

// ── matmul ───────────────────────────────────────────────────
// Matrix-vector multiply: W (d×n) @ x (n,) → out (d,)
// This is the MOST CALLED function in the entire forward pass.
// Memory layout of W is ROW-MAJOR:
//   w[i * n + j] = element at row i, column j
//   each row i of W is a vector of length n
//   dot product of row i with x gives output[i]
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    for i in 0..d {
        let row = &w[i * n..(i + 1) * n]; // row i of W
        out[i] = row.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
    }
}

// ── accum ────────────────────────────────────────────────────
// Element-wise add b into a: a[i] += b[i]
// Used for residual connections.
pub fn accum(a: &mut [f32], b: &[f32]) {
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        *ai += bi;
    }
}

pub fn copy_into(dst: &mut [f32], src: &[f32]) {
    dst.copy_from_slice(src);
}