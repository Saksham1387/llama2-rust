mod tokenizer;
mod model;
mod math;
mod forward;
mod sampler;

use tokenizer::Tokenizer;
use model::Transformer;
use forward::{RunState, forward};
use sampler::Sampler;
use std::time::Instant;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage:   cargo run --release -- <model.bin> <tokenizer.bin> [options]");
        eprintln!("Example: cargo run --release -- stories15M.bin tokenizer.bin");
        eprintln!("Options (all optional):");
        eprintln!("  -i <prompt>      input prompt  (default: empty)");
        eprintln!("  -n <steps>       number of tokens to generate (default: 256)");
        eprintln!("  -t <temperature> 0.0=greedy, 1.0=original (default: 1.0)");
        eprintln!("  -p <topp>        top-p sampling 0..1 (default: 0.9)");
        eprintln!("  -s <seed>        random seed (default: time-based)");
        return;
    }

    let model_path     = &args[1];
    let tokenizer_path = &args[2];

    // ── parse optional flags ──────────────────────────────────
    let mut prompt      = String::new();
    let mut steps       = 256usize;
    let mut temperature = 1.0f32;
    let mut topp        = 0.9f32;
    let mut seed        = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap().as_secs();

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "-i" => { i += 1; prompt = args[i].clone(); }
            "-n" => { i += 1; steps = args[i].parse().unwrap_or(256); }
            "-t" => { i += 1; temperature = args[i].parse().unwrap_or(1.0); }
            "-p" => { i += 1; topp = args[i].parse().unwrap_or(0.9); }
            "-s" => { i += 1; seed = args[i].parse().unwrap_or(42); }
            _    => {}
        }
        i += 1;
    }

    // ── load model ────────────────────────────────────────────
    eprint!("Loading model... ");
    let transformer = Transformer::from_file(model_path)
        .expect("Failed to load model");
    let cfg = &transformer.config;
    eprintln!("done. ({} layers, dim={}, vocab={})",
              cfg.n_layers, cfg.dim, cfg.vocab_size);

    // clamp steps to max sequence length
    let steps = steps.min(cfg.seq_len);

    // ── load tokenizer ────────────────────────────────────────
    let tok = Tokenizer::from_file(tokenizer_path, cfg.vocab_size)
        .expect("Failed to load tokenizer");

    // ── allocate run state ────────────────────────────────────
    // All the temporary buffers needed during the forward pass.
    let mut state = RunState::new(cfg);

    // ── build sampler ─────────────────────────────────────────
    let mut sampler = Sampler::new(cfg.vocab_size, temperature, topp, seed);

    // ── encode prompt ─────────────────────────────────────────
    let prompt_tokens = tok.encode(&prompt, true, false);
    assert!(!prompt_tokens.is_empty(), "prompt encoding produced no tokens");

    // ── generation loop ───────────────────────────────────────
    // 
    // Two phases:
    //   Phase 1 (prefill):  feed all prompt tokens one by one
    //                       — don't sample, just process
    //   Phase 2 (generate): after prompt, start sampling new tokens
    //
    let mut token = prompt_tokens[0];
    let mut pos   = 0usize;
    let mut start: Option<Instant> = None;

    while pos < steps {
        // run the full transformer forward pass for this token
        let logits = forward(cfg, &transformer.weights, &mut state, token, pos);
        let mut logits_vec = logits.to_vec();

        // decide next token
        let next = if pos < prompt_tokens.len() - 1 {
            // still consuming prompt — force the next prompt token
            prompt_tokens[pos + 1]
        } else {
            // prompt fully consumed — sample from model output
            sampler.sample(&mut logits_vec)
        };

        pos += 1;

        // stop if model produces EOS token 
        if next == 1 { break; }

        // decode and print the token
        let piece = tok.decode(token, next);
        print!("{}", piece);
        std::io::stdout().flush().unwrap();

        token = next;

        // start timer after first token (first iter is slower due to cache warmup)
        if start.is_none() && pos > 1 {
            start = Some(Instant::now());
        }
    }

    println!(); 

    // ── report tokens/sec ─────────────────────────────────────
    if let Some(t) = start {
        let elapsed = t.elapsed().as_secs_f64();
        let toks = (pos - 1) as f64;
        eprintln!("\nachieved: {:.1} tokens/sec", toks / elapsed);
    }
}