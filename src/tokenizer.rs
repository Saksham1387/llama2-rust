use std::{fs::File};
use std::io::{self, BufReader, Read};

pub struct TokenIndex {
    str_val: String,
    id: u32
}

pub struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    sorted_vocab: Vec<TokenIndex>,
    vocab_size: usize,
    max_token_length: usize,

    byte_pieces: Vec<String>
}


impl Tokenizer {
    pub fn from_file(path: &str,vocab_size:usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        let max_token_length = read_u32(&mut reader)? as usize;

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut vocab_scores = Vec::with_capacity(vocab_size);

        for _ in 0..vocab_size {
            let score = read_f32(&mut reader)?;
            let len   = read_u32(&mut reader)? as usize;
            let s     = read_string(&mut reader, len)?;
 
            vocab_scores.push(score);
            vocab.push(s);
        };

        let byte_pieces: Vec<String> = (0u8..=255)
        .map(|b| {
            // store as a 1-byte string; non-UTF8 bytes stay as raw bytes
            String::from_utf8(vec![b]).unwrap_or_else(|_| format!("<0x{:02X}>", b))
        })
        .collect();

        let mut sorted_vocab: Vec<TokenIndex> = vocab.iter().enumerate().map(|(id,s)| TokenIndex {str_val: s.clone(), id:id as u32}).collect();
        sorted_vocab.sort_by(|a, b| a.str_val.cmp(&b.str_val));

        Ok(Tokenizer { vocab, vocab_scores, sorted_vocab, vocab_size, max_token_length, byte_pieces })
    }

    fn str_lookup(&self, s: &str) -> Option<u32> {
        self.sorted_vocab
            .binary_search_by(|ti| ti.str_val.as_str().cmp(s))
            .ok()
            .map(|idx| self.sorted_vocab[idx].id)
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let mut tokens: Vec<u32> = Vec::new();
 
        // add BOS token first if requested
        if bos {
            tokens.push(1);
        }
 
        // ── dummy prefix ─────────────────────────────────────
        // SentencePiece always prepends a space to the input.
        if !text.is_empty() {
            if let Some(id) = self.str_lookup(" ") {
                tokens.push(id);
            }
        }
 
        // ── Phase 1: character / UTF-8 byte encoding ─────────
        // Walk the raw UTF-8 bytes. For each full Unicode codepoint,
        // look it up in vocab. If not found, fall back to byte tokens.
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // figure out how many bytes this UTF-8 codepoint uses
            let codepoint_len = utf8_codepoint_len(bytes[i]);
            let end = (i + codepoint_len).min(bytes.len());
            let slice = &bytes[i..end];
 
            // try to interpret as a UTF-8 string
            let codepoint_str = std::str::from_utf8(slice)
                .unwrap_or(""); // if bad UTF-8, will fall through to byte fallback
 
            if !codepoint_str.is_empty() {
                if let Some(id) = self.str_lookup(codepoint_str) {
                    // found the codepoint in vocab directly
                    tokens.push(id);
                } else {
                    // byte fallback — encode each byte individually
                    // +3 because first 3 ids are <unk>=0, <s>=1, </s>=2
                    for &b in slice {
                        tokens.push(b as u32 + 3);
                    }
                }
            }
            i = end;
        }
 
        // ── Phase 2: BPE merge loop ───────────────────────────
        // Repeatedly find the pair of adjacent tokens with the HIGHEST
        // vocab_score, merge them into one token, repeat until no merges left.
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id: Option<u32>    = None;
            let mut best_idx: Option<usize> = None;
 
            // scan every consecutive pair
            for idx in 0..tokens.len().saturating_sub(1) {
                // concatenate the two token strings
                let merged = format!(
                    "{}{}",
                    self.vocab[tokens[idx] as usize],
                    self.vocab[tokens[idx + 1] as usize]
                );
 
                // look the merged string up in vocab
                if let Some(id) = self.str_lookup(&merged) {
                    let score = self.vocab_scores[id as usize];
                    if score > best_score {
                        best_score = score;
                        best_id    = Some(id);
                        best_idx   = Some(idx);
                    }
                }
            }
 
            // no merge found → done
            let (Some(id), Some(idx)) = (best_id, best_idx) else {
                break;
            };
 
            // apply merge: replace pair (idx, idx+1) with merged token
            tokens[idx] = id;
            tokens.remove(idx + 1); // shift left after merging
        }
 
        // add EOS token at the end if requested
        if eos {
            tokens.push(2);
        }
 
        tokens
    }



    pub fn decode(&self, prev_token:u32,token:u32) -> &str {
        let mut piece = self.vocab[token as usize].as_str();

        if prev_token == 1 && piece.starts_with(' ') {
            piece = &piece[1..];
        }

        if let Some(byte_val) = parse_byte_token(piece) {
            return self.byte_pieces[byte_val as usize].as_str();
        }

        piece
    }

    pub fn decode_string(&self, tokens: &[u32]) -> String {
        let mut out = String::new();
        for (i, &tok) in tokens.iter().enumerate() {
            let prev = if i == 0 { 1 } else { tokens[i - 1] };
            let piece = self.decode(prev, tok);
            // skip non-printable single bytes
            if is_safe_piece(piece) {
                out.push_str(piece);
            }
        }
        out
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size }
    pub fn token_to_str(&self, id: u32) -> &str { &self.vocab[id as usize] }
    pub fn bos_token(&self) -> u32 { 1 }
    pub fn eos_token(&self) -> u32 { 2 }
}

fn utf8_codepoint_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xFF => 4,
        _           => 1, // continuation byte — treat as 1
    }
}

fn is_safe_piece(piece: &str) -> bool {
    if piece.is_empty() { return false; }
    let bytes = piece.as_bytes();
    if bytes.len() == 1 {
        let b = bytes[0];
        // only allow printable ASCII or whitespace
        return b.is_ascii_graphic() || b.is_ascii_whitespace();
    }
    true
}

fn parse_byte_token(s: &str) -> Option<u8> {
    // must be exactly "<0xXX>"  (6 chars)
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        let hex = &s[3..5];
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
 
fn read_string(r: &mut impl Read, len: usize) -> io::Result<String> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    // SentencePiece uses UTF-8; fall back to lossy if bytes are invalid
    Ok(String::from_utf8_lossy(&buf).into_owned())
}
