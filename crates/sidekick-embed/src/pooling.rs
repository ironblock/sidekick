use sidekick_core::manifest::Pooling;

/// Mean-pool token vectors `[seq, dims]` (flattened row-major) under an
/// attention mask. Padding positions (mask 0) are excluded.
pub fn mean_pool(hidden: &[f32], dims: usize, mask: &[u32]) -> Vec<f32> {
    debug_assert_eq!(hidden.len(), mask.len() * dims);
    let mut out = vec![0.0f32; dims];
    let mut count = 0u32;
    for (i, &m) in mask.iter().enumerate() {
        if m == 0 {
            continue;
        }
        count += 1;
        let row = &hidden[i * dims..(i + 1) * dims];
        for (o, x) in out.iter_mut().zip(row) {
            *o += x;
        }
    }
    if count > 0 {
        let inv = 1.0 / count as f32;
        for o in &mut out {
            *o *= inv;
        }
    }
    out
}

/// Pool one input's token vectors `[seq, dims]` (row-major) as `pooling`
/// says, under its attention mask (1 for a real token): the mean of the
/// real tokens, the first token (`cls`), or the last real one (`last`).
/// `None` takes `hidden` as the pooled vector. The caller checks the
/// shape: `hidden` holds `mask.len()` rows of `dims`, or `dims` values for
/// `None`. Backends differ only in their forward pass; this is shared.
pub fn pool(hidden: &[f32], dims: usize, mask: &[u32], pooling: Pooling) -> Vec<f32> {
    match pooling {
        Pooling::None => hidden.to_vec(),
        Pooling::Mean => mean_pool(hidden, dims, mask),
        Pooling::Cls => hidden[..dims].to_vec(),
        Pooling::Last => {
            let last = mask.iter().rposition(|&m| m != 0).unwrap_or(0);
            hidden[last * dims..(last + 1) * dims].to_vec()
        }
    }
}

/// Scale to unit L2 norm (no-op on the zero vector).
pub fn normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_respects_mask() {
        // Two real tokens and one padding token that must be ignored.
        let hidden = [1.0, 2.0, 3.0, 4.0, 100.0, 100.0];
        let pooled = mean_pool(&hidden, 2, &[1, 1, 0]);
        assert_eq!(pooled, vec![2.0, 3.0]);
    }

    #[test]
    fn pool_takes_the_token_its_pooling_names() {
        // Three tokens of two dims; the third is padding.
        let hidden = [1.0, 2.0, 3.0, 4.0, 100.0, 100.0];
        let mask = [1, 1, 0];
        assert_eq!(pool(&hidden, 2, &mask, Pooling::Mean), vec![2.0, 3.0]);
        assert_eq!(pool(&hidden, 2, &mask, Pooling::Cls), vec![1.0, 2.0]);
        assert_eq!(pool(&hidden, 2, &mask, Pooling::Last), vec![3.0, 4.0], "the last real token, not the pad");
        assert_eq!(pool(&[5.0, 6.0], 2, &[1], Pooling::None), vec![5.0, 6.0]);
    }

    #[test]
    fn normalize_unit_length() {
        let mut v = [3.0, 4.0];
        normalize_in_place(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }
}
