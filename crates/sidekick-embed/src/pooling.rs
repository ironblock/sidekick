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
    fn normalize_unit_length() {
        let mut v = [3.0, 4.0];
        normalize_in_place(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }
}
