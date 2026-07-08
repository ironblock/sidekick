use crate::Result;

/// What the caller will do with the vectors. Models like EmbeddingGemma use
/// different prompt prefixes for queries vs. documents; backends apply them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbedPurpose {
    #[default]
    Document,
    Query,
}

/// A text-embedding backend. Implementations: Core ML encoder on ANE,
/// static (model2vec-style) CPU floor tier.
///
/// Synchronous by design — Core ML predictions and static lookups are both
/// blocking; async callers wrap in `spawn_blocking`.
pub trait Embedder: Send + Sync {
    /// Stable identifier, used as the OpenAI `model` name.
    fn id(&self) -> &str;

    /// Native output dimensionality.
    fn dims(&self) -> usize;

    /// Dimensions this model was trained to truncate to (Matryoshka), largest
    /// first, including the native size. Empty means truncation is lossy and
    /// unsupported.
    fn matryoshka_dims(&self) -> &[usize] {
        &[]
    }

    /// Embed a batch. Returns one unit-normalized vector of `dims()` length
    /// per input, in order.
    fn embed(&self, texts: &[&str], purpose: EmbedPurpose) -> Result<Vec<Vec<f32>>>;
}

/// Truncate a unit vector to `dims` and re-normalize (Matryoshka truncation).
/// Callers must check `matryoshka_dims()` before using this on model output.
pub fn truncate_normalized(v: &[f32], dims: usize) -> Vec<f32> {
    let mut out: Vec<f32> = v[..dims.min(v.len())].to_vec();
    let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut out {
            *x /= norm;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_renormalizes() {
        let v = vec![3.0, 4.0, 12.0];
        let t = truncate_normalized(&v, 2);
        assert_eq!(t.len(), 2);
        let norm = t.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        // Direction preserved: 3:4 ratio.
        assert!((t[0] / t[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn truncation_handles_zero_vector() {
        let t = truncate_normalized(&[0.0, 0.0], 2);
        assert_eq!(t, vec![0.0, 0.0]);
    }
}
