//! Metrics over embedding vectors. Pure functions, unit-tested on every
//! platform. All of them are NaN-safe: a non-finite value is reported, never
//! folded away (`f32::min` returns the other operand when one is NaN, which
//! once hid a NaN-producing path behind "parity 1.000000").

/// Cosine similarity; `None` when either vector is non-finite or zero.
pub fn cosine(a: &[f32], b: &[f32]) -> Option<f64> {
    assert_eq!(a.len(), b.len(), "vector lengths differ");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let c = dot / (na.sqrt() * nb.sqrt());
    c.is_finite().then_some(c)
}

/// Summary of per-case scores where `None` means non-finite.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// Lowest score, or `None` if any score is non-finite.
    pub worst: Option<f64>,
    /// Index of the worst case (the first non-finite one, if any).
    pub worst_at: usize,
    pub mean: f64,
    pub non_finite: usize,
}

pub fn summarize(scores: &[Option<f64>]) -> Summary {
    assert!(!scores.is_empty(), "nothing to summarize");
    let non_finite = scores.iter().filter(|s| s.is_none()).count();
    let finite: Vec<f64> = scores.iter().flatten().copied().collect();
    let mean = if finite.is_empty() {
        f64::NAN
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    };
    if let Some(i) = scores.iter().position(Option::is_none) {
        return Summary {
            worst: None,
            worst_at: i,
            mean,
            non_finite,
        };
    }
    let (worst_at, worst) = finite
        .iter()
        .copied()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("non-empty");
    Summary {
        worst: Some(worst),
        worst_at,
        mean,
        non_finite,
    }
}

/// Largest change in pairwise similarity between two embeddings of the same
/// cases: max over pairs |cos_a(i, j) − cos_b(i, j)|. `None` if any
/// similarity is non-finite.
pub fn similarity_drift(a: &[Vec<f32>], b: &[Vec<f32>]) -> Option<f64> {
    assert_eq!(a.len(), b.len());
    let mut drift = 0f64;
    for i in 0..a.len() {
        for j in i + 1..a.len() {
            let d = (cosine(&a[i], &a[j])? - cosine(&b[i], &b[j])?).abs();
            drift = drift.max(d);
        }
    }
    Some(drift)
}

/// A ranking the reference makes clearly that a path reverses: for anchor
/// `anchor`, the reference puts `closer` ahead of `farther` by at least the
/// margin, and the path doesn't.
#[derive(Debug, Clone, PartialEq)]
pub struct RankFlip {
    pub anchor: usize,
    pub closer: usize,
    pub farther: usize,
    /// Reference similarity gap (≥ margin).
    pub reference_gap: f64,
    /// Path similarity gap (≤ 0).
    pub path_gap: f64,
}

/// Every (anchor, closer, farther) triple where the reference's similarity
/// gap is ≥ `margin` and the path's gap is ≤ 0. Near-ties (gap < margin)
/// are excused: fp16 may legitimately swap them.
pub fn rank_flips(reference: &[Vec<f32>], path: &[Vec<f32>], margin: f64) -> Vec<RankFlip> {
    let n = reference.len();
    let sim = |v: &[Vec<f32>], i: usize, j: usize| cosine(&v[i], &v[j]).unwrap_or(f64::NAN);
    let mut flips = Vec::new();
    for anchor in 0..n {
        let r: Vec<f64> = (0..n).map(|j| sim(reference, anchor, j)).collect();
        let p: Vec<f64> = (0..n).map(|j| sim(path, anchor, j)).collect();
        for closer in 0..n {
            for farther in 0..n {
                if closer == anchor || farther == anchor || closer == farther {
                    continue;
                }
                let reference_gap = r[closer] - r[farther];
                let path_gap = p[closer] - p[farther];
                // A NaN path gap is a flip too.
                if reference_gap >= margin && (path_gap.is_nan() || path_gap <= 0.0) {
                    flips.push(RankFlip {
                        anchor,
                        closer,
                        farther,
                        reference_gap,
                        path_gap,
                    });
                }
            }
        }
    }
    flips
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-12);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).unwrap().abs() < 1e-12);
        assert_eq!(cosine(&[f32::NAN, 0.0], &[1.0, 0.0]), None);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), None);
    }

    #[test]
    fn summary_never_hides_non_finite() {
        let s = summarize(&[Some(0.99), None, Some(0.5)]);
        assert_eq!(s.worst, None);
        assert_eq!(s.worst_at, 1);
        assert_eq!(s.non_finite, 1);
        let s = summarize(&[Some(0.99), Some(0.5), Some(0.7)]);
        assert_eq!((s.worst, s.worst_at), (Some(0.5), 1));
    }

    #[test]
    fn drift_is_zero_for_identical_and_positive_otherwise() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.6, 0.8]];
        assert_eq!(similarity_drift(&a, &a), Some(0.0));
        let mut b = a.clone();
        b[2] = vec![0.8, 0.6];
        assert!(similarity_drift(&a, &b).unwrap() > 0.1);
    }

    #[test]
    fn rank_flips_respect_margin() {
        // Anchor 0; reference: 1 much closer than 2. Path swaps them.
        let reference = vec![vec![1.0, 0.0], vec![0.9, 0.436], vec![0.0, 1.0]];
        let swapped = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.9, 0.436]];
        let flips = rank_flips(&reference, &swapped, 0.02);
        assert!(flips
            .iter()
            .any(|f| f.anchor == 0 && f.closer == 1 && f.farther == 2));
        assert!(rank_flips(&reference, &reference, 0.02).is_empty());
        // A near-tie is excused.
        let tie = vec![vec![1.0, 0.0], vec![0.70, 0.714], vec![0.71, 0.704]];
        let tie_swapped = vec![vec![1.0, 0.0], vec![0.71, 0.704], vec![0.70, 0.714]];
        assert!(rank_flips(&tie, &tie_swapped, 0.02).is_empty());
    }

    #[test]
    fn nan_path_counts_as_flip() {
        let reference = vec![vec![1.0, 0.0], vec![0.9, 0.436], vec![0.0, 1.0]];
        let broken = vec![vec![1.0, 0.0], vec![f32::NAN, 0.0], vec![0.0, 1.0]];
        assert!(!rank_flips(&reference, &broken, 0.02).is_empty());
    }
}
