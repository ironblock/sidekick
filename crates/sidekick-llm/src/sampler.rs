//! Choosing each token from its logits (docs/design/open-llm.md, "Sampling
//! and the wire contract").
//!
//! - **Temperature 0** is greedy: the largest logit, ties to the lower
//!   token id, so a request repeats byte for byte on the same build, Mac
//!   and macOS.
//! - **Otherwise** a token is drawn from the softmax of the logits at that
//!   temperature, restricted to the nucleus when `top_p` is below 1: the
//!   most likely tokens (ties to the lower id) whose probabilities first add
//!   up to `top_p`, at least one.
//! - **`seed`** seeds a ChaCha20 generator, so sampling repeats too; without
//!   one, the generator is seeded from the OS. Each sampled token takes one
//!   64-bit draw, mapped to a float in [0, 1) by its top 53 bits, so a
//!   seed's sequence doesn't depend on a library's distribution code.
//! - **A grammar's mask** (guided output) sets every token it doesn't allow
//!   to −∞ first, so greedy picks the most likely allowed token and the
//!   nucleus is taken over the masked distribution.

use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use toktrie::SimpleVob;

/// One request's sampling settings, its own or the model's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: Option<u64>,
}

/// A request's sampler.
pub struct Sampler {
    params: SamplingParams,
    rng: ChaCha20Rng,
    /// Unnormalized probabilities, reused from step to step.
    weights: Vec<f64>,
    nucleus: Vec<u32>,
    /// The masked logits, reused from step to step.
    masked: Vec<f32>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = match params.seed {
            Some(seed) => ChaCha20Rng::seed_from_u64(seed),
            None => ChaCha20Rng::from_os_rng(),
        };
        Self { params, rng, weights: Vec::new(), nucleus: Vec::new(), masked: Vec::new() }
    }

    pub fn params(&self) -> SamplingParams {
        self.params
    }

    /// The next token. `logits` are finite (the decoder replaces its state
    /// and fails a call that returns anything else).
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.params.temperature <= 0.0 {
            return argmax(logits);
        }
        let inv_t = 1.0 / f64::from(self.params.temperature);
        let max = f64::from(logits[argmax(logits) as usize]);
        self.weights.clear();
        self.weights.extend(logits.iter().map(|&l| ((f64::from(l) - max) * inv_t).exp()));
        let total: f64 = self.weights.iter().sum();
        let u = self.uniform();
        if self.params.top_p >= 1.0 {
            return pick(self.weights.iter().copied().enumerate().map(|(i, w)| (i as u32, w)), u * total);
        }
        self.nucleus(total);
        let mass: f64 = self.nucleus.iter().map(|&i| self.weights[i as usize]).sum();
        pick(self.nucleus.iter().map(|&i| (i, self.weights[i as usize])), u * mass)
    }

    /// The next token among those `mask` allows (it allows at least one).
    pub fn sample_masked(&mut self, logits: &[f32], mask: Option<&SimpleVob>) -> u32 {
        let Some(mask) = mask else { return self.sample(logits) };
        let mut masked = std::mem::take(&mut self.masked);
        masked.clear();
        masked.resize(logits.len(), f32::NEG_INFINITY);
        mask.iter_set_entries(|i| {
            if let Some(&l) = logits.get(i) {
                masked[i] = l;
            }
        });
        let token = self.sample(&masked);
        self.masked = masked;
        token
    }

    /// The nucleus, most likely first (ties to the lower id): the shortest
    /// such run whose weight reaches `top_p` of `total`, at least one token.
    /// Only tokens above a threshold are sorted, the threshold lowered until
    /// they hold the nucleus' mass, so a 130k-token row isn't sorted whole.
    fn nucleus(&mut self, total: f64) {
        let target = f64::from(self.params.top_p) * total;
        // The largest weight is 1 (the max logit's).
        let mut threshold = 1e-3;
        loop {
            self.nucleus.clear();
            let mut mass = 0.0;
            for (i, &w) in self.weights.iter().enumerate() {
                if w >= threshold {
                    self.nucleus.push(i as u32);
                    mass += w;
                }
            }
            if mass >= target || threshold < 1e-300 {
                break;
            }
            threshold *= 1e-3;
        }
        let w = &self.weights;
        self.nucleus.sort_unstable_by(|&a, &b| w[b as usize].total_cmp(&w[a as usize]).then(a.cmp(&b)));
        let mut mass = 0.0;
        let mut keep = 0;
        for &i in &self.nucleus {
            mass += w[i as usize];
            keep += 1;
            if mass >= target {
                break;
            }
        }
        self.nucleus.truncate(keep.max(1));
    }

    /// A float in [0, 1) from one 64-bit draw.
    fn uniform(&mut self) -> f64 {
        (self.rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// The first token whose running weight passes `at`; the last one with
/// any weight if rounding leaves `at` past the end.
fn pick(weights: impl Iterator<Item = (u32, f64)>, at: f64) -> u32 {
    let mut acc = 0.0;
    let mut last = 0;
    for (id, w) in weights {
        if w <= 0.0 {
            continue;
        }
        acc += w;
        last = id;
        if acc > at {
            return id;
        }
    }
    last
}

/// The largest logit's token, ties to the lower id.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(temperature: f32, top_p: f32, seed: Option<u64>) -> SamplingParams {
        SamplingParams { temperature, top_p, seed }
    }

    #[test]
    fn greedy_takes_the_largest_and_ties_go_to_the_lower_id() {
        let mut s = Sampler::new(params(0.0, 0.5, None));
        assert_eq!(s.sample(&[0.0, 3.0, 1.0, 3.0]), 1);
        assert_eq!(argmax(&[-1.0, -1.0]), 0);
    }

    #[test]
    fn a_seed_repeats_and_another_seed_differs() {
        let logits: Vec<f32> = (0..1000).map(|i| ((i * 7919) % 101) as f32 / 25.0).collect();
        let run = |seed| {
            let mut s = Sampler::new(params(1.0, 0.95, Some(seed)));
            (0..64).map(|_| s.sample(&logits)).collect::<Vec<_>>()
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }

    #[test]
    fn top_p_keeps_the_nucleus() {
        // Probabilities about 0.64, 0.24, 0.09, 0.03.
        let logits = [3.0, 2.0, 1.0, 0.0];
        let mut seen = [0usize; 4];
        let mut s = Sampler::new(params(1.0, 0.8, Some(1)));
        for _ in 0..4000 {
            seen[s.sample(&logits) as usize] += 1;
        }
        assert_eq!(&seen[2..], &[0, 0], "outside the 0.8 nucleus: {seen:?}");
        // Inside it, renormalized: 0.64 / 0.88 = 0.73.
        let share = seen[0] as f64 / 4000.0;
        assert!((share - 0.73).abs() < 0.03, "{seen:?}");
        // A nucleus of one is greedy, at any temperature.
        let mut s = Sampler::new(params(5.0, 0.0, None));
        assert!((0..100).all(|_| s.sample(&logits) == 0));
    }

    #[test]
    fn temperature_reshapes_the_distribution() {
        let logits = [1.0, 0.0];
        let share = |t: f32| {
            let mut s = Sampler::new(params(t, 1.0, Some(3)));
            (0..4000).filter(|_| s.sample(&logits) == 0).count() as f64 / 4000.0
        };
        // softmax([1, 0] / t)[0]: 0.73 at t = 1, 0.95 at t = 1/3, 0.62 at t = 2.
        assert!((share(1.0) - 0.731).abs() < 0.03);
        assert!((share(1.0 / 3.0) - 0.953).abs() < 0.02);
        assert!((share(2.0) - 0.622).abs() < 0.03);
    }

    #[test]
    fn a_mask_restricts_greedy_and_sampling_to_what_it_allows() {
        let logits = [5.0, 4.0, 3.0, 2.0, 1.0];
        let mut mask = SimpleVob::alloc(5);
        mask.allow_token(2);
        mask.allow_token(4);
        let mut s = Sampler::new(params(0.0, 1.0, None));
        assert_eq!(s.sample_masked(&logits, Some(&mask)), 2);
        assert_eq!(s.sample_masked(&logits, None), 0);
        // Sampling: only 2 and 4, in proportion e^3 : e^1 (0.88 : 0.12).
        let mut s = Sampler::new(params(1.0, 1.0, Some(9)));
        let picks: Vec<u32> = (0..2000).map(|_| s.sample_masked(&logits, Some(&mask))).collect();
        assert!(picks.iter().all(|&p| p == 2 || p == 4));
        let share = picks.iter().filter(|&&p| p == 2).count() as f64 / 2000.0;
        assert!((share - 0.881).abs() < 0.03, "{share}");
        // The nucleus is over the masked distribution: 0.8 keeps only 2.
        let mut s = Sampler::new(params(1.0, 0.8, Some(9)));
        assert!((0..200).all(|_| s.sample_masked(&logits, Some(&mask)) == 2));
    }

    #[test]
    fn a_wide_row_with_a_flat_tail_samples_the_nucleus_exactly() {
        // One likely token and a long flat tail: the threshold must drop
        // until the tail joins the nucleus.
        let mut logits = vec![0.0f32; 130_560];
        logits[42] = 12.0;
        let mut s = Sampler::new(params(1.0, 0.99, Some(5)));
        let picks: Vec<u32> = (0..60).map(|_| s.sample(&logits)).collect();
        // exp(12) / (exp(12) + 130559) is about 0.55: the tail is sampled.
        assert!(picks.iter().any(|&p| p != 42) && picks.contains(&42));
    }
}
