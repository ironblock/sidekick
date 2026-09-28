//! Grading one path's results against the reference: pure, unit-tested.

use crate::expect::{ceiling_for, floor_for, letter, Floor, Gates, Path3};
use crate::metrics::{cosine, rank_flips, similarity_drift, summarize};
use crate::reference::Reference;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Cases tagged this way are stress inputs, floored separately.
pub const STRESS_TAG: &str = "degenerate";

/// A comparison repeated over several runs: `n` runs, lowest cosine
/// (`None` if any run was non-finite). `n == 0` means not applicable.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Check {
    pub n: usize,
    pub min: Option<f64>,
}

impl Check {
    pub fn add(&mut self, c: Option<f64>) {
        self.min = match (self.n, self.min, c) {
            (0, _, c) => c,
            (_, Some(m), Some(c)) => Some(m.min(c)),
            _ => None, // poisoned: once non-finite, stays non-finite
        };
        self.n += 1;
    }

    pub fn merge(&mut self, other: Check) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other;
            return;
        }
        self.min = match (self.min, other.min) {
            (Some(a), Some(b)) => Some(a.min(b)),
            _ => None,
        };
        self.n += other.n;
    }
}

/// One case as a worker ran it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    pub id: String,
    /// sidekick's token ids equal the reference pipeline's.
    pub ids_match: bool,
    pub bucket: usize,
    /// The product output (`embed`); non-finite values replaced by 0.
    pub vector: Vec<f32>,
    pub finite: bool,
    /// When ids differ: the model's output on the reference's ids, to
    /// separate tokenization from numerics.
    pub model_only: Option<Vec<f32>>,
    /// vs the same ids in each larger bucket.
    pub bucket_invariance: Check,
    /// vs random pad ids in the same bucket.
    pub pad_invariance: Check,
    pub ms: f64,
}

/// Everything one worker (one model, one path, one process) measured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResult {
    pub model: String,
    pub path: String,
    pub cases: Vec<CaseResult>,
    /// Re-running the first cases at the end gave identical bits.
    pub repeat_bitwise: bool,
    pub load_ms: f64,
}

/// A path's grade: the numbers, and every gate or floor it failed.
#[derive(Debug, Clone, Serialize)]
pub struct PathGrade {
    pub path: String,
    pub letter: char,
    /// Worst cosine vs torch over non-stress cases, and where.
    pub worst: Option<f64>,
    pub worst_case: String,
    pub stress: Option<f64>,
    pub mean: f64,
    /// Max |Δ pairwise similarity| vs torch, and the mean signed Δ (a
    /// systematic inflation or deflation of similarity scores).
    pub drift: Option<f64>,
    pub drift_bias: f64,
    pub rank_flips: usize,
    pub bucket_invariance: Check,
    pub pad_invariance: Check,
    pub by_tag: BTreeMap<String, Option<f64>>,
    /// Worst cosine vs each non-torch oracle (report only).
    pub vs_oracles: BTreeMap<String, Option<f64>>,
    /// Worst cosine at each truncated Matryoshka dimension.
    pub matryoshka: BTreeMap<usize, Option<f64>>,
    pub median_ms: f64,
    /// Cosine vs torch per case, in reference order.
    pub per_case: Vec<Option<f64>>,
    /// Hard gates failed: a broken artifact or product path.
    pub failures: Vec<String>,
    /// This chip's floors failed: less accurate than when last measured.
    pub regressions: Vec<String>,
}

fn truncate(v: &[f32], dims: usize) -> Vec<f32> {
    sidekick_core::truncate_normalized(v, dims)
}

fn worst_of(scores: &[Option<f64>], select: &[bool]) -> Option<(Option<f64>, usize)> {
    let picked: Vec<(usize, Option<f64>)> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| select[*i])
        .collect();
    if picked.is_empty() {
        return None;
    }
    let s = summarize(&picked.iter().map(|p| p.1).collect::<Vec<_>>());
    Some((s.worst, picked[s.worst_at].0))
}

/// Grade `result` against `reference`. `floor` is this chip's expectation
/// for the model and path, if one is recorded.
pub fn grade(
    reference: &Reference,
    result: &WorkerResult,
    path: Path3,
    gates: &Gates,
    floor: Option<&Floor>,
    matryoshka: &[usize],
) -> PathGrade {
    let cases = &reference.cases;
    let torch = &reference.vectors["torch"];
    assert_eq!(
        result.cases.len(),
        cases.len(),
        "worker ran a different case list"
    );
    let vectors: Vec<Vec<f32>> = result.cases.iter().map(|c| c.vector.clone()).collect();
    let per_case: Vec<Option<f64>> = result
        .cases
        .iter()
        .zip(torch)
        .map(|(c, t)| if c.finite { cosine(&c.vector, t) } else { None })
        .collect();

    let stress: Vec<bool> = cases
        .iter()
        .map(|c| c.tags.iter().any(|t| t == STRESS_TAG))
        .collect();
    let core: Vec<bool> = stress.iter().map(|s| !s).collect();
    let (worst, worst_at) = worst_of(&per_case, &core).expect("the corpus has non-stress cases");
    let stress_worst = worst_of(&per_case, &stress).and_then(|w| w.0);
    let mean = summarize(&per_case).mean;

    let mut by_tag: BTreeMap<String, Option<f64>> = BTreeMap::new();
    for tag in cases.iter().flat_map(|c| c.tags.iter()) {
        if by_tag.contains_key(tag) {
            continue;
        }
        let sel: Vec<bool> = cases.iter().map(|c| c.tags.contains(tag)).collect();
        by_tag.insert(tag.clone(), worst_of(&per_case, &sel).and_then(|w| w.0));
    }

    let any_non_finite = per_case.iter().any(Option::is_none);
    let drift = if any_non_finite {
        None
    } else {
        similarity_drift(&vectors, torch)
    };
    let drift_bias = if any_non_finite {
        f64::NAN
    } else {
        let (mut sum, mut n) = (0f64, 0usize);
        for i in 0..vectors.len() {
            for j in i + 1..vectors.len() {
                if let (Some(a), Some(b)) = (
                    cosine(&vectors[i], &vectors[j]),
                    cosine(&torch[i], &torch[j]),
                ) {
                    sum += a - b;
                    n += 1;
                }
            }
        }
        sum / n.max(1) as f64
    };
    let rank_flips = rank_flips(torch, &vectors, gates.rank_margin).len();

    let mut bucket_invariance = Check::default();
    let mut pad_invariance = Check::default();
    for c in &result.cases {
        bucket_invariance.merge(c.bucket_invariance);
        pad_invariance.merge(c.pad_invariance);
    }

    let mut vs_oracles = BTreeMap::new();
    for (name, oracle) in &reference.vectors {
        if name == "torch" {
            continue;
        }
        let scores: Vec<Option<f64>> = result
            .cases
            .iter()
            .zip(oracle)
            .map(|(c, o)| if c.finite { cosine(&c.vector, o) } else { None })
            .collect();
        vs_oracles.insert(name.clone(), worst_of(&scores, &core).and_then(|w| w.0));
    }

    let mut mrl = BTreeMap::new();
    for &d in matryoshka.iter().filter(|&&d| d < reference.model.dims) {
        let scores: Vec<Option<f64>> = result
            .cases
            .iter()
            .zip(torch)
            .map(|(c, t)| {
                if c.finite {
                    cosine(&truncate(&c.vector, d), &truncate(t, d))
                } else {
                    None
                }
            })
            .collect();
        mrl.insert(d, worst_of(&scores, &core).and_then(|w| w.0));
    }

    let mut ms: Vec<f64> = result.cases.iter().map(|c| c.ms).collect();
    ms.sort_by(f64::total_cmp);
    let median_ms = ms[ms.len() / 2];

    // Hard gates.
    let mut failures = Vec::new();
    // No usable output: non-finite, or a vector cosine can't use (all zeros).
    let bad: Vec<&str> = result
        .cases
        .iter()
        .zip(&per_case)
        .filter(|(_, c)| c.is_none())
        .map(|(r, _)| r.id.as_str())
        .collect();
    if !bad.is_empty() {
        failures.push(format!(
            "no usable output (non-finite, or all zeros) on {} case(s): {}",
            bad.len(),
            bad.join(", ")
        ));
    }
    let bad: Vec<&str> = result
        .cases
        .iter()
        .filter(|c| !c.ids_match)
        .map(|c| c.id.as_str())
        .collect();
    if !bad.is_empty() {
        failures.push(format!(
            "token ids differ from the reference pipeline on {} case(s): {}",
            bad.len(),
            bad.join(", ")
        ));
    }
    if !result.repeat_bitwise {
        failures.push("re-running cases in the same process changed the output".into());
    }
    let gate = gates.bucket_invariance.get(path);
    if bucket_invariance.n > 0 && !bucket_invariance.min.is_some_and(|m| m >= gate) {
        failures.push(format!(
            "bucket invariance {} < {gate}",
            fmt(bucket_invariance.min)
        ));
    }
    if pad_invariance.n > 0
        && !pad_invariance
            .min
            .is_some_and(|m| m >= gates.pad_invariance)
    {
        failures.push(format!(
            "pad invariance {} < {}: the output depends on pad content (a dropped attention mask, or a convolution reading pad states)",
            fmt(pad_invariance.min),
            gates.pad_invariance
        ));
    }

    // Floors (this chip's regression tests).
    let mut regressions = Vec::new();
    if let Some(f) = floor {
        if !worst.is_some_and(|w| w >= f.worst) {
            regressions.push(format!(
                "worst cosine {} ({}) below this chip's floor {}",
                fmt(worst),
                cases[worst_at].id,
                f.worst
            ));
        }
        if let Some(sf) = f.stress {
            if !stress_worst.is_some_and(|w| w >= sf) {
                regressions.push(format!(
                    "stress-case worst {} below floor {sf}",
                    fmt(stress_worst)
                ));
            }
        }
        if !drift.is_some_and(|d| d <= f.drift) {
            regressions.push(format!(
                "similarity drift {} above ceiling {}",
                fmt(drift),
                f.drift
            ));
        }
    }

    let letter = if failures.is_empty() {
        letter(worst)
    } else {
        'F'
    };

    PathGrade {
        path: path.name().to_string(),
        letter,
        worst,
        worst_case: cases[worst_at].id.clone(),
        stress: stress_worst,
        mean,
        drift,
        drift_bias,
        rank_flips,
        bucket_invariance,
        pad_invariance,
        by_tag,
        vs_oracles,
        matryoshka: mrl,
        median_ms,
        per_case,
        failures,
        regressions,
    }
}

/// Floors for this chip from a grade, as TOML for expectations.toml.
pub fn suggest_floor(chip: &str, model: &str, g: &PathGrade) -> Option<String> {
    let worst = g.worst?;
    let drift = g.drift?;
    let mut s = format!(
        "[[floor]]\nchip = {chip:?}\nmodel = {model:?}\npath = {:?}\nworst = {}\n",
        g.path,
        floor_for(worst)
    );
    if let Some(st) = g.stress {
        s += &format!("stress = {}\n", floor_for(st));
    }
    s += &format!("drift = {}\n", ceiling_for(drift));
    Some(s)
}

pub fn fmt(x: Option<f64>) -> String {
    match x {
        Some(v) => format!("{v:.6}"),
        None => "non-finite".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expect::PerPath;
    use crate::reference::{Case, RefModel, RefPrefixes};

    fn gates() -> Gates {
        Gates {
            bucket_invariance: PerPath {
                cpu: 0.999999,
                gpu: 0.9999,
                ane: 0.995,
            },
            pad_invariance: 0.99999,
            determinism: 0.999999,
            rank_margin: 0.02,
        }
    }

    fn reference(vectors: Vec<Vec<f32>>, tags: &[&[&str]]) -> Reference {
        let cases = tags
            .iter()
            .enumerate()
            .map(|(i, t)| Case {
                id: format!("c{i}"),
                purpose: "document".into(),
                tags: t.iter().map(|s| s.to_string()).collect(),
                text: String::new(),
                ids: vec![1, 2],
            })
            .collect();
        Reference {
            format: 1,
            corpus_sha256: String::new(),
            model: RefModel {
                id: "m".into(),
                dims: vectors[0].len(),
                buckets: vec![128],
                max_seq_len: 128,
                prefixes: RefPrefixes {
                    query: String::new(),
                    document: String::new(),
                },
            },
            source: serde_json::Value::Null,
            tokenizer_sha256: String::new(),
            prompt_source: BTreeMap::new(),
            pooling: "cls".into(),
            oracles: vec!["torch".into()],
            versions: BTreeMap::new(),
            cases,
            vectors: BTreeMap::from([("torch".to_string(), vectors)]),
        }
    }

    fn result(vectors: Vec<Vec<f32>>) -> WorkerResult {
        WorkerResult {
            model: "m".into(),
            path: "ane".into(),
            cases: vectors
                .into_iter()
                .enumerate()
                .map(|(i, v)| CaseResult {
                    id: format!("c{i}"),
                    ids_match: true,
                    bucket: 128,
                    finite: v.iter().all(|x| x.is_finite()),
                    vector: v,
                    model_only: None,
                    bucket_invariance: Check {
                        n: 1,
                        min: Some(1.0),
                    },
                    pad_invariance: Check {
                        n: 1,
                        min: Some(1.0),
                    },
                    ms: 1.0,
                })
                .collect(),
            repeat_bitwise: true,
            load_ms: 0.0,
        }
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn check_poisons_on_non_finite() {
        let mut c = Check::default();
        c.add(Some(0.99));
        c.add(None);
        c.add(Some(0.5));
        assert_eq!(c, Check { n: 3, min: None });
        let mut m = Check {
            n: 1,
            min: Some(0.9),
        };
        m.merge(Check { n: 2, min: None });
        assert_eq!(m.min, None);
        m.merge(Check::default());
        assert_eq!(m.n, 3);
    }

    #[test]
    fn exact_path_grades_a_and_stress_is_separate() {
        let t = vec![
            unit(&[1.0, 0.0, 0.1]),
            unit(&[0.0, 1.0, 0.2]),
            unit(&[0.5, 0.5, 0.0]),
        ];
        let r = reference(t.clone(), &[&[], &["tiny"], &["degenerate"]]);
        let mut v = t.clone();
        v[2] = unit(&[0.5, 0.6, 0.3]); // stress case off; core exact
        let g = grade(&r, &result(v), Path3::Ane, &gates(), None, &[]);
        assert_eq!(g.letter, 'A');
        assert!(g.stress.unwrap() < 0.99);
        assert!(g.failures.is_empty(), "{:?}", g.failures);
    }

    #[test]
    fn non_finite_is_f_and_never_hidden() {
        let t = vec![unit(&[1.0, 0.0]), unit(&[0.0, 1.0]), unit(&[0.6, 0.8])];
        let r = reference(t.clone(), &[&[], &[], &[]]);
        let mut v = t.clone();
        v[1] = vec![f32::NAN, 0.0];
        let mut res = result(v);
        res.cases[1].vector = vec![0.0, 0.0];
        let g = grade(&r, &res, Path3::Cpu, &gates(), None, &[]);
        assert_eq!(g.letter, 'F');
        assert_eq!(g.worst, None);
        assert_eq!(g.drift, None);
        assert!(g.failures[0].contains("non-finite"));
    }

    #[test]
    fn zero_vector_is_f() {
        let t = vec![unit(&[1.0, 0.0]), unit(&[0.0, 1.0])];
        let r = reference(t.clone(), &[&[], &[]]);
        let mut v = t;
        v[1] = vec![0.0, 0.0];
        let g = grade(&r, &result(v), Path3::Ane, &gates(), None, &[]);
        assert_eq!(g.letter, 'F');
        assert!(
            g.failures[0].contains("all zeros") && g.failures[0].contains("c1"),
            "{:?}",
            g.failures
        );
    }

    #[test]
    fn invariance_gates_are_per_path_and_nan_fails() {
        let t = vec![unit(&[1.0, 0.0]), unit(&[0.0, 1.0])];
        let r = reference(t.clone(), &[&[], &[]]);
        let mut res = result(t.clone());
        res.cases[0].bucket_invariance = Check {
            n: 2,
            min: Some(0.998),
        };
        assert!(grade(&r, &res, Path3::Ane, &gates(), None, &[])
            .failures
            .is_empty());
        assert_eq!(grade(&r, &res, Path3::Cpu, &gates(), None, &[]).letter, 'F');
        res.cases[0].bucket_invariance = Check { n: 2, min: None };
        assert_eq!(grade(&r, &res, Path3::Ane, &gates(), None, &[]).letter, 'F');
        let mut res = result(t);
        res.cases[1].pad_invariance = Check {
            n: 1,
            min: Some(0.9),
        };
        let g = grade(&r, &res, Path3::Ane, &gates(), None, &[]);
        assert!(g.failures.iter().any(|f| f.contains("pad content")));
    }

    #[test]
    fn ids_mismatch_fails() {
        let t = vec![unit(&[1.0, 0.0]), unit(&[0.0, 1.0])];
        let r = reference(t.clone(), &[&[], &[]]);
        let mut res = result(t);
        res.cases[1].ids_match = false;
        let g = grade(&r, &res, Path3::Ane, &gates(), None, &[]);
        assert_eq!(g.letter, 'F');
        assert!(g.failures[0].contains("c1"));
    }

    #[test]
    fn floors_fail_without_turning_the_letter_to_f() {
        let t = vec![
            unit(&[1.0, 0.0, 0.0]),
            unit(&[0.0, 1.0, 0.0]),
            unit(&[0.0, 0.0, 1.0]),
        ];
        let r = reference(t.clone(), &[&[], &[], &[]]);
        let mut v = t.clone();
        v[0] = unit(&[1.0, 0.1, 0.0]); // cos ~0.995
        let floor = Floor {
            chip: "c".into(),
            model: "m".into(),
            path: "ane".into(),
            worst: 0.999,
            stress: None,
            drift: 1.0,
        };
        let g = grade(&r, &result(v), Path3::Ane, &gates(), Some(&floor), &[]);
        assert_eq!(g.letter, 'C');
        assert!(g.failures.is_empty());
        assert!(g.regressions[0].contains("floor"));
        let s = suggest_floor("c", "m", &g).unwrap();
        assert!(s.contains("worst = 0.99"), "{s}");
    }

    #[test]
    fn matryoshka_truncates_both_sides() {
        let t = vec![unit(&[1.0, 0.0, 0.0, 0.3]), unit(&[0.0, 1.0, 0.2, 0.0])];
        let r = reference(t.clone(), &[&[], &[]]);
        let mut v = t.clone();
        v[0] = unit(&[1.0, 0.0, 0.0, -0.3]); // differs only past dim 2
        let g = grade(&r, &result(v), Path3::Ane, &gates(), None, &[4, 2]);
        assert!((g.matryoshka[&2].unwrap() - 1.0).abs() < 1e-9);
        assert!(g.worst.unwrap() < 0.9);
    }
}
