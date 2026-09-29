//! Grading a classifier path against its reference: pure, unit-tested.
//!
//! Classifiers are compared in probability space, where their outputs are
//! used: Δp is the largest |difference| in any label's probability after
//! the model's activation at temperature 1 (docs/design/classify.md).
//! - **Gates** (fail the model): finite output; token ids, markers and
//!   qtype equal to the reference's; bucket and pad invariance; repeatable
//!   output.
//! - **Graded**: raw Δp and Δlogit against the fp32 reference, and argmax
//!   flips where the reference's top-2 logit margin is at least
//!   `flip_margin`. Flips below it are near-ties, reported, not graded.
//! - **Reported only**: Δp with the manifest's calibration temperature, and
//!   accuracy against gold labels, which measures the corpus as much as
//!   the model.

use crate::classify_reference::ClassifyReference;
use crate::expect::{Path3, PerPath};
use crate::grade::STRESS_TAG;
use serde::{Deserialize, Serialize};
use sidekick_core::manifest::ClassifierManifest;
use sidekick_core::{activate, ProblemType};
use std::collections::BTreeMap;

/// `[classify_gates]` in expectations.toml. Every value is a maximum |Δp|.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ClassifyGates {
    /// A case in its own bucket vs each larger one.
    pub bucket_invariance: PerPath,
    /// Pad ids 0 vs random pad ids, same bucket.
    pub pad_invariance: f64,
    /// Two ANE runs in separate processes.
    pub determinism: f64,
    /// Reference top-2 logit margin below which an argmax flip is a
    /// near-tie (reported, not graded).
    pub flip_margin: f64,
}

impl Default for ClassifyGates {
    fn default() -> Self {
        Self {
            bucket_invariance: PerPath { cpu: 1e-5, gpu: 1e-3, ane: 1e-3 },
            pad_invariance: 1e-4,
            determinism: 1e-5,
            flip_margin: 0.05,
        }
    }
}

/// A difference measured over several runs: `n` runs, largest |Δp| (`None`
/// once any run was non-finite). `n == 0` means not applicable.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Delta {
    pub n: usize,
    pub max: Option<f64>,
}

impl Delta {
    pub fn add(&mut self, d: Option<f64>) {
        self.max = match (self.n, self.max, d) {
            (0, _, d) => d,
            (_, Some(m), Some(d)) => Some(m.max(d)),
            _ => None,
        };
        self.n += 1;
    }

    /// Within `limit`: nothing measured, or finite and at most `limit`.
    pub fn within(&self, limit: f64) -> bool {
        self.n == 0 || self.max.is_some_and(|m| m <= limit)
    }
}

/// max |a − b| elementwise; `None` for non-finite input or a length
/// mismatch.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return None;
    }
    Some(a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max))
}

/// Δp: max |Δ probability| after the activation, at `temperature`.
pub fn delta_p(problem: ProblemType, a: &[f32], b: &[f32], temperature: Option<f32>) -> Option<f64> {
    max_abs_diff(&activate(problem, a, temperature), &activate(problem, b, temperature))
}

pub fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |best, (i, &x)| if x > v[best] { i } else { best })
}

/// Gap between the largest and second-largest logit (∞ for one label).
pub fn top2_margin(v: &[f32]) -> f64 {
    let mut s: Vec<f64> = v.iter().map(|&x| x as f64).collect();
    s.sort_by(|a, b| b.total_cmp(a));
    if s.len() < 2 {
        f64::INFINITY
    } else {
        s[0] - s[1]
    }
}

/// One case as a worker ran it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyCaseResult {
    pub id: String,
    /// sidekick's ids, markers and qtype equal the reference's.
    pub ids_match: bool,
    pub bucket: usize,
    /// The product output; non-finite values replaced by 0.
    pub logits: Vec<f32>,
    pub finite: bool,
    /// When ids differ: the model's logits on the reference's inputs, to
    /// separate the input builder from numerics.
    pub model_only: Option<Vec<f32>>,
    pub bucket_invariance: Delta,
    pub pad_invariance: Delta,
    pub ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyWorkerResult {
    pub model: String,
    pub path: String,
    pub cases: Vec<ClassifyCaseResult>,
    pub repeat_bitwise: bool,
    pub load_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassifyGrade {
    pub path: String,
    pub letter: char,
    /// Worst raw Δp vs torch over non-stress cases, and where.
    pub worst_dp: Option<f64>,
    pub worst_case: String,
    pub stress_dp: Option<f64>,
    pub mean_dp: f64,
    pub worst_dlogit: Option<f64>,
    /// Argmax changes where the reference's top-2 margin ≥ flip_margin.
    pub flips: Vec<String>,
    /// Argmax changes below it (reported only).
    pub near_ties: Vec<String>,
    /// Worst Δp with the manifest's temperature, over cases that have one.
    pub calibrated_dp: Option<f64>,
    /// Gold-label accuracy of this path and of the reference, over cases
    /// with gold labels: (correct, total).
    pub gold: Option<(usize, usize)>,
    pub reference_gold: Option<(usize, usize)>,
    pub bucket_invariance: Delta,
    pub pad_invariance: Delta,
    pub by_tag: BTreeMap<String, Option<f64>>,
    /// Worst Δp of each other oracle vs this path (report only).
    pub vs_oracles: BTreeMap<String, Option<f64>>,
    /// Worst Δp on the reference's own inputs, for cases whose ids differ.
    pub model_only_dp: Option<f64>,
    pub median_ms: f64,
    /// Raw Δp per case, in reference order.
    pub per_case: Vec<Option<f64>>,
    pub failures: Vec<String>,
}

/// Documentation vocabulary for a worst raw Δp (never gated): A ≤ 1e-3,
/// B ≤ 5e-3, C ≤ 2e-2, D above; a graded flip caps the grade at C; F for
/// non-finite output or a failed gate.
pub fn letter(worst_dp: Option<f64>, flips: usize, failed: bool) -> char {
    let l = match worst_dp {
        _ if failed => return 'F',
        None => return 'F',
        Some(d) if d <= 1e-3 => 'A',
        Some(d) if d <= 5e-3 => 'B',
        Some(d) if d <= 2e-2 => 'C',
        Some(_) => 'D',
    };
    if flips > 0 { l.max('C') } else { l }
}

/// Worst of the selected per-case values (poisoned by a non-finite one),
/// and its index.
fn worst(values: &[Option<f64>], select: impl Fn(usize) -> bool) -> (Option<f64>, Option<usize>) {
    let mut out: (Option<f64>, Option<usize>) = (Some(0.0), None);
    for (i, v) in values.iter().enumerate().filter(|(i, _)| select(*i)) {
        match (out.0, v) {
            (Some(w), Some(d)) if out.1.is_none() || *d > w => out = (Some(*d), Some(i)),
            (Some(_), Some(_)) => {}
            (_, None) => return (None, Some(i)),
            (None, _) => unreachable!(),
        }
    }
    out
}

pub fn grade(
    reference: &ClassifyReference,
    result: &ClassifyWorkerResult,
    path: Path3,
    gates: &ClassifyGates,
    manifest: &ClassifierManifest,
) -> ClassifyGrade {
    let problem = manifest.problem_type;
    let torch = &reference.logits["torch"];
    let cases = &reference.cases;
    let mut failures = Vec::new();

    let n = result.cases.len().min(cases.len());
    if result.cases.len() != cases.len() {
        failures.push(format!("{} results for {} cases", result.cases.len(), cases.len()));
    }
    let non_finite: Vec<&str> =
        result.cases.iter().filter(|c| !c.finite).map(|c| c.id.as_str()).collect();
    if !non_finite.is_empty() {
        failures.push(format!("non-finite output on {}", non_finite.join(", ")));
    }
    let ids_differ: Vec<&str> =
        result.cases.iter().filter(|c| !c.ids_match).map(|c| c.id.as_str()).collect();
    if !ids_differ.is_empty() {
        failures.push(format!(
            "token ids, markers or qtype differ from the reference on {} case(s): {}",
            ids_differ.len(),
            ids_differ.join(", ")
        ));
    }
    let mut bucket_invariance = Delta::default();
    let mut pad_invariance = Delta::default();
    for c in &result.cases {
        merge(&mut bucket_invariance, c.bucket_invariance);
        merge(&mut pad_invariance, c.pad_invariance);
    }
    let limit = gates.bucket_invariance.get(path);
    if !bucket_invariance.within(limit) {
        failures.push(format!("bucket invariance Δp {} > {limit}", fmt(bucket_invariance.max)));
    }
    if !pad_invariance.within(gates.pad_invariance) {
        failures.push(format!("pad invariance Δp {} > {}", fmt(pad_invariance.max), gates.pad_invariance));
    }
    if !result.repeat_bitwise {
        failures.push("output changes when re-run in the same process".into());
    }

    fn finite(c: &ClassifyCaseResult) -> Option<&[f32]> {
        c.finite.then_some(c.logits.as_slice())
    }
    let per_case: Vec<Option<f64>> = (0..n)
        .map(|i| finite(&result.cases[i]).and_then(|l| delta_p(problem, l, &torch[i], None)))
        .collect();
    let dlogit: Vec<Option<f64>> = (0..n)
        .map(|i| finite(&result.cases[i]).and_then(|l| max_abs_diff(l, &torch[i])))
        .collect();
    let stress = |i: usize| cases[i].tags.iter().any(|t| t == STRESS_TAG);
    let (worst_dp, worst_at) = worst(&per_case, |i| !stress(i));
    // `None` when there are no stress cases, as when one is non-finite;
    // the report tells the two apart by the tags.
    let (stress_dp, stress_at) = worst(&per_case, stress);
    let stress_dp = stress_at.and(stress_dp);
    let finite_dp: Vec<f64> = per_case.iter().flatten().copied().collect();
    let mean_dp = if finite_dp.is_empty() { f64::NAN } else { finite_dp.iter().sum::<f64>() / finite_dp.len() as f64 };

    let mut flips = Vec::new();
    let mut near_ties = Vec::new();
    for i in 0..n {
        let c = &result.cases[i];
        if !c.finite || torch[i].len() < 2 || argmax(&c.logits) == argmax(&torch[i]) {
            continue;
        }
        let labels = cases[i].labels(&manifest.classify.labels);
        let label = |j: usize| labels.get(j).cloned().unwrap_or_else(|| j.to_string());
        let what = format!(
            "{}: {} → {} (margin {:.4})",
            cases[i].id,
            label(argmax(&torch[i])),
            label(argmax(&c.logits)),
            top2_margin(&torch[i])
        );
        if top2_margin(&torch[i]) >= gates.flip_margin {
            flips.push(what);
        } else {
            near_ties.push(what);
        }
    }

    // Rerankers: one score per case, so the flips that matter are order
    // changes between documents of one query. A pair counts where the
    // reference's score gap (after the activation) is at least the margin.
    let score = |l: &[f32]| activate(problem, l, None).first().copied().unwrap_or(f32::NAN) as f64;
    for i in 0..n {
        for j in i + 1..n {
            let (Some(gi), Some(gj)) = (&cases[i].group, &cases[j].group) else { continue };
            let (ci, cj) = (&result.cases[i], &result.cases[j]);
            if gi != gj || !ci.finite || !cj.finite || torch[i].len() != 1 {
                continue;
            }
            let gap = score(&torch[i]) - score(&torch[j]);
            let now = score(&ci.logits) - score(&cj.logits);
            // Same order, or no reference order to keep. A reference gap
            // that sidekick collapses to an exact tie counts: the order
            // is lost.
            if gap == 0.0 || gap * now > 0.0 {
                continue;
            }
            let what = format!("{gi}: {} vs {} reordered (reference gap {:.4})", cases[i].id, cases[j].id, gap.abs());
            if gap.abs() >= gates.flip_margin {
                flips.push(what);
            } else {
                near_ties.push(what);
            }
        }
    }

    let calibrated: Vec<Option<f64>> = (0..n)
        .filter_map(|i| {
            let t = manifest.temperature(cases[i].question_type, cases[i].k)?;
            Some(finite(&result.cases[i]).and_then(|l| delta_p(problem, l, &torch[i], Some(t))))
        })
        .collect();
    let calibrated_dp = (!calibrated.is_empty()).then(|| worst(&calibrated, |_| true).0).flatten();

    let gold_accuracy = |logits: &dyn Fn(usize) -> Option<Vec<f32>>| {
        let mut correct = 0;
        let mut total = 0;
        for (i, case) in cases.iter().enumerate().take(n) {
            let Some(gold) = case.gold.as_ref().filter(|g| !g.is_empty()) else { continue };
            total += 1;
            let labels = case.labels(&manifest.classify.labels);
            if let Some(l) = logits(i) {
                correct += usize::from(labels.get(argmax(&l)).is_some_and(|p| gold.contains(p)));
            }
        }
        (total > 0).then_some((correct, total))
    };
    let gold = gold_accuracy(&|i| finite(&result.cases[i]).map(<[f32]>::to_vec));
    let reference_gold = gold_accuracy(&|i| Some(torch[i].clone()));

    let mut by_tag: BTreeMap<String, Option<f64>> = BTreeMap::new();
    for (i, case) in cases.iter().enumerate().take(n) {
        for tag in &case.tags {
            let e = by_tag.entry(tag.clone()).or_insert(Some(0.0));
            *e = match (*e, per_case[i]) {
                (Some(a), Some(b)) => Some(a.max(b)),
                _ => None,
            };
        }
    }
    let vs_oracles = reference
        .logits
        .iter()
        .filter(|(name, _)| *name != "torch")
        .map(|(name, oracle)| {
            let d: Vec<Option<f64>> = (0..n)
                .map(|i| finite(&result.cases[i]).and_then(|l| delta_p(problem, l, &oracle[i], None)))
                .collect();
            (name.clone(), worst(&d, |_| true).0)
        })
        .collect();
    let model_only: Vec<Option<f64>> = (0..n)
        .filter_map(|i| result.cases[i].model_only.as_ref().map(|l| delta_p(problem, l, &torch[i], None)))
        .collect();
    let model_only_dp = (!model_only.is_empty()).then(|| worst(&model_only, |_| true).0).flatten();

    let mut ms: Vec<f64> = result.cases.iter().map(|c| c.ms).collect();
    ms.sort_by(f64::total_cmp);
    let median_ms = ms.get(ms.len() / 2).copied().unwrap_or(f64::NAN);

    ClassifyGrade {
        path: path.name().into(),
        letter: letter(worst_dp, flips.len(), !failures.is_empty()),
        worst_dp,
        worst_case: worst_at.map(|i| cases[i].id.clone()).unwrap_or_default(),
        stress_dp,
        mean_dp,
        worst_dlogit: worst(&dlogit, |_| true).0,
        flips,
        near_ties,
        calibrated_dp,
        gold,
        reference_gold,
        bucket_invariance,
        pad_invariance,
        by_tag,
        vs_oracles,
        model_only_dp,
        median_ms,
        per_case,
        failures,
    }
}

fn merge(into: &mut Delta, other: Delta) {
    if other.n == 0 {
        return;
    }
    if into.n == 0 {
        *into = other;
        return;
    }
    into.max = match (into.max, other.max) {
        (Some(a), Some(b)) => Some(a.max(b)),
        _ => None,
    };
    into.n += other.n;
}

pub fn fmt(x: Option<f64>) -> String {
    match x {
        Some(v) => format!("{v:.2e}"),
        None => "non-finite".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify_reference::tests::sample;
    use sidekick_core::manifest::{ClassifierIo, ClassifyFormat, ClassifySection};
    use sidekick_core::ClassifyTask;

    fn manifest() -> ClassifierManifest {
        ClassifierManifest {
            id: "z".into(),
            task: ClassifyTask::ZeroShotClassification,
            source: None,
            artifact: "m".into(),
            tokenizer: "t".into(),
            buckets: vec![128],
            max_seq_len: 128,
            max_batch: 1,
            problem_type: ProblemType::SingleLabel,
            classify: ClassifySection {
                format: Some(ClassifyFormat::Laya),
                max_labels: Some(4),
                calibration: [("noul:2".to_string(), 2.0)].into(),
                io: ClassifierIo::default(),
                ..Default::default()
            },
        }
    }

    fn case(id: &str, logits: Vec<f32>) -> ClassifyCaseResult {
        ClassifyCaseResult {
            id: id.into(),
            ids_match: true,
            bucket: 128,
            finite: logits.iter().all(|x| x.is_finite()),
            logits,
            model_only: None,
            bucket_invariance: Delta { n: 1, max: Some(0.0) },
            pad_invariance: Delta { n: 1, max: Some(0.0) },
            ms: 1.0,
        }
    }

    fn run(cases: Vec<ClassifyCaseResult>) -> ClassifyGrade {
        let (json, st) = sample();
        let reference = ClassifyReference::parse(&json, &st).unwrap();
        let result = ClassifyWorkerResult { model: "z".into(), path: "ane".into(), cases, repeat_bitwise: true, load_ms: 0.0 };
        grade(&reference, &result, Path3::Ane, &ClassifyGates::default(), &manifest())
    }

    #[test]
    fn exact_output_grades_a_and_reports_gold_and_calibration() {
        // Reference logits: a = [1, 2] (gold q), b = [0.5, 0, -1].
        let g = run(vec![case("a", vec![1.0, 2.0]), case("b", vec![0.5, 0.0, -1.0])]);
        assert!(g.failures.is_empty(), "{:?}", g.failures);
        assert_eq!(g.letter, 'A');
        assert_eq!(g.worst_dp, Some(0.0));
        assert_eq!(g.gold, Some((1, 1)));
        assert_eq!(g.reference_gold, Some((1, 1)));
        // Only case a (noul, k = 2) has a temperature.
        assert_eq!(g.calibrated_dp, Some(0.0));
    }

    #[test]
    fn a_confident_flip_is_graded_and_a_near_tie_only_reported() {
        // Case a's reference margin is 1.0: a flip there is graded.
        let g = run(vec![case("a", vec![2.0, 1.0]), case("b", vec![0.5, 0.0, -1.0])]);
        assert_eq!(g.flips.len(), 1, "{:?}", g.flips);
        assert!(g.flips[0].starts_with("a: q → p"), "{:?}", g.flips);
        assert!(g.letter >= 'C');
        assert_eq!(g.gold, Some((0, 1)));

        // Case b's reference margin is 0.5; make a reference-like near tie
        // by grading against a tighter margin gate instead.
        let (json, st) = sample();
        let reference = ClassifyReference::parse(&json, &st).unwrap();
        let result = ClassifyWorkerResult {
            model: "z".into(),
            path: "ane".into(),
            cases: vec![case("a", vec![1.0, 2.0]), case("b", vec![0.0, 0.5, -1.0])],
            repeat_bitwise: true,
            load_ms: 0.0,
        };
        let gates = ClassifyGates { flip_margin: 0.6, ..Default::default() };
        let g = grade(&reference, &result, Path3::Ane, &gates, &manifest());
        assert!(g.flips.is_empty());
        assert_eq!(g.near_ties.len(), 1);
    }

    #[test]
    fn gates_fail_non_finite_ids_and_invariance() {
        let mut b = case("b", vec![0.5, 0.0, f32::NAN]);
        b.ids_match = false;
        b.pad_invariance = Delta { n: 1, max: Some(0.5) };
        let mut a = case("a", vec![1.0, 2.0]);
        a.bucket_invariance = Delta { n: 1, max: Some(0.01) };
        let g = run(vec![a, b]);
        assert_eq!(g.letter, 'F');
        let all = g.failures.join(" | ");
        for want in ["non-finite output on b", "differ from the reference on 1 case(s): b", "bucket invariance", "pad invariance"] {
            assert!(all.contains(want), "{all}");
        }
        assert_eq!(g.worst_dp, None, "a non-finite case poisons the worst");
    }

    #[test]
    fn letters_follow_the_thresholds() {
        assert_eq!(letter(Some(1e-3), 0, false), 'A');
        assert_eq!(letter(Some(4e-3), 0, false), 'B');
        assert_eq!(letter(Some(1e-2), 0, false), 'C');
        assert_eq!(letter(Some(0.1), 0, false), 'D');
        assert_eq!(letter(Some(0.0), 1, false), 'C');
        assert_eq!(letter(Some(0.1), 1, false), 'D');
        assert_eq!(letter(Some(0.0), 0, true), 'F');
        assert_eq!(letter(None, 0, false), 'F');
    }

    #[test]
    fn delta_helpers() {
        assert_eq!(max_abs_diff(&[1.0, 2.0], &[1.5, 2.0]), Some(0.5));
        assert_eq!(max_abs_diff(&[1.0], &[f32::NAN]), None);
        assert_eq!(top2_margin(&[0.1, 3.0, 2.5]), 0.5);
        assert_eq!(argmax(&[0.1, 3.0, 2.5]), 1);
        let mut d = Delta::default();
        assert!(d.within(0.0));
        d.add(Some(1e-4));
        d.add(Some(2e-4));
        assert_eq!(d.max, Some(2e-4));
        d.add(None);
        assert!(!d.within(1.0));
    }

    #[test]
    fn rerank_flips_count_within_a_group_above_the_margin() {
        let json = serde_json::json!({
            "format": 1, "corpus_sha256": "c", "tokenizer_sha256": "t",
            "model": {"id": "r", "task": "text-ranking", "buckets": [128], "max_seq_len": 128, "max_labels": 1, "labels": ["score"]},
            "source": {"repo": "org/r", "revision": null}, "oracles": ["torch"],
            "cases": [
                {"id": "a0", "group": "a", "query": "q", "input": "x", "ids": [1], "k": 1},
                {"id": "a1", "group": "a", "query": "q", "input": "y", "ids": [1], "k": 1},
                {"id": "a2", "group": "a", "query": "q", "input": "z", "ids": [1], "k": 1},
                {"id": "b0", "group": "b", "query": "p", "input": "x", "ids": [1], "k": 1},
            ],
        })
        .to_string();
        // Raw scores (regression): a0 3.0, a1 1.0, a2 1.02, b0 0.0.
        let data: Vec<u8> = [3.0f32, 1.0, 1.02, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4, 1], &data).unwrap();
        let st = safetensors::serialize([("torch", view)], &None).unwrap();
        let reference = ClassifyReference::parse(&json, &st).unwrap();
        let mut m = manifest();
        m.task = sidekick_core::ClassifyTask::TextRanking;
        m.problem_type = ProblemType::Regression;
        m.classify.labels = vec!["score".into()];
        // a1 and a2 swap (gap 0.02: a near tie); a0 falls below a2 (gap
        // 1.98: a flip); b0 is another query, never compared with a.
        let got = [0.9f32, 1.05, 1.0, 5.0];
        // (Below: a0 tied with a1 exactly also loses a0's order.)
        let cases = reference.cases.iter().zip(got).map(|(c, s)| case(&c.id, vec![s])).collect();
        let result = ClassifyWorkerResult { model: "r".into(), path: "ane".into(), cases, repeat_bitwise: true, load_ms: 0.0 };
        let g = grade(&reference, &result, Path3::Ane, &ClassifyGates::default(), &m);
        assert_eq!(g.flips.len(), 2, "{:?}", g.flips);
        assert!(g.flips.iter().all(|f| f.starts_with("a: a0 vs")), "{:?}", g.flips);
        assert_eq!(g.near_ties.len(), 1, "{:?}", g.near_ties);
        assert!(g.near_ties[0].starts_with("a: a1 vs a2"));
        assert!(g.letter >= 'C');

        // A 2.0 reference gap collapsed to an exact tie is a flip.
        let tied = [1.0f32, 1.0, 1.02, 5.0];
        let cases = reference.cases.iter().zip(tied).map(|(c, s)| case(&c.id, vec![s])).collect();
        let result = ClassifyWorkerResult { model: "r".into(), path: "ane".into(), cases, repeat_bitwise: true, load_ms: 0.0 };
        let g = grade(&reference, &result, Path3::Ane, &ClassifyGates::default(), &m);
        assert!(g.flips.iter().any(|f| f.starts_with("a: a0 vs a1")), "{:?}", g.flips);
    }
}
