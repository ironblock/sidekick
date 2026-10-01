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
//! - **The fp16 ceiling**, when the reference carries it (the `fp16`
//!   oracle: the model's outputs computed in ideal fp16, simulated by the
//!   conversion library): the best any fp16 path could do. Each path is
//!   also graded by its ratio to the ceiling's worst Δp, and its grade is
//!   the better of the absolute and the ratio grade. So a model is credited
//!   either for being practically exact or for being as good as fp16
//!   allows. On the fp16 paths (GPU, ANE), the bucket-invariance gate is
//!   `max(gate, ceiling max Δp)`: a lenient bound for catching bugs. A reference without the oracle grades
//!   exactly as before.

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

/// The activation a model's outputs are graded in. A reranker's score is
/// graded as a probability, sigmoid(logit), whatever its manifest serves:
/// cross-encoders are trained with a sigmoid (BCE) objective, and a raw
/// logit of magnitude ~10 against thresholds meant for probabilities would
/// grade an fp16-exact model D. Raw |Δlogit| is reported alongside, and rank
/// flips are judged on raw logits, which never saturate.
pub fn graded_problem(manifest: &ClassifierManifest) -> ProblemType {
    if manifest.task == sidekick_core::ClassifyTask::TextRanking {
        ProblemType::SingleLabel
    } else {
        manifest.problem_type
    }
}

/// The activation one case is graded in: sigmoid per label for a case whose
/// request asked for multi-label decoding, the model's graded problem
/// otherwise.
pub fn case_problem(manifest: &ClassifierManifest, case: &crate::classify_reference::ClassifyCase) -> ProblemType {
    if case.multi_label == Some(true) {
        ProblemType::MultiLabel
    } else {
        graded_problem(manifest)
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
    /// The final grade (see `final_letter`).
    pub letter: char,
    /// The grade on absolute Δp alone, as without a ceiling.
    pub absolute_letter: char,
    /// The fp16 ceiling, when the reference carries it.
    pub ceiling: Option<Ceiling>,
    /// p99 Δp over the graded cases (`None` if any is non-finite).
    pub p99_dp: Option<f64>,
    /// p99_dp / ceiling.p99, when there's a nonzero ceiling: what the ratio
    /// grade is graded on.
    pub ratio: Option<f64>,
    /// worst_dp / ceiling.max, reported only: the ceiling's single worst
    /// case varies with the fp16 implementation.
    pub max_ratio: Option<f64>,
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

/// The reference oracle with the model's ideal-fp16 outputs.
pub const CEILING_ORACLE: &str = "fp16";

/// The fp16 ceiling: the ideal-fp16 oracle's Δp against fp32 over the
/// graded (non-stress) cases.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct Ceiling {
    pub max: f64,
    pub p99: f64,
    pub mean: f64,
    /// How many Δp values it summarizes. Under 100, `p99` is the maximum.
    pub n: usize,
}

/// The ceiling for these cases, or `None` without the oracle (or with a
/// non-finite or missing value in it, which grades as no ceiling).
pub fn ceiling(
    reference: &ClassifyReference,
    problem: impl Fn(usize) -> ProblemType,
    select: impl Fn(usize) -> bool,
) -> Option<Ceiling> {
    let oracle = reference.logits.get(CEILING_ORACLE)?;
    let torch = &reference.logits["torch"];
    let mut d: Vec<f64> = Vec::new();
    for i in (0..reference.cases.len()).filter(|&i| select(i)) {
        d.push(delta_p(problem(i), &oracle[i], &torch[i], None)?);
    }
    if d.is_empty() {
        return None;
    }
    d.sort_by(f64::total_cmp);
    Some(Ceiling { max: d[d.len() - 1], p99: p99(&d), mean: d.iter().sum::<f64>() / d.len() as f64, n: d.len() })
}

/// The 99th percentile of sorted values (nearest rank): the value at rank
/// ceil(0.99·n). For fewer than 100 values it is the maximum.
pub fn p99(sorted: &[f64]) -> f64 {
    sorted[((sorted.len() as f64 * 0.99).ceil() as usize).max(1) - 1]
}

/// Grade for a path's p99 Δp as a multiple of the ceiling's p99: A ≤ 1.25×,
/// B ≤ 2×, C ≤ 4×, D beyond.
pub fn ratio_letter(ratio: f64) -> char {
    match ratio {
        r if r <= 1.25 => 'A',
        r if r <= 2.0 => 'B',
        r if r <= 4.0 => 'C',
        _ => 'D',
    }
}

/// The final grade: the better of the absolute grade and, with a ceiling,
/// the ratio grade. A graded flip still caps it at C, and a failed gate or
/// a non-finite worst case is F.
pub fn final_letter(worst_dp: Option<f64>, flips: usize, failed: bool, ratio: Option<f64>) -> char {
    let absolute = letter(worst_dp, 0, failed);
    if absolute == 'F' {
        return 'F';
    }
    let best = ratio.map_or(absolute, |r| absolute.min(ratio_letter(r)));
    if flips > 0 { best.max('C') } else { best }
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
    let problem = |i: usize| case_problem(manifest, &reference.cases[i]);
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
    let stress = |i: usize| cases[i].tags.iter().any(|t| t == STRESS_TAG);
    let ceiling = ceiling(reference, problem, |i| !stress(i));
    // fp16 paths may move across buckets as much as ideal fp16 itself
    // differs from fp32; the CPU stays exact.
    let limit = match (path, ceiling) {
        (Path3::Gpu | Path3::Ane, Some(c)) => gates.bucket_invariance.get(path).max(c.max),
        _ => gates.bucket_invariance.get(path),
    };
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
        .map(|i| finite(&result.cases[i]).and_then(|l| delta_p(problem(i), l, &torch[i], None)))
        .collect();
    let dlogit: Vec<Option<f64>> = (0..n)
        .map(|i| finite(&result.cases[i]).and_then(|l| max_abs_diff(l, &torch[i])))
        .collect();
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
    // reference's raw-logit gap is at least the margin: every activation is
    // monotonic, and a saturated sigmoid would hide real reorderings.
    let score = |l: &[f32]| l.first().copied().unwrap_or(f32::NAN) as f64;
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
            Some(finite(&result.cases[i]).and_then(|l| delta_p(problem(i), l, &torch[i], Some(t))))
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
                .map(|i| finite(&result.cases[i]).and_then(|l| delta_p(problem(i), l, &oracle[i], None)))
                .collect();
            (name.clone(), worst(&d, |_| true).0)
        })
        .collect();
    let model_only: Vec<Option<f64>> = (0..n)
        .filter_map(|i| result.cases[i].model_only.as_ref().map(|l| delta_p(problem(i), l, &torch[i], None)))
        .collect();
    let model_only_dp = (!model_only.is_empty()).then(|| worst(&model_only, |_| true).0).flatten();

    let graded: Option<Vec<f64>> = (0..n).filter(|&i| !stress(i)).map(|i| per_case[i]).collect();
    let p99_dp = graded.filter(|d| !d.is_empty()).map(|mut d| {
        d.sort_by(f64::total_cmp);
        p99(&d)
    });
    let ratio = match (p99_dp, ceiling) {
        (Some(p), Some(c)) if c.p99 > 0.0 => Some(p / c.p99),
        _ => None,
    };
    let max_ratio = match (worst_dp, ceiling) {
        (Some(w), Some(c)) if c.max > 0.0 => Some(w / c.max),
        _ => None,
    };

    let mut ms: Vec<f64> = result.cases.iter().map(|c| c.ms).collect();
    ms.sort_by(f64::total_cmp);
    let median_ms = ms.get(ms.len() / 2).copied().unwrap_or(f64::NAN);

    ClassifyGrade {
        path: path.name().into(),
        letter: final_letter(worst_dp, flips.len(), !failures.is_empty(), ratio),
        absolute_letter: letter(worst_dp, flips.len(), !failures.is_empty()),
        ceiling,
        p99_dp,
        ratio,
        max_ratio,
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

    #[test]
    fn the_better_of_the_absolute_and_ratio_grades_counts() {
        assert_eq!([0.9, 1.25, 1.3, 2.0, 2.1, 4.0, 4.1].map(ratio_letter), ['A', 'A', 'B', 'B', 'C', 'C', 'D']);
        // As bad as fp16 allows: credited by the ratio.
        assert_eq!(final_letter(Some(0.03), 0, false, Some(1.1)), 'A');
        // Practically exact against a tiny ceiling (nlptown's is ~1e-4):
        // the honest absolute B stands; the ratio never demotes.
        assert_eq!(final_letter(Some(4e-3), 0, false, Some(40.0)), 'B');
        // A graded flip still caps at C; a failed gate is F.
        assert_eq!(final_letter(Some(0.0), 1, false, Some(1.0)), 'C');
        assert_eq!(final_letter(Some(0.03), 0, true, Some(1.0)), 'F');
        assert_eq!(final_letter(None, 0, false, Some(1.0)), 'F');
        // No ceiling: exactly the absolute grade.
        assert_eq!(final_letter(Some(4e-3), 0, false, None), letter(Some(4e-3), 0, false));
    }

    /// `sample()` with an `fp16` oracle: case a's ideal-fp16 logits are
    /// [1.1, 2.0] against fp32's [1, 2] (Δp 0.0202); case b's are exact.
    fn sample_with_ceiling() -> ClassifyReference {
        let (json, _) = sample();
        assert!(json.contains(r#""oracles":["torch"]"#));
        let json = json.replace(r#""oracles":["torch"]"#, r#""oracles":["torch","fp16"]"#);
        let nan = f32::NAN;
        let bytes = |v: [f32; 8]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        let torch = bytes([1.0, 2.0, nan, nan, 0.5, 0.0, -1.0, nan]);
        let fp16 = bytes([1.1, 2.0, nan, nan, 0.5, 0.0, -1.0, nan]);
        let view = |b| safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 4], b).unwrap();
        let st = safetensors::serialize([("torch", view(&torch)), ("fp16", view(&fp16))], &None).unwrap();
        ClassifyReference::parse(&json, &st).unwrap()
    }

    #[test]
    fn a_path_as_good_as_ideal_fp16_grades_by_its_ratio() {
        let reference = sample_with_ceiling();
        let c = ceiling(&reference, |_| ProblemType::SingleLabel, |_| true).unwrap();
        assert!((c.max - 0.0202).abs() < 1e-4, "{c:?}");
        assert!((c.mean - c.max / 2.0).abs() < 1e-9, "one exact case of two");
        assert_eq!(c.p99, c.max);

        // The path matches ideal fp16 exactly: absolute D (Δp 0.0202), ratio
        // 1.0 (A). Its ANE bucket invariance (0.015) is over the 1e-3 gate
        // but within the ceiling.
        let mut a = case("a", vec![1.1, 2.0]);
        a.bucket_invariance = Delta { n: 1, max: Some(0.015) };
        let cases = vec![a, case("b", vec![0.5, 0.0, -1.0])];
        let result = ClassifyWorkerResult { model: "z".into(), path: "ane".into(), cases: cases.clone(), repeat_bitwise: true, load_ms: 0.0 };
        let g = grade(&reference, &result, Path3::Ane, &ClassifyGates::default(), &manifest());
        assert!(g.failures.is_empty(), "{:?}", g.failures);
        assert_eq!((g.absolute_letter, g.letter), ('D', 'A'));
        // Two cases: p99 is the maximum, so both ratios are 1.
        assert!((g.ratio.unwrap() - 1.0).abs() < 1e-6);
        assert!((g.max_ratio.unwrap() - 1.0).abs() < 1e-6);

        // The CPU's exact-invariance gate doesn't move.
        let result = ClassifyWorkerResult { path: "cpu".into(), ..result };
        let g = grade(&reference, &result, Path3::Cpu, &ClassifyGates::default(), &manifest());
        assert!(g.failures.iter().any(|f| f.starts_with("bucket invariance")), "{:?}", g.failures);
        assert_eq!(g.letter, 'F');
    }

    #[test]
    fn without_the_oracle_nothing_changes() {
        let (json, st) = sample();
        let reference = ClassifyReference::parse(&json, &st).unwrap();
        assert!(ceiling(&reference, |_| ProblemType::SingleLabel, |_| true).is_none());
        let result = ClassifyWorkerResult {
            model: "z".into(),
            path: "ane".into(),
            cases: vec![case("a", vec![1.1, 2.0]), case("b", vec![0.5, 0.0, -1.0])],
            repeat_bitwise: true,
            load_ms: 0.0,
        };
        let g = grade(&reference, &result, Path3::Ane, &ClassifyGates::default(), &manifest());
        assert_eq!((g.letter, g.absolute_letter, g.ratio, g.ceiling), ('D', 'D', None, None));
    }

    #[test]
    fn a_rerankers_ceiling_is_in_sigmoid_space() {
        // Raw logits 10 (fp32) vs 11 (ideal fp16): a whole logit apart, but
        // sigmoid(11) - sigmoid(10) is 2.9e-5. The ceiling is graded where
        // the score is: as a probability.
        let json = serde_json::json!({
            "format": 1, "corpus_sha256": "c", "tokenizer_sha256": "t",
            "model": {"id": "r", "task": "text-ranking", "buckets": [128], "max_seq_len": 128, "max_labels": 1, "labels": ["score"]},
            "source": {"repo": "org/r", "revision": null}, "oracles": ["torch", "fp16"],
            "cases": [{"id": "a0", "group": "a", "query": "q", "input": "x", "ids": [1], "k": 1}],
        })
        .to_string();
        let (t, f) = (10.0f32.to_le_bytes(), 11.0f32.to_le_bytes());
        let view = |b| safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1, 1], b).unwrap();
        let st = safetensors::serialize([("torch", view(&t)), ("fp16", view(&f))], &None).unwrap();
        let reference = ClassifyReference::parse(&json, &st).unwrap();
        let mut m = manifest();
        m.task = sidekick_core::ClassifyTask::TextRanking;
        m.problem_type = ProblemType::Regression;
        let c = ceiling(&reference, |_| graded_problem(&m), |_| true).unwrap();
        assert!((c.max - 2.9e-5).abs() < 1e-6, "{c:?}");
    }

    #[test]
    fn the_ratio_is_anchored_on_p99_not_on_one_worst_case() {
        // 100 two-label cases with fp32 logits [0, 0]. Ideal fp16 shifts one
        // case by 0.1 and the rest by 0.05: its max is that one case, its
        // p99 the common 0.05 shift. The path matches the common shift but
        // has its own outlier at 0.17, 1.7x the ceiling's max.
        let n = 100;
        let cases: Vec<serde_json::Value> = (0..n)
            .map(|i| serde_json::json!({"id": format!("c{i}"), "input": "x", "candidate_labels": ["p", "q"],
                                         "question_type": "choice", "ids": [1], "markers": [1], "qtype": 0, "k": 2}))
            .collect();
        let json = serde_json::json!({
            "format": 1, "corpus_sha256": "c", "tokenizer_sha256": "t",
            "model": {"id": "z", "task": "zero-shot-classification", "format": "laya",
                      "buckets": [128], "max_seq_len": 128, "max_labels": 2},
            "source": {"repo": "org/z", "revision": null}, "oracles": ["torch", "fp16"], "cases": cases,
        })
        .to_string();
        let row = |i: usize, outlier: f32, common: f32| [if i == 0 { outlier } else { common }, 0.0f32];
        let bytes = |rows: Vec<[f32; 2]>| rows.iter().flatten().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        let torch = bytes((0..n).map(|_| [0.0, 0.0]).collect());
        let fp16 = bytes((0..n).map(|i| row(i, 0.1, 0.05)).collect());
        let view = |b| safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![n, 2], b).unwrap();
        let st = safetensors::serialize([("torch", view(&torch)), ("fp16", view(&fp16))], &None).unwrap();
        let reference = ClassifyReference::parse(&json, &st).unwrap();

        let dp = |x: f32| (1.0 / (1.0 + (-x as f64).exp()) - 0.5).abs();
        let c = ceiling(&reference, |_| ProblemType::SingleLabel, |_| true).unwrap();
        assert!((c.max - dp(0.1)).abs() < 1e-6 && (c.p99 - dp(0.05)).abs() < 1e-6, "{c:?}");

        let results = (0..n).map(|i| case(&format!("c{i}"), row(i, 0.17, 0.05).to_vec())).collect();
        let result = ClassifyWorkerResult { model: "z".into(), path: "ane".into(), cases: results, repeat_bitwise: true, load_ms: 0.0 };
        let g = grade(&reference, &result, Path3::Ane, &ClassifyGates::default(), &manifest());
        assert!((g.ratio.unwrap() - 1.0).abs() < 1e-6, "p99 vs p99: {:?}", g.ratio);
        assert!((g.max_ratio.unwrap() - dp(0.17) / dp(0.1)).abs() < 1e-3, "reported: {:?}", g.max_ratio); // f32 softmax
        // Absolute D (worst Δp 0.042); the ratio grades A, and that counts.
        assert_eq!((g.absolute_letter, g.letter), ('D', 'A'));
        assert_eq!(p99(&[1.0, 2.0, 3.0]), 3.0, "fewer than 100 values: the maximum");
    }

    #[test]
    fn a_multi_label_case_is_graded_with_sigmoids() {
        // Case b ([0.5, 0, -1] in fp32) asked for multi-label decoding; the
        // path is off by 0.2 on its first logit. Softmax would spread that
        // over all three labels, but each label is its own sigmoid.
        let (json, st) = sample();
        let json = json.replacen(r#""k":3"#, r#""k":3,"multi_label":true"#, 1);
        let reference = ClassifyReference::parse(&json, &st).unwrap();
        assert_eq!(reference.cases[1].multi_label, Some(true));
        let m = manifest();
        assert_eq!(case_problem(&m, &reference.cases[0]), ProblemType::SingleLabel);
        assert_eq!(case_problem(&m, &reference.cases[1]), ProblemType::MultiLabel);

        let got = vec![0.7f32, 0.0, -1.0];
        let result = ClassifyWorkerResult {
            model: "z".into(),
            path: "ane".into(),
            cases: vec![case("a", vec![1.0, 2.0]), case("b", got.clone())],
            repeat_bitwise: true,
            load_ms: 0.0,
        };
        let g = grade(&reference, &result, Path3::Ane, &ClassifyGates::default(), &m);
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x as f64).exp());
        let want = sigmoid(0.7) - sigmoid(0.5);
        assert!((g.per_case[1].unwrap() - want).abs() < 1e-6, "{:?} vs {want}", g.per_case[1]);
        let softmax = delta_p(ProblemType::SingleLabel, &got, &[0.5, 0.0, -1.0], None).unwrap();
        assert!((softmax - want).abs() > 1e-3, "the two activations must differ here");
        assert_eq!(g.per_case[0], Some(0.0));
    }
}
