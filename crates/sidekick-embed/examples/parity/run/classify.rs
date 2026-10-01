//! Classifiers in the live suite: the same structure as embedders (compute
//! plans before any prediction, one worker process per compute path, a
//! second ANE process for determinism), graded in probability space by
//! `classify_grade`.

use super::{
    bitwise_eq, check_plans, spawn_worker, units, BucketInfo, BucketInvariance, Options, XorShift,
    DETERMINISM_CASES,
};
use crate::classify_grade::{
    delta_p, fmt, grade, BucketPair, ClassifyCaseResult, ClassifyGrade, ClassifyWorkerResult, Delta,
};
use crate::classify_reference::{corpus_sha256, tokenizer_sha, ClassifyCase, ClassifyReference};
use crate::expect::{Expectations, Path3};
use serde::Serialize;
use sidekick_core::manifest::ResolvedClassifier;
use sidekick_core::{Classifier, Prepared};
use sidekick_embed::CoremlClassifier;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub fn reference_dir(model: &ResolvedClassifier, refs: Option<&Path>) -> PathBuf {
    match refs {
        Some(r) => r.join(&model.manifest.id),
        None => model.dir.join("parity"),
    }
}

/// One classifier on one compute path: every reference case through the
/// product path (`prepare` + `run`), plus the invariance checks.
pub fn worker(
    model: &ResolvedClassifier,
    refs: Option<&Path>,
    path: Path3,
    out: &str,
    limit: Option<usize>,
    bucket_mode: BucketInvariance,
) -> Result<(), String> {
    let reference = ClassifyReference::load(&reference_dir(model, refs))?;
    let vocab = tokenizers::Tokenizer::from_file(model.tokenizer_path())
        .map_err(|e| e.to_string())?
        .get_vocab_size(true) as u64;

    let t0 = Instant::now();
    let clf = CoremlClassifier::load_with(model, units(path)).map_err(|e| e.to_string())?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let buckets = clf.buckets().to_vec();
    let cases = &reference.cases[..limit.unwrap_or(reference.cases.len()).min(reference.cases.len())];
    let full = limit.is_none();
    let mut rng = XorShift(0x5eed_1234_abcd_0002);
    let coverage = BucketCoverage::new(bucket_mode, &buckets);

    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let e = |e: sidekick_core::Error| format!("case {}: {e}", case.id);
        let problem = crate::classify_grade::case_problem(&model.manifest, case);
        let max = model.manifest.max_seq_len;
        let self_truncating = model.manifest.classify.format.is_some();
        let prepared = match &case.query {
            Some(query) => clf.prepare_pair(query, &case.input, &case.pair_params(max)),
            None => clf.prepare(&case.input, &case.params(max, self_truncating)),
        }
        .map_err(e)?;
        let t = Instant::now();
        let logits = clf.run(&prepared).map_err(e)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;

        // Segment ids count only where the model takes them: an XLM-R
        // reference may list all-zero ones that sidekick never builds.
        let segments = model.manifest.classify.io.token_type_ids.is_some();
        let ids_match = prepared.ids == case.ids
            && (!segments || prepared.type_ids == case.type_ids)
            && prepared.markers == case.markers
            && prepared.qtype == case.qtype;
        let model_only = if ids_match {
            None
        } else {
            let bucket = *buckets
                .iter()
                .find(|&&b| b >= case.ids.len())
                .ok_or("reference ids exceed buckets")?;
            let theirs = Prepared {
                ids: case.ids.clone(),
                type_ids: case.type_ids.clone(),
                markers: case.markers.clone(),
                qtype: case.qtype,
                bucket,
                decide_pos: None,
            };
            Some(clf.run(&theirs).map_err(e)?)
        };

        let mut bucket_invariance = Delta::default();
        let mut bucket_pairs = Vec::new();
        let mut pad_invariance = Delta::default();
        if full {
            for b in coverage.larger(&buckets, prepared.bucket, results.len(), &case.tags) {
                let other = clf.run_in(&prepared, b, &[]).map_err(e)?;
                let dp = delta_p(problem, &logits, &other, None);
                bucket_invariance.add(dp);
                bucket_pairs.push(BucketPair { from: prepared.bucket, to: b, dp });
            }
            let pads = prepared.bucket - prepared.ids.len();
            if pads > 0 {
                let pad_ids: Vec<i32> = (0..pads).map(|_| (rng.next() % vocab) as i32).collect();
                let other = clf.run_in(&prepared, prepared.bucket, &pad_ids).map_err(e)?;
                pad_invariance.add(delta_p(problem, &logits, &other, None));
            }
        }

        let finite = logits.iter().all(|x| x.is_finite());
        results.push(ClassifyCaseResult {
            id: case.id.clone(),
            ids_match,
            bucket: prepared.bucket,
            logits: logits.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect(),
            finite,
            model_only,
            bucket_invariance,
            bucket_pairs,
            pad_invariance,
            ms,
        });
    }

    // After every bucket has run: the first cases again, bit for bit.
    let mut repeat_bitwise = true;
    for (case, first) in cases.iter().zip(&results).take(5) {
        let max = model.manifest.max_seq_len;
        let self_truncating = model.manifest.classify.format.is_some();
        let again = match &case.query {
            Some(query) => clf.prepare_pair(query, &case.input, &case.pair_params(max)),
            None => clf.prepare(&case.input, &case.params(max, self_truncating)),
        }
        .and_then(|p| clf.run(&p))
        .map_err(|e| e.to_string())?;
        let again: Vec<f32> = again.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect();
        repeat_bitwise &= bitwise_eq(&first.logits, &again);
    }

    let result = ClassifyWorkerResult {
        model: model.manifest.id.clone(),
        path: path.name().into(),
        bucket_coverage: Some(coverage.describe(bucket_mode, &buckets, cases)),
        cases: results,
        repeat_bitwise,
        load_ms,
    };
    std::fs::write(out, serde_json::to_vec(&result).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

/// Which larger buckets a case re-runs in for bucket invariance. Every
/// bucket is its own artifact (D15), and a mask or position defect shows in
/// any bucket larger than the case's, so `Sampled` keeps both kinds of
/// coverage at a fraction of the cost: every case in its next larger bucket
/// (every bucket but the smallest sees every case that falls just below
/// it), and every larger bucket for the adversarial cases and a
/// deterministic every-`SAMPLE_EVERY`th case, so each bucket also sees
/// inputs of every length below it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketCoverage {
    Full,
    Sampled,
}

/// The sampled stride: every 10th case re-runs in every larger bucket.
pub const SAMPLE_EVERY: usize = 10;

/// Above this largest bucket, `auto` samples: a larger bucket's prediction
/// costs about its length's square on a transformer, and at 2,048 tokens
/// re-running every case in every larger bucket takes hours.
pub const SAMPLE_ABOVE: usize = 512;

impl BucketCoverage {
    pub fn new(mode: BucketInvariance, buckets: &[usize]) -> Self {
        match mode {
            BucketInvariance::Full => Self::Full,
            BucketInvariance::Sampled => Self::Sampled,
            BucketInvariance::Auto if buckets.last().is_some_and(|&b| b > SAMPLE_ABOVE) => Self::Sampled,
            BucketInvariance::Auto => Self::Full,
        }
    }

    /// Whether case `index` (with `tags`) re-runs in every larger bucket.
    fn every_bucket(self, index: usize, tags: &[String]) -> bool {
        self == Self::Full || index.is_multiple_of(SAMPLE_EVERY) || tags.iter().any(|t| t == "adversarial")
    }

    /// The larger buckets case `index`, prepared for `bucket`, re-runs in.
    pub fn larger(self, buckets: &[usize], bucket: usize, index: usize, tags: &[String]) -> Vec<usize> {
        let mut larger = buckets.iter().copied().filter(|&b| b > bucket);
        if self.every_bucket(index, tags) {
            larger.collect()
        } else {
            larger.next().into_iter().collect()
        }
    }

    /// The report line: what was run, and why (`mode` as asked).
    pub fn describe(self, mode: BucketInvariance, buckets: &[usize], cases: &[ClassifyCase]) -> String {
        match self {
            Self::Full => "every larger bucket for every case".into(),
            Self::Sampled => {
                let n = cases.iter().enumerate().filter(|(i, c)| self.every_bucket(*i, &c.tags)).count();
                let why = match mode {
                    BucketInvariance::Auto => format!(
                        "the largest bucket, {}, is over {SAMPLE_ABOVE}",
                        buckets.last().copied().unwrap_or(0)
                    ),
                    _ => "--bucket-invariance sampled".into(),
                };
                format!(
                    "sampled ({why}): the next larger bucket for every case; every larger bucket for {n} of \
                     {} cases (the adversarial ones and every {SAMPLE_EVERY}th)",
                    cases.len()
                )
            }
        }
    }
}

#[derive(Serialize)]
pub struct ClassifierReport {
    id: String,
    task: String,
    source: String,
    buckets: Vec<BucketInfo>,
    paths: Vec<ClassifyGrade>,
    crashed: Vec<(String, String)>,
    /// Largest |Δp| between two ANE processes.
    determinism: Option<Delta>,
    /// Which larger buckets the bucket-invariance gate re-ran cases in.
    bucket_coverage: Option<String>,
    failures: Vec<String>,
    warnings: Vec<String>,
}

/// Grade one classifier. `None` when it has no reference and wasn't asked
/// for by name (skipped, as embedders are); otherwise the report and
/// whether it passed.
pub fn grade_model(
    model: &ResolvedClassifier,
    models_dir: &Path,
    o: &Options,
    expectations: &Expectations,
    scratch: &Path,
) -> Option<(ClassifierReport, bool)> {
    let m = &model.manifest;
    let id = &m.id;
    let gates = &expectations.classify_gates;
    let refs_dir = reference_dir(model, o.refs.as_deref());
    println!("\n=== {id} ({})", serde_json::to_value(m.task).ok()?.as_str()?);
    let mut report = ClassifierReport {
        id: id.clone(),
        task: serde_json::to_value(m.task).ok()?.as_str()?.to_string(),
        source: String::new(),
        buckets: Vec::new(),
        paths: Vec::new(),
        crashed: Vec::new(),
        determinism: None,
        bucket_coverage: None,
        failures: Vec::new(),
        warnings: Vec::new(),
    };
    let reference = match ClassifyReference::load(&refs_dir) {
        Ok(r) => r,
        Err(e) if !refs_dir.join("reference.json").exists() && o.models.is_empty() => {
            println!("  SKIPPED: no reference ({e}); generate one with the classifier reference generator");
            return None;
        }
        Err(e) => {
            println!("  FAIL: unusable reference: {e}");
            report.failures.push(format!("unusable reference: {e}"));
            return Some((report, false));
        }
    };
    report.source = reference.source_label();
    let stale = match corpus_sha256(m) {
        Ok(corpus) => reference.stale(m, &corpus, &tokenizer_sha(&model.tokenizer_path())),
        Err(e) => Some(format!("no corpus to check the reference against: {e}")),
    };
    if let Some(why) = stale {
        println!("  FAIL: {why}");
        report.failures.push(why);
        return Some((report, false));
    }
    let version = |p: &str| reference.versions.get(p).cloned().flatten().unwrap_or("?".into());
    println!(
        "  reference: {}, {} cases, oracles {}; torch {}, transformers {}",
        reference.source_label(),
        reference.cases.len(),
        reference.oracles.join(", "),
        version("torch"),
        version("transformers"),
    );

    let plans = check_plans(
        id,
        &m.buckets,
        |b| model.artifact_path_for_bucket(b),
        m.compute_units,
        o,
        scratch,
        &mut report.failures,
        &mut report.warnings,
    );
    report.buckets = plans.buckets;
    if !plans.ok {
        println!("  FAIL: not ANE-eligible; skipping predictions (such an artifact can abort at predict)");
        return Some((report, false));
    }

    let mut results: Vec<(Path3, ClassifyWorkerResult)> = Vec::new();
    for &path in &o.paths {
        match spawn_worker::<ClassifyWorkerResult>(id, models_dir, path, None, scratch, o) {
            Ok(r) => results.push((path, r)),
            Err(e) => {
                println!("  {}: CRASHED: {e}", path.name());
                report.crashed.push((path.name().into(), e.clone()));
                report.failures.push(format!("{}: worker failed: {e}", path.name()));
            }
        }
    }

    report.bucket_coverage = results.iter().find_map(|(_, r)| r.bucket_coverage.clone());

    // Determinism across processes: a second, independent ANE load.
    if let Some((_, first)) = results.iter().find(|(p, _)| *p == Path3::Ane) {
        match spawn_worker::<ClassifyWorkerResult>(id, models_dir, Path3::Ane, Some(DETERMINISM_CASES), scratch, o) {
            Ok(second) => {
                let mut d = Delta::default();
                for ((a, b), case) in first.cases.iter().zip(&second.cases).zip(&reference.cases) {
                    d.add(if a.finite && b.finite {
                        delta_p(crate::classify_grade::case_problem(m, case), &a.logits, &b.logits, None)
                    } else {
                        None
                    });
                }
                if !d.within(gates.determinism) {
                    report.failures.push(format!(
                        "ANE output differs between processes: Δp {} > {}",
                        fmt(d.max),
                        gates.determinism
                    ));
                }
                report.determinism = Some(d);
            }
            Err(e) => report.failures.push(format!("ane (second process): worker failed: {e}")),
        }
    }

    let grades: Vec<ClassifyGrade> =
        results.iter().map(|(p, r)| grade(&reference, r, *p, gates, m)).collect();

    // As for embedders: a bucket graded without a compute plan must show it
    // ran somewhere other than the CPU.
    let cpu = results.iter().find(|(p, _)| *p == Path3::Cpu).map(|(_, r)| r);
    let ane = results.iter().find(|(p, _)| *p == Path3::Ane).map(|(_, r)| r);
    for &b in &plans.unverified {
        let (Some(c), Some(a)) = (cpu, ane) else {
            report.failures.push(format!(
                "bucket {b}: without a compute plan, grading needs both the cpu and ane paths \
                 to rule out a CPU fallback"
            ));
            continue;
        };
        let in_bucket: Vec<usize> = (0..a.cases.len()).filter(|&i| a.cases[i].bucket == b).collect();
        if in_bucket.is_empty() {
            report.failures.push(format!("bucket {b}: no compute plan and no case to check it with"));
        } else if in_bucket.iter().all(|&i| bitwise_eq(&a.cases[i].logits, &c.cases[i].logits)) {
            report.failures.push(format!(
                "bucket {b}: no compute plan, and its ANE output is bit-identical to CPU_ONLY: \
                 it runs on the CPU"
            ));
        }
    }
    if let (Some(c), Some(a)) = (cpu, ane) {
        if c.cases.iter().zip(&a.cases).all(|(x, y)| bitwise_eq(&x.logits, &y.logits)) {
            report.warnings.push(
                "ANE output is bit-identical to CPU_ONLY: the model may be running entirely on the CPU".into(),
            );
        }
    }

    print_grades(&reference, &grades, &report);
    for g in &grades {
        report.failures.extend(g.failures.iter().map(|f| format!("{}: {f}", g.path)));
    }
    for f in &report.failures {
        println!("  FAIL: {f}");
    }
    for w in &report.warnings {
        println!("  warning: {w}");
    }
    let pass = report.failures.is_empty();
    if pass {
        println!("  PASS");
    }
    report.paths = grades;
    Some((report, pass))
}

fn accuracy(x: Option<(usize, usize)>) -> String {
    match x {
        Some((c, n)) => format!("{c}/{n} ({:.1}%)", 100.0 * c as f64 / n as f64),
        None => "-".into(),
    }
}

fn print_grades(reference: &ClassifyReference, grades: &[ClassifyGrade], report: &ClassifierReport) {
    if grades.is_empty() {
        return;
    }
    let has_stress = reference.cases.iter().any(|c| c.tags.iter().any(|t| t == crate::grade::STRESS_TAG));
    println!(
        "  {:<4} {:<5} {:>10} {:<24} {:>10} {:>10} {:>10} {:>5} {:>5} {:>10} {:>10} {:>10} {:>8}",
        "path", "grade", "worst Δp", "(case)", "stress", "mean Δp", "Δlogit", "flips", "ties",
        "calib Δp", "buckets", "pads", "ms/case"
    );
    let inv = |d: &Delta| if d.n == 0 { "-".to_string() } else { fmt(d.max) };
    for g in grades {
        println!(
            "  {:<4} {:<5} {:>10} {:<24} {:>10} {:>10.2e} {:>10} {:>5} {:>5} {:>10} {:>10} {:>10} {:>8.1}",
            g.path,
            g.letter,
            fmt(g.worst_dp),
            format!("({})", g.worst_case),
            if has_stress { fmt(g.stress_dp) } else { "-".into() },
            g.mean_dp,
            fmt(g.worst_dlogit),
            g.flips.len(),
            g.near_ties.len(),
            g.calibrated_dp.map_or("-".into(), |d| fmt(Some(d))),
            inv(&g.bucket_invariance),
            inv(&g.pad_invariance),
            g.median_ms,
        );
    }
    if let Some(c) = grades.iter().find_map(|g| g.ceiling) {
        println!(
            "  fp16 ceiling (ideal fp16 vs fp32): max {}, p99 {}, mean {}",
            fmt(Some(c.max)),
            fmt(Some(c.p99)),
            fmt(Some(c.mean))
        );
        let ratios: Vec<String> = grades
            .iter()
            .map(|g| match (g.ratio, g.max_ratio) {
                (Some(r), max) => format!(
                    "{} {r:.2}x ({}; absolute {}; max {})",
                    g.path,
                    crate::classify_grade::ratio_letter(r),
                    g.absolute_letter,
                    max.map_or("-".into(), |m| format!("{m:.2}x")),
                ),
                (None, _) => format!("{} -", g.path),
            })
            .collect();
        println!(
            "  p99 Δp / ceiling p99, the better of it and the absolute grade counts: {}",
            ratios.join(", ")
        );
        if c.n < 100 {
            println!(
                "  note: p99 = max at this corpus size ({} values); the ratio inherits the \
                 ceiling max's sensitivity",
                c.n
            );
        }
    }
    if let Some(d) = &report.determinism {
        println!("  ANE across two processes: Δp {} over {} cases", fmt(d.max), d.n);
    }
    for g in grades.iter().filter(|g| g.bucket_past_cpu_cap.n > 0) {
        println!(
            "  {} bucket check past {} tokens: Δp {} over {} comparisons (hardware limit, D33: Core ML's CPU \
             sums in a length-dependent order past 1,024; reported, not gated)",
            g.path,
            sidekick_core::MAX_CPU_INVARIANT_SEQ,
            fmt(g.bucket_past_cpu_cap.max),
            g.bucket_past_cpu_cap.n
        );
    }
    if let Some(c) = &report.bucket_coverage {
        println!("  bucket invariance: {c}");
    }
    for g in grades {
        for f in &g.flips {
            println!("  {} flip: {f}", g.path);
        }
        for t in &g.near_ties {
            println!("  {} near tie (not graded): {t}", g.path);
        }
        if let Some(d) = g.model_only_dp {
            println!("  {} Δp on the reference's own inputs: {}", g.path, fmt(Some(d)));
        }
    }
    // Gold accuracy, report only: it measures the corpus as much as the model.
    if let Some(g) = grades.iter().find(|g| g.reference_gold.is_some()) {
        let paths: Vec<String> = grades.iter().map(|g| format!("{} {}", g.path, accuracy(g.gold))).collect();
        println!(
            "  gold accuracy (report only): reference {}; {}",
            accuracy(g.reference_gold),
            paths.join(", ")
        );
    }
    println!("  worst Δp by tag: {}", grades.iter().map(|g| format!("{:>10}", g.path)).collect::<String>());
    for tag in grades[0].by_tag.keys() {
        let cells: String = grades.iter().map(|g| format!("{:>10}", fmt(g.by_tag[tag]))).collect();
        println!("    {tag:<14}{cells}");
    }
    for (name, _) in reference.logits.iter().filter(|(n, _)| *n != "torch") {
        let vs: Vec<String> = grades.iter().map(|g| format!("{} {}", g.path, fmt(g.vs_oracles[name]))).collect();
        println!("  {name} (report only): sidekick vs it: {}", vs.join(", "));
    }
    // The highest-Δp cases, side by side across paths (see docs/MODELS.md
    // for reading a conversion defect vs fp16 numerics vs the ANE).
    let n = reference.cases.len();
    let high = |i: usize| {
        grades.iter().map(|g| g.per_case.get(i).copied().flatten().unwrap_or(f64::INFINITY)).fold(0.0, f64::max)
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| high(b).total_cmp(&high(a)));
    println!("  highest-Δp cases:");
    for &i in order.iter().take(6) {
        let c = &reference.cases[i];
        let cells: Vec<String> = grades
            .iter()
            .map(|g| format!("{} {}", g.path, fmt(g.per_case.get(i).copied().flatten())))
            .collect();
        println!("    {:<26} {:>4} tok k={:<2} {}  [{}]", c.id, c.ids.len(), c.k, cells.join("  "), c.tags.join(","));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(t: &[&str]) -> Vec<String> {
        t.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn auto_samples_only_above_512_tokens() {
        let auto = |b: &[usize]| BucketCoverage::new(BucketInvariance::Auto, b);
        assert_eq!(auto(&[128, 256, 512]), BucketCoverage::Full);
        assert_eq!(auto(&[128, 512, 1024]), BucketCoverage::Sampled);
        assert_eq!(BucketCoverage::new(BucketInvariance::Full, &[128, 2048]), BucketCoverage::Full);
        assert_eq!(BucketCoverage::new(BucketInvariance::Sampled, &[128, 256]), BucketCoverage::Sampled);
    }

    #[test]
    fn sampled_cases_run_in_the_next_bucket_and_a_sample_in_every_one() {
        let b = [128, 256, 512, 1024, 2048];
        let s = BucketCoverage::Sampled;
        // Every case: its next larger bucket.
        assert_eq!(s.larger(&b, 128, 3, &[]), vec![256]);
        assert_eq!(s.larger(&b, 1024, 7, &[]), vec![2048]);
        // Every 10th case and every adversarial one: all larger buckets.
        assert_eq!(s.larger(&b, 128, 20, &[]), vec![256, 512, 1024, 2048]);
        assert_eq!(s.larger(&b, 256, 3, &tags(&["adversarial"])), vec![512, 1024, 2048]);
        // The largest bucket has none above it.
        assert!(s.larger(&b, 2048, 0, &[]).is_empty());
        // Full: every larger bucket for every case.
        assert_eq!(BucketCoverage::Full.larger(&b, 512, 3, &[]), vec![1024, 2048]);
    }

    #[test]
    fn the_report_says_what_was_sampled() {
        let case = |id: &str, t: &[&str]| ClassifyCase {
            id: id.into(),
            tags: tags(t),
            input: String::new(),
            query: None,
            group: None,
            candidate_labels: vec![],
            question_type: None,
            instructions: None,
            ids: vec![],
            type_ids: vec![],
            markers: vec![],
            qtype: None,
            k: 2,
            multi_label: None,
            gold: None,
        };
        // 25 cases: indexes 0, 10 and 20, plus one adversarial at 5.
        let cases: Vec<ClassifyCase> =
            (0..25).map(|i| case(&i.to_string(), if i == 5 { &["adversarial"] } else { &[] })).collect();
        let line = BucketCoverage::Sampled.describe(BucketInvariance::Auto, &[128, 2048], &cases);
        assert!(line.contains("every larger bucket for 4 of 25 cases"), "{line}");
        assert!(line.contains("the largest bucket, 2048, is over 512"), "{line}");
        let forced = BucketCoverage::Sampled.describe(BucketInvariance::Sampled, &[128, 256], &cases);
        assert!(forced.starts_with("sampled (--bucket-invariance sampled)"), "{forced}");
        assert_eq!(
            BucketCoverage::Full.describe(BucketInvariance::Full, &[128], &cases),
            "every larger bucket for every case"
        );
    }
}
