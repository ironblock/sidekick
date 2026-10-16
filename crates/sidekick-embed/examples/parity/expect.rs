//! fixtures/parity/expectations.toml: the gates every model must pass, and
//! the per-chip floors that turn measured accuracy into a regression test.

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// A compute path a model is graded on: Core ML's three, and ONNX
/// Runtime's CPU (D40).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Path3 {
    Cpu,
    Gpu,
    Ane,
    /// An `onnx` model on ONNX Runtime's CPU: graded from `--external`
    /// results until the suite runs the backend itself.
    OnnxCpu,
}

impl Path3 {
    /// Core ML's paths, which the suite runs.
    pub const ALL: [Path3; 3] = [Path3::Cpu, Path3::Gpu, Path3::Ane];

    pub fn name(self) -> &'static str {
        match self {
            Path3::Cpu => "cpu",
            Path3::Gpu => "gpu",
            Path3::Ane => "ane",
            Path3::OnnxCpu => "onnx-cpu",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().chain([Path3::OnnxCpu]).find(|p| p.name() == s)
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct PerPath {
    pub cpu: f64,
    pub gpu: f64,
    pub ane: f64,
    /// ONNX Runtime's CPU; Core ML's CPU value when absent.
    #[serde(default, rename = "onnx-cpu")]
    pub onnx_cpu: Option<f64>,
    /// Core ML's CPU on an Intel Mac; `cpu` when absent. Its CPU rounds the
    /// last partial tile of the attention-score matmul differently by
    /// bucket, so the exact gate Apple silicon meets doesn't hold there
    /// (D38 amendment, October 2026).
    #[serde(default, rename = "cpu-x86_64")]
    pub cpu_x86_64: Option<f64>,
}

impl PerPath {
    /// The value for `path` on the architecture this suite runs on.
    pub fn get(&self, path: Path3) -> f64 {
        self.get_on(path, std::env::consts::ARCH)
    }

    /// The value for `path` on `arch` (`std::env::consts::ARCH` names).
    pub fn get_on(&self, path: Path3, arch: &str) -> f64 {
        match path {
            Path3::Cpu if arch == "x86_64" => self.cpu_x86_64.unwrap_or(self.cpu),
            Path3::Cpu => self.cpu,
            Path3::Gpu => self.gpu,
            Path3::Ane => self.ane,
            Path3::OnnxCpu => self.onnx_cpu.unwrap_or(self.cpu),
        }
    }
}

/// Hard gates: they hold on every chip, for every model.
#[derive(Debug, Clone, Deserialize)]
pub struct Gates {
    /// Minimum cosine between a case's output in its own bucket and in each
    /// larger one.
    pub bucket_invariance: PerPath,
    /// Minimum cosine between pad ids 0 and random pad ids, same bucket.
    pub pad_invariance: f64,
    /// Minimum cosine between two ANE runs in separate processes.
    pub determinism: f64,
    /// Reference similarity gap a path must not reverse (report only).
    pub rank_margin: f64,
}

/// Measured accuracy for one model on one path of one chip, less a margin.
#[derive(Debug, Clone, Deserialize)]
pub struct Floor {
    pub chip: String,
    pub model: String,
    pub path: String,
    /// Minimum worst-case cosine vs the fp32 reference, stress cases excluded.
    pub worst: f64,
    /// Minimum worst-case cosine over stress cases (tagged `degenerate`).
    pub stress: Option<f64>,
    /// Maximum |change| in any pairwise similarity vs the reference.
    pub drift: f64,
}

/// One letter scale: the bound for A, B and C; past C is D.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Scale {
    #[serde(rename = "A")]
    pub a: f64,
    #[serde(rename = "B")]
    pub b: f64,
    #[serde(rename = "C")]
    pub c: f64,
}

impl Scale {
    /// The letter for `v` where higher is better (each bound a minimum).
    pub fn at_least(&self, v: f64) -> char {
        if v >= self.a {
            'A'
        } else if v >= self.b {
            'B'
        } else if v >= self.c {
            'C'
        } else {
            'D'
        }
    }

    /// The letter for `v` where lower is better (each bound a maximum).
    pub fn at_most(&self, v: f64) -> char {
        if v <= self.a {
            'A'
        } else if v <= self.b {
            'B'
        } else if v <= self.c {
            'C'
        } else {
            'D'
        }
    }
}

/// `[grades]`: the letter scales, the one definition the suite and
/// tools/models_table.py both read. Letters describe accuracy; they are
/// never gates. F is a failed gate or non-finite output.
#[derive(Debug, Clone, Deserialize)]
pub struct Grades {
    /// Embedders: worst-case cosine against fp32, a minimum per letter.
    pub cosine: Scale,
    /// Classifiers: worst |Δp| against fp32, a maximum per letter.
    pub dp: Scale,
    /// Classifiers with an fp16 ceiling: p99 Δp over the ceiling's p99, a
    /// maximum per letter.
    pub ratio: Scale,
}

impl Grades {
    /// The committed `[grades]`, for tests.
    #[cfg(test)]
    pub fn committed() -> Self {
        Expectations::load(&Expectations::default_path()).unwrap().grades
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Expectations {
    pub gates: Gates,
    pub grades: Grades,
    /// Classifier gates; built-in defaults when absent.
    #[serde(default)]
    pub classify_gates: crate::classify_grade::ClassifyGates,
    #[serde(default)]
    pub floor: Vec<Floor>,
}

impl Expectations {
    /// `parity/expectations.toml` in the fixtures directory.
    pub fn default_path() -> PathBuf {
        crate::fixtures::dir().join("parity/expectations.toml")
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn floor(&self, chip: &str, model: &str, path: Path3) -> Option<&Floor> {
        self.floor
            .iter()
            .find(|f| f.chip == chip && f.model == model && f.path == path.name())
    }
}

/// A floor for a measured worst-case cosine: 1.5x the measured error, and
/// at least 1e-5 below the measurement, rounded down to six decimals. A
/// relative margin because the error spans three orders of magnitude
/// (bge-small ~1e-5, EmbeddingGemma on the ANE ~2e-2).
pub fn floor_for(measured: f64) -> f64 {
    let f = (1.0 - 1.5 * (1.0 - measured)).min(measured - 1e-5);
    (f * 1e6).floor() / 1e6
}

/// A ceiling for a measured similarity drift, by the same rule.
pub fn ceiling_for(measured: f64) -> f64 {
    let c = (1.5 * measured).max(measured + 1e-5);
    (c * 1e6).ceil() / 1e6
}

/// The letter for a worst-case cosine: `[grades].cosine`, or F when it
/// isn't finite.
pub fn letter(grades: &Grades, worst: Option<f64>) -> char {
    worst.map_or('F', |w| grades.cosine.at_least(w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margins_scale_with_error() {
        let close = |a: f64, b: f64| (a - b).abs() < 2e-6;
        // bge-small: error 2e-5 -> floor 1e-5 lower (absolute minimum wins).
        assert!(close(floor_for(0.99998), 0.99997), "{}", floor_for(0.99998));
        // gemma on the ANE: error 2.3e-2 -> 1.5x.
        assert!(close(floor_for(0.977), 0.9655), "{}", floor_for(0.977));
        assert!(floor_for(0.977) <= 0.9655);
        assert!(close(ceiling_for(0.038), 0.057));
        assert!(close(ceiling_for(0.0), 0.00001));
    }

    #[test]
    fn letters() {
        let g = Grades::committed();
        assert_eq!(letter(&g, Some(0.99995)), 'A');
        assert_eq!(letter(&g, Some(0.9995)), 'B');
        assert_eq!(letter(&g, Some(0.99)), 'C');
        assert_eq!(letter(&g, Some(0.9)), 'D');
        assert_eq!(letter(&g, None), 'F');
        // bounds are inclusive
        assert_eq!(g.cosine.at_least(g.cosine.a), 'A');
        assert_eq!(g.dp.at_most(g.dp.a), 'A');
    }

    #[test]
    fn committed_grades_are_ordered() {
        let g = Grades::committed();
        assert!(g.cosine.a > g.cosine.b && g.cosine.b > g.cosine.c && g.cosine.c < 1.0);
        for s in [g.dp, g.ratio] {
            assert!(0.0 < s.a && s.a < s.b && s.b < s.c, "{s:?}");
        }
    }

    #[test]
    fn committed_expectations_parse() {
        let e = Expectations::load(&Expectations::default_path()).unwrap();
        assert!(e.gates.bucket_invariance.cpu >= e.gates.bucket_invariance.ane);
        // Intel's CPU gate is looser than Apple silicon's, never looser than the GPU's.
        let b = &e.gates.bucket_invariance;
        assert!(b.get_on(Path3::Cpu, "x86_64") < b.get_on(Path3::Cpu, "aarch64"));
        assert!(b.get_on(Path3::Cpu, "x86_64") >= b.gpu);
        assert_eq!(b.get_on(Path3::Gpu, "x86_64"), b.get_on(Path3::Gpu, "aarch64"));
        let c = &e.classify_gates.bucket_invariance;
        assert!(c.get_on(Path3::Cpu, "x86_64") > c.get_on(Path3::Cpu, "aarch64"));
        assert!(c.get_on(Path3::Cpu, "x86_64") <= c.gpu);
        for f in &e.floor {
            assert!(Path3::parse(&f.path).is_some(), "bad path {}", f.path);
            assert!(f.worst > 0.0 && f.worst <= 1.0 && f.drift >= 0.0);
        }
    }
}
