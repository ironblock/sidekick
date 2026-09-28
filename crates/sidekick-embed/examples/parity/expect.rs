//! fixtures/parity/expectations.toml: the gates every model must pass, and
//! the per-chip floors that turn measured accuracy into a regression test.

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// A compute path sidekick can run a Core ML model on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Path3 {
    Cpu,
    Gpu,
    Ane,
}

impl Path3 {
    pub const ALL: [Path3; 3] = [Path3::Cpu, Path3::Gpu, Path3::Ane];

    pub fn name(self) -> &'static str {
        match self {
            Path3::Cpu => "cpu",
            Path3::Gpu => "gpu",
            Path3::Ane => "ane",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == s)
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct PerPath {
    pub cpu: f64,
    pub gpu: f64,
    pub ane: f64,
}

impl PerPath {
    pub fn get(&self, path: Path3) -> f64 {
        match path {
            Path3::Cpu => self.cpu,
            Path3::Gpu => self.gpu,
            Path3::Ane => self.ane,
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

#[derive(Debug, Clone, Deserialize)]
pub struct Expectations {
    pub gates: Gates,
    #[serde(default)]
    pub floor: Vec<Floor>,
}

impl Expectations {
    pub fn default_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/parity/expectations.toml")
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

/// Documentation vocabulary for a worst-case cosine (never gated):
/// A ≥ 0.9999, indistinguishable from fp32; B ≥ 0.999; C ≥ 0.985, the
/// converters' ANE acceptance gate; D below that; F for non-finite output
/// or a failed hard gate.
pub fn letter(worst: Option<f64>) -> char {
    match worst {
        None => 'F',
        Some(w) if w >= 0.9999 => 'A',
        Some(w) if w >= 0.999 => 'B',
        Some(w) if w >= 0.985 => 'C',
        Some(_) => 'D',
    }
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
        assert_eq!(letter(Some(0.99995)), 'A');
        assert_eq!(letter(Some(0.9995)), 'B');
        assert_eq!(letter(Some(0.99)), 'C');
        assert_eq!(letter(Some(0.9)), 'D');
        assert_eq!(letter(None), 'F');
    }

    #[test]
    fn committed_expectations_parse() {
        let e = Expectations::load(&Expectations::default_path()).unwrap();
        assert!(e.gates.bucket_invariance.cpu >= e.gates.bucket_invariance.ane);
        for f in &e.floor {
            assert!(Path3::parse(&f.path).is_some(), "bad path {}", f.path);
            assert!(f.worst > 0.0 && f.worst <= 1.0 && f.drift >= 0.0);
        }
    }
}
