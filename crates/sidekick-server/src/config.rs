use serde::Deserialize;
use sidekick_core::ComputeUnits;
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Server configuration. File values (TOML) are overridden by CLI flags.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Listen address. Loopback by default: this daemon fronts on-device
    /// models and has no business on the network unless you say so.
    pub addr: SocketAddr,
    /// Directory scanned for `<model>/manifest.toml` (embedders) and
    /// `<model>/classifier.toml` (classifiers) entries.
    pub models_dir: Option<PathBuf>,
    /// Require `Authorization: Bearer <key>` on /v1 routes when set.
    pub api_key: Option<String>,
    /// How long a Foundation Models session is kept for conversation
    /// follow-ups before being dropped. Shorter than `model_idle_ttl_secs`
    /// because a session is one conversation's context (cheap to rebuild via
    /// replay) while a resident model serves every request (expensive to
    /// reload — seconds of Core ML compile for large encoders).
    pub session_ttl_secs: u64,
    /// How long a loaded model (embedder or classifier) stays resident after
    /// its last use.
    pub model_idle_ttl_secs: u64,
    /// Hard cap on a single generation call. A hung Foundation Models call
    /// otherwise hangs its request forever. For embeddings, classify and
    /// rerank requests, it bounds the work after any model or bucket load
    /// the request waited on: a first load compiles the bucket, which can
    /// take minutes, and is bounded by `load_timeout_secs` instead.
    pub request_timeout_secs: u64,
    /// How long an embeddings, classify or rerank request waits on loading
    /// its model and the buckets it needs. Far longer than any measured
    /// first load (about 4 minutes for a 2,048-token bucket); a load that
    /// outlasts it continues in the background, so a retry finds it.
    pub load_timeout_secs: u64,
    /// Load models served on the ANE even when their compiled weights exceed
    /// Core ML's 1 GiB limit, which Core ML would quietly run off the ANE.
    /// For experimentation; a manifest's `ane_weight_limit = "ignore"` does
    /// the same for one model.
    pub ignore_ane_weight_cap: bool,
    /// Serve models on `cpu_only` past 1,024 tokens, accepting Core ML's
    /// slightly bucket-dependent CPU results there (D33). A manifest's
    /// `cpu_seq_limit = "ignore"` does the same for one model.
    pub ignore_cpu_seq_cap: bool,
    /// Read where Core ML places each loaded bucket's operations on this
    /// machine (`placement` in /v1/models, `compute_plans` in /health). Off
    /// by default: reading a plan compiles the bucket a second time, as
    /// long as its first load took (minutes for a large bucket), and Core
    /// ML caches that compile on disk at about the bucket's weight size.
    /// Each bucket's plan is read once, in the background, after its first
    /// load, and the result is cached. Without it, the plan recorded at
    /// conversion time is reported, when the manifest has one.
    pub report_compute_plans: bool,
    /// Per-model settings, by model id (`[models."agent-jev"]`).
    pub models: BTreeMap<String, ModelConfig>,
}

/// One model's settings in the daemon config (D38).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// Serve the model on these compute units instead of its manifest's: a
    /// Core ML model only. The weight limit (D32) and the CPU cap (D33)
    /// judge it as they would the manifest's choice, and /v1/models and
    /// /health report that the operator chose.
    pub compute_units: Option<ComputeUnits>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8790".parse().unwrap(),
            models_dir: None,
            api_key: None,
            session_ttl_secs: 300,
            model_idle_ttl_secs: 900,
            request_timeout_secs: 60,
            load_timeout_secs: 900,
            ignore_ane_weight_cap: false,
            ignore_cpu_seq_cap: false,
            report_compute_plans: false,
            models: BTreeMap::new(),
        }
    }
}

impl Config {
    pub fn load(path: Option<&PathBuf>) -> anyhow::Result<Self> {
        let path = match path {
            Some(p) => p.clone(),
            None => match default_config_path() {
                Some(p) if p.is_file() => p,
                _ => return Ok(Self::default()),
            },
        };
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        toml::from_str(&raw).map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))
    }

    /// The operator's compute units, by model id (D38).
    pub fn compute_units(&self) -> BTreeMap<String, ComputeUnits> {
        self.models.iter().filter_map(|(id, m)| Some((id.clone(), m.compute_units?))).collect()
    }

    pub fn models_dir(&self) -> PathBuf {
        self.models_dir.clone().unwrap_or_else(|| {
            dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("sidekick")
                .join("models")
        })
    }

    /// Where compute plans are cached: `~/Library/Caches/sidekick/compute-plans`.
    pub fn compute_plan_cache_dir(&self) -> Option<PathBuf> {
        dirs::cache_dir().map(|d| d.join("sidekick").join("compute-plans"))
    }

    pub fn session_ttl(&self) -> Duration {
        Duration::from_secs(self.session_ttl_secs)
    }

    pub fn model_idle_ttl(&self) -> Duration {
        Duration::from_secs(self.model_idle_ttl_secs)
    }

    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs)
    }

    pub fn load_timeout(&self) -> Duration {
        Duration::from_secs(self.load_timeout_secs)
    }
}

pub fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("sidekick").join("config.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_section_sets_its_compute_units() {
        let c: Config = toml::from_str("[models.\"agent-jev\"]\ncompute_units = \"cpu_and_ne\"\n\n[models.other]\n").unwrap();
        assert_eq!(c.compute_units(), [("agent-jev".to_string(), ComputeUnits::CpuAndNeuralEngine)].into());
        assert!(Config::default().compute_units().is_empty());
        // A typo is an error, not a silently ignored setting.
        assert!(toml::from_str::<Config>("[models.x]\ncompute_unit = \"cpu_only\"\n").is_err());
        assert!(toml::from_str::<Config>("[models.x]\ncompute_units = \"gpu\"\n").is_err());
    }

    #[test]
    fn defaults_are_loopback_and_sane() {
        let c = Config::default();
        assert!(c.addr.ip().is_loopback());
        assert_eq!(c.session_ttl(), Duration::from_secs(300));
    }

    #[test]
    fn parses_partial_toml() {
        let c: Config = toml::from_str("addr = \"127.0.0.1:9000\"").unwrap();
        assert_eq!(c.addr.port(), 9000);
        assert_eq!(c.model_idle_ttl_secs, 900);
        assert!(!c.ignore_ane_weight_cap, "the ANE weight cap is enforced by default");
        let c: Config = toml::from_str("ignore_ane_weight_cap = true").unwrap();
        assert!(c.ignore_ane_weight_cap);
        assert!(!Config::default().ignore_cpu_seq_cap, "the CPU sequence cap is enforced by default");
        let c: Config = toml::from_str("ignore_cpu_seq_cap = true").unwrap();
        assert!(c.ignore_cpu_seq_cap);
        assert!(!Config::default().report_compute_plans, "live plan reads are opt-in");
        let c: Config = toml::from_str("report_compute_plans = true").unwrap();
        assert!(c.report_compute_plans);
    }
}
