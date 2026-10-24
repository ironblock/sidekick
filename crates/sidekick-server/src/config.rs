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
    /// Require `Authorization: Bearer <key>` on every route but /health
    /// when set (D13).
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
    /// Serve models on the ANE (`cpu_and_ne` or `all`) past their ANE cap:
    /// 2,048 tokens, or the manifest's `ane_max_seq_len` (D48). Past it a
    /// bucket can take half an hour to compile for the ANE and then fail
    /// some predictions. For measurement; to serve long inputs, serve the
    /// model on the GPU (`[models."<id>"] compute_units = "cpu_and_gpu"`).
    pub ignore_ane_seq_cap: bool,
    /// Read where Core ML places each loaded bucket's operations on this
    /// machine (`placement` in /v1/models, `compute_plans` in /health). Off
    /// by default: reading a plan compiles the bucket a second time, as
    /// long as its first load took (minutes for a large bucket), and Core
    /// ML caches that compile on disk at about the bucket's weight size.
    /// Each bucket's plan is read once, in the background, after its first
    /// load, and the result is cached. Without it, the plan recorded at
    /// conversion time is reported, when the manifest has one.
    pub report_compute_plans: bool,
    /// Load each Core ML bucket the daemon serves once at startup, in the
    /// background, so that Core ML's compile cache holds it before a
    /// request needs it (D52). Cold buckets go first. Each is released
    /// after loading, so nothing stays resident that a request didn't ask
    /// for; a request for a bucket being warmed waits for that load and
    /// takes it. `[models."<id>"] warm = false` leaves one model out.
    pub warm_on_start: bool,
    /// How long the startup warm-up may run, in seconds: no bucket starts
    /// loading after this. A load in progress finishes, since Core ML can't
    /// stop one; the buckets left compile on first use, as without
    /// warming. 0 means no limit.
    pub warm_budget_secs: u64,
    /// Inference threads for each `onnx` model, unless its
    /// `[models."<id>"]` says otherwise (D40). Default: the physical cores
    /// of the machine's fastest tier (an Apple silicon Mac's P-cores), or
    /// 4 if they can't be detected.
    pub onnx_intra_op_threads: Option<usize>,
    /// The ONNX Runtime library to load (`libonnxruntime.dylib` or `.so`),
    /// when the daemon has an onnx model: before `ORT_DYLIB_PATH` and the
    /// one beside sidekickd. For an existing install, and for a daemon
    /// under launchd, which doesn't pass a shell's environment (D40).
    pub onnx_runtime: Option<PathBuf>,
    /// Per-model settings, by model id (`[models."agent-jev"]`).
    pub models: BTreeMap<String, ModelConfig>,
}

/// One model's settings in the daemon config (D38).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// Serve the model on these compute units instead of its manifest's: a
    /// Core ML model only. The weight limit (D32) and the length caps (D33,
    /// D48) judge it as they would the manifest's choice, and /v1/models and
    /// /health report that the operator chose.
    pub compute_units: Option<ComputeUnits>,
    /// Inference threads for an `onnx` model (D40).
    pub intra_op_threads: Option<usize>,
    /// `false` leaves a Core ML model out of the startup warm-up and of
    /// fetch's (D52); `sidekickd warm <id>` still warms it.
    pub warm: Option<bool>,
}

/// Each `onnx` model's inference threads (D40): the operator's for that
/// model, else the operator's for every model, else the computed default,
/// `min(max(8, the model's [onnx] default), the machine's fast cores)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OnnxThreads {
    /// `onnx_intra_op_threads`.
    pub configured: Option<usize>,
    /// `[models."<id>"] intra_op_threads`.
    pub by_model: BTreeMap<String, usize>,
    /// The machine's fast physical cores (`sidekick_embed::onnx::detected_cores`).
    pub cores: usize,
}

impl OnnxThreads {
    /// A model's threads, and who chose them: `config (model)`, `config`,
    /// or `default`.
    pub fn resolve(&self, id: &str, model_default: Option<usize>) -> (usize, &'static str) {
        match (self.by_model.get(id), self.configured) {
            (Some(&n), _) => (n, "config (model)"),
            (None, Some(n)) => (n, "config"),
            (None, None) => (sidekick_embed::onnx::default_threads(self.cores, model_default), "default"),
        }
    }

    /// The default for a model without its own: the machine's fast cores,
    /// capped at 8.
    pub fn machine_default(&self) -> usize {
        sidekick_embed::onnx::default_threads(self.cores, None)
    }
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
            ignore_ane_seq_cap: false,
            report_compute_plans: false,
            warm_on_start: true,
            warm_budget_secs: 1800,
            onnx_intra_op_threads: None,
            onnx_runtime: None,
            models: BTreeMap::new(),
        }
    }
}

impl Config {
    /// The file `load` reads: `path` when given, else the default path if a
    /// file is there, else none (the defaults apply).
    pub fn resolve_path(path: Option<&PathBuf>) -> Option<PathBuf> {
        match path {
            Some(p) => Some(p.clone()),
            None => default_config_path().filter(|p| p.is_file()),
        }
    }

    pub fn load(path: Option<&PathBuf>) -> anyhow::Result<Self> {
        let Some(path) = Self::resolve_path(path) else {
            return Ok(Self::default());
        };
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        toml::from_str(&raw).map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))
    }

    /// The operator's compute units, by model id (D38).
    pub fn compute_units(&self) -> BTreeMap<String, ComputeUnits> {
        self.models.iter().filter_map(|(id, m)| Some((id.clone(), m.compute_units?))).collect()
    }

    /// Models the config leaves out of automatic warm-ups (D52).
    pub fn warm_skipped(&self) -> std::collections::BTreeSet<String> {
        self.models.iter().filter(|(_, m)| m.warm == Some(false)).map(|(id, _)| id.clone()).collect()
    }

    /// The startup warm-up's budget; `None` for no limit.
    pub fn warm_budget(&self) -> Option<Duration> {
        (self.warm_budget_secs > 0).then(|| Duration::from_secs(self.warm_budget_secs))
    }

    /// The `onnx` models' inference threads. Zero threads is a config
    /// error: ONNX Runtime would read it as "as many as there are cores",
    /// the opposite of a limit.
    pub fn onnx_threads(&self) -> anyhow::Result<OnnxThreads> {
        if self.onnx_intra_op_threads == Some(0) {
            anyhow::bail!("onnx_intra_op_threads must be at least 1");
        }
        let mut by_model = BTreeMap::new();
        for (id, m) in &self.models {
            match m.intra_op_threads {
                Some(0) => anyhow::bail!("[models.\"{id}\"] intra_op_threads must be at least 1"),
                Some(n) => {
                    by_model.insert(id.clone(), n);
                }
                None => {}
            }
        }
        Ok(OnnxThreads { configured: self.onnx_intra_op_threads, by_model, cores: sidekick_embed::onnx::detected_cores().cores })
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

/// `<config dir>/sidekick/config.toml`: on macOS
/// `~/Library/Application Support/sidekick/config.toml`, on Linux
/// `$XDG_CONFIG_HOME/sidekick/config.toml` (`~/.config` by default).
pub fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("sidekick").join("config.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_threads_have_a_default_and_per_model_overrides() {
        let c: Config = toml::from_str("onnx_intra_op_threads = 3\n[models.e]\nintra_op_threads = 1\n").unwrap();
        let t = c.onnx_threads().unwrap();
        assert_eq!(t.resolve("e", Some(16)), (1, "config (model)"), "the model's setting wins");
        assert_eq!(t.resolve("other", Some(16)), (3, "config"), "then the daemon's, over a manifest default");
        let t = Config::default().onnx_threads().unwrap();
        let detected = sidekick_embed::onnx::detected_cores();
        assert_eq!(t.cores, detected.cores);
        assert_eq!(t.machine_default(), detected.cores.min(8), "the fast cores ({}), capped at 8", detected.source);
        assert_eq!(t.resolve("m", None), (t.machine_default(), "default"));
        assert_eq!(t.resolve("m", Some(64)).0, detected.cores, "a model's default never passes the cores");
        if cfg!(target_os = "macos") {
            assert_ne!(detected.source, "fallback", "a Mac reports its cores");
        }
        // Zero would mean "all cores" to ONNX Runtime: refused.
        assert!(toml::from_str::<Config>("onnx_intra_op_threads = 0").unwrap().onnx_threads().is_err());
        assert!(toml::from_str::<Config>("[models.e]\nintra_op_threads = 0").unwrap().onnx_threads().is_err());
    }

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
        assert!(!Config::default().ignore_ane_seq_cap, "the ANE sequence cap is enforced by default");
        let c: Config = toml::from_str("ignore_ane_seq_cap = true").unwrap();
        assert!(c.ignore_ane_seq_cap);
        assert!(!Config::default().report_compute_plans, "live plan reads are opt-in");
        let c: Config = toml::from_str("report_compute_plans = true").unwrap();
        assert!(c.report_compute_plans);
    }

    #[test]
    fn warming_is_on_by_default_with_a_budget_and_a_per_model_opt_out() {
        let c = Config::default();
        assert!(c.warm_on_start);
        assert_eq!(c.warm_budget(), Some(Duration::from_secs(1800)));
        assert!(c.warm_skipped().is_empty());
        let c: Config =
            toml::from_str("warm_on_start = false\nwarm_budget_secs = 0\n[models.big]\nwarm = false\n[models.small]\nwarm = true\n")
                .unwrap();
        assert!(!c.warm_on_start);
        assert_eq!(c.warm_budget(), None, "0 is no limit");
        assert_eq!(c.warm_skipped(), ["big".to_string()].into());
    }
}
