use serde::Deserialize;
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
    /// Directory scanned for `<model>/manifest.toml` entries.
    pub models_dir: Option<PathBuf>,
    /// Require `Authorization: Bearer <key>` on /v1 routes when set.
    pub api_key: Option<String>,
    /// How long a Foundation Models session is kept for conversation
    /// follow-ups before being dropped. Shorter than `model_idle_ttl_secs`
    /// because a session is one conversation's context (cheap to rebuild via
    /// replay) while a resident model serves every request (expensive to
    /// reload — seconds of Core ML compile for large encoders).
    pub session_ttl_secs: u64,
    /// How long a loaded embedding model stays resident after its last use.
    pub model_idle_ttl_secs: u64,
    /// Hard cap on a single generation call. A hung Foundation Models call
    /// otherwise hangs its request forever.
    pub request_timeout_secs: u64,
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

    pub fn models_dir(&self) -> PathBuf {
        self.models_dir.clone().unwrap_or_else(|| {
            dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("sidekick")
                .join("models")
        })
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
}

pub fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("sidekick").join("config.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
