//! The chat models the daemon serves, by model id (D53).
//!
//! `/v1/chat/completions` routes a request by its exact `model` to one of
//! these: no default, no alias and no fallback between backends. Each entry
//! is a [`ChatBackend`] with what `/v1/models` and `/health` say about it.
//! Foundation Models (`apple-fm`) is registered by [`foundation_models`],
//! each open model in the models directory by [`open_model`] (`sidekick-llm`);
//! when there are open models, they and `apple-fm` share one ANE chat lane
//! ([`sidekick_llm::ChatLane`]).

use crate::config::RequestTimeouts;
use serde_json::{Map, Value};
use sidekick_core::{ChatBackend, ModelInfo};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// Foundation Models' model id.
pub const APPLE_FM: &str = "apple-fm";

/// What `/v1/models` lists about a chat model besides its id and context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatListing {
    /// What serves the model: `foundation_models`, or `coreml` for an open
    /// model.
    pub backend: &'static str,
    /// The optional request fields the model honors. Others are accepted
    /// and ignored, or refused, as API.md's chat section says (D22).
    pub honored_fields: Vec<&'static str>,
    /// Whether `response_format: json_schema` output streams as it is
    /// generated, rather than arriving whole.
    pub guided_streaming: bool,
    /// Whether the model has guardrails that can refuse a request (a 400
    /// `content_filter`, #178).
    pub guardrails: bool,
    /// An open model's Core ML compute units.
    pub compute_units: Option<&'static str>,
    /// The macOS on which an open model's build was graded.
    pub graded_os: Option<String>,
}

/// One chat model: its backend, what it lists, its request timeout, and the
/// id its provenance header names.
pub struct ChatModel {
    backend: Arc<dyn ChatBackend>,
    pub listing: ChatListing,
    /// The timeout the backend was built with, and who chose it
    /// ([`RequestTimeouts::resolve`]). The backend enforces it over the
    /// request's whole work, streamed or not.
    pub request_timeout: (Duration, &'static str),
    /// Backend-specific facts for `/health`.
    pub health: Map<String, Value>,
    /// Facts read at each `/health`: an open model's state, queue and
    /// decoder.
    live_health: Option<LiveHealth>,
    /// A fixed `sidekick-model`: an open model's `<id>@<revision>`.
    fixed_provenance: Option<String>,
    provenance: ProvenanceCache,
}

/// Reads a chat model's live `/health` facts.
pub type LiveHealth = Arc<dyn Fn() -> Map<String, Value> + Send + Sync>;

impl ChatModel {
    pub fn new(backend: Arc<dyn ChatBackend>, listing: ChatListing, request_timeout: (Duration, &'static str)) -> Self {
        Self {
            backend,
            listing,
            request_timeout,
            health: Map::new(),
            live_health: None,
            fixed_provenance: None,
            provenance: Default::default(),
        }
    }

    /// With backend-specific facts for `/health`.
    pub fn with_health(self, health: Map<String, Value>) -> Self {
        Self { health, ..self }
    }

    /// With facts `/health` reads live.
    pub fn with_live_health(self, live: LiveHealth) -> Self {
        Self { live_health: Some(live), ..self }
    }

    /// With a fixed `sidekick-model` header.
    pub fn with_provenance(self, model: String) -> Self {
        Self { fixed_provenance: Some(model), ..self }
    }

    /// Every `/health` fact about the model: the fixed ones, then the live.
    pub fn health_facts(&self) -> Map<String, Value> {
        let mut facts = self.health.clone();
        if let Some(live) = &self.live_health {
            facts.extend(live());
        }
        facts
    }

    pub fn id(&self) -> &str {
        self.backend.id()
    }

    pub fn backend(&self) -> &Arc<dyn ChatBackend> {
        &self.backend
    }

    /// The model for provenance: the variant id the backend reported (a
    /// Foundation Models variant), otherwise the model id. Never waits on
    /// the backend: a stale entry is refreshed by a detached task.
    pub fn provenance_id(self: &Arc<Self>) -> String {
        if let Some(fixed) = &self.fixed_provenance {
            return fixed.clone();
        }
        let (id, refresh) = self.provenance.read();
        if refresh {
            let model = self.clone();
            tokio::spawn(async move { model.refresh_provenance().await });
        }
        id.unwrap_or_else(|| self.id().to_string())
    }

    /// Ask the backend for its model facts, and cache the variant id.
    pub async fn refresh_provenance(&self) -> Option<ModelInfo> {
        let info = self.backend.model_info().await;
        self.provenance.set(info.as_ref().and_then(|i| i.variant_id.clone()));
        info
    }
}

/// The chat models, by id.
#[derive(Default)]
pub struct ChatModels {
    models: BTreeMap<String, Arc<ChatModel>>,
    /// The ANE chat lane `apple-fm` and the open models share, when there
    /// are open models.
    lane: Option<Arc<sidekick_llm::ChatLane>>,
}

impl ChatModels {
    /// Register a model under its backend's id. Two models with one id is
    /// an error: a request could reach only one of them.
    pub fn insert(&mut self, model: ChatModel) -> anyhow::Result<()> {
        let id = model.id().to_string();
        if self.models.contains_key(&id) {
            anyhow::bail!("two chat models have the id `{id}`");
        }
        self.models.insert(id, Arc::new(model));
        Ok(())
    }

    /// The model a request names, exactly.
    pub fn get(&self, id: &str) -> Option<&Arc<ChatModel>> {
        self.models.get(id)
    }

    /// Every chat model, by id.
    pub fn iter(&self) -> impl Iterator<Item = &Arc<ChatModel>> {
        self.models.values()
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(String::as_str)
    }

    /// The models' shared ANE chat lane, for `/health`.
    pub fn lane(&self) -> Option<&Arc<sidekick_llm::ChatLane>> {
        self.lane.as_ref()
    }

    pub fn set_lane(&mut self, lane: Arc<sidekick_llm::ChatLane>) {
        self.lane = Some(lane);
    }
}

impl FromIterator<ChatModel> for ChatModels {
    /// Panics on a duplicate id; for tests and fixed sets.
    fn from_iter<I: IntoIterator<Item = ChatModel>>(iter: I) -> Self {
        let mut models = Self::default();
        for m in iter {
            models.insert(m).expect("chat model ids are unique");
        }
        models
    }
}

/// Foundation Models as `apple-fm`, with the backend `build` makes for its
/// request timeout.
pub fn foundation_models(
    timeouts: &RequestTimeouts,
    build: impl FnOnce(Duration) -> Arc<dyn ChatBackend>,
) -> ChatModel {
    let timeout = timeouts.resolve(APPLE_FM, None);
    let backend = build(timeout.0);
    debug_assert_eq!(backend.id(), APPLE_FM);
    let listing = ChatListing {
        backend: "foundation_models",
        honored_fields: vec![
            "max_completion_tokens",
            "max_tokens",
            "response_format",
            "seed",
            "stop",
            "temperature",
            "top_p",
        ],
        // Guided snapshots aren't prefix-stable, so guided output is sent whole.
        guided_streaming: false,
        guardrails: true,
        compute_units: None,
        graded_os: None,
    };
    let mut health = Map::new();
    // The SDK the Foundation Models shim was compiled against; the macOS 27
    // features are only present when it is 27 or newer.
    health.insert("fm_sdk".into(), Value::from(sidekick_fm::FM_SDK));
    ChatModel::new(backend, listing, timeout).with_health(health)
}

/// `/v1/models`' listing of an open model (D53).
pub fn open_listing() -> ChatListing {
    ChatListing {
        backend: "coreml",
        honored_fields: vec!["max_completion_tokens", "max_tokens", "seed", "stop", "temperature", "top_p"],
        // No guided output until llguidance (#186): `response_format`
        // json_schema is a 400 on open models.
        guided_streaming: false,
        guardrails: false,
        compute_units: Some(sidekick_core::ComputeUnits::CpuAndNeuralEngine.name()),
        graded_os: None,
    }
}

/// An open model as a chat model: its backend, listing, timeout,
/// `<id>@<revision>` provenance and live `/health` facts.
pub fn open_model(
    model: Arc<sidekick_llm::OpenModel>,
    revision: Option<&str>,
    request_timeout: (Duration, &'static str),
) -> ChatModel {
    let id = model.id().to_string();
    let provenance = match revision {
        Some(rev) => format!("{id}@{rev}"),
        None => id,
    };
    let live = model.clone();
    ChatModel::new(model, open_listing(), request_timeout)
        .with_provenance(provenance)
        .with_live_health(Arc::new(move || live.health()))
}

/// How long a known variant id is reused before a background refresh. A
/// Foundation Models variant changes only when the OS updates the model.
const PROVENANCE_TTL: Duration = Duration::from_secs(600);
/// The same while the backend reports none (Apple Intelligence off, assets
/// still downloading), so the header catches up soon after it's available.
const PROVENANCE_RETRY: Duration = Duration::from_secs(30);

/// The variant id a chat model's `sidekick-model` header names. Asking the
/// backend can be a blocking call (the Foundation Models shim), so requests
/// read this cache and never wait on it: a stale entry is refreshed by a
/// detached task, one at a time.
#[derive(Default)]
struct ProvenanceCache {
    state: Mutex<CacheState>,
}

#[derive(Default)]
struct CacheState {
    variant_id: Option<String>,
    fetched: Option<Instant>,
    refreshing: bool,
}

impl ProvenanceCache {
    fn set(&self, variant_id: Option<String>) {
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        s.variant_id = variant_id;
        s.fetched = Some(Instant::now());
        s.refreshing = false;
    }

    /// The cached id, and whether this caller should start a refresh.
    fn read(&self) -> (Option<String>, bool) {
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let ttl = if s.variant_id.is_some() { PROVENANCE_TTL } else { PROVENANCE_RETRY };
        let stale = s.fetched.is_none_or(|t| t.elapsed() >= ttl);
        let refresh = stale && !s.refreshing;
        if refresh {
            s.refreshing = true;
        }
        (s.variant_id.clone(), refresh)
    }
}
