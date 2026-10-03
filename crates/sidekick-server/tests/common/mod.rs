//! Shared fixtures for the server's integration tests: a mock chat
//! backend, a real on-disk static embedding model, and mock classifiers
//! behind real `classifier.toml` manifests.

#![allow(dead_code)]

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sidekick_core::manifest::{ClassifierManifest, ClassifyFormat};
use sidekick_core::{
    Availability, ChatBackend, ChatRequest, ChatResponse, Classifier, ClassifyParams,
    ClassifyTask, DeltaSink, Error, FinishReason, ModelInfo, ModelRegistry, PairParams, Prepared,
    ProblemType, Result, Source, Usage,
};
use sidekick_server::{build_router, AppState, ClassifierPool, EmbedderPool};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tower::ServiceExt;

pub struct MockChat {
    pub available: bool,
    /// How long `model_info` takes: a slow Swift shim.
    pub info_delay: Duration,
}

#[async_trait::async_trait]
impl ChatBackend for MockChat {
    fn id(&self) -> &str {
        "apple-fm"
    }

    fn context_limit(&self) -> Option<usize> {
        Some(4096)
    }

    async fn model_info(&self) -> Option<ModelInfo> {
        tokio::time::sleep(self.info_delay).await;
        self.available.then(|| ModelInfo {
            variant: Some("AFM 3 Core".into()),
            variant_id: Some("core3".into()),
            context_size: Some(4096),
            capabilities: Some(vec!["guided_generation".into(), "tool_calling".into()]),
        })
    }

    async fn availability(&self) -> Availability {
        if self.available {
            Availability::Available
        } else {
            Availability::unavailable(
                sidekick_core::UnavailableReason::AppleIntelligenceNotEnabled,
            )
        }
    }

    /// A last message `stream:a|b|c` streams those deltas; a `FAIL` part
    /// errors mid-stream and a `FILTER` part is a mid-stream guardrail stop.
    async fn complete_stream(&self, req: ChatRequest, mut sink: DeltaSink) -> Result<ChatResponse> {
        let last = req.messages.last().unwrap().content.clone();
        let Some(parts) = last.strip_prefix("stream:") else {
            let response = self.complete(req).await?;
            sink(&response.content);
            return Ok(response);
        };
        let mut content = String::new();
        for part in parts.split('|') {
            match part {
                "FAIL" => return Err(sidekick_core::Error::Inference("broke mid-stream".into())),
                "FILTER" => return Err(sidekick_core::Error::ContentFiltered("guardrail".into())),
                _ => {
                    sink(part);
                    content.push_str(part);
                }
            }
        }
        Ok(ChatResponse {
            content,
            finish: FinishReason::Stop,
            usage: Usage { prompt_tokens: 10, completion_tokens: 3, cached_tokens: Some(4), reasoning_tokens: None },
            constrained: false,
        })
    }

    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse> {
        if !self.available {
            return Err(sidekick_core::Error::Unavailable(
                sidekick_core::UnavailableReason::AppleIntelligenceNotEnabled,
            ));
        }
        // A last message of the form `error:<kind>` makes the mock fail with
        // that error, to exercise the HTTP error mapping.
        if let Some(kind) = req.messages.last().unwrap().content.strip_prefix("error:") {
            use sidekick_core::Error;
            return Err(match kind {
                "overflow" => Error::ContextOverflow { limit: 4096, actual: Some(4459) },
                "overflow_unknown" => Error::ContextOverflow { limit: 4096, actual: None },
                "content_filter" => Error::ContentFiltered("guardrailViolation".into()),
                "rate_limited" => Error::RateLimited {
                    message: "slow down".into(),
                    retry_after_secs: Some(7),
                },
                "transient" => Error::Transient("concurrentRequests".into()),
                "language" => Error::UnsupportedLanguage("xx".into()),
                "guided" => Error::GuidedGeneration("unsupported JSON schema type: null".into()),
                "not_ready" => Error::Unavailable(sidekick_core::UnavailableReason::ModelNotReady),
                other => Error::Inference(other.into()),
            });
        }
        let content = if let Some(schema) = &req.schema {
            format!("{{\"schema_props\": {}}}", schema["properties"].to_string().len())
        } else if !req.stop.is_empty() {
            format!("stops: {}", req.stop.join("|"))
        } else {
            format!("echo: {}", req.messages.last().unwrap().content)
        };
        Ok(ChatResponse {
            content,
            finish: FinishReason::Stop,
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                cached_tokens: Some(3),
                reasoning_tokens: None,
            },
            constrained: req.schema.is_some(),
        })
    }
}

/// Write the same static-model fixture used in sidekick-embed's unit tests.
pub fn write_embedding_fixture(dir: &std::path::Path) {
    let model_dir = dir.join("test-static");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(
        model_dir.join("manifest.toml"),
        r#"
id = "test-static"
backend = "static"
artifact = "model.safetensors"
tokenizer = "tokenizer.json"
dims = 4
matryoshka = [4, 2]
max_seq_len = 512
"#,
    )
    .unwrap();
    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"hello": 0, "world": 1, "[UNK]": 2},
            "unk_token": "[UNK]"
        }
    });
    std::fs::write(model_dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
    let rows: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let bytes: Vec<u8> = rows.iter().flatten().flat_map(|f| f.to_le_bytes()).collect();
    let view =
        safetensors_view(&bytes, vec![3, 4]);
    let data = safetensors::serialize([("embeddings", view)], &None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), data).unwrap();
}

fn safetensors_view(bytes: &[u8], shape: Vec<usize>) -> safetensors::tensor::TensorView<'_> {
    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, bytes).unwrap()
}

/// `classifier.toml` for a fixed-label model: nlptown-style sentiment.
pub const SENTIMENT: &str = r#"
id = "sentiment"
task = "text-classification"
source = { repo = "nlptown/bert-base-multilingual-uncased-sentiment", revision = "abc123" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [16, 64]
max_seq_len = 64
max_batch = 4

[classify]
labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
output = "logits"
"#;

/// The text classifier served on the CPU with buckets past 1,024 tokens,
/// which the registry caps at 1,024 (D33).
pub fn cpu_capped() -> String {
    SENTIMENT
        .replace("id = \"sentiment\"", "id = \"cpu-capped\"")
        .replace("buckets = [16, 64]", "buckets = [16, 1024, 2048]")
        .replace("max_seq_len = 64", "max_seq_len = 2048\ncompute_units = \"cpu_only\"")
}

/// A text classifier whose manifest records its conversion-time compute
/// plan for buckets 16 and 64 (128 isn't recorded), read on this machine
/// or, with `build`, on another macOS build. With `served_on`, the model is
/// served with other compute units than the plan was read for.
pub fn placed(id: &str, build: Option<&str>, served_on: Option<&str>) -> String {
    let machine = sidekick_embed::placement::this_machine();
    let units = served_on.map(|u| format!("\ncompute_units = \"{u}\"")).unwrap_or_default();
    let plan = |ane: usize| format!("ane = {ane}\ngpu = 0\ncpu = 2\nunassigned = 5\ntotal = {}\noff_ane_ops = {{ gather = 2 }}\n", ane + 7);
    format!(
        "{}\n[placement]\ncompute_units = \"cpu_and_ne\"\nchip = \"{}\"\nmacos_build = \"{}\"\ndate = \"2026-10-01\"\n\
         [placement.buckets.16]\n{}[placement.buckets.64]\n{}",
        SENTIMENT
            .replace("id = \"sentiment\"", &format!("id = \"{id}\""))
            .replace("buckets = [16, 64]", "buckets = [16, 64, 128]")
            .replace("max_seq_len = 64", &format!("max_seq_len = 128{units}")),
        machine.chip,
        build.unwrap_or(&machine.macos_build),
        plan(30),
        plan(31),
    )
}

/// `classifier.toml` for SENTIMENT chunked in two programs per bucket (D37),
/// served on the GPU, with its recorded placement for the GPU and, as an
/// alternative, for the ANE.
pub fn chunked() -> String {
    let machine = sidekick_embed::placement::this_machine();
    let counts = |ane: usize, gpu: usize| {
        format!("ane = {ane}\ngpu = {gpu}\ncpu = 1\nunassigned = 2\ntotal = {}\noff_ane_ops = {{}}\n", ane + gpu + 3)
    };
    let tables = |prefix: &str, ane: [usize; 2], gpu: [usize; 2]| {
        format!(
            "\n[{prefix}.buckets.16]\nane = {}\ngpu = {}\ncpu = 2\nunassigned = 4\ntotal = {}\noff_ane_ops = {{}}\n\
             \n[[{prefix}.buckets.16.chunks]]\n{}\n[[{prefix}.buckets.16.chunks]]\n{}",
            ane[0] + ane[1],
            gpu[0] + gpu[1],
            ane[0] + ane[1] + gpu[0] + gpu[1] + 6,
            counts(ane[0], gpu[0]),
            counts(ane[1], gpu[1]),
        )
    };
    format!(
        "{}\n[chunking]\nchunks = 2\nweight_budget_bytes = 966367641\n\n[placement]\ncompute_units = \"cpu_and_gpu\"\n\
         chip = \"{}\"\nmacos_build = \"{}\"\n{}{}",
        SENTIMENT
            .replace("id = \"sentiment\"", "id = \"chunked\"")
            .replace("model_{seq}.mlmodelc", "model_{seq}.{chunk}.mlmodelc")
            .replace("buckets = [16, 64]", "buckets = [16]")
            .replace("max_seq_len = 64", "max_seq_len = 16\ncompute_units = \"cpu_and_gpu\""),
        machine.chip,
        machine.macos_build,
        tables("placement", [0, 0], [10, 20]),
        tables("placement.alternatives.cpu_and_ne", [9, 19], [0, 0]),
    )
}

/// `classifier.toml` for a zero-shot model in the laya format.
pub const ZERO_SHOT: &str = r#"
id = "decider"
task = "zero-shot-classification"
source = { repo = "example/decider" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [64]
max_seq_len = 64
max_batch = 2

[classify]
format = "laya"
max_labels = 4

[classify.laya]
head_max_len = 32
default_instructions = { choice = "Which option fits?", score = "Which level fits?", noul = "Does it hold?" }

[classify.calibration]
"choice:3-5" = 2.0
"noul:2" = 0.5

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
marker_pos = "marker_pos"
qtype = "qtype"
output = "logits"
"#;

/// `classifier.toml` for a zero-shot model in the gliner2 format.
pub const SCHEMA_ZERO_SHOT: &str = r#"
id = "schema-decider"
task = "zero-shot-classification"
source = { repo = "example/schema-decider" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [64]
max_seq_len = 64
max_batch = 2
compute_units = "cpu_and_gpu"

[classify]
format = "gliner2"
max_labels = 4

[classify.gliner2]
default_instructions = "label"

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
output = "logits"
"#;
/// The zero-shot model with Julia-1's option rendering, which has no
/// default instructions.
pub fn julia_zero_shot() -> String {
    ZERO_SHOT
        .replace("id = \"decider\"", "id = \"julia\"")
        .replace("\ndefault_instructions", "\noption_rendering = \"julia\"\n# default_instructions")
}

/// `classifier.toml` for a zero-shot model in the fev format.
pub const FEV_ZERO_SHOT: &str = r#"
id = "fev-decider"
task = "zero-shot-classification"
source = { repo = "example/fev-decider" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [64]
max_seq_len = 64
max_batch = 2

[classify]
format = "fev"
max_labels = 4

[classify.fev]
state_max_len = 32
delimiters = { state = "<|r0|>", question = "<|r1|>", option = "<|r2|>", option_end = "<|r3|>", decide = "<|r4|>" }

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
marker_pos = "marker_pos"
decide_pos = "decide_pos"
output = "logits"
"#;

/// `classifier.toml` for a zero-shot model in the agentjev format. AgentJev's
/// temperatures are per question type, whatever the label count, so the
/// manifest declares each type's at every label-count bucket it can reach.
pub const AGENTJEV_ZERO_SHOT: &str = r#"
id = "jev-decider"
task = "zero-shot-classification"
source = { repo = "example/jev-decider" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [64]
max_seq_len = 64
max_batch = 2
compute_units = "cpu_and_gpu"

[classify]
format = "agentjev"
max_labels = 4

[classify.calibration]
"noul:2" = 1.0718
"choice:2" = 1.0353
"choice:3-5" = 1.0353
"score:2" = 1.0718
"score:3-5" = 1.0718

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
seg = "seg"
position_ids = "position_ids"
marker_pos = "cand_end"
output = "logits"
"#;

/// `classifier.toml` for a reranker that reports raw logits (the ms-marco
/// cross-encoders' `Identity` activation).
pub const RERANKER: &str = r#"
id = "reranker"
task = "text-ranking"
source = { repo = "example/reranker", revision = "r1" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [64]
max_seq_len = 64
max_batch = 4
problem_type = "regression"

[classify]
labels = ["score"]

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
token_type_ids = "token_type_ids"
output = "logits"
"#;

/// The same with the single-output default activation: sigmoid.
pub const SIGMOID_RERANKER: &str = r#"
id = "sigmoid-reranker"
task = "text-ranking"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [64]
max_seq_len = 64

[classify]
labels = ["score"]

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
output = "logits"
"#;

/// A classifier that answers from its manifest without a model, and records
/// the params each input was prepared with. Its token ids are the input's
/// bytes (truncated to `truncate_prompt_tokens`), which `run` reads back:
/// - text-classification: logits favor the last label when the input
///   contains "good", the first otherwise;
/// - zero-shot: labels are rendered by laya's own `render_options` (so bad
///   noul labels are a 400), and a label's logit is 3 if the input
///   contains it, else 0;
/// - an input containing "nan" returns NaN logits; "fail" fails the run;
///   "reject" fails `prepare` with a client error.
///
/// As a reranker (`text-ranking`), a pair's ids are the bytes of
/// `query|document` and its logit is the number of the query's words the
/// document contains; "nan", "fail" and "reject" in the document behave as
/// above.
pub struct MockClassifier {
    pub manifest: ClassifierManifest,
    pub seen: Arc<Mutex<Vec<ClassifyParams>>>,
    /// The pair params each rerank pair was prepared with.
    pub pairs: Arc<Mutex<Vec<PairParams>>>,
    /// Inputs run so far.
    pub runs: Arc<std::sync::atomic::AtomicUsize>,
}

impl MockClassifier {
    /// The manifest's smallest bucket holding `n` tokens, as the real
    /// backends pick it, or its largest.
    fn bucket(&self, n: usize) -> usize {
        let buckets = &self.manifest.buckets;
        *buckets.iter().find(|&&b| b >= n).unwrap_or_else(|| buckets.last().expect("validated non-empty"))
    }
}

impl Classifier for MockClassifier {
    fn id(&self) -> &str {
        &self.manifest.id
    }
    fn task(&self) -> ClassifyTask {
        self.manifest.task
    }
    fn problem_type(&self) -> ProblemType {
        self.manifest.problem_type
    }
    fn labels(&self) -> &[String] {
        &self.manifest.classify.labels
    }
    fn max_labels(&self) -> usize {
        self.manifest.max_labels()
    }
    fn max_batch(&self) -> usize {
        self.manifest.max_batch
    }
    fn calibration(&self, params: &ClassifyParams, k: usize) -> Option<f32> {
        self.manifest.temperature(params.question_type, k)
    }
    fn source(&self) -> Option<&Source> {
        self.manifest.source.as_ref()
    }

    fn prepare(&self, input: &str, params: &ClassifyParams) -> Result<Prepared> {
        self.seen.lock().unwrap().push(params.clone());
        if input.contains("reject") {
            return Err(Error::InvalidRequest("rejected by prepare".into()));
        }
        let mut ids: Vec<i32> = input.bytes().map(i32::from).collect();
        if let Some(n) = params.truncate_prompt_tokens {
            ids.truncate(n);
        }
        let (markers, labels) = match self.manifest.classify.format {
            Some(ClassifyFormat::Laya) => {
                let qt = params.question_type.expect("the server requires question_type");
                let rendering = self.manifest.classify.laya.as_ref().map(|l| l.option_rendering).unwrap_or_default();
                sidekick_embed::laya::render_options(qt, &params.candidate_labels, rendering)?;
                (vec![0; params.candidate_labels.len()], params.candidate_labels.clone())
            }
            Some(ClassifyFormat::Gliner2) => {
                assert!(params.question_type.is_none(), "the server sends no question_type to gliner2");
                (vec![0; params.candidate_labels.len()], params.candidate_labels.clone())
            }
            Some(ClassifyFormat::Fev) => {
                let qt = params.question_type.expect("the server requires question_type");
                sidekick_embed::fev::render_options(qt, &params.candidate_labels)?;
                (vec![0; params.candidate_labels.len()], params.candidate_labels.clone())
            }
            Some(ClassifyFormat::Agentjev) => {
                let qt = params.question_type.expect("the server requires question_type");
                assert!(params.instructions.is_some(), "the server requires instructions");
                assert!(params.truncate_prompt_tokens.is_none(), "agentjev never truncates");
                sidekick_embed::agentjev::render_candidates(qt, &params.candidate_labels)?;
                (vec![0; params.candidate_labels.len()], params.candidate_labels.clone())
            }
            None => (vec![], vec![]),
        };
        LABELS.with(|l| *l.borrow_mut() = labels);
        let fev = self.manifest.classify.format == Some(ClassifyFormat::Fev);
        let agentjev = self.manifest.classify.format == Some(ClassifyFormat::Agentjev);
        let tree = |f: fn(usize) -> i32| if agentjev { (0..ids.len()).map(f).collect() } else { vec![] };
        let (seg, position_ids) = (tree(|_| 0), tree(|i| i as i32));
        Ok(Prepared {
            bucket: self.bucket(ids.len()),
            type_ids: vec![],
            markers,
            qtype: params.question_type.filter(|_| !fev && !agentjev).map(|q| q.index()),
            decide_pos: fev.then_some(0),
            seg,
            position_ids,
            ids,
        })
    }

    fn prepare_pair(&self, query: &str, document: &str, params: &PairParams) -> Result<Prepared> {
        assert_eq!(self.manifest.task, ClassifyTask::TextRanking, "the server sends pairs to rerankers only");
        self.pairs.lock().unwrap().push(params.clone());
        if document.contains("reject") {
            return Err(Error::InvalidRequest("rejected by prepare".into()));
        }
        let ids: Vec<i32> = format!("{query}|{document}").bytes().map(i32::from).collect();
        Ok(Prepared {
            type_ids: vec![0; ids.len()],
            bucket: self.bucket(ids.len()),
            ids,
            markers: vec![],
            qtype: None,
            decide_pos: None,
            seg: vec![],
            position_ids: vec![],
        })
    }

    fn run(&self, prepared: &Prepared) -> Result<Vec<f32>> {
        self.runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let text: String = prepared.ids.iter().map(|&b| b as u8 as char).collect();
        if self.manifest.task == ClassifyTask::TextRanking {
            let (query, document) = text.split_once('|').unwrap();
            if document.contains("fail") {
                return Err(Error::Inference("mock failure".into()));
            }
            if document.contains("nan") {
                return Ok(vec![f32::NAN]);
            }
            let shared = query.split_whitespace().filter(|w| document.split_whitespace().any(|d| d == *w)).count();
            return Ok(vec![shared as f32]);
        }
        if text.contains("fail") {
            return Err(Error::Inference("mock failure".into()));
        }
        let labels = LABELS.with(|l| l.borrow().clone());
        let k = if labels.is_empty() { self.manifest.classify.labels.len() } else { labels.len() };
        if text.contains("nan") {
            return Ok(vec![f32::NAN; k]);
        }
        Ok(if labels.is_empty() {
            let rising = text.contains("good");
            (0..k).map(|i| if rising { i as f32 } else { (k - i) as f32 }).collect()
        } else {
            labels.iter().map(|l| if text.contains(l.as_str()) { 3.0 } else { 0.0 }).collect()
        })
    }
}

thread_local! {
    /// The candidate labels of the input last prepared on this thread: `run`
    /// sees only the `Prepared`, and the server runs each input right after
    /// preparing it, on the same thread.
    static LABELS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub fn test_state(chat_available: bool, api_key: Option<&str>) -> AppState {
    test_state_with(chat_available, api_key).0
}

/// Also returns the params every mock classifier input was prepared with.
pub fn test_state_with(
    chat_available: bool,
    api_key: Option<&str>,
) -> (AppState, Arc<Mutex<Vec<ClassifyParams>>>) {
    let (state, seen, _) = test_state_probe(chat_available, api_key);
    (state, seen)
}

/// Also returns how many inputs the mock classifiers have run.
pub fn test_state_probe(
    chat_available: bool,
    api_key: Option<&str>,
) -> (AppState, Arc<Mutex<Vec<ClassifyParams>>>, Arc<std::sync::atomic::AtomicUsize>) {
    let p = test_state_full(chat_available, api_key);
    (p.state, p.seen, p.runs)
}

/// A test state and everything its mock classifiers record.
pub struct Probes {
    pub state: AppState,
    pub seen: Arc<Mutex<Vec<ClassifyParams>>>,
    pub pairs: Arc<Mutex<Vec<PairParams>>>,
    pub runs: Arc<std::sync::atomic::AtomicUsize>,
}

pub fn test_state_full(chat_available: bool, api_key: Option<&str>) -> Probes {
    test_state_scanned(chat_available, api_key, &sidekick_core::ScanOptions::default())
}

/// [`test_state_full`], its models directory scanned with `options`.
pub fn test_state_scanned(chat_available: bool, api_key: Option<&str>, options: &sidekick_core::ScanOptions) -> Probes {
    // A process-wide counter, not a timestamp: `Instant::now().elapsed()` is
    // ~0ns and collided across concurrently-running tests, letting one test
    // scan another's half-written fixture (observed as a ~1-in-5 flake).
    static FIXTURE_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "sk-server-test-{}-{}",
        std::process::id(),
        FIXTURE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    write_embedding_fixture(&dir);
    let capped = cpu_capped();
    let chunked = chunked();
    let julia = julia_zero_shot();
    let (fresh, stale, gpu) =
        (placed("placed", None, None), placed("placed-stale", Some("0Z000"), None), placed("placed-gpu", None, Some("cpu_and_gpu")));
    for (name, body) in [
        ("sentiment", SENTIMENT),
        ("chunked", chunked.as_str()),
        ("placed", fresh.as_str()),
        ("placed-stale", stale.as_str()),
        ("placed-gpu", gpu.as_str()),
        ("decider", ZERO_SHOT),
        ("schema-decider", SCHEMA_ZERO_SHOT),
        ("cpu-capped", capped.as_str()),
        ("fev-decider", FEV_ZERO_SHOT),
        ("jev-decider", AGENTJEV_ZERO_SHOT),
        ("julia", julia.as_str()),
        ("reranker", RERANKER),
        ("sigmoid-reranker", SIGMOID_RERANKER),
        ("broken", "id = "),
    ] {
        std::fs::create_dir_all(dir.join(name)).unwrap();
        std::fs::write(dir.join(name).join("classifier.toml"), body).unwrap();
    }
    let registry = Arc::new(ModelRegistry::scan_with(&dir, options).unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let pairs = Arc::new(Mutex::new(Vec::new()));
    let classifiers = {
        let registry = registry.clone();
        let (seen, runs, pairs) = (seen.clone(), runs.clone(), pairs.clone());
        ClassifierPool::new("classifier", Duration::from_secs(60), move |id| {
            let manifest = registry.classifier(id)?.manifest.clone();
            Ok(Arc::new(MockClassifier { manifest, seen: seen.clone(), pairs: pairs.clone(), runs: runs.clone() })
                as Arc<dyn Classifier>)
        })
    };
    let state = AppState {
        chat: Arc::new(MockChat { available: chat_available, info_delay: Duration::ZERO }),
        embedders: Arc::new(EmbedderPool::embedders(registry.clone(), Duration::from_secs(60))),
        classifiers: Arc::new(classifiers),
        registry,
        classifiers_supported: true,
        chat_model: Default::default(),
        api_key: api_key.map(Arc::from),
        started_at: Instant::now(),
        request_timeout: Duration::from_secs(60),
        placements: None,
    };
    Probes { state, seen, pairs, runs }
}

/// `call` that also returns the response headers.
pub async fn call_with_headers(state: AppState, req: Request<Body>) -> (StatusCode, HeaderMap, Value) {
    let response = build_router(state).oneshot(req).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let value = serde_json::from_str(&text).unwrap_or(Value::String(text));
    (status, headers, value)
}

pub async fn call(state: AppState, req: Request<Body>) -> (StatusCode, Value) {
    let response = build_router(state).oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let value = serde_json::from_str(&text).unwrap_or(Value::String(text));
    (status, value)
}

pub fn post_json(uri: &str, body: Value) -> Request<Body> {
    Request::post(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

