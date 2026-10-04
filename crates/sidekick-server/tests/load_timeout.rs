//! Loading isn't counted against `request_timeout_secs` (api::deadline):
//! fakes whose model loads, bucket loads and predictions take as long as
//! a test says, behind the real classify, rerank and embeddings routes.
//! A load longer than the request timeout doesn't 504; a prediction that
//! outlasts it still does, and says what it was; a load past
//! `load_timeout_secs` is its own 504, and keeps going.

mod common;

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use common::*;
use serde_json::{json, Value};
use sidekick_core::{
    Classifier, ClassifyParams, ClassifyTask, EmbedLimits, EmbedPurpose, Embedder, PairParams, Prepared,
    ProblemType, Result, Source,
};
use sidekick_server::{AppState, ClassifierPool, EmbedderPool};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// Wide margins, so a slow or busy machine (a shared CI runner) can't turn
// a pass into a fail: every step is a second or more away from the bound
// it is tested against.
const REQUEST: Duration = Duration::from_secs(2);
/// A load, past the request timeout.
const SLOW: Duration = Duration::from_secs(3);
/// A prediction past the request timeout.
const HUNG: Duration = Duration::from_secs(4);
/// A prediction well inside the request timeout, even two in a row, but
/// still running when a deadline counted from arrival would have passed.
const QUICK: Duration = Duration::from_millis(300);

/// How long each step of a fake takes.
#[derive(Clone, Copy, Default)]
struct Delays {
    /// The pool's load of the model.
    model: Duration,
    /// Each bucket's first load.
    bucket: Duration,
    /// Each prediction.
    run: Duration,
}

/// Buckets that load once, taking `delay`: a caller that finds the bucket
/// being loaded waits on that load, as `BucketModels` makes it.
#[derive(Default)]
struct Buckets {
    loaded: Mutex<HashSet<usize>>,
}

impl Buckets {
    fn load(&self, buckets: &[usize], delay: Duration) -> Vec<usize> {
        let mut waited = Vec::new();
        let mut unique = buckets.to_vec();
        unique.sort_unstable();
        unique.dedup();
        for b in unique {
            // The lock is held for the load: a concurrent caller waits on it.
            let (mut loaded, contended) = match self.loaded.try_lock() {
                Ok(l) => (l, false),
                Err(_) => (self.loaded.lock().unwrap(), true),
            };
            if !loaded.contains(&b) {
                std::thread::sleep(delay);
                loaded.insert(b);
                waited.push(b);
            } else if contended {
                waited.push(b);
            }
        }
        waited
    }
}

/// A [`MockClassifier`] with delays.
struct SlowClassifier {
    inner: MockClassifier,
    delays: Delays,
    buckets: Arc<Buckets>,
}

impl Classifier for SlowClassifier {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn task(&self) -> ClassifyTask {
        self.inner.task()
    }
    fn problem_type(&self) -> ProblemType {
        self.inner.problem_type()
    }
    fn labels(&self) -> &[String] {
        self.inner.labels()
    }
    fn max_labels(&self) -> usize {
        self.inner.max_labels()
    }
    fn max_batch(&self) -> usize {
        self.inner.max_batch()
    }
    fn calibration(&self, params: &ClassifyParams, k: usize) -> Option<f32> {
        self.inner.calibration(params, k)
    }
    fn source(&self) -> Option<&Source> {
        self.inner.source()
    }
    fn prepare(&self, input: &str, params: &ClassifyParams) -> Result<Prepared> {
        self.inner.prepare(input, params)
    }
    fn prepare_pair(&self, query: &str, document: &str, params: &PairParams) -> Result<Prepared> {
        self.inner.prepare_pair(query, document, params)
    }
    fn load_buckets(&self, buckets: &[usize]) -> Result<Vec<usize>> {
        Ok(self.buckets.load(buckets, self.delays.bucket))
    }
    fn run(&self, prepared: &Prepared) -> Result<Vec<f32>> {
        std::thread::sleep(self.delays.run);
        self.inner.run(prepared)
    }
}

/// A 4-dimensional embedder with delays and one 64-token bucket.
struct SlowEmbedder {
    delays: Delays,
    buckets: Arc<Buckets>,
}

impl Embedder for SlowEmbedder {
    fn id(&self) -> &str {
        "test-static"
    }
    fn dims(&self) -> usize {
        4
    }
    fn embed(&self, texts: &[&str], purpose: EmbedPurpose) -> Result<Vec<Vec<f32>>> {
        Ok(self.embed_bucketed(texts, purpose, EmbedLimits::default())?.0)
    }
    fn embed_bucketed(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        limits: EmbedLimits,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<usize>>)> {
        self.embed_staged(texts, purpose, limits, &mut |_| {})
    }
    fn embed_staged(
        &self,
        texts: &[&str],
        _: EmbedPurpose,
        _: EmbedLimits,
        loaded: &mut dyn FnMut(&[usize]),
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<usize>>)> {
        loaded(&self.buckets.load(&[64], self.delays.bucket));
        std::thread::sleep(self.delays.run);
        Ok((texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect(), Some(vec![64; texts.len()])))
    }
}

/// The test state, its classifiers and its embedder slowed by `delays`,
/// with the given request and load timeouts.
fn slow_state(delays: Delays, load_timeout: Duration) -> AppState {
    slow_state_with(delays, load_timeout).0
}

/// [`slow_state`], and the fakes' buckets.
fn slow_state_with(delays: Delays, load_timeout: Duration) -> (AppState, Arc<Buckets>) {
    let mut state = test_state(true, None);
    let registry = state.registry.clone();
    let buckets = Arc::new(Buckets::default());
    let (b, out) = (buckets.clone(), buckets.clone());
    state.classifiers = Arc::new(ClassifierPool::new("classifier", Duration::from_secs(60), move |id| {
        std::thread::sleep(delays.model);
        let inner = MockClassifier {
            manifest: registry.classifier(id)?.manifest.clone(),
            seen: Default::default(),
            pairs: Default::default(),
            runs: Default::default(),
        };
        Ok(Arc::new(SlowClassifier { inner, delays, buckets: b.clone() }) as Arc<dyn Classifier>)
    }));
    state.embedders = Arc::new(EmbedderPool::new("embedding model", Duration::from_secs(60), move |_| {
        std::thread::sleep(delays.model);
        Ok(Arc::new(SlowEmbedder { delays, buckets: buckets.clone() }) as Arc<dyn Embedder>)
    }));
    state.request_timeout = REQUEST;
    state.load_timeout = load_timeout;
    (state, out)
}

/// `state` with its models and every fake bucket already loaded.
async fn warm(state: &AppState, buckets: &Buckets) {
    buckets.loaded.lock().unwrap().extend([16, 64]);
    for id in ["sentiment", "reranker"] {
        state.classifiers.get(id).await.unwrap();
    }
    state.embedders.get("test-static").await.unwrap();
}

const LONG_LOAD: Duration = Duration::from_secs(30);

fn classify() -> Request<Body> {
    post_json("/v1/classify", json!({"model": "sentiment", "input": ["a good film", "a much longer input, past 16 bytes"]}))
}

fn rerank() -> Request<Body> {
    post_json("/v1/rerank", json!({"model": "reranker", "query": "cat", "documents": ["a cat", "a dog"]}))
}

fn embed() -> Request<Body> {
    post_json("/v1/embeddings", json!({"model": "test-static", "input": "hello"}))
}

fn embed_v2() -> Request<Body> {
    post_json("/v2/embed", json!({"model": "test-static", "texts": ["hello"], "input_type": "search_document"}))
}

fn message(body: &Value) -> &str {
    body["error"]["message"].as_str().unwrap_or_default()
}

async fn status(state: AppState, req: Request<Body>) -> (StatusCode, HeaderMap, Value) {
    call_with_headers(state, req).await
}

/// Each request, on its own fresh state, concurrently.
async fn on_fresh_states(delays: Delays, requests: &[fn() -> Request<Body>]) -> Vec<(StatusCode, HeaderMap, Value)> {
    let calls = requests.iter().map(|req| {
        let (state, req) = (slow_state(delays, LONG_LOAD), req());
        tokio::spawn(async move { status(state, req).await })
    });
    futures::future::join_all(calls).await.into_iter().map(|r| r.unwrap()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_model_load_longer_than_the_request_timeout_doesnt_504() {
    let delays = Delays { model: SLOW, run: QUICK, ..Default::default() };
    for (code, _, body) in on_fresh_states(delays, &[classify, rerank, embed, embed_v2]).await {
        assert_eq!(code, StatusCode::OK, "{body}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bucket_load_longer_than_the_request_timeout_doesnt_504() {
    let delays = Delays { bucket: SLOW, run: QUICK, ..Default::default() };
    for (code, headers, body) in on_fresh_states(delays, &[classify, rerank, embed, embed_v2]).await {
        assert_eq!(code, StatusCode::OK, "{body}");
        assert!(headers.contains_key("sidekick-buckets"));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn requests_waiting_on_another_requests_bucket_load_dont_504() {
    let state = slow_state(Delays { bucket: SLOW, run: QUICK, ..Default::default() }, LONG_LOAD);
    // The model resident first, so every request below waits on the bucket
    // load alone: one of them loads it, the others wait on that load.
    let _ = state.classifiers.get("sentiment").await.unwrap();
    let requests = (0..4).map(|_| {
        let (state, req) = (state.clone(), post_json("/v1/classify", json!({"model": "sentiment", "input": "short"})));
        tokio::spawn(async move { status(state, req).await })
    });
    for done in futures::future::join_all(requests).await {
        let (code, _, body) = done.unwrap();
        assert_eq!(code, StatusCode::OK, "{body}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_slow_prediction_still_times_out_and_says_what_it_was() {
    let delays = Delays { run: HUNG, ..Default::default() };
    let routes =
        [(classify as fn() -> Request<Body>, "Classification"), (rerank, "Reranking"), (embed, "Embedding"), (embed_v2, "Embedding")];
    let calls = routes.map(|(req, work)| async move {
        let (state, buckets) = slow_state_with(delays, LONG_LOAD);
        warm(&state, &buckets).await;
        (status(state, req()).await, work)
    });
    for ((code, _, body), work) in futures::future::join_all(calls).await {
        assert_eq!(code, StatusCode::GATEWAY_TIMEOUT, "{body}");
        assert_eq!(body["error"]["code"], "timeout");
        assert_eq!(message(&body), format!("{work} did not complete within 2s"));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_slow_prediction_after_a_load_times_out_from_the_end_of_the_load() {
    let classified = async {
        let delays = Delays { bucket: SLOW, run: HUNG, ..Default::default() };
        status(slow_state(delays, LONG_LOAD), classify()).await
    };
    let embedded = async {
        let delays = Delays { model: SLOW, run: HUNG, ..Default::default() };
        status(slow_state(delays, LONG_LOAD), embed()).await
    };
    let ((code, _, body), (_, _, embedded)) = tokio::join!(classified, embedded);
    assert_eq!(code, StatusCode::GATEWAY_TIMEOUT, "{body}");
    let m = message(&body);
    assert!(
        m.starts_with(
            "Classification did not complete within 2s of model `sentiment` finishing its load (the model and buckets 16, 64, "
        ),
        "{m}"
    );
    assert!(m.ends_with("s, not counted)"), "{m}");
    assert!(message(&embedded).contains("finishing its load (the model and bucket 64, "), "{embedded}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_load_past_the_load_timeout_is_its_own_504_and_keeps_loading() {
    let state = slow_state(Delays { bucket: SLOW, ..Default::default() }, Duration::from_secs(1));
    let one = || post_json("/v1/classify", json!({"model": "sentiment", "input": "a good film"}));
    let (code, headers, body) = status(state.clone(), one()).await;
    assert_eq!(code, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert_eq!(body["error"]["code"], "load_timeout");
    assert!(message(&body).starts_with("Model `sentiment` was still loading after 1s"), "{body}");
    assert_eq!(headers["retry-after"], "30");
    // The load finishes in the background; a retry after it finds the bucket
    // resident and succeeds.
    tokio::time::sleep(SLOW + Duration::from_secs(1)).await;
    let (code, _, body) = status(state, one()).await;
    assert_eq!(code, StatusCode::OK, "{body}");
}
