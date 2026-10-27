//! Warming Core ML's compile cache before requests need it (D52): in the
//! background at startup, on demand (`sidekickd warm`), and after
//! `sidekickd fetch`.
//!
//! A bucket's first load in a process compiles it for its compute path,
//! unless Core ML's cache already holds it for this executable (its name,
//! and on the ANE its code-signing identifier), home, program and macOS
//! build. Warming loads each bucket once, in
//! sidekickd and through the same loader requests use
//! (`sidekick_embed::warm`), then releases it: what's left is the cache
//! entry, which every later load of the bucket hits.
//!
//! The daemon warms in a child process: its own executable file (so the
//! same name and signing identifier), home and environment, so the same
//! cache. In one process, a Core ML load waits for an ANE compile in
//! progress, so a warm-up in the daemon's own process held up requests'
//! cache-hit loads of other buckets; in another process they don't wait.
//! Compiles themselves run one at a time across the machine (D52 has the
//! measurements).

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sidekick_core::{ComputeUnits, EmbeddingBackendKind, ModelRegistry};
use sidekick_embed::warm::{self, Key, Label, State, Status, Warmed};
use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// One bucket to warm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    pub model: String,
    pub bucket: usize,
    pub key: Key,
    /// An open chat model's directory: its decoder's chunks are the
    /// bucket, listed under its cache capacity, and load as a decoder.
    pub decoder: Option<PathBuf>,
}

impl Target {
    fn label(&self) -> Label<'_> {
        Label { model: &self.model, bucket: self.bucket }
    }

    /// Load the bucket once and release it.
    fn warm(&self) -> sidekick_core::Result<Warmed> {
        warm_one(&self.key, self.label(), self.decoder.as_deref())
    }
}

/// Warm one bucket, or an open chat model's decoder when `decoder` names
/// its directory.
fn warm_one(key: &Key, label: Label<'_>, decoder: Option<&std::path::Path>) -> sidekick_core::Result<Warmed> {
    match decoder {
        Some(dir) => crate::open_models::warm(dir, key, label),
        None => warm::warm(key, label),
    }
}

/// Every Core ML bucket the registry serves, on the compute units it
/// serves it on: the registry has already applied the length caps (D33,
/// D48) and the operator's compute units (D38), so these are the buckets a
/// request can load and nothing else. Embedders first, then classifiers
/// (none in a build that can't run them), each model's buckets smallest
/// first.
pub fn targets(registry: &ModelRegistry) -> Vec<Target> {
    let mut out = Vec::new();
    for r in registry.iter().filter(|r| r.manifest.backend == EmbeddingBackendKind::Coreml) {
        let m = &r.manifest;
        let units = m.compute_units.unwrap_or_default();
        for &bucket in &m.buckets {
            let key = Key::bucket(&r.dir, &m.artifact, bucket, m.chunks(), units);
            out.push(Target { model: m.id.clone(), bucket, key, decoder: None });
        }
    }
    for c in registry.classifiers().filter(|_| sidekick_embed::CLASSIFIERS_SUPPORTED) {
        let m = &c.manifest;
        for &bucket in &m.buckets {
            let key = Key::bucket(&c.dir, &m.artifact, bucket, m.chunks(), m.compute_units);
            out.push(Target { model: m.id.clone(), bucket, key, decoder: None });
        }
    }
    for c in registry.chats() {
        let m = &c.manifest;
        let (key, bucket) = (crate::open_models::key(&c.dir, m), crate::open_models::bucket(m));
        out.push(Target { model: m.id.clone(), bucket, key, decoder: Some(c.dir.clone()) });
    }
    out
}

/// Whether a Core ML model the registry serves has this id.
fn is_core_ml(registry: &ModelRegistry, id: &str) -> bool {
    registry.get(id).is_ok_and(|r| r.manifest.backend == EmbeddingBackendKind::Coreml)
        || (sidekick_embed::CLASSIFIERS_SUPPORTED && registry.classifier(id).is_ok())
        || registry.chat(id).is_ok()
}

/// The order to warm `targets` in: buckets not known warm first, since
/// those are what a request would wait on; then by bucket, since short
/// inputs are the most common; then by model.
pub fn ordered(targets: Vec<Target>) -> Vec<Target> {
    let mut keyed: Vec<(bool, Target)> =
        targets.into_iter().map(|t| (status(&t).is_some_and(|s| s.state == State::Warm), t)).collect();
    keyed.sort_by(|(aw, a), (bw, b)| (aw, a.bucket, &a.model).cmp(&(bw, b.bucket, &b.model)));
    keyed.into_iter().map(|(_, t)| t).collect()
}

fn status(target: &Target) -> Option<Status> {
    warm::status(&target.key)
}

/// The startup warm-up's progress, for `/health`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Not started: `warm_on_start` is off, or the build has no Core ML.
    Off,
    Running,
    /// Every bucket was visited.
    Done,
    /// The budget ran out with buckets left.
    BudgetSpent,
    /// The warm-up process couldn't start, or stopped early.
    Failed,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Phase::Off => "off",
            Phase::Running => "running",
            Phase::Done => "done",
            Phase::BudgetSpent => "budget_spent",
            Phase::Failed => "failed",
        }
    }
}

/// What the warm-up process last said about one bucket.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Reported {
    Warming,
    /// Loaded, in ms, or `None` when it didn't need loading.
    Warm(Option<u64>),
    Failed(String),
}

#[derive(Debug)]
struct Pass {
    phase: Phase,
    budget: Option<Duration>,
    started: Option<Instant>,
    finished: Option<Duration>,
    total: usize,
    /// Buckets visited: loaded, already loaded, or failed.
    done: usize,
    /// Of those, loaded by the warm-up.
    loaded: usize,
    failed: usize,
    current: Option<(String, usize)>,
    /// Why the warm-up process stopped early.
    error: Option<String>,
    buckets: HashMap<(String, usize), Reported>,
}

/// The startup warm-up: a `sidekickd warm` child process loading one bucket
/// at a time, and its progress as it reports it. A request for a bucket it
/// is compiling waits for that compile (the lock file in
/// `sidekick_embed::warm`), then loads from the cache; requests for other
/// buckets don't wait on it.
pub struct Warmup {
    pass: Mutex<Pass>,
    /// The warm-up process, and its stdin: it exits when that closes, as
    /// it does when the daemon exits, however it exits.
    child: Mutex<Option<(std::process::Child, std::process::ChildStdin)>>,
}

impl Default for Warmup {
    fn default() -> Self {
        Self::off()
    }
}

/// One bucket, as the daemon hands it to the warm-up process: exactly the
/// key its requests load, so that the child needn't scan or read config.
#[derive(Debug, Serialize, Deserialize)]
struct ChildTarget {
    model: String,
    bucket: usize,
    units: ComputeUnits,
    paths: Vec<PathBuf>,
    /// An open chat model's directory, when the bucket is its decoder.
    #[serde(default)]
    decoder: Option<PathBuf>,
}

/// What the warm-up process reports on its stdout, one JSON object a line.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
enum Event {
    Start { model: String, bucket: usize },
    Done { model: String, bucket: usize, load_ms: Option<u64> },
    Failed { model: String, bucket: usize, error: String },
    BudgetSpent { left: usize },
}

impl Warmup {
    /// A warm-up that never runs.
    pub fn off() -> Self {
        Self {
            pass: Mutex::new(Pass {
                phase: Phase::Off,
                budget: None,
                started: None,
                finished: None,
                total: 0,
                done: 0,
                loaded: 0,
                failed: 0,
                current: None,
                error: None,
                buckets: HashMap::new(),
            }),
            child: Mutex::new(None),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Pass> {
        self.pass.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Start warming the registry's Core ML buckets, but `skip`'s models', in
    /// a child process, starting no load after `budget`. A build without
    /// Core ML has nothing to warm and stays off.
    pub fn start(registry: &ModelRegistry, skip: &BTreeSet<String>, budget: Option<Duration>) -> Arc<Self> {
        let warmup = Arc::new(Self::off());
        if warm::enable().is_none() {
            return warmup;
        }
        let todo = ordered(targets(registry).into_iter().filter(|t| !skip.contains(&t.model)).collect());
        {
            let mut pass = warmup.lock();
            pass.phase = Phase::Running;
            pass.budget = budget;
            pass.started = Some(Instant::now());
            pass.total = todo.len();
        }
        let not_warm = todo.iter().filter(|t| status(t).is_none_or(|s| s.state != State::Warm)).count();
        tracing::info!(
            "warming {} Core ML bucket{} in a background process, {not_warm} not known warm",
            todo.len(),
            plural(todo.len())
        );
        let running = warmup.clone();
        let spawned = std::thread::Builder::new().name("sidekick-warmup".into()).spawn(move || {
            if let Err(e) = running.supervise(&todo, budget) {
                tracing::warn!("warm-up stopped: {e}; the buckets left compile on first use");
                running.finish(Phase::Failed, Some(e.to_string()));
            }
        });
        if let Err(e) = spawned {
            warmup.finish(Phase::Failed, Some(format!("no warm-up thread: {e}")));
        }
        warmup
    }

    /// Run the warm-up process over `todo` and follow its reports.
    fn supervise(&self, todo: &[Target], budget: Option<Duration>) -> anyhow::Result<()> {
        if todo.is_empty() {
            self.finish(Phase::Done, None);
            return Ok(());
        }
        let exe = std::env::current_exe()?;
        let mut command = std::process::Command::new(exe);
        command.args(["warm", "--child"]);
        if let Some(b) = budget {
            command.args(["--budget-secs", &b.as_secs().to_string()]);
        }
        // Its logs go where the daemon's go.
        let mut child = command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()?;
        let targets: Vec<ChildTarget> = todo
            .iter()
            .map(|t| ChildTarget {
                model: t.model.clone(),
                bucket: t.bucket,
                units: t.key.units,
                paths: t.key.paths.clone(),
                decoder: t.decoder.clone(),
            })
            .collect();
        let mut stdin = child.stdin.take().expect("piped");
        let stdout = child.stdout.take().expect("piped");
        serde_json::to_writer(&mut stdin, &targets)?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        *self.child.lock().unwrap_or_else(PoisonError::into_inner) = Some((child, stdin));
        for line in std::io::BufReader::new(stdout).lines() {
            match serde_json::from_str::<Event>(&line?) {
                Ok(event) => self.apply(event),
                Err(e) => tracing::debug!("warm-up process said something else: {e}"),
            }
        }
        let status = match self.child.lock().unwrap_or_else(PoisonError::into_inner).take() {
            Some((mut child, stdin)) => {
                drop(stdin);
                child.wait()?
            }
            None => return Ok(()), // stopped by `stop`
        };
        anyhow::ensure!(status.success(), "the warm-up process exited with {status}");
        // Done, unless it said its budget ran out.
        self.finish(Phase::Done, None);
        Ok(())
    }

    fn apply(&self, event: Event) {
        let mut pass = self.lock();
        match event {
            Event::Start { model, bucket } => {
                pass.current = Some((model.clone(), bucket));
                pass.buckets.insert((model, bucket), Reported::Warming);
            }
            Event::Done { model, bucket, load_ms } => {
                pass.done += 1;
                pass.loaded += usize::from(load_ms.is_some());
                pass.current = None;
                pass.buckets.insert((model, bucket), Reported::Warm(load_ms));
            }
            Event::Failed { model, bucket, error } => {
                pass.done += 1;
                pass.failed += 1;
                pass.current = None;
                pass.buckets.insert((model, bucket), Reported::Failed(error));
            }
            Event::BudgetSpent { .. } => pass.phase = Phase::BudgetSpent,
        }
    }

    fn finish(&self, phase: Phase, error: Option<String>) {
        let mut pass = self.lock();
        if pass.phase == Phase::Running {
            pass.phase = phase;
        }
        pass.finished = pass.started.map(|s| s.elapsed());
        pass.current = None;
        pass.error = pass.error.take().or(error);
        if pass.phase == Phase::Done {
            tracing::info!(
                "warm-up done in {:.1?}: {} loaded, {} failed, {} already warm",
                pass.finished.unwrap_or_default(),
                pass.loaded,
                pass.failed,
                pass.done - pass.loaded - pass.failed
            );
        }
    }

    /// Stop the warm-up process, as the daemon shuts down.
    pub fn stop(&self) {
        if let Some((mut child, stdin)) = self.child.lock().unwrap_or_else(PoisonError::into_inner).take() {
            drop(stdin);
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// A bucket's state: this daemon's own loads first, then what the
    /// warm-up process reported, then the ledger. `None` without Core ML.
    fn state(&self, target: &Target) -> Option<(State, Status, Option<Reported>)> {
        let own = status(target)?;
        let reported = self.lock().buckets.get(&(target.model.clone(), target.bucket)).cloned();
        let state = match (&own.state, &reported) {
            (State::Warming | State::Warm, _) => own.state,
            (_, Some(Reported::Warming)) => State::Warming,
            (_, Some(Reported::Warm(_))) => State::Warm,
            (State::Cold, Some(Reported::Failed(_))) => State::Failed,
            _ => own.state,
        };
        Some((state, own, reported))
    }

    /// `/v1/models`' `compile_cache` for one model: each bucket's state, or
    /// `None` for a model that isn't Core ML (or a build without it).
    pub fn compile_cache(&self, targets: &[Target], model: &str) -> Option<std::collections::BTreeMap<String, &'static str>> {
        let states: std::collections::BTreeMap<String, &'static str> = targets
            .iter()
            .filter(|t| t.model == model)
            .filter_map(|t| Some((t.bucket.to_string(), self.state(t)?.0.name())))
            .collect();
        (!states.is_empty()).then_some(states)
    }

    /// `/health`'s `warmup`: the startup warm-up, and every Core ML bucket's
    /// state, by model.
    pub fn health(&self, registry: &ModelRegistry) -> Value {
        let mut models = Map::new();
        for target in targets(registry) {
            let Some((state, own, reported)) = self.state(&target) else { continue };
            let entry = models
                .entry(target.model.clone())
                .or_insert_with(|| json!({"compute_units": target.key.units.name(), "buckets": {}}));
            let mut bucket = json!({"state": state.name(), "loaded": own.loaded});
            if let Some(ms) = own.load_ms {
                bucket["load_ms"] = json!(ms);
            }
            match reported {
                Some(Reported::Warm(Some(ms))) => bucket["warmup_load_ms"] = json!(ms),
                Some(Reported::Failed(e)) => bucket["warmup_error"] = json!(e),
                _ => {}
            }
            if let Some(ms) = own.recorded_ms {
                bucket["recorded_load_ms"] = json!(ms);
            }
            if let Some(e) = own.error {
                bucket["error"] = json!(e);
            }
            entry["buckets"][target.bucket.to_string()] = bucket;
        }
        let pass = self.lock();
        let elapsed = pass.finished.or_else(|| pass.started.map(|s| s.elapsed()));
        json!({
            "state": pass.phase.name(),
            "budget_secs": pass.budget.map(|b| b.as_secs()),
            "elapsed_secs": elapsed.map(|e| (e.as_secs_f64() * 10.0).round() / 10.0),
            "buckets": {"total": pass.total, "done": pass.done, "loaded": pass.loaded, "failed": pass.failed},
            "current": pass.current.as_ref().map(|(model, bucket)| json!({"model": model, "bucket": bucket})),
            "error": pass.error,
            "models": models,
        })
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// `sidekickd warm`.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct WarmArgs {
    /// The models to warm (default: every Core ML model the daemon serves,
    /// but those whose `[models."<id>"]` says `warm = false`)
    pub models: Vec<String>,
    /// Only these sequence-length buckets, comma-separated
    #[arg(long, value_delimiter = ',')]
    pub buckets: Vec<usize>,
    /// Start no load after this many seconds (default: no limit)
    #[arg(long, value_name = "SECS")]
    pub budget_secs: Option<u64>,
    /// List each bucket's state without loading anything
    #[arg(long)]
    pub list: bool,
    /// The daemon's warm-up process: buckets as JSON on stdin, progress as
    /// JSON lines on stdout
    #[arg(long, hide = true)]
    pub child: bool,
}

/// What `sidekickd warm` chose to warm: `args`' models (all of the
/// registry's Core ML models but `skip`'s when it names none) and buckets.
pub fn chosen(registry: &ModelRegistry, args: &WarmArgs, skip: &BTreeSet<String>) -> anyhow::Result<Vec<Target>> {
    for id in &args.models {
        let known = registry.get(id).is_ok() || registry.classifier(id).is_ok() || registry.chat(id).is_ok();
        anyhow::ensure!(known, "no model `{id}` is served from the models directory");
        anyhow::ensure!(is_core_ml(registry, id), "`{id}` isn't a Core ML model: it has no compile cache to warm");
    }
    let chosen: Vec<Target> = targets(registry)
        .into_iter()
        .filter(|t| if args.models.is_empty() { !skip.contains(&t.model) } else { args.models.contains(&t.model) })
        .filter(|t| args.buckets.is_empty() || args.buckets.contains(&t.bucket))
        .collect();
    for &b in &args.buckets {
        anyhow::ensure!(
            chosen.iter().any(|t| t.bucket == b),
            "no chosen model is served with a {b}-token bucket (the length caps, D33 and D48, can leave one out)"
        );
    }
    Ok(chosen)
}

/// Run `sidekickd warm`: load each chosen bucket in this process, one at a
/// time, and print what happened. Fails when any load failed.
pub fn run(registry: &ModelRegistry, args: &WarmArgs, skip: &BTreeSet<String>) -> anyhow::Result<()> {
    let ledger = warm::enable();
    anyhow::ensure!(ledger.is_some() || args.list, "this build has no Core ML: nothing to warm");
    let todo = ordered(chosen(registry, args, skip)?);
    if todo.is_empty() {
        println!("no Core ML buckets to warm");
        return Ok(());
    }
    if args.list {
        for t in &todo {
            let s = status(t);
            let state = s.as_ref().map_or("unknown", |s| s.state.name());
            let recorded = s.and_then(|s| s.recorded_ms).map(|ms| format!("  (recorded load {})", secs(ms))).unwrap_or_default();
            println!("{:<32} {:>6}  {:<11} {state}{recorded}", t.model, t.bucket, t.key.units.name());
        }
        return Ok(());
    }
    let failed = load_all(&todo, args.budget_secs.map(Duration::from_secs));
    anyhow::ensure!(failed == 0, "{failed} bucket{} failed to load", plural(failed));
    Ok(())
}

/// Load `todo` in order, printing a line per bucket; returns how many
/// failed.
fn load_all(todo: &[Target], budget: Option<Duration>) -> usize {
    let started = Instant::now();
    let mut failed = 0;
    for (i, t) in todo.iter().enumerate() {
        if budget.is_some_and(|b| started.elapsed() >= b) {
            println!("budget spent after {:.0?}: {} bucket{} not warmed", started.elapsed(), todo.len() - i, plural(todo.len() - i));
            break;
        }
        let before = status(t).map_or(State::Cold, |s| s.state);
        let line = format!("{:<32} {:>6}  {:<11}", t.model, t.bucket, t.key.units.name());
        match t.warm() {
            Ok(Warmed::Loaded { load_ms, .. }) => println!("{line} {} -> warm  {}", before.name(), secs(load_ms)),
            Ok(Warmed::AlreadyLoaded | Warmed::LoadedElsewhere) => println!("{line} warm (loaded by another process)"),
            Err(e) => {
                failed += 1;
                println!("{line} failed: {e}");
            }
        }
    }
    println!("done in {:.1?}", started.elapsed());
    failed
}

fn secs(ms: u64) -> String {
    format!("{:.2} s", ms as f64 / 1000.0)
}

/// The daemon's warm-up process (`sidekickd warm --child`): the buckets as
/// one JSON line on stdin, loaded in order, each reported on stdout. Exits
/// when stdin closes, as it does when the daemon exits, even mid-compile.
pub fn run_child(budget: Option<Duration>) -> anyhow::Result<()> {
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    let todo: Vec<ChildTarget> = serde_json::from_str(&line)?;
    std::thread::spawn(|| {
        let _ = std::io::copy(&mut std::io::stdin().lock(), &mut std::io::sink());
        tracing::debug!("the daemon is gone: the warm-up process stops");
        std::process::exit(0);
    });
    anyhow::ensure!(warm::enable().is_some(), "this build has no Core ML: nothing to warm");
    let mut out = std::io::stdout().lock();
    let mut say = |event: Event| -> anyhow::Result<()> {
        serde_json::to_writer(&mut out, &event)?;
        out.write_all(b"\n")?;
        Ok(out.flush()?)
    };
    let started = Instant::now();
    for (i, t) in todo.iter().enumerate() {
        if budget.is_some_and(|b| started.elapsed() >= b) {
            tracing::warn!(
                "warm-up budget spent after {:.0?}: {} bucket{} left to compile on first use",
                started.elapsed(),
                todo.len() - i,
                plural(todo.len() - i)
            );
            return say(Event::BudgetSpent { left: todo.len() - i });
        }
        let (model, bucket) = (t.model.clone(), t.bucket);
        say(Event::Start { model: model.clone(), bucket })?;
        let key = Key { paths: t.paths.clone(), units: t.units };
        let event = match warm_one(&key, Label { model: &model, bucket }, t.decoder.as_deref()) {
            Ok(Warmed::Loaded { load_ms, recorded }) => {
                tracing::info!(
                    model = %model,
                    bucket,
                    units = t.units.name(),
                    load_ms,
                    "warmed{}",
                    if recorded { " (recorded warm, so a cache hit)" } else { "" }
                );
                Event::Done { model, bucket, load_ms: Some(load_ms) }
            }
            Ok(Warmed::AlreadyLoaded | Warmed::LoadedElsewhere) => Event::Done { model, bucket, load_ms: None },
            Err(e) => {
                tracing::warn!(model = %model, bucket, units = t.units.name(), "warm-up load failed: {e}");
                Event::Failed { model, bucket, error: e.to_string() }
            }
        };
        say(event)?;
    }
    Ok(())
}

/// After `sidekickd fetch` installed `id`: warm its buckets, unless the
/// daemon wouldn't serve it, it isn't Core ML, or the config leaves it
/// out. Never fails the fetch: a bucket that fails to load is reported.
pub fn after_fetch(registry: &ModelRegistry, id: &str, skip: &BTreeSet<String>) {
    if skip.contains(id) || !is_core_ml(registry, id) || warm::enable().is_none() {
        return;
    }
    let todo: Vec<Target> = targets(registry).into_iter().filter(|t| t.model == id).collect();
    if todo.is_empty() {
        return;
    }
    println!(
        "warming {id}'s {} bucket{} on {} so that its first requests don't wait on a compile (--no-warm skips this)",
        todo.len(),
        plural(todo.len()),
        todo[0].key.units.name()
    );
    let failed = load_all(&ordered(todo), None);
    if failed > 0 {
        println!("  warning: {failed} bucket{} failed to load; sidekickd will try again on first use", plural(failed));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn models_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sk-server-warm-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn write(dir: &Path, id: &str, file: &str, body: &str) {
        std::fs::create_dir_all(dir.join(id)).unwrap();
        std::fs::write(dir.join(id).join(file), body).unwrap();
        std::fs::write(dir.join(id).join("tokenizer.json"), "{}").unwrap();
    }

    fn embedder(id: &str, buckets: &str, max: usize, extra: &str) -> String {
        format!(
            "id = \"{id}\"\nbackend = \"coreml\"\nartifact = \"model_{{seq}}.mlmodelc\"\ntokenizer = \"tokenizer.json\"\n\
             dims = 4\npooling = \"none\"\nbuckets = {buckets}\nmax_seq_len = {max}\n{extra}\
             [io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\noutput = \"embedding\"\n"
        )
    }

    #[test]
    fn targets_are_the_served_buckets_on_the_served_units() {
        let dir = models_dir("targets");
        write(&dir, "small", "manifest.toml", &embedder("small", "[128, 512]", 512, ""));
        // Past the ANE cap (D48): 4,096 isn't served, so it isn't warmed.
        write(&dir, "long", "manifest.toml", &embedder("long", "[1024, 2048, 4096]", 4096, ""));
        write(&dir, "gpu", "manifest.toml", &embedder("gpu", "[128]", 128, "compute_units = \"cpu_and_gpu\"\n"));
        let options = sidekick_core::ScanOptions {
            compute_units: [("small".to_string(), ComputeUnits::CpuOnly)].into(),
            ..Default::default()
        };
        let registry = ModelRegistry::scan_with(&dir, &options).unwrap();
        let mut got: Vec<(String, usize, ComputeUnits)> =
            targets(&registry).into_iter().map(|t| (t.model, t.bucket, t.key.units)).collect();
        got.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        assert_eq!(
            got,
            [
                ("gpu".into(), 128, ComputeUnits::CpuAndGpu),
                ("long".into(), 1024, ComputeUnits::CpuAndNeuralEngine),
                ("long".into(), 2048, ComputeUnits::CpuAndNeuralEngine),
                // The operator's compute units (D38).
                ("small".into(), 128, ComputeUnits::CpuOnly),
                ("small".into(), 512, ComputeUnits::CpuOnly),
            ]
        );
        let t = targets(&registry).into_iter().find(|t| t.model == "small" && t.bucket == 512).unwrap();
        assert_eq!(t.key.paths, [dir.join("small").join("model_512.mlmodelc")], "the path a request loads");

        // Cold first, then by bucket, then by model.
        let order: Vec<(String, usize)> = ordered(targets(&registry)).into_iter().map(|t| (t.model, t.bucket)).collect();
        assert_eq!(order[0].1, 128);
        assert!(order.windows(2).all(|w| w[0].1 <= w[1].1), "{order:?}");

        // `sidekickd warm`'s choices.
        let skip: BTreeSet<String> = ["long".to_string()].into();
        let all = chosen(&registry, &WarmArgs::default(), &skip).unwrap();
        assert!(all.iter().all(|t| t.model != "long"), "skipped by config");
        let named = WarmArgs { models: vec!["long".into()], buckets: vec![2048], ..Default::default() };
        let one = chosen(&registry, &named, &skip).unwrap();
        assert_eq!(one.iter().map(|t| (t.model.as_str(), t.bucket)).collect::<Vec<_>>(), [("long", 2048)], "named overrides the config");
        let capped = WarmArgs { models: vec!["long".into()], buckets: vec![4096], ..Default::default() };
        assert!(chosen(&registry, &capped, &skip).unwrap_err().to_string().contains("4096-token bucket"));
        let unknown = WarmArgs { models: vec!["nope".into()], ..Default::default() };
        assert!(chosen(&registry, &unknown, &skip).unwrap_err().to_string().contains("no model `nope`"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_off_warmup_reports_off() {
        let health = Warmup::off().health(&ModelRegistry::default());
        assert_eq!(health["state"], "off");
        assert_eq!(health["buckets"]["total"], 0);
    }

    #[test]
    fn the_warm_up_processs_reports_update_progress_and_bucket_states() {
        let warmup = Warmup::off();
        warmup.lock().phase = Phase::Running;
        warmup.lock().total = 3;
        let lines = [
            r#"{"event":"start","model":"m","bucket":128}"#,
            r#"{"event":"done","model":"m","bucket":128,"load_ms":4200}"#,
            r#"{"event":"start","model":"m","bucket":256}"#,
        ];
        for line in lines {
            warmup.apply(serde_json::from_str(line).unwrap());
        }
        {
            let pass = warmup.lock();
            assert_eq!((pass.done, pass.loaded, pass.failed), (1, 1, 0));
            assert_eq!(pass.current, Some(("m".to_string(), 256)));
            assert_eq!(pass.buckets[&("m".to_string(), 128)], Reported::Warm(Some(4200)));
            assert_eq!(pass.buckets[&("m".to_string(), 256)], Reported::Warming);
        }
        warmup.apply(serde_json::from_str(r#"{"event":"failed","model":"m","bucket":256,"error":"E5RT"}"#).unwrap());
        warmup.apply(serde_json::from_str(r#"{"event":"budget_spent","left":1}"#).unwrap());
        let pass = warmup.lock();
        assert_eq!((pass.done, pass.loaded, pass.failed, pass.phase), (2, 1, 1, Phase::BudgetSpent));
        // What the child writes, the daemon reads.
        let written = serde_json::to_string(&Event::Done { model: "m".into(), bucket: 1, load_ms: None }).unwrap();
        assert_eq!(written, r#"{"event":"done","model":"m","bucket":1,"load_ms":null}"#);
    }
}
