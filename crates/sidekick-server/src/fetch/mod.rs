//! `sidekickd fetch`: download a published build from Hugging Face, verify
//! it, and install it in the models directory (#137, D51).
//!
//! 1. Files come into the Hugging Face cache: through the user's own `hf`
//!    CLI when it is installed (Xet, its token and configuration), else
//!    through sidekickd's anonymous downloader, which writes the same
//!    layout (`hfcache`), so each sees the other's downloads as cached.
//! 2. Every variant's manifest comes first, with the repository's terms
//!    (TERMS.md, a tier-2b model's), which must be accepted before anything
//!    else downloads. The chosen variant's buckets become `--include`
//!    patterns (`plan`), and the rest of that build follows.
//! 3. Every file is checked against the Hub's metadata and against
//!    sidekick's checksum ledger, built into sidekickd (`verify`).
//! 4. The model directory is a real directory of hard links to the cached
//!    files, with the manifest for exactly the buckets fetched (`install`).
//!
//! The repositories are docs/PUBLISHING.md's: `sidekickd/<model id>-coreml`
//! and `-onnx`, one directory per variant holding the install tree as is.

mod hfcache;
mod hub;
mod install;
mod pattern;
mod plan;
mod source;
mod verify;

use anyhow::{bail, Context};
use clap::ValueEnum;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// `sidekickd fetch` options.
#[derive(clap::Args, Debug, Clone)]
pub struct FetchArgs {
    /// The model id, as docs/MODELS.md lists it (bge-small-en-v1.5)
    pub model: String,
    /// The backend whose build to fetch: each backend's builds are their own
    /// repository
    #[arg(long, value_enum, default_value_t = Backend::default_for_platform())]
    pub backend: Backend,
    /// The variant to fetch (`fp16`, `fp16-max2048`): its directory in the
    /// repository. Default: the one most recently published, or the only one
    #[arg(long)]
    pub variant: Option<String>,
    /// Only these sequence-length buckets of a Core ML build, comma-separated
    /// (default: all of them). The model is then served up to the longest
    #[arg(long, value_delimiter = ',')]
    pub buckets: Vec<usize>,
    /// Accept the model's terms (its repository's TERMS.md), naming the
    /// model; fetch shows them and stops without it
    #[arg(long, value_name = "MODEL")]
    pub accept_terms: Option<String>,
    /// The Hugging Face repository (`<owner>/<name>`). Default: where sidekick's
    /// checksum ledger records the build as published
    #[arg(long)]
    pub repo: Option<String>,
    /// The branch, tag or commit to fetch. Default: the published revision
    /// the ledger records, else `main`
    #[arg(long)]
    pub revision: Option<String>,
    /// Download with sidekickd's own downloader even when `hf` is installed
    #[arg(long)]
    pub no_hf: bool,
    /// Verify against this checksum ledger file instead of the one built
    /// into sidekickd
    #[arg(long, value_name = "FILE")]
    pub ledger: Option<PathBuf>,
    /// Replace a model directory that `sidekickd fetch` didn't install
    #[arg(long)]
    pub force: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Backend {
    Coreml,
    Onnx,
}

impl Backend {
    fn default_for_platform() -> Self {
        if cfg!(target_os = "macos") {
            Backend::Coreml
        } else {
            Backend::Onnx
        }
    }

    fn name(self) -> &'static str {
        match self {
            Backend::Coreml => "coreml",
            Backend::Onnx => "onnx",
        }
    }
}

/// A repository's terms, which a tier-2b model ships (docs/PUBLISHING.md).
const TERMS: &str = "TERMS.md";

/// A Hub repo id: `<owner>/<name>`, of the characters the Hub allows.
fn check_repo_id(repo: &str) -> anyhow::Result<()> {
    let part_ok = |p: &str| {
        !p.is_empty() && p != "." && p != ".." && p.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
    };
    let parts: Vec<&str> = repo.split('/').collect();
    anyhow::ensure!(parts.len() == 2 && parts.iter().all(|p| part_ok(p)), "`{repo}` isn't a Hugging Face repository id (`<owner>/<name>`)");
    Ok(())
}

/// What a fetch of a gated repository needs from the user, which fetch
/// can't do for them.
fn gated_hint(repo_id: &str, model: &str) -> String {
    format!(
        "If the repository is gated (a model whose terms must be accepted), sign in at {}, read and accept its \
         terms there, sign the `hf` CLI in with the same account (`hf auth login`), and fetch again with \
         --accept-terms {model}",
        hub::repo_page(repo_id)
    )
}

/// Run `sidekickd fetch`.
pub fn run(args: &FetchArgs, models_dir: &Path) -> anyhow::Result<()> {
    let backend = args.backend.name();
    let ledger_source = match &args.ledger {
        Some(path) => verify::LedgerSource::File(path.clone()),
        None => verify::LedgerSource::BuiltIn,
    };
    let asked_variant = args.variant.as_deref().filter(|v| !v.is_empty());
    if let Some(v) = asked_variant {
        anyhow::ensure!(hfcache::check_repo_path(v).is_ok() && !v.contains('/'), "`{v}` isn't a variant directory name");
    }
    let ledger = verify::builds(&ledger_source, &args.model, backend)?;
    let published = verify::latest_published(&ledger, asked_variant);
    let repo_id = match (&args.repo, published) {
        (Some(repo), _) => repo.clone(),
        (None, Some((_, p))) => p.repo.clone(),
        (None, None) => bail!(
            "sidekick's checksum ledger records no published {backend} build of `{}`{}; name a repository with --repo",
            args.model,
            asked_variant.map(|v| format!(" ({v})")).unwrap_or_default()
        ),
    };
    check_repo_id(&repo_id)?;
    let revision = args
        .revision
        .clone()
        .or_else(|| published.filter(|(_, p)| p.repo == repo_id).map(|(_, p)| p.revision.clone()))
        .unwrap_or_else(|| "main".into());
    hfcache::check_repo_path(&revision).map_err(|_| anyhow::anyhow!("`{revision}` isn't a revision"))?;

    let source = choose_source(args.no_hf)?;
    let mut verified = HashMap::new();
    eprintln!("fetching {repo_id}@{revision} via {}", source.name());

    // Every variant's manifest, and the terms: small, and they decide the rest.
    let first: Vec<String> =
        plan::manifest_patterns("").into_iter().chain(plan::manifest_patterns("*/")).chain([TERMS.to_string()]).collect();
    let snapshot = source
        .fetch(&repo_id, &revision, &first, &[], &mut verified)
        .map_err(|e| anyhow::anyhow!("{e}\n{}", gated_hint(&repo_id, &args.model)))?;
    let found = snapshot.files(&first, &[])?;
    let manifests: Vec<&source::SnapFile> = found.iter().filter(|f| f.path != TERMS).collect();
    let preferred = published.filter(|(_, p)| p.repo == repo_id).map(|(b, _)| b.variant.as_str());
    let variant = choose_variant(&manifests, asked_variant, preferred, &repo_id, &revision)?;
    let root = variant.as_ref().map(|v| format!("{v}/")).unwrap_or_default();
    eprintln!("variant {}", variant.as_deref().unwrap_or("(the repository's root)"));

    let accepted_terms = match found.iter().find(|f| f.path == TERMS) {
        None => None,
        Some(terms) => {
            let digest = check_file(terms, &mut verified)?;
            if args.accept_terms.as_deref() != Some(args.model.as_str()) {
                let text = std::fs::read_to_string(&terms.blob).context("reading TERMS.md")?;
                eprintln!("\n{}\n", text.trim_end());
                bail!(
                    "{repo_id} publishes the terms above for `{model}`. To accept them, fetch again with \
                     --accept-terms {model}; fetch records that you did, with the terms' sha256",
                    model = args.model
                );
            }
            Some(digest.sha256)
        }
    };

    let in_root: Vec<&&source::SnapFile> =
        manifests.iter().filter(|f| f.path.strip_prefix(&root).is_some_and(|rest| !rest.contains('/'))).collect();
    let manifest_file = match in_root.as_slice() {
        [one] => **one,
        _ => bail!("{repo_id}@{revision} has both a manifest.toml and a classifier.toml in `{root}`; fetch can't tell which to serve"),
    };
    let kind = if manifest_file.path.ends_with(plan::Kind::Classifier.file()) { plan::Kind::Classifier } else { plan::Kind::Embedder };
    check_file(manifest_file, &mut verified)?;
    let original = std::fs::read_to_string(&manifest_file.blob).context("reading the manifest")?;
    let build = plan::Build::parse(kind, &original)?;
    anyhow::ensure!(
        build.backend == backend || build.backend == "static",
        "{repo_id} holds a {} build, and --backend is {backend}",
        build.backend
    );
    let chosen = build.choose(&args.buckets)?;
    let include = build.includes(&root, &chosen);
    install::check_target(models_dir, &build.id, args.force)?;

    // The rest of the build.
    let snapshot = source
        .fetch(&repo_id, &revision, &include, &[], &mut verified)
        .map_err(|e| anyhow::anyhow!("{e}\n{}", gated_hint(&repo_id, &args.model)))?;
    let files = snapshot.files(&include, &[])?;
    let mut placed = Vec::new();
    for f in &files {
        let digest = check_file(f, &mut verified)?;
        let path = f.path.strip_prefix(&root).unwrap_or(&f.path).to_string();
        placed.push(install::Placed { path, source: f.blob.clone(), sha256: digest.sha256, size: digest.size });
    }
    for name in std::iter::once(kind.file().to_string()).chain(required_files(&build, &chosen)) {
        anyhow::ensure!(
            placed.iter().any(|p| p.path == name || p.path.starts_with(&format!("{name}/"))),
            "{repo_id}@{revision} has no `{root}{name}`"
        );
    }
    let ledger_build = match_ledger(&ledger, variant.as_deref(), &placed, &include, &root)?;
    let served_manifest = build.served_manifest(&original, &chosen)?;
    // The served manifest is written fresh; the published one stays in the cache.
    placed.retain(|p| p.path != kind.file());

    let provenance = install::Provenance {
        repo: repo_id.clone(),
        revision: revision.clone(),
        commit: snapshot.commit.clone(),
        variant: variant.clone(),
        buckets: chosen.clone(),
        downloaded_by: source.name().into(),
        ledger_build: ledger_build.clone(),
        accepted_terms_sha256: accepted_terms,
        files: placed.iter().map(|p| install::ProvenanceFile { path: p.path.clone(), sha256: p.sha256.clone(), size: p.size }).collect(),
    };
    let done = install::install(models_dir, &build.id, &placed, (kind.file(), &served_manifest), &provenance, args.force)?;

    println!(
        "installed {} ({} build{}) at {}",
        build.id,
        build.backend,
        if chosen.is_empty() { String::new() } else { format!(", buckets {chosen:?}") },
        done.dir.display()
    );
    println!(
        "  from {repo_id}@{} ({} files, {} hard-linked to {}{})",
        &snapshot.commit[..12],
        placed.len(),
        source::human_bytes(done.linked_bytes),
        snapshot.repo.dir.display(),
        if done.copied_bytes > 0 { format!(", {} copied", source::human_bytes(done.copied_bytes)) } else { String::new() }
    );
    match &ledger_build {
        Some(id) => println!("  verified against sidekick's checksum ledger: build {id}"),
        None => println!("  verified against the Hub's hashes only: sidekick's checksum ledger has no build of it"),
    }
    match done.skipped {
        Some(reason) => println!("  warning: sidekickd won't serve it as installed: {reason}"),
        None => println!("  sidekickd serves it from its next start"),
    }
    Ok(())
}

/// The variant to fetch, from the directories that hold a manifest (`""`
/// for one at the repository's root): the one asked for, else the one the
/// ledger last published, else the only one.
fn choose_variant(
    manifests: &[&source::SnapFile],
    asked: Option<&str>,
    preferred: Option<&str>,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Option<String>> {
    let mut dirs: Vec<&str> = manifests.iter().map(|f| f.path.rsplit_once('/').map_or("", |(dir, _)| dir)).collect();
    dirs.sort_unstable();
    dirs.dedup();
    let named: Vec<&str> = dirs.iter().copied().filter(|d| !d.is_empty()).collect();
    let pick = |d: &str| Ok((!d.is_empty()).then(|| d.to_string()));
    match asked {
        Some(v) if dirs.contains(&v) => pick(v),
        Some(v) => bail!("{repo_id}@{revision} has no variant `{v}`; its variants are {named:?}"),
        None => match (preferred.filter(|p| dirs.contains(p)), dirs.as_slice()) {
            (Some(p), _) => pick(p),
            (None, [one]) => pick(one),
            (None, []) => bail!("{repo_id}@{revision} has no manifest.toml or classifier.toml, so it isn't a sidekick build"),
            (None, _) => bail!("{repo_id}@{revision} has several variants, {named:?}: choose one with --variant"),
        },
    }
}

/// The downloader: `hf` when it is installed and recent, else sidekickd's
/// own, which needs `curl`.
fn choose_source(no_hf: bool) -> anyhow::Result<source::Source> {
    let why = if no_hf {
        "--no-hf".to_string()
    } else {
        match source::find_hf() {
            Ok(hf) => return Ok(source::Source::Hf(hf)),
            Err(why) => why,
        }
    };
    anyhow::ensure!(hub::curl_available() || hfcache::offline(), "{why}, and `curl` isn't either: install one of them to fetch models");
    let cache = hfcache::cache_dir()?;
    eprintln!("{why}: downloading anonymously into the Hugging Face cache at {}", cache.display());
    Ok(source::Source::Native { hub: hub::Hub::from_env(), cache, offline: hfcache::offline() })
}

/// Verify one file against the Hub's etag for it, hashing each blob once
/// (a weights file shared by several buckets is one blob, D49).
fn check_file(f: &source::SnapFile, verified: &mut HashMap<PathBuf, verify::FileDigest>) -> anyhow::Result<verify::FileDigest> {
    let Some(etag) = &f.etag else {
        bail!("`{}`: the cache records no Hub hash for it, so it can't be verified; fetch again with network access", f.path)
    };
    let needs_sha1 = !verify::etag_is_sha256(etag);
    let digest = match verified.get(&f.blob) {
        Some(d) if !needs_sha1 || d.git_sha1.is_some() => d.clone(),
        _ => {
            let d = verify::hash_file(&f.blob, needs_sha1).with_context(|| format!("reading {}", f.path))?;
            verified.insert(f.blob.clone(), d.clone());
            d
        }
    };
    verify::check_etag(&f.path, etag, &digest)
        .map_err(|e| anyhow::anyhow!("{e}. The cached file {} is corrupt: delete it and fetch again", f.blob.display()))?;
    Ok(digest)
}

/// The ledger build the installed files are, by the ledger's builds of
/// this model, backend and variant. With none recorded, the Hub's hashes
/// are all there is; with some recorded, the files must be one of them.
fn match_ledger(
    ledger: &[verify::LedgerBuild],
    variant: Option<&str>,
    placed: &[install::Placed],
    include: &[String],
    root: &str,
) -> anyhow::Result<Option<String>> {
    let candidates: Vec<&verify::LedgerBuild> = ledger.iter().filter(|b| variant.is_none_or(|v| b.variant == v)).collect();
    let mut problems = Vec::new();
    for build in &candidates {
        match check_ledger(build, placed, include, root) {
            Ok(()) => return Ok(Some(build.id.clone())),
            Err(e) => problems.push(format!("{}: {e}", build.id)),
        }
    }
    match problems.as_slice() {
        [] => Ok(None),
        [one] => bail!("refusing to install: the files don't match sidekick's checksum ledger. {one}"),
        many => bail!(
            "refusing to install: the files match none of the {} builds sidekick's checksum ledger records for this \
             model.\n  {}",
            many.len(),
            many.join("\n  ")
        ),
    }
}

/// Every installed file must be one the build records, with its hash and
/// size, and every file the build records for the buckets fetched must be
/// installed.
fn check_ledger(build: &verify::LedgerBuild, placed: &[install::Placed], include: &[String], root: &str) -> anyhow::Result<()> {
    for p in placed {
        let Some(entry) = build.file(&p.path) else { bail!("it doesn't record `{}`", p.path) };
        if entry.sha256 != p.sha256 || entry.size != p.size {
            bail!(
                "`{}` has sha256 {} ({} bytes), and the ledger records {} ({} bytes)",
                p.path,
                p.sha256,
                p.size,
                entry.sha256,
                entry.size
            );
        }
    }
    let missing: Vec<&str> = build
        .paths()
        .filter(|path| pattern::selected(&format!("{root}{path}"), include, &[]))
        .filter(|path| !placed.iter().any(|p| p.path == *path))
        .collect();
    anyhow::ensure!(missing.is_empty(), "the repository lacks files it records: {missing:?}");
    Ok(())
}

/// The tokenizer and each chosen program, which the build must have.
fn required_files(build: &plan::Build, chosen: &[usize]) -> Vec<String> {
    let mut names = vec![build.tokenizer.clone()];
    if build.buckets.is_empty() {
        names.push(build.artifact.clone());
    } else {
        names.extend(chosen.iter().flat_map(|&b| sidekick_core::artifact_files(&build.artifact, b, build.chunks)));
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_ids_are_checked() {
        for ok in ["hf-internal-testing/tiny-random-bert", "o/n.v1_x"] {
            assert!(check_repo_id(ok).is_ok(), "{ok}");
        }
        for bad in ["n", "o/n/x", "o/", "../n", "o/..", "o/n?x", "o n/x"] {
            assert!(check_repo_id(bad).is_err(), "{bad}");
        }
    }

    fn placed(path: &str, sha: &str, size: u64) -> install::Placed {
        install::Placed { path: path.into(), source: PathBuf::new(), sha256: sha.into(), size }
    }

    fn ledger() -> Vec<verify::LedgerBuild> {
        serde_json::from_str(
            r#"[{"id": "m/coreml/fp16/old", "model": "m", "backend": "coreml", "variant": "fp16", "files": [
                    {"sha256": "w0", "size": 7, "paths": ["model_128.mlmodelc/weights/weight.bin", "model_256.mlmodelc/weights/weight.bin"]},
                    {"sha256": "t", "size": 2, "paths": ["tokenizer.json"]}]},
                {"id": "m/coreml/fp16/new", "model": "m", "backend": "coreml", "variant": "fp16", "files": [
                    {"sha256": "w", "size": 7, "paths": ["model_128.mlmodelc/weights/weight.bin", "model_256.mlmodelc/weights/weight.bin"]},
                    {"sha256": "t", "size": 2, "paths": ["tokenizer.json"]}]}]"#,
        )
        .unwrap()
    }

    #[test]
    fn the_files_must_be_a_build_the_ledger_records() {
        let include = vec!["fp16/tokenizer.json".to_string(), "fp16/model_128.mlmodelc/*".into()];
        let good = [placed("model_128.mlmodelc/weights/weight.bin", "w", 7), placed("tokenizer.json", "t", 2)];
        // A subset of buckets matches the build it came from, whichever entry that is.
        assert_eq!(match_ledger(&ledger(), Some("fp16"), &good, &include, "fp16/").unwrap().as_deref(), Some("m/coreml/fp16/new"));
        // Nothing recorded for the variant: the Hub's hashes alone.
        assert_eq!(match_ledger(&ledger(), Some("int8"), &good, &include, "int8/").unwrap(), None);
        // A changed file is refused, naming each build it was compared with.
        let changed = [placed("model_128.mlmodelc/weights/weight.bin", "x", 7), placed("tokenizer.json", "t", 2)];
        let err = match_ledger(&ledger(), Some("fp16"), &changed, &include, "fp16/").unwrap_err().to_string();
        assert!(err.contains("none of the 2 builds") && err.contains("has sha256 x"), "{err}");
        // So is a file no build records, and a recorded file of a fetched bucket that's missing.
        let one = &ledger()[1..];
        let extra = [good[0].clone(), good[1].clone(), placed("extra.bin", "e", 1)];
        assert!(match_ledger(one, Some("fp16"), &extra, &include, "fp16/").unwrap_err().to_string().contains("doesn't record `extra.bin`"));
        let err = match_ledger(one, Some("fp16"), &good[1..], &include, "fp16/").unwrap_err().to_string();
        assert!(err.contains("lacks files it records") && err.contains("model_128.mlmodelc/weights/weight.bin"), "{err}");
    }

    #[test]
    fn the_variant_is_asked_for_published_or_the_only_one() {
        let f = |path: &str| source::SnapFile { path: path.into(), blob: PathBuf::new(), etag: None };
        let (a, b, root) = (f("fp16/manifest.toml"), f("int8/manifest.toml"), f("manifest.toml"));
        let choose = |m: &[&source::SnapFile], asked, preferred| choose_variant(m, asked, preferred, "o/n", "main");
        assert_eq!(choose(&[&a], None, None).unwrap().as_deref(), Some("fp16"));
        assert_eq!(choose(&[&a, &b], Some("int8"), Some("fp16")).unwrap().as_deref(), Some("int8"));
        assert_eq!(choose(&[&a, &b], None, Some("fp16")).unwrap().as_deref(), Some("fp16"));
        let err = choose(&[&a, &b], None, None).unwrap_err().to_string();
        assert!(err.contains("several variants") && err.contains("--variant"), "{err}");
        let err = choose(&[&a], Some("fp32"), None).unwrap_err().to_string();
        assert!(err.contains("no variant `fp32`") && err.contains("[\"fp16\"]"), "{err}");
        assert!(choose(&[], None, None).unwrap_err().to_string().contains("isn't a sidekick build"));
        // A build at the repository's root (not how sidekick publishes, but a valid install tree).
        assert_eq!(choose(&[&root], None, None).unwrap(), None);
    }
}
