//! Getting a repo's files into the Hugging Face cache: through the user's
//! own `hf` CLI when it is installed, else by sidekickd itself, into the
//! same cache with the same layout.

use super::hfcache::{self, check_repo_path, RepoCache, Tree, TreeEntry};
use super::hub::Hub;
use super::pattern::selected;
use super::verify::{self, FileDigest};
use anyhow::{bail, Context};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The oldest `hf` this drives: 1.0 made `--include` a repeatable option.
const MIN_HF: (u32, u32) = (1, 0);

/// How files reach the cache.
pub enum Source {
    /// The user's `hf` CLI, with its own configuration and token.
    Hf(PathBuf),
    /// sidekickd's own anonymous downloader.
    /// `offline` is huggingface_hub's `HF_HUB_OFFLINE`: the cache only.
    Native { hub: Hub, cache: PathBuf, offline: bool },
}

impl Source {
    pub fn name(&self) -> &'static str {
        match self {
            Source::Hf(_) => "hf",
            Source::Native { .. } => "sidekickd",
        }
    }
}

/// A commit's snapshot in the cache.
pub struct Snapshot {
    pub repo: RepoCache,
    pub commit: String,
    /// The commit's listing, when cached (huggingface_hub 1.x and
    /// sidekickd write one).
    pub tree: Option<Tree>,
}

/// One file of a snapshot.
#[derive(Debug, Clone)]
pub struct SnapFile {
    /// The repo path.
    pub path: String,
    /// The file's bytes: the blob the snapshot's link resolves to.
    pub blob: PathBuf,
    /// The Hub's etag for it: from the cached listing, else the name of the
    /// blob the link points to. `None` only for a snapshot with neither.
    pub etag: Option<String>,
}

impl Snapshot {
    /// The snapshot's files selected by the patterns. With a listing, every
    /// selected file must be in the snapshot; without one, the snapshot's
    /// own files are all there is to go on.
    pub fn files(&self, include: &[String], exclude: &[String]) -> anyhow::Result<Vec<SnapFile>> {
        let dir = self.repo.snapshot(&self.commit);
        let paths: Vec<String> = match &self.tree {
            Some(tree) => tree.keys().filter(|p| selected(p, include, exclude)).cloned().collect(),
            None => {
                let mut found = Vec::new();
                if dir.is_dir() {
                    walk(&dir, &dir, &mut found)?;
                }
                found.retain(|p| selected(p, include, exclude));
                found.sort();
                found
            }
        };
        let mut files = Vec::new();
        let mut missing = Vec::new();
        for path in paths {
            check_repo_path(&path)?;
            let pointer = dir.join(&path);
            let Ok(blob) = std::fs::canonicalize(&pointer) else {
                missing.push(path);
                continue;
            };
            let etag = self.tree.as_ref().and_then(|t| t.get(&path)).map(|e| e.etag().to_string()).or_else(|| {
                let target = std::fs::read_link(&pointer).ok()?;
                let in_blobs = target.parent().and_then(|p| p.file_name()).is_some_and(|n| n == "blobs");
                in_blobs.then(|| target.file_name()?.to_str().map(str::to_string)).flatten()
            });
            files.push(SnapFile { path, blob, etag });
        }
        if !missing.is_empty() {
            bail!(
                "the cached snapshot of {} is missing {} file(s) ({}{}); fetch again with network access",
                self.commit,
                missing.len(),
                missing.iter().take(3).cloned().collect::<Vec<_>>().join(", "),
                if missing.len() > 3 { ", …" } else { "" }
            );
        }
        Ok(files)
    }
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        // Snapshot files are links into blobs/; directories are real.
        if std::fs::symlink_metadata(&path)?.is_dir() {
            walk(root, &path, out)?;
        } else if let Ok(rel) = path.strip_prefix(root) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

/// `hf` on PATH, if it is new enough to drive; otherwise why not.
pub fn find_hf() -> Result<PathBuf, String> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let hf = std::env::split_paths(&path)
        .map(|d| d.join("hf"))
        .find(|p| p.is_file())
        .ok_or_else(|| "the `hf` CLI isn't installed".to_string())?;
    let out = Command::new(&hf).arg("--version").stderr(Stdio::null()).output().map_err(|e| format!("`hf --version` failed: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    match parse_version(&text) {
        Some(v) if v >= MIN_HF => Ok(hf),
        Some((major, minor)) => Err(format!("`hf` {major}.{minor} is older than {}.{} (`hf update` updates it)", MIN_HF.0, MIN_HF.1)),
        None => Err("`hf --version` printed no version".into()),
    }
}

/// The first `major.minor` in `hf --version`'s output.
fn parse_version(text: &str) -> Option<(u32, u32)> {
    text.split(|c: char| !(c.is_ascii_digit() || c == '.')).find_map(|word| {
        let mut parts = word.split('.');
        Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
    })
}

impl Source {
    /// Bring the selected files of `repo_id@revision` into the cache.
    /// Downloads sidekickd makes itself are verified before they are
    /// placed, and their digests recorded in `verified`.
    pub fn fetch(
        &self,
        repo_id: &str,
        revision: &str,
        include: &[String],
        exclude: &[String],
        verified: &mut HashMap<PathBuf, FileDigest>,
    ) -> anyhow::Result<Snapshot> {
        match self {
            Source::Hf(hf) => fetch_hf(hf, repo_id, revision, include, exclude),
            Source::Native { hub, cache, offline } => fetch_native(hub, cache, *offline, repo_id, revision, include, exclude, verified),
        }
    }
}

fn fetch_hf(hf: &Path, repo_id: &str, revision: &str, include: &[String], exclude: &[String]) -> anyhow::Result<Snapshot> {
    let mut cmd = Command::new(hf);
    cmd.args(["download", repo_id, "--revision", revision]);
    for p in include {
        cmd.args(["--include", p]);
    }
    for p in exclude {
        cmd.args(["--exclude", p]);
    }
    // hf's progress goes to the terminal; its result (the snapshot path) to us.
    let out = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::inherit()).output().context("running hf")?;
    if !out.status.success() {
        bail!("`hf download {repo_id} --revision {revision}` failed ({}); its message is above", out.status);
    }
    let path = snapshot_path(&String::from_utf8_lossy(&out.stdout))
        .ok_or_else(|| anyhow::anyhow!("`hf download` printed no snapshot path"))?;
    let (repo, commit) = RepoCache::of_snapshot(&path)
        .filter(|(_, c)| hfcache::is_commit(c))
        .ok_or_else(|| anyhow::anyhow!("`hf download` returned {}, which isn't a cache snapshot", path.display()))?;
    let tree = repo.read_tree(&commit);
    Ok(Snapshot { repo, commit, tree })
}

/// The snapshot path in `hf download`'s output, in any of its formats: the
/// bare path (`--quiet`, and hf before 1.20), `path=<path>` (its agent
/// format), or `{"path": "<path>"}` (`--json`).
fn snapshot_path(stdout: &str) -> Option<PathBuf> {
    let line = stdout.lines().map(str::trim).rfind(|l| !l.is_empty())?;
    if line.starts_with('{') {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        return v.get("path")?.as_str().map(PathBuf::from);
    }
    Some(PathBuf::from(line.strip_prefix("path=").unwrap_or(line)))
}

#[allow(clippy::too_many_arguments)]
fn fetch_native(
    hub: &Hub,
    cache: &Path,
    offline: bool,
    repo_id: &str,
    revision: &str,
    include: &[String],
    exclude: &[String],
    verified: &mut HashMap<PathBuf, FileDigest>,
) -> anyhow::Result<Snapshot> {
    let repo = RepoCache::new(cache, repo_id);
    if offline {
        let commit = if hfcache::is_commit(revision) { Some(revision.to_string()) } else { repo.read_ref(revision) };
        let commit = commit.filter(|c| repo.snapshot(c).is_dir()).ok_or_else(|| {
            anyhow::anyhow!("HF_HUB_OFFLINE is set, and {repo_id}@{revision} isn't in the cache ({})", cache.display())
        })?;
        let tree = repo.read_tree(&commit);
        return Ok(Snapshot { repo, commit, tree });
    }
    let commit = hub.commit(repo_id, revision)?;
    let tree = match repo.read_tree(&commit) {
        Some(tree) => tree,
        None => {
            let tree = hub.tree(repo_id, &commit)?;
            repo.write_tree(&commit, &tree).with_context(|| format!("writing to the cache {}", cache.display()))?;
            tree
        }
    };
    repo.write_ref(revision, &commit)?;
    repo.tag_cache_dir();
    let wanted: Vec<(&String, &TreeEntry)> = tree.iter().filter(|(p, _)| selected(p, include, exclude)).collect();
    // Each distinct blob once: a file at several paths (D49) is one download.
    let mut seen = HashSet::new();
    let missing: Vec<(&String, &TreeEntry)> =
        wanted.iter().copied().filter(|(_, e)| !repo.blob(e.etag()).exists() && seen.insert(e.etag())).collect();
    if !missing.is_empty() {
        let bytes: u64 = missing.iter().map(|(_, e)| e.size).sum();
        eprintln!("downloading {} file(s), {}, into {}", missing.len(), human_bytes(bytes), repo.dir.display());
    }
    for (path, entry) in missing {
        check_repo_path(path)?;
        download_blob(hub, &repo, repo_id, &commit, path, entry, verified)?;
    }
    for (path, entry) in &wanted {
        check_repo_path(path)?;
        repo.link_pointer(&commit, path, entry.etag())?;
    }
    Ok(Snapshot { repo, commit, tree: Some(tree) })
}

/// Download one blob under huggingface_hub's lock for it, to
/// `blobs/<etag>.incomplete` (resumed if a previous run left one), and
/// verify it against its etag and size before it is renamed into place.
fn download_blob(
    hub: &Hub,
    repo: &RepoCache,
    repo_id: &str,
    commit: &str,
    path: &str,
    entry: &TreeEntry,
    verified: &mut HashMap<PathBuf, FileDigest>,
) -> anyhow::Result<()> {
    let etag = entry.etag();
    let blob = repo.blob(etag);
    std::fs::create_dir_all(blob.parent().expect("blobs/"))?;
    let _lock = repo.lock(etag)?;
    if blob.exists() {
        // Another download finished it while this one waited for the lock.
        return Ok(());
    }
    eprintln!("  {path} ({})", human_bytes(entry.size));
    let incomplete = blob.with_file_name(format!("{etag}.incomplete"));
    let have = std::fs::metadata(&incomplete).map(|m| m.len()).unwrap_or(0);
    if have > entry.size {
        std::fs::remove_file(&incomplete)?;
    }
    if have != entry.size {
        hub.download(repo_id, commit, path, &incomplete)?;
    }
    let digest = verify::hash_file(&incomplete, !verify::etag_is_sha256(etag))?;
    let problem = if digest.size != entry.size {
        Some(format!("`{path}`: downloaded {} bytes, and the Hub lists {}", digest.size, entry.size))
    } else {
        verify::check_etag(path, etag, &digest).err()
    };
    if let Some(problem) = problem {
        let _ = std::fs::remove_file(&incomplete);
        bail!("{problem}; the download was discarded");
    }
    std::fs::rename(&incomplete, &blob)?;
    verified.insert(blob, digest);
    Ok(())
}

pub fn human_bytes(n: u64) -> String {
    match n {
        n if n >= 1 << 30 => format!("{:.2} GiB", n as f64 / (1u64 << 30) as f64),
        n if n >= 1 << 20 => format!("{:.1} MiB", n as f64 / (1u64 << 20) as f64),
        n if n >= 1 << 10 => format!("{:.1} KiB", n as f64 / 1024.0),
        n => format!("{n} B"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_hf_versions_and_snapshot_paths() {
        assert_eq!(parse_version("1.27.0\n"), Some((1, 27)));
        assert_eq!(parse_version("huggingface_hub version: 0.34.4"), Some((0, 34)));
        assert_eq!(parse_version("version=1.0.1"), Some((1, 0)));
        assert_eq!(parse_version("no digits"), None);
        let p = "/c/models--o--n/snapshots/f171d7baecaf37b5da5a3616d8833b9969753535";
        for out in [format!("{p}\n"), format!("Hint: x\npath={p}\n"), format!("{{\"path\": \"{p}\"}}\n\n")] {
            assert_eq!(snapshot_path(&out), Some(PathBuf::from(p)), "{out}");
        }
        assert_eq!(snapshot_path("\n"), None);
    }

    /// Every file under a cache root, by relative path: a link's target, or
    /// a file's inode, so a second download that replaced anything shows.
    fn cache_state(root: &Path) -> std::collections::BTreeMap<String, String> {
        use std::os::unix::fs::MetadataExt;
        fn walk(root: &Path, dir: &Path, out: &mut std::collections::BTreeMap<String, String>) {
            for e in std::fs::read_dir(dir).unwrap() {
                let p = e.unwrap().path();
                let rel = p.strip_prefix(root).unwrap().to_string_lossy().into_owned();
                let meta = std::fs::symlink_metadata(&p).unwrap();
                if meta.is_dir() {
                    walk(root, &p, out);
                } else if meta.file_type().is_symlink() {
                    out.insert(rel, format!("-> {}", std::fs::read_link(&p).unwrap().display()));
                } else if !rel.starts_with(".locks") && rel != "CACHEDIR.TAG" {
                    out.insert(rel, format!("inode {}", meta.ino()));
                }
            }
        }
        let mut out = Default::default();
        walk(root, root, &mut out);
        out
    }

    /// Runs `hf download` against a cache, offline or not.
    fn hf_download(hf: &Path, cache: &Path, offline: bool, include: &[String]) -> std::process::Output {
        let mut cmd = Command::new(hf);
        cmd.args(["download", INTEROP_REPO, "--revision", "main"]).env("HF_HUB_CACHE", cache).env("HF_HUB_DISABLE_TELEMETRY", "1");
        cmd.env("HF_HUB_OFFLINE", if offline { "1" } else { "0" });
        for p in include {
            cmd.args(["--include", p]);
        }
        cmd.output().unwrap()
    }

    const INTEROP_REPO: &str = "hf-internal-testing/tiny-random-bert";

    /// sidekickd's downloader and `hf` each find the other's cache complete.
    /// Downloads about 1 MB from the Hub, so it needs the network and the
    /// `hf` CLI (1.0 or later), and is ignored by default:
    /// `cargo test -p sidekick-server --lib -- --ignored hf_and_sidekickd`.
    #[test]
    #[ignore]
    fn hf_and_sidekickd_share_one_cache() {
        let hf = find_hf().expect("this test needs the hf CLI");
        let root = std::env::temp_dir().join(format!("sk-interop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // A Git LFS (Xet) file in a subdirectory, and plain Git files.
        let include: Vec<String> = ["onnx/*", "config.json", "model.safetensors", "*.json"].map(String::from).to_vec();
        let native = |cache: &Path, offline: bool| Source::Native { hub: Hub::from_env(), cache: cache.to_path_buf(), offline };
        let check = |snap: &Snapshot| {
            let files = snap.files(&include, &[]).unwrap();
            assert!(files.iter().any(|f| f.path == "onnx/model.onnx") && files.len() >= 4, "{files:?}");
            for f in &files {
                let etag = f.etag.as_deref().unwrap();
                let digest = verify::hash_file(&f.blob, !verify::etag_is_sha256(etag)).unwrap();
                verify::check_etag(&f.path, etag, &digest).unwrap();
            }
            files.len()
        };

        // sidekickd first, then hf: hf finds it complete, offline and online,
        // and downloads nothing.
        let ours = root.join("sidekickd-first");
        let snap = native(&ours, false).fetch(INTEROP_REPO, "main", &include, &[], &mut HashMap::new()).unwrap();
        let n = check(&snap);
        let before = cache_state(&ours);
        let out = hf_download(&hf, &ours, true, &include);
        assert!(out.status.success(), "hf offline: {}", String::from_utf8_lossy(&out.stderr));
        let said = snapshot_path(&String::from_utf8_lossy(&out.stdout)).unwrap();
        assert_eq!(std::fs::canonicalize(said).unwrap(), std::fs::canonicalize(snap.repo.snapshot(&snap.commit)).unwrap());
        let out = hf_download(&hf, &ours, false, &include);
        assert!(out.status.success(), "hf online: {}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(cache_state(&ours), before, "hf changed nothing in sidekickd's cache");

        // hf first, then sidekickd: the same, offline and online.
        let theirs = root.join("hf-first");
        let out = hf_download(&hf, &theirs, false, &include);
        assert!(out.status.success(), "hf: {}", String::from_utf8_lossy(&out.stderr));
        let before = cache_state(&theirs);
        let snap = native(&theirs, true).fetch(INTEROP_REPO, "main", &include, &[], &mut HashMap::new()).unwrap();
        assert_eq!(check(&snap), n);
        native(&theirs, false).fetch(INTEROP_REPO, "main", &include, &[], &mut HashMap::new()).unwrap();
        assert_eq!(cache_state(&theirs), before, "sidekickd changed nothing in hf's cache");

        // And the two caches are the same, file for file.
        assert_eq!(cache_state(&ours).keys().collect::<Vec<_>>(), cache_state(&theirs).keys().collect::<Vec<_>>());
        let links = |s: std::collections::BTreeMap<String, String>| s.into_iter().filter(|(_, v)| v.starts_with("->")).collect::<Vec<_>>();
        assert_eq!(links(cache_state(&ours)), links(cache_state(&theirs)));
        let repo = |c: &Path| RepoCache::new(c, INTEROP_REPO);
        assert_eq!(repo(&ours).read_ref("main"), repo(&theirs).read_ref("main"));
        assert_eq!(repo(&ours).read_tree(&snap.commit), repo(&theirs).read_tree(&snap.commit));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A cache-shaped snapshot without a listing: files are found by
    /// walking it, and each one's etag is the blob its link names.
    #[test]
    fn a_snapshot_without_a_listing_is_read_from_its_links() {
        let root = std::env::temp_dir().join(format!("sk-source-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let repo = RepoCache::new(&root, "o/n");
        let commit = "a".repeat(40);
        std::fs::create_dir_all(repo.dir.join("blobs")).unwrap();
        for (path, etag) in [("manifest.toml", "11"), ("model_128.mlmodelc/weights/weight.bin", "22"), ("model_256.mlmodelc/weights/weight.bin", "22")] {
            std::fs::write(repo.blob(etag), etag).unwrap();
            repo.link_pointer(&commit, path, etag).unwrap();
        }
        let snap = Snapshot { repo: repo.clone(), commit: commit.clone(), tree: None };
        let files = snap.files(&["model_*/*".into()], &[]).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| f.etag.as_deref() == Some("22") && f.blob == std::fs::canonicalize(repo.blob("22")).unwrap()));
        // With a listing that names a file the snapshot lacks: incomplete.
        let mut tree = Tree::new();
        for p in ["manifest.toml", "tokenizer.json"] {
            tree.insert(p.into(), TreeEntry { size: 2, blob_id: "11".into(), lfs_sha256: None, lfs_size: None, xet_hash: None });
        }
        let snap = Snapshot { repo, commit, tree: Some(tree) };
        let err = snap.files(&["*.json".into(), "*.toml".into()], &[]).unwrap_err().to_string();
        assert!(err.contains("missing 1 file(s) (tokenizer.json)"), "{err}");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
