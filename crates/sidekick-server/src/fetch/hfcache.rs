//! The Hugging Face cache, laid out exactly as huggingface_hub lays it out,
//! so `hf`, Python's huggingface_hub and `sidekickd fetch` each see the
//! others' downloads as cached.
//!
//! ```text
//! <cache>/                                   $HF_HUB_CACHE, else $HF_HOME/hub, else
//!   CACHEDIR.TAG                             $XDG_CACHE_HOME/huggingface/hub, else
//!   .locks/models--<org>--<name>/<etag>.lock  ~/.cache/huggingface/hub
//!   models--<org>--<name>/
//!     blobs/<etag>                           each file's bytes, once
//!     refs/<branch or tag>                   the commit it resolved to
//!     snapshots/<commit>/<path>              relative symlink to ../../blobs/<etag>
//!     trees/<commit>.json                    the commit's file listing
//! ```
//!
//! A blob is named by its etag: the sha256 of a Git LFS (or Xet) file, the
//! Git blob id (sha1) of any other. `trees/<commit>.json` is
//! huggingface_hub's cached listing of a commit (format 1), which it reads
//! to tell a complete snapshot from a partial one when offline.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

/// The cache root, from the environment as huggingface_hub reads it.
pub fn cache_dir() -> anyhow::Result<PathBuf> {
    cache_dir_from(|k| std::env::var(k).ok().filter(|v| !v.is_empty()), dirs::home_dir())
}

fn cache_dir_from(env: impl Fn(&str) -> Option<String>, home: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    let home_str = home.as_ref().map(|h| h.to_string_lossy().into_owned());
    let expand = |v: String| expand_path(&v, &env, home_str.as_deref());
    if let Some(v) = env("HF_HUB_CACHE").or_else(|| env("HUGGINGFACE_HUB_CACHE")) {
        return Ok(PathBuf::from(expand(v)));
    }
    if let Some(v) = env("HF_HOME") {
        return Ok(PathBuf::from(expand(v)).join("hub"));
    }
    if let Some(v) = env("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(expand(v)).join("huggingface").join("hub"));
    }
    let home = home.ok_or_else(|| anyhow::anyhow!("no home directory: set HF_HUB_CACHE or HF_HOME"))?;
    Ok(home.join(".cache").join("huggingface").join("hub"))
}

/// Python's `os.path.expandvars(os.path.expanduser(v))`, for the forms an
/// environment variable takes: a leading `~`, `$NAME` and `${NAME}`. An
/// unset variable is left as written, as Python leaves it.
fn expand_path(v: &str, env: &impl Fn(&str) -> Option<String>, home: Option<&str>) -> String {
    let v = match (v.strip_prefix('~'), home) {
        (Some(rest), Some(home)) if rest.is_empty() || rest.starts_with('/') => format!("{home}{rest}"),
        _ => v.to_string(),
    };
    let mut out = String::new();
    let mut rest = v.as_str();
    while let Some(i) = rest.find('$') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let (name, len) = if let Some(braced) = after.strip_prefix('{') {
            match braced.find('}') {
                Some(end) => (&braced[..end], end + 2),
                None => ("", 0),
            }
        } else {
            let end = after.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(after.len());
            (&after[..end], end)
        };
        match (name.is_empty(), env(name)) {
            (false, Some(value)) => out.push_str(&value),
            _ => out.push_str(&rest[i..i + 1 + len]),
        }
        rest = &rest[i + 1 + len..];
    }
    out.push_str(rest);
    out
}

/// Whether the Hub must not be contacted: huggingface_hub's
/// `HF_HUB_OFFLINE` (or `TRANSFORMERS_OFFLINE`).
pub fn offline() -> bool {
    let truthy = |k: &str| std::env::var(k).is_ok_and(|v| ["1", "ON", "YES", "TRUE"].contains(&v.to_uppercase().as_str()));
    truthy("HF_HUB_OFFLINE") || truthy("TRANSFORMERS_OFFLINE")
}

/// A full commit id: 40 lowercase hex digits.
pub fn is_commit(rev: &str) -> bool {
    rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A repo path that stays inside the snapshot: relative, `/`-separated,
/// with no empty, `.` or `..` part. Every path the Hub lists is checked
/// before it becomes a file name.
pub fn check_repo_path(path: &str) -> anyhow::Result<()> {
    let ok = !path.is_empty()
        && !path.contains('\\')
        && !path.contains('\0')
        && path.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
        && Path::new(path).components().all(|c| matches!(c, Component::Normal(_)));
    anyhow::ensure!(ok, "the repository lists an unsafe path `{path}`");
    Ok(())
}

/// One file of a commit's listing (huggingface_hub's `TreeCacheEntry`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeEntry {
    pub size: u64,
    /// The Git blob id (sha1 of `blob <size>\0<bytes>`; of the LFS pointer
    /// for an LFS file).
    pub blob_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lfs_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lfs_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub xet_hash: Option<String>,
}

impl TreeEntry {
    /// What the Hub sends as the file's etag, and so its blob's name.
    pub fn etag(&self) -> &str {
        self.lfs_sha256.as_deref().unwrap_or(&self.blob_id)
    }
}

/// A commit's files, by repo path.
pub type Tree = BTreeMap<String, TreeEntry>;

#[derive(Serialize, Deserialize)]
struct TreeFile {
    format_version: u32,
    files: Tree,
}

const TREE_FORMAT: u32 = 1;

const CACHEDIR_TAG: &str = "Signature: 8a477f597d28d172789f06886806bc55\n\
# This file is a cache directory tag created by huggingface_hub.\n\
# For information about cache directory tags, see:\n\
#\thttps://bford.info/cachedir/\n";

/// One model repo's folder in the cache.
#[derive(Debug, Clone)]
pub struct RepoCache {
    /// The cache root.
    pub root: PathBuf,
    /// `<root>/models--<org>--<name>`.
    pub dir: PathBuf,
    folder: String,
}

impl RepoCache {
    pub fn new(root: &Path, repo_id: &str) -> Self {
        let folder = repo_folder_name(repo_id);
        Self { root: root.to_path_buf(), dir: root.join(&folder), folder }
    }

    /// The repo folder a snapshot directory belongs to
    /// (`<dir>/snapshots/<commit>`).
    pub fn of_snapshot(snapshot: &Path) -> Option<(Self, String)> {
        let commit = snapshot.file_name()?.to_str()?.to_string();
        let dir = snapshot.parent()?.parent()?;
        let folder = dir.file_name()?.to_str()?.to_string();
        let root = dir.parent()?.to_path_buf();
        Some((Self { root, dir: dir.to_path_buf(), folder }, commit))
    }

    pub fn blob(&self, etag: &str) -> PathBuf {
        self.dir.join("blobs").join(etag)
    }

    pub fn snapshot(&self, commit: &str) -> PathBuf {
        self.dir.join("snapshots").join(commit)
    }

    pub fn pointer(&self, commit: &str, path: &str) -> PathBuf {
        self.snapshot(commit).join(path)
    }

    /// The commit a branch or tag last resolved to.
    pub fn read_ref(&self, rev: &str) -> Option<String> {
        let commit = std::fs::read_to_string(self.dir.join("refs").join(rev)).ok()?;
        Some(commit.trim().to_string()).filter(|c| is_commit(c))
    }

    /// Record what `rev` resolved to, unless it is itself the commit; like
    /// huggingface_hub, write only on a change, through a temporary file.
    pub fn write_ref(&self, rev: &str, commit: &str) -> std::io::Result<()> {
        if rev == commit || self.read_ref(rev).as_deref() == Some(commit) {
            return Ok(());
        }
        write_atomic(&self.dir.join("refs").join(rev), commit.as_bytes())
    }

    fn tree_path(&self, commit: &str) -> PathBuf {
        self.dir.join("trees").join(format!("{commit}.json"))
    }

    /// The cached listing of a commit, if one is cached in a format this
    /// reads.
    pub fn read_tree(&self, commit: &str) -> Option<Tree> {
        let file: TreeFile = serde_json::from_slice(&std::fs::read(self.tree_path(commit)).ok()?).ok()?;
        (file.format_version == TREE_FORMAT).then_some(file.files)
    }

    /// Cache a commit's listing, as huggingface_hub writes it (a commit's
    /// listing never changes, so an existing file is left alone).
    pub fn write_tree(&self, commit: &str, tree: &Tree) -> std::io::Result<()> {
        let path = self.tree_path(commit);
        if path.exists() {
            return Ok(());
        }
        let file = TreeFile { format_version: TREE_FORMAT, files: tree.clone() };
        let mut out = Vec::new();
        let mut ser = serde_json::Serializer::with_formatter(&mut out, serde_json::ser::PrettyFormatter::with_indent(b" "));
        file.serialize(&mut ser).map_err(std::io::Error::other)?;
        write_atomic(&path, &out)
    }

    /// Hold huggingface_hub's per-blob download lock
    /// (`.locks/<folder>/<etag>.lock`) until the returned file is dropped.
    pub fn lock(&self, etag: &str) -> std::io::Result<std::fs::File> {
        let dir = self.root.join(".locks").join(&self.folder);
        std::fs::create_dir_all(&dir)?;
        let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(dir.join(format!("{etag}.lock")))?;
        file.lock()?;
        Ok(file)
    }

    /// Point `snapshots/<commit>/<path>` at `blobs/<etag>` with a relative
    /// symlink, unless something is there already.
    pub fn link_pointer(&self, commit: &str, path: &str, etag: &str) -> std::io::Result<()> {
        let pointer = self.pointer(commit, path);
        if pointer.symlink_metadata().is_ok() {
            return Ok(());
        }
        std::fs::create_dir_all(pointer.parent().expect("a pointer has a parent"))?;
        std::os::unix::fs::symlink(relative_blob_target(path, etag), &pointer)
    }

    /// Tag the cache root for backup tools, as huggingface_hub does.
    pub fn tag_cache_dir(&self) {
        let tag = self.root.join("CACHEDIR.TAG");
        if !tag.exists() {
            let _ = std::fs::write(tag, CACHEDIR_TAG);
        }
    }
}

/// `models--<org>--<name>`.
pub fn repo_folder_name(repo_id: &str) -> String {
    std::iter::once("models").chain(repo_id.split('/')).collect::<Vec<_>>().join("--")
}

/// The symlink target from `snapshots/<commit>/<path>` to `blobs/<etag>`:
/// one `..` per directory of `path`, and two more for `snapshots/<commit>`.
pub fn relative_blob_target(path: &str, etag: &str) -> PathBuf {
    let depth = path.split('/').count() - 1 + 2;
    let mut target = PathBuf::new();
    for _ in 0..depth {
        target.push("..");
    }
    target.join("blobs").join(etag)
}

/// Write a file through a temporary file in the same directory and a
/// rename, so a reader never sees half of it.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().expect("a file has a parent");
    std::fs::create_dir_all(dir)?;
    let name = path.file_name().expect("a file has a name").to_string_lossy();
    let tmp = dir.join(format!("{name}.{}.tmp", std::process::id()));
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
    }

    #[test]
    fn the_cache_root_follows_huggingface_hub() {
        let home = Some(PathBuf::from("/home/u"));
        let at = |vars: &[(&str, &str)]| cache_dir_from(env(vars), home.clone()).unwrap();
        assert_eq!(at(&[]), PathBuf::from("/home/u/.cache/huggingface/hub"));
        assert_eq!(at(&[("XDG_CACHE_HOME", "/x")]), PathBuf::from("/x/huggingface/hub"));
        assert_eq!(at(&[("HF_HOME", "/h"), ("XDG_CACHE_HOME", "/x")]), PathBuf::from("/h/hub"));
        assert_eq!(at(&[("HUGGINGFACE_HUB_CACHE", "/old"), ("HF_HOME", "/h")]), PathBuf::from("/old"));
        assert_eq!(at(&[("HF_HUB_CACHE", "/c"), ("HUGGINGFACE_HUB_CACHE", "/old")]), PathBuf::from("/c"));
        // expanduser and expandvars, as Python applies them.
        assert_eq!(at(&[("HF_HOME", "~/hf")]), PathBuf::from("/home/u/hf/hub"));
        assert_eq!(at(&[("HF_HUB_CACHE", "$VOL/${SUB}/hub"), ("VOL", "/v"), ("SUB", "s")]), PathBuf::from("/v/s/hub"));
        assert_eq!(at(&[("HF_HUB_CACHE", "/a/$UNSET/b")]), PathBuf::from("/a/$UNSET/b"));
        assert!(cache_dir_from(env(&[]), None).is_err());
    }

    #[test]
    fn repo_folders_and_pointers_match_huggingface_hub() {
        assert_eq!(repo_folder_name("julien-c/EsperBERTo-small"), "models--julien-c--EsperBERTo-small");
        // As `hf download` links them (checked against a real cache).
        assert_eq!(relative_blob_target("config.json", "e1"), PathBuf::from("../../blobs/e1"));
        assert_eq!(relative_blob_target("onnx/model.onnx", "e2"), PathBuf::from("../../../blobs/e2"));
        assert_eq!(
            relative_blob_target("model_128.mlmodelc/weights/weight.bin", "e3"),
            PathBuf::from("../../../../blobs/e3")
        );
        let repo = RepoCache::new(Path::new("/c"), "org/name");
        assert_eq!(repo.dir, PathBuf::from("/c/models--org--name"));
        assert_eq!(repo.pointer("abc", "a/b"), PathBuf::from("/c/models--org--name/snapshots/abc/a/b"));
        let (back, commit) = RepoCache::of_snapshot(Path::new("/c/models--org--name/snapshots/abc")).unwrap();
        assert_eq!((back.dir, back.root, commit.as_str()), (repo.dir, repo.root, "abc"));
    }

    #[test]
    fn unsafe_repo_paths_are_refused() {
        for ok in ["a", "a/b.bin", "model_128.mlmodelc/weights/weight.bin", ".gitattributes"] {
            assert!(check_repo_path(ok).is_ok(), "{ok}");
        }
        for bad in ["", "/etc/passwd", "../x", "a/../../x", "a//b", "./a", "a/.", "a\\b", "a/"] {
            assert!(check_repo_path(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn refs_trees_and_pointers_round_trip() {
        let root = std::env::temp_dir().join(format!("sk-hfcache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let repo = RepoCache::new(&root, "org/name");
        let commit = "f171d7baecaf37b5da5a3616d8833b9969753535";
        repo.write_ref("main", commit).unwrap();
        assert_eq!(std::fs::read_to_string(repo.dir.join("refs/main")).unwrap(), commit, "no newline, as hf writes it");
        assert_eq!(repo.read_ref("main").as_deref(), Some(commit));
        repo.write_ref(commit, commit).unwrap();
        assert!(!repo.dir.join("refs").join(commit).exists(), "a commit gets no ref");
        repo.write_ref("refs/pr/1", commit).unwrap();
        assert_eq!(repo.read_ref("refs/pr/1").as_deref(), Some(commit), "a nested ref, as hf stores it");

        let mut tree = Tree::new();
        tree.insert("config.json".into(), TreeEntry { size: 3, blob_id: "b1".into(), lfs_sha256: None, lfs_size: None, xet_hash: None });
        tree.insert(
            "onnx/model.onnx".into(),
            TreeEntry { size: 9, blob_id: "b2".into(), lfs_sha256: Some("s2".into()), lfs_size: Some(9), xet_hash: Some("x2".into()) },
        );
        repo.write_tree(commit, &tree).unwrap();
        assert_eq!(repo.read_tree(commit).unwrap(), tree);
        let json = std::fs::read_to_string(repo.dir.join(format!("trees/{commit}.json"))).unwrap();
        assert!(json.starts_with("{\n \"format_version\": 1,\n \"files\": {"), "{json}");
        assert!(!json.contains("\"lfs_sha256\": null"), "absent fields are left out, as hf leaves them: {json}");
        assert_eq!(tree["onnx/model.onnx"].etag(), "s2");
        assert_eq!(tree["config.json"].etag(), "b1");

        std::fs::create_dir_all(repo.dir.join("blobs")).unwrap();
        std::fs::write(repo.blob("s2"), b"onnx bytes").unwrap();
        repo.link_pointer(commit, "onnx/model.onnx", "s2").unwrap();
        let pointer = repo.pointer(commit, "onnx/model.onnx");
        assert_eq!(std::fs::read_link(&pointer).unwrap(), PathBuf::from("../../../blobs/s2"));
        assert_eq!(std::fs::read(&pointer).unwrap(), b"onnx bytes");
        let _lock = repo.lock("s2").unwrap();
        assert!(root.join(".locks/models--org--name/s2.lock").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
