//! Verifying what was downloaded, twice over:
//!
//! - **against the Hub's own metadata**: a Git LFS (or Xet) file's etag is
//!   the sha256 of its bytes, any other file's is its Git blob id, the sha1
//!   of `blob <size>\0<bytes>`. That catches a corrupted or truncated
//!   download, whichever tool made it.
//! - **against sidekick's checksum ledger** (#136, D49): the sha256 and
//!   size of every file of every build sidekick published, recorded in
//!   this repository rather than on the Hub. That catches a repository
//!   whose content isn't what sidekick published.

use serde::Deserialize;
use sha1::Sha1;
use sha2::{Digest as _, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

/// A file's hashes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileDigest {
    pub size: u64,
    pub sha256: String,
    /// The Git blob id, when it was asked for.
    pub git_sha1: Option<String>,
}

/// Hash a file in one pass: sha256 always, the Git blob id when
/// `git_sha1` (it costs a second hash, and only non-LFS files, which are
/// small, need it).
pub fn hash_file(path: &Path, git_sha1: bool) -> std::io::Result<FileDigest> {
    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let mut sha256 = Sha256::new();
    let mut sha1 = git_sha1.then(|| {
        let mut h = Sha1::new();
        h.update(format!("blob {size}\0").as_bytes());
        h
    });
    let mut buf = vec![0u8; 1 << 20];
    let mut read = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        read += n as u64;
        sha256.update(&buf[..n]);
        if let Some(h) = sha1.as_mut() {
            h.update(&buf[..n]);
        }
    }
    if read != size {
        return Err(std::io::Error::other(format!("{} changed size while it was read", path.display())));
    }
    Ok(FileDigest { size, sha256: hex(&sha256.finalize()), git_sha1: sha1.map(|h| hex(&h.finalize())) })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Whether an etag is a sha256 (an LFS or Xet file) rather than a Git blob
/// id (sha1).
pub fn etag_is_sha256(etag: &str) -> bool {
    etag.len() == 64
}

/// Check a file's digest against the etag the Hub gave it.
pub fn check_etag(path: &str, etag: &str, digest: &FileDigest) -> Result<(), String> {
    let (kind, got) = if etag_is_sha256(etag) {
        ("sha256", Some(&digest.sha256))
    } else {
        ("Git blob id", digest.git_sha1.as_ref())
    };
    match got {
        Some(got) if got.eq_ignore_ascii_case(etag) => Ok(()),
        Some(got) => Err(format!("`{path}`: its {kind} is {got}, and the Hub lists {etag}")),
        None => Err(format!("`{path}`: its Git blob id wasn't computed")),
    }
}

/// One build in sidekick's checksum ledger (docs/data/ledger.json,
/// docs/PUBLISHING.md): each distinct file once, with every installed path
/// that holds it (D49), and each upload. Other fields are left unread.
#[derive(Debug, Clone, Deserialize)]
pub struct LedgerBuild {
    /// `<model>/<backend>/<variant>/<digest[:12]>`.
    pub id: String,
    pub model: String,
    pub backend: String,
    /// The build's directory in its repository (`fp16`, `fp16-max2048`).
    pub variant: String,
    pub files: Vec<LedgerFile>,
    #[serde(default)]
    pub published: Vec<Published>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LedgerFile {
    pub sha256: String,
    pub size: u64,
    /// Paths relative to the installed model directory.
    pub paths: Vec<String>,
}

/// One upload of a build.
#[derive(Debug, Clone, Deserialize)]
pub struct Published {
    pub repo: String,
    pub revision: String,
    #[serde(default)]
    pub date: Option<String>,
}

impl LedgerBuild {
    /// The recorded hash and size of an installed path.
    pub fn file(&self, path: &str) -> Option<&LedgerFile> {
        self.files.iter().find(|f| f.paths.iter().any(|p| p == path))
    }

    /// Every recorded installed path.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.files.iter().flat_map(|f| f.paths.iter().map(String::as_str))
    }
}

#[derive(Deserialize)]
struct LedgerFormat {
    builds: Vec<LedgerBuild>,
}

/// The ledger built into sidekickd (build.rs): the repository's
/// docs/data/ledger.json when it was built, else empty.
const BUILT_IN: &str = include_str!(concat!(env!("OUT_DIR"), "/ledger.json"));

/// Where fetch reads the ledger: the one built into sidekickd, or a file
/// named on the command line (a newer ledger, or a test's).
pub enum LedgerSource {
    File(PathBuf),
    BuiltIn,
}

/// The ledger's builds of one model for one backend.
pub fn builds(source: &LedgerSource, model: &str, backend: &str) -> anyhow::Result<Vec<LedgerBuild>> {
    let text = match source {
        LedgerSource::BuiltIn => BUILT_IN.to_string(),
        LedgerSource::File(path) => {
            std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading the ledger {}: {e}", path.display()))?
        }
    };
    let ledger: LedgerFormat = serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("parsing the ledger: {e}"))?;
    Ok(ledger.builds.into_iter().filter(|b| b.model == model && b.backend == backend).collect())
}

/// The most recent upload among `builds` (of `variant`, when given): the
/// last recorded, by date.
pub fn latest_published<'a>(builds: &'a [LedgerBuild], variant: Option<&str>) -> Option<(&'a LedgerBuild, &'a Published)> {
    builds
        .iter()
        .filter(|b| variant.is_none_or(|v| b.variant == v))
        .flat_map(|b| b.published.iter().map(move |p| (b, p)))
        .enumerate()
        .max_by(|(i, (_, a)), (j, (_, b))| a.date.cmp(&b.date).then(i.cmp(j)))
        .map(|(_, found)| found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_match_sha256sum_and_git_hash_object() {
        let dir = std::env::temp_dir().join(format!("sk-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f");
        std::fs::write(&path, b"hello\n").unwrap();
        let d = hash_file(&path, true).unwrap();
        assert_eq!(d.size, 6);
        // `shasum -a 256` and `git hash-object` of "hello\n".
        assert_eq!(d.sha256, "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03");
        assert_eq!(d.git_sha1.as_deref(), Some("ce013625030ba8dba906f756967f9e9ca394464a"));
        assert_eq!(hash_file(&path, false).unwrap().git_sha1, None);

        assert!(check_etag("f", "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03", &d).is_ok());
        assert!(check_etag("f", "ce013625030ba8dba906f756967f9e9ca394464a", &d).is_ok());
        let wrong = check_etag("f", &"0".repeat(64), &d).unwrap_err();
        assert!(wrong.contains("sha256") && wrong.contains("the Hub lists"), "{wrong}");
        assert!(check_etag("f", &"0".repeat(40), &d).unwrap_err().contains("Git blob id"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_ledger_is_read_by_model_and_backend() {
        let dir = std::env::temp_dir().join(format!("sk-ledger-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ledger.json");
        // The shape docs/data/ledger.json has (docs/PUBLISHING.md), fields fetch doesn't read included.
        std::fs::write(
            &path,
            r#"{"schema": 1, "builds": [
                {"id": "m/coreml/fp16/aaa", "model": "m", "backend": "coreml", "variant": "fp16", "layout": "shared",
                 "buckets": [128, 256], "grades": [],
                 "files": [{"sha256": "aa", "size": 1, "paths": ["model_128.mlmodelc/weights/weight.bin", "model_256.mlmodelc/weights/weight.bin"]}],
                 "published": [{"repo": "o/m-coreml", "revision": "r1", "date": "2026-11-01", "sidekick": "0.16.0"}]},
                {"id": "m/coreml/int8/bbb", "model": "m", "backend": "coreml", "variant": "int8", "files": [],
                 "published": [{"repo": "o/m-coreml", "revision": "r2", "date": "2026-12-01", "sidekick": "0.17.0"}]},
                {"id": "m/onnx/fp32/ccc", "model": "m", "backend": "onnx", "variant": "fp32", "files": [], "published": []}
            ]}"#,
        )
        .unwrap();
        let src = LedgerSource::File(path);
        let b = builds(&src, "m", "coreml").unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].file("model_256.mlmodelc/weights/weight.bin").unwrap().sha256, "aa", "one entry, two paths (D49)");
        assert_eq!(b[0].paths().count(), 2);
        let (build, p) = latest_published(&b, None).unwrap();
        assert_eq!((build.variant.as_str(), p.revision.as_str()), ("int8", "r2"), "the latest upload");
        assert_eq!(latest_published(&b, Some("fp16")).unwrap().1.revision, "r1");
        assert!(latest_published(&builds(&src, "m", "onnx").unwrap(), None).is_none(), "never uploaded");
        // The built-in ledger parses, whatever it holds.
        builds(&LedgerSource::BuiltIn, "m", "coreml").unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
