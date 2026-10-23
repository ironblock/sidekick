//! Installing a fetched build into the models directory: a real directory
//! whose files are hard links to the cache's blobs, beside the manifest it
//! is served with and a record of where it came from.
//!
//! Hard links rather than symlinks into the snapshot (D51): Core ML plans a
//! program with any symlinked file in it off the ANE, silently, and the
//! registry refuses an ONNX graph whose external weights resolve outside
//! its directory (D40). A hard link is the cache's own file, so the build
//! takes no more disk, and the cache and the install stay independent:
//! deleting either leaves the other whole. Where the filesystem refuses a
//! link (the cache on another volume), the file is copied, and fetch says
//! so.

use super::source::human_bytes;
use anyhow::{bail, Context};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// The record of a fetch, in the installed directory. sidekickd doesn't
/// serve from it; fetch reads it to know a directory is its own to replace.
pub const PROVENANCE: &str = "fetch.toml";

/// One file to install.
#[derive(Debug, Clone)]
pub struct Placed {
    /// Path inside the model directory.
    pub path: String,
    /// The cache blob holding it.
    pub source: PathBuf,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Serialize)]
pub struct Provenance {
    pub repo: String,
    pub revision: String,
    pub commit: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub buckets: Vec<usize>,
    /// `hf` or `sidekickd`.
    pub downloaded_by: String,
    /// The checksum ledger's build the files matched; none when the ledger
    /// records no build of the model, and the Hub's hashes were all there
    /// was to check against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ledger_build: Option<String>,
    /// The sha256 of the TERMS.md the user accepted with --accept-terms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_terms_sha256: Option<String>,
    pub files: Vec<ProvenanceFile>,
}

#[derive(Debug, Serialize)]
pub struct ProvenanceFile {
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug)]
pub struct Installed {
    pub dir: PathBuf,
    pub linked_bytes: u64,
    pub copied_bytes: u64,
    /// Why the registry would skip the model, if it would.
    pub skipped: Option<String>,
}

/// Install `files` and the served manifest as `<models_dir>/<id>`,
/// replacing a directory a previous fetch installed (or any, with
/// `force`). The new tree is built beside it and moved into place, so a
/// running daemon never scans half of one.
pub fn install(
    models_dir: &Path,
    id: &str,
    files: &[Placed],
    manifest: (&str, &str),
    provenance: &Provenance,
    force: bool,
) -> anyhow::Result<Installed> {
    let target = check_target(models_dir, id, force)?;
    // Staged one level below the models directory's children, where the
    // registry doesn't look.
    let staging_root = models_dir.join(".fetch").join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&staging_root);
    let staging = staging_root.join(id);
    std::fs::create_dir_all(&staging).with_context(|| format!("creating {}", staging.display()))?;
    let result = stage_and_swap(&staging_root, &staging, &target, files, manifest, provenance);
    let _ = std::fs::remove_dir_all(&staging_root);
    let _ = std::fs::remove_dir(models_dir.join(".fetch"));
    result
}

/// Where `id` installs, if fetch may install there: nothing is there, a
/// previous fetch installed it, or `force`.
pub fn check_target(models_dir: &Path, id: &str, force: bool) -> anyhow::Result<PathBuf> {
    anyhow::ensure!(
        !id.is_empty() && !id.starts_with('.') && !id.contains(['/', '\\', '\0']),
        "the manifest's id `{id}` can't name a directory"
    );
    let target = models_dir.join(id);
    if target.symlink_metadata().is_ok() && !force && !target.join(PROVENANCE).is_file() {
        bail!(
            "{} exists and wasn't installed by `sidekickd fetch`; move it aside, or pass --force to replace it",
            target.display()
        );
    }
    Ok(target)
}

fn stage_and_swap(
    staging_root: &Path,
    staging: &Path,
    target: &Path,
    files: &[Placed],
    (manifest_file, manifest_text): (&str, &str),
    provenance: &Provenance,
) -> anyhow::Result<Installed> {
    let (mut linked_bytes, mut copied_bytes) = (0, 0);
    let mut link_error = None;
    for f in files {
        let dest = staging.join(&f.path);
        std::fs::create_dir_all(dest.parent().expect("a file has a parent"))?;
        match std::fs::hard_link(&f.source, &dest) {
            Ok(()) => linked_bytes += f.size,
            Err(e) => {
                link_error.get_or_insert(e);
                std::fs::copy(&f.source, &dest).with_context(|| format!("copying {} into place", f.path))?;
                copied_bytes += f.size;
            }
        }
    }
    if let Some(e) = link_error {
        eprintln!(
            "note: the filesystem refused hard links into the cache ({e}), so {} was copied; with the models \
             directory on the cache's volume, the install takes no extra disk",
            human_bytes(copied_bytes)
        );
    }
    std::fs::write(staging.join(manifest_file), manifest_text)?;
    let record = format!(
        "# Written by `sidekickd fetch`: where this model came from. sidekickd serves the\n\
         # directory by its {manifest_file}; fetch replaces a directory that has this file.\n\n{}",
        toml::to_string(provenance).context("writing fetch.toml")?
    );
    std::fs::write(staging.join(PROVENANCE), record)?;

    // What the daemon will make of it, scanned as it will scan it.
    let reg = sidekick_core::ModelRegistry::scan(staging_root)?;
    let skipped = reg.skipped().first().map(|s| s.reason.clone());

    if target.symlink_metadata().is_ok() {
        let previous = staging_root.join(".previous");
        std::fs::rename(target, &previous).with_context(|| format!("moving {} aside", target.display()))?;
        if let Err(e) = std::fs::rename(staging, target) {
            let _ = std::fs::rename(&previous, target);
            return Err(e).with_context(|| format!("moving the new install to {}", target.display()));
        }
    } else {
        std::fs::rename(staging, target).with_context(|| format!("moving the new install to {}", target.display()))?;
    }
    Ok(Installed { dir: target.to_path_buf(), linked_bytes, copied_bytes, skipped })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    fn provenance() -> Provenance {
        Provenance {
            repo: "o/n".into(),
            revision: "main".into(),
            commit: "a".repeat(40),
            variant: None,
            buckets: vec![128],
            downloaded_by: "sidekickd".into(),
            ledger_build: None,
            accepted_terms_sha256: None,
            files: vec![],
        }
    }

    const MANIFEST: &str = "id = \"m\"\nbackend = \"coreml\"\nartifact = \"model_{seq}.mlmodelc\"\ntokenizer = \"tokenizer.json\"\n\
                            dims = 4\nbuckets = [128]\nmax_seq_len = 128\n";

    #[test]
    fn installs_hard_links_and_replaces_only_its_own() {
        let root = std::env::temp_dir().join(format!("sk-install-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let blobs = root.join("cache/blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("w"), b"weights").unwrap();
        std::fs::write(blobs.join("t"), b"{}").unwrap();
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let files = vec![
            Placed { path: "model_128.mlmodelc/weights/weight.bin".into(), source: blobs.join("w"), sha256: "x".into(), size: 7 },
            Placed { path: "tokenizer.json".into(), source: blobs.join("t"), sha256: "y".into(), size: 2 },
        ];
        let done = install(&models, "m", &files, ("manifest.toml", MANIFEST), &provenance(), false).unwrap();
        assert_eq!((done.linked_bytes, done.copied_bytes), (9, 0));
        let w = models.join("m/model_128.mlmodelc/weights/weight.bin");
        assert!(!w.symlink_metadata().unwrap().file_type().is_symlink(), "a real file, not a link");
        assert_eq!(w.metadata().unwrap().ino(), blobs.join("w").metadata().unwrap().ino(), "the cache's own file");
        assert!(models.join("m/fetch.toml").is_file());
        assert!(!models.join(".fetch").exists(), "staging is cleaned up");
        assert_eq!(done.skipped, None);
        assert!(sidekick_core::ModelRegistry::scan(&models).unwrap().get("m").is_ok());

        // A second fetch replaces its own install.
        install(&models, "m", &files[..1], ("manifest.toml", MANIFEST), &provenance(), false).unwrap();
        assert!(!models.join("m/tokenizer.json").exists(), "the new install, not a merge");
        assert_eq!(std::fs::read(blobs.join("t")).unwrap(), b"{}", "replacing an install leaves the cache whole");

        // A directory fetch didn't install is left alone, unless forced.
        std::fs::create_dir_all(models.join("hand")).unwrap();
        let err = install(&models, "hand", &files, ("manifest.toml", MANIFEST), &provenance(), false).unwrap_err();
        assert!(err.to_string().contains("wasn't installed by `sidekickd fetch`"), "{err}");
        install(&models, "hand", &files, ("manifest.toml", MANIFEST), &provenance(), true).unwrap();
        assert!(models.join("hand/fetch.toml").is_file());

        // The registry's verdict comes back.
        let bad = MANIFEST.replace("max_seq_len = 128", "max_seq_len = 64");
        let done = install(&models, "m", &files, ("manifest.toml", &bad), &provenance(), false).unwrap();
        assert!(done.skipped.unwrap().contains("largest bucket must equal max_seq_len"));

        for id in ["", ".fetch", "a/b", "../x"] {
            assert!(install(&models, id, &files, ("manifest.toml", MANIFEST), &provenance(), false).is_err(), "{id}");
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
}
