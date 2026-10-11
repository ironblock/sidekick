//! Where the suite finds its fixtures (the corpora, the expectations): a
//! binary copied to another machine has no checkout at the path it was
//! built from.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The environment variable naming the fixtures directory.
pub const ENV: &str = "SIDEKICK_FIXTURES";

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// The fixtures directory: the first of `--fixtures`, `$SIDEKICK_FIXTURES`,
/// `./fixtures` and the checkout the suite was built from that holds
/// `parity/corpus.toml`. An error names every path tried.
pub fn resolve(flag: Option<&Path>, env: Option<PathBuf>, cwd: &Path, built: &Path) -> Result<PathBuf, String> {
    let candidates: Vec<PathBuf> = match flag {
        // Named outright: that or nothing.
        Some(dir) => vec![dir.to_path_buf()],
        None => env.into_iter().chain([cwd.join("fixtures"), built.to_path_buf()]).collect(),
    };
    candidates.iter().find(|d| d.join("parity/corpus.toml").is_file()).cloned().ok_or_else(|| {
        let tried: Vec<String> = candidates.iter().map(|d| d.display().to_string()).collect();
        format!(
            "no fixtures directory (with parity/corpus.toml) at {}; pass --fixtures DIR or set {ENV}",
            tried.join(", ")
        )
    })
}

/// The checkout's fixtures, where this binary was built.
fn built() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// Settle the fixtures directory for this process from `--fixtures`, and
/// export it, so worker processes find the same one.
pub fn init(flag: Option<&Path>) -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let dir = resolve(flag, std::env::var_os(ENV).map(PathBuf::from), &cwd, &built())?;
    std::env::set_var(ENV, &dir);
    Ok(DIR.get_or_init(|| dir).clone())
}

/// The fixtures directory: as [`init`] settled it, else by the same lookup
/// without a flag; the checkout's path if nothing else holds one (where
/// the error then surfaces, naming the file).
pub fn dir() -> PathBuf {
    DIR.get_or_init(|| {
        let cwd = std::env::current_dir().unwrap_or_default();
        resolve(None, std::env::var_os(ENV).map(PathBuf::from), &cwd, &built()).unwrap_or_else(|_| built())
    })
    .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures_in(root: &Path, name: &str) -> PathBuf {
        let d = root.join(name);
        std::fs::create_dir_all(d.join("parity")).unwrap();
        std::fs::write(d.join("parity/corpus.toml"), "").unwrap();
        d
    }

    #[test]
    fn the_flag_then_the_env_then_the_cwd_then_the_checkout() {
        let root = std::env::temp_dir().join(format!("sk-fixtures-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (flag, env, built) = (fixtures_in(&root, "flag"), fixtures_in(&root, "env"), fixtures_in(&root, "built"));
        let cwd = root.join("cwd");
        let cwd_fixtures = fixtures_in(&cwd, "fixtures");
        let missing = root.join("missing");

        assert_eq!(resolve(Some(&flag), Some(env.clone()), &cwd, &built).unwrap(), flag);
        assert_eq!(resolve(None, Some(env.clone()), &cwd, &built).unwrap(), env);
        assert_eq!(resolve(None, None, &cwd, &built).unwrap(), cwd_fixtures);
        assert_eq!(resolve(None, Some(missing.clone()), &missing, &built).unwrap(), built, "the checkout last");
        // A flag that names no fixtures is an error, not a fallback.
        let e = resolve(Some(&missing), Some(env), &cwd, &built).unwrap_err();
        assert!(e.contains(&missing.display().to_string()) && e.contains("--fixtures"), "{e}");
        // Nothing anywhere: every path tried is named.
        let e = resolve(None, Some(missing.clone()), &missing, &root.join("nowhere")).unwrap_err();
        assert!(e.contains("missing, ") && e.contains("nowhere") && e.contains(ENV), "{e}");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
