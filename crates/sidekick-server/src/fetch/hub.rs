//! The Hugging Face Hub's HTTP API, through the system's `curl`.
//!
//! sidekick has no HTTP client of its own, and adding one would bring a TLS
//! stack (and with rustls, C code to cross-compile) into a daemon that only
//! serves loopback HTTP. `curl` ships with macOS and nearly every Linux, and
//! brings the system's certificates, proxy settings (`HTTPS_PROXY`), retries
//! and resumable downloads. The fallback is anonymous by design: sidekickd
//! never reads a Hugging Face token. A private or gated repo needs `hf`.

use super::hfcache::{Tree, TreeEntry};
use anyhow::{bail, Context};
use serde::Deserialize;
use std::io::IsTerminal;
use std::path::Path;
use std::process::{Command, Stdio};

pub struct Hub {
    endpoint: String,
    user_agent: String,
}

/// An HTTP status the Hub answered with, beyond the request itself failing.
fn explain(status: u32, what: &str) -> String {
    match status {
        401 | 403 => format!(
            "{what}: the Hub refused an anonymous request ({status}): the repository is private or gated. \
             sidekickd's own downloader never uses a Hugging Face token; install the `hf` CLI, and fetch \
             downloads through it with your account"
        ),
        404 => format!("{what}: not found on the Hub (404). Check the repository and revision"),
        429 => format!("{what}: the Hub is rate-limiting anonymous requests (429). Retry later, or install the `hf` CLI and sign in"),
        _ => format!("{what}: the Hub answered HTTP {status}"),
    }
}

impl Hub {
    /// The endpoint from `HF_ENDPOINT`, as huggingface_hub reads it.
    pub fn from_env() -> Self {
        let endpoint = std::env::var("HF_ENDPOINT")
            .ok()
            .filter(|e| !e.is_empty())
            .unwrap_or_else(|| "https://huggingface.co".into());
        Self { endpoint: endpoint.trim_end_matches('/').to_string(), user_agent: format!("sidekickd/{}", env!("CARGO_PKG_VERSION")) }
    }

    /// The commit a revision (branch, tag or commit) names.
    pub fn commit(&self, repo: &str, revision: &str) -> anyhow::Result<String> {
        #[derive(Deserialize)]
        struct Info {
            sha: String,
        }
        let url = format!("{}/api/models/{repo}/revision/{}", self.endpoint, quote(revision, ""));
        let (body, _) = self.get(&url, &format!("{repo}@{revision}"))?;
        let info: Info = serde_json::from_slice(&body).with_context(|| format!("reading the Hub's answer for {repo}@{revision}"))?;
        anyhow::ensure!(super::hfcache::is_commit(&info.sha), "the Hub named an invalid commit `{}` for {repo}@{revision}", info.sha);
        Ok(info.sha)
    }

    /// Every file at a commit, from the Hub's recursive tree listing,
    /// followed across pages.
    pub fn tree(&self, repo: &str, commit: &str) -> anyhow::Result<Tree> {
        #[derive(Deserialize)]
        struct Lfs {
            oid: String,
            size: u64,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Item {
            #[serde(rename = "type")]
            kind: String,
            path: String,
            oid: String,
            size: u64,
            lfs: Option<Lfs>,
            xet_hash: Option<String>,
        }
        let mut tree = Tree::new();
        let mut url = Some(format!("{}/api/models/{repo}/tree/{commit}?recursive=true&expand=false", self.endpoint));
        while let Some(page) = url.take() {
            let (body, headers) = self.get(&page, &format!("listing {repo}@{commit}"))?;
            let items: Vec<Item> = serde_json::from_slice(&body).with_context(|| format!("reading the Hub's listing of {repo}"))?;
            for item in items.into_iter().filter(|i| i.kind == "file") {
                super::hfcache::check_repo_path(&item.path)?;
                let entry = TreeEntry {
                    size: item.size,
                    blob_id: item.oid,
                    lfs_sha256: item.lfs.as_ref().map(|l| l.oid.clone()),
                    lfs_size: item.lfs.map(|l| l.size),
                    xet_hash: item.xet_hash,
                };
                check_etag(entry.etag())?;
                tree.insert(item.path, entry);
            }
            url = next_link(&headers);
        }
        Ok(tree)
    }

    /// Download one file of a commit to `dest`, resuming a partial `dest`.
    pub fn download(&self, repo: &str, commit: &str, path: &str, dest: &Path) -> anyhow::Result<()> {
        let url = format!("{}/{repo}/resolve/{commit}/{}", self.endpoint, quote(path, "/"));
        let progress = std::io::stderr().is_terminal();
        let mut cmd = curl();
        cmd.args(["--location", "--retry", "5", "--retry-delay", "2", "--continue-at", "-"])
            .args(["--user-agent", &self.user_agent, "--write-out", "%{http_code}", "--output"])
            .arg(dest)
            .arg(&url)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        cmd.arg(if progress { "--progress-bar" } else { "--silent" });
        if !progress {
            cmd.arg("--show-error");
        }
        let out = cmd.output().context("running curl")?;
        let status: u32 = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0);
        if !(200..300).contains(&status) {
            // A body curl wrote for an error status isn't the file.
            if status >= 400 {
                let _ = std::fs::remove_file(dest);
            }
            if status == 416 {
                bail!("downloading {path}: the partial download is longer than the file; it was removed, retry");
            }
            bail!("{}", if status == 0 { format!("downloading {path}: curl failed ({})", out.status) } else { explain(status, &format!("downloading {path}")) });
        }
        Ok(())
    }

    /// GET a URL: the body and the last response's headers.
    fn get(&self, url: &str, what: &str) -> anyhow::Result<(Vec<u8>, String)> {
        let headers_file = std::env::temp_dir().join(format!("sidekickd-fetch-headers-{}-{}", std::process::id(), unique()));
        let out = curl()
            .args(["--silent", "--show-error", "--location", "--retry", "3", "--user-agent", &self.user_agent])
            .arg("--dump-header")
            .arg(&headers_file)
            .args(["--write-out", "\n%{http_code}"])
            .arg(url)
            .stderr(Stdio::inherit())
            .output()
            .context("running curl")?;
        let headers = std::fs::read_to_string(&headers_file).unwrap_or_default();
        let _ = std::fs::remove_file(&headers_file);
        anyhow::ensure!(out.status.success(), "{what}: curl failed ({})", out.status);
        let mut body = out.stdout;
        let split = body.iter().rposition(|&b| b == b'\n').unwrap_or(0);
        let status: u32 = String::from_utf8_lossy(&body[split..]).trim().parse().unwrap_or(0);
        body.truncate(split);
        if !(200..300).contains(&status) {
            bail!("{}", explain(status, what));
        }
        // With --location the dump holds every hop's headers: keep the last.
        let last = headers.rsplit("\r\n\r\n").find(|block| !block.trim().is_empty()).unwrap_or("").to_string();
        Ok((body, last))
    }
}

/// A repository's page on the Hub (`HF_ENDPOINT`'s), where a user accepts
/// a gated repository's terms.
pub fn repo_page(repo: &str) -> String {
    format!("{}/{repo}", Hub::from_env().endpoint)
}

fn curl() -> Command {
    Command::new("curl")
}

/// Whether `curl` can be run.
pub fn curl_available() -> bool {
    Command::new("curl").arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

fn unique() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    N.fetch_add(1, Ordering::Relaxed)
}

/// An etag becomes a file name: only hex digits.
fn check_etag(etag: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !etag.is_empty() && etag.len() <= 128 && etag.bytes().all(|b| b.is_ascii_hexdigit()),
        "the Hub listed an invalid file hash `{etag}`"
    );
    Ok(())
}

/// The `rel="next"` URL of a `Link` header, which the Hub paginates
/// listings with.
fn next_link(headers: &str) -> Option<String> {
    headers
        .lines()
        .filter_map(|l| l.split_once(':').filter(|(k, _)| k.trim().eq_ignore_ascii_case("link")).map(|(_, v)| v.to_string()))
        .flat_map(|v| v.split(',').map(str::to_string).collect::<Vec<_>>())
        .find(|part| part.contains("rel=\"next\""))
        .and_then(|part| Some(part[part.find('<')? + 1..part.find('>')?].to_string()))
}

/// Percent-encode as Python's `urllib.parse.quote(s, safe)`: letters,
/// digits, `_.-~` and the `safe` characters stay as they are.
pub fn quote(s: &str, safe: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"_.-~".contains(&b) || safe.as_bytes().contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotes_like_urllib() {
        assert_eq!(quote("refs/pr/1", ""), "refs%2Fpr%2F1");
        assert_eq!(quote("v0.16.0", ""), "v0.16.0");
        assert_eq!(quote("model_128.mlmodelc/weights/weight.bin", "/"), "model_128.mlmodelc/weights/weight.bin");
        assert_eq!(quote("a b/ü", "/"), "a%20b/%C3%BC");
    }

    #[test]
    fn follows_the_next_link() {
        let headers = "HTTP/2 200\r\ncontent-type: application/json\r\nlink: <https://huggingface.co/api/models/o/n/tree/c?recursive=true&cursor=abc>; rel=\"next\"\r\n";
        assert_eq!(next_link(headers).as_deref(), Some("https://huggingface.co/api/models/o/n/tree/c?recursive=true&cursor=abc"));
        assert_eq!(next_link("HTTP/2 200\r\ncontent-type: application/json\r\n"), None);
        let two = "Link: <https://a/prev>; rel=\"prev\", <https://a/next>; rel=\"next\"\r\n";
        assert_eq!(next_link(two).as_deref(), Some("https://a/next"));
    }

    #[test]
    fn etags_must_be_hex() {
        assert!(check_etag("083459756ab4f6e37a2ebbd0c9ab4cebbda7638d2015958dd2a7ac9047588f93").is_ok());
        assert!(check_etag("22ffb3454131a71d4144340befb799c66ad0c670").is_ok());
        for bad in ["", "../x", "abc/def", "zz"] {
            assert!(check_etag(bad).is_err(), "{bad}");
        }
    }
}
