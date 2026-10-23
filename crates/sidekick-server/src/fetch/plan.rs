//! What to fetch for a model, and the manifest it is served with.
//!
//! A published build's repository is its install tree (D49): the manifest
//! (`manifest.toml` for an embedder, `classifier.toml` for a classifier),
//! the tokenizer, and the artifact, one program per bucket for Core ML
//! (`model_128.mlmodelc/`, …), each chunk its own (D37). A variant (a
//! quantization) is a subdirectory holding its own install tree. Fetch
//! reads the manifest first, then maps the buckets asked for to
//! `--include` patterns, so `hf` and the fallback fetch the same files.

use sidekick_core::manifest::{CLASSIFIER_MANIFEST, EMBEDDER_MANIFEST};
use sidekick_core::{artifact_files, ClassifierManifest, ModelManifest};

/// Which manifest a build has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Embedder,
    Classifier,
}

impl Kind {
    pub fn file(self) -> &'static str {
        match self {
            Kind::Embedder => EMBEDDER_MANIFEST,
            Kind::Classifier => CLASSIFIER_MANIFEST,
        }
    }
}

/// The parts of a build's manifest that decide its files.
#[derive(Debug, Clone)]
pub struct Build {
    pub kind: Kind,
    pub id: String,
    /// `coreml`, `onnx` or `static`; a classifier is Core ML.
    pub backend: String,
    pub artifact: String,
    pub tokenizer: String,
    /// Empty for a model without buckets (ONNX, static).
    pub buckets: Vec<usize>,
    pub chunks: usize,
    /// The manifest as written, to rewrite for a subset.
    table: toml::Table,
}

/// The manifest patterns for a variant's root (`""` or `"<variant>/"`).
pub fn manifest_patterns(root: &str) -> Vec<String> {
    [EMBEDDER_MANIFEST, CLASSIFIER_MANIFEST].iter().map(|m| format!("{root}{m}")).collect()
}

impl Build {
    pub fn parse(kind: Kind, text: &str) -> anyhow::Result<Self> {
        let table: toml::Table = toml::from_str(text).map_err(|e| anyhow::anyhow!("parsing {}: {e}", kind.file()))?;
        let build = match kind {
            Kind::Embedder => {
                let m: ModelManifest = toml::from_str(text).map_err(|e| anyhow::anyhow!("parsing {}: {e}", kind.file()))?;
                let chunks = m.chunks();
                Self { kind, id: m.id, backend: m.backend.name().into(), artifact: m.artifact, tokenizer: m.tokenizer, buckets: m.buckets, chunks, table }
            }
            Kind::Classifier => {
                let m: ClassifierManifest = toml::from_str(text).map_err(|e| anyhow::anyhow!("parsing {}: {e}", kind.file()))?;
                let chunks = m.chunks();
                Self { kind, id: m.id, backend: "coreml".into(), artifact: m.artifact, tokenizer: m.tokenizer, buckets: m.buckets, chunks, table }
            }
        };
        for path in [&build.artifact, &build.tokenizer] {
            super::hfcache::check_repo_path(&path.replace("{seq}", "0").replace("{chunk}", "0"))
                .map_err(|_| anyhow::anyhow!("the manifest names an unsafe path `{path}`"))?;
        }
        Ok(build)
    }

    /// The buckets to fetch: those asked for, every one of which the build
    /// must have, or all of them.
    pub fn choose(&self, asked: &[usize]) -> anyhow::Result<Vec<usize>> {
        if asked.is_empty() {
            return Ok(self.buckets.clone());
        }
        anyhow::ensure!(!self.buckets.is_empty(), "`{}` has no sequence-length buckets to choose from ({} build)", self.id, self.backend);
        let mut chosen: Vec<usize> = asked.to_vec();
        chosen.sort_unstable();
        chosen.dedup();
        let missing: Vec<usize> = chosen.iter().copied().filter(|b| !self.buckets.contains(b)).collect();
        anyhow::ensure!(missing.is_empty(), "`{}` has no bucket {missing:?}; its buckets are {:?}", self.id, self.buckets);
        Ok(chosen)
    }

    /// The `--include` patterns for the chosen buckets, under `root`: the
    /// manifest, the tokenizer, and each chosen program (a directory, so
    /// `name/*`, or a file). An ONNX graph's external weights sit beside
    /// it, named by the exporter's convention (`model.onnx_data`, or
    /// `model.onnx.data`); the registry checks at scan that the graph
    /// finds them.
    pub fn includes(&self, root: &str, chosen: &[usize]) -> Vec<String> {
        let mut patterns = vec![format!("{root}{}", self.kind.file()), format!("{root}{}", self.tokenizer)];
        let names: Vec<String> = if self.buckets.is_empty() {
            vec![self.artifact.clone()]
        } else {
            chosen.iter().flat_map(|&b| artifact_files(&self.artifact, b, self.chunks)).collect()
        };
        for name in names {
            patterns.push(format!("{root}{name}"));
            patterns.push(format!("{root}{name}/*"));
            if self.backend == "onnx" {
                patterns.push(format!("{root}{name}_data"));
                patterns.push(format!("{root}{name}.data"));
            }
        }
        patterns
    }

    /// The manifest to serve the chosen buckets with: the build's own, with
    /// `buckets`, `max_seq_len` and any recorded compute plans limited to
    /// what was fetched. The whole build is served by its manifest as it
    /// is, byte for byte.
    pub fn served_manifest(&self, original: &str, chosen: &[usize]) -> anyhow::Result<String> {
        if chosen == self.buckets.as_slice() {
            return Ok(original.to_string());
        }
        let mut table = self.table.clone();
        let largest = *chosen.last().expect("a subset has a bucket");
        table.insert("buckets".into(), toml::Value::Array(chosen.iter().map(|&b| toml::Value::Integer(b as i64)).collect()));
        table.insert("max_seq_len".into(), toml::Value::Integer(largest as i64));
        let keep = |plans: &mut toml::Value| {
            if let Some(t) = plans.as_table_mut() {
                t.retain(|k, _| k.parse::<usize>().is_ok_and(|b| chosen.contains(&b)));
            }
        };
        if let Some(placement) = table.get_mut("placement").and_then(|p| p.as_table_mut()) {
            if let Some(b) = placement.get_mut("buckets") {
                keep(b);
            }
            if let Some(alts) = placement.get_mut("alternatives").and_then(|a| a.as_table_mut()) {
                for (_, alt) in alts.iter_mut() {
                    if let Some(b) = alt.as_table_mut().and_then(|a| a.get_mut("buckets")) {
                        keep(b);
                    }
                }
            }
        }
        // fev's state budget is the window less its question's share; the
        // registry shrinks it the same way when a length cap shortens the
        // window (manifest.rs, fev_state_max_len).
        if let Some(fev) = table.get_mut("classify").and_then(|c| c.get_mut("fev")).and_then(|f| f.as_table_mut()) {
            let budget = largest.saturating_sub(640) as i64;
            if fev.get("state_max_len").and_then(|v| v.as_integer()).is_some_and(|s| s > budget) {
                fev.insert("state_max_len".into(), toml::Value::Integer(budget));
            }
        }
        let body = toml::to_string(&table).map_err(|e| anyhow::anyhow!("writing the manifest: {e}"))?;
        Ok(format!(
            "# Written by `sidekickd fetch`: the published {file}, limited to the buckets\n\
             # fetched ({chosen:?} of {all:?}).\n\n{body}",
            file = self.kind.file(),
            all = self.buckets,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::pattern::selected;

    const BGE: &str = r#"# a comment
id = "bge-small-en-v1.5"
backend = "coreml"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
dims = 384
pooling = "none"
buckets = [128, 256, 512]
max_seq_len = 512

[io]
input_ids = "input_ids"
attention_mask = "attention_mask"
output = "embedding"

[placement]
compute_units = "cpu_and_ne"
chip = "Apple M1 Max"
macos_build = "27A1"

[placement.buckets.128]
ane = 1
gpu = 0
cpu = 0
unassigned = 0
total = 1

[placement.buckets.512]
ane = 1
gpu = 0
cpu = 0
unassigned = 0
total = 1
"#;

    /// The files of a three-bucket, two-chunk Core ML build with an int8
    /// variant, as a repo lists them.
    fn repo_files() -> Vec<String> {
        let mut files = vec![".gitattributes".to_string(), "README.md".into(), "manifest.toml".into(), "tokenizer.json".into()];
        for v in ["", "int8/"] {
            for b in [128, 256, 1024] {
                for c in 0..2 {
                    for f in ["model.mil", "coremldata.bin", "metadata.json", "weights/weight.bin", "analytics/coremldata.bin"] {
                        files.push(format!("{v}model_{b}.{c}.mlmodelc/{f}"));
                    }
                }
            }
        }
        files.extend(["int8/manifest.toml".into(), "int8/tokenizer.json".into()]);
        files
    }

    fn chunked() -> Build {
        let text = BGE.replace("model_{seq}.mlmodelc", "model_{seq}.{chunk}.mlmodelc").replace("[128, 256, 512]", "[128, 256, 1024]")
            .replace("max_seq_len = 512", "max_seq_len = 1024")
            + "\n[chunking]\nchunks = 2\nweight_budget_bytes = 966367641\n";
        Build::parse(Kind::Embedder, &text).unwrap()
    }

    #[test]
    fn buckets_map_to_include_patterns() {
        let b = chunked();
        let inc = b.includes("", &b.choose(&[256, 128]).unwrap());
        let got: Vec<String> = repo_files().into_iter().filter(|f| selected(f, &inc, &[])).collect();
        let mut want = vec!["manifest.toml".to_string(), "tokenizer.json".into()];
        for bucket in [128, 256] {
            for c in 0..2 {
                for f in ["model.mil", "coremldata.bin", "metadata.json", "weights/weight.bin", "analytics/coremldata.bin"] {
                    want.push(format!("model_{bucket}.{c}.mlmodelc/{f}"));
                }
            }
        }
        let (mut got, mut want) = (got, want);
        got.sort();
        want.sort();
        assert_eq!(got, want, "both chunks of each chosen bucket, nothing of 1024, the variant, or the card");

        // A variant's patterns stay inside its directory.
        let inc = b.includes("int8/", &[1024]);
        let got: Vec<String> = repo_files().into_iter().filter(|f| selected(f, &inc, &[])).collect();
        assert_eq!(got.len(), 2 + 2 * 5, "{got:?}");
        assert!(got.iter().all(|f| f.starts_with("int8/")), "{got:?}");
    }

    #[test]
    fn an_onnx_build_fetches_its_graph_and_external_weights() {
        let onnx = "id = \"q\"\nbackend = \"onnx\"\nartifact = \"model.onnx\"\ntokenizer = \"tokenizer.json\"\ndims = 4\n\
                    pooling = \"last_token\"\nmax_seq_len = 64\n";
        let b = Build::parse(Kind::Embedder, onnx).unwrap();
        assert!(b.choose(&[128]).unwrap_err().to_string().contains("no sequence-length buckets"));
        let inc = b.includes("", &b.choose(&[]).unwrap());
        for (file, want) in [
            ("model.onnx", true),
            ("model.onnx_data", true),
            ("model.onnx.data", true),
            ("tokenizer.json", true),
            ("manifest.toml", true),
            ("fp16w/model.onnx", false),
            ("README.md", false),
        ] {
            assert_eq!(selected(file, &inc, &[]), want, "{file}");
        }
    }

    #[test]
    fn unknown_buckets_and_unsafe_paths_are_refused() {
        let b = Build::parse(Kind::Embedder, BGE).unwrap();
        let err = b.choose(&[128, 384]).unwrap_err().to_string();
        assert!(err.contains("no bucket [384]") && err.contains("[128, 256, 512]"), "{err}");
        assert_eq!(b.choose(&[]).unwrap(), vec![128, 256, 512]);
        let bad = BGE.replace("tokenizer = \"tokenizer.json\"", "tokenizer = \"../../secret\"");
        assert!(Build::parse(Kind::Embedder, &bad).unwrap_err().to_string().contains("unsafe path"));
    }

    #[test]
    fn a_subset_is_served_with_its_own_manifest() {
        let b = Build::parse(Kind::Embedder, BGE).unwrap();
        // All buckets: the published manifest, unchanged.
        assert_eq!(b.served_manifest(BGE, &[128, 256, 512]).unwrap(), BGE);
        let text = b.served_manifest(BGE, &[128, 256]).unwrap();
        assert!(text.starts_with("# Written by `sidekickd fetch`"), "{text}");
        let m: ModelManifest = toml::from_str(&text).unwrap();
        assert_eq!((m.buckets.clone(), m.max_seq_len), (vec![128, 256], 256));
        assert_eq!((m.dims, m.io.output.as_str()), (384, "embedding"), "everything else carries over");
        let p = m.placement.unwrap();
        assert_eq!(p.buckets.keys().collect::<Vec<_>>(), ["128"], "plans for buckets not fetched are dropped");

        // The registry serves it: the subset manifest validates.
        let dir = std::env::temp_dir().join(format!("sk-plan-{}", std::process::id()));
        let model = dir.join("bge-small-en-v1.5");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("manifest.toml"), &text).unwrap();
        let reg = sidekick_core::ModelRegistry::scan(&dir).unwrap();
        assert!(reg.skipped().is_empty(), "{:?}", reg.skipped());
        assert_eq!(reg.get("bge-small-en-v1.5").unwrap().manifest.buckets, vec![128, 256]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
