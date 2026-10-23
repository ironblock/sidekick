use clap::{CommandFactory, FromArgMatches, Parser};
use sidekick_server::{build_router, build_state, Config};
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "sidekickd",
    about = "OpenAI-compatible server for on-device embeddings, classification and chat (Apple Foundation Models, Core ML, and ONNX Runtime on the CPU)"
)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
    /// Config file (default, if present: ~/Library/Application
    /// Support/sidekick/config.toml on macOS, ~/.config/sidekick/config.toml
    /// on Linux; the startup log names the file read)
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Listen address (overrides config)
    #[arg(long)]
    addr: Option<SocketAddr>,
    /// Models directory (overrides config)
    #[arg(long, global = true)]
    models_dir: Option<PathBuf>,
    /// API key required as `Authorization: Bearer <key>` (overrides config;
    /// also settable via SIDEKICK_API_KEY)
    #[arg(long, env = "SIDEKICK_API_KEY")]
    api_key: Option<String>,
    /// Hard cap in seconds on a single generation call (overrides config;
    /// also settable via SIDEKICK_TIMEOUT_SECS). Deliberately no clap
    /// default: a default value would always override the config file.
    #[arg(long, env = "SIDEKICK_TIMEOUT_SECS")]
    request_timeout_secs: Option<u64>,
    /// Load models served on the ANE even past Core ML's 1 GiB weight
    /// limit, which it would run off the ANE (for experimentation)
    #[arg(long)]
    ignore_ane_weight_cap: bool,
    /// Serve models on cpu_only past 1,024 tokens, accepting Core ML's
    /// slightly bucket-dependent CPU results there (D33)
    #[arg(long)]
    ignore_cpu_seq_cap: bool,
    /// Serve models on the ANE past their ANE length cap (2,048 tokens, or
    /// the manifest's ane_max_seq_len), where long buckets compile slowly
    /// and can fail predictions (D48)
    #[arg(long)]
    ignore_ane_seq_cap: bool,
}

/// With no command, sidekickd serves.
#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Download a published model build from Hugging Face, verify every
    /// file, and install it in the models directory (D51). Uses the `hf`
    /// CLI when it is installed, else downloads into the same Hugging Face
    /// cache itself.
    Fetch(sidekick_server::fetch::FetchArgs),
}

/// `sidekickd --version`: the crate version plus the SDK the Foundation
/// Models shim was built with, since that decides which macOS 27 features
/// the binary has, the ONNX Runtime it expects, and the model backends
/// compiled in. Called once per process, so leaking it for clap's
/// `'static` requirement is fine.
fn version() -> &'static str {
    let mut text = match sidekick_fm::FM_SDK {
        "none" => format!("{} (Foundation Models: stub build)", env!("CARGO_PKG_VERSION")),
        sdk => format!(
            "{} (Foundation Models shim built with macOS SDK {sdk})",
            env!("CARGO_PKG_VERSION")
        ),
    };
    if sidekick_embed::onnx::SUPPORTED {
        // Loaded at startup (D40); /health reports the version that loaded.
        text.push_str(&format!(" (ONNX Runtime {} expected)", sidekick_embed::onnx::PINNED_VERSION));
    }
    text.push_str(&format!(" (backends: {})", sidekick_embed::compiled_backends().join(", ")));
    Box::leak(text.into_boxed_str())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sidekick=info,sidekickd=info,sidekick_server=info".into()),
        )
        .init();

    let args = Args::from_arg_matches(&Args::command().version(version()).get_matches())?;
    let config_file = Config::resolve_path(args.config.as_ref());
    let mut config = Config::load(args.config.as_ref())?;
    if let Some(addr) = args.addr {
        config.addr = addr;
    }
    if let Some(dir) = args.models_dir {
        config.models_dir = Some(dir);
    }
    if let Some(Command::Fetch(fetch)) = &args.command {
        return sidekick_server::fetch::run(fetch, &config.models_dir());
    }
    if args.api_key.is_some() {
        config.api_key = args.api_key;
    }
    if let Some(secs) = args.request_timeout_secs {
        config.request_timeout_secs = secs;
    }
    if args.ignore_ane_weight_cap {
        config.ignore_ane_weight_cap = true;
    }
    if args.ignore_cpu_seq_cap {
        config.ignore_cpu_seq_cap = true;
    }
    if args.ignore_ane_seq_cap {
        config.ignore_ane_seq_cap = true;
    }

    let state = build_state(&config)?;
    let availability = state.chat.availability().await;
    let model = state.chat.model_info().await.unwrap_or_default();
    state.chat_model.set(model.variant_id.clone());
    let config_file = match config_file {
        Some(p) => p.display().to_string(),
        None => match sidekick_server::config::default_config_path() {
            Some(p) => format!("none (defaults; no file at {})", p.display()),
            None => "none (defaults)".into(),
        },
    };
    tracing::info!(
        config = %config_file,
        addr = %config.addr,
        models_dir = %config.models_dir().display(),
        chat_availability = ?availability,
        chat_model = model.variant.as_deref().unwrap_or("unknown"),
        chat_context = state.chat.context_limit().unwrap_or(0),
        fm_sdk = sidekick_fm::FM_SDK,
        "sidekickd starting"
    );

    let listener = tokio::net::TcpListener::bind(config.addr).await?;
    axum::serve(listener, build_router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ane_weight_cap_flag_parses() {
        let args = Args::try_parse_from(["sidekickd", "--ignore-ane-weight-cap"]).unwrap();
        assert!(args.ignore_ane_weight_cap);
        assert!(!Args::try_parse_from(["sidekickd"]).unwrap().ignore_ane_weight_cap);
        assert!(Args::try_parse_from(["sidekickd", "--ignore-cpu-seq-cap"]).unwrap().ignore_cpu_seq_cap);
        assert!(Args::try_parse_from(["sidekickd", "--ignore-ane-seq-cap"]).unwrap().ignore_ane_seq_cap);
        assert!(!Args::try_parse_from(["sidekickd"]).unwrap().ignore_ane_seq_cap);
    }

    #[test]
    fn fetch_is_a_subcommand_and_serving_is_the_default() {
        assert!(Args::try_parse_from(["sidekickd"]).unwrap().command.is_none());
        let args = Args::try_parse_from([
            "sidekickd", "fetch", "bge-small-en-v1.5", "--repo", "o/n", "--buckets", "128,512", "--models-dir", "/m",
        ])
        .unwrap();
        let Some(Command::Fetch(f)) = args.command else { panic!("fetch") };
        assert_eq!((f.model.as_str(), f.repo.as_deref(), f.buckets.as_slice()), ("bge-small-en-v1.5", Some("o/n"), &[128, 512][..]));
        assert_eq!(args.models_dir, Some(PathBuf::from("/m")), "--models-dir is shared with serving");
        let onnx = Args::try_parse_from(["sidekickd", "fetch", "m", "--backend", "onnx"]).unwrap();
        assert!(matches!(onnx.command, Some(Command::Fetch(sidekick_server::fetch::FetchArgs { backend: sidekick_server::fetch::Backend::Onnx, .. }))));
    }
}
