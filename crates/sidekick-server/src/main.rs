use clap::{CommandFactory, FromArgMatches, Parser};
use sidekick_server::{build_router, build_state, Config};
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "sidekickd",
    about = "OpenAI-compatible server over Apple on-device inference (Foundation Models + ANE encoders)"
)]
struct Args {
    /// Config file (default: ~/.config/sidekick/config.toml if present)
    #[arg(long)]
    config: Option<PathBuf>,
    /// Listen address (overrides config)
    #[arg(long)]
    addr: Option<SocketAddr>,
    /// Models directory (overrides config)
    #[arg(long)]
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
}

/// `sidekickd --version`: the crate version plus the SDK the Foundation
/// Models shim was built with, since that decides which macOS 27 features
/// the binary has. Called once per process, so leaking it for clap's
/// `'static` requirement is fine.
fn version() -> &'static str {
    let text = match sidekick_fm::FM_SDK {
        "none" => format!("{} (Foundation Models: stub build)", env!("CARGO_PKG_VERSION")),
        sdk => format!(
            "{} (Foundation Models shim built with macOS SDK {sdk})",
            env!("CARGO_PKG_VERSION")
        ),
    };
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
    let mut config = Config::load(args.config.as_ref())?;
    if let Some(addr) = args.addr {
        config.addr = addr;
    }
    if let Some(dir) = args.models_dir {
        config.models_dir = Some(dir);
    }
    if args.api_key.is_some() {
        config.api_key = args.api_key;
    }
    if let Some(secs) = args.request_timeout_secs {
        config.request_timeout_secs = secs;
    }

    let state = build_state(&config)?;
    let availability = state.chat.availability().await;
    let model = state.chat.model_info().await.unwrap_or_default();
    tracing::info!(
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
