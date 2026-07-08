use clap::Parser;
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sidekick=info,sidekickd=info,sidekick_server=info".into()),
        )
        .init();

    let args = Args::parse();
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
    tracing::info!(
        addr = %config.addr,
        models_dir = %config.models_dir().display(),
        chat_availability = ?availability,
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
