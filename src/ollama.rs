//! Optionally owning the local model server.
//!
//! When the container sets `NEKORA_MANAGE_OLLAMA=1`, this process starts
//! `ollama serve` itself, waits for it to answer, and pulls the embedder and
//! vision model before the heartbeat begins — so a fresh container comes up
//! self-contained. Left unset, she just talks to whatever Ollama is already
//! running, and none of this runs.

use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use ollama_rs::Ollama;
use tokio::process::{Child, Command};

use crate::brain::required_ollama_models;
use crate::config::env_or;

/// A running `ollama serve` we started and are responsible for stopping.
pub struct Managed {
    child: Child,
}

impl Drop for Managed {
    fn drop(&mut self) {
        // Best-effort: the OS reaps it once we exit anyway, but ask it to stop.
        let _ = self.child.start_kill();
    }
}

/// Build an Ollama client for `host`. OLLAMA_HOST carries the port
/// (http://127.0.0.1:11434), but the builder wants the base and port apart, so
/// split on the last colon of the authority.
pub fn client_from_host(host: &str) -> Ollama {
    let (scheme, authority) = host.split_once("://").unwrap_or(("http", host));
    let (name, port) = authority
        .rsplit_once(':')
        .and_then(|(name, port)| port.parse::<u16>().ok().map(|port| (name, port)))
        .unwrap_or((authority, 11434));
    Ollama::builder()
        .host(format!("{scheme}://{name}"))
        .port(port)
        .build()
}

/// Start and prepare a local Ollama when this process owns it; otherwise nothing.
pub async fn start_if_managed(vision_model: &str) -> Result<Option<Managed>> {
    if env_or("NEKORA_MANAGE_OLLAMA", "") != "1" {
        return Ok(None);
    }
    let host = env_or("OLLAMA_HOST", "http://127.0.0.1:11434");
    let mut child = Command::new("ollama")
        .arg("serve")
        // `ollama serve` reads OLLAMA_HOST as a bare host:port, without the scheme.
        .env("OLLAMA_HOST", authority(&host))
        .spawn()?;

    let ollama = client_from_host(&host);
    let timeout: u64 = env_or("NEKORA_OLLAMA_START_TIMEOUT", "120")
        .parse()
        .unwrap_or(120);
    let deadline = Instant::now() + Duration::from_secs(timeout);
    loop {
        if let Some(status) = child.try_wait()? {
            bail!("ollama serve exited early with {status}");
        }
        if ollama.list_local_models().await.is_ok() {
            break;
        }
        if Instant::now() >= deadline {
            bail!("ollama did not become ready within {timeout}s");
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    let installed = ollama.list_local_models().await?;
    for model in required_ollama_models(vision_model) {
        if has_model(&installed, &model) {
            println!("ollama: {model} already installed");
            continue;
        }
        println!("ollama: pulling {model}");
        ollama.pull_model(model, false).await?;
    }
    Ok(Some(Managed { child }))
}

fn authority(host: &str) -> String {
    host.split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(host)
        .trim_end_matches('/')
        .to_string()
}

// Ollama reports a bare tag as `name:latest`, so a required `bge-m3` matches an
// installed `bge-m3:latest`.
fn has_model(installed: &[ollama_rs::models::LocalModel], model: &str) -> bool {
    installed.iter().any(|local| {
        local.name == model || (!model.contains(':') && local.name == format!("{model}:latest"))
    })
}
