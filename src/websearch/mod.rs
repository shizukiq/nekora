mod ollama;
mod openrouter;

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use reqwest::{Client, Response, StatusCode, Url};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::config;

const DEFAULT_PROVIDER_CHAIN: &str = "ollama,openrouter";
const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_COOLDOWN_SECS: u64 = 5 * 60;
const MAX_TIMEOUT_SECS: u64 = 5 * 60;
const MAX_COOLDOWN_SECS: u64 = 60 * 60;
const MAX_REDIRECTS: usize = 10;

pub(crate) const MAX_QUERY_CHARS: usize = 500;
pub(crate) const MAX_RESULTS: usize = 10;
pub(crate) const MAX_RESPONSE_BYTES: usize = 1_000_000;
pub(crate) const MAX_TITLE_CHARS: usize = 300;
pub(crate) const MAX_URL_CHARS: usize = 4096;
pub(crate) const MAX_SNIPPET_CHARS: usize = 2000;

pub(crate) type ProviderResult<T> = std::result::Result<T, SearchError>;
pub(crate) type SearchFuture<'a> =
    Pin<Box<dyn Future<Output = ProviderResult<SearchResults>> + Send + 'a>>;

pub(crate) trait SearchProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn search<'a>(&'a self, query: &'a str, limit: usize) -> SearchFuture<'a>;
}

#[derive(Debug)]
pub(crate) enum SearchError {
    Configuration(String),
    RateLimited {
        message: String,
        retry_after: Option<Duration>,
    },
    Temporary(String),
    Permanent(String),
}

impl SearchError {
    fn fallback_delay(&self, default: Duration) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after, .. } => Some(retry_after.unwrap_or(default)),
            Self::Temporary(_) => Some(default),
            Self::Configuration(_) | Self::Permanent(_) => None,
        }
    }
}

impl fmt::Display for SearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(message) | Self::Temporary(message) | Self::Permanent(message) => {
                formatter.write_str(message)
            }
            Self::RateLimited {
                message,
                retry_after,
            } => {
                formatter.write_str(message)?;
                if let Some(retry_after) = retry_after {
                    write!(formatter, " (retry after {}s)", retry_after.as_secs())?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchResults {
    pub(crate) results: Vec<SearchResult>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchResult {
    pub(crate) title: String,
    pub(crate) url: String,
    pub(crate) snippet: String,
}

impl SearchResult {
    pub(crate) fn from_parts(
        title: Option<&str>,
        url: &str,
        snippet: Option<&str>,
    ) -> Option<Self> {
        let url = url.trim();
        if url.chars().count() > MAX_URL_CHARS {
            return None;
        }
        let Ok(parsed_url) = Url::parse(url) else {
            return None;
        };
        if parsed_url.host_str().is_none() || !matches!(parsed_url.scheme(), "http" | "https") {
            return None;
        }

        let title = clip(
            title
                .map(str::trim)
                .filter(|title| !title.is_empty())
                .unwrap_or("untitled result"),
            MAX_TITLE_CHARS,
        );
        let snippet = snippet
            .map(str::trim)
            .filter(|snippet| !snippet.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| title.clone());

        Some(Self {
            title,
            url: strip_tracking_params(&parsed_url).to_string(),
            snippet: clip(&snippet, MAX_SNIPPET_CHARS),
        })
    }
}

struct ProviderSlot {
    provider: Box<dyn SearchProvider>,
    cooldown_until: Mutex<Option<Instant>>,
}

pub(crate) struct ProviderChain {
    client: Client,
    providers: Vec<ProviderSlot>,
    cooldown: Duration,
}

impl ProviderChain {
    pub(crate) fn from_env() -> Result<Self> {
        let timeout = duration_from_env(
            "NEKORA_WEB_SEARCH_TIMEOUT",
            DEFAULT_TIMEOUT_SECS,
            MAX_TIMEOUT_SECS,
        )?;
        let cooldown = duration_from_env(
            "NEKORA_WEB_SEARCH_COOLDOWN",
            DEFAULT_COOLDOWN_SECS,
            MAX_COOLDOWN_SECS,
        )?;
        let client = Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
            .user_agent("nekora/0.1")
            .build()
            .context("could not create web search HTTP client")?;

        let chain = config::env_or("NEKORA_WEB_SEARCH_CHAIN", DEFAULT_PROVIDER_CHAIN);
        let mut providers = Vec::new();
        for raw_name in chain.split(',') {
            let name = raw_name.trim().to_ascii_lowercase();
            if name.is_empty() {
                continue;
            }
            let provider: Box<dyn SearchProvider> = match name.as_str() {
                "ollama" => Box::new(ollama::OllamaProvider::new(client.clone())?),
                "openrouter" => Box::new(openrouter::OpenRouterProvider::new(client.clone())?),
                other => bail!(
                    "unknown web search provider {other:?}; \
                     supported providers: ollama, openrouter"
                ),
            };
            providers.push(ProviderSlot {
                provider,
                cooldown_until: Mutex::new(None),
            });
        }

        if providers.is_empty() {
            bail!("NEKORA_WEB_SEARCH_CHAIN must contain at least one provider");
        }

        Ok(Self {
            client,
            providers,
            cooldown,
        })
    }

    pub(crate) async fn search(&self, query: &str, limit: usize) -> Result<SearchResults> {
        let query = query.trim();
        if query.is_empty() {
            bail!("search query must not be empty");
        }
        if query.chars().count() > MAX_QUERY_CHARS {
            bail!("search query is too long");
        }
        if !(1..=MAX_RESULTS).contains(&limit) {
            bail!("search result limit must be between 1 and {MAX_RESULTS}");
        }

        // Search providers may treat an exact URL as a lookup key and not follow its redirect.
        let requested_url = exact_query_url(query);
        let resolved_url = if let Some(url) = requested_url.as_ref() {
            Some(
                resolve_requested_url(&self.client, url)
                    .await
                    .unwrap_or_else(|_| url.clone()),
            )
        } else {
            None
        };
        let provider_query = resolved_url
            .as_ref()
            .map(|url| url.as_str())
            .unwrap_or(query);

        let mut failures = Vec::new();
        let mut attempted = false;
        let mut had_search_response = false;
        for slot in &self.providers {
            if slot.is_cooling_down() {
                continue;
            }
            attempted = true;
            match slot.provider.search(provider_query, limit).await {
                Ok(results) if results.results.is_empty() => {
                    had_search_response = true;
                    slot.clear_cooldown();
                    failures.push(format!("{}: no results", slot.provider.name()));
                }
                Ok(results)
                    if requested_url.as_ref().is_some_and(|requested_url| {
                        !results.results.iter().any(|result| {
                            same_page(requested_url, &result.url)
                                || resolved_url.as_ref().is_some_and(|resolved_url| {
                                    same_page(resolved_url, &result.url)
                                })
                        })
                    }) =>
                {
                    had_search_response = true;
                    slot.clear_cooldown();
                }
                Ok(results) => {
                    slot.clear_cooldown();
                    return Ok(results);
                }
                Err(error) => {
                    if matches!(
                        &error,
                        SearchError::Configuration(_) | SearchError::Permanent(_)
                    ) {
                        failures.push(format!("{}: {error}", slot.provider.name()));
                        continue;
                    }
                    let Some(delay) = error.fallback_delay(self.cooldown) else {
                        return Err(anyhow!(
                            "web search provider {} failed: {error}",
                            slot.provider.name()
                        ));
                    };
                    slot.cool_down(delay);
                    failures.push(format!("{}: {error}", slot.provider.name()));
                }
            }
        }

        if !attempted {
            bail!("all configured web search providers are cooling down");
        }
        if requested_url.is_some() && had_search_response {
            // A provider answering without the requested page means "not found", not a provider
            // outage. Do not surface unrelated citations as an exact-URL match.
            return Ok(SearchResults {
                results: Vec::new(),
            });
        }
        if failures.is_empty() {
            bail!("all configured web search providers failed");
        }
        Err(anyhow!(
            "all web search providers failed: {}",
            failures.join("; ")
        ))
    }
}

async fn resolve_requested_url(
    client: &Client,
    url: &Url,
) -> std::result::Result<Url, reqwest::Error> {
    let response = client.get(url.clone()).send().await?;
    Ok(response.url().clone())
}

pub(crate) fn exact_query_url(query: &str) -> Option<Url> {
    let url = Url::parse(query.trim()).ok()?;
    if url.host_str().is_some() && matches!(url.scheme(), "http" | "https") {
        Some(url)
    } else {
        None
    }
}

fn same_page(expected: &Url, actual: &str) -> bool {
    let Ok(actual) = Url::parse(actual) else {
        return false;
    };
    expected.scheme() == actual.scheme()
        && expected.host_str() == actual.host_str()
        && expected.port_or_known_default() == actual.port_or_known_default()
        && expected.path().trim_end_matches('/') == actual.path().trim_end_matches('/')
        && comparable_query(expected) == comparable_query(&actual)
}

fn comparable_query(url: &Url) -> Vec<(String, String)> {
    let mut query = url
        .query_pairs()
        .filter(|(key, _)| !is_tracking_param(key))
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    query.sort_unstable();
    query
}

fn strip_tracking_params(url: &Url) -> Url {
    let retained = url
        .query_pairs()
        .filter(|(key, _)| !is_tracking_param(key))
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    let mut cleaned = url.clone();
    cleaned.set_query(None);
    if !retained.is_empty() {
        let mut query = cleaned.query_pairs_mut();
        for (key, value) in retained {
            query.append_pair(&key, &value);
        }
    }
    cleaned
}

fn is_tracking_param(key: &str) -> bool {
    key.to_ascii_lowercase().starts_with("utm_")
}

impl ProviderSlot {
    fn is_cooling_down(&self) -> bool {
        let now = Instant::now();
        let mut cooldown_until = self
            .cooldown_until
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match *cooldown_until {
            Some(deadline) if deadline > now => true,
            Some(_) => {
                *cooldown_until = None;
                false
            }
            None => false,
        }
    }

    fn clear_cooldown(&self) {
        *self
            .cooldown_until
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
    }

    fn cool_down(&self, delay: Duration) {
        let Some(deadline) = Instant::now().checked_add(delay) else {
            return;
        };
        let mut cooldown_until = self
            .cooldown_until
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if cooldown_until.is_none_or(|current| deadline > current) {
            *cooldown_until = Some(deadline);
        }
    }
}

fn duration_from_env(key: &str, default_secs: u64, max_secs: u64) -> Result<Duration> {
    let default = default_secs.to_string();
    let raw = config::env_or(key, &default);
    let seconds: u64 = raw
        .trim()
        .parse()
        .with_context(|| format!("{key} must be a positive number of seconds"))?;
    if seconds == 0 || seconds > max_secs {
        bail!("{key} must be between 1 and {max_secs} seconds");
    }
    Ok(Duration::from_secs(seconds))
}

pub(crate) fn parse_http_url(raw: &str, key: &str) -> Result<Url> {
    let url = Url::parse(raw.trim()).with_context(|| format!("{key} must be a valid URL"))?;
    if url.host_str().is_none() || !matches!(url.scheme(), "http" | "https") {
        bail!("{key} must use an http or https URL");
    }
    if url.query().is_some() || url.fragment().is_some() {
        bail!("{key} must not contain a query or fragment");
    }
    Ok(url)
}

pub(crate) fn append_path(mut base: Url, path: &str, key: &str) -> Result<Url> {
    let joined = format!("{}{}", base.as_str().trim_end_matches('/'), path);
    base = Url::parse(&joined).with_context(|| format!("{key} must be a valid URL"))?;
    Ok(base)
}

pub(crate) async fn response_json<T: DeserializeOwned>(
    mut response: Response,
    provider: &'static str,
) -> ProviderResult<T> {
    let status = response.status();
    if !status.is_success() {
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|seconds| *seconds > 0)
            .map(|seconds| Duration::from_secs(seconds.min(MAX_COOLDOWN_SECS)));
        let message = format!("{provider} returned HTTP {status}");
        let error = if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            SearchError::Configuration(message)
        } else if status == StatusCode::TOO_MANY_REQUESTS {
            SearchError::RateLimited {
                message,
                retry_after,
            }
        } else if status == StatusCode::REQUEST_TIMEOUT
            || status == StatusCode::TOO_EARLY
            || status.is_server_error()
        {
            SearchError::Temporary(message)
        } else {
            SearchError::Permanent(message)
        };
        return Err(error);
    }

    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        return Err(SearchError::Temporary(format!(
            "{provider} returned an oversized response"
        )));
    }

    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| transport_error(provider, error))?
    {
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(SearchError::Temporary(format!(
                "{provider} returned an oversized response"
            )));
        }
        body.extend_from_slice(&chunk);
    }

    serde_json::from_slice(&body).map_err(|error| {
        SearchError::Temporary(format!("{provider} returned invalid JSON: {error}"))
    })
}

pub(crate) fn transport_error(provider: &'static str, error: reqwest::Error) -> SearchError {
    SearchError::Temporary(format!("{provider} request failed: {error}"))
}

fn clip(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}
