use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::config;

use super::{
    parse_http_url, response_json, transport_error, SearchError, SearchFuture, SearchProvider,
    SearchResult, SearchResults, MAX_RESULTS,
};

const DEFAULT_ENDPOINT: &str = "https://ollama.com/api/web_search";

pub(crate) struct OllamaProvider {
    client: Client,
    endpoint: reqwest::Url,
    api_key: String,
}

impl OllamaProvider {
    pub(crate) fn new(client: Client) -> anyhow::Result<Self> {
        let endpoint = parse_http_url(
            &config::env_or("OLLAMA_WEB_SEARCH_URL", DEFAULT_ENDPOINT),
            "OLLAMA_WEB_SEARCH_URL",
        )?;
        Ok(Self {
            client,
            endpoint,
            api_key: config::env_or("OLLAMA_API_KEY", ""),
        })
    }
}

impl SearchProvider for OllamaProvider {
    fn name(&self) -> &'static str {
        "ollama"
    }

    fn search<'a>(&'a self, query: &'a str, limit: usize) -> SearchFuture<'a> {
        Box::pin(async move {
            if self.api_key.trim().is_empty() {
                return Err(SearchError::Configuration(
                    "OLLAMA_API_KEY is required for the ollama web search provider".to_string(),
                ));
            }

            let response = self
                .client
                .post(self.endpoint.clone())
                .bearer_auth(&self.api_key)
                .json(&SearchRequest {
                    query,
                    max_results: limit.min(MAX_RESULTS),
                })
                .send()
                .await
                .map_err(|error| transport_error(self.name(), error))?;
            let payload: SearchResponse = response_json(response, self.name()).await?;
            let results = payload
                .results
                .into_iter()
                .filter_map(|result| {
                    SearchResult::from_parts(
                        Some(&result.title),
                        &result.url,
                        result.content.as_deref(),
                    )
                })
                .take(limit)
                .collect();
            Ok(SearchResults { results })
        })
    }
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    query: &'a str,
    max_results: usize,
}

#[derive(Deserialize)]
struct SearchResponse {
    #[serde(default)]
    results: Vec<OllamaResult>,
}

#[derive(Deserialize)]
struct OllamaResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: Option<String>,
}
