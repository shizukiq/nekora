use std::collections::HashSet;

use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::config;
use crate::promptsall;

use super::{
    append_path, exact_query_url, parse_http_url, response_json, transport_error, SearchError,
    SearchFuture, SearchProvider, SearchResult, SearchResults, MAX_RESULTS, MAX_SNIPPET_CHARS,
};

const DEFAULT_API_BASE: &str = "https://openrouter.ai/api/v1";
const DEFAULT_MODEL: &str = "openai/gpt-4.1-mini";
const DEFAULT_ENGINE: &str = "auto";

pub(crate) struct OpenRouterProvider {
    client: Client,
    endpoint: reqwest::Url,
    api_key: String,
    model: String,
    engine: String,
}

impl OpenRouterProvider {
    pub(crate) fn new(client: Client) -> anyhow::Result<Self> {
        let api_base = parse_http_url(
            &config::env_or("OPENROUTER_API_BASE", DEFAULT_API_BASE),
            "OPENROUTER_API_BASE",
        )?;
        let endpoint = append_path(api_base, "/chat/completions", "OPENROUTER_API_BASE")?;
        Ok(Self {
            client,
            endpoint,
            api_key: config::env_or("OPENROUTER_API_KEY", ""),
            model: config::env_or("OPENROUTER_WEB_SEARCH_MODEL", DEFAULT_MODEL),
            engine: config::env_or("OPENROUTER_WEB_SEARCH_ENGINE", DEFAULT_ENGINE),
        })
    }
}

impl SearchProvider for OpenRouterProvider {
    fn name(&self) -> &'static str {
        "openrouter"
    }

    fn search<'a>(&'a self, query: &'a str, limit: usize) -> SearchFuture<'a> {
        Box::pin(async move {
            if self.api_key.trim().is_empty() {
                return Err(SearchError::Configuration(
                    "OPENROUTER_API_KEY is required for the openrouter web search provider"
                        .to_string(),
                ));
            }
            if self.model.trim().is_empty() {
                return Err(SearchError::Configuration(
                    "OPENROUTER_WEB_SEARCH_MODEL must not be empty".to_string(),
                ));
            }
            if self.engine.trim().is_empty() {
                return Err(SearchError::Configuration(
                    "OPENROUTER_WEB_SEARCH_ENGINE must not be empty".to_string(),
                ));
            }

            let response = self
                .client
                .post(self.endpoint.clone())
                .bearer_auth(&self.api_key)
                .json(&json!({
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": format!("{}\nQuery: {query}", promptsall::WEB_SEARCH_INSTRUCTION)
                    }],
                    "tools": [{
                        "type": "openrouter:web_search",
                        "parameters": {
                            "engine": self.engine,
                            "max_results": limit.min(MAX_RESULTS),
                            "max_total_results": limit.min(MAX_RESULTS),
                            "max_uses": 1,
                            "max_characters": MAX_SNIPPET_CHARS
                        }
                    }]
                }))
                .send()
                .await
                .map_err(|error| transport_error(self.name(), error))?;
            let payload: OpenRouterResponse = response_json(response, self.name()).await?;
            let message = payload
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| {
                    SearchError::Temporary("openrouter returned no search response".to_string())
                })?
                .message;

            let fallback_snippet = exact_query_url(query)
                .and(message.content.as_deref())
                .filter(|content| !content.trim().is_empty());
            let mut seen_urls = HashSet::new();
            let mut results = message
                .annotations
                .into_iter()
                .filter_map(|annotation| annotation.into_result(fallback_snippet))
                .filter(|result| seen_urls.insert(result.url.clone()))
                .take(limit)
                .collect::<Vec<_>>();
            if results.is_empty() {
                if let (Some(url), Some(content)) = (exact_query_url(query), message.content) {
                    if let Some(result) =
                        SearchResult::from_parts(Some(url.as_str()), url.as_str(), Some(&content))
                    {
                        results.push(result);
                    }
                }
            }
            Ok(SearchResults { results })
        })
    }
}

#[derive(Deserialize)]
struct OpenRouterResponse {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    annotations: Vec<Annotation>,
}

#[derive(Deserialize)]
struct Annotation {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    url_citation: Option<Citation>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct Citation {
    url: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    content: Option<String>,
}

impl Annotation {
    fn into_result(self, fallback_snippet: Option<&str>) -> Option<SearchResult> {
        if self.kind != "url_citation" {
            return None;
        }

        let (url, title, content) = match self.url_citation {
            Some(citation) => (citation.url, citation.title, citation.content),
            None => (self.url?, self.title, self.content),
        };
        let snippet = content
            .as_deref()
            .filter(|content| !content.trim().is_empty())
            .or(fallback_snippet);
        SearchResult::from_parts(title.as_deref(), &url, snippet)
    }
}
