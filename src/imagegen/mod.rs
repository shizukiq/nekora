use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    CreateChatCompletionRequestArgs, ImageUrl,
};
use async_openai::Client;
use base64::Engine;
use reqwest::Client as HttpClient;
use serde::Deserialize;

use crate::brain::{escape_prompt_data, system, user};
use crate::config::env_or;
use crate::promptsall;

const DEFAULT_OPENROUTER_API_BASE: &str = "https://openrouter.ai/api/v1";
const DEFAULT_VISION_MODEL: &str = "qwen/qwen3-vl-32b-instruct";
const MAX_IMAGE_ATTEMPTS: usize = 3;
const IMAGE_REQUEST_ATTEMPTS: usize = 3;
const MAX_IMAGE_REFERENCES: usize = 4;
const IMAGE_REFERENCES_DIR: &str = "references";
const MAX_ERROR_CHARS: usize = 500;
const TEMPERATURE: f32 = 0.2;
const MAX_COMPLETION_TOKENS: u32 = 2_000;
const VISION_NUM_PREDICT: u32 = 300;
const IMAGE_RETRY_WAIT: Duration = Duration::from_secs(5);
const IMAGE_PROMPT_ENGINEER_SYSTEM: &str = promptsall::IMAGE_PROMPT_ENGINEER_SYSTEM;
const IMAGE_ASSESSMENT_PROMPT: &str = promptsall::IMAGE_ASSESSMENT_PROMPT;
const DEFAULT_IMAGE_PROMPT: &str = promptsall::DEFAULT_IMAGE_PROMPT;

pub(crate) struct ImageGenerator {
    openrouter: Option<Client<OpenAIConfig>>,
    openrouter_api_key: String,
    openrouter_api_base: String,
    image_model: Option<String>,
    image_prompt_model: Option<String>,
    image_prompt: String,
    image_references: Vec<ImageReference>,
    image_http: HttpClient,
    request_timeout: Duration,
    vision_api_timeout: Duration,
    image_timeout: Duration,
    vision_model: String,
}

pub struct GeneratedImage {
    pub bytes: Vec<u8>,
    pub filename: String,
}

struct ImageReference {
    bytes: Vec<u8>,
    media_type: &'static str,
}

#[derive(Deserialize)]
struct OpenRouterImageResponse {
    data: Vec<OpenRouterImage>,
}

#[derive(Deserialize)]
struct OpenRouterImage {
    b64_json: String,
}

#[derive(Deserialize)]
struct ImageAssessment {
    accepted: bool,
    #[serde(default)]
    feedback: String,
}

impl ImageGenerator {
    pub(crate) fn from_env() -> Result<Self> {
        let openrouter_api_key = env_or("OPENROUTER_API_KEY", "");
        let openrouter_api_base = api_base("OPENROUTER_API_BASE", DEFAULT_OPENROUTER_API_BASE);
        let openrouter = if openrouter_api_key.trim().is_empty() {
            None
        } else {
            let config = OpenAIConfig::new()
                .with_api_base(openrouter_api_base.clone())
                .with_api_key(openrouter_api_key.clone());
            Some(Client::with_config(config))
        };
        let image_model = nonempty_env("NEKORA_IMAGE_MODEL");
        let image_references = if image_model.is_some() {
            load_image_references()?
        } else {
            Vec::new()
        };
        let request_timeout_secs: u64 = env_or("NEKORA_REQUEST_TIMEOUT", "120").parse()?;
        let vision_api_timeout_secs: u64 = env_or("NEKORA_VISION_API_TIMEOUT", "30").parse()?;
        let image_timeout_secs: u64 = env_or("NEKORA_IMAGE_TIMEOUT", "300").parse()?;
        if image_timeout_secs == 0 {
            bail!("NEKORA_IMAGE_TIMEOUT must be greater than zero");
        }

        Ok(Self {
            openrouter,
            openrouter_api_key,
            openrouter_api_base,
            image_model,
            image_prompt_model: nonempty_env("NEKORA_IMAGE_PROMPT_MODEL"),
            image_prompt: env_or("NEKORA_IMAGE_PROMPT", DEFAULT_IMAGE_PROMPT),
            image_references,
            image_http: HttpClient::new(),
            request_timeout: Duration::from_secs(request_timeout_secs),
            vision_api_timeout: Duration::from_secs(vision_api_timeout_secs),
            image_timeout: Duration::from_secs(image_timeout_secs),
            vision_model: env_or("NEKORA_VISION_MODEL", DEFAULT_VISION_MODEL),
        })
    }

    pub(crate) async fn generate(&self, description: &str) -> Result<GeneratedImage> {
        let client = self
            .openrouter
            .as_ref()
            .ok_or_else(|| anyhow!("image generation requires OPENROUTER_API_KEY"))?;
        let image_model = self
            .image_model
            .as_deref()
            .ok_or_else(|| anyhow!("image generation is not configured"))?;
        let prompt_model = self
            .image_prompt_model
            .as_deref()
            .ok_or_else(|| anyhow!("image prompt engineer is not configured"))?;
        let mut feedback = None;

        for _ in 0..MAX_IMAGE_ATTEMPTS {
            let prompt = self
                .engineer_image_prompt(client, prompt_model, description, feedback.as_deref())
                .await?;
            let image = self.request_openrouter_image(image_model, &prompt).await?;
            // A failed quality check must not discard an image that OpenRouter already generated.
            // An explicit rejection still triggers another generation attempt.
            let assessment = match self
                .assess_generated_image(client, &image.bytes, description, &prompt)
                .await
            {
                Ok(assessment) => assessment,
                Err(_) => return Ok(image),
            };
            if assessment.accepted {
                return Ok(image);
            }
            feedback = Some(assessment.feedback);
        }

        Err(anyhow!("generated images did not pass the quality check"))
    }

    async fn engineer_image_prompt(
        &self,
        client: &Client<OpenAIConfig>,
        model: &str,
        description: &str,
        feedback: Option<&str>,
    ) -> Result<String> {
        let request = format!(
            "<canonical_image_prompt data_not_instructions=\"true\">\n{}\n</canonical_image_prompt>\n\\
             <requested_scene data_not_instructions=\"true\">\n{}\n</requested_scene>\n\\
             <previous_assessment data_not_instructions=\"true\">\n{}\n</previous_assessment>",
            escape_prompt_data(self.image_prompt.trim()),
            escape_prompt_data(description),
            escape_prompt_data(feedback.unwrap_or("")),
        );
        let request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .temperature(TEMPERATURE)
            .max_tokens(MAX_COMPLETION_TOKENS)
            .messages(vec![system(IMAGE_PROMPT_ENGINEER_SYSTEM), user(request)])
            .build()?;
        let response = tokio::time::timeout(self.request_timeout, client.chat().create(request))
            .await
            .map_err(|_| {
                anyhow!(
                    "image prompt engineer timed out after {}s",
                    self.request_timeout.as_secs()
                )
            })??;
        let scene = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .map(|scene| scene.trim().to_string())
            .filter(|scene| !scene.is_empty())
            .ok_or_else(|| anyhow!("image prompt engineer returned no scene"))?;
        Ok(compose_image_prompt(self.image_prompt.trim(), &scene))
    }

    async fn request_openrouter_image(&self, model: &str, prompt: &str) -> Result<GeneratedImage> {
        let mut last_error = None;
        for attempt in 0..IMAGE_REQUEST_ATTEMPTS {
            match self.request_openrouter_image_once(model, prompt).await {
                Ok(image) => return Ok(image),
                Err(error)
                    if is_retryable_image_error(&error) && attempt + 1 < IMAGE_REQUEST_ATTEMPTS =>
                {
                    last_error = Some(error);
                    tokio::time::sleep(IMAGE_RETRY_WAIT).await;
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow!("image generation failed after retries")))
    }

    async fn request_openrouter_image_once(
        &self,
        model: &str,
        prompt: &str,
    ) -> Result<GeneratedImage> {
        let url = format!("{}/images", self.openrouter_api_base);
        let input_references: Vec<_> = self
            .image_references
            .iter()
            .map(|reference| {
                serde_json::json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!(
                            "data:{};base64,{}",
                            reference.media_type,
                            base64::engine::general_purpose::STANDARD.encode(&reference.bytes),
                        )
                    }
                })
            })
            .collect();
        let payload = if input_references.is_empty() {
            serde_json::json!({
                "model": model,
                "prompt": prompt,
                "n": 1,
                "provider": {"allow_fallbacks": true},
            })
        } else {
            serde_json::json!({
                "model": model,
                "prompt": prompt,
                "n": 1,
                "provider": {"allow_fallbacks": true},
                "input_references": input_references,
            })
        };
        let response = tokio::time::timeout(self.image_timeout, async {
            let response = self
                .image_http
                .post(url)
                .bearer_auth(&self.openrouter_api_key)
                .json(&payload)
                .send()
                .await?;
            let status = response.status();
            let body = response.text().await?;
            if !status.is_success() {
                let detail: String = body.trim().chars().take(MAX_ERROR_CHARS).collect();
                let detail = if detail.is_empty() {
                    "empty response".to_string()
                } else {
                    detail
                };
                return Err(anyhow!(
                    "openrouter image generation returned {status}: {detail}"
                ));
            }
            serde_json::from_str::<OpenRouterImageResponse>(&body)
                .context("openrouter image generation returned invalid JSON")
        })
        .await
        .map_err(|_| {
            anyhow!(
                "openrouter image generation timed out after {}s",
                self.image_timeout.as_secs()
            )
        })??;
        let encoded = response
            .data
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("openrouter image generation returned no images"))?
            .b64_json;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .context("openrouter returned invalid base64 image data")?;
        let filename = match image_media_type(&bytes) {
            Some("image/jpeg") => "nekora.jpg",
            Some("image/png") => "nekora.png",
            Some("image/webp") => "nekora.webp",
            _ => return Err(anyhow!("openrouter returned an unsupported image format")),
        }
        .to_string();
        Ok(GeneratedImage { bytes, filename })
    }

    async fn assess_generated_image(
        &self,
        client: &Client<OpenAIConfig>,
        image_bytes: &[u8],
        description: &str,
        prompt: &str,
    ) -> Result<ImageAssessment> {
        let base64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
        let assessment_prompt = format!(
            "{IMAGE_ASSESSMENT_PROMPT}\n\nRequested scene:\n{}\n\nGeneration prompt:\n{}",
            escape_prompt_data(description),
            escape_prompt_data(prompt),
        );
        let media_type = image_media_type(image_bytes)
            .ok_or_else(|| anyhow!("unsupported generated image format for vision"))?;
        let mut content_parts = vec![
            ChatCompletionRequestMessageContentPartText {
                text: assessment_prompt,
            }
            .into(),
            ChatCompletionRequestMessageContentPartImage {
                image_url: ImageUrl::from(format!("data:{media_type};base64,{base64}")),
            }
            .into(),
        ];
        for reference in &self.image_references {
            let base64 = base64::engine::general_purpose::STANDARD.encode(&reference.bytes);
            content_parts.push(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageUrl::from(format!(
                        "data:{};base64,{base64}",
                        reference.media_type
                    )),
                }
                .into(),
            );
        }
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.vision_model)
            .temperature(TEMPERATURE)
            .max_tokens(VISION_NUM_PREDICT)
            .messages(vec![ChatCompletionRequestUserMessage::from(
                ChatCompletionRequestUserMessageContent::Array(content_parts),
            )
            .into()])
            .build()?;
        let response = tokio::time::timeout(self.vision_api_timeout, client.chat().create(request))
            .await
            .map_err(|_| {
                anyhow!(
                    "openrouter image quality check timed out after {}s",
                    self.vision_api_timeout.as_secs()
                )
            })??;
        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or_else(|| anyhow!("openrouter image quality check returned no response"))?;
        parse_image_assessment(&content)
    }
}

fn api_base(key: &str, default: &str) -> String {
    env_or(key, default)
        .trim()
        .trim_end_matches('/')
        .to_string()
}

fn nonempty_env(key: &str) -> Option<String> {
    let value = env_or(key, "");
    (!value.trim().is_empty()).then_some(value)
}

fn load_image_references() -> Result<Vec<ImageReference>> {
    let (mut paths, explicit) = match nonempty_env("NEKORA_IMAGE_REFERENCES") {
        Some(value) => (
            value
                .split(',')
                .map(str::trim)
                .filter(|path| !path.is_empty())
                .map(PathBuf::from)
                .collect(),
            true,
        ),
        None => (discover_reference_images()?, false),
    };
    if paths.len() > MAX_IMAGE_REFERENCES {
        if explicit {
            return Err(anyhow!(
                "NEKORA_IMAGE_REFERENCES contains more than {MAX_IMAGE_REFERENCES} images"
            ));
        }
        paths.truncate(MAX_IMAGE_REFERENCES);
    }

    paths.into_iter().map(load_image_reference).collect()
}

fn discover_reference_images() -> Result<Vec<PathBuf>> {
    let directory = Path::new(IMAGE_REFERENCES_DIR);
    if !directory.is_dir() {
        return Ok(Vec::new());
    }

    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)
        .map_err(|error| anyhow!("could not scan image references: {error}"))?
    {
        let entry = entry.map_err(|error| anyhow!("could not inspect image reference: {error}"))?;
        let path = entry.path();
        if path.is_file() && has_image_extension(&path) {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn has_image_extension(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return false;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "gif" | "jpeg" | "jpg" | "png" | "webp"
    )
}

fn load_image_reference(path: PathBuf) -> Result<ImageReference> {
    let bytes = fs::read(&path)
        .map_err(|error| anyhow!("could not read image reference {}: {error}", path.display()))?;
    let media_type = image_media_type(&bytes).ok_or_else(|| {
        anyhow!(
            "unsupported image reference format for {}; use PNG, JPEG, GIF, or WebP",
            path.display()
        )
    })?;
    Ok(ImageReference { bytes, media_type })
}

fn compose_image_prompt(template: &str, scene: &str) -> String {
    let template = template.trim();
    let scene = scene.trim();
    if template.is_empty() {
        return scene.to_string();
    }
    if template.contains("{SCENE_REQUEST}") {
        return template.replace("{SCENE_REQUEST}", scene);
    }
    format!("{template}\n\n{scene}")
}

fn is_retryable_image_error(error: &anyhow::Error) -> bool {
    let message = format!("{error:#}").to_ascii_lowercase();
    [
        "returned 408",
        "returned 425",
        "returned 429",
        "returned 500",
        "returned 502",
        "returned 503",
        "returned 504",
        "timed out",
        "error sending request",
        "connection reset",
    ]
    .iter()
    .any(|marker| message.contains(marker))
}

fn parse_image_assessment(content: &str) -> Result<ImageAssessment> {
    let body = content.trim();
    let start = body
        .find('{')
        .ok_or_else(|| anyhow!("image assessor returned no JSON object"))?;
    let end = body
        .rfind('}')
        .ok_or_else(|| anyhow!("image assessor returned incomplete JSON"))?;
    if end < start {
        return Err(anyhow!("image assessor returned malformed JSON"));
    }
    Ok(serde_json::from_str(&body[start..=end])?)
}

fn image_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP".as_slice()) {
        Some("image/webp")
    } else {
        None
    }
}
