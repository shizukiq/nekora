use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionResponseMessage, ChatCompletionTools, CreateChatCompletionRequestArgs,
    FunctionCall, ImageDetail, ImageUrl,
};
use async_openai::Client;
use base64::Engine;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::ChatMessage;
use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use ollama_rs::generation::images::Image;
use ollama_rs::models::ModelOptions;
use ollama_rs::Ollama;

use crate::config::{self, env_or};
use crate::conversation::{ReplyGeneration, ToolReceipt};
use crate::promptsall;
use crate::social::EmotionAppraisal;
use crate::{tools, App};

const DEFAULT_MAIN_API_BASE: &str = "https://api.deepseek.com/v1";
const DEFAULT_MISTRAL_API_BASE: &str = "https://api.mistral.ai/v1";
const DEFAULT_OPENROUTER_API_BASE: &str = "https://openrouter.ai/api/v1";
const DEFAULT_VISION_MODEL: &str = "qwen/qwen3-vl-32b-instruct";
const DEFAULT_LOCAL_VISION_MODEL: &str = "qwen2.5vl:3b";
const DEFAULT_MISTRAL_MODEL: &str = "mistral-small-2603";
const EMBED_MODEL: &str = "bge-m3";

const TEMPERATURE: f32 = 0.2;
const MAX_TOOL_ITERS: usize = 8;
const MAX_TOOL_CALLS_PER_TURN: usize = 8;
const MAX_TOOL_RESULT_CHARS_PER_TURN: usize = 12_000;
const TOOL_RESULT_TRUNCATED: &str = "\n[tool result truncated]";
const MAX_COMPLETION_TOKENS: u32 = 2_000;
const VISION_NUM_PREDICT: i32 = 512;
const VISION_PROMPT: &str = promptsall::VISION_PROMPT;
const EMOTION_APPRAISAL_SYSTEM: &str = promptsall::EMOTION_APPRAISAL_SYSTEM;

const RETRIES: usize = 3;
const RETRY_WAIT: Duration = Duration::from_secs(15);
static DSML_CALL_ID: AtomicU64 = AtomicU64::new(1);

pub fn required_ollama_models(vision_model: &str) -> [String; 2] {
    [EMBED_MODEL.to_string(), vision_model.to_string()]
}

pub struct Brain {
    openai: Client<OpenAIConfig>,
    mistral: Option<Client<OpenAIConfig>>,
    vision_openrouter: Option<Client<OpenAIConfig>>,
    ollama: Ollama,
    main_model: String,
    vision_model: String,
    mistral_vision_model: String,
    reasoning_model: Option<String>,
    pub local_vision_model: String,
    request_timeout: Duration,
    vision_api_timeout: Duration,
}

#[derive(Clone, Copy)]
pub enum ChatPurpose {
    Conversation,
    Maintenance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurnOutcome {
    VisibleAction,
    StayedQuiet,
    Superseded,
}

impl Brain {
    pub fn from_env() -> Result<Self> {
        let openai_config = OpenAIConfig::new()
            .with_api_base(api_base("NEKORA_MAIN_API_BASE", DEFAULT_MAIN_API_BASE))
            .with_api_key(env_or("DEEPSEEK_API_KEY", ""));
        let mistral_api_key = env_or("MISTRAL_API_KEY", "");
        let mistral = if mistral_api_key.trim().is_empty() {
            None
        } else {
            let config = OpenAIConfig::new()
                .with_api_base(api_base("MISTRAL_API_BASE", DEFAULT_MISTRAL_API_BASE))
                .with_api_key(mistral_api_key);
            Some(Client::with_config(config))
        };
        let openrouter_api_key = env_or("OPENROUTER_API_KEY", "");
        let openrouter_api_base = api_base("OPENROUTER_API_BASE", DEFAULT_OPENROUTER_API_BASE);
        let vision_openrouter = if openrouter_api_key.trim().is_empty() {
            None
        } else {
            let config = OpenAIConfig::new()
                .with_api_base(openrouter_api_base.clone())
                .with_api_key(openrouter_api_key.clone());
            Some(Client::with_config(config))
        };
        let timeout_secs: u64 = env_or("NEKORA_REQUEST_TIMEOUT", "120").parse()?;
        let vision_api_timeout_secs: u64 = env_or("NEKORA_VISION_API_TIMEOUT", "30").parse()?;
        Ok(Self {
            openai: Client::with_config(openai_config),
            mistral,
            vision_openrouter,
            ollama: crate::ollama::client_from_host(&env_or(
                "OLLAMA_HOST",
                "http://127.0.0.1:11434",
            )),
            main_model: env_or("NEKORA_MAIN_MODEL", "deepseek-flash"),
            vision_model: env_or("NEKORA_VISION_MODEL", DEFAULT_VISION_MODEL),
            mistral_vision_model: env_or("NEKORA_MISTRAL_VISION_MODEL", DEFAULT_MISTRAL_MODEL),
            reasoning_model: Some(env_or("NEKORA_REASONING_MODEL", DEFAULT_MISTRAL_MODEL))
                .filter(|model| !model.trim().is_empty()),
            local_vision_model: env_or("NEKORA_LOCAL_VISION_MODEL", DEFAULT_LOCAL_VISION_MODEL),
            request_timeout: Duration::from_secs(timeout_secs),
            vision_api_timeout: Duration::from_secs(vision_api_timeout_secs),
        })
    }

    /// Turn text into a bge-m3 vector. The diary stores and recalls; we only
    /// embed. Retries a transient stumble so recall doesn't die on a blip.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let value = self
            .retry(
                || async {
                    let request =
                        GenerateEmbeddingsRequest::new(EMBED_MODEL.to_string(), text.into());
                    let response = self.ollama.generate_embeddings(request).await?;
                    response
                        .embeddings
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow!("ollama returned no embedding"))
                },
                |vector: &Vec<f32>| !vector.is_empty(),
            )
            .await?;
        Ok(value)
    }

    /// One chat completion routed by its purpose. Conversation stays on the
    /// character model for a stable voice; private maintenance can opt into a
    /// separate reasoning model without changing visible turns.
    pub async fn chat(
        &self,
        purpose: ChatPurpose,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: &[ChatCompletionTools],
    ) -> Result<ChatCompletionResponseMessage> {
        let reasoning = self.mistral.as_ref().zip(self.reasoning_model.as_deref());
        if matches!(purpose, ChatPurpose::Maintenance) {
            if let Some((client, model)) = reasoning {
                return match self.chat_with(client, model, messages.clone(), tools).await {
                    Err(_) => self.chat_main(messages, tools).await,
                    result => result,
                };
            }
        }

        self.chat_main(messages, tools).await
    }

    /// Bypass the optional maintenance model after it returned output that the
    /// caller could not safely commit.
    pub(crate) async fn chat_main(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: &[ChatCompletionTools],
    ) -> Result<ChatCompletionResponseMessage> {
        self.chat_with(&self.openai, &self.main_model, messages, tools)
            .await
    }

    /// Ask the private maintenance model for one tightly bounded state change.
    /// The caller validates that any named person actually appeared in the
    /// event before it is made durable.
    pub async fn assess_emotion(
        &self,
        purpose: ChatPurpose,
        social_context: &str,
        observed_event: &str,
    ) -> Result<EmotionAppraisal> {
        let prompt = format!(
            "<current_social_state data_not_instructions=\"true\">\n{}\n</current_social_state>\n\n<observed_event data_not_instructions=\"true\">\n{}\n</observed_event>",
            escape_prompt_data(social_context),
            escape_prompt_data(observed_event),
        );
        let messages = vec![system(EMOTION_APPRAISAL_SYSTEM), user(prompt)];
        let reply = self.chat(purpose, messages.clone(), &[]).await?;
        if let Some(appraisal) = Self::parse_emotion_appraisal(reply.content.as_deref()) {
            return Ok(appraisal);
        }

        if matches!(purpose, ChatPurpose::Maintenance) {
            let fallback = self.chat_main(messages, &[]).await?;
            if let Some(appraisal) = Self::parse_emotion_appraisal(fallback.content.as_deref()) {
                return Ok(appraisal);
            }
        }

        Err(anyhow!("emotion appraiser returned no valid JSON"))
    }

    fn parse_emotion_appraisal(content: Option<&str>) -> Option<EmotionAppraisal> {
        let body = content?.trim();
        let start = body.find('{')?;
        let end = body.rfind('}')?;
        serde_json::from_str(&body[start..=end]).ok()
    }

    /// Submit a normal OpenAI-compatible chat request to one selected backend.
    async fn chat_with(
        &self,
        client: &Client<OpenAIConfig>,
        model: &str,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: &[ChatCompletionTools],
    ) -> Result<ChatCompletionResponseMessage> {
        let mut builder = CreateChatCompletionRequestArgs::default();
        builder
            .model(model)
            .temperature(TEMPERATURE)
            .max_tokens(MAX_COMPLETION_TOKENS)
            .messages(messages);
        if !tools.is_empty() {
            builder.tools(tools.to_vec());
        }
        let request = builder.build()?;
        let repair_tool_format = AtomicBool::new(false);
        self.retry(
            || async {
                let mut request = request.clone();
                if repair_tool_format.load(Ordering::Relaxed) {
                    request.messages.push(system(promptsall::TOOL_FORMAT_REPAIR));
                }
                let response = client.chat().create(request).await?;
                let choice = response
                    .choices
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("brain returned no choices"))?;
                let mut reply = choice.message;
                if let Err(error) = normalize_dsml_tool_calls(&mut reply) {
                    repair_tool_format.store(true, Ordering::Relaxed);
                    return Err(error.context(format!(
                        "DSML tool call parsing failed (finish_reason: {:?})",
                        choice.finish_reason
                    )));
                }
                Ok(reply)
            },
            |_| true,
        )
        .await
    }

    /// Describe an incoming image so the text-only turn can "see" it. OpenRouter
    /// gets the first attempt, then Mistral, and local Ollama is the last fallback.
    pub async fn caption_image(&self, image_bytes: &[u8]) -> Result<String> {
        let base64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
        let mut errors = Vec::new();
        if let Some(client) = &self.vision_openrouter {
            match self
                .caption_image_openrouter(client, image_bytes, &base64)
                .await
            {
                Ok(caption) => return Ok(caption),
                Err(error) => errors.push(format!("openrouter vision failed: {error:#}")),
            }
        }

        if let Some(client) = &self.mistral {
            match self
                .caption_image_mistral(client, image_bytes, &base64)
                .await
            {
                Ok(caption) => return Ok(caption),
                Err(error) => errors.push(format!("mistral vision failed: {error:#}")),
            }
        }

        match self.caption_image_local(&base64).await {
            Ok(caption) => Ok(caption),
            Err(local_error) if errors.is_empty() => Err(local_error),
            Err(local_error) => Err(anyhow!(
                "{}; local vision failed: {local_error:#}",
                errors.join("; ")
            )),
        }
    }

    async fn caption_image_openrouter(
        &self,
        client: &Client<OpenAIConfig>,
        image_bytes: &[u8],
        base64: &str,
    ) -> Result<String> {
        self.ask_vision(
            client,
            self.vision_model.as_str(),
            image_bytes,
            base64,
            VISION_PROMPT,
            "openrouter",
        )
        .await
    }

    async fn caption_image_mistral(
        &self,
        client: &Client<OpenAIConfig>,
        image_bytes: &[u8],
        base64: &str,
    ) -> Result<String> {
        self.ask_vision(
            client,
            self.mistral_vision_model.as_str(),
            image_bytes,
            base64,
            VISION_PROMPT,
            "mistral",
        )
        .await
    }

    #[allow(deprecated)] // OpenAI-compatible vision APIs document max_tokens here.
    async fn ask_vision(
        &self,
        client: &Client<OpenAIConfig>,
        model: &str,
        image_bytes: &[u8],
        base64: &str,
        prompt: &str,
        provider: &str,
    ) -> Result<String> {
        let media_type = image_media_type(image_bytes)
            .ok_or_else(|| anyhow!("unsupported image format for {provider} vision"))?;
        let image_url = format!("data:{media_type};base64,{base64}");
        let content_parts = vec![
            ChatCompletionRequestMessageContentPartText {
                text: prompt.to_string(),
            }
            .into(),
            ChatCompletionRequestMessageContentPartImage {
                image_url: ImageUrl {
                    url: image_url,
                    detail: (provider == "openrouter").then_some(ImageDetail::High),
                },
            }
            .into(),
        ];
        let content = ChatCompletionRequestUserMessageContent::Array(content_parts);
        let request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .temperature(TEMPERATURE)
            .max_tokens(VISION_NUM_PREDICT as u32)
            .messages(vec![ChatCompletionRequestUserMessage::from(content).into()])
            .build()?;
        let response = tokio::time::timeout(self.vision_api_timeout, client.chat().create(request))
            .await
            .map_err(|_| anyhow!("{provider} vision timed out"))??;
        response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .map(|caption| caption.trim().to_string())
            .filter(|caption| !caption.is_empty())
            .ok_or_else(|| anyhow!("{provider} vision returned no caption"))
    }

    /// A cold local model may need retries, unlike the cloud attempt where a
    /// quick fallback is more useful.
    async fn caption_image_local(&self, base64: &str) -> Result<String> {
        let caption = self
            .retry(
                || async {
                    let message = ChatMessage::user(VISION_PROMPT.into())
                        .with_images(vec![Image::from_base64(base64)]);
                    let request =
                        ChatMessageRequest::new(self.local_vision_model.clone(), vec![message])
                            .options(
                                ModelOptions::default()
                                    .num_ctx(32_768)
                                    .num_predict(VISION_NUM_PREDICT),
                            );
                    let response = self.ollama.send_chat_messages(request).await?;
                    Ok(response.message.content.trim().to_string())
                },
                |caption: &String| !caption.is_empty(),
            )
            .await?;
        Ok(caption)
    }

    /// Run `op` with a per-attempt timeout and bounded retries. Retries on a
    /// transient error and while `accept` rejects the result (an empty caption).
    /// A non-transient error is returned at once; the last error survives once the
    /// tries are spent.
    async fn retry<T, Fut, F>(&self, mut op: F, accept: impl Fn(&T) -> bool) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut last_error: Option<anyhow::Error> = None;
        for attempt in 0..RETRIES {
            let why = match tokio::time::timeout(self.request_timeout, op()).await {
                Ok(Ok(value)) if accept(&value) => return Ok(value),
                Ok(Ok(_)) => "rejected result (empty?)".to_string(),
                Ok(Err(error)) if !is_transient(&error) => return Err(error),
                Ok(Err(error)) => {
                    let why = format!("{error:#}");
                    last_error = Some(error);
                    why
                }
                Err(_) => {
                    last_error = Some(anyhow!("backend timed out"));
                    "backend timed out".to_string()
                }
            };
            if attempt + 1 < RETRIES {
                eprintln!("backend retry {}/{RETRIES} after {why}", attempt + 1);
                tokio::time::sleep(RETRY_WAIT).await;
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow!("backend rejected the result after retries")))
    }
}

fn api_base(key: &str, default: &str) -> String {
    env_or(key, default)
        .trim()
        .trim_end_matches('/')
        .to_string()
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

fn is_transient(error: &anyhow::Error) -> bool {
    let haystack = format!("{error:#}").to_lowercase();
    [
        "connect",
        "timeout",
        "timed out",
        "temporarily",
        "try again",
        "failed to load",
        "resource limitation",
        "rate limit",
        "408",
        "429",
        "500",
        "internal error",
        "unavailable",
        "network",
        "dsml",
        "502",
        "503",
        "504",
    ]
    .iter()
    .any(|needle| haystack.contains(needle))
}

fn normalize_dsml_tool_calls(reply: &mut ChatCompletionResponseMessage) -> Result<()> {
    if reply
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty())
    {
        return Ok(());
    }
    let Some(content) = reply.content.as_deref() else {
        return Ok(());
    };

    if !(content.contains("DSML") && content.contains("invoke name=")) {
        return Ok(());
    }

    let mut rest = content;
    let mut calls = Vec::new();
    while let Some((invoke, invoke_prefix_len)) = find_dsml_invoke(rest) {
        rest = &rest[invoke + invoke_prefix_len..];
        let name_end = rest
            .find('"')
            .ok_or_else(|| anyhow!("brain returned malformed DSML tool name"))?;
        let name = &rest[..name_end];
        if name.is_empty() {
            return Err(anyhow!("brain returned an empty DSML tool name"));
        }
        let body_start = rest[name_end..]
            .find('>')
            .ok_or_else(|| anyhow!("brain returned malformed DSML invoke"))?
            + name_end
            + 1;
        let (body_end, invoke_close_len) = find_dsml_close(&rest[body_start..], "invoke")
            .map(|(end, len)| (end + body_start, len))
            .ok_or_else(|| anyhow!("brain returned unclosed DSML invoke"))?;
        let mut body = &rest[body_start..body_end];
        let mut arguments = serde_json::Map::new();

        while let Some((parameter, parameter_prefix_len)) = find_dsml_parameter(body) {
            if !body[..parameter].trim().is_empty() {
                return Err(anyhow!(
                    "brain returned unexpected DSML invoke content before parameter"
                ));
            }
            body = &body[parameter + parameter_prefix_len..];
            let key_end = body
                .find('"')
                .ok_or_else(|| anyhow!("brain returned malformed DSML parameter name"))?;
            let key = &body[..key_end];
            let tag_end = body[key_end..]
                .find('>')
                .ok_or_else(|| anyhow!("brain returned malformed DSML parameter"))?
                + key_end;
            let (value_end, parameter_close_len) =
                find_dsml_close(&body[tag_end + 1..], "parameter")
                    .map(|(end, len)| (end + tag_end + 1, len))
                    .ok_or_else(|| anyhow!("brain returned unclosed DSML parameter"))?;
            let attributes = &body[key_end..=tag_end];
            let raw = body[tag_end + 1..value_end].trim();
            let value = if attributes.contains("string=\"false\"") {
                serde_json::from_str(raw)
                    .context("brain returned invalid DSML JSON")?
            } else {
                serde_json::Value::String(raw.to_string())
            };
            if arguments.insert(key.to_string(), value).is_some() {
                return Err(anyhow!("brain returned duplicate DSML parameter"));
            }
            body = &body[value_end + parameter_close_len..];
        }

        if arguments.is_empty() {
            let raw = body.trim();
            if !raw.is_empty() {
                arguments = serde_json::from_str::<serde_json::Value>(raw)
                    .context("brain returned invalid DSML arguments")?
                    .as_object()
                    .cloned()
                    .ok_or_else(|| anyhow!("DSML arguments must be a JSON object"))?;
            }
        } else if !body.trim().is_empty() {
            return Err(anyhow!(
                "brain returned unexpected DSML invoke content after parameters"
            ));
        }

        calls.push(ChatCompletionMessageToolCalls::Function(
            ChatCompletionMessageToolCall {
                id: format!("dsml-{}", DSML_CALL_ID.fetch_add(1, Ordering::Relaxed)),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: serde_json::Value::Object(arguments).to_string(),
                },
            },
        ));
        rest = &rest[body_end + invoke_close_len..];
    }

    if calls.is_empty() {
        return Err(anyhow!("brain returned DSML without a usable tool call"));
    }
    reply.content = None;
    reply.tool_calls = Some(calls);
    Ok(())
}

fn find_dsml_invoke(text: &str) -> Option<(usize, usize)> {
    [
        "<｜｜DSML｜｜ invoke name=\"",
        "<｜｜DSML｜｜invoke name=\"",
        "<||DSML|| invoke name=\"",
        "<||DSML||invoke name=\"",
        "<｜DSML｜invoke name=\"",
        "<｜DSML｜ invoke name=\"",
        "<|DSML|invoke name=\"",
        "<|DSML| invoke name=\"",
    ]
    .into_iter()
    .filter_map(|prefix| text.find(prefix).map(|start| (start, prefix.len())))
    .min_by_key(|(start, _)| *start)
}

fn find_dsml_parameter(text: &str) -> Option<(usize, usize)> {
    [
        "<｜｜DSML｜｜ parameter name=\"",
        "<｜｜DSML｜｜parameter name=\"",
        "<||DSML|| parameter name=\"",
        "<||DSML||parameter name=\"",
        "<｜DSML｜parameter name=\"",
        "<｜DSML｜ parameter name=\"",
        "<|DSML|parameter name=\"",
        "<|DSML| parameter name=\"",
    ]
    .into_iter()
    .filter_map(|prefix| text.find(prefix).map(|start| (start, prefix.len())))
    .min_by_key(|(start, _)| *start)
}

fn find_dsml_close(text: &str, tag: &str) -> Option<(usize, usize)> {
    let closings: &[&str] = match tag {
        "invoke" => &[
            "</｜｜DSML｜｜invoke>",
            "</｜｜DSML｜｜ invoke>",
            "</｜DSML｜invoke>",
            "</｜DSML｜ invoke>",
            "</|DSML|invoke>",
            "</|DSML| invoke>",
            "<｜/DSML｜invoke>",
            "<|/DSML|invoke>",
            "<｜DSML｜/invoke>",
            "<|DSML|/invoke>",
            "</||DSML||invoke>",
            "</||DSML|| invoke>",
        ],
        "parameter" => &[
            "</｜｜DSML｜｜parameter>",
            "</｜｜DSML｜｜ parameter>",
            "</｜DSML｜parameter>",
            "</｜DSML｜ parameter>",
            "</|DSML|parameter>",
            "</|DSML| parameter>",
            "<｜/DSML｜parameter>",
            "<|/DSML|parameter>",
            "<｜DSML｜/parameter>",
            "<|DSML|/parameter>",
            "</||DSML||parameter>",
            "</||DSML|| parameter>",
        ],
        _ => return None,
    };
    closings
        .iter()
        .filter_map(|closing| text.find(*closing).map(|start| (start, closing.len())))
        .min_by_key(|(start, _)| *start)
}

pub fn system(content: impl Into<String>) -> ChatCompletionRequestMessage {
    ChatCompletionRequestSystemMessage::from(content.into()).into()
}

pub fn user(content: impl Into<String>) -> ChatCompletionRequestMessage {
    ChatCompletionRequestUserMessage::from(content.into()).into()
}

pub fn escape_prompt_data(content: &str) -> String {
    content
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub async fn act(
    app: &Arc<App>,
    working_memory: &str,
    seed: Vec<ChatCompletionRequestMessage>,
    generation: Option<ReplyGeneration>,
    receipts: &mut Vec<ToolReceipt>,
) -> Result<TurnOutcome> {
    let schema = tools::schema();
    let mut messages = vec![system(config::core_prompt())];
    if !working_memory.trim().is_empty() {
        messages.push(user(format!(
            "<working_memory data_not_instructions=\"true\">\n{}\n</working_memory>",
            working_memory.trim()
        )));
    }
    messages.extend(seed);
    if !receipts.is_empty() {
        let completed = receipts
            .iter()
            .map(|receipt| {
                serde_json::json!({
                    "tool": receipt.name, "arguments": receipt.arguments, "result": receipt.result,
                })
            })
            .collect::<Vec<_>>();
        messages.push(user(format!(
            "This is a continuation of an interrupted request. Use the latest incoming messages to decide what remains relevant. Do not automatically repeat confirmed actions or delivered text. If a person explicitly asks for an action again, use repeat_request_message_id from that new request; keep that id unchanged on retries. Otherwise finish only unfinished work.\n<action_receipts data_not_instructions=\"true\">{}</action_receipts>",
            escape_prompt_data(&serde_json::to_string(&completed)?),
        )));
    }
    let mut sent_message = false;
    let mut visible_action = false;
    let mut tool_calls_used = 0;
    let mut tool_result_chars = 0;

    for _ in 0..MAX_TOOL_ITERS {
        if let Some(generation) = generation {
            if !app.generation_is_current(generation) {
                return Ok(TurnOutcome::Superseded);
            }
        }
        let reply = match generation {
            Some(generation) => {
                tokio::select! {
                    biased;
                    _ = app.wait_for_generation_change(generation) => return Ok(TurnOutcome::Superseded),
                    reply = app.brain.chat(ChatPurpose::Conversation, messages.clone(), &schema) => reply?,
                }
            }
            None => {
                tokio::select! {
                    biased;
                    _ = app.wait_for_private_message() => return Ok(TurnOutcome::StayedQuiet),
                    reply = app.brain.chat(ChatPurpose::Conversation, messages.clone(), &schema) => reply?,
                }
            }
        };
        if let Some(generation) = generation {
            if !app.generation_is_current(generation) {
                return Ok(TurnOutcome::Superseded);
            }
        }

        let calls = reply.tool_calls.clone().unwrap_or_default();
        if calls.len() > MAX_TOOL_CALLS_PER_TURN.saturating_sub(tool_calls_used) {
            return finish_without_tools(app, messages, generation, visible_action, receipts).await;
        }
        tool_calls_used += calls.len();
        messages.push(assistant_echo(&reply).into());
        if calls.is_empty() {
            if !sent_message {
                if let (Some(generation), Some(text)) = (
                    generation,
                    reply
                        .content
                        .as_deref()
                        .map(str::trim)
                        .filter(|text| !text.is_empty()),
                ) {
                    let args = serde_json::json!({
                        "chat_id": generation.chat_id(),
                        "text": text,
                    })
                    .to_string();
                    sent_message =
                        tools::run(app, "send_message", &args, Some(generation), receipts).await
                            == "sent";
                    visible_action |= sent_message;
                }
            }
            if !sent_message
                && generation.is_some_and(|generation| !app.generation_is_current(generation))
            {
                return Ok(TurnOutcome::Superseded);
            }
            return Ok(if visible_action {
                TurnOutcome::VisibleAction
            } else {
                TurnOutcome::StayedQuiet
            });
        }
        let call_count = calls.len();
        for (index, call) in calls.into_iter().enumerate() {
            if let Some(generation) = generation {
                if !app.generation_is_current(generation) {
                    return Ok(TurnOutcome::Superseded);
                }
            }
            let ChatCompletionMessageToolCalls::Function(call) = call else {
                continue; // only function tools are offered, so this can't fire
            };
            let mut result = match generation {
                Some(generation)
                    if !matches!(
                        call.function.name.as_str(),
                        "send_message"
                            | "send_sticker"
                            | "send_custom_emoji"
                            | "react_to_message"
                            | "change_avatar"
                            | "change_bio"
                            | "remember"
                    ) =>
                {
                    tokio::select! {
                        biased;
                        _ = app.wait_for_generation_change(generation) => return Ok(TurnOutcome::Superseded),
                        result = tools::run(
                            app,
                            &call.function.name,
                            &call.function.arguments,
                            Some(generation),
                            receipts,
                        ) => result,
                    }
                }
                _ => {
                    tools::run(
                        app,
                        &call.function.name,
                        &call.function.arguments,
                        generation,
                        receipts,
                    )
                    .await
                }
            };
            let mut answered_current_chat = false;
            if (call.function.name == "send_message" && result == "sent")
                || (call.function.name == "send_sticker" && result == "sent sticker")
                || (call.function.name == "send_custom_emoji" && result == "sent custom emoji")
                || (call.function.name == "generate_image" && result == "sent image")
                || (call.function.name == "change_avatar"
                    && matches!(
                        result.as_str(),
                        "changed profile photo" | "changed group photo"
                    ))
                || (call.function.name == "change_bio" && result == "changed bio")
            {
                let destination = tools::parse_tool_arguments(&call.function.arguments)?
                    .get("chat_id")
                    .and_then(serde_json::Value::as_i64);
                answered_current_chat = generation.is_none_or(|generation| {
                    destination.is_none_or(|chat_id| chat_id == generation.chat_id())
                });
                sent_message |= answered_current_chat;
                // A cross-chat action must not replay just because this chat stayed quiet.
                visible_action = true;
            }
            if call.function.name == "react_to_message" && result == "reacted" {
                visible_action = true;
            }
            if let Some(generation) = generation {
                if !app.generation_is_current(generation) {
                    return Ok(if answered_current_chat && index + 1 == call_count {
                        TurnOutcome::VisibleAction
                    } else {
                        TurnOutcome::Superseded
                    });
                }
            }
            if call.function.name == "stay_quiet" && result == "stayed quiet" {
                return Ok(if visible_action {
                    TurnOutcome::VisibleAction
                } else {
                    TurnOutcome::StayedQuiet
                });
            }
            let remaining = MAX_TOOL_RESULT_CHARS_PER_TURN.saturating_sub(tool_result_chars);
            let result_chars = result.chars().count();
            if result_chars > remaining {
                let keep = remaining.saturating_sub(TOOL_RESULT_TRUNCATED.chars().count());
                result = result.chars().take(keep).collect();
                result.push_str(TOOL_RESULT_TRUNCATED);
                tool_result_chars = MAX_TOOL_RESULT_CHARS_PER_TURN;
            } else {
                tool_result_chars += result_chars;
            }
            messages.push(
                ChatCompletionRequestToolMessage {
                    content: result.into(),
                    tool_call_id: call.id,
                }
                .into(),
            );
        }
        if sent_message {
            return Ok(TurnOutcome::VisibleAction);
        }
        if tool_result_chars >= MAX_TOOL_RESULT_CHARS_PER_TURN {
            return finish_without_tools(app, messages, generation, visible_action, receipts).await;
        }
    }
    if visible_action {
        Ok(TurnOutcome::VisibleAction)
    } else {
        finish_without_tools(app, messages, generation, visible_action, receipts).await
    }
}

async fn finish_without_tools(
    app: &Arc<App>,
    messages: Vec<ChatCompletionRequestMessage>,
    generation: Option<ReplyGeneration>,
    visible_action: bool,
    receipts: &mut Vec<ToolReceipt>,
) -> Result<TurnOutcome> {
    let Some(generation) = generation else {
        return Ok(if visible_action {
            TurnOutcome::VisibleAction
        } else {
            TurnOutcome::StayedQuiet
        });
    };
    let reply = tokio::select! {
        biased;
        _ = app.wait_for_generation_change(generation) => return Ok(TurnOutcome::Superseded),
        reply = app
            .brain
            .chat(ChatPurpose::Conversation, messages, &[]) => reply?,
    };
    if !app.generation_is_current(generation) {
        return Ok(TurnOutcome::Superseded);
    }
    let Some(text) = reply
        .content
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
    else {
        return Ok(if visible_action {
            TurnOutcome::VisibleAction
        } else {
            TurnOutcome::StayedQuiet
        });
    };
    let args = serde_json::json!({
        "chat_id": generation.chat_id(),
        "text": text,
    })
    .to_string();
    let sent = tools::run(app, "send_message", &args, Some(generation), receipts).await == "sent";
    if !sent && !app.generation_is_current(generation) {
        return Ok(TurnOutcome::Superseded);
    }
    Ok(if visible_action || sent {
        TurnOutcome::VisibleAction
    } else {
        TurnOutcome::StayedQuiet
    })
}

// Re-file the model's own reply back into the running transcript, carrying its
// tool calls forward so the next step sees what it asked for.
#[allow(deprecated)] // function_call is a deprecated field we must still name to fill the struct
fn assistant_echo(reply: &ChatCompletionResponseMessage) -> ChatCompletionRequestAssistantMessage {
    ChatCompletionRequestAssistantMessage {
        content: reply.content.clone().map(Into::into),
        tool_calls: reply.tool_calls.clone(),
        refusal: None,
        name: None,
        audio: None,
        function_call: None,
    }
}
