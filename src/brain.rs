//! Every request Nekora makes, and the agentic turn that strings them together.
//!
//! The split pool from the Python build, unchanged: the main brain is DeepSeek
//! over its OpenAI-compatible `/v1`, while vision (qwen2.5vl) and the bge-m3
//! embedder stay LOCAL on Ollama — free, offline, and the vault's vectors must
//! never change embedder once written. The core already decided *whether* to
//! act; [`act`] decides *what*, letting the model reach for tools until it has
//! nothing left to do.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestToolMessage, ChatCompletionRequestUserMessage,
    ChatCompletionResponseMessage, ChatCompletionToolChoiceOption, ChatCompletionTools,
    CreateChatCompletionRequestArgs,
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
use crate::conversation::ReplyGeneration;
use crate::{tools, App};

// The brain is on DeepSeek; vision and embeddings are local. bge-m3 is fixed for
// the life of a vault: swap it and the stored vectors stop comparing.
const DEFAULT_MAIN_API_BASE: &str = "https://api.deepseek.com/v1/";
const EMBED_MODEL: &str = "bge-m3";

// Low temperature keeps her in character rather than loose.
const TEMPERATURE: f32 = 0.2;
// A turn may chain at most this many tool calls before we force it to conclude,
// the guard against a model that keeps calling tools forever.
const MAX_TOOL_ITERS: usize = 8;
// Cap the vision model's output so it can't run away reasoning instead of just
// describing the picture.
const VISION_NUM_PREDICT: i32 = 300;

// A local backend stumbles — a model still cold-loading under memory pressure, a
// connection blip — and we retry rather than surface it, or she reads the failure
// and narrates her own plumbing being down. A cold model can take tens of seconds
// to load, so the wait between tries is generous.
const RETRIES: usize = 3;
const RETRY_WAIT: Duration = Duration::from_secs(15);

/// What the local Ollama must have pulled before she can run. The brain is on
/// DeepSeek (cloud), so it is deliberately not here.
pub fn required_ollama_models(vision_model: &str) -> [String; 2] {
    [EMBED_MODEL.to_string(), vision_model.to_string()]
}

pub struct Brain {
    openai: Client<OpenAIConfig>,
    ollama: Ollama,
    main_model: String,
    pub vision_model: String,
    // Every request is capped here so a slow backend can't wedge the heartbeat.
    // DeepSeek is fast, but a local vision model can cold-load for tens of
    // seconds, so the default is generous and env-overridable.
    request_timeout: Duration,
}

impl Brain {
    pub fn from_env() -> Result<Self> {
        let openai_config = OpenAIConfig::new()
            .with_api_base(env_or("NEKORA_MAIN_API_BASE", DEFAULT_MAIN_API_BASE))
            .with_api_key(env_or("DEEPSEEK_API_KEY", ""));
        let timeout_secs: u64 = env_or("NEKORA_REQUEST_TIMEOUT", "120").parse()?;
        Ok(Self {
            openai: Client::with_config(openai_config),
            ollama: crate::ollama::client_from_host(&env_or(
                "OLLAMA_HOST",
                "http://127.0.0.1:11434",
            )),
            main_model: env_or("NEKORA_MAIN_MODEL", "deepseek-v4-flash"),
            vision_model: env_or("NEKORA_VISION_MODEL", "qwen2.5vl:3b"),
            request_timeout: Duration::from_secs(timeout_secs),
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

    /// One chat completion on the brain. Returns the assistant message, which may
    /// carry tool calls. Wrapped in the retry because a cloud endpoint blips and
    /// there is no failover.
    pub async fn chat(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: &[ChatCompletionTools],
        force_tool: Option<&str>,
    ) -> Result<ChatCompletionResponseMessage> {
        let mut builder = CreateChatCompletionRequestArgs::default();
        builder
            .model(self.main_model.as_str())
            .temperature(TEMPERATURE)
            .messages(messages);
        if !tools.is_empty() {
            builder.tools(tools.to_vec());
        }
        if let Some(name) = force_tool {
            builder.tool_choice(force_choice(name));
        }
        let mut request = builder.build()?;
        loop {
            let result = self
                .retry(
                    || async {
                        let response = self.openai.chat().create(request.clone()).await?;
                        response
                            .choices
                            .into_iter()
                            .next()
                            .map(|choice| choice.message)
                            .ok_or_else(|| anyhow!("brain returned no choices"))
                    },
                    |_| true,
                )
                .await;

            // Some reasoning backends accept tools but reject a named tool choice.
            // Keep the strict request for compatible backends, then let that one
            // turn fall back to the provider's normal automatic tool selection.
            if request.tool_choice.is_some()
                && result
                    .as_ref()
                    .err()
                    .is_some_and(is_unsupported_thinking_tool_choice)
            {
                request.tool_choice = None;
                continue;
            }
            return result;
        }
    }

    /// Describe an incoming image so the text-only turn can "see" it, on the local
    /// vision model. Retries transient load failures and an empty caption — a local
    /// model that overflows its reasoning returns nothing.
    pub async fn caption_image(&self, image_bytes: &[u8]) -> Result<String> {
        let base64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
        let caption = self
            .retry(
                || async {
                    let message = ChatMessage::user(
                        "Look at this image carefully. Describe the main visible subject first, \
                         then one or two details that are actually clear. Use plain natural \
                         wording with no preamble such as 'the image shows'. Do not guess from \
                         a blurry background; if something is unclear, say so in one short \
                         sentence."
                            .into(),
                    )
                    .with_images(vec![Image::from_base64(base64.as_str())]);
                    let request = ChatMessageRequest::new(self.vision_model.clone(), vec![message])
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

// A stumble worth retrying — a load-under-pressure, a timeout, a 5xx, a connection
// blip — as opposed to a permanent bad request or auth failure that won't fix
// itself. Matched against the whole error chain, since the useful text is often on
// the cause, not the top.
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
        "internal error",
        "unavailable",
        "network",
        "502",
        "503",
        "504",
    ]
    .iter()
    .any(|needle| haystack.contains(needle))
}

fn is_unsupported_thinking_tool_choice(error: &anyhow::Error) -> bool {
    let haystack = format!("{error:#}").to_ascii_lowercase();
    haystack.contains("thinking mode") && haystack.contains("tool_choice")
}

fn force_choice(name: &str) -> ChatCompletionToolChoiceOption {
    serde_json::from_value(serde_json::json!({
        "type": "function",
        "function": {"name": name},
    }))
    .expect("a well-formed forced tool choice")
}

pub fn system(content: impl Into<String>) -> ChatCompletionRequestMessage {
    ChatCompletionRequestSystemMessage::from(content.into()).into()
}

pub fn user(content: impl Into<String>) -> ChatCompletionRequestMessage {
    ChatCompletionRequestUserMessage::from(content.into()).into()
}

// Some prompts (a wall of "what do you remember") should not be left to the
// model's discretion; forcing the listing tool guarantees the honest answer.
fn memory_listing_requested(seed: &[ChatCompletionRequestMessage]) -> bool {
    const MARKERS: [&str; 6] = [
        "what do you remember",
        "what is in your memory",
        "что ты помнишь",
        "что помнишь",
        "что есть в твоей памяти",
        "что у тебя в памяти",
    ];
    seed.iter().any(|message| {
        let ChatCompletionRequestMessage::User(user) = message else {
            return false;
        };
        let async_openai::types::chat::ChatCompletionRequestUserMessageContent::Text(text) =
            &user.content
        else {
            return false;
        };
        let text = text.to_lowercase();
        MARKERS.iter().any(|marker| text.contains(marker))
    })
}

/// The turn: let the model reach for tools until it is done, then stop.
///
/// `seed` is what set her off — an incoming burst, or a diary page she drifted
/// to. The effects are the tool calls themselves (a sent message, a filed
/// memory); there is nothing to return.
pub async fn act(
    app: &Arc<App>,
    seed: Vec<ChatCompletionRequestMessage>,
    generation: Option<ReplyGeneration>,
) -> Result<()> {
    let schema = tools::schema();
    let mut forced_tool = memory_listing_requested(&seed).then_some("list_memories");

    let mut messages = vec![system(config::persona())];
    messages.extend(seed);
    let mut sent_message = false;
    let mut stayed_quiet = false;

    for _ in 0..MAX_TOOL_ITERS {
        let reply = app
            .brain
            .chat(messages.clone(), &schema, forced_tool)
            .await?;
        if let Some(generation) = generation {
            if !app.generation_is_current(generation) {
                return Ok(());
            }
        }
        forced_tool = None;

        let calls = reply.tool_calls.clone().unwrap_or_default();
        messages.push(assistant_echo(&reply).into());
        if calls.is_empty() {
            if !sent_message && !stayed_quiet {
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
                    let _ = tools::run(app, "send_message", &args, Some(generation)).await;
                }
            }
            break; // she sent, or chose to stay quiet
        }
        for call in calls {
            let ChatCompletionMessageToolCalls::Function(call) = call else {
                continue; // only function tools are offered, so this can't fire
            };
            let result = tools::run(
                app,
                &call.function.name,
                &call.function.arguments,
                generation,
            )
            .await;
            if call.function.name == "send_message" && result == "sent" {
                sent_message = true;
            }
            if call.function.name == "stay_quiet" && result == "stayed quiet" {
                stayed_quiet = true;
            }
            messages.push(
                ChatCompletionRequestToolMessage {
                    content: result.into(),
                    tool_call_id: call.id,
                }
                .into(),
            );
        }
    }
    Ok(())
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
