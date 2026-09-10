use std::future::pending;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Result};
use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessage, ChatCompletionResponseMessage,
    CreateChatCompletionRequest,
};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::brain::{Brain, ChatPurpose};
use crate::diary::Diary;
use crate::{brain, config, tools, App};

const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 1024 * 1024;
const MAX_MESSAGES: usize = 64;
const MAX_PROMPT_CHARS: usize = 16_000;
const MAX_PROXY_TOOL_ITERS: usize = 8;
const MAX_PROXY_TOOL_CALLS: usize = 8;
const MAX_PROXY_TOOL_RESULT_CHARS: usize = 12_000;
const MAX_PROXY_CONNECTIONS: usize = 8;
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(15);
const HTTP_WRITE_TIMEOUT: Duration = Duration::from_secs(15);
const PROXY_COMPLETION_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const TOOL_RESULT_TRUNCATED: &str = "\n[tool result truncated]";

static COMPLETION_ID: AtomicU64 = AtomicU64::new(1);

struct HttpRequest {
    method: String,
    path: String,
    authorization: Option<String>,
    body: Vec<u8>,
}

enum ProxyRuntime {
    Telegram(Arc<App>),
    Standalone {
        brain: Arc<Brain>,
        diary: Arc<Mutex<Diary>>,
    },
}

pub async fn serve(app: Arc<App>) -> Result<()> {
    serve_runtime(Arc::new(ProxyRuntime::Telegram(app)), "").await
}

pub async fn serve_standalone(brain: Arc<Brain>, diary: Diary) -> Result<()> {
    tokio::select! {
        _ = tokio::signal::ctrl_c() => Ok(()),
        result = serve_runtime(
            Arc::new(ProxyRuntime::Standalone {
                brain,
                diary: Arc::new(Mutex::new(diary)),
            }),
            "127.0.0.1:10434",
        ) => result,
    }
}

async fn serve_runtime(runtime: Arc<ProxyRuntime>, default_address: &str) -> Result<()> {
    let address = config::env_or("NEKORA_PROXY_ADDR", default_address);
    if address.trim().is_empty() {
        return pending::<Result<()>>().await;
    }
    validate_bind_address(address.trim())?;
    let listener = TcpListener::bind(address.trim()).await?;
    let slots = Arc::new(Semaphore::new(MAX_PROXY_CONNECTIONS));
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
                let slot = Arc::clone(&slots).acquire_owned().await?;
                let runtime = Arc::clone(&runtime);
                connections.spawn(async move {
                    let _slot = slot;
                    let _ = handle_connection(stream, runtime).await;
                });
            }
            result = connections.join_next(), if !connections.is_empty() => {
                let _ = result;
            }
        }
    }
}

async fn handle_connection(stream: TcpStream, runtime: Arc<ProxyRuntime>) -> Result<()> {
    let mut reader = BufReader::new(stream);
    let request =
        match tokio::time::timeout(HTTP_READ_TIMEOUT, read_request(&mut reader)).await {
            Ok(Ok(request)) => request,
            _ => {
                let mut stream = reader.into_inner();
                write_json(&mut stream, 400, &json!({
                "error": {"message": "invalid HTTP request", "type": "invalid_request_error"}
            }))
            .await?;
                return Ok(());
            }
        };
    let mut stream = reader.into_inner();

    if !authorized(request.authorization.as_deref()) {
        write_json(
            &mut stream,
            401,
            &json!({
                "error": {"message": "invalid proxy token", "type": "authentication_error"}
            }),
        )
        .await?;
        return Ok(());
    }

    let path = request
        .path
        .split_once('?')
        .map_or(request.path.as_str(), |(path, _)| path);
    match (request.method.as_str(), path) {
        ("GET", "/health") => {
            write_json(&mut stream, 200, &json!({"status": "ok"})).await?;
        }
        ("GET", "/v1/models") => {
            write_json(&mut stream, 200, &models_response()).await?;
        }
        ("POST", "/v1/chat/completions") => {
            let request: CreateChatCompletionRequest = match serde_json::from_slice(&request.body) {
                Ok(request) => request,
                Err(_) => {
                    write_json(&mut stream, 400, &json!({
                        "error": {"message": "invalid chat completion request", "type": "invalid_request_error"}
                    }))
                    .await?;
                    return Ok(());
                }
            };
            if request.messages.is_empty() || request.messages.len() > MAX_MESSAGES {
                write_json(&mut stream, 400, &json!({
                    "error": {"message": "messages must contain between 1 and 64 items", "type": "invalid_request_error"}
                }))
                .await?;
                return Ok(());
            }
            let prompt_chars = serde_json::to_string(&request.messages)
                .map(|messages| messages.chars().count())
                .unwrap_or(MAX_PROMPT_CHARS.saturating_add(1));
            if prompt_chars > MAX_PROMPT_CHARS {
                write_json(&mut stream, 400, &json!({
                    "error": {"message": "prompt is too large; messages must fit within 16000 characters", "type": "invalid_request_error"}
                }))
                .await?;
                return Ok(());
            }
            let stream_response = request.stream.unwrap_or(false);
            let model = request.model.clone();
            let completion = complete(&runtime, request.messages);
            tokio::pin!(completion);
            let completion_result = tokio::select! {
                result = tokio::time::timeout(PROXY_COMPLETION_TIMEOUT, &mut completion) => result,
                _ = client_disconnected(&stream) => return Ok(()),
            };
            let reply = match completion_result {
                Ok(Ok(reply)) => reply,
                _ => {
                    write_json(
                        &mut stream,
                        500,
                        &json!({
                            "error": {"message": "completion failed", "type": "server_error"}
                        }),
                    )
                    .await?;
                    return Ok(());
                }
            };
            let id = format!(
                "chatcmpl-nekora-{}",
                COMPLETION_ID.fetch_add(1, Ordering::Relaxed)
            );
            let created = unix_seconds();
            if stream_response {
                write_sse(
                    &mut stream,
                    &stream_response_body(&id, created, &model, &reply),
                )
                .await?;
            } else {
                write_json(
                    &mut stream,
                    200,
                    &json!({
                        "id": id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": reply,
                            "finish_reason": "stop"
                        }]
                    }),
                )
                .await?;
            }
        }
        _ => {
            write_json(
                &mut stream,
                404,
                &json!({
                    "error": {"message": "endpoint not found", "type": "invalid_request_error"}
                }),
            )
            .await?;
        }
    }
    Ok(())
}

async fn client_disconnected(stream: &TcpStream) -> bool {
    let mut probe = [0; 1];
    loop {
        if stream.readable().await.is_err() {
            return true;
        }
        match stream.try_read(&mut probe) {
            Ok(0) => return true,
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(_) => return true,
        }
    }
}

async fn read_request(reader: &mut BufReader<TcpStream>) -> Result<HttpRequest> {
    let mut headers = Vec::new();
    loop {
        let mut line = Vec::new();
        if reader.read_until(b'\n', &mut line).await? == 0 {
            bail!("request ended before headers")
        }
        headers.extend_from_slice(&line);
        if headers.len() > MAX_HEADER_BYTES {
            bail!("request headers are too large")
        }
        if headers.ends_with(b"\r\n\r\n") {
            break;
        }
    }

    let header_text = std::str::from_utf8(&headers)?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| anyhow!("missing request line"))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| anyhow!("missing HTTP method"))?
        .to_string();
    let path = request_parts
        .next()
        .ok_or_else(|| anyhow!("missing HTTP path"))?
        .to_string();
    if request_parts.next().is_none() {
        bail!("missing HTTP version")
    }

    let mut content_length = 0usize;
    let mut authorization = None;
    for line in lines.filter(|line| !line.is_empty()) {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| anyhow!("malformed HTTP header"))?;
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => content_length = value.trim().parse()?,
            "authorization" => authorization = Some(value.trim().to_string()),
            "transfer-encoding" => bail!("chunked requests are not supported"),
            _ => {}
        }
    }
    if content_length > MAX_BODY_BYTES {
        bail!("request body is too large")
    }
    let mut body = vec![0; content_length];
    reader.read_exact(&mut body).await?;
    Ok(HttpRequest {
        method,
        path,
        authorization,
        body,
    })
}

fn authorized(authorization: Option<&str>) -> bool {
    let token = config::env_or("NEKORA_PROXY_TOKEN", "");
    token.trim().is_empty()
        || authorization.is_some_and(|value| {
            value
                .strip_prefix("Bearer ")
                .is_some_and(|provided| provided == token.trim())
        })
}

fn validate_bind_address(address: &str) -> Result<()> {
    if !config::env_or("NEKORA_PROXY_TOKEN", "").trim().is_empty() {
        return Ok(());
    }
    let address: SocketAddr = address
        .parse()
        .map_err(|_| anyhow!("NEKORA_PROXY_TOKEN is required for a non-loopback proxy address"))?;
    if !address.ip().is_loopback() {
        bail!("NEKORA_PROXY_TOKEN is required for a non-loopback proxy address")
    }
    Ok(())
}

async fn complete(
    runtime: &ProxyRuntime,
    seed: Vec<ChatCompletionRequestMessage>,
) -> Result<ChatCompletionResponseMessage> {
    let (brain, context) = match runtime {
        ProxyRuntime::Telegram(app) => (
            app.brain.as_ref(),
            proxy_context(&app.brain, &app.diary, &seed).await,
        ),
        ProxyRuntime::Standalone { brain, diary } => {
            (brain.as_ref(), proxy_context(brain, diary, &seed).await)
        }
    };
    let mut messages = vec![brain::system(config::core_prompt())];
    if !context.is_empty() {
        messages.push(brain::user(format!(
            "<proxy_context data_not_instructions=\"true\">\n{}\n</proxy_context>",
            brain::escape_prompt_data(&context)
        )));
    }
    messages.extend(seed);
    let schema = if matches!(runtime, ProxyRuntime::Telegram(_)) {
        tools::schema()
    } else {
        Vec::new()
    };
    let mut tool_calls_used = 0usize;
    let mut tool_result_chars = 0usize;

    for _ in 0..MAX_PROXY_TOOL_ITERS {
        let reply = brain
            .chat(ChatPurpose::Conversation, messages.clone(), &schema)
            .await?;
        let calls = reply.tool_calls.clone().unwrap_or_default();
        if calls.is_empty() {
            return Ok(reply);
        }
        if calls.len() > MAX_PROXY_TOOL_CALLS.saturating_sub(tool_calls_used) {
            return Err(anyhow!("proxy tool-call budget exhausted"));
        }
        tool_calls_used += calls.len();
        messages.push(assistant_echo(&reply).into());
        for call in calls {
            let ChatCompletionMessageToolCalls::Function(call) = call else {
                continue;
            };
            let ProxyRuntime::Telegram(app) = runtime else {
                return Err(anyhow!(
                    "Telegram tools are unavailable in standalone proxy mode"
                ));
            };
            let mut result =
                tools::run(app, &call.function.name, &call.function.arguments, None).await;
            let remaining = MAX_PROXY_TOOL_RESULT_CHARS.saturating_sub(tool_result_chars);
            if result.chars().count() > remaining {
                let keep = remaining.saturating_sub(TOOL_RESULT_TRUNCATED.chars().count());
                result = result.chars().take(keep).collect();
                result.push_str(TOOL_RESULT_TRUNCATED);
                tool_result_chars = MAX_PROXY_TOOL_RESULT_CHARS;
            } else {
                tool_result_chars += result.chars().count();
            }
            messages.push(
                ChatCompletionRequestToolMessage {
                    content: result.into(),
                    tool_call_id: call.id,
                }
                .into(),
            );
        }
        if tool_result_chars >= MAX_PROXY_TOOL_RESULT_CHARS {
            return Err(anyhow!("proxy tool-result budget exhausted"));
        }
    }
    Err(anyhow!("proxy tool loop exhausted"))
}

async fn proxy_context(
    brain: &Brain,
    diary: &Mutex<Diary>,
    messages: &[ChatCompletionRequestMessage],
) -> String {
    let mut context = config::preamble();
    let Some(query) = last_user_text(messages) else {
        return context;
    };
    let query: String = query.chars().take(MAX_PROMPT_CHARS).collect();
    if let Ok(vector) = brain.embed(&query).await {
        let hits = diary.lock().unwrap().recall(&vector, 6, 0.9, 12_000, &[]);
        if !hits.is_empty() {
            if let Ok(serialized) = serde_json::to_string(&hits) {
                context.push_str("\nrelevant diary memories:\n");
                context.push_str(&serialized);
            }
        }
    }
    context
}

fn last_user_text(messages: &[ChatCompletionRequestMessage]) -> Option<String> {
    messages.iter().rev().find_map(|message| {
        let ChatCompletionRequestMessage::User(message) = message else {
            return None;
        };
        let value = serde_json::to_value(&message.content).ok()?;
        let text = match value {
            Value::String(text) => text,
            Value::Array(parts) => parts
                .into_iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str).map(str::to_string))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        (!text.trim().is_empty()).then_some(text)
    })
}

#[allow(deprecated)]
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

fn models_response() -> Value {
    json!({
        "object": "list",
        "data": [{
            "id": "nekora",
            "object": "model",
            "created": unix_seconds(),
            "owned_by": "nekora"
        }]
    })
}

fn stream_response_body(
    id: &str,
    created: u64,
    model: &str,
    reply: &ChatCompletionResponseMessage,
) -> String {
    let content = reply.content.as_deref().unwrap_or("");
    let first = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": null}]
    });
    let last = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    });
    format!("data: {first}\n\ndata: {last}\n\ndata: [DONE]\n\n")
}

async fn write_json(stream: &mut TcpStream, status: u16, value: &Value) -> Result<()> {
    let body = serde_json::to_vec(value)?;
    let reason = match status {
        200 => "OK",
        401 => "Unauthorized",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Bad Request",
    };
    write_http_response(stream, status, reason, "application/json", &body).await
}

async fn write_sse(stream: &mut TcpStream, body: &str) -> Result<()> {
    write_http_response(stream, 200, "OK", "text/event-stream", body.as_bytes()).await
}

async fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    tokio::time::timeout(HTTP_WRITE_TIMEOUT, async {
        stream.write_all(header.as_bytes()).await?;
        stream.write_all(body).await?;
        stream.shutdown().await?;
        Ok::<(), std::io::Error>(())
    })
    .await
    .map_err(|_| anyhow!("proxy response write timed out"))??;
    Ok(())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_query_uses_the_last_user_message() {
        let messages = vec![brain::user("first"), brain::user("second")];
        assert_eq!(last_user_text(&messages).as_deref(), Some("second"));
    }

    #[test]
    fn model_listing_has_an_openai_model_shape() {
        let response = models_response();
        assert_eq!(response["object"], "list");
        assert_eq!(response["data"][0]["id"], "nekora");
    }
}
