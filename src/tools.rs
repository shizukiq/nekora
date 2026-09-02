//! Her action set, as the function calls the turn can reach for.
//!
//! Two halves: memory tools that touch the diary (recall is a RAG lookup,
//! remember is how a thought becomes a durable note) and world tools that go out
//! through the userbot (message, list chats). Every tool returns a short string
//! the model reads on its next step; a tool that fails returns an in-character
//! "not right now" while the real error goes to the operator, so one bad call
//! narrows the turn instead of killing it.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObject};
use serde_json::{json, Value};

use crate::conversation::ReplyGeneration;
use crate::App;

// How many notes recall hands back, and the floor to bother injecting one. The
// core reports (cosine+1)/2, so 0.8 raw cosine is ~0.9 here; below it a memory is
// noise, not a memory.
const RECALL_K: usize = 6;
const RECALL_MIN_RELATEDNESS: f64 = 0.9;
// A fresh memory starts trusted; usage in the diary earns or erodes it later.
const DEFAULT_CONFIDENCE: f32 = 0.7;

/// The tools the model sees. Kept deliberately small: the fewer the tools, the
/// less she forgets she has them.
pub fn schema() -> Vec<ChatCompletionTools> {
    [
        (
            "recall_memory",
            "Search your diary for what you already know about something before you claim to remember it.",
            json!({"type": "object", "properties": {
                "query": {"type": "string", "description": "what you are trying to remember"}},
                "required": ["query"]}),
        ),
        (
            "list_memories",
            "List durable diary entries when you need to answer what you remember.",
            json!({"type": "object", "properties": {
                "limit": {"type": "integer", "minimum": 0, "maximum": 100}}}),
        ),
        (
            "remember",
            "Write a lasting note to your diary. Use for things worth keeping, not small talk.",
            json!({"type": "object", "properties": {
                "text": {"type": "string", "description": "the memory, in your own words"}},
                "required": ["text"]}),
        ),
        (
            "inspect_user",
            "Look up a Telegram user by user_id or @username and inspect their profile avatar when you feel like it.",
            json!({"type": "object", "properties": {
                "user_id": {"type": "integer", "description": "Telegram user id from the conversation"},
                "username": {"type": "string", "description": "public username, with or without @"}}}),
        ),
        (
            "inspect_message_media",
            "Look closely at a photo or sticker from a recent message using its chat_id and message_id.",
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"}, "message_id": {"type": "integer"}},
                "required": ["chat_id", "message_id"]}),
        ),
        (
            "get_current_time",
            "Ask Telegram for the current server time and return it in UTC+04:00.",
            json!({"type": "object", "properties": {}}),
        ),
        (
            "send_message",
            "Send a text message to a Telegram chat, if you actually want to say something.",
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["chat_id", "text"]}),
        ),
        (
            "list_chats",
            "See your recent Telegram chats to decide who to talk to.",
            json!({"type": "object", "properties": {}}),
        ),
        (
            "stay_quiet",
            "Choose to do nothing this time. Silence is a valid answer.",
            json!({"type": "object", "properties": {"reason": {"type": "string"}}}),
        ),
    ]
    .into_iter()
    .map(|(name, description, parameters)| {
        ChatCompletionTools::Function(ChatCompletionTool {
            function: FunctionObject {
                name: name.to_string(),
                description: Some(description.to_string()),
                parameters: Some(parameters),
                strict: None,
            },
        })
    })
    .collect()
}

/// Dispatch one tool call; always return a string for the model to read next.
pub async fn run(
    app: &Arc<App>,
    name: &str,
    args_json: &str,
    generation: Option<ReplyGeneration>,
) -> String {
    match dispatch(app, name, args_json, generation).await {
        Ok(result) => result,
        Err(error) => {
            // Never hand the raw error to the model: it names the backend and she
            // narrates her own plumbing out of character. Operator gets it on stderr.
            eprintln!("tool {name} failed: {error:#}");
            "(couldn't do that just now)".to_string()
        }
    }
}

async fn dispatch(
    app: &Arc<App>,
    name: &str,
    args_json: &str,
    generation: Option<ReplyGeneration>,
) -> Result<String> {
    let args: Value = if args_json.trim().is_empty() {
        json!({})
    } else {
        serde_json::from_str(args_json)?
    };

    match name {
        "recall_memory" => {
            let vector = app.brain.embed(str_arg(&args, "query")?).await?;
            let hits: Vec<_> = app
                .diary
                .lock()
                .unwrap()
                .recall(&vector, RECALL_K)
                .into_iter()
                .filter(|hit| hit.relatedness >= RECALL_MIN_RELATEDNESS)
                .collect();
            Ok(if hits.is_empty() {
                "nothing in the diary about that".to_string()
            } else {
                serde_json::to_string(&hits)?
            })
        }
        "list_memories" => {
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(0) as usize;
            let listing = app.diary.lock().unwrap().list_memories(limit);
            Ok(serde_json::to_string(&listing)?)
        }
        "remember" => {
            let text = str_arg(&args, "text")?;
            let vector = app.brain.embed(text).await?;
            let stored = app
                .diary
                .lock()
                .unwrap()
                .remember(text, &vector, DEFAULT_CONFIDENCE)?;
            Ok(match stored {
                None => "already knew that".to_string(),
                Some(_) => "noted".to_string(),
            })
        }
        "inspect_user" => {
            let user_id = args.get("user_id").and_then(Value::as_i64);
            let username = args.get("username").and_then(Value::as_str);
            Ok(serde_json::to_string(
                &app.userbot.inspect_user(user_id, username).await?,
            )?)
        }
        "inspect_message_media" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing chat_id"))?;
            let message_id = args
                .get("message_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing message_id"))?;
            Ok(serde_json::to_string(
                &app.userbot
                    .inspect_message_media(chat_id, message_id)
                    .await?,
            )?)
        }
        "get_current_time" => Ok(serde_json::to_string(&app.userbot.current_time().await?)?),
        "send_message" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing chat_id"))?;
            let text = str_arg(&args, "text")?;
            app.userbot.send(app, chat_id, text, generation).await?;
            if generation.is_none_or(|generation| app.generation_is_current(generation)) {
                app.record_outgoing(chat_id, text);
            }
            Ok("sent".to_string())
        }
        "list_chats" => Ok(serde_json::to_string(&app.userbot.recent_chats().await?)?),
        "stay_quiet" => Ok("stayed quiet".to_string()),
        other => Ok(format!("unknown tool: {other}")),
    }
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing {key}"))
}
