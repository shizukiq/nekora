use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObject};
use serde_json::{json, Value};

use crate::conversation::ReplyGeneration;
use crate::diary::{is_valid_generated_memory, MemoryRevision};
use crate::promptsall;
use crate::App;

const RECALL_K: usize = 6;
const RECALL_MIN_RELATEDNESS: f64 = 0.9;
const MAX_RECALL_BODY_CHARS: usize = 12_000;
const DEFAULT_CONFIDENCE: f32 = 0.7;

pub fn schema() -> Vec<ChatCompletionTools> {
    [
        (
            "recall_memory",
            promptsall::TOOL_RECALL_MEMORY,
            json!({"type": "object", "properties": {
                "query": {"type": "string", "description": "a self-contained retrieval cue with names, topic, and relevant event context"}},
                "required": ["query"]}),
        ),
        (
            "web_search",
            promptsall::TOOL_WEB_SEARCH,
            json!({"type": "object", "properties": {
                "query": {"type": "string", "description": "what you want to search for"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10, "description": "maximum number of results"}},
                "required": ["query"]}),
        ),
        (
            "list_memories",
            promptsall::TOOL_LIST_MEMORIES,
            json!({"type": "object", "properties": {
                "limit": {"type": "integer", "minimum": 0, "maximum": 100}}}),
        ),
        (
            "remember",
            promptsall::TOOL_REMEMBER,
            json!({"type": "object", "properties": {
                "text": {"type": "string", "description": "a flowing first-person Russian diary page with concrete details and a final Retrieval cues line"}},
                "required": ["text"]}),
        ),
        (
            "revise_memory",
            promptsall::TOOL_REVISE_MEMORY,
            json!({"type": "object", "properties": {
                "memory_id": {"type": "string", "description": "id of the active memory to replace"},
                "text": {"type": "string", "description": "complete corrected self-contained memory"}},
                "required": ["memory_id", "text"]}),
        ),
        (
            "archive_memory",
            promptsall::TOOL_ARCHIVE_MEMORY,
            json!({"type": "object", "properties": {
                "memory_id": {"type": "string", "description": "id of the active memory to archive"}},
                "required": ["memory_id"]}),
        ),
        (
            "inspect_user",
            promptsall::TOOL_INSPECT_USER,
            json!({"type": "object", "properties": {
                "user_id": {"type": "integer", "description": "participant id from the chat context"},
                "name": {"type": "string", "description": "the display name shown in the conversation"},
                "username": {"type": "string", "description": "public username, with or without @; empty if unavailable"}},
                "required": ["user_id", "name", "username"]}),
        ),
        (
            "inspect_own_profile",
            promptsall::TOOL_INSPECT_OWN_PROFILE,
            json!({"type": "object", "properties": {
                "avatar_limit": {"type": "integer", "minimum": 1, "maximum": 4, "description": "number of recent profile photos to inspect; defaults to 1"}}}),
        ),
        (
            "list_received_gifts",
            promptsall::TOOL_LIST_RECEIVED_GIFTS,
            json!({"type": "object", "properties": {
                "offset": {"type": "string", "description": "pagination offset returned by the service; empty for the first page"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "number of gifts; defaults to 20"}}}),
        ),
        (
            "list_sticker_sets",
            promptsall::TOOL_LIST_STICKER_SETS,
            json!({"type": "object", "properties": {
                "kind": {"type": "string", "enum": ["sticker", "custom_emoji"]}},
                "required": ["kind"]}),
        ),
        (
            "list_stickers",
            promptsall::TOOL_LIST_STICKERS,
            json!({"type": "object", "properties": {
                "set_id": {"type": "integer"},
                "emoji": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "number of items; defaults to 20"}},
                "required": ["set_id"]}),
        ),
        (
            "find_custom_emojis",
            promptsall::TOOL_FIND_CUSTOM_EMOJIS,
            json!({"type": "object", "properties": {
                "emoji": {"type": "string", "description": "one ordinary emoji to find variants for"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "number of variants; defaults to 10"}},
                "required": ["emoji"]}),
        ),
        (
            "inspect_message_media",
            promptsall::TOOL_INSPECT_MESSAGE_MEDIA,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"}, "message_id": {"type": "integer"}},
                "required": ["chat_id", "message_id"]}),
        ),
        (
            "search_messages",
            promptsall::TOOL_SEARCH_MESSAGES,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer", "description": "optional chat to search; omit for a scoped global search"},
                "query": {"type": "string", "description": "text to search for"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "maximum number of matches; defaults to 20"}},
                "required": ["query"]}),
        ),
        (
            "search_chats",
            promptsall::TOOL_SEARCH_CHATS,
            json!({"type": "object", "properties": {
                "query": {"type": "string", "description": "part of a chat title or username"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "maximum number of chats; defaults to 10"}},
                "required": ["query"]}),
        ),
        (
            "view_messages_around",
            promptsall::TOOL_VIEW_MESSAGES_AROUND,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"},
                "message_id": {"type": "integer"},
                "before": {"type": "integer", "minimum": 0, "maximum": 50, "description": "messages before the target; defaults to 5"},
                "after": {"type": "integer", "minimum": 0, "maximum": 50, "description": "messages after the target; defaults to 5"}},
                "required": ["chat_id", "message_id"]}),
        ),
        (
            "edit_message",
            promptsall::TOOL_EDIT_MESSAGE,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"},
                "message_id": {"type": "integer"},
                "text": {"type": "string", "description": "the complete replacement text"}},
                "required": ["chat_id", "message_id", "text"]}),
        ),
        (
            "remove_message",
            promptsall::TOOL_REMOVE_MESSAGE,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"},
                "message_id": {"type": "integer"}},
                "required": ["chat_id", "message_id"]}),
        ),
        (
            "forward_message",
            promptsall::TOOL_FORWARD_MESSAGE,
            json!({"type": "object", "properties": {
                "source_chat_id": {"type": "integer"},
                "destination_chat_id": {"type": "integer"},
                "message_id": {"type": "integer"}},
                "required": ["source_chat_id", "destination_chat_id", "message_id"]}),
        ),
        (
            "join_chat",
            promptsall::TOOL_JOIN_CHAT,
            json!({"type": "object", "properties": {
                "username": {"type": "string", "description": "public @username without an invite link"}},
                "required": ["username"]}),
        ),
        (
            "leave_chat",
            promptsall::TOOL_LEAVE_CHAT,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"},
                "username": {"type": "string"}}}),
        ),
        (
            "ban_user",
            promptsall::TOOL_BAN_USER,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer", "description": "group chat id"},
                "user_id": {"type": "integer", "description": "positive participant id"},
                "duration_minutes": {"type": "integer", "minimum": 0, "maximum": 43200, "description": "0 for permanent, otherwise temporary duration"}},
                "required": ["chat_id", "user_id"]}),
        ),
        (
            "get_current_time",
            promptsall::TOOL_GET_CURRENT_TIME,
            json!({"type": "object", "properties": {}}),
        ),
        (
            "generate_image",
            promptsall::TOOL_GENERATE_IMAGE,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"},
                "description": {"type": "string", "description": "the scene, subject, composition, and mood to depict"},
                "caption": {"type": "string", "description": "optional short caption"},
                "reply_to_message_id": {"type": "integer", "description": "message_id to reply to; omit for a normal image message"}},
                "required": ["chat_id", "description"]}),
        ),
        (
            "change_avatar",
            promptsall::TOOL_CHANGE_AVATAR,
            json!({"type": "object", "properties": {
                "description": {"type": "string", "description": "the new avatar's subject, mood, colors, and composition"}},
                "required": ["description"]}),
        ),
        (
            "send_message",
            promptsall::TOOL_SEND_MESSAGE,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"},
                "text": {"type": "string"},
                "reply_to_message_id": {"type": "integer", "description": "message_id from chat context; omit for a normal message"}},
                "required": ["chat_id", "text"]}),
        ),
        (
            "send_sticker",
            promptsall::TOOL_SEND_STICKER,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"},
                "document_id": {"type": "integer", "description": "document_id returned by list_stickers"},
                "reply_to_message_id": {"type": "integer"}},
                "required": ["chat_id", "document_id"]}),
        ),
        (
            "send_custom_emoji",
            promptsall::TOOL_SEND_CUSTOM_EMOJI,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"},
                "document_id": {"type": "integer"},
                "emoji": {"type": "string"},
                "reply_to_message_id": {"type": "integer"}},
                "required": ["chat_id", "document_id", "emoji"]}),
        ),
        (
            "react_to_message",
            promptsall::TOOL_REACT_TO_MESSAGE,
            json!({"type": "object", "properties": {
                "chat_id": {"type": "integer"},
                "message_id": {"type": "integer"},
                "reaction": {"type": "string", "description": "one supported emoji, custom_emoji:<document_id> from context, or empty to remove your reaction"}},
                "required": ["chat_id", "message_id", "reaction"]}),
        ),
        (
            "list_chats",
            promptsall::TOOL_LIST_CHATS,
            json!({"type": "object", "properties": {}}),
        ),
        (
            "stay_quiet",
            promptsall::TOOL_STAY_QUIET,
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
            // Keep backend details out of the persona, but let a failed image
            // action produce an honest visible explanation instead of silence.
            eprintln!("tool {name} failed: {error:#}");
            if matches!(name, "generate_image" | "change_avatar") {
                image_generation_failure_for_model(name, &error)
            } else if is_invalid_tool_arguments(&error) {
                "(tool arguments were invalid JSON; retry the same action with a shorter valid JSON object)".to_string()
            } else if name == "react_to_message" && is_reaction_invalid(&error) {
                "(Telegram rejected that reaction; it is unavailable for this chat or message. Do not retry the same reaction.)".to_string()
            } else {
                "(couldn't do that just now)".to_string()
            }
        }
    }
}

fn is_reaction_invalid(error: &anyhow::Error) -> bool {
    format!("{error:#}").contains("REACTION_INVALID")
}

fn is_invalid_tool_arguments(error: &anyhow::Error) -> bool {
    format!("{error:#}").contains("tool arguments")
}

fn image_generation_failure_for_model(name: &str, error: &anyhow::Error) -> String {
    let details = format!("{error:#}").to_ascii_lowercase();
    let cause = if details.contains("returned 502") {
        "temporary upstream image-provider failure (HTTP 502)"
    } else if details.contains("returned 429") {
        "temporary image-provider rate limit (HTTP 429)"
    } else if details.contains("timed out") {
        "the image provider timed out"
    } else if details.contains("requires openrouter_api_key") || details.contains("not configured")
    {
        "image generation is not configured"
    } else {
        "the image service returned an error"
    };
    if name == "change_avatar" {
        format!(
            "(avatar generation failed; the profile photo was not changed. Cause: {cause}. Do not claim that the avatar changed, and do not call change_avatar again in this turn; mention briefly that it failed and can be retried later.)"
        )
    } else {
        format!(
            "(image generation failed; no image was sent. Cause: {cause}. Do not claim that an image was sent, and do not call generate_image again in this turn; tell the person briefly that image generation failed and they can try again.)"
        )
    }
}

async fn dispatch(
    app: &Arc<App>,
    name: &str,
    args_json: &str,
    generation: Option<ReplyGeneration>,
) -> Result<String> {
    let args = parse_tool_arguments(args_json)?;

    match name {
        "recall_memory" => {
            let vector = app.brain.embed(str_arg(&args, "query")?).await?;
            let hits: Vec<_> = app.diary.lock().unwrap().recall(
                &vector,
                RECALL_K,
                RECALL_MIN_RELATEDNESS,
                MAX_RECALL_BODY_CHARS,
                &[],
            );
            Ok(if hits.is_empty() {
                "nothing in the diary about that".to_string()
            } else {
                serde_json::to_string(&hits)?
            })
        }
        "web_search" => {
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(5) as usize;
            if !(1..=10).contains(&limit) {
                return Err(anyhow!("limit must be between 1 and 10"));
            }
            let query = str_arg(&args, "query")?;
            let results = app.web_search.search(query, limit).await?;
            let results = serde_json::to_string(&results)?;
            let social = app.assess_search_results(&results).await;
            Ok(format!("{results}\n\n{social}"))
        }
        "list_memories" => {
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(0) as usize;
            let listing = app.diary.lock().unwrap().list_memories(limit);
            Ok(serde_json::to_string(&listing)?)
        }
        "remember" => {
            let text = str_arg(&args, "text")?;
            if !is_valid_generated_memory(text) {
                return Err(anyhow!(
                    "memory must be diary prose with a complete final Retrieval cues paragraph"
                ));
            }
            let vector = app.brain.embed(text).await?;
            if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
                return Ok("turn became outdated before the memory was stored".to_string());
            }
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
        "revise_memory" => {
            let memory_id = str_arg(&args, "memory_id")?;
            let text = str_arg(&args, "text")?;
            if !is_valid_generated_memory(text) {
                return Err(anyhow!(
                    "replacement memory must be diary prose with a complete final Retrieval cues paragraph"
                ));
            }
            let vector = app.brain.embed(text).await?;
            if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
                return Ok("turn became outdated before the memory was revised".to_string());
            }
            let revision =
                app.diary
                    .lock()
                    .unwrap()
                    .revise(memory_id, text, &vector, DEFAULT_CONFIDENCE)?;
            Ok(match revision {
                MemoryRevision::Replaced(id) => {
                    format!("revised as {id}; previous memory removed")
                }
                MemoryRevision::AlreadyKnown => {
                    "correction already existed; previous memory removed".to_string()
                }
                MemoryRevision::Unchanged => "memory already says that".to_string(),
                MemoryRevision::NotEditable => {
                    "memory was not active or is an immutable anchor".to_string()
                }
            })
        }
        "archive_memory" => {
            let memory_id = str_arg(&args, "memory_id")?.to_string();
            if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
                return Ok("turn became outdated before the memory was removed".to_string());
            }
            let retired = app
                .diary
                .lock()
                .unwrap()
                .retire(std::slice::from_ref(&memory_id))?;
            Ok(if retired == 1 {
                "removed".to_string()
            } else {
                "memory was not active or is an immutable anchor".to_string()
            })
        }
        "inspect_user" => {
            let user_id = args
                .get("user_id")
                .and_then(Value::as_i64)
                .filter(|user_id| *user_id != 0);
            let name = str_arg(&args, "name")?;
            let username = str_arg(&args, "username")?;
            let username = (!username.trim().is_empty()).then_some(username);
            Ok(serde_json::to_string(
                &app.userbot.inspect_user(user_id, name, username).await?,
            )?)
        }
        "inspect_own_profile" => {
            let avatar_limit = args
                .get("avatar_limit")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize;
            Ok(serde_json::to_string(
                &app.userbot.inspect_own_profile(avatar_limit).await?,
            )?)
        }
        "list_received_gifts" => {
            let offset = args.get("offset").and_then(Value::as_str).unwrap_or("");
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(20) as usize;
            Ok(serde_json::to_string(
                &app.userbot.received_gifts(offset, limit).await?,
            )?)
        }
        "list_sticker_sets" => {
            let kind = str_arg(&args, "kind")?;
            let custom_emoji = match kind {
                "sticker" => false,
                "custom_emoji" => true,
                _ => return Err(anyhow!("kind must be sticker or custom_emoji")),
            };
            Ok(serde_json::to_string(
                &app.userbot.sticker_sets(custom_emoji).await?,
            )?)
        }
        "list_stickers" => {
            let set_id = args
                .get("set_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing set_id"))?;
            let emoji = args
                .get("emoji")
                .and_then(Value::as_str)
                .filter(|emoji| !emoji.trim().is_empty());
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(20) as usize;
            Ok(serde_json::to_string(
                &app.userbot.stickers_in_set(set_id, emoji, limit).await?,
            )?)
        }
        "find_custom_emojis" => {
            let emoji = str_arg(&args, "emoji")?;
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(10) as usize;
            Ok(serde_json::to_string(
                &app.userbot.find_custom_emojis(emoji, limit).await?,
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
        "search_messages" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .or_else(|| generation.map(|generation| generation.chat_id()));
            let query = str_arg(&args, "query")?;
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(20) as usize;
            Ok(serde_json::to_string(
                &app.userbot.search_messages(chat_id, query, limit).await?,
            )?)
        }
        "search_chats" => {
            let query = str_arg(&args, "query")?;
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(10) as usize;
            Ok(serde_json::to_string(
                &app.userbot.search_chats(query, limit).await?,
            )?)
        }
        "view_messages_around" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing chat_id"))?;
            let message_id = args
                .get("message_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing message_id"))?;
            let before = args.get("before").and_then(Value::as_u64).unwrap_or(5) as usize;
            let after = args.get("after").and_then(Value::as_u64).unwrap_or(5) as usize;
            Ok(serde_json::to_string(
                &app.userbot
                    .view_messages_around(chat_id, message_id, before, after)
                    .await?,
            )?)
        }
        "edit_message" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing chat_id"))?;
            if generation.is_some_and(|generation| generation.chat_id() != chat_id) {
                return Err(anyhow!(
                    "a conversational turn can only edit its current chat"
                ));
            }
            let message_id = args
                .get("message_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing message_id"))?;
            let text = str_arg(&args, "text")?;
            app.userbot
                .edit_message(app, chat_id, message_id, text, generation)
                .await?;
            Ok("edited".to_string())
        }
        "remove_message" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing chat_id"))?;
            if generation.is_some_and(|generation| generation.chat_id() != chat_id) {
                return Err(anyhow!(
                    "a conversational turn can only remove a message in its current chat"
                ));
            }
            let message_id = args
                .get("message_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing message_id"))?;
            if app
                .userbot
                .remove_message(app, chat_id, message_id, generation)
                .await?
            {
                Ok("removed".to_string())
            } else {
                Ok("turn became outdated before the message was removed".to_string())
            }
        }
        "forward_message" => {
            let source_chat_id = args
                .get("source_chat_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing source_chat_id"))?;
            let destination_chat_id = args
                .get("destination_chat_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing destination_chat_id"))?;
            if generation.is_some_and(|generation| generation.chat_id() != destination_chat_id) {
                return Err(anyhow!(
                    "a conversational turn can only forward into its current chat"
                ));
            }
            let message_id = args
                .get("message_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing message_id"))?;
            let count = app
                .userbot
                .forward_message(
                    app,
                    source_chat_id,
                    destination_chat_id,
                    message_id,
                    generation,
                )
                .await?;
            Ok(if count == 0 {
                "turn became outdated before the message was forwarded".to_string()
            } else {
                format!("forwarded {count} message")
            })
        }
        "join_chat" => {
            let username = str_arg(&args, "username")?;
            Ok(
                match app.userbot.join_chat(app, username, generation).await? {
                    Some(chat) => serde_json::to_string(&chat)?,
                    None => "turn became outdated before the chat was joined".to_string(),
                },
            )
        }
        "leave_chat" => {
            let chat_id = args.get("chat_id").and_then(Value::as_i64);
            let username = args.get("username").and_then(Value::as_str);
            if chat_id.is_none() && username.is_none() {
                return Err(anyhow!("missing chat_id or username"));
            }
            Ok(
                if app
                    .userbot
                    .leave_chat(app, chat_id, username, generation)
                    .await?
                {
                    "left chat".to_string()
                } else {
                    "turn became outdated before the chat was left".to_string()
                },
            )
        }
        "ban_user" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing chat_id"))?;
            if generation.is_some_and(|generation| generation.chat_id() != chat_id) {
                return Err(anyhow!(
                    "a conversational turn can only moderate its current chat"
                ));
            }
            let user_id = args
                .get("user_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing user_id"))?;
            let duration_minutes = args
                .get("duration_minutes")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            if app
                .userbot
                .ban_user(app, chat_id, user_id, duration_minutes, generation)
                .await?
            {
                Ok("user banned".to_string())
            } else {
                Ok("turn became outdated before the user was banned".to_string())
            }
        }
        "get_current_time" => Ok(serde_json::to_string(&app.userbot.current_time().await?)?),
        "generate_image" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .or_else(|| generation.map(|generation| generation.chat_id()))
                .ok_or_else(|| anyhow!("missing chat_id"))?;
            if generation.is_some_and(|generation| generation.chat_id() != chat_id) {
                return Err(anyhow!(
                    "a conversational turn can only answer its current chat"
                ));
            }
            let description = str_arg(&args, "description")?;
            let caption = match args.get("caption") {
                Some(value) => value
                    .as_str()
                    .ok_or_else(|| anyhow!("caption must be a string"))?,
                None => "",
            }
            .to_string();
            let reply_to_message_id = optional_message_id(&args, "reply_to_message_id")?;
            let image = app.image_generator.generate(description).await?;
            // Generation may be cancelled when a newer message arrives. Once
            // Telegram sending begins it must finish its own generation checks
            // and record a successful send even if the calling turn is dropped.
            let app = Arc::clone(app);
            let userbot = Arc::clone(&app.userbot);
            tokio::spawn(async move {
                userbot
                    .send_image(
                        &app,
                        chat_id,
                        image,
                        &caption,
                        reply_to_message_id,
                        generation,
                    )
                    .await
            })
            .await
            .map_err(|error| anyhow!("image sender task failed: {error}"))??;
            Ok("sent image".to_string())
        }
        "change_avatar" => {
            let description = str_arg(&args, "description")?;
            let image = app.image_generator.generate(description).await?;
            if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
                return Ok("turn became outdated before the avatar was changed".to_string());
            }
            if app
                .userbot
                .change_profile_photo(app, image, generation)
                .await?
            {
                Ok("changed profile photo".to_string())
            } else {
                Ok("turn became outdated before the avatar was changed".to_string())
            }
        }
        "send_message" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .or_else(|| generation.map(|generation| generation.chat_id()))
                .ok_or_else(|| anyhow!("missing chat_id"))?;
            if generation.is_some_and(|generation| generation.chat_id() != chat_id) {
                return Err(anyhow!(
                    "a conversational turn can only answer its current chat"
                ));
            }
            let text = str_arg(&args, "text")?;
            let reply_to_message_id = optional_message_id(&args, "reply_to_message_id")?;
            app.userbot
                .send(app, chat_id, text, reply_to_message_id, generation)
                .await?;
            if generation.is_none() {
                app.social.lock().unwrap().complete_intention_for(chat_id)?;
            }
            Ok("sent".to_string())
        }
        "send_sticker" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .or_else(|| generation.map(|generation| generation.chat_id()))
                .ok_or_else(|| anyhow!("missing chat_id"))?;
            if generation.is_some_and(|generation| generation.chat_id() != chat_id) {
                return Err(anyhow!(
                    "a conversational turn can only answer its current chat"
                ));
            }
            let document_id = args
                .get("document_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing document_id"))?;
            let reply_to_message_id = optional_message_id(&args, "reply_to_message_id")?;
            app.userbot
                .send_sticker(app, chat_id, document_id, reply_to_message_id, generation)
                .await?;
            Ok("sent sticker".to_string())
        }
        "send_custom_emoji" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .or_else(|| generation.map(|generation| generation.chat_id()))
                .ok_or_else(|| anyhow!("missing chat_id"))?;
            if generation.is_some_and(|generation| generation.chat_id() != chat_id) {
                return Err(anyhow!(
                    "a conversational turn can only answer its current chat"
                ));
            }
            let document_id = args
                .get("document_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing document_id"))?;
            let emoji = str_arg(&args, "emoji")?;
            let reply_to_message_id = optional_message_id(&args, "reply_to_message_id")?;
            app.userbot
                .send_custom_emoji(
                    app,
                    chat_id,
                    document_id,
                    emoji,
                    reply_to_message_id,
                    generation,
                )
                .await?;
            Ok("sent custom emoji".to_string())
        }
        "react_to_message" => {
            let chat_id = args
                .get("chat_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing chat_id"))?;
            if generation.is_some_and(|generation| generation.chat_id() != chat_id) {
                return Err(anyhow!(
                    "a conversational turn can only answer its current chat"
                ));
            }
            let message_id = args
                .get("message_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| anyhow!("missing message_id"))?;
            let reaction = str_arg(&args, "reaction")?;
            if app
                .userbot
                .react(app, chat_id, message_id, reaction, generation)
                .await?
            {
                Ok("reacted".to_string())
            } else {
                Ok("turn became outdated before the reaction was sent".to_string())
            }
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

fn parse_tool_arguments(args_json: &str) -> Result<Value> {
    let trimmed = args_json.trim();
    if trimmed.is_empty() {
        return Ok(json!({}));
    }
    let unwrapped = trimmed
        .strip_prefix("```json")
        .and_then(|value| value.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(trimmed);
    let value: Value = serde_json::from_str(unwrapped)
        .map_err(|error| anyhow!("tool arguments are invalid JSON: {error}"))?;
    if !value.is_object() {
        return Err(anyhow!("tool arguments must be a JSON object"));
    }
    Ok(value)
}

fn optional_message_id(args: &Value, key: &str) -> Result<Option<i64>> {
    let Some(value) = args.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_i64()
        .ok_or_else(|| anyhow!("{key} must be an integer"))
        .map(Some)
}
