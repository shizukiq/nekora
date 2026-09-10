use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Result};
use chrono::{DateTime, FixedOffset, Utc};

use crate::promptsall;

const PROMPT_FILE: &str = "prompts/system.md";

const CORE_SYSTEM: &str = promptsall::CORE_SYSTEM;

const DEFAULT_PERSONA: &str = promptsall::DEFAULT_PERSONA;

const NEKORA_UTC_OFFSET_SECONDS: i32 = 4 * 60 * 60;

/// Read `key` from the environment, or fall back to `default`.
pub fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Load a `.env` file into the process environment without overriding anything
/// already set, so real environment variables win over the file.
pub fn load_env(path: &str) {
    let Ok(contents) = fs::read_to_string(path) else {
        return;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if env::var_os(key).is_none() {
            let value = value.trim().trim_matches(['"', '\'']);
            unsafe { env::set_var(key, value) };
        }
    }
}

pub fn nekora_name() -> String {
    env_or("NEKORA_NAME", "Nekora")
}

/// The one Telegram user permitted to discuss Nekora's implementation and
/// development. An unset value means there is no privileged developer chat.
pub fn creator_user_id() -> Result<Option<i64>> {
    let value = env_or("NEKORA_CREATOR_USER_ID", "");
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    let user_id = value.parse::<i64>().map_err(|_| {
        anyhow::anyhow!("NEKORA_CREATOR_USER_ID must be a positive Telegram user ID")
    })?;
    if user_id <= 0 {
        bail!("NEKORA_CREATOR_USER_ID must be a positive Telegram user ID");
    }
    Ok(Some(user_id))
}

pub fn vault_dir() -> PathBuf {
    PathBuf::from(env_or("NEKORA_VAULT", "vault"))
}

/// Runtime files are kept beside the diary notes, but outside the note directory
/// scan. This lets the process recover an unfinished day without turning its
/// checkpoint into a memory entry.
pub fn runtime_dir() -> PathBuf {
    vault_dir().join("runtime")
}

/// Nekora's self-description: `prompts/system.md` if an operator wrote one,
/// otherwise the default from `promptsall.rs`.
pub fn persona() -> String {
    fs::read_to_string(PROMPT_FILE).unwrap_or_else(|_| DEFAULT_PERSONA.to_string())
}

/// The stable core prefix shared by conversational turns. Runtime-derived data
/// is deliberately kept out of this system message.
pub fn core_prompt() -> String {
    format!("{CORE_SYSTEM}\n\n{}", persona().trim())
}

pub fn nekora_utc_offset() -> FixedOffset {
    FixedOffset::east_opt(NEKORA_UTC_OFFSET_SECONDS).expect("UTC+4 is a valid fixed offset")
}

pub fn nekora_time() -> DateTime<FixedOffset> {
    Utc::now().with_timezone(&nekora_utc_offset())
}

/// The factual runtime line each turn opens with. Read it fresh every turn
/// because the time is part of it.
pub fn preamble() -> String {
    format!(
        "Current date and time in Nekora's timezone: {} (GMT+4). Account name: {}. Preferred person: {}.",
        nekora_time().format("%Y-%m-%d %H:%M:%S %:z"),
        nekora_name(),
        env_or("PAPIK_NAME", "your person"),
    )
}
