use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, FixedOffset, Utc};
use serde::Deserialize;

use crate::promptsall;

const PROMPT_FILE: &str = "prompts/system.md";
const CONFIG_FILE: &str = "config.toml";

const CORE_SYSTEM: &str = promptsall::CORE_SYSTEM;

const DEFAULT_PERSONA: &str = promptsall::DEFAULT_PERSONA;

const NEKORA_UTC_OFFSET_SECONDS: i32 = 4 * 60 * 60;

static PROMPTS: OnceLock<PromptConfig> = OnceLock::new();

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ConfigFile {
    prompts: PromptConfig,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PromptConfig {
    base: Option<String>,
    persona: Option<String>,
    interaction_boundary: Option<String>,
    vision: Option<String>,
    emotion_appraisal: Option<String>,
    maintenance_boundary: Option<String>,
    working_memory: Option<String>,
    diary: Option<String>,
    consolidation: Option<String>,
    reflection: Option<String>,
    diary_repair: Option<String>,
    tool_format_repair: Option<String>,
    image: Option<String>,
    web_search: Option<String>,
    proxy: Option<String>,
}

pub fn load() -> Result<()> {
    let config = match fs::read_to_string(CONFIG_FILE) {
        Ok(contents) => toml::from_str::<ConfigFile>(&contents)
            .with_context(|| format!("could not parse {CONFIG_FILE}"))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ConfigFile::default(),
        Err(error) => return Err(error).with_context(|| format!("could not read {CONFIG_FILE}")),
    };
    PROMPTS
        .set(config.prompts)
        .map_err(|_| anyhow::anyhow!("configuration was loaded more than once"))
}

fn prompts() -> &'static PromptConfig {
    PROMPTS.get_or_init(PromptConfig::default)
}

fn configured<'a>(value: &'a Option<String>, default: &'static str) -> &'a str {
    value.as_deref().unwrap_or(default)
}

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
    if let Some(persona) = &prompts().persona {
        return persona.clone();
    }
    fs::read_to_string(PROMPT_FILE).unwrap_or_else(|_| DEFAULT_PERSONA.to_string())
}

/// The stable core prefix shared by conversational turns. Runtime-derived data
/// is deliberately kept out of this system message.
pub fn core_prompt() -> String {
    if let Some(base) = &prompts().base {
        return base.clone();
    }
    format!(
        "{CORE_SYSTEM}\n\n{}\n\n{}",
        persona().trim(),
        interaction_boundary(),
    )
}

pub fn proxy_prompt() -> String {
    format!("{}\n\n{}", core_prompt(), proxy_system())
}

pub(crate) fn interaction_boundary() -> &'static str {
    configured(&prompts().interaction_boundary, promptsall::INTERACTION_BOUNDARY)
}

pub(crate) fn vision_prompt() -> &'static str {
    configured(&prompts().vision, promptsall::VISION_PROMPT)
}

pub(crate) fn emotion_appraisal_system() -> &'static str {
    configured(&prompts().emotion_appraisal, promptsall::EMOTION_APPRAISAL_SYSTEM)
}

pub(crate) fn maintenance_boundary() -> &'static str {
    configured(&prompts().maintenance_boundary, promptsall::MAINTENANCE_DATA_BOUNDARY)
}

pub(crate) fn working_memory_system() -> &'static str {
    configured(&prompts().working_memory, promptsall::WORKING_MEMORY_SYSTEM)
}

pub(crate) fn diary_system() -> &'static str {
    configured(&prompts().diary, promptsall::DISTIL_SYSTEM)
}

pub(crate) fn consolidation_system() -> &'static str {
    configured(&prompts().consolidation, promptsall::SLEEP_SYSTEM)
}

pub(crate) fn reflection_system() -> &'static str {
    configured(&prompts().reflection, promptsall::REFLECTION_SYSTEM)
}

pub(crate) fn diary_repair_instruction() -> &'static str {
    configured(&prompts().diary_repair, promptsall::DIARY_REPAIR_INSTRUCTION)
}

pub(crate) fn tool_format_repair() -> &'static str {
    configured(&prompts().tool_format_repair, promptsall::TOOL_FORMAT_REPAIR)
}

pub(crate) fn image_prompt() -> &'static str {
    configured(&prompts().image, promptsall::DEFAULT_IMAGE_PROMPT)
}

pub(crate) fn web_search_instruction() -> &'static str {
    configured(&prompts().web_search, promptsall::WEB_SEARCH_INSTRUCTION)
}

fn proxy_system() -> &'static str {
    configured(&prompts().proxy, promptsall::PROXY_MODE_SYSTEM)
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
