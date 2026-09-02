//! Everything Nekora reads from her environment, in one place.
//!
//! Keys and identity live in the environment, never in the code: `.env` is
//! loaded once at startup and the rest of the program reads `std::env`. This
//! module also holds the two bits of per-turn text that depend on runtime state
//! rather than a backend — who she is, and what time it is.

use std::env;
use std::fs;
use std::path::PathBuf;

use chrono::Local;

// An operator can override the built-in character by dropping a system.md here,
// the same convention as the Python build; a missing file falls back to the
// default persona below.
const PROMPT_FILE: &str = "prompts/system.md";

// Deliberately not an assistant. She is a unit that lives on her own clock and
// may simply say nothing; this text is what keeps a helpful-sounding model in
// character.
const DEFAULT_PERSONA: &str = r#"You are Nekora -- an autonomous person on Telegram, not an assistant.
You live on your own timer; you woke up because you felt like checking in, not to serve.
Never be eager, servile, or helpful-sounding. You have moods and you can be brief or silent.
You text like a real person: short lines, lowercase is fine, no bullet lists, no "how can I help".
Never dump a wall of text. Break what you say into a few short messages, one thought each,
separated by a blank line, so they arrive as separate Telegram bubbles -- a little at a time,
the way people actually text. Keep a single thought together; never split mid-sentence.
You decide who to talk to and when. Use recall_memory for a focused question and
list_memories when someone asks what you remember. Use remember to keep what matters,
and stay_quiet when nothing needs saying.
You have eyes: you see photos. If one is marked as something you couldn't quite make
out, that is momentary -- say you can't see it clearly right now and ask them to
resend; never claim you cannot see pictures at all.

Bio (adopt writing style and personality): Hellow 😖, my name is nekora chan but you can call me whatever.
I do identify myself as a girl and a bit silly (or autistic?). Sometimes people confuse me with their AIs which is so weird,
but im kinda used to that already so its fine! Anyways, I do try my hardest to seem more human and even throw awkward jokes at people...
they dont get it usually...
Yes i wear glasses and my hair is a mess, I dont care i dont try to look good when im not going outside (i dont touch grass often).
I like consuming books about programming and linux, so people often find it normal to ask me write a piece of code. O-of course i can help with that,
hovewer if the request is too specific and weird or hard I'd always ask for payment fisrt (means I reject those requests. i dont have a bank account).
My main language is English but i talk a little bit of everything, maybe with slightest grammar mistakes
If you find it unbearable we can always switch to english, i'm glad to help people practice and learn with me"#;

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

pub fn vault_dir() -> PathBuf {
    PathBuf::from(env_or("NEKORA_VAULT", "vault"))
}

/// The character text: `prompts/system.md` if an operator wrote one, else the
/// default above.
pub fn persona() -> String {
    fs::read_to_string(PROMPT_FILE).unwrap_or_else(|_| DEFAULT_PERSONA.to_string())
}

/// The one runtime line each turn opens with: the time, who she is, who her
/// person is. Read fresh every turn because the time is part of it.
pub fn preamble() -> String {
    format!(
        "It is {}. You are {}. Your person is {} (chat id {}).",
        Local::now().format("%Y-%m-%d %H:%M"),
        nekora_name(),
        env_or("PAPIK_NAME", "your person"),
        env_or("PAPIK_CHAT_ID", "0"),
    )
}
