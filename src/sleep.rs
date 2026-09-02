//! Sleep and reflection: how a day of talking turns into lasting memory.
//!
//! Sleep (consolidation) is the context dump — once the working buffer grows past
//! a token budget she distils it into a few durable first-person notes and starts
//! again on a clean context, the way a night's sleep keeps what mattered and drops
//! the chatter. Reflection is the awake counterpart: she drifts to a random old
//! page, lets it meet what's recently on her mind, and sometimes forms a new
//! thought worth keeping. Neither embeds or stores directly; both hand finished
//! text to the diary, which dedups and persists.

use std::sync::Arc;

use anyhow::Result;

use crate::brain::{system, user};
use crate::{config, App};

// Past this many tokens of working context she dumps to the diary and resets. Big
// enough to hold a real conversation, small enough to stay well under a model's
// window.
const CONTEXT_DUMP_TRIGGER: usize = 40_000;
// English averages ~4 chars/token. This only decides *when* to sleep; nothing is
// billed against it, so the crude estimate is fine.
const CHARS_PER_TOKEN: usize = 4;
// A distilled line shorter than this is a stray fragment, not a memory.
const MIN_MEMORY_CHARS: usize = 8;
// A reflection is softer than a stated fact, so it is filed less confidently.
const REFLECTION_CONFIDENCE: f32 = 0.6;
const MEMORY_CONFIDENCE: f32 = 0.7;

const DISTIL_INSTRUCTION: &str =
    "Below is what happened while you were awake (lines are timestamped). Write the \
few things worth keeping as lasting memories -- one per line, first person, in \
your own voice. Keep real dates and times when they matter. Skip small talk and \
anything you will not care about next week. No numbering.";

/// Sleep: if the buffer is heavy, distil it into the diary and return it emptied.
///
/// Under the trigger the buffer is left alone unless `force` is set for a day
/// boundary. Over it, each distilled line is embedded and remembered (the diary
/// dedups), and a clean buffer is handed back.
pub async fn consolidate(
    app: &Arc<App>,
    short_term: Vec<String>,
    force: bool,
) -> Result<Vec<String>> {
    let joined = short_term.join("\n");
    if short_term.is_empty() || (!force && estimate_tokens(&joined) < CONTEXT_DUMP_TRIGGER) {
        return Ok(short_term);
    }

    let reply = app
        .brain
        .chat(
            vec![
                system(config::persona()),
                user(format!("{DISTIL_INSTRUCTION}\n\n{joined}")),
            ],
            &[],
            None,
        )
        .await?;

    for line in reply.content.unwrap_or_default().lines() {
        let line = line.trim().trim_start_matches(['-', '*', ' ']).trim();
        if line.chars().count() < MIN_MEMORY_CHARS {
            continue;
        }
        let vector = app.brain.embed(line).await?;
        app.diary
            .lock()
            .unwrap()
            .remember(line, &vector, MEMORY_CONFIDENCE)?;
    }
    Ok(Vec::new()) // woke on a clean context, the day filed away
}

/// Reflection: drift to a random diary page, meet it with the present, keep the
/// thought. Returns the reflection (also filed), or `None` when the diary is
/// still empty. `recent` is a short string of what's lately on her mind.
pub async fn reflect(app: &Arc<App>, recent: &str) -> Result<Option<String>> {
    let Some(page) = app.diary.lock().unwrap().random_page() else {
        return Ok(None);
    };

    let recent = if recent.is_empty() {
        "nothing in particular"
    } else {
        recent
    };
    let prompt = format!(
        "This is an old note from your diary:\n\n{page}\n\n\
         And this is what's been on your mind lately:\n\n{recent}\n\n\
         Reflect: in 1-3 sentences, first person, notice something that connects them \
         or something new. If nothing connects, say so in one line."
    );

    let reply = app
        .brain
        .chat(vec![system(config::persona()), user(prompt)], &[], None)
        .await?;
    let thought = reply.content.unwrap_or_default().trim().to_string();
    if thought.is_empty() {
        return Ok(None);
    }
    let vector = app.brain.embed(&thought).await?;
    app.diary
        .lock()
        .unwrap()
        .remember(&thought, &vector, REFLECTION_CONFIDENCE)?;
    Ok(Some(thought))
}

fn estimate_tokens(text: &str) -> usize {
    text.len() / CHARS_PER_TOKEN
}
