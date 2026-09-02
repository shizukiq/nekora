//! Sleep and reflection: how a day of talking turns into lasting memory.
//!
//! Sleep first keeps the short-lived promises and tasks separately, then turns
//! the durable diary into fewer, better-connected notes. Source notes are
//! archived instead of deleted, so a bad consolidation can always be inspected.

use std::sync::Arc;

use anyhow::Result;

use crate::brain::{system, user};
use crate::{config, persistence, App};

// Leave room for the system prompt and the current turn inside the 32k model
// window; this is an estimate based on the buffered text, not a model setting.
const CONTEXT_DUMP_TRIGGER: usize = 20_000;
// English averages ~4 chars/token. This only decides when to sleep.
const CHARS_PER_TOKEN: usize = 4;
const MIN_MEMORY_CHARS: usize = 8;
const REFLECTION_CONFIDENCE: f32 = 0.6;
const MEMORY_CONFIDENCE: f32 = 0.7;
const SLEEP_PASSES: usize = 2;
const RELATED_MEMORIES: usize = 1;
const SLEEP_RELATEDNESS: f64 = 0.86;
const WORKING_MEMORY_FILE: &str = "working_memory.md";
const MAX_WORKING_MEMORY_CHARS: usize = 4_000;

const DISTIL_INSTRUCTION: &str =
    "Below is what happened while you were awake (lines are timestamped). Open the diary and \
keep only what will matter later. Write short, self-contained first-person memory pieces, \
50-300 words each, separated by exactly --- on its own line. Preserve timestamps, source \
events, outcomes, promises, canonical names of people/places/objects, important messages, \
topics, relationships, emotional or physical state, useful photo details, and 3-7 retrieval \
cues. Keep contradictions and uncertainty explicit. Skip small talk, repeated context, and \
transient chatter. Do not invent facts, do not copy old diary wording, and output no numbering \
or code fences.";

const WORKING_MEMORY_INSTRUCTION: &str =
    "Maintain a short-term working memory for the next one to three days. Keep unfinished \
tasks, promises made, reminders with dates, responsibilities, and important emotional or \
physical state. Preserve items from the existing memory unless they are clearly completed \
or older than three days. Include dates or last-updated times when known. Keep it concise, \
under 500 words. Output only the memory text; output EMPTY if nothing remains.";

const SLEEP_INSTRUCTION: &str =
    "You are the Sleep-time Consolidator for a retrieval-augmented diary. Each input piece \
starts with its confidence and then its text. confidence=1 is an immutable anchor: do not \
rewrite it or archive it. Pieces with confidence<1 are mutable: merge near-duplicates, split \
mixed topics, remove noise, and preserve the factual core. Normalize names when clear, keep \
events, outcomes, relationships, exact important messages, photo details, contradictions, \
and uncertainty. Never invent facts, never turn a theory into a fact, and keep 3-7 distinctive \
retrieval cues in each piece. Return only new self-contained first-person pieces, each under \
500 tokens, separated by exactly --- on its own line. Do not include IDs, confidence labels, \
explanations, numbering, or code fences. Return NO_MEMORY if no active replacement is needed.";

/// Text that should be visible on every turn, but must not compete with durable facts.
pub fn working_memory_context() -> String {
    let path = config::vault_dir().join(WORKING_MEMORY_FILE);
    let Some(body) = persistence::read_file(&path) else {
        return String::new();
    };
    let body = body.trim();
    if body.is_empty() {
        return String::new();
    }
    let body: String = body.chars().take(MAX_WORKING_MEMORY_CHARS).collect();
    format!("short-term working memory (recent tasks and state; not permanent fact):\n{body}\n")
}

/// Pull a few relevant durable memories into the turn without making the model
/// remember to call the recall tool first.
pub async fn relevant_memories_context(app: &Arc<App>, query: &str) -> String {
    if query.trim().is_empty() {
        return String::new();
    }
    let Ok(vector) = app.brain.embed(query).await else {
        return String::new();
    };
    let memories = app
        .diary
        .lock()
        .unwrap()
        .recall(&vector, 4)
        .into_iter()
        .filter(|memory| memory.relatedness >= SLEEP_RELATEDNESS)
        .collect::<Vec<_>>();
    if memories.is_empty() {
        return String::new();
    }
    let notes = memories
        .into_iter()
        .map(|memory| format!("- {}", memory.body))
        .collect::<Vec<_>>()
        .join("\n");
    format!("relevant long-term memories (use only if they actually match):\n{notes}\n")
}

/// Sleep: if the buffer is heavy, refresh short-term memory, distil the day,
/// consolidate related diary notes, and return it emptied.
pub async fn consolidate(
    app: &Arc<App>,
    short_term: Vec<String>,
    force: bool,
) -> Result<Vec<String>> {
    let joined = short_term.join("\n");
    if short_term.is_empty() || (!force && estimate_tokens(&joined) < CONTEXT_DUMP_TRIGGER) {
        return Ok(short_term);
    }

    refresh_working_memory(app, &joined).await?;

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

    for chunk in reply.content.unwrap_or_default().split("\n---\n") {
        let memory = chunk.trim().trim_start_matches(['-', '*', ' ']).trim();
        if memory.chars().count() < MIN_MEMORY_CHARS {
            continue;
        }
        let vector = app.brain.embed(memory).await?;
        app.diary
            .lock()
            .unwrap()
            .remember(memory, &vector, MEMORY_CONFIDENCE)?;
    }
    consolidate_diary(app).await?;
    Ok(Vec::new())
}

async fn refresh_working_memory(app: &Arc<App>, joined: &str) -> Result<()> {
    let path = config::vault_dir().join(WORKING_MEMORY_FILE);
    let previous = persistence::read_file(&path).unwrap_or_default();
    let prompt = format!(
        "{WORKING_MEMORY_INSTRUCTION}\n\nExisting working memory:\n{}\n\nToday's events:\n{joined}",
        previous.trim()
    );
    let reply = app
        .brain
        .chat(vec![system(config::persona()), user(prompt)], &[], None)
        .await?;
    let body = reply.content.unwrap_or_default().trim().to_string();
    if body.is_empty() {
        return Ok(());
    }
    let body = if body.eq_ignore_ascii_case("EMPTY") {
        String::new()
    } else {
        body.chars().take(MAX_WORKING_MEMORY_CHARS).collect()
    };
    persistence::write_file_atomic(&path, &body)?;
    Ok(())
}

async fn consolidate_diary(app: &Arc<App>) -> Result<()> {
    let mut excluded = Vec::new();
    for _ in 0..SLEEP_PASSES {
        let Some(target) = app.diary.lock().unwrap().sleep_target(&excluded) else {
            break;
        };
        let vector = app.brain.embed(&target.body).await?;
        let related = app.diary.lock().unwrap().sleep_related(
            &target.id,
            &vector,
            RELATED_MEMORIES,
            SLEEP_RELATEDNESS,
        );
        let mut source_ids = vec![target.id.clone()];
        source_ids.extend(
            related
                .iter()
                .filter(|memory| memory.confidence < 1.0)
                .map(|memory| memory.id.clone()),
        );
        let mut pieces = vec![format!(
            "confidence: {}\n{}",
            target.confidence, target.body
        )];
        pieces.extend(
            related
                .into_iter()
                .map(|memory| format!("confidence: {}\n{}", memory.confidence, memory.body)),
        );
        let reply = app
            .brain
            .chat(
                vec![
                    system(config::persona()),
                    user(format!("{SLEEP_INSTRUCTION}\n\n{}", pieces.join("\n---\n"))),
                ],
                &[],
                None,
            )
            .await?;

        let mut saved = false;
        for chunk in reply.content.unwrap_or_default().split("\n---\n") {
            let memory = chunk.trim().trim_start_matches(['-', '*', ' ']).trim();
            if memory.eq_ignore_ascii_case("NO_MEMORY") || memory.chars().count() < MIN_MEMORY_CHARS
            {
                continue;
            }
            let vector = app.brain.embed(memory).await?;
            if app
                .diary
                .lock()
                .unwrap()
                .remember(memory, &vector, target.confidence)?
                .is_some()
            {
                saved = true;
            }
        }
        if saved {
            app.diary.lock().unwrap().retire(&source_ids)?;
        }
        excluded.push(target.id);
    }
    Ok(())
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
