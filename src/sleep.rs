//! Sleep and reflection: how a day of talking turns into lasting memory.
//!
//! Sleep first keeps the short-lived promises and tasks separately, then turns
//! the durable diary into fewer, better-connected notes. Source notes are
//! archived instead of deleted, so a bad consolidation can always be inspected.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde_json::Value;

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
const MAX_RECALL_QUERY_CHARS: usize = 12_000;
const MAX_MEMORY_CONTEXT_CHARS: usize = 4_000;
const RAG_TIMEOUT: Duration = Duration::from_secs(8);

const DISTIL_INSTRUCTION: &str =
    "It's time to open the diary and preserve what actually happened while you were awake. \
Write shortly, but do not lose details that will matter later. The input is a notification stream, \
not a list of facts: distinguish real events from explicit tests, examples, mock data, and synthetic \
context. Never save a piece that is explicitly declared unreal or created only to test the diary. \
Avoid copying prior diary wording. Always divide the result into small, self-sufficient, semantically \
coherent pieces of 50-300 words with --- on its own line between pieces. For each piece, choose the \
natural form and include whatever is known about timestamps, source event, outcomes, canonical entities, \
important messages without changing their meaning, topics, importance and rationale, affect, relationships, \
retrieval cues, useful visual details, similarities, and contradictions or uncertainty. Do not invent facts. \
If unsure, keep the uncertainty visible and do not make a weak conclusion. Output only the pieces, with no \
code fences.";

const WORKING_MEMORY_INSTRUCTION: &str =
    "Maintain a short-term working memory for the next one to three days. Keep unfinished \
tasks, promises made, reminders with dates, responsibilities, and important emotional or \
physical state. Preserve items from the existing memory unless they are clearly completed \
or older than three days. Include dates or last-updated times when known. Keep it concise, \
under 500 words. Output only the memory text; output EMPTY if nothing remains.";

const SLEEP_INSTRUCTION: &str =
    "You are the Sleep-time Consolidator. Restructure the supplied diary pieces for reliable future \
retrieval. Each input piece begins with a JSON object containing confidence, followed by its text. \
confidence=1 is an immutable anchor: do not rewrite or archive it. Mutable pieces with confidence<1 \
may be merged, split, renamed, shortened, or dropped. Sanity-check lower-confidence claims against \
higher-confidence pieces, preserve factual cores, and make speculation or contradictions explicit. \
Never invent facts and never turn a theory into a fact. Normalize entities only when clear. Optimize \
each piece for embedding retrieval with specific wording and 3-7 short retrieval cues. Keep each result \
under 500 tokens. Choose the natural form and do not force a category, title, or voice, but include a \
kind when useful: ENTITY_DESCRIPTION, THOUGHT, EVENT, FACT, or OTHER. Include a short rationale when \
changing confidence. Never output confidence=1; use -1 only for a piece that is clearly false. Return \
the resulting self-sufficient pieces, 50-300 words when possible, separated by --- on its own line. \
Each result may begin with a JSON object containing confidence, kind, rationale, and retrieval_cues; \
the host stores confidence separately. Return NO_MEMORY if no replacement is needed. Output no code fences.";

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
    let query: String = query.chars().take(MAX_RECALL_QUERY_CHARS).collect();
    let anchors = {
        let mut diary = app.diary.lock().unwrap();
        diary.reload_if_needed();
        diary.anchors(4)
    };
    let memories = match tokio::time::timeout(RAG_TIMEOUT, app.brain.embed(&query)).await {
        Ok(Ok(vector)) => app
            .diary
            .lock()
            .unwrap()
            .recall(&vector, 4)
            .into_iter()
            .filter(|memory| memory.relatedness >= SLEEP_RELATEDNESS)
            .collect::<Vec<_>>(),
        Ok(Err(_)) | Err(_) => Vec::new(),
    };
    if memories.is_empty() && anchors.is_empty() {
        return String::new();
    }
    let anchor_notes = anchors
        .iter()
        .map(|memory| {
            let body: String = memory.body.chars().take(MAX_MEMORY_CONTEXT_CHARS).collect();
            format!("- {body}")
        })
        .collect::<Vec<_>>();
    let recalled_notes = memories
        .into_iter()
        .filter(|memory| !anchors.iter().any(|anchor| anchor.id == memory.id))
        .map(|memory| {
            let body: String = memory.body.chars().take(MAX_MEMORY_CONTEXT_CHARS).collect();
            format!("- {body}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut context = String::new();
    if !anchor_notes.is_empty() {
        context.push_str(
            "canonical diary notes (treat as known unless contradicted by newer context):\n",
        );
        context.push_str(&anchor_notes.join("\n"));
        context.push('\n');
    }
    if !recalled_notes.is_empty() {
        context.push_str("relevant long-term memories (use only if they actually match):\n");
        context.push_str(&recalled_notes);
        context.push('\n');
    }
    context
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
        let Some((memory, confidence)) = memory_piece(chunk, MEMORY_CONFIDENCE) else {
            continue;
        };
        let vector = app.brain.embed(&memory).await?;
        app.diary
            .lock()
            .unwrap()
            .remember(&memory, &vector, confidence)?;
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
            "{{\"confidence\":{}}}\n{}",
            target.confidence, target.body
        )];
        pieces.extend(
            related
                .into_iter()
                .map(|memory| format!("{{\"confidence\":{}}}\n{}", memory.confidence, memory.body)),
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
            let Some((memory, confidence)) = memory_piece(chunk, target.confidence) else {
                continue;
            };
            if memory.eq_ignore_ascii_case("NO_MEMORY") || confidence <= -0.99 {
                continue;
            }
            let vector = app.brain.embed(&memory).await?;
            if app
                .diary
                .lock()
                .unwrap()
                .remember(&memory, &vector, confidence)?
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

fn memory_piece(chunk: &str, fallback_confidence: f32) -> Option<(String, f32)> {
    let chunk = chunk.trim().trim_start_matches(['-', '*', ' ']).trim();
    if chunk.is_empty() {
        return None;
    }
    let (confidence, body) = if chunk.starts_with('{') {
        let end = chunk.find('\n')?;
        let metadata: Value = serde_json::from_str(&chunk[..end]).ok()?;
        let confidence = metadata
            .get("confidence")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(fallback_confidence);
        (confidence, chunk[end..].trim())
    } else {
        (fallback_confidence, chunk)
    };
    if body.chars().count() < MIN_MEMORY_CHARS || !confidence.is_finite() {
        return None;
    }
    Some((body.to_string(), confidence.clamp(-1.0, 0.99)))
}
