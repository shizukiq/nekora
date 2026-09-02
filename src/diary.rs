//! Long-term memory: the markdown vault and the cosine recall over it.
//!
//! Each memory is one `.md` note — a frontmatter block (confidence, usage,
//! last_used, and the raw embedding) over a first-person body. Recall is a plain
//! linear cosine scan; the vault is small enough that a note is a page she flips
//! to, not a row in a database. The embeddings are handed in from outside — the
//! diary never decides what a vector means, only which stored ones are nearest.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::persistence;

// Two memories this close are the same memory said twice; the second is dropped
// rather than filed, so the vault doesn't fill with paraphrases.
const DEDUP_RELATEDNESS: f64 = 0.95;
// The ceiling on a single `list_memories` answer, so "what do you remember" can
// never dump an unbounded wall of notes.
const MAX_LISTED_MEMORIES: usize = 100;

struct DiaryEntry {
    id: String,
    body: String,
    embedding: Vec<f32>,
    confidence: f32,
    usage: u32,
    last_used: i64,
    retired: bool,
}

#[derive(Serialize)]
pub struct Recall {
    pub id: String,
    pub relatedness: f64,
    pub body: String,
}

#[derive(Serialize)]
pub struct Memory {
    pub id: String,
    pub confidence: f32,
    pub usage: u32,
    pub body: String,
}

#[derive(Serialize)]
pub struct MemoryList {
    pub memories: Vec<Memory>,
    pub truncated: bool,
}

pub struct Diary {
    directory: PathBuf,
    entries: Vec<DiaryEntry>,
    counter: u64,
}

// Cosine similarity remapped from [-1, 1] to [0, 1], the same scale the tools
// threshold against. Mismatched or empty vectors score 0 rather than panic, so a
// half-written note can't take recall down.
fn relatedness(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&left, &right) in a.iter().zip(b) {
        if !left.is_finite() || !right.is_finite() {
            return 0.0;
        }
        let (left, right) = (left as f64, right as f64);
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt()) + 1.0) / 2.0
}

impl Diary {
    pub fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            entries: Vec::new(),
            counter: 0,
        }
    }

    /// Create the vault directory if needed and load every note it holds. Safe to
    /// call again to reload from disk.
    pub fn open(&mut self) -> bool {
        self.entries.clear();
        self.counter = 0;
        if !persistence::ensure_directory(&self.directory) {
            return false;
        }
        let Ok(mut files) = persistence::markdown_files(&self.directory) else {
            return false;
        };
        files.sort();
        for path in files {
            if let Some(raw) = persistence::read_file(&path) {
                let id = path.file_stem().unwrap_or_default().to_string_lossy();
                if let Some(entry) = parse_note(&id, &raw) {
                    self.entries.push(entry);
                }
            }
        }
        true
    }

    /// File a memory, unless it is a near-duplicate of one already held. Returns
    /// the new note's index, or `None` when it was dropped as a duplicate; `Err`
    /// only when the note could not be written to disk.
    pub fn remember(
        &mut self,
        body: &str,
        embedding: &[f32],
        confidence: f32,
    ) -> std::io::Result<Option<usize>> {
        let too_close = self.entries.iter().any(|entry| {
            !entry.retired && relatedness(embedding, &entry.embedding) > DEDUP_RELATEDNESS
        });
        if too_close {
            return Ok(None);
        }
        let entry = DiaryEntry {
            id: self.next_id(),
            body: body.to_string(),
            embedding: embedding.to_vec(),
            confidence,
            usage: 0,
            last_used: 0,
            retired: false,
        };
        self.write_note(&entry)?;
        self.entries.push(entry);
        Ok(Some(self.entries.len() - 1))
    }

    /// The `limit` nearest memories to `embedding`, most related first. Each one
    /// returned is touched — usage up, last_used now — so heavily recalled notes
    /// can be told apart from dead weight later.
    pub fn recall(&mut self, embedding: &[f32], limit: usize) -> Vec<Recall> {
        let mut scored: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| !entry.retired)
            .map(|(index, entry)| (index, relatedness(embedding, &entry.embedding)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(limit);

        let now = unix_seconds();
        scored
            .into_iter()
            .map(|(index, score)| {
                self.touch(index, now);
                Recall {
                    id: self.entries[index].id.clone(),
                    relatedness: score,
                    body: self.entries[index].body.clone(),
                }
            })
            .collect()
    }

    /// The most recent memories, newest id first, capped so the answer stays
    /// bounded. `limit` of 0 means "as many as the cap allows".
    pub fn list_memories(&self, limit: usize) -> MemoryList {
        let mut order: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| !entry.retired)
            .map(|(index, _)| index)
            .collect();
        order.sort_by(|&a, &b| self.entries[b].id.cmp(&self.entries[a].id));
        let requested = if limit == 0 {
            MAX_LISTED_MEMORIES
        } else {
            limit
        };
        let count = requested.min(MAX_LISTED_MEMORIES).min(order.len());
        let memories = order[..count]
            .iter()
            .map(|&index| {
                let entry = &self.entries[index];
                Memory {
                    id: entry.id.clone(),
                    confidence: entry.confidence,
                    usage: entry.usage,
                    body: entry.body.clone(),
                }
            })
            .collect();
        MemoryList {
            memories,
            truncated: count < order.len(),
        }
    }

    /// A random note's body, the seed for a "reflect on an old page" tick.
    pub fn random_page(&self) -> Option<String> {
        let active: Vec<&DiaryEntry> = self.entries.iter().filter(|entry| !entry.retired).collect();
        if active.is_empty() {
            return None;
        }
        // Reflection happens minutes apart, so the low bits of the wall clock are
        // effectively random for the purpose of picking a page.
        let index = unix_nanos() as usize % active.len();
        Some(active[index].body.clone())
    }

    /// Pick a mutable source for diary sleep: usually the newest active note,
    /// with an occasional random page so old memories get a chance to merge too.
    /// Confidence 1.0 notes are anchors and are never rewritten.
    pub fn sleep_target(&self, excluded: &[String]) -> Option<Memory> {
        let active: Vec<&DiaryEntry> = self
            .entries
            .iter()
            .filter(|entry| {
                !entry.retired
                    && entry.confidence < 1.0
                    && !excluded.iter().any(|id| id == &entry.id)
            })
            .collect();
        if active.is_empty() {
            return None;
        }
        let entry = if unix_nanos() % 5 == 0 {
            active.get(unix_nanos() as usize % active.len())?
        } else {
            active.iter().max_by(|left, right| left.id.cmp(&right.id))?
        };
        Some(Memory {
            id: entry.id.clone(),
            confidence: entry.confidence,
            usage: entry.usage,
            body: entry.body.clone(),
        })
    }

    /// Find active notes close to a sleep target without changing recall stats.
    pub fn sleep_related(
        &self,
        target_id: &str,
        embedding: &[f32],
        limit: usize,
        minimum_relatedness: f64,
    ) -> Vec<Memory> {
        let mut scored: Vec<(&DiaryEntry, f64)> = self
            .entries
            .iter()
            .filter(|entry| !entry.retired && entry.id != target_id)
            .map(|entry| (entry, relatedness(embedding, &entry.embedding)))
            .filter(|(_, score)| *score >= minimum_relatedness)
            .collect();
        scored.sort_by(|left, right| right.1.total_cmp(&left.1));
        scored.truncate(limit);
        scored
            .into_iter()
            .map(|(entry, _)| Memory {
                id: entry.id.clone(),
                confidence: entry.confidence,
                usage: entry.usage,
                body: entry.body.clone(),
            })
            .collect()
    }

    /// Archive source notes after a replacement memory was successfully saved.
    /// The markdown stays on disk for recovery, but normal recall ignores it.
    pub fn retire(&mut self, ids: &[String]) -> std::io::Result<usize> {
        let directory = self.directory.clone();
        let mut retired = 0;
        for entry in &mut self.entries {
            if ids.iter().any(|id| id == &entry.id) && !entry.retired {
                entry.retired = true;
                write_note_to(&directory, entry)?;
                retired += 1;
            }
        }
        Ok(retired)
    }

    fn touch(&mut self, index: usize, now: i64) {
        let entry = &mut self.entries[index];
        entry.usage = entry.usage.saturating_add(1);
        entry.last_used = now;
        // A failure to persist the refreshed metadata is not worth losing the
        // recall hit over; the in-memory bump survives until the next write.
        let _ = write_note_to(&self.directory, entry);
    }

    fn write_note(&self, entry: &DiaryEntry) -> std::io::Result<()> {
        write_note_to(&self.directory, entry)
    }

    fn next_id(&mut self) -> String {
        let id = format!("{}-{}", unix_millis(), self.counter);
        self.counter += 1;
        id
    }
}

fn note_path(directory: &Path, id: &str) -> PathBuf {
    directory.join(format!("{id}.md"))
}

fn write_note_to(directory: &Path, entry: &DiaryEntry) -> std::io::Result<()> {
    let mut text = format!(
        "---\nconfidence: {}\nusage: {}\nlast_used: {}\nembedding:",
        entry.confidence, entry.usage, entry.last_used
    );
    for value in &entry.embedding {
        text.push(' ');
        text.push_str(&value.to_string());
    }
    if entry.retired {
        text.push_str("\nretired: true");
    }
    text.push_str("\n---\n");
    text.push_str(&entry.body);
    text.push('\n');
    persistence::write_file_atomic(&note_path(directory, &entry.id), &text)
}

fn parse_note(id: &str, raw: &str) -> Option<DiaryEntry> {
    let rest = raw.strip_prefix("---\n")?;
    let separator = rest.find("\n---\n")?;
    let header = &rest[..separator];
    let body = rest[separator + "\n---\n".len()..]
        .trim_matches(|c| c == '\n' || c == '\r')
        .to_string();

    let mut entry = DiaryEntry {
        id: id.to_string(),
        body,
        embedding: Vec::new(),
        confidence: 0.0,
        usage: 0,
        last_used: 0,
        retired: false,
    };
    for line in header.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "confidence" => entry.confidence = parse_finite(value)?,
            "usage" => entry.usage = value.parse().ok()?,
            "last_used" => entry.last_used = value.parse().ok()?,
            "retired" => entry.retired = value == "true",
            "embedding" => {
                for token in value.split_whitespace() {
                    entry.embedding.push(parse_finite(token)?);
                }
            }
            _ => {}
        }
    }
    Some(entry)
}

// A note whose stored number is garbage is a broken note, not a zero: reject the
// whole entry so a corrupt embedding never silently ranks against real ones.
fn parse_finite(value: &str) -> Option<f32> {
    let parsed: f32 = value.parse().ok()?;
    parsed.is_finite().then_some(parsed)
}

fn unix_seconds() -> i64 {
    duration_since_epoch().as_secs() as i64
}

fn unix_millis() -> u128 {
    duration_since_epoch().as_millis()
}

fn unix_nanos() -> u128 {
    duration_since_epoch().as_nanos()
}

fn duration_since_epoch() -> std::time::Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_ranks_dedups_and_reloads() {
        let configured_vault = std::env::var_os("NEKORA_TEST_VAULT").map(PathBuf::from);
        let vault = configured_vault
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join(format!("nekora-diary-{}", unix_nanos())));
        if configured_vault.is_some() && vault.exists() {
            assert!(
                persistence::markdown_files(&vault)
                    .expect("test vault should be readable")
                    .is_empty(),
                "NEKORA_TEST_VAULT must not already contain markdown notes"
            );
        }
        let mut diary = Diary::new(vault.clone());
        assert!(diary.open());

        assert_eq!(
            diary.remember("apple", &[1.0, 0.0, 0.0, 0.0], 0.0).unwrap(),
            Some(0)
        );
        assert_eq!(
            diary
                .remember("banana", &[0.0, 1.0, 0.0, 0.0], 0.0)
                .unwrap(),
            Some(1)
        );
        let escaped = "cherry \"red\"\nline";
        assert_eq!(
            diary.remember(escaped, &[0.0, 0.0, 1.0, 0.0], 0.0).unwrap(),
            Some(2)
        );

        let hits = diary.recall(&[0.9, 0.1, 0.0, 0.0], 2);
        assert_eq!(hits[0].body, "apple");
        assert!(hits[0].relatedness > hits[1].relatedness);

        // A body with quotes and a newline round-trips through the vault intact.
        let escaped_hits = diary.recall(&[0.0, 0.0, 1.0, 0.0], 1);
        assert_eq!(escaped_hits[0].body, escaped);

        // A near-copy of apple is refused as a duplicate.
        assert_eq!(
            diary
                .remember("apple2", &[0.99, 0.01, 0.0, 0.0], 0.0)
                .unwrap(),
            None
        );

        let listed = diary.list_memories(0);
        assert_eq!(listed.memories.len(), 3);
        assert!(!listed.truncated);

        // Reopening reads the same three notes back off disk.
        assert!(diary.open());
        assert_eq!(diary.list_memories(0).memories.len(), 3);

        if configured_vault.is_some() {
            for path in persistence::markdown_files(&vault).unwrap_or_default() {
                std::fs::remove_file(path).ok();
            }
        } else {
            std::fs::remove_dir_all(&vault).ok();
        }
    }
}
