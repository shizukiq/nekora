use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::persistence;

const DEDUP_RELATEDNESS: f64 = 0.95;
const MAX_LISTED_MEMORIES: usize = 100;
const MAX_LISTED_MEMORY_CHARS: usize = 12_000;
const WORKING_MEMORY_FILE: &str = "working_memory";
const MIN_GENERATED_MEMORY_CHARS: usize = 8;
const FORBIDDEN_DIARY_LABELS: &[&str] = &[
    "source",
    "outcome",
    "entities",
    "topics",
    "emotion / relationship",
    "emotion",
    "importance",
    "uncertainty",
    "источник",
    "итог",
    "сущности",
    "темы",
    "эмоция / отношения",
    "эмоции",
    "важность",
    "неопределённость",
    "неопределенность",
];

struct DiaryEntry {
    id: String,
    body: String,
    embedding: Vec<f32>,
    confidence: f32,
    usage: u32,
    last_used: i64,
    retired: bool,
}

#[derive(Deserialize, Serialize)]
struct StoredMetadata {
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    usage: u32,
    #[serde(default)]
    last_used: i64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    embedding: Vec<f32>,
    #[serde(default, skip_serializing_if = "is_false")]
    retired: bool,
}

#[derive(Serialize)]
pub struct Recall {
    pub id: String,
    pub relatedness: f64,
    pub confidence: f32,
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

pub enum MemoryRevision {
    Replaced(String),
    AlreadyKnown,
    Unchanged,
    NotEditable,
}

pub struct Diary {
    directory: PathBuf,
    entries: Vec<DiaryEntry>,
    counter: u64,
}

pub fn is_valid_generated_memory(memory: &str) -> bool {
    let mut cue_line = None;
    for (index, line) in memory.lines().enumerate() {
        if line.trim_start().starts_with("Retrieval cues:") {
            cue_line = Some((index, line));
            break;
        }
    }
    let Some((index, line)) = cue_line else {
        return false;
    };
    let body_chars = memory
        .lines()
        .take(index)
        .flat_map(str::chars)
        .filter(|character| !character.is_whitespace())
        .count();
    if body_chars < MIN_GENERATED_MEMORY_CHARS {
        return false;
    }
    if memory.lines().take(index).any(has_forbidden_diary_label) {
        return false;
    }
    if memory
        .lines()
        .skip(index + 1)
        .any(|line| !line.trim().is_empty())
    {
        return false;
    }
    let Some((_, cues)) = line.split_once(':') else {
        return false;
    };
    let count = cues
        .split([',', ';', '|'])
        .map(str::trim)
        .filter(|cue| !cue.is_empty())
        .count();
    (3..=7).contains(&count)
}

fn has_forbidden_diary_label(line: &str) -> bool {
    let line = line
        .trim_start()
        .trim_start_matches(['#', '*', '_'])
        .trim()
        .to_lowercase();
    FORBIDDEN_DIARY_LABELS.iter().any(|label| {
        line.strip_prefix(label)
            .is_some_and(|rest| rest.trim_start().starts_with(':'))
    })
}

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

    pub fn open(&mut self) -> bool {
        if !persistence::ensure_directory(&self.directory) {
            return false;
        }
        let Ok(mut files) = persistence::markdown_files(&self.directory) else {
            return false;
        };
        self.entries.clear();
        self.counter = 0;
        files.sort();
        let old_names: HashSet<String> = files
            .iter()
            .filter_map(|path| path.file_stem())
            .map(|stem| stem.to_string_lossy().into_owned())
            .collect();
        let mut used_ids = HashSet::new();
        let mut pending = Vec::new();
        for path in files {
            let old_name = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            if old_name == WORKING_MEMORY_FILE {
                continue;
            }
            let Some(raw) = persistence::read_file(&path) else {
                continue;
            };
            let Some(mut entry) = parse_note(&old_name, &raw) else {
                continue;
            };
            if entry.retired {
                let _ = fs::remove_file(path);
                continue;
            }
            if !is_canonical_id(&entry.id)
                || used_ids.contains(&entry.id)
                || (old_names.contains(&entry.id) && entry.id != old_name)
            {
                entry.id = self.next_id_avoiding(&used_ids, &old_names);
            }
            used_ids.insert(entry.id.clone());
            let canonical = old_name == entry.id
                && raw.starts_with("---\n{")
                && !raw.contains("\n\nRelated notes:\n");
            pending.push((path, old_name, entry, canonical));
        }
        for (path, old_name, entry, canonical) in pending {
            if !canonical {
                if write_note_to(&self.directory, &entry).is_err() {
                    return false;
                }
                if old_name != entry.id && fs::remove_file(path).is_err() {
                    return false;
                }
            }
            self.entries.push(entry);
        }
        true
    }

    pub fn reload_if_needed(&mut self) {
        let Ok(files) = persistence::markdown_files(&self.directory) else {
            return;
        };
        let known = self
            .entries
            .iter()
            .map(|entry| entry.id.as_str())
            .collect::<HashSet<_>>();
        let has_new_note = files.iter().any(|path| {
            path.file_stem().is_some_and(|stem| {
                let file_name = stem.to_string_lossy();
                file_name != WORKING_MEMORY_FILE && !known.contains(file_name.as_ref())
            })
        });
        if has_new_note {
            let _ = self.open();
        }
    }

    pub fn remember(
        &mut self,
        body: &str,
        embedding: &[f32],
        confidence: f32,
    ) -> std::io::Result<Option<String>> {
        self.remember_excluding(body, embedding, confidence, &[])
    }

    pub fn remember_replacement(
        &mut self,
        body: &str,
        embedding: &[f32],
        confidence: f32,
        source_ids: &[String],
    ) -> std::io::Result<Option<String>> {
        self.remember_excluding(body, embedding, confidence, source_ids)
    }

    pub fn revise(
        &mut self,
        id: &str,
        body: &str,
        embedding: &[f32],
        confidence: f32,
    ) -> std::io::Result<MemoryRevision> {
        let Some(source) = self
            .entries
            .iter()
            .find(|entry| entry.id == id && entry.confidence < 1.0)
        else {
            return Ok(MemoryRevision::NotEditable);
        };
        if source.body.trim() == body.trim() {
            return Ok(MemoryRevision::Unchanged);
        }

        let source_ids = [id.to_string()];
        let replacement = self.remember_replacement(body, embedding, confidence, &source_ids)?;
        if let Err(error) = self.retire(&source_ids) {
            if let Some(replacement_id) = replacement.as_ref() {
                let _ = self.retire(std::slice::from_ref(replacement_id));
            }
            return Err(error);
        }

        Ok(match replacement {
            Some(id) => MemoryRevision::Replaced(id),
            None => MemoryRevision::AlreadyKnown,
        })
    }

    fn remember_excluding(
        &mut self,
        body: &str,
        embedding: &[f32],
        confidence: f32,
        excluded_ids: &[String],
    ) -> std::io::Result<Option<String>> {
        let too_close = self.entries.iter().any(|entry| {
            !excluded_ids.iter().any(|id| id == &entry.id)
                && relatedness(embedding, &entry.embedding) > DEDUP_RELATEDNESS
        });
        if too_close {
            return Ok(None);
        }
        let id = self.next_id();
        let entry = DiaryEntry {
            id,
            body: body.to_string(),
            embedding: embedding.to_vec(),
            confidence,
            usage: 0,
            last_used: 0,
            retired: false,
        };
        self.write_note(&entry)?;
        let id = entry.id.clone();
        self.entries.push(entry);
        Ok(Some(id))
    }

    pub fn recall(
        &mut self,
        embedding: &[f32],
        limit: usize,
        minimum_relatedness: f64,
        max_body_chars: usize,
        excluded: &[String],
    ) -> Vec<Recall> {
        let mut scored: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| !excluded.iter().any(|id| id == &entry.id))
            .map(|(index, entry)| (index, relatedness(embedding, &entry.embedding)))
            .filter(|(_, score)| *score >= minimum_relatedness)
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(limit);

        let now = unix_seconds();
        let mut remaining = max_body_chars;
        scored
            .into_iter()
            .filter_map(|(index, score)| {
                if remaining == 0 {
                    return None;
                }
                let mut chars = self.entries[index].body.chars();
                let mut body: String = chars.by_ref().take(remaining).collect();
                remaining -= body.chars().count();
                if chars.next().is_some() {
                    body.pop();
                    body.push('…');
                }
                self.touch(index, now);
                Some(Recall {
                    id: self.entries[index].id.clone(),
                    relatedness: score,
                    confidence: self.entries[index].confidence,
                    body,
                })
            })
            .collect()
    }

    pub fn list_memories(&self, limit: usize) -> MemoryList {
        let mut order: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .map(|(index, _)| index)
            .collect();
        order.sort_by(|&a, &b| self.entries[b].id.cmp(&self.entries[a].id));
        let requested = if limit == 0 {
            MAX_LISTED_MEMORIES
        } else {
            limit
        };
        let mut remaining_chars = MAX_LISTED_MEMORY_CHARS;
        let mut body_truncated = false;
        let mut memories = Vec::new();
        for &index in order.iter().take(requested.min(MAX_LISTED_MEMORIES)) {
            if remaining_chars == 0 {
                break;
            }
            let entry = &self.entries[index];
            let mut chars = entry.body.chars();
            let body: String = chars.by_ref().take(remaining_chars).collect();
            body_truncated |= chars.next().is_some();
            remaining_chars -= body.chars().count();
            memories.push(Memory {
                id: entry.id.clone(),
                confidence: entry.confidence,
                usage: entry.usage,
                body,
            });
        }
        MemoryList {
            truncated: body_truncated || memories.len() < order.len(),
            memories,
        }
    }

    pub fn anchors(&self, limit: usize) -> Vec<Memory> {
        let mut anchors: Vec<&DiaryEntry> = self
            .entries
            .iter()
            .filter(|entry| entry.confidence >= 1.0)
            .collect();
        anchors.sort_by(|left, right| left.id.cmp(&right.id));
        anchors.truncate(limit);
        anchors
            .into_iter()
            .map(|entry| Memory {
                id: entry.id.clone(),
                confidence: entry.confidence,
                usage: entry.usage,
                body: entry.body.clone(),
            })
            .collect()
    }

    pub fn random_page(&self) -> Option<String> {
        let active: Vec<&DiaryEntry> = self.entries.iter().collect();
        if active.is_empty() {
            return None;
        }
        let index = unix_nanos() as usize % active.len();
        Some(active[index].body.clone())
    }

    pub fn sleep_target(&self, excluded: &[String]) -> Option<Memory> {
        let active: Vec<&DiaryEntry> = self
            .entries
            .iter()
            .filter(|entry| entry.confidence < 1.0 && !excluded.iter().any(|id| id == &entry.id))
            .collect();
        if active.is_empty() {
            return None;
        }
        let mut rng = rand::rng();
        let entry = if rng.random_bool(0.2) {
            active.get(rng.random_range(0..active.len()))?
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

    pub fn sleep_related(
        &self,
        target_id: &str,
        embedding: &[f32],
        limit: usize,
        minimum_relatedness: f64,
        excluded: &[String],
    ) -> Vec<Memory> {
        let mut scored: Vec<(&DiaryEntry, f64)> = self
            .entries
            .iter()
            .filter(|entry| entry.id != target_id && !excluded.iter().any(|id| id == &entry.id))
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

    pub fn retire(&mut self, ids: &[String]) -> std::io::Result<usize> {
        let mut kept = Vec::with_capacity(self.entries.len());
        let mut removed = 0;
        let entries = std::mem::take(&mut self.entries);
        let mut entries = entries.into_iter();
        while let Some(entry) = entries.next() {
            let should_remove = ids.iter().any(|id| id == &entry.id) && entry.confidence < 1.0;
            if !should_remove {
                kept.push(entry);
                continue;
            }

            match fs::remove_file(note_path(&self.directory, &entry.id)) {
                Ok(()) => removed += 1,
                Err(error) if error.kind() == io::ErrorKind::NotFound => removed += 1,
                Err(error) => {
                    kept.push(entry);
                    kept.extend(entries);
                    self.entries = kept;
                    return Err(error);
                }
            }
        }
        self.entries = kept;
        Ok(removed)
    }

    fn touch(&mut self, index: usize, now: i64) {
        let entry = &mut self.entries[index];
        entry.usage = entry.usage.saturating_add(1);
        entry.last_used = now;
        let _ = write_note_to(&self.directory, entry);
    }

    fn write_note(&self, entry: &DiaryEntry) -> std::io::Result<()> {
        write_note_to(&self.directory, entry)
    }

    fn next_id(&mut self) -> String {
        loop {
            let id = unix_millis()
                .saturating_add(u128::from(self.counter))
                .to_string();
            self.counter += 1;
            if !self.entries.iter().any(|entry| entry.id == id) {
                return id;
            }
        }
    }

    fn next_id_avoiding(&mut self, used: &HashSet<String>, old_names: &HashSet<String>) -> String {
        loop {
            let id = unix_millis()
                .saturating_add(u128::from(self.counter))
                .to_string();
            self.counter += 1;
            if !used.contains(&id)
                && !old_names.contains(&id)
                && !self.entries.iter().any(|entry| entry.id == id)
            {
                return id;
            }
        }
    }
}

fn note_path(directory: &Path, id: &str) -> PathBuf {
    directory.join(format!("{id}.md"))
}

fn write_note_to(directory: &Path, entry: &DiaryEntry) -> std::io::Result<()> {
    let metadata = StoredMetadata {
        confidence: entry.confidence,
        usage: entry.usage,
        last_used: entry.last_used,
        embedding: entry.embedding.clone(),
        retired: entry.retired,
    };
    let metadata = serde_json::to_string(&metadata)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut text = format!("---\n{metadata}\n---\n");
    text.push_str(entry.body.trim());
    text.push('\n');
    persistence::write_file_atomic(&note_path(directory, &entry.id), &text)
}

fn parse_note(id: &str, raw: &str) -> Option<DiaryEntry> {
    let Some(rest) = raw.strip_prefix("---\n") else {
        let body = raw.trim_matches(|c| c == '\n' || c == '\r').trim();
        if body.is_empty() {
            return None;
        }
        return Some(DiaryEntry {
            id: id.to_string(),
            body: body.to_string(),
            embedding: Vec::new(),
            confidence: 1.0,
            usage: 0,
            last_used: 0,
            retired: false,
        });
    };
    let separator = rest.find("\n---\n")?;
    let header = &rest[..separator];
    let raw_body = rest[separator + "\n---\n".len()..]
        .trim_matches(|c| c == '\n' || c == '\r')
        .to_string();
    let body = strip_related_notes(&raw_body);

    if header.trim_start().starts_with('{') {
        let metadata: StoredMetadata = serde_json::from_str(header.trim()).ok()?;
        return Some(DiaryEntry {
            id: id.to_string(),
            body,
            embedding: metadata.embedding,
            confidence: metadata.confidence,
            usage: metadata.usage,
            last_used: metadata.last_used,
            retired: metadata.retired,
        });
    }

    let mut entry = DiaryEntry {
        id: id.to_string(),
        body,
        embedding: Vec::new(),
        confidence: 0.0,
        usage: 0,
        last_used: 0,
        retired: false,
    };
    let mut has_diary_metadata = false;
    for line in header.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "id" => entry.id = value.to_string(),
            "confidence" => {
                has_diary_metadata = true;
                entry.confidence = parse_finite(value)?;
            }
            "usage" => {
                has_diary_metadata = true;
                entry.usage = value.parse().ok()?;
            }
            "last_used" => {
                has_diary_metadata = true;
                entry.last_used = value.parse().ok()?;
            }
            "retired" => {
                has_diary_metadata = true;
                entry.retired = value == "true";
            }
            "embedding" => {
                has_diary_metadata = true;
                for token in value.split_whitespace() {
                    entry.embedding.push(parse_finite(token)?);
                }
            }
            _ => {}
        }
    }
    if !has_diary_metadata {
        entry.confidence = 1.0;
    }
    Some(entry)
}

fn strip_related_notes(body: &str) -> String {
    let Some((body, links)) = body.rsplit_once("\n\nRelated notes:\n") else {
        return body.to_string();
    };
    let lines = links
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if !lines.is_empty()
        && lines
            .iter()
            .all(|line| line.starts_with("- [[") && line.ends_with("]]"))
    {
        return body.trim_end().to_string();
    }
    body.to_string()
}

fn is_canonical_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|character| character.is_ascii_digit())
}

fn is_false(value: &bool) -> bool {
    !*value
}

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

        assert!(diary
            .remember("apple", &[1.0, 0.0, 0.0, 0.0], 0.0)
            .unwrap()
            .is_some());
        assert!(diary
            .remember("banana", &[0.0, 1.0, 0.0, 0.0], 0.0)
            .unwrap()
            .is_some());
        let escaped = "cherry \"red\"\nline";
        assert!(diary
            .remember(escaped, &[0.0, 0.0, 1.0, 0.0], 0.0)
            .unwrap()
            .is_some());

        let hits = diary.recall(&[0.9, 0.1, 0.0, 0.0], 2, 0.0, 12_000, &[]);
        assert_eq!(hits[0].body, "apple");
        assert!(hits[0].relatedness > hits[1].relatedness);

        // A body with quotes and a newline round-trips through the vault intact.
        let escaped_hits = diary.recall(&[0.0, 0.0, 1.0, 0.0], 1, 0.0, 12_000, &[]);
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
