use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::Rng;
use serde::Serialize;

use crate::persistence;

const DEDUP_RELATEDNESS: f64 = 0.95;
const MAX_LISTED_MEMORIES: usize = 100;
const MAX_LISTED_MEMORY_CHARS: usize = 12_000;
const MAX_GRAPH_LINKS: usize = 4;
const GRAPH_RELATEDNESS: f64 = 0.78;
const WORKING_MEMORY_FILE: &str = "working_memory";
const MIN_GENERATED_MEMORY_CHARS: usize = 8;

struct DiaryEntry {
    id: String,
    file_name: String,
    title: String,
    body: String,
    embedding: Vec<f32>,
    confidence: f32,
    usage: u32,
    last_used: i64,
    retired: bool,
    links: Vec<String>,
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
        let mut occupied: HashSet<String> = files
            .iter()
            .filter_map(|path| path.file_stem())
            .map(|stem| stem.to_string_lossy().into_owned())
            .collect();
        for path in files {
            let old_file_name = path.file_stem().unwrap_or_default().to_string_lossy();
            if old_file_name == WORKING_MEMORY_FILE {
                continue;
            }
            let Some(raw) = persistence::read_file(&path) else {
                continue;
            };
            let Some(mut entry) = parse_note(&old_file_name, &raw) else {
                continue;
            };
            if is_legacy_file_name(&old_file_name) {
                let new_file_name = unique_file_name(&entry.title, &occupied);
                if new_file_name != old_file_name {
                    let new_path = note_path(&self.directory, &new_file_name);
                    if fs::rename(&path, &new_path).is_ok() {
                        occupied.insert(new_file_name.clone());
                        entry.file_name = new_file_name;
                        let _ = write_note_to(&self.directory, &entry);
                    }
                }
            }
            occupied.insert(entry.file_name.clone());
            self.entries.push(entry);
        }
        self.connect_notes();
        true
    }

    pub fn reload_if_needed(&mut self) {
        let Ok(files) = persistence::markdown_files(&self.directory) else {
            return;
        };
        let known = self
            .entries
            .iter()
            .map(|entry| entry.file_name.as_str())
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
            .find(|entry| entry.id == id && !entry.retired && entry.confidence < 1.0)
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
            !entry.retired
                && !excluded_ids.iter().any(|id| id == &entry.id)
                && relatedness(embedding, &entry.embedding) > DEDUP_RELATEDNESS
        });
        if too_close {
            return Ok(None);
        }
        let title = memory_title(body);
        let occupied = self
            .entries
            .iter()
            .map(|entry| entry.file_name.clone())
            .collect::<HashSet<_>>();
        let file_name = unique_file_name(&title, &occupied);
        let mut links = self
            .entries
            .iter()
            .filter(|entry| excluded_ids.iter().any(|id| id == &entry.id))
            .map(|entry| entry.file_name.clone())
            .take(MAX_GRAPH_LINKS)
            .collect::<Vec<_>>();
        for link in self.graph_links(embedding, None, excluded_ids) {
            if links.len() == MAX_GRAPH_LINKS {
                break;
            }
            if !links.contains(&link) {
                links.push(link);
            }
        }
        let id = self.next_id();
        let entry = DiaryEntry {
            id,
            file_name,
            title,
            body: body.to_string(),
            embedding: embedding.to_vec(),
            confidence,
            usage: 0,
            last_used: 0,
            retired: false,
            links,
        };
        self.write_note(&entry)?;
        let id = entry.id.clone();
        for existing in &mut self.entries {
            if entry.links.iter().any(|link| link == &existing.file_name)
                && !existing.links.contains(&entry.file_name)
                && existing.links.len() < MAX_GRAPH_LINKS
            {
                existing.links.push(entry.file_name.clone());
                let _ = write_note_to(&self.directory, existing);
            }
        }
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
            .filter(|(_, entry)| !entry.retired && !excluded.iter().any(|id| id == &entry.id))
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
            .filter(|(_, entry)| !entry.retired)
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
            .filter(|entry| !entry.retired && entry.confidence >= 1.0)
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
        let active: Vec<&DiaryEntry> = self.entries.iter().filter(|entry| !entry.retired).collect();
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
            .filter(|entry| {
                !entry.retired
                    && entry.confidence < 1.0
                    && !excluded.iter().any(|id| id == &entry.id)
            })
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
            .filter(|entry| {
                !entry.retired
                    && entry.id != target_id
                    && !excluded.iter().any(|id| id == &entry.id)
            })
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
        let directory = self.directory.clone();
        let mut retired = 0;
        for entry in &mut self.entries {
            if ids.iter().any(|id| id == &entry.id) && !entry.retired && entry.confidence < 1.0 {
                entry.retired = true;
                if let Err(error) = write_note_to(&directory, entry) {
                    entry.retired = false;
                    return Err(error);
                }
                retired += 1;
            }
        }
        Ok(retired)
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

    fn graph_links(
        &self,
        embedding: &[f32],
        excluded_file_name: Option<&str>,
        excluded_ids: &[String],
    ) -> Vec<String> {
        if embedding.is_empty() {
            return Vec::new();
        }
        let mut scored = self
            .entries
            .iter()
            .filter(|entry| {
                !entry.retired
                    && Some(entry.file_name.as_str()) != excluded_file_name
                    && !excluded_ids.iter().any(|id| id == &entry.id)
                    && !entry.embedding.is_empty()
            })
            .map(|entry| {
                (
                    entry.file_name.clone(),
                    relatedness(embedding, &entry.embedding),
                )
            })
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| right.1.total_cmp(&left.1));

        let mut links = scored
            .iter()
            .filter(|(_, score)| *score >= GRAPH_RELATEDNESS)
            .take(MAX_GRAPH_LINKS)
            .map(|(file_name, _)| file_name.clone())
            .collect::<Vec<_>>();
        if links.is_empty() {
            if let Some((file_name, _)) = scored.first() {
                links.push(file_name.clone());
            }
        }
        links
    }

    fn connect_notes(&mut self) {
        for index in 0..self.entries.len() {
            if self.entries[index].retired {
                continue;
            }
            let file_name = self.entries[index].file_name.clone();
            let embedding = self.entries[index].embedding.clone();
            for link in self.graph_links(&embedding, Some(&file_name), &[]) {
                if self.entries[index].links.len() >= MAX_GRAPH_LINKS {
                    break;
                }
                if !self.entries[index].links.contains(&link) {
                    self.entries[index].links.push(link);
                }
            }
        }
        for index in 0..self.entries.len() {
            if self.entries[index].retired {
                continue;
            }
            let file_name = self.entries[index].file_name.clone();
            let links = self.entries[index].links.clone();
            for link in links {
                let Some(other) = self
                    .entries
                    .iter()
                    .position(|entry| entry.file_name == link)
                else {
                    continue;
                };
                if !self.entries[other].links.contains(&file_name)
                    && self.entries[other].links.len() < MAX_GRAPH_LINKS
                {
                    self.entries[other].links.push(file_name.clone());
                }
            }
        }
        for entry in &self.entries {
            let _ = write_note_to(&self.directory, entry);
        }
    }

    fn next_id(&mut self) -> String {
        loop {
            let id = format!("{}-{}", unix_millis(), self.counter);
            self.counter += 1;
            if !self.entries.iter().any(|entry| entry.id == id) {
                return id;
            }
        }
    }
}

fn note_path(directory: &Path, id: &str) -> PathBuf {
    directory.join(format!("{id}.md"))
}

fn write_note_to(directory: &Path, entry: &DiaryEntry) -> std::io::Result<()> {
    let mut text = format!(
        "---\nid: {}\ntitle: {}\nconfidence: {}\nusage: {}\nlast_used: {}\nembedding:",
        entry.id,
        quote_frontmatter(&entry.title),
        entry.confidence,
        entry.usage,
        entry.last_used
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
    if !entry.links.is_empty() {
        text.push_str("\n\nRelated notes:\n");
        for link in &entry.links {
            text.push_str("- [[");
            text.push_str(link);
            text.push_str("]]\n");
        }
    }
    text.push('\n');
    persistence::write_file_atomic(&note_path(directory, &entry.file_name), &text)
}

fn parse_note(id: &str, raw: &str) -> Option<DiaryEntry> {
    let Some(rest) = raw.strip_prefix("---\n") else {
        let body = raw.trim_matches(|c| c == '\n' || c == '\r').trim();
        if body.is_empty() {
            return None;
        }
        return Some(DiaryEntry {
            id: id.to_string(),
            file_name: id.to_string(),
            title: memory_title(body),
            body: body.to_string(),
            embedding: Vec::new(),
            confidence: 1.0,
            usage: 0,
            last_used: 0,
            retired: false,
            links: Vec::new(),
        });
    };
    let separator = rest.find("\n---\n")?;
    let header = &rest[..separator];
    let raw_body = rest[separator + "\n---\n".len()..]
        .trim_matches(|c| c == '\n' || c == '\r')
        .to_string();
    let (body, body_links) = match raw_body
        .rsplit_once("\n\nRelated notes:\n")
        .and_then(|(body, links)| related_note_links(links).map(|links| (body, links)))
    {
        Some((body, links)) => (body.to_string(), links),
        None => (raw_body, Vec::new()),
    };

    let mut entry = DiaryEntry {
        id: id.to_string(),
        file_name: id.to_string(),
        title: String::new(),
        body,
        embedding: Vec::new(),
        confidence: 0.0,
        usage: 0,
        last_used: 0,
        retired: false,
        links: body_links,
    };
    let mut has_diary_metadata = false;
    for line in header.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "id" => entry.id = value.to_string(),
            "title" => entry.title = parse_frontmatter_text(value),
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
            "links" => {
                has_diary_metadata = true;
                entry.links.extend(parse_links(value));
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
    if entry.title.is_empty() {
        entry.title = memory_title(&entry.body);
    }
    if !has_diary_metadata {
        entry.confidence = 1.0;
    }
    Some(entry)
}

fn parse_links(value: &str) -> Vec<String> {
    value
        .split("[[")
        .skip(1)
        .filter_map(|part| part.split("]]").next())
        .map(|link| link.split('|').next().unwrap_or(link).trim().to_string())
        .filter(|link| !link.is_empty())
        .collect()
}

fn related_note_links(value: &str) -> Option<Vec<String>> {
    let lines = value
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty()
        || lines
            .iter()
            .any(|line| !line.starts_with("- [[") || !line.ends_with("]]"))
    {
        return None;
    }
    Some(parse_links(value))
}

fn memory_title(body: &str) -> String {
    let line = body
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('{'))
        .unwrap_or("memory");
    let line = line
        .trim_start_matches(['#', '-', '*', ' '])
        .trim()
        .split_once(" — ")
        .map(|(_, title)| title)
        .unwrap_or(line);
    let title = line
        .chars()
        .filter(|character| !matches!(character, '[' | ']' | '|'))
        .take(80)
        .collect::<String>();
    if title.is_empty() {
        "memory".to_string()
    } else {
        title
    }
}

fn unique_file_name(title: &str, occupied: &HashSet<String>) -> String {
    let base = match slugify(title) {
        base if base.is_empty() => "memory".to_string(),
        base => base,
    };
    if !occupied.contains(&base) {
        return base;
    }
    let mut suffix = 2;
    loop {
        let candidate = format!("{base}-{suffix}");
        if !occupied.contains(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

fn slugify(title: &str) -> String {
    let mut slug = String::new();
    for character in title.chars() {
        if character.is_alphanumeric() {
            for lower in character.to_lowercase() {
                slug.push(lower);
            }
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.trim_matches('-').to_string()
}

fn is_legacy_file_name(file_name: &str) -> bool {
    let mut parts = file_name.split('-');
    let Some(timestamp) = parts.next() else {
        return false;
    };
    let Some(counter) = parts.next() else {
        return false;
    };
    parts.next().is_none()
        && !timestamp.is_empty()
        && !counter.is_empty()
        && timestamp
            .chars()
            .all(|character| character.is_ascii_digit())
        && counter.chars().all(|character| character.is_ascii_digit())
}

fn quote_frontmatter(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn parse_frontmatter_text(value: &str) -> String {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .map(|value| value.replace("\\\"", "\"").replace("\\\\", "\\"))
        .unwrap_or_else(|| value.to_string())
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
