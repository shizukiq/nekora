//! Grouping an incoming burst into one thought, and splitting her reply back out.
//!
//! People send several short messages in a row and mean them as one turn. This
//! buffer waits out a quiet gap (and any "typing…") before handing the whole
//! burst to the brain, so she answers the thought and not each line. The reverse
//! move, `split_message`, breaks her one reply back into human-sized bubbles on
//! blank lines, never cutting inside a code fence.

use std::collections::BTreeMap;

// Once someone stops typing, wait this long before assuming the thought is
// finished. Short enough to feel responsive, long enough to catch a trailing
// line.
const QUIET_MS: i64 = 3_000;
// A "typing…" notification extends the wait by this much, so she doesn't cut in
// while a longer message is still being written.
const TYPING_HOLD_MS: i64 = 4_000;
// …but a burst is never held longer than this from its first message, so a
// non-stop typer can't defer her forever.
const MAX_BATCH_MS: i64 = 10_000;
// A hard cap on messages per batch, the other escape hatch from an endless burst.
const MAX_BATCH_MESSAGES: usize = 32;

#[derive(Clone)]
pub struct ConversationMessage {
    pub sender_id: i64,
    pub message_id: i64,
    pub sender: String,
    pub username: Option<String>,
    pub text: String,
}

pub struct ConversationBatch {
    pub chat_id: i64,
    pub messages: Vec<ConversationMessage>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplyGeneration {
    chat_id: i64,
    revision: u64,
}

struct Pending {
    messages: Vec<ConversationMessage>,
    first_message_at: i64,
    last_message_at: i64,
    typing_until: i64,
    // Arrival order, so when several chats are ready at once the oldest goes first.
    sequence: u64,
}

impl Pending {
    fn is_ready(&self, now_ms: i64) -> bool {
        if self.messages.len() >= MAX_BATCH_MESSAGES
            || now_ms >= self.first_message_at + MAX_BATCH_MS
        {
            return true;
        }
        now_ms >= self.last_message_at + QUIET_MS && now_ms >= self.typing_until
    }

    fn deadline(&self, now_ms: i64) -> i64 {
        if self.messages.len() >= MAX_BATCH_MESSAGES {
            return now_ms;
        }
        (self.first_message_at + MAX_BATCH_MS)
            .min((self.last_message_at + QUIET_MS).max(self.typing_until))
    }
}

#[derive(Default)]
pub struct Conversation {
    pending: BTreeMap<i64, Pending>,
    revisions: BTreeMap<i64, u64>,
    next_sequence: u64,
}

impl Conversation {
    /// Invalidate any reply currently being generated for this chat.
    pub fn message_arrived(&mut self, chat_id: i64) {
        let revision = self.revisions.entry(chat_id).or_default();
        *revision = revision.wrapping_add(1);
    }

    /// Capture the revision a reply must still match before it can be sent.
    pub fn start_generation(&self, chat_id: i64) -> ReplyGeneration {
        ReplyGeneration {
            chat_id,
            revision: self.revisions.get(&chat_id).copied().unwrap_or_default(),
        }
    }

    pub fn generation_is_current(&self, generation: ReplyGeneration) -> bool {
        self.revisions
            .get(&generation.chat_id)
            .copied()
            .unwrap_or_default()
            == generation.revision
    }

    pub fn push(&mut self, chat_id: i64, message: ConversationMessage, now_ms: i64) {
        let sequence = &mut self.next_sequence;
        let pending = self.pending.entry(chat_id).or_insert_with(|| {
            let seq = *sequence;
            *sequence += 1;
            Pending {
                messages: Vec::new(),
                first_message_at: now_ms,
                last_message_at: now_ms,
                typing_until: 0,
                sequence: seq,
            }
        });
        pending.last_message_at = now_ms;
        pending.messages.push(message);
    }

    /// Extend a chat's typing hold, but only if the typer already has a message
    /// pending: a bare "typing…" from someone who hasn't said anything yet is not
    /// a thought to wait on.
    pub fn note_typing(&mut self, chat_id: i64, sender_id: i64, now_ms: i64) -> bool {
        let Some(pending) = self.pending.get_mut(&chat_id) else {
            return false;
        };
        if !pending.messages.iter().any(|m| m.sender_id == sender_id) {
            return false;
        }
        pending.typing_until = pending.typing_until.max(now_ms + TYPING_HOLD_MS);
        true
    }

    /// Take the oldest batch whose quiet/typing window has closed, if any.
    pub fn take_ready(&mut self, now_ms: i64) -> Option<ConversationBatch> {
        let chat_id = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.is_ready(now_ms))
            .min_by_key(|(_, pending)| pending.sequence)
            .map(|(chat_id, _)| *chat_id)?;
        let pending = self.pending.remove(&chat_id).unwrap();
        Some(ConversationBatch {
            chat_id,
            messages: pending.messages,
        })
    }

    /// Remove and return whatever is pending for one chat, ready or not. Used for
    /// the post-batch grace: fold in anything the person sent while she waited.
    pub fn drain_chat(&mut self, chat_id: i64) -> Vec<ConversationMessage> {
        self.pending
            .remove(&chat_id)
            .map(|pending| pending.messages)
            .unwrap_or_default()
    }

    /// When the loop must next wake to check for a ready batch, or -1 when empty.
    pub fn next_deadline(&self, now_ms: i64) -> i64 {
        self.pending
            .values()
            .map(|pending| pending.deadline(now_ms))
            .min()
            .unwrap_or(-1)
    }
}

fn append_limited(text: &str, max_bytes: usize, parts: &mut Vec<String>) {
    if text.is_empty() || max_bytes == 0 {
        return;
    }
    let mut start = 0;
    while text.len() - start > max_bytes {
        let mut limit = start + max_bytes;
        while limit > start && !text.is_char_boundary(limit) {
            limit -= 1;
        }
        // Prefer to break on a newline, then a space, so a bubble ends on a word.
        let window = &text[start..limit];
        let mut cut = window
            .rfind('\n')
            .or_else(|| window.rfind(' '))
            .map(|offset| start + offset)
            .filter(|&cut| cut > start)
            .unwrap_or(limit);
        // A single word longer than the limit has no break; force one at the next
        // char boundary so we still make progress.
        if cut <= start {
            cut = start + 1;
            while cut < text.len() && !text.is_char_boundary(cut) {
                cut += 1;
            }
        }
        parts.push(text[start..cut].to_string());
        start = cut;
        if text[start..].starts_with([' ', '\n']) {
            start += 1;
        }
    }
    if start < text.len() {
        parts.push(text[start..].to_string());
    }
}

/// Split one reply into the bubbles it should arrive as: on blank lines, but
/// never inside a ``` fence, and never a bubble longer than `max_bytes`.
pub fn split_message(text: &str, max_bytes: usize) -> Vec<String> {
    let mut parts = Vec::new();
    if text.is_empty() {
        return parts;
    }
    let bytes = text.as_bytes();
    let mut start = 0;
    let mut position = 0;
    let mut in_code_fence = false;
    while position < bytes.len() {
        if bytes[position..].starts_with(b"```") {
            in_code_fence = !in_code_fence;
            position += 3;
            continue;
        }
        if !in_code_fence && bytes[position] == b'\n' && bytes.get(position + 1) == Some(&b'\n') {
            append_limited(&text[start..position], max_bytes, &mut parts);
            position += 2;
            start = position;
            continue;
        }
        position += 1;
    }
    append_limited(&text[start..], max_bytes, &mut parts);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(sender_id: i64, text: &str) -> ConversationMessage {
        ConversationMessage {
            sender_id,
            message_id: 1,
            sender: "someone".into(),
            username: None,
            text: text.into(),
        }
    }

    #[test]
    fn a_burst_becomes_one_batch_after_the_quiet_window() {
        let mut conversation = Conversation::default();
        conversation.push(7, message(1, "hi"), 0);
        conversation.push(7, message(1, "you there?"), 500);
        assert!(conversation.take_ready(1_000).is_none()); // still inside the quiet gap
        let batch = conversation.take_ready(4_000).expect("ready after quiet");
        assert_eq!(batch.chat_id, 7);
        assert_eq!(batch.messages.len(), 2);
    }

    #[test]
    fn grace_folds_a_late_message_into_the_same_reply() {
        let mut conversation = Conversation::default();
        conversation.push(7, message(1, "hey"), 0);
        // the burst goes quiet and is taken as a batch
        let batch = conversation.take_ready(4_000).expect("ready after quiet");
        assert_eq!(batch.messages.len(), 1);
        // during the response grace the person is still typing and sends more
        conversation.push(7, message(1, "wait, also this"), 4_500);
        // draining pulls that late line so it joins the reply she's about to make
        let late = conversation.drain_chat(7);
        assert_eq!(late.len(), 1);
        assert_eq!(late[0].text, "wait, also this");
        // and nothing is left pending for that chat afterwards
        assert!(conversation.drain_chat(7).is_empty());
    }

    #[test]
    fn typing_defers_readiness() {
        let mut conversation = Conversation::default();
        conversation.push(7, message(1, "wait"), 0);
        assert!(conversation.note_typing(7, 1, 3_500));
        assert!(conversation.take_ready(4_000).is_none()); // held by typing_until
        assert!(conversation.take_ready(8_000).is_some());
    }

    #[test]
    fn splits_on_blank_lines_but_keeps_code_fences_whole() {
        let parts = split_message("first thought\n\nsecond thought", 4096);
        assert_eq!(parts, vec!["first thought", "second thought"]);

        let fenced = "```\nline one\n\nline two\n```";
        assert_eq!(split_message(fenced, 4096), vec![fenced]);
    }

    #[test]
    fn a_long_line_is_broken_on_a_space_within_the_byte_cap() {
        let parts = split_message("aaaa bbbb cccc", 6);
        assert!(parts.iter().all(|p| p.len() <= 6), "{parts:?}");
        assert_eq!(parts.concat().replace(' ', ""), "aaaabbbbcccc");
    }

    #[test]
    fn new_message_invalidates_generation_before_old_result_returns() {
        let mut conversation = Conversation::default();
        let generation = conversation.start_generation(7);

        conversation.message_arrived(7);

        assert!(!conversation.generation_is_current(generation));
    }

    #[test]
    fn new_message_stops_remaining_reply_chunks() {
        let mut conversation = Conversation::default();
        let generation = conversation.start_generation(7);
        let parts = split_message("first\n\nsecond\n\nthird", 4096);
        let mut sent = Vec::new();

        for part in &parts {
            if !conversation.generation_is_current(generation) {
                break;
            }
            sent.push(part.as_str());
            if sent.len() == 1 {
                conversation.message_arrived(7);
            }
        }

        assert_eq!(sent, vec!["first"]);
    }

    #[test]
    fn new_message_during_typing_delay_blocks_pending_chunk() {
        let mut conversation = Conversation::default();
        let generation = conversation.start_generation(7);
        assert!(conversation.generation_is_current(generation));

        // The cancellable delay wakes on the incoming message; this is the
        // guard checked again immediately before send_message.
        conversation.message_arrived(7);

        assert!(!conversation.generation_is_current(generation));
    }

    #[test]
    fn uninterrupted_reply_keeps_chunks_in_order() {
        let conversation = Conversation::default();
        let generation = conversation.start_generation(7);
        let parts = split_message("first\n\nsecond\n\nthird", 4096);
        let sent: Vec<_> = parts
            .iter()
            .filter(|_| conversation.generation_is_current(generation))
            .map(String::as_str)
            .collect();

        assert_eq!(sent, vec!["first", "second", "third"]);
    }
}
