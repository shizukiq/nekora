//! Nekora's heartbeat loop -- the conductor that lives on a timer.
//!
//! The core decides *whether*; the brain decides *what*; this file schedules and,
//! when the container asks, owns the local Ollama process. Every 27 minutes it
//! flips the core's coin: on "act" she drifts to a random diary page and
//! reflects, maybe messaging someone. An incoming burst arrives sooner -- it wakes
//! her from any nap, waits until the person finishes a thought, then hands the
//! whole burst to the brain. When the day's talk grows heavy she sleeps: the
//! context is distilled into the diary and she wakes on a clean slate.
//!
//! Nothing here is instant and nothing here is eager -- that is the whole point of
//! modelling a unit instead of an assistant.

mod brain;
mod config;
mod conversation;
mod diary;
mod heartbeat;
mod ollama;
mod persistence;
mod sleep;
mod tools;
mod userbot;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Result};
use chrono::Local;
use grammers_client::client::UpdatesConfiguration;
use grammers_client::session::storages::SqliteSession;
use grammers_client::session::types::{PeerId, PeerKind};
use grammers_client::tl;
use grammers_client::update::Update;
use grammers_client::{Client, SenderPool};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

use brain::Brain;
use conversation::{Conversation, ConversationBatch, ConversationMessage, ReplyGeneration};
use diary::Diary;
use heartbeat::Heartbeat;
use userbot::{Incoming, Userbot};

// The core ticks about every 27 minutes -- long enough that she is plainly living
// on her own clock, not watching the chat.
const TICK: Duration = Duration::from_secs(27 * 60);
// Sometimes she reads an incoming message and lets it lie, so she can't be driven
// purely by whoever texts most.
const SUGGEST_IGNORE_CHANCE: f64 = 0.1;
// After a burst goes quiet she still waits this long before answering, in case the
// person is mid-thought and about to send more. Anything that lands during the
// grace is folded into the same reply, so she reads the whole thing instead of
// cutting in -- the same reason she sends more than one message herself.
const RESPONSE_GRACE: Duration = Duration::from_secs(5);
// Occasionally re-list her tools so she doesn't forget she can search, remember,
// or send a photo.
const TOOL_REMINDER_CHANCE: f64 = 0.02;
// How much of today she carries into each turn: the recent, timestamped tail of
// the buffer. Without it every message looks timeless. Bounded so a long day does
// not resend the whole context on every message.
const RECENT_LINES: usize = 40;
const TODAY_FILE: &str = "today.json";

struct Today {
    day: String,
    lines: Vec<String>,
    path: PathBuf,
}

#[derive(Deserialize, Serialize)]
struct SavedToday {
    day: String,
    lines: Vec<String>,
}

struct TodaySnapshot {
    day: String,
    lines: Vec<String>,
}

impl Today {
    fn open() -> Result<Self> {
        let path = config::runtime_dir().join(TODAY_FILE);
        let saved = if path.exists() {
            let raw = persistence::read_file(&path)
                .ok_or_else(|| anyhow!("could not read today's journal at {path:?}"))?;
            if raw.trim().is_empty() {
                None
            } else {
                Some(serde_json::from_str::<SavedToday>(&raw)?)
            }
        } else {
            None
        };
        let today = Self {
            day: saved
                .as_ref()
                .map(|saved| saved.day.clone())
                .unwrap_or_else(today_str),
            lines: saved.map(|saved| saved.lines).unwrap_or_default(),
            path,
        };
        if !today.path.exists() {
            today.persist()?;
        }
        Ok(today)
    }

    fn snapshot(&self) -> TodaySnapshot {
        TodaySnapshot {
            day: self.day.clone(),
            lines: self.lines.clone(),
        }
    }

    fn append(&mut self, line: String) -> Result<()> {
        self.lines.push(line);
        self.persist()
    }

    /// Remove exactly the lines that were handed to sleep. Messages arriving
    /// while the model is working stay at the front of the next day/turn.
    fn finish(&mut self, snapshot: &TodaySnapshot, next_day: Option<String>) -> Result<()> {
        if self.day != snapshot.day || self.lines.len() < snapshot.lines.len() {
            return Err(anyhow!(
                "today journal changed while it was being consolidated"
            ));
        }
        let removed: Vec<_> = self.lines.drain(..snapshot.lines.len()).collect();
        let previous_day = self.day.clone();
        if let Some(next_day) = next_day {
            self.day = next_day;
        }
        if let Err(error) = self.persist() {
            self.day = previous_day;
            self.lines.splice(0..0, removed);
            return Err(error);
        }
        Ok(())
    }

    fn persist(&self) -> Result<()> {
        let contents = serde_json::to_string(&SavedToday {
            day: self.day.clone(),
            lines: self.lines.clone(),
        })?;
        persistence::write_file_atomic(&self.path, &contents)?;
        Ok(())
    }
}

/// Everything shared between the update ingest and the heartbeat loop. One
/// process, one Nekora: a single instance of each subsystem, guarded where two
/// tasks touch it.
pub struct App {
    pub brain: Arc<Brain>,
    pub userbot: Arc<Userbot>,
    pub diary: Mutex<Diary>,
    heartbeat: Mutex<Heartbeat>,
    conversation: Mutex<Conversation>,
    today: Mutex<Today>,
    // Pulses the heartbeat loop when a message arrives, so it re-checks at once
    // instead of sleeping out the tick.
    wake: Notify,
    // The monotonic origin the conversation's millisecond timers are measured from.
    started: Instant,
}

impl App {
    fn new(brain: Arc<Brain>, userbot: Arc<Userbot>, diary: Diary, today: Today) -> Self {
        Self {
            brain,
            userbot,
            diary: Mutex::new(diary),
            heartbeat: Mutex::new(Heartbeat::new(unix_seconds() as u64)),
            conversation: Mutex::new(Conversation::default()),
            today: Mutex::new(today),
            wake: Notify::new(),
            started: Instant::now(),
        }
    }

    fn monotonic_ms(&self) -> i64 {
        self.started.elapsed().as_millis() as i64
    }

    pub(crate) fn message_arrived(&self, chat_id: i64) {
        self.conversation.lock().unwrap().message_arrived(chat_id);
    }

    pub(crate) fn generation_is_current(&self, generation: ReplyGeneration) -> bool {
        self.conversation
            .lock()
            .unwrap()
            .generation_is_current(generation)
    }

    /// Wait for a reply delay, waking early when the chat revision changes.
    pub(crate) async fn wait_for_generation_delay(
        &self,
        generation: ReplyGeneration,
        delay: Duration,
    ) -> bool {
        let deadline = tokio::time::Instant::now() + delay;
        loop {
            if !self.generation_is_current(generation) {
                return false;
            }
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return true;
            }
            tokio::select! {
                biased;
                _ = self.wake.notified() => {}
                _ = tokio::time::sleep(remaining) => {
                    return self.generation_is_current(generation);
                }
            }
        }
    }

    /// Keep an incoming message in today's context even when the turn is ignored.
    fn record_event(&self, event: &Incoming) {
        let line = message_block(
            event.chat_id,
            &event.sender,
            event.username.as_deref(),
            event.sender_id,
            event.message_id,
            &event.timestamp,
            &event.metadata,
            &event.text,
        );
        if let Err(error) = self.today.lock().unwrap().append(line) {
            eprintln!("today journal write failed: {error:#}");
        }
    }

    /// Keep a sent answer in today's context.
    pub fn record_outgoing(&self, chat_id: i64, text: &str, reply_to_message_id: Option<i64>) {
        let metadata = reply_to_message_id.map_or_else(String::new, |message_id| {
            format!("telegram_context:\ntelegram_reply_to_message_id={message_id}\n")
        });
        let line = message_block(
            chat_id,
            &config::nekora_name(),
            None,
            0,
            0,
            &now_stamp(),
            &metadata,
            text,
        );
        if let Err(error) = self.today.lock().unwrap().append(line) {
            eprintln!("today journal write failed: {error:#}");
        }
    }

    /// Keep a reaction Nekora sent in today's context, including its target.
    pub fn record_reaction(&self, chat_id: i64, message_id: i64, reaction: &str) {
        let reaction = reaction.trim();
        let action = if reaction.is_empty() {
            "removed reaction".to_string()
        } else {
            format!("reacted with {reaction}")
        };
        let metadata =
            format!("telegram_context:\ntelegram_reaction_target_message_id={message_id}\n");
        let line = message_block(
            chat_id,
            &config::nekora_name(),
            None,
            0,
            0,
            &now_stamp(),
            &metadata,
            &format!("[{action} to message_id={message_id}]"),
        );
        if let Err(error) = self.today.lock().unwrap().append(line) {
            eprintln!("today journal write failed: {error:#}");
        }
    }

    fn today_snapshot(&self) -> TodaySnapshot {
        self.today.lock().unwrap().snapshot()
    }

    fn finish_today(&self, snapshot: &TodaySnapshot, next_day: Option<String>) -> Result<()> {
        self.today.lock().unwrap().finish(snapshot, next_day)
    }

    /// Today's timestamped tail, so the turn can reason about elapsed real time.
    fn recent_context(&self) -> String {
        let lines = &self.today.lock().unwrap().lines;
        let start = lines.len().saturating_sub(RECENT_LINES);
        if start == lines.len() {
            return String::new();
        }
        format!(
            "recently (real times, today):\n{}\n",
            lines[start..].join("\n")
        )
    }
}

fn maybe_tool_reminder() -> String {
    if rand::rng().random::<f64>() < TOOL_REMINDER_CHANCE {
        "(you can recall_memory, list_memories, remember, inspect_user, inspect_message_media, \
         get_current_time, \
         send_message, react_to_message, list_chats.)\n"
            .to_string()
    } else {
        String::new()
    }
}

/// An "act" tick with nobody talking: reflect on an old page, then let her act.
async fn proactive(app: &Arc<App>) -> Result<()> {
    let recent = app.recent_context();
    let thought = sleep::reflect(app, &recent).await?;
    let drift = match thought {
        Some(thought) => format!("you just reflected on an old diary page:\n{thought}"),
        None => "a quiet moment; your diary is still empty".to_string(),
    };
    let working_memory = sleep::working_memory_context();
    let content = format!(
        "{}\n{}{}{recent}({drift})",
        config::preamble(),
        maybe_tool_reminder(),
        working_memory,
    );
    brain::act(app, vec![brain::user(content)], None).await
}

/// Give one burst of incoming messages to the brain as one conversational turn.
async fn respond(app: &Arc<App>, events: &[Incoming], generation: ReplyGeneration) -> Result<()> {
    let chat_id = events[0].chat_id;
    app.userbot.go_online().await;

    let context = app.recent_context();
    let lines = events
        .iter()
        .map(|event| {
            message_block(
                event.chat_id,
                &event.sender,
                event.username.as_deref(),
                event.sender_id,
                event.message_id,
                &event.timestamp,
                &event.metadata,
                &event.text,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let memories = sleep::relevant_memories_context(app, &lines).await;
    let working_memory = sleep::working_memory_context();
    let content = format!(
        "{}\n{}{}{memories}{context}current reply target chat_id: {chat_id}\nseveral messages arrived close together; read them as one thought \
         and answer the whole thing:\n{lines}",
        config::preamble(),
        maybe_tool_reminder(),
        working_memory,
    );
    // The "typing…" indicator runs for the whole turn -- the generation and the
    // sending -- so it tracks real thinking time instead of a delay pasted on after.
    app.userbot
        .keep_typing(
            chat_id,
            brain::act(app, vec![brain::user(content)], Some(generation)),
        )
        .await
}

/// Wait for a ready conversation batch or the autonomous tick, whichever comes
/// first. `None` means the tick fired with nobody talking.
async fn wait_for_turn(app: &Arc<App>) -> Option<ConversationBatch> {
    loop {
        let now_ms = app.monotonic_ms();
        if let Some(batch) = app.conversation.lock().unwrap().take_ready(now_ms) {
            return Some(batch);
        }
        let deadline = app.conversation.lock().unwrap().next_deadline(now_ms);
        let has_conversation = deadline >= 0;
        let timeout = if has_conversation {
            TICK.min(Duration::from_millis((deadline - now_ms).max(0) as u64))
        } else {
            TICK
        };
        tokio::select! {
            _ = app.wake.notified() => continue,
            _ = tokio::time::sleep(timeout) => {
                if has_conversation {
                    continue; // the batch's window has closed; take it next loop
                }
                return None;
            }
        }
    }
}

/// The loop: wake on a message or every tick, act, then sleep if the day is heavy.
async fn heartbeat_loop(app: &Arc<App>) {
    loop {
        if let Err(error) = run_turn(app).await {
            // One bad turn -- both LLM endpoints down, a backend blip mid-turn --
            // must never take the whole unit down. Log it and keep beating.
            eprintln!("heartbeat: turn failed, continuing: {error:#}");
        }
    }
}

async fn run_turn(app: &Arc<App>) -> Result<()> {
    let batch = wait_for_turn(app).await;

    let now = unix_seconds();
    let day = today_str();
    if app.today.lock().unwrap().day != day {
        let snapshot = app.today_snapshot();
        match sleep::consolidate(app, snapshot.lines.clone(), true).await {
            Ok(fresh) if fresh.is_empty() => {
                app.finish_today(&snapshot, Some(day.clone()))?;
            }
            Ok(_) => {}
            Err(error) => {
                // The day rolled over but sleep failed: put the pending burst back
                // so it is answered next turn rather than lost.
                if let Some(batch) = &batch {
                    let mut conversation = app.conversation.lock().unwrap();
                    for message in &batch.messages {
                        conversation.push(batch.chat_id, message.clone(), app.monotonic_ms());
                    }
                }
                return Err(error);
            }
        }
    }

    match batch {
        None => {
            if app.heartbeat.lock().unwrap().tick(now) {
                proactive(app).await?;
            }
        }
        Some(batch) => {
            let chat_id = batch.chat_id;
            let mut events = to_events(chat_id, batch.messages);
            app.userbot.mark_read(chat_id).await;
            if rand::rng().random::<f64>() >= SUGGEST_IGNORE_CHANCE {
                tokio::time::sleep(RESPONSE_GRACE).await;
                let (late, generation) = {
                    let mut conversation = app.conversation.lock().unwrap();
                    let late = conversation.drain_chat(chat_id);
                    // Snapshot while holding the same lock as drain_chat. A
                    // message arriving after this point must invalidate the
                    // generation instead of being silently omitted from it.
                    let generation = conversation.start_generation(chat_id);
                    (late, generation)
                };
                events.extend(to_events(chat_id, late));
                respond(app, &events, generation).await?;
            }
        }
    }

    let snapshot = app.today_snapshot();
    match sleep::consolidate(app, snapshot.lines.clone(), false).await {
        Ok(fresh) if fresh.is_empty() => {
            app.finish_today(&snapshot, None)?;
        }
        Ok(_) => {}
        Err(error) => return Err(error),
    }
    Ok(())
}

fn to_events(chat_id: i64, messages: Vec<ConversationMessage>) -> Vec<Incoming> {
    messages
        .into_iter()
        .map(|message| Incoming {
            chat_id,
            sender_id: message.sender_id,
            message_id: message.message_id,
            sender: message.sender,
            username: message.username,
            timestamp: message.timestamp,
            metadata: message.metadata,
            text: message.text,
        })
        .collect()
}

fn message_block(
    chat_id: i64,
    sender: &str,
    username: Option<&str>,
    sender_id: i64,
    message_id: i64,
    timestamp: &str,
    metadata: &str,
    text: &str,
) -> String {
    let mut header = format!(
        "<message chat_id=\"{chat_id}\" sender=\"{}\" time=\"{}\"",
        escape_message_attribute(sender),
        escape_message_attribute(timestamp),
    );
    if let Some(username) = username
        .map(str::trim)
        .filter(|username| !username.is_empty())
    {
        header.push_str(" username=\"@");
        header.push_str(&escape_message_attribute(username.trim_start_matches('@')));
        header.push('\"');
    }
    if sender_id > 0 {
        header.push_str(&format!(" user_id=\"{sender_id}\""));
    }
    if message_id > 0 {
        header.push_str(&format!(" message_id=\"{message_id}\""));
    }
    let metadata = metadata.trim_end();
    let body = if metadata.is_empty() {
        text.to_string()
    } else {
        format!("{metadata}\n{text}")
    };
    format!("{header}>\n{body}\n</message>")
}

fn escape_message_attribute(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Drain the update stream forever, feeding incoming messages to the core. A
/// message pushes into the conversation buffer, wakes her from any nap, and
/// pulses the heartbeat loop to re-check.
async fn ingest(app: &Arc<App>, updates: &mut grammers_client::client::UpdateStream) {
    loop {
        match updates.next().await {
            Ok(Update::NewMessage(message) | Update::MessageEdited(message))
                if !message.outgoing() =>
            {
                let chat_id = message.peer_id().bot_api_dialog_id_unchecked();
                if !app.userbot.accepts_incoming(&message).await {
                    continue;
                }
                let incoming = app.userbot.describe(&message).await;
                app.record_event(&incoming);
                if app.userbot.is_broadcast_channel(chat_id).await {
                    continue;
                }
                // A new private message supersedes a reply that is still being
                // generated. Group messages stay queued for the next batch, so a
                // busy chat does not repeatedly cancel the current answer.
                if message.peer_id().kind() == PeerKind::User {
                    app.message_arrived(chat_id);
                }
                app.heartbeat.lock().unwrap().wake();
                app.wake.notify_one();
                let now_ms = app.monotonic_ms();
                app.conversation.lock().unwrap().push(
                    incoming.chat_id,
                    ConversationMessage {
                        sender_id: incoming.sender_id,
                        message_id: incoming.message_id,
                        sender: incoming.sender,
                        username: incoming.username,
                        timestamp: incoming.timestamp,
                        metadata: incoming.metadata,
                        text: incoming.text,
                    },
                    now_ms,
                );
                app.heartbeat.lock().unwrap().wake();
                app.wake.notify_one();
            }
            // A "typing…" from someone with a message already pending should hold
            // the batch open a little longer, so she doesn't cut into a thought
            // that is still being written.
            Ok(Update::Raw(raw)) => {
                if let tl::enums::Update::MessageReactions(reactions) = &raw.raw {
                    if let Some(chat_id) = PeerId::from(&reactions.peer).bot_api_dialog_id() {
                        if app.userbot.chat_is_in_contact_scope(chat_id).await {
                            if let Some(event) =
                                app.userbot.describe_reaction_update(reactions).await
                            {
                                app.record_event(&event);
                            }
                        }
                    }
                }
                if let tl::enums::Update::UserTyping(typing) = &raw.raw {
                    if let Some(chat_id) =
                        PeerId::user(typing.user_id).map(PeerId::bot_api_dialog_id_unchecked)
                    {
                        if app.userbot.chat_is_in_contact_scope(chat_id).await {
                            let now_ms = app.monotonic_ms();
                            if app
                                .conversation
                                .lock()
                                .unwrap()
                                .note_typing(chat_id, chat_id, now_ms)
                            {
                                app.wake.notify_one();
                            }
                        }
                    }
                }
            }
            Ok(_) => {}
            Err(error) => {
                eprintln!("update stream error, retrying: {error:#}");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

async fn run() -> Result<()> {
    let brain = Arc::new(Brain::from_env()?);
    // Kept alive for the whole run: dropping this stops a managed Ollama.
    let _ollama = ollama::start_if_managed(&brain.vision_model).await?;

    let mut diary = Diary::new(config::vault_dir());
    if !diary.open() {
        bail!("could not open vault at {:?}", config::vault_dir());
    }

    let api_id: i32 = config::env_or("TELEGRAM_API_ID", "0").parse().unwrap_or(0);
    let session_path = format!("{}.session", config::env_or("NEKORA_SESSION", "nekora"));
    let session = Arc::new(SqliteSession::open(&session_path).await?);
    let SenderPool {
        runner,
        updates,
        handle,
    } = SenderPool::new(Arc::clone(&session), api_id);
    let client = Client::new(handle);
    let pool_task = tokio::spawn(runner.run());

    userbot::login(&client).await?;
    let mut update_stream = client
        .stream_updates(updates, UpdatesConfiguration::default())
        .await
        .map_err(|error| anyhow::anyhow!("could not start update stream: {error}"))?;

    let userbot = Arc::new(Userbot::new(client, Arc::clone(&session), brain.clone()));
    let today = Today::open()?;
    let app = Arc::new(App::new(brain, userbot, diary, today));

    println!("nekora is up; waiting on her own clock");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => eprintln!("bye"),
        _ = async {
            tokio::join!(ingest(&app, &mut update_stream), heartbeat_loop(&app));
        } => {}
    }
    pool_task.abort();
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    config::load_env(".env");
    run().await
}

fn unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn today_str() -> String {
    Local::now().format("%Y-%m-%d").to_string()
}

fn now_stamp() -> String {
    Local::now().format("%Y-%m-%d %H:%M").to_string()
}
