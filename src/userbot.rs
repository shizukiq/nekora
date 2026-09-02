//! The Telegram side: a real userbot on a real account, via grammers (never the
//! Bot API).
//!
//! Purely hands — it receives, sends, and reports; it never decides. Incoming
//! media is flattened to text the brain can read (a photo becomes a caption, a
//! voice note a transcription), always keeping at least a label so nothing
//! arrives as empty text. Outgoing text is paced like a person typing, not a bot
//! blasting: a short "typing…" and a delay drawn from a words-per-minute band, so
//! a long line takes longer to land than a short one.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Result};
use grammers_client::media::Media;
use grammers_client::update::Message;
use grammers_client::{tl, Client};
use grammers_session::storages::SqliteSession;
use grammers_session::types::{PeerId, PeerRef};
use grammers_session::Session;
use rand::Rng;
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::brain::Brain;
use crate::config::env_or;
use crate::conversation::{split_message, ReplyGeneration};
use crate::App;

// Keep bubbles human-paced without making a reply feel stuck. The word-based
// estimate gives short, medium, and long chunks distinct bands while the bounds
// keep random jitter from producing an awkward outlier.
const BUBBLE_DELAY_MIN_MS: u64 = 500;
const BUBBLE_DELAY_MAX_MS: u64 = 3_000;
const BUBBLE_DELAY_PER_WORD_MS: u64 = 220;

// How many recent chats list_chats shows the model, and how much of each last line.
const RECENT_CHATS: usize = 20;
const LAST_LINE_CHARS: usize = 120;

// Telegram's per-message ceiling; the splitter breaks a long reply on this.
const MAX_BUBBLE_BYTES: usize = 4096;

// The markers we wrap incoming media in. A caption or a transcription is
// attacker-controlled text: it must not be able to forge these and smuggle
// instructions past the brain as if they were our labels.
const MEDIA_TOKENS: [&str; 8] = [
    "[photo]",
    "[voice message]",
    "[voice transcription]:",
    "[sticker]",
    "[video]",
    "[video note]",
    "[audio]",
    "[file]",
];

/// One incoming message, flattened to what the brain and the core need.
pub struct Incoming {
    pub chat_id: i64,
    pub sender_id: i64,
    pub message_id: i64,
    pub sender: String,
    pub username: Option<String>,
    pub text: String,
}

#[derive(Serialize)]
pub struct ChatSummary {
    pub id: i64,
    pub name: String,
    pub last: String,
}

pub struct Userbot {
    client: Client,
    // Shared with the sender pool: grammers caches every peer's access authority
    // here (and SqliteSession keeps it across restarts), which is how a bare chat
    // id gets turned back into something Telegram will accept.
    session: Arc<SqliteSession>,
    brain: Arc<Brain>,
    // A Bot-API chat id to the reference needed to message it, learned from every
    // incoming message and every listed dialog — grammers needs the peer's access
    // authority to send, which a bare id doesn't carry.
    peers: Mutex<HashMap<i64, PeerRef>>,
}

impl Userbot {
    pub fn new(client: Client, session: Arc<SqliteSession>, brain: Arc<Brain>) -> Self {
        Self {
            client,
            session,
            brain,
            peers: Mutex::new(HashMap::new()),
        }
    }

    /// Flatten an incoming message to text. Every non-text message yields at least
    /// a label ([photo], [voice message], …) so media never arrives empty — an
    /// empty turn makes her confabulate. Extracted content is appended when it
    /// works; when it fails the label alone remains and the real error goes to the
    /// operator, never into her context.
    pub async fn describe(&self, message: &Message) -> Incoming {
        let chat_id = message.peer_id().bot_api_dialog_id_unchecked();
        if let Ok(Some(peer_ref)) = message.peer_ref().await {
            self.peers.lock().unwrap().insert(chat_id, peer_ref);
        }

        let caption = clean(message.text());
        let media = match message.media() {
            Some(Media::Photo(_)) => Some(self.describe_photo(message).await),
            Some(Media::Sticker(_)) => Some("[sticker]".to_string()),
            Some(Media::Document(document)) => {
                Some(self.describe_document(message, document.mime_type()).await)
            }
            Some(Media::WebPage(_)) | None => None,
            Some(_) => Some("[media]".to_string()),
        };

        let text = match media {
            Some(media) if !caption.is_empty() => format!("{media}\n{caption}"),
            Some(media) => media,
            None => caption,
        };
        let sender_peer = message.sender();
        let sender = sender_peer
            .and_then(|peer| peer.name())
            .unwrap_or("someone")
            .to_string();
        let username = sender_peer
            .and_then(|peer| peer.username())
            .map(str::to_string);
        Incoming {
            chat_id,
            sender_id: message
                .sender_id()
                .and_then(|id| id.bot_api_dialog_id())
                .unwrap_or(0),
            message_id: message.id() as i64,
            sender,
            username,
            text,
        }
    }

    async fn describe_photo(&self, message: &Message) -> String {
        let mut label = "[photo]".to_string();
        match self.download(message).await {
            Ok(bytes) => match self.brain.caption_image(&bytes).await {
                Ok(caption) => {
                    label.push('\n');
                    label.push_str(&caption);
                }
                Err(error) => {
                    eprintln!("caption failed: {error:#}");
                    label.push_str(" (you glance at it but can't quite make it out right now)");
                }
            },
            Err(error) => {
                eprintln!("photo download failed: {error:#}");
                label.push_str(" (you glance at it but can't quite make it out right now)");
            }
        }
        label
    }

    // Voice notes are audio/ogg and can be transcribed with Premium; other audio is
    // music, video is video, everything else is just a file. The label alone is
    // enough for the turn when the finer detail isn't available.
    async fn describe_document(&self, message: &Message, mime: Option<&str>) -> String {
        let mime = mime.unwrap_or("");
        if mime.starts_with("video/") {
            "[video]".to_string()
        } else if mime == "audio/ogg" {
            let mut label = "[voice message]".to_string();
            match self.transcribe(message).await {
                Some(text) => {
                    label.push_str("\n[voice transcription]: ");
                    label.push_str(&clean(&text));
                }
                None => label.push_str(" (you can't quite catch it right now)"),
            }
            label
        } else if mime.starts_with("audio/") {
            "[audio]".to_string()
        } else {
            "[file]".to_string()
        }
    }

    async fn download(&self, message: &Message) -> Result<Vec<u8>> {
        let photo = message
            .photo()
            .ok_or_else(|| anyhow!("message has no photo"))?;
        let mut download = self.client.iter_download(&photo);
        let mut bytes = Vec::new();
        while let Some(chunk) = download.next().await? {
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }

    // Telegram Premium transcription. Returns text or None (no premium, backend
    // error, still pending); never an error string, so a failure reaches the
    // operator, not her context.
    async fn transcribe(&self, message: &Message) -> Option<String> {
        let peer_ref = message.peer_ref().await.ok()??;
        let request = tl::functions::messages::TranscribeAudio {
            peer: peer_ref.into(),
            msg_id: message.id(),
        };
        let tl::enums::messages::TranscribedAudio::Audio(audio) =
            self.client.invoke(&request).await.ok()?;
        let text = audio.text.trim();
        (!text.is_empty()).then(|| text.to_string())
    }

    /// Appear online while she is answering, so "typing…" reads as a person at
    /// the keyboard, not a bot poking an offline account. Best-effort.
    pub async fn go_online(&self) {
        let _ = self
            .client
            .invoke(&tl::functions::account::UpdateStatus { offline: false })
            .await;
    }

    /// Show a live "typing…" in `chat_id` for exactly as long as `fut` runs, then
    /// hand back its output. This ties the indicator to the real generation time
    /// instead of a fixed delay tacked on afterwards. If the peer can't be
    /// resolved the work still runs, just without the indicator.
    pub async fn keep_typing<T>(&self, chat_id: i64, fut: impl Future<Output = T>) -> T {
        let Ok(peer) = self.resolve(chat_id).await else {
            return fut.await;
        };
        // repeat() wants an Unpin future; boxing pins it.
        let (output, _) = self
            .client
            .action(peer)
            .repeat(|| tl::types::SendMessageTypingAction {}, Box::pin(fut))
            .await;
        output
    }

    /// Send one logical answer as sequential, human-sized Telegram bubbles. Each
    /// part gets its own typing indicator and short delay; an incoming message
    /// wakes that delay and makes the remaining parts stale.
    pub async fn send(
        &self,
        app: &App,
        chat_id: i64,
        text: &str,
        generation: Option<ReplyGeneration>,
    ) -> Result<()> {
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(());
        }
        let peer = self.resolve(chat_id).await?;
        let mut parts = split_message(text, MAX_BUBBLE_BYTES);
        if parts.is_empty() {
            parts.push(text.to_string());
        }
        for part in &parts {
            if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
                return Ok(());
            }

            let _ = self
                .client
                .action(peer)
                .oneshot(tl::types::SendMessageTypingAction {})
                .await;
            let delay = type_delay(part);
            let delay_finished = match generation {
                Some(generation) => app.wait_for_generation_delay(generation, delay).await,
                None => {
                    tokio::time::sleep(delay).await;
                    true
                }
            };
            if !delay_finished
                || generation.is_some_and(|generation| !app.generation_is_current(generation))
            {
                return Ok(());
            }

            // This is deliberately immediately before the network send. A new
            // message can arrive during typing or the delay above.
            if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
                return Ok(());
            }
            self.client.send_message(peer, part.as_str()).await?;
        }
        Ok(())
    }

    /// Mark the sender's chat read up to its latest message. Cosmetic, so a
    /// failure is swallowed rather than allowed to kill the turn.
    pub async fn mark_read(&self, chat_id: i64) {
        if let Ok(peer) = self.resolve(chat_id).await {
            let _ = self.client.mark_as_read(peer).await;
        }
    }

    /// The last handful of chats, so the model can choose who to talk to. Also
    /// caches each peer so a later send_message to it can resolve.
    pub async fn recent_chats(&self) -> Result<Vec<ChatSummary>> {
        let mut dialogs = self.client.iter_dialogs().limit(RECENT_CHATS);
        let mut out = Vec::new();
        while let Some(dialog) = dialogs.next().await? {
            let id = dialog.peer.id().bot_api_dialog_id_unchecked();
            self.peers.lock().unwrap().insert(id, dialog.peer_ref());
            let last = dialog
                .last_message
                .as_ref()
                .map(|message| message.text())
                .unwrap_or("")
                .chars()
                .take(LAST_LINE_CHARS)
                .collect();
            out.push(ChatSummary {
                id,
                name: dialog.peer.name().unwrap_or("").to_string(),
                last,
            });
        }
        Ok(out)
    }

    async fn resolve(&self, chat_id: i64) -> Result<PeerRef> {
        if let Some(peer) = self.peers.lock().unwrap().get(&chat_id).copied() {
            return Ok(peer);
        }
        let id = PeerId::from_bot_api_dialog_id(chat_id)
            .ok_or_else(|| anyhow!("not a valid chat id: {chat_id}"))?;
        // The session holds the access authority for every peer she has ever seen,
        // even across restarts; a bare id doesn't carry it, and without it Telegram
        // rejects the send with PEER_ID_INVALID. Only a peer the session has truly
        // never cached falls through to the ambient reference.
        if let Some(peer) = self.session.peer_ref(id).await.ok().flatten() {
            self.peers.lock().unwrap().insert(chat_id, peer);
            return Ok(peer);
        }
        Ok(id.to_ambient_ref())
    }
}

fn clean(text: &str) -> String {
    let text = text.trim();
    if MEDIA_TOKENS.iter().any(|token| text.contains(token)) {
        "(text withheld)".to_string()
    } else {
        text.to_string()
    }
}

fn type_delay(text: &str) -> Duration {
    let words = text.split_whitespace().count().max(1) as u64;
    let base_ms = words
        .saturating_mul(BUBBLE_DELAY_PER_WORD_MS)
        .clamp(BUBBLE_DELAY_MIN_MS, BUBBLE_DELAY_MAX_MS);
    let jitter = rand::rng().random_range(0.9..1.1);
    let milliseconds = ((base_ms as f64) * jitter).round() as u64;
    Duration::from_millis(milliseconds.clamp(BUBBLE_DELAY_MIN_MS, BUBBLE_DELAY_MAX_MS))
}

/// Log in interactively the first time, then never again — the session is
/// persisted, so this only prompts on a fresh account. Uses the account phone (a
/// real userbot, not a bot token).
pub async fn login(client: &Client) -> Result<()> {
    if client.is_authorized().await? {
        return Ok(());
    }
    let api_hash = env_or("TELEGRAM_API_HASH", "");
    let phone = match env_or("TELEGRAM_PHONE", "") {
        phone if !phone.is_empty() => phone,
        _ => prompt("phone (international format): ").await?,
    };
    let token = client.request_login_code(&phone, &api_hash).await?;
    let code = prompt("login code: ").await?;
    match client.sign_in(&token, &code).await {
        Ok(_) => Ok(()),
        Err(grammers_client::SignInError::PasswordRequired(password_token)) => {
            let password = prompt("2FA password: ").await?;
            client.check_password(password_token, password).await?;
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

async fn prompt(label: &str) -> Result<String> {
    let mut stdout = tokio::io::stdout();
    stdout.write_all(label.as_bytes()).await?;
    stdout.flush().await?;
    let mut line = String::new();
    BufReader::new(tokio::io::stdin())
        .read_line(&mut line)
        .await?;
    Ok(line.trim().to_string())
}
