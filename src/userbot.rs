use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use grammers_client::media::{Document, Downloadable, Media, PhotoSize, Sticker};
use grammers_client::message::{InputMessage, InputReactions, Message as TelegramMessage};
use grammers_client::peer::Peer;
use grammers_client::update::Message;
use grammers_client::{tl, Client};
use grammers_session::storages::SqliteSession;
use grammers_session::types::{PeerId, PeerKind, PeerRef};
use grammers_session::Session;
use rand::Rng;
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::brain::Brain;
use crate::config::{self, env_or};
use crate::conversation::{split_message, ReplyGeneration};
use crate::heartbeat::PresencePlan;
use crate::imagegen::GeneratedImage;
use crate::App;

const BUBBLE_DELAY_MIN_MS: u64 = 500;
const BUBBLE_DELAY_MAX_MS: u64 = 3_000;
const BUBBLE_DELAY_PER_WORD_MS: u64 = 220;

const RECENT_CHATS: usize = 20;
const LAST_LINE_CHARS: usize = 120;
const MAX_TEXT_DOCUMENT_BYTES: usize = 96 * 1024;
const MAX_MEDIA_BYTES: usize = 32 * 1024 * 1024;
const MAX_CONTEXT_ITEMS: usize = 16;
const MAX_CONTEXT_TEXT_CHARS: usize = 1_500;
const REPLY_LOOKUP_TIMEOUT: Duration = Duration::from_secs(5);

const MAX_BUBBLE_BYTES: usize = 4096;
const MAX_OUTGOING_CHARS: usize = 12_000;
const MAX_IMAGE_CAPTION_CHARS: usize = 1_024;

// Escaped before insertion so media text cannot forge its surrounding marker.
const MEDIA_TOKENS: [&str; 11] = [
    "[photo]",
    "[voice message]",
    "[voice transcription]:",
    "[sticker]",
    "[video]",
    "[video note]",
    "[audio]",
    "[file]",
    "[gif]",
    "[image]",
    "[preview frame]",
];

/// One incoming message, flattened to what the brain and the core need.
pub struct Incoming {
    pub chat_id: i64,
    pub sender_id: i64,
    pub message_id: i64,
    pub sender: String,
    pub username: Option<String>,
    pub timestamp: String,
    pub metadata: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct ChatSummary {
    pub id: i64,
    pub name: String,
    pub username: Option<String>,
    pub last: String,
}

#[derive(Serialize)]
pub struct UserInspection {
    pub user_id: i64,
    pub name: String,
    pub username: Option<String>,
    pub avatar_description: Option<String>,
}

#[derive(Serialize)]
pub struct OwnProfile {
    pub user_id: i64,
    pub name: String,
    pub username: Option<String>,
    pub bio: Option<String>,
    pub premium: bool,
    pub emoji_status_document_id: Option<i64>,
    pub profile_photo_count: usize,
    pub avatars: Vec<AvatarInspection>,
}

#[derive(Serialize)]
pub struct AvatarInspection {
    pub photo_id: i64,
    pub date: Option<i32>,
    pub description: String,
}

#[derive(Serialize)]
pub struct ReceivedGift {
    pub gift_id: i64,
    pub title: Option<String>,
    pub slug: Option<String>,
    pub number: Option<i32>,
    pub emoji: Option<String>,
    pub appearance: Vec<String>,
    pub from_user_id: Option<i64>,
    pub sender_hidden: bool,
    pub date: i32,
    pub message: Option<String>,
    pub saved_to_profile: bool,
    pub pinned: bool,
    pub refunded: bool,
}

#[derive(Serialize)]
pub struct ReceivedGifts {
    pub total: i32,
    pub next_offset: Option<String>,
    pub gifts: Vec<ReceivedGift>,
}

#[derive(Serialize)]
pub struct StickerSetSummary {
    pub set_id: i64,
    pub title: String,
    pub short_name: String,
    pub kind: &'static str,
    pub count: i32,
}

#[derive(Serialize)]
pub struct TelegramAsset {
    pub document_id: i64,
    pub emoji: String,
    pub kind: &'static str,
}

#[derive(Serialize)]
pub struct MediaInspection {
    pub kind: String,
    pub emoji: Option<String>,
    pub description: String,
}

#[derive(Serialize)]
pub struct CurrentTime {
    pub source: &'static str,
    pub unix: i64,
    pub utc_offset: &'static str,
    pub datetime: String,
}

#[derive(Default)]
struct Presence {
    active_turns: usize,
    revision: u64,
}

#[derive(Clone)]
struct CachedTelegramAsset {
    document: tl::enums::InputDocument,
    emoji: String,
    custom_emoji: bool,
}

pub struct Userbot {
    client: Client,
    account_user_id: i64,
    // SqliteSession preserves peer access hashes across restarts.
    session: Arc<SqliteSession>,
    brain: Arc<Brain>,
    peers: Mutex<HashMap<i64, PeerRef>>,
    private_contacts: Mutex<HashSet<i64>>,
    sticker_sets: Mutex<HashMap<i64, tl::enums::InputStickerSet>>,
    telegram_assets: Mutex<HashMap<i64, CachedTelegramAsset>>,
    presence: Mutex<Presence>,
}

impl Userbot {
    pub fn new(
        client: Client,
        account_user_id: i64,
        session: Arc<SqliteSession>,
        brain: Arc<Brain>,
    ) -> Self {
        Self {
            client,
            account_user_id,
            session,
            brain,
            peers: Mutex::new(HashMap::new()),
            private_contacts: Mutex::new(HashSet::new()),
            sticker_sets: Mutex::new(HashMap::new()),
            telegram_assets: Mutex::new(HashMap::new()),
            presence: Mutex::new(Presence::default()),
        }
    }

    /// Flatten an incoming message to text, retaining a label when media parsing fails.
    pub async fn describe(&self, message: &Message) -> Incoming {
        let chat_id = message.peer_id().bot_api_dialog_id_unchecked();
        if let Ok(Some(peer_ref)) = message.peer_ref().await {
            self.peers.lock().unwrap().insert(chat_id, peer_ref);
        }

        let caption = clean(message.text());
        let service_event = (!message.outgoing())
            .then(|| message.action())
            .flatten()
            .and_then(describe_gift_action);
        let media = match message.media() {
            Some(Media::Photo(_)) => Some(self.describe_photo(message).await),
            Some(Media::Sticker(sticker)) => Some(self.describe_sticker(&sticker).await),
            Some(Media::Document(document)) => {
                Some(self.describe_document(message, &document).await)
            }
            Some(Media::WebPage(_)) | None => None,
            Some(_) => Some("[media]".to_string()),
        };

        let text = match media {
            Some(media) if !caption.is_empty() => format!("{media}\n{caption}"),
            Some(media) => media,
            None if caption.is_empty() => service_event.unwrap_or_default(),
            None => caption,
        };
        let sender_peer = message.sender();
        let sender_id = message
            .sender_id()
            .and_then(|id| id.bot_api_dialog_id())
            .unwrap_or(0);
        if let Some(peer) = sender_peer {
            if let Ok(Some(peer_ref)) = peer.to_ref().await {
                self.peers
                    .lock()
                    .unwrap()
                    .insert(peer.id().bot_api_dialog_id_unchecked(), peer_ref);
            }
        }
        let sender = sender_peer
            .map(display_name)
            .unwrap_or_else(|| "someone".to_string());
        let username = sender_peer
            .and_then(|peer| peer.username())
            .map(str::to_string);
        let metadata = self.describe_message_context(message).await;
        Incoming {
            chat_id,
            sender_id,
            message_id: message.id() as i64,
            sender,
            username,
            timestamp: message
                .date()
                .with_timezone(&config::nekora_utc_offset())
                .to_rfc3339(),
            metadata,
            text,
        }
    }

    /// Accept private messages only from Telegram contacts.
    pub async fn accepts_incoming(&self, message: &Message) -> bool {
        if message.peer_id().kind() != PeerKind::User {
            return true;
        }
        let chat_id = message.peer_id().bot_api_dialog_id_unchecked();
        if let Ok(Some(peer_ref)) = message.peer_ref().await {
            self.peers.lock().unwrap().insert(chat_id, peer_ref);
        }
        let accepted = self.chat_is_in_contact_scope(chat_id).await;
        if accepted {
            self.private_contacts.lock().unwrap().insert(chat_id);
        }
        accepted
    }

    pub fn is_known_private_contact(&self, chat_id: i64) -> bool {
        self.private_contacts.lock().unwrap().contains(&chat_id)
    }

    /// Check a chat before a delayed or autonomous action reaches Telegram.
    pub async fn chat_is_in_contact_scope(&self, chat_id: i64) -> bool {
        if chat_id < 0 {
            return true;
        }
        self.resolve_contact_scoped_peer(chat_id).await.is_ok()
    }

    /// Record a reaction update without starting a new conversational turn.
    pub async fn describe_reaction_update(
        &self,
        update: &tl::types::UpdateMessageReactions,
    ) -> Option<Incoming> {
        let peer_id = PeerId::from(&update.peer);
        let chat_id = peer_id.bot_api_dialog_id()?;
        let (target, reactions, reaction_list) = if let Ok(peer_ref) = self.resolve(chat_id).await {
            let target = self
                .client
                .get_messages_by_id(peer_ref, &[update.msg_id])
                .await
                .ok()
                .and_then(|mut messages| messages.pop().flatten());
            let reactions = if message_reactions_is_min(&update.reactions) {
                self.fetch_full_message_reactions(peer_ref, update.msg_id)
                    .await
                    .unwrap_or_else(|| update.reactions.clone())
            } else {
                update.reactions.clone()
            };
            let reaction_list = self.fetch_reaction_list(peer_ref, update.msg_id).await;
            (target, reactions, reaction_list)
        } else {
            (None, update.reactions.clone(), None)
        };
        let reaction_summary =
            reaction_summary_with_actors(&reactions, reaction_list.as_deref(), chat_id);
        let mut metadata = format!(
            "telegram_context:\ntelegram_chat_type={}\ntelegram_message_reaction_update=true\ntelegram_reaction_message_id={}\ntelegram_reactions={reaction_summary}\n",
            chat_type_for_peer(peer_id),
            update.msg_id,
        );
        if let Some(top_message_id) = update.top_msg_id {
            metadata.push_str(&format!("telegram_reaction_topic_id={top_message_id}\n"));
        }
        metadata.push_str(&format!("telegram_reaction_target_chat_id={chat_id}\n"));
        if let Some(target) = target.as_ref() {
            let target_text = compact_context_text(target.text());
            if !target_text.is_empty() {
                metadata.push_str(&format!("telegram_reaction_target_text={target_text}\n"));
            }
            if target.outgoing() {
                metadata.push_str("telegram_reaction_target_outgoing=true\n");
            }
            if let Some(sender) = target.sender() {
                metadata.push_str(&format!(
                    "telegram_reaction_target_sender={}\n",
                    compact_context_text(&display_name(sender))
                ));
            }
        }
        Some(Incoming {
            chat_id,
            sender_id: 0,
            message_id: i64::from(update.msg_id),
            sender: "Telegram reactions".to_string(),
            username: None,
            timestamp: config::nekora_time().to_rfc3339(),
            metadata,
            text: format!("[Telegram reaction update on message_id={}]", update.msg_id),
        })
    }

    async fn describe_message_reactions(&self, message: &TelegramMessage) -> Option<String> {
        let reactions = message_reactions(message)?.clone();
        let chat_id = message.peer_id().bot_api_dialog_id()?;
        let peer_ref = self.resolve_contact_scoped_peer(chat_id).await.ok()?;
        let reactions = if message_reactions_is_min(&reactions) {
            self.fetch_full_message_reactions(peer_ref, message.id())
                .await
                .unwrap_or(reactions)
        } else {
            reactions
        };
        let reaction_list = self.fetch_reaction_list(peer_ref, message.id()).await;
        Some(reaction_summary_with_actors(
            &reactions,
            reaction_list.as_deref(),
            message.peer_id().bot_api_dialog_id_unchecked(),
        ))
    }

    async fn fetch_reply_target_reactions(&self, message: &TelegramMessage) -> Option<String> {
        let chat_id = message.peer_id().bot_api_dialog_id()?;
        let peer_ref = self.resolve_contact_scoped_peer(chat_id).await.ok()?;
        self.fetch_message_reaction_summary(peer_ref, message.id(), chat_id)
            .await
    }

    async fn fetch_reply_target_reactions_by_id(
        &self,
        message: &Message,
        message_id: i32,
    ) -> Option<String> {
        let chat_id = message.peer_id().bot_api_dialog_id()?;
        let peer_ref = self.resolve_contact_scoped_peer(chat_id).await.ok()?;
        self.fetch_message_reaction_summary(peer_ref, message_id, chat_id)
            .await
    }

    async fn fetch_message_reaction_summary(
        &self,
        peer: PeerRef,
        message_id: i32,
        chat_id: i64,
    ) -> Option<String> {
        let reactions = self.fetch_full_message_reactions(peer, message_id).await?;
        let reaction_list = self.fetch_reaction_list(peer, message_id).await;
        if !message_reactions_have_any(&reactions)
            && !reaction_list.as_ref().is_some_and(|list| !list.is_empty())
        {
            return None;
        }
        Some(reaction_summary_with_actors(
            &reactions,
            reaction_list.as_deref(),
            chat_id,
        ))
    }

    async fn fetch_full_message_reactions(
        &self,
        peer: PeerRef,
        message_id: i32,
    ) -> Option<tl::enums::MessageReactions> {
        let response = self
            .client
            .invoke(&tl::functions::messages::GetMessagesReactions {
                peer: peer.into(),
                id: vec![message_id],
            })
            .await
            .ok()?;
        let updates = match response {
            tl::enums::Updates::Updates(updates) => updates.updates,
            tl::enums::Updates::Combined(updates) => updates.updates,
            tl::enums::Updates::UpdateShort(update) => vec![update.update],
            _ => return None,
        };
        updates.into_iter().find_map(|update| match update {
            tl::enums::Update::MessageReactions(update) if update.msg_id == message_id => {
                Some(update.reactions)
            }
            _ => None,
        })
    }

    async fn fetch_reaction_list(
        &self,
        peer: PeerRef,
        message_id: i32,
    ) -> Option<Vec<tl::enums::MessagePeerReaction>> {
        let response = self
            .client
            .invoke(&tl::functions::messages::GetMessageReactionsList {
                peer: peer.into(),
                id: message_id,
                reaction: None,
                offset: None,
                limit: MAX_CONTEXT_ITEMS as i32,
            })
            .await
            .ok()?;
        let tl::enums::messages::MessageReactionsList::List(response) = response;
        Some(response.reactions)
    }

    async fn describe_message_context(&self, message: &Message) -> String {
        let mut lines = vec![format!("telegram_chat_type={}", chat_type(message))];
        if let Some(peer) = message.peer() {
            lines.push(format!(
                "telegram_chat_name={}",
                compact_context_text(&display_name(peer))
            ));
            if let Some(username) = peer.username() {
                lines.push(format!(
                    "telegram_chat_username=@{}",
                    compact_context_text(username)
                ));
            }
        }

        if message.mentioned() {
            lines.push(
                "telegram_addressed_to_account=true (mention or reply to Nekora)".to_string(),
            );
        }

        let mentions = mention_targets(message);
        if !mentions.is_empty() {
            lines.push(format!(
                "telegram_explicit_mentions={}",
                mentions.join(", ")
            ));
        }

        if let Some(header) = message.reply_header() {
            append_reply_header_context(&mut lines, &header);
        }

        if let Some(reply_message_id) = message.reply_to_message_id() {
            let mut reply_reactions_loaded = false;
            if let Ok(Ok(Some(reply))) =
                tokio::time::timeout(REPLY_LOOKUP_TIMEOUT, message.get_reply()).await
            {
                append_reply_target(&mut lines, &reply);
                if let Some(reactions) = self.fetch_reply_target_reactions(&reply).await {
                    lines.push(format!("telegram_reply_target_reactions={reactions}"));
                    reply_reactions_loaded = true;
                }
            }
            if !reply_reactions_loaded {
                if let Some(reactions) = self
                    .fetch_reply_target_reactions_by_id(message, reply_message_id)
                    .await
                {
                    lines.push(format!("telegram_reply_target_reactions={reactions}"));
                }
            }
        }

        if let Some(reactions) = self.describe_message_reactions(message).await {
            lines.push(format!("telegram_reactions={reactions}"));
        }
        if let Some(reply_count) = message.reply_count() {
            lines.push(format!("telegram_reply_count={reply_count}"));
        }
        if let Some(grouped_id) = message.grouped_id() {
            lines.push(format!("telegram_media_group_id={grouped_id}"));
        }
        if let Some(tl::enums::MessageFwdHeader::Header(header)) = message.forward_header() {
            lines.push("telegram_forwarded_by_sender=true".to_string());
            if let Some(origin) = header.from_id.as_ref() {
                let origin_id = PeerId::from(origin).bot_api_dialog_id_unchecked();
                lines.push(format!("telegram_forward_origin_peer_id={origin_id}"));
                if origin_id == self.account_user_id {
                    lines.push("telegram_forward_origin_is_nekora=true".to_string());
                }
            }
            if let Some(name) = header.from_name.as_deref() {
                lines.push(format!(
                    "telegram_forward_origin_name={}",
                    compact_context_text(name)
                ));
            }
            if let Some(author) = header.post_author.as_deref() {
                lines.push(format!(
                    "telegram_forward_origin_post_author={}",
                    compact_context_text(author)
                ));
            }
        }
        if let Some(post_author) = message.post_author() {
            lines.push(format!(
                "telegram_post_author={}",
                compact_context_text(post_author)
            ));
        }
        if let Some(via_bot_id) = message.via_bot_id() {
            lines.push(format!("telegram_via_bot_id={via_bot_id}"));
        }
        if message.pinned() {
            lines.push("telegram_pinned=true".to_string());
        }
        if message.edit_date().is_some() {
            lines.push("telegram_edited=true".to_string());
        }

        format!("telegram_context:\n{}\n", lines.join("\n"))
    }

    pub async fn is_broadcast_channel(&self, chat_id: i64) -> bool {
        let Ok(peer_ref) = self.resolve(chat_id).await else {
            return false;
        };
        matches!(
            self.client.resolve_peer(peer_ref).await,
            Ok(Peer::Channel(_))
        )
    }

    pub async fn current_time(&self) -> Result<CurrentTime> {
        let tl::enums::updates::State::State(state) = self
            .client
            .invoke(&tl::functions::updates::GetState {})
            .await?;
        let utc = DateTime::<Utc>::from_timestamp_secs(i64::from(state.date))
            .ok_or_else(|| anyhow!("Telegram returned an invalid server timestamp"))?;
        let offset = config::nekora_utc_offset();
        let datetime = utc
            .with_timezone(&offset)
            .format("%Y-%m-%d %H:%M:%S %:z")
            .to_string();
        Ok(CurrentTime {
            source: "telegram",
            unix: i64::from(state.date),
            utc_offset: "+04:00",
            datetime,
        })
    }

    pub async fn inspect_user(
        &self,
        user_id: Option<i64>,
        display_name: &str,
        username: Option<&str>,
    ) -> Result<UserInspection> {
        if let Some(user_id) = user_id {
            if user_id <= 0 {
                return Err(anyhow!("user_id must be positive"));
            }
        }
        let username = username
            .map(str::trim)
            .filter(|username| !username.is_empty())
            .map(|username| username.trim_start_matches('@'))
            .filter(|username| !username.is_empty());
        let peer = if let Some(username) = username {
            let peer = self
                .client
                .resolve_username(username)
                .await?
                .ok_or_else(|| anyhow!("username not found: @{username}"))?;
            if let Some(expected_id) = user_id {
                let actual_id = match &peer {
                    Peer::User(user) => user.id().bot_api_dialog_id_unchecked(),
                    _ => return Err(anyhow!("target is not a user")),
                };
                if actual_id != expected_id {
                    return Err(anyhow!("user_id and username refer to different users"));
                }
            }
            peer
        } else {
            let user_id = user_id.ok_or_else(|| anyhow!("missing user_id or username"))?;
            let id = PeerId::user_unchecked(user_id).bot_api_dialog_id_unchecked();
            let peer_ref = self.resolve(id).await?;
            self.client.resolve_peer(peer_ref).await?
        };

        let Peer::User(user) = &peer else {
            return Err(anyhow!("target is not a user"));
        };
        let actual_name = user.full_name();
        let name = if actual_name.is_empty() && !display_name.trim().is_empty() {
            display_name.trim().to_string()
        } else {
            actual_name
        };
        let avatar_description = match peer.photo(true).await {
            Ok(Some(photo)) => match self.download_media(&photo).await {
                Ok(bytes) => match self.brain.caption_image(&bytes).await {
                    Ok(description) => Some(description),
                    Err(error) => {
                        eprintln!("avatar caption failed: {error:#}");
                        Some("profile photo exists, but could not be viewed right now".to_string())
                    }
                },
                Err(error) => {
                    eprintln!("avatar download failed: {error:#}");
                    Some("profile photo exists, but could not be viewed right now".to_string())
                }
            },
            Ok(None) => None,
            Err(error) => {
                eprintln!("profile photo lookup failed: {error:#}");
                None
            }
        };
        Ok(UserInspection {
            user_id: peer.id().bot_api_dialog_id_unchecked(),
            name,
            username: user.username().map(str::to_string),
            avatar_description,
        })
    }

    pub async fn inspect_own_profile(&self, avatar_limit: usize) -> Result<OwnProfile> {
        if !(1..=4).contains(&avatar_limit) {
            return Err(anyhow!("avatar_limit must be between 1 and 4"));
        }
        let me = self.client.get_me().await?;
        let full: tl::types::users::UserFull = self
            .client
            .invoke(&tl::functions::users::GetFullUser {
                id: tl::enums::InputUser::UserSelf,
            })
            .await?
            .into();
        let tl::enums::UserFull::Full(full_user) = full.full_user;
        let raw_me = full.users.into_iter().find_map(|user| match user {
            tl::enums::User::User(user) if user.id == self.account_user_id => Some(user),
            _ => None,
        });
        let premium = raw_me.as_ref().is_some_and(|user| user.premium);
        let emoji_status_document_id = raw_me
            .as_ref()
            .and_then(|user| user.emoji_status.as_ref())
            .and_then(emoji_status_document_id);

        let photos = self
            .client
            .invoke(&tl::functions::photos::GetUserPhotos {
                user_id: tl::enums::InputUser::UserSelf,
                offset: 0,
                max_id: 0,
                limit: avatar_limit as i32,
            })
            .await?;
        let (profile_photo_count, photos) = match photos {
            tl::enums::photos::Photos::Photos(photos) => (photos.photos.len(), photos.photos),
            tl::enums::photos::Photos::Slice(photos) => {
                (photos.count.max(0) as usize, photos.photos)
            }
        };
        let mut avatars = Vec::with_capacity(photos.len());
        for raw in photos {
            let (photo_id, date) = match &raw {
                tl::enums::Photo::Photo(photo) => (photo.id, Some(photo.date)),
                tl::enums::Photo::Empty(photo) => (photo.id, None),
            };
            let photo = grammers_client::media::Photo::from_raw(raw);
            let description = match self.download_media(&photo).await {
                Ok(bytes) => self
                    .brain
                    .caption_image(&bytes)
                    .await
                    .unwrap_or_else(|error| {
                        eprintln!("own avatar caption failed: {error:#}");
                        "profile photo exists, but could not be viewed right now".to_string()
                    }),
                Err(error) => {
                    eprintln!("own avatar download failed: {error:#}");
                    "profile photo exists, but could not be viewed right now".to_string()
                }
            };
            avatars.push(AvatarInspection {
                photo_id,
                date,
                description,
            });
        }

        Ok(OwnProfile {
            user_id: self.account_user_id,
            name: me.full_name(),
            username: me.username().map(str::to_string),
            bio: full_user.about,
            premium,
            emoji_status_document_id,
            profile_photo_count,
            avatars,
        })
    }

    pub async fn received_gifts(&self, offset: &str, limit: usize) -> Result<ReceivedGifts> {
        if !(1..=20).contains(&limit) {
            return Err(anyhow!("limit must be between 1 and 20"));
        }
        let response: tl::types::payments::SavedStarGifts = self
            .client
            .invoke(&tl::functions::payments::GetSavedStarGifts {
                exclude_unsaved: false,
                exclude_saved: false,
                exclude_unlimited: false,
                exclude_unique: false,
                sort_by_value: false,
                exclude_upgradable: false,
                exclude_unupgradable: false,
                peer_color_available: false,
                exclude_hosted: false,
                peer: tl::enums::InputPeer::PeerSelf,
                collection_id: None,
                offset: offset.to_string(),
                limit: limit as i32,
            })
            .await?
            .into();

        let total = response.count;
        let next_offset = response.next_offset;
        let gifts = response
            .gifts
            .into_iter()
            .map(|gift| {
                let tl::enums::SavedStarGift::Gift(gift) = gift;
                let (gift_id, title, slug, number, emoji, appearance) = match gift.gift {
                    tl::enums::StarGift::Gift(star_gift) => (
                        star_gift.id,
                        star_gift.title,
                        None,
                        None,
                        document_emoji(&star_gift.sticker),
                        Vec::new(),
                    ),
                    tl::enums::StarGift::Unique(star_gift) => {
                        let appearance = star_gift
                            .attributes
                            .iter()
                            .filter_map(|attribute| match attribute {
                                tl::enums::StarGiftAttribute::Model(attribute) => {
                                    Some(format!("model: {}", attribute.name))
                                }
                                tl::enums::StarGiftAttribute::Pattern(attribute) => {
                                    Some(format!("pattern: {}", attribute.name))
                                }
                                tl::enums::StarGiftAttribute::Backdrop(attribute) => {
                                    Some(format!("backdrop: {}", attribute.name))
                                }
                                tl::enums::StarGiftAttribute::OriginalDetails(_) => None,
                            })
                            .collect();
                        (
                            star_gift.id,
                            Some(star_gift.title),
                            Some(star_gift.slug),
                            Some(star_gift.num),
                            None,
                            appearance,
                        )
                    }
                };
                ReceivedGift {
                    gift_id,
                    title,
                    slug,
                    number,
                    emoji,
                    appearance,
                    from_user_id: gift.from_id.as_ref().and_then(raw_user_id),
                    sender_hidden: gift.name_hidden,
                    date: gift.date,
                    message: gift.message.map(|message| {
                        let tl::enums::TextWithEntities::Entities(message) = message;
                        message.text
                    }),
                    saved_to_profile: !gift.unsaved,
                    pinned: gift.pinned_to_top,
                    refunded: gift.refunded,
                }
            })
            .collect();
        Ok(ReceivedGifts {
            total,
            next_offset,
            gifts,
        })
    }

    pub async fn sticker_sets(&self, custom_emoji: bool) -> Result<Vec<StickerSetSummary>> {
        let response = if custom_emoji {
            self.client
                .invoke(&tl::functions::messages::GetEmojiStickers { hash: 0 })
                .await?
        } else {
            self.client
                .invoke(&tl::functions::messages::GetAllStickers { hash: 0 })
                .await?
        };
        let tl::enums::messages::AllStickers::Stickers(response) = response else {
            return Ok(Vec::new());
        };
        let kind = if custom_emoji {
            "custom_emoji"
        } else {
            "sticker"
        };
        let mut cache = self.sticker_sets.lock().unwrap();
        Ok(response
            .sets
            .into_iter()
            .take(50)
            .map(|set| {
                let tl::enums::StickerSet::Set(set) = set;
                cache.insert(
                    set.id,
                    tl::types::InputStickerSetId {
                        id: set.id,
                        access_hash: set.access_hash,
                    }
                    .into(),
                );
                StickerSetSummary {
                    set_id: set.id,
                    title: set.title,
                    short_name: set.short_name,
                    kind,
                    count: set.count,
                }
            })
            .collect())
    }

    pub async fn stickers_in_set(
        &self,
        set_id: i64,
        emoji: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TelegramAsset>> {
        if !(1..=50).contains(&limit) {
            return Err(anyhow!("limit must be between 1 and 50"));
        }
        let sticker_set = self
            .sticker_sets
            .lock()
            .unwrap()
            .get(&set_id)
            .cloned()
            .ok_or_else(|| anyhow!("list sticker sets before opening one"))?;
        let response = self
            .client
            .invoke(&tl::functions::messages::GetStickerSet {
                stickerset: sticker_set,
                hash: 0,
            })
            .await?;
        let tl::enums::messages::StickerSet::Set(response) = response else {
            return Ok(Vec::new());
        };
        let wanted = emoji.map(str::trim).filter(|emoji| !emoji.is_empty());
        let mut assets = Vec::new();
        let mut cache = self.telegram_assets.lock().unwrap();
        for document in response.documents {
            let tl::enums::Document::Document(document) = document else {
                continue;
            };
            let Some((asset_emoji, custom_emoji)) = document_asset_kind(&document) else {
                continue;
            };
            if wanted.is_some_and(|wanted| !asset_emoji.contains(wanted)) {
                continue;
            }
            let input = tl::types::InputDocument {
                id: document.id,
                access_hash: document.access_hash,
                file_reference: document.file_reference,
            }
            .into();
            cache.insert(
                document.id,
                CachedTelegramAsset {
                    document: input,
                    emoji: asset_emoji.clone(),
                    custom_emoji,
                },
            );
            assets.push(TelegramAsset {
                document_id: document.id,
                emoji: asset_emoji,
                kind: if custom_emoji {
                    "custom_emoji"
                } else {
                    "sticker"
                },
            });
            if assets.len() == limit {
                break;
            }
        }
        Ok(assets)
    }

    pub async fn find_custom_emojis(
        &self,
        emoji: &str,
        limit: usize,
    ) -> Result<Vec<TelegramAsset>> {
        if emoji.trim().is_empty() {
            return Err(anyhow!("emoji must not be empty"));
        }
        if !(1..=20).contains(&limit) {
            return Err(anyhow!("limit must be between 1 and 20"));
        }
        let ids = match self
            .client
            .invoke(&tl::functions::messages::SearchCustomEmoji {
                emoticon: emoji.trim().to_string(),
                hash: 0,
            })
            .await?
        {
            tl::enums::EmojiList::List(list) => list.document_id,
            tl::enums::EmojiList::NotModified => Vec::new(),
        };
        let documents = self
            .client
            .invoke(&tl::functions::messages::GetCustomEmojiDocuments {
                document_id: ids.into_iter().take(limit).collect(),
            })
            .await?;
        let mut assets = Vec::new();
        let mut cache = self.telegram_assets.lock().unwrap();
        for document in documents {
            let tl::enums::Document::Document(document) = document else {
                continue;
            };
            let Some((asset_emoji, true)) = document_asset_kind(&document) else {
                continue;
            };
            let input = tl::types::InputDocument {
                id: document.id,
                access_hash: document.access_hash,
                file_reference: document.file_reference,
            }
            .into();
            cache.insert(
                document.id,
                CachedTelegramAsset {
                    document: input,
                    emoji: asset_emoji.clone(),
                    custom_emoji: true,
                },
            );
            assets.push(TelegramAsset {
                document_id: document.id,
                emoji: asset_emoji,
                kind: "custom_emoji",
            });
        }
        Ok(assets)
    }

    pub async fn inspect_message_media(
        &self,
        chat_id: i64,
        message_id: i64,
    ) -> Result<MediaInspection> {
        let message_id = i32::try_from(message_id)
            .map_err(|_| anyhow!("message_id is outside Telegram's range"))?;
        let peer = self.resolve_contact_scoped_peer(chat_id).await?;
        let message = self
            .client
            .get_messages_by_id(peer, &[message_id])
            .await?
            .into_iter()
            .next()
            .flatten()
            .ok_or_else(|| anyhow!("message not found"))?;
        let media = message
            .media()
            .ok_or_else(|| anyhow!("message has no inspectable media"))?;
        let (kind, emoji, bytes) = match &media {
            Media::Photo(photo) => ("photo", None, self.download_media(photo).await?),
            Media::Sticker(sticker) => {
                let bytes = self.download_sticker_image(sticker).await?;
                ("sticker", Some(sticker.emoji().to_string()), bytes)
            }
            Media::Document(document) => {
                let kind = visual_document_kind(document)
                    .ok_or_else(|| anyhow!("document has no inspectable visual content"))?;
                let bytes = self.download_visual_document(document, kind).await?;
                (kind, None, bytes)
            }
            _ => return Err(anyhow!("message has no inspectable visual content")),
        };
        let description = self.brain.caption_image(&bytes).await?;
        Ok(MediaInspection {
            kind: kind.to_string(),
            emoji,
            description,
        })
    }

    async fn describe_photo(&self, message: &Message) -> String {
        let Some(photo) = message.photo() else {
            return "[photo] (you glance at it but can't quite make it out right now)".to_string();
        };
        self.describe_image_source("[photo]", &photo).await
    }

    async fn describe_sticker(&self, sticker: &Sticker) -> String {
        let emoji = clean(sticker.emoji());
        let label = if emoji.is_empty() {
            "[sticker]".to_string()
        } else {
            format!("[sticker] emoji: {emoji}")
        };
        if let Some(thumbnail) = largest_thumbnail(&sticker.document) {
            self.describe_image_source(&label, &thumbnail).await
        } else {
            self.describe_image_source(&label, &sticker.document).await
        }
    }

    async fn describe_document(&self, message: &Message, document: &Document) -> String {
        if let Some(kind) = visual_document_kind(document) {
            return self.describe_visual_document(document, kind).await;
        }
        let mime = document.mime_type().unwrap_or("");
        let extension = document
            .name()
            .and_then(|name| name.rsplit_once('.'))
            .map(|(_, extension)| extension);
        let is_text = mime.starts_with("text/")
            || extension.is_some_and(|extension| {
                extension.eq_ignore_ascii_case("txt")
                    || extension.eq_ignore_ascii_case("md")
                    || extension.eq_ignore_ascii_case("markdown")
            });
        if is_text {
            let mut download = self.client.iter_download(document);
            let mut bytes = Vec::new();
            while bytes.len() < MAX_TEXT_DOCUMENT_BYTES {
                let chunk = match download.next().await {
                    Ok(Some(chunk)) => chunk,
                    Ok(None) => break,
                    Err(_) => return "[text file] (couldn't read it right now)".to_string(),
                };
                let remaining = MAX_TEXT_DOCUMENT_BYTES - bytes.len();
                bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
            }
            let Ok(mut text) = String::from_utf8(bytes) else {
                return "[text file] (couldn't decode it right now)".to_string();
            };
            if text.trim().is_empty() {
                return "[text file] (empty)".to_string();
            }
            if document
                .size()
                .is_some_and(|size| size > MAX_TEXT_DOCUMENT_BYTES)
            {
                text.push_str("\n[text file truncated at 96 KiB]");
            }
            return format!("[text file]\n{text}");
        }
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

    async fn describe_visual_document(&self, document: &Document, kind: &str) -> String {
        let label = visual_label(document, kind);
        if matches!(kind, "gif" | "video") {
            let Some(thumbnail) = largest_thumbnail(document) else {
                return format!("{label}\n(no preview frame was attached)");
            };
            return self.describe_image_source(&label, &thumbnail).await;
        }
        self.describe_image_source(&label, document).await
    }

    async fn describe_image_source<D: Downloadable>(&self, label: &str, source: &D) -> String {
        let mut label = label.to_string();
        match self.download_media(source).await {
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
                eprintln!("media download failed: {error:#}");
                label.push_str(" (you glance at it but can't quite make it out right now)");
            }
        }
        label
    }

    async fn download_sticker_image(&self, sticker: &Sticker) -> Result<Vec<u8>> {
        if let Some(thumbnail) = largest_thumbnail(&sticker.document) {
            self.download_media(&thumbnail).await
        } else {
            self.download_media(&sticker.document).await
        }
    }

    async fn download_visual_document(&self, document: &Document, kind: &str) -> Result<Vec<u8>> {
        if matches!(kind, "gif" | "video") {
            let thumbnail = largest_thumbnail(document)
                .ok_or_else(|| anyhow!("visual document has no preview thumbnail"))?;
            self.download_media(&thumbnail).await
        } else {
            self.download_media(document).await
        }
    }

    async fn download_media<D: Downloadable>(&self, media: &D) -> Result<Vec<u8>> {
        let mut download = self.client.iter_download(media);
        let mut bytes = Vec::new();
        while let Some(chunk) = download.next().await? {
            if bytes.len().saturating_add(chunk.len()) > MAX_MEDIA_BYTES {
                return Err(anyhow!("media exceeds {MAX_MEDIA_BYTES} bytes"));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }

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

    async fn go_online(&self) {
        let _ = self
            .client
            .invoke(&tl::functions::account::UpdateStatus { offline: false })
            .await;
    }

    async fn go_offline(&self) {
        let _ = self
            .client
            .invoke(&tl::functions::account::UpdateStatus { offline: true })
            .await;
    }

    async fn online_refresh_period(&self) -> Option<Duration> {
        let tl::enums::Config::Config(config) = self
            .client
            .invoke(&tl::functions::help::GetConfig {})
            .await
            .ok()?;
        u64::try_from(config.online_update_period_ms)
            .ok()
            .filter(|period| *period > 0)
            .map(Duration::from_millis)
    }

    fn presence_is_idle(&self, revision: u64) -> bool {
        let presence = self.presence.lock().unwrap();
        presence.active_turns == 0 && presence.revision == revision
    }

    pub async fn stay_online<T>(
        self: &Arc<Self>,
        plan: PresencePlan,
        future: impl Future<Output = T>,
    ) -> T {
        {
            let mut presence = self.presence.lock().unwrap();
            presence.active_turns += 1;
            presence.revision = presence.revision.wrapping_add(1);
        }
        self.go_online().await;
        let refresh = self.online_refresh_period().await;
        tokio::pin!(future);
        let output = match refresh {
            Some(period) => loop {
                tokio::select! {
                    output = &mut future => break output,
                    _ = tokio::time::sleep(period) => self.go_online().await,
                }
            },
            None => future.await,
        };

        let idle_revision = {
            let mut presence = self.presence.lock().unwrap();
            presence.active_turns -= 1;
            if presence.active_turns == 0 {
                presence.revision = presence.revision.wrapping_add(1);
                Some(presence.revision)
            } else {
                None
            }
        };
        if let Some(revision) = idle_revision {
            self.go_offline().await;
            if let Some(delay) = plan.idle_return_after {
                let userbot = Arc::clone(self);
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    if !userbot.presence_is_idle(revision) {
                        return;
                    }
                    userbot.go_online().await;
                    tokio::time::sleep(plan.idle_return_for).await;
                    if userbot.presence_is_idle(revision) {
                        userbot.go_offline().await;
                    }
                });
            }
        }
        output
    }

    pub async fn keep_typing<T>(&self, chat_id: i64, fut: impl Future<Output = T>) -> T {
        let Ok(peer) = self.resolve_contact_scoped_peer(chat_id).await else {
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

    pub async fn send(
        &self,
        app: &App,
        chat_id: i64,
        text: &str,
        reply_to_message_id: Option<i64>,
        generation: Option<ReplyGeneration>,
    ) -> Result<()> {
        if text.contains("DSML") && text.contains("invoke name=\"") {
            return Err(anyhow!("refusing to send internal tool-call markup"));
        }
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(());
        }
        let peer = self.resolve_contact_scoped_peer(chat_id).await?;
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(());
        }
        let telegram_reply_to_message_id = reply_to_message_id
            .map(|message_id| {
                i32::try_from(message_id)
                    .map_err(|_| anyhow!("reply_to_message_id is outside Telegram's range"))
            })
            .transpose()?;
        let text: String = text.chars().take(MAX_OUTGOING_CHARS).collect();
        let mut parts = split_message(&text, MAX_BUBBLE_BYTES);
        if parts.is_empty() {
            parts.push(text);
        }
        for (index, part) in parts.iter().enumerate() {
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

            // Recheck after typing and delay, immediately before the send.
            if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
                return Ok(());
            }
            let message = InputMessage::new().text(part.as_str()).reply_to(
                (index == 0)
                    .then_some(telegram_reply_to_message_id)
                    .flatten(),
            );
            self.client.send_message(peer, message).await?;
            app.record_outgoing(
                chat_id,
                part,
                if index == 0 {
                    reply_to_message_id
                } else {
                    None
                },
            );
        }
        Ok(())
    }

    pub async fn send_image(
        &self,
        app: &App,
        chat_id: i64,
        image: GeneratedImage,
        caption: &str,
        reply_to_message_id: Option<i64>,
        generation: Option<ReplyGeneration>,
    ) -> Result<()> {
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(());
        }
        let peer = self.resolve_contact_scoped_peer(chat_id).await?;
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(());
        }
        let telegram_reply_to_message_id = reply_to_message_id
            .map(|message_id| {
                i32::try_from(message_id)
                    .map_err(|_| anyhow!("reply_to_message_id is outside Telegram's range"))
            })
            .transpose()?;
        let image_len = image.bytes.len();
        let mut stream = Cursor::new(image.bytes);
        let uploaded = self
            .client
            .upload_stream(&mut stream, image_len, image.filename)
            .await?;
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(());
        }
        let caption: String = caption.chars().take(MAX_IMAGE_CAPTION_CHARS).collect();
        let message = InputMessage::new()
            .text(caption.as_str())
            .photo(uploaded)
            .reply_to(telegram_reply_to_message_id);
        self.client.send_message(peer, message).await?;
        app.record_outgoing(
            chat_id,
            if caption.is_empty() {
                "[generated image]"
            } else {
                &caption
            },
            reply_to_message_id,
        );
        Ok(())
    }

    pub async fn send_sticker(
        &self,
        app: &App,
        chat_id: i64,
        document_id: i64,
        reply_to_message_id: Option<i64>,
        generation: Option<ReplyGeneration>,
    ) -> Result<()> {
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(());
        }
        let asset = self
            .telegram_assets
            .lock()
            .unwrap()
            .get(&document_id)
            .cloned()
            .ok_or_else(|| anyhow!("list the sticker before sending it"))?;
        if asset.custom_emoji {
            return Err(anyhow!(
                "document_id belongs to a custom emoji, not a sticker"
            ));
        }
        let peer = self.resolve_contact_scoped_peer(chat_id).await?;
        let reply_to = reply_to_message_id
            .map(|message_id| {
                i32::try_from(message_id)
                    .map_err(|_| anyhow!("reply_to_message_id is outside Telegram's range"))
            })
            .transpose()?;
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(());
        }
        let message = InputMessage::new()
            .media(tl::types::InputMediaDocument {
                spoiler: false,
                id: asset.document,
                video_cover: None,
                video_timestamp: None,
                ttl_seconds: None,
                query: None,
            })
            .reply_to(reply_to);
        self.client.send_message(peer, message).await?;
        app.record_outgoing(
            chat_id,
            &format!("[sticker] {}", asset.emoji),
            reply_to_message_id,
        );
        Ok(())
    }

    pub async fn send_custom_emoji(
        &self,
        app: &App,
        chat_id: i64,
        document_id: i64,
        emoji: &str,
        reply_to_message_id: Option<i64>,
        generation: Option<ReplyGeneration>,
    ) -> Result<()> {
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(());
        }
        let asset = self
            .telegram_assets
            .lock()
            .unwrap()
            .get(&document_id)
            .cloned()
            .ok_or_else(|| anyhow!("find or list the custom emoji before sending it"))?;
        if !asset.custom_emoji {
            return Err(anyhow!(
                "document_id belongs to a sticker, not a custom emoji"
            ));
        }
        let emoji = emoji.trim();
        if emoji != asset.emoji {
            return Err(anyhow!("emoji must match the selected custom emoji"));
        }
        let length = i32::try_from(emoji.encode_utf16().count())
            .map_err(|_| anyhow!("emoji is too long"))?;
        let peer = self.resolve_contact_scoped_peer(chat_id).await?;
        let reply_to = reply_to_message_id
            .map(|message_id| {
                i32::try_from(message_id)
                    .map_err(|_| anyhow!("reply_to_message_id is outside Telegram's range"))
            })
            .transpose()?;
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(());
        }
        let message = InputMessage::new()
            .text(emoji)
            .fmt_entities([tl::types::MessageEntityCustomEmoji {
                offset: 0,
                length,
                document_id,
            }
            .into()])
            .reply_to(reply_to);
        self.client.send_message(peer, message).await?;
        app.record_outgoing(
            chat_id,
            &format!("[custom emoji] {emoji}"),
            reply_to_message_id,
        );
        Ok(())
    }

    pub async fn react(
        &self,
        app: &App,
        chat_id: i64,
        message_id: i64,
        reaction: &str,
        generation: Option<ReplyGeneration>,
    ) -> Result<bool> {
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(false);
        }
        let telegram_message_id = i32::try_from(message_id)
            .map_err(|_| anyhow!("message_id is outside Telegram's range"))?;
        let peer = self.resolve_contact_scoped_peer(chat_id).await?;
        if generation.is_some_and(|generation| !app.generation_is_current(generation)) {
            return Ok(false);
        }
        let reaction = reaction.trim();
        if reaction.is_empty() {
            self.client
                .send_reactions(peer, telegram_message_id, InputReactions::remove())
                .await?;
        } else {
            self.client
                .send_reactions(peer, telegram_message_id, reaction_input(reaction)?)
                .await?;
        }
        app.record_reaction(chat_id, message_id, reaction);
        Ok(true)
    }

    pub async fn mark_read(&self, chat_id: i64) {
        if let Ok(peer) = self.resolve_contact_scoped_peer(chat_id).await {
            let _ = self.client.mark_as_read(peer).await;
        }
    }

    pub async fn recent_chats(&self) -> Result<Vec<ChatSummary>> {
        let mut dialogs = self.client.iter_dialogs();
        let mut out = Vec::new();
        while let Some(dialog) = dialogs.next().await? {
            let id = dialog.peer.id().bot_api_dialog_id_unchecked();
            if !peer_is_in_contact_scope(&dialog.peer) {
                continue;
            }
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
                name: display_name(&dialog.peer),
                username: dialog.peer.username().map(str::to_string),
                last,
            });
            if out.len() == RECENT_CHATS {
                break;
            }
        }
        Ok(out)
    }

    async fn resolve_contact_scoped_peer(&self, chat_id: i64) -> Result<PeerRef> {
        let peer_ref = self.resolve(chat_id).await?;
        if chat_id > 0 {
            let peer = self.client.resolve_peer(peer_ref).await?;
            if !peer_is_in_contact_scope(&peer) {
                return Err(anyhow!("private chat is outside Telegram contacts"));
            }
        }
        Ok(peer_ref)
    }

    async fn resolve(&self, chat_id: i64) -> Result<PeerRef> {
        if let Some(peer) = self.peers.lock().unwrap().get(&chat_id).copied() {
            return Ok(peer);
        }
        let id = PeerId::from_bot_api_dialog_id(chat_id)
            .ok_or_else(|| anyhow!("not a valid chat id: {chat_id}"))?;
        // A bare peer id cannot replace the access hash cached by the session.
        if let Some(peer) = self.session.peer_ref(id).await.ok().flatten() {
            self.peers.lock().unwrap().insert(chat_id, peer);
            return Ok(peer);
        }
        Ok(id.to_ambient_ref())
    }
}

fn chat_type(message: &Message) -> &'static str {
    match message.peer_id().kind() {
        PeerKind::User => "private",
        PeerKind::Chat => "group",
        PeerKind::Channel => match message.peer() {
            Some(Peer::Channel(_)) => "broadcast_channel",
            _ => "supergroup_or_channel",
        },
    }
}

fn peer_is_in_contact_scope(peer: &Peer) -> bool {
    match peer {
        Peer::User(user) => user.contact(),
        Peer::Group(_) | Peer::Channel(_) => true,
    }
}

fn reaction_input(reaction: &str) -> Result<InputReactions> {
    if let Some(document_id) = reaction.strip_prefix("custom_emoji:") {
        let document_id = document_id
            .trim()
            .parse::<i64>()
            .map_err(|_| anyhow!("custom emoji reaction must contain a document id"))?;
        if document_id <= 0 {
            return Err(anyhow!("custom emoji document id must be positive"));
        }
        return Ok(InputReactions::custom_emoji(document_id));
    }
    Ok(InputReactions::emoticon(reaction))
}

fn chat_type_for_peer(peer_id: PeerId) -> &'static str {
    match peer_id.kind() {
        PeerKind::User => "private",
        PeerKind::Chat => "group",
        PeerKind::Channel => "supergroup_or_channel",
    }
}

fn append_reply_header_context(lines: &mut Vec<String>, header: &tl::enums::MessageReplyHeader) {
    match header {
        tl::enums::MessageReplyHeader::Header(header) => {
            if let Some(message_id) = header.reply_to_msg_id {
                lines.push(format!("telegram_reply_to_message_id={message_id}"));
            }
            if let Some(top_message_id) = header.reply_to_top_id {
                lines.push(format!("telegram_reply_to_topic_id={top_message_id}"));
            }
            if let Some(peer_id) = header
                .reply_to_peer_id
                .as_ref()
                .and_then(|peer| PeerId::from(peer).bot_api_dialog_id())
            {
                lines.push(format!("telegram_reply_to_chat_id={peer_id}"));
            }
            if header.forum_topic {
                lines.push("telegram_reply_is_forum_topic=true".to_string());
            }
            if let Some(quote) = header.quote_text.as_deref().map(compact_context_text) {
                if !quote.is_empty() {
                    lines.push(format!("telegram_reply_quote={quote}"));
                }
            }
        }
        tl::enums::MessageReplyHeader::MessageReplyStoryHeader(header) => {
            let peer_id = PeerId::from(&header.peer)
                .bot_api_dialog_id()
                .map(|id| id.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            lines.push(format!(
                "telegram_reply_to_story_id={} telegram_reply_to_chat_id={peer_id}",
                header.story_id
            ));
        }
    }
}

fn append_reply_target(lines: &mut Vec<String>, reply: &TelegramMessage) {
    let sender_id = reply
        .sender_id()
        .and_then(|id| id.bot_api_dialog_id())
        .unwrap_or(0);
    let sender = reply
        .sender()
        .map(display_name)
        .unwrap_or_else(|| "unknown".to_string());
    let username = reply.sender().and_then(|peer| peer.username());
    let mut target = format!(
        "telegram_reply_target_message_id={} sender={} user_id={sender_id}",
        reply.id(),
        compact_context_text(&sender)
    );
    if let Some(username) = username {
        target.push_str(" username=@");
        target.push_str(&compact_context_text(username));
    }
    lines.push(target);
    let text = compact_context_text(reply.text());
    if !text.is_empty() {
        lines.push(format!("telegram_reply_target_text={text}"));
    }
}

fn mention_targets(message: &Message) -> Vec<String> {
    let mut targets = Vec::new();
    let Some(entities) = message.fmt_entities() else {
        return targets;
    };
    for entity in entities {
        let target = match entity {
            tl::enums::MessageEntity::Mention(entity) => {
                utf16_slice(message.text(), entity.offset, entity.length).map(|mention| {
                    if mention.starts_with('@') {
                        mention.to_string()
                    } else {
                        format!("@{mention}")
                    }
                })
            }
            tl::enums::MessageEntity::MentionName(entity) => {
                Some(format!("user_id={}", entity.user_id))
            }
            _ => None,
        };
        if let Some(target) = target {
            targets.push(compact_context_text(&target));
            if targets.len() >= MAX_CONTEXT_ITEMS {
                break;
            }
        }
    }
    targets
}

fn utf16_slice(text: &str, offset: i32, length: i32) -> Option<&str> {
    let offset = usize::try_from(offset).ok()?;
    let length = usize::try_from(length).ok()?;
    let end = offset.checked_add(length)?;
    let mut boundaries = vec![(0, 0)];
    let mut units = 0;
    for (byte, character) in text.char_indices() {
        units += character.len_utf16();
        boundaries.push((units, byte + character.len_utf8()));
    }
    let start_byte = boundaries
        .iter()
        .find_map(|(units, byte)| (*units == offset).then_some(*byte))?;
    let end_byte = boundaries
        .iter()
        .find_map(|(units, byte)| (*units == end).then_some(*byte))?;
    text.get(start_byte..end_byte)
}

fn message_reactions(message: &TelegramMessage) -> Option<&tl::enums::MessageReactions> {
    let reactions = match &message.raw {
        tl::enums::Message::Message(message) => message.reactions.as_ref(),
        tl::enums::Message::Service(message) => message.reactions.as_ref(),
        tl::enums::Message::Empty(_) => None,
    }?;
    Some(reactions)
}

fn message_reactions_is_min(reactions: &tl::enums::MessageReactions) -> bool {
    let tl::enums::MessageReactions::Reactions(reactions) = reactions;
    reactions.min
}

fn message_reactions_have_any(reactions: &tl::enums::MessageReactions) -> bool {
    let tl::enums::MessageReactions::Reactions(reactions) = reactions;
    !reactions.results.is_empty()
        || reactions
            .recent_reactions
            .as_ref()
            .is_some_and(|reactions| !reactions.is_empty())
        || reactions
            .top_reactors
            .as_ref()
            .is_some_and(|reactors| !reactors.is_empty())
}

fn reaction_summary_with_actors(
    reactions: &tl::enums::MessageReactions,
    reaction_list: Option<&[tl::enums::MessagePeerReaction]>,
    chat_id: i64,
) -> String {
    let mut summary = reaction_summary_raw(reactions);
    let reactions = reaction_list
        .filter(|reactions| !reactions.is_empty())
        .or(match reactions {
            tl::enums::MessageReactions::Reactions(reactions) => {
                reactions.recent_reactions.as_deref()
            }
        });
    let Some(reactions) = reactions else {
        return summary;
    };
    let actors: Vec<_> = reactions
        .iter()
        .take(MAX_CONTEXT_ITEMS)
        .map(|reaction| match reaction {
            tl::enums::MessagePeerReaction::Reaction(reaction) => {
                let actor = if reaction.my {
                    "self".to_string()
                } else {
                    PeerId::from(&reaction.peer_id)
                        .bot_api_dialog_id()
                        .map(|id| {
                            if id == chat_id && chat_id > 0 {
                                "chat_partner".to_string()
                            } else {
                                format!("user_id={id}")
                            }
                        })
                        .unwrap_or_else(|| "unknown".to_string())
                };
                format!(
                    "from={actor} reaction={}",
                    reaction_label(&reaction.reaction)
                )
            }
        })
        .collect();
    if !actors.is_empty() {
        summary.push_str(&format!("; reactors=[{}]", actors.join(", ")));
    }
    summary
}

fn reaction_summary_raw(reactions: &tl::enums::MessageReactions) -> String {
    let tl::enums::MessageReactions::Reactions(reactions) = reactions;
    let mut sections = Vec::new();

    let counts: Vec<_> = reactions
        .results
        .iter()
        .take(MAX_CONTEXT_ITEMS)
        .map(|reaction| match reaction {
            tl::enums::ReactionCount::Count(reaction) => {
                format!("{} x{}", reaction_label(&reaction.reaction), reaction.count)
            }
        })
        .collect();
    if !counts.is_empty() {
        sections.push(format!("counts=[{}]", counts.join(", ")));
    }

    let recent: Vec<_> = reactions
        .recent_reactions
        .as_ref()
        .into_iter()
        .flatten()
        .take(MAX_CONTEXT_ITEMS)
        .map(|reaction| match reaction {
            tl::enums::MessagePeerReaction::Reaction(reaction) => {
                let peer_id = PeerId::from(&reaction.peer_id)
                    .bot_api_dialog_id()
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let mut flags = Vec::new();
                if reaction.my {
                    flags.push("my");
                }
                if reaction.big {
                    flags.push("big");
                }
                if reaction.unread {
                    flags.push("unread");
                }
                let suffix = if flags.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", flags.join(","))
                };
                format!(
                    "user_id={peer_id} {}{suffix}",
                    reaction_label(&reaction.reaction)
                )
            }
        })
        .collect();
    if !recent.is_empty() {
        sections.push(format!("recent=[{}]", recent.join(", ")));
    }

    let top_reactors: Vec<_> = reactions
        .top_reactors
        .as_ref()
        .into_iter()
        .flatten()
        .take(MAX_CONTEXT_ITEMS)
        .map(|reactor| match reactor {
            tl::enums::MessageReactor::Reactor(reactor) => {
                let peer_id = reactor
                    .peer_id
                    .as_ref()
                    .and_then(|peer| PeerId::from(peer).bot_api_dialog_id())
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "anonymous".to_string());
                format!("user_id={peer_id} count={}", reactor.count)
            }
        })
        .collect();
    if !top_reactors.is_empty() {
        sections.push(format!("top_reactors=[{}]", top_reactors.join(", ")));
    }

    if sections.is_empty() {
        "present (details unavailable)".to_string()
    } else {
        sections.join("; ")
    }
}

fn reaction_label(reaction: &tl::enums::Reaction) -> String {
    match reaction {
        tl::enums::Reaction::Empty => "removed".to_string(),
        tl::enums::Reaction::Emoji(reaction) => reaction.emoticon.clone(),
        tl::enums::Reaction::CustomEmoji(reaction) => {
            format!("custom_emoji:{}", reaction.document_id)
        }
        tl::enums::Reaction::Paid => "paid".to_string(),
    }
}

fn emoji_status_document_id(status: &tl::enums::EmojiStatus) -> Option<i64> {
    match status {
        tl::enums::EmojiStatus::Status(status) => Some(status.document_id),
        tl::enums::EmojiStatus::Collectible(status) => Some(status.document_id),
        tl::enums::EmojiStatus::Empty | tl::enums::EmojiStatus::InputEmojiStatusCollectible(_) => {
            None
        }
    }
}

fn raw_user_id(peer: &tl::enums::Peer) -> Option<i64> {
    match peer {
        tl::enums::Peer::User(peer) => Some(peer.user_id),
        tl::enums::Peer::Chat(_) | tl::enums::Peer::Channel(_) => None,
    }
}

fn document_emoji(document: &tl::enums::Document) -> Option<String> {
    let tl::enums::Document::Document(document) = document else {
        return None;
    };
    document_asset_kind(document).map(|(emoji, _)| emoji)
}

fn document_asset_kind(document: &tl::types::Document) -> Option<(String, bool)> {
    document
        .attributes
        .iter()
        .find_map(|attribute| match attribute {
            tl::enums::DocumentAttribute::Sticker(sticker) => Some((sticker.alt.clone(), false)),
            tl::enums::DocumentAttribute::CustomEmoji(emoji) => Some((emoji.alt.clone(), true)),
            _ => None,
        })
}

fn describe_gift_action(action: &tl::enums::MessageAction) -> Option<String> {
    let gift = match action {
        tl::enums::MessageAction::StarGift(action) => &action.gift,
        tl::enums::MessageAction::StarGiftUnique(action) => &action.gift,
        _ => return None,
    };
    Some(match gift {
        tl::enums::StarGift::Gift(gift) => {
            match (gift.title.as_deref(), document_emoji(&gift.sticker)) {
                (Some(title), Some(emoji)) => format!("[Telegram gift received: {title}, {emoji}]"),
                (Some(title), None) => format!("[Telegram gift received: {title}]"),
                (None, Some(emoji)) => format!("[Telegram gift received: {emoji}]"),
                (None, None) => "[Telegram gift received]".to_string(),
            }
        }
        tl::enums::StarGift::Unique(gift) => {
            format!(
                "[Telegram collectible gift received: {} #{}]",
                gift.title, gift.num
            )
        }
    })
}

fn compact_context_text(text: &str) -> String {
    text.chars()
        .take(MAX_CONTEXT_TEXT_CHARS)
        .map(|character| match character {
            '\n' | '\r' => ' ',
            character => character,
        })
        .collect::<String>()
        .trim()
        .to_string()
}

fn display_name(peer: &Peer) -> String {
    let name = match peer {
        Peer::User(user) => user.full_name(),
        _ => peer.name().unwrap_or_default().to_string(),
    };
    if name.is_empty() {
        "someone".to_string()
    } else {
        name
    }
}

fn visual_document_kind(document: &Document) -> Option<&'static str> {
    let mime = document.mime_type().unwrap_or("");
    if mime.eq_ignore_ascii_case("image/gif") || document_has_extension(document, &["gif"]) {
        return Some("gif");
    }
    if mime.starts_with("video/")
        || document_has_extension(document, &["mp4", "mov", "webm", "mkv", "avi"])
    {
        return Some("video");
    }
    mime.starts_with("image/").then_some("image")
}

fn document_has_extension(document: &Document, extensions: &[&str]) -> bool {
    document
        .name()
        .and_then(|name| name.rsplit_once('.'))
        .is_some_and(|(_, extension)| {
            extensions
                .iter()
                .any(|candidate| extension.eq_ignore_ascii_case(candidate))
        })
}

fn largest_thumbnail(document: &Document) -> Option<PhotoSize> {
    document
        .thumbs()
        .into_iter()
        .max_by_key(|thumbnail| thumbnail.size())
}

fn visual_label(document: &Document, kind: &str) -> String {
    let mut details = Vec::new();
    if let Some((width, height)) = document.resolution() {
        details.push(format!("{width}x{height}"));
    }
    if let Some(duration) = document.duration() {
        details.push(format!("{duration:.1}s"));
    }
    let suffix = if details.is_empty() {
        String::new()
    } else {
        format!(" ({})", details.join(", "))
    };
    let preview = matches!(kind, "gif" | "video")
        .then_some("\n[preview frame]")
        .unwrap_or("");
    format!("[{kind}]{suffix}{preview}")
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
