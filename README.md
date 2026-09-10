<div align="center">

# Nekora

=^..^=

![Rust](https://img.shields.io/badge/Rust-2021-%23dea584?style=for-the-badge&logo=rust&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-MTProto-%232AABEE?style=for-the-badge&logo=telegram&logoColor=white)

**An autonomous Telegram character with persistent memory, social state, and her own rhythm.**

</div>

> [!WARNING]
> Nekora is experimental userbot software. It controls a real Telegram account
> through MTProto, not the Bot API. Use a dedicated account, protect its session
> file, and follow Telegram's terms and rate limits.

## Overview

Nekora is a long-running Telegram character, not a command bot. She groups short message bursts into one thought,
remembers selected events, maintains a small relationship state, can inspect media and public information, and may
choose to reply, react, start a conversation, or remain silent.

The project is a single Rust process with explicit boundaries: the heartbeat decides whether to act, the brain decides
what to do, tools expose bounded capabilities, and the userbot owns Telegram I/O.

## How it works

```text
Telegram updates ──▶ conversation window ──▶ brain + tools ──▶ Telegram
                              │                    ▲
                              ▼                    │
                    social appraisal       memory context

heartbeat ──▶ reflection ──▶ brain + tools ──▶ message, reaction, or silence
```

Incoming messages wait for a three-second quiet window; typing can extend the window, and a five-second grace catches
late messages before generation starts. The resulting reply is split into natural Telegram bubbles and sent with typing
delays. A newer private message invalidates an obsolete in-flight reply. Deliberately unanswered batches remain pending
and are reconsidered with increasing delays: private chats start at one minute, groups at five, and both cap at the
27-minute heartbeat interval. A new message makes its pending chat eligible immediately.

Every 27 minutes the heartbeat may trigger an autonomous turn independently of chat activity. A waking tick has roughly
a 50% chance to act and a 1% chance to begin a 15–120 minute nap. Acting still does not guarantee a message: silence is
an explicit outcome.

## Community

Questions, updates, and the living Nekora instance:

- Telegram: [@nekora_shiz](https://t.me/nekora_shiz)
- Support / author: [@shizukiqx](https://t.me/shizukiqx)

## Model routing

Conversation, maintenance, vision, embeddings, search, and image generation are separate paths. Private maintenance uses
Mistral directly. Image recognition tries OpenRouter first, then Mistral, and finally local Ollama; OpenRouter is not a
proxy for the Mistral API.

| Path                 | Default                                    | Notes                                                                                                        |
|----------------------|--------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Visible conversation | `deepseek-v4-flash`                        | OpenAI-compatible main endpoint; also appraises incoming social events                                       |
| Private maintenance  | `mistral-small-2603` via direct Mistral API  | configurable with `NEKORA_REASONING_MODEL`; an empty value disables it, and request failures fall back to the main model |
| Embeddings           | `bge-m3` on Ollama                         | fixed local vector space for diary recall                                                                    |
| Vision               | `qwen/qwen3-vl-32b-instruct` on OpenRouter | OpenRouter first, then `mistral-small-2603` via Mistral, then `qwen2.5vl:3b` locally                       |
| Web search           | Ollama Cloud, then OpenRouter              | provider order is configurable; results are normalized before entering the turn                              |
| Image generation     | disabled                                   | requires separate OpenRouter prompt and image models; explicit quality rejections trigger a retry            |

If `MISTRAL_API_KEY` is empty, private maintenance uses the main model and vision goes from OpenRouter directly to the
local fallback. `MISTRAL_API_BASE` defaults to `https://api.mistral.ai/v1`.

Do not change the embedding model for an existing vault: old and new vectors would no longer be comparable. Changing
`NEKORA_REASONING_MODEL` does not move visible conversations to Mistral.

## Memory and social state

Nekora does retrieval, not training. Runtime context is split by lifetime:

| State           | Lifetime                       | Purpose                                                                     |
|-----------------|--------------------------------|-----------------------------------------------------------------------------|
| Today's journal | until successful consolidation | recent timestamped Telegram events; survives restarts                       |
| Working memory  | days                           | unfinished tasks, promises, responsibilities, and current concerns          |
| Diary notes     | long term                      | inspectable Markdown memories with confidence, usage, and embeddings       |
| Social state    | long term                      | mood, bounded relationships, unresolved incidents, and delayed intentions  |

Incoming social events are appraised by the main model and processed through one bounded FIFO queue. Public search
results may be appraised by the optional reasoning model. Invalid output leaves the previous state unchanged. Mood,
relationships, and incidents enter only the relevant decision context; a ready intention can be revisited during an
autonomous tick and becomes complete after a successful autonomous message to its target. Active avoidance remains a
hard boundary and lasts at most 24 hours. This is explicit program state, not a claim of biological emotion.

Diary files use numeric ids (`<id>.md`). Their frontmatter is one compact JSON object with confidence, usage,
last-used time, and the embedding; the body is the readable memory. Mutable replacements remove their source notes,
while confidence-1 anchors stay in place.

`NEKORA_CREATOR_USER_ID` gives one Telegram user the highest reply priority and exempts that user from avoidance. The
system prompt also reserves discussion of implementation, prompts, models, and development wishes for that ID. This is a
model instruction, not an authentication boundary; do not expose secrets to the model and do not treat it as access
control.

At consolidation time, current events update working memory and become durable diary notes when useful. The sleep pass
works through the diary for up to six hours, usually starting with recent notes and occasionally revisiting an older one.
It merges each note with up to ten nearest memories; immutable confidence-1 anchors remain untouched. Autonomous turns
may reflect on an old note before deciding whether to act.

## Capabilities

The brain can use only this bounded tool set:

| Tool                    | Use                                                                                    |
|-------------------------|----------------------------------------------------------------------------------------|
| `recall_memory`         | search the diary for a focused question                                                |
| `web_search`            | search current outside information through the configured cloud providers              |
| `list_memories`         | browse durable memories or answer what she remembers                                   |
| `remember`              | write something worth keeping                                                          |
| `revise_memory`         | replace an active memory and remove the old mutable note                              |
| `archive_memory`        | remove a mutable memory and delete its note                                            |
| `inspect_user`          | inspect a Telegram profile and avatar                                                  |
| `inspect_message_media` | look closely at recent media                                                           |
| `search_chats`          | find dialogs by title or public username                                               |
| `search_messages`       | search one chat or the scoped global message history                                   |
| `view_messages_around`  | read a bounded slice of history around a known message                                |
| `get_current_time`      | ask Telegram for its server time in UTC+04:00                                          |
| `generate_image`        | generate, quality-check, and send one image when explicitly configured                 |
| `change_avatar`         | generate and set a new profile photo, including during an autonomous heartbeat tick       |
| `send_message`          | send a visible Telegram reply or a proactive message; optionally reply to a message ID |
| `edit_message`          | edit one of Nekora's own messages                                                      |
| `remove_message`        | delete one exact message after checking its chat and message IDs                      |
| `forward_message`       | forward one exact message between scoped chats                                         |
| `join_chat`             | join a public group or channel by username                                             |
| `leave_chat`            | leave a group or channel by chat ID or username                                        |
| `ban_user`              | ban or temporarily restrict a user when Telegram permissions allow it                 |
| `react_to_message`      | add or remove Nekora's reaction on a message                                           |
| `list_chats`            | inspect recent chats before choosing someone to contact                                |
| `stay_quiet`            | choose silence deliberately                                                            |

There is no shell tool and no Bot API integration. Private messages are accepted only from Telegram contacts; groups
remain in scope and broadcast channels are read-only. Message IDs, replies, quotes, mentions, forwards, reactions, media
groups, and timestamps are preserved as model-readable context. Photos and a representative video preview can be
inspected by the vision path.

## Security and privacy

- `*.session` is equivalent to a logged-in Telegram account. Never publish or share it.
- The main provider receives conversation text, recalled memories, and runtime context used for a turn.
- When configured, Mistral may receive diary and working-memory data for maintenance, public search results for
  emotional appraisal, and incoming images after OpenRouter vision fails. Requests go directly to `MISTRAL_API_BASE`.
- When configured, OpenRouter may receive incoming images before Mistral, configured image references during
  generation, plus generation prompts and generated images.
- Configured search providers receive search queries.
- Local Ollama receives diary text for embeddings and media for fallback vision.
- The vault contains message checkpoints, social state, working memory, diary notes, and embeddings. Back it up as
  private data.
- Message bodies, memory, media descriptions, and search results are treated as untrusted prompt data, but model-level
  prompt isolation is not a security sandbox.

## Setup

### Requirements

For a local build you need:

- Rust and Cargo
- a Telegram API ID and API hash
- a DeepSeek API key, or another OpenAI-compatible main endpoint
- Ollama with `bge-m3` and the configured fallback vision model
- a Mistral API key for direct maintenance and the second vision attempt
- an OpenRouter key for primary cloud vision, or when OpenRouter search/image generation is enabled
- credentials for at least one provider in `NEKORA_WEB_SEARCH_CHAIN`

Get the Telegram API credentials from Telegram's developer portal. Nekora uses an account phone number and an
interactive login code on the first run — this is a userbot, not a bot-token application.

### Local run

Start Ollama separately and pull the two default local models:

```sh
ollama serve
ollama pull bge-m3
ollama pull qwen2.5vl:3b
```

Create a `.env` in the current working directory, normally the repository root:

```dotenv
TELEGRAM_API_ID=123456
TELEGRAM_API_HASH=your_telegram_api_hash
TELEGRAM_PHONE=+10000000000
DEEPSEEK_API_KEY=your_deepseek_api_key
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_API_BASE=https://api.mistral.ai/v1
# OPENROUTER_API_KEY=your_openrouter_api_key  # optional fallback/search/images
NEKORA_WEB_SEARCH_CHAIN=ollama,openrouter
OLLAMA_API_KEY=your_ollama_cloud_api_key
NEKORA_VISION_MODEL=qwen/qwen3-vl-32b-instruct
NEKORA_LOCAL_VISION_MODEL=qwen2.5vl:3b
NEKORA_REASONING_MODEL=mistral-small-2603
NEKORA_MISTRAL_VISION_MODEL=mistral-small-2603

PAPIK_NAME=your name
NEKORA_NAME=Nekora
NEKORA_CREATOR_USER_ID=123456789
```

`TELEGRAM_PHONE` is optional; if it is absent, Nekora asks for it interactively. On the first start she also asks for
the login code and, when enabled, the 2FA password. The resulting session is saved and reused on later starts.

Build and run:

```sh
cargo run --release
```

The process prints `nekora is up; waiting on her own clock` after login and once the vault is open. Leave it running:
the process itself is the 24/7 presence.

### OpenAI-compatible proxy

The same process can expose a separate local HTTP endpoint for OpenAI-compatible clients. It is disabled unless
`NEKORA_PROXY_ADDR` is set:

```dotenv
NEKORA_PROXY_ADDR=127.0.0.1:10434
NEKORA_PROXY_TOKEN=replace-with-a-long-random-token
```

It provides `GET /health`, `GET /v1/models`, and `POST /v1/chat/completions`. In the normal process it shares Nekora's
character, diary recall, and bounded Telegram tools; `stream: true` returns a compatible SSE response after the completion
is ready. Set `NEKORA_PROXY_ONLY=1` to run only the proxy without Telegram login; that mode keeps persona and diary context
but has no Telegram tools. Keep the listener on loopback or set `NEKORA_PROXY_TOKEN` before binding it to another interface.

`NEKORA_PROXY_ADDR` is only the local listener address. `NEKORA_PROXY_TOKEN` is the optional Bearer password for clients
of that endpoint; neither value is a Telegram credential and Nekora must never ask a person to send either one in chat.

Example request:

```sh
curl http://127.0.0.1:10434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer replace-with-a-long-random-token' \
  -d '{"model":"nekora","messages":[{"role":"user","content":"Привет"}]}'
```

### Docker

The included image builds the Rust binary, installs Ollama, starts it, pulls missing local models, and keeps runtime
state on one volume.

```sh
docker build -t nekora .

docker run --rm -it \
  --env-file .env \
  -v nekora-data:/app/vault \
  nekora
```

Keep the terminal attached for the first interactive Telegram login. The Docker image already sets
`NEKORA_MANAGE_OLLAMA=1`, `NEKORA_VAULT=/app/vault`,
`NEKORA_SESSION=/app/vault/nekora`, and the Ollama model directory inside the volume. The first start can take a while
while the models are downloaded. The local Ollama process handles embeddings and vision fallback. Cloud search does not
need another container or an inbound port.

### VPS notes

A VPS only needs outbound HTTPS access to the configured providers. Put the search keys in `.env`, keep
`NEKORA_WEB_SEARCH_CHAIN=ollama,openrouter`, and use `--env-file .env` with Docker or
`EnvironmentFile=/etc/nekora/nekora.env`
with systemd. Keep that environment file readable only by Nekora. No SearXNG installation, local search service, or
inbound port is required.

## Configuration

Every setting is read from the environment. A `.env` file is loaded at startup, but already-exported environment
variables win over it.

| Variable                       | Default                             | Description                                                                                               |
|--------------------------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `TELEGRAM_API_ID`              | `0`                                 | Telegram application ID; required in practice                                                             |
| `TELEGRAM_API_HASH`            | empty                               | Telegram application hash; required for login                                                             |
| `TELEGRAM_PHONE`               | prompt                              | account phone in international format                                                                     |
| `DEEPSEEK_API_KEY`             | empty                               | key for the main OpenAI-compatible endpoint                                                               |
| `NEKORA_MAIN_API_BASE`         | `https://api.deepseek.com/v1`       | main chat endpoint                                                                                        |
| `NEKORA_MAIN_MODEL`            | `deepseek-v4-flash`                 | main chat model                                                                                           |
| `MISTRAL_API_KEY`              | empty                               | direct Mistral key for private maintenance and second vision attempt                                     |
| `MISTRAL_API_BASE`             | `https://api.mistral.ai/v1`         | direct Mistral OpenAI-compatible endpoint                                                                |
| `NEKORA_WEB_SEARCH_CHAIN`      | `ollama,openrouter`                 | ordered cloud search providers                                                                            |
| `OLLAMA_API_KEY`               | empty                               | Ollama Cloud web search key                                                                               |
| `OLLAMA_WEB_SEARCH_URL`        | `https://ollama.com/api/web_search` | Ollama Search endpoint                                                                                    |
| `OPENROUTER_API_KEY`           | empty                               | optional OpenRouter key for primary vision, web search, and images                                        |
| `OPENROUTER_API_BASE`          | `https://openrouter.ai/api/v1`      | OpenRouter API base                                                                                       |
| `OPENROUTER_WEB_SEARCH_MODEL`  | `openai/gpt-4.1-mini`               | model used by the OpenRouter search tool                                                                  |
| `OPENROUTER_WEB_SEARCH_ENGINE` | `auto`                              | OpenRouter search engine selection                                                                        |
| `NEKORA_VISION_MODEL`          | `qwen/qwen3-vl-32b-instruct`        | OpenRouter primary vision model                                                                            |
| `NEKORA_LOCAL_VISION_MODEL`    | `qwen2.5vl:3b`                      | local Ollama vision fallback                                                                              |
| `NEKORA_REASONING_MODEL`       | `mistral-small-2603`               | Mistral model for private maintenance and public-result appraisal; set empty to use the main model       |
| `NEKORA_MISTRAL_VISION_MODEL`  | `mistral-small-2603`               | second cloud model for analyzing incoming images                                                          |
| `NEKORA_IMAGE_MODEL`           | empty                               | OpenRouter model slug for the dedicated `/images` API                                                     |
| `NEKORA_IMAGE_PROMPT_MODEL`    | empty                               | OpenRouter chat model that engineers generation prompts                                                   |
| `NEKORA_IMAGE_REFERENCES`      | image files in `references/` (up to 4) | comma-separated local reference image paths; unset uses supported images in `references/`                |
| `NEKORA_IMAGE_PROMPT`          | built-in Nekora template            | optional full canonical image prompt override; `{SCENE_REQUEST}` is replaced per image                   |
| `NEKORA_IMAGE_TIMEOUT`         | `300`                               | seconds allowed for one OpenRouter image generation request                                               |
| `NEKORA_VISION_API_TIMEOUT`    | `30`                                | seconds allowed for each cloud vision attempt before the next provider is tried                           |
| `NEKORA_WEB_SEARCH_TIMEOUT`    | `30`                                | seconds allowed for one search request                                                                    |
| `NEKORA_WEB_SEARCH_COOLDOWN`   | `300`                               | seconds to skip a rate-limited provider                                                                   |
| `OLLAMA_HOST`                  | `http://127.0.0.1:11434`            | Ollama server address                                                                                     |
| `NEKORA_MANAGE_OLLAMA`         | unset                               | set to `1` to start and prepare Ollama automatically                                                      |
| `NEKORA_OLLAMA_START_TIMEOUT`  | `120`                               | seconds to wait for managed Ollama                                                                        |
| `NEKORA_REQUEST_TIMEOUT`       | `120`                               | seconds allowed for a model request                                                                       |
| `NEKORA_NAME`                  | `Nekora`                            | name used in the character preamble                                                                       |
| `PAPIK_NAME`                   | `your person`                       | the person's name used in the character preamble                                                          |
| `NEKORA_PROXY_ADDR`            | unset                               | optional OpenAI-compatible listener, for example `127.0.0.1:10434`                                       |
| `NEKORA_PROXY_TOKEN`           | empty                               | optional bearer token; required when the proxy is exposed beyond loopback                                |
| `NEKORA_PROXY_ONLY`            | unset                               | set to `1` to skip Telegram login and run the standalone proxy only                                       |
| `NEKORA_CREATOR_USER_ID`       | empty                               | positive Telegram user ID of the privileged developer chat                                                |
| `NEKORA_VAULT`                 | `vault`                             | directory for Markdown memories and runtime state                                                         |
| `NEKORA_SESSION`               | `nekora`                            | session path base; `.session` is appended                                                                 |

To change her personality, create `prompts/system.md`. When that file is not present, the built-in Nekora persona is
used. This file supplies the character profile; the core Telegram, context, and tool workflow remains built in so a
personality edit cannot remove it accidentally. The prompt is read relative to the current working directory.

Image generation uses the local reference images as OpenRouter `input_references`. The prompt engineer receives the
canonical image template and returns only the scene-specific `{SCENE_REQUEST}` text; the fixed identity, rendering, and
anti-drift sections are assembled by Rust. Set `NEKORA_IMAGE_MODEL` and `NEKORA_IMAGE_PROMPT_MODEL` explicitly before
using `generate_image`, because image requests may incur provider charges. Image generation has its own
`NEKORA_IMAGE_TIMEOUT` because it can take longer than an ordinary model request.

Incoming image recognition tries OpenRouter first, then Mistral, and uses local Ollama only when both cloud providers
fail. The same generator and references are used by `change_avatar`; the tool uploads the result as Nekora's Telegram
profile photo.

Conversational requests keep only the core workflow and character profile in the stable system prefix. Working memory is
runtime-derived data and follows in its own untrusted user-role block, before per-turn time, recalled diary notes,
recent events, and incoming messages. A turn receives a larger history from its current chat plus a smaller private
awareness window from other chats. Providers with automatic prefix caching can therefore reuse the stable part even when
working memory changes.

## Data and state

With the default local settings, the important state looks like this:

```text
vault/
├── diary notes (*.md)      durable memories
├── working_memory.md       short-term working context
└── runtime/
    ├── today.json          crash-safe checkpoint for today's messages
    └── social.json         mood, relationships, incidents, and intentions

nekora.session              Telegram authorization, next to the process
```

The Docker image puts the session and Ollama models inside `/app/vault`, so the volume contains all persistent runtime
data. The repository's `.gitignore`
excludes `.env`, session files, the vault, and build output.

## Project structure

| File                          | Responsibility                                                    |
|-------------------------------|-------------------------------------------------------------------|
| `src/heartbeat.rs`            | autonomous ticks, acting, and naps                                |
| `src/conversation.rs`         | burst grouping and reply splitting                                |
| `src/brain.rs`                | model requests, vision, embeddings, and the tool loop             |
| `src/tools.rs`                | the memory and Telegram action set                                |
| `src/websearch/mod.rs`        | provider chain, fallback policy, and normalized results           |
| `src/websearch/ollama.rs`     | Ollama Cloud Search adapter                                       |
| `src/websearch/openrouter.rs` | OpenRouter web-search adapter                                     |
| `src/imagegen/mod.rs`         | OpenRouter image generation, references, and quality gate         |
| `src/diary.rs`                | Markdown notes, compact metadata, recall, and confidence          |
| `src/sleep.rs`                | working-memory refresh and diary consolidation                    |
| `src/social.rs`               | persistent mood, relationships, social incidents, and intentions |
| `src/userbot.rs`              | MTProto login, updates, media, and paced sending                  |
| `src/proxy.rs`                | bounded OpenAI-compatible HTTP endpoint and SSE response          |
| `src/persistence.rs`          | atomic filesystem operations for the vault                        |
| `src/promptsall.rs`           | centralized model instructions, persona, maintenance, and tool prompts |
| `src/config.rs`               | environment, identity, and character profile loading              |

## Development

```sh
cargo fmt
cargo build --release
```

The core is intentionally kept readable: one process, a small number of subsystems, Markdown as the durable memory
source of truth, and no database or external vector store.

## License

GPL-3. See `Cargo.toml` for the project metadata.
