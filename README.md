<div align="center">

# Nekora

=^..^=

![Rust](https://img.shields.io/badge/Rust-2021-%23dea584?style=for-the-badge&logo=rust&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-MTProto-%232AABEE?style=for-the-badge&logo=telegram&logoColor=white)
![License](https://img.shields.io/badge/License-GPL--3-green?style=for-the-badge)

**A 24/7 Telegram character AI with a memory, a pair of local eyes, and her own clock.**

</div>

> **Experimental software.** Nekora is designed to stay alive 24/7. She is not
> quick, noisy, or constantly visible in a chat — the point is that at any time
> she may decide to answer or start a conversation on her own. Use a separate
> account, keep the session private, and follow Telegram's rules.

## What it is

Nekora is not a chatbot waiting behind a command prompt. She is a character AI
that keeps time, notices the world, remembers selected things, and occasionally
decides that silence is the right answer.

She is always running, but she is intentionally not an instant assistant. A
message can sit in a short conversational window while she waits for the rest
of the thought; a model can take time to answer; and the heartbeat only checks
its own impulse every so often. She may be quiet for a long while and still be
present — ready to make her own decision without being summoned.

The project is one Rust process built around a simple separation:

```text
Telegram updates ──▶ conversation window ──▶ brain + tools ──▶ paced Telegram bubbles
                              ▲                    │
                              │                    ▼
                    heartbeat every 27 min ◀── diary + working memory
```

Incoming messages are grouped into one thought instead of answered line by
line. A short quiet period closes the burst, typing notifications can extend it,
and a final grace period catches messages that arrive while she is preparing a
reply. The answer is split back into separate Telegram messages and sent with
human-sized typing delays.

When nobody is talking, the heartbeat gives her a chance to act on her own. A
tick may lead to a diary reflection, a message, or nothing at all. She acts on
roughly half of waking ticks, occasionally takes a 15–120 minute nap, and a
message can wake her from it.

## Goals

- Give an AI character a sense of time instead of a request/response loop.
- Make Telegram interaction feel paced and social: read states, typing, bursts,
  media, and separate message bubbles.
- Keep memory inspectable and durable without pretending that the model trained
  itself on the conversation.
- Let the character choose between acting, waiting, and staying quiet.

## Community

Questions, updates, and the living Nekora instance:

- Telegram: [@nekora_shiz](https://t.me/nekora_shiz)

## Technical details

The main model handles conversation and tool calls through an OpenAI-compatible
API. Ollama stays local and provides the two things that benefit most from
being close to the vault: embeddings and vision.

| Part | Default | Role |
| --- | --- | --- |
| Main model | `deepseek-v4-flash` | conversation, reflection, memory distillation, tool loop |
| Main endpoint | `https://api.deepseek.com/v1/` | OpenAI-compatible chat API |
| Embeddings | `bge-m3` through Ollama | semantic recall over diary notes |
| Vision | `qwen2.5vl:3b` through Ollama | photos, stickers, GIFs, and video preview frames |

The main endpoint and model are configurable, so another compatible API can be
used without changing the Telegram or memory layers. The local embedder should
remain stable for an existing vault: stored vectors were made by that model.

# Human-like behavior

## Feelings

Nekora does not change the model's weights and does not need to claim that it
has a biological emotional system. Instead, the character prompt gives the model
a voice with moods, hesitation, curiosity, and the option to say nothing. When a
conversation matters, its emotional context can be preserved in working memory
or in the diary and influence a later turn.

## Learning

This is retrieval, not training. A meaningful event can be written to the diary
with `remember`, and later turns receive relevant notes through embedding recall.
Working memory keeps unfinished tasks, promises, dates, and recent state for the
next few days. The model can therefore change its behavior through remembered
context while the underlying model stays untouched.

## Sleeping

Sleep is the maintenance pass for memory. When the current day's context grows
large, Nekora updates working memory, distils the day's events into durable
notes, finds related entries, and asks the model to merge, refine, or retire
redundant pieces. At a day boundary she performs the same consolidation before
moving on to the new day.

## Thoughts

There is no separate human-like stream of thoughts hidden in the program. What
looks like a thought is a carefully assembled turn: the current time and
identity, working memory, relevant diary notes, recent events, and the new
message burst. During an autonomous tick she can first reflect on an old diary
page, then decide whether that reflection deserves an action.

## Memory that lives in Markdown

Nekora's long-term memory is a small Markdown vault, not a hosted database.
Each durable memory is a note with its confidence, usage history, links to
related notes, and an embedding in its frontmatter. Recall is a cosine scan —
deliberately simple, inspectable, and appropriate for a personal-sized diary.

There are three kinds of context:

| Context | Lifetime | Purpose |
| --- | --- | --- |
| Recent conversation | current process | the last messages needed for a natural reply |
| Working memory | across turns | short-term tasks, promises, and state |
| Diary notes | long-term | facts and experiences worth keeping |

The result is a memory model with a short-term layer and a durable layer:
working memory carries near-term obligations, while the diary keeps facts,
events, relationships, uncertainty, and useful emotional context.

Incoming Telegram messages also keep their platform context: the original
message ID and timestamp, chat type, replies and quoted text, explicit mentions,
whether Nekora was addressed, reactions and reactors, forwards, and media-group
membership. The model can use that metadata to choose a visible Telegram reply
(`reply_to_message_id`) or a reaction. Private chats are limited to Telegram
contacts; group conversations remain available.

## Capabilities

The tool set is intentionally small:

| Tool | Use |
| --- | --- |
| `recall_memory` | search the diary for a focused question |
| `list_memories` | list durable memories when asked what she remembers |
| `remember` | write something worth keeping |
| `inspect_user` | inspect a Telegram profile and avatar |
| `inspect_message_media` | look closely at recent media |
| `get_current_time` | ask Telegram for its server time in UTC+04:00 |
| `send_message` | send a visible Telegram reply or a proactive message; optionally reply to a message ID |
| `react_to_message` | add or remove Nekora's reaction on a message |
| `list_chats` | inspect recent chats before choosing someone to contact |
| `stay_quiet` | choose silence deliberately |

There is no arbitrary shell execution, web search, or Bot API integration.
Media is flattened into model-readable descriptions; moving video is represented
by the best Telegram preview frame available to the model.

## Security concerns

The session file is equivalent to a logged-in Telegram account. The main model
provider receives the text and context sent to it; Ollama receives local vision
and embedding requests. The vault may contain messages, memories, and their
embeddings. Treat all three as private state and use a dedicated account.

# Deployment

## Prerequisites

For a local build you need:

- Rust and Cargo
- a Telegram API ID and API hash
- a DeepSeek API key, or another OpenAI-compatible main endpoint
- Ollama with `bge-m3` and the configured vision model available

Get the Telegram API credentials from Telegram's developer portal. Nekora uses
an account phone number and an interactive login code on the first run — this
is a userbot, not a bot-token application.

## Local

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

PAPIK_NAME=your name
NEKORA_NAME=Nekora
```

`TELEGRAM_PHONE` is optional; if it is absent, Nekora asks for it interactively.
On the first start she also asks for the login code and, when enabled, the 2FA
password. The resulting session is saved and reused on later starts.

Build and run:

```sh
cargo run --release
```

The process prints `nekora is up; waiting on her own clock` after login and once
the vault is open. Leave it running: the process itself is the 24/7 presence.

## Docker

The included image builds the Rust binary and installs Ollama in the runtime
image. It starts and owns `ollama serve`, pulls missing local models, and keeps
the session, diary, checkpoint, and model files on one volume.

```sh
docker build -t nekora .

docker run --rm -it \
  --env-file .env \
  -v nekora-data:/app/vault \
  nekora
```

Keep the terminal attached for the first interactive Telegram login. The Docker
image already sets `NEKORA_MANAGE_OLLAMA=1`, `NEKORA_VAULT=/app/vault`,
`NEKORA_SESSION=/app/vault/nekora`, and the Ollama model directory inside the
volume. The first start can take a while while the models are downloaded.

## Configuration

Every setting is read from the environment. A `.env` file is loaded at startup,
but already-exported environment variables win over it.

| Variable | Default | Description |
| --- | --- | --- |
| `TELEGRAM_API_ID` | `0` | Telegram application ID; required in practice |
| `TELEGRAM_API_HASH` | empty | Telegram application hash; required for login |
| `TELEGRAM_PHONE` | prompt | account phone in international format |
| `DEEPSEEK_API_KEY` | empty | key for the main OpenAI-compatible endpoint |
| `NEKORA_MAIN_API_BASE` | DeepSeek `/v1/` | main chat endpoint |
| `NEKORA_MAIN_MODEL` | `deepseek-v4-flash` | main chat model |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server address |
| `NEKORA_VISION_MODEL` | `qwen2.5vl:3b` | local vision model |
| `NEKORA_MANAGE_OLLAMA` | unset | set to `1` to start and prepare Ollama automatically |
| `NEKORA_OLLAMA_START_TIMEOUT` | `120` | seconds to wait for managed Ollama |
| `NEKORA_REQUEST_TIMEOUT` | `120` | seconds allowed for a model request |
| `NEKORA_NAME` | `Nekora` | name used in the character preamble |
| `PAPIK_NAME` | `your person` | the person's name used in the character preamble |
| `NEKORA_VAULT` | `vault` | directory for Markdown memories and runtime state |
| `NEKORA_SESSION` | `nekora` | session path base; `.session` is appended |

To change her personality, create `prompts/system.md`. When that file is not
present, the built-in Nekora persona is used. The prompt is read relative to
the current working directory.

## Data and state

With the default local settings, the important state looks like this:

```text
vault/
├── diary notes (*.md)     durable memories
├── working_memory.md      short-term working context
└── runtime/
    └── today.json        crash-safe checkpoint for today's messages

nekora.session            Telegram authorization, next to the process
```

The Docker image puts the session inside `/app/vault` so the volume contains the
persistent session, diary, checkpoint, and model files. The vault and session both contain private data;
back them up carefully and never publish them. The repository's `.gitignore`
already excludes `.env`, session files, the vault, and build output.

## Project structure

| File | Responsibility |
| --- | --- |
| `src/heartbeat.rs` | autonomous ticks, acting, and naps |
| `src/conversation.rs` | burst grouping and reply splitting |
| `src/brain.rs` | model requests, vision, embeddings, and the tool loop |
| `src/tools.rs` | the memory and Telegram action set |
| `src/diary.rs` | Markdown notes, recall, confidence, and links |
| `src/sleep.rs` | working-memory refresh and diary consolidation |
| `src/userbot.rs` | MTProto login, updates, media, and paced sending |
| `src/persistence.rs` | atomic filesystem operations for the vault |
| `src/config.rs` | environment, identity, and persona loading |

## Development

```sh
cargo fmt
cargo build --release
```

The core is intentionally kept readable: one process, a small number of
subsystems, Markdown as the source of truth, and no extra service required for
the diary.

## License

GPL-3. See `Cargo.toml` for the project metadata.
