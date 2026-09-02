# syntax=docker/dockerfile:1

# One binary now, not a C++ core plus a Python hand. Build it, then drop it into a
# slim runtime that also carries the local Ollama the vision model and embedder run on.
FROM rust:1-slim-bookworm AS build
WORKDIR /build
# libsql (the grammers session store) compiles a bundled SQLite, so a C toolchain
# is needed. rustls is used throughout, so no system OpenSSL is required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/target/release/nekora /usr/local/bin/nekora

RUN useradd --create-home --uid 10001 nekora \
    && mkdir -p /app/vault /app/vault/.ollama/models \
    && chown -R nekora:nekora /app
USER nekora
VOLUME ["/app/vault"]

# The session, vault, and Ollama models all live on the persisted volume, and this
# process owns the Ollama server (starts it and pulls bge-m3 + qwen2.5vl on boot).
ENV OLLAMA_HOST=http://127.0.0.1:11434
ENV OLLAMA_MODELS=/app/vault/.ollama/models
ENV NEKORA_MANAGE_OLLAMA=1
ENV NEKORA_SESSION=/app/vault/nekora
ENV NEKORA_VAULT=/app/vault

CMD ["nekora"]
