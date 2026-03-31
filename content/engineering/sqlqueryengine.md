---
layout: post
title: "SQL Query Engine — Natural Language to PostgreSQL via a Self-Healing LLM Pipeline"
date: 2026-03-31
tags: [fastapi, llm, postgresql, redis, docker, openai, natural-language, sql, openwebui, self-hosted]
---

![SQL Query Engine Cover](https://storage.googleapis.com/codeadeel-github/Generic/Blogger/sqlqueryenginecover.png)

# SQL Query Engine

> Point it at a database. Ask questions in plain English. Get back SQL results.

---

## The Problem

You have a PostgreSQL database full of data, and stakeholders who need answers — fast. They don't write SQL. They shouldn't have to. But every text-to-SQL tool you've tried either chokes on schema complexity, generates broken queries with no recovery, or needs you to adopt someone else's entire platform.

**SQL Query Engine** closes that gap. It's a self-hosted [FastAPI](https://fastapi.tiangolo.com/) service that translates natural language into executed PostgreSQL queries through a two-stage LLM pipeline — powered by any OpenAI-compatible model (Ollama, vLLM, OpenAI, LiteLLM). It also ships with full [OpenWebUI](https://github.com/open-webui/open-webui) integration out of the box.

## How It Works

At its core, SQL Query Engine is a two-stage inference pipeline sitting between your users and PostgreSQL. Here's the full picture:

```mermaid
flowchart LR
    A([🧑 User<br>Natural Language]) -->|HTTP / OpenAI API| B[🖥️ SQL Query Engine<br>FastAPI :5181]
    B -->|Stage 1 · NL → SQL| C[🤖 LLM<br>OpenAI-compatible]
    C --> B
    B -->|Stage 2 · Execute + Repair| D[(🐘 PostgreSQL)]
    D --> B
    B -->|Cache + Pub/Sub| E[(📦 Redis)]
    B -->|Results| A
```

The user asks a question. The engine asks the LLM to write a SQL query. The query is executed against PostgreSQL. If it fails, the engine captures the error and asks the LLM to fix it — automatically, up to N times. That's it.

## Two-Stage Pipeline

The engine splits every request into two stages, each handled by a dedicated module:

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI :5181
    participant GEN as QueryGenerator<br>Stage 1 · NL → SQL
    participant PG as PostgreSQL
    participant Redis
    participant LLM as LLM<br>OpenAI-compatible
    participant EVAL as QueryEvaluator<br>Stage 2 · Execute + Repair

    Client->>API: POST /inference/sqlQueryEngine/{chatID}
    API->>GEN: forward(chatID, prompt)

    GEN->>PG: introspectSchema() [first request only]
    PG-->>GEN: DDL + sample rows
    GEN->>Redis: cacheContext(chatID)
    GEN->>LLM: generateSQL(schema, prompt)
    LLM-->>GEN: sqlQuery

    GEN->>EVAL: evaluate(sqlQuery)
    EVAL->>PG: execute(sqlQuery)
    PG-->>EVAL: rows | SQLException

    alt Query fails
        EVAL->>LLM: repairSQL(query, error)
        LLM-->>EVAL: repairedQuery
        EVAL->>PG: execute(repairedQuery)
        PG-->>EVAL: rows
    end

    EVAL->>Redis: publish progressEvent(chatID)
    EVAL-->>API: QueryResult
    API-->>Client: 200 JSON
```

- **Stage 1 — SQL Generation** — On the first request for a session, the engine introspects the database schema from PostgreSQL, samples rows from every table, and asks the LLM to produce a detailed schema description. This context is cached in Redis per `chatID`, so subsequent questions skip introspection entirely. The LLM then generates a SQL query using structured output (Pydantic schema enforcement via LangChain).

- **Stage 2 — SQL Evaluation & Repair** — The generated query is executed against PostgreSQL. If it fails — syntax errors, schema mismatches, empty results — the engine captures the full error output (SQLSTATE codes, PostgreSQL diagnostics, tracebacks) and feeds it back to the LLM along with the schema context. The LLM returns a structured fix. This loop repeats until the query succeeds or the retry limit is reached. Every step publishes real-time progress to a Redis Pub/Sub channel.

## The Self-Healing Loop

This is the key differentiator. When a query goes wrong, the engine doesn't give up — it enters an iterative repair loop:

```mermaid
flowchart TD
    A[Execute SQL Query] --> B{Success?}
    B -->|Yes + Rows| C[✅ Return Results]
    B -->|No or Empty| D[Capture Error Details<br>SQLSTATE · Diagnostics · Traceback]
    D --> E[Feed Error + Schema to LLM]
    E --> F[LLM Returns Fix<br>isValid · observation · fixedQuery]
    F --> G{Retry Limit?}
    G -->|No| A
    G -->|Yes| H[⚠️ Return Failure]
```

The evaluator captures PostgreSQL errors at the `psycopg` level — including SQLSTATE codes, diagnostic messages, and hints — and feeds the full context back to the LLM through a structured evaluation schema. If the query is valid but returns empty results, the loop continues — because an empty result on a reasonable question usually means the query is wrong, not the data. The database connection is set to read-only mode, so the engine can never accidentally modify data.

## OpenAI-Compatible API

The engine exposes `/v1/chat/completions`, `/v1/completions`, and `/v1/models` endpoints — making it a drop-in replacement for any OpenAI-compatible client.

```mermaid
flowchart TB
    subgraph Clients
        OW[OpenWebUI]
        CU[curl / http]
        PY[Python SDK]
        ANY[Any OpenAI Client]
    end

    subgraph Engine["SQL Query Engine (Docker)"]
        direction LR
        V1["/v1/chat/completions"]
        NATIVE["/inference/sqlQueryEngine"]
        AUTH[Bearer Token Auth]
        V1 --> AUTH
        NATIVE --> AUTH
    end

    OW -->|OpenAI Protocol| V1
    CU -->|REST| NATIVE
    PY -->|OpenAI Protocol| V1
    ANY -->|OpenAI Protocol| V1
```

The streaming implementation subscribes to Redis Pub/Sub *before* launching the engine in a thread pool, then forwards progress messages as SSE chunks wrapped in `<think>` tags — so reasoning-capable clients like OpenWebUI display them as chain-of-thought while the query is being built and validated. The final result (SQL + markdown table) renders cleanly below.

Bearer token authentication is supported via `OPENAI_API_KEY`. Comma-separate multiple keys for multi-user setups. Leave empty to disable auth entirely.

## Quick Start

Clone, configure, deploy:

```bash
git clone https://github.com/codeadeel/sqlqueryengine.git
cd sqlqueryengine
```

Edit `docker-compose.yml` with your LLM and PostgreSQL details:

```yaml
# LLM endpoint — Ollama, vLLM, OpenAI, LiteLLM, etc.
- LLM_BASE_URL=http://host.docker.internal:11434/v1
- LLM_MODEL=qwen2.5-coder:7b
- LLM_API_KEY=ollama

# Your PostgreSQL instance
- POSTGRES_HOST=host.docker.internal
- POSTGRES_PORT=5432
- POSTGRES_DB=mydb
- POSTGRES_USER=myuser
- POSTGRES_PASSWORD=mypassword
```

Start the stack:

```bash
docker compose up --build
```

Done. The engine, Redis, and OpenWebUI are all running.

| Service | URL |
|---|---|
| SQL Query Engine API | `http://localhost:5181` |
| Swagger UI | `http://localhost:5181/docs` |
| OpenWebUI | `http://localhost:5182` |

## Architecture at a Glance

```mermaid
flowchart TB
    subgraph Clients
        OW[OpenWebUI]
        SW[Swagger UI]
        CU[curl / SDK]
    end

    subgraph Engine["SQL Query Engine (Docker)"]
        direction LR
        MAIN[FastAPI · main.py]
        COMPAT[OpenAI Compat · openaiCompat.py]
        ENG[Engine · engine.py]
        GEN[QueryGenerator · Stage 1]
        EVAL[QueryEvaluator · Stage 2]
        DB[DBHandler · dbHandler.py]
        SM[SessionManager · sessionManager.py]
        MAIN --> ENG
        COMPAT --> ENG
        ENG --> GEN
        ENG --> EVAL
        GEN --> DB
        EVAL --> DB
        GEN --> SM
        EVAL --> SM
    end

    subgraph External
        PG[(PostgreSQL)]
        RD[(Redis<br>Cache + Pub/Sub)]
        LLM[LLM<br>OpenAI-compatible]
    end

    OW -->|OpenAI Protocol| COMPAT
    SW -->|REST| MAIN
    CU -->|REST / OpenAI| MAIN
    DB --> PG
    SM --> RD
    GEN --> LLM
    EVAL --> LLM
```

Each module has a single responsibility. `dbHandler.py` handles PostgreSQL introspection and safe query execution (read-only connections). `sessionManager.py` manages per-user context in Redis hashes. `promptTemplates.py` holds the LLM prompt templates. `connConfig.py` reads environment variables and provides a shared FastAPI dependency for connection parameters.

## Features Summary

| Feature | Details |
|---|---|
| Pipeline | Two-stage: NL → SQL generation + execution with self-healing repair loop |
| LLM Support | Any OpenAI-compatible endpoint (Ollama, vLLM, OpenAI, LiteLLM) |
| Database | PostgreSQL (read-only connections enforced) |
| Caching | Redis — schema context cached per session, skips re-introspection |
| Streaming | Redis Pub/Sub → SSE with `<think>` tag support for chain-of-thought |
| API | REST + OpenAI-compatible `/v1/chat/completions` |
| Auth | Bearer token (comma-separate multiple keys) |
| Chat UI | OpenWebUI pre-configured in Docker Compose |
| Deployment | Docker Compose, single command |
| Module Mode | Import `SQLQueryEngine` directly in Python, no HTTP layer needed |

---

## Links

- **GitHub**: [codeadeel/sqlqueryengine](https://github.com/codeadeel/sqlqueryengine)
- **Wiki**: [Full documentation](https://github.com/codeadeel/sqlqueryengine/wiki)
- **Wiki — REST API & Python Module**: [usage examples & endpoints](https://github.com/codeadeel/sqlqueryengine/wiki)
- **Wiki — OpenWebUI Setup**: [connection & configuration](https://github.com/codeadeel/sqlqueryengine/wiki)
- **Wiki — Environment Variables**: [full reference](https://github.com/codeadeel/sqlqueryengine/wiki)
- **License**: MIT
