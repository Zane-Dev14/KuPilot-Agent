# K8s Failure Intelligence Copilot

RAG-powered Kubernetes failure diagnosis. Ask questions about pod crashes, OOMKills, scheduling failures — get root-cause analysis backed by your own runbooks, events, and manifests.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue) ![macOS](https://img.shields.io/badge/os-macOS-lightgrey) ![Ollama](https://img.shields.io/badge/LLM-Ollama-orange)

---

## How It Works

```
User question
     ↓
Query Classifier (LLM + heuristics)
     ├── diagnostic     → RAG pipeline (retrieve → rerank → LLM → structured JSON)
     ├── conversational  → memory lookup ("what did I ask before?")
     ├── operational     → kubectl command suggestions
     └── out_of_scope   → polite refusal

RAG pipeline:
  1. Milvus vector search (BGE embeddings, 384-dim)
  2. Cross-encoder reranking (BGE reranker v2)
  3. Adaptive model: simple queries → llama3.1:8b, complex → Qwen3-coder:30b
  4. Structured JSON diagnosis with confidence + sources
```

---

## Quick Start

```bash
# 1. Clone & enter project
cd Week-2

# 2. Run setup (starts Milvus, installs deps, runs tests, ingests sample data)
chmod +x setup.sh
./setup.sh

# 3. Start Ollama (separate terminal) & pull models
ollama serve
ollama pull llama3.1:8b-instruct-q8_0
ollama pull Qwen3-coder:30b

# 4a. Web UI
python -m uvicorn src.api:app --reload
# → http://localhost:8000

# 4b. CLI chat
python scripts/chat.py
```

> **Milvus broken?** Run `./setup.sh --fix-milvus` — it wipes volumes and rebuilds from scratch.

---

## Project Structure

```
src/
  config.py          Settings (Pydantic BaseSettings, reads .env)
  vectorstore.py     Embeddings + reranker + Milvus wrapper
  ingestion.py       File loaders (YAML, JSON events, Markdown, logs) + chunking
  memory.py          Per-session chat memory (LRU eviction, disk persistence)
  rag_chain.py       Core RAG: classify → retrieve → rerank → generate → parse
  api.py             FastAPI server (web UI + REST + SSE streaming)

scripts/
  chat.py            Interactive CLI chat (rich output)
  ingest.py          CLI document ingestion

static/              Frontend assets (Three.js cinematic UI)
templates/           Jinja2 HTML template
data/sample/         Sample K8s manifests, events, runbooks
tests/               Offline unit tests (no Milvus/Ollama needed)
```

---

## API

| Method | Endpoint | What it does |
|--------|----------|-------------|
| `GET`  | `/` | Web UI |
| `GET`  | `/health` | Milvus connection status |
| `POST` | `/diagnose` | Diagnose a failure → structured JSON |
| `POST` | `/diagnose/stream` | Same, but SSE token streaming |
| `POST` | `/query-analysis` | Debug: complexity score + model pick |
| `POST` | `/ingest` | Ingest documents into Milvus |
| `POST` | `/memory/clear` | Clear a session's memory |

**Example — diagnose:**

```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"question": "Why is my pod OOMKilled?", "session_id": "user1"}'
```

```json
{
  "diagnosis": {
    "root_cause": "Container exceeded 512Mi memory limit",
    "explanation": "The data-processor pod allocated more than its limit...",
    "recommended_fix": "1. Increase limit to 1Gi  2. Profile memory usage",
    "confidence": 0.85,
    "sources": ["data/sample/docs/oomkilled-runbook.md"]
  },
  "complexity_score": 0.35
}
```

---

## Configuration

All settings via environment variables or `.env` file:

| Variable | Default | What |
|----------|---------|------|
| `MILVUS_URI` | `http://localhost:19530` | Milvus connection |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (384-dim) |
| `EMBEDDING_DEVICE` | `mps` | `mps` for Apple Silicon, `cpu` otherwise |
| `SIMPLE_MODEL` | `llama3.1:8b-instruct-q8_0` | Fast model for simple queries |
| `COMPLEX_MODEL` | `Qwen3-coder:30b` | Large model for complex reasoning |
| `QUERY_COMPLEXITY_THRESHOLD` | `0.7` | Score above → complex model |
| `RETRIEVAL_TOP_K` | `4` | Docs returned after reranking |
| `CHUNK_SIZE` | `1000` | Text chunk size for ingestion |

---

## Requirements

- **macOS** with Docker Desktop
- **Python 3.10+**
- **Ollama** ([ollama.ai](https://ollama.ai)) for local LLM inference
- ~8 GB RAM for embeddings + Milvus + small model

---

## Tests

```bash
# Offline tests — no Milvus or Ollama needed
pytest tests/test_basic.py -v
```

Covers: config defaults, memory (LRU, eviction, persistence), model selector (complexity scoring), JSON parser, ingestion (all file types), query classifier, conversational handler.

---

## Adding Your Own Data

Drop files into `data/sample/` (or any directory):

- `.yaml` / `.yml` — K8s manifests
- `.json` — K8s events
- `.md` — Runbooks / documentation
- `.log` / `.txt` — Plain text logs

Then ingest:

```bash
python scripts/ingest.py --path data/sample/
```

Use `--no-drop` to append instead of replacing the collection.
