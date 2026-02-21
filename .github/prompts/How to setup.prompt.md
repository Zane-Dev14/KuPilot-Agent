Plan: DeepAgents Multi-Agent Migration

TL;DR — Incrementally transform your existing RAG-powered K8s diagnosis app into a DeepAgents-based multi-agent system. The existing `src/` modules (vectorstore, ingestion, memory, config, rag_chain) become the foundation that new tool classes wrap. Five agents (Orchestrator, Investigator, Knowledge, Remediation, Verification) are built as DeepAgents subagents with shared tools. Both CLI and Web UI remain functional at every step. A GitHub Actions workflow covers lint, tests, Docker build, and push.

---

Steps

Phase 0 — Structural Preparation

1. Create new directories alongside the existing `src/`:
   - `agents/`          # Agent definitions (one file per agent + factory)
   - `tools/`           # Tool interfaces & implementations
   - `rag/`             # RAG abstractions (wraps existing src/vectorstore.py + src/ingestion.py)
   - `models/`          # Model configuration (Ollama init_chat_model wrapper)
   - `orchestration/`   # Incident state store, message bus, agent wiring
   - `storage/`         # Incident store (JSON-backed shared state)
   - `.github/workflows/`  # CI/CD

   Each directory gets an `__init__.py`. No existing files are moved or deleted.

2. Update `requirements.txt` — add:
   - `deepagents>=0.4.0`
   - `langgraph>=0.4.0` (DeepAgents dependency, pin explicitly)
   - `ruff>=0.9.0` (linter for CI)

3. Create `models/config.py` — a single reusable model factory:
   - Function `get_model()` that calls `langchain.chat_models.init_chat_model("ollama:qwen2.5-coder:14b")` with shared defaults (temperature, timeout, base_url from existing `Settings`)
   - Falls back to existing `src/config.py` `Settings` for Ollama host/port
   - All agents import from here — single model, single config point

Phase 1 — Tool Interfaces (Scaffolding Only)

Create 8 tool files in `tools/`, each exporting a decorated `@tool` function (or class with `__call__`) with full docstrings, typed parameters, and Pydantic return schemas. No heavy logic — just the interface contract.

4. `tools/kubectl.py` — `KubectlTool`
   - `kubectl_exec(command: str, namespace: str = "default", output_format: str = "json") -> dict`
   - Allowlist: `get`, `describe`, `logs`, `top`, `events`. Blocks `delete`, `apply` (unless `dry_run=True`)
   - Real cluster mode: calls `subprocess.run(["kubectl", ...])` with timeout
   - Simulation fallback: if kubeconfig missing, reads from `data/sample/` files matching the resource type
   - Returns structured `{"status": "ok"|"error", "output": ..., "command": ...}`

5. `tools/cluster_snapshot.py` — `ClusterSnapshotTool`
   - `collect_snapshot(namespace: str = "default") -> dict`
   - Orchestrates multiple kubectl calls (pods, events, resource quotas)
   - Returns `{"pods": [...], "events": [...], "metrics": {...}, "timestamp": ...}`

6. `tools/log_analysis.py` — `LogAnalysisTool`
   - `analyze_logs(pod_name: str, namespace: str, tail_lines: int = 100) -> dict`
   - Fetches logs via KubectlTool, summarizes, detects anomaly patterns (OOM, crash, connection refused)
   - Returns `{"summary": str, "anomalies": [...], "raw_tail": str}`

7. `tools/manifest_validator.py` — `ManifestValidatorTool`
   - `validate_manifest(yaml_content: str, dry_run: bool = True) -> dict`
   - YAML parse check, resource limits check, image format check
   - Optional `kubectl apply --dry-run=client` via KubectlTool
   - Returns `{"valid": bool, "issues": [...], "dry_run_result": ...}`

8. `tools/rag_retrieval.py` — `RAGRetrievalTool`
   - `retrieve(query: str, top_k: int = 5, source_type: str | None = None) -> dict`
   - Wraps existing `src/vectorstore.py` `MilvusStore.search()` + `rerank()`
   - `source_type` filter: `"runbook"`, `"event"`, `"manifest"`, `"log"`, `"incident"`
   - Returns `{"results": [{"content": str, "metadata": dict, "score": float}]}`

9. `tools/root_cause.py` — `RootCauseHypothesisTool`
   - `generate_hypotheses(symptoms: dict) -> dict`
   - Takes cluster snapshot + log anomalies, produces ranked hypotheses
   - Returns `{"hypotheses": [{"cause": str, "confidence": float, "tests": [str]}]}`

10. `tools/fix_generator.py` — `FixGeneratorTool`
    - `generate_fix(hypothesis: dict, manifest: str | None = None) -> dict`
    - Produces patch YAML, kubectl commands, risk score
    - Returns `{"patches": [...], "commands": [...], "risk_score": float, "explanation": str}`

11. `tools/verification.py` — `VerificationTool`
    - `verify_fix(fix: dict, cluster_state: dict) -> dict`
    - Predicts outcome, checks for missing steps, flags risks
    - Returns `{"likely_effective": bool, "missing_steps": [...], "risks": [...]}`

Phase 2 — Agent Definitions

Each agent is a DeepAgents subagent dict (or `create_deep_agent` call) registered in `agents/`.

12. `agents/knowledge.py` — Knowledge Agent
    - Tools: `RAGRetrievalTool`
    - System prompt: "You retrieve and explain Kubernetes failure documentation."
    - Wraps existing `src/rag_chain.py` retrieval + reranking logic

13. `agents/investigator.py` — Investigator Agent (ReAct)
    - Tools: `KubectlTool`, `ClusterSnapshotTool`, `LogAnalysisTool`, `RootCauseHypothesisTool`
    - System prompt: "You are a K8s investigator. Follow observe → hypothesize → test loops."
    - This is the primary ReAct agent — DeepAgents' built-in planning (`write_todos`) drives the loop

14. `agents/remediation.py` — Remediation Agent
    - Tools: `FixGeneratorTool`, `ManifestValidatorTool`
    - System prompt: "You generate safe remediations for Kubernetes failures."

15. `agents/verification.py` — Verification Agent
    - Tools: `VerificationTool`, `KubectlTool` (read-only subset)
    - System prompt: "You evaluate proposed fixes and flag risks."

16. `agents/orchestrator.py` — Orchestrator Agent (top-level)
    - Created via `create_deep_agent()` with subagents: Investigator, Knowledge, Remediation, Verification
    - Tools: `ClusterSnapshotTool` (for initial triage)
    - System prompt: "You are the K8s Failure Intelligence coordinator. Delegate diagnostic, knowledge, remediation, and verification tasks to your subagents."
    - Uses `model="ollama:qwen2.5-coder:14b"` (from `models/config.py`)

Phase 3 — Shared State & Wiring

17. `storage/incident_store.py` — Shared incident state
    - Pydantic model `Incident` with fields: `id`, `status`, `symptoms`, `snapshots`, `hypotheses`, `selected_fix`, `verification_result`, `timeline`
    - JSON-backed persistence (extends pattern from `src/memory.py` `DiskChatMemory`)
    - Agents read/write incident state between delegations

18. `orchestration/wiring.py` — Agent factory
    - `create_k8s_agent() -> CompiledStateGraph` that assembles the full orchestrator with all subagents and tools
    - Single entry point for both CLI and API

Phase 4 — Integration with Existing App

19. Update `src/api.py` — add new endpoint:
    - `POST /agent/diagnose` — invokes orchestrator agent, streams back via SSE (same pattern as existing `/diagnose/stream`)
    - Keep existing `/diagnose` and `/diagnose/stream` endpoints working unchanged
    - Add `GET /agent/incidents` and `GET /agent/incidents/{id}` for incident history

20. Update `scripts/chat.py` — add `--agent` flag:
    - When `--agent` is passed, use the orchestrator agent instead of the existing RAG chain
    - Same rich terminal output

21. Update `templates/index.html` / `static/app.js`:
    - Add toggle or new panel for "Agent Mode" vs "Classic RAG Mode"
    - Agent mode hits `/agent/diagnose`, classic uses existing `/diagnose/stream`

Phase 5 — RAG Enhancement

22. `rag/vectorstore_adapter.py` — thin adapter over existing `src/vectorstore.py`:
    - Adds multi-collection support (separate collections for: runbooks, incidents, live_logs, snapshots)
    - Exposes metadata-filtered search for the `RAGRetrievalTool`
    - The existing `MilvusStore` class stays untouched — adapter calls into it

23. `rag/indexer.py` — extends `src/ingestion.py`:
    - Adds `index_incident()` — saves resolved incidents back into the vector store
    - Adds `index_log_chunk()` — for live log ingestion during investigation
    - Wraps existing `ingest_file()` / `ingest_directory()`

Phase 6 — CI/CD (GitHub Actions)

24. Create `.github/workflows/ci.yml`:

    How it works:
    - Trigger: Runs on every push to `main` and on all pull requests
    - Lint job: Checks out code → installs `ruff` → runs `ruff check .` and `ruff format --check .` to enforce code style
    - Test job: Checks out code → starts Milvus via `docker compose up -d` → installs Python deps → runs `pytest tests/ -v` → reports results. Uses a service container for Milvus or the existing `docker-compose.yml`
    - Build job (depends on lint + test passing): Builds a Docker image from a new `Dockerfile` (multi-stage: Python 3.11-slim, copies src/agents/tools/rag/models/orchestration/storage, installs deps, runs uvicorn)
    - Push job (only on `main` branch merge): Tags image with commit SHA + `latest`, pushes to GitHub Container Registry (ghcr.io) using `GITHUB_TOKEN`
    - Each job runs independently where possible (lint ∥ test), build waits for both, push waits for build
    - Secrets needed: none beyond `GITHUB_TOKEN` (auto-provided) for GHCR

25. Create `Dockerfile` — multi-stage build:
    - Stage 1: install Python dependencies
    - Stage 2: copy app code, expose port 8000, run `uvicorn src.api:app`

Migration Order (What to do first)

Order | What | Why
1 | Create directories + `models/config.py` | Foundation, zero breakage
2 | Tool interfaces (scaffolding) | Define contracts before agents
3 | `KubectlTool` + `RAGRetrievalTool` (implement) | Most critical tools, prove the pattern
4 | Knowledge Agent + Investigator Agent | First two agents using real tools
5 | Orchestrator with subagent wiring | Minimum viable multi-agent
6 | `POST /agent/diagnose` endpoint + CLI flag | Both interfaces get agent mode
7 | Remaining tools + agents | Remediation, Verification
8 | RAG enhancements (multi-collection, live indexing) | Advanced features
9 | CI/CD + Dockerfile | Production readiness

Components Reused From Current Codebase

Current File | Reuse Strategy
`src/config.py` | Directly imported by `models/config.py` for Ollama host, Milvus settings
`src/vectorstore.py` | Wrapped by `rag/vectorstore_adapter.py` and `tools/rag_retrieval.py`
`src/ingestion.py` | Wrapped by `rag/indexer.py` — same loaders, new entry points
`src/memory.py` | Reused as-is for session memory; pattern extended for `storage/incident_store.py`
`src/rag_chain.py` | Classification logic reused by Orchestrator for initial triage; RAG pipeline wrapped by Knowledge Agent
`src/api.py` | Extended with new `/agent/*` endpoints — existing endpoints unchanged
`scripts/chat.py` | Extended with `--agent` flag
`docker-compose.yml` | Reused as-is for Milvus; extended with app service in Dockerfile
`tests/test_basic.py` | Kept; new tests added in `tests/test_tools.py`, `tests/test_agents.py`
`data/sample/*` | Used as simulation fallback for KubectlTool when no real cluster is available

Verification

- After Phase 0: `pytest tests/test_basic.py` still passes, `pip install -e .` works
- After Phase 1-2: `python -c "from tools.kubectl import kubectl_exec"` imports without error
- After Phase 3-4: `python -c "from orchestration.wiring import create_k8s_agent; a = create_k8s_agent()"` creates the agent graph
- After Phase 5: Existing `/diagnose/stream` still works; new `/agent/diagnose` returns agent-driven diagnosis
- After Phase 6: `gh workflow run ci.yml` passes all jobs; `docker build .` succeeds

Decisions

- Single model: `qwen2.5-coder:14b` via Ollama for all agents
- No rewrite: Every existing file stays where it is; new code wraps/extends
- Safety: `KubectlTool` allowlists read-only commands; `apply` forced to `--dry-run=client` unless explicit override
- DeepAgents subagent pattern: Orchestrator is the top-level `create_deep_agent()`; other agents are subagent dicts passed via `subagents=[...]`
- Simulation fallback: KubectlTool detects missing kubeconfig and transparently serves sample data

Ready when you are — tell me which phase to start implementing, or if you'd like adjustments to the plan.
