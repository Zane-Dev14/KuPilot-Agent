# Kubernetes Failure Intelligence Copilot — Phase 1 (Revised)

**TL;DR:** Switch embeddings to `bge-m3` (384-dim, multilingual), add `bge-reranker-v2-m3` as a reranking layer in retrieval, implement a query analyzer that extracts K8s metadata (namespace, pod, error type) and decomposes complex queries, ingest Kubernetes Events as structured documents, and use `deepseek-r1:32b` as a conditional fallback model for high-uncertainty or reasoning-heavy diagnoses. The default model remains lighter; deepseek activates only when query complexity warrants detailed reasoning.

## Steps

### Phase 1a: Core infrastructure (unchanged from original plan)

1. Create project folder structure (same layout)
2. Create `requirements.txt` — updated:
   - Change `BAAI/bge-large-en-v1.5` → embedding service will use `bge-m3` (384-dim)
   - Add `sentence-transformers>=3.0.0` (for bge-reranker-v2-m3)
   - No new packages needed for reranker or query analyzer (use Hugging Face & LCEL)
3. Create `config.py` — add:
   - `EMBEDDING_MODEL` default → `"BAAI/bge-m3"`
   - `EMBEDDING_DIMENSION` → `384` (was 1024)
   - `RERANKER_MODEL` → `"BAAI/bge-reranker-v2-m3"`
   - `SIMPLE_MODEL` → `"llama3.1"` (default for simple queries)
   - `COMPLEX_MODEL` → `"deepseek-r1:32b"` (fallback for reasoning)
   - `QUERY_COMPLEXITY_THRESHOLD` → `0.7` (confidence score to trigger deepseek)
4. Create Milvus `docker-compose.yml` (same)
5. Create `src/embeddings/embedding_service.py` — use `bge-m3`, verify 384-dim output
6. Create `src/vectorstore/milvus_client.py` — update collection schema dim to `384`

### Phase 1b: Retrieval pipeline with query analysis & reranking

7. **Create `src/retrieval/query_analyzer.py`** (new module)
   - `QueryMetadata` class: holds extracted `namespace`, `pod`, `container`, `node`, `error_type`, `labels_dict`
   - `QueryAnalyzer` class with methods:
     - `extract_k8s_metadata(query: str) -> QueryMetadata` — use regex + heuristics to extract K8s object references (e.g., "pod crashed in namespace prod" → namespace="prod", pod=None, error_type="crashed")
     - `decompose_query(query: str) -> list[str]` — if multi-part (using semicolon, "and", "also"), split into sub-queries
     - `analyze(query: str) -> tuple[QueryMetadata, list[str]]` — unified entry point
   - Include type hints, logging, structured output

8. **Create `src/retrieval/reranker.py`** (new module)
   - `RerankerService` class wrapping `bge-reranker-v2-m3`
   - Use `sentence-transformers.CrossEncoder` or `langchain_community.CrossEncoderReranker`
   - Methods:
     - `rerank(documents: list[Document], query: str, top_k: int = 4) -> list[Document]` — retrieve K+N docs, rerank, return top K
     - Support metadata-based filtering (filter by extracted namespace/pod before reranking)
   - Singleton pattern, lazy load model
   
9. **Update `src/retrieval/retriever.py`** (new file)
   - `EnhancedRetriever` LCEL chain composing:
     - Input: `query: str`
     - Query analyzer (extract metadata + decompose)
     - For each sub-query:
       - Vector search via Milvus (with metadata filters if applicable)
       - Rerank results via `RerankerService`
     - Merge and deduplicate results
     - Output: `list[Document]`
   - Use `RunnableParallel` for parallel sub-query retrieval
   - Use `RunnableLambda` for filter application and reranking

### Phase 1c: Document ingestion (extended)

10. **Create `src/ingestion/events_loader.py`** (new module)
    - `KubernetesEventsLoader` extends `BaseDocumentLoader`
    - Parses Kubernetes Event objects (JSON or YAML format)
    - Extracts:
      - `event_type` (Warning, Normal)
      - `reason` (CrashLoopBackOff, OOMKilled, FailedScheduling, etc.)
      - `involved_object` (Kind, Name, Namespace, etc.)
      - `first_timestamp`, `last_timestamp`, `count`
      - `message` (human-readable description)
    - Metadata: `kind="Event"`, `reason`, `involved_object_kind`, `involved_object_name`, `namespace`, `event_type`
    - Documents grouped by reason + involved object, chunked per event or time window
    - High relevance for RAG — events are the primary signal of failures

11. **Update `src/ingestion/pipeline.py`**
    - Extend auto-detect to handle event files (`.events.json`, `events/` directories)
    - Select `KubernetesEventsLoader` for event inputs

### Phase 1d: LLM strategy & chains

12. **Create `src/chains/model_selector.py`** (new module)
    - `ModelSelector` class to choose between simple and complex models
    - `estimate_query_complexity(query: str) -> float` — heuristic scoring (0.0-1.0)
      - Check for keywords like "why", "explain", "diagnose", "troubleshoot" → +0.3
      - Check for multi-part decomposition count → +0.2 per part above 1
      - Check for long query length → +0.1
      - Check for ambiguous language (`maybe`, `could`, `possibly`) → +0.2
    - If score >= `QUERY_COMPLEXITY_THRESHOLD` → use `deepseek-r1:32b`; else use `SIMPLE_MODEL`
    - Support override via optional `force_model` parameter in request

13. **Update `src/chains/rag_chain.py`** (revised)
    - Retrieve: call `EnhancedRetriever`
    - Assess complexity: call `ModelSelector.select_model(query)`
    - Generate diagnosis:
      - Prompt format includes retrieved documents + K8s metadata extracted by query analyzer
      - For deepseek: expose reasoning chain in response (include `thinking` tokens if available)
      - For simple model: standard diagnosis output
    - Output: `FailureDiagnosis` with added field `reasoning_model_used: str`

### Phase 1e: Updated models & API

14. **Update `src/models/schemas.py`**
    - `QueryMetadata`: namespace, pod, container, node, error_type, labels_dict
    - `FailureDiagnosis`: add `reasoning_model_used: str`, `thinking_chain: str | None` (for deepseek reasoning)
    - `DiagnoseRequest`: add optional `force_model: str` (override model selection)

15. **Update `src/api/server.py`** (FastAPI)
    - `/diagnose` endpoint now receives query, passes through `ModelSelector`, streams reasoning if applicable
    - `/query-analysis` endpoint (debug) to show extracted metadata + decomposed sub-queries

### Phase 1f: Sample data & testing

16. **Extend `data/sample/`**
    - Add `events/` directory with sample K8s event JSONs:
      - CrashLoopBackOff events
      - OOMKilled events
      - ImagePullBackOff events
      - FailedScheduling events
    - Coordinate with manifest and log samples to reference the same pods/nodes

17. **Update `scripts/ingest.py`**
    - Add `--events-dir` or auto-detection

### Phase 1g: Verification

- Vector dimension now 384 (bge-m3) — verify in Milvus collection schema
- Query analyzer extracts metadata correctly — test with "Why is pod web-app-123 crashing in namespace prod?"
- Reranker reorders documents by relevance — test with complex query retrieving 10 docs, rerank to top 4
- Events loader parses events correctly — test with sample event JSON
- Model selector activates deepseek for complex queries — test with reasoning-heavy prompts
- Integration test: ingest manifests + logs + events, query, check response includes model used

## Decisions

- **`bge-m3` over `bge-large-en-v1.5`**: Newer, multilingual, 384-dim is sufficiently rich for K8s context while reducing embedding storage and search latency by ~3x. Trade-off: slightly lower absolute recall, but reranking recovers it.
- **`bge-reranker-v2-m3` (not lighter alternative)**: Consistency with embedding family and superior accuracy worth the computation cost. Reranking is done once per query (not every vector search), so latency impact is ~100-200ms per request — acceptable for diagnosis workflows.
- **Query analyzer with both metadata extraction and decomposition**: Metadata filtering enables precise retrieval (e.g., "failures in namespace X" filters before search). Decomposition helps with multi-part diagnostic questions like "Why is my pod crashing AND how do I fix the image registry issue?"
- **`deepseek-r1:32b` as fallback, not default**: Reasoning models are 4-8x slower inference; reserve for genuinely complex cases. Simple queries (e.g., "What does CrashLoopBackOff mean?") should use fast model for immediate response.
- **Events as first-class documents**: K8s Events are high-signal, structured, and queryable. Ingesting them alongside logs/manifests enables queries like "Show me all Pod events in the last hour" and powers pattern detection.
- **Expose reasoning chain**: For transparency and debugging, deepseek's thinking process is included in `thinking_chain` field when used, so users understand the diagnosis.
