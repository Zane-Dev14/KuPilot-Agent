Plan: DeepAgents Multi-Agent K8s Failure Intelligence — Simplified

TL;DR — Extend your working K8s diagnosis app into a multi-agent DeepAgents system. Keep the existing structure flat: no new directories. Add four new subagent definitions directly in `src/agents.py`, extend `scripts/agent.py` with orchestrator mode, and add `/agent/diagnose` endpoint to `src/api.py`. Reuse all existing tools (src/tools.py), config (src/config.py), vectorstore (src/vectorstore.py), and incident state (src/incident.py).

---

## Completed Work

✅ **Phase 0 (Done)**
- `requirements.txt`: deepagents==0.4.0, langgraph>=0.4.0, langchain>=1.2.0, all dependencies resolved and working
- `src/config.py`: Existing Settings reused globally
- `src/tools.py`: 8 @tool functions fully implemented (kubectl_exec, cluster_snapshot, analyze_logs, validate_manifest, retrieve_docs, generate_hypotheses, generate_fix, verify_fix)
- `src/agents.py`: create_investigator_agent() defined and working

✅ **Phase 1 (Done)**
- `scripts/agent.py`: Unified CLI with three modes (interactive, test, demo)
  - interactive: Full ReAct loop with tool execution
  - test: Direct 4-step tool test (no agent)
  - demo: Synchronous 5-step workflow demo
- Tool execution loop working: agents detect and execute tool calls, feed results back

✅ **Requirements Status**
- All pip conflicts resolved (pydantic, pymilvus, sentence-transformers, langchain-core)
- Environment fully functional: `./.venv/bin/python scripts/agent.py --mode test` ✓

---

## Next Steps (What Remains)

### Phase 2 — Add Subagents to `src/agents.py`

1. **Knowledge Agent** — RAG + Documentation
   - Tools: `retrieve_docs` only
   - Prompt: "You are a Kubernetes knowledge expert. Retrieve and explain relevant runbooks and documentation from the enterprise knowledge base."
   - Function: `create_knowledge_agent()` → DeepAgent instance

2. **Remediation Agent** — Fix Generation & Validation
   - Tools: `generate_fix`, `validate_manifest`
   - Prompt: "You are a remediation specialist. Generate safe, risk-assessed Kubernetes fixes and validate their YAML."
   - Function: `create_remediation_agent()` → DeepAgent instance

3. **Verification Agent** — Safety & Risk Assessment
   - Tools: `verify_fix`, `kubectl_exec` (read-only: get, describe, logs)
   - Prompt: "You are a verification expert. Evaluate proposed fixes for safety, missing steps, and unintended consequences."
   - Function: `create_verification_agent()` → DeepAgent instance

4. **Orchestrator Agent** — Multi-Agent Coordinator
   - **Subagents**: Investigator, Knowledge, Remediation, Verification
   - Tools: `cluster_snapshot` (for initial triage)
   - Prompt: "You are the K8s Failure Intelligence Orchestrator. When a failure is reported: (1) Collect cluster snapshot, (2) Delegate investigation to Investigator agent, (3) Retrieve knowledge via Knowledge agent, (4) Generate fixes via Remediation agent, (5) Validate via Verification agent. Synthesize all results into a final diagnosis."
   - Function: `create_orchestrator_agent()` → DeepAgent with subagents
   - **Implementation detail**: Pass other agents via the `subagents=[...]` parameter of `create_deep_agent()`, or langgraph StateGraph if needed

---

### Phase 3 — Update CLI (`scripts/agent.py`)

1. Add new mode: `--mode orchestrator`
   - Invokes the orchestrator agent from `src/agents.py`
   - Same tool execution loop as current interactive mode
   - Example: `python scripts/agent.py --mode orchestrator`

2. Keep existing modes (interactive, test, demo) unchanged

3. Example test:
   ```bash
   ./.venv/bin/python scripts/agent.py --mode orchestrator
   # Enter: "Why is my data-processor pod crashing?"
   # Agent orchestrates all subagents and returns synthesized diagnosis
   ```

---

### Phase 4 — Update Web API (`src/api.py`)

1. Add endpoint: `POST /agent/diagnose`
   - Request body: `{"query": "Why is pod X crashing?", "max_steps": 5}`
   - Delegates to orchestrator agent
   - Returns: `{"type": "diagnosis", "root_cause": str, "evidence": [...], "fix": {...}, "verification": {...}}`
   - Status code: 200 on success, 500 on agent error

2. Keep existing `/diagnose` and `/diagnose/stream` unchanged (classic RAG mode)

3. Example:
   ```bash
   curl -X POST http://localhost:8000/agent/diagnose \
     -H "Content-Type: application/json" \
     -d '{"query": "Pod OOMKilled", "max_steps": 5}'
   ```

---

### Phase 5 — Testing & Validation

1. Test each subagent independently:
   ```python
   from src.agents import create_knowledge_agent, create_remediation_agent
   agent = create_knowledge_agent()
   result = agent.invoke({"messages": [{"role": "user", "content": "OOMKilled"}]})
   ```

2. Test orchestrator end-to-end:
   ```bash
   python scripts/agent.py --mode orchestrator
   ```

3. Verify all tools are accessible:
   ```bash
   python -c "from src.tools import ALL_TOOLS; print(len(ALL_TOOLS))"
   # Expected: 8
   ```

---

## Architecture Summary

```
User Query
    ↓
Orchestrator Agent (DeepAgents, subagents enabled)
    ├→ Investigator (subagent) → tools: kubectl_exec, cluster_snapshot, analyze_logs, generate_hypotheses
    ├→ Knowledge (subagent) → tools: retrieve_docs
    ├→ Remediation (subagent) → tools: generate_fix, validate_manifest
    └→ Verification (subagent) → tools: verify_fix
         ↓
    Synthesized Diagnosis + Fix + Verification Result
         ↓
    [CLI output] or [/agent/diagnose JSON response]
```

---

## File Changes Summary

| File | Change | Why |
|------|---------|----|
| `src/agents.py` | Add 4 agent factory functions (Knowledge, Remediation, Verification, Orchestrator) | Enable multi-agent coordination |
| `scripts/agent.py` | Add `--mode orchestrator` | CLI access to orchestrator |
| `src/api.py` | Add `POST /agent/diagnose` | Web API access to orchestrator |
| `src/tools.py` | No change | Reuse all 8 existing tools |
| `src/config.py` | No change | Reuse existing model factory |
| `src/incident.py` | No change | Reuse existing state model |
| `src/vectorstore.py` | No change | Reuse existing MilvusStore |
| `requirements.txt` | No change | Already has deepagents, langgraph, all deps |

---

## Verification Checklist

- [ ] `from src.agents import create_knowledge_agent, create_remediation_agent, create_verification_agent, create_orchestrator_agent` — all import without error
- [ ] `python scripts/agent.py --mode orchestrator` — runs and accepts queries
- [ ] `curl -X POST http://localhost:8000/agent/diagnose -d '{"query": "Pod failing"}'` — returns 200 with diagnosis
- [ ] Existing `/diagnose` endpoint unchanged — still works
- [ ] All 8 tools callable via orchestrator agent

---

## Migration Order

1. **Add subagent factory functions to `src/agents.py`** (Knowledge, Remediation, Verification, Orchestrator)
2. **Extend `scripts/agent.py` with `--mode orchestrator`**
3. **Add `POST /agent/diagnose` to `src/api.py`**
4. **Test each subagent and orchestrator**
5. **Verify backward compatibility of existing endpoints**

---

## Decisions Made

- **No new directories**: Everything stays in `src/` and `scripts/` for simplicity
- **Reuse all existing code**: No rewrites — only additions/extensions
- **Single model**: All agents use same `qwen2.5-coder:14b` via Ollama
- **Subagent pattern**: Orchestrator uses DeepAgents' native subagent support (`subagents=[...]` in `create_deep_agent()`)
- **Tool allowlisting**: Investigator gets full kubectl read access; Verification gets read-only subset
- **Backward compatible**: Existing RAG chain, CLI modes, and API endpoints unchanged
