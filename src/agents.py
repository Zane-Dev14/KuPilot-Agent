"""DeepAgents definitions for Kubernetes diagnosis."""

import json
import logging
import os
import re
import time

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config import get_settings
from src.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model factory — configurable via AGENT_MODEL env var
# ─────────────────────────────────────────────────────────────────────────────

_MAX_STEPS = 6  # hard ceiling for tool-call iterations


def _get_model():
    """Create the LLM. Override model with AGENT_MODEL env var."""
    settings = get_settings()
    model_name = os.environ.get("AGENT_MODEL", "ollama:qwen2.5-coder:14b")
    return init_chat_model(
        model_name,
        base_url=settings.ollama_base_url,
        temperature=0,       # deterministic output
        top_p=1,             # disable nucleus sampling
        num_predict=4096,    # allow full answers
        num_ctx=32768,       # large context prevents loops & memory loss
    )

# ─────────────────────────────────────────────────────────────────────────────
# Investigator Agent (ReAct loop)
# ─────────────────────────────────────────────────────────────────────────────

_INVESTIGATOR_PROMPT = """\
You are a Kubernetes failure investigator. Your job is to diagnose why \
workloads fail and provide actionable remediation.

═══════════════════════════════════════════════════════════════
FORBIDDEN TOOLS — You MUST NOT call any of these:
  write_todos, ls, read_file, write_file, edit_file, glob,
  grep, execute, task
They are NOT available for Kubernetes diagnosis. Ignore them.
═══════════════════════════════════════════════════════════════

AVAILABLE TOOLS (ONLY these 8 exist):
1. kubectl_exec(command, namespace)
   Run read-only kubectl: get pods, describe pod <name>, logs <name>,
   top pods, events. Use REAL pod names, never placeholders like <pod>.
2. cluster_snapshot(namespace)
   Quick snapshot of pods + events in a namespace.
3. analyze_logs(pod_name, namespace, tail_lines)
   Detect anomaly patterns (OOM, CrashLoop, ImagePull, Scheduling)
   in a pod's logs.
4. retrieve_docs(query, top_k, source_type)
   Search runbooks and event docs in the vector store.
5. generate_hypotheses(symptoms)
   Produce ranked root-cause hypotheses from a symptom description.
6. generate_fix(hypothesis, manifest_yaml)
   Produce remediation commands and patches for a hypothesis.
7. verify_fix(fix_commands, cluster_health_check)
   Evaluate whether a proposed fix is likely to succeed.
8. validate_manifest(yaml_content, dry_run)
   Validate a Kubernetes YAML manifest.

WORKFLOW:
  1. Gather state   → kubectl_exec / cluster_snapshot
  2. Analyze        → analyze_logs / retrieve_docs
  3. Hypothesize    → generate_hypotheses
  4. Fix            → generate_fix → verify_fix
  5. STOP and give final answer.

TERMINATION RULES (critical):
- Call at most 4 tools total, then produce your final answer.
- After each tool result, decide: do I have enough info? If yes → answer.
- If a tool returns an error, proceed with what you have — do NOT retry.
- NEVER output JSON in your final answer. Plain text only.
- Your final response MUST NOT contain a tool call.

OUTPUT FORMAT (plain text):
  Root Cause: <one-line summary>
  Evidence: <specific logs, events, or metrics from tools>
  Fix: <commands or steps>
  Risk Level: LOW | MEDIUM | HIGH
  Next Steps: <monitoring or follow-up actions>

CONTEXT-AWARE QUESTIONS:
- If user asks about resource properties (memory, CPU, limits, requests),
  call kubectl_exec with "describe pod <name>" and report the resources
  section.
- If user says "that pod" or "the pod", infer the pod name from earlier
  context in the conversation.
- For simple knowledge questions (what is OOMKilled), answer directly
  from your knowledge — no tools needed.

Do NOT use write_todos or any planning tools. Only use the 8 tools above."""


def create_investigator_agent():
    """Create a ReAct-based Kubernetes investigator agent."""
    model = _get_model()

    agent = create_deep_agent(
        name="k8s-investigator",
        model=model,
        tools=ALL_TOOLS,
        system_prompt=_INVESTIGATOR_PROMPT,
    )

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry & text-parsed tool calls
# ─────────────────────────────────────────────────────────────────────────────

def _build_tool_registry():
    """Map tool name → tool object for manual invocation."""
    return {
        getattr(t, "name", "") or getattr(t, "__name__", ""): t
        for t in ALL_TOOLS
        if getattr(t, "name", None) or getattr(t, "__name__", None)
    }


def _parse_tool_calls(text):
    """Extract tool calls from text when model emits them as JSON in content."""
    if not isinstance(text, str):
        return []
    calls = []
    for m in re.finditer(
        r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', text, re.DOTALL
    ):
        try:
            p = json.loads(m.group(1))
            if isinstance(p, dict) and "name" in p and "arguments" in p:
                calls.append({"name": p["name"], "arguments": p["arguments"]})
        except json.JSONDecodeError:
            pass
    if not calls:
        s = text.strip()
        if s.startswith("{"):
            try:
                p = json.loads(s)
                if isinstance(p, dict) and "name" in p and "arguments" in p:
                    calls.append({"name": p["name"], "arguments": p["arguments"]})
            except json.JSONDecodeError:
                pass
    return calls


def _truncate(text, max_len=800):
    """Truncate tool output to prevent prompt bloat."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n...(truncated, {len(text)} total chars)"


# Forbidden tool names injected by DeepAgents middleware
_FORBIDDEN_TOOLS = frozenset({
    "write_todos", "ls", "read_file", "write_file", "edit_file",
    "glob", "grep", "execute", "task",
})


# ─────────────────────────────────────────────────────────────────────────────
# Diagnose — step-guarded agent execution
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(query: str, max_steps: int | None = None) -> dict:
    """Run the investigator agent with a step-limited tool loop.

    Args:
        query: user's question about a k8s failure.
        max_steps: override default step ceiling (default: ``_MAX_STEPS``).

    Returns:
        ``{"response": str, "steps": list[dict]}``.
    """
    max_steps = max_steps or _MAX_STEPS
    agent = create_investigator_agent()
    registry = _build_tool_registry()
    messages = [HumanMessage(content=query)]
    steps: list[dict] = []
    last_ai: AIMessage | None = None

    for step in range(1, max_steps + 1):
        t0 = time.time()
        result = agent.invoke({"messages": messages})
        llm_ms = int((time.time() - t0) * 1000)

        # Extract messages from result
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
        elif hasattr(result, "messages"):
            messages = result["messages"]
        else:
            return {"response": str(result), "steps": steps}

        if not messages:
            break

        last = messages[-1]
        if not isinstance(last, AIMessage):
            continue
        last_ai = last

        # Collect tool calls (structured first, then text-parsed fallback)
        tool_calls = getattr(last, "tool_calls", None) or []
        parsed = _parse_tool_calls(last.content) if not tool_calls else []

        normalized: list[tuple[str, dict, str]] = []
        for c in tool_calls:
            normalized.append((
                c.get("name"),
                c.get("args") or c.get("arguments") or {},
                c.get("id", c.get("name")),
            ))
        for c in parsed:
            normalized.append((c["name"], c["arguments"], c["name"]))

        if not normalized:
            # No tool calls → final answer
            steps.append({"step": step, "type": "final", "llm_ms": llm_ms})
            logger.info("Step %d: final answer (%dms)", step, llm_ms)
            return {"response": last.content, "steps": steps}

        # Execute each tool call
        for tool_name, tool_args, call_id in normalized:
            si: dict = {
                "step": step, "type": "tool_call",
                "name": tool_name, "args": tool_args, "llm_ms": llm_ms,
            }

            # Block forbidden built-in tools
            if tool_name in _FORBIDDEN_TOOLS:
                msg = (
                    f"Tool '{tool_name}' is not available for Kubernetes "
                    f"diagnosis. Use ONLY: {', '.join(registry.keys())}"
                )
                si["error"] = msg
                steps.append(si)
                messages.append(
                    ToolMessage(content=msg, name=tool_name, tool_call_id=call_id)
                )
                logger.warning("Step %d: blocked forbidden tool %s", step, tool_name)
                continue

            tool = registry.get(tool_name)
            if not tool:
                err = (
                    f"Tool '{tool_name}' not found. "
                    f"Available: {', '.join(registry.keys())}"
                )
                si["error"] = err
                steps.append(si)
                messages.append(
                    ToolMessage(content=err, name=tool_name, tool_call_id=call_id)
                )
                logger.warning("Step %d: unknown tool %s", step, tool_name)
                continue

            try:
                t1 = time.time()
                tool_result = tool.invoke(tool_args)
                tool_ms = int((time.time() - t1) * 1000)
                result_str = json.dumps(tool_result, indent=2)
                si["result"] = result_str[:500]
                si["tool_ms"] = tool_ms
                steps.append(si)
                logger.info(
                    "Step %d: %s → %d chars (%dms)",
                    step, tool_name, len(result_str), tool_ms,
                )
                messages.append(
                    ToolMessage(
                        content=_truncate(result_str),
                        name=tool_name,
                        tool_call_id=call_id,
                    )
                )
            except Exception as exc:
                si["error"] = str(exc)
                steps.append(si)
                logger.error("Step %d: %s failed: %s", step, tool_name, exc)
                messages.append(
                    ToolMessage(
                        content=f"Error: {exc}",
                        name=tool_name,
                        tool_call_id=call_id,
                    )
                )

    # ── Max steps reached — force final synthesis ────────────────────────
    logger.warning("Max steps (%d) reached — forcing final synthesis", max_steps)
    messages.append(
        HumanMessage(
            content=(
                "You have reached the maximum number of tool calls. "
                "Provide your FINAL diagnosis now based on ALL evidence "
                "gathered so far. Do NOT call any more tools. "
                "Use plain text only."
            )
        )
    )
    result = agent.invoke({"messages": messages})
    if isinstance(result, dict) and "messages" in result:
        msgs = result["messages"]
    elif hasattr(result, "messages"):
        msgs = result["messages"]
    else:
        return {"response": str(result), "steps": steps}

    for msg in reversed(msgs):
        if isinstance(msg, AIMessage) and msg.content:
            steps.append({"step": max_steps + 1, "type": "forced_final"})
            return {"response": msg.content, "steps": steps}

    # Absolute fallback
    fallback = last_ai.content if last_ai and last_ai.content else "Unable to produce diagnosis."
    return {"response": fallback, "steps": steps}


async def diagnose_stream(query: str):
    """Stream agent diagnosis as a single yielded result.

    Runs the synchronous ``diagnose()`` and yields the response.

    Args:
        query: user's question.

    Yields:
        Response text.
    """
    import asyncio

    result = await asyncio.to_thread(diagnose, query)
    yield result.get("response", "")
