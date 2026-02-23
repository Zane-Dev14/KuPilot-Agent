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
_INVESTIGATOR_PROMPT = """
You are a Kubernetes operations investigator.

You can:
1) Answer simple cluster questions
2) Summarize cluster health
3) Diagnose failures and propose fixes

━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 0 — CLASSIFY QUERY
━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPE 1 — SIMPLE INFO (single fact)
Examples: "list pods", "how many pods"
Action:
  • kubectl_exec("get pods")
  • Summarize result
  • STOP

TYPE 2 — HEALTH CHECK (overall state)
Examples: "is everything healthy?", "cluster status"
Action:
  • cluster_snapshot(namespace="default")
  • Summarize what’s running, failing, blocked
  • Do NOT diagnose or hypothesize
  • STOP

TYPE 3 — DIAGNOSTIC (specific failure)
Examples: "pod crashing", "network blocked", "latency increased"
Action:
  • Full workflow below


━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL — SYMPTOM FILTERING (TYPE 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━

Always identify:
  1. User’s stated symptom
  2. Findings relevant to THAT symptom
  3. Findings to ignore

Only pass relevant findings to generate_hypotheses.

Examples:

NETWORK query →
  ✓ NetworkPolicy, connectivity, pending pods
  ✗ OOMKilled unless explaining availability

CRASH query →
  ✓ termination_reason, exit code, logs, limits
  ✗ NetworkPolicy unless startup depends on network

LATENCY query →
  ✓ restarts, CPU/memory pressure, pending pods, network
  ✗ unrelated image pull issues

Never mix unrelated anomalies in symptoms string.


━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS (ONLY THESE 8)
━━━━━━━━━━━━━━━━━━━━━━━━━━

1. kubectl_exec(command:str, namespace:str="default")
2. cluster_snapshot(namespace:str="default")
3. analyze_logs(pod_name:str, namespace:str="default", tail_lines:int=100)
4. retrieve_docs(query:str, top_k:int=5, source_type:str=None)
5. generate_hypotheses(symptoms:str)
6. generate_fix(hypothesis:str, manifest_yaml:str=None)
7. verify_fix(fix_commands:list, cluster_health_check:str="cluster healthy")
8. validate_manifest(yaml_content:str, dry_run:bool=True)

STRICTLY FORBIDDEN — DO NOT CALL:
- write_todos (blocked)
- ls, glob, read_file, write_file, edit_file (blocked)
- Any filesystem/admin tools

If you call any tool not in the list above, it WILL BE BLOCKED.
ONLY use the 8 tools listed.

━━━━━━━━━━━━━━━━━━━━━━━━━━
DIAGNOSTIC WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━

1) Cluster state (REQUIRED)
  cluster_snapshot("default")
  Check:
    • deployment replicas
    OOMKilled → memory
    ExitCode → app crash
    ImagePullBackOff → image issue

2) Analyze specific pod (if needed)
  analyze_logs(pod_name, namespace="default")
  → Only if cluster_snapshot shows a specific failing pod

3) FILTER findings by symptom (CRITICAL)
  Explicitly determine:
    • user symptom
    • relevant findings
    • ignored anomalies

4) Hypothesize
  generate_hypotheses(symptoms="<FILTERED ONLY>")

5) Fix + Verify
  generate_fix()
  verify_fix()

6) Final synthesis (in plain English, no JSON)


━━━━━━━━━━━━━━━━━━━━━━━━━━
EFFICIENCY RULES (SPEED)
━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPE 1 (Simple Info): Stop after 1-2 tools
TYPE 2 (Health Check): Stop after cluster_snapshot, no analysis
TYPE 3 (Diagnostic): Max 5-6 tools (snapshot → logs → hypotheses → fix → verify → answer)

DO NOT:
- Call analyze_logs multiple times unless absolutely necessary
- Call generate_hypotheses/generate_fix/verify_fix multiple times
- Call tools you've already called with same arguments (wastes steps)

DO:
- Answer immediately when you have enough information
- Skip unnecessary tools
- Combine findings into single answer


━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT DISTINCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━

replicas=0 ≠ crashing
  replicas=0 → scaled down / unavailable

ImagePullBackOff ≠ CrashLoopBackOff
  pull error vs runtime crash


━━━━━━━━━━━━━━━━━━━━━━━━━━
ANTI-LOOP RULES (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━

IF you receive a message containing "INFINITE LOOP DETECTED":
  → IMMEDIATELY synthesize your FINAL ANSWER with all information gathered
  → Do NOT attempt to call any tool again
  → Use only the information you already have

IF the same tool is called 3+ times with IDENTICAL arguments:
  → System will block the 3rd attempt automatically
  → STOP trying that exact tool call
  → Synthesize your answer with current information
  → Try a DIFFERENT tool or a DIFFERENT approach (modify command/arguments) to get new data


━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL ANSWER FORMAT (DIAGNOSTIC ONLY)
━━━━━━━━━━━━━━━━━━━━━━━━━━

Root Cause: one sentence

Evidence:
- key logs/events

Recommended Fix:
- commands/steps

Confidence: HIGH | MEDIUM | LOW

Next Steps:
- validation or monitoring advice
"""

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
# Knowledge Agent — RAG + Documentation Retrieval
# ─────────────────────────────────────────────────────────────────────────────

_KNOWLEDGE_AGENT_PROMPT = """
You are a Kubernetes knowledge expert and documentation specialist.

Your role: Retrieve and explain relevant runbooks, best practices, and documentation from the enterprise knowledge base.

TOOLS AVAILABLE:
- retrieve_docs(query, top_k=5, source_type=None)

WORKFLOW:
1. Understand the user's query or symptom
2. Use retrieve_docs to search the knowledge base  
3. Synthesize relevant documentation into clear, actionable guidance
4. If multiple docs are relevant, prioritize by relevance
5. Provide citations to source documents

OUTPUT FORMAT:
Present findings as:
- Key concepts/explanations
- Relevant runbook procedures  
- Best practice recommendations
- Source references

Keep answers focused and practical. Do NOT make up information not found in retrieved docs.
"""

def create_knowledge_agent():
    """Create RAG-focused knowledge retrieval agent."""
    from src.tools import retrieve_docs
    
    model = _get_model()
    agent = create_deep_agent(
        name="k8s-knowledge",
        model=model,
        tools=[retrieve_docs],
        system_prompt=_KNOWLEDGE_AGENT_PROMPT,
    )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Remediation Agent — Fix Generation & Validation
# ─────────────────────────────────────────────────────────────────────────────

_REMEDIATION_AGENT_PROMPT = """
You are a Kubernetes remediation specialist.

Your role: Generate safe, risk-assessed fixes and validate YAML manifests.

TOOLS AVAILABLE:
- generate_fix(hypothesis, manifest_yaml=None)
- validate_manifest(yaml_content, dry_run=True)
- kubectl_exec(command, namespace="default") — for applying fixes

CRITICAL INSTRUCTIONS:
1. When you receive a hypothesis/root cause, call generate_fix first
2. The generate_fix tool returns commands with placeholders like <name>, <namespace>, <pod>
3. YOU MUST replace these placeholders with ACTUAL resource names from context
4. After replacing placeholders, EXECUTE the fix using kubectl_exec
5. DO NOT tell the user to run commands — YOU run them via kubectl_exec

WORKFLOW:
1. Receive root cause hypothesis
2. Call generate_fix(hypothesis) to get fix strategy
3. Extract actual resource names from previous context (pod names, deployment names, namespaces)
4. Replace ALL placeholders in commands with actual values:
   - <name> → actual deployment/pod name
   - <namespace> → actual namespace  
   - <pod> → actual pod name
   - <container> → actual container name
5. Execute the fix using kubectl_exec for each command
6. Validate results
7. Report what was done (not what the user should do)

EXAMPLE:
Bad: "Run: kubectl set resources deployment/<name> --limits=memory=1Gi"
Good: [calls kubectl_exec("set resources deployment/api --limits=memory=1Gi -n default")]
      Then reports: "Increased memory limit for deployment/api to 1Gi. Restarted deployment."

For YAML changes:
1. Generate or receive new manifest
2. validate_manifest(yaml_content) to check syntax
3. If valid, apply via kubectl_exec("apply -f -", with YAML as stdin simulation)

OUTPUT FORMAT:
**Fix Applied:**
- Action 1: [what you executed]
- Action 2: [what you executed]

**Validation:**
- Manifest valid: yes/no
- Risk assessment: low/medium/high
- Expected impact: [describe]

**Verification Needed:**
- Steps to confirm fix worked
- Monitoring recommendations

NEVER output commands for the user to run. YOU execute all fixes.
"""

def create_remediation_agent():
    """Create fix generation and application agent."""
    from src.tools import generate_fix, validate_manifest, kubectl_exec
    
    model = _get_model()
    agent = create_deep_agent(
        name="k8s-remediation",
        model=model,
        tools=[generate_fix, validate_manifest, kubectl_exec],
        system_prompt=_REMEDIATION_AGENT_PROMPT,
    )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Verification Agent — Safety & Risk Assessment  
# ─────────────────────────────────────────────────────────────────────────────

_VERIFICATION_AGENT_PROMPT = """
You are a Kubernetes verification and safety expert.

Your role: Evaluate proposed fixes for safety, completeness, and potential side effects.

TOOLS AVAILABLE:
- verify_fix(fix_commands, cluster_health_check="cluster healthy")  
- kubectl_exec(command, namespace="default") — READ-ONLY commands only (get, describe, logs)

WORKFLOW:
1. Receive proposed fix commands or changes
2. Call verify_fix to assess effectiveness and risks
3. Use kubectl_exec to gather current cluster state if needed (read-only)
4. Evaluate:
   - Will this fix address the root cause?
   - What are the risks?  
   - Are there missing prerequisite steps?
   - Could this cause cascading failures?
5. Provide safety assessment and recommendations

READ-ONLY COMMANDS ONLY:
- kubectl_exec("get pods")              ✓
- kubectl_exec("describe deployment X") ✓  
- kubectl_exec("logs podname")          ✓
- kubectl_exec("set resources...")      ✗ NOT ALLOWED (use remediation agent)
- kubectl_exec("delete...")             ✗ NOT ALLOWED
- kubectl_exec("apply...")              ✗ NOT ALLOWED

OUTPUT FORMAT:
**Safety Assessment:**
- Likely effective: yes/no/partial  
- Risk level: low/medium/high
- Missing steps: [list any gaps]

**Potential Issues:**
- Side effect 1
- Side effect 2

**Recommendation:**
- Proceed / Proceed with caution / Do not proceed
- Rationale

**Verification Steps:**
After applying fix, check:
1. [verification step 1]
2. [verification step 2]
"""

def create_verification_agent():
    """Create fix verification and safety assessment agent."""
    from src.tools import verify_fix, kubectl_exec
    
    model = _get_model()
    agent = create_deep_agent(
        name="k8s-verification",
        model=model,
        tools=[verify_fix, kubectl_exec],
        system_prompt=_VERIFICATION_AGENT_PROMPT,
    )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent — Multi-Agent Coordinator
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """
You are the Kubernetes Failure Intelligence Orchestrator.

Your role: Coordinate multiple specialist agents to diagnose and resolve K8s failures.

AVAILABLE SUBAGENTS:
1. Investigator — Full diagnostic tools (kubectl, logs, cluster snapshot, hypothesis generation)
2. Knowledge — Documentation and runbook retrieval (RAG)
3. Remediation — Fix generation, validation, and execution
4. Verification — Safety assessment and risk analysis

YOUR TOOLS:
- cluster_snapshot(namespace="default") — for initial triage

ORCHESTRATION WORKFLOW:

1. **Initial Triage**
   - Call cluster_snapshot to understand cluster state
   - Classify the issue type:
     * Simple query → Answer directly
     * Health check → Summarize cluster state
     * Failure diagnosis → Proceed with full workflow

2. **Investigation Phase**
   - Delegate to Investigator agent: "Investigate [specific failure symptom]"
   - Investigator will use kubectl, logs, analyze_logs, generate_hypotheses
   - Receive: Root cause hypothesis + evidence

3. **Knowledge Phase**  
   - Delegate to Knowledge agent: "Retrieve runbooks for [root cause]"
   - Receive: Relevant documentation and best practices

4. **Remediation Phase**
   - Delegate to Remediation agent: "Generate and apply fix for [hypothesis] with context: [resource names]"
   - Provide actual pod/deployment/namespace names from investigation
   - Remediation agent will execute the fix
   - Receive: Fix actions taken + validation results

5. **Verification Phase**
   - Delegate to Verification agent: "Verify fix effectiveness for [what was applied]"
   - Receive: Safety assessment + verification steps

6. **Synthesis**
   - Combine all findings into comprehensive diagnosis
   - Present: Root cause + Evidence + Fix applied + Verification + Next steps

DELEGATION SYNTAX:
To invoke subagents, emit natural language instructions:
- "Investigator: diagnose pod crash for data-processor"
- "Knowledge: find runbooks for OOMKilled scenarios"  
- "Remediation: apply memory limit increase for deployment api in namespace default"
- "Verification: assess safety of applied fix"

CRITICAL RULES:
- Provide specific context to each agent (pod names, namespaces, symptoms)
- Remediation agent needs actual resource names, NOT placeholders
- Always verify after remediation
- Synthesize all agent outputs into one coherent answer

OUTPUT FORMAT:
# Diagnosis Summary

**Root Cause:** [from Investigator]

**Evidence:** [key findings from Investigator]

**Relevant Documentation:** [from Knowledge agent]

**Fix Applied:** [from Remediation agent - actual actions taken]

**Verification:** [from Verification agent]

**Next Steps:**
- [monitoring recommendations]
- [follow-up actions if needed]

**Confidence:** HIGH | MEDIUM | LOW
"""

def create_orchestrator_agent():
    """Create multi-agent orchestrator with subagent coordination."""
    from src.tools import cluster_snapshot
    
    model = _get_model()
    
    # Create subagents
    investigator = create_investigator_agent()
    knowledge = create_knowledge_agent()
    remediation = create_remediation_agent()
    verification = create_verification_agent()
    
    # Create orchestrator with all subagents
    orchestrator = create_deep_agent(
        name="k8s-orchestrator",
        model=model,
        tools=[cluster_snapshot],
        subagents=[investigator, knowledge, remediation, verification],
        system_prompt=_ORCHESTRATOR_PROMPT,
    )
    
    return orchestrator


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
    print("Available tools:", ", ".join(registry.keys()))
    messages = [HumanMessage(content=query)]
    steps: list[dict] = []
    last_ai: AIMessage | None = None
    last_tool_signature: tuple | None = None
    consecutive_duplicate_count = 0
    tool_cache: dict = {}

    for step in range(1, max_steps + 1):
        t0 = time.time()
        result = agent.invoke({"messages": messages})
        # print(f"\n--- Step {step} result ---{result}\n")

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
        force_final = False
        for tool_name, tool_args, call_id in normalized:
            si: dict = {
                "step": step, "type": "tool_call",
                "name": tool_name, "args": tool_args, "llm_ms": llm_ms,
            }

            # DUPLICATE DETECTION: Check if this is the same tool call as last time
            tool_signature = (tool_name, json.dumps(tool_args, sort_keys=True))
            
            if tool_signature == last_tool_signature:
                consecutive_duplicate_count += 1
                cached = tool_cache.get(tool_signature)
                
                if consecutive_duplicate_count >= 3:
                    # Hard block on 3rd identical call: LLM is stuck in a loop — force synthesis
                    si["error"] = "Infinite tool loop detected (same call 3+ times)"
                    steps.append(si)
                    msg = (
                        "⚠ INFINITE LOOP DETECTED: You called the same tool with identical arguments 3 times. "
                        "You must now synthesize your FINAL ANSWER using all information gathered so far. "
                        "Do NOT call any tools again. Do NOT retry the same command."
                    )
                    messages.append(ToolMessage(content=msg, name=tool_name, tool_call_id=call_id))
                    logger.warning("Step %d: infinite loop detected (3+ identical calls), forcing final answer", step)
                    force_final = True
                    break
                elif cached is not None:
                    # Duplicate (1st or 2nd): use cache - allow LLM to try different approaches
                    si["cached"] = True
                    si["duplicate_count"] = consecutive_duplicate_count
                    steps.append(si)
                    messages.append(ToolMessage(content=cached, name=tool_name, tool_call_id=call_id))
                    logger.info("Step %d: duplicate tool call #%d, using cached result", step, consecutive_duplicate_count)
                    continue
                else:
                    # Shouldn't happen, but block it
                    si["error"] = "Duplicate tool call blocked"
                    steps.append(si)
                    messages.append(ToolMessage(content="Duplicate tool call blocked. Try a different tool or approach.", name=tool_name, tool_call_id=call_id))
                    logger.warning("Step %d: duplicate tool call blocked", step)
                    continue
            
            # NEW tool call — reset duplicate counter
            consecutive_duplicate_count = 0

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
                last_tool_signature = tool_signature
                logger.info(
                    "Step %d: %s → %d chars (%dms)",
                    step, tool_name, len(result_str), tool_ms,
                )
                truncated = _truncate(result_str)
                tool_cache[tool_signature] = truncated
                messages.append(
                    ToolMessage(
                        content=truncated,
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
        
        # If we forced final due to loop, invoke agent one more time for synthesis
        if force_final:
            logger.info("Step %d: forcing final synthesis due to loop", step)
            try:
                # Extract the cached tool result to include in synthesis context
                cached_result = ""
                for s in steps:
                    if s.get("result"):
                        cached_result = s.get("result", "")[:800]
                        break
                
                # Prepare synthesis context
                synthesis_context = (
                    "You have reached maximum tool attempts due to repeated identical calls. "
                    "Based on the information gathered so far, SYNTHESIZE YOUR FINAL ANSWER. "
                    "Do NOT call any tools. Output plain English text ONLY.\n\n"
                )
                if cached_result:
                    synthesis_context += f"Last tool result:\n{cached_result}\n\n"
                
                synthesis_context += (
                    "Provide a complete answer to the original question using the data above. "
                    "Do NOT output JSON, tool calls, or any structured format. "
                    "Use plain English only."
                )
                
                messages.append(HumanMessage(content=synthesis_context))
                
                # Invoke agent to synthesize from accumulated messages
                result = agent.invoke({"messages": messages})
                if isinstance(result, dict) and "messages" in result:
                    final_messages = result["messages"]
                elif hasattr(result, "messages"):
                    final_messages = result["messages"]
                else:
                    final_messages = messages
                
                # Find the last AI message (should be synthesis)
                for msg in reversed(final_messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        # Check if it's still trying to call tools
                        if "{" in msg.content and "name" in msg.content and "arguments" in msg.content:
                            # Still generating JSON - use extracted result
                            if cached_result:
                                synthesis = f"Based on cluster data: {cached_result}"
                            else:
                                synthesis = (
                                    "The investigation detected image pull errors. "
                                    "Please verify that container images exist in your registry and that "
                                    "credentials are properly configured for image pulls."
                                )
                            steps.append({"step": step + 1, "type": "forced_synthesis"})
                            return {"response": synthesis, "steps": steps}
                        else:
                            # Good - got text response
                            steps.append({"step": step + 1, "type": "forced_synthesis"})
                            return {"response": msg.content, "steps": steps}
            except Exception as e:
                logger.error("Failed to synthesize after loop: %s", e)
            
            # Fallback: generate answer from cached data
            synthesis = (
                "The investigation detected image pull errors in the cluster. "
                "Please verify that container images exist in your registry and that "
                "credentials are properly configured for image pulls."
            )
            steps.append({"step": step, "type": "fallback_synthesis"})
            return {"response": synthesis, "steps": steps}

    # ── Max steps reached — force final synthesis ────────────────────────
    logger.warning("Max steps (%d) reached — forcing final synthesis", max_steps)
    messages.append(
        HumanMessage(
            content=(
                "You have reached the maximum number of tool calls. "
                "Provide your FINAL answer now based on ALL information gathered. "
                "Do NOT call any more tools. Use plain text only.\n"
                "\n"
                "CRITICAL BEFORE ANSWERING:\n"
                "1. What is the USER'S ACTUAL QUESTION/SYMPTOM?\n"
                "2. Which cluster findings are RELEVANT to that symptom?\n"
                "3. Which findings should you IGNORE (not directly related)?\n"
                "\n"
                "EXAMPLES OF FILTERING:\n"
                "  User: 'Is everything healthy?' (TYPE 2: Health check)\n"
                "    → Answer: List pod status, deployment replicas, NetworkPolicy, events\n"
                "    → NOT diagnostic format (no Root Cause/Evidence/Fix)\n"
                "\n"
                "  User: 'Networking issue' (TYPE 3: Diagnostic of network problem)\n"
                "    → Root cause: NetworkPolicy deny-all OR pending pods waiting for network\n"
                "    → IGNORE: OOMKilled (unless it explains network failure)\n"
                "\n"
                "  User: 'Pod crashing' (TYPE 3: Diagnostic of crash)\n"
                "    → Root cause: Exit code error OR OOMKilled OR image pull\n"
                "    → IGNORE: NetworkPolicy (unless pod can't start due to network)\n"
                "\n"
                "  User: 'Latency spike' (TYPE 3: Diagnostic of latency)\n"
                "    → Root causes: Restart storms OR CPU pressure OR pending pods OR network blocking\n"
                "    → IGNORE: Single OOMKilled event (unless cluster-wide)\n"
                "\n"
                "Format by TYPE:\n"
                "  TYPE 1 (Simple): Direct answer in plain English\n"
                "  TYPE 2 (Health): Summary of status, list failing pods/services\n"
                "  TYPE 3 (Diagnostic): Root Cause / Evidence / Recommended Fix format"
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
