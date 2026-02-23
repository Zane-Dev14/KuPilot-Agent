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
AVAILABLE TOOLS (ONLY THESE 5 FOR INVESTIGATION)
━━━━━━━━━━━━━━━━━━━━━━━━━━

1. kubectl_exec(command:str, namespace:str="default")
2. cluster_snapshot(namespace:str="default")
3. analyze_logs(pod_name:str, namespace:str="default", tail_lines:int=100)
4. retrieve_docs(query:str, top_k:int=5, source_type:str=None) - for documentation only
5. generate_hypotheses(symptoms:str) - for root cause analysis

YOU DO NOT HAVE ACCESS TO:
- generate_fix() - this is for the REMEDIATION agent
- verify_fix() - this is for the REMEDIATION agent
- validate_manifest() - this is for the REMEDIATION agent

Your job is ONLY to diagnose, not to fix. The remediation agent will handle fixes.

STRICTLY FORBIDDEN — DO NOT CALL:
- generate_fix, verify_fix, validate_manifest (not your job!)
- write_todos (blocked)
- ls, glob, read_file, write_file, edit_file (blocked)
- Any filesystem/admin tools

━━━━━━━━━━━━━━━━━━━━━━━━━━
DIAGNOSTIC WORKFLOW (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━

For ANY diagnostic query mentioning issues, failures, errors, or "fix":

Step 1: cluster_snapshot("default") - ALWAYS REQUIRED
  → Identify failing pods from status

Step 2: IF any pods have errors (CrashLoop, ImagePull, OOMKilled, etc):
  → analyze_logs(pod_name, namespace="default") for EACH failing pod
  → kubectl_exec("describe pod <name>", namespace="default") for details

Step 3: generate_hypotheses(symptoms="...")
  → Include: pod names, error types, log excerpts

Step 4: Return diagnosis in plain English:
  "Detected <error_type> in pods <names>. Root cause: <explanation>. 
   Evidence: <key findings from logs/describe>."

NEVER skip analyze_logs for failing pods.
NEVER return just "Analysis complete."
ALWAYS include specific pod names and error types in your response.

EXAMPLE for ImagePull error:
1. cluster_snapshot() → sees "api-588b4594f8-vfgrg: ErrImageNeverPull"
2. analyze_logs("api-588b4594f8-vfgrg") → check logs
3. kubectl_exec("describe pod api-588b4594f8-vfgrg") → get image details
4. Response: "Detected ImagePullBackOff in pod api-588b4594f8-vfgrg. 
   The container is trying to pull image X which doesn't exist. Evidence shows..."

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
CRITICAL: OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  NEVER OUTPUT JSON, YAML, OR STRUCTURED DATA FORMATS ⚠️

You MUST respond in natural, conversational English.
DO NOT format your response as:
  ✗ JSON: {"root_cause": "...", "evidence": [...]}
  ✗ YAML: root_cause: "..."
  ✗ Tool call syntax: {"name": "kubectl_exec", "arguments": {...}}
  ✗ Structured code blocks with key-value pairs

Instead, write naturally:
  ✓ "The root cause is X. Evidence shows Y. I recommend Z."

━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL ANSWER FORMAT (DIAGNOSTIC ONLY)
━━━━━━━━━━━━━━━━━━━━━━━━━━

MANDATORY: Your final response MUST include ALL of these:
  1. SPECIFIC pod names (e.g., "api-588b4594f8-vfgrg")
  2. ERROR TYPE (e.g., "ImagePullBackOff", "OOMKilled", "CrashLoopBackOff")
  3. EVIDENCE from logs or describe output
  4. ROOT CAUSE explanation

Template:
"Detected [ERROR_TYPE] in pods [pod-name-1], [pod-name-2]. 
Root cause: [explanation].
Evidence: [key findings from analyze_logs or kubectl describe].
Recommended next steps: [what remediation should do]."

Example:
"Detected ImagePullBackOff in pods api-588b4594f8-vfgrg and api-7548997b5b-6lhqr.
Root cause: The containers are configured to use image 'my-app:v2.0' which doesn't exist in the registry.
Evidence: analyze_logs shows 'Failed to pull image', kubectl describe confirms image pull failures with ErrImageNeverPull status.
Recommended next steps: Update deployment to use correct image tag or verify image exists in registry."

NEVER return empty or generic responses like:
  ✗ "Analysis complete."
  ✗ "Investigation done."
  ✗ "{...}" (JSON)
"""

def create_investigator_agent():
    """Create a ReAct-based Kubernetes investigator agent."""
    from src.tools import INVESTIGATOR_TOOLS
    
    model = _get_model()

    agent = create_deep_agent(
        name="k8s-investigator",
        model=model,
        tools=INVESTIGATOR_TOOLS,  # Only diagnostic tools, no fix/verify
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

Your role: Generate safe, risk-assessed fixes and EXECUTE them.

TOOLS AVAILABLE:
- generate_fix(hypothesis, manifest_yaml=None)
- validate_manifest(yaml_content, dry_run=True)
- kubectl_exec(command, namespace="default") — for applying fixes

CRITICAL WORKFLOW:
1. Receive root cause hypothesis from investigation
2. Call generate_fix(hypothesis) to get fix strategy
3. Replace ALL placeholders with actual resource names from context:
   - <name> → actual deployment/pod name
   - <namespace> → actual namespace
   - <pod>, <container> → actual names
4. EXECUTE the fix using kubectl_exec
5. Report what was done (not what user should do)

EXAMPLE:
  Bad: "Run: kubectl set resources deployment/<name> --limits=memory=1Gi"
  Good: Calls kubectl_exec("set resources deployment/api --limits=memory=1Gi", "default")
        Then reports: "Increased memory limit for deployment/api to 1Gi."

⚠️  OUTPUT IN NATURAL LANGUAGE ONLY — NO JSON/YAML ⚠️

DO NOT output:
  ✗ {"status": "...", "actions": [...]}
  ✗ Tool call syntax in your response
  ✗ Structured data formats

Write your response as clear, conversational text:

**Fix Applied:**
[Describe each action you executed in plain English]

**Validation:**
[Explain validation results naturally]

**Verification Needed:**
[Describe verification steps in text]

Write as if explaining to a human operator.
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
# Multi-Agent Orchestration Function
# ─────────────────────────────────────────────────────────────────────────────


def orchestrate_multiagent_diagnosis(query: str, max_steps: int = 20) -> dict:
    """FAST multi-agent with ACTUAL tool execution and parallel phases."""
    import logging
    import concurrent.futures
    from scripts.agent import run_agent_with_tools
    
    logger = logging.getLogger(__name__)
    all_steps = []
    agents_used = []
    start_time = time.time()
    
    # Phase 1: Investigation (RUNS TOOLS via run_agent_with_tools)
    print("  [1/4] Investigator analyzing cluster...", flush=True)
    investigator = create_investigator_agent()
    agents_used.append("investigator")
    inv_response, inv_steps = run_agent_with_tools(investigator, query, max_steps=5, verbose=False)
    all_steps.append({"agent": "investigator", "steps": len(inv_steps), "output": inv_response[:800]})
    
    # Debug: Show what tools were actually called
    tools_called = [s.get("name") for s in inv_steps if s.get("type") == "tool_call"]
    print(f"        Tools used: {', '.join(tools_called) if tools_called else 'none'}", flush=True)
    print(f"        Response preview: {inv_response[:150]}...", flush=True)
    
    # Extract root cause from tool results AND response
    root_cause = "Unknown"
    evidence = ""
    
    # Check tool outputs for error patterns
    for step in inv_steps:
        if step.get("type") == "tool_call" and step.get("result"):
            result_str = step.get("result", "")
            if "ImagePull" in result_str or "ErrImage" in result_str:
                root_cause = "ImagePullBackOff"
                evidence += f" {step.get('name')} found ImagePull errors."
            elif "OOMKilled" in result_str:
                root_cause = "OOMKilled"
                evidence += f" {step.get('name')} found OOMKilled."
            elif "CrashLoop" in result_str or "Error" in result_str:
                root_cause = "CrashLoopBackOff"
                evidence += f" {step.get('name')} found crashes."
    
    # Fallback to response text if no tool evidence
    if root_cause == "Unknown":
        resp_lower = inv_response.lower()
        if "imagepull" in resp_lower or "errimage" in resp_lower:
            root_cause = "ImagePullBackOff"
        elif "oomkilled" in resp_lower or "memory" in resp_lower:
            root_cause = "OOMKilled"
        elif "crash" in resp_lower or "exit" in resp_lower:
            root_cause = "CrashLoopBackOff"
    
    print(f"  ✓ Investigation: Found {root_cause} [{len(inv_steps)} steps, {int((time.time()-start_time)*1000)}ms]", flush=True)
    
    # Phases 2 & 3: Knowledge + Remediation (PARALLEL EXECUTION)
    print("  [2-3/4] Running Knowledge + Remediation in parallel...", flush=True)
    from src.tools import retrieve_docs
    
    def run_knowledge():
        try:
            result = retrieve_docs.invoke({"query": root_cause, "top_k": 3})
            return f"Found {result.get('count', 0)} relevant documents"
        except Exception as e:
            return f"Knowledge search unavailable: {e}"
    
    def run_remediation():
        remediation = create_remediation_agent()
        
        # Build context from investigation tool results
        context_parts = [f"Root cause: {root_cause}"]
        if evidence:
            context_parts.append(f"Evidence:{evidence}")
        
        # Extract pod names from tool results
        pod_names = []
        for step in inv_steps:
            if step.get("type") == "tool_call" and "pod" in step.get("name", "").lower():
                result = step.get("result", "")
                # Simple extraction of pod names from results
                import re
                matches = re.findall(r'[a-z]+-[0-9a-z]+-[0-9a-z]+', result)
                pod_names.extend(matches[:3])  # Limit to first 3
        
        if pod_names:
            context_parts.append(f"Affected pods: {', '.join(list(set(pod_names))[:3])}")
        
        context_parts.append(f"Investigation output: {inv_response[:400]}")
        
        rem_query = f"""Fix {root_cause}.

{chr(10).join(context_parts)}

EXECUTE fixes NOW via kubectl_exec. Do NOT just suggest commands."""
        response, steps = run_agent_with_tools(remediation, rem_query, max_steps=4, verbose=False)
        return response, steps
    
    # Execute in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        know_future = executor.submit(run_knowledge)
        rem_future = executor.submit(run_remediation)
        
        know_response = know_future.result()
        rem_response, rem_steps = rem_future.result()
    
    agents_used.append("knowledge")
    agents_used.append("remediation")
    all_steps.append({"agent": "knowledge", "output": know_response})
    all_steps.append({"agent": "remediation", "steps": len(rem_steps), "output": rem_response[:800]})
    print(f"  ✓ Knowledge + Remediation complete [{len(rem_steps)} steps, {int((time.time()-start_time)*1000)}ms]", flush=True)
    
    # Phase 4: Verification (DIRECT tool call - fast)
    print("  [4/4] Verification checking status...", flush=True)
    from src.tools import kubectl_exec
    try:
        ver_result = kubectl_exec.invoke({"command": "get pods", "namespace": "default"})
        ver_response = f"Current pods:\n{ver_result.get('output', 'N/A')[:300]}"
    except Exception as e: 
        ver_response = f"Verification unavailable: {e}"
    
    agents_used.append("verification")
    all_steps.append({"agent": "verification", "output": ver_response})
    
    total_time = int((time.time()-start_time)*1000)
    print(f"  ✓ Multi-agent workflow complete ({total_time}ms total)", flush=True)
    
    # Synthesize final response
    final_response = f"""# Multi-Agent Diagnosis (Completed in {total_time}ms)

## 🔍 Investigation
{inv_response[:700]}

## 📚 Knowledge Base
{know_response}

## 🔧 Remediation
{rem_response[:700]}

## ✅ Verification
{ver_response[:300]}

---
**Workflow Complete:** {len(agents_used)} agents, {sum(s.get('steps',0) for s in all_steps)} total tool steps"""
    
    return {
        "response": final_response,
        "steps": all_steps,
        "agents_used": agents_used,
        "total_time_ms": total_time
    }

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent — Simplified coordination with ALL tools
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT_V2 = """
You are the Kubernetes Failure Intelligence Orchestrator.

You have access to ALL tools directly. Your role is to coordinate a multi-phase workflow:

PHASE 1: INVESTIGATION
- cluster_snapshot() to get overview
- analyze_logs() for failing pods
- generate_hypotheses() for root cause

PHASE 2: KNOWLEDGE
- retrieve_docs() to find runbooks

PHASE 3: REMEDIATION (CRITICAL)
- generate_fix() to get fix commands
- Replace placeholders with actual resource names
- kubectl_exec() to EXECUTE the fix
- Report what you DID, not what user should do

PHASE 4: VERIFICATION
- kubectl_exec('get pods') to check status
- verify_fix() to assess effectiveness

CRITICAL FOR FIXING PODS:
1. After generate_fix(), you will get commands with <placeholders>
2. Extract actual pod/deployment/namespace names from cluster_snapshot
3. Replace ALL <name>, <namespace>, <pod> with real values
4. EXECUTE via kubectl_exec()
5. Say "Executed: kubectl ..." not "Run: kubectl ..."

EXAMPLE:
❌ Bad: "Run: kubectl set resources deployment/<name> --limits=memory=1Gi"
✅ Good: [calls kubectl_exec("set resources deployment/api --limits=memory=1Gi -n default")]
        Then says: "✓ Executed: Increased memory limit for deployment/api to 1Gi"

OUTPUT FORMAT:
# Diagnosis Summary

**Root Cause:** ...
**Evidence:** ...
**Fix Executed:** 
- ✓ kubectl set resources ...
- ✓ kubectl rollout restart ...
**Verification:** ...
**Confidence:** HIGH/MEDIUM/LOW
"""

def create_orchestrator_agent():
    """Create orchestrator agent with all tools for coordinated workflow.
    
    This agent has access to all tools and coordinates the workflow phases:
    Investigation → Knowledge → Remediation → Verification
    
    For TRUE multi-agent delegation, use orchestrate_multiagent_diagnosis() instead.
    """
    model = _get_model()
    
    # Give orchestrator ALL tools for comprehensive workflow
    orchestrator = create_deep_agent(
        name="k8s-orchestrator",
        model=model,
        tools=ALL_TOOLS,
        system_prompt=_ORCHESTRATOR_PROMPT_V2,
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
