"""DeepAgents definitions for Kubernetes diagnosis."""

import os
from typing import Optional
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

from src.tools import ALL_TOOLS
from src.config import get_settings

# ─────────────────────────────────────────────────────────────────────────────
# Model factory — configurable via AGENT_MODEL env var
# ─────────────────────────────────────────────────────────────────────────────

def _get_model():
    """Create the LLM. Override model with AGENT_MODEL env var."""
    settings = get_settings()
    model_name = os.environ.get("AGENT_MODEL", "ollama:qwen2.5-coder:14b")
    return init_chat_model(
        model_name,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        num_predict=1024,   # limit output tokens for speed
    )

# ─────────────────────────────────────────────────────────────────────────────
# Investigator Agent (ReAct loop)
# ─────────────────────────────────────────────────────────────────────────────

_INVESTIGATOR_PROMPT = """\
You are a Kubernetes failure investigator. Diagnose why workloads fail.

Tools: kubectl_exec, cluster_snapshot, analyze_logs, retrieve_docs, generate_hypotheses, generate_fix, verify_fix, validate_manifest.

Process: collect state → analyze logs → retrieve docs → hypothesize → fix → verify.

Call ONE tool at a time. After getting results, call the next tool or give your final answer.

When done, respond in plain text with:
Root Cause: [issue]
Evidence: [logs/events]
Fix: [commands]
Risk Level: [low/medium/high]

Do NOT call tools after giving the final answer."""


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
# Helper: Run investigator agent on a query
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(query: str) -> dict:
    """
    Run the investigator agent on a query.
    
    Args:
        query: user's question about a k8s failure
    
    Returns:
        dict with agent response
    """
    agent = create_investigator_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    # Extract the last message from agent
    return result


async def diagnose_stream(query: str):
    """
    Stream agent response tokens.
    
    Args:
        query: user's question
    
    Yields:
        token strings
    """
    agent = create_investigator_agent()
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": query}]},
        version="v2"
    ):
        if event["event"] == "on_chain_stream":
            chunk = event.get("data", {}).get("chunk", {})
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    if hasattr(msg, "content"):
                        yield msg.content
