"""DeepAgents definitions for Kubernetes diagnosis."""

from typing import Optional
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

from src.tools import ALL_TOOLS
from src.config import get_settings

# ─────────────────────────────────────────────────────────────────────────────
# Investigator Agent (ReAct loop)
# ─────────────────────────────────────────────────────────────────────────────

_INVESTIGATOR_PROMPT = """\
You are a Kubernetes failure investigator. Your job is to diagnose why workloads are failing.

You have access to tools for:
- Cluster inspection: kubectl_exec, cluster_snapshot, analyze_logs
- Knowledge retrieval: retrieve_docs
- Root cause analysis: generate_hypotheses
- Remediation: generate_fix, verify_fix, validate_manifest

Your process:
1. Collect cluster state and logs via cluster_snapshot and analyze_logs
2. Retrieve relevant documentation with retrieve_docs
3. Generate hypotheses with generate_hypotheses
4. For the most likely cause, generate_fix and verify_fix
5. Return a structured diagnosis with root cause, evidence, and fix

Be thorough but concise. Always cite evidence from logs, events, and documentation.
"""


def create_investigator_agent():
    """Create a ReAct-based Kubernetes investigator agent."""
    settings = get_settings()
    
    model = init_chat_model(
        "ollama:qwen2.5-coder:14b",
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )
    
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
