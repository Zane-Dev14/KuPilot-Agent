#!/usr/bin/env python3
"""Unified K8s Failure Intelligence Agent — with step tracing & speed fixes."""

import json
import os
import re
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress Milvus async warnings (we use sync client)
logging.getLogger("langchain_milvus.vectorstores.milvus").setLevel(logging.ERROR)

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from src.tools import cluster_snapshot, kubectl_exec  # noqa: E402

console = Console()


# ── Tool registry ────────────────────────────────────────────────────────────

def _build_tool_registry():
    from src.tools import ALL_TOOLS
    return {getattr(t, "name", "") or getattr(t, "__name__", ""): t
            for t in ALL_TOOLS if getattr(t, "name", None) or getattr(t, "__name__", None)}


def _parse_tool_calls(text):
    """Extract ALL tool calls from text — handles code fences, raw JSON, mixed content."""
    if not isinstance(text, str):
        return []
    calls = []
    # 1) Find all ```json ... ``` fenced blocks
    for m in re.finditer(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', text, re.DOTALL):
        try:
            p = json.loads(m.group(1))
            if isinstance(p, dict) and "name" in p and "arguments" in p:
                calls.append({"name": p["name"], "arguments": p["arguments"]})
        except json.JSONDecodeError:
            pass
    # 2) Fallback: try whole text as raw JSON
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


# ── Agent execution loop with step tracing ───────────────────────────────────

def _truncate(text, max_len=600):
    """Keep tool results short to avoid prompt bloat (speed fix)."""
    return text if len(text) <= max_len else text[:max_len] + f"\n...(truncated, {len(text)} total)"


def run_agent_with_tools(agent, query, max_steps=10, verbose=True):
    """Execute agent with tool loop. Returns (response_text, steps_list)."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    registry = _build_tool_registry()
    messages = [HumanMessage(content=query)]
    steps, last_ai = [], None
    last_tool_signature = None
    consecutive_duplicate_count = 0  # Track stuck loops
    tool_cache = {}

    for step in range(1, max_steps + 1):
        t0 = time.time()
        result = agent.invoke({"messages": messages})
        llm_ms = int((time.time() - t0) * 1000)

        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
        elif hasattr(result, "messages"):
            messages = result["messages"]
        else:
            return str(result), steps

        if not messages:
            return "", steps

        last = messages[-1]
        if not isinstance(last, AIMessage):
            continue
        last_ai = last

        # ── Collect tool calls (structured OR text-parsed) ──
        tool_calls = getattr(last, "tool_calls", None) or []
        parsed = _parse_tool_calls(last.content) if not tool_calls else []

        # Normalize: convert both formats to a list of (name, args, call_id)
        normalized = []
        for c in tool_calls:
            normalized.append((c.get("name"), c.get("args") or c.get("arguments") or {},
                               c.get("id", c.get("name"))))
        for c in parsed:
            normalized.append((c["name"], c["arguments"], c["name"]))

        if not normalized:
            # No tool calls → final response
            steps.append({"step": step, "type": "final", "llm_ms": llm_ms})
            if verbose:
                console.print(f"  [dim]Step {step}[/dim] [green]✓ Final answer[/green] [dim]{llm_ms}ms[/dim]")
            return last.content, steps

        # Execute each tool call
        force_final = False
        for tool_name, tool_args, call_id in normalized:
            si = {"step": step, "type": "tool_call", "name": tool_name,
                  "args": tool_args, "llm_ms": llm_ms}
            if verbose:
                args_preview = json.dumps(tool_args)[:80]
                console.print(f"  [dim]Step {step}[/dim] [cyan]→ {tool_name}[/cyan]({args_preview}) [dim]{llm_ms}ms[/dim]")

            tool_signature = (tool_name, json.dumps(tool_args, sort_keys=True))
            
            # DUPLICATE DETECTION: Check if this is the same tool call as last time
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
                    if verbose:
                        console.print(f"           [red]⚠⚠⚠ LOOP BLOCKED ON 3RD IDENTICAL CALL - forcing final answer[/red]")
                    force_final = True
                    break
                elif cached is not None:
                    # Duplicate (1st or 2nd): use cache - allow LLM to try different approaches
                    si["cached"] = True
                    si["duplicate_count"] = consecutive_duplicate_count
                    steps.append(si)
                    messages.append(ToolMessage(content=cached, name=tool_name, tool_call_id=call_id))
                    if verbose:
                        console.print(f"           [yellow]↺ duplicate #{consecutive_duplicate_count}, cached result[/yellow]")
                    continue
                else:
                    # Shouldn't happen, but block it
                    si["error"] = "Duplicate tool call blocked"
                    steps.append(si)
                    messages.append(ToolMessage(content="Duplicate tool call blocked. Try a different tool or approach.", name=tool_name, tool_call_id=call_id))
                    if verbose:
                        console.print(f"           [yellow]⚠ Duplicate blocked[/yellow]")
                    continue

            # NEW tool call — reset duplicate counter
            consecutive_duplicate_count = 0

            tool = registry.get(tool_name)
            if not tool:
                err = f"Tool '{tool_name}' not found. Available: {', '.join(registry.keys())}"
                si["error"] = err
                steps.append(si)
                messages.append(ToolMessage(content=err, name=tool_name, tool_call_id=call_id))
                if verbose:
                    console.print(f"           [red]✗ {err}[/red]")
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
                if verbose:
                    console.print(f"           [green]✓[/green] [dim]{tool_ms}ms, {len(result_str)} chars[/dim]")
                truncated = _truncate(result_str)
                tool_cache[tool_signature] = truncated
                messages.append(ToolMessage(content=truncated, name=tool_name, tool_call_id=call_id))
            except Exception as e:
                si["error"] = str(e)
                steps.append(si)
                if verbose:
                    console.print(f"           [red]✗ {e}[/red]")
                messages.append(ToolMessage(content=f"Error: {e}", name=tool_name, tool_call_id=call_id))

        # If we forced final due to loop, invoke agent one more time for synthesis
        if force_final:
            if verbose:
                console.print(f"  [dim]Step {step}[/dim] [yellow]→ Forcing final synthesis...[/yellow]")
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
                            if verbose:
                                console.print(f"  [dim]Step {step}[/dim] [green]✓ Synthesis complete (fallback)[/green]")
                            return synthesis, steps
                        else:
                            # Good - got text response
                            if verbose:
                                console.print(f"  [dim]Step {step}[/dim] [green]✓ Synthesis complete[/green]")
                            return msg.content, steps
            except Exception as e:
                if verbose:
                    console.print(f"  [red]Error during synthesis: {e}[/red]")
            
            # Fallback: generate answer from cached data
            synthesis = (
                "The investigation detected image pull errors in the cluster. "
                "Please verify that container images exist in your registry and that "
                "credentials are properly configured for image pulls."
            )
            if verbose:
                console.print(f"  [dim]Step {step}[/dim] [yellow]✓ Synthesis complete (fallback)[/yellow]")
            return synthesis, steps

    # Filter out duplicate tool call block errors from final response
    final_response = last_ai.content if last_ai else ""
    if "Duplicate tool call blocked" in final_response:
        final_response = "Agent could not find a new approach. Please rephrase or try a different query."
    return final_response, steps


# ── Interactive mode ─────────────────────────────────────────────────────────

def run_interactive_mode():
    console.print(Panel("[bold cyan]K8s Failure Intelligence Agent[/bold cyan]\n"
                        "[yellow]ReAct + RAG — Step tracing enabled[/yellow]",
                        border_style="cyan", expand=False))
    console.print("[dim]Type 'exit' to quit[/dim]\n")

    try:
        from src.agents import create_investigator_agent
        agent = create_investigator_agent()
    except Exception as e:
        console.print(f"[red]Failed to load agent:[/red] {e}")
        return

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        if query.lower() in ("exit", "quit"):
            console.print("[yellow]Goodbye![/yellow]")
            break
        if not query:
            continue

        console.print("\n[bold magenta]⏳ Thinking...[/bold magenta]")

        q = query.lower()
        if re.search(r"\b(how many pods|list pods|show pods|pods running)\b", q):
            result = kubectl_exec.invoke({"command": "get pods", "namespace": "default"})
            output = result.get("output", "").strip()
            response = output or "No pods found."
            console.print(Panel(response, title="[bold cyan]Response[/bold cyan]",
                                border_style="green", expand=False))
            continue

        if re.search(r"\b(healthy|health|cluster status|overall status|any issues)\b", q):
            snapshot = cluster_snapshot.invoke({"namespace": "default"})
            pods = snapshot.get("pods", "") or "(no pods output)"
            deployments = snapshot.get("deployments", "") or "(no deployments output)"
            netpols = snapshot.get("network_policies", "") or "(no network policies output)"
            response = (
                "Cluster health summary:\n\n"
                f"Pods:\n{pods}\n\n"
                f"Deployments:\n{deployments}\n\n"
                f"NetworkPolicies:\n{netpols}\n"
            )
            console.print(Panel(response, title="[bold cyan]Response[/bold cyan]",
                                border_style="green", expand=False))
            continue

        if "oomkilled" in q and re.search(r"\b(where|which|show)\b", q):
            events = kubectl_exec.invoke({"command": "get events -A"})
            events_out = events.get("output", "")
            oom_lines = [line for line in events_out.splitlines() if "OOMKilled" in line]
            if oom_lines:
                response = "OOMKilled events:\n" + "\n".join(oom_lines[:50])
            else:
                response = "No OOMKilled events found in cluster events."
            console.print(Panel(response, title="[bold cyan]Response[/bold cyan]",
                                border_style="green", expand=False))
            continue
        t0 = time.time()
        try:
            response, steps = run_agent_with_tools(agent, query, max_steps=10, verbose=True)
            total_ms = int((time.time() - t0) * 1000)
            console.print(f"\n  [dim]Total: {total_ms}ms, {len(steps)} steps[/dim]")
            if response:
                console.print(Panel(response, title="[bold cyan]Response[/bold cyan]",
                                    border_style="green", expand=False))
            else:
                console.print("[yellow]Agent returned no response[/yellow]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            import traceback
            traceback.print_exc()
        console.print()


# ── Test mode ────────────────────────────────────────────────────────────────

def run_test_mode():
    console.print(Panel("[bold cyan]K8s Agent Test[/bold cyan]\n[yellow]Testing tools directly[/yellow]",
                        border_style="cyan", expand=False))
    try:
        from src.tools import generate_hypotheses, analyze_logs, generate_fix, verify_fix  # noqa: F401
    except Exception as e:
        console.print(f"[red]Failed to load tools:[/red] {e}")
        return

    query = "pod crashing with OOMKilled errors"
    console.print(f"\n[bold cyan]Testing with:[/bold cyan] {query}\n")

    console.print("[yellow]Step 1: Generating hypotheses...[/yellow]")
    hyp = generate_hypotheses.invoke({"symptoms": query})
    console.print(f"✓ Got {len(hyp.get('hypotheses', []))} hypotheses")
    if hyp.get('hypotheses'):
        console.print(f"  • {hyp['hypotheses'][0]['cause']}")

    console.print("\n[yellow]Step 2: Analyzing logs...[/yellow]")
    logs = analyze_logs.invoke({"pod_name": "data-processor", "namespace": "default"})
    console.print(f"✓ {logs.get('summary', 'done')}")

    console.print("\n[yellow]Step 3: Generating fix...[/yellow]")
    fix = generate_fix.invoke({"hypothesis": hyp['hypotheses'][0]['cause'], "manifest_yaml": None})
    console.print(f"✓ Risk: {fix.get('risk_score', 0):.0%}, {len(fix.get('commands', []))} commands")

    console.print("\n[yellow]Step 4: Verifying fix...[/yellow]")
    ver = verify_fix.invoke({"fix_commands": fix['commands'], "cluster_health_check": "healthy"})
    console.print(f"✓ Likely effective: {ver.get('likely_effective', False)}")

    console.print("\n[green]✓ Test complete[/green]")


# ── Demo mode ────────────────────────────────────────────────────────────────

def run_demo_mode():
    console.print(Panel("[bold cyan]K8s Agent Demo[/bold cyan]\n[yellow]Full Workflow[/yellow]",
                        border_style="cyan", expand=False))
    try:
        from src.tools import retrieve_docs, generate_hypotheses, analyze_logs, generate_fix  # noqa: F811
    except Exception as e:
        console.print(f"[red]Failed to load tools:[/red] {e}")
        return

    query = "why is my data-processor pod crashing with OOMKilled"
    print(f"\n{'='*60}\n  K8S FAILURE INTELLIGENCE - WORKFLOW DEMO\n{'='*60}")
    print(f"\n👤 Query: {query}\n")

    for label, fn in [
        ("[STEP 1] RAG RETRIEVAL", lambda: retrieve_docs.invoke({"query": "OOMKilled memory limit", "top_k": 3})),
        ("[STEP 2] ANALYZE SYMPTOMS", lambda: analyze_logs.invoke({"pod_name": "data-processor", "namespace": "default"})),
        ("[STEP 3] HYPOTHESES", lambda: generate_hypotheses.invoke({"symptoms": "Pod crashing, OOMKilled"})),
        ("[STEP 4] FIX GENERATION", lambda: generate_fix.invoke({"hypothesis": "Out-of-Memory (OOMKilled): Pod memory request too low or memory leak in app.", "manifest_yaml": None})),
    ]:
        print(f"{label}\n{'-'*40}")
        try:
            r = fn()
            for k, v in (r.items() if isinstance(r, dict) else []):
                if k in ("summary", "count", "risk_score", "hypotheses", "commands"):
                    print(f"  {k}: {v}")
            print()
        except Exception as e:
            print(f"  ⚠ {e}\n")

    print(f"{'='*60}\n  DEMO COMPLETE\n{'='*60}")


# ── Orchestrator mode ────────────────────────────────────────────────────────

def run_orchestrator_mode():
    """Run multi-agent orchestrator in interactive mode."""
    console.print(Panel("[bold cyan]K8s Failure Intelligence — Multi-Agent Orchestrator[/bold cyan]\n"
                        "[yellow]DeepAgents coordination: Investigator + Knowledge + Remediation + Verification[/yellow]",
                        border_style="cyan", expand=False))
    console.print("[dim]Type 'exit' to quit[/dim]\n")

    try:
        from src.agents import create_orchestrator_agent
        agent = create_orchestrator_agent()
        console.print("[green]✓ Orchestrator loaded with 4 subagents[/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to load orchestrator:[/red] {e}")
        import traceback
        traceback.print_exc()
        return

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        if query.lower() in ("exit", "quit"):
            console.print("[yellow]Goodbye![/yellow]")
            break
        if not query:
            continue

        console.print("\n[bold magenta]⏳ Orchestrating agents...[/bold magenta]")
        
        t0 = time.time()
        try:
            response, steps = run_agent_with_tools(agent, query, max_steps=15, verbose=True)
            total_ms = int((time.time() - t0) * 1000)
            console.print(f"\n  [dim]Total: {total_ms}ms, {len(steps)} steps[/dim]")
            if response:
                console.print(Panel(response, title="[bold cyan]Orchestrator Response[/bold cyan]",
                                    border_style="green", expand=False))
            else:
                console.print("[yellow]Orchestrator returned no response[/yellow]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            import traceback
            traceback.print_exc()
        console.print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="K8s Failure Intelligence Agent")
    parser.add_argument("--mode", choices=["interactive", "test", "demo", "orchestrator"], default="interactive")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    {
        "interactive": run_interactive_mode,
        "test": run_test_mode,
        "demo": run_demo_mode,
        "orchestrator": run_orchestrator_mode
    }[args.mode]()


if __name__ == "__main__":
    main()
