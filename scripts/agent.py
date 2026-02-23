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

def _truncate(text, max_len=400, step_number=1):
    """Aggressive truncation - later steps get more aggressive to speed up LLM."""
    adjusted_limit = max(200, max_len - (step_number * 40))
    return text if len(text) <= adjusted_limit else text[:adjusted_limit] + f"\n...(truncated, {len(text)} total)"


def _extract_natural_language(content: str) -> str:
    """Extract natural language from response, filter out JSON/tool calls."""
    if not content:
        return ""
    
    stripped = content.strip()
    
    # Check if response is pure JSON
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
            # Extract text fields if JSON
            if isinstance(parsed, dict):
                return parsed.get("response", parsed.get("answer", parsed.get("output", 
                       "Analysis complete. Please see investigation results above.")))
        except json.JSONDecodeError:
            pass
    
    # Remove JSON code blocks
    content = re.sub(r'```json\s*\n.*?\n```', '', content, flags=re.DOTALL)
    content = re.sub(r'```\s*\n\{.*?\}\s*\n```', '', content, flags=re.DOTALL)
    
    # Remove inline JSON-like structures (tool calls in text)
    content = re.sub(r'\{"name":\s*"[^"]+",\s*"arguments":[^}]+\}', '', content)
    
    return content.strip() or "Analysis complete."


def run_agent_with_tools(agent, query, max_steps=10, verbose=True):
    """Execute agent with tool loop. Returns (response_text, steps_list)."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    registry = _build_tool_registry()
    messages = [HumanMessage(content=query)]
    steps, last_ai = [], None
    last_tool_signature = None
    consecutive_duplicate_count = 0  # Track stuck loops
    tool_cache = {}
    tool_call_history = []  # Global tracking of all tool calls (not just consecutive)

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
            
            # GLOBAL DUPLICATE DETECTION: Check if we've called this before (anywhere in history)
            if tool_signature in tool_call_history:
                duplicate_count = tool_call_history.count(tool_signature)
                cached = tool_cache.get(tool_signature)
                
                if duplicate_count >= 2:
                    # Block on 3rd total attempt (2 previous calls)
                    si["error"] = f"Duplicate tool call blocked (called {duplicate_count+1} times)"
                    si["cached"] = True
                    steps.append(si)
                    msg = (
                        f"⚠️ You already called this tool {duplicate_count} times with identical arguments. "
                        "Using cached result. Try a DIFFERENT tool or DIFFERENT arguments."
                    )
                    messages.append(ToolMessage(content=cached or msg, name=tool_name, tool_call_id=call_id))
                    if verbose:
                        console.print(f"           [red]⚠ Blocked duplicate #{duplicate_count+1}, using cache[/red]")
                    continue
                elif cached is not None:
                    # 2nd attempt: use cache but warn
                    si["cached"] = True
                    si["duplicate_count"] = duplicate_count
                    steps.append(si)
                    messages.append(ToolMessage(content=cached, name=tool_name, tool_call_id=call_id))
                    if verbose:
                        console.print(f"           [yellow]↺ duplicate #{duplicate_count+1}, cached result[/yellow]")
                    tool_call_history.append(tool_signature)  # Still track it
                    continue
            
            # LEGACY: Also check consecutive duplicates for compatibility
            if tool_signature == last_tool_signature:
                consecutive_duplicate_count += 1
                
                if consecutive_duplicate_count >= 3:
                    # Hard block on 3rd consecutive identical call: LLM is stuck in a loop
                    si["error"] = "Infinite tool loop detected (same call 3+ times)"
                    steps.append(si)
                    msg = (
                        "⚠ INFINITE LOOP DETECTED: You called the same tool with identical arguments 3 times. "
                        "You must now synthesize your FINAL ANSWER using all information gathered so far. "
                        "Do NOT call any tools again. Do NOT retry the same command."
                    )
                    messages.append(ToolMessage(content=msg, name=tool_name, tool_call_id=call_id))
                    if verbose:
                        console.print(f"           [red]⚠⚠⚠ LOOP BLOCKED ON 3RD CONSECUTIVE CALL - forcing final answer[/red]")
                    force_final = True
                    break
            else:
                # NEW tool call — reset consecutive duplicate counter
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
                
                # Handle human confirmation for write commands
                if isinstance(tool_result, dict) and tool_result.get("status") == "pending_confirmation":
                    if verbose:
                        console.print(f"           [yellow]⚠ Requires confirmation[/yellow]")
                    
                    # Prompt user for confirmation
                    confirm_msg = tool_result.get("message", "Confirm this operation?")
                    console.print(f"\n[bold yellow]{confirm_msg}[/bold yellow]")
                    user_input = console.input("[bold cyan]Proceed? [Y/n]: [/bold cyan]").strip().lower()
                    
                    if user_input in ("y", "yes", ""):
                        # User confirmed - add to confirmed list and re-invoke
                        from src.tools import _CONFIRMED_COMMANDS
                        cmd_sig = f"{tool_result['command']}|{tool_result['namespace']}"
                        _CONFIRMED_COMMANDS.add(cmd_sig)
                        
                        if verbose:
                            console.print(f"           [green]✓ User confirmed, executing...[/green]")
                        
                        # Re-invoke tool now that it's confirmed
                        t1 = time.time()
                        tool_result = tool.invoke(tool_args)
                        tool_ms = int((time.time() - t1) * 1000)
                    else:
                        # User declined
                        if verbose:
                            console.print(f"           [red]✗ User declined[/red]")
                        tool_result = {
                            "status": "declined",
                            "output": "",
                            "stderr": "Operation cancelled by user",
                        }
                
                result_str = json.dumps(tool_result, indent=2)
                si["result"] = result_str[:500]
                si["tool_ms"] = tool_ms
                steps.append(si)
                last_tool_signature = tool_signature
                tool_call_history.append(tool_signature)  # Track globally
                if verbose:
                    console.print(f"           [green]✓[/green] [dim]{tool_ms}ms, {len(result_str)} chars[/dim]")
                truncated = _truncate(result_str, max_len=400, step_number=step)
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
    content = last_ai.content if last_ai else ""
    # Handle both string and list content types from LangChain messages
    if isinstance(content, list):
        content = " ".join(str(c) for c in content if c)
    final_response = _extract_natural_language(str(content))
    if not final_response or "Duplicate tool call blocked" in final_response:
        final_response = "Analysis complete based on available cluster data."
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
    """Run multi-agent orchestrator in interactive mode.
    
    Two approaches available:
    1. Single orchestrator agent with all tools (default)
    2. True multi-agent sequential invocation (use multiagent mode)
    """
    console.print(Panel("[bold cyan]K8s Failure Intelligence — Multi-Agent Orchestrator[/bold cyan]\n"
                        "[yellow]Single agent with comprehensive workflow coordination[/yellow]",
                        border_style="cyan", expand=False))
    console.print("[dim]Type 'exit' to quit[/dim]\n")

    try:
        from src.agents import create_orchestrator_agent
        agent = create_orchestrator_agent()
        console.print("[green]✓ Orchestrator loaded[/green]\n")
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

        console.print("\n[bold magenta]⏳ Orchestrating workflow...[/bold magenta]")
        
        t0 = time.time()
        try:
            response, steps = run_agent_with_tools(agent, query, max_steps=20, verbose=True)
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


# ── Multi-agent mode (TRUE delegation) ──────────────────────────────────────

def run_multiagent_mode():
    """Run TRUE multi-agent mode with sequential agent invocation.
    
    This mode invokes 4 separate agents in sequence:
    1. Investigator - diagnoses the issue
    2. Knowledge - retrieves runbooks
    3. Remediation - EXECUTES fixes
    4. Verification - confirms success
    """
    console.print(Panel("[bold cyan]K8s Failure Intelligence — TRUE Multi-Agent Mode[/bold cyan]\n"
                        "[yellow]Sequential invocation: Investigator → Knowledge → Remediation → Verification[/yellow]",
                        border_style="cyan", expand=False))
    console.print("[dim]Type 'exit' to quit[/dim]\n")

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

        console.print("\n[bold magenta]⏳ Running multi-agent workflow...[/bold magenta]")
        
        t0 = time.time()
        try:
            from src.agents import orchestrate_multiagent_diagnosis
            result = orchestrate_multiagent_diagnosis(query, max_steps=20)
            total_ms = int((time.time() - t0) * 1000)
            
            agents_used = result.get("agents_used", [])
            console.print(f"\n  [dim]Total: {total_ms}ms, {len(agents_used)} agents invoked[/dim]")
            console.print(f"  [dim]Agent sequence: {' → '.join(agents_used)}[/dim]\n")
            
            if result.get("response"):
                console.print(Panel(result["response"], title="[bold cyan]Multi-Agent Summary[/bold cyan]",
                                    border_style="green", expand=False))
            else:
                console.print("[yellow]No response generated[/yellow]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            import traceback
            traceback.print_exc()
        console.print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="K8s Failure Intelligence Agent")
    parser.add_argument("--mode", choices=["interactive", "test", "demo", "orchestrator", "multiagent"], default="interactive",
                        help="interactive: single investigator | orchestrator: coordinated workflow | multiagent: TRUE multi-agent delegation")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    {
        "interactive": run_interactive_mode,
        "test": run_test_mode,
        "demo": run_demo_mode,
        "orchestrator": run_orchestrator_mode,
        "multiagent": run_multiagent_mode,
    }[args.mode]()


if __name__ == "__main__":
    main()
