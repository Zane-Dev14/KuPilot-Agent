#!/usr/bin/env python3
"""Unified K8s Failure Intelligence Agent — with step tracing & speed fixes."""

import json, re, sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel

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
        for tool_name, tool_args, call_id in normalized:
            si = {"step": step, "type": "tool_call", "name": tool_name,
                  "args": tool_args, "llm_ms": llm_ms}
            if verbose:
                args_preview = json.dumps(tool_args)[:80]
                console.print(f"  [dim]Step {step}[/dim] [cyan]→ {tool_name}[/cyan]({args_preview}) [dim]{llm_ms}ms[/dim]")

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
                if verbose:
                    console.print(f"           [green]✓[/green] [dim]{tool_ms}ms, {len(result_str)} chars[/dim]")
                messages.append(ToolMessage(content=_truncate(result_str), name=tool_name, tool_call_id=call_id))
            except Exception as e:
                si["error"] = str(e)
                steps.append(si)
                if verbose:
                    console.print(f"           [red]✗ {e}[/red]")
                messages.append(ToolMessage(content=f"Error: {e}", name=tool_name, tool_call_id=call_id))

    return (last_ai.content if last_ai else ""), steps


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
            import traceback; traceback.print_exc()
        console.print()


# ── Test mode ────────────────────────────────────────────────────────────────

def run_test_mode():
    console.print(Panel("[bold cyan]K8s Agent Test[/bold cyan]\n[yellow]Testing tools directly[/yellow]",
                        border_style="cyan", expand=False))
    try:
        from src.tools import generate_hypotheses, analyze_logs, generate_fix, verify_fix
    except Exception as e:
        console.print(f"[red]Failed to load tools:[/red] {e}"); return

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
        from src.tools import retrieve_docs, generate_hypotheses, analyze_logs, generate_fix, verify_fix
    except Exception as e:
        console.print(f"[red]Failed to load tools:[/red] {e}"); return

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="K8s Failure Intelligence Agent")
    parser.add_argument("--mode", choices=["interactive", "test", "demo"], default="interactive")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        import logging; logging.basicConfig(level=logging.DEBUG)

    {"interactive": run_interactive_mode, "test": run_test_mode, "demo": run_demo_mode}[args.mode]()


if __name__ == "__main__":
    main()
