#!/usr/bin/env python3
"""Unified K8s Failure Intelligence Agent - Simplified, Reliable Version."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def _build_tool_registry():
    from src.tools import ALL_TOOLS
    registry = {}
    for tool in ALL_TOOLS:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
        if name:
            registry[name] = tool
    return registry

def _parse_tool_call(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "name" in payload and "arguments" in payload:
        return {"name": payload["name"], "arguments": payload["arguments"]}
    return None


def _run_agent_with_tools(agent, query, max_steps=4):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    registry = _build_tool_registry()
    messages = [HumanMessage(content=query)]
    last_ai = None

    for _ in range(max_steps):
        result = agent.invoke({"messages": messages})
        print(f"Result: {result}")
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
        elif hasattr(result, "messages"):
            messages = result['messages']
        else:
            return str(result), messages

        if not messages:
            return "", messages

        last = messages[-1]
        if isinstance(last, AIMessage):
            last_ai = last
            tool_calls = getattr(last, "tool_calls", None) or []
            if tool_calls:
                for call in tool_calls:
                    tool_name = call.get("name")
                    tool_args = call.get("args") or call.get("arguments") or {}
                    tool = registry.get(tool_name)
                    if not tool:
                        continue
                    tool_result = tool.invoke(tool_args)
                    messages.append(ToolMessage(
                        content=json.dumps(tool_result, indent=2),
                        name=tool_name,
                        tool_call_id=call.get("id", tool_name),
                    ))
                continue

            parsed = _parse_tool_call(last.content)
            if parsed:
                tool = registry.get(parsed["name"])
                if tool:
                    tool_result = tool.invoke(parsed["arguments"])
                    messages.append(ToolMessage(
                        content=json.dumps(tool_result, indent=2),
                        name=parsed["name"],
                        tool_call_id=parsed["name"],
                    ))
                    continue

            return last.content, messages

    return last_ai.content if last_ai else "", messages
    
def run_interactive_mode():
    """Interactive chat mode."""
    console.print(Panel("[bold cyan]K8s Failure Intelligence Agent[/bold cyan]\n[yellow]ReAct + RAG[/yellow]",
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
        
        console.print("\n[bold magenta]⏳ Agent invoking...[/bold magenta]\n")
        
        try:
            response_text, _ = _run_agent_with_tools(agent, query, max_steps=5)
            if response_text:
                console.print("\n[bold green]═════════════════════[/bold green]")
                console.print("[bold cyan]Response:[/bold cyan]\n")
                console.print(response_text)
                console.print("[bold green]═════════════════════[/bold green]\n")
            else:
                console.print("[yellow]Agent returned no response[/yellow]\n")
        
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}\n")
            import traceback
            traceback.print_exc()


def run_test_mode():
    """Simple test mode."""
    console.print(Panel("[bold cyan]K8s Agent Test[/bold cyan]\n[yellow]Testing tools[/yellow]",
                       border_style="cyan", expand=False))
    
    try:
        from src.tools import (
            generate_hypotheses, analyze_logs, generate_fix, verify_fix
        )
    except Exception as e:
        console.print(f"[red]Failed to load tools:[/red] {e}")
        return
    
    query = "pod crashing with OOMKilled errors"
    console.print(f"\n[bold cyan]Testing with:[/bold cyan] {query}\n")
    
    # Step 1: Hypotheses
    console.print("[yellow]Step 1: Generating hypotheses...[/yellow]")
    try:
        hyp = generate_hypotheses.invoke({"symptoms": query})
        console.print(f"✓ Got {len(hyp.get('hypotheses', []))} hypotheses")
        if hyp.get('hypotheses'):
            print(f"  • {hyp['hypotheses'][0]['cause']}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    
    # Step 2: Logs
    console.print("\n[yellow]Step 2: Analyzing logs...[/yellow]")
    try:
        logs = analyze_logs.invoke({"pod_name": "data-processor", "namespace": "default"})
        console.print(f"✓ {logs.get('summary', 'Analyzed logs')}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
    
    # Step 3: Fix
    console.print("\n[yellow]Step 3: Generating fix...[/yellow]")
    try:
        if hyp.get('hypotheses'):
            fix = generate_fix.invoke({"hypothesis": hyp['hypotheses'][0]['cause'], "manifest_yaml": None})
            console.print(f"✓ Fix risk score: {fix.get('risk_score', 0):.0%}")
            if fix.get('commands'):
                console.print(f"✓ {len(fix['commands'])} commands to execute")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
    
    # Step 4: Verify
    console.print("\n[yellow]Step 4: Verifying fix...[/yellow]")
    try:
        if hyp.get('hypotheses') and fix.get('commands'):
            ver = verify_fix.invoke({"fix_commands": fix['commands'], "cluster_health_check": "healthy"})
            console.print(f"✓ Likely effective: {ver.get('likely_effective', False)}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
    
    console.print("\n[green]✓ Test complete[/green]")


def run_demo_mode():
    """Demo mode - show workflow."""
    console.print(Panel("[bold cyan]K8s Agent Demo[/bold cyan]\n[yellow]Full Workflow[/yellow]",
                       border_style="cyan", expand=False))
    
    try:
        from src.tools import (
            retrieve_docs, generate_hypotheses, analyze_logs,
            generate_fix, verify_fix
        )
    except Exception as e:
        console.print(f"[red]Failed to load tools:[/red] {e}")
        return
    
    print("\n" + "="*60)
    print("  K8S FAILURE INTELLIGENCE - WORKFLOW DEMO")
    print("="*60)
    
    query = "why is my data-processor pod crashing with OOMKilled"
    print(f"\n👤 Query: {query}\n")
    
    # Step 1: RAG
    print("[STEP 1] RAG RETRIEVAL")
    print("-" * 40)
    try:
        rag = retrieve_docs.invoke({"query": "OOMKilled memory limit", "top_k": 3})
        print(f"✓ Retrieved {rag.get('count', 0)} documents\n")
    except Exception as e:
        print(f"⚠ RAG unavailable: {e}\n")
    
    # Step 2: Logs
    print("[STEP 2] ANALYZE SYMPTOMS")
    print("-" * 40)
    try:
        logs = analyze_logs.invoke({"pod_name": "data-processor", "namespace": "default"})
        print(f"✓ {logs.get('summary', 'Analyzed logs')}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Step 3: Hypotheses
    print("[STEP 3] ROOT CAUSE HYPOTHESES")
    print("-" * 40)
    try:
        hyp = generate_hypotheses.invoke({"symptoms": "Pod crashing, OOMKilled"})
        hyps = hyp.get('hypotheses', [])
        print(f"✓ Generated {len(hyps)} hypotheses")
        if hyps:
            print(f"  Top: {hyps[0]['cause']}")
            print(f"  Confidence: {hyps[0].get('confidence', 0):.0%}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Step 4: Fix
    print("[STEP 4] FIX GENERATION")
    print("-" * 40)
    try:
        if hyps:
            fix = generate_fix.invoke({"hypothesis": hyps[0]['cause'], "manifest_yaml": None})
            print(f"✓ Risk Score: {fix.get('risk_score', 0):.0%}")
            print(f"✓ Commands: {len(fix.get('commands', []))}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Step 5: Verify
    print("[STEP 5] VERIFICATION")
    print("-" * 40)
    try:
        if fix:
            ver = verify_fix.invoke({"fix_commands": fix.get('commands', []), "cluster_health_check": "healthy"})
            print(f"✓ Effective: {ver.get('likely_effective', False)}")
            print(f"✓ Missing steps: {len(ver.get('missing_steps', []))}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*60)
    print("  DEMO COMPLETE")
    print("="*60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="K8s Failure Intelligence Agent")
    parser.add_argument("--mode", choices=["interactive", "test", "demo"], default="interactive",
                       help="Operation mode")
    parser.add_argument("--verbose", action="store_true", help="Debug output")
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    if args.mode == "interactive":
        run_interactive_mode()
    elif args.mode == "test":
        run_test_mode()
    else:
        run_demo_mode()


if __name__ == "__main__":
    main()
