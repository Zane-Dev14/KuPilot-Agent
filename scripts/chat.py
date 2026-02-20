#!/usr/bin/env python3
"""Interactive CLI chat — multi-turn K8s failure diagnosis with rich output.

Usage:
    python scripts/chat.py
    python scripts/chat.py --force-model llama3.1

Commands inside the chat:
    exit   — quit
    clear  — reset conversation memory
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
# Suppress noisy async Milvus warning (we only use sync)
logging.getLogger("langchain_milvus.vectorstores.milvus").setLevel(logging.ERROR)

console = Console()


def _confidence_colour(c: float) -> str:
    if c >= 0.8:
        return "green"
    if c >= 0.5:
        return "yellow"
    return "red"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="K8s Failure Intelligence — interactive chat")
    parser.add_argument("--force-model", type=str, default=None, help="Override model selection")
    parser.add_argument("--session", type=str, default="cli", help="Session ID for memory")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]Kubernetes Failure Intelligence Copilot[/bold cyan]\n"
        "Ask about pod crashes, OOMKills, scheduling failures, etc.\n"
        "Type [bold]exit[/bold] to quit  ·  [bold]clear[/bold] to reset memory",
        border_style="cyan",
    ))

    # Lazy imports so startup banner shows immediately
    from src.rag_chain import RAGChain, estimate_complexity

    console.print("\n[dim]Loading models … (first run downloads embeddings + reranker)[/dim]")
    chain = RAGChain()
    console.print("[green]Ready![/green]\n")

    while True:
        try:
            query = console.input("[bold cyan]You>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() == "exit":
            console.print("[dim]Goodbye![/dim]")
            break
        if query.lower() == "clear":
            chain.memory.clear(args.session)
            console.print("[yellow]Memory cleared.[/yellow]\n")
            continue

        # Show complexity score
        complexity = estimate_complexity(query)
        console.print(f"[dim]Complexity: {complexity:.2f}[/dim]")

        with console.status("[bold green]Thinking …[/bold green]"):
            try:
                dx = chain.diagnose(query, session_id=args.session, force_model=args.force_model)
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]\n")
                continue

        # Build output panel
        cc = _confidence_colour(dx.confidence)
        body = (
            f"[bold]Root Cause:[/bold]  {dx.root_cause}\n\n"
            f"[bold]Explanation:[/bold] {dx.explanation}\n\n"
            f"[bold]Fix:[/bold]         {dx.recommended_fix}\n\n"
            f"[bold]Confidence:[/bold]  [{cc}]{dx.confidence:.0%}[/{cc}]\n"
            f"[bold]Model:[/bold]       {dx.model_used}\n"
            f"[bold]Sources:[/bold]     {', '.join(set(dx.sources)) or '(none)'}"
        )
        console.print(Panel(body, title="[bold]Diagnosis[/bold]", border_style="green", expand=False))
        console.print()


if __name__ == "__main__":
    main()
