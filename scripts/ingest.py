#!/usr/bin/env python3
"""Ingest sample data into the K8s Failure Intelligence knowledge base.

Usage:
    python scripts/ingest.py                          # ingest all sample data
    python scripts/ingest.py --path data/sample/docs  # ingest specific dir
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from src.config import get_settings
from src.ingestion import ingest_directory, ingest_file
from src.vectorstore import MilvusStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
console = Console()

SAMPLE_DIR = ROOT / "data" / "sample"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the knowledge base")
    parser.add_argument("--path", type=str, default=str(SAMPLE_DIR),
                        help="File or directory to ingest (default: data/sample/)")
    parser.add_argument("--no-drop", action="store_true",
                        help="Append to collection instead of wiping it (default: wipe for clean schema)")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        console.print(f"[red]Path not found:[/red] {target}")
        sys.exit(1)

    console.print(f"\n[bold]Ingesting:[/bold] {target}\n")

    # Load documents
    if target.is_dir():
        docs = ingest_directory(target)
    else:
        docs = ingest_file(target)

    if not docs:
        console.print("[yellow]No documents found.[/yellow]")
        return

    # Store in Milvus
    drop_old = not args.no_drop
    store = MilvusStore(drop_old=drop_old)
    if args.no_drop:
        console.print("[yellow]📎 Appending to existing collection[/yellow]")
    else:
        console.print("[yellow]🗑️  Wiping collection for clean schema[/yellow]")
    ids = store.add_documents(docs)

    # Summary table
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Chunks created", str(len(docs)))
    table.add_row("Chunks stored", str(len(ids)))

    # Count by doc_type
    types: dict[str, int] = {}
    for d in docs:
        t = d.metadata.get("doc_type", "unknown")
        types[t] = types.get(t, 0) + 1
    for t, n in sorted(types.items()):
        table.add_row(f"  └ {t}", str(n))

    console.print(table)
    console.print("\n[green]✓ Ingestion complete![/green]\n")


if __name__ == "__main__":
    main()
