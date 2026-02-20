"""Unified document ingestion — auto-detects file type, chunks, adds metadata."""

import json, logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import get_settings

logger = logging.getLogger(__name__)

_CHAR_SPLITTER = None


def _splitter():
    global _CHAR_SPLITTER
    if _CHAR_SPLITTER is None:
        s = get_settings()
        _CHAR_SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    return _CHAR_SPLITTER


def _stamp(docs):
    """Normalize metadata keys (Milvus requires uniform schemas)."""
    now = datetime.now(timezone.utc).isoformat()
    canonical = ("source", "doc_type", "kind", "name", "namespace", "reason")
    allowed = set(canonical) | {"ingested_at", "chunk_index"}
    for i, d in enumerate(docs):
        d.metadata.setdefault("ingested_at", now)
        d.metadata["chunk_index"] = i
        for key in canonical:
            d.metadata.setdefault(key, "")
        d.metadata = {k: v for k, v in d.metadata.items() if k in allowed}
    return docs


def _flat(d, indent=0):
    prefix = " " * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.extend(_flat(v, indent + 2))
        elif isinstance(v, list):
            lines.append(f"{prefix}{k}: [{len(v)} items]")
        else:
            lines.append(f"{prefix}{k}: {v}")
    return lines


def _load_yaml(path):
    try:
        objs = list(yaml.safe_load_all(path.read_text()))
    except yaml.YAMLError as e:
        logger.error("YAML error in %s: %s", path, e)
        return []
    docs = []
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        md = obj.get("metadata", {})
        meta = {"source": str(path), "doc_type": "kubernetes_manifest",
                "kind": obj.get("kind", "Unknown"),
                "name": md.get("name", "unknown"), "namespace": md.get("namespace", "default")}
        lines = [f"Kind: {obj.get('kind', 'Unknown')}", f"Name: {md.get('name', 'unknown')}",
                 f"Namespace: {md.get('namespace', 'default')}", f"API Version: {obj.get('apiVersion', 'v1')}"]
        for sec in ("spec", "status"):
            if sec in obj:
                lines.append(f"{sec.title()}:")
                lines.extend(_flat(obj[sec], indent=2))
        docs.append(Document(page_content="\n".join(lines), metadata=meta))
    return docs


def _load_events(path):
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        logger.error("JSON error in %s: %s", path, e)
        return []
    events = raw if isinstance(raw, list) else [raw]
    docs = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        inv = ev.get("involvedObject", {})
        meta = {"source": str(path), "doc_type": "kubernetes_event",
                "reason": ev.get("reason", "Unknown"),
                "namespace": inv.get("namespace", "default"), "name": inv.get("name", "unknown")}
        content = "\n".join([
            f"Event: {ev.get('reason', 'Unknown')}", f"Type: {ev.get('type', 'Normal')}",
            f"Object: {inv.get('kind', '?')}/{inv.get('name', '?')} (ns: {inv.get('namespace', 'default')})",
            f"Count: {ev.get('count', 1)}", f"Message: {ev.get('message', '')}"])
        docs.append(Document(page_content=content, metadata=meta))
    return docs


_MD_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])


def _load_markdown(path):
    splits = _MD_SPLITTER.split_text(path.read_text())
    for s in splits:
        s.metadata["source"] = str(path)
        s.metadata["doc_type"] = "markdown_document"
    return splits


def _load_log(path):
    meta = {"source": str(path), "doc_type": "kubernetes_log"}
    return [Document(page_content=c, metadata={**meta})
            for c in _splitter().split_text(path.read_text())]


_LOADERS = {".yaml": _load_yaml, ".yml": _load_yaml, ".json": _load_events,
            ".md": _load_markdown, ".log": _load_log, ".txt": _load_log}


def ingest_file(path: Path) -> list[Document]:
    """Load + chunk a single file. Returns list of Document chunks."""
    loader = _LOADERS.get(path.suffix.lower())
    if not loader:
        logger.warning("Unsupported file type: %s", path)
        return []
    docs = loader(path)
    if path.suffix.lower() in (".yaml", ".yml", ".json"):
        out = []
        for d in docs:
            if len(d.page_content) > get_settings().chunk_size:
                out.extend(_splitter().split_documents([d]))
            else:
                out.append(d)
        docs = out
    return _stamp(docs)


def ingest_directory(directory: Path) -> list[Document]:
    """Recursively ingest all supported files under directory."""
    all_docs = []
    for ext in _LOADERS:
        for fp in sorted(directory.rglob(f"*{ext}")):
            logger.info("Ingesting %s", fp)
            all_docs.extend(ingest_file(fp))
    return all_docs  # ingest_file already stamps — no double-stamp
