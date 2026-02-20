"""Per-session conversation memory with LRU eviction and optional disk persistence."""

import json, logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import get_settings

logger = logging.getLogger(__name__)


class ChatMemory:
    """In-memory per-session history with LRU eviction."""

    def __init__(self, max_messages=20, max_sessions=256):
        self.max_messages, self.max_sessions = max_messages, max_sessions
        self._store: OrderedDict[str, list[BaseMessage]] = OrderedDict()

    def add_user_message(self, session_id, content):
        self._append(session_id, HumanMessage(content=content))

    def add_ai_message(self, session_id, content):
        self._append(session_id, AIMessage(content=content))

    def get_history(self, session_id) -> list[BaseMessage]:
        return list(self._store.get(session_id, []))

    def clear(self, session_id):
        self._store.pop(session_id, None)
        self._persist()

    def clear_all(self):
        self._store.clear()
        self._persist()

    @property
    def active_sessions(self) -> int:
        return len(self._store)

    def _append(self, sid, msg):
        if sid not in self._store:
            if len(self._store) >= self.max_sessions:
                self._store.popitem(last=False)
            self._store[sid] = []
        else:
            self._store.move_to_end(sid)
        self._store[sid].append(msg)
        if len(self._store[sid]) > self.max_messages:
            self._store[sid] = self._store[sid][-self.max_messages:]
        self._persist()

    def _persist(self):
        """Override in subclass for disk persistence."""


class DiskChatMemory(ChatMemory):
    """Extends ChatMemory with JSON file persistence (atomic writes)."""

    def __init__(self, path: Path, **kw):
        super().__init__(**kw)
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _persist(self):
        data = {sid: [{"type": m.type,
                        "content": m.content if isinstance(m.content, str) else str(m.content)}
                       for m in msgs]
                for sid, msgs in self._store.items()}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self.path)
        except Exception as exc:
            logger.warning("Failed to persist memory: %s", exc)

    def _load(self):
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        for sid, msgs in raw.items():
            if not isinstance(msgs, list):
                continue
            restored = []
            for item in msgs:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", "")
                if item.get("type") == "human":
                    restored.append(HumanMessage(content=content))
                elif item.get("type") == "ai":
                    restored.append(AIMessage(content=content))
            if restored:
                self._store[sid] = restored[-self.max_messages:]


# ── Module singleton ──────────────────────────────────────────────────────────

_memory: Optional[ChatMemory] = None


def get_chat_memory() -> ChatMemory:
    global _memory
    if _memory is None:
        s = get_settings()
        base = Path(__file__).resolve().parents[1]
        path = Path(s.memory_path)
        if not path.is_absolute():
            path = base / path
        _memory = DiskChatMemory(path=path, max_messages=s.memory_max_messages,
                                 max_sessions=s.memory_max_sessions)
    return _memory
