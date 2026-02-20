"""Tests that run without Milvus, Ollama, or downloaded models.

Covers: config defaults, memory, model selector, ingestion parsing.
Run:  pytest tests/test_basic.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest


# ─── Config ──────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        from src.config import Settings
        s = Settings()  # no .env needed — uses defaults
        assert s.milvus_uri == "http://localhost:19530"
        assert s.milvus_collection == "k8s_failures"
        assert s.embedding_dimension == 384
        assert s.retrieval_top_k == 4
        assert 0.0 < s.query_complexity_threshold < 1.0

    def test_singleton(self):
        from src.config import get_settings
        a = get_settings()
        b = get_settings()
        assert a is b


# ─── Memory ──────────────────────────────────────────────────────────────────

class TestMemory:
    def _new_mem(self, **kw):
        from src.memory import ChatMemory
        return ChatMemory(**kw)

    def test_add_and_retrieve(self):
        mem = self._new_mem()
        mem.add_user_message("s1", "hello")
        mem.add_ai_message("s1", "hi")
        history = mem.get_history("s1")
        assert len(history) == 2
        assert history[0].content == "hello"
        assert history[1].content == "hi"

    def test_empty_session(self):
        mem = self._new_mem()
        assert mem.get_history("nonexistent") == []

    def test_trim(self):
        mem = self._new_mem(max_messages=4)
        for i in range(6):
            mem.add_user_message("s1", f"msg-{i}")
        assert len(mem.get_history("s1")) == 4

    def test_lru_eviction(self):
        mem = self._new_mem(max_sessions=2)
        mem.add_user_message("a", "x")
        mem.add_user_message("b", "x")
        mem.add_user_message("c", "x")  # should evict "a"
        assert mem.get_history("a") == []
        assert mem.active_sessions == 2

    def test_clear(self):
        mem = self._new_mem()
        mem.add_user_message("s1", "x")
        mem.clear("s1")
        assert mem.get_history("s1") == []
        assert mem.active_sessions == 0

    def test_clear_all(self):
        mem = self._new_mem()
        mem.add_user_message("a", "x")
        mem.add_user_message("b", "x")
        mem.clear_all()
        assert mem.active_sessions == 0


# ─── Model selector ─────────────────────────────────────────────────────────

class TestModelSelector:
    def test_simple_query_low_complexity(self):
        from src.rag_chain import estimate_complexity
        score = estimate_complexity("list pods")
        assert score < 0.2, f"Expected < 0.2, got {score}"

    def test_definition_query_low(self):
        from src.rag_chain import estimate_complexity
        score = estimate_complexity("What is CrashLoopBackOff?")
        assert score < 0.5, f"Expected < 0.5, got {score}"

    def test_reasoning_query(self):
        from src.rag_chain import estimate_complexity
        score = estimate_complexity("Why is my pod in CrashLoopBackOff?")
        assert score >= 0.3, f"Expected >= 0.3, got {score}"

    def test_complex_multi_question(self):
        from src.rag_chain import estimate_complexity
        score = estimate_complexity("Why is it crashing? How do I fix it? What caused the OOM?")
        assert score >= 0.5, f"Expected >= 0.5, got {score}"

    def test_compound_query_high(self):
        from src.rag_chain import estimate_complexity
        score = estimate_complexity(
            "Why is my pod crashing with CrashLoopBackOff AND how do I troubleshoot it?"
        )
        assert score >= 0.5, f"Expected >= 0.5, got {score}"

    def test_multi_signal_reaches_threshold(self):
        from src.rag_chain import estimate_complexity
        score = estimate_complexity(
            "Why is my pod OOMKilled even though memory limit is 512Mi? "
            "How do I diagnose and fix this intermittent issue?"
        )
        assert score >= 0.7, f"Expected >= 0.7, got {score}"

    def test_word_boundary_no_false_positive(self):
        from src.rag_chain import estimate_complexity
        # 'show' should NOT match 'how', 'shower' should NOT match 'how'
        score = estimate_complexity("show me the pods")
        assert score < 0.2, f"Expected < 0.2, got {score}"

    def test_force_model_override(self):
        from src.rag_chain import select_model
        result = select_model("hi", force="my-custom-model")
        assert result == "my-custom-model"


# ─── JSON parser ─────────────────────────────────────────────────────────────

class TestParseJson:
    def test_direct_json(self):
        from src.rag_chain import _parse_json
        raw = '{"root_cause": "OOM", "confidence": 0.9}'
        assert _parse_json(raw)["root_cause"] == "OOM"

    def test_fenced_code_block(self):
        from src.rag_chain import _parse_json
        raw = 'Some preamble\n```json\n{"root_cause": "X"}\n```\nAftermath'
        assert _parse_json(raw)["root_cause"] == "X"

    def test_embedded_braces(self):
        from src.rag_chain import _parse_json
        raw = 'Here is the answer: {"root_cause": "Y"}'
        assert _parse_json(raw)["root_cause"] == "Y"

    def test_garbage_returns_empty(self):
        from src.rag_chain import _parse_json
        assert _parse_json("no json here!!!") == {}


# ─── Ingestion ───────────────────────────────────────────────────────────────

class TestIngestion:
    def test_load_yaml(self):
        from src.ingestion import ingest_file

        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "test-app", "namespace": "dev"},
            "spec": {"replicas": 2},
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            import yaml
            yaml.dump(manifest, f)
            p = Path(f.name)
        docs = ingest_file(p)
        assert len(docs) >= 1
        assert docs[0].metadata["kind"] == "Deployment"
        assert "test-app" in docs[0].page_content
        p.unlink()

    def test_load_json_events(self):
        from src.ingestion import ingest_file

        events = [
            {
                "reason": "OOMKilled",
                "type": "Warning",
                "message": "Container killed",
                "involvedObject": {"kind": "Pod", "name": "web", "namespace": "dev"},
                "count": 3,
            }
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(events, f)
            p = Path(f.name)
        docs = ingest_file(p)
        assert len(docs) >= 1
        assert "OOMKilled" in docs[0].page_content
        p.unlink()

    def test_load_markdown(self):
        from src.ingestion import ingest_file

        md = "# Title\nParagraph.\n## Section\nContent here."
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(md)
            p = Path(f.name)
        docs = ingest_file(p)
        assert len(docs) >= 1
        p.unlink()

    def test_load_log(self):
        from src.ingestion import ingest_file

        log = "2024-01-01T00:00:00Z ERROR something went wrong\n" * 10
        with tempfile.NamedTemporaryFile(suffix=".log", mode="w", delete=False) as f:
            f.write(log)
            p = Path(f.name)
        docs = ingest_file(p)
        assert len(docs) >= 1
        assert "ERROR" in docs[0].page_content
        p.unlink()

    def test_unsupported_extension(self):
        from src.ingestion import ingest_file
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            p = Path(f.name)
        docs = ingest_file(p)
        assert docs == []
        p.unlink()


# ─── Query classifier ────────────────────────────────────────────────────────

class TestQueryClassifier:
    def _cls(self, q):
        from src.rag_chain import _classify_query
        return _classify_query(q)

    def test_have_i_asked_you_about(self):
        assert self._cls("Have i asked you about OOMKilled today? where?") == "conversational"

    def test_have_i_asked_about(self):
        assert self._cls("Have i asked about CrashLoopBackOff?") == "conversational"

    def test_did_i_ask_you_about(self):
        assert self._cls("Did i ask you about scheduling failures?") == "conversational"

    def test_did_i_ask_about(self):
        assert self._cls("Did i ask about imagepull?") == "conversational"

    def test_first_question(self):
        assert self._cls("What was my first question?") == "conversational"

    def test_last_question(self):
        assert self._cls("What was my last question?") == "conversational"

    def test_summary(self):
        assert self._cls("Can you give me a summary?") == "conversational"

    def test_kubectl_operational(self):
        assert self._cls("kubectl get pods -n kube-system") == "operational"

    def test_k8s_diagnostic(self):
        assert self._cls("Why is my pod OOMKilled?") == "diagnostic"

    def test_out_of_scope(self):
        assert self._cls("What is the weather today?") == "out_of_scope"


# ─── Conversational handler (no I/O) ─────────────────────────────────────────

class TestAnswerConversational:
    """Tests _answer_conversational() with hand-crafted message history.

    No Milvus or Ollama calls — KB fallback is only triggered when no session
    mentions exist, and since we mock history with mentions, that path is not
    exercised here (it requires Milvus to be up).
    """

    def _make_history(self, pairs):
        """Build a list of BaseMessage from [(role, content)] pairs."""
        from langchain_core.messages import HumanMessage, AIMessage
        msgs = []
        for role, content in pairs:
            msgs.append(HumanMessage(content=content) if role == "human" else AIMessage(content=content))
        return msgs

    def _call(self, query, history):
        from src.rag_chain import _answer_conversational
        return _answer_conversational(query, history)

    def test_returns_5_tuple(self):
        result = self._call("Have i asked you about OOMKilled?", [])
        assert isinstance(result, tuple) and len(result) == 5

    def test_have_i_asked_with_mention_in_history(self):
        history = self._make_history([
            ("human", "Why is my pod OOMKilled?"),
            ("ai", "The pod exceeded its memory limit and was OOMKilled."),
        ])
        root, explanation, conf, sources, evidence = self._call(
            "Have i asked you about OOMKilled today? where?", history
        )
        assert "yes" in root.lower() or "oomkilled" in root.lower()
        assert conf >= 0.8
        # evidence snippets should reference messages from history
        assert len(evidence) > 0

    def test_have_i_asked_no_mention_returns_gracefully(self):
        """When topic was not mentioned and KB is unavailable, should return valid tuple."""
        history = self._make_history([
            ("human", "What is a Pod?"),
        ])
        # MilvusStore will fail here (no server); the function must not raise
        root, explanation, conf, sources, evidence = self._call(
            "Have i asked you about CrashLoopBackOff?", history
        )
        assert isinstance(root, str) and len(root) > 0
        assert isinstance(conf, float)

    def test_first_question(self):
        history = self._make_history([
            ("human", "What is OOMKilled?"),
            ("ai", "It means the container was out-of-memory killed."),
        ])
        root, _, conf, _, _ = self._call("What was my first question?", history)
        assert "What is OOMKilled?" in root
        assert conf >= 0.8

    def test_last_question(self):
        history = self._make_history([
            ("human", "First question"),
            ("human", "Second question"),
        ])
        root, _, conf, _, _ = self._call("What was my last question?", history)
        assert "Second question" in root

    def test_empty_history_returns_valid(self):
        root, _, conf, _, _ = self._call("Have i asked you about OOMKilled?", [])
        # No session history + Milvus down → graceful fallback
        assert isinstance(root, str) and len(root) > 0


# ─── Find mentions ────────────────────────────────────────────────────────────

class TestFindMentions:
    def _make_history(self, pairs):
        from langchain_core.messages import HumanMessage, AIMessage
        return [
            HumanMessage(content=c) if r == "human" else AIMessage(content=c)
            for r, c in pairs
        ]

    def test_finds_topic_in_user_message(self):
        from src.rag_chain import _find_mentions
        history = self._make_history([
            ("human", "Why is my pod OOMKilled?"),
            ("ai", "Memory limit exceeded."),  # does NOT contain oomkilled
        ])
        mentions = _find_mentions(history, "oomkilled")
        assert len(mentions) == 1  # only the human message matches

    def test_case_insensitive(self):
        from src.rag_chain import _find_mentions
        history = self._make_history([("human", "OOMKilled pod issue")])
        assert len(_find_mentions(history, "oomkilled")) == 1

    def test_no_match_returns_empty(self):
        from src.rag_chain import _find_mentions
        history = self._make_history([("human", "pod scheduling failure")])
        assert _find_mentions(history, "oomkilled") == []

    def test_none_history_via_conversational(self):
        """_answer_conversational with None history should not raise."""
        from src.rag_chain import _answer_conversational
        result = _answer_conversational("Have i asked about oomkilled?", None)
        assert len(result) == 5
