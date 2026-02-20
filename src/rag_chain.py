"""RAG diagnosis chain — retrieval + adaptive model selection + generation."""

import json, logging, os, re
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.config import get_settings
from src.memory import get_chat_memory
from src.vectorstore import MilvusStore

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

class FailureDiagnosis(BaseModel):
    root_cause: str = Field(default="Unknown")
    explanation: str = Field(default="")
    recommended_fix: str = Field(default="")
    confidence: float = Field(default=0.5)
    sources: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    response_type: str = Field(default="diagnostic")
    model_used: Optional[str] = None


# ── Keyword lists ─────────────────────────────────────────────────────────────

_REASONING_KW = ["why", "explain", "diagnose", "troubleshoot", "root cause",
                 "how", "analyze", "investigate", "debug", "cause"]
_UNCERTAIN_KW = ["maybe", "could", "might", "possibly", "not sure",
                 "intermittent", "sometimes", "random"]
_COMPOUND_KW  = ["and", "but", "also", "plus", "additionally", "as well",
                 "however", "although", "even though", "while"]
_TECHNICAL_KW = ["oomkilled", "oom", "crashloopbackoff", "crashloop",
                 "imagepullbackoff", "imagepull", "failedscheduling",
                 "evicted", "cordon", "taint", "affinity", "pdb",
                 "hpa", "vpa", "resource quota", "limitrange",
                 "network policy", "init container", "sidecar",
                 "liveness", "readiness", "startup probe",
                 "persistent volume", "configmap", "secret"]
_K8S_TERMS = ["k8s", "kubernetes", "pod", "pods", "deployment", "replicaset",
              "statefulset", "daemonset", "namespace", "node", "cluster",
              "crashloop", "oomkilled", "imagepull", "ingress", "service",
              "configmap", "secret", "pvc", "pv", "kube-system", "k3d"]
_CONVERSATION_PATTERNS = [
    r"\bfirst (question|message)\b", r"\bwhat did i ask\b",
    r"\bwhat was my (first|last|previous) question\b",
    r"\bhave i asked\b", r"\bdid i ask\b",
    r"\bsummary\b", r"\bsummarise\b", r"\brecap\b", r"\bremember\b",
]
_OPERATIONAL_PATTERNS = [
    r"\b(list|show|get)\s+pods\b", r"\bpods\s+in\s+k3d\b",
    r"\bk3d\b.*\bpods\b", r"\bpods\b.*\bk3d\b", r"\bkubectl\b",
]


# ── Small helpers ─────────────────────────────────────────────────────────────

def _word_in(word, text):
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text))

def _term_in(term, text):
    return bool(re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text))

def _fuzzy_term_in(term, text):
    for tok in re.findall(r"[a-z0-9-]+", text):
        if len(tok) >= 6 and (term.startswith(tok) or tok.startswith(term)):
            return True
    return False

def _clean_source(s):
    idx = s.find("data/")
    return s[idx:] if idx != -1 else s

def _build_evidence(docs, n=3):
    return [f"{_clean_source(d.metadata.get('source', '?'))}: "
            f"{d.page_content.strip().replace(chr(10), ' ')[:220]}"
            for d in docs[:n]]


# ── Complexity & model selection ──────────────────────────────────────────────

def estimate_complexity(query: str) -> float:
    q = query.lower()
    score = min(sum(1 for kw in _REASONING_KW if _word_in(kw, q)) * 0.20, 0.50)
    if any(_word_in(kw, q) for kw in _UNCERTAIN_KW):
        score += 0.15
    qmarks = q.count("?")
    if qmarks > 1:
        score += min(qmarks * 0.10, 0.30)
    compound = sum(1 for kw in _COMPOUND_KW if _word_in(kw, q))
    if compound:
        score += min(compound * 0.10, 0.20)
    score += min(sum(1 for kw in _TECHNICAL_KW if kw in q) * 0.10, 0.30)
    words = len(query.split())
    if words > 15:
        score += 0.10
    if words > 30:
        score += 0.10
    return round(min(score, 1.0), 2)


def select_model(query: str, force: str | None = None) -> str:
    if force:
        return force
    s = get_settings()
    c = estimate_complexity(query)
    model = s.complex_model if c >= s.query_complexity_threshold else s.simple_model
    logger.info("Complexity %.2f -> %s", c, model)
    return model


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a Kubernetes failure diagnosis expert integrated with a RAG knowledge base.

RULES:
1. Base your answers ONLY on the CONTEXT documents and CHAT HISTORY provided.
   Do NOT invent pod names, namespaces, or details that are not in the context.
2. If the user asks a conversational question (e.g. "what did I ask before?",
   "summarise our chat"), answer naturally using CHAT HISTORY.
   Still use JSON format, put your answer in "root_cause",
   set "recommended_fix" to "N/A", and do NOT cite sources.
3. If the question is outside Kubernetes operations, reply helpfully
   but set confidence to 0.0 and recommended_fix to "N/A".
4. When diagnosing failures, cite specific evidence from the context.
5. For the fix, give concrete steps or commands.

Always respond with ONLY this JSON (no markdown fences, no extra text):
{{
  "root_cause": "<concise root cause or direct answer>",
  "explanation": "<detailed explanation citing evidence from context>",
  "recommended_fix": "<specific actionable steps>",
  "confidence": 0.0-1.0
}}"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "CONTEXT (retrieved documents):\n{context}\n\n"
              "CHAT HISTORY (recent conversation):\n{history}\n\n"
              "USER QUERY:\n{query}"),
])

_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a router for a Kubernetes diagnosis assistant.\n"
               "Decide: diagnostic | conversational | operational | out_of_scope.\n"
               "Use chat history to interpret short follow-ups.\n"
               'Return ONLY JSON: {"response_type": "..."}'),
    ("human", "CHAT HISTORY:\n{history}\n\nUSER QUERY:\n{query}"),
])


# ── Formatting ────────────────────────────────────────────────────────────────

def _format_docs(docs):
    if not docs:
        return "(no documents found)"
    return "\n---\n".join(
        f"[{i}] {d.metadata.get('kind') or d.metadata.get('doc_type', '')} — "
        f"{d.metadata.get('source', '?')}\n{d.page_content[:1200]}"
        for i, d in enumerate(docs, 1))

def _format_history(messages):
    if not messages:
        return "(no previous conversation)"
    return "\n".join(f"{'User' if m.type == 'human' else 'AI'}: {m.content[:500]}"
                     for m in messages[-10:])

def _format_memory_text(dx):
    parts = [f"Root Cause: {dx.root_cause}"]
    if dx.explanation:
        parts.append(f"Explanation: {dx.explanation}")
    if dx.recommended_fix and dx.recommended_fix != "N/A":
        parts.append(f"Recommended Fix: {dx.recommended_fix}")
    return "\n".join(parts)


# ── Classification ────────────────────────────────────────────────────────────

def _classify_query(query, history=None):
    history = history or []
    q = query.lower().strip()

    # LLM classification (skip during pytest)
    if "PYTEST_CURRENT_TEST" not in os.environ:
        try:
            s = get_settings()
            llm = ChatOllama(model=s.simple_model, temperature=0, base_url=s.ollama_base_url)
            raw = (_CLASSIFIER_PROMPT | llm).invoke(
                {"history": _format_history(history), "query": query})
            raw = raw.content if hasattr(raw, "content") else str(raw)
            rt = (_parse_json(str(raw)).get("response_type") or "").strip()
            if rt in {"diagnostic", "conversational", "operational", "out_of_scope"}:
                return rt
        except Exception as exc:
            logger.warning("LLM classifier failed: %s", exc)

    # Heuristic fallback
    if any(re.search(p, q) for p in _CONVERSATION_PATTERNS):
        return "conversational"
    if any(re.search(p, q) for p in _OPERATIONAL_PATTERNS):
        return "operational"
    if any(_term_in(t, q) for t in _K8S_TERMS):
        return "diagnostic"
    if any(_fuzzy_term_in(t, q) for t in _TECHNICAL_KW):
        return "diagnostic"
    if history:
        recent = " ".join(
            (m.content if isinstance(m.content, str) else str(m.content)).lower()
            for m in history[-6:])
        if any(_term_in(t, recent) for t in _K8S_TERMS) or \
           any(kw in recent for kw in _TECHNICAL_KW):
            return "diagnostic"
    return "out_of_scope"


# ── Non-diagnostic handlers ──────────────────────────────────────────────────

def _user_messages(messages):
    return [m.content if isinstance(m.content, str) else str(m.content)
            for m in messages if m.type == "human"]


def _find_mentions(messages, topic):
    t = topic.lower()
    out = []
    for i, m in enumerate(messages, 1):
        content = m.content if isinstance(m.content, str) else str(m.content)
        if t in content.lower():
            role = "User" if m.type == "human" else "AI"
            out.append(f"[{role} #{i}] {content.strip().replace(chr(10), ' ')[:300]}")
    return out


def _answer_conversational(query, history, store=None):
    q = query.lower().strip()
    if history is None:
        history = []
    user_msgs = _user_messages(history)
    logger.debug("_answer_conversational: q=%s user_msgs=%d", q, len(user_msgs))

    # First / last question
    if re.search(r"\bfirst (question|message)\b", q):
        if user_msgs:
            return f"Your first question was: {user_msgs[0]}", "", 0.9, [], []
        return "I do not have any prior questions in this session.", "", 0.3, [], []

    if re.search(r"\b(last|previous) question\b", q) or "what did i ask" in q:
        if user_msgs:
            return f"Your last question was: {user_msgs[-1]}", "", 0.9, [], []
        return "I do not have any prior questions in this session.", "", 0.3, [], []

    # "Have I asked about <topic>?"
    logger.debug("_answer_conversational: checking 'have I asked' pattern")
    m = re.search(
        r"(?:have i|did i)\s+(?:ask(?:ed)?|mention(?:ed)?|talk(?:ed)?\s+about|"
        r"bring\s+up|discuss(?:ed)?)\s+(?:you\s+)?(?:about\s+)?(.+)", q)
    logger.debug("_answer_conversational: have_i_match=%s", bool(m))
    if m:
        topic = m.group(1).strip("?.! ")
        topic = topic.split("?")[0].strip()
        topic = re.sub(r"\b(today|now|recently|where|when|what|how)\b", "", topic)
        topic = re.split(r"\s+(?:and|or|but|nor)\b", topic)[0]
        topic = re.sub(r"[?.!,]+", "", topic).strip()
        logger.debug("_answer_conversational: extracted topic='%s'", topic)

        mentions = _find_mentions(history, topic)
        logger.debug("_answer_conversational: mentions_found=%d", len(mentions))
        if mentions:
            if "where" in q:
                return "Yes — mentioned in session.", "; ".join(mentions[:5]), 0.9, [], mentions
            return f"Yes, you mentioned {topic} earlier.", f"{mentions[0]}", 0.9, [], [mentions[0]]

        # KB fallback
        docs = []
        try:
            logger.debug("_answer_conversational: KB search for '%s'", topic)
            docs = (store if store is not None else MilvusStore()).search(topic)
        except Exception:
            pass

        if docs:
            sources = [_clean_source(d.metadata.get("source", "?")) for d in docs[:4]]
            evidence = _build_evidence(docs, n=4)
            return ("No prior mentions in session. I found related documents in the KB.",
                    "See evidence and sources.", 0.7, sources, evidence)

        return f"No, I don't see any prior mentions of {topic} in this session.", "", 0.6, [], []

    return "I can answer questions about our conversation in this session.", "", 0.4, [], []


def _answer_operational(query):
    hint = ""
    if "k3d" in query.lower():
        hint = (" If you are using k3d, select the correct context with "
                "`kubectl config get-contexts` and `kubectl config use-context <name>`.")
    return ("I cannot run cluster commands from here, but you can list pods locally.",
            "Use kubectl to query the cluster." + hint,
            "Run: kubectl get pods -A (or kubectl get pods -n <namespace>)", 0.2)


def _answer_out_of_scope():
    return ("I specialize in Kubernetes failure diagnosis and cannot help with that topic.",
            "Ask a Kubernetes or cluster troubleshooting question to continue.", 0.0)


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"```(?:json)?\s*(\{.*?\})\s*```", r"\{.*\}"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1) if m.lastindex else m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# ── Shared builders ───────────────────────────────────────────────────────────

def _build_non_diagnostic(response_type, query, history, store):
    """Build FailureDiagnosis for non-diagnostic query types."""
    if response_type == "conversational":
        root, expl, conf, sources, evidence = _answer_conversational(query, history, store)
        return FailureDiagnosis(root_cause=root, explanation=expl, recommended_fix="N/A",
            confidence=conf, sources=sources, evidence_snippets=evidence,
            response_type=response_type, model_used="rules")
    if response_type == "operational":
        root, expl, fix, conf = _answer_operational(query)
        return FailureDiagnosis(root_cause=root, explanation=expl, recommended_fix=fix,
            confidence=conf, response_type=response_type, model_used="rules")
    root, expl, conf = _answer_out_of_scope()
    return FailureDiagnosis(root_cause=root, explanation=expl, recommended_fix="N/A",
        confidence=conf, response_type=response_type, model_used="rules")


def _parse_llm_diagnosis(raw_text, docs, model_name, response_type):
    """Parse LLM output + docs into FailureDiagnosis."""
    parsed = _parse_json(str(raw_text))
    try:
        conf = max(0.0, min(1.0, float(parsed.get("confidence") or 0.5)))
    except (ValueError, TypeError):
        conf = 0.5
    return FailureDiagnosis(
        root_cause=parsed.get("root_cause") or str(raw_text)[:300],
        explanation=parsed.get("explanation") or "",
        recommended_fix=parsed.get("recommended_fix") or "",
        confidence=conf,
        sources=[_clean_source(d.metadata.get("source", "?")) for d in docs],
        evidence_snippets=_build_evidence(docs),
        response_type=response_type, model_used=model_name)


# ── RAG Chain ─────────────────────────────────────────────────────────────────

class RAGChain:
    def __init__(self):
        self.store = MilvusStore()
        self.memory = get_chat_memory()

    def _save_turn(self, sid, query, dx):
        self.memory.add_user_message(sid, query)
        self.memory.add_ai_message(sid, _format_memory_text(dx)[:1200])

    def diagnose(self, query, session_id="default", force_model=None):
        history = self.memory.get_history(session_id)
        rtype = _classify_query(query, history)

        if rtype != "diagnostic":
            dx = _build_non_diagnostic(rtype, query, history, self.store)
            self._save_turn(session_id, query, dx)
            return dx

        docs = self.store.search(query)
        logger.info("Retrieved %d docs for query", len(docs))
        model_name = select_model(query, force_model)
        llm = ChatOllama(model=model_name, temperature=0,
                         base_url=get_settings().ollama_base_url)
        result = (_PROMPT | llm).invoke({
            "context": _format_docs(docs),
            "history": _format_history(history), "query": query})
        raw_text = result.content if hasattr(result, "content") else str(result)

        dx = _parse_llm_diagnosis(raw_text, docs, model_name, rtype)
        self._save_turn(session_id, query, dx)
        return dx

    async def diagnose_stream(self, query, session_id="default", force_model=None):
        import asyncio
        history = self.memory.get_history(session_id)
        rtype = _classify_query(query, history)

        # Non-diagnostic: yield complete answer at once
        if rtype != "diagnostic":
            dx = _build_non_diagnostic(rtype, query, history, self.store)
            self._save_turn(session_id, query, dx)
            full = f"{dx.root_cause}\n\n{dx.explanation}"
            if dx.recommended_fix and dx.recommended_fix != "N/A":
                full += f"\n\n**Recommended Fix:**\n{dx.recommended_fix}"
            yield f"data: {json.dumps({'token': full})}\n\n"
            yield f"data: {json.dumps({'done': True, 'diagnosis': dx.model_dump()})}\n\n"
            return

        # Diagnostic: stream LLM tokens
        docs = await asyncio.to_thread(self.store.search, query)
        logger.info("Retrieved %d docs for streaming query", len(docs))
        model_name = select_model(query, force_model)
        llm = ChatOllama(model=model_name, temperature=0,
                         base_url=get_settings().ollama_base_url)
        raw_text = ""
        async for chunk in (_PROMPT | llm).astream({
            "context": _format_docs(docs),
            "history": _format_history(history), "query": query}):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                raw_text += str(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

        dx = _parse_llm_diagnosis(raw_text, docs, model_name, rtype)
        self._save_turn(session_id, query, dx)
        yield f"data: {json.dumps({'done': True, 'diagnosis': dx.model_dump()})}\n\n"
