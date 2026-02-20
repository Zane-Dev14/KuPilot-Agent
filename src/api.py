"""FastAPI server for K8s Failure Intelligence Copilot."""

import json, logging, sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.config import get_settings
from src.rag_chain import RAGChain, FailureDiagnosis, estimate_complexity
from src.ingestion import ingest_file, ingest_directory
from src.vectorstore import MilvusStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("src.rag_chain").setLevel(logging.DEBUG)
logging.getLogger("src.vectorstore").setLevel(logging.DEBUG)


# ── Schemas ───────────────────────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    question: str = Field(..., min_length=1)
    namespace: Optional[str] = None
    force_model: Optional[str] = None
    session_id: Optional[str] = "default"

class DiagnoseResponse(BaseModel):
    diagnosis: FailureDiagnosis
    session_id: Optional[str] = None
    complexity_score: float

class IngestRequest(BaseModel):
    path: str
    doc_type: Optional[str] = None
    no_drop: bool = False

class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    chunks_stored: int
    errors: list = Field(default_factory=list)


# ── App setup ─────────────────────────────────────────────────────────────────

_rag_chain: Optional[RAGChain] = None

def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain

def _clean_force(v):
    return None if v in (None, "", "string") else v

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("K8s Failure Intelligence Copilot starting — %s", get_settings())
    yield
    logger.info("Shutting down")

app = FastAPI(title="Kubernetes Failure Intelligence Copilot", version="0.1.0",
              lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    s = get_settings()
    return templates.TemplateResponse("index.html", {
        "request": request, "simple_model": s.simple_model, "complex_model": s.complex_model})

@app.get("/health")
async def health():
    try:
        store = MilvusStore()
        ok = await run_in_threadpool(store.health_check)
        payload = {"status": "ok" if ok else "degraded",
                   "milvus": "connected" if ok else "disconnected"}
        return payload if ok else JSONResponse(status_code=503, content=payload)
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})

@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    try:
        chain = get_rag_chain()
        dx = await run_in_threadpool(
            chain.diagnose, request.question,
            request.session_id or "default",
            _clean_force(request.force_model))
        return DiagnoseResponse(diagnosis=dx, session_id=request.session_id,
                                complexity_score=estimate_complexity(request.question))
    except Exception as e:
        logger.error("Diagnosis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diagnose/stream")
async def diagnose_stream(request: DiagnoseRequest):
    try:
        chain = get_rag_chain()
        force = _clean_force(request.force_model)
        async def sse():
            try:
                async for event in chain.diagnose_stream(
                    request.question, request.session_id or "default", force):
                    yield event
            except Exception as e:
                logger.error("Stream error: %s", e, exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-analysis")
async def analyze_query(request: DiagnoseRequest):
    c = estimate_complexity(request.question)
    s = get_settings()
    model = s.complex_model if c >= s.query_complexity_threshold else s.simple_model
    return {"analysis": {"question": request.question, "complexity_score": c, "estimated_model": model}}

@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    try:
        target = Path(request.path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        store = MilvusStore(drop_old=not request.no_drop)
        loader = ingest_directory if target.is_dir() else ingest_file
        docs = await run_in_threadpool(loader, target)
        if not docs:
            return IngestResponse(documents_loaded=0, chunks_created=0, chunks_stored=0,
                                  errors=["No documents found"])
        ids = await run_in_threadpool(store.add_documents, docs)
        return IngestResponse(documents_loaded=len(docs), chunks_created=len(docs),
                              chunks_stored=len(ids))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Ingestion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/clear")
async def clear_memory(request: Request):
    body = await request.json()
    sid = body.get("session_id", "default")
    get_rag_chain().memory.clear(sid)
    return {"status": "ok", "session_id": sid}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
