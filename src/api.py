"""FastAPI server for K8s Failure Intelligence Copilot."""

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.ingestion import ingest_file, ingest_directory  # noqa: E402
from src.vectorstore import MilvusStore  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("src.rag_chain").setLevel(logging.DEBUG)
logging.getLogger("src.vectorstore").setLevel(logging.DEBUG)


# ── Schemas ───────────────────────────────────────────────────────────────────

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
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    try:
        target = Path(request.path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        store = MilvusStore(drop_old=not request.no_drop)
        
        # Determine if target is file or directory and load accordingly
        if target.is_dir():
            docs = await run_in_threadpool(ingest_directory, target)
        else:
            docs = await run_in_threadpool(ingest_file, target)
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

# ── Agent endpoints ──────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_steps: int = Field(default=6, ge=1, le=20)


@app.post("/agent/diagnose")
async def agent_diagnose_orchestrator(request: AgentRequest):
    """Run the multi-agent orchestrator and return structured diagnosis.
    
    This endpoint uses the orchestrator agent which coordinates:
    - Investigator: root cause analysis
    - Knowledge: runbook/documentation retrieval
    - Remediation: fix generation and execution
    - Verification: safety assessment
    
    Returns structured diagnosis with root cause, evidence, fix applied, and verification.
    """
    try:
        from src.agents import create_orchestrator_agent
        from scripts.agent import run_agent_with_tools
        
        # Create orchestrator with all subagents
        orchestrator = await run_in_threadpool(create_orchestrator_agent)
        
        # Run orchestrator with tool execution loop
        response, steps = await run_in_threadpool(
            run_agent_with_tools,
            orchestrator,
            request.query,
            max_steps=request.max_steps,
            verbose=False
        )
        
        # Parse response to extract structured components
        # (orchestrator should output structured format per prompt)
        result = {
            "type": "diagnosis",
            "response": response,
            "steps": steps,
            "query": request.query,
        }
        
        # Try to parse structured fields from response
        # Look for sections like "Root Cause:", "Evidence:", "Fix Applied:", etc.
        if response:
            sections = {}
            current_section = None
            lines = response.split("\n")
            for line in lines:
                if line.startswith("**Root Cause:**"):
                    current_section = "root_cause"
                    sections[current_section] = line.replace("**Root Cause:**", "").strip()
                elif line.startswith("**Evidence:**"):
                    current_section = "evidence"
                    sections[current_section] = []
                elif line.startswith("**Relevant Documentation:**"):
                    current_section = "documentation"
                    sections[current_section] = []
                elif line.startswith("**Fix Applied:**"):
                    current_section = "fix"
                    sections[current_section] = []
                elif line.startswith("**Verification:**"):
                    current_section = "verification"
                    sections[current_section] = []
                elif line.startswith("**Next Steps:**"):
                    current_section = "next_steps"
                    sections[current_section] = []
                elif line.startswith("**Confidence:**"):
                    sections["confidence"] = line.replace("**Confidence:**", "").strip()
                    current_section = None
                elif current_section and line.strip():
                    if isinstance(sections.get(current_section), list):
                        sections[current_section].append(line.strip())
                    elif current_section in sections:
                        sections[current_section] += " " + line.strip()
            
            result.update(sections)
        
        return result
        
    except Exception as e:
        logger.error("Orchestrator diagnose failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




