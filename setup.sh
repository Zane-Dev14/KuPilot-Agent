#!/bin/zsh
# All-in-one setup for K8s Failure Intelligence Copilot (macOS / zsh)
# Usage: ./setup.sh [--fix-milvus]
set -e
cd "$(dirname "$0")"

G='\033[0;32m' Y='\033[1;33m' R='\033[0;31m' B='\033[0;34m' N='\033[0m'
step() { echo "\n${Y}── $1${N}" }
ok()   { echo "${G}✓ $1${N}" }

FIX_MILVUS=false
for arg in "$@"; do [[ "$arg" == "--fix-milvus" ]] && FIX_MILVUS=true; done

echo "${B}═══════════════════════════════════════════════════════${N}"
echo "${B}  K8s Failure Intelligence Copilot — Setup${N}"
echo "${B}═══════════════════════════════════════════════════════${N}"

# ── 1. Prerequisites ─────────────────────────────────────────────────────────
step "Checking prerequisites"
command -v python3 &>/dev/null || { echo "${R}❌ python3 not found${N}"; exit 1; }
ok "Python 3 ($(python3 --version 2>&1))"
command -v docker &>/dev/null  || { echo "${R}❌ Docker not found${N}"; exit 1; }
ok "Docker"
(docker compose version &>/dev/null || docker-compose version &>/dev/null) \
    || { echo "${R}❌ Docker Compose not found${N}"; exit 1; }
ok "Docker Compose"
command -v ollama &>/dev/null && ok "Ollama" \
    || echo "${Y}⚠  Ollama not found — install from https://ollama.ai${N}"

# ── 2. Milvus ────────────────────────────────────────────────────────────────
step "Starting Milvus"
if [[ "$FIX_MILVUS" == true ]]; then
    echo "  Clean rebuild (--fix-milvus)..."
    docker compose down 2>/dev/null || true
    docker volume rm $(docker volume ls -q | grep -E 'milvus|etcd|minio' 2>/dev/null) 2>/dev/null || true
    sleep 2
fi
docker compose up -d
echo "  Waiting for Milvus (up to 60s)..."
for i in {1..60}; do
    curl -s http://localhost:19530 &>/dev/null \
        && docker ps | grep milvus-standalone | grep -q healthy 2>/dev/null \
        && break
    [[ $((i % 15)) -eq 0 ]] && echo "    ${i}s..."
    sleep 1
done
curl -s http://localhost:19530 &>/dev/null && ok "Milvus ready" \
    || echo "${Y}⚠  Milvus may not be ready yet${N}"

# ── 3. Python env ────────────────────────────────────────────────────────────
step "Python environment"
[[ -d venv ]] || python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt
ok "Dependencies installed"

# ── 4. Tests ─────────────────────────────────────────────────────────────────
step "Running offline tests"
pytest tests/test_basic.py -v --tb=short || { echo "${R}❌ Tests failed${N}"; exit 1; }
ok "All tests passed"

# ── 5. Diagnostics ───────────────────────────────────────────────────────────
step "System diagnostics"
python3 -c "
from src.config import get_settings
from src.vectorstore import MilvusStore
s = get_settings()
print(f'  Embedding:  {s.embedding_model} ({s.embedding_dimension}-dim)')
print(f'  LLMs:       {s.simple_model} / {s.complex_model}')
print(f'  Milvus:     {s.milvus_uri}')
ok = MilvusStore().health_check()
print(f'  Connection: {\"✓ connected\" if ok else \"❌ not connected\"}')
if not ok: exit(1)
"
ok "All systems go"

# ── 6. Ingest sample data ───────────────────────────────────────────────────
step "Ingesting sample data"
python3 scripts/ingest.py
ok "Data ingested"

# ── Done ──────────────────────────────────────────────────────────────────────
echo "\n${B}═══════════════════════════════════════════════════════${N}"
echo "${G}✅ Setup complete!${N}"
echo "${B}═══════════════════════════════════════════════════════${N}"
echo "
Next steps:
  1. Start Ollama:     ${Y}ollama serve${N}
  2. Pull models:      ${Y}ollama pull llama3.1:8b-instruct-q8_0${N}
  3. CLI chat:         ${Y}python scripts/chat.py${N}
  4. Web UI:           ${Y}python -m uvicorn src.api:app --reload${N}
     Open ${Y}http://localhost:8000${N} in your browser
"
