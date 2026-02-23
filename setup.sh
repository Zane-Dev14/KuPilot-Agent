#!/bin/bash
set -e

# ────────────────────────────────────────────────────────────────────────────
# K8s Failure Intelligence Copilot — Setup Script
# 
# Orchestrates:
#   1. Docker + docker-compose verification
#   2. Milvus startup (via docker-compose) + health check
#   3. Python venv + dependencies
#   4. Ollama model availability
# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILVUS_PORT="19530"
MILVUS_HEALTH_PORT="9091"
MILVUS_HEALTH_CHECK="http://localhost:${MILVUS_HEALTH_PORT}/healthz"
MAX_WAIT_SECONDS=60

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────

log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Wait for service to be healthy
wait_for_service() {
    local service_name="$1"
    local health_check_url="$2"
    local max_wait="$3"
    
    local elapsed=0
    while [ $elapsed -lt "$max_wait" ]; do
        if curl -sf "$health_check_url" >/dev/null 2>&1; then
            log_info "$service_name is healthy"
            return 0
        fi
        echo -n "."
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    log_error "$service_name did not become healthy within ${max_wait}s"
    return 1
}

# ────────────────────────────────────────────────────────────────────────────
# Main Setup
# ────────────────────────────────────────────────────────────────────────────

echo ""
echo "🚀 Setting up K8s Failure Intelligence Copilot..."
echo ""

# Handle --fix-milvus argument
if [ "$1" == "--fix-milvus" ]; then
    echo "🔧 Fixing Milvus (wiping volumes and rebuilding)..."
    log_info "Stopping docker-compose services"
    docker-compose down -v || true
    sleep 2
    log_info "Starting fresh Milvus cluster"
    docker-compose up -d
    log_info "Waiting for Milvus to be healthy..."
    if ! wait_for_service "Milvus" "$MILVUS_HEALTH_CHECK" "$MAX_WAIT_SECONDS"; then
        log_error "Milvus failed to start. Check: docker-compose logs milvus-standalone"
        exit 1
    fi
    log_info "Milvus recovery complete"
    echo ""
    echo "🚀 Ready to start the server:"
    echo "   source venv/bin/activate"
    echo "   python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000"
    exit 0
fi

# ✓ Step 1: Check Docker & docker-compose
echo "1. Checking Docker environment..."
if ! command -v docker >/dev/null 2>&1; then
    log_error "Docker not found. Install from https://www.docker.com/products/docker-desktop"
    exit 1
fi
log_info "Docker installed"

if ! command -v docker-compose >/dev/null 2>&1; then
    log_error "docker-compose not found"
    exit 1
fi
log_info "docker-compose installed"
echo ""

# ✓ Step 2: Start Milvus via docker-compose
echo "2. Starting Milvus via docker-compose..."
cd "$SCRIPT_DIR"
log_info "Running docker-compose up -d"
docker-compose up -d

log_info "Waiting for Milvus to be healthy (max ${MAX_WAIT_SECONDS}s)..."
if ! wait_for_service "Milvus" "$MILVUS_HEALTH_CHECK" "$MAX_WAIT_SECONDS"; then
    log_error "Milvus failed to start. Run './setup.sh --fix-milvus' to rebuild, or check:"
    echo "   docker-compose logs milvus-standalone"
    exit 1
fi
echo ""

# ✓ Step 3: Check Python version
echo "3. Checking Python..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo "$python_version" | cut -d. -f1)
python_minor=$(echo "$python_version" | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
    log_error "Python 3.10+ required, found $python_version"
    exit 1
fi
log_info "Python $python_version detected"
echo ""

# ✓ Step 4: Create & activate virtual environment
echo "4. Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    log_info "Creating venv"
    python3 -m venv venv
fi
source venv/bin/activate
log_info "venv activated"
echo ""

# ✓ Step 5: Install dependencies
echo "5. Installing dependencies..."
log_info "Upgrading pip/setuptools/wheel"
pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
log_info "Installing requirements.txt"
pip install -r requirements.txt >/dev/null 2>&1
echo ""

# ✓ Step 6: Check Ollama
echo "6. Checking Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
    log_warn "Ollama not found. Install from https://ollama.ai then run:"
    echo "   ollama pull qwen2.5-coder:7b"
    echo ""
    echo "   (You can also try: brew install ollama)"
    echo ""
else
    log_info "Ollama found"
    
    # Auto-pull model if missing
    if ! ollama list 2>/dev/null | grep -q "qwen2.5-coder"; then
        log_warn "qwen2.5-coder not found. Pulling (may take a few minutes)..."
        ollama pull qwen2.5-coder:14b
    else
        log_info "qwen2.5-coder:14b available"
    fi
fi
echo ""

# ✓ Summary
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. (Optional) Start Ollama in a separate terminal:"
echo "       ollama serve"
echo ""
echo "  2. Start the server:"
echo "       source venv/bin/activate"
echo "       python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "  3. Open http://localhost:8000 in your browser"
echo ""
echo "Troubleshooting:"
echo "  • Milvus not starting? → ./setup.sh --fix-milvus"
echo "  • Check service logs → docker-compose logs -f"
echo ""