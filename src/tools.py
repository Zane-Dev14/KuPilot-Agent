"""Kubernetes diagnosis tools for DeepAgents."""

import json
import logging
import subprocess
import yaml
from pathlib import Path
from typing import Optional
from datetime import datetime

from langchain.tools import tool
from pydantic import BaseModel

from src.config import get_settings
from src.vectorstore import MilvusStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Kubectl Tool — Real cluster + simulation fallback
# ─────────────────────────────────────────────────────────────────────────────

_KUBECTL_ALLOWLIST = {"get", "describe", "logs", "top", "events"}


def _load_sample_data(resource_type: str, name: str = "") -> dict:
    """Load sample data from data/sample/ when kubectl unavailable."""
    sample_dir = Path("data/sample")
    if resource_type == "events":
        events_dir = sample_dir / "events"
        if events_dir.exists():
            for f in events_dir.glob("*.json"):
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                        events = data if isinstance(data, list) else [data]
                        return {"items": events[:5]}
                except Exception:
                    pass
    elif resource_type == "pods":
        return {"items": []}
    return {"items": []}


@tool
def kubectl_exec(command: str, namespace: str = "default") -> dict:
    """
    Execute safe kubernetes commands. Read-only by default.
    
    Args:
        command: kubectl verb (get, describe, logs, top, events)
        namespace: kubernetes namespace
    
    Returns:
        {"status": "ok"|"error", "output": str, "stderr": str}
    """
    return _kubectl_exec_impl(command, namespace)


def _kubectl_exec_impl(command: str, namespace: str = "default") -> dict:
    """Internal implementation of kubectl execution."""
    parts = command.strip().split()
    if not parts or parts[0] not in _KUBECTL_ALLOWLIST:
        return {"status": "error", "output": "", "stderr": "Command not allowed (allowlist: get, describe, logs, top, events)"}
    
    try:
        # Try real kubectl first
        full_cmd = ["kubectl", "-n", namespace] + parts
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=10)
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "output": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(full_cmd)
        }
    except FileNotFoundError:
        # kubectl not found, try simulation
        logger.info("kubectl not found, loading sample data")
        if "events" in command:
            data = _load_sample_data("events")
            return {"status": "ok", "output": json.dumps(data, indent=2), "stderr": "", "mode": "simulated"}
        return {"status": "error", "output": "", "stderr": "kubectl not available and no simulated data for this command"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "output": "", "stderr": "Command timed out"}
    except Exception as e:
        return {"status": "error", "output": "", "stderr": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Cluster Snapshot Tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def cluster_snapshot(namespace: str = "default") -> dict:
    """
    Collect a snapshot of cluster state: pods, events, resource status.
    
    Args:
        namespace: kubernetes namespace to snapshot
    
    Returns:
        {"pods": [...], "events": [...], "timestamp": str}
    """
    snapshot = {"namespace": namespace, "pods": [], "events": []}
    
    # Get pods
    pods_result = _kubectl_exec_impl("get pods", namespace)
    if pods_result["status"] == "ok":
        try:
            # Parse simple output or JSON
            snapshot["pods"] = pods_result["output"][:500]  # Stub
        except Exception:
            pass
    
    # Get events
    events_result = _kubectl_exec_impl("events", namespace)
    if events_result["status"] == "ok":
        try:
            data = json.loads(events_result["output"])
            snapshot["events"] = data.get("items", [])[:10]
        except Exception:
            pass
    
    snapshot["timestamp"] = datetime.utcnow().isoformat()
    return snapshot


# ─────────────────────────────────────────────────────────────────────────────
# Log Analysis Tool
# ─────────────────────────────────────────────────────────────────────────────

_ANOMALIES = {
    "OOM": ["OutOfMemory", "OOMKilled", "out of memory"],
    "CrashLoop": ["CrashLoopBackOff", "Back-off pulling image"],
    "ImagePull": ["InvalidImageName", "ErrImagePull"],
    "Scheduling": ["FailedScheduling", "insufficient"],
}


@tool
def analyze_logs(pod_name: str, namespace: str = "default", tail_lines: int = 100) -> dict:
    """
    Analyze logs from a pod, detect anomalies.
    
    Args:
        pod_name: name of the pod
        namespace: namespace of the pod
        tail_lines: number of log lines to fetch
    
    Returns:
        {"summary": str, "anomalies": [...], "raw_tail": str}
    """
    result = _kubectl_exec_impl(f"logs {pod_name}", namespace)
    logs = result.get("output", "")
    
    anomalies = []
    for anom_type, keywords in _ANOMALIES.items():
        for kw in keywords:
            if kw.lower() in logs.lower():
                anomalies.append(anom_type)
                break
    
    # Simple summary
    lines = logs.split("\n")[-tail_lines:]
    summary = f"Analyzed {len(lines)} lines of logs. Found {len(set(anomalies))} anomaly patterns: {', '.join(set(anomalies)) or 'none'}"
    
    return {
        "summary": summary,
        "anomalies": list(set(anomalies)),
        "raw_tail": "\n".join(lines[-10:])
    }


# ─────────────────────────────────────────────────────────────────────────────
# Manifest Validator Tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def validate_manifest(yaml_content: str, dry_run: bool = True) -> dict:
    """
    Validate a Kubernetes manifest YAML.
    
    Args:
        yaml_content: YAML as string
        dry_run: if True, don't apply, just validate
    
    Returns:
        {"valid": bool, "issues": [...]}
    """
    issues = []
    
    try:
        objs = list(yaml.safe_load_all(yaml_content))
    except yaml.YAMLError as e:
        return {"valid": False, "issues": [f"YAML parse error: {e}"]}
    
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        
        kind = obj.get("kind", "Unknown")
        name = obj.get("metadata", {}).get("name", "unknown")
        
        # Check for required fields
        if "metadata" not in obj:
            issues.append(f"{kind}/{name}: missing metadata")
        
        # Check resource limits (for Pods/Deployments)
        if kind in ("Pod", "Deployment", "StatefulSet"):
            spec = obj.get("spec", {})
            containers = spec.get("containers", [])
            for i, c in enumerate(containers):
                if not c.get("resources", {}).get("limits"):
                    issues.append(f"{kind}/{name} container[{i}]: no resource limits")
                if not c.get("image"):
                    issues.append(f"{kind}/{name} container[{i}]: no image specified")
    
    return {"valid": len(issues) == 0, "issues": issues}


# ─────────────────────────────────────────────────────────────────────────────
# RAG Retrieval Tool — Wraps existing vectorstore
# ─────────────────────────────────────────────────────────────────────────────

@tool
def retrieve_docs(query: str, top_k: int = 5, source_type: Optional[str] = None) -> dict:
    """
    Query the RAG vector store for relevant documentation.
    
    Args:
        query: search query (e.g., "pod Ooomkilled")
        top_k: number of results
        source_type: filter by "runbook", "event", "manifest", or None for all
    
    Returns:
        {"results": [{"content": str, "source": str, "score": float}]}
    """
    try:
        store = MilvusStore()
        docs = store.search(query, k=top_k)
        
        results = []
        for doc in docs:
            result = {
                "content": doc.page_content[:500],
                "source": doc.metadata.get("source", "unknown"),
                "doc_type": doc.metadata.get("doc_type", "unknown"),
            }
            results.append(result)
        
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error("RAG retrieval failed: %s", e)
        return {"results": [], "count": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Root Cause Hypothesis Tool
# ─────────────────────────────────────────────────────────────────────────────

class Hypothesis(BaseModel):
    cause: str
    confidence: float
    tests: list[str]


@tool
def generate_hypotheses(symptoms: str) -> dict:
    """
    Generate ranked root cause hypotheses based on symptoms.
    
    Args:
        symptoms: description of failures (e.g., "Pod crashing, high memory use")
    
    Returns:
        {"hypotheses": [{"cause": str, "confidence": float, "tests": [...]}]}
    """
    # Simple rule-based hypothesis generator
    hypotheses = []
    
    if "oom" in symptoms.lower() or "memory" in symptoms.lower():
        hypotheses.append({
            "cause": "Out-of-Memory (OOMKilled): Pod memory request too low or memory leak in app.",
            "confidence": 0.9,
            "tests": [
                "kubectl top pods -n namespace",
                "Check pod memory requests/limits",
                "Check application logs for memory issues",
                "Increase memory limit and redeploy"
            ]
        })
    
    if "crash" in symptoms.lower() or "crash" in symptoms.lower():
        hypotheses.append({
            "cause": "CrashLoopBackOff: Application error or missing dependencies.",
            "confidence": 0.85,
            "tests": [
                "kubectl logs <pod> -n namespace",
                "kubectl describe pod <pod> -n namespace",
                "Check liveness/readiness probes",
                "Verify container image exists"
            ]
        })
    
    if "image" in symptoms.lower() or "pull" in symptoms.lower():
        hypotheses.append({
            "cause": "ImagePullBackOff: Image registry unreachable or invalid image name.",
            "confidence": 0.8,
            "tests": [
                "kubectl describe pod <pod> -n namespace",
                "Check image name spelling",
                "Verify registry credentials",
                "Test: docker pull <image>"
            ]
        })
    
    if "schedule" in symptoms.lower() or "pending" in symptoms.lower():
        hypotheses.append({
            "cause": "FailedScheduling: Insufficient node resources (CPU, memory) or node affinity mismatch.",
            "confidence": 0.8,
            "tests": [
                "kubectl top nodes",
                "kubectl describe nodes",
                "Check pod resource requests",
                "Check node affinities/taints"
            ]
        })
    
    # Default
    if not hypotheses:
        hypotheses.append({
            "cause": "General pod failure — retrieve diagnostic runbooks.",
            "confidence": 0.5,
            "tests": ["Retrieve documentation", "Analyze logs", "Check events"]
        })
    
    return {"hypotheses": hypotheses}


# ─────────────────────────────────────────────────────────────────────────────
# Fix Generator Tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def generate_fix(hypothesis: str, manifest_yaml: Optional[str] = None) -> dict:
    """
    Generate remediation steps and patches.
    
    Args:
        hypothesis: the root cause (from generate_hypotheses)
        manifest_yaml: current deployment manifest (optional)
    
    Returns:
        {"patches": [...], "commands": [...], "risk_score": float, "explanation": str}
    """
    commands = []
    patches = []
    risk_score = 0.3
    explanation = "Review and test before applying."
    
    if "OOMKilled" in hypothesis:
        commands = [
            "kubectl set resources deployment/<name> -n <namespace> --limits=memory=1Gi",
            "kubectl rollout restart deployment/<name> -n <namespace>"
        ]
        patches = [{"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value": "1Gi"}]
        risk_score = 0.2
        explanation = "Increase memory limit. Monitor using 'kubectl top pods' after applying."
    
    elif "CrashLoopBackOff" in hypothesis:
        commands = [
            "kubectl logs <pod> -n <namespace> --tail=50",
            "kubectl describe pod <pod> -n <namespace>"
        ]
        explanation = "Examine logs to find the app error, then fix the application code or configuration."
        risk_score = 0.5
    
    elif "ImagePullBackOff" in hypothesis:
        commands = [
            "kubectl set image deployment/<name> <container>=<new-image> -n <namespace>",
            "kubectl rollout status deployment/<name> -n <namespace>"
        ]
        explanation = "Correct the image name or ensure registry is accessible."
        risk_score = 0.2
    
    elif "FailedScheduling" in hypothesis:
        commands = [
            "kubectl top nodes",
            "kubectl set resources deployment/<name> -n <namespace> --requests=cpu=100m,memory=128Mi"
        ]
        explanation = "Lower resource requests or add more nodes to the cluster."
        risk_score = 0.3
    
    return {
        "patches": patches,
        "commands": commands,
        "risk_score": risk_score,
        "explanation": explanation
    }


# ─────────────────────────────────────────────────────────────────────────────
# Verification Tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def verify_fix(fix_commands: list[str], cluster_health_check: str = "cluster healthy") -> dict:
    """
    Evaluate if a fix is likely to succeed.
    
    Args:
        fix_commands: list of commands to be executed
        cluster_health_check: description of current cluster state
    
    Returns:
        {"likely_effective": bool, "missing_steps": [...], "risks": [...]}
    """
    risks = []
    missing_steps = []
    
    # Check for apply without dry-run
    for cmd in fix_commands:
        if "apply" in cmd and "--dry-run" not in cmd:
            risks.append("apply command without --dry-run (should test first)")
            missing_steps.append("Add --dry-run=client to validate first")
    
    # Check for rollout restart
    if any("rollout restart" in cmd for cmd in fix_commands):
        missing_steps.append("Wait for rollout to complete: kubectl rollout status <deployment> -n <namespace>")
    
    # Check monitoring
    if not any("top" in cmd or "logs" in cmd for cmd in fix_commands):
        missing_steps.append("Monitor fix: kubectl top pods, kubectl logs <pod>")
    
    likely_effective = "healthy" in cluster_health_check.lower() and len(risks) == 0
    
    return {
        "likely_effective": likely_effective,
        "missing_steps": missing_steps,
        "risks": risks,
        "recommendation": "dry-run first, monitor health metrics post-deployment"
    }


# ─────────────────────────────────────────────────────────────────────────────
# Export all tools for DeepAgents
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    kubectl_exec,
    cluster_snapshot,
    analyze_logs,
    validate_manifest,
    retrieve_docs,
    generate_hypotheses,
    generate_fix,
    verify_fix,
]
