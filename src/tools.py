"""Kubernetes diagnosis tools for DeepAgents."""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from langchain.tools import tool

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
    """Execute safe read-only kubectl commands against the cluster.

    ALLOWED commands (first word must be one of these):
      get pods, get pod <real-pod-name>, describe pod <real-pod-name>,
      logs <real-pod-name>, top pods, events

    IMPORTANT: Always use actual pod names (e.g. "data-processor-7b8f9"),
    never placeholders like <pod> or <your-pod>.

    Args:
        command: the kubectl sub-command, e.g. "get pods" or
                 "describe pod data-processor-7b8f9".
        namespace: kubernetes namespace (default: "default").

    Returns:
        {"status": "ok" | "error", "output": "...",
         "stderr": "...", "command": "..."}
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
    """Collect a quick snapshot of pods and events in a namespace.

    Use this for a broad overview before diving into specific pods.

    Args:
        namespace: kubernetes namespace to snapshot (default: "default").

    Returns:
        {"namespace": str, "pods": str, "events": list, "timestamp": str}
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
    
    snapshot["timestamp"] = datetime.now().isoformat()
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
    """Analyze a pod's logs and detect anomaly patterns.

    Detects: OOM, CrashLoop, ImagePull, Scheduling issues.
    Use an actual pod name, not a placeholder.

    Args:
        pod_name: real pod name (e.g. "data-processor-7b8f9").
        namespace: kubernetes namespace (default: "default").
        tail_lines: how many recent log lines to analyze (default: 100).

    Returns:
        {"summary": str, "anomalies": [str, ...], "raw_tail": str}
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
    """Validate a Kubernetes manifest YAML string.

    Checks for missing metadata, resource limits, and image specs.

    Args:
        yaml_content: the raw YAML string of the manifest.
        dry_run: if True, only validate — do not apply (default: True).

    Returns:
        {"valid": bool, "issues": ["...", ...]}
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
    """Search the RAG vector store for runbooks, events, and manifests.

    Use this to find documentation about failure patterns (e.g.
    "OOMKilled memory limit", "CrashLoopBackOff troubleshooting").

    Args:
        query: natural-language search query.
        top_k: max results to return (default: 5).
        source_type: optional filter — "runbook", "event", "manifest",
                     or None for all types.

    Returns:
        {"results": [{"content": str, "source": str, "doc_type": str}],
         "count": int}
    """
    try:
        store = MilvusStore()
        docs = store.search(query, k=top_k)

        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content[:500],
                "source": doc.metadata.get("source", "unknown"),
                "doc_type": doc.metadata.get("doc_type", "unknown"),
            })

        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error("RAG retrieval failed (Milvus may be unavailable): %s", e)
        return {"results": [], "count": 0, "error": f"Vector store unavailable: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Root Cause Hypothesis Tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def generate_hypotheses(symptoms: str) -> dict:
    """Generate ranked root-cause hypotheses from a symptom description.

    Call this AFTER gathering evidence (logs, events). Pass a plain-text
    summary of symptoms, e.g. "Pod crashing, OOMKilled, high memory".

    Args:
        symptoms: plain-text description of observed problems.

    Returns:
        {"hypotheses": [{"cause": str, "confidence": float,
                         "tests": [str, ...]}]}
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
    """Generate remediation commands and patches for a root cause.

    Pass the "cause" string from generate_hypotheses output.

    Args:
        hypothesis: the root-cause string, e.g.
            "Out-of-Memory (OOMKilled): Pod memory request too low".
        manifest_yaml: optional current deployment manifest YAML string.

    Returns:
        {"patches": [dict], "commands": [str], "risk_score": float,
         "explanation": str}
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
    """Evaluate whether proposed fix commands are likely to succeed.

    Pass the "commands" list from generate_fix output.

    Args:
        fix_commands: list of kubectl / remediation commands.
        cluster_health_check: brief description of current cluster state
            (default: "cluster healthy").

    Returns:
        {"likely_effective": bool, "missing_steps": [str],
         "risks": [str], "recommendation": str}
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
