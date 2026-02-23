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

    EXAMPLES of CORRECT usage:
      "get pods"                             → list all pods in namespace
      "describe pod data-processor-abc123"   → show full pod details
      "logs data-processor-abc123"           → fetch pod logs
      "top pods"                             → show resource usage
      "events"                               → show cluster events

    NEVER use placeholders. Use REAL pod names you discovered from "get pods".

    Args:
        command: the kubectl sub-command (verb + resource + name if applicable).
        namespace: kubernetes namespace (default: "default").

    Returns:
        {"status": "ok" | "error", "output": "...",
         "stderr": "...", "command": "..."}
    """
    return _kubectl_exec_impl(command, namespace)


def _kubectl_exec_impl(command: str, namespace: str = "default") -> dict:
    """Internal implementation of kubectl execution."""
    cmd = command.strip()
    if cmd == "events":
        cmd = "get events"

    parts = cmd.split()
    if not parts or parts[0] not in _KUBECTL_ALLOWLIST:
        return {
            "status": "error",
            "output": "",
            "stderr": "Command not allowed (allowlist: get, describe, logs, top, events)",
        }

    try:
        # Try real kubectl first
        if "--all-namespaces" in parts or "-A" in parts:
            full_cmd = ["kubectl"] + parts
        else:
            full_cmd = ["kubectl", "-n", namespace] + parts
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=10)
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "output": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(full_cmd),
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
    """Collect comprehensive snapshot of pods, deployments, network policies, and events.

    Checks:
    - Pod status, restarts, and container state
    - Deployment replicas (desired vs running)
    - NetworkPolicy (checks if deny-all exists)
    - Cluster events

    Args:
        namespace: kubernetes namespace to snapshot (default: "default").

    Returns:
        {"namespace": str, "pods": list, "deployments": list, 
         "network_policies": list, "events": list, "timestamp": str}
    """
    snapshot = {
        "namespace": namespace,
        "pods": [],
        "deployments": [],
        "network_policies": [],
        "events": []
    }
    
    # Get pods with full status
    pods_result = _kubectl_exec_impl("get pods", namespace)
    if pods_result["status"] == "ok":
        # Parse pod output for key info
        snapshot["pods"] = pods_result["output"][:1000]
    
    # Get deployments to check replica state
    deployments_result = _kubectl_exec_impl("get deployments", namespace)
    if deployments_result["status"] == "ok":
        snapshot["deployments"] = deployments_result["output"][:500]
    
    # Get network policies (critical for networking diagnostics)
    netpol_result = _kubectl_exec_impl("get networkpolicy", namespace)
    if netpol_result["status"] == "ok":
        snapshot["network_policies"] = netpol_result["output"][:500]
    
    # Get events
    events_result = _kubectl_exec_impl("get events", namespace)
    if events_result["status"] == "ok":
        try:
            data = json.loads(events_result["output"])
            snapshot["events"] = data.get("items", [])[:15]
        except Exception:
            pass
    
    snapshot["timestamp"] = datetime.now().isoformat()
    return snapshot


# ─────────────────────────────────────────────────────────────────────────────
# Log Analysis Tool
# ─────────────────────────────────────────────────────────────────────────────

_ANOMALIES = {
    "OOM": ["OutOfMemory", "OOMKilled", "out of memory"],
    "CrashLoop": ["CrashLoopBackOff"],
    "ImagePull": ["InvalidImageName", "ErrImagePull", "ErrImageNeverPull", "ImagePullBackOff", "Back-off pulling image"],
    "Scheduling": ["FailedScheduling", "insufficient"],
}


@tool
def analyze_logs(pod_name: str, namespace: str = "default", tail_lines: int = 100) -> dict:
    """Analyze a pod's logs AND container state for anomalies.

    Checks:
    - Container logs for crash errors
    - kubectl describe pod for termination reason (OOMKilled, ImagePullBackOff, etc.)
    - Pod status and restart count

    Args:
        pod_name: The actual pod name (copy from 'kubectl get pods' output).
        namespace: kubernetes namespace (default: "default").
        tail_lines: how many recent log lines to analyze (default: 100).

    Returns:
        {"summary": str, "anomalies": [str], "last_state": str, "termination_reason": str, "raw_tail": str}
    """
    # Get logs
    logs_result = _kubectl_exec_impl(f"logs {pod_name}", namespace)
    logs = logs_result.get("output", "")
    
    # Get pod description for termination state
    describe_result = _kubectl_exec_impl(f"describe pod {pod_name}", namespace)
    describe_output = describe_result.get("output", "")
    
    anomalies = []
    termination_reason = ""
    last_state = ""
    
    # Check logs for anomalies
    for anom_type, keywords in _ANOMALIES.items():
        for kw in keywords:
            if kw.lower() in logs.lower():
                anomalies.append(anom_type)
                break
    
    # Parse kubectl describe output for Last State and termination reason
    lines = describe_output.split("\n")
    in_last_state = False
    for i, line in enumerate(lines):
        if "Last State:" in line:
            in_last_state = True
            # Capture next few lines for context
            last_state = "\n".join(lines[i:min(i+5, len(lines))])
            if "ContainerTerminated" not in anomalies:
                anomalies.append("ContainerTerminated")
        if in_last_state and "Reason:" in line:
            termination_reason = line.split("Reason:")[1].strip() if "Reason:" in line else ""
            if "OOMKilled" in termination_reason:
                if "OOMKilled" not in anomalies:
                    anomalies.append("OOMKilled")
            elif "Error" in termination_reason or "exit code" in line.lower():
                if "CrashLoop" not in anomalies:
                    anomalies.append("CrashLoop")
            in_last_state = False
        if "ErrImagePull" in line or "ImagePullBackOff" in line or "ErrImageNeverPull" in line:
            if "ImagePull" not in anomalies:
                anomalies.append("ImagePull")
    
    # Build summary
    summary_parts = [f"Pod: {pod_name}", f"Logs analyzed: {len(logs.split())} lines"]
    if termination_reason:
        summary_parts.append(f"Termination reason: {termination_reason}")
    if anomalies:
        summary_parts.append(f"Anomalies detected: {', '.join(set(anomalies))}")
    
    return {
        "summary": " | ".join(summary_parts),
        "anomalies": list(set(anomalies)),
        "last_state": last_state,
        "termination_reason": termination_reason,
        "raw_tail": "\n".join(logs.split("\n")[-10:])
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

    EXAMPLES of useful queries:
      "OOMKilled memory pod"     → find docs about OOM failures
      "CrashLoopBackOff fix"     → find remediation for crash loops
      "ImagePull error recovery" → find image pull troubleshooting
      "high memory usage"        → find memory management docs

    Args:
        query: natural-language search query (describe the problem).
        top_k: max results to return (default: 5).
        source_type: optional filter — "runbook", "event", "manifest",
                     or None to search all types.

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
    """Generate ranked root-cause hypotheses from symptoms and observed pod state.

    Distinguishes between:
    - OOMKilled (termination reason, memory limits)
    - CrashLoopBackOff (app crash, exit code)
    - ImagePullBackOff (image pull failure)
    - Replicas=0 (deployment not running)
    - NetworkPolicy blocking (deny-all traffic)
    - Scheduling failures (resource constraints)

    Args:
        symptoms: plain-text description of observed problems from pod state.

    Returns:
        {"hypotheses": [{"cause": str, "confidence": float, "tests": [str]}]}
    """
    hypotheses = []
    
    # CRITICAL: Check for replicas=0 first (service unavailable)
    if "replicas: 0" in symptoms.lower() or "desired 0" in symptoms.lower():
        hypotheses.append({
            "cause": "Deployment replicas=0: Service is scaled down or deployment is not running. No pods exist.",
            "confidence": 0.99,
            "tests": ["kubectl get deployment <name>", "kubectl scale deployment <name> --replicas=1"]
        })
    
    # Check for NetworkPolicy deny-all
    if "deny-all" in symptoms.lower() or "networkpolicy" in symptoms.lower():
        hypotheses.append({
            "cause": "Deny-All NetworkPolicy: A deny-all network policy is blocking all ingress/egress traffic.",
            "confidence": 0.95,
            "tests": ["kubectl get networkpolicy", "kubectl describe networkpolicy <policy-name>", "Review ingress/egress rules"]
        })
    
    # OOMKilled termination reason
    if "oomkilled" in symptoms.lower() or "out of memory" in symptoms.lower():
        hypotheses.append({
            "cause": "Out-of-Memory (OOMKilled): Container ran out of memory. Check memory limit vs actual usage.",
            "confidence": 0.95,
            "tests": ["kubectl describe pod <pod>", "kubectl top pods", "Increase memory limit in deployment"]
        })
    
    # ImagePullBackOff vs CrashLoopBackOff distinction
    if (
        "imagepull" in symptoms.lower()
        or "errimagepull" in symptoms.lower()
        or "errimageneverpull" in symptoms.lower()
    ):
        hypotheses.append({
            "cause": "ImagePullBackOff: Container image cannot be pulled. Check image name, registry, or credentials.",
            "confidence": 0.9,
            "tests": ["kubectl describe pod <pod>", "Verify image name in deployment", "Check registry access"]
        })
    
    if "crashloop" in symptoms.lower() or "exit code" in symptoms.lower():
        hypotheses.append({
            "cause": "CrashLoopBackOff: Container is crashing and restarting. Check application logs and exit reason.",
            "confidence": 0.9,
            "tests": ["kubectl logs <pod>", "kubectl logs <pod> --previous", "Check liveness/readiness probes", "Fix application startup logic"]
        })
    
    # Scheduling issues
    if "pending" in symptoms.lower() or "failedscheduling" in symptoms.lower():
        hypotheses.append({
            "cause": "FailedScheduling: Pod cannot be scheduled due to insufficient resources or affinity constraints.",
            "confidence": 0.85,
            "tests": ["kubectl top nodes", "kubectl describe pod <pod>", "Add nodes or reduce resource requests"]
        })
    
    # Generic fallback
    if not hypotheses:
        hypotheses.append({
            "cause": "Unidentified pod failure — gather more evidence from logs and pod description.",
            "confidence": 0.5,
            "tests": ["kubectl describe pod <pod>", "kubectl logs <pod>", "kubectl get events"]
        })
    
    return {"hypotheses": hypotheses}


# ─────────────────────────────────────────────────────────────────────────────
# Fix Generator Tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def generate_fix(hypothesis: str, manifest_yaml: Optional[str] = None) -> dict:
    """Generate remediation commands and patches for a root cause.

    Pass the "cause" string from generate_hypotheses output.

    Example:
      hypothesis="Out-of-Memory (OOMKilled): Pod memory request too low or memory leak in app."
      → returns {"commands": ["kubectl set resources...", "kubectl rollout restart..."], ...}

    IMPORTANT: Extract the "commands" field from the returned dict and pass it to verify_fix.

    Args:
        hypothesis: the root-cause string from generate_hypotheses (the "cause" field).
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
def verify_fix(fix_commands, cluster_health_check: str = "cluster healthy") -> dict:
    """Evaluate whether proposed fix commands are likely to succeed.

    Pass the "commands" list from generate_fix output.

    Args:
        fix_commands: either a list of kubectl commands OR a string command.
        cluster_health_check: brief description of current cluster state
            (default: "cluster healthy").

    Returns:
        {"likely_effective": bool, "missing_steps": [str],
         "risks": [str], "recommendation": str}
    """
    # Normalize: convert string to list if needed
    if isinstance(fix_commands, str):
        if not fix_commands or fix_commands.startswith("<"):
            # Placeholder or empty string
            return {
                "likely_effective": False,
                "missing_steps": ["Actual fix commands required"],
                "risks": ["No valid commands provided"],
                "recommendation": "Extract the 'commands' field from generate_fix output"
            }
        fix_commands = [cmd.strip() for cmd in fix_commands.split(";") if cmd.strip()]
    elif not isinstance(fix_commands, list):
        fix_commands = []

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
