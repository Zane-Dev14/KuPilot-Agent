# CrashLoopBackOff Troubleshooting Runbook

## Symptoms
- Pod repeatedly restarts (restart count increasing)
- `kubectl get pods` shows `CrashLoopBackOff` status
- Back-off delay increases exponentially (10s, 20s, 40s, ... up to 5min)

## Common Root Causes

### 1. Application Error on Startup
The container starts but immediately exits with a non-zero exit code.

**Diagnosis:**
```bash
kubectl logs <pod-name> --previous
kubectl describe pod <pod-name>
```

**Common causes:**
- Missing environment variables or config files
- Database connection failures
- Permission errors
- Missing dependencies

**Fix:**
- Check `kubectl logs` for the specific error
- Verify all ConfigMaps and Secrets are mounted
- Ensure dependent services (DB, cache) are running

### 2. Liveness Probe Failure
The container starts but fails liveness checks, causing kubelet to restart it.

**Diagnosis:**
```bash
kubectl describe pod <pod-name> | grep -A 5 "Liveness"
```

**Fix:**
- Increase `initialDelaySeconds` if app needs more startup time
- Verify the health endpoint returns 200
- Check resource limits — probe may timeout under CPU throttling

### 3. OOMKilled
Container exceeds memory limits and is killed by the OOM killer.

**Diagnosis:**
```bash
kubectl describe pod <pod-name> | grep -i "oom\|killed\|memory"
```

**Fix:**
- Increase memory limits in the pod spec
- Profile application memory usage
- Fix memory leaks in application code

## Escalation
If the issue persists after troubleshooting, escalate to the platform team with:
1. Pod describe output
2. Container logs (current and previous)
3. Events for the namespace
4. Resource utilization metrics
