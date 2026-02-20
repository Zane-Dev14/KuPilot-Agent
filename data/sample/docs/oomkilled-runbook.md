# OOMKilled Troubleshooting Runbook

## Symptoms
- Pod status shows `OOMKilled` (exit code 137)
- `kubectl describe pod` shows `Reason: OOMKilled`
- Container restarts with increasing back-off

## Diagnosis Steps

### Step 1: Confirm OOM
```bash
kubectl describe pod <pod-name> -n <namespace>
# Look for: "Last State: Terminated, Reason: OOMKilled, Exit Code: 137"
```

### Step 2: Check Resource Limits
```bash
kubectl get pod <pod-name> -o jsonpath='{.spec.containers[*].resources}'
```

### Step 3: Check Actual Usage
```bash
kubectl top pod <pod-name> -n <namespace>
```

## Common Causes

### 1. Memory Limit Too Low
The application legitimately needs more memory than allocated.

**Fix:**
- Increase `resources.limits.memory` in the deployment spec
- Right-size using metrics: `kubectl top pod` over time
- Consider using VPA (Vertical Pod Autoscaler) for automatic right-sizing

### 2. Memory Leak
Application memory grows unbounded over time.

**Fix:**
- Profile the application (heap dumps, memory profilers)
- Check for unclosed connections, growing caches, or event listener leaks
- Implement graceful shutdown and connection pooling

### 3. JVM Heap Not Aligned with Container Limit
Java applications may not respect container memory limits.

**Fix:**
- Set `-XX:MaxRAMPercentage=75.0` to limit JVM heap to 75% of container memory
- Always set both `-Xmx` and container memory limits
- Leave headroom for non-heap memory (metaspace, threads, NIO buffers)

## Prevention
- Set resource requests AND limits for all containers
- Use `LimitRange` to enforce defaults at namespace level
- Monitor memory trends with Prometheus/Grafana
- Set up alerts for containers approaching memory limits (>80%)
