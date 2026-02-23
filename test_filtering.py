#!/usr/bin/env python3
"""Test that agent properly filters findings by symptom."""

import logging
logging.getLogger("langchain_milvus").setLevel(logging.ERROR)

from src.agents import diagnose

queries = [
    ("how many pods are running", "SIMPLE INFO", 1),
    ("Is everything healthy in the cluster?", "HEALTH CHECK", 2),
    ("Pod is OOMKilled. What should I do to fix it safely?", "DIAGNOSTIC: OOM", 1),
    ("Services are not connecting to each other. Network is blocked.", "DIAGNOSTIC: Network", 1),
    ("Latency suddenly increased across services.", "DIAGNOSTIC: Latency", 1),
    ("Why is my pod crashing?", "DIAGNOSTIC: Crash", 1),
]

print("=" * 80)
print("FILTERED FINDINGS TEST")
print("=" * 80)

for query, query_type, max_steps in queries:
    print(f"\n[{query_type}] Query: {query}")
    print("-" * 80)
    try:
        result = diagnose(query, max_steps=max_steps)
        response = result.get("response", "")
        
        # Extract key parts
        lines = response.split("\n")
        
        # Look for symptom, findings, ignored, root cause
        showing = False
        for line in lines:
            if "Actual Question" in line or "Relevant" in line or "Ignored" in line or "Root Cause" in line:
                showing = True
            if showing:
                print(line)
                if "Root Cause" in line or "Confidence" in line:
                    break
        
        if not showing:
            print(response[:300])
        
    except Exception as e:
        print(f"ERROR: {e}")
    
    print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
