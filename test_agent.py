#!/usr/bin/env python3
"""Test the agent with various queries."""

import sys
import logging

# Suppress Milvus warnings
logging.getLogger("langchain_milvus").setLevel(logging.ERROR)

from src.agents import diagnose

queries = [
    "One of my pods keeps restarting but others are fine. Can you investigate?",
    "I think this is a networking issue. Pod is failing to start.",
    "Pod is OOMKilled. What should I do to fix it safely?",
    # EDGE CASE: Image error - requires tool retry with different approach
    "has image error happened to me yet? where? show me",
]

for query in queries:
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")
    try:
        # Increase max_steps to allow retries and different approaches
        result = diagnose(query, max_steps=11)
        steps = result.get("steps", [])
        print(f"Steps: {len(steps)} | Response length: {len(result.get('response', ''))} chars\n")
        
        # Show step summary
        for step in steps[:10]:
            t = step.get("type", "?")
            n = step.get("name", "final")
            e = step.get("error", "")
            cached = " [CACHED]" if step.get("cached") else ""
            llm_ms = step.get("llm_ms", 0)
            tool_ms = step.get("tool_ms", 0)
            if e:
                print(f"  [{t:12}] {n:25} ERROR: {e[:45]}{cached}")
            elif t == "tool_call":
                print(f"  [{t:12}] {n:25} {tool_ms:5d}ms{cached}")
            else:
                print(f"  [{t:12}] {n:25}{cached}")
        
        # Show response
        print("\nResponse:")
        response = result.get("response", "")
        if len(response) > 1000:
            print(f"  {response[:1000]}\n  ... (truncated)")
        else:
            print(f"  {response}")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)[:200]}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("Test complete")
print(f"{'='*70}")
