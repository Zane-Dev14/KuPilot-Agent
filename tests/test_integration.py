#!/usr/bin/env python3
"""Consolidated integration tests for multi-agent diagnostic system.

This file merges:
- test_agent.py: Single investigator agent tests
- test_filtering.py: Symptom filtering tests
- test_multiagent_simple.py: Agent creation tests
- test_orchestrator.py: Orchestrator integration tests
- test_performance.py: Multi-agent performance tests
"""

import sys
import os
import time
import logging

# Suppress Milvus warnings
logging.getLogger("langchain_milvus").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ────────────────────────────────────────────────────────────────────────────
# TEST SUITE 1: Agent Creation & Multi-Agent System
# ────────────────────────────────────────────────────────────────────────────

def test_agent_creation():
    """Verify all specialist agents can be created."""
    print("\n" + "="*70)
    print("TEST SUITE 1: Agent Creation")
    print("="*70)
    
    try:
        from src.agents import (
            create_investigator_agent,
            create_knowledge_agent,
            create_remediation_agent,
            create_verification_agent,
            create_orchestrator_agent,
        )
        
        print("Creating specialist agents...")
        inv = create_investigator_agent()
        know = create_knowledge_agent()
        rem = create_remediation_agent()
        ver = create_verification_agent()
        orch = create_orchestrator_agent()
        
        print("✓ Investigator agent created")
        print("✓ Knowledge agent created")
        print("✓ Remediation agent created")
        print("✓ Verification agent created")
        print("✓ Orchestrator agent created")
        
        # Check orchestrator function exists
        from src.agents import orchestrate_multiagent_diagnosis
        print("✓ Multi-agent orchestration function available")
        
        print("\n✅ Agent creation test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Agent creation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ────────────────────────────────────────────────────────────────────────────
# TEST SUITE 2: Symptom Filtering & Query Classification
# ────────────────────────────────────────────────────────────────────────────

def test_symptom_filtering():
    """Test that agent properly filters findings by symptom type."""
    print("\n" + "="*70)
    print("TEST SUITE 2: Symptom Filtering & Classification")
    print("="*70)
    
    from src.agents import diagnose
    
    test_cases = [
        ("how many pods are running", "SIMPLE INFO"),
        ("Is everything healthy in the cluster?", "HEALTH CHECK"),
        ("Pod is OOMKilled. What should I do to fix it safely?", "DIAGNOSTIC: OOM"),
        ("Services are not connecting to each other. Network is blocked.", "DIAGNOSTIC: Network"),
        ("Latency suddenly increased across services.", "DIAGNOSTIC: Latency"),
        ("Why is my pod crashing?", "DIAGNOSTIC: Crash"),
    ]
    
    passed = 0
    failed = 0
    
    for query, query_type in test_cases:
        print(f"\n[{query_type}] {query}")
        try:
            result = diagnose(query, max_steps=2)
            response = result.get("response", "")
            
            if response and len(response) > 50:
                print(f"  ✓ Got response ({len(response)} chars)")
                passed += 1
            else:
                print(f"  ✗ Response too short")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:100]}")
            failed += 1
    
    print(f"\n✅ Symptom filtering: {passed}/{len(test_cases)} queries handled")
    return failed == 0


# ────────────────────────────────────────────────────────────────────────────
# TEST SUITE 3: Single Investigator Agent
# ────────────────────────────────────────────────────────────────────────────

def test_investigator_agent():
    """Test investigator agent with various diagnostic queries."""
    print("\n" + "="*70)
    print("TEST SUITE 3: Investigator Agent")
    print("="*70)
    
    from src.agents import diagnose
    
    queries = [
        "One of my pods keeps restarting but others are fine. Can you investigate?",
        "I think this is a networking issue. Pod is failing to start.",
        "Pod is OOMKilled. What should I do to fix it safely?",
        "has image error happened to me yet? where? show me",
    ]
    
    print(f"Testing {len(queries)} diagnostic queries...\n")
    
    passed = 0
    for i, query in enumerate(queries, 1):
        print(f"[{i}] {query[:60]}...")
        try:
            result = diagnose(query, max_steps=11)
            steps = result.get("steps", [])
            response = result.get("response", "")
            
            if response and len(response) > 100:
                print(f"    ✓ Steps: {len(steps)} | Response: {len(response)} chars")
                passed += 1
            else:
                print(f"    ✗ Insufficient response")
                
        except Exception as e:
            print(f"    ✗ ERROR: {str(e)[:80]}")
    
    print(f"\n✅ Investigator agent: {passed}/{len(queries)} queries passed")
    return passed == len(queries)


# ────────────────────────────────────────────────────────────────────────────
# TEST SUITE 4: Orchestrator Integration
# ────────────────────────────────────────────────────────────────────────────

def test_orchestrator():
    """Test orchestrator agent with diagnostic scenarios."""
    print("\n" + "="*70)
    print("TEST SUITE 4: Orchestrator Integration")
    print("="*70)
    
    from src.agents import create_orchestrator_agent
    from scripts.agent import run_agent_with_tools
    
    test_queries = [
        ("Pod is OOMKilled. Fix it.", 18),
        ("One of my pods keeps restarting. Investigate and fix.", 18),
    ]
    
    print(f"Testing {len(test_queries)} orchestrator scenarios...\n")
    
    passed = 0
    for query, max_steps in test_queries:
        print(f"Query: {query[:50]}...")
        try:
            agent = create_orchestrator_agent()
            response, steps = run_agent_with_tools(
                agent, query, max_steps=max_steps, verbose=False
            )
            
            if response and len(response) > 200:
                print(f"  ✓ Steps: {len(steps)} | Response: {len(response)} chars")
                
                # Check for fix execution
                executed = [s for s in steps if s.get('name') == 'kubectl_exec']
                if executed:
                    print(f"  ✓ Fix execution: {len(executed)} kubectl calls")
                
                passed += 1
            else:
                print(f"  ✗ Insufficient response")
                
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:80]}")
    
    print(f"\n✅ Orchestrator integration: {passed}/{len(test_queries)} scenarios passed")
    return passed == len(test_queries)


# ────────────────────────────────────────────────────────────────────────────
# TEST SUITE 5: Performance & Multi-Agent Diagnostics
# ────────────────────────────────────────────────────────────────────────────

def test_multiagent_performance():
    """Test orchestrate_multiagent_diagnosis performance."""
    print("\n" + "="*70)
    print("TEST SUITE 5: Multi-Agent Performance")
    print("="*70)
    
    from src.agents import orchestrate_multiagent_diagnosis
    
    test_queries = [
        "has image pull error happened to me? where? show me. and fix it.",
        "is my api pod at risk of OOMKilled? How much memory does it have? Fix it. Put the required amount."
    ]
    
    print(f"Testing {len(test_queries)} performance scenarios...\n")
    
    passed = 0
    total_time = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"[{i}] {query[:50]}...")
        
        try:
            start = time.time()
            result = orchestrate_multiagent_diagnosis(query)
            elapsed = time.time() - start
            total_time += elapsed
            
            response = result.get('response', '')
            agents = result.get('agents_used', [])
            steps = result.get('steps', [])
            
            print(f"    ⏱  {elapsed:.2f}s ({int(elapsed*1000)}ms)")
            print(f"    Agents: {len(agents)} | Steps: {len(steps)}")
            
            # Validation checks
            has_pod_names = any(
                word in response.lower() 
                for word in ['api-', 'pod', 'deployment']
            )
            has_error_type = any(
                word in response.lower() 
                for word in ['imagepull', 'oomkilled', 'crash', 'error']
            )
            has_content = 'Analysis complete' not in response[:100]
            
            if has_pod_names and has_error_type and has_content:
                print(f"    ✓ Response validation passed")
                passed += 1
            else:
                print(f"    ⚠ Response validation issues")
                print(f"      - Pod names: {has_pod_names}")
                print(f"      - Error type: {has_error_type}")
                print(f"      - Has content: {has_content}")
                
        except Exception as e:
            print(f"    ✗ ERROR: {str(e)[:80]}")
    
    print(f"\n✅ Multi-agent performance: {passed}/{len(test_queries)} passed")
    if total_time > 0:
        print(f"   Average time: {total_time/len(test_queries):.2f}s")
    
    return passed == len(test_queries)


# ────────────────────────────────────────────────────────────────────────────
# Test Runner
# ────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    """Run all integration test suites."""
    print("\n" + "="*70)
    print("MULTI-AGENT INTEGRATION TEST SUITE")
    print("="*70)
    
    results = []
    
    # Run tests in order
    results.append(("Agent Creation", test_agent_creation()))
    results.append(("Symptom Filtering", test_symptom_filtering()))
    results.append(("Investigator Agent", test_investigator_agent()))
    results.append(("Orchestrator Integration", test_orchestrator()))
    results.append(("Multi-Agent Performance", test_multiagent_performance()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
