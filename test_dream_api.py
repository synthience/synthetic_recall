#!/usr/bin/env python3
"""
Simple test script for connecting to the Dream API server running in the Docker container.
Tests both basic functionality and advanced features like HypersphereManager and convergence mechanisms.
"""

import asyncio
import aiohttp
import json
import sys
import socket
import time
from typing import Dict, Any, List, Optional

# Default URL to try first
BASE_URL = "http://localhost:8080"

# Check if a port was specified as a command-line argument
if len(sys.argv) > 1:
    try:
        port = int(sys.argv[1])
        BASE_URL = f"http://localhost:{port}"
        print(f"Using specified port: {port}")
    except ValueError:
        print(f"Invalid port number: {sys.argv[1]}. Using default: 8080")

# API endpoints based on router configuration
DREAM_API_URL = f"{BASE_URL}/api/dream"

# Test data
TEST_MEMORY = "This is a test memory for the Lucidia system, designed to verify the HypersphereDispatcher integration."
TEST_MEMORIES = [
    "Hypersphere technology provides efficient embedding generation for large batches of text.",
    "The convergence mechanisms in Lucidia's reflection engine prevent infinite loops during report refinement.",
    "Lucidia uses a multi-stage retrieval process that blends semantic and graph-based approaches.",
    "Self-critique processes benefit from diminishing returns logic to stabilize confidence evaluations."
]

def check_port(host, port):
    """Check if a port is open on the given host."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)  # 2 second timeout
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

async def test_basic_connectivity():
    """Test basic connection to the Dream API server."""
    print("\n===== TESTING BASIC CONNECTIVITY =====")
    
    # Parse the port from the URL
    port = int(BASE_URL.split(':')[-1])
    if not check_port('localhost', port):
        print(f"Port {port} is not open on localhost. The server might not be running on this port.")
        return False
    else:
        print(f"Port {port} is open on localhost. Attempting to connect to the API...")
    
    async with aiohttp.ClientSession() as session:
        # Test the health endpoint
        health_url = f"{DREAM_API_URL}/health"
        try:
            async with session.get(health_url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check successful: {data}")
                    return True
                else:
                    print(f"❌ Failed health check. Status: {response.status}")
                    body = await response.text()
                    print(f"Response: {body}")
                    return False
        except Exception as e:
            print(f"❌ Error connecting to Dream API health endpoint: {e}")
            return False

async def test_dream_api_status():
    """Test the dream status endpoint to verify API is operational."""
    print("\n===== TESTING DREAM API STATUS =====")
    
    async with aiohttp.ClientSession() as session:
        dream_status_url = f"{DREAM_API_URL}/status"
        try:
            async with session.get(dream_status_url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Dream status check successful: {data}")
                    return data
                else:
                    print(f"❌ Failed dream status check. Status: {response.status}")
                    body = await response.text()
                    print(f"Response: {body}")
                    return None
        except Exception as e:
            print(f"❌ Error connecting to Dream API status endpoint: {e}")
            return None

async def test_hypersphere_manager():
    """Test the HypersphereManager integration by processing embedding requests."""
    print("\n===== TESTING HYPERSPHERE MANAGER INTEGRATION =====")
    
    async with aiohttp.ClientSession() as session:
        # Test batch embedding processing
        embedding_url = f"{DREAM_API_URL}/test/batch_embedding"
        embedding_data = {
            "texts": TEST_MEMORIES,
            "use_hypersphere": True
        }
        
        try:
            async with session.post(embedding_url, json=embedding_data) as response:
                if response.status == 200:
                    data = await response.json()
                    if "embeddings" in data and len(data["embeddings"]) == len(TEST_MEMORIES):
                        print(f"✅ Batch embedding processing successful: {len(data['embeddings'])} embeddings generated")
                        print(f"   First embedding dimensions: {len(data['embeddings'][0])}")
                        return data
                    else:
                        print(f"❌ Received invalid batch embedding response: {data}")
                        return None
                else:
                    print(f"❌ Failed to process batch embeddings. Status: {response.status}")
                    body = await response.text()
                    print(f"Response: {body}")
                    return None
        except Exception as e:
            print(f"❌ Error connecting to batch embedding endpoint: {e}")
            return None

async def test_similarity_search():
    """Test the similarity search functionality."""
    print("\n===== TESTING SIMILARITY SEARCH WITH HYPERSPHERE =====")
    
    async with aiohttp.ClientSession() as session:
        # First add test memories to the system
        add_memories_url = f"{DREAM_API_URL}/test/add_test_memories"
        memories_data = {
            "memories": [{
                "content": memory,
                "importance": 0.8,
                "metadata": {"test": True, "created_at": time.time()}
            } for memory in TEST_MEMORIES]
        }
        
        try:
            # Step 1: Add test memories
            async with session.post(add_memories_url, json=memories_data) as response:
                if response.status == 200:
                    data = await response.json()
                    memory_ids = data.get("memory_ids", [])
                    print(f"✅ Successfully added {len(memory_ids)} test memories")
                else:
                    print(f"❌ Failed to add test memories. Status: {response.status}")
                    return None
            
            # Step 2: Perform similarity search
            search_url = f"{DREAM_API_URL}/test/similarity_search"
            search_data = {
                "query": "How does Lucidia handle embedding generation?",
                "top_k": 2,
                "use_hypersphere": True
            }
            
            async with session.post(search_url, json=search_data) as response:
                if response.status == 200:
                    data = await response.json()
                    if "results" in data and len(data["results"]) > 0:
                        print(f"✅ Similarity search successful: {len(data['results'])} results found")
                        for i, result in enumerate(data["results"]):
                            print(f"   Result {i+1}: Score={result['score']:.4f}, Content={result['content'][:50]}...")
                        return data
                    else:
                        print(f"❌ Similarity search returned no results: {data}")
                        return None
                else:
                    print(f"❌ Failed to perform similarity search. Status: {response.status}")
                    body = await response.text()
                    print(f"Response: {body}")
                    return None
        except Exception as e:
            print(f"❌ Error testing similarity search: {e}")
            return None

async def test_convergence_mechanisms():
    """Test the convergence mechanisms in the self-critique process."""
    print("\n===== TESTING SELF-CRITIQUE CONVERGENCE MECHANISMS =====")
    
    async with aiohttp.ClientSession() as session:
        # Create a test dream report with initial fragments
        create_report_url = f"{DREAM_API_URL}/test/create_test_report"
        report_data = {
            "title": "Test Convergence Report",
            "fragments": [
                {"content": "Convergence mechanisms prevent infinite loops in self-critique processes", 
                 "type": "insight", "confidence": 0.7},
                {"content": "How effective are diminishing returns in stabilizing confidence evaluations?", 
                 "type": "question", "confidence": 0.5},
                {"content": "Oscillation detection requires at least 4 data points to identify patterns", 
                 "type": "hypothesis", "confidence": 0.6}
            ]
        }
        
        try:
            # Step 1: Create test report
            async with session.post(create_report_url, json=report_data) as response:
                if response.status == 200:
                    data = await response.json()
                    report_id = data.get("report_id")
                    if not report_id:
                        print(f"❌ Failed to get report ID from response: {data}")
                        return None
                    print(f"✅ Created test report with ID: {report_id}")
                else:
                    print(f"❌ Failed to create test report. Status: {response.status}")
                    body = await response.text()
                    print(f"Response: {body}")
                    return None
            
            # Step 2: Perform multiple refinements to test convergence
            refine_url = f"{DREAM_API_URL}/test/refine_report"
            refinement_results = []
            
            for i in range(5):  # Test 5 refinement iterations
                refine_data = {"report_id": report_id}
                async with session.post(refine_url, json=refine_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        refinement_results.append(result)
                        refinement_count = result.get("refinement_count", "unknown")
                        confidence = result.get("confidence", "unknown")
                        status = result.get("status", "unknown")
                        
                        print(f"✅ Refinement {i+1}: Status={status}, Refinement Count={refinement_count}, Confidence={confidence}")
                        
                        # If the system skipped refinement due to convergence, we've verified it's working
                        if status == "skipped":
                            print(f"✅ Convergence mechanism successfully prevented further refinement: {result.get('reason')}")
                            break
                    else:
                        print(f"❌ Failed to refine report (iteration {i+1}). Status: {response.status}")
                        body = await response.text()
                        print(f"Response: {body}")
            
            # Step 3: Get the final report to check convergence stats
            get_report_url = f"{DREAM_API_URL}/test/get_report?report_id={report_id}"
            async with session.get(get_report_url) as response:
                if response.status == 200:
                    report = await response.json()
                    print(f"\nFinal Report Details:")
                    print(f"  Refinement Count: {report.get('refinement_count')}")
                    print(f"  Confidence History: {report.get('confidence_history')}")
                    print(f"  Self-Assessment: {report.get('analysis', {}).get('self_assessment')}")
                    return {"report": report, "refinements": refinement_results}
                else:
                    print(f"❌ Failed to retrieve final report. Status: {response.status}")
                    return {"refinements": refinement_results}
                    
        except Exception as e:
            print(f"❌ Error testing convergence mechanisms: {e}")
            return None

async def test_dream_api():
    """Run all tests for the Dream API functionality."""
    print("\n========= DREAM API TEST SUITE =========")
    print(f"Testing connection to Dream API at {DREAM_API_URL}")
    
    # First check basic connectivity
    connected = await test_basic_connectivity()
    if not connected:
        print("\n❌ Failed to connect to the Dream API. Please ensure the server is running.")
        return
    
    # Run component tests
    api_status = await test_dream_api_status()
    hypersphere_results = await test_hypersphere_manager()
    similarity_results = await test_similarity_search()
    convergence_results = await test_convergence_mechanisms()
    
    # Display summary
    print("\n========= TEST SUMMARY ==========")
    print(f"✅ API Status: {'Online' if api_status else 'Error'}")
    print(f"✅ HypersphereManager: {'Functional' if hypersphere_results else 'Error or Not Available'}")
    print(f"✅ Similarity Search: {'Functional' if similarity_results else 'Error or Not Available'}")
    print(f"✅ Convergence Mechanisms: {'Functional' if convergence_results else 'Error or Not Available'}")
    print("\n=================================")
    print("\nTest suite completed.")

if __name__ == "__main__":
    asyncio.run(test_dream_api())
