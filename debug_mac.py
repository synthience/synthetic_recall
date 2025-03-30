#!/usr/bin/env python

import requests
import json
import time

# Helper function to make API calls
def api_call(endpoint, payload=None, method="POST"):
    url = f"http://localhost:8002{endpoint}"
    print(f"Calling {url} with payload: {json.dumps(payload)}")
    if method == "POST":
        response = requests.post(url, json=payload)
    else:
        response = requests.get(url)
    
    print(f"Status code: {response.status_code}")
    try:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        return result, response.status_code
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {"error": str(e)}, response.status_code

# Helper to check container status
def check_container_logs():
    print("\n=== Checking container logs (last 10 lines) ===")
    print("To see more logs, run: docker-compose logs -f context-cascade-orchestrator")
    import subprocess
    result = subprocess.run(
        ["docker-compose", "logs", "--tail=10", "context-cascade-orchestrator"], 
        capture_output=True, text=True)
    print(result.stdout)

# Main debugging sequence
def main():
    print("=== Starting MAC Variant Debug Sequence ===")
    
    # 1. Reset to NONE variant first (safe state)
    result, status = api_call("/set_variant", {"variant": "NONE", "reset_neural_memory": True})
    if status != 200:
        print("Failed to reset to NONE variant")
        return
    
    # 2. Switch to MAC variant
    print("\n=== Switching to MAC variant ===")
    result, status = api_call("/set_variant", {"variant": "MAC", "reset_neural_memory": False})
    if status != 200 or not result.get("success"):
        print("Failed to switch to MAC variant")
        return
        
    # 3. Process a memory with MAC variant
    print("\n=== Processing memory with MAC variant ===")
    test_content = "Test memory for MAC variant debugging"
    test_embedding = [0.1] * 768  # Simple test embedding
    
    result, status = api_call("/process_memory", {
        "content": test_content,
        "embedding": test_embedding
    })
    
    # 4. Extract and examine variant_output
    print("\n=== Examining variant_output ===")
    variant_output = result.get("variant_output", {})
    print(f"variant_output: {json.dumps(variant_output, indent=2)}")
    
    if "mac" in variant_output:
        print("SUCCESS: MAC metrics found in variant_output")
    else:
        print("FAILURE: MAC metrics missing from variant_output")
        check_container_logs()

if __name__ == "__main__":
    main()
