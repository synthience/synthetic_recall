import requests
import json
import numpy as np
import os
import time
import sys

# Base URL for the trainer-server
BASE_URL = os.environ.get('TRAINER_URL', 'http://localhost:8001')
SAVE_LOAD_PATH = os.environ.get('TRAINER_SAVE_PATH', '/tmp/neural_memory_test_state.json') # Use /tmp for safety

# ---- Configuration for Neural Memory ----
# Ensure these dimensions match what your application expects
# and what the NeuralMemoryModule defaults to or is configured with.
TEST_CONFIG = {
    "input_dim": 768,
    "key_dim": 128,
    "value_dim": 768, # Often same as input_dim
    "query_dim": 128, # Often same as key_dim
    "memory_hidden_dims": [512], # Example: one hidden layer of 512
    "outer_learning_rate": 1e-4
    # Add other config keys if needed, otherwise defaults from NeuralMemoryConfig will be used
}
# ---- End Configuration ----


# --- Helper Functions ---
def _make_request(method, endpoint, **kwargs):
    """Helper to make requests and handle basic errors."""
    url = f'{BASE_URL}{endpoint}'
    try:
        response = requests.request(method, url, **kwargs, timeout=30) # Added timeout
        print(f"{method} {endpoint}: {response.status_code}")
        print(f"Response Text: {response.text[:500]}") # Print beginning of response text
        try:
            response_json = response.json()
            print("Response JSON:")
            print(json.dumps(response_json, indent=2))
            return response_json, response.status_code
        except json.JSONDecodeError:
            print("Response was not valid JSON")
            return response.text, response.status_code # Return raw text on decode error
    except requests.exceptions.RequestException as e:
        print(f"Error calling {endpoint}: {str(e)}")
        return {"error": str(e)}, 500

def _generate_embedding(dim):
    """Generates a random numpy embedding and converts to list."""
    return np.random.rand(dim).astype(np.float32).tolist()

# --- Test Functions ---

def test_health():
    print("\n" + "="*50)
    print("Testing Health Endpoint")
    print("="*50)
    return _make_request('GET', '/health')

def test_status(expected_status="not initialized"):
    print("\n" + "="*50)
    print(f"Testing Status Endpoint (Expecting: {expected_status})")
    print("="*50)
    data, code = _make_request('GET', '/status')
    if code == 200 and isinstance(data, dict):
        print(f"Status: {data.get('status')}")
        if data.get('config'):
             print("Config found in status.")
        assert expected_status in data.get('status', '').lower(), f"Expected status '{expected_status}', got '{data.get('status')}'"
    else:
         assert False, f"Status check failed with code {code}"
    return data, code

def test_init(load_path=None):
    print("\n" + "="*50)
    print(f"Testing Initialization (Load Path: {load_path})")
    print("="*50)
    payload = {
        "config": TEST_CONFIG,
        "memory_core_url": os.environ.get("MEMORY_CORE_URL", "http://localhost:5020") # Example URL
    }
    if load_path:
        payload["load_path"] = load_path

    data, code = _make_request('POST', '/init', json=payload)
    assert code == 200, f"Init failed with status {code}"
    assert "message" in data, "Init response missing 'message'"
    assert "config" in data, "Init response missing 'config'"
    # Verify some config values returned match sent values
    assert data["config"]["input_dim"] == TEST_CONFIG["input_dim"]
    assert data["config"]["key_dim"] == TEST_CONFIG["key_dim"]
    return data, code

def test_update_memory():
    print("\n" + "="*50)
    print("Testing Update Memory Endpoint")
    print("="*50)
    input_emb = _generate_embedding(TEST_CONFIG["input_dim"])
    payload = {"input_embedding": input_emb}
    data, code = _make_request('POST', '/update_memory', json=payload)
    assert code == 200, f"Update memory failed with status {code}"
    assert data.get("status") == "success", "Update memory status was not 'success'"
    # Check if optional metrics are returned
    print(f"Update Loss (Surprise Proxy): {data.get('loss')}")
    print(f"Update Grad Norm (Surprise Proxy): {data.get('grad_norm')}")
    return data, code

def test_retrieve():
    print("\n" + "="*50)
    print("Testing Retrieve Endpoint")
    print("="*50)
    query_emb = _generate_embedding(TEST_CONFIG["query_dim"])
    payload = {"query_embedding": query_emb}
    data, code = _make_request('POST', '/retrieve', json=payload)
    assert code == 200, f"Retrieve memory failed with status {code}"
    assert "retrieved_embedding" in data, "Retrieve response missing 'retrieved_embedding'"
    assert len(data["retrieved_embedding"]) == TEST_CONFIG["value_dim"], \
        f"Retrieved embedding has wrong dimension ({len(data['retrieved_embedding'])} vs {TEST_CONFIG['value_dim']})"
    return data, code

def test_analyze_surprise():
    print("\n" + "="*50)
    print("Testing Analyze Surprise Endpoint")
    print("="*50)
    pred_emb = _generate_embedding(TEST_CONFIG["input_dim"])
    actual_emb = _generate_embedding(TEST_CONFIG["input_dim"])
    # Make them slightly different
    actual_emb[0] += 0.1
    payload = {
        "predicted_embedding": pred_emb,
        "actual_embedding": actual_emb
    }
    data, code = _make_request('POST', '/analyze_surprise', json=payload)
    assert code == 200, f"Analyze surprise failed with status {code}"
    assert "surprise" in data, "Analyze surprise response missing 'surprise'"
    assert "is_surprising" in data, "Analyze surprise response missing 'is_surprising'"
    assert "quickrecal_boost" in data, "Analyze surprise response missing 'quickrecal_boost'"
    return data, code

def test_save_state():
    print("\n" + "="*50)
    print(f"Testing Save State Endpoint (Path: {SAVE_LOAD_PATH})")
    print("="*50)
    payload = {"path": SAVE_LOAD_PATH}
    data, code = _make_request('POST', '/save', json=payload)
    assert code == 200, f"Save state failed with status {code}"
    assert "message" in data, "Save state response missing 'message'"
    # Check if file actually exists
    # Handle potential "file://" prefix if added by save_state
    actual_path = SAVE_LOAD_PATH[7:] if SAVE_LOAD_PATH.startswith("file://") else SAVE_LOAD_PATH
    assert os.path.exists(actual_path), f"State file not found at {actual_path} after save"
    return data, code

def test_load_state():
    print("\n" + "="*50)
    print(f"Testing Load State Endpoint (Path: {SAVE_LOAD_PATH})")
    print("="*50)
    payload = {"path": SAVE_LOAD_PATH}
    # Ensure file exists before loading
    actual_path = SAVE_LOAD_PATH[7:] if SAVE_LOAD_PATH.startswith("file://") else SAVE_LOAD_PATH
    if not os.path.exists(actual_path):
         print(f"WARNING: State file {actual_path} does not exist. Skipping load test.")
         pytest.skip(f"State file {actual_path} not found for loading.")
         return None, 404 # Simulate file not found skip

    data, code = _make_request('POST', '/load', json=payload)
    assert code == 200, f"Load state failed with status {code}"
    assert "message" in data, "Load state response missing 'message'"
    return data, code

def test_outer_training():
    print("\n" + "="*50)
    print("Testing Outer Training Endpoint")
    print("="*50)
    seq_len = 5
    input_seq = [_generate_embedding(TEST_CONFIG["input_dim"]) for _ in range(seq_len)]
    # Target sequence should match value_dim
    target_seq = [_generate_embedding(TEST_CONFIG["value_dim"]) for _ in range(seq_len)]
    payload = {
        "input_sequence": input_seq,
        "target_sequence": target_seq
    }
    data, code = _make_request('POST', '/train_outer', json=payload)
    assert code == 200, f"Outer training failed with status {code}"
    assert "average_loss" in data, "Outer training response missing 'average_loss'"
    assert isinstance(data["average_loss"], float), "'average_loss' should be a float"
    return data, code


# --- Main Test Sequence ---
if __name__ == "__main__":
    results = {}
    print(f"\n===== Testing Neural Memory API at {BASE_URL} =====\n")

    results["health_initial"] = test_health()[1] == 200
    results["status_initial"] = test_status(expected_status="not initialized")[1] == 200

    # Initialize
    init_data, init_code = test_init()
    results["init"] = init_code == 200

    if results["init"]:
        results["status_after_init"] = test_status(expected_status="initialized")[1] == 200
        # Perform operations only if initialized
        results["update_memory"] = test_update_memory()[1] == 200
        results["retrieve"] = test_retrieve()[1] == 200
        results["analyze_surprise"] = test_analyze_surprise()[1] == 200
        results["outer_training"] = test_outer_training()[1] == 200

        # Save/Load Test
        results["save_state"] = test_save_state()[1] == 200
        if results["save_state"]:
             # Re-initialize before loading to test loading into fresh instance
             reinit_data, reinit_code = test_init()
             if reinit_code == 200:
                 results["load_state"] = test_load_state()[1] == 200
                 # Verify status after load
                 results["status_after_load"] = test_status(expected_status="initialized")[1] == 200
             else:
                 print("Failed to re-initialize before load test.")
                 results["load_state"] = False
                 results["status_after_load"] = False
             # Clean up saved file
             try:
                 actual_path = SAVE_LOAD_PATH[7:] if SAVE_LOAD_PATH.startswith("file://") else SAVE_LOAD_PATH
                 if os.path.exists(actual_path): os.remove(actual_path)
             except Exception as e: print(f"Error cleaning up save file: {e}")
        else:
             results["load_state"] = False # Cannot load if save failed
             results["status_after_load"] = False


    else:
        print("Skipping further tests due to initialization failure.")
        results["update_memory"] = "skipped"
        results["retrieve"] = "skipped"
        results["analyze_surprise"] = "skipped"
        results["outer_training"] = "skipped"
        results["save_state"] = "skipped"
        results["load_state"] = "skipped"
        results["status_after_load"] = "skipped"


    print("\n" + "="*50)
    print("Final Test Summary")
    print("="*50)
    passed_count = 0
    failed_count = 0
    skipped_count = 0
    for name, result in results.items():
        if result is True:
            status = "✅ PASSED"
            passed_count += 1
        elif result is False:
            status = "❌ FAILED"
            failed_count += 1
        else:
            status = "⚠️ SKIPPED"
            skipped_count += 1
        print(f"- {name}: {status}")

    print("-" * 50)
    print(f"Total Passed: {passed_count}")
    print(f"Total Failed: {failed_count}")
    print(f"Total Skipped: {skipped_count}")
    print("="*50)

    # Exit with non-zero code if any tests failed
    sys.exit(1 if failed_count > 0 else 0)