#!/usr/bin/env python3
# test_neural_memory_api.py

import requests
import numpy as np
import time
import matplotlib.pyplot as plt
import json
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API base URL
BASE_URL = "http://localhost:8001"  # Default port for trainer-server

# Configuration for testing
VALIDATION_DIMS = True  # Set to true to validate embedding dimensions
EMBEDDING_DIM = 768  # Match your configured input_dim
QUERY_DIM = 128    # Match your configured query_dim

# Test helper functions
def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")

def generate_random_embedding(dim: int) -> List[float]:
    """Generate a random normalized embedding."""
    embedding = np.random.normal(0, 1, dim).astype(np.float32)
    # Normalize to unit length
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()

def post_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Send a POST request to the API."""
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise exception for error status
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling {endpoint}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        raise

def get_request(endpoint: str) -> Dict[str, Any]:
    """Send a GET request to the API."""
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling {endpoint}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        raise

# Main test functions
def test_health():
    """Test the health endpoint."""
    print_section("Testing Health Endpoint")
    try:
        result = get_request("/health")
        print(f"Health check result: {result}")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def test_init():
    """Test initializing the Neural Memory module."""
    print_section("Testing Initialization")
    
    # Customize config as needed
    config = {
        "input_dim": EMBEDDING_DIM,
        "key_dim": 128,
        "value_dim": EMBEDDING_DIM,
        "query_dim": QUERY_DIM,
        "memory_hidden_dims": [512, 512],
        "inner_learning_rate": 0.01,
        "outer_learning_rate": 0.001,
        "momentum_decay": 0.9,
        "alpha_init": -2.0,  # Starting with low forgetting (sigmoid will be ~0.12)
        "theta_init": 0.0,   # Default momentum scaling
        "eta_init": 0.0,     # Default learning rate scaling
    }
    
    init_data = {
        "config": config,
        "memory_core_url": "http://memory-core:5020"
    }
    
    try:
        result = post_request("/init", init_data)
        print(f"Init response: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"Init failed: {e}")
        return False

def test_status():
    """Test getting the status of the Neural Memory module."""
    print_section("Testing Status Endpoint")
    try:
        result = get_request("/status")
        print(f"Status: {result['status']}")
        if 'config' in result and result['config']:
            # Print key configuration values
            config = result['config']
            print(f"Configuration settings:")
            print(f"- input_dim: {config.get('input_dim')}")
            print(f"- key_dim: {config.get('key_dim')}")
            print(f"- value_dim: {config.get('value_dim')}")
            print(f"- query_dim: {config.get('query_dim')}")
        return True
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return False

def test_update_memory():
    """Test updating the memory with a new embedding."""
    print_section("Testing Update Memory Endpoint")
    
    # Generate a random embedding
    embedding = generate_random_embedding(EMBEDDING_DIM)
    
    update_data = {
        "input_embedding": embedding
    }
    
    try:
        result = post_request("/update_memory", update_data)
        print(f"Update memory result: {result}")
        
        # Test boundary cases - malformed embeddings
        print("\nTesting with different embedding dimensions...")
        if VALIDATION_DIMS:
            try:
                # Test with wrong dimension
                wrong_dim_embedding = generate_random_embedding(EMBEDDING_DIM // 2)
                wrong_dim_data = {"input_embedding": wrong_dim_embedding}
                result = post_request("/update_memory", wrong_dim_data)
                print("Warning: API accepted wrong dimension embedding!")
            except Exception as e:
                print(f"✅ API correctly rejected wrong dimension embedding: {e}")
                
            # Test with NaN values
            try:
                nan_embedding = embedding.copy()
                nan_embedding[0] = float('nan')
                nan_data = {"input_embedding": nan_embedding}
                result = post_request("/update_memory", nan_data)
                print("Warning: API accepted NaN embedding!")
            except Exception as e:
                print(f"✅ API correctly rejected NaN embedding: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Update memory failed: {e}")
        return False

def test_retrieve():
    """Test retrieving from memory using a query embedding."""
    print_section("Testing Retrieve Endpoint")
    
    # Generate a random query embedding
    query_embedding = generate_random_embedding(QUERY_DIM)
    
    retrieve_data = {
        "query_embedding": query_embedding
    }
    
    try:
        result = post_request("/retrieve", retrieve_data)
        retrieved = result.get("retrieved_embedding", [])
        print(f"Retrieved embedding length: {len(retrieved)}")
        print(f"First few values: {retrieved[:5]}...")
        
        return True
    except Exception as e:
        logger.error(f"Retrieve failed: {e}")
        return False

def test_analyze_surprise():
    """Test analyzing surprise between predicted and actual embeddings."""
    print_section("Testing Analyze Surprise Endpoint")
    
    # Generate two random embeddings
    predicted = generate_random_embedding(EMBEDDING_DIM)
    actual = generate_random_embedding(EMBEDDING_DIM)
    
    surprise_data = {
        "predicted_embedding": predicted,
        "actual_embedding": actual
    }
    
    try:
        result = post_request("/analyze_surprise", surprise_data)
        print(f"Surprise analysis: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"Analyze surprise failed: {e}")
        return False

def test_memory_retention():
    """Test memory retention by storing and retrieving specific patterns."""
    print_section("Testing Memory Retention")
    
    # Create some distinctive test patterns
    patterns = [
        (f"pattern_{i}", generate_random_embedding(EMBEDDING_DIM)) 
        for i in range(5)
    ]
    
    # Store each pattern in memory
    for name, embedding in patterns:
        print(f"Storing {name}...")
        update_data = {"input_embedding": embedding}
        try:
            post_request("/update_memory", update_data)
        except Exception as e:
            logger.error(f"Failed to store {name}: {e}")
            return False
    
    # Query for each pattern - use the pattern itself as query
    # In a real scenario, you'd likely use a different query projection
    print("\nRetrieving patterns...")
    for name, embedding in patterns:
        try:
            # Project to query dimension if needed
            query = embedding[:QUERY_DIM] if QUERY_DIM < EMBEDDING_DIM else embedding
            retrieve_data = {"query_embedding": query}
            result = post_request("/retrieve", retrieve_data)
            retrieved = result.get("retrieved_embedding", [])
            
            # Calculate similarity with original
            retrieved_array = np.array(retrieved)
            original_array = np.array(embedding)
            similarity = np.dot(retrieved_array, original_array) / (
                np.linalg.norm(retrieved_array) * np.linalg.norm(original_array))
            
            print(f"{name} - cosine similarity with retrieved: {similarity:.4f}")
        except Exception as e:
            logger.error(f"Failed to retrieve {name}: {e}")
            return False
    
    return True

def test_train_outer():
    """Test the outer loop training with a simple sequence."""
    print_section("Testing Outer Loop Training")
    
    # Create a simple sequence of embeddings
    seq_length = 10
    sequence = [generate_random_embedding(EMBEDDING_DIM) for _ in range(seq_length)]
    
    # Use the same sequence as both input and target for simplicity
    train_data = {
        "input_sequence": sequence,
        "target_sequence": sequence  # In real use, these would typically differ
    }
    
    try:
        result = post_request("/train_outer", train_data)
        print(f"Outer loop training result: {result}")
        print(f"Average loss: {result.get('average_loss', 'N/A')}")
        return True
    except Exception as e:
        logger.error(f"Outer loop training failed: {e}")
        return False

def test_save_load():
    """Test saving and loading the memory state."""
    print_section("Testing Save/Load State")
    
    save_path = "/app/memory/saved_state.json"
    
    # Save current state
    save_data = {"path": save_path}
    try:
        save_result = post_request("/save", save_data)
        print(f"Save result: {save_result}")
        
        # Load the state back
        load_data = {"path": save_path}
        load_result = post_request("/load", load_data)
        print(f"Load result: {load_result}")
        
        return True
    except Exception as e:
        logger.error(f"Save/load failed: {e}")
        return False

# Run all tests
def run_tests():
    """Run all the test functions and return the success count."""
    tests = [
        ("Health Check", test_health),
        ("Initialization", test_init),
        ("Status Check", test_status),
        ("Update Memory", test_update_memory),
        ("Memory Retrieval", test_retrieve),
        ("Analyze Surprise", test_analyze_surprise),
        ("Memory Retention", test_memory_retention),
        ("Outer Loop Training", test_train_outer),
        ("Save/Load State", test_save_load)
    ]
    
    results = {}
    success_count = 0
    
    for name, test_func in tests:
        print(f"\n{'*' * 60}\nRunning test: {name}\n{'*' * 60}")
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            
            if success:
                status = "✅ Passed"
                success_count += 1
            else:
                status = "❌ Failed"
                
            results[name] = (success, duration)
            print(f"\nTest {name}: {status} ({duration:.2f}s)")
        except Exception as e:
            logger.error(f"Test {name} raised exception: {e}")
            results[name] = (False, 0)
            print(f"\nTest {name}: ❌ Failed (Exception)")
    
    # Print summary
    print_section("Test Results Summary")
    for name, (success, duration) in results.items():
        status = "✅ Passed" if success else "❌ Failed"
        print(f"{name}: {status} ({duration:.2f}s)")
    
    print(f"\nOverall: {success_count}/{len(tests)} tests passed")
    return success_count == len(tests)

if __name__ == "__main__":
    print(f"Testing Neural Memory API at {BASE_URL}")
    success = run_tests()
    exit(0 if success else 1)
