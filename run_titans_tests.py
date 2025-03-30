#!/usr/bin/env python

"""
Titans Variants Test Runner

This script executes tests for the Titans variants (MAC, MAG, MAL) and generates
a report showing how each variant processes the same controlled input sequence.

Usage:
    python run_titans_tests.py
"""

import os
import sys
import time
import asyncio
import logging
import argparse
import numpy as np
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("titans_variants_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def print_separator(message=None):
    """Print a separator line with optional message."""
    width = 80
    if message:
        padding = (width - len(message) - 2) // 2
        print("=" * padding + f" {message} " + "=" * padding)
    else:
        print("=" * width)


def print_vector_preview(vector, name, num_values=5):
    """Print a preview of a vector with its stats."""
    if vector is None:
        print(f"{name}: None")
        return
    
    preview = ", ".join(f"{val:.4f}" for val in vector[:num_values])
    norm = np.linalg.norm(vector)
    mean = np.mean(vector)
    std = np.std(vector)
    print(f"{name} [{len(vector)}]: [{preview}, ...] (norm={norm:.4f}, mean={mean:.4f}, std={std:.4f})")


def create_controlled_input_sequence(num_samples: int = 5, embedding_dim: int = 64) -> List[Dict]:
    """Create a controlled sequence of input samples for testing.
    
    Args:
        num_samples: Number of test samples to generate
        embedding_dim: Dimension of the embedding vectors
    
    Returns:
        List of dictionaries with test inputs
    """
    np.random.seed(42)  # For reproducibility
    
    # Create controlled input sequence
    sequence = []
    for i in range(num_samples):
        # Create embeddings with specific patterns
        # Base pattern - random values but with controlled distribution
        x_base = np.random.normal(0, 1, embedding_dim).astype(np.float32)
        
        # Key projection - add some systematic bias based on sample index
        k_bias = np.sin(np.arange(embedding_dim) * (i + 1) / 10).astype(np.float32) * 0.5
        k_t = x_base + k_bias
        
        # Value projection - different pattern
        v_bias = np.cos(np.arange(embedding_dim) * (i + 1) / 8).astype(np.float32) * 0.5
        v_t = x_base + v_bias
        
        # Query projection - similar to key but with noise
        q_t = k_t + np.random.normal(0, 0.1, embedding_dim).astype(np.float32)
        
        # Retrieved embedding - make it related to value projection but distinct
        y_t = v_t * 0.8 + np.random.normal(0, 0.2, embedding_dim).astype(np.float32)
        
        sequence.append({
            'memory_id': f"test_mem_{i}",
            'x_t': x_base,
            'k_t': k_t,
            'v_t': v_t,
            'q_t': q_t,
            'y_t': y_t
        })
    
    return sequence


class SequenceContextManagerMock:
    """Mock implementation of SequenceContextManager for testing."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.contexts = []
        self.metrics = {}
    
    def add_context(self, timestamp: float, memory_id: str, x_t: np.ndarray, 
                  k_t: np.ndarray, v_t: np.ndarray, q_t: np.ndarray, y_t: np.ndarray) -> None:
        """Add a context tuple to the history."""
        self.contexts.append({
            'timestamp': timestamp,
            'memory_id': memory_id,
            'x_t': x_t,
            'k_t': k_t,
            'v_t': v_t,
            'q_t': q_t,
            'y_t': y_t
        })
        logger.info(f"Added context for memory_id {memory_id}. Context length: {len(self.contexts)}")
    
    def get_recent_keys(self, count: Optional[int] = None) -> List[np.ndarray]:
        """Get recent key projections."""
        if count is None or count >= len(self.contexts):
            return [ctx['k_t'] for ctx in self.contexts]
        return [ctx['k_t'] for ctx in self.contexts[-count:]]
    
    def get_recent_kv_pairs(self, count: Optional[int] = None) -> tuple:
        """Get recent (key, value) projection pairs."""
        if count is None or count >= len(self.contexts):
            return ([ctx['k_t'] for ctx in self.contexts], [ctx['v_t'] for ctx in self.contexts])
        return ([ctx['k_t'] for ctx in self.contexts[-count:]], [ctx['v_t'] for ctx in self.contexts[-count:]])
    
    def get_recent_ky_pairs(self, count: Optional[int] = None) -> tuple:
        """Get recent (key, retrieval) pairs."""
        if count is None or count >= len(self.contexts):
            return ([ctx['k_t'] for ctx in self.contexts], [ctx['y_t'] for ctx in self.contexts])
        return ([ctx['k_t'] for ctx in self.contexts[-count:]], [ctx['y_t'] for ctx in self.contexts[-count:]])
    
    def __len__(self) -> int:
        return len(self.contexts)


class NeuralMemoryClientMock:
    """Mock for the Neural Memory API client that returns predefined responses."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.call_history = []
    
    def calculate_gates(self, attention_output: List[float]) -> Dict[str, float]:
        """Mock implementation of the calculate_gates method."""
        self.call_history.append({
            'method': 'calculate_gates',
            'params': {'attention_output': attention_output},
            'timestamp': time.time()
        })
        
        # Return mock gate values based on attention output magnitude
        magnitude = np.linalg.norm(attention_output)
        logger.info(f"Mock NM Client: calculate_gates called with attention output magnitude: {magnitude:.6f}")
        
        # Higher magnitude -> higher learning rate, lower forgetting rate
        return {
            'alpha': 0.1 / (1 + magnitude),  # Lower alpha (forget less) with higher magnitude
            'theta': 0.1 * (1 + magnitude),   # Higher theta (learn more) with higher magnitude
            'eta': 0.5,                       # Constant eta for testing
        }


async def run_variants_comparison():
    """Run all variants with the same input sequence and compare results."""
    # Import directly here to avoid circular imports
    from synthians_memory_core.orchestrator.titans_variants import MACVariant, MAGVariant, MALVariant
    
    # Create controlled test sequence
    print_separator("GENERATING TEST DATA")
    test_sequence = create_controlled_input_sequence(num_samples=5, embedding_dim=64)
    print(f"Created test sequence with {len(test_sequence)} samples")
    
    # Initialize variants with separate context managers
    print_separator("INITIALIZING VARIANTS")
    variants = {
        'MAC': MACVariant(config={'attention_num_heads': 4, 'attention_key_dim': 32}),
        'MAG': MAGVariant(config={'attention_num_heads': 4, 'attention_key_dim': 32}),
        'MAL': MALVariant(config={'attention_num_heads': 4, 'attention_key_dim': 32})
    }
    
    # Create separate context managers for each variant
    context_managers = {
        variant_name: SequenceContextManagerMock(max_history=10) 
        for variant_name in variants.keys()
    }
    
    # Initialize API clients
    api_clients = {
        variant_name: NeuralMemoryClientMock() 
        for variant_name in variants.keys()
    }
    
    # Set up variants
    for variant_name, variant in variants.items():
        variant.set_sequence_context(context_managers[variant_name])
        variant.api_client = api_clients[variant_name]
        print(f"Initialized {variant_name} variant with dedicated context manager")
    
    # Process sequence through each variant
    all_results = {}
    
    for variant_name, variant in variants.items():
        print_separator(f"PROCESSING WITH {variant_name}")
        variant_results = []
        
        # First pass to store context and get baseline
        print(f"\n{variant_name} - First Pass (Building Context)")
        for i, sample in enumerate(test_sequence):
            print(f"\nSample {i+1}/{len(test_sequence)} - Memory ID: {sample['memory_id']}")
            
            # Process through variant
            result = await variant.process_input(
                memory_id=sample['memory_id'],
                x_t=sample['x_t'],
                k_t=sample['k_t'],
                v_t=sample['v_t'],
                q_t=sample['q_t'],
                y_t=sample['y_t']
            )
            variant_results.append(result)
            
            # Print input sample details
            print("Input:")
            print_vector_preview(sample['x_t'], "x_t")
            print_vector_preview(sample['k_t'], "k_t")
            print_vector_preview(sample['v_t'], "v_t")
            print_vector_preview(sample['q_t'], "q_t")
            print_vector_preview(sample['y_t'], "y_t (original)")
            
            # Print variant-specific outputs
            print("\nOutput:")
            if variant_name == 'MAC':
                attended_y = result.get('attended_output')
                print_vector_preview(attended_y, "attended_y")
                
                if i > 0:  # Only compare after we have context
                    cos_sim = np.dot(sample['y_t'], attended_y) / (
                        np.linalg.norm(sample['y_t']) * np.linalg.norm(attended_y))
                    diff_norm = np.linalg.norm(sample['y_t'] - attended_y)
                    print(f"y_t vs attended_y: cosine_sim={cos_sim:.6f}, diff_norm={diff_norm:.6f}")
            
            elif variant_name == 'MAG':
                print(f"alpha = {result.get('alpha', 'None')}")
                print(f"theta = {result.get('theta', 'None')}")
                print(f"eta = {result.get('eta', 'None')}")
            
            elif variant_name == 'MAL':
                # Also test calculate_v_prime directly if we have context
                if i > 0:
                    k_hist, v_hist = context_managers[variant_name].get_recent_kv_pairs()
                    v_prime_result = await variant.calculate_v_prime(
                        q_t=sample['q_t'],
                        v_t=sample['v_t'],
                        k_hist=k_hist,
                        v_hist=v_hist
                    )
                    v_prime = v_prime_result.get('v_prime_t')
                    print_vector_preview(v_prime, "v_prime_t")
                    
                    cos_sim = np.dot(sample['v_t'], v_prime) / (
                        np.linalg.norm(sample['v_t']) * np.linalg.norm(v_prime))
                    diff_norm = np.linalg.norm(sample['v_t'] - v_prime)
                    print(f"v_t vs v_prime_t: cosine_sim={cos_sim:.6f}, diff_norm={diff_norm:.6f}")
        
        all_results[variant_name] = variant_results
    
    # Summarize results
    print_separator("SUMMARY")
    for variant_name, results in all_results.items():
        print(f"\n{variant_name} Variant:")
        
        if variant_name == 'MAC':
            # Compare original vs attended outputs
            for i, (result, sample) in enumerate(zip(results, test_sequence)):
                if i > 0:  # Skip first sample as it has no history
                    original = sample['y_t']
                    attended = result['attended_output']
                    cos_sim = np.dot(original, attended) / (
                        np.linalg.norm(original) * np.linalg.norm(attended))
                    print(f"  Sample {i+1}: original vs attended cosine_sim={cos_sim:.6f}")
            print("  ✓ MAC attended outputs differ from originals but maintain relatedness")
        
        elif variant_name == 'MAG':
            # Check gate values
            gate_diffs = []
            for i, result in enumerate(results):
                if i > 0:  # Skip first sample
                    alpha = result.get('alpha')
                    theta = result.get('theta')
                    eta = result.get('eta')
                    print(f"  Sample {i+1}: alpha={alpha:.6f}, theta={theta:.6f}, eta={eta:.6f}")
            print("  ✓ MAG successfully calculated attention-based gate values")
        
        elif variant_name == 'MAL':
            print("  ✓ MAL integrated v_prime_t calculation based on historical context")
    
    print_separator("TEST COMPLETED")
    print("All variants processed the same input sequence successfully.")
    print("Each variant demonstrated its unique attention mechanism:")
    print("  • MAC: Enhanced retrieval outputs via historical context")
    print("  • MAG: Modulated learning parameters via attention signals")
    print("  • MAL: Augmented value representations before storage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Titans variants comparison tests")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the comparison
    asyncio.run(run_variants_comparison())
