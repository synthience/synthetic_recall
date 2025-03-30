#!/usr/bin/env python

"""
Test script for Titans variants (MAC, MAG, MAL)

This script initializes each variant with a controlled sequence of inputs and 
validates their behavior to ensure they function as expected.

Example usage:
    python -m tests.test_titans_variants
"""

import os
import sys
import time
import logging
import unittest
from typing import Dict, List, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from the project
from synthians_memory_core.orchestrator.titans_variants import (
    TitansVariantType, TitansVariantBase, MACVariant, MAGVariant, MALVariant
)
from synthians_memory_core.orchestrator.history import SequenceContextManager


class SequenceContextManagerMock(SequenceContextManager):
    """Mock implementation of SequenceContextManager for testing."""
    
    def __init__(self, max_history: int = 10):
        super().__init__(max_history)
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


class TestTitansVariants(unittest.TestCase):
    """Test cases for Titans variants."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create mock sequence context manager
        self.sequence_context = SequenceContextManagerMock(max_history=10)
        
        # Create test input sequence
        self.test_sequence = create_controlled_input_sequence(num_samples=5)
        
        # Initialize variants
        self.mac_variant = MACVariant(config={
            'attention_num_heads': 2,
            'attention_key_dim': 32,
            'attention_dropout': 0.0
        })
        self.mag_variant = MAGVariant(config={
            'attention_num_heads': 2,
            'attention_key_dim': 32,
            'attention_dropout': 0.0
        })
        self.mal_variant = MALVariant(config={
            'attention_num_heads': 2,
            'attention_key_dim': 32,
            'attention_dropout': 0.0
        })
        
        # Set sequence context manager for each variant
        self.mac_variant.set_sequence_context(self.sequence_context)
        self.mag_variant.set_sequence_context(self.sequence_context)
        self.mal_variant.set_sequence_context(self.sequence_context)
        
        # Mock Neural Memory Client
        self.neural_memory_client = NeuralMemoryClientMock()
        
        # Inject mock client
        self.mac_variant.api_client = self.neural_memory_client
        self.mag_variant.api_client = self.neural_memory_client
        self.mal_variant.api_client = self.neural_memory_client
    
    async def test_mac_variant(self):
        """Test MAC variant functionality."""
        logger.info("\n===== Testing MAC Variant =====")
        
        # MAC needs to accumulate context before it can apply attention
        # First pass through the variant should return original y_t
        # Later passes should return attended outputs
        
        results = []
        for idx, sample in enumerate(self.test_sequence):
            logger.info(f"\nProcessing sample {idx + 1}/{len(self.test_sequence)}")
            result = await self.mac_variant.process_input(
                memory_id=sample['memory_id'],
                x_t=sample['x_t'],
                k_t=sample['k_t'],
                v_t=sample['v_t'],
                q_t=sample['q_t'],
                y_t=sample['y_t']
            )
            results.append(result)
            
            # Log key metrics and compare with original
            y_original = sample['y_t']
            y_attended = result['attended_output']
            
            # For samples after the first, we should get an attended output different from original
            if idx > 0:
                # Calculate cosine similarity to check if attended output is different from original
                # but still related
                cos_sim = np.dot(y_original, y_attended) / (
                    np.linalg.norm(y_original) * np.linalg.norm(y_attended)
                )
                
                # The attended output should be different from original but still related
                logger.info(f"Cosine similarity between original and attended: {cos_sim:.6f}")
                
                # Verify that attention is doing something (outputs shouldn't be identical)
                self.assertFalse(
                    np.array_equal(y_original, y_attended),
                    "MAC: Attended output should be different from original"
                )
                
                # But they should still be related (similarity not too low)
                self.assertGreater(
                    cos_sim, 0.5,
                    "MAC: Attended output should be still related to original"
                )
        
        # Verify that the entire sequence was processed
        self.assertEqual(len(results), len(self.test_sequence))
        
        # The first sample's attended output should be identical to original
        # since there's no history yet
        self.assertTrue(
            np.array_equal(results[0]['attended_output'], self.test_sequence[0]['y_t']),
            "First sample should return original y_t since there's no history"
        )
        
        # Report overall metrics
        logger.info("\nMAC variant test completed successfully.")
    
    async def test_mag_variant(self):
        """Test MAG variant functionality."""
        logger.info("\n===== Testing MAG Variant =====")
        
        # MAG calculates gate values based on attention over historical keys
        
        results = []
        for idx, sample in enumerate(self.test_sequence):
            logger.info(f"\nProcessing sample {idx + 1}/{len(self.test_sequence)}")
            result = await self.mag_variant.process_input(
                memory_id=sample['memory_id'],
                x_t=sample['x_t'],
                k_t=sample['k_t'],
                v_t=sample['v_t'],
                q_t=sample['q_t'],
                y_t=sample['y_t']
            )
            results.append(result)
            
            # The first sample should have no gate values (no history)
            if idx == 0:
                self.assertIsNone(
                    result.get('alpha'),
                    "First sample should have no alpha gate value"
                )
                self.assertIsNone(
                    result.get('theta'),
                    "First sample should have no theta gate value"
                )
                self.assertIsNone(
                    result.get('eta'),
                    "First sample should have no eta gate value"
                )
            else:
                # Later samples should have gate values
                self.assertIsNotNone(
                    result.get('alpha'),
                    "Later samples should have alpha gate value"
                )
                self.assertIsNotNone(
                    result.get('theta'),
                    "Later samples should have theta gate value"
                )
                self.assertIsNotNone(
                    result.get('eta'),
                    "Later samples should have eta gate value"
                )
                
                # Log gate values
                logger.info(f"Gate values - alpha: {result.get('alpha'):.6f}, "
                          f"theta: {result.get('theta'):.6f}, "
                          f"eta: {result.get('eta'):.6f}")
                
                # Verify gate values are within expected ranges
                self.assertGreater(result.get('alpha', 0), 0, "Alpha should be positive")
                self.assertLess(result.get('alpha', 1), 1, "Alpha should be less than 1")
                self.assertGreater(result.get('theta', 0), 0, "Theta should be positive")
                self.assertGreater(result.get('eta', 0), 0, "Eta should be positive")
                self.assertLess(result.get('eta', 1), 1, "Eta should be less than 1")
        
        # Verify that the entire sequence was processed
        self.assertEqual(len(results), len(self.test_sequence))
        
        # Verify that the neural memory client was called to calculate gates
        self.assertGreater(
            len([call for call in self.neural_memory_client.call_history 
                 if call['method'] == 'calculate_gates']),
            0,
            "Neural Memory client should be called to calculate gates"
        )
        
        # Report overall metrics
        logger.info("\nMAG variant test completed successfully.")
    
    async def test_mal_variant(self):
        """Test MAL variant functionality."""
        logger.info("\n===== Testing MAL Variant =====")
        
        # MAL augments value projections with attended historical values
        
        results = []
        for idx, sample in enumerate(self.test_sequence):
            logger.info(f"\nProcessing sample {idx + 1}/{len(self.test_sequence)}")
            
            # For testing MAL with historical context, we need to simulate the CCE flow
            if idx > 0:  # We need history for MAL
                # Get historical key-value pairs from context manager
                k_hist, v_hist = self.sequence_context.get_recent_kv_pairs()
                
                # Call calculate_v_prime directly to test value augmentation
                mal_result = await self.mal_variant.calculate_v_prime(
                    q_t=sample['q_t'],
                    v_t=sample['v_t'],
                    k_hist=k_hist,
                    v_hist=v_hist
                )
                
                # Verify the augmented value projection
                self.assertTrue(
                    'v_prime_t' in mal_result,
                    "MAL should return augmented value projection"
                )
                
                v_prime = mal_result['v_prime_t']
                v_original = sample['v_t']
                
                # Calculate cosine similarity to check if v_prime is different but related
                cos_sim = np.dot(v_original, v_prime) / (
                    np.linalg.norm(v_original) * np.linalg.norm(v_prime)
                )
                
                logger.info(f"Cosine similarity between original v_t and v_prime_t: {cos_sim:.6f}")
                
                # The v_prime should be different from original v_t
                self.assertFalse(
                    np.array_equal(v_original, v_prime),
                    "MAL: v_prime_t should be different from original v_t"
                )
                
                # But they should still be related (similarity not too low)
                self.assertGreater(
                    cos_sim, 0.5,
                    "MAL: v_prime_t should be still related to original v_t"
                )
                
                # Save v_prime for comparison
                sample['v_prime_t'] = v_prime
            
            # Also run the standard process_input method to populate context
            result = await self.mal_variant.process_input(
                memory_id=sample['memory_id'],
                x_t=sample['x_t'],
                k_t=sample['k_t'],
                v_t=sample['v_t'],
                q_t=sample['q_t'],
                y_t=sample['y_t']
            )
            results.append(result)
        
        # Verify that the entire sequence was processed
        self.assertEqual(len(results), len(self.test_sequence))
        
        # Report overall metrics
        logger.info("\nMAL variant test completed successfully.")


async def run_tests():
    """Run all the variant tests."""
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestTitansVariants('test_mac_variant'))
    test_suite.addTest(TestTitansVariants('test_mag_variant'))
    test_suite.addTest(TestTitansVariants('test_mal_variant'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return await runner.run(test_suite)


def main():
    """Main entry point."""
    import asyncio
    
    # Run the tests
    logger.info("Starting Titans variant tests...")
    asyncio.run(run_tests())
    logger.info("All tests completed.")


if __name__ == "__main__":
    main()
