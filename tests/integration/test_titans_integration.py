#!/usr/bin/env python

"""
Integration tests for Titans variants using the CCE API.

These tests validate the behavior of each Titans variant (MAC, MAG, MAL)
by interacting with the Context Cascade Engine through its HTTP API.

Prerequisites:
- Docker services must be running with the following containers:
  - Memory Core service
  - Neural Memory service
  - Context Cascade Engine service

Run with:
    python -m pytest tests/integration/test_titans_integration.py -v
"""

import os
import json
import pytest
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
import aiohttp
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CCE_URL = os.environ.get("CCE_URL", "http://localhost:8002")
MEMORY_CORE_URL = os.environ.get("MEMORY_CORE_URL", "http://localhost:5010")
NEURAL_MEMORY_URL = os.environ.get("NEURAL_MEMORY_URL", "http://localhost:8001")


@pytest.fixture
async def api_client():
    """Create an aiohttp client session for making API requests."""
    async with aiohttp.ClientSession() as session:
        yield session


def generate_test_data(num_samples: int = 5, embedding_dim: int = 384) -> List[Dict]:
    """Generate controlled test data with predictable patterns.
    
    Args:
        num_samples: Number of test samples to generate
        embedding_dim: Dimension of the embedding vectors (384 is default for Lucidia)
    
    Returns:
        List of dictionaries with test inputs (content and embedding)
    """
    np.random.seed(42)  # For reproducibility
    
    # Create controlled input sequence with patterns
    sequence = []
    for i in range(num_samples):
        # Create base embedding with specific pattern
        embedding = np.zeros(embedding_dim, dtype=np.float32)
        
        # Create a pattern: first third rising, middle third flat, last third falling
        third = embedding_dim // 3
        
        # First third: rising pattern (0.1 to 0.9) with sample-specific offset
        embedding[:third] = np.linspace(0.1, 0.9, third) + (i * 0.05)
        
        # Middle third: constant value based on sample index
        embedding[third:2*third] = 0.5 + (i * 0.1)
        
        # Last third: falling pattern (0.9 to 0.1) with sample-specific offset
        embedding[2*third:] = np.linspace(0.9, 0.1, embedding_dim - 2*third) + (i * 0.05)
        
        # Add some controlled noise
        noise = np.random.normal(0, 0.05, embedding_dim).astype(np.float32)
        embedding += noise
        
        # Normalize the embedding to unit length (important for cosine similarity)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Create content string that's unique but semantically coherent
        content = f"Test content {i+1}: This is a controlled test input with specific embedding patterns. This sample has ID {i} and unique characteristics."
        
        sequence.append({
            "content": content,
            "embedding": embedding.tolist()
        })
    
    return sequence


async def set_titans_variant(api_client, variant: str) -> bool:
    """Set the active Titans variant in the CCE service.
    
    Note: In the actual implementation, Titans variant is set via environment variable
    TITANS_VARIANT when starting the CCE service. This function simulates that by checking
    if the CCE service is available and then logging the variant we're testing.
    
    Args:
        api_client: aiohttp client session
        variant: Variant name ('NONE', 'MAC', 'MAG', or 'MAL')
    
    Returns:
        True if CCE service is available, False otherwise
    """
    try:
        # Ensure variant is valid
        variant = variant.upper()
        if variant not in ["NONE", "MAC", "MAG", "MAL"]:
            logger.error(f"Invalid variant: {variant}")
            return False
        
        # Try multiple possible endpoints to check if CCE service is available
        health_endpoints = ["/health", "/", "/status", "/process_memory"]
        
        for endpoint in health_endpoints:
            try:
                async with api_client.get(f"{CCE_URL}{endpoint}", timeout=2) as response:
                    if response.status < 500:  # Any non-server error response indicates service is up
                        logger.info(f"CCE service available at {endpoint}. Testing with variant {variant}")
                        logger.warning(f"NOTE: Variant {variant} must be manually set in the CCE service environment")
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                continue
        
        logger.error("CCE service unavailable at all tested endpoints")
        return False
    except Exception as e:
        logger.error(f"Error connecting to CCE service: {e}")
        return False


async def process_memory(api_client, content: str, embedding: List[float]) -> Dict:
    """Process a memory through the CCE API.
    
    Args:
        api_client: aiohttp client session
        content: Text content to process
        embedding: Embedding vector for the content
    
    Returns:
        Response JSON from the CCE API
    """
    try:
        payload = {
            "content": content,
            "embedding": embedding
        }
        
        logger.info(f"Sending to CCE API: {json.dumps({k: '...' if k == 'embedding' else v for k, v in payload.items()})[:100]}...")
        
        async with api_client.post(
            f"{CCE_URL}/process_memory",
            json=payload,
            timeout=10
        ) as response:
            status = response.status
            if status == 200:
                result = await response.json()
                # Log response structure (without embedding arrays)
                result_log = {k: ('array' if isinstance(v, list) and len(v) > 10 else v) 
                             for k, v in result.items()}
                logger.info(f"Received from CCE API: {json.dumps(result_log)[:200]}...")
                return result
            else:
                error = await response.text()
                logger.error(f"Failed to process memory (status {status}): {error}")
                return {"error": error, "status": status}
    except Exception as e:
        logger.error(f"Error processing memory: {e}")
        return {"error": str(e)}


async def clear_memory_stores():
    """Clear both Memory Core and Neural Memory stores to start with a clean slate.
    
    This ensures tests run with a clean environment.
    This function makes best-effort attempts to clear the stores, handling connection errors gracefully.
    """
    try:
        # First clear Memory Core
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{MEMORY_CORE_URL}/clear", timeout=3) as response:
                    if response.status == 200:
                        logger.info("Cleared Memory Core store.")
                    else:
                        logger.warning(f"Failed to clear Memory Core: {await response.text()}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Could not connect to Memory Core: {e}")
            
            # Then reset Neural Memory using the /init endpoint instead of /clear
            try:
                async with session.post(f"{NEURAL_MEMORY_URL}/init", json={}, timeout=3) as response:
                    if response.status == 200:
                        logger.info("Reset Neural Memory via /init endpoint.")
                    else:
                        logger.warning(f"Failed to reset Neural Memory: {await response.text()}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Could not connect to Neural Memory: {e}")
    except Exception as e:
        logger.warning(f"Error during memory store clearing: {e}")
        logger.info("Continuing test despite memory clearing failure")


@pytest.mark.asyncio
async def test_base_variant():
    """Test the baseline (NONE) variant to establish a control baseline."""
    # Clear memory stores first
    await clear_memory_stores()
    
    # Create aiohttp client
    async with aiohttp.ClientSession() as client:
        # Set variant to NONE
        assert await set_titans_variant(client, "NONE")
        
        # Generate test data
        test_data = generate_test_data(num_samples=3)
        
        # Process each sample and store results
        results = []
        for i, sample in enumerate(test_data):
            logger.info(f"Processing sample {i+1}/{len(test_data)} with variant NONE")
            result = await process_memory(client, sample["content"], sample["embedding"])
            
            # Log the full result structure for debugging
            logger.info(f"Result structure keys: {list(result.keys())}")
            
            # Error check
            if "error" in result:
                logger.error(f"Error in API response: {result['error']}")
                if i == 0:  # Only skip on first sample
                    logger.warning("Skipping first sample due to error, trying next")
                    continue
            
            results.append(result)
            
            # Verify response structure based on actual CCE API format
            assert "neural_memory_retrieval" in result, "Response missing 'neural_memory_retrieval' section."
            retrieval_info = result["neural_memory_retrieval"]
            assert "retrieved_embedding" in retrieval_info, "'retrieved_embedding' missing within 'neural_memory_retrieval'."
            assert retrieval_info["retrieved_embedding"] is not None, "'retrieved_embedding' should not be null."
            
            # Additional basic structure validation
            assert "memory_id" in result, "Response missing memory_id"
            assert "status" in result, "Response missing status"
            assert "timestamp" in result, "Response missing timestamp"
        
        # Basic assertions for NONE variant
        for result in results:
            # Skip error results
            if "error" in result:
                continue
            
            # Check for presence of variant-specific outputs
            variant_output = result.get("variant_output", {})
            
            # NONE variant should not have attended_embedding or v_prime_t or gate metrics
            assert "attended_embedding" not in variant_output, "NONE variant should not have attended_embedding"
            assert "v_prime" not in variant_output, "NONE variant should not have v_prime"
            assert not any(gate in variant_output for gate in ["alpha", "theta", "eta"]), "NONE variant should not have gate values"


@pytest.mark.asyncio
async def test_mac_variant():
    """Test the MAC (Memory-Attended Computation) variant."""
    # Clear memory stores first
    await clear_memory_stores()
    
    # Create aiohttp client
    async with aiohttp.ClientSession() as client:
        # Set variant to MAC
        assert await set_titans_variant(client, "MAC")
        
        # Generate test data
        test_data = generate_test_data(num_samples=5)
        
        # Process each sample and store results
        results = []
        for i, sample in enumerate(test_data):
            logger.info(f"Processing sample {i+1}/{len(test_data)} with variant MAC")
            result = await process_memory(client, sample["content"], sample["embedding"])
            
            # Log the full response structure for debugging
            logger.info(f"Result keys: {list(result.keys())}")
            if "variant_output" in result:
                logger.info(f"Variant output keys: {list(result['variant_output'].keys())}")
                logger.info(f"Variant output: {json.dumps(result['variant_output'])}")
            
            # Skip errors
            if "error" in result:
                logger.error(f"Error in API response: {result['error']}")
                continue
                
            results.append(result)
            
            # Basic structure checks
            assert "neural_memory_retrieval" in result, "Response missing neural_memory_retrieval"
            assert "retrieved_embedding" in result["neural_memory_retrieval"], "Retrieved embedding missing"
            
            # Validate MAC-specific fields (after first sample to build context)
            if i > 0:
                # Check if variant output is present
                assert "variant_output" in result, "Response should include variant_output"
                variant_output = result["variant_output"]
                
                # Check that variant_type is correctly set to MAC
                assert "variant_type" in variant_output, "Response missing variant_type in variant_output"
                assert variant_output["variant_type"] == "MAC", f"Expected variant_type 'MAC', got '{variant_output.get('variant_type')}'"
                
                # Since MAC modifies the retrieved embedding, it should be different from the input embedding
                # but we can't directly compare without the original y_t_raw which isn't in the response
                retrieved = np.array(result["neural_memory_retrieval"]["retrieved_embedding"])
                
                # Log the MAC metrics we found
                logger.info(f"MAC retrieved embedding norm: {np.linalg.norm(retrieved):.6f}")
                assert np.linalg.norm(retrieved) > 0.1, "Retrieved embedding has too small magnitude"


@pytest.mark.asyncio
async def test_mag_variant():
    """Test the MAG (Memory-Attended Gates) variant."""
    # Clear memory stores first
    await clear_memory_stores()
    
    # Create aiohttp client
    async with aiohttp.ClientSession() as client:
        # Set variant to MAG
        assert await set_titans_variant(client, "MAG")
        
        # Generate test data
        test_data = generate_test_data(num_samples=5)
        
        # Process each sample and store results
        results = []
        for i, sample in enumerate(test_data):
            logger.info(f"Processing sample {i+1}/{len(test_data)} with variant MAG")
            result = await process_memory(client, sample["content"], sample["embedding"])
            
            # Log the response structure
            logger.info(f"Result keys: {list(result.keys())}")
            if "variant_output" in result:
                logger.info(f"Variant output keys: {list(result['variant_output'].keys())}")
                logger.info(f"Variant output: {json.dumps(result['variant_output'])}")
            
            # Skip errors
            if "error" in result:
                logger.error(f"Error in API response: {result['error']}")
                continue
                
            results.append(result)
            
            # Basic structure checks
            assert "neural_memory_retrieval" in result, "Response missing neural_memory_retrieval"
            assert "retrieved_embedding" in result["neural_memory_retrieval"], "Retrieved embedding missing"
            
            # Validate MAG-specific fields (after first sample to build context)
            if i > 0:
                # Check if variant output is present
                assert "variant_output" in result, "MAG should include variant_output"
                variant_output = result["variant_output"]
                
                # Check that variant_type is correctly set to MAG
                assert "variant_type" in variant_output, "Response missing variant_type in variant_output"
                assert variant_output["variant_type"] == "MAG", f"Expected variant_type 'MAG', got '{variant_output.get('variant_type')}'"
                
                # Check if neural_memory_update is present with relevant data
                if "neural_memory_update" in result:
                    logger.info(f"Neural memory update keys: {list(result['neural_memory_update'].keys())}")
                else: 
                    logger.info("No neural_memory_update in response")


@pytest.mark.asyncio
async def test_mal_variant():
    """Test the MAL (Memory-Augmented Learning) variant."""
    # Clear memory stores first
    await clear_memory_stores()
    
    # Create aiohttp client
    async with aiohttp.ClientSession() as client:
        # Set variant to MAL
        assert await set_titans_variant(client, "MAL")
        
        # Generate test data
        test_data = generate_test_data(num_samples=5)
        
        # Process each sample and store results
        results = []
        for i, sample in enumerate(test_data):
            logger.info(f"Processing sample {i+1}/{len(test_data)} with variant MAL")
            result = await process_memory(client, sample["content"], sample["embedding"])
            
            # Log the response structure
            logger.info(f"Result keys: {list(result.keys())}")
            if "variant_output" in result:
                logger.info(f"Variant output keys: {list(result['variant_output'].keys())}")
                logger.info(f"Variant output: {json.dumps(result['variant_output'])}")
            
            # Skip errors
            if "error" in result:
                logger.error(f"Error in API response: {result['error']}")
                continue
                
            results.append(result)
            
            # Basic structure checks
            assert "neural_memory_retrieval" in result, "Response missing neural_memory_retrieval"
            assert "retrieved_embedding" in result["neural_memory_retrieval"], "Retrieved embedding missing"
            
            # Validate MAL-specific fields (after first sample to build context)
            if i > 0:
                # Check if variant output is present
                assert "variant_output" in result, "MAL should include variant_output"
                variant_output = result["variant_output"]
                
                # Check that variant_type is correctly set to MAL
                assert "variant_type" in variant_output, "Response missing variant_type in variant_output"
                assert variant_output["variant_type"] == "MAL", f"Expected variant_type 'MAL', got '{variant_output.get('variant_type')}'"
                
                # Check if neural_memory_update is present
                if "neural_memory_update" in result:
                    logger.info(f"Neural memory update keys: {list(result['neural_memory_update'].keys())}")
                else:
                    logger.info("No neural_memory_update in response")


@pytest.mark.asyncio
async def test_all_variants_sequentially():
    """Run through all variants sequentially with the same test data to compare."""
    # Generate a fixed test sequence
    test_data = generate_test_data(num_samples=3)
    
    variant_results = {}
    variants = ["NONE", "MAC", "MAG", "MAL"]
    
    # Test each variant
    async with aiohttp.ClientSession() as client:
        for variant in variants:
            # Clear memory stores for a clean start
            await clear_memory_stores()
            
            # Set variant
            assert await set_titans_variant(client, variant)
            
            # Process all samples
            results = []
            for i, sample in enumerate(test_data):
                logger.info(f"Processing sample {i+1}/{len(test_data)} with variant {variant}")
                result = await process_memory(client, sample["content"], sample["embedding"])
                results.append(result)
            
            # Store results for comparison
            variant_results[variant] = results
    
    # Compare results across variants
    for i, sample in enumerate(test_data):
        logger.info(f"\nCOMPARISON FOR SAMPLE {i+1}:")
        
        # Skip first sample as it won't have attention effects
        if i == 0:
            continue
        
        # Compare retrieved embeddings across variants
        embeddings = {}
        for variant in variants:
            result = variant_results[variant][i]
            if "retrieved_embedding" in result:
                embeddings[f"{variant}_retrieved"] = np.array(result["retrieved_embedding"])
            if "attended_embedding" in result:
                embeddings[f"{variant}_attended"] = np.array(result["attended_embedding"])
        
        # Calculate cross-similarities
        logger.info("Cross-variant embedding similarities:")
        for name1, emb1 in embeddings.items():
            for name2, emb2 in embeddings.items():
                if name1 != name2:
                    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    logger.info(f"  {name1} vs {name2}: {cos_sim:.6f}")


if __name__ == "__main__":
    # This allows running the tests directly with python instead of pytest
    asyncio.run(test_all_variants_sequentially())
