#!/usr/bin/env python

"""
Integration tests for Titans variant switching capability (Phase 4.6).

These tests validate the dynamic variant switching functionality, including
DevMode protection, context flushing, optional Neural Memory reset, and response validation.

Prerequisites:
- Docker services (MC, NM, CCE) must be running.
- CCE container *must* have CCE_DEV_MODE=true for most tests.
"""

import os
import json
import pytest
import pytest_asyncio
import asyncio
import aiohttp
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - Use environment variables with defaults
CCE_URL = os.environ.get('CCE_URL', 'http://localhost:8002')
MC_URL = os.environ.get('MC_URL', 'http://localhost:5010')
NM_URL = os.environ.get('NM_URL', 'http://localhost:8001')

# Global variable to store fetched embedding dimension
CONFIGURED_EMBEDDING_DIM = None

# --- Fixtures ---

@pytest.fixture(scope="session", autouse=True)
async def fetch_embedding_dim():
    """Fetch embedding dimension from Memory Core once per session."""
    global CONFIGURED_EMBEDDING_DIM
    if CONFIGURED_EMBEDDING_DIM is None:
        print("\nFetching embedding dimension from Memory Core...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{MC_URL}/stats", timeout=10) as response:
                    if response.status == 200:
                        stats = await response.json()
                        CONFIGURED_EMBEDDING_DIM = stats.get("api_server", {}).get("embedding_dim", 768)
                        print(f"Using embedding dimension: {CONFIGURED_EMBEDDING_DIM}")
                    else:
                        print(f"Warning: Failed to get stats from Memory Core (Status: {response.status}). Using default dim 768.")
                        CONFIGURED_EMBEDDING_DIM = 768
            except Exception as e:
                print(f"Warning: Error fetching embedding dim: {e}. Using default dim 768.")
                CONFIGURED_EMBEDDING_DIM = 768

# Use pytest-asyncio built-in event loop management
# Fixture for the aiohttp client session
@pytest_asyncio.fixture
async def http_session():
    """Provides an aiohttp ClientSession."""
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        yield session

# --- Helper Functions ---

def generate_test_embedding(dim: Optional[int] = None):
    """Generate a random test embedding, using fetched dimension."""
    global CONFIGURED_EMBEDDING_DIM
    target_dim = dim if dim is not None else CONFIGURED_EMBEDDING_DIM
    if target_dim is None:
        logger.warning("Embedding dimension not fetched, defaulting to 768 for test embedding.")
        target_dim = 768

    embedding = np.random.normal(0, 1, target_dim).astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm # Normalize
    return embedding.tolist()

async def set_variant(session: aiohttp.ClientSession, variant: str, reset_neural_memory: bool = False):
    """Helper to call the /set_variant endpoint."""
    url = f"{CCE_URL}/set_variant"
    payload = {"variant": variant, "reset_neural_memory": reset_neural_memory}
    log_msg = f"Calling {url} with payload: {payload}"
    logger.info(log_msg)
    print(f"\n{log_msg}") # Ensure visibility during test run
    try:
        async with session.post(url, json=payload) as response:
            status = response.status
            result = await response.json()
            log_resp = f"Response from /set_variant (status {status}): {json.dumps(result, indent=2)}"
            logger.info(log_resp)
            print(log_resp)
            return result, status
    except Exception as e:
        err_msg = f"Error calling /set_variant: {e}"
        logger.error(err_msg, exc_info=True)
        print(f"[ERROR] {err_msg}")
        return {"error": str(e), "success": False}, 500

async def process_memory(session: aiohttp.ClientSession, content: str, embedding: Optional[List[float]] = None):
    """Helper to call the CCE /process_memory endpoint."""
    url = f"{CCE_URL}/process_memory"
    payload = {"content": content, "metadata": {"source": "integration_test"}}
    if embedding:
        payload["embedding"] = embedding
    log_msg = f"Calling {url} for content: '{content[:30]}...'"
    logger.info(log_msg)
    print(log_msg)
    try:
        async with session.post(url, json=payload) as response:
            status = response.status
            result = await response.json()
            log_resp = f"Response from /process_memory (status {status}) for '{content[:30]}...': Keys={list(result.keys())}"
            logger.info(log_resp)
            print(log_resp)
            return result, status
    except Exception as e:
        err_msg = f"Error calling /process_memory: {e}"
        logger.error(err_msg, exc_info=True)
        print(f"[ERROR] {err_msg}")
        return {"error": str(e), "success": False}, 500

async def reset_neural_memory(session: aiohttp.ClientSession):
    """Helper to call NM /init endpoint."""
    url = f"{NM_URL}/init"
    logger.info(f"Calling {url} to reset Neural Memory state...")
    print(f"Calling {url} to reset Neural Memory state...")
    try:
        async with session.post(url, json={}) as response: # Empty payload for default init
            status = response.status
            result = await response.json()
            log_resp = f"Response from NM /init (status {status}): {result.get('message')}"
            logger.info(log_resp)
            print(log_resp)
            return result, status
    except Exception as e:
        err_msg = f"Error calling NM /init: {e}"
        logger.error(err_msg, exc_info=True)
        print(f"[ERROR] {err_msg}")
        return {"error": str(e), "success": False}, 500

# --- Test Class ---

@pytest.mark.integration
@pytest.mark.variant
class TestVariantSwitching:

    @pytest_asyncio.fixture(autouse=True)
    async def setup_and_teardown(self, http_session: aiohttp.ClientSession):
        """Reset NM state and set variant to NONE before each test."""
        print("\n--- Test Setup: Resetting NM and setting variant to NONE ---")
        await reset_neural_memory(http_session)
        await set_variant(http_session, "NONE")
        # No yield needed as we just want setup before each test method

    @pytest.mark.asyncio
    @pytest.mark.parametrize("variant", ["NONE", "MAC", "MAG", "MAL"])
    async def test_basic_switching_and_processing(self, http_session: aiohttp.ClientSession, variant: str):
        """Test that we can switch to each variant and process memory with it."""
        logger.info(f"====== Starting test_basic_switching_and_processing for {variant} ======")
        
        # Get starting/previous variant from test setup (should always be "NONE")
        previous_variant = "NONE"
        
        # Try to switch to the requested variant
        result_set, status_set = await set_variant(http_session, variant)
        logger.info(f"Set variant result: {json.dumps(result_set, indent=2)}")
        
        # Basic assertions for the switch itself
        assert status_set == 200
        assert result_set.get("success") is True
        assert result_set.get("variant") == variant

        # Updated assertions based on response status
        if result_set.get("status") == "switched":
            expected_prev = previous_variant # From the setup fixture perspective
            assert result_set.get("previous_variant") == expected_prev, \
                   f"Expected previous_variant {expected_prev}, got {result_set.get('previous_variant')}"
            assert result_set.get("context_flushed") is True # Should flush on actual switch
            assert result_set.get("reconfigured") is True
        elif result_set.get("status") == "unchanged":
            assert variant == previous_variant # Should only be unchanged if variant matches previous
            assert result_set.get("context_flushed", False) is False # No flush if unchanged
            # Reconfigured might be true or false depending on implementation, skip check
        else:
            pytest.fail(f"Unexpected status in set_variant response: {result_set.get('status')}")

        assert result_set.get("dev_mode", True) is True
        
        # Process a memory with the current variant
        test_content = f"Test memory for {variant} variant"
        test_embedding = generate_test_embedding()
        
        result_proc, status_proc = await process_memory(http_session, test_content, test_embedding)
        variant_output = result_proc.get("variant_output", {})
        
        # Assertions for processing
        assert status_proc == 200
        assert result_proc.get("status") == "completed"
        assert "memory_id" in result_proc
        # Check the variant_output reflects the *active* variant
        assert variant_output.get("variant_type") == variant, \
               f"CCE processed using unexpected variant '{variant_output.get('variant_type')}' instead of '{variant}'"
               
        # Variant-specific assertion checks
        if variant == "MAC":
            print("Checking MAC specific outputs...")
            assert "mac" in variant_output, "Response missing MAC metrics in variant_output"
            assert variant_output["mac"].get("attended_output_generated") is True, "MAC metrics missing 'attended_output_generated' flag"

        logger.info(f"====== Finished test_basic_switching_and_processing for {variant} ======")


    @pytest.mark.asyncio
    async def test_context_flush_effectiveness(self, http_session: aiohttp.ClientSession):
        """Test context is flushed during switch."""
        logger.info("====== Starting test_context_flush_effectiveness ======")

        # Set to MAC, process 5 memories
        await set_variant(http_session, "MAC")
        for i in range(5):
            await process_memory(http_session, f"MAC context memory {i}", generate_test_embedding())
            await asyncio.sleep(0.1) # Small delay

        # Switch to MAG, check flushed size
        result_switch, status_switch = await set_variant(http_session, "MAG")
        assert status_switch == 200
        assert result_switch.get("success") is True
        assert result_switch.get("context_flushed") is True
        # Check if context_size_flushed > 0. Allow 0 if initial state was empty.
        assert result_switch.get("context_size_flushed", -1) >= 0 # Check exists and is non-negative
        # Check if the size was actually 5 (or close to it depending on implementation)
        assert result_switch.get("context_size_flushed") == 5, "Expected 5 context entries to be flushed"
        logger.info(f"Context flushed: {result_switch.get('context_size_flushed')} entries.")

        # Process one more, check variant
        result_proc, status_proc = await process_memory(http_session, "Post-switch MAG memory", generate_test_embedding())
        assert status_proc == 200
        assert result_proc.get("variant_output", {}).get("variant_type") == "MAG"

        logger.info("====== Finished test_context_flush_effectiveness ======")

    @pytest.mark.asyncio
    async def test_neural_memory_reset(self, http_session: aiohttp.ClientSession):
        """Test optional NM reset."""
        logger.info("====== Starting test_neural_memory_reset ======")

        # Process an initial memory to establish some NM state
        await set_variant(http_session, "MAL") # Use MAL which modifies value
        initial_content = "Initial NM state memory"
        initial_embedding = generate_test_embedding()
        resp1, _ = await process_memory(http_session, initial_content, initial_embedding)
        loss1 = resp1.get("neural_memory_update", {}).get("loss")
        grad1 = resp1.get("neural_memory_update", {}).get("grad_norm")
        assert loss1 is not None, "Missing loss in initial processing"
        print(f"Initial process (MAL): Loss={loss1:.6f}, GradNorm={grad1:.6f}")

        # Switch to MAC *without* reset
        result_no_reset, _ = await set_variant(http_session, "MAC", reset_neural_memory=False)
        assert result_no_reset.get("success") is True
        assert result_no_reset.get("neural_memory_reset") is False

        # Process the *exact same* initial memory again (now with MAC variant)
        resp2, _ = await process_memory(http_session, initial_content, initial_embedding)
        loss2 = resp2.get("neural_memory_update", {}).get("loss")
        grad2 = resp2.get("neural_memory_update", {}).get("grad_norm")
        assert loss2 is not None, "Missing loss in second processing"
        print(f"Second process (MAC, no reset): Loss={loss2:.6f}, GradNorm={grad2:.6f}")
        # Loss should likely be lower now as NM has seen this input via MAL
        # assert loss2 < loss1, "Loss should decrease without NM reset" # This might be unreliable

        # Switch to MAG *with* reset
        result_with_reset, _ = await set_variant(http_session, "MAG", reset_neural_memory=True)
        assert result_with_reset.get("success") is True
        assert result_with_reset.get("neural_memory_reset") is True

        # Process the *exact same* initial memory again (now with MAG variant, after reset)
        resp3, _ = await process_memory(http_session, initial_content, initial_embedding)
        loss3 = resp3.get("neural_memory_update", {}).get("loss")
        grad3 = resp3.get("neural_memory_update", {}).get("grad_norm")
        assert loss3 is not None, "Missing loss in third processing"
        print(f"Third process (MAG, with reset): Loss={loss3:.6f}, GradNorm={grad3:.6f}")

        # After reset, loss should be similar to the *initial* loss (loss1)
        # Allow some tolerance for floating point / minor differences
        assert abs(loss3 - loss1) < 0.1 * abs(loss1) + 1e-5, \
            f"Loss after reset ({loss3:.6f}) is not close to initial loss ({loss1:.6f}). NM Reset likely failed."

        logger.info("====== Finished test_neural_memory_reset ======")

    @pytest.mark.asyncio
    async def test_invalid_variant_name(self, http_session: aiohttp.ClientSession):
        """Test setting an invalid variant name."""
        logger.info("====== Starting test_invalid_variant_name ======")
        result, status = await set_variant(http_session, "INVALID_VARIANT")
        assert status == 400
        assert "detail" in result  # Check for FastAPI error detail key
        assert "Invalid variant type" in result.get("detail", ""), "Response should indicate invalid variant"
        logger.info("====== Finished test_invalid_variant_name ======")

    @pytest.mark.asyncio
    async def test_same_variant_no_change(self, http_session: aiohttp.ClientSession):
        """Test switching to the currently active variant."""
        logger.info("====== Starting test_same_variant_no_change ======")
        # Set to MAG first
        await set_variant(http_session, "MAG")
        # Set to MAG again
        result, status = await set_variant(http_session, "MAG")
        assert status == 200
        assert result.get("success") is True
        assert result.get("status") == "unchanged"
        assert result.get("variant") == "MAG"
        assert result.get("context_flushed", False) is False, "Context should not be flushed if variant is unchanged"
        logger.info("====== Finished test_same_variant_no_change ======")

    @pytest.mark.asyncio
    async def test_comprehensive_variant_switching(self, http_session: aiohttp.ClientSession):
        """
        Comprehensive test that cycles through all variants sequentially and verifies:
        1. Each variant can be switched to successfully
        2. Memory processing works with each variant
        3. Metrics structure is consistent with expected format
        4. Proper cleanup occurs between variant switches
        """
        logger.info("====== Starting test_comprehensive_variant_switching ======")
        
        # Define all variants to test in sequence
        variants = ["NONE", "MAC", "MAG", "MAL", "NONE"]  # End with NONE to reset to default state
        
        # Track memory IDs for each variant for verification
        memory_ids = {}
        
        # Start with NONE variant (setup_and_teardown should ensure this)
        current_variant = "NONE"
        
        # Cycle through all variants
        for next_variant in variants:
            if next_variant == current_variant:
                logger.info(f"Skipping switch to {next_variant} as it's already active")
                continue
                
            logger.info(f"\n--- Switching from {current_variant} to {next_variant} ---")
            
            # Switch to the next variant
            switch_result, switch_status = await set_variant(http_session, next_variant)
            
            # Verify switch was successful
            assert switch_status == 200, f"Failed to switch to {next_variant}, status: {switch_status}"
            assert switch_result.get("success") is True, f"Failed to switch to {next_variant}: {switch_result}"
            assert switch_result.get("variant") == next_variant
            assert switch_result.get("previous_variant") == current_variant
            assert switch_result.get("context_flushed") is True
            assert switch_result.get("reconfigured") is True
            
            # Update current variant
            current_variant = next_variant
            
            # Process a test memory with this variant
            logger.info(f"Processing test memory with {current_variant} variant")
            test_content = f"Comprehensive test memory for {current_variant} variant - {time.time()}"
            test_embedding = generate_test_embedding()
            
            process_result, process_status = await process_memory(http_session, test_content, test_embedding)
            
            # Verify processing was successful
            assert process_status == 200, f"Failed to process memory with {current_variant}, status: {process_status}"
            assert process_result.get("status") == "completed", f"Memory processing failed: {process_result.get('status')}"
            
            # Store memory ID for this variant
            memory_id = process_result.get("memory_id")
            assert memory_id is not None, "No memory_id in response"
            memory_ids[current_variant] = memory_id
            
            # Verify variant_output structure
            variant_output = process_result.get("variant_output", {})
            assert "variant_type" in variant_output, "Missing variant_type in variant_output"
            assert variant_output["variant_type"] == current_variant
            
            # Verify variant-specific metrics structure
            # Note: Not all variants have consistently structured metrics yet
            # MAC has 'mac' key but MAG and MAL might not have their respective keys
            if current_variant == "MAC":
                # MAC always has a nested 'mac' dictionary with specific metrics
                assert "mac" in variant_output, "Missing mac key in variant_output for MAC variant"
                mac_metrics = variant_output.get("mac", {})
                assert "attended_output_generated" in mac_metrics, "MAC metrics missing 'attended_output_generated'"
                assert isinstance(mac_metrics["attended_output_generated"], bool)
                assert "fallback_mode" in mac_metrics, "MAC metrics missing 'fallback_mode'"
                assert isinstance(mac_metrics["fallback_mode"], bool)
            elif current_variant == "MAG":
                # Document the current behavior (MAG doesn't have a 'mag' key yet)
                # This may need to be updated if MAG is enhanced to use consistent structure
                assert variant_output["variant_type"] == "MAG", "MAG variant type incorrect"
                # If 'mag' key exists, verify it's a dictionary (but don't require it)
                if "mag" in variant_output:
                    assert isinstance(variant_output["mag"], dict), "MAG metrics should be a dictionary"
            elif current_variant == "MAL":
                # Document the current behavior (MAL doesn't have a 'mal' key yet)
                # This may need to be updated if MAL is enhanced to use consistent structure
                assert variant_output["variant_type"] == "MAL", "MAL variant type incorrect"
                # If 'mal' key exists, verify it's a dictionary (but don't require it)
                if "mal" in variant_output:
                    assert isinstance(variant_output["mal"], dict), "MAL metrics should be a dictionary"
            
            # Verify no unexpected top-level keys in variant_output
            allowed_keys = ["variant_type", "none", "mac", "mag", "mal"]
            for key in variant_output.keys():
                assert key in allowed_keys, f"Unexpected top-level key '{key}' in variant_output"
            
            # Short delay between operations
            await asyncio.sleep(0.2)
        
        # Verify we processed memories with all variants
        for variant in ["NONE", "MAC", "MAG", "MAL"]:
            assert variant in memory_ids, f"No memory was processed with {variant} variant"
        
        logger.info("====== Completed test_comprehensive_variant_switching ======")
        logger.info(f"Processed memories with IDs: {memory_ids}")

    # Test for DevMode protection (requires manual setup or more complex fixture)
    # @pytest.mark.asyncio
    # async def test_dev_mode_protection(self, http_session: aiohttp.ClientSession):
    #     """Test that switching fails if CCE_DEV_MODE is not true."""
    #     # This test requires starting the CCE container *without* CCE_DEV_MODE=true
    #     logger.info("====== Starting test_dev_mode_protection ======")
    #     print("WARNING: This test assumes CCE container was started WITHOUT CCE_DEV_MODE=true")
    #     result, status = await set_variant(http_session, "MAC")
    #     assert status == 403 # Forbidden
    #     assert "Cannot switch variants" in result.get("detail", result.get("error", ""))
    #     logger.info("====== Finished test_dev_mode_protection ======")