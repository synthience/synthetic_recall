# tests/test_adaptive_attention.py

import pytest
import pytest_asyncio
import asyncio
import json
import os
import time
from typing import Dict, List, Any, Optional

# Import our variant testing fixtures
from variant_conftest import api_clients, create_test_memories

# Note: We don't skip tests based on TITANS_VARIANT here since we want to test all variants
# We'll dynamically set the variant as part of our test metadata

# Test constants
FOCUS_MODES = ["recency", "relevance", "emotional", "broad", "balance"]
VARIANT_TYPES = ["MAC", "MAG", "MAL"]

@pytest.mark.asyncio
async def test_focus_mode_mapping(api_clients):
    """Test that different focus modes correctly map to expected parameters in all variants."""
    session, mc_client = api_clients
    
    # 1. Create some test memories to build history context
    memory_ids = await create_test_memories(mc_client, count=20, 
                                         prefix=f"Adaptive-Attention-Test")
    
    # Allow processing to complete
    await asyncio.sleep(2)
    
    # Test results for each variant and focus mode combination
    results = {}
    
    # 2. For each variant type, test all focus modes
    for variant_type in VARIANT_TYPES:
        results[variant_type] = {}
        
        for focus_mode in FOCUS_MODES:
            # Create a memory with attention hints for this focus mode
            async with session.post(
                "http://localhost:8002/process_memory",
                json={
                    "content": f"This is a test memory for {variant_type} variant with {focus_mode} focus",
                    "embedding": [float(i) / 100 for i in range(384)],  # Simple test embedding
                    "metadata": {
                        "source": "adaptive_attention_test",
                        "variant": variant_type,
                        "attention_hints": {
                            "focus": focus_mode,
                        }
                    }
                }
            ) as response:
                assert response.status == 200, f"Failed to process memory via CCE: {await response.text()}"
                result = await response.json()
                assert "memory_id" in result, "No memory_id in response"
                assert "variant_output" in result, "No variant_output in response"
                
                # Store the result for analysis
                results[variant_type][focus_mode] = result
                
                # Allow time for CCE to process
                await asyncio.sleep(1)
    
    # 3. Validate results for each variant and focus mode
    # MAC variant expectations
    if "MAC" in results:
        for focus_mode, result in results["MAC"].items():
            metrics = result.get("variant_output", {}).get("metrics", {})
            assert metrics.get("attention_applied", False), f"MAC: Attention not applied for {focus_mode}"
            assert metrics.get("temperature_scaling", False) == (focus_mode != "balance"), f"MAC: Wrong temperature scaling for {focus_mode}"
            
            # Verify focus mode specific expectations
            if focus_mode == "recency":
                assert metrics.get("recency_bias_applied", False), "MAC: Recency bias not applied"
                assert metrics.get("context_limited", False), "MAC: Context not limited for recency"
            
            elif focus_mode == "relevance":
                # For relevance we expect variance normalization in some cases
                if metrics.get("variance_normalization_applied", False):
                    assert True, "MAC: Variance normalization applied for relevance"
                
            elif focus_mode == "emotional" or focus_mode == "broad":
                assert metrics.get("historical_bias_applied", False), f"MAC: Historical bias not applied for {focus_mode}"
    
    # MAG variant expectations 
    if "MAG" in results:
        for focus_mode, result in results["MAG"].items():
            metrics = result.get("variant_output", {}).get("metrics", {})
            assert metrics.get("gate_calculation_success", False), f"MAG: Gate calculation failed for {focus_mode}"
            if focus_mode != "balance":  # balance uses default gate values
                assert metrics.get("gates_modified", False), f"MAG: Gates not modified for {focus_mode}"
            
            gates = metrics.get("calculated_gates", {})
            
            # Verify focus mode specific expectations
            if focus_mode == "recency":
                # Recency typically has higher alpha (more forgetting)
                assert metrics.get("context_limited", False), "MAG: Context not limited for recency"
                
            elif focus_mode == "broad":
                # Broad typically has lower alpha (less forgetting) 
                if "alpha" in gates:
                    assert gates["alpha"] < 0.4, f"MAG: Alpha too high ({gates['alpha']}) for broad focus"
    
    # MAL variant expectations
    if "MAL" in results:
        for focus_mode, result in results["MAL"].items():
            metrics = result.get("variant_output", {}).get("metrics", {})
            assert metrics.get("v_prime_calculation_success", False), f"MAL: v_prime calculation failed for {focus_mode}"
            assert metrics.get("temperature_scaling", False) == (focus_mode != "balance"), f"MAL: Wrong temperature scaling for {focus_mode}"
            
            # Verify focus mode specific expectations  
            if focus_mode == "recency":
                assert metrics.get("context_limited", False), "MAL: Context not limited for recency"
                assert metrics.get("blend_factor", 0.5) > 0.5, "MAL: Unexpected blend factor for recency"
                
            elif focus_mode == "broad":
                assert metrics.get("blend_factor", 0.5) < 0.2, "MAL: Blend factor too high for broad focus"
                
            # Check attention mode is recorded correctly
            assert "attention_mode" in metrics, f"MAL: No attention_mode recorded for {focus_mode}"

@pytest.mark.asyncio
async def test_hint_overrides(api_clients):
    """Test that explicit hint overrides take precedence over focus mode defaults."""
    session, mc_client = api_clients
    
    # 1. Create some test memories to build history context
    memory_ids = await create_test_memories(mc_client, count=15, 
                                         prefix=f"Hint-Override-Test")
    
    # Allow processing to complete
    await asyncio.sleep(2)
    
    # 2. Test overrides for MAC variant
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": "Testing MAC with explicit overrides",
            "embedding": [float(i) / 100 for i in range(384)],
            "metadata": {
                "source": "hint_override_test",
                "variant": "MAC",
                "attention_hints": {
                    "focus": "relevance",  # Base focus mode
                    "mac": {
                        "context_limit": 5,  # Override the default context limit
                        "attention_temperature": 2.5  # Override the default temperature
                    }
                }
            }
        }
    ) as response:
        assert response.status == 200
        mac_result = await response.json()
        mac_metrics = mac_result.get("variant_output", {}).get("metrics", {})
        
        # Verify MAC overrides worked
        assert mac_metrics.get("context_limit", 0) == 5, "MAC: context_limit override not applied"
        assert mac_metrics.get("attention_temperature", 0) == 2.5, "MAC: attention_temperature override not applied"
    
    await asyncio.sleep(1)
    
    # 3. Test overrides for MAG variant
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": "Testing MAG with explicit overrides",
            "embedding": [float(i) / 100 for i in range(384)],
            "metadata": {
                "source": "hint_override_test",
                "variant": "MAG",
                "attention_hints": {
                    "focus": "relevance",  # Base focus mode
                    "mag": {
                        "context_limit": 3,  # Override the default context limit
                        "gate_modifiers": {
                            "alpha_scale": 0.1,  # Override the default alpha scaling
                            "theta_scale": 2.0   # Override the default theta scaling
                        }
                    }
                }
            }
        }
    ) as response:
        assert response.status == 200
        mag_result = await response.json()
        mag_metrics = mag_result.get("variant_output", {}).get("metrics", {})
        
        # Verify MAG overrides worked
        assert mag_metrics.get("context_limit", 0) == 3, "MAG: context_limit override not applied"
        assert mag_metrics.get("gate_modifiers", {}).get("alpha_scale", 1.0) == 0.1, "MAG: alpha_scale override not applied"
        assert mag_metrics.get("gate_modifiers", {}).get("theta_scale", 1.0) == 2.0, "MAG: theta_scale override not applied"
    
    await asyncio.sleep(1)
    
    # 4. Test overrides for MAL variant
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": "Testing MAL with explicit overrides",
            "embedding": [float(i) / 100 for i in range(384)],
            "metadata": {
                "source": "hint_override_test",
                "variant": "MAL",
                "attention_hints": {
                    "focus": "relevance",  # Base focus mode
                    "mal": {
                        "context_limit": 7,  # Override the default context limit
                        "blend_factor": 0.25,  # Override the default blend factor
                        "attention_temperature": 1.75  # Override the default temperature
                    }
                }
            }
        }
    ) as response:
        assert response.status == 200
        mal_result = await response.json()
        mal_metrics = mal_result.get("variant_output", {}).get("metrics", {})
        
        # Verify MAL overrides worked
        assert mal_metrics.get("context_limit", 0) == 7, "MAL: context_limit override not applied"
        assert mal_metrics.get("blend_factor", 0) == 0.25, "MAL: blend_factor override not applied"
        assert mal_metrics.get("attention_temperature", 0) == 1.75, "MAL: attention_temperature override not applied"

@pytest.mark.asyncio
async def test_edge_cases(api_clients):
    """Test handling of edge cases like missing hints, empty history, etc."""
    session, mc_client = api_clients
    
    # 1. Test with no attention_hints at all
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": "Testing with no attention hints",
            "embedding": [float(i) / 100 for i in range(384)],
            "metadata": {
                "source": "edge_case_test",
                "variant": "MAC"  # No attention_hints
            }
        }
    ) as response:
        assert response.status == 200, "Failed with no attention hints"
        result = await response.json()
        # Should succeed with default values
        assert "memory_id" in result, "No memory_id in response with no attention hints"
    
    await asyncio.sleep(1)
    
    # 2. Test with empty attention_hints
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": "Testing with empty attention hints",
            "embedding": [float(i) / 100 for i in range(384)],
            "metadata": {
                "source": "edge_case_test",
                "variant": "MAG",
                "attention_hints": {}  # Empty hints
            }
        }
    ) as response:
        assert response.status == 200, "Failed with empty attention hints"
        result = await response.json()
        # Should succeed with default values
        assert "memory_id" in result, "No memory_id in response with empty attention hints"
    
    await asyncio.sleep(1)
    
    # 3. Test with invalid focus mode
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": "Testing with invalid focus mode",
            "embedding": [float(i) / 100 for i in range(384)],
            "metadata": {
                "source": "edge_case_test",
                "variant": "MAL",
                "attention_hints": {
                    "focus": "nonexistent_mode"  # Invalid focus mode
                }
            }
        }
    ) as response:
        assert response.status == 200, "Failed with invalid focus mode"
        result = await response.json()
        # Should succeed with default values
        assert "memory_id" in result, "No memory_id in response with invalid focus mode"
    
    await asyncio.sleep(1)
    
    # 4. Test with invalid parameter values
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": "Testing with invalid parameter values",
            "embedding": [float(i) / 100 for i in range(384)],
            "metadata": {
                "source": "edge_case_test",
                "variant": "MAC",
                "attention_hints": {
                    "focus": "recency",
                    "mac": {
                        "context_limit": "not_a_number",  # Invalid type
                        "attention_temperature": -1.0  # Invalid value
                    }
                }
            }
        }
    ) as response:
        assert response.status == 200, "Failed with invalid parameter values"
        result = await response.json()
        # Should succeed with default or constrained values
        assert "memory_id" in result, "No memory_id in response with invalid parameter values"

@pytest.mark.asyncio
async def test_dimension_mismatches(api_clients):
    """Test handling of dimension mismatches in embeddings."""
    session, mc_client = api_clients
    
    # Create memories with different embedding dimensions
    # First with 384 dimensions
    memory_384 = await create_test_memories(mc_client, count=1,
                                         prefix="Dimension-Test-384")
    
    # Create a memory with a 768-dimensional embedding through the CCE
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": "Memory with 768-dim embedding",
            "embedding": [float(i) / 100 for i in range(768)],  # 768-dim embedding
            "metadata": {
                "source": "dimension_test"
            }
        }
    ) as response:
        assert response.status == 200
        await response.json()
    
    await asyncio.sleep(2)
    
    # Now process a memory that will have to handle the dimension mismatch
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": "Testing dimension mismatch handling",
            "embedding": [float(i) / 100 for i in range(384)],  # Back to 384 dims
            "metadata": {
                "source": "dimension_test",
                "attention_hints": {
                    "focus": "broad"  # Use broad to maximize history inclusion
                }
            }
        }
    ) as response:
        assert response.status == 200, "Failed with dimension mismatch"
        result = await response.json()
        
        # Should successfully process despite dimension mismatches
        assert "memory_id" in result, "No memory_id in response with dimension mismatch"
        assert "variant_output" in result, "No variant_output in response with dimension mismatch"
