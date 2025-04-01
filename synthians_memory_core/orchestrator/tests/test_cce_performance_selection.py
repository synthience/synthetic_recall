#!/usr/bin/env python

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, call
import json
import numpy as np
import aiohttp
import os
import sys

# Define test constants directly
CCE_URL = "http://localhost:8002"
MC_URL = "http://localhost:5010"
NM_URL = "http://localhost:8001"

# Create a fixture for API clients
@pytest.fixture
async def api_clients():
    """Create API client session for testing."""
    session = aiohttp.ClientSession()
    mc_client = session  # Simplified for testing
    
    yield session, mc_client
    
    # Cleanup
    await session.close()

# Mock data preparation
mock_mc_store = {"success": True, "memory_id": "mem-test", "embedding": [0.1]*768, "quickrecal_score": 0.5}
mock_nm_projections = {"success": True, "key_projection": [0.2]*128, "value_projection": [0.3]*768, "query_projection": [0.4]*128}
mock_nm_retrieve = {"success": True, "retrieved_embedding": [0.5]*768, "query_projection": [0.4]*128}
mock_mc_boost = {"success": True}
mock_llm_advice = {"store": True, "metadata_tags": [], "boost_score_mod": 0.0, "variant_hint": None, "attention_focus": "broad", "notes": "", "decision_trace": []}

# Performance test scenarios
HIGH_SURPRISE_UPDATES = [
    {"success": True, "loss": 0.8, "grad_norm": 5.0},
    {"success": True, "loss": 0.9, "grad_norm": 5.5},
    {"success": True, "loss": 0.85, "grad_norm": 5.2},
    {"success": True, "loss": 0.95, "grad_norm": 6.0},
    {"success": True, "loss": 0.9, "grad_norm": 5.8},
]

LOW_SURPRISE_UPDATES = [
    {"success": True, "loss": 0.05, "grad_norm": 0.1},
    {"success": True, "loss": 0.03, "grad_norm": 0.08},
    {"success": True, "loss": 0.04, "grad_norm": 0.12},
    {"success": True, "loss": 0.02, "grad_norm": 0.05},
    {"success": True, "loss": 0.03, "grad_norm": 0.07},
]

INCREASING_TREND_UPDATES = [
    {"success": True, "loss": 0.1, "grad_norm": 0.5},
    {"success": True, "loss": 0.2, "grad_norm": 1.0},
    {"success": True, "loss": 0.3, "grad_norm": 1.5},
    {"success": True, "loss": 0.4, "grad_norm": 2.0},
    {"success": True, "loss": 0.5, "grad_norm": 2.5},
]

DECREASING_TREND_UPDATES = [
    {"success": True, "loss": 0.5, "grad_norm": 2.5},
    {"success": True, "loss": 0.4, "grad_norm": 2.0},
    {"success": True, "loss": 0.3, "grad_norm": 1.5},
    {"success": True, "loss": 0.2, "grad_norm": 1.0},
    {"success": True, "loss": 0.1, "grad_norm": 0.5},
]


@pytest.mark.asyncio
async def test_cce_selects_mag_on_high_surprise(api_clients):
    """Verify CCE selects MAG when NM performance shows consistently high surprise."""
    update_call_count = 0
    session, mc_client = api_clients
    cce_process_url = f"{CCE_URL}/process_memory"

    with patch('aiohttp.ClientSession.request', new_callable=AsyncMock) as mock_request:
        async def side_effect(method, url, **kwargs):
            nonlocal update_call_count
            json_payload = kwargs.get('json', {})
            
            # Configure mock response
            resp = AsyncMock(spec=aiohttp.ClientResponse)
            resp.status = 200
            
            if "process_memory" in url and MC_URL in url:
                resp.json.return_value = mock_mc_store
            elif "get_projections" in url and NM_URL in url:
                resp.json.return_value = mock_nm_projections
            elif "update_memory" in url and NM_URL in url:
                # Return sequence of high surprise updates
                idx = min(update_call_count, len(HIGH_SURPRISE_UPDATES) - 1)
                resp.json.return_value = HIGH_SURPRISE_UPDATES[idx]
                update_call_count += 1
            elif "retrieve" in url and NM_URL in url:
                resp.json.return_value = mock_nm_retrieve
            elif "update_quickrecal_score" in url and MC_URL in url:
                resp.json.return_value = mock_mc_boost
            elif "chat/completions" in url: # Mock LLM
                resp.json.return_value = {"choices": [{"message": {"content": json.dumps(mock_llm_advice)}}]}
            else: # Default success response
                resp.json.return_value = {"success": True, "message": f"Default mock for {url}"}
            
            # Setup async context manager for response
            response_context = AsyncMock()
            response_context.__aenter__.return_value = resp
            response_context.__aexit__.return_value = None
            return response_context
        
        mock_request.side_effect = side_effect
        
        # Make multiple calls to build performance history
        final_response = None
        for i in range(5):
            async with session.post(cce_process_url, json={"content": f"High surprise test {i}"}) as response:
                assert response.status == 200
                final_response = await response.json()
            await asyncio.sleep(0.1)
        
        # Verify final response
        assert final_response is not None
        selector_decision = final_response.get("selector_decision", {})
        assert selector_decision.get("selected") == "MAG"
        assert "High Surprise" in selector_decision.get("reason", "")


@pytest.mark.asyncio
async def test_cce_selects_none_on_low_surprise(api_clients):
    """Verify CCE selects NONE when NM performance shows consistently low surprise."""
    update_call_count = 0
    session, mc_client = api_clients
    cce_process_url = f"{CCE_URL}/process_memory"

    with patch('aiohttp.ClientSession.request', new_callable=AsyncMock) as mock_request:
        async def side_effect(method, url, **kwargs):
            nonlocal update_call_count
            
            # Configure mock response
            resp = AsyncMock(spec=aiohttp.ClientResponse)
            resp.status = 200
            
            if "process_memory" in url and MC_URL in url:
                resp.json.return_value = mock_mc_store
            elif "get_projections" in url and NM_URL in url:
                resp.json.return_value = mock_nm_projections
            elif "update_memory" in url and NM_URL in url:
                # Return sequence of low surprise updates
                idx = min(update_call_count, len(LOW_SURPRISE_UPDATES) - 1)
                resp.json.return_value = LOW_SURPRISE_UPDATES[idx]
                update_call_count += 1
            elif "retrieve" in url and NM_URL in url:
                resp.json.return_value = mock_nm_retrieve
            elif "update_quickrecal_score" in url and MC_URL in url:
                resp.json.return_value = mock_mc_boost
            elif "chat/completions" in url: # Mock LLM
                resp.json.return_value = {"choices": [{"message": {"content": json.dumps(mock_llm_advice)}}]}
            else: # Default success response
                resp.json.return_value = {"success": True, "message": f"Default mock for {url}"}
            
            # Setup async context manager for response
            response_context = AsyncMock()
            response_context.__aenter__.return_value = resp
            response_context.__aexit__.return_value = None
            return response_context
        
        mock_request.side_effect = side_effect
        
        # Make multiple calls to build performance history
        final_response = None
        for i in range(5):
            async with session.post(cce_process_url, json={"content": f"Low surprise test {i}"}) as response:
                assert response.status == 200
                final_response = await response.json()
            await asyncio.sleep(0.1)
        
        # Verify final response
        assert final_response is not None
        selector_decision = final_response.get("selector_decision", {})
        assert selector_decision.get("selected") == "NONE"
        assert "Low Surprise" in selector_decision.get("reason", "")


@pytest.mark.asyncio
async def test_cce_selects_mag_on_increasing_trend(api_clients):
    """Verify CCE selects MAG when NM performance shows an increasing surprise trend."""
    update_call_count = 0
    session, mc_client = api_clients
    cce_process_url = f"{CCE_URL}/process_memory"

    with patch('aiohttp.ClientSession.request', new_callable=AsyncMock) as mock_request:
        async def side_effect(method, url, **kwargs):
            nonlocal update_call_count
            
            # Configure mock response
            resp = AsyncMock(spec=aiohttp.ClientResponse)
            resp.status = 200
            
            if "process_memory" in url and MC_URL in url:
                resp.json.return_value = mock_mc_store
            elif "get_projections" in url and NM_URL in url:
                resp.json.return_value = mock_nm_projections
            elif "update_memory" in url and NM_URL in url:
                # Return sequence of increasing trend updates
                idx = min(update_call_count, len(INCREASING_TREND_UPDATES) - 1)
                resp.json.return_value = INCREASING_TREND_UPDATES[idx]
                update_call_count += 1
            elif "retrieve" in url and NM_URL in url:
                resp.json.return_value = mock_nm_retrieve
            elif "update_quickrecal_score" in url and MC_URL in url:
                resp.json.return_value = mock_mc_boost
            elif "chat/completions" in url: # Mock LLM
                resp.json.return_value = {"choices": [{"message": {"content": json.dumps(mock_llm_advice)}}]}
            else: # Default success response
                resp.json.return_value = {"success": True, "message": f"Default mock for {url}"}
            
            # Setup async context manager for response
            response_context = AsyncMock()
            response_context.__aenter__.return_value = resp
            response_context.__aexit__.return_value = None
            return response_context
        
        mock_request.side_effect = side_effect
        
        # Make multiple calls to build performance history
        final_response = None
        for i in range(5):
            async with session.post(cce_process_url, json={"content": f"Increasing trend test {i}"}) as response:
                assert response.status == 200
                final_response = await response.json()
            await asyncio.sleep(0.1)
        
        # Verify final response
        assert final_response is not None
        selector_decision = final_response.get("selector_decision", {})
        assert selector_decision.get("selected") == "MAG"
        assert "Increasing Surprise" in selector_decision.get("reason", "")


@pytest.mark.asyncio
async def test_cce_selects_mal_on_decreasing_trend(api_clients):
    """Verify CCE selects MAL when NM performance shows a decreasing surprise trend in moderate range."""
    update_call_count = 0
    session, mc_client = api_clients
    cce_process_url = f"{CCE_URL}/process_memory"

    with patch('aiohttp.ClientSession.request', new_callable=AsyncMock) as mock_request:
        async def side_effect(method, url, **kwargs):
            nonlocal update_call_count
            
            # Configure mock response
            resp = AsyncMock(spec=aiohttp.ClientResponse)
            resp.status = 200
            
            if "process_memory" in url and MC_URL in url:
                resp.json.return_value = mock_mc_store
            elif "get_projections" in url and NM_URL in url:
                resp.json.return_value = mock_nm_projections
            elif "update_memory" in url and NM_URL in url:
                # Return sequence of decreasing trend updates
                idx = min(update_call_count, len(DECREASING_TREND_UPDATES) - 1)
                resp.json.return_value = DECREASING_TREND_UPDATES[idx]
                update_call_count += 1
            elif "retrieve" in url and NM_URL in url:
                resp.json.return_value = mock_nm_retrieve
            elif "update_quickrecal_score" in url and MC_URL in url:
                resp.json.return_value = mock_mc_boost
            elif "chat/completions" in url: # Mock LLM
                resp.json.return_value = {"choices": [{"message": {"content": json.dumps(mock_llm_advice)}}]}
            else: # Default success response
                resp.json.return_value = {"success": True, "message": f"Default mock for {url}"}
            
            # Setup async context manager for response
            response_context = AsyncMock()
            response_context.__aenter__.return_value = resp
            response_context.__aexit__.return_value = None
            return response_context
        
        mock_request.side_effect = side_effect
        
        # Make multiple calls to build performance history
        final_response = None
        for i in range(5):
            async with session.post(cce_process_url, json={"content": f"Decreasing trend test {i}"}) as response:
                assert response.status == 200
                final_response = await response.json()
            await asyncio.sleep(0.1)
        
        # Verify final response
        assert final_response is not None
        selector_decision = final_response.get("selector_decision", {})
        assert selector_decision.get("selected") == "MAL"
        assert "Decreasing" in selector_decision.get("reason", "")
