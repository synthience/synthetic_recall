# synthians_memory_core/orchestrator/tests/test_memory_llm_router.py

import pytest
import pytest_asyncio
import asyncio
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock, ANY, call 
from typing import Dict, Any as TypingAny, Optional, List
import aiohttp

# Import memory_logic_proxy directly
from synthians_memory_core.orchestrator.memory_logic_proxy import MemoryLLMRouter

# Create a mock exception class to avoid aiohttp's ClientConnectorError.__str__ issue
class MockClientError(Exception):
    """Mock client error that doesn't break when stringified in error handling"""
    def __init__(self, message):
        self.message = message
        super().__init__(message)
    
    def __str__(self):
        return f"Mock Client Error: {self.message}"

# Fixture for testing parameters
@pytest.fixture
def sample_metadata():
    # Use keys expected by the router's prompt template
    return {
        "task_type": "explanation",
        "user_emotion": "curiosity", # Correct key
        "complexity": 0.75 # Example, not directly used in default prompt
    }

@pytest.fixture
def sample_nm_feedback():
    # Use keys expected by the router's prompt template
    return {
        "loss": 0.25,         # Correct key
        "grad_norm": 0.18     # Correct key
    }

@pytest.fixture
def sample_nm_performance():
    """Performance metrics with trend data for Phase 5.6."""
    return {
        "loss": 0.25,
        "grad_norm": 0.18,
        "avg_loss": 0.32,
        "avg_grad_norm": 0.22,
        "sample_count": 15,
        "std_dev_loss": 0.04,
        "confidence_level": "high",
        "trend_status": "decreasing",
        "trend_increasing": False,
        "trend_decreasing": True,
        "trend_slope": -0.08
    }

@pytest_asyncio.fixture
async def mock_aiohttp_session():
    """Mock aiohttp.ClientSession for tests."""
    # Create a proper mock that can be awaited
    mock_response = AsyncMock()
    mock_response.status = 200
    
    # Create context manager mock
    context_manager = AsyncMock()
    context_manager.__aenter__.return_value = mock_response
    
    # Create session mock
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.post.return_value = context_manager
    mock_session.closed = False
    mock_session.close = AsyncMock()

    # Patch the class, returning our instance
    with patch('aiohttp.ClientSession', return_value=mock_session) as patched_session_class:
        yield mock_session # Yield the instance for the test to use if needed

# --- CORRECTED FIXTURES ---
@pytest.fixture
def memory_llm_router():
    """Basic MemoryLLMRouter fixture with default settings."""
    # Use correct __init__ arguments
    return MemoryLLMRouter(
        mode="llmstudio", # Correct parameter name
        llama_endpoint="http://localhost:1234/v1/chat/completions", # Correct parameter name
        llama_model="test_model", # Correct parameter name
        retry_attempts=1, # Correct parameter name
        timeout=5.0 # Correct parameter name
    )

@pytest.fixture
def disabled_memory_llm_router():
    """MemoryLLMRouter fixture with disabled setting."""
    # Use correct __init__ arguments, map 'disabled=True' to 'mode="disabled"'
    return MemoryLLMRouter(
        mode="disabled", # Correct parameter name
        llama_endpoint="http://localhost:1234/v1/chat/completions", # Correct parameter name
        llama_model="test_model", # Correct parameter name
        retry_attempts=1,
        timeout=5.0
    )
# --- END CORRECTED FIXTURES ---

# --- Test Class ---
class TestMemoryLLMRouter:

    # Test uses correct args now
    def test_initialization(self):
        """Test basic initialization of MemoryLLMRouter."""
        router = MemoryLLMRouter(
            mode="llmstudio",
            llama_endpoint="http://test.endpoint/v1/chat/completions",
            llama_model="test_model",
            retry_attempts=5,
            timeout=10.0
        )

        assert router.mode == "llmstudio"
        assert router.llama_endpoint == "http://test.endpoint/v1/chat/completions"
        assert router.llama_model == "test_model"
        assert router.retry_attempts == 5
        assert router.timeout == 10.0
        assert router.session is None

    # Test uses correct args now
    @pytest.mark.asyncio
    async def test_disabled_mode(self, disabled_memory_llm_router, sample_metadata, sample_nm_feedback):
        """Test that router returns default advice when disabled."""
        result = await disabled_memory_llm_router.request_llama_guidance(
            user_input="Test query",
            nm_feedback=sample_nm_feedback,
            metadata=sample_metadata, # Pass the fixture directly
            current_variant="MAC",
            history_summary="No history"
        )

        # Assert against the actual default advice structure
        expected_default = disabled_memory_llm_router._default_advice("Router not in llmstudio mode")
        # Compare relevant fields, ignore trace for simplicity or use ANY
        assert result['store'] == expected_default['store']
        assert result['notes'] == expected_default['notes']
        assert result['variant_hint'] == expected_default['variant_hint']

    @pytest.mark.asyncio
    async def test_successful_guidance(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_feedback):
        """Test successful guidance request and response parsing."""
        # Set up the mock response
        successful_advice = {
            "store": True,
            "metadata_tags": ["explanation", "quantum"],
            "boost_score_mod": 0.2,
            "variant_hint": "MAL",
            "attention_focus": "relevance",
            "notes": "Input is explanatory and novel.",
            "decision_trace": ["Identified task type: explanation", "Surprise level moderate", "Selected MAL"]
        }
        
        # Setup the response json to return the expected structure
        mock_response = mock_aiohttp_session.post.return_value.__aenter__.return_value
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(successful_advice)}}]
        }

        result = await memory_llm_router.request_llama_guidance(
            user_input="Explain quantum entanglement",
            nm_feedback=sample_nm_feedback,
            metadata=sample_metadata, # Pass the fixture directly
            current_variant="MAC",
            history_summary="Recent discussion on physics."
        )

        # Verify the result matches the mock response content
        assert result is not None
        assert result["store"] == successful_advice["store"]
        assert result["metadata_tags"] == successful_advice["metadata_tags"]
        assert result["boost_score_mod"] == successful_advice["boost_score_mod"]
        assert result["variant_hint"] == successful_advice["variant_hint"]
        assert result["attention_focus"] == successful_advice["attention_focus"]
        assert result["notes"] == successful_advice["notes"]
        # Don't test the decision_trace exactly as it will include timestamps and generated content
        assert "Starting LLM guidance request" in result["decision_trace"]

        # Verify the API was called correctly
        mock_aiohttp_session.post.assert_called_once()
        call_args = mock_aiohttp_session.post.call_args
        url, kwargs = call_args[0][0], call_args[1]
        assert url == memory_llm_router.llama_endpoint
        assert "json" in kwargs
        payload = kwargs["json"]
        assert payload["model"] == memory_llm_router.llama_model
        assert payload["temperature"] <= 0.3
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"] == MemoryLLMRouter.DEFAULT_LLM_SCHEMA

    @pytest.mark.asyncio
    async def test_error_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_feedback):
        """Test handling of API errors after retries."""
        # Configure the mock post call to raise an error using our mock class
        mock_aiohttp_session.post.side_effect = MockClientError("Connection refused")
        memory_llm_router.retry_attempts = 1

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test error",
            nm_feedback=sample_nm_feedback,
            metadata=sample_metadata, # Pass the fixture directly
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice structure on error after retries
        expected_default = memory_llm_router._default_advice("LM Studio connection error")
        assert result["store"] == expected_default["store"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "connection error" in result["notes"].lower() or "Connection refused" in result["notes"]
        # Should call post twice (1 initial + 1 retry)
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_session_management(self, memory_llm_router):
        """Test proper management of the aiohttp session."""
        # Patch ClientSession to return a mock
        with patch('aiohttp.ClientSession') as mock_session_class:
            # Create two different mock instances to test properly
            mock_session1 = AsyncMock()
            mock_session1.closed = False
            mock_session1.close = AsyncMock()
            
            mock_session2 = AsyncMock()
            mock_session2.closed = False
            mock_session2.close = AsyncMock()
            
            # Set up the side effect to return different mocks on consecutive calls
            mock_session_class.side_effect = [mock_session1, mock_session2]
            
            assert memory_llm_router.session is None

            session1 = await memory_llm_router._get_session()
            assert session1 is mock_session1
            assert memory_llm_router.session is session1
            assert not session1.closed

            session2 = await memory_llm_router._get_session()
            assert session2 is session1  # Should still be the same session

            await memory_llm_router.close_session()
            assert memory_llm_router.session is None
            # Verify close was called
            mock_session1.close.assert_called_once()

            # Test getting a new session after closing
            session3 = await memory_llm_router._get_session()
            assert session3 is mock_session2  # Should be a new instance
            assert session3 is not session1  # And different from the first one
            assert not session3.closed
            
            await memory_llm_router.close_session()

    @pytest.mark.asyncio
    async def test_retry_logic(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_feedback):
        """Test the retry mechanism for failed requests."""
        memory_llm_router.retry_attempts = 2 # Allow 2 retries (3 attempts total)
        memory_llm_router.retry_delay = 0.01 # Faster retry for test

        # Define the sequence of responses/errors
        successful_advice = {
            "store": False, "metadata_tags": ["retry_test"], "boost_score_mod": -0.1,
            "variant_hint": "NONE", "attention_focus": "broad", "notes": "Retry succeeded",
            "decision_trace": ["LLM: Succeeded on retry"]
        }
        
        # Create a successful response for the third attempt
        success_context = AsyncMock()
        success_response = AsyncMock()
        success_response.status = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(successful_advice)}}]
        }
        success_context.__aenter__.return_value = success_response
        
        # Setup the sequence of side effects using our mock class
        mock_aiohttp_session.post.side_effect = [
            MockClientError("Connection refused"),
            asyncio.TimeoutError("Request timed out"),
            success_context
        ]

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test retry",
            nm_feedback=sample_nm_feedback,
            metadata=sample_metadata, # Pass fixture
            current_variant="MAG",
            history_summary=""
        )

        # Should have the result from the successful third attempt
        assert result is not None
        assert result["store"] == successful_advice["store"]
        assert result["metadata_tags"] == successful_advice["metadata_tags"]
        assert result["boost_score_mod"] == successful_advice["boost_score_mod"]
        assert result["variant_hint"] == successful_advice["variant_hint"]
        assert result["attention_focus"] == successful_advice["attention_focus"]

        # Verify that post was called 3 times
        assert mock_aiohttp_session.post.call_count == 3

    @pytest.mark.asyncio
    async def test_json_error_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_feedback):
        """Test handling of malformed JSON responses after retries."""
        memory_llm_router.retry_attempts = 1
        
        # Create a response with a JSON error
        bad_json_context = AsyncMock()
        bad_json_response = AsyncMock()
        bad_json_response.status = 200
        bad_json_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "{invalid:", 0)
        bad_json_response.text.return_value = "{invalid:"
        bad_json_context.__aenter__.return_value = bad_json_response
        
        # Setup the sequence of side effects
        mock_aiohttp_session.post.side_effect = [
            MockClientError("Connection refused"), 
            bad_json_context
        ]

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test JSON error",
            nm_feedback=sample_nm_feedback,
            metadata=sample_metadata, # Pass fixture
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice structure after retries fail on JSON error
        expected_default = memory_llm_router._default_advice("LLM JSON parse error")
        assert result["store"] == expected_default["store"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "Invalid JSON" in result["notes"] or "invalid json" in result["notes"].lower()
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_feedback):
        """Test handling of response with missing expected structure after retries."""
        memory_llm_router.retry_attempts = 1
        
        # Create a response with malformed content (missing choices key)
        malformed_context = AsyncMock()
        malformed_response = AsyncMock()
        malformed_response.status = 200
        malformed_response.json.return_value = {"unexpected_key": "value"}
        malformed_response.text.return_value = '{"unexpected_key": "value"}'
        malformed_context.__aenter__.return_value = malformed_response
        
        # Setup the sequence of side effects
        mock_aiohttp_session.post.side_effect = [
            MockClientError("Connection refused"), 
            malformed_context
        ]

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test malformed response",
            nm_feedback=sample_nm_feedback,
            metadata=sample_metadata, # Pass fixture
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice for malformed response after retry
        expected_default = memory_llm_router._default_advice("LLM response empty content")
        assert result["store"] == expected_default["store"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "empty content" in result["notes"].lower() or "missing content" in result["notes"].lower()
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_schema_mismatch_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_feedback):
        """Test handling of response JSON not matching the required schema after retries."""
        memory_llm_router.retry_attempts = 1
        
        # Create a response with schema mismatch (missing required fields)
        schema_mismatch_context = AsyncMock()
        schema_mismatch_response = AsyncMock()
        schema_mismatch_response.status = 200
        schema_mismatch_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"store": True})}}]
        }
        schema_mismatch_response.text.return_value = json.dumps({
            "choices": [{"message": {"content": json.dumps({"store": True})}}]
        })
        schema_mismatch_context.__aenter__.return_value = schema_mismatch_response
        
        # Setup the sequence of side effects
        mock_aiohttp_session.post.side_effect = [
            MockClientError("Connection refused"), 
            schema_mismatch_context
        ]

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test schema mismatch",
            nm_feedback=sample_nm_feedback,
            metadata=sample_metadata, # Pass fixture
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice when schema validation fails after retry
        expected_default = memory_llm_router._default_advice("LLM response missing keys")
        assert result["store"] == expected_default["store"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "missing keys" in result["notes"].lower() or "schema" in result["notes"].lower()
        assert result["metadata_tags"] == ["llm_guidance_failed"]
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_missing_content_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_feedback):
        """Test handling of response where the message content is missing."""
        memory_llm_router.retry_attempts = 1
        
        # Create a response with missing content field
        missing_content_context = AsyncMock()
        missing_content_response = AsyncMock()
        missing_content_response.status = 200
        missing_content_response.json.return_value = {
            "choices": [{"message": {"role": "assistant"}}]
        }
        missing_content_response.text.return_value = json.dumps({
            "choices": [{"message": {"role": "assistant"}}]
        })
        missing_content_context.__aenter__.return_value = missing_content_response
        
        # Setup the sequence of side effects
        mock_aiohttp_session.post.side_effect = [
            MockClientError("Connection refused"), 
            missing_content_context
        ]

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test missing content",
            nm_feedback=sample_nm_feedback,
            metadata=sample_metadata, # Pass fixture
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice when content is missing after retry
        expected_default = memory_llm_router._default_advice("LLM response empty content")
        assert result["store"] == expected_default["store"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "empty content" in result["notes"].lower() or "missing content" in result["notes"].lower()
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_feedback):
        """Test handling of timeout errors after retries."""
        memory_llm_router.retry_attempts = 1
        # Mock post to raise TimeoutError on both attempts
        mock_aiohttp_session.post.side_effect = asyncio.TimeoutError("Request timed out")

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test timeout",
            nm_feedback=sample_nm_feedback,
            metadata=sample_metadata, # Pass fixture
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice on timeout after retries
        expected_default = memory_llm_router._default_advice("LM Studio timeout")
        assert result["store"] == expected_default["store"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "timeout" in result["notes"].lower()
        assert mock_aiohttp_session.post.call_count == 2 # 1 initial + 1 retry

    @pytest.mark.asyncio
    async def test_multiple_retries_fail(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_feedback):
        """Test multiple retry attempts all failing."""
        memory_llm_router.retry_attempts = 2 # Allow 2 retries (3 attempts total)
        memory_llm_router.retry_delay = 0.01 # Faster retry for test

        # Setup the sequence of side effects with our mock class
        mock_aiohttp_session.post.side_effect = [
            MockClientError("Connection refused"),
            asyncio.TimeoutError("Request timed out"),
            MockClientError("Another connection error")
        ]

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test multiple retries fail",
            nm_feedback=sample_nm_feedback,
            metadata=sample_metadata, # Pass fixture
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice after all retries fail
        expected_default = memory_llm_router._default_advice("LM Studio connection error") # Uses last error type
        assert result["store"] == expected_default["store"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "connection error" in result["notes"].lower() or "Another connection error" in result["notes"]
        assert mock_aiohttp_session.post.call_count == 3 # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_phase_5_6_performance_metrics(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
        """Test that performance metrics are correctly included in the prompt and the correct model is used."""
        # Set up the mock response
        successful_advice = {
            "store": True,
            "metadata_tags": ["technical", "performance"],
            "boost_score_mod": -0.1,  # Negative because trend is decreasing
            "variant_hint": "NONE",  # NONE because low surprise is expected with decreasing trend
            "attention_focus": "relevance",
            "notes": "System is adapting well with decreasing loss.",
            "decision_trace": ["Identified decreasing trend", "High confidence in metrics"]
        }
        
        # Setup the response json to return the expected structure
        mock_response = mock_aiohttp_session.post.return_value.__aenter__.return_value
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(successful_advice)}}]
        }
        
        # Explicitly set the model to check in assertions
        memory_llm_router.llama_model = "bartowski/llama-3.2-1b-instruct"
        memory_llm_router.high_surprise_threshold = 0.5
        memory_llm_router.low_surprise_threshold = 0.1

        # Call the function with performance metrics
        result = await memory_llm_router.request_llama_guidance(
            user_input="Testing performance metrics",
            nm_performance=sample_nm_performance,
            metadata=sample_metadata,
            current_variant="MAC",
            history_summary="Recent system adaptation."
        )

        # Verify the API was called correctly
        mock_aiohttp_session.post.assert_called_once()
        call_args = mock_aiohttp_session.post.call_args
        url, kwargs = call_args[0][0], call_args[1]
        
        # Check that the model name is correct
        assert kwargs["json"]["model"] == "bartowski/llama-3.2-1b-instruct"
        
        # Check that the prompt includes performance metrics
        prompt = kwargs["json"]["messages"][0]["content"]
        assert "PROMPT VERSION: 5.6.3" in prompt
        assert "Average Loss: 0.3200" in prompt
        assert "Average Grad Norm: 0.2200" in prompt
        assert "Performance Trend: decreasing" in prompt
        assert "Sample Count: 15" in prompt
        assert "Standard Deviation (Loss): 0.0400" in prompt
        assert "System Confidence: high" in prompt
        
        # Check that threshold values are included in the heuristics
        assert "High surprise (loss/grad_norm > 0.50)" in prompt
        assert "Low surprise (loss/grad_norm < 0.10)" in prompt
        
        # Verify the result includes performance info in the decision trace
        assert "Performance metrics:" in result["decision_trace"][-1]
        assert "trend=decreasing" in result["decision_trace"][-1]
