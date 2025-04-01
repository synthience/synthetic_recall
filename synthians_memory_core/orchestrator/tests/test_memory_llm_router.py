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
    
    # Set default return values for both text() and json() methods
    # This ensures all tests have proper response handling
    default_json = {"choices": [{"message": {"content": "{}"}}]}
    mock_response.text = AsyncMock(return_value=json.dumps(default_json))
    mock_response.json = AsyncMock(return_value=default_json)
    
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
    async def test_disabled_mode(self, disabled_memory_llm_router, sample_metadata, sample_nm_performance):
        """Test that router returns default advice when disabled."""
        result = await disabled_memory_llm_router.request_llama_guidance(
            user_input="Test query",
            nm_performance=sample_nm_performance,
            metadata=sample_metadata, # Pass the fixture directly
            current_variant="MAC",
            history_summary="No history"
        )

        # Assert against the actual default advice structure
        expected_default = disabled_memory_llm_router._get_default_llm_guidance("Router not in llmstudio mode")
        # Compare relevant fields, ignore trace for simplicity or use ANY
        assert result['store'] == expected_default['store']
        assert result['notes'] == expected_default['notes']
        assert result['variant_hint'] == expected_default['variant_hint']

    @pytest.mark.asyncio
    async def test_successful_guidance(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
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
        
        # Reset the mock to ensure correct call count
        mock_aiohttp_session.post.reset_mock()
        
        # Setup the response with both text and json return values
        mock_response = mock_aiohttp_session.post.return_value.__aenter__.return_value
        mock_response.status = 200  # Ensure status is 200
        response_json = {
            "choices": [{"message": {"content": json.dumps(successful_advice)}}]
        }
        mock_response.text = AsyncMock(return_value=json.dumps(response_json))
        mock_response.json.return_value = response_json

        result = await memory_llm_router.request_llama_guidance(
            user_input="Explain quantum entanglement",
            nm_performance=sample_nm_performance,
            metadata=sample_metadata,
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
        # Ensure decision_trace contains both original elements and added ones
        assert any(trace for trace in result["decision_trace"] if "LLM guidance request successful" in trace)
        assert any(trace for trace in result["decision_trace"] if "Performance metrics" in trace)

        # Verify the API was called exactly once
        mock_aiohttp_session.post.assert_called_once()
        call_args = mock_aiohttp_session.post.call_args
        url, kwargs = call_args[0][0], call_args[1]
        assert url == memory_llm_router.llama_endpoint
        assert "json" in kwargs
        payload = kwargs["json"]
        assert payload["model"] == memory_llm_router.llama_model
        assert payload["temperature"] <= 0.3
        assert payload["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_error_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
        """Test handling of API errors after retries."""
        # Configure the mock post call to raise an error using our mock class
        mock_aiohttp_session.post.side_effect = MockClientError("Connection refused")
        memory_llm_router.retry_attempts = 1

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test error",
            nm_performance=sample_nm_performance,
            metadata=sample_metadata, # Pass the fixture directly
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice structure on error after retries
        expected_default = memory_llm_router._get_default_llm_guidance("LM Studio connection error")
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
    async def test_retry_logic(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
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
        # Set both text and json return values
        success_response.text = AsyncMock(return_value=json.dumps({
            "choices": [{"message": {"content": json.dumps(successful_advice)}}]
        }))
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
            nm_performance=sample_nm_performance,
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
    async def test_phase_5_6_performance_metrics(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
        """Test that performance metrics are correctly included in the prompt and the correct model is used."""
        # Reset the mock to ensure correct call count
        mock_aiohttp_session.post.reset_mock()
        mock_aiohttp_session.post.side_effect = None  # Clear any side effects from other tests
        
        # Set up the mock to check what was sent to the API
        successful_advice = {
            "store": True,
            "metadata_tags": ["metrics", "test"],
            "boost_score_mod": 0.3,
            "variant_hint": "MAG",
            "attention_focus": "relevance",
            "notes": "Based on performance metrics"
            # No decision_trace here in the expected dict
        }
        
        # Setup the response with both text and json return values
        mock_response = mock_aiohttp_session.post.return_value.__aenter__.return_value
        mock_response.status = 200  # Ensure status is 200
        response_json = {
            "choices": [{"message": {"content": json.dumps(successful_advice)}}]
        }
        mock_response.text = AsyncMock(return_value=json.dumps(response_json))
        mock_response.json.return_value = response_json

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test with metrics",
            nm_performance=sample_nm_performance,  # Using performance metrics
            metadata=sample_metadata,
            current_variant="NONE",
            history_summary="Sample history"
        )

        # Verify the call count and payload
        assert mock_aiohttp_session.post.call_count == 1, f"Expected 1 call, got {mock_aiohttp_session.post.call_count}"
        
        call_args = mock_aiohttp_session.post.call_args
        url, kwargs = call_args[0][0], call_args[1]
        payload = kwargs["json"]
        
        # Check model
        assert payload["model"] == memory_llm_router.llama_model
        
        # Check that metrics are included in the prompt
        prompt_content = payload["messages"][0]["content"]
        assert "Average Loss: 0.32" in prompt_content
        assert "Average Grad Norm: 0.22" in prompt_content
        assert "Sample Count: 15" in prompt_content
        assert "Standard Deviation (Loss): 0.04" in prompt_content
        assert "System Confidence: high" in prompt_content
        assert "Performance Trend: decreasing" in prompt_content
        
        # UPDATED ASSERTION: Compare relevant fields, exclude decision_trace
        assert result is not None
        for key, value in successful_advice.items():
            assert result.get(key) == value, f"Mismatch on key '{key}'"
        
        # Add specific checks for decision_trace
        assert "decision_trace" in result
        assert isinstance(result["decision_trace"], list)
        assert len(result["decision_trace"]) >= 2  # Should have at least success msg + perf summary
        assert "LLM guidance request successful." in result["decision_trace"][0]
        assert any("Performance metrics:" in trace for trace in result["decision_trace"])

    @pytest.mark.asyncio
    async def test_json_error_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
        """Test handling of JSON decoding errors in the response."""
        memory_llm_router.retry_attempts = 1
        
        # Reset the mock to ensure correct call count
        mock_aiohttp_session.post.reset_mock()
        
        # Create a response with invalid JSON
        bad_json_context = AsyncMock()
        bad_json_response = AsyncMock()
        bad_json_response.status = 200
        bad_json_response.text = AsyncMock(return_value="{Invalid JSON}")
        bad_json_response.json = AsyncMock(side_effect=json.JSONDecodeError("Expecting property name", "{Invalid JSON}", 1))
        bad_json_context.__aenter__.return_value = bad_json_response
        
        # Setup the sequence of side effects
        side_effects = [
            MockClientError("Connection refused"), 
            bad_json_context
        ]
        mock_aiohttp_session.post.side_effect = side_effects

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test JSON error",
            nm_performance=sample_nm_performance,
            metadata=sample_metadata,
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice after retries fail on JSON error
        expected_default = memory_llm_router._get_default_llm_guidance("LLM JSON parse error")
        assert result["store"] == expected_default["store"]
        assert result["variant_hint"] == expected_default["variant_hint"]
        assert "LLM Guidance Error:" in result["notes"]
        
        # The actual error message could be either format based on where the JSON error occurs
        assert ("JSON parse error" in result["notes"] or 
                "Response processing error" in result["notes"] or 
                "Expecting property name" in result["notes"])
        
        # Verify the post was called twice (initial + retry)
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
        """Test handling of response with missing expected structure after retries."""
        memory_llm_router.retry_attempts = 1
        
        # Reset the mock to ensure correct call count
        mock_aiohttp_session.post.reset_mock()
        
        # Create a response with malformed content (missing choices key)
        malformed_context = AsyncMock()
        malformed_response = AsyncMock()
        malformed_response.status = 200
        malformed_response.json.return_value = {"unexpected_key": "value"}
        malformed_response.text.return_value = json.dumps({
            "unexpected_key": "value"
        })
        malformed_context.__aenter__.return_value = malformed_response
        
        # Setup the sequence of side effects
        side_effects = [
            MockClientError("Connection refused"), 
            malformed_context
        ]
        mock_aiohttp_session.post.side_effect = side_effects

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test malformed response",
            nm_performance=sample_nm_performance,
            metadata=sample_metadata,
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice for malformed response after retry
        expected_default = memory_llm_router._get_default_llm_guidance("LLM response empty content")
        assert result["store"] == expected_default["store"]
        assert result["variant_hint"] == expected_default["variant_hint"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "empty content" in result["notes"].lower()
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_schema_mismatch_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
        """Test handling of response that fails schema validation after retries."""
        memory_llm_router.retry_attempts = 1
        
        # Reset the mock to ensure correct call count
        mock_aiohttp_session.post.reset_mock()
        
        # Create a response with missing required fields
        schema_mismatch_context = AsyncMock()
        schema_mismatch_response = AsyncMock()
        schema_mismatch_response.status = 200
        
        # Setup the response with an incomplete schema that will fail validation
        incomplete_advice = {"store": True} # Missing required fields
        response_json = {"choices": [{"message": {"content": json.dumps(incomplete_advice)}}]}
        
        schema_mismatch_response.text = AsyncMock(return_value=json.dumps(response_json))
        schema_mismatch_response.json.return_value = response_json
        schema_mismatch_context.__aenter__.return_value = schema_mismatch_response
        
        # Setup the sequence of side effects
        side_effects = [
            MockClientError("Connection refused"), 
            schema_mismatch_context
        ]
        mock_aiohttp_session.post.side_effect = side_effects

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test schema mismatch",
            nm_performance=sample_nm_performance,
            metadata=sample_metadata,
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice when schema validation fails after retry
        expected_default = memory_llm_router._get_default_llm_guidance("LLM response missing keys")
        assert result["store"] == expected_default["store"]
        assert result["variant_hint"] == expected_default["variant_hint"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "missing keys" in result["notes"].lower() or "schema" in result["notes"].lower()
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_missing_content_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
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
            nm_performance=sample_nm_performance,
            metadata=sample_metadata, # Pass fixture
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice when content is missing after retry
        expected_default = memory_llm_router._get_default_llm_guidance("LLM response empty content")
        assert result["store"] == expected_default["store"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "empty content" in result["notes"].lower() or "missing content" in result["notes"].lower()
        assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_handling(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
        """Test handling of timeout errors after retries."""
        memory_llm_router.retry_attempts = 1
        # Mock post to raise TimeoutError on both attempts
        mock_aiohttp_session.post.side_effect = asyncio.TimeoutError("Request timed out")

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test timeout",
            nm_performance=sample_nm_performance,
            metadata=sample_metadata, # Pass fixture
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice on timeout after retries
        expected_default = memory_llm_router._get_default_llm_guidance("LM Studio timeout")
        assert result["store"] == expected_default["store"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "timeout" in result["notes"].lower()
        assert mock_aiohttp_session.post.call_count == 2 # 1 initial + 1 retry

    @pytest.mark.asyncio
    async def test_multiple_retries_fail(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
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
            nm_performance=sample_nm_performance,
            metadata=sample_metadata, # Pass fixture
            current_variant="MAC",
            history_summary=""
        )

        # Should return default advice after all retries fail
        expected_default = memory_llm_router._get_default_llm_guidance("LM Studio connection error") # Uses last error type
        assert result["store"] == expected_default["store"]
        assert "LLM Guidance Error:" in result["notes"]
        assert "connection error" in result["notes"].lower() or "Another connection error" in result["notes"]
        assert mock_aiohttp_session.post.call_count == 3 # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_summarize_history_blended(self, memory_llm_router):
        """Test the blended history summarization method."""
        import numpy as np
        
        # Create mock history entries in the format of ContextTuple
        # (timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)
        mock_history = [
            # Create 3 entries with different norms
            (
                1648000000.0,  # timestamp
                "mem123",      # memory_id
                np.array([0.1, 0.2, 0.3, 0.4]),  # x_t - input embedding
                np.array([0.2, 0.3, 0.4, 0.5]),  # k_t - key projection
                np.array([0.3, 0.4, 0.5, 0.6]),  # v_t - value projection
                np.array([0.4, 0.5, 0.6, 0.7]),  # q_t - query projection
                np.array([0.5, 0.6, 0.7, 0.8]),  # y_t - output embedding
            ),
            (
                1648000001.0,
                "mem456",
                np.array([0.2, 0.3, 0.4, 0.5]),
                np.array([0.3, 0.4, 0.5, 0.6]),
                np.array([0.4, 0.5, 0.6, 0.7]),
                np.array([0.5, 0.6, 0.7, 0.8]),
                np.array([0.3, 0.4, 0.5, 0.6]),  # Different output to test surprise
            ),
            (
                1648000002.0,
                "mem789",
                np.array([0.5, 0.6, 0.7, 0.8]),
                np.array([0.6, 0.7, 0.8, 0.9]),
                np.array([0.7, 0.8, 0.9, 1.0]),
                np.array([0.8, 0.9, 1.0, 1.1]),
                np.array([0.9, 1.0, 1.1, 1.2]),
            )
        ]
        
        # Call the summarization method
        summary = memory_llm_router._summarize_history_blended(mock_history)
        
        # Verify the summary contains the expected elements
        assert summary is not None
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # Check that it contains the pattern analysis and embedding norm information
        assert "ID:mem789" in summary
        assert "ID:mem456" in summary
        assert "ID:mem123" in summary
        assert "In:" in summary  # Should have input norm
        assert "Out:" in summary  # Should have output norm
        assert "Diff:" in summary  # Should have difference norm
        assert "SR:" in summary  # Should have surprise ratio
        
        # Test empty history case
        empty_summary = memory_llm_router._summarize_history_blended([])
        assert empty_summary == "[No history available]"
        
        # Test error handling
        bad_history = [(1648000000.0, "bad_mem", None, None, None, None, None)]
        error_summary = memory_llm_router._summarize_history_blended(bad_history)
        expected_error_msg = "[History Summary Error: Could not process entries]"
        assert expected_error_msg in error_summary, f"Expected '{expected_error_msg}' in '{error_summary}'"

    @pytest.mark.asyncio
    async def test_history_summary_in_prompt(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
        """Test that history summary is correctly included in the prompt."""
        # Reset the mock to ensure correct call count
        mock_aiohttp_session.post.reset_mock()
        mock_aiohttp_session.post.side_effect = None  # Clear any side effects from other tests
        
        # Set up the mock to check what was sent to the API
        successful_advice = {
            "store": True,
            "metadata_tags": ["history", "test"],
            "boost_score_mod": 0.2,
            "variant_hint": "MAC",
            "attention_focus": "recency",
            "notes": "Based on history context"
            # No decision_trace here in the expected dict
        }
        
        # Setup the response with both text and json return values
        mock_response = mock_aiohttp_session.post.return_value.__aenter__.return_value
        mock_response.status = 200  # Ensure status is 200
        response_json = {
            "choices": [{"message": {"content": json.dumps(successful_advice)}}]
        }
        mock_response.text = AsyncMock(return_value=json.dumps(response_json))
        mock_response.json.return_value = response_json

        # Create a detailed history summary
        test_history_summary = """[3] ID:mem123 | In:0.52 Out:0.78 Diff:0.34 SR:0.65
[2] ID:mem456 | In:0.71 Out:0.65 Diff:0.22 SR:0.31
[1] ID:mem789 | In:1.34 Out:1.21 Diff:0.18 SR:0.13

[Pattern: Decreasing surprise - likely reinforcement of familiar concepts]"""

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test with history",
            nm_performance=sample_nm_performance,
            metadata=sample_metadata,
            current_variant="MAC",
            history_summary=test_history_summary  # Pass the detailed history summary
        )

        # Verify the call count and payload
        assert mock_aiohttp_session.post.call_count == 1, f"Expected 1 call, got {mock_aiohttp_session.post.call_count}"
        
        call_args = mock_aiohttp_session.post.call_args
        url, kwargs = call_args[0][0], call_args[1]
        payload = kwargs["json"]
        
        # Check that history is included in the prompt
        prompt_content = payload["messages"][0]["content"]
        assert "RECENT HISTORY SUMMARY:" in prompt_content
        assert test_history_summary in prompt_content
        
        # Verify the prompt has instructions for interpreting history
        assert "Look for patterns in the embedding norms" in prompt_content
        
        # UPDATED ASSERTION: Compare relevant fields, exclude decision_trace
        assert result is not None
        for key, value in successful_advice.items():
            assert result.get(key) == value, f"Mismatch on key '{key}'"
        
        # Add specific checks for decision_trace
        assert "decision_trace" in result
        assert isinstance(result["decision_trace"], list)
        assert len(result["decision_trace"]) >= 1
        assert "LLM guidance request successful." in result["decision_trace"][0]
        
    @pytest.mark.asyncio
    async def test_meta_reasoning_field(self, memory_llm_router, mock_aiohttp_session, sample_metadata, sample_nm_performance):
        """Test handling of the meta_reasoning field in responses."""
        # Set up the mock with response including meta_reasoning
        advice_with_meta_reasoning = {
            "store": True,
            "metadata_tags": ["meta", "reasoning"],
            "boost_score_mod": 0.3,
            "variant_hint": "MAG",
            "attention_focus": "relevance",
            "notes": "Basic note",
            "decision_trace": ["Step 1", "Step 2"],
            "meta_reasoning": "This is detailed reasoning explaining why I chose MAG variant based on the increasing surprise trend in recent interactions."
        }
        
        # Setup the response
        mock_response = mock_aiohttp_session.post.return_value.__aenter__.return_value
        # Set both text and json return values
        mock_response.text.return_value = json.dumps({
            "choices": [{
                "message": {"content": json.dumps(advice_with_meta_reasoning)}
            }]
        })
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": json.dumps(advice_with_meta_reasoning)}
            }]
        }

        result = await memory_llm_router.request_llama_guidance(
            user_input="Test with meta reasoning",
            nm_performance=sample_nm_performance,
            metadata=sample_metadata,
            current_variant="NONE",
            history_summary="Sample history"
        )

        # Verify the schema definition includes meta_reasoning
        payload = mock_aiohttp_session.post.call_args[1]["json"]
        schema = payload["response_format"]["json_schema"]["schema"]
        assert "meta_reasoning" in schema["properties"]
        
        # Check that meta_reasoning is passed through
        assert "meta_reasoning" in result
        assert result["meta_reasoning"] == advice_with_meta_reasoning["meta_reasoning"]
        
        # Test default advice has meta_reasoning field too
        with patch.object(mock_aiohttp_session, 'post', side_effect=Exception("Test error")):
            default_result = await memory_llm_router.request_llama_guidance(
                user_input="Error test",
                nm_performance=sample_nm_performance,
                metadata=sample_metadata,
                current_variant="NONE"
            )
            assert "meta_reasoning" in default_result
            print(f"EXPECTED META: 'automatically generated' to be in: '{default_result['meta_reasoning']}'")
            assert "automatically generated" in default_result["meta_reasoning"].lower()
