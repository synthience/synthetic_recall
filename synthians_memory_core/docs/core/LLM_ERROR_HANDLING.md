# LLM Guidance System Error Handling (Phase 5.7.3)

## Overview

This document details the comprehensive error handling system implemented for the LLM guidance component of the Synthians cognitive architecture. These improvements enhance reliability, provide better debugging information, and ensure graceful degradation when external LLM services are unavailable or return unexpected responses.

## Key Improvements

### 1. Structured Exception Hierarchy

Implemented a clear exception handling structure in `MemoryLLMRouter.request_llama_guidance()` that prioritizes specific exceptions before more general ones:

- Network connectivity issues (ClientConnectorError, TimeoutError)
- HTTP status errors (non-200 responses)
- Response parsing errors (JSON decoding)
- Schema validation errors (jsonschema validation)
- General exceptions (as a fallback)

This hierarchy ensures proper identification of error types and appropriate recovery mechanisms.

### 2. Enhanced Retry Logic

- Implemented retry logic for transient failures (timeouts, connection issues)
- Added configurable retry parameters:
  - `retry_attempts`: Number of retry attempts (default: 2)
  - `retry_delay`: Base delay between retries in seconds (default: 1.0)
  - Uses exponential backoff with jitter for optimal retry timing
- Clear logging of retry attempts and outcomes

### 3. Detailed Error Reporting

- Improved error messages with context about what failed
- Enhanced logging with detailed error information, including:
  - HTTP status codes
  - Error response bodies
  - Schema validation errors
  - Exception details
- Added decision trace entries to document error handling path

### 4. Robust Response Parsing

- Enhanced JSON response handling with proper error trapping
- Added schema validation using jsonschema
- Fixed edge cases in async response handling for both text and JSON formats
- Proper handling of empty or malformed responses
- Fixed prompt formatting by correctly escaping braces `{{ }}` in JSON examples within the prompt template

### 5. Graceful Fallbacks

- Implemented `_get_default_llm_guidance()` method (renamed from previous `_default_advice`) for consistent fallback responses
- Fallback responses include error reason in notes field
- Decision trace includes information about fallback reason
- All failures return a properly structured response object
- Ensured specific error reasons are correctly captured and passed to the default advice function

### 6. Testing Infrastructure Improvements

- Fixed mock response fixtures to correctly return JSON as a string via `.text()` method, resolving TypeError issues
- Enhanced mock setup to provide both `.text()` and `.json()` methods with proper response structures
- Updated test assertions to compare specific fields rather than entire dictionaries
- Added specific checks for dynamically added fields like `decision_trace`
- Improved `_summarize_history_blended` error handling and corresponding test assertions
- Ensured proper resetting of mocks between tests to maintain clean test state

## Testing Strategy

Comprehensive tests have been implemented to verify error handling capabilities:

- `test_json_error_handling`: Tests JSON parsing errors
- `test_malformed_response_handling`: Tests malformed content in responses
- `test_schema_mismatch_handling`: Tests schema validation failures
- `test_connection_error_handling`: Tests network connectivity issues
- `test_timeout_handling`: Tests request timeout scenarios
- `test_multiple_retries_fail`: Tests exhaustion of retry attempts

Each test verifies:
1. Proper error detection
2. Correct retry behavior
3. Appropriate fallback response
4. Accurate error messaging

## Monitoring and Debugging

To monitor LLM guidance system health and troubleshoot issues:

1. Check logs for `synthians_memory_core.orchestrator.memory_logic_proxy` entries
2. Review error patterns in the notes field of LLM guidance responses
3. Examine decision_trace arrays for detailed processing information
4. Validate LLM endpoint connectivity and API compatibility

## Future Improvements

- Consider implementing circuit breaker pattern for persistent LLM service failures
- Add metrics collection for error rates and retry statistics
- Enhance local fallback capabilities with simpler models or rules-based systems
- Develop more sophisticated response validation beyond schema checking

---

*This documentation represents the state of the LLM error handling system as of Phase 5.7.3, April 2025.*
