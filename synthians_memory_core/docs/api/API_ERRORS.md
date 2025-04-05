# API Error Responses for Synthians Cognitive Architecture

This document details the possible error responses from the Synthians Cognitive Architecture API endpoints, with a special focus on the new Phase 5.9 explainability and diagnostics endpoints.

## Common Error Response Format

All API endpoints follow a common error response format:

```json
{
  "success": false,
  "error": "Human-readable error message",
  "code": "ERROR_CODE",  // Optional machine-readable error code
  "details": {}          // Optional additional error details
}
```

## Standard HTTP Status Codes

| Code | Meaning | Common Scenarios |
|------|---------|-----------------|
| 200 | OK | Successful operation |
| 400 | Bad Request | Invalid parameters, malformed request |
| 403 | Forbidden | Feature flag disabled, authorization failed |
| 404 | Not Found | Resource doesn't exist |
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | Component (MC, NM, CCE) not available |

## Phase 5.9 Explainability API Errors

### GET `/assemblies/{id}/explain_activation`

| Status | Code | Message | Description |
|--------|------|---------|-------------|
| 403 | `EXPLAINABILITY_DISABLED` | "Explainability is disabled. Set ENABLE_EXPLAINABILITY=true in configuration." | Feature flag is not enabled |
| 404 | `ASSEMBLY_NOT_FOUND` | "Assembly with ID {id} not found." | The specified assembly doesn't exist |
| 404 | `MEMORY_NOT_FOUND` | "Memory with ID {memory_id} not found." | The specified memory doesn't exist (if memory_id provided) |
| 400 | `INVALID_MEMORY_ID` | "Invalid memory_id parameter." | memory_id parameter is malformed |
| 500 | `CALCULATION_ERROR` | "Error calculating activation explanation." | Internal error during explanation generation |

Example error response:
```json
{
  "success": false,
  "error": "Explainability is disabled. Set ENABLE_EXPLAINABILITY=true in configuration.",
  "code": "EXPLAINABILITY_DISABLED"
}
```

### GET `/assemblies/{id}/explain_merge`

| Status | Code | Message | Description |
|--------|------|---------|-------------|
| 403 | `EXPLAINABILITY_DISABLED` | "Explainability is disabled. Set ENABLE_EXPLAINABILITY=true in configuration." | Feature flag is not enabled |
| 404 | `ASSEMBLY_NOT_FOUND` | "Assembly with ID {id} not found." | The specified assembly doesn't exist |
| 404 | `MERGE_EVENT_NOT_FOUND` | "No merge event found for this assembly." | The assembly exists but wasn't created by a merge |
| 500 | `LOG_ACCESS_ERROR` | "Error accessing merge log." | Problem reading from the merge log file |

### GET `/assemblies/{id}/lineage`

| Status | Code | Message | Description |
|--------|------|---------|-------------|
| 403 | `EXPLAINABILITY_DISABLED` | "Explainability is disabled. Set ENABLE_EXPLAINABILITY=true in configuration." | Feature flag is not enabled |
| 404 | `ASSEMBLY_NOT_FOUND` | "Assembly with ID {id} not found." | The specified assembly doesn't exist |
| 404 | `NO_LINEAGE` | "Assembly has no lineage information." | The assembly has no merged_from information |
| 500 | `LINEAGE_TRACE_ERROR` | "Error tracing assembly lineage." | Internal error during lineage traversal |

## Phase 5.9 Diagnostics API Errors

### GET `/diagnostics/merge_log`

| Status | Code | Message | Description |
|--------|------|---------|-------------|
| 403 | `EXPLAINABILITY_DISABLED` | "Explainability is disabled. Set ENABLE_EXPLAINABILITY=true in configuration." | Feature flag is not enabled |
| 400 | `INVALID_LIMIT` | "Invalid limit parameter. Must be a positive integer." | limit parameter is not a positive integer |
| 500 | `LOG_READ_ERROR` | "Error reading merge log file." | Problem accessing or parsing the merge log file |

### GET `/config/runtime/{service_name}`

| Status | Code | Message | Description |
|--------|------|---------|-------------|
| 403 | `EXPLAINABILITY_DISABLED` | "Explainability is disabled. Set ENABLE_EXPLAINABILITY=true in configuration." | Feature flag is not enabled |
| 404 | `UNKNOWN_SERVICE` | "Unknown service: {service_name}" | The service name is not recognized |
| 400 | `INVALID_SERVICE_NAME` | "Invalid service_name parameter." | service_name parameter is malformed |
| 500 | `CONFIG_ACCESS_ERROR` | "Error accessing runtime configuration." | Problem retrieving configuration |

## Feature Flag Behavior

All Phase 5.9 explainability and diagnostics endpoints are gated by the `ENABLE_EXPLAINABILITY` flag in the Memory Core configuration. When this flag is set to `false`, these endpoints will return a 403 Forbidden response with the `EXPLAINABILITY_DISABLED` error code.

## Handling Errors in Clients

Client code should be prepared to handle these error responses. Here's an example in Python:

```python
async def get_activation_explanation(client, assembly_id, memory_id=None):
    try:
        response = await client.explain_activation(assembly_id, memory_id=memory_id)
        if response["success"]:
            return response["explanation"]
        else:
            print(f"Error: {response['error']}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None
```

## Testing Error Responses

When implementing tests for these endpoints, be sure to include tests for the error cases, particularly:

1. Feature flag disabled
2. Non-existent resources (assemblies, memories)
3. Invalid parameters
4. Internal server errors (can be mocked)

Example test for explainability disabled:

```python
async def test_explain_activation_disabled():
    # Create app with explainability disabled
    app = create_test_app(
        memory_core=mock_memory_core,
        config={"ENABLE_EXPLAINABILITY": False}
    )
    client = TestClient(app)
    
    # Attempt to access explainability endpoint
    response = client.get("/assemblies/asm_123/explain_activation")
    
    # Verify response
    assert response.status_code == 403
    data = response.json()
    assert data["success"] == False
    assert "Explainability is disabled" in data["error"]
    assert data["code"] == "EXPLAINABILITY_DISABLED"