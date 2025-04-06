# Phase 5.9 API Models (Implementation Release)

**Document Version:** 2.0 (Final Implementation)
**Target Phase:** 5.9 (COMPLETED)

This document defines the data models for the API endpoints implemented in Phase 5.9. These models reflect the current implementation, including the append-only merge log strategy and enhanced context in explanations.

## Explainability Endpoints

### Explain Assembly Activation

**Endpoint:** `GET /assemblies/{assembly_id}/explain_activation`

**Purpose:** Explains why a specific memory was or wasn't considered part of an assembly during an activation check.

**Response Model:**

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union

# Represents a snapshot of assembly state
class AssemblyStateSnapshot(BaseModel):
    memory_count: Optional[int] = Field(None, description="Number of members at the time")
    last_activation_time: Optional[str] = Field(None, description="ISO timestamp of the last activation")

# Represents the detailed explanation when data is available
class ExplainActivationData(BaseModel):
    assembly_id: str = Field(..., description="ID of the assembly being explained")
    memory_id: str = Field(..., description="ID of the specific memory being checked")
    check_timestamp: str = Field(..., description="ISO format timestamp of when this explanation was generated")
    trigger_context: Optional[str] = Field(None, description="Context of the activation check (e.g., 'retrieval_query:abc', 'assembly_update')") 
    assembly_state_before_check: Optional[AssemblyStateSnapshot] = Field(None, description="Simplified state of the assembly before check")
    calculated_similarity: float = Field(..., description="Calculated similarity score between memory and assembly")
    activation_threshold: float = Field(..., description="Activation threshold used for the decision")
    passed_threshold: bool = Field(..., description="Whether the similarity met or exceeded the threshold")

# Represents the explanation when detailed data isn't applicable or found
class ExplainActivationEmpty(BaseModel):
    assembly_id: str = Field(..., description="ID of the assembly being explained")
    memory_id: Optional[str] = Field(None, description="ID of the specific memory being checked")
    notes: str = Field(..., description="Explanation for why no detailed data is available")

# The actual API response structure
class ExplainActivationResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    explanation: Union[ExplainActivationData, ExplainActivationEmpty] = Field(..., description="Explanation details")
    error: Optional[str] = Field(None, description="Error message if success is False")
```

### Explain Assembly Merge

**Endpoint:** `GET /assemblies/{assembly_id}/explain_merge`

**Purpose:** Provides details about the merge event that resulted in this assembly, using the append-only log.

**Response Model:**

```python
# Source assembly information
class SourceAssemblyInfo(BaseModel):
    id: str = Field(..., description="ID of the source assembly")
    name: Optional[str] = Field(None, description="Name of the source assembly")

# Represents the explanation when the assembly was formed by a merge
class ExplainMergeData(BaseModel):
    assembly_id: str = Field(..., description="ID of the assembly created by the merge")
    is_merged: bool = Field(True, description="Whether this assembly was formed by a merge")
    source_assemblies: List[SourceAssemblyInfo] = Field(..., description="Information about source assemblies")
    similarity_at_merge: float = Field(..., description="Similarity score that triggered the merge")
    merge_threshold: float = Field(..., description="Threshold used for the merge decision")
    merge_timestamp: str = Field(..., description="ISO format timestamp of when the merge occurred")
    cleanup_status: str = Field(..., description="Status of post-merge cleanup (pending/completed/failed)")
    cleanup_timestamp: Optional[str] = Field(None, description="Timestamp of cleanup completion or failure")

# Represents the explanation when the assembly was not formed by a merge
class ExplainMergeEmpty(BaseModel):
    assembly_id: str = Field(..., description="ID of the assembly checked")
    is_merged: bool = Field(False, description="Whether this assembly was formed by a merge")

# The actual API response structure
class ExplainMergeResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    explanation: Union[ExplainMergeData, ExplainMergeEmpty] = Field(..., description="Explanation details")
    error: Optional[str] = Field(None, description="Error message if success is False")
```

### Get Assembly Lineage

**Endpoint:** `GET /assemblies/{assembly_id}/lineage`

**Purpose:** Traces the ancestry of an assembly through its merge history.

**Response Model:**

```python
class LineageEntry(BaseModel):
    assembly_id: str = Field(..., description="ID of the assembly in the lineage")
    name: Optional[str] = Field(None, description="Name of the assembly")
    depth: int = Field(..., description="Depth in the lineage tree (0 = target assembly)")
    status: str = Field("normal", description="Status of this entry in the trace (normal, cycle_detected, depth_limit_reached)")
    created_at: Optional[str] = Field(None, description="ISO timestamp when this assembly was created")
    memory_count: Optional[int] = Field(None, description="Number of memories in this assembly")

class LineageResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    lineage: List[LineageEntry] = Field(..., description="List of assemblies in the lineage")
    error: Optional[str] = Field(None, description="Error message if success is False")
```

## Diagnostics Endpoints

### Get Merge Log

**Endpoint:** `GET /diagnostics/merge_log`

**Purpose:** Retrieves recent merge events, reconciling creation and status updates to show the final state.

**Response Model:**

```python
# Represents a reconciled merge event for the API response
class MergeLogEntry(BaseModel):
    merge_event_id: str = Field(..., description="Unique ID of the merge event")
    timestamp: str = Field(..., description="ISO timestamp when the merge was initiated")
    source_assembly_ids: List[str] = Field(..., description="IDs of the source assemblies involved")
    target_assembly_id: str = Field(..., description="ID of the assembly created by the merge")
    similarity_at_merge: float = Field(..., description="Similarity score that triggered merge")
    merge_threshold: float = Field(..., description="Threshold used for merge decision")
    cleanup_status: str = Field(..., description="Current cleanup status (pending, completed, failed)")
    cleanup_timestamp: Optional[str] = Field(None, description="ISO timestamp of the cleanup status update")
    error: Optional[str] = Field(None, description="Error details if cleanup_status is 'failed'")

# API Response Structure
class MergeLogResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    entries: List[MergeLogEntry] = Field(..., description="List of reconciled merge events")
    total_count: int = Field(..., description="Total number of merge events in the log")
    error: Optional[str] = Field(None, description="Error message if success is False")
```

### Get Runtime Configuration

**Endpoint:** `GET /config/runtime/{service_name}`

**Purpose:** Retrieves the current, sanitized runtime configuration for a specific service.

**Response Model:**

```python
class RuntimeConfigResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    config: Dict[str, Any] = Field(..., description="Dictionary containing only the sanitized configuration key-value pairs")
    error: Optional[str] = Field(None, description="Error message if success is False")
```

## Log Format and Storage

### Merge Log JSONL Format (Append-Only)

The `merge_log.jsonl` file contains a stream of individual JSON objects per line, representing different event types related to merges.

*   **Type 1: Merge Event**
    ```json
    {
      "event_type": "merge",
      "merge_event_id": "merge_uuid_123",
      "timestamp": "2025-04-01T15:32:45.123Z",
      "source_assembly_ids": ["asm_abc", "asm_def"],
      "target_assembly_id": "asm_merged_123",
      "similarity_at_merge": 0.92,
      "merge_threshold": 0.85,
      "cleanup_status": "pending"
    }
    ```
*   **Type 2: Cleanup Update Event**
    ```json
    {
      "event_type": "cleanup_update",
      "target_merge_event_id": "merge_uuid_123",
      "timestamp": "2025-04-01T15:33:10.456Z",
      "status": "completed", 
      "error": null
    }
    ```

### Assembly Activation Statistics

The `stats/assembly_activation_stats.json` file stores counters for assembly activations:

```json
{
  "activation_counts": {
    "asm_uuid1": 42,
    "asm_uuid2": 17,
    "asm_uuid3": 8
  },
  "last_updated": "2025-04-01T16:45:30Z"
}
```

## Allowed Configuration Keys

The following configuration keys are considered safe for exposure via the runtime configuration endpoint:

*   **`SAFE_CONFIG_KEYS_MEMORY_CORE`:**
    ```python
    SAFE_CONFIG_KEYS_MEMORY_CORE = [
        "assembly_activation_threshold",
        "default_assembly_size",
        "merge_log_max_entries",
        "assembly_metrics_persist_interval",
        "enable_explainability",
        "max_memories_per_retrieval",
        "similarity_threshold"
    ]
    ```
*   **`SAFE_CONFIG_KEYS_VECTOR_INDEX`:**
    ```python
    SAFE_CONFIG_KEYS_VECTOR_INDEX = [
        "index_type",
        "embedding_dim",
        "metric_type",
        "vector_precision",
        "max_vectors_per_batch"
    ]
    ```
*   **`SAFE_CONFIG_KEYS_API`:**
    ```python
    SAFE_CONFIG_KEYS_API = [
        "enable_compression",
        "default_page_size",
        "max_page_size",
        "max_content_length",
        "allow_cors"
    ]
    ```

## Performance Considerations

1. **Log Rotation**: The merge log is rotated when it exceeds the configured maximum entry count (`merge_log_max_entries`).
2. **API Caching**: Responses for explainability endpoints are cached to improve performance for repeated queries.
3. **Lazy Loading**: The lineage tracing algorithm uses lazy loading to minimize memory consumption during traversal.
4. **Feature Flag**: All explainability and diagnostics features can be disabled via the `ENABLE_EXPLAINABILITY` flag when performance is critical.