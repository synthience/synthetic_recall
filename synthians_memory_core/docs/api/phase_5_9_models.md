# Phase 5.9 API Models (Revised)

**Document Version:** 1.1 (Reflecting Expert Review)
**Target Phase:** 5.9

This document defines the data models for the new API endpoints planned in Phase 5.9, incorporating expert review feedback. **These models reflect the revised implementation plan, including the append-only merge log strategy and enhanced context in explanations.**

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
    member_count: Optional[int] = Field(None, description="Number of members at the time")
    activation_level: Optional[float] = Field(None, description="Activation level at the time")
    # Add other relevant simple state fields if applicable

# Represents the detailed explanation when data is available
class ExplainActivationData(BaseModel):
    assembly_id: str = Field(..., description="ID of the assembly being explained")
    memory_id: Optional[str] = Field(None, description="ID of the specific memory being checked (if provided)")
    check_timestamp: str = Field(..., description="ISO format timestamp of when this explanation was generated")
    trigger_context: Optional[str] = Field(None, description="Context of the activation check (e.g., 'retrieval_query:abc', 'assembly_update')") # Added
    assembly_state_before_check: Optional[AssemblyStateSnapshot] = Field(None, description="Simplified state of the assembly before check") # Refined
    calculated_similarity: Optional[float] = Field(None, description="Calculated similarity score between memory and assembly")
    activation_threshold: Optional[float] = Field(None, description="Activation threshold used for the decision")
    passed_threshold: Optional[bool] = Field(None, description="Whether the similarity met or exceeded the threshold")
    notes: Optional[str] = Field(None, description="Additional explanation notes (e.g., 'Similarity >= threshold', 'Assembly not synchronized', 'Memory embedding invalid')")

# Represents the explanation when detailed data isn't applicable or found
class ExplainActivationEmpty(BaseModel):
    assembly_id: str = Field(..., description="ID of the assembly being explained")
    memory_id: Optional[str] = Field(None, description="ID of the specific memory being checked (if provided)")
    notes: str = Field(..., description="Explanation for why no detailed data is available (e.g., 'Memory not found', 'Assembly not found', 'Activation check not applicable')")

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
# Represents details about merge cleanup
class MergeCleanupDetails(BaseModel):
    update_timestamp: Optional[str] = Field(None, description="Timestamp of the latest cleanup status update")
    error_message: Optional[str] = Field(None, description="Error message if cleanup failed")

# Represents the explanation when the assembly was formed by a merge
class ExplainMergeData(BaseModel):
    target_assembly_id: str = Field(..., description="ID of the assembly created by the merge")
    merge_event_id: Optional[str] = Field(None, description="ID of the *merge_creation* event in the log") # Clarified source event
    merge_timestamp: Optional[str] = Field(None, description="ISO format timestamp of when the merge occurred (from creation event)")
    source_assembly_ids: List[str] = Field(..., description="IDs of the source assemblies that were merged (from target assembly's merged_from field)")
    source_assembly_names: Optional[List[str]] = Field(None, description="Names of the source assemblies (if available)") # Added
    similarity_at_merge: Optional[float] = Field(None, description="Similarity score that triggered the merge (from creation event)")
    threshold_at_merge: Optional[float] = Field(None, description="Threshold used for the merge decision (from creation event)")
    reconciled_cleanup_status: Optional[str] = Field(None, description="Final cleanup status ('pending', 'completed', 'failed') derived by finding the latest related status update event in the log") # Clarified reconciliation
    cleanup_details: Optional[MergeCleanupDetails] = Field(None, description="Details about the cleanup status (timestamp of update, error message if failed)") # Refined
    notes: Optional[str] = Field(None, description="Additional explanation notes")

# Represents the explanation when the assembly was not formed by a merge
class ExplainMergeEmpty(BaseModel):
    target_assembly_id: str = Field(..., description="ID of the assembly checked")
    notes: str = Field("Assembly was not formed by a merge.", description="Explanation for non-merged assemblies (checked merged_from field)")

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
    name: Optional[str] = Field(None, description="Name of the assembly") # Made optional as it requires extra lookups
    depth: int = Field(..., description="Depth in the lineage tree (0 = target assembly)")
    status: Optional[str] = Field(None, description="Status of this entry in the trace (e.g., 'origin', 'merged', 'cycle_detected', 'depth_limit_reached', 'not_found')") # Added status detail
    created_at: Optional[str] = Field(None, description="ISO timestamp when this specific assembly was created (if available)")
    memory_count: Optional[int] = Field(None, description="Number of memories in this assembly at the time of its creation/merge (if available)")

class LineageResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    target_assembly_id: str = Field(..., description="The ID of the assembly whose lineage was traced")
    lineage: List[LineageEntry] = Field(..., description="List of assemblies in the lineage, typically ordered breadth-first or depth-first from the target")
    max_depth_reached: bool = Field(..., description="Whether the tracing stopped due to reaching the max_depth limit")
    cycles_detected: bool = Field(..., description="Whether any cycles were detected during tracing")
    error: Optional[str] = Field(None, description="Error message if success is False")
```

## Diagnostics Endpoints

### Get Merge Log (Revised for Reconciliation)

**Endpoint:** `GET /diagnostics/merge_log`

**Purpose:** Retrieves recent merge events, reconciling creation and status updates to show the final state.

**Response Model:**

```python
# Represents a reconciled merge event for the API response
class ReconciledMergeLogEntry(BaseModel):
    merge_event_id: str = Field(..., description="Unique ID of the original merge creation event")
    creation_timestamp: str = Field(..., description="ISO timestamp when the merge was initiated")
    source_assembly_ids: List[str] = Field(..., description="IDs of the source assemblies involved")
    target_assembly_id: str = Field(..., description="ID of the assembly created by the merge")
    similarity_at_merge: Optional[float] = Field(None, description="Similarity score that triggered merge (from creation event)")
    merge_threshold: Optional[float] = Field(None, description="Threshold used for merge decision (from creation event)")
    final_cleanup_status: str = Field(..., description="The latest known cleanup status ('pending', 'completed', 'failed') based on log events")
    cleanup_timestamp: Optional[str] = Field(None, description="ISO timestamp of the *last* cleanup status update event, if any")
    cleanup_error: Optional[str] = Field(None, description="Error details if the final cleanup status is 'failed'")

# API Response Structure
class MergeLogResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    reconciled_log_entries: List[ReconciledMergeLogEntry] = Field(..., description="List of recent, reconciled merge events") # Renamed field for clarity
    count: int = Field(..., description="Total number of reconciled merge creation events returned")
    query_limit: int = Field(..., description="The limit parameter used for the query (applied to creation events)")
    error: Optional[str] = Field(None, description="Error message if success is False")
```

### Get Runtime Configuration

**Endpoint:** `GET /config/runtime/{service_name}`

**Purpose:** Retrieves the current, **sanitized (allow-listed)** runtime configuration for a specific service.

**Response Model:**

```python
class RuntimeConfigResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    service: str = Field(..., description="Name of the service queried (e.g., 'memory-core', 'neural-memory', 'cce')")
    config: Dict[str, Any] = Field(..., description="Dictionary containing only the sanitized configuration key-value pairs allowed for this service")
    retrieval_timestamp: str = Field(..., description="ISO timestamp when the configuration was retrieved")
    error: Optional[str] = Field(None, description="Error message if success is False")
```

## Log Format and Storage (Revised for Option B)

### Merge Log JSONL Format (Append-Only)

The `merge_log.jsonl` file contains a stream of individual JSON objects per line, representing different event types related to merges.

*   **Type 1: Merge Creation Event (`event_type: "merge_creation"`)**
    ```json
    {
      "event_type": "merge_creation",
      "merge_event_id": "merge_uuid_123",
      "timestamp": "2025-04-01T15:32:45.123Z", // Time merge was initiated
      "source_assembly_ids": ["asm_abc", "asm_def"],
      "target_assembly_id": "asm_merged_123",
      "similarity_at_merge": 0.92,
      "merge_threshold": 0.85
      // Note: No explicit cleanup_status here; it's implicitly "pending".
    }
    ```
*   **Type 2: Cleanup Status Update Event (`event_type: "cleanup_status_update"`)**
    ```json
    {
      "event_type": "cleanup_status_update",
      "update_timestamp": "2025-04-01T15:35:10.456Z", // Time this update occurred
      "target_merge_event_id": "merge_uuid_123",     // Links to the creation event via its ID
      "new_status": "completed",                    // "completed" or "failed"
      "error": null                                 // Optional: Error details string if status is "failed"
    }
    ```

### Implementation Considerations (Revised)

1.  **Log Rotation**: `merge_log.jsonl` rotation managed by `MergeTracker` based on configured size (`MERGE_LOG_ROTATION_SIZE_MB`) and/or entry count (`MERGE_LOG_MAX_ENTRIES`). Uses atomic operations.
2.  **Cleanup Status Querying**: The `/diagnostics/merge_log` API endpoint implementation reads the raw log file, identifies `merge_creation` events within the requested limit/timeframe, and then searches *forward* in the log (or uses an optimized index if implemented later) for the *most recent* `cleanup_status_update` event corresponding to each creation event's `merge_event_id`. This reconciled status is returned in the `ReconciledMergeLogEntry`.
3.  **Performance**: Reading and reconciling the log can become I/O intensive for very large logs. Future optimizations might include indexing the log file by `merge_event_id` or using a more suitable storage mechanism if query performance becomes a bottleneck.

## Runtime Configuration Sanitization

The specific keys considered safe for exposure via `GET /config/runtime/{service_name}` must be explicitly defined in an allow-list within the API implementation (e.g., `api/diagnostics_routes.py`).

*   **`SAFE_CONFIG_KEYS_MEMORY_CORE`:**
    ```python
    SAFE_CONFIG_KEYS_MEMORY_CORE = [
        "embedding_dim", "geometry", "assembly_activation_threshold",
        "assembly_boost_mode", "assembly_boost_factor",
        "max_allowed_drift_seconds", "enable_assembly_pruning",
        "enable_assembly_merging", "enable_explainability", # Include the flag itself
        "assembly_sync_check_interval", "persistence_interval",
        "decay_interval", "prune_check_interval",
        "assembly_threshold", "assembly_merge_threshold",
        "adaptive_threshold_enabled", "initial_retrieval_threshold",
        "vector_index_type", "check_index_on_retrieval",
        "index_check_interval", "vector_index_retry_interval",
        # Phase 5.9 Configs:
        "MERGE_LOG_MAX_ENTRIES", "MERGE_LOG_ROTATION_SIZE_MB",
        "ASSEMBLY_METRICS_PERSIST_INTERVAL", "MAX_LINEAGE_DEPTH",
        "EXPLAINABILITY_LOG_LEVEL"
    ]
    ```
*   **`SAFE_CONFIG_KEYS_NEURAL_MEMORY`:**
    ```python
    SAFE_CONFIG_KEYS_NEURAL_MEMORY = [
        "input_dim", "key_dim", "value_dim", "query_dim", # Confirm if these should be exposed
        "memory_hidden_dims", "gate_hidden_dims",
        "alpha_init", "theta_init", "eta_init", # Expose initial values, not runtime logits
        "outer_learning_rate", "use_complex_gates",
        # Phase 5.9 relevant (if applicable):
        "window_size", # If used for context/diagnostics
        "learning_rate", # Alias for outer_learning_rate?
        "surprise_threshold",
        "batch_size",
        "model_type",
        "enable_attention_maps" # If applicable
    ]
    ```
*   **`SAFE_CONFIG_KEYS_CCE`:**
    ```python
    SAFE_CONFIG_KEYS_CCE = [
        "default_variant", "variant_selection_mode",
        "variant_selection_threshold", "llm_guidance_weight",
        "cached_variants", "history_window_size",
        "high_surprise_threshold", "low_surprise_threshold", # From VariantSelector
        "llm_studio_endpoint", "llm_model", # Expose endpoint/model for info
        # Phase 5.9 relevant:
        "METRICS_RESPONSE_LIMIT", "INCLUDE_TRACE_INFO",
        "INCLUDE_LLM_ADVICE_RAW", "embedding_dim" # If CCE has its own dim config
    ]
    ```

**(Implementation Example `get_safe_config` - remains the same as previously provided)**
```python
from typing import Dict, Any

# Assume SAFE_CONFIG_KEYS dict is defined as above

def get_safe_config(service_name: str, full_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a sanitized version of the configuration."""
    if service_name not in SAFE_CONFIG_KEYS:
        raise ValueError(f"Unknown service: {service_name}")

    allow_list = SAFE_CONFIG_KEYS.get(service_name, [])
    sanitized = {}
    for key in allow_list:
        if key in full_config:
            # Basic type checking or serialization might be needed here
            # Ensure no complex objects leak unintentionally
            value = full_config[key]
            if isinstance(value, (str, int, float, bool, list, dict)):
                 try:
                     # Ensure the value is JSON serializable
                     json.dumps(value)
                     sanitized[key] = value
                 except TypeError:
                     sanitized[key] = f"[Unserializable type: {type(value).__name__}]"
            else:
                 # For other types, maybe just return their string representation or type name
                 sanitized[key] = f"[Type: {type(value).__name__}]"

    return sanitized