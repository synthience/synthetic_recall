# Explainability Module (Phase 5.9)

**Document Version:** 2.0 (Implementation Release)
**Target Phase:** 5.9

This document outlines the implemented explainability module for the Synthians Memory Core, providing transparency into system decisions.

## Overview

The explainability module provides mechanisms to understand *why* certain internal decisions were made, focusing on assembly activation, merging, and lineage tracing. This enhances transparency beyond basic diagnostics by providing contextual reasons for system behavior. These features are controlled by the `ENABLE_EXPLAINABILITY` configuration flag (defaulting to `False`) to manage potential performance overhead.

## Key Components

The explainability logic resides within the `synthians_memory_core/explainability/` directory.

### 1. Activation Explainer

**Purpose**: Explain why a specific memory was (or wasn't) considered part of an assembly during an activation check (typically triggered during retrieval).

**Dependencies**:
*   `GeometryManager`: To calculate or retrieve similarity scores between memory and assembly embeddings.
*   `MemoryPersistence`: To load memory and assembly data (embeddings, metadata) if not already available in the cache.
*   Core configuration (`config` dict): To access the `assembly_activation_threshold`.
*   **Trigger Context**: Information passed in about what initiated the activation check (e.g., the specific retrieval query ID or context).

**Implementation**:
*   The asynchronous function `generate_activation_explanation` in `explainability/activation.py`.
*   **Function Signature**:
    ```python
    async def generate_activation_explanation(
        assembly_id: str,
        memory_id: str,
        trigger_context: Optional[str],
        persistence: MemoryPersistence,
        geometry_manager: GeometryManager,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        # ... implementation ...
    ```
*   **Logic**:
    1.  Loads the specified `MemoryAssembly` and `MemoryEntry` using `persistence` with `safe_load_assembly` and `safe_load_memory` helpers. Handles cases where they don't exist.
    2.  Retrieves or recalculates the similarity between the memory's embedding and the assembly's composite embedding using `geometry_manager` via the `calculate_similarity` helper. Handles potential alignment/validation issues.
    3.  Retrieves the relevant `assembly_activation_threshold` from the `config`.
    4.  Compares the similarity score against the threshold.
    5.  Returns assembly state information at the time of the check.
*   **Output**: Returns a dictionary matching the structure defined in the `ExplainActivationData` Pydantic model (`docs/api/phase_5_9_models.md`). Key fields include: `assembly_id`, `memory_id`, `check_timestamp`, `calculated_similarity`, `activation_threshold`, `passed_threshold`, `trigger_context`, and `assembly_state_before_check`.

**API Endpoint**: `GET /assemblies/{id}/explain_activation?memory_id={memory_id}` (Requires `ENABLE_EXPLAINABILITY=true`).

### 2. Merge Explainer

**Purpose**: Explain how a merged assembly was created from its source assemblies, including similarity levels and merge decisions.

**Dependencies**:
*   `MemoryPersistence`: To load assembly data, including the critical `merged_from` list.
*   `MergeTracker`: To access the historical merge event log containing details about the merge operation.
*   `GeometryManager`: Required for loading assemblies from persistence.

**Implementation**:
*   The asynchronous function `generate_merge_explanation` in `explainability/merge.py`.
*   **Function Signature**:
    ```python
    async def generate_merge_explanation(
        assembly_id: str,
        merge_tracker,
        persistence: MemoryPersistence,
        geometry_manager: GeometryManager
    ) -> Dict[str, Any]:
        # ... implementation ...
    ```
*   **Logic**:
    1.  Loads the target `MemoryAssembly` using `persistence` and `geometry_manager`. If not found or `merged_from` is empty, returns the `ExplainMergeEmpty` structure.
    2.  Queries the `merge_tracker` to find the `merge_creation` event where `target_assembly_id` matches the input `assembly_id`.
    3.  If a creation event is found, queries the `merge_tracker` again to find the *most recent* `cleanup_status_update` event for that specific `merge_event_id`.
    4.  Retrieves the **names** of the source assemblies (listed in the `merged_from` field) using `persistence.load_assembly`. Handles cases where source assemblies may have been deleted.
    5.  Combines assembly data (`merged_from`) with merge log data (similarity/threshold values, timestamps) to build a comprehensive explanation.
*   **Output**: Returns a dictionary matching the structure defined in the `ExplainMergeData` Pydantic model. Key fields include: `assembly_id`, `is_merged`, `source_assemblies` (with IDs and names), `similarity_at_merge`, `merge_threshold`, `merge_timestamp`, `cleanup_status`, and optional `cleanup_timestamp` and `error`.

**API Endpoint**: `GET /assemblies/{id}/explain_merge` (Requires `ENABLE_EXPLAINABILITY=true`).

### 3. Lineage Tracer

**Purpose**: Trace the ancestry of a merged assembly through its chain of source assemblies, supporting historical analysis of memory assemblies.

**Dependencies**:
*   `MemoryPersistence`: To recursively load assemblies in the lineage chain via the `merged_from` field.
*   `GeometryManager`: Required for loading assemblies from persistence.

**Implementation**:
*   The asynchronous function `trace_lineage` in `explainability/lineage.py`.
*   **Function Signature**:
    ```python
    async def trace_lineage(
        assembly_id: str,
        persistence: MemoryPersistence,
        geometry_manager: GeometryManager,
        max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        # ... implementation ...
    ```
*   **Logic**:
    1.  Implements a recursive traversal algorithm starting from the target assembly.
    2.  Follows the `merged_from` field on each assembly to identify parent assemblies.
    3.  Uses a `visited` set to detect cycles in the lineage graph.
    4.  Enforces a `max_depth` limit to prevent excessive traversal or stack overflow.
    5.  Collects metadata about each assembly in the chain, including status (normal, cycle_detected, depth_limit_reached).
*   **Output**: Returns a list of dictionaries matching the structure defined in the `LineageEntry` Pydantic model. Each entry includes: `assembly_id`, `name`, `depth`, `status`, `created_at`, and `memory_count`. Special status values (`cycle_detected`, `depth_limit_reached`) are used to indicate early termination conditions.

**API Endpoint**: `GET /assemblies/{id}/lineage?max_depth={max_depth}` (Requires `ENABLE_EXPLAINABILITY=true`).

## Helper Functions

The module includes several helper functions in `_explain_helpers.py`:

*   `safe_load_assembly`: Safely loads an assembly, handling errors and returning appropriate messages.
*   `safe_load_memory`: Safely loads a memory, handling errors and returning appropriate messages.
*   `calculate_similarity`: Calculates similarity between memory and assembly embeddings using the geometry manager.
*   `get_assembly_names`: Retrieves human-readable names for a list of assembly IDs.

## Performance & Security Considerations

*   **Performance**:
    *   The explainability features are designed to be lightweight, leveraging already-cached data where possible.
    *   The `max_depth` parameter on lineage tracing prevents excessive recursion and resource consumption.
    *   API endpoints include optional caching for frequently-requested explanations.
    *   All features are disabled by default, requiring explicit activation via the `ENABLE_EXPLAINABILITY` flag.

*   **Security**:
    *   **Data Sensitivity:** Assembly IDs or names might contain contextual clues. The implementation ensures that IDs logged by `MergeTracker` or returned by `/lineage` do not expose sensitive information or PII.
    *   **Endpoint Access:** Access to explainability and diagnostic endpoints is appropriately secured via the `ENABLE_EXPLAINABILITY` flag.

## API & Dashboard Integration

The explainability features are exposed through well-defined REST endpoints in `api/explainability_routes.py`, which handle:

*   Request validation and parameter parsing.
*   Dependency injection to access core components (`persistence`, `merge_tracker`, `geometry_manager`).
*   Appropriate error handling and response formatting.
*   Response caching where appropriate (particularly for `/lineage` traversals).

These endpoints are conditionally mounted by the API server based on the `ENABLE_EXPLAINABILITY` flag, ensuring zero overhead when the feature is disabled.

Dashboard integration is planned for the diagnostic dashboard, which will provide a visual interface to these explainability features.