# Explainability Module (Revised for Phase 5.9)

**Document Version:** 1.1 (Reflecting Expert Review)
**Target Phase:** 5.9

This document outlines the planned explainability module for the Synthians Memory Core, incorporating expert feedback for Phase 5.9.

## Overview

The explainability module provides mechanisms to understand *why* certain internal decisions were made, focusing on assembly activation, merging, and lineage tracing. This enhances transparency beyond basic diagnostics by providing contextual reasons for system behavior. These features are controlled by the `ENABLE_EXPLAINABILITY` configuration flag (defaulting to `False`) to manage potential performance overhead.

## Key Components

The explainability logic will reside within the `synthians_memory_core/explainability/` directory.

### 1. Activation Explainer

**Purpose**: Explain why a specific memory was (or wasn't) considered part of an assembly during an activation check (typically triggered during retrieval).

**Dependencies**:
*   `GeometryManager`: To calculate or retrieve similarity scores between memory and assembly embeddings.
*   `MemoryPersistence`: To load memory and assembly data (embeddings, metadata) if not already available in the cache.
*   Core configuration (`config` dict): To access the `assembly_activation_threshold`.
*   **Trigger Context**: Information passed in about what initiated the activation check (e.g., the specific retrieval query ID or context).

**Implementation Plan**:
*   Implement the asynchronous function `generate_activation_explanation` in `explainability/activation.py`.
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
    1.  Load the specified `MemoryAssembly` and `MemoryEntry` using `persistence`. Handle cases where they don't exist.
    2.  Retrieve or recalculate the similarity between the memory's embedding and the assembly's composite embedding using `geometry_manager`. Handle potential alignment/validation issues.
    3.  Retrieve the relevant `assembly_activation_threshold` from the `config`.
    4.  Compare the similarity score against the threshold.
    5.  Optionally, retrieve simplified state of the assembly *before* the check (e.g., last activation level, member count) if readily available without significant performance cost.
*   **Output**: Return a dictionary matching the structure defined in the updated `ExplainActivationData` Pydantic model (`docs/api/phase_5_9_models.md`). Key fields include: `assembly_id`, `memory_id`, `check_timestamp`, `calculated_similarity`, `activation_threshold`, `passed_threshold`, **`trigger_context`**, and potentially simplified `assembly_state_before_check`.

**API Endpoint**: `GET /assemblies/{id}/explain_activation?memory_id={memory_id}` (Requires `ENABLE_EXPLAINABILITY=true`).

### 2. Merge Explainer

**Purpose**: Explain how a specific assembly was formed by a merge operation, combining information from the assembly's state and the persistent merge log.

**Dependencies**:
*   `MemoryPersistence`: To load the target assembly (to access its `merged_from` field) and to fetch the **names** of the source assemblies.
*   `MergeTracker`: To query the **append-only** `merge_log.jsonl` file and **reconcile the final status** (creation event + latest status update event) for the merge event that created the target assembly.
*   `GeometryManager`: Needed internally by `get_assembly_names` (which uses `safe_load_assembly`) to handle assembly loading correctly.

**Implementation Plan**:
*   Implement the asynchronous function `generate_merge_explanation` in `explainability/merge.py`.
*   **Function Signature**:
    ```python
    async def generate_merge_explanation(
        assembly_id: str,
        merge_tracker: MergeTracker,
        persistence: MemoryPersistence,
        geometry_manager: GeometryManager
    ) -> Dict[str, Any]:
        # ... implementation ...
    ```
*   **Logic**:
    1.  Load the target `MemoryAssembly` using `persistence`. If not found or `merged_from` is empty, return the `ExplainMergeEmpty` structure.
    2.  Query the `merge_tracker` to find the `merge_creation` event where `target_assembly_id` matches the input `assembly_id`.
    3.  If a creation event is found, query the `merge_tracker` again to find the *most recent* `cleanup_status_update` event for that specific `merge_event_id`.
    4.  Retrieve the **names** of the source assemblies (listed in the `merged_from` field) using `persistence.load_assembly`. Handle cases where source assemblies may have been deleted.
    5.  Combine information from the assembly (`merged_from`), the merge creation event (timestamp, similarity, threshold), and the latest status update event (final status, error details).
*   **Output**: Return a dictionary matching the structure defined in the updated `ExplainMergeData` Pydantic model (`docs/api/phase_5_9_models.md`). Key fields include: `target_assembly_id`, `merge_event_id`, `merge_timestamp`, `source_assembly_ids`, `source_assembly_names`, `similarity_at_merge`, `threshold_at_merge`, **reconciled `cleanup_status`**, and `cleanup_details`.

**API Endpoint**: `GET /assemblies/{id}/explain_merge` (Requires `ENABLE_EXPLAINABILITY=true`).

### 3. Lineage Tracker

**Purpose**: Trace the ancestry of a given assembly through its merge history, visualizing how it evolved from earlier assemblies.

**Dependencies**:
*   `MemoryPersistence`: To recursively load parent assemblies using the `merged_from` field.
*   `GeometryManager`: Needed to correctly load assemblies via `safe_load_assembly` during the recursive trace.

**Implementation Plan**:
*   Implement the asynchronous function `trace_lineage` in `explainability/lineage.py`.
*   **Function Signature**:
    ```python
    async def trace_lineage(
        assembly_id: str,
        persistence: MemoryPersistence,
        geometry_manager: GeometryManager,
        max_depth: int
    ) -> List[Dict[str, Any]]:
        # ... implementation ...
    ```
*   **Logic**:
    1.  Use recursion or an iterative approach (like Breadth-First or Depth-First Search) to traverse the assembly graph upwards via the `merged_from` links.
    2.  **Implement robust cycle detection:** Pass a `visited: set` argument through recursive calls. If an `assembly_id` is already in `visited`, mark it as a cycle detection point and stop traversing that branch.
    3.  Strictly enforce the `max_depth` limit to prevent unbounded traversal. Mark nodes where traversal stopped due to depth limit.
    4.  For each assembly encountered, retrieve its basic information (ID, name, creation time, memory count).
*   **Output**: Return a list of dictionaries matching the `LineageEntry` Pydantic model (`docs/api/phase_5_9_models.md`). Each entry should include `assembly_id`, `name`, `depth` (relative to the starting assembly), and a **`status` field ("origin", "merged", "cycle_detected", "depth_limit_reached", "not_found")**.

**API Endpoint**: `GET /assemblies/{id}/lineage` (**Caching is strongly recommended** at the API layer due to potentially expensive recursive loading). Requires `ENABLE_EXPLAINABILITY=true`.

### 4. MergeTracker Integration

*   **Role**: The `MergeTracker` component (implemented in `metrics/merge_tracker.py` as per `diagnostics.md`) provides the persistent, append-only log of merge events (`merge_log.jsonl`). This log is the source of truth for the `Merge Explainer`.
*   **Interaction**:
    *   The `SynthiansMemoryCore._execute_merge` method logs the `merge_creation` event.
    *   The `SynthiansMemoryCore._cleanup_and_index_after_merge` method logs the `cleanup_status_update` event.
    *   The `generate_merge_explanation` function queries the `MergeTracker` to read and reconcile these events.

## Performance & Security Considerations

*   **Performance**:
    *   **Activation Explanation:** Can be computationally intensive due to potential data loading (if items aren't cached) and similarity recalculation. Minimize redundant calculations.
    *   **Lineage Tracing:** Cost is directly proportional to the depth and breadth of the lineage. **API-level caching (e.g., TTL cache)** is crucial to mitigate performance impact for repeated requests on the same assembly.
    *   **Merge Explanation:** Primarily involves log querying and data loading. Performance depends on the efficiency of log reading/reconciliation and persistence loading. The append-only strategy shifts complexity to reads.
*   **Security**:
    *   **Data Sensitivity:** Assembly IDs or names might contain contextual clues. Ensure that IDs logged by `MergeTracker` or returned by `/lineage` do not expose sensitive information or PII. Consider using opaque internal IDs or applying masking/aliasing at the API layer if necessary. **A security review of logged data is required.**
    *   **Endpoint Access:** Access to explainability and diagnostic endpoints should be appropriately secured, especially if the system is deployed outside a trusted internal environment. The `ENABLE_EXPLAINABILITY` flag provides the primary control.

## API & Dashboard Integration

*   The core explainability functions (`generate_...`, `trace_lineage`) provide the data served by the new `/explain_*` and `/lineage` API endpoints.
*   These APIs are specifically designed to provide the necessary structured data for the planned Phase 5.9.1 **Synthians Cognitive Dashboard**.
*   The API responses should prioritize clarity and structure, enabling the dashboard to easily parse the information and present it visually (e.g., lineage trees, activation score breakdowns), aligning with user experience goals ("make it hot", i.e., clear and visually appealing).

## Configuration

The explainability features are controlled by the `ENABLE_EXPLAINABILITY` flag. Other related configurations (managed by respective components) influence behavior:

```json
{
    "ENABLE_EXPLAINABILITY": false, // Default to False for production
    // From Diagnostics/MergeTracker:
    // "MERGE_LOG_PATH": "data/logs/merge_log.jsonl",
    // "MERGE_LOG_MAX_ENTRIES": 50000,
    // "MERGE_LOG_ROTATION_SIZE_MB": 100,
    // From Core/Lineage:
    "MAX_LINEAGE_DEPTH": 10,
    // Optional: "LINEAGE_CACHE_TTL_SECONDS": 300
}
```
*(Refer to `docs/guides/CONFIGURATION_GUIDE.md` for the full list)*

## Implementation Roadmap for Phase 5.9

1.  **Core Logic:** Implement `generate_activation_explanation`, `generate_merge_explanation`, and `trace_lineage` functions within the `explainability/` directory, incorporating cycle detection and status reporting.
2.  **Dependencies:** Ensure functions correctly utilize `MemoryPersistence`, `GeometryManager`, `MergeTracker`, and `config`. Fetch source assembly names in `generate_merge_explanation`.
3.  **Models:** Define and use the updated Pydantic models (`ExplainActivationData`, `ExplainMergeData`, `LineageEntry`, etc.) reflecting added context.
4.  **API Integration:** Implement the corresponding API endpoints in `api/explainability_routes.py`, including feature flag checks and **implement caching for `/lineage`**.
5.  **Testing:** Write comprehensive unit and integration tests covering logic, API endpoints, edge cases (cycles, limits), flag behavior, and **add performance benchmarks**.
6.  **Security Review:** Explicitly review assembly ID naming and log contents during implementation.

## Best Practices for Implementation

1.  **Separation of Concerns:** Keep core calculation logic separate from API routing/handling. Use helper functions in `_explain_helpers.py`.
2.  **Asynchronous Operations:** Ensure all I/O (persistence loading, log reading) is performed asynchronously using `await` or `asyncio.to_thread` where appropriate.
3.  **Error Handling:** Gracefully handle missing data (e.g., assembly not found, merge event missing, source assembly deleted) and return informative error structures or appropriate empty/status-marked responses (like `ExplainMergeEmpty` or `LineageEntry` with status "not_found").
4.  **Performance Optimization:** Minimize redundant data loading and calculations. Use caching for `/lineage`. Be mindful of the potential cost of `/explain_activation`.
5.  **Clarity & Usability:** Design explanation outputs (the Pydantic models) to be structured and informative, facilitating clear presentation in the dashboard.