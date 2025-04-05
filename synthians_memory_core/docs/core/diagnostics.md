# Diagnostics Module (Phase 5.9)

**Document Version:** 2.0 (Implementation Release)
**Target Phase:** 5.9

This document outlines the implemented diagnostics module for the Synthians Memory Core, providing tools for monitoring and troubleshooting.

## Overview

The diagnostics module provides tools to monitor, inspect, and troubleshoot the Synthians Cognitive Architecture. It focuses on exposing runtime metrics, **sanitized** configuration, and **robustly logged** system events to help developers understand behavior and identify issues. This module's features are controlled by the `ENABLE_EXPLAINABILITY` configuration flag, which defaults to `False` in production environments to minimize performance impact.

## Key Components

### 1. MergeTracker (Append-Only Log)

**Purpose**: Track and log assembly merge operations reliably for historical analysis and debugging, **using an append-only strategy for robustness.** This approach avoids complex and potentially risky file rewrites for status updates.

**Implementation**:
*   **Class:** `MergeTracker` is implemented in `synthians_memory_core/metrics/merge_tracker.py`.
*   **Log Strategy:** Utilizes an append-only approach for the `merge_log.jsonl` file. Each significant merge-related action (creation, cleanup completion, cleanup failure) generates a distinct log entry.
    *   **`log_merge_event(...)`**: Logs the initial merge details (source IDs, target ID, similarity, threshold) with a unique `merge_event_id` and `event_type: "merge"`. The initial `cleanup_status` is set to "pending".
    *   **`update_cleanup_status(...)`**: Logs a *separate* event with `event_type: "cleanup_update"`. This event references the original `merge_event_id` and provides the `status` ("completed" or "failed") along with a timestamp and optional `error` details if the status is "failed".
*   **Storage**: Events are written as individual JSON lines (JSONL format) to `merge_log.jsonl`. The path is configurable via the configuration. The `aiofiles` library is used for asynchronous file writes.
*   **Querying & Reconciliation**: Reading the merge log via `MergeTracker.read_log_entries(limit)` involves fetching recent raw events. To determine the *current* status of a specific merge for API responses (like `/diagnostics/merge_log`), the implementation:
    1.  Identifies the relevant `merge` event.
    2.  Scans subsequent log entries for the *latest* `cleanup_update` event matching the `merge_event_id`.
    3.  Combines this information to present a reconciled view.
*   **Log Rotation**: Implements log rotation triggered when *new* entries are added. Rotation is based on maximum entry count (config key `merge_log_max_entries`). Rotation uses atomic file operations to prevent data loss during the rotation process.
*   **Security Note:** Ensures assembly IDs stored in the log do not contain PII by using opaque internal IDs.

**API Endpoint**: `GET /diagnostics/merge_log` (Returns *reconciled* merge events).

#### JSONL Event Schemas

*   **Merge Event:**
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
*   **Cleanup Update Event:**
    ```json
    {
      "event_type": "cleanup_update",
      "merge_event_id": "merge_uuid_123", // References the original merge event
      "timestamp": "2025-04-01T15:32:50.456Z",
      "status": "completed" // or "failed"
      // "error": "Error details" // Optional, present only if status is "failed"
    }
    ```

**Integration with Memory Core**:
1.  In `SynthiansMemoryCore._execute_merge`:
    *   After merge completion: `await merge_tracker.log_merge_event(...)` with source/target IDs, similarity, and threshold.
2.  In `SynthiansMemoryCore._cleanup_and_index_after_merge`:
    *   On success: `await merge_tracker.update_cleanup_status(merge_event_id, "completed")`.
    *   On failure: `await merge_tracker.update_cleanup_status(merge_event_id, "failed", error=error_details)`.

### 2. Runtime Configuration Exposure

**Purpose**: Securely expose the current runtime configuration for diagnostic visibility, applying appropriate sanitization to protect sensitive values.

**Implementation**:
*   **API Endpoint**: `GET /config/runtime/{service_name}` in `api/diagnostics_routes.py`.
*   **Security**: Implements **strict allow-list sanitization** using predefined `SAFE_CONFIG_KEYS` lists. Only explicitly allowed configuration keys are exposed via the API. Sensitive keys (credentials, secrets, internal paths) are filtered out.
*   **Validation**: Configurable via `ENABLE_EXPLAINABILITY` flag to provide control over exposure.
*   **Service Segmentation**: Configuration is segmented by `service_name` (e.g., "memory_core", "geometry", "api"), allowing targeted visibility into specific components.

**Example Endpoint**:
```
GET /config/runtime/memory_core -> {"assembly_activation_threshold": 0.82, "default_assembly_size": 10, ...}
GET /config/runtime/api -> {"enable_compression": true, "default_page_size": 25, ...}
```

### 3. Assembly Activation Statistics

**Purpose**: Track which assemblies are being activated most frequently during memory operations, providing insights into assembly utilization patterns.

**Implementation**:
*   **Tracking**: Within `SynthiansMemoryCore`, an in-memory dictionary `_assembly_activation_counts` is maintained, incrementing counters when assemblies are activated. This provides a real-time view of assembly utilization.
*   **Persistence**: Statistics are periodically saved to disk at `stats/assembly_activation_stats.json` using the `_persist_activation_stats` method, preserving data across service restarts. The persistence interval is configurable via `assembly_metrics_persist_interval`.
*   **API Exposure**: Statistics are exposed via the enhanced `/stats` endpoint, which now includes assembly activation counts alongside other system metrics.

**Example Stats Endpoint Response**:
```json
{
  "memory_stats": { ... },
  "assembly_stats": {
    "count": 42,
    "activation_counts": {
      "assembly_123": 156,
      "assembly_456": 89,
      ...
    }
  }
}
```

## Integration with Explainability

The Diagnostics Module works in concert with the Explainability Module to provide a comprehensive view of system behavior:

*   **MergeTracker** provides the data foundation for the **Merge Explainer** (`generate_merge_explanation`), allowing explanations of how assemblies were formed through merge operations.
*   **Activation Statistics** complement the **Activation Explainer** (`generate_activation_explanation`), offering insights into which assemblies are most frequently activated.
*   **Runtime Configuration** exposure provides context for all explainability functions, helping to understand the settings that influence system behavior.

## API Integration

The diagnostics features are accessed through well-defined API endpoints in `api/diagnostics_routes.py`:

*   `GET /diagnostics/merge_log`: Returns a reconciled view of recent merge events from the `MergeTracker`.
*   `GET /config/runtime/{service_name}`: Returns a sanitized view of the current runtime configuration for the specified service.
*   `GET /stats`: Enhanced to include assembly activation statistics alongside existing system metrics.

These endpoints are conditionally mounted based on the `ENABLE_EXPLAINABILITY` flag, ensuring zero overhead when diagnostics are disabled.

## Configuration

The diagnostics features are controlled by several configuration options:

*   `ENABLE_EXPLAINABILITY` (bool, default: `false`): Master switch for diagnostics and explainability features.
*   `merge_log_max_entries` (int, default: `1000`): Maximum number of entries to retain in the merge log.
*   `assembly_metrics_persist_interval` (float, default: `600.0`): Seconds between persisting assembly activation statistics.

## Security & Performance Considerations

*   **Security**:
    *   Configuration exposure uses strict allow-listing to prevent leakage of sensitive information.
    *   Assembly IDs in logs and statistics are opaque identifiers without embedded sensitive data.
    *   All diagnostics endpoints are controlled by the `ENABLE_EXPLAINABILITY` flag, which defaults to `false` in production.

*   **Performance**:
    *   The append-only log strategy minimizes write contention for the merge log.
    *   Activation statistics are maintained in memory with efficient counter increments, minimizing overhead.
    *   Configuration exposure has negligible performance impact, as it simply returns a filtered view of in-memory data.
    *   All features can be disabled via configuration when performance is critical.