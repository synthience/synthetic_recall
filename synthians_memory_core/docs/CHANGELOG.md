# Synthians Cognitive Architecture Changelog

This document tracks significant changes to the Synthians Cognitive Architecture.

## [Released] - Phase 5.9.1 - Backend API Stability (2025-04-06)

### Fixed
- **Memory Core Service**:
    - Fixed 500 error in `/stats` endpoint by implementing robust fallbacks for vector index integrity checks
    - Added missing `/config/runtime/{service_name}` endpoint for dashboard configuration access
    - Resolved TypeError in `detect_and_repair_index_drift` by removing incorrect `await` call
- **Neural Memory Service**:
    - Fixed UnboundLocalError in `/diagnose_emoloop` endpoint by properly initializing emotion entropy variable
- **Context Cascade Engine**:
    - Implemented missing `/status` endpoint with CCEStatusPayload model
    - Fixed AttributeError in `/health` endpoint by using safer attribute checks
    - Corrected TypeError in `/metrics/recent_cce_responses` by properly awaiting the coroutine
- **Dashboard Integration**:
    - Aligned proxy route configuration with implemented backend endpoints
    - Added CCEStatusData and CCEStatusResponse interfaces to shared schema
    - Updated API client hooks to use correct interface types

### Added
- Comprehensive error logging and handling across all endpoints
- Safe attribute access with sensible defaults for configuration endpoints
- Additional TypeScript interfaces for proper type checking

## [Released] - Phase 5.9 - Explainability & Diagnostics (2025-04-05)

### Added
- **Explainability Module (`explainability/`)**:
    - `generate_activation_explanation`: Core logic to explain assembly activation based on similarity vs. threshold.
    - `generate_merge_explanation`: Core logic to explain assembly merges by combining assembly data (`merged_from`) and reconciled merge log events. Requires `GeometryManager` for loading.
    - `trace_lineage`: Core logic to trace assembly ancestry via `merged_from` links, including cycle detection and max depth handling. Requires `GeometryManager` for loading.
    - `_explain_helpers.py`: Utility functions for safe data loading and calculations.
- **Diagnostics Module (`metrics/`)**:
    - `MergeTracker`: Manages an **append-only** log (`merge_log.jsonl`) of merge creation and cleanup status events for robustness and history. Implements reconciliation logic. Includes configurable log rotation (size/entry count).
    - Activation Statistics: Basic in-memory tracking (`_assembly_activation_counts`) with periodic persistence (`stats/assembly_activation_stats.json`).
- **API Endpoints (`api/`)**:
    - `GET /assemblies/{id}/explain_activation`: Exposes activation explanation logic.
    - `GET /assemblies/{id}/explain_merge`: Exposes merge explanation logic.
    - `GET /assemblies/{id}/lineage`: Exposes lineage tracing logic (with caching).
    - `GET /diagnostics/merge_log`: Exposes reconciled merge log entries.
    - `GET /config/runtime/{service_name}`: Exposes sanitized (allow-listed) runtime configuration.
- **Configuration**:
    - `ENABLE_EXPLAINABILITY`: Master flag to enable/disable all new features (default: `False`).
    - `MERGE_LOG_PATH`, `MERGE_LOG_MAX_ENTRIES`, `MERGE_LOG_ROTATION_SIZE_MB`: For `MergeTracker`.
    - `ASSEMBLY_METRICS_PERSIST_INTERVAL`: For activation stats persistence.
    - `MAX_LINEAGE_DEPTH`: For lineage tracing.
- **Testing (`tests/test_phase_5_9_explainability.py`)**:
    - Comprehensive unit, integration, and API tests for all new features.
    - Tests cover core logic, API endpoints, feature flag behavior, edge cases (cycles, depth limits, nonexistent items), and error handling.
    - Improved test fixture setup and teardown reliability (async cleanup, retry mechanisms, robust directory removal).

### Changed
- **`SynthiansMemoryCore`**: Integrated calls to `MergeTracker` during assembly merge operations (`_execute_merge`, `cleanup_and_index_after_merge`). Integrated activation tracking (`_track_assembly_activation`).
- **`MemoryPersistence`**: Added helper (`safe_write_json`) for atomic writes used by metrics/stats persistence. `load_assembly` now requires `GeometryManager`.
- **API Server (`api/server.py`)**: Conditionally mounts new routers based on `ENABLE_EXPLAINABILITY`. Passes necessary dependencies (core, persistence, geometry_manager) to route handlers.
- **API Client (`api/client/client.py`)**: Added methods to interact with new explainability/diagnostics endpoints.
- **Documentation**: Updated architecture, component guides, API references, configuration guide, etc.

### Fixed
- Addressed various bugs identified during testing: `AttributeError` (dict vs object), `TypeError` (mocking), `KeyError` (API schema), `ValueError` (test setup), persistence loading issues, lineage chain persistence in tests, teardown `PermissionError` mitigation.
- Corrected argument passing (e.g., `geometry_manager`) between API routes and core logic functions.
- Ensured Pydantic models (`docs/api/phase_5_9_models.md`) accurately reflect API request/response structures.
- Replaced deprecated `datetime.utcnow()` calls.

## Upcoming: Phase 5.9 (Planned)

- **Explainability Layer:**
  - New backend module for explaining system decisions
  - Tracking and logging of assembly merges
  - Assembly lineage tracing via `merged_from` field
  - API endpoints for accessing explanations
- **New API Endpoints (Planned):**
  - `GET /assemblies/{id}/explain_activation` - Explains assembly activation decisions
  - `GET /assemblies/{id}/explain_merge` - Provides merge event details 
  - `GET /assemblies/{id}/lineage` - Traces assembly ancestry
  - `GET /config/runtime/{service_name}` - Shows sanitized runtime configuration
  - `GET /diagnostics/merge_log` - Shows merge event history
- **Memory Assembly Enhancements:**
  - Full utilization of `merged_from` field for ancestry tracking
  - Persistent merge event logging
- **Enhanced Metrics:**
  - Improved CCE response metrics with detailed variant selection info
  - Assembly activation statistics

## Phase 5.8 (Current)

- **Memory Assembly Enhancements:**
  - Added **timestamp-based vector index drift detection**
  - Only synchronized assemblies contribute to boosting
  - Added `vector_index_updated_at` field to track synchronization status
  - Added pending update queue for failed vector index operations
- **Vector Index Reliability:**
  - Implemented assembly sync status tracking
  - Added `check_index_integrity` and `repair_index` endpoints
  - Added retry logic for failed vector operations
- **Configuration:**
  - Added `ASSEMBLY_MAX_DRIFT_SECONDS` to control sync freshness requirements
  - Added `ASSEMBLY_SYNC_CHECK_INTERVAL` for performance optimization
- **API Enhancements:**
  - Enhanced `/stats` endpoint with vector index and assembly sync details
  - Added `/assemblies` and `/assemblies/{id}` endpoints
  - Added detailed vector index diagnostics
- **Stability:**
  - Improved error handling for vector operations
  - Implemented safe FAISS index saves
  - Added backup JSON mapping for index recovery

## Phase 5.7

- **Integration Points:**
  - Enhanced CCE metrics endpoints
  - Improved Neural Memory diagnostics
- **Performance:**
  - Optimized FAISS usage
  - Reduced memory footprint

## Phase 5.5 - 5.6

- **Variant Selection:**
  - Adaptive variant switching based on performance
  - LLM advice integration for variant hints
- **Attention Mechanisms:**
  - Focus shift based on performance metrics
  - Weighted memory relevance by attention type

## Phase 5.0 - 5.4

- **Initial Titans Integration:**
  - MAC variant (Memory-Attention-Compression)
  - MAG variant (Memory-Attention-Gates)
  - MAL variant (Memory-Attention-Learning)
- **Core Architecture:**
  - Design and implementation of the Bi-Hemispheric model
  - Memory Core persistence layer
  - Neural Memory surprise detection
  - Context Cascade Engine cognitive flow
