# Synthians Memory Core Documentation

This directory contains detailed documentation specifically for the `synthians_memory_core` package, the heart of the Synthians memory system.

## Core Components & Concepts

*   [**Architecture**](./Architecture.md): Detailed internal architecture, including planned Explainability Layer.
*   [**Internal Mechanisms**](./INTERNAL_MECHANISMS.md): Key internal mechanisms for concurrency, persistence, retry loops, and background tasks.
*   [**Memory Structures**](./memory_structures.md): `MemoryEntry`, `MemoryAssembly` (incl. `merged_from` field and drift-aware gating).
*   [**Persistence**](./persistence.md): Save/load for entries/assemblies, with improved visualizations of storage structure.
*   [**Vector Index (FAISS)**](./vector_index.md): Implementation details, including GPU acceleration limitations.
*   [**Geometry Management**](./geometry.md): Role of `GeometryManager`.
*   [**QuickRecall Scoring**](./quickrecal.md): Explanation of `UnifiedQuickRecallCalculator`.
*   [**Emotional Intelligence**](./emotion.md): Details on emotion components.
*   [**Metadata Synthesis**](./metadata.md): How `MetadataSynthesizer` enriches memories.

## Planned Phase 5.9 Explainability & Diagnostics

*   [**Explainability**](./explainability.md): **(Planned for Phase 5.9)** Details on the planned `explainability/` module.
*   [**Diagnostics**](./diagnostics.md): **(Planned for Phase 5.9)** Details on the planned `MergeTracker`, runtime config, activation stats.
*   [**API Models**](../api/phase_5_9_models.md): **(Planned for Phase 5.9)** Detailed Pydantic models for new API endpoints.
*   [**Testing Strategy**](../testing/PHASE_5_9_TESTING.md): **(Planned for Phase 5.9)** Comprehensive testing approach for new features.

## Current Features (Phase 5.8)

*   Memory storage and retrieval with QuickRecal scoring.
*   Assembly management with vector index synchronization.
*   Vector index drift detection and repair mechanisms.
*   Emotional gating and analysis.
*   Surprise feedback from Neural Memory.
*   Assembly boost based on synchronization status.

## Phase 5.9 Planned Enhancements (Not Yet Implemented)

*   Backend logic and APIs for explaining assembly activation/merges.
*   Tracking and persistence of assembly merge history using the existing `merged_from` field.
*   Persistent merge event logging via a new `MergeTracker` to `merge_log.jsonl`.
*   New API endpoints (`/diagnostics/merge_log`, `/config/runtime`) for diagnostics.
*   Basic tracking of assembly activation statistics.
*   `ENABLE_EXPLAINABILITY` feature flag to control these new features.

## Additional Resources

*   [**API Reference**](../api/API_REFERENCE.md): *(Link to main API Docs)*
*   [**Development Guide**](../guides/DEVELOPMENT_GUIDE.md): Contribution guidelines.
*   [**Configuration**](../guides/CONFIGURATION_GUIDE.md): *(Link to main Config Guide)*
*   [**Changelog**](../CHANGELOG.md): Complete history of features and improvements.
