# Synthians Cognitive Architecture: Component Guide

## Core Components (Updated for Phase 5.8)

### 1. Synthians Memory Core (`synthians_memory_core`)

*   **Role:** Persistent, indexed, context-aware memory storage and retrieval. Manages `MemoryEntry` and `MemoryAssembly`.
*   **Key Modules:**
    *   `SynthiansMemoryCore`: Main orchestrator.
    *   `memory_structures`: `MemoryEntry`, `MemoryAssembly` classes.
    *   `memory_persistence`: Async save/load for entries and assemblies.
    *   `vector_index`: FAISS `IndexIDMap` wrapper with repair functionality.
    *   `geometry_manager`: Vector math with different embedding spaces.
    *   `hpc_quickrecal`: Relevance scoring with emotional context.
    *   `emotional_intelligence`: Emotion analysis and gating.
    *   `metadata_synthesizer`: Metadata enrichment for memories.
    *   **Internal Stability Mechanisms**: 
        * `_pending_vector_updates`: Queue for tracking failed vector operations
        * `_vector_update_retry_loop`: Background task for retrying failed operations
        * Note: The older `assembly_sync_manager.py` is deprecated and has been replaced with these internal mechanisms
*   **API:** Exposed via `api/server.py`. Includes memory processing, retrieval, assembly management, and configuration endpoints.
*   **Phase 5.8:** Added vector index drift detection via `vector_index_updated_at` timestamps, retry mechanism for failed vector operations, and automatic index integrity checking/repair.
*   **Integration:** Stores memories and assemblies in a structured persistence layer with vector indexing.
*   **See:** `docs/core/README.md`

### 2. Neural Memory Server (`synthians_trainer_server`)

*   **Role:** Adaptive associative sequence memory (Titans-based). Test-time learning. Surprise calculation.
*   **Key Modules:** `neural_memory`, `http_server`, `metrics_store`, `surprise_detector`.
*   **API:** Provides endpoints for update, retrieval, projections, gate calculation, etc.
*   **Phase 5.8:** Enhanced stability with better error handling and diagnostic endpoints.
*   **Integration:** Receives embeddings from CCE; returns loss/grad_norm (surprise) and retrieved embeddings to CCE.
*   **See:** `docs/trainer/README.md`

### 3. Context Cascade Engine (CCE) (`orchestrator`)

*   **Role:** Orchestrates MC <-> NM flow, implements cognitive cycle, dynamic variant selection, LLM guidance integration.
*   **Key Modules:** `context_cascade_engine`, `titans_variants`, `history`, `variant_selector`, `memory_logic_proxy`.
*   **API:** Exposes `/process_memory` (primary entry point).
*   **Phase 5.8:** Enhanced error handling, better performance metrics for variant selection, improved attention mechanics.
*   **Integration:** Calls MC and NM APIs; receives input, sends back final response.
*   **See:** `docs/orchestrator/README.md`

## Background Tasks and Stability Mechanisms

### Vector Index Reliability (Phase 5.8)

*   **Implementation:** Internal queue and background task in `SynthiansMemoryCore`:
    *   `_pending_vector_updates`: Queue for failed vector operations
    *   `_vector_update_retry_loop`: Background task that processes the queue
*   **Purpose:** Handle failures in FAISS operations gracefully without blocking main threads
*   **Mechanism:** When vector operations fail, they're added to the queue with operation type, ID, and embedding
*   **Gating:** `vector_index_updated_at` timestamp on assemblies enables drift-aware retrieval
*   **Integration:** Used in process_memory, create_assembly, update_memory, update_assembly, etc.
*   **See:** `docs/core/INTERNAL_MECHANISMS.md`

### Additional Background Tasks

*   **Persistence Loop:** `_persistence_loop` saves dirty memories/assemblies periodically
*   **Decay & Pruning Loop:** `_decay_and_pruning_loop` handles QuickRecal decay and optional assembly pruning/merging
*   **Index Integrity Check:** `_check_index_integrity` verifies consistency on startup
*   **Index Repair:** `_repair_index_async` can rebuild the index from stored data

## Shared Utilities

*   **Geometry Manager:** Vector operations across different embedding spaces
*   **Embedding Generator:** Wrapper around different embedding models
*   **API Clients:** Python clients for MC, NM, and CCE
*   **Configuration:** Layer for loading and validating config

## Testing Components

*   **Mock Components:** MockMemoryCore, MockGeometryManager, etc.
*   **Test Fixtures:** Common setups for unit and integration tests
*   **Benchmarking:** Tools for performance testing

## Phase 5.9 Planned Enhancements (Not Yet Implemented)

*   **Explainability Module:** For explaining activation and merge decisions
*   **Merge Tracker:** For logging and querying merge events
*   **Runtime Configuration:** For exposing sanitized configuration options
*   **Activation Statistics:** For tracking assembly usage patterns
*   **Integration with Dashboard:** APIs for data visualization

## Dependency Graph

```
SynthiansMemoryCore
├── MemoryPersistence
├── VectorIndex (FAISS wrapper)
├── GeometryManager
├── QuickRecalCalculator
├── EmotionalIntelligence
└── Background Tasks
    ├── _persistence_loop
    ├── _vector_update_retry_loop
    └── _decay_and_pruning_loop

ContextCascadeEngine
├── SequenceHistoryManager
├── TitansVariants (MAC, MAG, MAL)
├── VariantSelector
└── MemoryLogicProxy (calls Memory Core)

NeuralMemoryServer
├── NeuralMemoryModule
├── MetricsStore
└── SurpriseDetector