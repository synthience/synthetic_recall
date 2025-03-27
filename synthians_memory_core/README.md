# synthians_memory_core/README.md

# Synthians Memory Core

This directory contains the unified and optimized memory system for the Synthians AI architecture, integrating the best features from the Lucid Recall system.

## Overview

The Synthians Memory Core provides a lean, efficient, yet powerful memory system incorporating:

-   **Advanced Relevance Scoring:** Uses the `UnifiedQuickRecallCalculator` (HPC-QR) for multi-factor memory importance assessment.
-   **Flexible Geometry:** Supports Euclidean, Hyperbolic, Spherical, and Mixed geometries for embedding representation via the `GeometryManager`.
-   **Emotional Intelligence:** Integrates emotional analysis and gating (`EmotionalAnalyzer`, `EmotionalGatingService`) for nuanced retrieval.
-   **Memory Assemblies:** Groups related memories (`MemoryAssembly`) for complex concept representation.
-   **Robust Persistence:** Handles disk storage, backups, and atomic writes asynchronously (`MemoryPersistence`).
-   **Adaptive Thresholds:** Dynamically adjusts retrieval thresholds based on feedback (`ThresholdCalibrator`).
-   **Unified Interface:** Provides a cohesive API through `SynthiansMemoryCore`.
-   **Vector Search:** Fast retrieval using FAISS vector indexing with GPU acceleration support.

## Recent Improvements (March 2025)

The Synthians Memory Core has received significant enhancements in the `Synthience_memory_remaster` branch:

-   **Fixed Vector Index Persistence:** The FAISS vector index and ID mappings are now properly saved during the persistence cycle, ensuring memories can be retrieved after system restarts.
-   **Enhanced API Observability:** Added comprehensive vector index information to the `/stats` endpoint for better monitoring and debugging.
-   **Improved Embedding Handling:** Robust dimension handling to ensure vector operations work correctly regardless of embedding dimensions (384 vs 768).
-   **Retrieval Threshold Adjustments:** Lowered pre-filter threshold from 0.5 to 0.2 for improved recall while maintaining precision.
-   **Validation Tools:** Added comprehensive test scripts to validate the full memory lifecycle.

See `docs/memory_system_remaster.md` for detailed documentation on these improvements.

## Components

-   `synthians_memory_core.py`: The main orchestrator class.
-   `hpc_quickrecal.py`: Contains the `UnifiedQuickRecallCalculator`.
-   `geometry_manager.py`: Centralizes embedding and geometry operations.
-   `emotional_intelligence.py`: Provides emotion analysis and gating.
-   `memory_structures.py`: Defines `MemoryEntry` and `MemoryAssembly`.
-   `memory_persistence.py`: Manages disk storage and backups.
-   `adaptive_components.py`: Includes `ThresholdCalibrator`.
-   `vector_index.py`: Handles FAISS vector indexing with GPU support.
-   `custom_logger.py`: Simple logging utility.

## Usage

```python
import asyncio
from synthians_memory_core import SynthiansMemoryCore
import numpy as np

async def main():
    # Configuration (adjust paths and dimensions as needed)
    config = {
        'embedding_dim': 768,
        'geometry': 'hyperbolic',
        'storage_path': './synthians_memory_data'
    }

    # Initialize
    memory_core = SynthiansMemoryCore(config)
    await memory_core.initialize()

    # --- Example Operations ---

    # Generate a sample embedding (replace with your actual embedding generation)
    sample_embedding = np.random.rand(config['embedding_dim']).astype(np.float32)

    # 1. Store a new memory
    memory_entry = await memory_core.process_new_memory(
        content="Learned about hyperbolic embeddings today.",
        embedding=sample_embedding,
        metadata={"source": "learning_session"}
    )
    if memory_entry:
        print(f"Stored memory: {memory_entry.id}")

    # 2. Retrieve memories
    query_embedding = np.random.rand(config['embedding_dim']).astype(np.float32) # Use actual query embedding
    retrieved = await memory_core.retrieve_memories(
        query="hyperbolic geometry",
        query_embedding=query_embedding,
        top_k=3
    )
    print(f"\nRetrieved {len(retrieved)} memories:")
    for mem_dict in retrieved:
        print(f"- ID: {mem_dict.get('id')}, Score: {mem_dict.get('final_score', mem_dict.get('relevance_score')):.3f}, Content: {mem_dict.get('content', '')[:50]}...")


    # 3. Provide feedback (if adaptive thresholding enabled)
    if memory_entry and memory_core.threshold_calibrator:
         await memory_core.provide_feedback(
              memory_id=memory_entry.id,
              similarity_score=0.85, # Example score from retrieval
              was_relevant=True
         )
         print(f"\nProvided feedback. New threshold: {memory_core.threshold_calibrator.get_current_threshold():.3f}")

    # 4. Detect Contradictions
    # (Add potentially contradictory memories first)
    await memory_core.process_new_memory(content="A causes B", embedding=np.random.rand(config['embedding_dim']))
    await memory_core.process_new_memory(content="A prevents B", embedding=np.random.rand(config['embedding_dim']))
    contradictions = await memory_core.detect_contradictions()
    print(f"\nDetected {len(contradictions)} potential contradictions.")

    # Shutdown
    await memory_core.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Improvements

-   **Unified Structure:** Consolidates core logic into fewer files.
-   **Centralized Geometry:** `GeometryManager` handles all geometric operations consistently.
-   **Direct Integration:** HPC-QR, Emotion, Assemblies are integral parts, not separate layers added via mixins.
-   **Improved Efficiency:** Leverages `asyncio` and dedicated persistence class.
-   **Clearer Interfaces:** Simplified API focused on core memory operations.
-   **Hyperbolic First-Class:** Hyperbolic geometry is treated as a core configuration option.

This new structure provides a more streamlined, maintainable, and potentially more efficient implementation while capturing the core value propositions (HPC-QR, Hyperbolic, Emotion, Assemblies) of the original system.