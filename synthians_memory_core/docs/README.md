# Synthians Cognitive Architecture - Documentation

Welcome to the documentation for the Synthians Cognitive Architecture, a system designed to emulate aspects of human memory and cognition.

## Overview

This documentation provides comprehensive details on the system's architecture, its core components (Memory Core, Neural Memory, Context Cascade Engine), the underlying APIs, and usage guidelines.

**Key Concepts:**

*   **Bi-Hemispheric Model:** The system loosely models the interaction between episodic/declarative memory (Memory Core - The Archive) and procedural/associative memory (Neural Memory - The Associator).
*   **QuickRecal:** A dynamic relevance score for memories, influenced by factors like recency, emotion, and surprise.
*   **Surprise Feedback:** The Neural Memory provides signals (loss, gradient norm) indicating how surprising new input is, which boosts the QuickRecal score of corresponding memories in the Core.
*   **Performance-Aware Adaptation:** The system dynamically selects optimal processing variants based on Neural Memory performance metrics and trend analysis.
*   **Asynchronous Processing:** Built with `asyncio` for efficient handling of I/O-bound operations.

## Navigation

*   **[Architecture](./ARCHITECTURE.md):** High-level overview of the system's design, principles, and the Bi-Hemispheric model.
*   **[Component Guide](./COMPONENT_GUIDE.md):** Detailed breakdown of each major component (Memory Core, Neural Memory, CCE, Tools, Testing) and their roles.
*   **[API Reference & Client Usage](./api/README.md):** Documentation for the HTTP APIs and the Python client library.
    *   [API Reference](./api/API_REFERENCE.md)
    *   [Client Usage Guide](./api/client_usage.md)
*   **[Guides](./guides/README.md):** Practical guides for setup, development, and specific use cases.
*   **[Architecture Changes](./architechture-changes.md):** Log of significant architectural decisions and evolution.
*   **[Changelog](./CHANGELOG.md):** Chronological list of changes and updates to the codebase.

## Getting Started

1.  Review the **[Architecture](./ARCHITECTURE.md)** to understand the core concepts.
2.  Explore the **[Component Guide](./COMPONENT_GUIDE.md)** for details on individual parts.
3.  If interacting programmatically, consult the **[API Reference & Client Usage](./api/README.md)**.
4.  For setup and development workflows, see the **[Guides](./guides/README.md)**.
## ARCHITECTURE OVERVIEW
---
graph TD
    subgraph "User/External Systems"
        ClientApp[Client Application / Other Services]
    end

    subgraph "API Layer (FastAPI)"
        APIServer[API Server (server.py)]
        APIEndpoints[Endpoints (/process_memory, /retrieve_memories, etc.)]
        EmbeddingModel[Embedding Model (SentenceTransformer)]
        EmotionAnalyzerAPI[Emotion Analyzer (local/service)]
        TranscriptionExtractorAPI[Transcription Feature Extractor]
    end

    subgraph "Orchestration Layer (Context Cascade Engine - CCE)"
        Orchestrator[ContextCascadeEngine (orchestrator/context_cascade_engine.py)]
        VariantSelector[VariantSelector]
        MemoryLLMRouter[MemoryLLMRouter]
        PerformanceTracker[NM Performance Tracker]
    end

    subgraph "Memory Core Layer (SynthiansMemoryCore - MC)"
        MemoryCore[SynthiansMemoryCore (synthians_memory_core.py)]
        Persistence[MemoryPersistence (memory_persistence.py)]
        VectorIndex[Vector Index (FAISS via vector_index.py)]
        QuickRecallCalc[UnifiedQuickRecallCalculator (hpc_quickrecal.py)]
        GeometryMgr[GeometryManager (geometry_manager.py)]
        ThresholdCalib[ThresholdCalibrator (adaptive_components.py)]
        MetadataSynth[MetadataSynthesizer (tools.py - implied)]
        TrainerIntegration[TrainerIntegrationManager (memory_core/trainer_integration.py)]
        InterruptionHandler[InterruptionAwareMemoryHandler (interruption.py)]
        MemoryStructures[MemoryEntry / MemoryAssembly (memory_structures.py)]
        Utils[Utilities (custom_logger.py, etc.)]
    end

    subgraph "External Services"
        NeuralMemory[Neural Memory (Trainer Server - Separate Service)]
        LLMService[LLM Guidance Service (e.g., Llama 3.2)]
    end

    %% Connections
    ClientApp -- HTTP Request --> APIEndpoints
    APIEndpoints -- Calls --> APIServer

    %% API Server Internal Dependencies
    APIServer -- Uses --> EmbeddingModel
    APIServer -- Uses --> EmotionAnalyzerAPI
    APIServer -- Uses --> TranscriptionExtractorAPI

    %% API to Orchestrator/Core Interaction (Based on Cheat Sheet)
    APIServer -- Request --> Orchestrator

    %% Orchestrator Interactions (Based on Cheat Sheet)
    Orchestrator -- Manages --> MemoryCore
    Orchestrator -- Manages --> NeuralMemory
    Orchestrator -- Uses --> VariantSelector
    Orchestrator -- Uses --> MemoryLLMRouter
    Orchestrator -- Uses --> PerformanceTracker
    Orchestrator -- Gets Guidance --> LLMService
    Orchestrator -- Sends Updates/Retrievals --> TrainerIntegration

    %% Memory Core Internal Interactions
    MemoryCore -- Manages --> MemoryStructures
    MemoryCore -- Uses --> QuickRecallCalc
    MemoryCore -- Uses --> GeometryMgr
    MemoryCore -- Uses --> Persistence
    MemoryCore -- Uses --> ThresholdCalib
    MemoryCore -- Uses --> MetadataSynth
    MemoryCore -- Uses --> TrainerIntegration
    MemoryCore -- Uses --> InterruptionHandler
    MemoryCore -- Uses --> Utils
    
    Persistence -- Interacts --> VectorIndex
    GeometryMgr -- May Use --> EmbeddingModel
    QuickRecallCalc -- Uses --> GeometryMgr
    QuickRecallCalc -- Uses --> EmotionAnalyzerAPI
    TrainerIntegration -- HTTP API Calls --> NeuralMemory

    %% Implicit Dependencies
    MetadataSynth -- Uses --> EmotionAnalyzerAPI

    %% Components used by API directly (or via Core)
    APIServer -- Feedback --> ThresholdCalib
    APIServer -- Retrieval/Processing --> MemoryCore
    APIServer -- Calculation --> QuickRecallCalc
    APIServer -- Analysis --> EmotionAnalyzerAPI 
*This documentation is actively maintained alongside the codebase.*
