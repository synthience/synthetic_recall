# Bi-Hemispheric Architecture Overview

## Introduction

The Synthians Memory Core implements a Bi-Hemispheric Cognitive Architecture that separates memory storage/retrieval from sequence prediction/surprise detection, mimicking how the brain's hemispheres handle different aspects of cognition. This document provides a technical overview of the architecture, component interactions, and the information flow between them.

## System Components

### 1. Memory Core

The Memory Core serves as the primary memory storage and retrieval system, similar to the brain's hippocampus and temporal lobes.

**Key Responsibilities:**
- Storing and indexing memory entries with associated embeddings and metadata
- Retrieval of memories based on semantic similarity and quickrecal scores
- Memory assembly management and persistence
- Emotional gating of memory retrieval based on emotional context
- Maintaining memory importance through quickrecal scores

**Key Classes:**
- `SynthiansMemoryCore`: Main interface for all memory operations
- `MemoryEntry`: Individual memory representation with embedding and metadata
- `MemoryAssembly`: Collection of related memories with a composite embedding
- `MemoryPersistence`: Handles saving and loading memories and assemblies
- `EmotionalGatingService`: Applies emotional context to memory retrieval

### 2. Trainer Server

The Trainer Server handles sequence prediction and surprise detection, similar to the brain's frontal lobes and predictive capabilities.

**Key Responsibilities:**
- Predicting the next embedding in a sequence using neural mechanisms
- Calculating surprise when expectations don't match reality
- Training on memory sequences to improve predictions
- Maintaining a stateless architecture that relies on explicit memory state passing

**Key Classes:**
- `SynthiansTrainer`: Neural model for sequence prediction
- `SurpriseDetector`: Detects and analyzes surprise in embedding sequences
- `HPCQRFlowManager`: Manages the QuickRecal factors for memory importance

### 3. Context Cascade Engine (Orchestrator)

The Context Cascade Engine connects the Memory Core and Trainer Server, orchestrating the flow of information between them and implementing the full cognitive cycle.

**Key Responsibilities:**
- Processing new memories through the complete cognitive pipeline
- Managing the interplay between prediction and memory storage
- Feeding surprise feedback to enhance memory retrieval
- Handling error states and coordinating between components

**Key Classes:**
- `ContextCascadeEngine`: Main orchestrator class
- `GeometryManager`: Shared utility for consistent vector operations across components

## Information Flow

### Full Cognitive Cycle

1. **Input Processing:**
   - New memory content and optional embedding arrive at the Context Cascade Engine
   - The Engine forwards the memory to the Memory Core for storage

2. **Prediction:**
   - The Engine sends the current embedding to the Trainer Server
   - Trainer generates a prediction for the next memory embedding
   - The prediction is stored for later comparison

3. **Reality and Surprise:**
   - When the next actual memory arrives, its embedding is compared to the prediction
   - The Trainer calculates surprise metrics between prediction and reality
   - High surprise indicates a memory that violated expectations

4. **Feedback:**
   - Surprise information is fed back to adjust the quickrecal score of the memory
   - Surprising memories receive a higher importance (quickrecal boost)
   - This feedback loop ensures important memories are more accessible

5. **Adaptation:**
   - The system continuously learns from sequences of memories
   - Prediction accuracy improves over time through training
   - Retrieval thresholds adapt based on results

## Stateless Design Pattern

A key refinement in the architecture is the stateless design of the Trainer Server:

1. **No Global State:**
   - The Trainer Server maintains no session or global state
   - Each prediction request must include all necessary context

2. **Memory State Passing:**
   - The `previous_memory_state` parameter contains the state from the last prediction
   - This state includes sequence history, surprise metrics, and momentum
   - The response includes a new `memory_state` to be passed in the next request

3. **Orchestrator State Management:**
   - The Context Cascade Engine manages the memory state between requests
   - It stores the state returned by the Trainer and passes it in the next prediction

4. **Benefits:**
   - Improved scalability and reliability
   - Multiple sessions can use the same Trainer Server concurrently
   - Simpler recovery from failures

## Assembly Management

Memory Assemblies provide a way to group related memories and treat them as a cohesive unit:

1. **Creation and Composition:**
   - Assemblies can be created with initial memories or built incrementally
   - Each assembly maintains a composite embedding representing its semantic center
   - When memories are added, the composite embedding is updated

2. **Persistence:**
   - Assemblies are saved to disk in JSON format with their constituent memories
   - On initialization, the system loads all saved assemblies
   - The `MemoryPersistence` class handles serialization, deserialization, and error recovery

3. **Memory-to-Assembly Mapping:**
   - The system maintains a mapping from individual memories to their assemblies
   - This allows for efficient querying of all assemblies that contain a specific memory

## Error Handling

The architecture implements robust error handling throughout:

1. **Specific Error Types:**
   - HTTP status codes (404, 400, 500) are handled with specific error messages
   - Connection errors, timeouts, and unexpected exceptions have clear handling paths

2. **Consistent Error Format:**
   - All errors follow a standard format with status, code, and detailed message
   - Custom error codes for specific failure scenarios

3. **Graceful Degradation:**
   - Components can continue functioning when dependent services are unavailable
   - Default behaviors are provided when specific features cannot be accessed

## Geometric Operations

All vector operations are standardized using the shared GeometryManager:

1. **Consistent Vector Handling:**
   - Normalization, similarity calculations, and vector alignment
   - Support for different embedding dimensions (384, 768)
   - Handling of different geometries (Euclidean, Hyperbolic, Spherical)

2. **Alignment for Comparison:**
   - Vectors of different dimensions are safely aligned
   - NaN/Inf values are detected and handled appropriately

## Conclusion

The Bi-Hemispheric Architecture provides a powerful framework for memory processing that combines storage, retrieval, prediction, and surprise detection in a cohesive system. The stateless design pattern enhances scalability while maintaining the rich context needed for effective cognition.

The architecture is designed to be modular, allowing components to be improved or replaced independently. This flexibility enables ongoing enhancements to specific aspects of the system while maintaining overall functionality.
