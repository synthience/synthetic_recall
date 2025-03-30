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
- `NeuralMemoryModule`: Provides key-value-query projections and memory update mechanisms

### 3. Context Cascade Engine (Orchestrator)

The Context Cascade Engine connects the Memory Core and Trainer Server, orchestrating the flow of information between them and implementing the full cognitive cycle.

**Key Responsibilities:**
- Processing new memories through the complete cognitive pipeline
- Managing the interplay between prediction and memory storage
- Feeding surprise feedback to enhance memory retrieval
- Handling error states and coordinating between components
- Orchestrating variant-specific processing pathways (Titans variants)

**Key Classes:**
- `ContextCascadeEngine`: Main orchestrator class with modular processing methods
- `GeometryManager`: Shared utility for consistent vector operations across components
- `TitansVariantBase`: Base class for all variant-specific processing
- `SequenceContextManager`: Manages historical context for attention-based operations

## Titans Architecture Variants

The system supports multiple cognitive processing variants through the Titans Architecture framework. Each variant implements different attention mechanisms and memory update strategies:

### NONE Variant (Default)

The standard cognitive flow without additional attention mechanisms.

**Key Characteristics:**
- Direct memory storage and retrieval
- Standard Neural Memory updates without attention-based modifications
- Baseline for comparison with other variants

### MAC Variant (Memory-Attended Content)

Enhances retrieved content using attention mechanisms over historical memory.

**Key Characteristics:**
- Processes input through standard Neural Memory update
- Applies attention between query and historical keys to modify retrieved output
- Post-retrieval enhancement of memory content

**Processing Flow:**
1. Standard memory update and retrieval
2. Apply attention between current query and historical keys
3. Create attended_y_t by combining retrieved and historical values
4. Return the attention-modified retrieved embedding

### MAG Variant (Memory-Attended Gates)

Modifies Neural Memory update gate values using attention mechanisms.

**Key Characteristics:**
- Calculates Neural Memory gate values (α, θ, η) using attention
- These gates control forgetting rate, learning rate, and momentum decay
- Pre-update influence on memory storage

**Processing Flow:**
1. Calculate projections from input
2. Apply attention between query and historical keys
3. Compute gate values from attention output
4. Update Neural Memory with custom gates
5. Standard memory retrieval

### MAL Variant (Memory-Attended Learning)

Modifies the value projection used in Neural Memory updates using attention.

**Key Characteristics:**
- Modifies the value projection (v_t) before Neural Memory update
- Uses attention between current query and historical keys/values
- Creates an enhanced representation for memory storage

**Processing Flow:**
1. Calculate projections from input
2. Apply attention between query and historical keys/values
3. Calculate modified value projection (v_prime) by combining original and attended values
4. Update Neural Memory with modified value projection
5. Standard memory retrieval

## Information Flow

### Refactored Cognitive Cycle

1. **Input Processing:**
   - New memory content and optional embedding arrive at the Context Cascade Engine
   - The Engine forwards the memory to the Memory Core for storage
   - Memory ID and embedding (x_t) are returned

2. **Projections and Variant Pre-Processing:**
   - The Engine obtains key, value, and query projections (k_t, v_t, q_t) from Neural Memory
   - For MAG: Calculate attention-based gates
   - For MAL: Calculate modified value projection

3. **Neural Memory Update:**
   - For NONE/MAC: Standard update with input embedding
   - For MAG: Include calculated gates in update
   - For MAL: Use modified value projection in update
   - Loss and gradient norm metrics are returned

4. **QuickRecal Feedback:**
   - Surprise metrics (loss, grad_norm) are used to calculate QuickRecal boost
   - Memory Core updates the memory's QuickRecal score accordingly

5. **Retrieval and Post-Processing:**
   - Neural Memory retrieves relevant embedding based on input
   - For MAC: Apply attention over historical context to modify retrieved embedding

6. **History Update:**
   - All context (embeddings, projections, results) is added to sequence history
   - This enriches the historical context for future attention operations

## Embedding Handling and Dimension Alignment

The system includes robust handling for embedding-related challenges:

### Dimension Mismatches

The system gracefully handles dimension mismatches between embeddings (e.g., 384 vs 768 dimensions):

- **Vector Alignment Utility**: Automatically aligns vectors to the same dimension for comparison operations
- **Normalization Methods**: Safe normalization with dimension handling (padding/truncation as needed)
- **Validation**: Detection and handling of malformed embeddings (NaN/Inf values)

**Implementation Details:**
- The `_align_vectors_for_comparison` method handles dimension mismatches
- The `_normalize_embedding` methods in multiple classes handle padding or truncation
- The `_validate_embedding` method checks for NaN/Inf values and provides fallbacks

### Embedding Conversion

The system includes utility methods to handle various embedding formats:

- `_to_list`: Safely converts numpy arrays, TensorFlow tensors, and other array-like objects to Python lists
- `_to_numpy`: Ensures consistent numpy array format for internal processing

## TensorFlow Lazy Loading

To prevent NumPy version conflicts, the system implements lazy loading for TensorFlow:

```python
# Global variable for TensorFlow instance
_tf = None

def _get_tf():
    """Lazy-load TensorFlow only when needed."""
    global _tf
    if _tf is None:
        try:
            import tensorflow as tf
            _tf = tf
        except ImportError as e:
            logger.error(f"Failed to import TensorFlow: {e}")
    return _tf
```

**Benefits:**
- Prevents NumPy version conflicts by deferring TensorFlow imports
- Allows the `fix_numpy.py` script to downgrade NumPy before TensorFlow is imported
- Keeps TensorFlow isolated to only those components that require it
- Enables all variants to function correctly regardless of NumPy version requirements

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
   - Improved scalability through horizontal scaling of the Trainer Server
   - Enhanced reliability as state is not dependent on server uptime
   - Simplified debugging and state inspection
   - Easier deployment and migration without state loss

## Memory Assemblies

Memory Assemblies represent related memories that are grouped together for enhanced retrieval and semantic organization.

1. **Creation and Composition:**
   - Assemblies can be created with initial memories or built incrementally
   - Each assembly maintains a composite embedding representing its semantic center
   - When memories are added, the composite embedding is updated

2. **Retrieval Benefits:**
   - Assemblies improve recall by activating related memories
   - They provide context for ambiguous queries
   - They enable higher-level semantic organization beyond individual memories

3. **Dynamic Updates:**
   - Assemblies can evolve over time as new memories are added
   - The system can merge similar assemblies or split diverging ones
   - Assembly strength is determined by member coherence and usage patterns

## Implementation and Integration Guidelines

1. **Component Communication:**
   - All inter-component communication uses well-defined APIs
   - The Context Cascade Engine handles all orchestration
   - Components should not directly communicate with each other

2. **Error Handling:**
   - Each component implements comprehensive error handling
   - The orchestrator manages overall system stability
   - Graceful degradation is provided when components are unavailable

3. **Configuration:**
   - Each component has its own configuration
   - The orchestrator manages system-wide settings
   - Environment variables like `TITANS_VARIANT` control high-level behavior

4. **Monitoring and Diagnostics:**
   - Each component provides health and performance metrics
   - The `lucidia_think_trace` tool offers system-wide diagnostics
   - Logging is standardized across components for easy aggregation
