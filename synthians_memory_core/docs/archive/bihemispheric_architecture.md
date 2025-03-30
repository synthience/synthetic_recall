# Bi-Hemispheric Cognitive Architecture

## Overview

The Bi-Hemispheric Cognitive Architecture implements a neural system inspired by human brain hemispheric specialization, creating a bidirectional flow between memory storage/retrieval and sequential prediction. This architecture enables Lucidia to develop a more nuanced understanding of sequential patterns and adapt memory retrieval based on prediction accuracy and surprise detection.

## Key Components

### 1. Memory Core (Left Hemisphere)

Responsible for storing, indexing, retrieving, and enriching memories:

- **Memory Storage**: Persists embeddings and metadata to disk
- **Vector Indexing**: Enables fast similarity-based retrieval using FAISS
- **Metadata Enrichment**: Adds contextual information to memories
- **Emotional Analysis**: Detects emotions in content and uses them for retrieval
- **HPC-QR**: Hippocampal-inspired Quick Recall scoring system

### 2. Trainer Server (Right Hemisphere)

Focuses on pattern recognition and sequence prediction:

- **Sequence Prediction**: Predicts the next embedding based on current input
- **Memory State Tracking**: Maintains internal memory state to track context
- **Surprise Analysis**: Detects unexpected patterns in embedding sequences

### 3. Context Cascade Engine (Corpus Callosum)

Orchestrates the bidirectional flow between the two hemispheres:

- **Prediction Integration**: Feeds predictions from the Trainer into Memory Core retrieval
- **Surprise Detection**: Identifies when reality diverges from predictions
- **Memory Enhancement**: Updates memory importance based on surprise signals
- **State Management**: Tracks the Trainer's memory state across interactions

## Neural Pathway Flow

1. **Input Processing**: New input is processed and embedded by the Memory Core
2. **Prediction**: Context Cascade Engine sends current embedding to Trainer for next embedding prediction
3. **Reality Check**: When new input arrives, it's compared against the prediction
4. **Surprise Detection**: Difference between prediction and reality is quantified
5. **Feedback Loop**: Surprising memories get importance boosts in Memory Core
6. **Retrieval Enhancement**: Future retrievals prioritize memories that were surprising

## Key Innovations

1. **Vector Alignment**: System handles embedding dimension mismatches (384 vs 768) seamlessly
2. **Surprise Metrics**: Measures both prediction error and context shifts
3. **Adaptive Thresholds**: Surprise detection adapts to current narrative volatility
4. **Memory State Continuity**: Maintains continuity of the prediction model's internal state
5. **Quickrecal Boosting**: Automatically enhances the retrieval priority of surprising memories

## Architecture Diagram

```
┌───────────────────┐              ┌─────────────────────┐
│   Memory Core     │              │   Trainer Server    │
│  (Left Hemisphere)│              │  (Right Hemisphere) │
│                   │              │                     │
│ ┌───────────────┐ │              │ ┌─────────────────┐ │
│ │   GeometryMgr │ │              │ │GeometryMgr (ref)│ │
│ └───────────────┘ │              │ └─────────────────┘ │
│ ┌───────────────┐ │              │ ┌─────────────────┐ │
│ │  VectorIndex  │ │              │ │SequencePredictor│ │
│ └───────────────┘ │              │ └─────────────────┘ │
│ ┌───────────────┐ │              │ ┌─────────────────┐ │
│ │   MetadataSyn │ │              │ │ SurpriseDetector│ │
│ └───────────────┘ │              │ └─────────────────┘ │
└────────┬──────────┘              └──────────┬──────────┘
         │                                     │
         │        ┌──────────────────┐        │
         │        │ Context Cascade  │        │
         └────────┤     Engine      ├────────┘
                  │ (Corpus Callosum)│
                  └──────────────────┘
```

## Implementation Notes

- The system is designed to handle embedding dimension mismatches, a critical requirement for systems using different embedding models
- The GeometryManager is shared across components to ensure vector operations are consistent
- All communication between components uses asynchronous HTTP calls with proper timeouts and error handling
- Memory state is preserved between calls to maintain prediction continuity
- The system adapts to the current context's volatility when determining surprise thresholds
