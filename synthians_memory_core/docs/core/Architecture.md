# Synthians Memory Core - Architecture

This document provides a detailed overview of the Synthians Memory Core architecture, including component interactions, data flow, and implementation details.

## System Overview

Synthians Memory Core is designed as a modular system with several specialized components that work together to provide efficient, context-aware memory management. The architecture follows a layered approach with clear separation of concerns.

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Synthians Memory Core                            │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │    Memory   │  │     API     │  │  Emotional  │  │   Trainer   │    │
│  │    Layer    │  │    Layer    │  │ Intelligence│  │ Integration │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │           │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐    │
│  │   Memory    │  │    API      │  │  Emotional  │  │   Trainer   │    │
│  │  Structures │  │ Server/Client│  │  Analysis  │  │   Models    │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │           │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐    │
│  │  Geometry   │  │ Persistence │  │  Adaptive   │  │Transcription│    │
│  │   Manager   │  │    Layer    │  │ Components  │  │   Features  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Memory Layer

The Memory Layer is responsible for managing memory entries and assemblies.

**Key Components:**
- **SynthiansMemoryCore**: The central orchestrator that coordinates all memory operations
- **MemoryEntry**: Represents individual memories with content, embeddings, and metadata
- **MemoryAssembly**: Groups related memories with a composite embedding
- **UnifiedQuickRecallCalculator**: Calculates memory relevance scores

**Responsibilities:**
- Memory storage and retrieval
- Memory assembly management
- QuickRecal score calculation
- Vector similarity search

#### 2. API Layer

The API Layer provides interfaces for external systems to interact with the memory core.

**Key Components:**
- **API Server**: FastAPI implementation of the RESTful API
- **API Client**: Python client for interacting with the API
- **Request/Response Models**: Pydantic models for validation

**Responsibilities:**
- Expose memory operations via RESTful endpoints
- Handle request validation
- Process API requests asynchronously
- Provide client interface for API consumers

#### 3. Emotional Intelligence

The Emotional Intelligence components handle emotion analysis and emotional gating.

**Key Components:**
- **EmotionalAnalyzer**: Analyzes emotional content in text
- **EmotionalGatingService**: Filters memories based on emotional context

**Responsibilities:**
- Detect emotions in text content
- Calculate emotional resonance between memories and user state
- Apply emotional gating to memory retrieval
- Enrich memory metadata with emotional context

#### 4. Trainer Integration

The Trainer Integration components interface with external training systems.

**Key Components:**
- **TrainerIntegrationManager**: Manages integration with external training systems
- **SequenceEmbeddingsResponse**: Provides sequential memory embeddings for training

**Responsibilities:**
- Provide sequential memory embeddings for training
- Accept feedback on QuickRecal scores
- Support continuous learning
- Track narrative surprise

#### 5. Adaptive Components

The Adaptive Components adjust system behavior based on feedback.

**Key Components:**
- **ThresholdCalibrator**: Dynamically adjusts similarity thresholds
- **AdaptiveBatchScheduler**: (Optional) Optimizes batch processing

**Responsibilities:**
- Adjust thresholds based on relevance feedback
- Optimize system parameters dynamically
- Track performance metrics
- Implement learning from feedback

#### 6. Geometry Manager

The Geometry Manager handles embedding spaces and transformations.

**Key Components:**
- **GeometryManager**: Manages different geometry types
- **GeometryType**: Enumeration of supported geometries

**Responsibilities:**
- Handle different embedding space geometries
- Perform transformations between spaces
- Calculate distances in appropriate spaces
- Optimize embedding representations

#### 7. Persistence Layer

The Persistence Layer handles storage and retrieval of memories.

**Key Components:**
- **MemoryPersistence**: Manages persistent storage of memories
- **Vector Index**: Efficient index for vector similarity search

**Responsibilities:**
- Store memories persistently
- Manage vector indices
- Handle serialization/deserialization
- Implement efficient retrieval

#### 8. Transcription Features

The Transcription Features components extract features from transcribed speech.

**Key Components:**
- **TranscriptionFeatureExtractor**: Extracts features from transcriptions
- **InterruptionAwareMemoryHandler**: Handles interruptions in processing

**Responsibilities:**
- Extract features from transcribed speech
- Analyze audio metadata
- Enrich transcription memories
- Handle processing interruptions

## Data Flow

### Memory Processing Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Input   │────▶│ Embedding│────▶│ Emotion  │────▶│ QuickRecal│
│  Content │     │Generation│     │ Analysis │     │Calculation│
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                                                        │
┌──────────┐     ┌──────────┐     ┌──────────┐         ▼
│  Memory  │◀────│ Metadata │◀────│ Memory   │◀────┌──────────┐
│  Storage │     │ Synthesis│     │ Creation │     │ Assembly │
└──────────┘     └──────────┘     └──────────┘     │ Check    │
                                                    └──────────┘
```

1. **Input Content**: Text content is received with optional metadata
2. **Embedding Generation**: Content is embedded using Sentence Transformers
3. **Emotion Analysis**: Emotional content is analyzed
4. **QuickRecal Calculation**: Relevance score is calculated
5. **Assembly Check**: Memory is checked against existing assemblies
6. **Memory Creation**: MemoryEntry object is created
7. **Metadata Synthesis**: Metadata is enriched with system-generated information
8. **Memory Storage**: Memory is stored in the persistence layer and vector index

### Memory Retrieval Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Query   │────▶│ Embedding│────▶│  Vector  │────▶│ Emotional│
│  Input   │     │Generation│     │  Search  │     │  Gating  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                                                        │
┌──────────┐     ┌──────────┐     ┌──────────┐         ▼
│  Result  │◀────│ Memory   │◀────│ Threshold│◀────┌──────────┐
│ Formatting│     │ Fetching │     │  Filter  │     │ Metadata │
└──────────┘     └──────────┘     └──────────┘     │  Filter  │
                                                    └──────────┘
```

1. **Query Input**: Query text is received with optional parameters
2. **Embedding Generation**: Query is embedded using Sentence Transformers
3. **Vector Search**: Similar vectors are found in the vector index
4. **Emotional Gating**: Results are filtered based on emotional context
5. **Metadata Filter**: Results are filtered based on metadata criteria
6. **Threshold Filter**: Results below the similarity threshold are removed
7. **Memory Fetching**: Full memory objects are fetched for remaining results
8. **Result Formatting**: Results are formatted for return

### Feedback Loop

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Retrieval│────▶│  User    │────▶│ Feedback │
│ Results  │     │Interaction│     │ Capture  │
└──────────┘     └──────────┘     └──────────┘
                                       │
┌──────────┐     ┌──────────┐         ▼
│  System  │◀────│ Threshold│◀────┌──────────┐
│Adjustment│     │Calibration│     │ Feedback │
└──────────┘     └──────────┘     │ Analysis │
                                   └──────────┘
```

1. **Retrieval Results**: Memory retrieval results are presented
2. **User Interaction**: User interacts with the results
3. **Feedback Capture**: Relevance feedback is captured
4. **Feedback Analysis**: Feedback is analyzed for patterns
5. **Threshold Calibration**: Similarity thresholds are adjusted
6. **System Adjustment**: System parameters are optimized

## Implementation Details

### Memory Structures

```python
class MemoryEntry:
    """Represents a single memory entry."""
    
    id: str                      # Unique identifier
    content: str                 # Text content
    embedding: np.ndarray        # Vector embedding
    timestamp: float             # Creation timestamp
    quickrecal_score: float      # Relevance score
    metadata: Dict[str, Any]     # Metadata dictionary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary representation."""
```

```python
class MemoryAssembly:
    """Represents a group of related memories."""
    
    id: str                      # Unique identifier
    name: str                    # Assembly name
    memories: Set[str]           # Set of memory IDs
    composite_embedding: np.ndarray  # Composite embedding
    last_activation: datetime    # Last activation time
    metadata: Dict[str, Any]     # Metadata dictionary
    
    def add_memory(self, memory_id: str) -> None:
        """Add a memory to the assembly."""
        
    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory from the assembly."""
        
    def update_composite_embedding(self, embeddings: List[np.ndarray]) -> None:
        """Update the composite embedding."""
```

### QuickRecal Calculation

The UnifiedQuickRecallCalculator uses a weighted combination of factors:

```python
def calculate_score(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
    """Calculate QuickRecal score based on multiple factors."""
    
    # Extract context factors
    recency = context.get("timestamp", time.time())
    importance = context.get("importance", 0.5)
    emotion_intensity = context.get("emotional_intensity", 0.5)
    
    # Calculate individual factor scores
    recency_score = self._calculate_recency_score(recency)
    importance_score = importance
    emotion_score = self._calculate_emotion_score(emotion_intensity)
    
    # Apply mode-specific weights
    if self.mode == QuickRecallMode.RECENCY_FOCUSED:
        weights = {"recency": 0.6, "importance": 0.2, "emotion": 0.2}
    elif self.mode == QuickRecallMode.IMPORTANCE_FOCUSED:
        weights = {"recency": 0.2, "importance": 0.6, "emotion": 0.2}
    else:  # BALANCED
        weights = {"recency": 0.33, "importance": 0.33, "emotion": 0.33}
    
    # Calculate weighted score
    score = (
        weights["recency"] * recency_score +
        weights["importance"] * importance_score +
        weights["emotion"] * emotion_score
    )
    
    return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1
```

### Emotional Gating

The EmotionalGatingService filters memories based on emotional context:

```python
def apply_gating(
    self, 
    memories: List[MemoryEntry], 
    user_emotion: Dict[str, Any],
    cognitive_load: float = 0.5
) -> List[Tuple[MemoryEntry, float]]:
    """Apply emotional gating to memories."""
    
    # Extract user's dominant emotion
    dominant_emotion = user_emotion.get("dominant_emotion", "neutral")
    
    results = []
    for memory in memories:
        # Extract memory's emotional context
        memory_emotion = memory.metadata.get("emotional_context", {})
        memory_dominant = memory_emotion.get("dominant_emotion", "neutral")
        
        # Calculate emotional resonance
        resonance = self._calculate_resonance(dominant_emotion, memory_dominant)
        
        # Apply cognitive load filter
        # Higher cognitive load = stricter filtering
        threshold = 0.3 + (cognitive_load * 0.4)  # Range: 0.3 - 0.7
        
        if resonance >= threshold:
            # Include memory with its resonance score
            results.append((memory, resonance))
    
    # Sort by resonance score
    return sorted(results, key=lambda x: x[1], reverse=True)
```

### Threshold Calibration

The ThresholdCalibrator adjusts similarity thresholds based on feedback:

```python
def adjust_threshold(self) -> float:
    """Adjust the similarity threshold based on recent feedback."""
    
    if len(self.feedback_history) < 10:  # Need minimum feedback
        return self.threshold
    
    # Calculate Precision and Recall from recent history
    recent_feedback = list(self.feedback_history)
    recent_tp = sum(1 for f in recent_feedback if f["predicted_relevant"] and f["relevant"])
    recent_fp = sum(1 for f in recent_feedback if f["predicted_relevant"] and not f["relevant"])
    recent_fn = sum(1 for f in recent_feedback if not f["predicted_relevant"] and f["relevant"])
    
    precision = recent_tp / max(1, recent_tp + recent_fp)
    recall = recent_tp / max(1, recent_tp + recent_fn)
    
    adjustment = 0.0
    # If precision is low (too many irrelevant items), increase threshold
    if precision < 0.6 and recall > 0.5:
        adjustment = self.learning_rate * (1.0 - precision)
    # If recall is low (too many relevant items missed), decrease threshold
    elif recall < 0.6 and precision > 0.5:
        adjustment = -self.learning_rate * (1.0 - recall)
    
    # Apply adjustment with diminishing returns near bounds
    current_threshold = self.threshold
    if adjustment > 0:
        adjustment *= (1.0 - current_threshold)  # Less adjustment as we approach 1.0
    else:
        adjustment *= current_threshold  # Less adjustment as we approach 0.0
    
    new_threshold = current_threshold + adjustment
    new_threshold = max(0.1, min(0.95, new_threshold))  # Keep within reasonable bounds
    
    self.threshold = new_threshold
    return self.threshold
```

## Integration Points

### External System Integration

Synthians Memory Core can be integrated with external systems through:

1. **Direct Library Usage**: Import and use the core components directly
2. **API Integration**: Interact with the system through the RESTful API
3. **Trainer Integration**: Connect external training systems for continuous learning

### Embedding Model Integration

The system supports custom embedding models:

```python
# Initialize with custom embedding model
from sentence_transformers import SentenceTransformer

custom_model = SentenceTransformer("custom-model-name")
memory_core = SynthiansMemoryCore(embedding_model=custom_model)
```

### Storage Backend Integration

The persistence layer can be configured to use different storage backends:

```python
# Initialize with custom storage path
memory_core = SynthiansMemoryCore(
    storage_path="/custom/storage/path",
    vector_index_type="Cosine"  # Options: "L2", "IP", "Cosine"
)
```

## Performance Considerations

### Memory Usage

- Embeddings are stored as 32-bit float arrays
- A typical 768-dimension embedding uses ~3KB of memory
- 100,000 memories would require ~300MB for embeddings alone
- Additional memory is used for content, metadata, and indices

### Computational Complexity

- Vector search: O(log n) with approximate nearest neighbor algorithms
- Memory processing: O(1) per memory
- Assembly operations: O(m) where m is the number of memories in the assembly
- Threshold calibration: O(w) where w is the feedback window size

### Scaling Strategies

- Implement sharding for large memory collections
- Use distributed vector indices for improved search performance
- Implement caching for frequently accessed memories
- Consider batch processing for large import operations

## Security Considerations

- The API does not implement authentication by default
- For production deployments, implement appropriate authentication and authorization
- Consider encrypting sensitive memory content
- Implement access controls for memory operations
- Regularly back up the memory storage

## Deployment Architecture

For production deployments, consider the following architecture:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   API       │────▶│   Memory    │
│ Applications│     │ Gateway     │     │   Core      │
└─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                          ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │ Authentication│    │  Storage    │
                    │    Service   │    │  Backend    │
                    └─────────────┘    └─────────────┘
```

- Deploy the API server behind an API gateway
- Implement authentication and rate limiting
- Use a scalable storage backend
- Consider containerization for easy deployment
- Implement monitoring and logging
