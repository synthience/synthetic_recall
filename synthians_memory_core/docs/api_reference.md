# API Reference

This document provides a detailed reference for the APIs exposed by each component in the Bi-Hemispheric Cognitive Architecture.

## Table of Contents

1. [Memory Core API](#memory-core-api)
2. [Trainer Server API](#trainer-server-api)
3. [Context Cascade Engine API](#context-cascade-engine-api)
4. [Memory Assembly Management](#memory-assembly-management)
5. [Error Handling](#error-handling)

## Memory Core API

Base URL: `http://localhost:8000`

### Process Memory

```
POST /api/memories/process
```

Processes a new memory, generating embeddings and enriching metadata.

**Request Body:**

```json
{
  "content": "Memory text content",
  "source": "source_identifier",
  "timestamp": "2025-03-27T20:10:30Z",
  "embedding": [0.1, 0.2, ...],  // Optional pre-computed embedding
  "metadata": {  // Optional metadata
    "user": "user_id",
    "topic": "conversation_topic",
    "emotions": {"joy": 0.8, "surprise": 0.2}
  }
}
```

**Response:**

```json
{
  "id": "memory_uuid",
  "content": "Memory text content",
  "embedding": [0.1, 0.2, ...],
  "timestamp": "2025-03-27T20:10:30Z",
  "quickrecal_score": 0.85,
  "metadata": {
    "user": "user_id",
    "topic": "conversation_topic",
    "emotions": {"joy": 0.8, "surprise": 0.2},
    "dominant_emotion": "joy",
    "importance": 0.75,
    "content_length": 120
  }
}
```

### Retrieve Memories

```
POST /api/memories/retrieve
```

Retrieves memories based on similarity to a query embedding.

**Request Body:**

```json
{
  "embedding": [0.1, 0.2, ...],
  "limit": 10,
  "threshold": 0.3,
  "filters": {
    "topic": "optional_topic_filter",
    "user": "optional_user_filter",
    "emotion": "optional_emotion_filter"
  }
}
```

**Response:**

```json
{
  "memories": [
    {
      "id": "memory_uuid",
      "content": "Memory text content",
      "embedding": [0.1, 0.2, ...],
      "timestamp": "2025-03-27T20:10:30Z",
      "similarity": 0.92,
      "quickrecal_score": 0.85,
      "metadata": {...}
    },
    // Additional memories
  ]
}
```

### Get Sequence Embeddings

```
POST /api/memories/get_sequence_embeddings
```

Retrieves a sequence of memory embeddings for training or prediction.

**Request Body:**

```json
{
  "topic": "optional_topic",
  "user": "optional_user",
  "emotion": "optional_emotion",
  "min_importance": 0.5,
  "limit": 100,
  "min_quickrecal_score": 0.3,
  "start_timestamp": "2025-03-20T00:00:00Z",
  "end_timestamp": "2025-03-27T23:59:59Z",
  "sort_by": "timestamp"  // or "quickrecal_score"
}
```

**Response:**

```json
{
  "embeddings": [
    {
      "id": "memory_uuid",
      "embedding": [0.1, 0.2, ...],
      "timestamp": "2025-03-27T20:10:30Z",
      "quickrecal_score": 0.85,
      "emotion": {"joy": 0.8, "surprise": 0.2},
      "dominant_emotion": "joy",
      "importance": 0.75,
      "topic": "conversation_topic",
      "user": "user_id"
    },
    // Additional embeddings
  ]
}
```

### Update QuickRecal Score

```
POST /api/memories/update_quickrecal_score
```

Updates the quickrecal score of a memory based on surprise feedback.

**Request Body:**

```json
{
  "memory_id": "memory_uuid",
  "delta": 0.2,
  "predicted_embedding": [0.1, 0.2, ...],
  "reason": "Surprise score: 0.8, context surprise: 0.3",
  "embedding_delta": [0.05, -0.03, ...]
}
```

**Response:**

```json
{
  "status": "success",
  "memory_id": "memory_uuid",
  "previous_score": 0.65,
  "new_score": 0.85,
  "delta": 0.2
}
```

## Trainer Server API

Base URL: `http://localhost:8001`

### Health Check

```
GET /health
```

Checks the health status of the Trainer Server.

**Response:**

```json
{
  "status": "ok",
  "timestamp": "2025-03-27T20:10:30Z"
}
```

### Initialize Trainer

```
POST /init
```

Initializes the sequence trainer model with configuration.

**Request Body:**

```json
{
  "inputDim": 768,
  "hiddenDim": 512,
  "outputDim": 768,
  "memoryDim": 256,
  "learningRate": 0.001
}
```

**Response:**

```json
{
  "message": "Sequence trainer model initialized",
  "config": {
    "inputDim": 768,
    "hiddenDim": 512,
    "outputDim": 768,
    "memoryDim": 256,
    "learningRate": 0.001
  }
}
```

### Predict Next Embedding

```
POST /predict_next_embedding
```

Predicts the next embedding based on a sequence of input embeddings. This endpoint is fully stateless, relying on the `previous_memory_state` parameter for continuity between calls.

**Request Body:**

```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
  "previous_memory_state": {  // Required for stateless operation
    "sequence": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "surprise_history": [0.1, 0.2],
    "momentum": [0.1, 0.2, ...]
  }
}
```

**Response:**

```json
{
  "predicted_embedding": [0.1, 0.2, ...],
  "surprise_score": 0.35,
  "memory_state": {  // State to pass in the next prediction request
    "sequence": [[0.3, 0.4, ...], [0.5, 0.6, ...]],
    "surprise_history": [0.2, 0.35],
    "momentum": [0.15, 0.25, ...]
  }
}
```

### Train Sequence

```
POST /train_sequence
```

Trains the model on a sequence of embeddings.

**Request Body:**

```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
}
```

**Response:**

```json
{
  "success": true,
  "loss": 0.015,
  "iterations": 10,
  "message": "Training successful"
}
```

### Analyze Surprise

```
POST /analyze_surprise
```

Analyzes the surprise between predicted and actual embeddings.

**Request Body:**

```json
{
  "predicted_embedding": [0.1, 0.2, ...],
  "actual_embedding": [0.1, 0.3, ...]
}
```

**Response:**

```json
{
  "surprise": 0.15,
  "cosine_surprise": 0.12,
  "context_surprise": 0.18,
  "delta_norm": 0.22,
  "is_surprising": true,
  "adaptive_threshold": 0.10,
  "volatility": 0.05,
  "delta": [0.0, 0.1, ...],
  "quickrecal_boost": 0.15
}
```

## Context Cascade Engine API

Base URL: `http://localhost:8002`

### Process Memory

```
POST /api/process_memory
```

Processes a memory through the full cognitive pipeline.

**Request Body:**

```json
{
  "content": "Memory text content",
  "embedding": [0.1, 0.2, ...],  // Optional pre-computed embedding
  "metadata": {  // Optional metadata
    "user": "user_id",
    "topic": "conversation_topic",
    "emotions": {"joy": 0.8, "surprise": 0.2}
  }
}
```

**Response:**

```json
{
  "memory_id": "memory_uuid",
  "status": "processed",
  "timestamp": "2025-03-27T20:10:30Z",
  "surprise": {
    "score": 0.35,
    "threshold": 0.6,
    "is_surprising": false,
    "factors": {
      "geometric": 0.28,
      "contextual": 0.42,
      "semantic": 0.35
    }
  },
  "prediction": {
    "predicted_embedding": [0.15, 0.25, ...],
    "confidence": 0.85
  },
  "memory_state": {
    "quickrecal_score": 0.85,
    "adjusted_score": 0.85  // Same as quickrecal_score if no surprise
  }
}
```

### Retrieve Memories

```
POST /api/retrieve_memories
```

Retrieves memories with enhanced context-aware filtering.

**Request Body:**

```json
{
  "query": "Query text content",
  "embedding": [0.1, 0.2, ...],  // Optional pre-computed query embedding
  "limit": 10,
  "threshold": 0.3,
  "current_emotion": {  // Optional emotional context
    "dominant": "joy",
    "values": {"joy": 0.8, "surprise": 0.2}
  },
  "cognitive_load": 0.5,  // Optional, 0.0-1.0
  "filters": {
    "topic": "optional_topic_filter",
    "user": "optional_user_filter",
    "emotion": "optional_emotion_filter"
  }
}
```

**Response:**

```json
{
  "memories": [
    {
      "id": "memory_uuid",
      "content": "Memory text content",
      "embedding": [0.1, 0.2, ...],
      "timestamp": "2025-03-27T20:10:30Z",
      "similarity": 0.92,
      "emotional_resonance": 0.85,
      "final_score": 0.89,  // Combined score after emotional gating
      "quickrecal_score": 0.85,
      "metadata": {...}
    },
    // Additional memories
  ],
  "context": {
    "query_emotion": "joy",
    "cognitive_load_applied": 0.5,
    "filters_applied": ["topic"],
    "emotional_gating_applied": true
  }
}
```

## Memory Assembly Management

The Memory Core provides APIs for creating and managing memory assemblies, which group related memories together.

### Create Assembly

```
POST /api/assemblies/create
```

Creates a new memory assembly.

**Request Body:**

```json
{
  "name": "Assembly Name",
  "description": "Optional assembly description",
  "initial_memories": ["memory_id_1", "memory_id_2", ...],
  "tags": ["tag1", "tag2"],
  "metadata": {  // Optional metadata
    "creator": "user_id",
    "category": "assembly_category"
  }
}
```

**Response:**

```json
{
  "assembly_id": "assembly_uuid",
  "name": "Assembly Name",
  "memory_count": 2,
  "composite_embedding": [0.1, 0.2, ...],
  "creation_time": "2025-03-27T20:10:30Z",
  "dominant_emotions": ["joy", "curiosity"],
  "keywords": ["keyword1", "keyword2"]
}
```

### Add Memory to Assembly

```
POST /api/assemblies/{assembly_id}/add_memory
```

Adds a memory to an existing assembly.

**Request Body:**

```json
{
  "memory_id": "memory_uuid"
}
```

**Response:**

```json
{
  "assembly_id": "assembly_uuid",
  "memory_id": "memory_uuid",
  "status": "added",
  "memory_count": 3,
  "updated_composite_embedding": [0.12, 0.22, ...]
}
```

### List Assemblies

```
GET /api/assemblies
```

Lists all available memory assemblies.

**Response:**

```json
{
  "assemblies": [
    {
      "assembly_id": "assembly_uuid_1",
      "name": "Assembly Name 1",
      "memory_count": 3,
      "creation_time": "2025-03-27T20:10:30Z",
      "dominant_emotions": ["joy", "curiosity"],
      "keywords": ["keyword1", "keyword2"]
    },
    // Additional assemblies
  ]
}
```

### Get Assembly Details

```
GET /api/assemblies/{assembly_id}
```

Retrieves detailed information about a specific assembly.

**Response:**

```json
{
  "assembly_id": "assembly_uuid",
  "name": "Assembly Name",
  "description": "Assembly description",
  "memory_count": 3,
  "composite_embedding": [0.12, 0.22, ...],
  "creation_time": "2025-03-27T20:10:30Z",
  "last_modified": "2025-03-27T21:10:30Z",
  "memories": [
    {
      "id": "memory_id_1",
      "content": "Memory text content 1",
      "timestamp": "2025-03-27T20:05:30Z"
    },
    // Additional memory summaries
  ],
  "dominant_emotions": ["joy", "curiosity"],
  "keywords": ["keyword1", "keyword2"],
  "metadata": {
    "creator": "user_id",
    "category": "assembly_category"
  }
}
```

### Delete Assembly

```
DELETE /api/assemblies/{assembly_id}
```

Deletes a memory assembly (this does not delete the individual memories).

**Response:**

```json
{
  "assembly_id": "assembly_uuid",
  "status": "deleted",
  "memory_count_released": 3
}
```

## Error Handling

All APIs follow a consistent error response format:

```json
{
  "error": "Error message description",
  "status": "error",
  "code": 404,  // HTTP status code or custom error code
  "details": {  // Optional additional error information
    "field": "field_with_error",
    "reason": "specific reason for error"
  }
}
```

Common HTTP status codes:

- **400 Bad Request**: Invalid request parameters or payload
- **404 Not Found**: Resource (memory, assembly, etc.) not found
- **500 Internal Server Error**: Server-side processing error
- **408 Request Timeout**: Service timeout (particularly for embedding generation)

Custom error codes:

- **"timeout"**: Connection timed out
- **"connection_refused"**: Service unavailable or cannot be reached
- **"unknown_error"**: Unspecified error
