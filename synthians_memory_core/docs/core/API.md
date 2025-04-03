# Synthians Memory Core API Reference

This document provides a comprehensive reference for the Synthians Memory Core API, including all endpoints, request/response models, and usage examples.

## Base URL

By default, the API server runs at `http://localhost:5010`.

## Authentication

Currently, the API does not implement authentication. For production deployments, it is recommended to implement an authentication layer.

## Core Memory Operations

### Process Memory

Process and store a new memory in the system.

- **Endpoint**: `/process_memory`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "content": "Memory content text",
    "embedding": [0.1, 0.2, ...],  // Optional pre-computed embedding
    "metadata": {                  // Optional metadata
      "source": "user_input",
      "importance": 0.8
    },
    "analyze_emotion": true        // Optional, defaults to true
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "memory_id": "mem_12345",
    "quickrecal_score": 0.85,
    "embedding": [0.1, 0.2, ...],
    "metadata": {
      "source": "user_input",
      "importance": 0.8,
      "timestamp": 1648756892.45,
      "emotional_context": {
        "dominant_emotion": "neutral",
        "emotions": {
          "joy": 0.1,
          "sadness": 0.05,
          "neutral": 0.8
        }
      }
    }
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

### Retrieve Memories

Retrieve relevant memories based on a query.

- **Endpoint**: `/retrieve_memories`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "query": "Search query text",
    "query_embedding": [0.1, 0.2, ...],  // Optional pre-computed embedding
    "top_k": 5,                          // Number of results to return
    "user_emotion": {                    // Optional emotional context
      "dominant_emotion": "focused"
    },
    "cognitive_load": 0.5,               // Optional (0.0-1.0)
    "threshold": 0.7,                    // Optional similarity threshold
    "metadata_filter": {                 // Optional metadata filter
      "source": "meeting_notes"
    }
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "memories": [
      {
        "id": "mem_12345",
        "content": "Memory content text",
        "embedding": [0.1, 0.2, ...],
        "timestamp": 1648756892.45,
        "quickrecal_score": 0.85,
        "metadata": { ... },
        "similarity": 0.92,
        "emotional_resonance": 0.88,
        "final_score": 0.90
      },
      // More memories...
    ]
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

### Get Memory by ID

Retrieve a specific memory by its ID.

- **Endpoint**: `/api/memories/{memory_id}`
- **Method**: `GET`
- **URL Parameters**:
  - `memory_id`: The unique ID of the memory to retrieve
- **Response**:
  ```json
  {
    "success": true,
    "memory": {
      "id": "mem_12345",
      "content": "Memory content text",
      "embedding": [0.1, 0.2, ...],
      "timestamp": 1648756892.45,
      "quickrecal_score": 0.85,
      "metadata": { ... }
    }
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Memory not found"
  }
  ```

## Utility Endpoints

### Generate Embedding

Generate an embedding vector for text.

- **Endpoint**: `/generate_embedding`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "Text to embed"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "embedding": [0.1, 0.2, ...],
    "dimension": 768
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

### Calculate QuickRecal Score

Calculate relevance score for text or embedding.

- **Endpoint**: `/calculate_quickrecal`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "Text to score",           // Optional if embedding provided
    "embedding": [0.1, 0.2, ...],      // Optional if text provided
    "context": {                       // Optional context factors
      "importance": 0.8,
      "recency": 0.9
    }
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "quickrecal_score": 0.85,
    "factors": {
      "recency": 0.9,
      "importance": 0.8,
      "emotion": 0.7,
      "context": 0.85
    }
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

### Analyze Emotion

Analyze emotional content in text.

- **Endpoint**: `/analyze_emotion`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "Text to analyze for emotional content"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "emotions": {
      "joy": 0.8,
      "sadness": 0.1,
      "anger": 0.05,
      "fear": 0.02,
      "surprise": 0.03
    },
    "dominant_emotion": "joy"
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

### Provide Feedback

Provide feedback on retrieval relevance for threshold calibration.

- **Endpoint**: `/provide_feedback`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "memory_id": "mem_12345",
    "similarity_score": 0.82,
    "was_relevant": true
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "new_threshold": 0.78
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

## Advanced Feature Endpoints

### Process Transcription

Process transcribed speech with feature extraction.

- **Endpoint**: `/process_transcription`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "Transcribed speech text",
    "audio_metadata": {                // Optional audio metadata
      "speaker": "Alice",
      "meeting_id": "meeting-123",
      "speaking_rate": 1.2,
      "pauses": [3.5, 8.2],
      "interruption": false,
      "confidence": 0.92
    },
    "embedding": [0.1, 0.2, ...],      // Optional pre-computed embedding
    "memory_id": "mem_12345",          // Optional for updating existing memory
    "importance": 0.8,                 // Optional importance score
    "force_update": false              // Optional, defaults to false
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "memory_id": "mem_12345",
    "metadata": {
      "input_modality": "spoken",
      "source": "transcription",
      "speaker": "Alice",
      "meeting_id": "meeting-123",
      "speaking_rate": 1.2,
      "dominant_emotion": "neutral",
      "complexity_estimate": 0.65,
      "timestamp": 1648756892.45
    },
    "embedding": [0.1, 0.2, ...]
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

### Detect Contradictions

Identify potentially contradictory memories.

- **Endpoint**: `/detect_contradictions`
- **Method**: `POST`
- **Query Parameters**:
  - `threshold`: Similarity threshold for considering memories potentially contradictory (default: 0.75)
- **Response**:
  ```json
  {
    "success": true,
    "contradictions": [
      {
        "memory_a_id": "mem_12345",
        "memory_a_content": "The project deadline is end of Q3.",
        "memory_b_id": "mem_67890",
        "memory_b_content": "All deliverables must be completed by end of Q2.",
        "similarity": 0.82,
        "overlap_ratio": 0.45
      }
      // More contradictions...
    ],
    "count": 1
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

### Get Sequence Embeddings

Retrieve sequential memory embeddings for training.

- **Endpoint**: `/api/memories/get_sequence_embeddings`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "topic": "project_planning",       // Optional topic filter
    "user": "alice",                   // Optional user filter
    "emotion": "focused",              // Optional emotion filter
    "min_importance": 0.7,             // Optional importance threshold
    "limit": 100,                      // Max number of embeddings to return
    "min_quickrecal_score": 0.6,       // Optional QuickRecal threshold
    "start_timestamp": "1648756892.45", // Optional start time
    "end_timestamp": "1648843292.45",  // Optional end time
    "sort_by": "timestamp"             // Sort field: "timestamp" or "quickrecal_score"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "embeddings": [
      [0.1, 0.2, ...],
      // More embeddings...
    ],
    "memory_ids": ["mem_12345", "mem_67890", ...],
    "timestamps": [1648756892.45, 1648756992.45, ...],
    "count": 100
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

### Update QuickRecal Score

Update a memory's QuickRecal score based on feedback.

- **Endpoint**: `/api/memories/update_quickrecal_score`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "memory_id": "mem_12345",
    "delta": 0.2,                      // Score adjustment (positive or negative)
    "reason": "high_surprise"          // Optional reason for adjustment
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "memory_id": "mem_12345",
    "old_score": 0.7,
    "new_score": 0.9,
    "delta": 0.2
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

### Repair Index

Repair the vector index (maintenance endpoint).

- **Endpoint**: `/repair_index`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "repair_type": "auto"              // Repair type: "auto", "rebuild", "verify"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "repair_type": "auto",
    "fixed_count": 5,
    "total_checked": 1000,
    "duration_seconds": 2.5
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

## System Endpoints

### Health Check

Check system health and uptime.

- **Endpoint**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "healthy",
    "uptime_seconds": 1234.56,
    "memory_count": 500,
    "assembly_count": 50,
    "version": "1.0.0"
  }
  ```
- **Error Response**:
  ```json
  {
    "status": "unhealthy",
    "error": "Description of the error"
  }
  ```

### Get Statistics

Retrieve detailed system statistics.

- **Endpoint**: `/stats`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "success": true,
    "api_server": {
      "uptime_seconds": 1234.56,
      "memory_count": 500,
      "embedding_dim": 768,
      "geometry": "hyperbolic",
      "model": "all-mpnet-base-v2"
    },
    "memory": {
      "total_memories": 500,
      "total_assemblies": 50,
      "storage_path": "/app/memory/stored/synthians",
      "threshold": 0.75
    },
    "vector_index": {
      "count": 500,
      "id_mappings": 500,
      "index_type": "Cosine"
    }
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Description of the error retrieving stats"
  }
  ```

### List Assemblies

List all memory assemblies.

- **Endpoint**: `/assemblies`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "success": true,
    "assemblies": [
      {
        "assembly_id": "assembly_12345",
        "name": "Project Alpha Documentation",
        "memory_count": 15,
        "last_activation": "2023-04-01T14:30:45.123Z"
      }
      // More assemblies...
    ],
    "count": 50
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Error message"
  }
  ```

### Get Assembly

Get details for a specific assembly.

- **Endpoint**: `/assemblies/{assembly_id}`
- **Method**: `GET`
- **URL Parameters**:
  - `assembly_id`: The unique ID of the assembly to retrieve
- **Response**:
  ```json
  {
    "success": true,
    "assembly_id": "assembly_12345",
    "name": "Project Alpha Documentation",
    "memory_count": 15,
    "last_activation": "2023-04-01T14:30:45.123Z",
    "sample_memories": [
      {
        "id": "mem_12345",
        "content": "Memory content text",
        "quickrecal_score": 0.85
      }
      // More memories (limited to 10)...
    ],
    "total_memories": 15,
    "memory_ids": ["mem_12345", "mem_67890", ...],
    "composite_embedding": [0.1, 0.2, ...],
    "assembly_schema_version": "1.0"
  }
  ```
- **Error Response**:
  ```json
  {
    "success": false,
    "error": "Assembly not found"
  }
  ```

## Client Usage Examples

### Basic Memory Operations

```python
import asyncio
from synthians_memory_core.api.client.client import SynthiansClient

async def memory_operations_example():
    async with SynthiansClient(base_url="http://localhost:5010") as client:
        try:
            # Store a memory
            store_response = await client.process_memory(
                content="Important meeting notes about the Q3 roadmap.",
                metadata={
                    "source": "meeting_notes",
                    "importance": 0.8,
                    "project": "RoadmapQ3"
                }
            )
            
            if not store_response.get("success"):
                print(f"Error storing memory: {store_response.get('error')}")
                return
                
            memory_id = store_response.get("memory_id")
            print(f"Stored memory with ID: {memory_id}")
            
            # Retrieve memories
            retrieve_response = await client.retrieve_memories(
                query="roadmap planning",
                top_k=3,
                metadata_filter={"source": "meeting_notes"}
            )
            
            if not retrieve_response.get("success"):
                print(f"Error retrieving memories: {retrieve_response.get('error')}")
                return
                
            # Print results
            for memory in retrieve_response.get("memories", []):
                print(f"ID: {memory.get('id')}, Score: {memory.get('similarity'):.4f}")
                print(f"Content: {memory.get('content')}")
                
            # Get memory by ID
            get_response = await client.get_memory_by_id(memory_id)
            
            if not get_response.get("success"):
                print(f"Error getting memory: {get_response.get('error')}")
                return
                
            print(f"Retrieved memory by ID: {get_response.get('memory').get('content')}")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(memory_operations_example())
```

### Advanced Features

```python
import asyncio
from synthians_memory_core.api.client.client import SynthiansClient

async def advanced_features_example():
    async with SynthiansClient(base_url="http://localhost:5010") as client:
        try:
            # Analyze emotion
            emotion_response = await client.analyze_emotion(
                "I'm thrilled about the progress we've made on the project!"
            )
            
            if not emotion_response.get("success"):
                print(f"Error analyzing emotion: {emotion_response.get('error')}")
                return
                
            print(f"Dominant emotion: {emotion_response.get('dominant_emotion')}")
            print(f"Emotion scores: {emotion_response.get('emotions')}")
            
            # Process transcription
            transcription_response = await client.process_transcription(
                text="We should prioritize the user experience improvements.",
                audio_metadata={
                    "speaker": "Alice",
                    "meeting_id": "planning-2023-05-15",
                    "speaking_rate": 1.2,
                    "pauses": [3.5, 8.2],
                    "interruption": False,
                    "confidence": 0.92
                },
                importance=0.8
            )
            
            if not transcription_response.get("success"):
                print(f"Error processing transcription: {transcription_response.get('error')}")
                return
                
            print(f"Transcription processed with ID: {transcription_response.get('memory_id')}")
            
            # Detect contradictions
            contradiction_response = await client.detect_contradictions(threshold=0.7)
            
            if not contradiction_response.get("success"):
                print(f"Error detecting contradictions: {contradiction_response.get('error')}")
                return
                
            print(f"Found {contradiction_response.get('count')} potential contradictions")
            
            # Repair index
            repair_response = await client.repair_index(repair_type="auto")
            
            if not repair_response.get("success"):
                print(f"Error repairing index: {repair_response.get('error')}")
                return
                
            print(f"Index repaired successfully. Fixed {repair_response.get('fixed_count')} issues.")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(advanced_features_example())
```

## Error Handling

The API returns appropriate HTTP status codes along with error messages in the response body. Common error scenarios include:

- `400 Bad Request`: Invalid request parameters or body
- `404 Not Found`: Resource not found (e.g., memory ID doesn't exist)
- `500 Internal Server Error`: Server-side error

Example error handling:

```python
import asyncio
from synthians_memory_core.api.client.client import SynthiansClient

async def error_handling_example():
    async with SynthiansClient(base_url="http://localhost:5010") as client:
        try:
            # Try to get a non-existent memory
            response = await client.get_memory_by_id("non_existent_id")
            
            if not response.get("success"):
                if "not found" in response.get("error", "").lower():
                    print("Memory not found - handle this specific case")
                else:
                    print(f"Other error occurred: {response.get('error')}")
                return
                
            # Process the successful response
            print(f"Memory found: {response.get('memory').get('content')}")
            
        except Exception as e:
            print(f"Network or other error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(error_handling_example())
```
