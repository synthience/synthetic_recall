# Synthians Cognitive Architecture: API Reference

This document provides a reference for the APIs exposed by the Synthians Memory Core service.

**Date:** 2025-03-30
**Version:** 1.0.0

## Table of Contents

1.  [Synthians Memory Core API](#synthians-memory-core-api)
2.  [Common Error Handling](#common-error-handling)

---

## 1. Synthians Memory Core API

**Base URL:** `http://localhost:5010` (Default)

This service manages persistent memory storage, retrieval, scoring, embedding generation, emotion analysis, and related functionalities for the Synthians system.

---

### Root

*   **Method:** `GET`
*   **Path:** `/`
*   **Description:** Basic endpoint indicating the API is running.
*   **Response (Success):**
    ```json
    {
      "message": "Synthians Memory Core API"
    }
    ```

---

### Health Check

*   **Method:** `GET`
*   **Path:** `/health`
*   **Description:** Checks the health status of the Memory Core service, including uptime and basic counts.
*   **Response (Success):**
    ```json
    {
      "status": "healthy",
      "uptime_seconds": 1234.56,         // Example value
      "memory_count": 500,              // Example value
      "assembly_count": 50,             // Example value
      "version": "1.0.0"
    }
    ```
*   **Response (Error):**
    ```json
    {
      "status": "unhealthy",
      "error": "Description of the error"
    }
    ```

---

### Get Statistics

*   **Method:** `GET`
*   **Path:** `/stats`
*   **Description:** Retrieves detailed statistics about the Memory Core system, including memory counts, vector index status, geometry configuration, and API server details.
*   **Response (Success):**
    ```json
    {
      "success": true,
      "api_server": {
        "uptime_seconds": 1234.56,        // Example value
        "memory_count": 500,             // Example value - In-memory cache count
        "embedding_dim": 768,
        "geometry": "hyperbolic",        // Current geometry setting
        "model": "all-mpnet-base-v2"      // Configured embedding model
      },
      "memory": {
        "total_memories": 500,           // Example value - Total indexed memories
        "total_assemblies": 50,          // Example value
        "storage_path": "/app/memory/stored/synthians",
        "threshold": 0.75                // Configured default threshold (may differ from active retrieval thresholds which can be adaptive or request-specific)
      },
      "vector_index": {
        "count": 500,                    // Example value
        "id_mappings": 500,              // Example value
        "index_type": "Cosine"           // e.g., L2, IP, Cosine
      }
    }
    ```
*   **Response (Error):**
    ```json
    {
      "success": false,
      "error": "Description of the error retrieving stats"
    }
    ```

---

### Process Memory

*   **Method:** `POST`
*   **Path:** `/process_memory`
*   **Description:** Processes and stores a new memory entry. Generates embedding if not provided, calculates QuickRecal score, performs emotion analysis (optional), synthesizes metadata, and saves the memory. Handles potential embedding dimension mismatches.
*   **Request Model:** (`ProcessMemoryRequest`)
    ```json
    {
      "content": "string", // The text content of the memory
      "embedding": "Optional[List[float]]", // Optional pre-computed embedding
      "metadata": "Optional[Dict[str, Any]]", // Optional base metadata
      "analyze_emotion": "Optional[bool]" // Default: true. Set to false to skip emotion analysis.
    }
    ```
*   **Response Model:** (`ProcessMemoryResponse`)
    ```json
    {
      "success": true,
      "memory_id": "string", // Unique ID assigned to the memory
      "quickrecal_score": "float", // Calculated relevance score
      "embedding": "List[float]", // The embedding used/generated (potentially aligned)
      "metadata": "Dict[str, Any]", // Enriched metadata after synthesis
      "error": null // Or error string on failure
    }
    ```

---

### Retrieve Memories

*   **Method:** `POST`
*   **Path:** `/retrieve_memories`
*   **Description:** Retrieves relevant memories based on a query string. Generates query embedding, performs vector search, applies emotional gating, and uses adaptive thresholding (if enabled).
*   **Request Model:** (`RetrieveMemoriesRequest`)
    ```json
    {
      "query": "string", // The search query text
      "query_embedding": "Optional[List[float]]", // Pre-computed embedding vector; rarely needed as the system will automatically generate an embedding from the query text
      "top_k": "int", // Default: 5. Max number of results to return.
      "user_emotion": "Optional[Union[Dict[str, Any], str]]", // e.g., {"dominant_emotion": "joy"} or "joy". Used for emotional gating.
      "cognitive_load": "float", // Default: 0.5. Influences emotional gating strictness.
      "threshold": "Optional[float]", // Explicit similarity threshold override (0.0-1.0). If None, uses adaptive threshold.
      "metadata_filter": "Optional[Dict[str, Any]]", // Filter memories by metadata fields (e.g., {"source": "user", "day_of_week": "monday"}). Supports nested keys with dots (e.g., "details.project").
      "search_strategy": "Optional[str]" // Determines the retrieval approach (e.g., "vector", "hybrid", "metadata"). If not specified, uses the system default.
    }
    ```
*   **Response Model:** (`RetrieveMemoriesResponse`)
    ```json
    {
      "success": true,
      "memories": [
        {
          "id": "string",
          "content": "string",
          "embedding": "List[float]",
          "timestamp": "float", // Unix timestamp
          "quickrecal_score": "float",
          "metadata": "Dict[str, Any]", // Includes synthesized metadata
          "similarity": "float", // Similarity score to the query
          "emotional_resonance": "Optional[float]", // Score from emotional gating (if applied)
          "final_score": "Optional[float]" // Combined score after gating (if applied)
          // ... other MemoryEntry fields serialized by to_dict()
        }
        // ... more memories up to top_k
      ],
      "error": null // Or error string on failure
    }
    ```

---

### Generate Embedding

*   **Method:** `POST`
*   **Path:** `/generate_embedding`
*   **Description:** Generates an embedding vector for the given text using the server's configured Sentence Transformer model.
*   **Request Model:** (`GenerateEmbeddingRequest`)
    ```json
    {
      "text": "string" // The text to embed
    }
    ```
*   **Response Model:** (`GenerateEmbeddingResponse`)
    ```json
    {
      "success": true,
      "embedding": "List[float]", // The generated embedding vector
      "dimension": "int", // The dimension of the embedding
      "error": null
    }
    ```

---

### Calculate QuickRecal Score

*   **Method:** `POST`
*   **Path:** `/calculate_quickrecal`
*   **Description:** Calculates the QuickRecal score for a given text or embedding, considering context factors. Generates embedding if only text is provided.
*   **Request Model:** (`QuickRecalRequest`)
    ```json
    {
      "embedding": "Optional[List[float]]", // Pre-computed embedding
      "text": "Optional[string]", // Text to generate embedding from if embedding not provided
      "context": "Optional[Dict[str, Any]]" // Context factors (e.g., timestamp, relevance, importance, metadata)
    }
    ```
*   **Response Model:** (`QuickRecalResponse`)
    ```json
    {
      "success": true,
      "quickrecal_score": "float", // The calculated score (0.0-1.0)
      "factors": "Optional[Dict[str, float]]", // Scores of individual contributing factors (e.g., recency, emotion)
      "error": null
    }
    ```

---

### Analyze Emotion

*   **Method:** `POST`
*   **Path:** `/analyze_emotion`
*   **Description:** Analyzes the emotional content of the given text using the server's `EmotionAnalyzer` (transformer model or keyword fallback).
*   **Request Model:** (`EmotionRequest`)
    ```json
    {
      "text": "string" // The text to analyze
    }
    ```
*   **Response Model:** (`EmotionResponse`)
    ```json
    {
      "success": true,
      "emotions": "Dict[str, float]", // Scores for different emotions (e.g., {"joy": 0.8, "sadness": 0.1})
      "dominant_emotion": "string", // The emotion with the highest score
      "error": null
    }
    ```

---

### Provide Feedback

*   **Method:** `POST`
*   **Path:** `/provide_feedback`
*   **Description:** Provides feedback on the relevance of a retrieved memory, used by the `ThresholdCalibrator` to adjust the adaptive similarity threshold.
*   **Request Model:** (`FeedbackRequest`)
    ```json
    {
      "memory_id": "string", // ID of the memory the feedback is about
      "similarity_score": "float", // The similarity score assigned during retrieval
      "was_relevant": "bool" // True if the user found it relevant, False otherwise
    }
    ```
*   **Response Model:** (`FeedbackResponse`)
    ```json
    {
      "success": true,
      "new_threshold": "Optional[float]", // The current adaptive threshold after adjustment
      "error": null
    }
    ```

---

### Detect Contradictions

*   **Method:** `POST`
*   **Path:** `/detect_contradictions`
*   **Description:** Attempts to detect potential contradictions among stored memories based on semantic similarity and content analysis (currently basic keyword checks for opposition).
*   **Query Parameter:** `threshold` (float, default: 0.75) - Similarity threshold for considering memories potentially contradictory.
*   **Response (Success):**
    ```json
    {
      "success": true,
      "contradictions": [
        {
           "memory_a_id": "string",
           "memory_a_content": "string",
           "memory_b_id": "string",
           "memory_b_content": "string",
           "similarity": "float",
           "overlap_ratio": "float" // Ratio of common words
        }
        // ... more potential contradictions
      ],
      "count": "int" // Number of contradiction pairs found
    }
    ```
*   **Response (Error):**
    ```json
    {
      "success": false,
      "error": "Description of the error"
    }
    ```

---

### Process Transcription

*   **Method:** `POST`
*   **Path:** `/process_transcription`
*   **Description:** Processes transcribed text, enriches it with features extracted from audio metadata (e.g., pauses, speaking rate, interruption info), performs emotion analysis, and stores it as a memory.
*   **Request Model:** (`TranscriptionRequest`)
    ```json
    {
      "text": "string", // The transcribed text
      "audio_metadata": "Optional[Dict[str, Any]]", // e.g., {"duration_sec": 5.2, "was_interrupted": true}
      "embedding": "Optional[List[float]]", // Optional pre-computed embedding
      "memory_id": "Optional[string]", // For updating an existing memory
      "importance": "Optional[float]", // Optional importance score (0-1)
      "force_update": "bool" // Default: false. Force update if memory_id exists.
    }
    ```
*   **Response Model:** (`TranscriptionResponse`)
    ```json
    {
      "success": true,
      "memory_id": "string", // ID of the created/updated memory
      "metadata": "Dict[str, Any]", // Enriched metadata including extracted audio features
      "embedding": "List[float]", // The embedding used/generated
      "error": null
    }
    ```

---

### Get Memory by ID

*   **Method:** `GET`
*   **Path:** `/api/memories/{memory_id}`
*   **Description:** Retrieves a specific memory entry by its unique identifier. Returns the complete memory object including content, embedding, and all metadata.
*   **Path Parameter:** `memory_id` (string) - The unique ID of the memory.
*   **Response Model:** (`GetMemoryResponse`)
    ```json
    {
      "success": true,
      "memory": { // Full MemoryEntry dictionary representation
        "id": "string",
        "content": "string",
        "embedding": "List[float]",
        "timestamp": "string", // ISO format UTC
        "quickrecal_score": "float",
        "quickrecal_updated": "Optional[string]", // ISO format UTC
        "metadata": "Dict[str, Any]",
        "access_count": "int",
        "last_access_time": "string", // ISO format UTC
        "hyperbolic_embedding": "Optional[List[float]]"
      },
      "error": null
    }
    ```
*   **Response (Not Found):**
    ```json
    {
      "success": false,
      "memory": null,
      "error": "Memory with ID '{memory_id}' not found"
    }
    ```

---

### List Assemblies

*   **Method:** `GET`
*   **Path:** `/assemblies`
*   **Description:** Lists basic information about all known memory assemblies.
*   **Response (Success):**
    ```json
    {
      "success": true,
      "assemblies": [
        {
          "assembly_id": "string",
          "name": "string",
          "memory_count": "int", // Number of memories in the assembly
          "last_activation": "string" // ISO format UTC timestamp
        }
        // ... more assemblies
      ],
      "count": "int" // Total number of assemblies
    }
    ```

---

### Get Assembly Details

*   **Method:** `GET`
*   **Path:** `/assemblies/{assembly_id}`
*   **Description:** Retrieves detailed information about a specific memory assembly, including a sample of its member memories.
*   **Path Parameter:** `assembly_id` (string) - The unique ID of the assembly.
*   **Response (Success):**
    ```json
    {
      "success": true,
      "assembly_id": "string",
      "name": "string",
      "memory_count": "int",
      "last_activation": "string", // ISO format UTC
      "sample_memories": [ // Limited sample (e.g., first 10) for brevity
        {
          "id": "string",
          "content": "string",
          "quickrecal_score": "float"
        }
        // ... up to 10 sample memories
      ],
      "total_memories": "int" // Total number of memories in the assembly
    }
    ```
*   **Response (Not Found):**
    ```json
    {
      "success": false,
      "error": "Assembly not found"
    }
    ```

---

### Get Sequence Embeddings (Trainer Integration)

*   **Method:** `POST`
*   **Path:** `/api/memories/get_sequence_embeddings`
*   **Description:** Retrieves a sequence of memory embeddings, ordered and filtered, suitable for feeding into a sequence trainer (e.g., Neural Memory Server).
*   **Request Model:** (Implicit, uses query parameters)
    *   **Query Parameters:** `topic`, `user`, `emotion`, `min_importance`, `limit`, `min_quickrecal_score`, `start_timestamp`, `end_timestamp`, `sort_by` (timestamp or quickrecal_score)
*   **Response Model:** (`SequenceEmbeddingsResponse`)
    ```json
    {
      "embeddings": [
        {
          "id": "string",
          "embedding": "List[float]",
          "timestamp": "string", // ISO format UTC
          "quickrecal_score": "Optional[float]",
          "emotion": "Optional[Dict[str, float]]",
          "dominant_emotion": "Optional[string]",
          "importance": "Optional[float]",
          "topic": "Optional[string]",
          "user": "Optional[string]"
        }
        // ... more embeddings up to limit
      ]
    }
    ```

---

### Update QuickRecal Score (Trainer Integration)

*   **Method:** `POST`
*   **Path:** `/api/memories/update_quickrecal_score`
*   **Description:** Allows an external system (like the Trainer or Orchestrator) to update a memory's QuickRecal score based on feedback, such as prediction surprise. Records the reason and context in the memory's metadata.
*   **Request Model:** (`UpdateQuickRecalScoreRequest`)
    ```json
    {
      "memory_id": "string", // ID of the memory to update
      "delta": "float", // Amount to change score by (+ve or -ve)
      "predicted_embedding": "Optional[List[float]]", // Embedding predicted by the trainer
      "reason": "Optional[string]", // e.g., "NM Surprise Loss: 0.65"
      "embedding_delta": "Optional[List[float]]" // Pre-calculated delta between actual and predicted
    }
    ```
*   **Response (Success):**
    ```json
    {
      "status": "success",
      "memory_id": "string",
      "previous_score": "float",
      "new_score": "float", // Score after applying delta (clamped 0-1)
      "delta": "float"
    }
    ```
*   **Response (Error):**
    ```json
    {
      "status": "error",
      "message": "Description of failure (e.g., memory not found, update failed)"
    }
    ```

---

## 2. Common Error Handling

API endpoints generally return errors using the standard FastAPI `HTTPException` mechanism, resulting in JSON responses like:

```json
{
  "detail": "Description of the error"
}

Specific endpoints might return structured error responses with a "success": false field and an "error" field:

json
CopyInsert
{
  "success": false,
  "error": "Detailed error message",
  "status_code": 400 // Optional HTTP status code
}
Common HTTP Status Codes:

200 OK: Request successful.
400 Bad Request: Invalid input parameters or payload format (e.g., malformed JSON, missing required fields).
404 Not Found: Requested resource (e.g., memory_id, assembly_id) not found.
500 Internal Server Error: An unexpected error occurred during processing on the server (e.g., embedding generation failure, persistence error).
503 Service Unavailable: A required internal component (e.g., vector index, emotion model) failed to initialize or is unavailable.