Okay, here is the `API_REFERENCE.md` document, reflecting the endpoints and models defined in the provided `api/server.py`, `synthians_trainer_server/http_server.py`, and `orchestrator/server.py` files.

```markdown
# Synthians Cognitive Architecture: API Reference (Current State)

This document provides a reference for the APIs exposed by the three core services: Synthians Memory Core, Neural Memory Server, and Context Cascade Engine.

**Date:** March 29, 2025

## Table of Contents

1.  [Synthians Memory Core API](#synthians-memory-core-api)
2.  [Neural Memory Server API](#neural-memory-server-api)
3.  [Context Cascade Engine API](#context-cascade-engine-api)
4.  [Common Error Handling](#common-error-handling)

---

## 1. Synthians Memory Core API

**Base URL:** `http://localhost:5010` (Default)

This service manages persistent memory storage, retrieval, scoring, and related functionalities.

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
*   **Description:** Checks the health status of the Memory Core service.
*   **Response (Success):**
    ```json
    {
      "status": "healthy",
      "uptime_seconds": 1234.56,
      "memory_count": 500,
      "assembly_count": 50,
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
*   **Description:** Retrieves detailed statistics about the Memory Core system, including memory counts, vector index status, and configuration.
*   **Response (Success):**
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
*   **Description:** Processes and stores a new memory entry. Generates embedding if not provided, calculates QuickRecal score, performs emotion analysis (optional), synthesizes metadata, and saves the memory.
*   **Request Model:** (`ProcessMemoryRequest`)
    ```json
    {
      "content": "string",
      "embedding": "Optional[List[float]]",
      "metadata": "Optional[Dict[str, Any]]",
      "analyze_emotion": "Optional[bool]" // Default: true
    }
    ```
*   **Response Model:** (`ProcessMemoryResponse`)
    ```json
    {
      "success": true,
      "memory_id": "string",
      "quickrecal_score": "float",
      "embedding": "List[float]",
      "metadata": "Dict[str, Any]",
      "error": null // Or error string on failure
    }
    ```

---

### Retrieve Memories

*   **Method:** `POST`
*   **Path:** `/retrieve_memories`
*   **Description:** Retrieves relevant memories based on a query string or embedding. Applies emotional gating and adaptive thresholding.
*   **Request Model:** (`RetrieveMemoriesRequest`)
    ```json
    {
      "query": "string",
      "query_embedding": "Optional[List[float]]", // Currently handled internally if query string provided
      "top_k": "int" // Default: 5
      "user_emotion": "Optional[Union[Dict[str, Any], str]]", // e.g., {"dominant_emotion": "joy"} or "joy"
      "cognitive_load": "float" // Default: 0.5
      "threshold": "Optional[float]" // Explicit retrieval threshold
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
          "timestamp": "float",
          "quickrecal_score": "float",
          "metadata": "Dict[str, Any]",
          "similarity": "float", // Similarity to query
          "emotional_resonance": "Optional[float]", // If emotional gating applied
          "final_score": "Optional[float]" // If emotional gating applied
          // ... other MemoryEntry fields
        }
        // ... more memories
      ],
      "error": null // Or error string on failure
    }
    ```

---

### Generate Embedding

*   **Method:** `POST`
*   **Path:** `/generate_embedding`
*   **Description:** Generates an embedding vector for the given text using the configured model.
*   **Request Model:** (`GenerateEmbeddingRequest`)
    ```json
    {
      "text": "string"
    }
    ```
*   **Response Model:** (`GenerateEmbeddingResponse`)
    ```json
    {
      "success": true,
      "embedding": "List[float]",
      "dimension": "int",
      "error": null
    }
    ```

---

### Calculate QuickRecal Score

*   **Method:** `POST`
*   **Path:** `/calculate_quickrecal`
*   **Description:** Calculates the QuickRecal score for a given text or embedding, considering context.
*   **Request Model:** (`QuickRecalRequest`)
    ```json
    {
      "embedding": "Optional[List[float]]",
      "text": "Optional[string]",
      "context": "Optional[Dict[str, Any]]" // Includes timestamp, relevance, etc.
    }
    ```
*   **Response Model:** (`QuickRecalResponse`)
    ```json
    {
      "success": true,
      "quickrecal_score": "float",
      "factors": "Optional[Dict[str, float]]", // Contributing factor scores
      "error": null
    }
    ```

---

### Analyze Emotion

*   **Method:** `POST`
*   **Path:** `/analyze_emotion`
*   **Description:** Analyzes the emotional content of the given text.
*   **Request Model:** (`EmotionRequest`)
    ```json
    {
      "text": "string"
    }
    ```
*   **Response Model:** (`EmotionResponse`)
    ```json
    {
      "success": true,
      "emotions": "Dict[str, float]", // Scores for different emotions
      "dominant_emotion": "string",
      "error": null
    }
    ```

---

### Provide Feedback

*   **Method:** `POST`
*   **Path:** `/provide_feedback`
*   **Description:** Provides feedback on the relevance of a retrieved memory, used for adaptive threshold calibration.
*   **Request Model:** (`FeedbackRequest`)
    ```json
    {
      "memory_id": "string",
      "similarity_score": "float",
      "was_relevant": "bool"
    }
    ```
*   **Response Model:** (`FeedbackResponse`)
    ```json
    {
      "success": true,
      "new_threshold": "Optional[float]", // Current threshold after adjustment
      "error": null
    }
    ```

---

### Detect Contradictions

*   **Method:** `POST`
*   **Path:** `/detect_contradictions`
*   **Description:** Attempts to detect potential contradictions among stored memories based on semantic similarity and content analysis.
*   **Query Parameter:** `threshold` (float, default: 0.75)
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
           "overlap_ratio": "float"
        }
        // ... more contradictions
      ],
      "count": "int"
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
*   **Description:** Processes transcribed text, enriches it with features extracted from audio metadata (pauses, rhythm, interruptions), and stores it as a memory.
*   **Request Model:** (`TranscriptionRequest`)
    ```json
    {
      "text": "string",
      "audio_metadata": "Optional[Dict[str, Any]]", // e.g., duration, pauses, interruption info
      "embedding": "Optional[List[float]]",
      "memory_id": "Optional[string]", // For updating existing
      "importance": "Optional[float]",
      "force_update": "bool" // Default: false
    }
    ```
*   **Response Model:** (`TranscriptionResponse`)
    ```json
    {
      "success": true,
      "memory_id": "string",
      "metadata": "Dict[str, Any]", // Extracted + synthesized metadata
      "embedding": "List[float]",
      "error": null
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
          "memory_count": "int",
          "last_activation": "float" // Timestamp
        }
        // ... more assemblies
      ],
      "count": "int"
    }
    ```

---

### Get Assembly Details

*   **Method:** `GET`
*   **Path:** `/assemblies/{assembly_id}`
*   **Description:** Retrieves detailed information about a specific memory assembly.
*   **Path Parameter:** `assembly_id` (string)
*   **Response (Success):**
    ```json
    {
      "success": true,
      "assembly_id": "string",
      "name": "string",
      "memory_count": "int",
      "last_activation": "float",
      "sample_memories": [ // Limited sample for brevity
        {
          "id": "string",
          "content": "string",
          "quickrecal_score": "float"
        }
      ],
      "total_memories": "int"
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
*   **Description:** Retrieves a sequence of memory embeddings, ordered and filtered, suitable for feeding into the sequence trainer (Neural Memory).
*   **Query Parameters:** `topic`, `user`, `emotion`, `min_importance`, `limit`, `min_quickrecal_score`, `start_timestamp`, `end_timestamp`, `sort_by`
*   **Response Model:** (`SequenceEmbeddingsResponse`)
    ```json
    {
      "embeddings": [
        {
          "id": "string",
          "embedding": "List[float]",
          "timestamp": "string" // ISO format
          "quickrecal_score": "Optional[float]",
          "emotion": "Optional[Dict[str, float]]",
          "dominant_emotion": "Optional[string]",
          "importance": "Optional[float]",
          "topic": "Optional[string]",
          "user": "Optional[string]"
        }
        // ... more embeddings
      ]
    }
    ```

---

### Update QuickRecal Score (Trainer Integration)

*   **Method:** `POST`
*   **Path:** `/api/memories/update_quickrecal_score`
*   **Description:** Allows the Trainer (or Orchestrator) to update a memory's QuickRecal score based on surprise feedback. **(Requires `update_memory` to be implemented in Memory Core)**.
*   **Request Model:** (`UpdateQuickRecalScoreRequest`)
    ```json
    {
      "memory_id": "string",
      "delta": "float", // Amount to change score by
      "predicted_embedding": "Optional[List[float]]",
      "reason": "Optional[string]", // e.g., "NM Surprise Loss: 0.65"
      "embedding_delta": "Optional[List[float]]"
    }
    ```
*   **Response (Success):** (`UpdateQuickRecalScoreResponse` - Example)
    ```json
    {
      "status": "success",
      "memory_id": "string",
      "previous_score": "float",
      "new_score": "float",
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

### Get Memory by ID

*   **Method:** `GET`
*   **Path:** `/api/memories/{memory_id}`
*   **Description:** Retrieves a specific memory by its unique identifier. Returns the complete memory object with all metadata and embedding if found.
*   **Path Parameters:** `memory_id` (string) - Unique identifier of the memory to retrieve
*   **Response Model:** (`GetMemoryResponse`)
    ```json
    {
      "success": true,
      "memory": {
        "id": "string",
        "content": "string",
        "embedding": "List[float]",
        "timestamp": "string", // ISO format
        "quickrecal_score": "float",
        "quickrecal_updated": "string", // ISO format
        "metadata": "Dict[str, Any]"
      },
      "error": null
    }
    ```
*   **Response (Error):**
    ```json
    {
      "success": false,
      "memory": null,
      "error": "Description of error (e.g., 'Memory with ID {memory_id} not found')"
    }
    ```

---

## 2. Neural Memory Server API

**Base URL:** `http://localhost:8001` (Default)

This service implements the adaptive, associative Neural Memory module inspired by the Titans paper.

---

### Initialize Neural Memory

*   **Method:** `POST`
*   **Path:** `/init`
*   **Description:** Initializes or re-initializes the Neural Memory module with the specified configuration. Can also load state from a file. Auto-initialization occurs on startup, but this allows manual re-init.
*   **Request Model:** (`InitRequest`)
    ```json
    {
      "config": "Optional[dict]", // Overrides default NeuralMemoryConfig
      "memory_core_url": "Optional[string]", // URL for potential callbacks (not currently used)
      "load_path": "Optional[string]" // Path to load state from
    }
    ```
*   **Response Model:** (`InitResponse`)
    ```json
    {
      "message": "Neural Memory module initialized successfully.",
      "config": { // The effective configuration used
        "input_dim": 768,
        "key_dim": 128,
        "value_dim": 768,
        "query_dim": 128,
        // ... other config fields
      }
    }
    ```

---

### Retrieve Associative Memory

*   **Method:** `POST`
*   **Path:** `/retrieve`
*   **Description:** Retrieves an associated value embedding from the Neural Memory based on the input embedding (after query projection).
*   **Request Model:** (`RetrieveRequest`)
    ```json
    {
      "input_embedding": "List[float]" // The raw input embedding
    }
    ```
*   **Response Model:** (`RetrieveResponse`)
    ```json
    {
      "retrieved_embedding": "List[float]", // The associated embedding y_t = M(q_t)
      "query_projection": "List[float]" // The q_t used for retrieval
    }
    ```

---

### Update Neural Memory (Test-Time Learning)

*   **Method:** `POST`
*   **Path:** `/update_memory`
*   **Description:** Performs the core test-time learning step. Updates the internal memory weights (`M`) based on the input embedding and optionally modified projections/gates (for MAG/MAL variants).
*   **Request Model:** (`UpdateMemoryRequest`)
    ```json
    {
      "input_embedding": "List[float]", // x_t
      // --- Optional for MAL ---
      "external_key_projection": "Optional[List[float]]", // k_t
      "external_value_projection": "Optional[List[float]]", // v'_t (modified by MAL attention)
      // --- Optional for MAG ---
      "external_alpha_gate": "Optional[float]", // alpha_t (forget rate)
      "external_theta_gate": "Optional[float]", // theta_t (inner LR)
      "external_eta_gate": "Optional[float]"    // eta_t (momentum)
    }
    ```
*   **Response Model:** (`UpdateMemoryResponse`)
    ```json
    {
      "status": "success",
      "loss": "float", // The loss ||M(k_t) - v_t||^2 calculated during update
      "grad_norm": "float", // Norm of gradients w.r.t internal memory weights (surprise metric)
      "key_projection": "List[float]", // k_t calculated/used
      "value_projection": "List[float]", // v_t (or v'_t if MAL) calculated/used
      "applied_alpha": "Optional[float]", // Actual alpha_t used
      "applied_theta": "Optional[float]", // Actual theta_t used
      "applied_eta": "Optional[float]" // Actual eta_t used
    }
    ```

---

### Train Outer Loop

*   **Method:** `POST`
*   **Path:** `/train_outer`
*   **Description:** Performs one step of outer loop training (adjusting projection and gate parameters) based on a sequence. (Note: Effective outer loop training is complex).
*   **Request Model:** (`TrainOuterRequest`)
    ```json
    {
      "input_sequence": "List[List[float]]", // Shape: [seq_len, input_dim]
      "target_sequence": "List[List[float]]" // Shape: [seq_len, value_dim]
    }
    ```
*   **Response Model:** (`TrainOuterResponse`)
    ```json
    {
      "average_loss": "float" // Loss over the sequence
    }
    ```

---

### Save State

*   **Method:** `POST`
*   **Path:** `/save`
*   **Description:** Saves the current state of the Neural Memory module (weights, config, momentum) to a file.
*   **Request Model:** (`SaveLoadRequest`)
    ```json
    {
      "path": "string" // File path to save to
    }
    ```
*   **Response (Success):**
    ```json
    {
      "message": "Neural Memory state saved to /path/to/state.json"
    }
    ```

---

### Load State

*   **Method:** `POST`
*   **Path:** `/load`
*   **Description:** Loads the Neural Memory module state from a file. Reinitializes the module based on the saved configuration.
*   **Request Model:** (`SaveLoadRequest`)
    ```json
    {
      "path": "string" // File path to load from
    }
    ```
*   **Response (Success):**
    ```json
    {
      "message": "Neural Memory state loaded from /path/to/state.json"
    }
    ```

---

### Get Status

*   **Method:** `GET`
*   **Path:** `/status`
*   **Description:** Returns the initialization status and current configuration of the Neural Memory module.
*   **Response Model:** (`StatusResponse`)
    ```json
    {
     "status": "Initialized", // or "Neural Memory module not initialized."
     "config": { // The effective configuration used
        "input_dim": 768,
        // ... other config fields
      }
    }
    ```

---

### Analyze Surprise

*   **Method:** `POST`
*   **Path:** `/analyze_surprise`
*   **Description:** Calculates surprise metrics between a predicted and an actual embedding using the integrated `SurpriseDetector`.
*   **Request Model:** (`AnalyzeSurpriseRequest`)
    ```json
    {
      "predicted_embedding": "List[float]",
      "actual_embedding": "List[float]"
    }
    ```
*   **Response (Success):** (Dictionary based on `SurpriseDetector.calculate_surprise`)
    ```json
    {
      "surprise": "float",
      "cosine_surprise": "float",
      "context_surprise": "float",
      "delta_norm": "float",
      "is_surprising": "bool",
      "adaptive_threshold": "float",
      "volatility": "float",
      "delta": "List[float]",
      "quickrecal_boost": "float" // Calculated boost based on surprise
    }
    ```

---

### Health Check

*   **Method:** `GET`
*   **Path:** `/health`
*   **Description:** Basic health check, includes TensorFlow status.
*   **Response (Success):**
    ```json
    {
     "status": "ok",
     "tensorflow_version": "2.x.x",
     "neural_memory_initialized": true,
     "timestamp": "2025-03-29T10:00:00.123Z"
    }
    ```

---

### Get Projections

*   **Method:** `POST`
*   **Path:** `/get_projections`
*   **Description:** Calculates and returns the Key, Value, and Query projections for a given input embedding *without* updating the memory state.
*   **Request Model:** (`GetProjectionsRequest`)
    ```json
    {
      "input_embedding": "List[float]",
      "embedding_model": "Optional[string]",
      "projection_adapter": "Optional[string]"
    }
    ```
*   **Response Model:** (`GetProjectionsResponse`)
    ```json
    {
      "input_embedding_norm": "float",
      "projection_adapter_used": "string",
      "key_projection": "List[float]", // k_t
      "value_projection": "List[float]", // v_t
      "query_projection": "List[float]", // q_t
      "projection_metadata": {
        "dim_key": "int",
        "dim_value": "int",
        "dim_query": "int",
        "projection_matrix_hash": "string", // Placeholder
        "input_dim": "int",
        "timestamp": "string" // ISO format
      }
    }
    ```

---

### Calculate Gates (for MAG)

*   **Method:** `POST`
*   **Path:** `/calculate_gates`
*   **Description:** Calculates dynamic gate values (`alpha`, `theta`, `eta`) based on the provided attention output, typically used by the MAG variant in the CCE.
*   **Request Model:** (`CalculateGatesRequest`)
    ```json
    {
      "attention_output": "List[float]",
      "current_alpha": "Optional[float]",
      "current_theta": "Optional[float]",
      "current_eta": "Optional[float]"
    }
    ```
*   **Response Model:** (`CalculateGatesResponse`)
    ```json
    {
      "alpha": "float", // Calculated forget rate
      "theta": "float", // Calculated inner learning rate
      "eta": "float",   // Calculated momentum decay
      "metadata": {
          "timestamp": "string", // ISO format
          "attention_output_dim": "int",
          // ... other metadata
      }
    }
    ```

---

### Get/Set Configuration

*   **Method:** `GET`, `POST`
*   **Path:** `/config`
*   **Description:** Retrieves the current configuration (Neural Memory + Attention) and detected capabilities (external gates/projections support). Can potentially set configuration (e.g., active variant) via POST, though this depends on CCE re-reading it.
*   **Request Model (POST):** (`ConfigRequest`)
    ```json
    {
       "variant": "Optional[string]" // e.g., "MAC", "MAG", "MAL"
    }
    ```
*   **Response Model (GET/POST):** (`ConfigResponse`)
    ```json
    {
      "neural_memory_config": { /* ... NeuralMemoryConfig dict ... */ },
      "attention_config": { /* ... Attention config dict ... */ },
      "titans_variant": "string", // e.g., "MAC", "MAG", "MAL", "NONE"
      "supports_external_gates": "bool", // Can /update_memory accept MAG gates?
      "supports_external_projections": "bool" // Can /update_memory accept MAL projections?
    }
    ```

---

### Diagnose Emotional Loop

*   **Method:** `GET`
*   **Path:** `/diagnose_emoloop`
*   **Description:** Retrieves diagnostic metrics related to the surprise -> QuickRecal feedback loop, analyzed over a specified window.
*   **Query Parameters:** `window` (string, e.g., "last_100"), `emotion_filter` (string, optional), `format` (string, optional "table")
*   **Response Model:** (`DiagnoseEmoLoopResponse`)
    ```json
    {
        "diagnostic_window": "string",
        "avg_loss": "float",
        "avg_grad_norm": "float",
        "avg_quickrecal_boost": "float",
        "dominant_emotions_boosted": "List[string]",
        "emotional_entropy": "float",
        "emotion_bias_index": "float",
        "user_emotion_match_rate": "float",
        "cluster_update_hotspots": [ { "cluster_id": "string", "updates": "int" } ],
        "alerts": "List[string]",
        "recommendations": "List[string]"
    }
    ```

---

## 3. Context Cascade Engine API

**Base URL:** `http://localhost:8002` (Default)

This service orchestrates the flow between the Memory Core and Neural Memory Server.

---

### Root

*   **Method:** `GET`
*   **Path:** `/`
*   **Description:** Basic endpoint indicating the CCE service is running.
*   **Response (Success):**
    ```json
    {
      "service": "Context Cascade Orchestrator",
      "status": "running"
    }
    ```

---

### Process Memory (Orchestrated)

*   **Method:** `POST`
*   **Path:** `/process_memory`
*   **Description:** Triggers the full orchestrated cognitive cycle for a new memory input. This involves calls to both the Memory Core and Neural Memory Server according to the refactored flow and active Titans variant.
*   **Request Model:** (`ProcessMemoryRequest`)
    ```json
    {
      "content": "string",
      "embedding": "Optional[List[float]]",
      "metadata": "Optional[Dict[str, Any]]"
    }
    ```
*   **Response (Success):** (Example structure, combines results)
    ```json
    {
        "memory_id": "string",
        "intent_id": "string",
        "status": "completed", // or "error_partial", "error_total"
        "timestamp": "string", // ISO format
        "neural_memory_update": { /* ... Response from NMS /update_memory ... */ },
        "neural_memory_retrieval": {
             "success": true,
             "retrieved_embedding": "List[float]", // y_t_final (potentially MAC-modified)
             "query_projection": "List[float]", // q_t used for retrieval
             "error": null
        },
        "surprise_metrics": {
            "loss": "Optional[float]",
            "grad_norm": "Optional[float]",
            "boost_calculated": "Optional[float]"
        },
        "quickrecal_feedback": { /* ... Response from MC /update_quickrecal_score ... */ },
        "variant_output": {
            "variant_type": "string", // e.g., "MAC"
             // ... other variant-specific metrics ...
        },
        "error": null // Consolidated error message if applicable
    }
    ```

---

### Get Sequence Embeddings (Passthrough)

*   **Method:** `POST`
*   **Path:** `/get_sequence_embeddings`
*   **Description:** Passthrough endpoint to request sequence embeddings from the Memory Core, potentially for external training use.
*   **Request Model:** (`SequenceEmbeddingsRequest`)
    ```json
    {
      "topic": "Optional[string]",
      "limit": "int", // Default: 10
      "min_quickrecal_score": "Optional[float]"
      // ... other filters supported by MC endpoint ...
    }
    ```
*   **Response (Success):** (Same as MC `/api/memories/get_sequence_embeddings`)
    ```json
    {
      "embeddings": [ /* ... SequenceEmbedding objects ... */ ]
    }
    ```

---

### Analyze Surprise (Passthrough)

*   **Method:** `POST`
*   **Path:** `/analyze_surprise`
*   **Description:** Passthrough endpoint to analyze surprise between two embeddings using the Neural Memory Server's `SurpriseDetector`.
*   **Request Model:** (`AnalyzeSurpriseRequest`)
    ```json
    {
      "predicted_embedding": "List[float]",
      "actual_embedding": "List[float]"
    }
    ```
*   **Response (Success):** (Same as NMS `/analyze_surprise`)
    ```json
    {
      "surprise": "float",
      // ... other surprise metrics ...
      "quickrecal_boost": "float"
    }
    ```

---

## 4. Common Error Handling

API endpoints generally return errors using the standard FastAPI `HTTPException` mechanism, resulting in JSON responses like:

```json
{
  "detail": "Description of the error"
}
```

Specific endpoints might return structured error responses with a `"success": false` field and an `"error"` field:

```json
{
  "success": false,
  "error": "Detailed error message",
  "status_code": 400 // Optional HTTP status code
}
```

Common HTTP Status Codes:

*   `200 OK`: Request successful.
*   `400 Bad Request`: Invalid input parameters or payload format.
*   `404 Not Found`: Requested resource (e.g., memory_id, assembly_id) not found.
*   `408 Request Timeout`: A downstream service call timed out.
*   `500 Internal Server Error`: An unexpected error occurred during processing on the server.
*   `503 Service Unavailable`: A required downstream service (Memory Core, Neural Memory) is unavailable or the module is not initialized.

```