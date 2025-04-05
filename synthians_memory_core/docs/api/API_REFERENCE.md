# Synthians Cognitive Architecture: API Reference

**Version:** 1.2.0 (Implemented as of Phase 5.9)  
**Date:** April 2025

This reference documents all HTTP API endpoints exposed by the Synthians Cognitive Architecture services, including Memory Core, Neural Memory Server, and Context Cascade Engine.

## Table of Contents

1. [Synthians Memory Core API](#1-synthians-memory-core-api-httplocalhost5010)
   - [Core Endpoints](#core-endpoints)
   - [Explainability Endpoints](#explainability-endpoints)
   - [Diagnostics Endpoints](#diagnostics-endpoints)
2. [Neural Memory Server API](#2-neural-memory-server-api-httplocalhost8001)
3. [Context Cascade Engine API](#3-context-cascade-engine-api-httplocalhost8002)
4. [Common Error Responses](#4-common-error-responses)

---

## 1. Synthians Memory Core API (`http://localhost:5010`)

The Memory Core API provides endpoints for memory storage, retrieval, embedding generation, and assembly management. It also includes endpoints for explainability and diagnostics (implemented in Phase 5.9).

### Core Endpoints

#### Root (`/`)

*   **Method:** `GET`
*   **Description:** Returns a simple message confirming the API is running.
*   **Response (Success):**
    ```json
    {
      "message": "Synthians Memory Core API"
    }
    ```

#### Health Check (`/health`)

*   **Method:** `GET`
*   **Description:** Provides basic health information about the service.
*   **Response (Success):**
    ```json
    {
      "status": "healthy",
      "uptime_seconds": 3600.5,
      "memory_count": 1024,
      "assembly_count": 42,
      "version": "1.0.0"
    }
    ```
*   **Response (Error):**
    ```json
    {
      "status": "unhealthy",
      "error": "Error message here"
    }
    ```

#### Get Statistics (`/stats`)

*   **Method:** `GET`
*   **Description:** Retrieves detailed statistics about the Memory Core system, including memory/assembly counts and vector index status.
*   **Response (Success):**
    ```json
    {
      "success": true,
      "api_server": {
        "uptime_seconds": 1850.7,
        "memory_count": 512,
        "embedding_dim": 768,
        "geometry": "hyperbolic",
        "model": "sentence-transformers/all-mpnet-base-v2"
      },
      "memory": {
        "total_memories": 512,
        "total_assemblies": 48,
        "storage_path": "/app/memory/stored/synthians",
        "threshold": 0.75
      },
      "assemblies": {
        "total_count": 48,
        "indexed_count": 45,
        "average_size": 10.7,
        "max_size": 24,
        "min_size": 3,
        "activated_count": 12,
        "active_ratio": 0.25
      },
      "vector_index": {
        "count": 560,
        "id_mappings": 560,
        "index_type": "IndexIDMap"
      },
      "assembly_sync": {
        "pending_updates_count": 3,
        "retry_queue_size": 3
      }
    }
    ```
*   **Response (Error):**
    ```json
    {
      "success": false,
      "error": "Error retrieving stats"
    }
    ```

*(Remaining core endpoints like process_memory, retrieve_memories, etc. - descriptions remain unchanged as they're already implemented)*

### Explainability Endpoints

*These endpoints require setting the `ENABLE_EXPLAINABILITY` configuration flag to `true`.*

#### Explain Activation

*   **Method:** `GET`
*   **Path:** `/assemblies/{assembly_id}/explain_activation`
*   **Description:** Explains why a specific memory was or wasn't considered part of an assembly during activation.
*   **Path Parameter:** `assembly_id` (string).
*   **Query Parameter:** `memory_id` (string, *required*).
*   **Response Model:**
    ```json
    {
      "success": true,
      "explanation": {
        "assembly_id": "asm_abc123",
        "memory_id": "mem_xyz789",
        "check_timestamp": "2025-04-15T10:23:45.123Z",
        "trigger_context": "Activation check during retrieval for query 'example query'",
        "calculated_similarity": 0.875,
        "activation_threshold": 0.75,
        "passed_threshold": true,
        "assembly_state_before_check": {
          "memory_count": 5,
          "last_activation_time": "2025-04-15T10:22:30.000Z"
        }
      }, 
      "error": null
    }
    ```
*   **Error Responses:** 404 (Assembly or Memory not found), 400 (Bad request), 500 (Server error), 403 (Forbidden if flag disabled).

#### Explain Assembly Merge

*   **Method:** `GET`
*   **Path:** `/assemblies/{assembly_id}/explain_merge`
*   **Description:** Provides details about the merge event that resulted in this assembly.
*   **Path Parameter:** `assembly_id` (string).
*   **Response Model:**
    ```json
    {
      "success": true,
      "explanation": {
        "assembly_id": "asm_merged123",
        "is_merged": true,
        "source_assemblies": [
          {"id": "asm_source_A", "name": "Source Assembly A"},
          {"id": "asm_source_B", "name": "Source Assembly B"}
        ],
        "similarity_at_merge": 0.882,
        "merge_threshold": 0.85,
        "merge_timestamp": "2025-04-14T18:32:15.678Z",
        "cleanup_status": "completed",
        "cleanup_timestamp": "2025-04-14T18:32:16.789Z"
      }, 
      "error": null
    }
    // Or if not merged: 
    {
      "success": true,
      "explanation": {
        "assembly_id": "asm_original456",
        "is_merged": false
      },
      "error": null
    }
    ```
*   **Error Responses:** 404 (Assembly not found), 500 (Server error), 403 (Forbidden if flag disabled).

#### Get Assembly Lineage

*   **Method:** `GET`
*   **Path:** `/assemblies/{assembly_id}/lineage`
*   **Description:** Traces the merge history (ancestry) of an assembly through its parent assemblies.
*   **Path Parameter:** `assembly_id` (string).
*   **Query Parameter:** `max_depth` (integer, *optional*, default: 10) - Maximum depth to trace lineage.
*   **Response Model:**
    ```json
    {
      "success": true,
      "lineage": [
        {
          "assembly_id": "asm_merged123", 
          "name": "Merged Assembly 123", 
          "depth": 0,
          "status": "normal",
          "created_at": "2025-04-14T18:32:15.678Z",
          "memory_count": 15
        },
        {
          "assembly_id": "asm_source_A", 
          "name": "Source Assembly A", 
          "depth": 1,
          "status": "normal",
          "created_at": "2025-04-14T15:20:10.456Z",
          "memory_count": 8
        },
        {
          "assembly_id": "asm_source_B", 
          "name": "Source Assembly B", 
          "depth": 1,
          "status": "normal",
          "created_at": "2025-04-14T16:12:45.789Z",
          "memory_count": 7
        },
        {
          "assembly_id": "asm_grand_B1", 
          "name": "Grandparent Assembly B1", 
          "depth": 2,
          "status": "cycle_detected", // Special status showing cycle detection
          "created_at": "2025-04-13T11:05:22.345Z",
          "memory_count": 5
        }
      ],
      "error": null
    }
    ```
*   **Status Values:** `normal` (standard entry), `cycle_detected` (lineage forms a cycle), `depth_limit_reached` (max depth reached).
*   **Error Responses:** 404 (Assembly not found), 500 (Server error), 403 (Forbidden if flag disabled).

### Diagnostics Endpoints

*These endpoints require setting the `ENABLE_EXPLAINABILITY` configuration flag to `true`.*

#### Get Merge Log

*   **Method:** `GET`
*   **Path:** `/diagnostics/merge_log`
*   **Description:** Returns a reconciled view of recent merge operations and their cleanup status.
*   **Query Parameter:** `limit` (integer, *optional*, default: 50) - Maximum number of entries to return.
*   **Response Model:**
    ```json
    {
      "success": true,
      "entries": [
        {
          "merge_event_id": "merge_uuid_123",
          "timestamp": "2025-04-15T09:45:12.345Z",
          "source_assembly_ids": ["asm_abc", "asm_def"],
          "target_assembly_id": "asm_merged_123",
          "similarity_at_merge": 0.92,
          "merge_threshold": 0.85,
          "cleanup_status": "completed",
          "cleanup_timestamp": "2025-04-15T09:45:13.456Z"
        },
        {
          "merge_event_id": "merge_uuid_124",
          "timestamp": "2025-04-15T09:50:22.678Z",
          "source_assembly_ids": ["asm_ghi", "asm_jkl"],
          "target_assembly_id": "asm_merged_124",
          "similarity_at_merge": 0.88,
          "merge_threshold": 0.85,
          "cleanup_status": "failed",
          "cleanup_timestamp": "2025-04-15T09:50:24.789Z",
          "error": "Failed to update vector index: dimension mismatch"
        }
      ],
      "error": null
    }
    ```
*   **Error Responses:** 500 (Server error), 403 (Forbidden if flag disabled).

#### Get Runtime Configuration

*   **Method:** `GET`
*   **Path:** `/config/runtime/{service_name}`
*   **Description:** Returns a sanitized view of the current runtime configuration for the specified service.
*   **Path Parameter:** `service_name` (string) - Name of the service to get configuration for (e.g., "memory_core", "geometry", "api").
*   **Response Model:**
    ```json
    {
      "success": true,
      "config": {
        "assembly_activation_threshold": 0.82,
        "default_assembly_size": 10,
        "merge_log_max_entries": 1000,
        "assembly_metrics_persist_interval": 600.0,
        "enable_explainability": true
        // Only non-sensitive configuration values are returned
      },
      "error": null
    }
    ```
*   **Error Responses:** 404 (Service not found), 500 (Server error), 403 (Forbidden if flag disabled).

#### Get Statistics

*   **Method:** `GET`
*   **Path:** `/stats`
*   **Description:** Returns enhanced system statistics including assembly activation counts.
*   **Response Model:**
    ```json
    {
      "success": true,
      "stats": {
        "memory_stats": {
          "total_count": 1245,
          "indexed_count": 1245,
          "by_corpus": {
            "corpus_A": 780,
            "corpus_B": 465
          }
        },
        "assembly_stats": {
          "count": 42,
          "activation_counts": {
            "assembly_123": 156,
            "assembly_456": 89,
            // Additional assembly IDs and their activation counts
          },
          "top_activated": [
            {"id": "assembly_123", "count": 156},
            {"id": "assembly_456", "count": 89},
            {"id": "assembly_789", "count": 67}
          ]
        },
        "system_stats": {
          "uptime_seconds": 86400.5,
          "version": "1.2.0"
        }
      },
      "error": null
    }
    ```
*   **Error Responses:** 500 (Server error).

{{ ... }}

---

## 2. Neural Memory Server API (`http://localhost:8001`)
*(Existing Endpoints - Descriptions generally unchanged. Add `/config/runtime/neural-memory` as a planned feature if implemented)*

---

## 3. Context Cascade Engine API (`http://localhost:8002`)
*(Existing Endpoints - Descriptions generally unchanged. Update `/metrics/recent_cce_responses` example as a planned enhancement.)*
*(Add `/config/runtime/cce` as a planned feature if implemented)*

### Get Recent CCE Metrics (`/metrics/recent_cce_responses`) - Planned Enhancement

*   **Method:** `GET`
*   **Description:** Retrieves recent CCE processing response objects. Planned enhancement will include detailed variant selection and LLM guidance info.
*   **Query Parameter:** `limit` (int, optional, default: 10).
*   **Response Model (Example Entry - Planned for Phase 5.9):**
    ```json
     {
        "timestamp": "...",
        "status": "completed",
        "memory_id": "mem_abc",
        "variant_output": { /* ... variant specific metrics ... */ },
        "variant_selection": { // Detailed selection info
            "selected": "MAG",
            "reason": "Performance (High Surprise 0.65 -> MAG)",
            "trace": ["Input metrics: ...", ...],
            "perf_metrics_used": {"avg_loss": 0.65, ...}
        },
        "llm_advice_used": { // Detailed LLM usage info
            "raw_advice": { /* Optional raw */ },
            "adjusted_advice": { /* Advice after confidence adjustment */ },
            "confidence_level": 0.95,
            "adjustment_reason": "High confidence...",
            "boost_modifier_applied": 0.1,
            "tags_added": ["quantum"],
            "variant_hint_followed": true,
            "attention_focus_used": "relevance"
        },
        "neural_memory_update": { /* ... loss, grad_norm ... */ },
        "quickrecal_feedback": { /* ... boost applied ... */ }
        // ... other fields ...
     }
    ```

## 4. Common Error Responses

All API endpoints follow a consistent error response format:

*   **400 Bad Request:** Invalid input parameters.
*   **404 Not Found:** Requested resource not found.
*   **500 Internal Server Error:** Server-side error.
*   **403 Forbidden:** Access denied (e.g., when trying to access planned explainability endpoints once implemented but with `ENABLE_EXPLAINABILITY=false`).

Error responses include:
```json
{
  "success": false,
  "error": "Detailed error message"
}