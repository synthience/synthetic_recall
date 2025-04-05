# Synthians Cognitive Dashboard - API Reference

This document provides details on the API endpoints used by the Synthians Cognitive Dashboard to interact with the underlying Cognitive Architecture services.

## Base URLs

The dashboard interacts with three primary services:

- **Memory Core**: `http://memory-core:8080` (configurable via `MEMORY_CORE_URL` env variable)
- **Neural Memory**: `http://neural-memory:8080` (configurable via `NEURAL_MEMORY_URL` env variable)
- **CCE (Controlled Context Exchange)**: `http://cce:8080` (configurable via `CCE_URL` env variable)

## Authentication

Currently, the API endpoints do not require authentication. This will be implemented in future versions.

## Response Format

All API responses follow a standard format:

```json
{
  "status": "success" | "error",
  "data": {
    // Response data specific to the endpoint
  },
  "message": "Optional message, typically for errors"
}
```

## Common Status Codes

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Resource Not Found
- `500` - Internal Server Error

## Memory Core Endpoints

### Health Check

```
GET /api/memory-core/health
```

Returns the health status of the Memory Core service.

**Response**:
```json
{
  "status": "success",
  "data": {
    "name": "Memory Core",
    "status": "Healthy" | "Unhealthy" | "Checking..." | "Error",
    "url": "http://memory-core:8080",
    "details": "Optional details about the service status",
    "uptime": "3d 4h 12m",
    "version": "1.2.3"
  }
}
```

### Memory Stats

```
GET /api/memory-core/stats
```

Returns statistics about the memory storage.

**Response**:
```json
{
  "status": "success",
  "data": {
    "total_memories": 12500,
    "total_assemblies": 450,
    "dirty_items": 12,
    "pending_vector_updates": 3,
    "vector_index": {
      "count": 12500,
      "mapping_count": 12500,
      "drift_count": 2,
      "index_type": "HNSW",
      "gpu_enabled": true
    },
    "assembly_stats": {
      "total_count": 450,
      "indexed_count": 450,
      "vector_indexed_count": 448,
      "average_size": 27.8,
      "pruning_enabled": true,
      "merging_enabled": true
    },
    "persistence": {
      "last_update": "2025-03-15T14:32:11Z",
      "last_backup": "2025-03-15T12:00:00Z"
    },
    "performance": {
      "quick_recall_rate": 0.954,
      "threshold_recall_rate": 0.892
    }
  }
}
```

### List Assemblies

```
GET /api/memory-core/assemblies
```

Returns a list of all memory assemblies.

**Response**:
```json
{
  "status": "success",
  "data": [
    {
      "id": "assembly-123",
      "name": "Core Concepts",
      "description": "Fundamental AI concepts",
      "member_count": 145,
      "keywords": ["AI", "concepts", "foundation"],
      "tags": ["important", "core"],
      "topics": ["learning", "reasoning"],
      "created_at": "2025-01-15T08:12:34Z",
      "updated_at": "2025-03-14T16:45:22Z",
      "vector_index_updated_at": "2025-03-14T16:46:01Z",
      "memory_ids": ["mem-123", "mem-124", "mem-125"]
    },
    // More assemblies...
  ]
}
```

### Get Assembly Details

```
GET /api/memory-core/assemblies/:id
```

Returns details about a specific assembly.

**Parameters**:
- `id` (path parameter): The ID of the assembly

**Response**:
```json
{
  "status": "success",
  "data": {
    "id": "assembly-123",
    "name": "Core Concepts",
    "description": "Fundamental AI concepts",
    "member_count": 145,
    "keywords": ["AI", "concepts", "foundation"],
    "tags": ["important", "core"],
    "topics": ["learning", "reasoning"],
    "created_at": "2025-01-15T08:12:34Z",
    "updated_at": "2025-03-14T16:45:22Z",
    "vector_index_updated_at": "2025-03-14T16:46:01Z",
    "memory_ids": ["mem-123", "mem-124", "mem-125"],
    "memories": [
      {
        "id": "mem-123",
        "content": "Understanding of basic neural networks",
        "created_at": "2025-01-15T08:12:34Z",
        "type": "concept"
      },
      // More memories...
    ]
  }
}
```

### Verify Vector Index

```
POST /api/memory-core/verify-index
```

Triggers a verification of the vector index.

**Response**:
```json
{
  "status": "success",
  "message": "Vector index verification initiated",
  "data": {
    "job_id": "verify-job-456",
    "estimated_completion_time": "2025-03-16T15:30:00Z"
  }
}
```

### Trigger Retry Loop

```
POST /api/memory-core/retry-loop
```

Triggers the retry loop for failed operations.

**Response**:
```json
{
  "status": "success",
  "message": "Retry loop triggered",
  "data": {
    "pending_operations": 3
  }
}
```

## Neural Memory Endpoints

### Health Check

```
GET /api/neural-memory/health
```

Returns the health status of the Neural Memory service.

**Response**:
```json
{
  "status": "success",
  "data": {
    "name": "Neural Memory",
    "status": "Healthy" | "Unhealthy" | "Checking..." | "Error",
    "url": "http://neural-memory:8080",
    "details": "Optional details about the service status",
    "uptime": "3d 4h 12m",
    "version": "1.2.3"
  }
}
```

### Neural Memory Status

```
GET /api/neural-memory/status
```

Returns the status of the Neural Memory system.

**Response**:
```json
{
  "status": "success",
  "data": {
    "initialized": true,
    "config": {
      "dimensions": 1536,
      "hidden_size": 768,
      "layers": 12
    }
  }
}
```

### Emotional Loop Diagnostics

```
GET /api/neural-memory/diagnose_emoloop
```

Returns diagnostic information about the emotional loop.

**Parameters**:
- `window` (query parameter): Time window for diagnostics (default: "24h")

**Response**:
```json
{
  "status": "success",
  "data": {
    "avg_loss": 0.0324,
    "avg_grad_norm": 0.0512,
    "avg_qr_boost": 0.1786,
    "emotional_loop": {
      "dominant_emotions": ["curiosity", "confidence"],
      "entropy": 0.7821,
      "bias_index": 0.1232,
      "match_rate": 0.8934
    },
    "alerts": [
      "Gradient instability detected at 14:23:11"
    ],
    "recommendations": [
      "Consider reducing learning rate to stabilize training"
    ]
  }
}
```

### Initialize Neural Memory

```
POST /api/neural-memory/initialize
```

Initializes or resets the Neural Memory system.

**Response**:
```json
{
  "status": "success",
  "message": "Neural Memory initialized successfully",
  "data": {
    "initialization_time": "2025-03-16T14:23:11Z",
    "config": {
      "dimensions": 1536,
      "hidden_size": 768,
      "layers": 12
    }
  }
}
```

## CCE (Controlled Context Exchange) Endpoints

### Health Check

```
GET /api/cce/health
```

Returns the health status of the CCE service.

**Response**:
```json
{
  "status": "success",
  "data": {
    "name": "Context Cascade Engine",
    "status": "Healthy" | "Unhealthy" | "Checking..." | "Error",
    "url": "http://cce:8080",
    "details": "Optional details about the service status",
    "uptime": "3d 4h 12m",
    "version": "1.2.3"
  }
}
```

### CCE Status

```
GET /api/cce/status
```

Returns the status of the CCE system.

**Response**:
```json
{
  "status": "success",
  "data": {
    "active_variant": "MAC-13b",
    "llm_guidance_enabled": true,
    "recent_success_rate": 0.978,
    "average_latency": 234.5
  }
}
```

### Recent CCE Responses

```
GET /api/cce/metrics/recent_cce_responses
```

Returns recent CCE responses with metrics.

**Response**:
```json
{
  "status": "success",
  "data": {
    "recent_responses": [
      {
        "timestamp": "2025-03-16T14:23:11Z",
        "status": "success",
        "variant_output": {
          "variant_type": "MAC-13b"
        },
        "variant_selection": {
          "selected_variant": "MAC-13b",
          "reason": "High precision required for technical context",
          "performance_used": true
        },
        "llm_advice_used": {
          "raw_advice": "Consider using MAC-13b for this technical query",
          "adjusted_advice": "Using MAC-13b for optimal technical reasoning",
          "confidence_level": 0.89,
          "adjustment_reason": "Enhanced with system parameters"
        }
      },
      // More responses...
    ]
  }
}
```

### Set CCE Variant

```
POST /api/cce/set-variant
```

Sets the active variant for the CCE.

**Request Body**:
```json
{
  "variant": "MAC-13b" | "MAC-7b" | "TITAN-7b"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Variant set successfully",
  "data": {
    "previous_variant": "MAC-7b",
    "new_variant": "MAC-13b",
    "change_timestamp": "2025-03-16T14:25:11Z"
  }
}
```

## System-wide Endpoints

### Alerts

```
GET /api/alerts
```

Returns system-wide alerts from all services.

**Response**:
```json
{
  "status": "success",
  "data": [
    {
      "id": "alert-1",
      "type": "error" | "warning" | "info",
      "title": "High gradient detected",
      "description": "Neural Memory training shows unusually high gradients",
      "timestamp": "2025-03-16T13:24:56Z",
      "source": "NeuralMemory",
      "action": "Consider pausing training"
    },
    // More alerts...
  ]
}
```

## Error Responses

When an error occurs, the API will return a response with an error status:

```json
{
  "status": "error",
  "message": "Detailed error message",
  "code": "ERROR_CODE"
}
```

Common error codes include:

- `SERVICE_UNAVAILABLE` - The service is not accessible
- `INVALID_PARAMETERS` - The request contains invalid parameters
- `RESOURCE_NOT_FOUND` - The requested resource does not exist
- `INTERNAL_ERROR` - An unexpected error occurred in the service

## Rate Limiting

Currently, there are no rate limits on the API endpoints. This may change in future versions.

## Versioning

The current API version is v1. The version is not included in the URL path as there is only one version currently.

## Future Endpoints

The following endpoints are planned for future releases:

- Streaming log endpoints via WebSockets
- Authentication endpoints
- User management endpoints
- Detailed memory search endpoints
- Batch operations for assemblies and memories