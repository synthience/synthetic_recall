# Synthians Cognitive Architecture Configuration Guide

This document provides a comprehensive guide to configuring the Synthians Cognitive Architecture, including the new Phase 5.9 configuration options.

## Overview

The Synthians Cognitive Architecture uses a combination of environment variables and configuration files to control its behavior. These settings are managed by each component's configuration manager and can be modified to tune the system's performance.

## Memory Core Configuration

### Core Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `EMBEDDING_DIM` | int | 768 | Dimension of embeddings |
| `GEOMETRY` | str | "hyperbolic" | Geometry for similarity calculation ("euclidean", "hyperbolic") |
| `ENABLE_EMBEDDING_VALIDATION` | bool | true | Whether to validate embedding dimensions |
| `LOG_LEVEL` | str | "INFO" | Logging level ("DEBUG", "INFO", "WARNING", "ERROR") |
| `STORAGE_PATH` | str | "memory_storage" | Path for memory storage |
| `ASSEMBLY_STORAGE_PATH` | str | "assembly_storage" | Path for assembly storage |
| `INDEX_PATH` | str | "memory_index.json" | Path for memory index |
| `VECTOR_INDEX_PATH` | str | "vector_indices" | Path for vector indices |

### Assembly Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ASSEMBLY_ACTIVATION_THRESHOLD` | float | 0.75 | Threshold for assembly activation |
| `ASSEMBLY_BOOST_FACTOR` | float | 1.5 | Boost factor for activated assemblies |
| `ASSEMBLY_BOOST_MODE` | str | "linear" | Boost calculation mode ("linear", "sigmoid") |
| `ENABLE_ASSEMBLY_PRUNING` | bool | true | Whether to prune stale assemblies |
| `ENABLE_ASSEMBLY_MERGING` | bool | true | Whether to merge similar assemblies |
| `ASSEMBLY_MERGE_THRESHOLD` | float | 0.85 | Threshold for assembly merging |
| `MAX_ASSEMBLY_SIZE` | int | 100 | Maximum number of memories in an assembly |

### Vector Index Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `VECTOR_INDEX_TYPE` | str | "flat" | FAISS index type ("flat", "ivf", "hnsw") |
| `FAISS_NPROBE` | int | 5 | FAISS nprobe parameter for IVF indices |
| `MAX_ALLOWED_DRIFT_SECONDS` | int | 3600 | Maximum allowed time drift for assembly synchronization |
| `AUTO_REPAIR_ON_INIT` | bool | true | Whether to repair the index on initialization |
| `FAIL_ON_INIT_DRIFT` | bool | false | Whether to fail initialization on drift detection |

### Explainability & Diagnostics Settings (Phase 5.9)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ENABLE_EXPLAINABILITY` | bool | false | Master switch for explainability and diagnostics features |
| `MERGE_LOG_MAX_ENTRIES` | int | 1000 | Maximum number of entries to retain in the merge log |
| `ASSEMBLY_METRICS_PERSIST_INTERVAL` | float | 600.0 | Seconds between persisting assembly activation stats |
| `MAX_LINEAGE_DEPTH` | int | 10 | Maximum depth to trace when retrieving assembly lineage |

### QuickRecal Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `RECENCY_WEIGHT` | float | 0.3 | Weight of recency in QuickRecal calculation |
| `SURPRISE_WEIGHT` | float | 0.2 | Weight of surprise in QuickRecal calculation |
| `EMOTION_WEIGHT` | float | 0.3 | Weight of emotion in QuickRecal calculation |
| `CONTEXT_WEIGHT` | float | 0.2 | Weight of context in QuickRecal calculation |
| `QUICKRECAL_DECAY_RATE` | float | 0.01 | Rate of QuickRecal score decay |
| `DEFAULT_QUICKRECAL` | float | 0.5 | Default QuickRecal score for new memories |

### Emotional Intelligence Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `EMOTION_MODEL_PATH` | str | "models/emotion" | Path to emotion model |
| `EMOTION_THRESHOLD` | float | 0.5 | Threshold for emotion gating |
| `EMOTION_BOOST_FACTOR` | float | 1.2 | Boost factor for emotional memories |

### Phase 5.9 Explainability Settings (New)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ENABLE_EXPLAINABILITY` | bool | true | Whether to enable explainability features |
| `MERGE_LOG_PATH` | str | "data/merge_log.jsonl" | Path for merge event logs |
| `MAX_TRACKED_ACTIVATIONS` | int | 1000 | Maximum number of activation events to track |
| `MAX_LINEAGE_DEPTH` | int | 10 | Maximum depth for lineage tracing |
| `EXPLAINABILITY_LOG_LEVEL` | str | "INFO" | Logging level for explainability module |

## Neural Memory Configuration

### Core Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `EMBEDDING_DIM` | int | 768 | Dimension of embeddings |
| `SEQUENCE_LENGTH` | int | 5 | Length of embedding sequence |
| `MODEL_TYPE` | str | "transformer" | Type of neural model |
| `LEARNING_RATE` | float | 0.001 | Learning rate for test-time learning |
| `BATCH_SIZE` | int | 32 | Batch size for training |
| `HIDDEN_DIM` | int | 512 | Hidden dimension for neural model |
| `USE_GPU` | bool | false | Whether to use GPU acceleration |

### Surprise Detection Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `SURPRISE_THRESHOLD` | float | 0.5 | Threshold for surprise detection |
| `GRAD_NORM_FACTOR` | float | 0.5 | Weight of gradient norm in surprise calculation |
| `LOSS_FACTOR` | float | 0.5 | Weight of loss in surprise calculation |
| `SMOOTHING_FACTOR` | float | 0.1 | Smoothing factor for surprise metrics |

### Metrics Store Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `METRICS_WINDOW` | str | "24h" | Time window for metrics collection |
| `METRICS_CAPACITY` | int | 1000 | Maximum number of metrics to store |
| `METRICS_LOG_INTERVAL` | int | 100 | Interval for metrics logging |

## Context Cascade Engine Configuration

### Core Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `DEFAULT_VARIANT` | str | "auto" | Default Titans variant ("auto", "mac", "mag", "mal") |
| `VARIANT_SELECTION_MODE` | str | "performance" | Mode for variant selection ("performance", "fixed", "random") |
| `ENABLE_LLM_GUIDANCE` | bool | true | Whether to enable LLM guidance |
| `LLM_GUIDANCE_URL` | str | "http://localhost:8080" | URL for LLM guidance service |
| `LLM_CONFIDENCE_THRESHOLD` | float | 0.7 | Threshold for LLM confidence |

### Variant Selection Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `SURPRISE_THRESHOLD_MAG` | float | 0.6 | Surprise threshold for selecting MAG variant |
| `COMPLEXITY_THRESHOLD_MAL` | float | 0.8 | Complexity threshold for selecting MAL variant |
| `DEFAULT_ATTENTION_FOCUS` | str | "recency" | Default attention focus ("recency", "relevance", "emotion") |
| `CONTEXT_WINDOW` | int | 10 | Size of context window for history |

### Phase 5.9 Metrics Settings (New)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `METRICS_DETAIL_LEVEL` | str | "full" | Detail level for metrics ("basic", "full") |
| `METRICS_RESPONSE_LIMIT` | int | 100 | Maximum number of recent response metrics to store |
| `INCLUDE_TRACE_INFO` | bool | true | Whether to include trace info in metrics |
| `INCLUDE_LLM_ADVICE_RAW` | bool | false | Whether to include raw LLM advice in metrics |

## Configuring the System

### Environment Variables

Configuration values can be set using environment variables, which take precedence over default values. For example:

```bash
export EMBEDDING_DIM=512
export ASSEMBLY_ACTIVATION_THRESHOLD=0.8
export ENABLE_EXPLAINABILITY=true
```

### Docker Environment Variables

When running with Docker, environment variables can be set in the docker-compose.yml file:

```yaml
services:
  memory-core:
    environment:
      - EMBEDDING_DIM=512
      - ASSEMBLY_ACTIVATION_THRESHOLD=0.8
      - ENABLE_EXPLAINABILITY=true
```

### Configuration Files

Some components also support loading configuration from files. For example:

```json
{
  "embedding_dim": 512,
  "assembly_activation_threshold": 0.8,
  "enable_explainability": true
}
```

## Phase 5.9 Configuration Changes

Phase 5.9 introduces several new configuration options focused on explainability and diagnostics:

1. **Memory Core**:
   - `ENABLE_EXPLAINABILITY`: Controls whether explainability features are enabled
   - `MERGE_LOG_PATH`: Path for storing merge event logs
   - `MAX_TRACKED_ACTIVATIONS`: Maximum number of activation events to track
   - `MAX_LINEAGE_DEPTH`: Maximum depth for lineage tracing
   - `EXPLAINABILITY_LOG_LEVEL`: Logging level for explainability module

2. **Context Cascade Engine**:
   - `METRICS_DETAIL_LEVEL`: Detail level for metrics responses
   - `METRICS_RESPONSE_LIMIT`: Maximum number of recent response metrics to store
   - `INCLUDE_TRACE_INFO`: Whether to include trace information in metrics
   - `INCLUDE_LLM_ADVICE_RAW`: Whether to include raw LLM advice in metrics

## Runtime Configuration Access

Phase 5.9 introduces a way to access sanitized runtime configuration through the Memory Core API:

```
GET /config/runtime/{service_name}
```

Where `service_name` can be:
- `memory-core`
- `neural-memory`
- `cce`

This endpoint requires `ENABLE_EXPLAINABILITY=true` and only returns a curated list of safe configuration values.

## Best Practices

1. **Testing**: Test configuration changes in a non-production environment first
2. **Monitoring**: Monitor system behavior after configuration changes
3. **Documentation**: Document any non-default configuration values
4. **Consistency**: Keep configuration consistent across related settings
5. **Security**: Treat configuration with sensitive values (e.g., API keys) as sensitive data

## Troubleshooting

If the system behaves unexpectedly after configuration changes:

1. Verify that environment variables are set correctly
2. Check logs for configuration-related warnings or errors
3. Verify that configuration files are valid JSON
4. Try resetting to default values to see if the issue persists
5. Consult the API to examine runtime configuration values
