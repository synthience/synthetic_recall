# Synthians Cognitive Architecture: Configuration Guide

**Version:** 1.2
**Date:** March 30, 2025

## 1. Overview

This guide details the configuration parameters for the Synthians Cognitive Architecture, focusing primarily on the Memory Core service which is the central component of the system.

**Core Services:**

1.  **Synthians Memory Core:** Manages persistent memory, retrieval, and scoring.
2.  **Neural Memory Server:** Handles adaptive associative memory and test-time learning. *(Documentation for this service is provided separately)*
3.  **Context Cascade Engine (CCE):** Orchestrates the flow between the Memory Core and Neural Memory. *(Documentation for this service is provided separately)*

## 3. Synthians Memory Core Configuration (`synthians_memory_core`)

These parameters configure the main memory storage and retrieval service, typically controlled via the `config` dictionary passed to the `SynthiansMemoryCore` class constructor and environment variables for the API server.

### 3.1. Core Parameters (`SynthiansMemoryCore` config dict)

| Parameter                       | Type              | Default                               | Description                                                                                                  | Passed To               |
| :------------------------------ | :---------------- | :------------------------------------ | :----------------------------------------------------------------------------------------------------------- | :---------------------- |
| `embedding_dim`                 | int               | 768                                   | **CRITICAL:** Dimension of embeddings used throughout the system. Must match embedding model output.         | All Components          |
| `geometry`                      | str               | "hyperbolic"                          | Geometric space for embedding operations: "euclidean", "hyperbolic", "spherical", or "mixed"               | `GeometryManager`       |
| `hyperbolic_curvature`          | float             | -1.0                                  | Curvature parameter for hyperbolic geometry (`< 0`). Lower magnitude = more curved.                         | `GeometryManager`       |
| `storage_path`                  | str               | "/app/memory/stored/synthians"        | Base path for persistent storage of memories, indices, and assemblies.                                     | `MemoryPersistence`     |
| `vector_index_type`             | str               | "Cosine"                              | Vector similarity metric: "L2" (Euclidean), "IP" (Inner Product), or "Cosine" (normalized inner product).   | `MemoryVectorIndex`     |
| `max_memory_entries`            | int               | 50000                                 | Maximum allowed memory entries before pruning is triggered.                                                | Core                    |
| `prune_threshold_percent`       | float             | 0.9                                   | Percentage of `max_memory_entries` at which pruning is triggered (0.0-1.0).                               | Core                    |
| `min_quickrecal_for_ltm`        | float             | 0.2                                   | Minimum QuickRecal score required to retain a memory after decay (0.0-1.0).                                | Core                    |
| `assembly_threshold`            | float             | 0.75                                  | Minimum similarity threshold for memories to form an assembly (0.0-1.0).                                   | Core                    |
| `max_assemblies_per_memory`     | int               | 3                                     | Maximum number of assemblies a single memory can belong to.                                                | Core                    |
| `adaptive_threshold_enabled`    | bool              | True                                  | Enable adaptive similarity threshold for retrieval based on feedback.                                       | `ThresholdCalibrator`   |
| `initial_retrieval_threshold`   | float             | 0.75                                  | Initial similarity threshold for memory retrieval (0.0-1.0).                                               | `ThresholdCalibrator`   |
| `persistence_interval`          | float             | 60.0                                  | Seconds between automated persistence operations.                                                          | Core                    |
| `decay_interval`                | float             | 3600.0                                | Seconds between automated QuickRecal decay checks.                                                         | Core                    |
| `prune_check_interval`          | float             | 600.0                                 | Seconds between automated memory pruning checks.                                                           | Core                    |
| `persistence_batch_size`        | int               | 100                                   | Number of memories to persist in a single batch.                                                           | Core                    |
| `check_index_on_retrieval`      | bool              | False                                 | Whether to check vector index integrity during retrieval operations.                                        | Core                    |
| `index_check_interval`          | float             | 3600                                  | Seconds between automated vector index verification checks.                                                | Core                    |
| `migrate_to_idmap`              | bool              | True                                  | Whether to migrate older FAISS indices to IndexIDMap format.                                                | `MemoryVectorIndex`     |

### 3.2. Component-Specific Parameters (Passed to Subcomponents)

#### 3.2.1 GeometryManager Parameters

The following parameters are extracted from the main config and passed to the `GeometryManager` constructor:

```python
self.geometry_manager = GeometryManager({
    'embedding_dim': self.config['embedding_dim'],
    'geometry_type': self.config['geometry'],
    'curvature': self.config['hyperbolic_curvature']
})
```

Additional `GeometryManager` parameters (with their own defaults if not specified):

| Parameter               | Type   | Default      | Description                                                               |
| :---------------------- | :----- | :----------- | :------------------------------------------------------------------------ |
| `alignment_strategy`    | str    | "truncate"   | Strategy for aligning embedding dimensions: "truncate", "pad", or "project" |
| `normalization_enabled` | bool   | True         | Whether to normalize vectors during operations                            |

#### 3.2.2 UnifiedQuickRecallCalculator Parameters

The following parameters are extracted from the main config and passed to the `UnifiedQuickRecallCalculator` constructor:

```python
self.quick_recal = UnifiedQuickRecallCalculator({
    'embedding_dim': self.config['embedding_dim'],
    'mode': QuickRecallMode.HPC_QR,  # Default to HPC-QR mode
    'geometry_type': self.config['geometry'],
    'curvature': self.config['hyperbolic_curvature']
}, geometry_manager=self.geometry_manager)
```

#### 3.2.3 MemoryVectorIndex Parameters

The following parameters are extracted from the main config and passed to the `MemoryVectorIndex` constructor:

```python
self.vector_index = MemoryVectorIndex({
    'embedding_dim': self.config['embedding_dim'],
    'storage_path': os.path.join(self.config['storage_path'], 'index'),
    'index_type': self.config['vector_index_type'],
    'use_gpu': self.config.get('use_gpu_for_index', False)
})
```

## 5. Recommended Configurations

### 5.1. Memory Core Production Configuration

```python
memory_core_config = {
    'embedding_dim': 768,
    'geometry': 'hyperbolic',  # Using hyperbolic geometry for better representation
    'storage_path': '/persistent/data/memory_storage',
    'vector_index_type': 'Cosine',  # Cosine similarity (normalized inner product)
    'max_memory_entries': 100000,  # Larger memory capacity
    'prune_threshold_percent': 0.95,  # Trigger pruning at 95% capacity
    'min_quickrecal_for_ltm': 0.25,  # Higher bar for long-term retention
    'persistence_interval': 30.0,  # More frequent saves
    'adaptive_threshold_enabled': True,
    'initial_retrieval_threshold': 0.72,  # Slightly more permissive initial threshold
    'use_gpu_for_index': True  # Use GPU acceleration if available
}
```

## Important Notes on Parameter Inheritance

1. **Embedding Dimension:** The `embedding_dim` parameter is particularly critical as it's passed to multiple components and must match the output dimension of your embedding model. If you're using a pre-trained model like `all-MiniLM-L6-v2` (384D) or `all-mpnet-base-v2` (768D), ensure this parameter matches exactly.

2. **Geometry Settings:** The `geometry` and `hyperbolic_curvature` parameters are passed to both the `GeometryManager` and `UnifiedQuickRecallCalculator`. If you want to override geometry settings for only one component, you'll need to initialize that component directly rather than relying on the SynthiansMemoryCore to do it for you.

3. **Storage Paths:** The base `storage_path` is used to derive component-specific paths:
   - Vector index: `{storage_path}/index/`
   - Memory files: `{storage_path}/memories/`
   - Memory index: `{storage_path}/memory_index.json`
   - Assemblies: `{storage_path}/assemblies/`
