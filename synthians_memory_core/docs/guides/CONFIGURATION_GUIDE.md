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
| `enable_assembly_pruning`       | bool              | True                                  | Whether to automatically prune assemblies based on configured criteria.                                    | Core                    |
| `assembly_prune_min_memories`   | int               | 2                                     | Minimum number of memories an assembly must contain to avoid pruning.                                      | Core                    |
| `assembly_prune_max_idle_days`  | float             | 30.0                                  | Maximum days an assembly can remain without activation before pruning.                                     | Core                    |
| `assembly_prune_max_age_days`   | float             | 90.0                                  | Maximum age in days before an assembly is eligible for pruning.                                           | Core                    |
| `assembly_prune_min_activation_level` | float        | 5                                     | Minimum number of activations required for an assembly to avoid age-based pruning.                        | Core                    |
| `enable_assembly_merging`       | bool              | True                                  | Whether to automatically merge similar assemblies.                                                        | Core                    |
| `assembly_merge_threshold`      | float             | 0.85                                  | Similarity threshold for merging two assemblies (0.0-1.0).                                               | Core                    |
| `assembly_max_merges_per_run`   | int               | 10                                    | Maximum number of assembly merges to perform in a single maintenance cycle.                              | Core                    |
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

### 3.3. Memory Assembly Lifecycle Management

The Memory Assembly system introduced in Phase 5.8 includes automatic lifecycle management features that can be configured to maintain assembly quality and prevent unbounded growth. These features include pruning of inactive or low-quality assemblies and merging of highly similar assemblies.

#### 3.3.1 Assembly Pruning

Assembly pruning automatically removes assemblies that meet certain criteria during background maintenance cycles. Pruning can be configured with the following parameters:

```python
# Enable or disable automatic assembly pruning
enable_assembly_pruning = True

# Minimum number of memories required to keep an assembly
assembly_prune_min_memories = 2

# Maximum days an assembly can exist without being activated
assembly_prune_max_idle_days = 30.0

# Maximum age in days for an assembly to be automatically pruned
assembly_prune_max_age_days = 90.0

# Minimum activation count required to prevent age-based pruning
assembly_prune_min_activation_level = 5
```

Pruning operations target the following types of assemblies:

1. **Empty Assemblies**: Assemblies with no memory members
2. **Old Idle Assemblies**: Assemblies not activated for longer than `assembly_prune_max_idle_days`
3. **Low-Activity Old Assemblies**: Assemblies older than `assembly_prune_max_age_days` with fewer than `assembly_prune_min_activation_level` activations

#### 3.3.2 Assembly Merging

Assembly merging combines assemblies with highly similar composite embeddings, reducing redundancy and improving retrieval consistency. Merging can be configured with the following parameters:

```python
# Enable or disable automatic assembly merging
enable_assembly_merging = True

# Similarity threshold for merging two assemblies (0.0-1.0)
assembly_merge_threshold = 0.85

# Maximum number of merges to perform in a single maintenance cycle
assembly_max_merges_per_run = 10
```

When assemblies are merged:

1. A new assembly is created containing all memory members from both source assemblies
2. The composite embedding is recalculated based on the combined memory set
3. All memory-to-assembly references are updated atomically
4. Original assemblies are removed from all storage locations

#### 3.3.3 Lifecycle Management Integration

Lifecycle management runs automatically as part of the `_decay_and_pruning_loop` background task, following the same interval as the memory pruning process (`prune_check_interval`). This ensures regular maintenance of the assembly store without requiring additional background threads.

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
    'use_gpu_for_index': True,  # Use GPU acceleration if available
    
    # Assembly lifecycle management
    'enable_assembly_pruning': True,
    'assembly_prune_min_memories': 3,  # Higher minimum for production
    'assembly_prune_max_idle_days': 45.0,  # More generous idle window
    'assembly_prune_max_age_days': 120.0,  # Longer retention
    'assembly_prune_min_activation_level': 10,  # Higher activation threshold
    'enable_assembly_merging': True,
    'assembly_merge_threshold': 0.88,  # More conservative merging
    'assembly_max_merges_per_run': 5  # More conservative merging pace
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
