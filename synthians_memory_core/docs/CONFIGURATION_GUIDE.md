Okay, here is the comprehensive configuration guide for the Synthians Cognitive Architecture, covering all three services and providing example scenarios.

# Synthians Cognitive Architecture: Configuration Guide

**Version:** 1.1
**Date:** March 29, 2025

## 1. Overview

This guide details the configuration parameters for the three core services of the Synthians Cognitive Architecture:

1.  **Synthians Memory Core:** Manages persistent memory, retrieval, and scoring.
2.  **Neural Memory Server:** Handles adaptive associative memory and test-time learning.
3.  **Context Cascade Engine (CCE):** Orchestrates the flow between the Memory Core and Neural Memory.

Configuration is primarily managed through **Environment Variables** and **Configuration Dictionaries** passed during component initialization. Environment variables are particularly useful for Docker deployments.

## 2. Configuration Mechanisms

*   **Environment Variables:** Used for service URLs, ports, log levels, model paths, and high-level settings like the active Titans variant. Easy to manage in Docker/orchestration environments.
*   **Configuration Dictionaries:** Passed to the Python constructors of core classes (`SynthiansMemoryCore`, `NeuralMemoryModule`, `ContextCascadeEngine`, `GeometryManager`, etc.). Allow for fine-grained control over component behavior.
*   **API Endpoints:** Some components (like Neural Memory Server's `/init` and `/config`) allow configuration at runtime via API calls, although startup configuration via environment variables or code is generally preferred for consistency.

## 3. Synthians Memory Core Configuration (`synthians_memory_core`)

These parameters configure the main memory storage and retrieval service, typically controlled via the `config` dictionary passed to the `SynthiansMemoryCore` class constructor and environment variables for the API server.

### 3.1. Core Parameters (`SynthiansMemoryCore` config dict)

| Parameter                       | Type              | Default                               | Description                                                                                                  | Example                  |
| :------------------------------ | :---------------- | :------------------------------------ | :----------------------------------------------------------------------------------------------------------- | :----------------------- |
| `embedding_dim`                 | int               | 768                                   | **CRITICAL:** Dimension of embeddings used throughout the system. Must match embedding model output.         | 768                      |
| `geometry`                      | str               | `hyperbolic`                          | Geometry space for embeddings (`euclidean`, `hyperbolic`, `spherical`, `mixed`). Affects distance calculations. | `hyperbolic`             |
| `hyperbolic_curvature`          | float             | -1.0                                  | Negative curvature for Hyperbolic geometry (magnitude affects scaling). Ignored otherwise.                   | -1.0                     |
| `storage_path`                  | str (Path)        | `/app/memory/stored/synthians`        | **CRITICAL:** Root directory for storing memory files, index, and assemblies. Mount as a volume in Docker. | `./memory_data`        |
| `persistence_interval`          | float             | 60.0                                  | Frequency (seconds) for the background task to persist changed memories and index.                           | 300.0 (5 mins)           |
| `decay_interval`                | float             | 3600.0                                | Frequency (seconds) for applying time decay to QuickRecal scores.                                           | 7200.0 (2 hours)         |
| `prune_check_interval`          | float             | 600.0                                 | Frequency (seconds) for checking if memory pruning is needed based on `max_memory_entries`.                | 900.0 (15 mins)          |
| `max_memory_entries`            | int               | 50000                                 | Maximum number of memories to keep before attempting to prune.                                               | 100000                   |
| `prune_threshold_percent`       | float             | 0.9                                   | Pruning starts when memory count exceeds this percentage of `max_memory_entries` (0.0-1.0).              | 0.95                     |
| `min_quickrecal_for_ltm`        | float             | 0.2                                   | Minimum *effective* QuickRecal score required for a memory to survive pruning (0.0-1.0).                 | 0.15                     |
| `assembly_threshold`            | float             | 0.75                                  | Minimum similarity score for a memory to be added to an existing assembly (0.0-1.0).                         | 0.8                      |
| `max_assemblies_per_memory`     | int               | 3                                     | Maximum number of assemblies a single memory can belong to.                                                | 5                        |
| `adaptive_threshold_enabled`    | bool              | True                                  | Enable/disable the `ThresholdCalibrator` for adaptive retrieval thresholds.                                  | False                    |
| `initial_retrieval_threshold`   | float             | 0.75                                  | Initial similarity threshold for retrieval if adaptive thresholding is disabled or not yet calibrated.       | 0.7                      |
| `vector_index_type`             | str               | `Cosine`                              | FAISS index type (`L2`, `IP`, `Cosine`). `Cosine` often works well with normalized sentence embeddings.       | `IP`                     |
| `use_gpu`                       | bool              | True                                  | Attempt to use GPU for FAISS index operations if available.                                                | True                     |
| `gpu_id`                        | int               | 0                                     | ID of the GPU to use for FAISS if `use_gpu` is True.                                                         | 0                        |
| `gpu_timeout_seconds`           | int               | 10                                    | Max seconds to wait for FAISS GPU initialization before falling back to CPU.                                 | 15                       |
| `geometry_alignment_strategy` | str               | `truncate`                            | How `GeometryManager` handles dimension mismatches (`truncate`, `pad`). Passed to `GeometryManager`.       | `pad`                    |
| `quickrecal_mode`               | str               | `hpc_qr`                              | Scoring mode for `UnifiedQuickRecallCalculator` (`standard`, `hpc_qr`, `minimal`, `custom`).               | `standard`               |
| `quickrecal_factor_weights`   | Dict[str, float]  | *(Depends on mode)*                   | Custom weights for QuickRecall factors if `quickrecal_mode` is `custom`.                                   | `{"recency": 0.5, ...}` |
| `emotion_model_path`            | str               | *(Auto-detects)*                      | Path to the local GoEmotions model for `EmotionAnalyzer`. Falls back to downloading.                     | `/models/emotion`        |
| `emotion_device`                | str               | *(Auto-detects)*                      | Device for emotion model (`cpu` or `cuda`).                                                              | `cuda`                   |

### 3.2. API Server Environment Variables (`api/server.py` via `run_server.py`)

| Variable          | Default             | Description                                                                      | Example               |
| :---------------- | :------------------ | :------------------------------------------------------------------------------- | :-------------------- |
| `HOST`            | `0.0.0.0`           | Host address for the API server to bind to.                                      | `127.0.0.1`           |
| `PORT`            | `5010`              | Port for the API server to listen on.                                            | `8000`                |
| `LOG_LEVEL`       | `INFO`              | Logging level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`).                       | `DEBUG`               |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | Name of the Sentence Transformer model to load for embedding generation.         | `all-MiniLM-L6-v2`    |
| `STORAGE_PATH`    | *(Inherited)*       | If set, overrides `storage_path` in the `SynthiansMemoryCore` config dict.       | `/persistent/memory`  |
| `EMOTION_MODEL_PATH`| *(Inherited)*       | If set, overrides `emotion_model_path` in the `SynthiansMemoryCore` config dict. | `/models/emotion_alt` |

## 4. Neural Memory Server Configuration (`synthians_trainer_server`)

These parameters configure the adaptive associative memory service. They are primarily set via the `NeuralMemoryConfig` class, which can be initialized from a dictionary (e.g., during `POST /init`).

### 4.1. Core Parameters (`NeuralMemoryConfig` dict)

| Parameter               | Type             | Default             | Description                                                                                                         | Example            |
| :---------------------- | :--------------- | :------------------ | :------------------------------------------------------------------------------------------------------------------ | :----------------- |
| `input_dim`             | int              | 768                 | **CRITICAL:** Dimension of the raw input embeddings (`x_t`). Must match Memory Core's `embedding_dim`.                | 768                |
| `key_dim`               | int              | 128                 | Dimension of the Key projections (`k_t`). Should match `query_dim`.                                                 | 128                |
| `value_dim`             | int              | 768                 | Dimension of the Value projections (`v_t`) and the output of the memory MLP (`y_t`). Often matches `input_dim`.         | 768                |
| `query_dim`             | int              | 128                 | Dimension of the Query projections (`q_t`). Should match `key_dim`.                                                 | 128                |
| `memory_hidden_dims`    | List[int]        | `[512]`             | List of hidden layer sizes for the internal Memory MLP (`M`).                                                       | `[1024, 512]`      |
| `gate_hidden_dims`      | List[int]        | `[64]`              | Hidden layer sizes for calculating complex gates (if `use_complex_gates` is True).                                  | `[128]`            |
| `alpha_init`            | float            | -2.0                | Initial value for the *logit* of the alpha gate (forget rate). Sigmoid applied later. Lower => lower forget rate. | -3.0               |
| `theta_init`            | float            | -3.0                | Initial value for the *logit* of the theta gate (inner learning rate). Sigmoid applied later. Lower => lower LR.  | -4.0               |
| `eta_init`              | float            | 2.0                 | Initial value for the *logit* of the eta gate (momentum decay). Sigmoid applied later. Higher => more momentum.     | 3.0                |
| `outer_learning_rate` | float            | 1e-4                | Learning rate for the Adam optimizer used in the outer training loop (`/train_outer`).                             | 5e-5               |
| `use_complex_gates`     | bool             | False               | Whether to use MLP-based gates instead of simple scalar gates (currently noted as not implemented).                 | False              |

### 4.2. API Server Environment Variables (`synthians_trainer_server/http_server.py`)

| Variable                | Default         | Description                                                                          | Example                  |
| :---------------------- | :-------------- | :----------------------------------------------------------------------------------- | :----------------------- |
| `HOST`                  | `0.0.0.0`       | Host address for the API server.                                                     | `127.0.0.1`              |
| `PORT`                  | `8001`          | Port for the API server.                                                             | `8081`                   |
| `LOG_LEVEL`             | `info`          | Logging level.                                                                       | `debug`                  |
| `NM_DEFAULT_STATE_PATH` | `None`          | Path to a state file to automatically load on startup.                                 | `/app/state/nm_init.json`|
| `MEMORY_CORE_URL`       | `http://localhost:5010` | URL of the Memory Core service (used for potential future callbacks).          | `http://memory:5010`     |
| `METRICS_LOG_DIR`       | `./logs`        | Directory where `MetricsStore` saves logs (`memory_updates.jsonl`, etc.).          | `/logs/neural_memory`    |

## 5. Context Cascade Engine (CCE) Configuration (`orchestrator`)

The CCE primarily uses environment variables for configuration.

### 5.1. Environment Variables (`orchestrator/context_cascade_engine.py`)

| Variable                  | Default                 | Description                                                                                   | Example                      |
| :------------------------ | :---------------------- | :-------------------------------------------------------------------------------------------- | :--------------------------- |
| `MEMORY_CORE_URL`         | `http://localhost:5010` | **CRITICAL:** URL of the running Synthians Memory Core service.                               | `http://memory-core-svc:5010`|
| `NEURAL_MEMORY_URL`       | `http://localhost:8001` | **CRITICAL:** URL of the running Neural Memory Server service.                                | `http://neural-memory-svc:8001`|
| `TITANS_VARIANT`          | `NONE`                  | Active Titans variant (`NONE`, `MAC`, `MAG`, `MAL`). Case-insensitive.                        | `MAG`                        |
| `CCE_METRICS_ENABLED`     | `True`                  | Enable/disable cognitive metrics collection via `MetricsStore`. (`True`/`False` string)       | `False`                      |
| `SEQUENCE_CONTEXT_LENGTH` | `50`                    | Max number of historical context tuples (`(ts, id, x, k, v, q, y)`) to keep for attention. | `100`                        |

*(Note: The CCE API server (`orchestrator/server.py`) uses standard `HOST`/`PORT`/`LOG_LEVEL` variables if run directly, typically defaulting to port 8002).*

## 6. Example Configurations

### Scenario 1: Local Development (CPU Only, Standard Memory)

*   **Goal:** Run all services locally on a machine without a dedicated GPU, focusing on basic memory functions.
*   **Memory Core (`SynthiansMemoryCore` config / env vars):**
    *   `embedding_dim: 384` (using `all-MiniLM-L6-v2`)
    *   `EMBEDDING_MODEL: all-MiniLM-L6-v2`
    *   `geometry: euclidean`
    *   `use_gpu: False`
    *   `vector_index_type: Cosine`
    *   `quickrecal_mode: standard`
    *   `storage_path: ./local_memory_data`
    *   `PORT: 5010`
*   **Neural Memory Server (Env vars / `/init` config):**
    *   `input_dim: 384`
    *   `key_dim: 64`
    *   `value_dim: 384`
    *   `query_dim: 64`
    *   `memory_hidden_dims: [256]`
    *   `PORT: 8001`
    *   `METRICS_LOG_DIR: ./local_nm_logs`
*   **CCE (Env vars):**
    *   `MEMORY_CORE_URL: http://localhost:5010`
    *   `NEURAL_MEMORY_URL: http://localhost:8001`
    *   `TITANS_VARIANT: NONE`
    *   `CCE_METRICS_ENABLED: True`
    *   *(Run on default port 8002)*

### Scenario 2: Docker Deployment (GPU Enabled, Hyperbolic, MAG Variant)

*   **Goal:** Deploy services in Docker, leveraging GPU for FAISS and potentially emotion/embedding models, using Hyperbolic geometry and the MAG attention variant.
*   **`docker-compose.yml` Environment Variables:**
    *   **Memory Core Service:**
        *   `PORT=5010`
        *   `EMBEDDING_MODEL=all-mpnet-base-v2` (Requires 768 dim)
        *   `STORAGE_PATH=/data/memory` (Mounted volume)
        *   `EMOTION_MODEL_PATH=/models/emotion` (Mounted volume)
        *   `USE_GPU=True` (Passed to `SynthiansMemoryCore` config)
        *   `GEOMETRY=hyperbolic` (Passed to `SynthiansMemoryCore` config)
        *   `QUICKRECAL_MODE=hpc_qr` (Passed to `SynthiansMemoryCore` config)
        *   `VECTOR_INDEX_TYPE=Cosine`
        *   **(Runtime must be configured for NVIDIA in compose file)**
    *   **Neural Memory Service:**
        *   `PORT=8001`
        *   `METRICS_LOG_DIR=/logs/neural_memory` (Mounted volume)
        *   `NM_DEFAULT_STATE_PATH=/data/state/nm_latest.json` (Optional, mounted volume)
        *   **(Runtime must be configured for NVIDIA if using TF-GPU)**
        *   *(Config dimensions like `input_dim: 768`, `key_dim: 128`, etc., are set via default or loaded state)*
    *   **CCE Service:**
        *   `MEMORY_CORE_URL=http://memory-core:5010` (Using Docker service names)
        *   `NEURAL_MEMORY_URL=http://neural-memory:8001`
        *   `TITANS_VARIANT=MAG`
        *   `CCE_METRICS_ENABLED=True`
        *   `SEQUENCE_CONTEXT_LENGTH=100`

### Scenario 3: Minimal Resource Footprint (e.g., Edge Device - Conceptual)

*   **Goal:** Configure for minimal CPU/RAM usage, sacrificing some features.
*   **Memory Core:**
    *   `embedding_dim: 384`
    *   `geometry: euclidean`
    *   `use_gpu: False`
    *   `vector_index_type: L2` (Potentially simpler index types if needed)
    *   `quickrecal_mode: minimal`
    *   `adaptive_threshold_enabled: False`
    *   `max_memory_entries: 5000`
    *   `persistence_interval: 600` (Less frequent saves)
    *   `decay_interval: 86400` (Daily decay)
*   **Neural Memory Server:**
    *   `input_dim: 384`, `key_dim: 32`, `value_dim: 384`, `query_dim: 32`
    *   `memory_hidden_dims: [128]` (Smaller MLP)
    *   (Likely wouldn't run Neural Memory on a very constrained edge device, but shows config options)
*   **CCE:**
    *   `TITANS_VARIANT: NONE`
    *   `CCE_METRICS_ENABLED: False`
    *   `SEQUENCE_CONTEXT_LENGTH: 20`

## 7. Key Considerations

*   **Consistency:** `embedding_dim` *must* be consistent between the Memory Core, the chosen `EMBEDDING_MODEL`, and the `input_dim`/`value_dim` of the Neural Memory Server.
*   **URLs:** Ensure service URLs (`MEMORY_CORE_URL`, `NEURAL_MEMORY_URL`) are correct for the deployment environment (e.g., `localhost` for local, service names for Docker Compose/Kubernetes).
*   **Storage Paths:** Directories specified in `storage_path`, `METRICS_LOG_DIR`, `EMOTION_MODEL_PATH`, `NM_DEFAULT_STATE_PATH` must exist and have correct permissions, especially when using Docker volumes.
*   **GPU:** For GPU usage, ensure NVIDIA drivers, CUDA toolkit, `nvidia-docker` (or equivalent), and GPU-enabled versions of libraries (FAISS-GPU, TensorFlow-GPU) are installed correctly in the environment/container.
*   **Titans Variants:** The CCE will dynamically use the variant specified by `TITANS_VARIANT`. Ensure the Neural Memory Server supports the required API endpoints for the selected variant (e.g., `/calculate_gates` for MAG, specific `/update_memory` parameters for MAG/MAL).

This guide provides a comprehensive overview of configuring the Synthians Cognitive Architecture. Always refer to the specific component code for the most up-to-date default values and parameter handling.
