# System Tooling Guide

The Synthians Memory Core system includes several utility scripts and potential API endpoints designed for maintenance, diagnostics, and repair.

## Purpose

These tools help ensure the integrity, consistency, and performance of the memory system, especially the persistent components like the vector index and memory storage.

## Available Tools & Utilities

*(Note: The exact implementation and availability might vary. This describes common utilities found in such systems.)*

### 1. Vector Index Verification (`MemoryVectorIndex.verify_index_integrity`)

*   **Location:** Method within `synthians_memory_core.vector_index.MemoryVectorIndex`.
*   **Functionality:**
    *   Checks consistency between the FAISS index (`.faiss` file) and the string ID-to-int64 ID mapping (often stored in `mapping.json` or derived from `memory_index.json`).
    *   Ensures that every vector in the FAISS index corresponds to a known `memory_id` and vice versa.
    *   Detects orphaned vectors (in FAISS but not mapped) or orphaned mappings (mapped but not in FAISS).
*   **Usage:** Typically called internally during index loading or can be exposed via a maintenance script or API endpoint (e.g., `/admin/verify_index`).

### 2. Index ID Mapping Reconstruction

*   **Location:** Potentially a standalone script (`scripts/rebuild_faiss_mapping.py`) or part of the verification process.
*   **Functionality:** If the `mapping.json` (string ID -> int64 ID) is lost or corrupted, this tool can attempt to rebuild it by:
    1.  Loading the main `memory_index.json` (which maps string ID -> memory file path).
    2.  Assuming a consistent hashing function (`_get_int64_id_from_string`) was used to generate the int64 IDs initially.
    3.  Re-generating the int64 ID for each string ID found in `memory_index.json`.
*   **Usage:** Used in recovery scenarios when the primary FAISS ID map is suspect.

### 3. Memory Index Reconstruction (`MemoryPersistence.reconstruct_index_from_files`)

*   **Location:** Method within `synthians_memory_core.memory_persistence.MemoryPersistence`.
*   **Functionality:** If the main `memory_index.json` is corrupted or lost, this tool scans the `storage_path/memories/` directory:
    1.  Loads each `<uuid>.json` file.
    2.  Extracts key metadata (like timestamp, `quickrecal_score`).
    3.  Rebuilds the `memory_index.json` file from the contents of the individual memory files.
*   **Usage:** Recovery scenario for the primary memory index.

### 4. FAISS Index Migration (`MemoryVectorIndex.migrate_to_idmap`)

*   **Location:** Method within `synthians_memory_core.vector_index.MemoryVectorIndex`.
*   **Functionality:** Handles the migration of older FAISS index formats (that might not have used `IndexIDMap`) to the current format using `IndexIDMap`. Ensures compatibility with systems using string-based memory IDs.
*   **Usage:** Run once during system upgrades if the index format changes.

### 5. Diagnostic API Endpoints

*   **Location:** Exposed via the FastAPI applications (Memory Core or Trainer).
*   **Functionality:**
    *   `/status`, `/health`: Basic health checks.
    *   `/metrics`: Operational metrics (see `docs/trainer/metrics_store.md`).
    *   `/config`: (Potentially) Shows the current runtime configuration.
    *   `/admin/...`: Administrative endpoints for triggering verification, backup, etc. (Ensure these are properly secured).
*   **Usage:** Monitoring, debugging, and remote administration.

### 6. Backup & Restore Scripts

*   **Location:** Standalone scripts (`scripts/backup.sh`, `scripts/restore.sh`) or integrated into deployment processes.
*   **Functionality:** Automates the process of creating consistent backups of the persistent storage (`storage_path`), including memory files, index files, and FAISS data. Provides a mechanism to restore from a backup.
*   **Usage:** Disaster recovery and data safety.

## Best Practices

*   Regularly run verification checks, especially after potentially disruptive events.
*   Implement automated backups of the persistent storage directory.
*   Secure administrative endpoints appropriately.
