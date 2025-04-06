
---

## **Phase 5.9: Clarity Emerges - Backend Explainability & Diagnostics CHEAT SHEET (v1.1)**

**🎯 Goal:** Implement backend APIs & logic for explaining Memory Core operations (activation, merge, lineage) and providing diagnostics (merge log, runtime config, activation stats) to support the Cognitive Dashboard (Phase 5.9.1).

---

### **📦 Key New Modules / Files**

*   `synthians_memory_core/explainability/`: Core Python logic for generating explanations.
    *   `activation.py`, `merge.py`, `lineage.py`, `_explain_helpers.py`
    *   *(Consideration: May refactor into an `ExplainabilityService` class later for better abstraction)*
*   `synthians_memory_core/metrics/`: Diagnostics data capture.
    *   `merge_tracker.py` (`MergeTracker` class)
    *   (Activation stats persistence logic likely integrated into `SynthiansMemoryCore`)
*   `synthians_memory_core/api/explainability_routes.py`: FastAPI routes for `/explain_*`.
*   `tests/integration/test_phase_5_9_explainability.py`: Integration tests.
*   `docs/api/phase_5_9_models.md`: **Definitive Pydantic models & JSON examples.**
*   `docs/testing/PHASE_5_9_TESTING.md`: Detailed testing strategy.
*   `docs/api/API_ERRORS.md` (Optional): Detailed error codes/formats.

---

### **🔄 Key Updated Modules / Files**

*   `synthians_memory_core/synthians_memory_core.py`: Integrates `MergeTracker`, activation counting, stats persistence.
*   `synthians_memory_core/api/diagnostics_routes.py`: Adds `/merge_log`, `/config/runtime/*`.
*   `synthians_memory_core/api/server.py`: Conditionally mounts new routers (`ENABLE_EXPLAINABILITY`), enhances `/stats`.
*   `synthians_memory_core/memory_structures.py`: `MemoryAssembly.merged_from` utilized.
*   `synthians_memory_core/memory_persistence.py`: Ensures `merged_from` persistence.
*   `docs/...`: `ARCHITECTURE.md`, `COMPONENT_GUIDE.md`, `API_REFERENCE.md`, `CONFIGURATION_GUIDE.md`, `CHANGELOG.md`, `core/diagnostics.md`, `core/explainability.md`.

---

### **💡 Core Logic & Mechanisms**

1.  **Explainability (`explainability/`)**
    *   **Activation (`generate_activation_explanation`):**
        *   *Input:* `assembly_id`, `memory_id`
        *   *Logic:* Load embeds (`Persistence`) -> Calc Similarity (`GeometryManager`) -> Compare vs Threshold (Core Config).
        *   *Output:* `ExplainActivationData`.
    *   **Merge (`generate_merge_explanation`):**
        *   *Input:* `assembly_id`, `merge_tracker_instance`
        *   *Logic:* Load Assembly (`Persistence`) -> Check `merged_from` -> Query `MergeTracker` log -> Get details (sources, sim, threshold, timestamp, cleanup status).
        *   *Output:* `ExplainMergeData`.
    *   **Lineage (`trace_lineage`):**
        *   *Input:* `assembly_id`, `persistence_instance`
        *   *Logic:* Recursive load via `merged_from`.
        *   *Output:* `List[LineageEntry]`.
        *   *(Future: May need compressed/summary views for deep lineage).*

2.  **Diagnostics (`metrics/` & Core)**
    *   **Merge Tracking (`MergeTracker`):**
        *   `log_merge_event`: Called by `_execute_merge`. Writes JSONL (`merge_log.jsonl`). `cleanup_status="pending"`.
        *   `update_cleanup_status`: Called by async cleanup task. **Logs separate `cleanup_status_update` event referencing original `merge_event_id` (Option B).**
        *   `read_log_entries`: Reads recent lines for API. Handles correlation of merge/cleanup events.
        *   Rotation: Based on `merge_log_max_entries`.
    *   **Runtime Config (`diagnostics_routes.py`):**
        *   Reads current config dict.
        *   **CRITICAL:** Applies **strict allow-list sanitization** (`SAFE_CONFIG_KEYS_*`).
    *   **Activation Stats (`SynthiansMemoryCore`):**
        *   In-memory dict `_assembly_activation_counts` incremented.
        *   Persisted periodically to `stats/assembly_activation_stats.json`.
        *   `/stats` API loads file for reporting.

---

### **📡 New API Endpoints (Memory Core API - `localhost:5010`)**

*   `GET /assemblies/{assembly_id}/explain_activation`: Why memory activated in assembly.
*   `GET /assemblies/{assembly_id}/explain_merge`: How this assembly was formed by merge.
*   `GET /assemblies/{assembly_id}/lineage`: Merge ancestry history.
*   `GET /diagnostics/merge_log`: Recent merge events (correlates merge/cleanup).
*   `GET /config/runtime/{service_name}`: *Sanitized* runtime config.
*   **Note:** See `docs/api/phase_5_9_models.md` for detailed request/response JSON examples and schemas.

---

### **💾 Key Data Structures / Fields / Files**

*   `MemoryAssembly.merged_from`: `List[str]` - Source assembly IDs. Basis for lineage.
*   `logs/merge_log.jsonl`: Append-only JSON Lines file. Main source for merge explanations.
    *   *Merge Event:* `{ "event_type": "merge", "merge_event_id": "...", "timestamp": "...", "source_assembly_ids": [...], ... "cleanup_status": "pending"}`
    *   *Cleanup Event (Option B):* `{ "event_type": "cleanup_update", "merge_event_id": "...", "timestamp": "...", "status": "completed" | "failed", "error": "..." }`
*   `stats/assembly_activation_stats.json`: `{"asm_id": count, ...}`.

---

### **⚙️ New Configuration Flags (Memory Core)**

*   `ENABLE_EXPLAINABILITY` (bool, default: `true`): Master switch for new Phase 5.9 API endpoints.
*   `merge_log_max_entries` (int, default: `1000`): Max lines in `merge_log.jsonl`.
*   `assembly_metrics_persist_interval` (float, default: `600.0`): Seconds between saving activation stats.

---

### **🔗 Dependencies & Flow Summary**

*   **API Layer:**
    *   Explain Routes -> `explainability` functions (or future Service).
    *   Diagnostics Routes -> `MergeTracker`, Config Sanitization.
    *   `/stats` -> Reads `assembly_activation_stats.json`.
*   **Explainability Module:**
    *   Uses `MemoryPersistence`, `GeometryManager`, `MergeTracker`, Core Config.
*   **Core Logic (`SynthiansMemoryCore`):**
    *   `_execute_merge` -> Logs `merge` event. Populates `merged_from`. Schedules async cleanup.
    *   `_cleanup_and_index_after_merge` -> Logs `cleanup_update` event.
    *   `_activate_assemblies` -> Increments activation counts.
    *   Background Loop -> Saves activation stats.

---

### **⚠️ Key Pitfalls Reminder**

*   **Config Sanitization:** Strict allow-list for `/config/runtime` is non-negotiable.
*   **Merge Log Updates:** Ensure Option B (Separate Event) logic is robustly implemented in `MergeTracker` and the `/diagnostics/merge_log` endpoint correctly correlates events.
*   **Async/Concurrency:** Use `aiofiles`. Protect shared resources (`MergeTracker` file access, activation counts) with locks. Beware race conditions.
*   **Error Handling:** Implement **uniform error response shapes** across all new API endpoints for client consistency (e.g., `{"success": false, "error_code": "...", "message": "..."}`). See `API_ERRORS.md`.
*   **Feature Flag:** Test *all* new endpoints with `ENABLE_EXPLAINABILITY` as both `true` and `false`.
*   **Performance:** Monitor `/explain_activation`, `/lineage`, `/diagnostics/merge_log` (especially with correlation).

---
