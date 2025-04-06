
---

## 📄 **Synthians Cognitive System Cheat Sheet (Phase 5.9.1 Complete)**

*“The blueprint remembers, the associator learns, the cascade connects, and now the dashboard reveals the inner workings.”*

---

### 🏛️ **MEMORY CORE (MC) — *The Archive & Introspection Hub***

*   **Core File:** `SynthiansMemoryCore` (`synthians_memory_core`)
*   **Role:** Persistent, indexed storage; relevance scoring (QuickRecal); assembly management; **provides backend for explainability and diagnostics**.
*   **Key Internal Modules (Phase 5.9 Additions):**
    *   `explainability/`: Contains logic for:
        *   `activation.py` (`generate_activation_explanation`)
        *   `merge.py` (`generate_merge_explanation`)
        *   `lineage.py` (`trace_lineage`)
    *   `metrics/`: Contains logic for:
        *   `merge_tracker.py` (`MergeTracker` - **Append-only** `merge_log.jsonl`)
        *   Assembly activation stats tracking (in `SynthiansMemoryCore`, persists to `stats/assembly_activation_stats.json`)
*   **Key APIs Exposed (Consumed by Dashboard Proxy):**
    *   `GET /assemblies/{id}/explain_activation?memory_id={mem_id}`: Explains memory activation within an assembly.
    *   `GET /assemblies/{id}/explain_merge`: Explains how a merged assembly was formed.
    *   `GET /assemblies/{id}/lineage`: Traces assembly ancestry.
    *   `GET /diagnostics/merge_log`: Returns **reconciled** merge events.
    *   `GET /config/runtime/{service_name}`: Returns **sanitized** runtime config (MC, NM, CCE).
    *   `GET /stats`: Now includes assembly activation counts.
*   **Configuration:**
    *   `ENABLE_EXPLAINABILITY` (bool): Master switch for all 5.9 features. **Crucial for dashboard functionality.**
    *   `MERGE_LOG_*`: Settings for the merge log file (path, rotation).
    *   `ASSEMBLY_METRICS_PERSIST_INTERVAL`: How often activation stats are saved.
    *   `MAX_LINEAGE_DEPTH`: Default limit for lineage traces.

---

### 🧠 **NEURAL MEMORY (NM) — *The Associator***

*   **Core File:** `NeuralMemoryModule` (`synthians_trainer_server`)
*   **Role:** Adaptive associative sequence memory (Titans-based); surprise calculation.
*   **Phase 5.9.1 Integration:**
    *   Provides runtime configuration data via MC API (`/config/runtime/neural-memory`).
    *   Provides diagnostic data (`/diagnose_emoloop`) for dashboard display.
*   *(No major internal changes from 5.8)*

---

### ⚙️ **Context Cascade Engine (CCE) — *The Orchestrator***

*   **Core File:** `ContextCascadeEngine` (`orchestrator`)
*   **Role:** Manages MC↔NM flow, implements cognitive cycle, dynamic variant selection, LLM guidance.
*   **Phase 5.9.1 Integration:**
    *   Provides runtime configuration data via MC API (`/config/runtime/cce`).
    *   Provides enhanced metrics (`/metrics/recent_cce_responses`) including variant selection reasons, LLM advice usage, performance data used for decisions.
*   **Titans Variant Naming:** Frontend representation corrected to use base names (e.g., "MAC", "MAG", "MAL") without inaccurate parameter counts (like "13b"). Actual model parameters are internal details, not part of the variant *type* name shown in UI.

---

### 🖥️ **SYNTHIANS COGNITIVE DASHBOARD — *The Interface***

*   **Core Files:** `Synthians_dashboard/client/` (React Frontend), `Synthians_dashboard/server/` (Express Proxy)
*   **Role:** Provides UI for monitoring, inspection, and limited interaction with MC, NM, and CCE services via its **backend proxy**.
*   **Key Phase 5.9.1 Integrations:**
    *   **Feature Flag Handling:** Uses `ENABLE_EXPLAINABILITY` (fetched via `/config/runtime/memory-core`) to conditionally show/hide explainability/diagnostics UI elements (`FeaturesContext`).
    *   **Runtime Configuration View:** Displays sanitized configs for MC, NM, CCE (`useRuntimeConfig` hook -> `/config` page).
    *   **Merge Log View:** Displays reconciled merge events from `MergeTracker` (`useMergeLog` hook -> `/logs` or diagnostics page -> `MergeLogView.tsx`).
    *   **Assembly Inspector Enhancements:**
        *   **Lineage:** Triggers fetch (`useAssemblyLineage` hook) and displays ancestry (`LineageView.tsx`). Allows setting `max_depth`.
        *   **Merge Explanation:** Triggers fetch (`useExplainMerge` hook) and displays merge details or "not merged" message (`MergeExplanationView.tsx`).
        *   **Activation Explanation:** Triggers fetch (`useExplainActivation` hook on memory select) and displays activation details (`ActivationExplanationView.tsx`).
    *   **Stats Display:** Integrates assembly activation stats from MC `/stats` endpoint into relevant views (e.g., Overview, Memory Core page).
    *   **API Client:** `lib/api.ts` updated with hooks for all new endpoints.
    *   **Shared Schema:** `shared/schema.ts` updated with TypeScript interfaces matching backend Pydantic models.
    *   **Proxy Server:** `server/routes.ts` updated with proxy routes for all new MC endpoints.

---

### ✨ **Key Explainability & Diagnostics Flow (Post 5.9.1)**

1.  **User Action (Dashboard):** Clicks "Explain Merge" on Assembly `asm_xyz`.
2.  **Frontend Request:** React component calls `explainMergeQuery.refetch()`. Hook (`useExplainMerge`) sends `GET /api/memory-core/assemblies/asm_xyz/explain_merge` to the **Dashboard Proxy**.
3.  **Proxy Forwarding:** Dashboard Proxy (`server/routes.ts`) forwards the request to `Memory Core API: GET /assemblies/asm_xyz/explain_merge`.
4.  **Memory Core Processing:**
    *   API route (`api/explainability_routes.py`) checks `ENABLE_EXPLAINABILITY`.
    *   Calls `explainability.merge.generate_merge_explanation(assembly_id="asm_xyz", ...)`.
    *   `generate_merge_explanation` loads assembly `asm_xyz` via `Persistence`. Checks `merged_from`.
    *   Queries `MergeTracker` (`metrics/merge_tracker.py`) for `merge_creation` and `cleanup_status_update` events related to `asm_xyz`.
    *   `MergeTracker` reads `merge_log.jsonl` and reconciles events.
    *   Explanation is constructed and returned as JSON matching `ExplainMergeResponse` model.
5.  **Response Journey:** MC API -> Proxy -> Frontend Hook -> React Component -> Rendered in `MergeExplanationView.tsx`.

*(Similar flows exist for Activation Explanation, Lineage, Merge Log, and Runtime Config)*

---

### ⚠️ **Key Considerations (Post 5.9.1)**

*   **`ENABLE_EXPLAINABILITY` Flag:** Controls visibility and accessibility of all new dashboard features related to Phase 5.9 backend capabilities. **Must be `true` on Memory Core for dashboard features to function.**
*   **Proxy Routes:** The Dashboard's backend proxy (`server/routes.ts`) **must** have routes defined for all new MC API endpoints.
*   **Schema Sync:** `shared/schema.ts` (Frontend TS) **must** match `docs/api/phase_5_9_models.md` (Backend Pydantic).
*   **Performance:** Lineage tracing can be I/O intensive; API caching is implemented. Merge log reconciliation reads from disk. Consider implications for large logs.
*   **"Titans Variant" Naming:** Frontend UI corrected to display base variant names (MAC, MAG, MAL) without parameter counts like "13b", which are internal implementation details not reflected in the variant *type*.

---
