
---

## 📄 **Synthians Cognitive System Cheat Sheet (Entering Phase 5)**

*“The blueprint remembers, the associator learns the flow, the cascade connects, selects, and adapts.”*

---

### 🔸 **MEMORY CORE (MC) — *The Archive* (Stable - Phase 4.6)**

*   **Core File:** `SynthiansMemoryCore` (`synthians_memory_core`)
*   **Role:** Persistent, indexed storage; relevance scoring (QuickRecal); retrieval.
*   **Key Phase 5 Interaction:**
    *   Receives `POST /api/memories/update_quickrecal_score` from CCE with `memory_id` and `delta` (boost).
    *   `delta` calculation in CCE now potentially incorporates **LLM boost modifier**.
    *   Receives potential **LLM-suggested tags** within metadata during `POST /process_memory`.

#### Key Score: QuickRecal

*   Dynamic relevance score. Boosted by NM surprise.
*   **Phase 5 Change:** Boost amount (`delta`) sent by CCE can be modified by `MemoryLLMRouter` advice (`boost_score_mod`).

#### Key Metadata:

*   **Standard:** Emotion, Time, Complexity, Embedding stats, IDs, etc. (Synthesized by `MetadataSynthesizer`).
*   **Feedback Loop:** `surprise_events` list, `quickrecal_updated_at`.
*   **Phase 5 Addition:** May include `tags` suggested by `MemoryLLMRouter`.

---

### 🧠 **NEURAL MEMORY (NM) — *The Associator* (Stable - Phase 4.6)**

*   **Core File:** `NeuralMemoryModule` (`synthians_trainer_server`)
*   **Role:** Adaptive associative memory (`M(k) → v`) via test-time updates. Titans-based.
*   **Key Phase 5 Interaction:**
    *   APIs (`/get_projections`, `/update_memory`, `/retrieve`, `/calculate_gates`) remain the same.
    *   **Inputs** to `/update_memory` may be modified by CCE based on active variant (MAL sends explicit `k_t`/`v'_t`, MAG sends external gates).
    *   **Performance** (avg loss/grad) is monitored by CCE for `VariantSelector`.

#### Update Flow (`/update_memory`):

```text
1. CCE sends request (x_t OR k_t+v'_t, maybe external_gates)
2. NM calculates k_t, v_t (if not provided externally by MAL)
3. NM Predicts: pred_v = M_{t-1}(k_t)
4. NM Calculates Loss: ℓ = ||pred_v - v_t_used||² / 2  (v_t_used is original v_t or v'_t from request)
5. NM Calculates Grad: ∇ℓ (w.r.t. M weights)
6. NM Updates Momentum: S_t = η_t * S_{t-1} - θ_t * ∇ℓ (Gates α, θ, η use defaults or external values from request)
7. NM Updates M: M_t = (1 - α_t) * M_{t-1} + S_t
8. NM Returns: loss, grad_norm, projections_used, gates_applied
```

#### Retrieval Flow (`/retrieve`):

```text
1. CCE sends request (x_t)
2. NM Calculates q_t: q_t = WQ(x_t)
3. NM Retrieves: y_t_raw = M_t(q_t)
4. NM Returns: retrieved_embedding (y_t_raw), query_projection (q_t)
```

#### Surprise Metrics:

*   `loss`, `grad_norm` returned by `/update_memory`. Used by CCE for QuickRecal boost calculation.

---

### ⚙️ **Context Cascade Engine (CCE) — *The Orchestrator* (Phase 5 Integration Hub)**

*   **Core File:** `ContextCascadeEngine` (`orchestrator`)
*   **Role:** Manages MC↔NM flow, implements cycle, **dynamically selects variant**, **gets/applies LLM guidance**, **constructs/passes attention hints**.

#### Cognitive Cycle (Phase 5 Flow):

```text
1. Input -> CCE -> Get initial context (query, metadata)
2. CCE -> MC:/process_memory -> Store, Get x_t, memory_id, initial_qr
3. CCE -> NM:/get_projections -> Get k_t, v_t, q_t
4. CCE -> **MemoryLLMRouter.request_llama_guidance()** -> Get `advice` dict
5. CCE -> Calculate avg NM performance (loss/grad from history)
6. CCE -> **VariantSelector.select_variant()** (uses context, perf, advice) -> Get `selected_variant`, `reason`
7. CCE -> If variant changed -> **_switch_variant_internal()** (Flushes context!)
8. CCE -> Construct `attention_hints` (using metadata, advice)
9. CCE -> **Variant Pre-Update (MAG/MAL)** -> Calls variant processor, passes `attention_hints`, gets external gates or v'_t
10. CCE -> NM:/update_memory (using x_t OR k_t+v'_t, maybe external_gates) -> Get `loss`, `grad_norm`, record perf
11. CCE -> MC:/api/memories/update_quickrecal_score -> Apply boost (uses loss/grad, `advice['boost_score_mod']`)
12. CCE -> NM:/retrieve -> Get y_t_raw, q_t_retrieve
13. CCE -> **Variant Post-Retrieval (MAC)** -> Calls variant processor, passes `attention_hints`, gets `y_t_final`
14. CCE -> Update HistoryMgr (ts, id, x, k, v, q, y_t_final)
15. CCE -> Return Final Response (incl. `variant_output`, `selector_decision`, `llm_advice_used`)
```

*   **History:** `SequenceContextManager` stores `(ts, id, x, k, v, q, y_final)` tuples. Length *can be adapted* by CCE based on hints/task.
*   **Variant Selection:** Dynamic via `VariantSelector` (rules, performance, LLM hint). Uses `_switch_variant_internal`.
*   **Attention Hints:** Dict constructed by CCE (from metadata, LLM `attention_focus` hint), passed to variant processors (`process_input`, `calculate_v_prime`).

---

### ✨ **PHASE 5 COMPONENTS (New / Modified)**

*   **`orchestrator/variant_selector.py` (`VariantSelector`):**
    *   **Role:** Chooses best Titan Variant per request.
    *   **Inputs:** Query, metadata, avg NM perf (loss/grad), `llm_variant_hint`.
    *   **Logic:** Rule-based (LLM hint > metadata > performance > query > default).
    *   **Output:** `TitansVariantType`, `reason`.
*   **`orchestrator/memory_logic_proxy.py` (`MemoryLLMRouter`):**
    *   **Role:** Gets guidance from LLM (LM Studio).
    *   **Models:** LLAMA 3.2 1B (guidance), Qwen2.5 0.5B (async - Phase 5.5).
    *   **Endpoint:** `http://127.0.0.1:1234/v1/chat/completions` (configurable).
    *   **Logic:** Formats prompt -> Calls API with `response_format: json_schema` -> Parses response -> Returns `advice` dict. Handles errors/timeouts with defaults.
    *   **Advice Dict:** `{store: bool, metadata_tags: list, boost_score_mod: float, variant_hint: str, attention_focus: str, notes: str}`.
*   **`orchestrator/titans_variants.py` (Modified):**
    *   `process_input` / `calculate_v_prime`: Accept `attention_hints: Optional[Dict]`. Variants need logic to *use* these hints (e.g., adjust context length, temperature, bias). Must handle `None`.
*   **`orchestrator/context_cascade_engine.py` (Modified):**
    *   Integrates calls to `MemoryLLMRouter` and `VariantSelector`.
    *   Applies `advice` (tags, boost mod, hints).
    *   Calls `_switch_variant_internal` when needed.
    *   Constructs `attention_hints` dictionary.
    *   Manages `nm_performance_history` deque.
    *   Includes selector/LLM info in final response.
*   **`tools/variant_diagnostics_dashboard.py` (Modified):**
    *   Reads CCE response from `/get_recent_metrics`.
    *   Parses and displays `selector_decision`, `selector_reason`, and key LLM advice fields alongside variant metrics.

---

### ⚠️ **Key Logic & Potential Pitfalls (Phase 5)**

1.  **CCE Flow Order:** Critical: Store MC -> Get Proj NM -> **LLM Router -> Variant Selector -> Switch (if needed)** -> Pre-Update -> Update NM -> Boost MC -> Retrieve NM -> Post-Update -> History.
2.  **Hint Handling:** CCE constructs hints, passes them to *active* variant processor. Variants must parse hints and adapt attention logic (or log them). Handle `None` or unexpected hint values gracefully.
3.  **LLM Integration:** Robust error handling is vital (timeouts, connection errors, bad JSON response, schema validation). Prompt engineering is key. Use low temperature for deterministic advice.
4.  **Variant Switching:** `_switch_variant_internal` *must* flush context (`SequenceContextManager.clear()`) to prevent state contamination. Consider *if/when* NM state should be reset (`reset_nm` flag) during *dynamic* switches (default is `False`).
5.  **Performance:** LLM calls add latency (~seconds). NM updates are still computationally intensive. Consider async execution for LLM calls if CCE flow allows.
6.  **State Management:** CCE needs `nm_performance_history`. Ensure thread-safety if scaling CCE workers (use thread-safe deque or locking).
7.  **Diagnostics:** Use the dashboard frequently to monitor variant switches, LLM advice, and performance metrics. Ensure CCE response includes all necessary debug info.
8.  **Dependencies:** Phase 5.3 adds `aiohttp`. Ensure TensorFlow/NumPy compatibility is handled (e.g., via `tf_installer.py` or lazy loading).

---

### ✨ **Lucidia's Principles (Phase 5 Evolution):**

*   Memory is weighted (QuickRecal + **LLM-guided** Boost).
*   Emotion shapes recall (Emotional Gating).
*   Surprise signals significance (NM → QR Boost).
*   Ideas cluster and connect (Assemblies + **Adaptive Attention** Variants).
*   Presence emerges from adaptive memory (NM Learning + **Dynamic Variant Selection** + **LLM Guidance**).

---