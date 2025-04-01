Okay, here is the updated Cheat Sheet reflecting the completion of Phase 5.5 (Performance-Aware Selection) and the start of Phase 5.6 (LLM Guidance Refinement - Prompt & Metrics).

---

## 📄 **Synthians Cognitive System Cheat Sheet (Entering Phase 5.6)**

*“The blueprint remembers, the associator learns the flow, the cascade connects, selects based on performance, and adapts with guidance.”*

---

### 🔸 **MEMORY CORE (MC) — *The Archive* (Stable - Phase 4.6)**

*   **Core File:** `SynthiansMemoryCore` (`synthians_memory_core`)
*   **Role:** Persistent, indexed storage; relevance scoring (QuickRecal); retrieval.
*   **Key Phase 5 Interaction:**
    *   Receives `POST /api/memories/update_quickrecal_score` from CCE with `memory_id` and `delta` (boost).
    *   `delta` calculation in CCE now incorporates **LLM boost modifier (potentially confidence-adjusted)**.
    *   Receives potential **LLM-suggested tags** within metadata during `POST /process_memory`.

#### Key Score: QuickRecal

*   Dynamic relevance score. Boosted by NM surprise.
*   **Phase 5 Change:** Boost amount (`delta`) sent by CCE is modified by `MemoryLLMRouter` advice (`boost_score_mod`), potentially scaled/capped by **performance confidence**.

#### Key Metadata:

*   **Standard:** Emotion, Time, Complexity, Embedding stats, IDs, etc. (Synthesized by `MetadataSynthesizer`).
*   **Feedback Loop:** `surprise_events` list, `quickrecal_updated_at`.
*   **Phase 5 Addition:** May include `tags` suggested by `MemoryLLMRouter`.

---

### 🧠 **NEURAL MEMORY (NM) — *The Associator* (Stable - Phase 4.6)**

*   **Core File:** `NeuralMemoryModule` (`synthians_trainer_server`)
*   **Role:** Adaptive associative memory (`M(k) → v`) via test-time updates. Titans-based.
*   **Key Phase 5 Interaction:**
    *   APIs (`/get_projections`, `/update_memory`, `/retrieve`, `/calculate_gates`) remain stable.
    *   Inputs to `/update_memory` may be modified by CCE based on active variant.
    *   **Performance** (loss/grad) is returned on `/update_memory` and tracked by CCE for `VariantSelector` and `MemoryLLMRouter`.

#### Update Flow (`/update_memory`):

```text
# (Same as previous phase - NM internals are stable)
1. CCE sends request (x_t OR k_t+v'_t, maybe external_gates)
2. NM calculates k_t, v_t (if not provided externally by MAL)
3. NM Predicts: pred_v = M_{t-1}(k_t)
4. NM Calculates Loss: ℓ = ||pred_v - v_t_used||² / 2
5. NM Calculates Grad: ∇ℓ (w.r.t. M weights)
6. NM Updates Momentum: S_t = η_t * S_{t-1} - θ_t * ∇ℓ
7. NM Updates M: M_t = (1 - α_t) * M_{t-1} + S_t
8. NM Returns: loss, grad_norm, projections_used, gates_applied
```

#### Retrieval Flow (`/retrieve`):

```text
# (Same as previous phase)
1. CCE sends request (x_t)
2. NM Calculates q_t: q_t = WQ(x_t)
3. NM Retrieves: y_t_raw = M_t(q_t)
4. NM Returns: retrieved_embedding (y_t_raw), query_projection (q_t)
```

#### Surprise Metrics:

*   `loss`, `grad_norm` returned by `/update_memory`. Used by CCE for QuickRecal boost calculation **and** performance tracking.

---

### ⚙️ **Context Cascade Engine (CCE) — *The Orchestrator* (Phase 5.6 Integration Hub)**

*   **Core File:** `ContextCascadeEngine` (`orchestrator`)
*   **Role:** Manages MC↔NM flow, implements cycle, **tracks NM performance (incl. trends, confidence)**, **dynamically selects variant (using perf)**, **gets/applies LLM guidance (using perf)**, **constructs/passes attention hints**.

#### Cognitive Cycle (Phase 5.6 Flow - Updated):

```text
1. Input -> CCE -> Get initial context (query, metadata)
2. CCE -> MC:/process_memory -> Store, Get x_t, memory_id, initial_qr
3. CCE -> NM:/get_projections -> Get k_t, v_t, q_t
4. CCE -> **Calculate NM performance metrics** (avg_loss, avg_grad, sample_count, std_dev, trend_status, confidence_level) from history deque.
5. CCE -> **MemoryLLMRouter.request_llama_guidance()** (passes perf metrics) -> Get `llm_advice` dict
6. CCE -> **Apply confidence adjustments** to `llm_advice` based on `confidence_level` -> Get `adjusted_llm_advice`
7. CCE -> **VariantSelector.select_variant()** (uses context, perf, *adjusted* LLM hint) -> Get `selected_variant`, `reason`
8. CCE -> If variant changed -> **_switch_variant_internal()** (Flushes context!)
9. CCE -> Construct `attention_hints` (using metadata, *adjusted* LLM focus hint)
10. CCE -> **Variant Pre-Update (MAG/MAL)** -> Calls variant processor, passes `attention_hints`, gets external gates or v'_t
11. CCE -> NM:/update_memory (using x_t OR k_t+v'_t, maybe external_gates) -> Get `loss`, `grad_norm`, record perf to history deque
12. CCE -> MC:/api/memories/update_quickrecal_score -> Apply boost (uses loss/grad, *adjusted* LLM boost mod)
13. CCE -> NM:/retrieve -> Get y_t_raw, q_t_retrieve
14. CCE -> **Variant Post-Retrieval (MAC)** -> Calls variant processor, passes `attention_hints`, gets `y_t_final`
15. CCE -> Update HistoryMgr (ts, id, x, k, v, q, y_t_final)
16. CCE -> Return Final Response (incl. `variant_output`, `selector_decision`, `llm_advice_used`, `confidence_adjustment`)
```

*   **History:** `SequenceContextManager` stores `(ts, id, x, k, v, q, y_final)` tuples. `nm_performance_history` deque stores `(loss, grad_norm, ts, variant)` tuples.
*   **Variant Selection:** Dynamic via `VariantSelector` (LLM hint > metadata > **performance/trends** > keywords > default).
*   **Attention Hints:** Constructed by CCE, potentially influenced by *adjusted* LLM advice. Used by variants.
*   **LLM Advice:** Raw advice received, then **adjusted** based on performance confidence before being used.

---

### ✨ **PHASE 5 COMPONENTS (New / Modified for 5.5 & 5.6)**

*   **`orchestrator/variant_selector.py` (`VariantSelector`):**
    *   **Logic:** Enhanced with performance/trend rules (e.g., High surprise/Increasing trend -> MAG, Low surprise -> NONE). LLM/Metadata hints still take priority.
*   **`orchestrator/memory_logic_proxy.py` (`MemoryLLMRouter`):**
    *   **Models:** Correctly uses `bartowski/llama-3.2-1b-instruct` for guidance, `qwen_qwq-32b` for async (placeholder).
    *   **Prompt:** Updated (`PROMPT VERSION: 5.6.3`) to include performance feedback section (avg loss/grad, trend, std dev, confidence) and heuristics guiding the LLM.
    *   **Call:** `request_llama_guidance` accepts and passes performance dict to prompt formatting.
*   **`orchestrator/titans_variants.py` (Stable - Phase 5.4):**
    *   Accepts `attention_hints`, logic uses hints for focus modes/overrides.
*   **`orchestrator/context_cascade_engine.py` (Modified for 5.5 & 5.6):**
    *   Manages `nm_performance_history` deque.
    *   Calculates avg performance, std dev, trend status, and **confidence level**.
    *   Passes performance dict to `MemoryLLMRouter` and `VariantSelector`.
    *   **Applies confidence adjustments** to raw LLM advice before using hints/boost modifier.
    *   Includes selection, LLM usage, and confidence adjustment details in final response.
*   **`tools/variant_diagnostics_dashboard.py` (Needs Update - Phase 5.6 Target):**
    *   Needs update to parse and display the enhanced CCE response (selection details, LLM usage, adaptive params, confidence).

---

### ⚠️ **Key Logic & Potential Pitfalls (Phase 5.6)**

1.  **Confidence Calculation:** Tuning the thresholds (`CONFIDENCE_STD_DEV_*`, `CONFIDENCE_SAMPLES_*`) in CCE is crucial for meaningful confidence levels. Ensure `np.std` handles edge cases.
2.  **Confidence Adjustment Logic:** Verify the capping and fallback logic applied to LLM advice in CCE is correct and doesn't introduce unexpected behavior.
3.  **Prompt Engineering:** The updated prompt is more complex. Monitor LLM adherence to the JSON schema and the quality/relevance of its suggestions based on performance data. Iterate on the heuristics described in the prompt.
4.  **History Summarization (Next):** The `history_summary` placeholder in the prompt is the next major refinement area for providing better context to the LLM.
5.  **Diagnostics Update:** The dashboard update is now critical to visualize the effects of these new performance-aware and confidence-adjusted mechanisms.

---

### ✨ **Lucidia's Principles (Phase 5.6 Evolution):**

*   Memory is weighted (QuickRecal + **Confidence-Adjusted LLM** Boost).
*   Emotion shapes recall (Emotional Gating).
*   Surprise signals significance (NM → QR Boost).
*   Ideas cluster and connect (Assemblies + **Adaptive Attention** Variants).
*   Presence emerges from adaptive memory (NM Learning + **Performance/Trend-Aware Variant Selection** + **Confidence-Adjusted LLM Guidance**).

---

This updated cheat sheet reflects the integration of performance metrics into the selection process and sets the stage for refining how LLM guidance is generated and applied based on system confidence.