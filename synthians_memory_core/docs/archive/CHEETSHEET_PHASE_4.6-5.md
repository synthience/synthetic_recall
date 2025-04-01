
---

## 📄 **Synthians Cognitive System Cheat Sheet (Phase 4.6 Complete, Entering Phase 5)**

*“The blueprint remembers, the associator learns the flow, the cascade connects and adapts.”*

---

### 🔸 **MEMORY CORE — *The Archive* (Phase 4.6 Stable)**

**Core File:** `SynthiansMemoryCore` (`synthians_memory_core`)
**Stores:** `MemoryEntry` (content + metadata + embedding)

#### Memory Flow (Ingestion):

```text
Input (Content/Embedding) → Enrich Metadata → Calculate QuickRecal → Store Entry → Index Embedding (FAISS)
```

#### Key Score: QuickRecal

*   Determines inherent relevance/importance.
*   **Dynamically boosted** by surprise feedback from NM/CCE via `/api/memories/update_quickrecal_score`.
*   **Factors:** Recency, Emotion, Relevance, Importance, Surprise, Diversity, Coherence, Overlap (Penalty), Geometric Novelty (HPC-QR Mode), etc.

#### Key Metadata (Synthesized & Preserved):

*   **Standard:** `dominant_emotion`, `intensity`, `sentiment_value`, `timestamp_iso`, `time_of_day`, `day_of_week`, `embedding_dim`, `embedding_valid`, `complexity_estimate`, `word_count`, `source`, `user_id`, `session_id` (if provided), `uuid` (memory\_id).
*   **Feedback Loop:** `surprise_events` (list recording QR boosts: reason, delta, timestamp), `quickrecal_updated_at`.
*   *(Note: CCE's `variant_output` is part of the response, not typically stored in MC metadata unless explicitly passed).*

#### Assemblies:

*   Groups related `MemoryEntry` IDs via embedding similarity.
*   Contribute to retrieval context. Maintain composite embeddings.

---

### 🧠 **NEURAL MEMORY (NM) — *The Associator* (Phase 4.6 Stable)**

**Core File:** `NeuralMemoryModule` (`synthians_trainer_server`)
**Role:** Adaptive associative memory (learns `M(key) → value`) via test-time weight changes.
**Based On:** Titans paper principles.

#### Update Flow (Test-Time Memorization - `/update_memory`):

```text
1. Project: x_t → k_t (WK), v_t (WV)          (Can be overridden by MAL using external_k_t, external_v_t)
2. Predict: pred_v = M_{t-1}(k_t)           (Recall via current Memory M)
3. Loss:    ℓ = ||pred_v - v_t||² / 2      (Uses v_t or v'_t provided in request)
4. Grad:    ∇ℓ (w.r.t. M_{t-1} weights)
5. Momentum: S_t = η_t * S_{t-1} - θ_t * ∇ℓ (Gates α, θ, η use internal defaults or external values from MAG via request)
6. Update M: M_t = (1 - α_t) * M_{t-1} + S_t (Apply forgetting & momentum)
```

*   **Gates (α, θ, η):** Control Forget Rate, Inner LR, Momentum Decay. Can be modulated externally by MAG.
*   **M:** Internal MLP weights (dynamically updated).
*   **S:** Momentum state.

#### Retrieval Flow (Inference - `/retrieve`):

```text
Input (embedding x_t) → WQ (q_t) → M_t(q_t) → Output (retrieved_embedding y_t_raw)
```

*(Uses current `M` weights, does not update them).*

#### Surprise Metrics (Output of `/update_memory`):

*   `loss`: Magnitude of the associative error `ℓ`.
*   `grad_norm`: Magnitude of the required weight change `∇ℓ`.
*   *(Sent to CCE -> MC to boost QuickRecal)*.

#### Key Integration APIs (Used by CCE):

*   `POST /get_projections`: Returns `k_t, v_t, q_t` for input `x_t` (no update).
*   `POST /calculate_gates`: Takes `attention_output` (from CCE/MAG), returns calculated `alpha, theta, eta`.
*   `POST /update_memory`: Performs update. Accepts `input_embedding` (for standard/MAC/MAG) **OR** `key_projection` + `value_projection` (for MAL). Accepts optional `external_alpha_gate`, `external_theta_gate`, `external_eta_gate` (for MAG). Returns `loss`, `grad_norm`, projections/gates used.
*   `POST /retrieve`: Takes `input_embedding`, returns `retrieved_embedding` and `query_projection`.
*   `GET /config`: Returns NM config (dims, capabilities like gate/projection support).

---

### ⚙️ **Context Cascade Engine (CCE) — *The Orchestrator* (Phase 4.6 Stable)**

**Core File:** `ContextCascadeEngine` (`orchestrator`)
**Role:** Manages bi-directional flow (MC↔NM), implements core cognitive cycle, handles **Titans Variant Logic (MAC, MAG, MAL, NONE)**, integrates **Phase 5 adaptive layers**.

#### Cognitive Cycle (Phase 4.6 Refactored Flow):

```text
1. Input -> CCE -> MC:/process_memory -> Get x_t, memory_id, initial_qr
2. CCE -> NM:/get_projections -> Get k_t, v_t, q_t
3. CCE -> **[Phase 5: Call MemoryLLMRouter -> Get Advice (store?, tags, boost_mod, variant_hint, attention_hint)]**
4. CCE -> **[Phase 5: Call VariantSelector (using context, history, advice) -> Select Variant]**
5. CCE -> **[Phase 5: If variant changed -> _switch_variant_internal()]**
6. CCE -> **Variant Pre-Update (MAG/MAL)** -> Apply attention (using history, maybe hints), get external gates or v'_t
7. CCE -> NM:/update_memory (with variant mods) -> Get loss, grad_norm
8. CCE -> MC:/api/memories/update_quickrecal_score -> Apply boost (using loss/grad_norm, maybe advice['boost_score_mod'])
9. CCE -> NM:/retrieve -> Get y_t_raw, q_t_retrieve
10. CCE -> **Variant Post-Retrieval (MAC)** -> Apply attention (using history, maybe hints), calculate y_t_final
11. CCE -> Update HistoryMgr (ts, id, x, k, v, q, y_final)
12. CCE -> Return Final Response (structured `variant_output`, metrics, etc.)
```

*   **History:** `SequenceContextManager` stores `(ts, id, x, k, v, q, y_final)` tuples for attention.
*   **Variant Selection (Phase 4.6):** Static via `TITANS_VARIANT` env var.
*   **Variant Selection (Phase 5):** Dynamic via `VariantSelector` using context, performance, and LLM hints.

#### Variant Impact Summary (How Variants Modify the Cycle):

| Variant | Modifies                  | Target           | Timing         | Mechanism                                        |
| :------ | :------------------------ | :--------------- | :------------- | :----------------------------------------------- |
| **MAC** | Retrieval Output          | `y_t_raw`→`y_t_final` | Post-retrieval | `Attend(q_t, K_hist, Y_hist)` + Combine          |
| **MAG** | NM Update Gates           | `α, θ, η`        | Pre-update     | `Attend(q_t, K_hist, K_hist)`→`/calculate_gates` |
| **MAL** | NM Update Value           | `v_t` → `v'_t`     | Pre-update     | `Attend(q_t, K_hist, V_hist)`→ Combine→`/update_memory` |
| **NONE** | No Modification          | N/A              | N/A            | Base NM operations                               |

---

### ✨ **PHASE 5: Adaptive Reasoning & Selection (Current Focus)**

*   **Goal:** Enable dynamic, context-aware cognitive processing.
*   **Key Components & Integration:**
    *   **`orchestrator/variant_selector.py` (`VariantSelector`):**
        *   **Role:** Intelligently chooses the best Titan Variant (NONE, MAC, MAG, MAL) per request.
        *   **Inputs:** Task type (from query/metadata), NM performance history (loss/grad), LLM hints.
        *   **Integration:** Called by CCE (Step 4 in cycle). Triggers internal variant switching.
    *   **`orchestrator/memory_logic_proxy.py` (`MemoryLLMRouter`):**
        *   **Role:** Interfaces with external LLMs (via LM Studio) for nuanced memory operations guidance.
        *   **Models:** hugging-quants/llama-3.2-1b-instruct (real-time guidance), qwen2.5-0.5b-instruct (async "dream" tasks).
        *   **Integration:** Called by CCE (Step 3 in cycle). Provides `advice` dict (store decision, tags, boost modifier, variant/attention hints).
    *   **`tools/variant_diagnostics_dashboard.py`:**
        *   **Role:** Monitors CCE's `variant_output` metrics.
        *   **Integration:** Reads CCE responses (via polling or dedicated `/metrics/recent_cce_responses` endpoint).
    *   **Adaptive Attention Heuristics:**
        *   **Role:** CCE dynamically adjusts attention parameters.
        *   **Mechanism:** Modifies `SequenceContextManager` length; passes `attention_hints` to variant processors.

---

### 🛠️ **SHARED UTILITIES (Stable)**

*   `GeometryManager`: Vector ops (normalization, alignment, distance/similarity).
*   `EmotionalGatingService`: Filters/re-ranks MC retrieval based on emotion.
*   `ThresholdCalibrator`: Adapts MC retrieval threshold based on feedback.
*   `MetadataSynthesizer`: Enriches `MemoryEntry` metadata.

---

### ✨ **Lucidia's Principles (Evolving):**

*   Memory is weighted (QuickRecal + **LLM-guided** Boost).
*   Emotion shapes recall (Emotional Gating).
*   Surprise signals significance (NM → QR Boost).
*   Ideas cluster and connect (Assemblies + **Adaptive** Attention Variants).
*   Presence emerges from adaptive memory (NM Learning + **Dynamic Variant Selection** + **LLM Guidance**).

---