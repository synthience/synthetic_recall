
---
## 📄 **UPDATED Lucidia Cognitive System Cheat Sheet (Phase 1–3 Complete, Entering Phase 4)**
*“The blueprint remembers, the associator learns the flow, the cascade connects.”*

---

### 🔸 **MEMORY CORE — *The Archive* (Stable, Indexed Storage)**

**Core File:** `SynthiansMemoryCore` (`synthians_memory_core`)
**Stores:** `MemoryEntry` (content + metadata + embedding)

#### Memory Flow (Ingestion):
```text
Input (Content/Embedding) → Enrich Metadata → Calculate QuickRecal → Store Entry → Index Embedding (FAISS)
```

#### Key Score: QuickRecal
*Determines inherent relevance/importance. **Dynamically boosted by surprise.** *
```text
QuickRecal = Function of factors including:
  - Relevance (e.g., to query), Recency, Emotion
  - Importance (explicit/inferred), Personal Context
  - Surprise (via Neural Memory feedback), Diversity, Coherence
  - Overlap (Penalty)
  - Geometric/Causal Novelty (Mode Dependent, e.g., HPC-QR)
```

#### Key Metadata (Synthesized & Preserved):
```text
(Includes time, emotion, complexity, embedding stats; preserves source/IDs if provided)
- dominant_emotion, sentiment_value, intensity
- timestamp_iso, time_of_day, day_of_week, etc.
- embedding_dim, embedding_valid, etc.
- complexity_estimate, word_count
- source, user_id, session_id (if input)
- uuid (memory_id)
- surprise_events: Records QuickRecal boosts from NM surprise (reason, delta, scores).
- **variant_used**: (Phase 4+) Name of the Titans variant used during processing.
- **surprise_boost_applied**: (Phase 4+) The calculated boost amount applied.
- **attention_trace_id**: (Phase 4+, Optional) ID linking to detailed attention metrics.
```

#### Assemblies:
- Groups of related `MemoryEntry` IDs.
- Dynamically updated based on embedding similarity.
- Contribute to retrieval via activation scoring.
- Hold composite embeddings representing the cluster's theme.

---

### 🧠 **NEURAL MEMORY — *The Associator* (Adaptive, Test-Time Learner)**

**Core File:** `NeuralMemoryModule` (`synthians_trainer_server`)
**Learns:** Associative mappings `M(key) → value` via weight changes.
**Supports:** Continuous adaptation during operation.

#### Update Flow (Test-Time Memorization - `/update_memory`):
```text
1. Project: x_t → k_t (WK), v_t (WV)          (Get Key/Value - Can be overridden by MAL)
2. Predict: pred_v = M_{t-1}(k_t)           (Recall via current Memory)
3. Loss:    ℓ = ||pred_v - v_t||² / 2      (Calculate Associative Error - Uses v_t or v'_t from MAL)
4. Grad:    ∇ℓ (w.r.t. M_{t-1} weights)     (Find required change)
5. Momentum: S_t = η_t * S_{t-1} - θ_t * ∇ℓ (Update gradient momentum - Gates can be overridden by MAG)
6. Update M: M_t = (1 - α_t) * M_{t-1} + S_t (Apply forgetting & momentum - Gates can be overridden by MAG)
```
```text
- α_t: Forget Rate Gate (0=keep all, 1=forget all)
- θ_t: Inner Learning Rate Gate (scales gradient influence)
- η_t: Momentum Decay Gate (controls persistence of past gradients)
- M: Neural weights of the internal memory MLP (These *change*)
- S: Gradient momentum state (tracks update direction)
```

#### Retrieval Flow (Inference - `/retrieve`):
```text
Input (query_embedding) → WQ (q_t) → M_t(q_t) → Output (retrieved_embedding y_t_raw)
```
*(Uses the **current** weights of M, does **not** update them)*

#### Surprise Metrics (Output of `/update_memory`):
- `loss`: Magnitude of the associative error `ℓ`.
- `grad_norm`: Magnitude of the required weight change `∇ℓ`.
*(Sent to Memory Core via CCE to boost QuickRecal)*

#### API Endpoints for Phase 4:
-   `POST /get_projections`: Returns `k_t, v_t, q_t` without updating `M`.
-   `POST /calculate_gates`: Takes `attention_output` (from CCE), returns `alpha, theta, eta` (for MAG).
-   `GET/POST /config`: Returns NM config details (dims, capabilities).

---

### ⚙️ **Context Cascade Engine (CCE) — *The Orchestrator* (Phase 3 Complete)**

**Core File:** `ContextCascadeEngine` (`orchestrator`)
**Role:** Manages the bi-directional flow between Memory Core and Neural Memory. Implements the core cognitive cycle and **variant-specific logic**.

#### Refactored Cognitive Cycle (Phase 3 Functional Flow):
```text
1. Input -> CCE -> MC:/process_memory -> Get x_t, memory_id, initial_qr
2. CCE -> NM:/get_projections -> Get k_t, v_t, q_t
3. CCE -> **Variant Pre-Update (MAG/MAL)** -> Apply attention, calculate external gates or v'_t
4. CCE -> NM:/update_memory (with variant mods) -> Get loss, grad_norm
5. CCE -> MC:/api/memories/update_quickrecal_score -> Apply boost from loss/grad_norm
6. CCE -> NM:/retrieve -> Get y_t_raw, q_t_retrieve
7. CCE -> **Variant Post-Retrieval (MAC)** -> Apply attention, calculate y_t_final
8. CCE -> Update HistoryMgr (ts, id, x, k, v, q, y_final) -> Store context for future attention
9. CCE -> Return Final Response (including y_t_final, metrics, **variant_used**)
```
-   **History:** Uses `SequenceContextManager` to store `(ts, id, x, k, v, q, y)` tuples for attention.
-   **Variant Selection:** Reads `TITANS_VARIANT` environment variable (`NONE`, `MAC`, `MAG`, `MAL`).

#### **Variant Flow Diagram (Phase 4):**
```mermaid
graph TD
    Input[Input: x_t] --> MCStore(MC:/process_memory)
    MCStore --> |x_t, mem_id, qr| NMProj(NM:/get_projections)
    NMProj --> |k_t, v_t, q_t| PreUpdate{Variant Pre-Update?}
    PreUpdate -- No (NONE/MAC) --> NMUpdate(NM:/update_memory)
    PreUpdate -- Yes (MAG/MAL) --> CalcVariant{Calc Gates (MAG) or v'_t (MAL)}
    CalcVariant --> NMUpdate
    NMUpdate --> |loss, grad_norm| MCBoost(MC:/update_quickrecal_score)
    NMUpdate --> NMRetrieve(NM:/retrieve)
    NMRetrieve --> |y_t_raw, q_t| PostRetrieve{Variant Post-Retrieval?}
    PostRetrieve -- No (NONE/MAG/MAL) --> FinalOutput[y_t_final = y_t_raw]
    PostRetrieve -- Yes (MAC) --> CalcMAC{Calc attended_y_t}
    CalcMAC --> FinalOutputMAC[y_t_final = attended_y_t]
    MCBoost --> HistoryUpdate(Update HistoryMgr)
    FinalOutput --> HistoryUpdate
    FinalOutputMAC --> HistoryUpdate
    HistoryUpdate --> Output(Return Response)
```

---

### ✨ **PHASE 4: Titans Variants (Current Focus)**

*Integrates Attention mechanisms into the CCE flow.*

#### **Variant Impact Summary:**

| Variant | Affects | Target | Timing | Mechanism |
|--------|---------|--------|--------|-----------|
| **MAC** | Retrieval Output | `y_t` → `y_t_final` | Post-retrieval | `Attend(q_t, K_hist, Y_hist)` + Combine |
| **MAG** | Learning Gates | `α, θ, η` | Pre-update | `Attend(q_t, K_hist, K_hist)` -> `/calculate_gates` |
| **MAL** | Stored Value | `v_t` → `v'_t` | Pre-update | `Attend(q_t, K_hist, V_hist)` + Combine |

1.  **MAC (Memory-Attended Computation):**
    *   **Goal:** Enhance retrieval output `y_t`.
    *   **Mechanism:** Uses attention `Attend(q_t, K_hist, Y_hist)` *after* NM `/retrieve` to combine historical outputs (`Y_hist`) with raw retrieval (`y_t_raw`) -> `y_t_final`.
2.  **MAG (Memory-Attended Gates):**
    *   **Goal:** Dynamically modulate NM learning.
    *   **Mechanism:** Uses attention `Attend(q_t, K_hist, K_hist)` *before* NM `/update_memory`. Calls NM `/calculate_gates` with attention output. Sends external gates (`alpha_t`, `theta_t`, `eta_t`) in `/update_memory` request.
3.  **MAL (Memory-Augmented Learning):**
    *   **Goal:** Enhance what gets stored in NM.
    *   **Mechanism:** Uses attention `Attend(q_t, K_hist, V_hist)` *before* NM `/update_memory`. Combines original `v_t` with attended value to get `v'_t`. Sends `k_t` and `v'_t` explicitly in `/update_memory` request payload.

---

### 🛠️ **SHARED UTILITIES**

-   `GeometryManager`: Handles vector normalization, alignment (e.g., 768D vs 384D), similarity/distance calculations across different geometric spaces (Euclidean, Hyperbolic). Ensures numerical consistency.
-   `EmotionalGatingService`: Filters/re-ranks `MemoryCore` retrieval results based on user's current emotional state and memory's emotional resonance.
-   `ThresholdCalibrator`: Dynamically adjusts the similarity threshold for `MemoryCore` retrieval based on explicit user feedback (relevant/not relevant).

---

### ✨ **Lucidia's Principles (Reminders):**

-   **Memory is weighted, not just chronological.** (QuickRecal + Surprise Boost)
-   **Emotion shapes recall.** (Emotional Gating)
-   **Surprise signals significance.** (NM Loss/Grad → QuickRecal Boost)
-   **Ideas cluster and connect.** (Assemblies + **Titans Attention Variants**)
-   **Presence emerges from adaptive memory.** (NM Test-Time Learning + **Contextual Adaptation via Variants**)

---

This version incorporates the diagram, table, and metadata suggestions. It feels even more comprehensive and directly maps the concepts to the implementation flow.

