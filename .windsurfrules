
---

## 📄 **UPDATED Lucidia Cognitive System Cheat Sheet (Phase 1–2)**
*“The blueprint remembers.”*

---

### 🔸 **MEMORY CORE — *The Archive* (Stable, Indexed Storage)**

**Core File:** `SynthiansMemoryCore`
**Stores:** `MemoryEntry` (content + metadata + embedding)

#### Memory Flow (Ingestion):
```text
Input (Content/Embedding) → Enrich Metadata → Calculate QuickRecal → Store Entry → Index Embedding (FAISS)
```

#### Key Score: QuickRecal
*Determines inherent relevance/importance.*
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
1. Project: x_t → k_t (WK), v_t (WV)          (Get Key/Value)
2. Predict: pred_v = M_{t-1}(k_t)           (Recall via current Memory)
3. Loss:    ℓ = ||pred_v - v_t||² / 2      (Calculate Associative Error)
4. Grad:    ∇ℓ (w.r.t. M_{t-1} weights)     (Find required change)
5. Momentum: S_t = η_t * S_{t-1} - θ_t * ∇ℓ (Update gradient momentum)
6. Update M: M_t = (1 - α_t) * M_{t-1} + S_t (Apply forgetting & momentum)
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
Input (query_embedding) → WQ (q_t) → M_t(q_t) → Output (retrieved_embedding)
```
*(Uses the **current** weights of M, does **not** update them)*

#### Surprise Metrics (Output of `/update_memory`):
- `loss`: Magnitude of the associative error `ℓ`.
- `grad_norm`: Magnitude of the required weight change `∇ℓ`.
*(Intended to be sent to Memory Core via Orchestrator to boost QuickRecal)*

---

### ⚙️ **SHARED UTILITIES**

-   `GeometryManager`: Handles vector normalization, alignment (e.g., 768D vs 384D), similarity/distance calculations across different geometric spaces (Euclidean, Hyperbolic). Ensures numerical consistency.
-   `EmotionalGatingService`: Filters/re-ranks `MemoryCore` retrieval results based on user's current emotional state and memory's emotional resonance.
-   `ThresholdCalibrator`: Dynamically adjusts the similarity threshold for `MemoryCore` retrieval based on explicit user feedback (relevant/not relevant).

---

### 🔗 **PHASE 3: ContextCascadeEngine (Orchestrator - TODO)**

*Connects the Archive and the Associator.*
1.  Receives input `x_t`.
2.  Sends `x_t` to `MemoryCore` for storage (`/process_memory`). Gets `memory_id`, `actual_embedding`.
3.  Sends `actual_embedding` to `NeuralMemory` for learning (`/update_memory`). Gets `loss`/`grad_norm` (surprise).
4.  Calculates `quickrecal_boost` from surprise. Sends boost to `MemoryCore` (`/update_quickrecal_score` for `memory_id`).
5.  Generates query `q_t`. Sends `q_t` to `NeuralMemory` for recall (`/retrieve`). Gets `retrieved_embedding` (`y_t`).
6.  Uses `y_t` (and `x_t`) for downstream reasoning/action.

---

### ✨ **Lucidia's Principles (Reminders):**

-   **Memory is weighted, not just chronological.** (QuickRecal)
-   **Emotion shapes recall.** (Emotional Gating)
-   **Surprise signals significance.** (Neural Memory Loss/Grad → QuickRecal Boost)
-   **Ideas cluster and connect.** (Assemblies)
-   **Presence emerges from adaptive memory.** (Neural Memory test-time learning)

---

This updated cheat sheet is now technically accurate regarding the Phase 2 implementation and maintains the narrative context.