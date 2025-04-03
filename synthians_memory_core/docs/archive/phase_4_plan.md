## Phase 4: Implementing Titans Architecture Variants (MAC, MAG, MAL)

### Overview

This phase involves integrating attention mechanisms with the Neural Memory module, as described in Section 4 of the Titans paper, to enhance its capabilities.

**Phase 4 Goal:** To implement, integrate, and provide configuration options for the Memory-Attended Computation (MAC), Memory-Attended Gates (MAG), and Memory-Augmented Learning (MAL) variants.

**Prerequisites:**

1.  **Stable Phase 3:** Ensure the current codebase (post-Phase 3 fixes) is stable, committed, and tests are passing. The core loop (MemCore Store -> NeuralMem Update -> QuickRecal Boost -> NeuralMem Retrieve) must be reliable.
2.  **Confirm Configuration:** Verify the `NeuralMemoryConfig` (in `neural_memory.py` defaults and `http_server.py` startup) has `key_dim` and `query_dim` set correctly and *identically* (e.g., both 128).
3.  **Confirm QuickRecal Fix:** Double-check Memory Core logs to ensure the `update_quickrecal_score` endpoint is working correctly after the `get_memory_by_id`/`update_memory` fixes.
4.  **Understand Attention:** Familiarity with standard multi-head self-attention and cross-attention mechanisms (as implemented in TensorFlow/Keras or described in "Attention Is All You Need").
5.  **Review Titans Paper (Sec 4):** Re-read Section 4 and study the diagrams for MAC, MAG, and MAL to understand the data flow and where attention interacts.

**Architectural Decisions:**

1.  **Attention Module Location:** A new, reusable attention module (`attention.py`?) should be created within `synthians_trainer_server`.
2.  **Orchestration Location:** The `ContextCascadeEngine` (CCE) remains the central orchestrator. It will be responsible for:
    *   Maintaining necessary context/history for attention (e.g., recent keys, values, memory outputs).
    *   Calling the appropriate attention module based on the active variant.
    *   Modifying the data flow and calls to the `NeuralMemoryServer` according to the variant's logic.
3.  **Parameter Location:**
    *   Core attention parameters (projection matrices within the attention module) will be part of the attention module itself.
    *   Any *new* trainable parameters needed specifically for MAG (projecting attention output to gates) or MAL (gating/combining values) should ideally reside within the `NeuralMemoryModule` (as *outer* parameters) to keep related components together, but the CCE might need to trigger their calculation via new API endpoints or modified existing ones.
4.  **Configuration:** Introduce a new configuration setting (e.g., environment variable `TITANS_VARIANT` or a config file entry) read by the CCE to determine which variant (`NONE`, `MAC`, `MAG`, `MAL`) is active.

## Phase 4 Implementation Plan

**Step 1: Setup & Attention Core Module**

1.  **Branching:** Create a new feature branch (e.g., `feature/phase4-attention-variants`).
2.  **Configuration:**
    *   Define how the active variant (`NONE`, `MAC`, `MAG`, `MAL`) will be configured (e.g., add `TITANS_VARIANT` environment variable).
    *   Modify `ContextCascadeEngine.__init__` to read this configuration and store the active variant mode.
3.  **Create Attention Module (`synthians_trainer_server/attention.py`):**
    *   Implement a `MultiHeadAttentionModule` class using `tf.keras.layers.MultiHeadAttention`.
    *   Make it configurable (num_heads, key_dim, value_dim, dropout).
    *   Ensure it handles mask inputs if necessary (though likely not needed for these variants initially).
    *   Add basic unit tests for this module.
4.  **Context History in CCE:**
    *   Modify the `ContextCascadeEngine.sequence_context` list. Instead of just storing embeddings and IDs, ensure it stores the necessary tuples for attention based on potential future needs: `(timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)` where `x_t` is the input embedding, `k/v/q_t` are projections, and `y_t` is the output from `NeuralMemoryModule.call`.
    *   This requires adding `/get_projections` calls *during* the CCE's `process_new_input` flow (likely after getting `actual_embedding` from MemCore) *before* calling `/update_memory` and `/retrieve`, and storing these projections. Modify the `/update_memory` and `/retrieve` request/response cycle if needed to avoid redundant calculations. **Alternative:** Modify `/update_memory` and `/retrieve` responses to *return* the `k_t, v_t, q_t` they calculated internally. The latter is probably more efficient.
        *   **Decision:** Let's modify `/update_memory` and `/retrieve` to return the projections they compute.
        *   **Action:** Update `UpdateMemoryResponse` and `RetrieveResponse` models (and handlers in `http_server.py`) to include optional `key_projection`, `value_projection`, `query_projection` fields. Modify `NeuralMemoryModule.update_step` and `call` to potentially return these. Update CCE to store these in `sequence_context`.

**Step 2: Implement MAC (Memory-Attended Computation) Variant**

1.  **Modify CCE (`process_new_input`):**
    *   Add logic branch: `if self.active_variant == 'MAC':`.
    *   Inside this branch, *after* the call to `NeuralMemoryServer:/retrieve` which returns the raw memory output `y_t = M(q_t)` (and also `q_t` itself, based on Step 1 refinement):
        *   Retrieve recent history pairs `(k_i, y_i)` from `self.sequence_context`. Let `Y_hist = [y_i]` and `K_hist = [k_i]`.
        *   Instantiate or get the `MultiHeadAttentionModule`.
        *   Calculate attended output: `attended_y_t = AttentionModule(query=q_t, keys=K_hist, values=Y_hist)`.
        *   **Crucially:** Replace the raw `retrieved_embedding` in the `response` dictionary and potentially `self.last_retrieved_embedding` with this `attended_y_t`. This attended value is what downstream components will use.
2.  **Testing:**
    *   Add integration tests (e.g., modifying `lucidia_think_trace.py` or creating new tests) that activate MAC mode.
    *   Verify that the final `retrieved_embedding` differs from the raw output of `/retrieve` when history is present.
    *   Check logs for attention calculations.

**Step 3: Implement MAG (Memory-Attended Gates) Variant**

1.  **Modify `NeuralMemoryModule` (`neural_memory.py`):**
    *   Add new trainable layers (e.g., `Dense` layers) responsible for projecting the attention output to scalar gate logits. These layers belong to the *outer* parameters.
        ```python
        # In __init__
        self.attention_to_alpha = tf.keras.layers.Dense(1, name="att_alpha_proj", kernel_initializer=initializer_outer)
        self.attention_to_theta = tf.keras.layers.Dense(1, name="att_theta_proj", kernel_initializer=initializer_outer)
        self.attention_to_eta = tf.keras.layers.Dense(1, name="att_eta_proj", kernel_initializer=initializer_outer)
        # Add these layers' variables to outer_trainable_variables property
        ```
    *   Add a new method like `calculate_gates_from_attention(self, attention_output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]`:
        ```python
        def calculate_gates_from_attention(self, attention_output):
            alpha_logit = self.attention_to_alpha(attention_output)
            theta_logit = self.attention_to_theta(attention_output)
            eta_logit = self.attention_to_eta(attention_output)
            # Return scalar tensors (remove batch dim if present)
            return tf.squeeze(tf.sigmoid(alpha_logit)), tf.squeeze(tf.sigmoid(theta_logit)), tf.squeeze(tf.sigmoid(eta_logit))
        ```
    *   Modify `update_step`: Add optional arguments `alpha_t_ext=None, theta_t_ext=None, eta_t_ext=None`. If these arguments are provided (not None), use them instead of calculating gates from the internal `alpha_logit`, etc.
        ```python
        # Inside update_step
        alpha_t = tf.sigmoid(self.alpha_logit) if alpha_t_ext is None else alpha_t_ext
        theta_t = tf.sigmoid(self.theta_logit) if theta_t_ext is None else theta_t_ext
        eta_t = tf.sigmoid(self.eta_init) if eta_t_ext is None else eta_t_ext # Corrected: Use eta_logit
        # eta_t = tf.sigmoid(self.eta_logit) if eta_t_ext is None else eta_t_ext # <-- Corrected Line
        ```
2.  **Modify Neural Memory Server API (`http_server.py`):**
    *   Add a new endpoint `/calculate_gates` (POST) that takes an `attention_output` vector and returns the calculated `alpha_t, theta_t, eta_t` by calling `nm.calculate_gates_from_attention`.
    *   Modify `UpdateMemoryRequest` to include optional `alpha_t`, `theta_t`, `eta_t` fields.
    *   Modify the `/update_memory` handler to pass these external gates to `nm.update_step` if they are present in the request.
3.  **Modify CCE (`process_new_input`):**
    *   Add logic branch: `if self.active_variant == 'MAG':`.
    *   Inside this branch, *before* calling `/update_memory`:
        *   Get `q_t` (either from the `/process_memory` response via Memory Core call if we modify that, or by calling `/get_projections` on NeuralMem). Let's assume we get it along with `k_t` from the initial processing step.
        *   Retrieve recent history keys `K_hist = [k_i]` from `self.sequence_context`.
        *   Instantiate or get the `MultiHeadAttentionModule`.
        *   Calculate attention output: `attention_output = AttentionModule(query=q_t, keys=K_hist, values=K_hist)` (Attending query to past keys).
        *   Call the *new* `NeuralMemoryServer:/calculate_gates` endpoint with `attention_output`.
        *   Receive `alpha_t, theta_t, eta_t` from the response.
        *   Modify the payload for the *subsequent* `/update_memory` call to include these calculated gates (`alpha_t`, `theta_t`, `eta_t`).
4.  **Outer Loop Training (`NeuralMemoryModule.train_step`):** Ensure the gradients flow back through the new gate projection layers (`attention_to_alpha`, etc.) when calculating `outer_grads`.
5.  **Testing:** Add integration tests for MAG mode. Verify that gate values passed externally influence the update step. Check gradients for the new layers.

**Step 4: Implement MAL (Memory-Augmented Learning) Variant**

1.  **Modify `NeuralMemoryModule` (`neural_memory.py`):**
    *   Modify `update_step`: Instead of calculating `k_t, v_t` from `x_t` internally, change the method signature to accept `k_t` and `v_prime_t` directly: `update_step(self, k_t: tf.Tensor, v_prime_t: tf.Tensor)`. Update the loss calculation to use `v_prime_t`: `loss = 0.5 * tf.reduce_sum(tf.square(predicted_v_t - v_prime_t))`. Remove the `get_projections` call from within `update_step`.
2.  **Modify Neural Memory Server API (`http_server.py`):**
    *   Modify `UpdateMemoryRequest`: Change `input_embedding` to `key_projection: List[float]` and `value_projection: List[float]` (representing `k_t` and `v'_t`).
    *   Modify the `/update_memory` handler:
        *   Validate `key_projection` against `key_dim` and `value_projection` against `value_dim`.
        *   Convert them to tensors.
        *   Call `nm.update_step(k_tensor, v_prime_tensor)`.
3.  **Modify CCE (`process_new_input`):**
    *   Add logic branch: `if self.active_variant == 'MAL':`.
    *   Inside this branch, *before* calling `/update_memory`:
        *   Get `k_t, v_t, q_t` for the current input `x_t` (e.g., via `/get_projections` or from refined response).
        *   Retrieve recent history pairs `(k_i, v_i)` from `self.sequence_context`. Let `K_hist = [k_i]` and `V_hist = [v_i]`.
        *   Instantiate or get the `MultiHeadAttentionModule`.
        *   Calculate attention output: `attended_v_t = AttentionModule(query=q_t, keys=K_hist, values=V_hist)`.
        *   Combine `attended_v_t` with the current `v_t` to get `v_prime_t`. (Start with simple addition: `v_prime_t = v_t + attended_v_t`. Later, this could be a learned gating mechanism requiring new outer parameters).
        *   Modify the payload for the `/update_memory` call to send `key_projection=k_t` and `value_projection=v_prime_t`.
4.  **Testing:** Add integration tests for MAL mode. Verify that the `v_prime_t` calculated in CCE is correctly used in the Neural Memory's loss calculation.

**Step 5: Refinement, Integration Testing & Benchmarking**

1.  **Code Review & Refactoring:** Clean up the CCE logic, ensure efficient history management, and refine error handling.
2.  **Configuration Testing:** Test switching between `NONE`, `MAC`, `MAG`, `MAL` modes using the configuration mechanism.
3.  **Comprehensive Integration Tests:** Create tests simulating longer sequences and verifying the distinct behaviors of each variant. Use `lucidia_think_trace.py` extensively.
4.  **(Optional/Future) Benchmarking:** If specific tasks (like those in the Titans paper) are defined, implement the necessary outer loop training (`/train_outer`) adjustments for each variant and benchmark performance on evaluation datasets. This is a significant undertaking beyond the core implementation.

**Step 6: Documentation**

1.  **Update `README.md` / `NEWEST-DOCUMENTATION.md`:** Reflect the completion of Phase 4 and the availability of the variants.
2.  **Update `architecture_overview.md` / `bihemispheric_architecture.md`:** Add descriptions and potentially diagrams illustrating the data flow for MAC, MAG, MAL.
3.  **Update `api_reference.md`:** Document any changes to the Neural Memory Server endpoints (e.g., `/calculate_gates`, modified `/update_memory` payload).
4.  **Create `attention.md`:** Document the `MultiHeadAttentionModule`.
5.  **Update `implementation_guide.md`:** Explain how to configure and use the different Titans variants.

This plan provides a structured approach to implementing the attention-based variants, focusing on modifying the CCE and the Neural Memory API/Module iteratively for each variant. Remember to test thoroughly at each step.