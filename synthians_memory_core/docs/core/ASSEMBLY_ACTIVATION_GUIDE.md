
---

# **Deep Dive Guide: Memory Assembly Activation & Boosting (Phase 5.8)**

## 1. Overview

This guide provides a technical deep dive into the Memory Assembly activation and retrieval boosting mechanism within the `SynthiansMemoryCore`. It explains the workflow, key code implementations, configuration parameters, and synchronization logic established in Phase 5.8 to ensure stable and effective contextual memory retrieval. Understanding this process is crucial for debugging retrieval relevance issues and extending assembly functionalities.

## 2. Core Concepts

*   **Memory Assembly:** A collection (`MemoryAssembly` object) of related `MemoryEntry` IDs, possessing a `composite_embedding` that represents the semantic center of its members.
*   **Vector Index Synchronization:** An assembly's `composite_embedding` must be present and up-to-date in the `MemoryVectorIndex` (FAISS) to be considered "synchronized". This is tracked by the `MemoryAssembly.vector_index_updated_at` timestamp.
*   **Assembly Activation:** During memory retrieval, assemblies whose synchronized `composite_embedding` is sufficiently similar to the `query_embedding` are "activated". Their activation level corresponds to this similarity score.
*   **Retrieval Boosting:** Memories belonging to activated, synchronized assemblies receive a boost to their base relevance score, making them more likely to appear higher in the final retrieval results.

## 3. Activation & Boosting Workflow in `retrieve_memories`

The process involves several key methods within `SynthiansMemoryCore`:

1.  **Query Embedding:** The input `query` string is converted into a validated `query_embedding` (numpy array) using `generate_embedding` and `geometry_manager`.
2.  **Candidate Generation (`_get_candidate_memories`):**
    *   **Assembly Activation Call:** This method first calls `_activate_assemblies(query_embedding)`.
    *   **Direct Memory Search:** It then performs a direct search in the `vector_index` for individual memory embeddings (`mem:*` IDs) similar to the `query_embedding`.
    *   **Candidate Loading:** It combines memory IDs from activated assemblies and direct search results, loads the full memory data (as dicts), and calculates the *base* `similarity` for each memory found in the direct search.
    *   **Return:** It returns a tuple: `(list_of_memory_dicts, assembly_activation_scores_dict)`.
3.  **Boost Calculation & Application (`retrieve_memories`):**
    *   The main `retrieve_memories` method iterates through the candidate `memory_dicts`.
    *   For each memory, it checks which assemblies it belongs to (using the `memory_to_assemblies` mapping).
    *   It finds the maximum activation score among the *associated, activated* assemblies (using the `assembly_activation_scores_dict`).
    *   It calculates the `assembly_boost` based on this `max_activation`, the configured `assembly_boost_factor`, and `assembly_boost_mode`.
    *   It adds the `assembly_boost` to the memory's base `similarity` to get the final `relevance_score` (clamped between 0.0 and 1.0).
    *   It stores diagnostic information (`boost_info`) in the memory dictionary.
4.  **Filtering & Ranking:** Candidates are filtered based on the `threshold` (using `similarity`, *not* the boosted score), emotional gating, and metadata filters.
5.  **Final Ranking:** The remaining candidates are sorted based on their final `relevance_score` (which includes the assembly boost).
6.  **Return:** The top `k` results are returned.

## 4. Code Implementation Details

### 4.1 `_activate_assemblies`

This asynchronous method is central to finding relevant assemblies.

```python
# synthians_memory_core/synthians_memory_core.py

async def _activate_assemblies(self, query_embedding: np.ndarray) -> List[Tuple[MemoryAssembly, float]]:
    """Find and activate assemblies based on query similarity."""
    # ... (Input validation: query_embedding, vector_index check) ...
    logger.debug(f"[Assembly Debug] Query embedding shape: {query_embedding.shape}...")

    # --- Get Config ---
    assembly_threshold = self.config.get('assembly_threshold', 0.1) # Lowered default
    drift_limit = self.config.get('max_allowed_drift_seconds', 3600)
    enable_sync = self.config.get('enable_assembly_sync', True)
    logger.debug(f"[Assembly Debug] Configs: Threshold={assembly_threshold}, DriftLimit={drift_limit}s, SyncEnabled={enable_sync}")

    activated_assemblies = []
    now_utc = datetime.now(timezone.utc) # Use timezone-aware datetime

    try:
        # --- Search Vector Index ---
        # No id_prefix needed, search includes both mem:* and asm:*
        search_results = await self.vector_index.search_async(
            query_embedding,
            k=200 # Search broadly
        )

        # --- Filter for Assemblies & Check Sync ---
        asm_results = [r for r in search_results if r[0].startswith("asm:")]
        logger.debug(f"[Assembly Debug] Found {len(asm_results)} potential assemblies after filtering prefix.")

        for asm_id_with_prefix, similarity in asm_results:
            # 1. Similarity Threshold Check
            if similarity < assembly_threshold:
                logger.debug(f"[ACTIVATE_DBG] Skipping '{asm_id_with_prefix}': similarity {similarity:.4f} < {assembly_threshold}")
                continue

            # 2. Extract ID & Look up Assembly Object
            assembly_id = asm_id_with_prefix[4:]
            assembly = self.assemblies.get(assembly_id) # Use in-memory dict
            if assembly is None:
                logger.warning(f"[ACTIVATE_DBG] Assembly '{assembly_id}' found in index but not in memory dict. Skipping.")
                continue

            # 3. Synchronization Check (if enabled)
            if enable_sync:
                updated_at = assembly.vector_index_updated_at
                logger.debug(f"[ACTIVATE_DBG] Checking sync for '{assembly_id}': updated_at={updated_at}")
                if updated_at is None:
                    logger.debug(f"[ACTIVATE_DBG] Skipping '{assembly_id}': updated_at is None.")
                    continue

                # Ensure updated_at is timezone-aware for comparison
                if updated_at.tzinfo is None:
                     updated_at = updated_at.replace(tzinfo=timezone.utc)

                drift_seconds = (now_utc - updated_at).total_seconds()
                logger.debug(f"[ACTIVATE_DBG] Checking drift for '{assembly_id}': drift={drift_seconds:.2f}s, limit={drift_limit}s")
                if drift_seconds > drift_limit:
                     logger.debug(f"[ACTIVATE_DBG] Skipping '{assembly_id}': Drift limit exceeded.")
                     continue
            else:
                 logger.debug(f"[ACTIVATE_DBG] Sync check disabled for '{assembly_id}'.")


            # 4. Activation Success
            logger.info(f"[ACTIVATE_DBG] ACTIVATE SUCCESS for '{assembly_id}' with similarity {similarity:.4f}")
            activated_assemblies.append((assembly, similarity))
            # Note: Actual activation level update (assembly.activate()) happens
            # in the main retrieve_memories flow AFTER filtering, to avoid
            # modifying state during the search phase here.

        # Sort by similarity before returning
        activated_assemblies.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"[Assembly Debug] Total activated assemblies passing checks: {len(activated_assemblies)}")
        return activated_assemblies

    except Exception as e:
        logger.error(f"Error during assembly activation: {str(e)}", exc_info=True)
        return []
```

**Key Logic Points:**

*   Searches the *entire* index using `search_async`.
*   Filters results client-side for IDs starting with `"asm:"`.
*   Applies the `assembly_threshold` to the similarity score.
*   Retrieves the `MemoryAssembly` object from the *in-memory* `self.assemblies` dictionary.
*   Checks `vector_index_updated_at`: If `None` or too old compared to `max_allowed_drift_seconds`, the assembly is skipped.

### 4.2 `retrieve_memories` (Boost Calculation)

This snippet shows the core boost logic within the main retrieval function.

```python
# synthians_memory_core/synthians_memory_core.py

async def retrieve_memories(...):
    # ... (previous steps: query embedding, get candidates & activation scores) ...

    candidates, assembly_activation_scores = await self._get_candidate_memories(query_embedding_np, top_k * 5)

    boost_mode = self.config.get('assembly_boost_mode', 'linear')
    boost_factor = self.config.get('assembly_boost_factor', 0.2)
    scored_candidates = []

    logger.debug(f"Applying assembly boost (Mode: {boost_mode}, Factor: {boost_factor}) to {len(candidates)} candidates...")

    for memory_dict in candidates:
        similarity = memory_dict.get("similarity", 0.0) # Base similarity from direct search
        assembly_boost = 0.0
        max_activation = 0.0
        boost_reason = "none"
        mem_id = memory_dict.get("id")
        associated_assembly_ids = set()

        # Get associated assemblies under lock
        async with self._lock:
            associated_assembly_ids = self.memory_to_assemblies.get(mem_id, set())

        if associated_assembly_ids:
            # Find the highest activation score among the *activated* assemblies for this memory
            max_activation = max(
                (assembly_activation_scores.get(asm_id, 0.0) for asm_id in associated_assembly_ids),
                default=0.0
            )

            if max_activation > 0:
                # --- Calculate Boost ---
                if boost_mode == "linear":
                    assembly_boost = max_activation * boost_factor
                    boost_reason = f"linear(act:{max_activation:.2f}*f:{boost_factor:.2f})"
                # ... (other boost modes) ...
                else: # Default additive
                    assembly_boost = max_activation * boost_factor
                    boost_reason = f"default_linear(act:{max_activation:.2f}*f:{boost_factor:.2f})"

                # Clamp boost to prevent score > 1.0
                assembly_boost = min(assembly_boost, max(0.0, 1.0 - similarity))
                memory_dict["relevance_score"] = min(1.0, similarity + assembly_boost)
                logger.debug(f"Memory {mem_id[:8]} Boost Applied: +{assembly_boost:.4f} (Act: {max_activation:.3f}) -> New Score: {memory_dict['relevance_score']:.4f}")
            else:
                boost_reason = "no_activated_assemblies" # Associated assemblies weren't activated enough or were unsynced
                memory_dict["relevance_score"] = similarity # No boost applied
        else:
            boost_reason = "no_associated_assemblies"
            memory_dict["relevance_score"] = similarity # No boost applied

        # Store diagnostic info
        memory_dict["boost_info"] = {
            "base_similarity": float(similarity),
            "assembly_boost": float(assembly_boost),
            "max_activation": float(max_activation),
            "boost_reason": boost_reason
        }
        scored_candidates.append(memory_dict)

    # --- Filtering Steps ---
    # Filter based on original similarity >= threshold_to_use
    # ...
    # Apply emotional gating
    # ...
    # Apply metadata filtering
    # ...

    # --- Final Sort & Return ---
    # Sort by the final relevance_score (which includes the boost)
    filtered_candidates.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    final_memories = filtered_candidates[:top_k]
    # ... (return final_memories) ...
```

**Key Logic Points:**

*   Retrieves pre-calculated `assembly_activation_scores` from `_get_candidate_memories`.
*   Looks up assemblies associated with *each candidate memory*.
*   Finds the `max_activation` score *only* among those associated assemblies that were actually activated (present in the `assembly_activation_scores` dict).
*   Calculates `assembly_boost` based on `max_activation` and config.
*   Adds boost to `similarity` to get `relevance_score`.
*   Stores `boost_info` for diagnostics.
*   **Important:** Filtering (threshold, emotion, metadata) happens *after* boost calculation but typically uses the *original* `similarity` for thresholding, while the final ranking uses the *boosted* `relevance_score`.

## 5. Configuration Parameters

*   **`assembly_threshold`** (`float`, default: 0.1): Minimum similarity score between a query embedding and an assembly's composite embedding for the assembly to be considered "activated".
*   **`enable_assembly_sync`** (`bool`, default: True): If True, only assemblies with a recent `vector_index_updated_at` timestamp are considered during activation.
*   **`max_allowed_drift_seconds`** (`int`, default: 3600): Maximum age (in seconds) of the `vector_index_updated_at` timestamp for an assembly to be considered synchronized.
*   **`assembly_boost_factor`** (`float`, default: 0.2): A multiplier controlling the strength of the boost applied based on the assembly's activation level. A higher factor means stronger boosting.
*   **`assembly_boost_mode`** (`str`, default: "linear"): How the boost is calculated.
    *   `linear` (or `additive`): `boost = activation_level * boost_factor`. Boost is added to base similarity.
    *   `multiplicative`: `boost = base_similarity * activation_level * boost_factor`. Boost is proportional to base similarity.

## 6. Synchronization & Drift (`vector_index_updated_at`)

*   **Purpose:** Prevents boosting based on stale assembly embeddings in the vector index. When an assembly's members change, its `composite_embedding` is recalculated, but the vector index update might fail or be delayed (handled by the retry queue).
*   **Mechanism:**
    *   `MemoryAssembly.add_memory()` recalculates the composite embedding but *does not* set the timestamp.
    *   An external process (like `_update_assemblies` or the `AssemblySyncManager`) attempts to update the vector index using `vector_index.add_async` or `update_entry_async`.
    *   *Only upon successful completion* of the vector index update, the `vector_index_updated_at` timestamp on the `MemoryAssembly` object is set to `datetime.now(timezone.utc)`.
    *   `_activate_assemblies` checks this timestamp against `max_allowed_drift_seconds` before considering an assembly for activation.

## 7. Debugging Common Issues

*   **Boost is 0.0 / Reason is 'no_activated_assemblies':**
    1.  **Check `_activate_assemblies` Logs:** Add `[ACTIVATE_DBG]` logs as shown above.
    2.  **Is the assembly embedding in the index?** Verify `vector_index.get_stats()` shows IDs starting with `asm:`. Ensure the test/setup code explicitly adds assembly embeddings using `vector_index.add_async(f"asm:{assembly.assembly_id}", assembly.composite_embedding)`.
    3.  **Is the similarity too low?** Check the logged similarity against `assembly_threshold`. Temporarily lower the threshold for testing.
    4.  **Is the assembly synchronized?** Check the `vector_index_updated_at` timestamp log in `_activate_assemblies`. Is it `None`? Is the drift exceeding `max_allowed_drift_seconds`? Ensure the code updating the index actually sets the timestamp upon success.
    5.  **Is the assembly lookup failing?** Check the `Assembly '{id}' present in self.assemblies?` log. Ensure the ID extracted from the index result matches the key in `memory_core.assemblies`.
*   **Boost seems too high/low:**
    1.  Check the `assembly_boost_factor` configuration.
    2.  Check the `max_activation` value logged in `boost_info`. Is the assembly activation level unexpectedly high/low?
    3.  Verify the `boost_mode` and calculation logic in `retrieve_memories`.

## 8. Future Improvements

*   More sophisticated boost calculation models (e.g., considering assembly size, coherence).
*   Dynamically adjusting `assembly_boost_factor` based on context or performance.
*   Visualizing assembly activation patterns and boost impact in the diagnostics dashboard.
*   Using assembly topic tags (future metadata) to influence activation or boosting.

---