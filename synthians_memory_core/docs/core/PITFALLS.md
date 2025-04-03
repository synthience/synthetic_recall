# ⚠️ Phase 5.8 – Common Pitfalls to Avoid
*A stability-first guide for implementers, reviewers, and automated systems.*

This guide outlines critical pitfalls uncovered during the implementation and debugging of Phase 5.8. It should be reviewed and referenced **before** attempting to modify or extend `MemoryVectorIndex`, `SynthiansMemoryCore`, or `MemoryAssembly`.

---

## 📌 Section I: Vector Index Consistency & Stability

### 1. Desynchronized Index and Mapping
- **Issue:** `faiss_index.bin` and `mapping.json` fall out of sync.
- **Symptoms:** `FAISS ntotal: X`, `mapping count: Y`, search returns unknown IDs.
- **Avoid by:**
  - Always updating `id_to_index` in lockstep with FAISS vector additions/removals.
  - Verifying persistence success in `save()`, not assuming.

---

### 2. Faulty Index Reset on Load
- **Issue:** `load()` resets the index if `faiss_index.bin` is missing, even if `mapping.json` is valid.
- **Avoid by:**
  - Only resetting when both files are missing or verified as corrupt.
  - Preserving `id_to_index` if partial recovery is viable.

---

### 3. Verification Method With Side Effects
- **Issue:** `verify_index_integrity()` mutates state (e.g., resets the index).
- **Avoid by:**
  - Making all verification routines **purely diagnostic**.
  - Moving side-effecting logic to a clearly named `repair_index()`.

---

### 4. Silent Persistence Failures
- **Issue:** `save()` doesn’t verify index file integrity or catch partial writes.
- **Avoid by:**
  - Checking file size > 0.
  - Logging or failing hard on any write error.
  - Writing to a temp file and only replacing on success.

---

## 📌 Section II: Core Logic Integration

### 5. Unchecked Index Write Failures
- **Issue:** `process_new_memory()` continues even if `vector_index.add()` fails.
- **Avoid by:**
  - Checking return value and logging failures.
  - Marking memory as "not searchable" if index update fails.

---

### 6. Repair Trigger Bugs
- **Issue:** Auto-repair on init doesn't always fire for partial mismatches.
- **Avoid by:**
  - Expanding mismatch detection conditions (e.g., `0 vs >5`, `mapping > 0 but index = 0`).
  - Clearly documenting what triggers automatic repair.

---

## 📌 Section III: Assembly Persistence

### 7. Invalid JSON Format or Schema Drift
- **Issue:** Old or manually created assemblies missing keys (`memories`, `embedding`).
- **Avoid by:**
  - Adding a `schema_version` check in `from_dict()`.
  - Providing a migration script or fallback parser for legacy formats.

---

### 8. `to_dict()` Errors on `None` Embeddings
- **Issue:** `save_assembly()` crashes when `composite_embedding` is `None`.
- **Avoid by:**
  - Validating all arrays before serialization.
  - Using a robust `default_serializer()` for edge cases.

---

## 📌 Section IV: Index Repair & Recovery

### 9. Invalid Callback in `repair_index()`
- **Issue:** Callback fails due to missing required params (e.g., `geometry_manager`).
- **Avoid by:**
  - Always passing required args to `load_assembly()`.
  - Using high-level methods like `get_memory_by_id_async()` during repair.

---

### 10. Missing Error Propagation in Repair
- **Issue:** Repair silently fails but API returns `200 OK`.
- **Avoid by:**
  - Returning error context in the API response.
  - Logging `exc_info=True` on all internal errors.

---

## 📌 Section V: Testing Pitfalls

### 11. Destroying Test Setup Mid-Test
- **Issue:** Calling `repair_index()` *after* creating test memories deletes them.
- **Avoid by:**
  - Running repair *before* tests or mocking persistence layer.
  - Verifying persisted state exists before triggering repair.

---

### 12. Mismatched Response Keys
- **Issue:** Tests fail due to accessing `.get("results")` instead of `.get("memories")`.
- **Avoid by:**
  - Ensuring test logic matches current API schema.
  - Using test constants for key names if schema is unstable.

---

## 📌 Section VI: Utility & Embedding Validation

### 13. Broken Type Comparison (`embedding_validators.py`)
- **Issue:** Comparing `int > str` → `TypeError`.
- **Avoid by:**
  - Converting all dimensions to `int` before comparison.
  - Adding type checks at function boundaries.

---

### 14. Redundant Embedding Utilities
- **Issue:** Validation logic duplicated across `utils/`, `geometry_manager`, `vector_index`.
- **Avoid by:**
  - Unifying embedding normalization in `GeometryManager`.
  - Removing conflicting helper logic from `utils/`.

---

## 📌 Section VII: Environment Setup

### 15. FAISS-GPU Fallbacks
- **Issue:** `faiss-gpu` fails to load silently; CPU used instead.
- **Avoid by:**
  - Logging fallback explicitly.
  - Verifying `faiss.StandardGpuResources()` exists before invoking GPU ops.

---

## 📌 Developer Reminders

- ✅ Validate vectors before any operation.
- ✅ Always wrap FAISS ops in `asyncio.to_thread()` if used inside async methods.
- ✅ Never assume file writes succeed — always check.
- ✅ Don’t rely on automatic repair unless you verify the trigger conditions.
- ✅ Include retry logic and expose retry queue stats in `/stats`.
- ✅ Consider implementing `/assemblies/{id}/timeline` for drift and update tracking.

---

## 🔒 Recommended Stability Enhancers

- `vector_index_updated_at`: track desyncs for assemblies.
- `vector_index.get_fingerprint()`: log consistency across save/load.
- `RecoveryTimeline`: track per-ID update attempts, failures, retries.
- `assembly.schema_version`: ensure forward compatibility.

---

## 🛑 Final Note

> This system is not just a persistence layer. It is the **epistemic core of contextual memory**.  
> A corrupted index is not just data loss—it’s **conceptual amnesia**.  
> Build accordingly.
