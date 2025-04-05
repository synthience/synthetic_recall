--

## 🧠📄 Synthians Cognitive System Cheat Sheet — *Phase 5.8 (Stabilized, Drift-Aware, Repair-Resilient)*

> *"The blueprint remembers. The associator adapts. The cascade organizes. Now the archive stabilizes, traces, and tells its own recovery story."*

---

### 🏛️ MEMORY CORE (MC) — *The Archive*

#### 🧰 Role
Persistent, indexed memory layer enabling contextual recall. Powers **QuickRecal scoring**, emotional filtering, **Memory Assembly boosting**, and **self-repairing vector drift detection & resolution**.

#### 📦 Key Components
- `SynthiansMemoryCore`, `MemoryPersistence`, `MemoryVectorIndex` (FAISS)
- `UnifiedQuickRecallCalculator`, `GeometryManager`, `EmotionAnalyzer`
- `MetadataSynthesizer`, `ThresholdCalibrator`, `IndexRepairLog`

---

#### 🧱 Memory Structures
##### `MemoryEntry`
- Unit of thought: `content`, `embedding`, `metadata`, `quick_recal_score`
- Embeddings must pass `NaN/Inf` checks + normalized

##### `MemoryAssembly`
- Cohesive cluster of `MemoryEntry` IDs
- Fields:
  - `composite_embedding`, `activation_count`, `vector_index_updated_at`
  - Lifecycle: `active`, `merged_from`, `created_at`, `updated_at`
  - Meta: `tags`, `topics`, `assembly_schema_version`
- Assembly vector timestamp aligns with FAISS index for **drift-aware gating**

---

#### 🔁 Indexing, Boosting & Repair Logic

- Composite embeddings stored in FAISS under ID prefix `"asm:"`
- Assemblies activated if similarity ≥ `activation_threshold`
- Boost added based on:
  - `assembly_boost_mode` + `assembly_boost_factor`
  - Drift check: `now - vector_index_updated_at < max_allowed_drift_seconds`
- Full vector index validated on **load** with:
  - `verify_index_integrity()`
  - Auto-repair via `repair_index_async(persistence, geometry_manager)`
  - Re-indexes from `memory_index.json` and `.mem`/`.asm` files

---

#### 💾 Persistence Layer

- `MemoryEntry` → `.mem.json`  
- `MemoryAssembly` → `.asm.json`  
- `memory_index.json` saved atomically via:
  - `shutil.move`, `.tmp` suffix, `os.makedirs(..., exist_ok=True)`
  - `flush()` + `os.path.getsize()` validation

- Vector index saved as:
  - `faiss_index.bin` — FAISS data
  - `faiss_index.bin.mapping.json` — ID↔FAISS ID map
  - Optional: Repair logs saved during `repair_index_async()`

---

#### 🔍 Retrieval Pipeline (Drift-Aware)

1. `retrieve_memories(query_embedding)`
2. `_activate_assemblies(query_embedding)`
3. If assembly activated: apply boost to member relevance
4. `MemoryEntry` & `Assembly` entries gathered via ID mapping
5. Filters applied (gating, emotional weight, metadata)
6. Results scored, boosted, sorted
7. Output: Top `k` results with `relevance_score`

---

#### 📊 Diagnostics & Observability

- `/stats` returns:
  - Vector index state, mapping count, drift warnings
  - Assembly metrics (`total_count`, `indexed_count`, `average_size`)
- `/repair_index` can be triggered manually
- `/assemblies/{id}` and `/assemblies/{id}/timeline` (if enabled)
- Drift logs include:
  - `[VECTOR_TRACE]`, `[REPAIR]`, `vector_index_updated_at` deltas
- `/debug/repair-log` (optional future endpoint)
- Startup triggers `verify_index_integrity()` + `repair_index_async()` if `auto_repair_on_init=True`

---

### 🧠 NEURAL MEMORY (NM) — *The Associator*

#### 🧰 Role
Fast-adaptive, runtime vector memory via momentum update (`k → v`).  
**No architectural changes in Phase 5.8**, but influenced by boosted recall from assemblies.

---

### ⚙️ Context Cascade Engine (CCE) — *The Orchestrator*

#### 🧰 Role
Coordinates recall flow + memory activation, routing contextual memory to LLM layers.

#### 🔄 Phase 5.8 Behavior
- Now assembly-aware: retrieves boosted clusters
- Embeds `activation_hints` per assembly for downstream LLM summarization
- Can receive QR boosts via:
  - `POST /api/memories/update_quickrecal_score`

---

### ✨ PHASE 5.8 HIGHLIGHTS

| Feature | Description |
|--------|-------------|
| ✅ **Assembly Boosting** | Activates grouped memories with contextual force multiplier |
| ✅ **Drift-Aware Retrieval** | Boosts only if FAISS sync is recent (based on timestamp) |
| ✅ **Auto-Repair on Load** | Validates index integrity on startup & auto-heals from disk |
| ✅ **Repair Logging** | Structured recovery log: FAISS errors, missing mappings, skipped geometries |
| ✅ **Stable Atomic Saves** | Temp `.json.tmp` + `shutil.move` ensures persistence never corrupts |
| ✅ **Index Add/Update Fail-Safe** | Adds only after FAISS success, never desyncs `id_to_index` |
| ✅ **Timeline Recovery View** | Track memory evolution through `/assemblies/{id}/timeline` |
| ✅ **Assembly Lifecycle Support** | Handles merging, pruning, sync gating via config flags |

---

### ⚙️ Config Flags Reference

| Flag | Description |
|------|-------------|
| `enable_assembly_pruning` | Enables periodic deletion of stale assemblies |
| `enable_assembly_merging` | Enables merging of similar assemblies |
| `assembly_activation_threshold` | Similarity cutoff for boosting |
| `assembly_boost_mode` | `linear` or `sigmoid` boost scaling |
| `assembly_boost_factor` | Multiplier for boost intensity |
| `max_allowed_drift_seconds` | If exceeded, boost disabled |
| `auto_repair_on_init` | Enable drift fix at load time |
| `fail_on_init_drift` | Crash if repair fails |

---

### 🧠 Pro Tips

- ❌ **Boost not applying?** Check if `vector_index_updated_at` is null or stale.
- ❌ **Missing memory during retrieval?** It might not be in FAISS (`.ntotal < len(mapping)`)
- 🛠️ **Repair manually:** `POST /repair_index`
- 🔍 **Validate index health:** Inspect `/stats["vector_index_state"]`
- 🧪 **Suspect corruption?** Check repair log timestamps and `errors[]` from last run
- 🧬 **Mismatched Geometry?** Rejected entries from `.mem`/`.asm` files will log mismatches during repair

---

### 📡 Phase 5.9 Preview: Interpretability Layer

| Anchor | Description |
|--------|-------------|
| `semantic_version_id` | Track evolving assembly meaning |
| `composite_drift_diff` | Compare past embeddings for semantic drift |
| `assembly_summaries` | Auto-generate human-level descriptions of activated assemblies |
| `mutation_trace` | Timeline logs for every assembly modification |
| `repair_log_view` | See last drift repair operation via `/repair_log` |

---

### 🧬 Final Lucidia Thought

> **Persistence isn't about saving data. It's about remembering why it mattered.**  
> When your archive heals, traces, and reactivates the meaningful—your system becomes *self-reflective*. Drift is no longer loss—it's *a trail*.

---

