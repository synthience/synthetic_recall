# Memory Structures

This document details the core memory structures used in the Synthians Memory Core, focusing on `MemoryEntry` and `MemoryAssembly` classes, including the `merged_from` field that will be fully utilized in Phase 5.9.

## MemoryEntry

`MemoryEntry` is the fundamental unit of storage in the Synthians Memory Core. It represents a single piece of information with its vector representation and metadata.

### Structure

```python
class MemoryEntry:
    def __init__(
        self,
        id: str,
        content: str,
        embedding: List[float],
        timestamp: Optional[str] = None,
        memory_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
        quick_recal_score: float = 0.5,
        tags: Optional[List[str]] = None,
        emotion_metadata: Optional[Dict[str, Any]] = None,
    ):
        # Unique identifier
        self.id = id
        
        # Core content and representation
        self.content = content
        self.embedding = embedding
        
        # Temporal data
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        
        # Type and classification
        self.memory_type = memory_type
        
        # Metadata and tags
        self.metadata = metadata or {}
        self.tags = tags or []
        
        # Emotional data
        self.emotion_metadata = emotion_metadata or {}
        
        # Relevance scoring
        self.quick_recal_score = quick_recal_score
```

### Key Features

- **Unique Identification**: Each memory has a unique ID.
- **Content and Embedding**: Stores both the textual content and its vector representation.
- **Temporal Information**: Timestamp of when the memory was created.
- **Metadata**: Flexible dictionary for additional attributes.
- **QuickRecal Score**: Dynamic relevance score that can be updated based on feedback.
- **Emotion Metadata**: Emotional context of the memory.
- **Tags**: List of categorical tags for organization.

## MemoryAssembly

`MemoryAssembly` represents a group of related `MemoryEntry` objects that form a cohesive unit. It maintains its own composite embedding and tracks relationships between memories.

### Structure

```python
class MemoryAssembly:
    def __init__(
        self,
        id: str,
        name: str,
        composite_embedding: List[float],
        memory_ids: List[str],
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        tags: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        merged_from: Optional[List[str]] = None,
        assembly_schema_version: str = "1.0",
        vector_index_updated_at: Optional[str] = None,
        activation_count: int = 0
    ):
        # Unique identifier and name
        self.id = id
        self.name = name
        
        # Vector representation (composite of member memories)
        self.composite_embedding = composite_embedding
        
        # Member memories
        self.memory_ids = memory_ids
        
        # Temporal data
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or self.created_at
        
        # Classification
        self.tags = tags or []
        self.topics = topics or []
        
        # Lineage tracking - used for merge ancestry
        self.merged_from = merged_from or []
        
        # Technical metadata
        self.assembly_schema_version = assembly_schema_version
        
        # Vector index synchronization timestamp
        self.vector_index_updated_at = vector_index_updated_at
        
        # Usage statistics
        self.activation_count = activation_count
```

### Key Features

- **Composite Embedding**: Vector representation of the entire assembly, not just the sum of its parts.
- **Member Management**: Tracks the IDs of member `MemoryEntry` objects.
- **Temporal Tracking**: Creation and update timestamps.
- **Classification**: Tags and topics for organization.
- **Merged From (Lineage)**: The `merged_from` field tracks the IDs of source assemblies that were merged to form this assembly.
- **Vector Index Synchronization**: The `vector_index_updated_at` timestamp is CRITICAL for system stability. It marks when the assembly's vector was last successfully synchronized with the FAISS vector index.
  - **Important**: Only assemblies with a recent timestamp (within `max_allowed_drift_seconds`) are considered "synchronized" and eligible to contribute to retrieval boosting.
  - If this timestamp is missing or too old, the assembly will not be used for boosting, even if it would otherwise be a relevant match.
  - This mechanism, introduced in Phase 5.8, prevents boosts from stale embeddings and ensures system stability during FAISS index operations.
- **Activation Statistics**: Tracks how often the assembly has been activated.

## Drift-Aware Gating Mechanism

**CRUCIAL:** The `vector_index_updated_at` timestamp determines if an assembly is "synchronized" with the vector index. Only synchronized assemblies (where the timestamp is recent, within `max_allowed_drift_seconds`) contribute to retrieval boosting. This prevents boosting based on stale embeddings.

This is a critical stability mechanism introduced in Phase 5.8:

1. When an assembly is created or its embedding is updated, the system attempts to update the vector index.
2. If the update succeeds, `vector_index_updated_at` is set to the current time:
   ```python
   # After successful vector index update:
   assembly.vector_index_updated_at = datetime.utcnow().isoformat()
   ```

3. If the update fails (e.g., due to FAISS errors or GPU memory issues), the update is queued for retry via the internal `_pending_vector_updates` queue (replacing the deprecated `AssemblySyncManager`) and the timestamp remains null or unchanged.
   ```python
   try:
       await self.vector_index.add_with_ids([assembly.id], [assembly.composite_embedding])
       assembly.vector_index_updated_at = datetime.utcnow().isoformat()
   except Exception as e:
       logger.warning(f"Failed to add assembly to vector index, queuing for retry: {e}")
       await self._pending_vector_updates.put({
           "operation": "add",
           "id": assembly.id,
           "embedding": assembly.composite_embedding,
           "is_assembly": True
       })
   ```

4. During retrieval in the `_activate_assemblies` method, the system explicitly checks this timestamp:
   ```python
   now = datetime.utcnow()
   for assembly_id, assembly in self._assemblies.items():
       # Skip assemblies without synchronized vectors
       if assembly.vector_index_updated_at is None or (
           now - datetime.fromisoformat(assembly.vector_index_updated_at)).total_seconds() > self.config.ASSEMBLY_MAX_DRIFT_SECONDS:
           # ⚠️ SKIPPED: Assembly vector not synchronized
           continue
           
       # Calculate similarity and possibly activate the assembly
       # ...
   ```

5. Only assemblies that pass this check are considered synchronized and used for boosting.

This mechanism ensures that:
- The system never boosts based on stale or inconsistent vector representations
- The Memory Core can continue functioning even if temporary vector index failures occur
- The system self-heals as the background retry process successfully updates the index

## Merged From Field

The `merged_from` field in `MemoryAssembly` is particularly important for the upcoming explainability features in Phase 5.9. It serves as the foundation for assembly lineage tracking.

### Current Implementation

The field exists in the data structure but is not fully utilized yet:

```python
self.merged_from = merged_from or []  # List of source assembly IDs
```

### Phase 5.9 Enhancements (Planned)

In Phase 5.9, this field will be fully utilized to:

1. **Track Merge Ancestry**: When two or more assemblies are merged, the source assembly IDs will be stored in the `merged_from` field of the resulting assembly.

2. **Enable Lineage Tracing**: The system will use this field to walk the tree of ancestors, allowing visualization of the complete lineage of an assembly.

3. **Support Merge Explanation**: Combined with merge logs, this field will help explain why and how assemblies were merged.

4. **Facilitate Explainability**: The field will power the `/assemblies/{id}/lineage` API endpoint for visualizing assembly ancestry.

### Example Usage (Planned for Phase 5.9)
```python
# During merge operation
async def merge_assemblies(self, source_assembly_ids: List[str]) -> str:
    # Create new assembly from sources
    new_assembly = MemoryAssembly(
        id=f"asm_{uuid.uuid4().hex[:8]}",
        name="Merged Assembly",
        composite_embedding=computed_composite_embedding,
        memory_ids=combined_memory_ids,
        # Store source assembly IDs for lineage tracking
        merged_from=source_assembly_ids
    )
```

### Critical Implementation Details for `merged_from` Population

**Timing and Atomicity Considerations:**

The correct population of the `merged_from` field is crucial for system integrity and the explainability features planned in Phase 5.9. Follow these strict guidelines:

1. **Mandatory Population**: The `merged_from` list **must** be populated with source assembly IDs *before* the new assembly is persisted for the first time.

2. **Atomicity Sequence**: The proper sequence of operations in `_execute_merge` is:
   ```
   1. Identify assemblies A and B to merge
   2. Create new assembly C with merged_from=[A.id, B.id]
   3. Persist assembly C to storage (via persistence.save_assembly)
   4. Optionally log the merge event (via merge_tracker.log_merge in Phase 5.9)
   5. Start async task for cleanup and source deletion
   ```

3. **Error Handling**: If an error occurs between persisting the new assembly and cleaning up source assemblies, the system must still maintain the lineage information. The persisted `merged_from` field ensures this even if cleanup fails or the system crashes.

4. **Potential Edge Case**: In the unlikely event of a system crash between saving the merged assembly and completing the cleanup of source assemblies, there could temporarily be both a new merged assembly *and* its source assemblies in the system. The `merged_from` field helps identify this situation during recovery.

5. **Note for Phase 5.9**: When the merge logging system is implemented, the `merged_from` field will be the primary mechanism for reconstructing the merge lineage. Any failure to properly set this field will result in incomplete explainability data.
    )
    
    # Log the merge operation
    await self.merge_tracker.log_merge(
        source_assembly_ids, 
        new_assembly.id,
        similarity_score,
        threshold
    )
    
    # Store and return
    await self.persistence.save_assembly(new_assembly)
    return new_assembly.id
```

## Persistence

Both `MemoryEntry` and `MemoryAssembly` objects are stored persistently:

- `MemoryEntry` objects are saved as `.mem.json` files.
- `MemoryAssembly` objects are saved as `.asm.json` files.
- The `merged_from` field is included in the JSON serialization of assemblies.
- The `vector_index_updated_at` timestamp is preserved across system restarts.

See the [Persistence](./persistence.md) documentation for details on the storage system.

## Memory Structure Schema Versions

The system supports versioned schemas to allow for future enhancements:

- **MemoryEntry**: Currently version "1.0"
- **MemoryAssembly**: Currently version "1.0"

New fields may be added in future versions while maintaining backward compatibility.