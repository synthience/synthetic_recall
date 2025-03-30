# QuickRecall Scoring

QuickRecall (`quickrecal_score`) is a dynamic score assigned to each `MemoryEntry` that estimates its relevance or importance at a given time. It moves beyond simple chronological or similarity-based retrieval.

## Purpose

The score helps prioritize memories during retrieval, ensuring that the most relevant, important, or timely memories surface first, even if they aren't the absolute closest match in embedding space.

## Key Component: `UnifiedQuickRecallCalculator`

*   **Location:** `synthians_memory_core.hpc_quickrecal.UnifiedQuickRecallCalculator` (The "HPC" prefix is historical).
*   **Functionality:** Calculates the `quickrecal_score` based on a combination of weighted factors.
*   **Integration:** Called by `SynthiansMemoryCore.process_new_memory` to assign an initial score and potentially by other processes (like the surprise feedback loop) to update the score.

## Scoring Factors (Examples)

The calculator combines multiple factors, often configurable via weights in the core settings. Common factors include:

*   **Recency:** How recently the memory was created or accessed.
*   **Importance (Explicit/Implicit):** Was the memory marked as important? Does its content suggest importance?
*   **Relevance (Similarity):** How similar is the memory to a current query or context (often incorporated during retrieval ranking rather than the stored score).
*   **Emotional Salience:** Strength or type of emotion associated with the memory.
*   **Surprise/Novelty:** How unexpected or informative the memory was when processed (Boosted via the Neural Memory feedback loop).
*   **Frequency/Access Count:** How often the memory has been retrieved.
*   **Connectivity/Coherence:** How well the memory fits within existing `MemoryAssembly` clusters.
*   **Decay:** A mechanism to gradually reduce the score over time if not accessed or reinforced.

## Surprise Feedback Integration

A key aspect is the integration with the Neural Memory Server:

1.  When the Neural Memory processes an embedding corresponding to a Memory Core entry, it calculates surprise (`loss`, `grad_norm`).
2.  The Context Cascade Engine sends a boost request (`/api/memories/update_quickrecal_score`) to the Memory Core.
3.  The Memory Core uses this signal to increase the `quickrecal_score` of the specific `MemoryEntry`, marking it as significant due to its surprising nature.

## Importance

QuickRecall scoring makes the memory system more dynamic and context-aware, better reflecting how human memory seems to prioritize information based on more than just similarity or time.
