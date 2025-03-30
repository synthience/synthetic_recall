# Metadata Synthesis

The `synthians_memory_core.metadata_synthesizer.MetadataSynthesizer` class is responsible for automatically generating and enriching the metadata associated with each `MemoryEntry`.

## Purpose

Metadata provides crucial context about a memory beyond its raw content and embedding. Synthesized metadata helps in:

*   **Enhanced Retrieval:** Filtering or boosting memories based on time, emotion, complexity, etc.
*   **Analysis & Understanding:** Providing insights into the nature and origin of memories.
*   **Scoring:** Contributing factors to the `quickrecal_score` calculation.

## Key Component: `MetadataSynthesizer`

*   **Functionality:** Takes the raw input (content, timestamp, source information, embedding) and generates a dictionary of derived metadata fields.
*   **Integration:** Called by `SynthiansMemoryCore.process_new_memory` after initial processing but before final storage.

## Synthesized Metadata Fields (Examples)

The synthesizer aims to add fields like:

*   **Temporal:**
    *   `timestamp_iso`: Standardized ISO 8601 format.
    *   `time_of_day`: Morning, Afternoon, Evening, Night.
    *   `day_of_week`: Monday, Tuesday, etc.
    *   `month`, `year`.
*   **Emotional (if `EmotionAnalyzer` is used):**
    *   `dominant_emotion`, `sentiment_label`, `sentiment_score`.
*   **Cognitive/Complexity:**
    *   `word_count`, `char_count`.
    *   `complexity_estimate`: A simple measure (e.g., based on sentence length or vocabulary).
*   **Embedding Information:**
    *   `embedding_dim`: Dimension of the stored embedding.
    *   `embedding_norm`: Magnitude of the embedding vector (before/after normalization).
    *   `embedding_provider`: Source of the embedding (e.g., model name).
*   **Identifiers:**
    *   `memory_id`: The unique UUID assigned to the memory entry.
    *   `source`, `user_id`, `session_id`: Preserved if provided in the initial input metadata.

## Configuration

*   The specific metadata fields generated might be influenced by the availability of other components (like the `EmotionAnalyzer`) and potential configuration flags (though currently less configurable than other components).

## Importance

Automated metadata synthesis ensures that memories are consistently tagged with rich contextual information without requiring manual input for every field, significantly enhancing the utility and searchability of the memory core.
