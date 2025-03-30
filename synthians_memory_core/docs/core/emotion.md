# Emotional Intelligence Components

The Synthians Memory Core incorporates emotional context into memory processing and retrieval through two key components within the `synthians_memory_core.emotional_intelligence` module.

## 1. `EmotionAnalyzer`

*   **Purpose:** Analyzes text content to determine its emotional profile.
*   **Functionality:**
    *   Typically utilizes an external library or model (like `transformers` with a sentiment/emotion classification model) to analyze input text.
    *   Outputs structured emotional data, often including:
        *   `dominant_emotion`: The most prominent emotion detected (e.g., joy, sadness, anger).
        *   `sentiment_label`: Positive, Negative, or Neutral.
        *   `sentiment_score`: A numerical value indicating sentiment polarity/intensity.
        *   Emotion scores: Confidence scores for various basic emotions.
    *   This information is added to the `metadata` of a `MemoryEntry` during processing.
*   **Configuration:** May require specifying the model name or path in the core configuration.

## 2. `EmotionalGatingService`

*   **Purpose:** Filters or re-ranks memory retrieval results based on emotional context.
*   **Functionality:**
    *   Takes the initial list of candidate memories retrieved (e.g., via vector search).
    *   Considers the user's current emotional state (if provided) and the emotional metadata stored within each candidate memory.
    *   Applies rules or scoring adjustments to:
        *   **Filter:** Remove memories that clash significantly with the user's current state or are deemed inappropriate given the context.
        *   **Re-rank:** Boost memories that resonate emotionally with the user's state or the query context.
    *   Aims to provide more contextually relevant and potentially more empathetic recall.
*   **Integration:** Used within the `SynthiansMemoryCore.retrieve_memories` method after initial candidate retrieval.

## Importance

Integrating emotional intelligence allows the memory system to:

*   Tag memories with their emotional context at the time of encoding.
*   Provide recall that is sensitive to the user's current emotional state.
*   Potentially prioritize memories associated with strong emotions, mimicking aspects of human memory.

## Recent Improvements

The emotion processing components have been enhanced to handle embedding dimension mismatches (384D vs 768D) through:

- Updates to the `_calculate_emotion` method to use vector alignment utilities
- Proper fallbacks when either the emotion service is unavailable or dimension mismatches occur
- Integration with the `MetadataSynthesizer` to ensure emotional metadata is consistently stored

## Configuration Options

*To be added: Documentation on configuration parameters for the emotion components*
