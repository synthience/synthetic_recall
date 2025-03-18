# Emotion Analyzer Integration

## Overview

The Emotion Analyzer is a WebSocket-based service that analyzes text to detect emotions. It's used to enhance memory context with emotional information, allowing for more personalized and contextually-aware responses.

## API Endpoints

### WebSocket API

- **Endpoint**: `ws://localhost:5007/ws`
- **Web Interface**: `http://localhost:8007`

### Authentication

No authentication is required for local development.

## Message Format

### Request Format

```json
{
  "type": "analyze",
  "text": "The text to analyze for emotional content."
}
```

### Health Check

```json
{
  "type": "health_check"
}
```

### Response Format

```json
{
  "detailed_emotions": {
    "excitement": 0.791286289691925
  },
  "primary_emotions": {
    "joy": 0.791286289691925,
    "sadness": 0.0,
    "anger": 0.0,
    "fear": 0.0,
    "surprise": 0.0,
    "neutral": 0.0,
    "other": 0.0
  },
  "dominant_detailed": {
    "emotion": "excitement",
    "confidence": 0.791286289691925
  },
  "dominant_primary": {
    "emotion": "joy",
    "confidence": 0.791286289691925
  },
  "type": "analysis_result",
  "input_text": "The text that was analyzed"
}
```

### Error Response

```json
{
  "type": "error",
  "message": "Error description"
}
```

## Integration

The `EmotionMixin` class in `memory_core/emotion.py` integrates with the Emotion Analyzer service. It provides two main methods:

1. `detect_emotion`: Returns the dominant emotion detected in the text
2. `detect_emotional_context`: Returns detailed emotional analysis including primary and detailed emotions

## Memory Integration

Emotional context is now integrated into the memory system in the following ways:

### Memory Storage with Emotional Context

When storing a memory using the `store_memory` method in `EnhancedMemoryClient`, the system now:

1. Automatically detects and analyzes the emotional context of the memory content
2. Stores the emotional data in the memory's metadata, including:
   - Emotional state (dominant emotion)
   - Sentiment value (ranging from -1.0 to 1.0)
   - Detailed emotion confidence scores
3. Adjusts the memory's quickrecal_score based on emotional intensity, making emotionally charged memories more likely to be retrieved

### Memory Retrieval by Emotional Context

A new method `retrieve_memories_by_emotion` has been added that allows filtering memories by:

1. Specific emotion (e.g., joy, anger, sadness)
2. Sentiment threshold and direction (positive vs. negative)
3. Minimum quickrecal_score

### LLM Tool Integration

A new tool `retrieve_emotional_memories` has been added to the LLM toolset, allowing AI agents to:

1. Query memories based on emotional criteria
2. Use emotional context for more personalized responses
3. Implement emotion-aware feedback weighting for recursive response refinement

## Fallback Mechanism

If the Emotion Analyzer service is unavailable, the integration will fall back to using the HPC service for emotion analysis.

## Environment Variables

- `EMOTION_ANALYZER_HOST`: Host for the emotion analyzer service (default: `localhost`)
- `EMOTION_ANALYZER_PORT`: Port for the emotion analyzer service (default: `5007`)

## Running the Emotion Analyzer

Use the Docker Compose file to run the Emotion Analyzer service:

```bash
docker-compose -f docker-compose.emotion.yml up -d
```

## Testing

Use the utility script to test the integration:

```bash
python utils/test_emotion_analyzer.py --with-delay "I am feeling excited about this project!"
```

## Use Cases

1. **Emotional Response Weighting**: Use emotional context to weight different responses in recursive refinement loops
2. **Personalized Response Generation**: Generate responses that match the user's emotional state
3. **Emotional Memory Filtering**: Retrieve memories that match specific emotional criteria
4. **Sentiment Tracking**: Track sentiment over time for trend analysis
5. **Memory Boosting**: Automatically increase the quickrecal_score of emotionally intense memories
