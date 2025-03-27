# Conversational Awareness and Interruption Management

## Overview

This documentation covers the integration of conversational rhythm and interruption tracking into the Lucidia memory system. These enhancements enable Lucidia to maintain awareness of the conversational flow, detect interruptions, and incorporate this rich contextual information into memory entries.

## Key Components

### 1. TranscriptionFeatureExtractor

The `TranscriptionFeatureExtractor` class enriches transcribed speech with emotional, semantic, and conversational metadata.

**Location**: `synthians_memory_core/utils/transcription_feature_extractor.py`

**Features**:
- Emotional analysis using EmotionAnalyzer
- Keyword extraction using KeyBERT (optional)
- Speech pattern analysis (speaking rate, duration)
- Interruption detection and tracking

**Example Usage**:
```python
from synthians_memory_core.utils.transcription_feature_extractor import TranscriptionFeatureExtractor
from synthians_memory_core.emotion_analyzer import EmotionAnalyzer

# Initialize components
emotion_analyzer = EmotionAnalyzer()
extractor = TranscriptionFeatureExtractor(emotion_analyzer=emotion_analyzer)

# Extract features from a transcription
metadata = await extractor.extract_features(
    transcript="I was explaining something important when you interrupted me.",
    meta={
        "duration_sec": 2.5,
        "was_interrupted": True,
        "user_interruptions": 3,
        "interruption_timestamps": [1678945201.23, 1678945210.45, 1678945220.12]
    }
)

# Example output metadata
# {
#   "dominant_emotion": "frustration",
#   "emotions": {"frustration": 0.72, "neutral": 0.15, "anger": 0.08, "sadness": 0.05},
#   "input_modality": "spoken",
#   "keywords": ["explaining", "important", "interrupted"],
#   "speaking_rate": 3.2,
#   "duration_sec": 2.5,
#   "was_interrupted": True,
#   "user_interruptions": 3,
#   "interruption_timestamps": [1678945201.23, 1678945210.45, 1678945220.12],
#   "requires_reflection": True,
#   "interruption_severity": "medium"
# }
```

### 2. InterruptionAwareMemoryHandler

The `InterruptionAwareMemoryHandler` serves as a bridge between the voice system's interruption tracking and the memory system, ensuring that conversation dynamics are preserved in memory.

**Location**: `synthians_memory_core/interruption/memory_handler.py`

**Features**:
- Processes transcripts with interruption metadata
- Enriches memory entries with conversational rhythm information
- Provides reflection prompts based on interruption patterns
- Integrates with the API's `/process_transcription` endpoint

**Example Usage**:
```python
from synthians_memory_core.interruption import InterruptionAwareMemoryHandler

# Initialize handler
memory_handler = InterruptionAwareMemoryHandler(api_url="http://localhost:8000")

# Process a transcript with interruption data
result = await memory_handler(
    text="I was trying to explain when you jumped in.",
    transcript_sequence=12,
    timestamp=time.time(),
    confidence=0.95,
    was_interrupted=True,
    user_interruptions=2,
    interruption_timestamps=[1678945330.45, 1678945342.12],
    session_id="abc123"
)

# Get a reflection prompt if needed
prompt = InterruptionAwareMemoryHandler.get_reflection_prompt({
    "was_interrupted": True,
    "user_interruptions": 6
})
# Example output: "You seem to be interrupting frequently. Would you like me to pause more often to let you speak?"
```

### 3. API Endpoint for Transcription Processing

A dedicated endpoint has been added to the API to process transcriptions with rich metadata extraction.

**Location**: `synthians_memory_core/api/server.py`

**Endpoint**: `/process_transcription`

**Request Format**:
```json
{
  "text": "The transcribed text content",
  "audio_metadata": {
    "duration_sec": 4.8,
    "was_interrupted": true,
    "user_interruptions": 2,
    "interruption_timestamps": [1678945330.45, 1678945342.12],
    "speaker_id": "user_123", 
    "confidence": 0.95
  },
  "embedding": [0.1, 0.2, ...],  // Optional pre-computed embedding
  "memory_id": "abc123",  // Optional for updating existing memory
  "importance": 0.8,  // Optional importance score
  "force_update": false  // Whether to force update if memory exists
}
```

**Response Format**:
```json
{
  "success": true,
  "memory_id": "def456",
  "metadata": { /* Full extracted metadata */ },
  "embedding": [0.1, 0.2, ...]
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/process_transcription \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I was explaining something when you jumped in.",
    "audio_metadata": {
      "duration_sec": 2.5,
      "was_interrupted": true,
      "user_interruptions": 1,
      "speaker_id": "user_123"
    }
  }'
```

### 4. VoiceStateManager Enhancements

The `VoiceStateManager` now tracks detailed interruption data and provides it to the memory system.

**Location**: `voice_core/state/voice_state_manager.py`

**New Tracking Variables**:
```python
self._session_interruptions = 0
self._interruption_timestamps = []
self._current_session_id = str(uuid.uuid4())
self._session_start_time = time.time()
```

**Enhanced Methods**:
- `handle_user_speech_detected`: Now tracks interruption timestamps and counts
- `handle_stt_transcript`: Enriches transcripts with interruption metadata
- `_get_status_for_ui`: Includes interruption metrics in status updates

## Integration Flow

1. **Interruption Detection**:
   - VoiceStateManager detects when a user interrupts Lucidia during speech
   - Tracks timestamp and increments interruption counter

2. **Transcript Processing**:
   - When a transcript is processed, interruption metadata is attached
   - Data includes: was_interrupted flag, count, timestamps, and session ID

3. **Memory Creation**:
   - TranscriptionFeatureExtractor processes the transcript
   - Extracts emotions, keywords, and integrates interruption data
   - Adds contextual markers like "requires_reflection" and "interruption_severity"

4. **Retrieval and Reflection**:
   - During memory retrieval, interruption metadata enables context-aware responses
   - Reflection prompts can be generated based on interruption patterns

## Handling Dimension Mismatches

This implementation leverages previous improvements for handling embedding dimension mismatches:

1. The TranscriptionFeatureExtractor works with both 384 and 768 dimension embeddings
2. Vector alignment utilities ensure compatibility during comparison operations
3. Robust error handling prevents system crashes from malformed inputs

## Configuration Options

### EmotionAnalyzer Configuration

```yaml
# Docker environment variables
EMOTION_MODEL_PATH: /workspace/models/emotion
CUDA_VISIBLE_DEVICES: "" # Set to GPU index if available
```

### TranscriptionFeatureExtractor Configuration

Key configuration options in the constructor:

```python
TranscriptionFeatureExtractor(
    emotion_analyzer=None,  # Optional EmotionAnalyzer instance
    use_keybert=False,  # Whether to use KeyBERT for keyword extraction
    keybert_model=None,  # Optional custom KeyBERT model
    min_ngram=1,  # Minimum keyword n-gram size
    max_ngram=2  # Maximum keyword n-gram size
)
```

## Docker Integration

This feature works within the existing Docker infrastructure with no need to rebuild containers:

1. All components are mounted via Docker volumes
2. Emotion models are accessible via mounted volumes
3. New endpoints are available through the existing API server

## Example: Complete Flow

```python
# 1. User interrupts Lucidia
# VoiceStateManager tracks this interruption
state_manager._session_interruptions += 1
state_manager._interruption_timestamps.append(time.time())

# 2. Speech is transcribed with interruption context
transcript_with_metadata = {
    "text": "I need to tell you something important.",
    "was_interrupted": True,
    "user_interruptions": state_manager._session_interruptions,
    "interruption_timestamps": state_manager._interruption_timestamps,
    "session_id": state_manager._current_session_id
}

# 3. Memory handler processes this and sends to API
from synthians_memory_core.interruption import InterruptionAwareMemoryHandler
memory_handler = InterruptionAwareMemoryHandler()
result = await memory_handler(**transcript_with_metadata)

# 4. Memory now contains rich contextual data about the interruption
# This enables more human-like responses during retrieval
```

## Future Considerations

1. **Analytics and Patterns**: Track conversation patterns over time to identify when interruptions are more frequent

2. **Adaptive Behavior**: Enable Lucidia to adapt its speaking style based on interruption patterns

3. **Multi-speaker Awareness**: Extend tracking to differentiate between different speakers' interruption patterns

4. **UI Integration**: Provide visual feedback when interruptions are detected to improve user experience
