# Interruption Tracking and Analysis Module

## Overview

The interruption module provides a bridge between Lucidia's voice interaction system and the memory core. It captures conversational rhythm, interruption patterns, and speaking behaviors to enhance the semantic understanding of conversations with rich contextual metadata.

## Key Components

### InterruptionAwareMemoryHandler

A specialized handler that processes transcripts with interruption metadata and stores them in the memory system with rich contextual information.

```python
from synthians_memory_core.interruption import InterruptionAwareMemoryHandler

# Initialize the handler
handler = InterruptionAwareMemoryHandler(api_url="http://localhost:8000")

# Process a transcript with interruption data
await handler(
    text="I wanted to explain something important.",
    was_interrupted=True,
    user_interruptions=2,
    interruption_timestamps=[1678945330.45, 1678945342.12]
)
```

## Integration with VoiceStateManager

The interruption module is designed to work with the `VoiceStateManager` from the voice_core package. The VoiceStateManager tracks interruptions in real-time and provides this data when processing transcripts.

### Configuration

To connect the VoiceStateManager with the InterruptionAwareMemoryHandler:

```python
from voice_core.state.voice_state_manager import VoiceStateManager
from synthians_memory_core.interruption import InterruptionAwareMemoryHandler

# Initialize components
state_manager = VoiceStateManager()
memory_handler = InterruptionAwareMemoryHandler(api_url="http://localhost:8000")

# Register the memory handler as the transcript handler
state_manager.register_transcript_handler(memory_handler)
```

## Memory Processing Flow

1. VoiceStateManager detects and tracks interruptions during conversation
2. When a transcript is processed, interruption metadata is attached
3. InterruptionAwareMemoryHandler sends this enriched data to the memory API
4. TranscriptionFeatureExtractor processes the text and metadata
5. The memory is stored with rich conversational context

## Using Interruption Data for Reflection

The module provides utilities to generate reflection prompts based on interruption patterns:

```python
from synthians_memory_core.interruption import InterruptionAwareMemoryHandler

# For a memory with high interruption count
prompt = InterruptionAwareMemoryHandler.get_reflection_prompt({
    "was_interrupted": True,
    "user_interruptions": 6
})
# Returns: "You seem to be interrupting frequently. Would you like me to pause more often to let you speak?"
```

## Compatibility with Embedding Handling

This module is fully compatible with Lucidia's robust embedding handling system:

- Works with both 384 and 768 dimension embeddings
- Properly handles vector alignment during comparison operations
- Validates embeddings to prevent NaN/Inf values
- Provides graceful fallbacks when embedding generation fails

## Metadata Structure

The interruption metadata schema includes:

```json
{
  "was_interrupted": true,            // Whether this specific utterance was interrupted
  "user_interruptions": 3,           // Total interruptions in the current session
  "interruption_timestamps": [       // Timestamps of interruptions (relative to session start)
    12.5, 24.1, 38.8
  ],
  "session_id": "abc123",            // Unique ID for the current conversation session
  "interruption_severity": "medium", // Classification of interruption pattern severity
  "requires_reflection": true        // Whether this memory might benefit from reflection
}
```

## Best Practices

1. **Session Management**: Generate a new session ID for each distinct conversation
2. **Timestamp Precision**: Store interruption timestamps as relative times (seconds from session start)
3. **Aggregation**: Consider aggregating interruption patterns across multiple sessions for deeper insights
4. **Memory Retrieval**: Use interruption metadata as a factor in memory prioritization
