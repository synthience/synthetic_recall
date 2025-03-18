# Dolphining Framework for STT Correction

## Overview

The Dolphining Framework is a comprehensive approach to speech-to-text (STT) correction that enhances transcription accuracy through multiple phases of processing. Named after the intelligent problem-solving capabilities of dolphins, this framework maintains multiple interpretations of ambiguous speech input and applies layered processing to determine the most likely intended meaning.

## Framework Phases

The Dolphining Framework consists of seven core phases:

1. **Dive into Ambiguity**: Identifies potentially ambiguous or incorrect elements in transcripts and generates multiple interpretations.

2. **Overlapping Realities**: Maintains multiple possible interpretations simultaneously rather than committing to a single correction too early.

3. **Layered Processing**: Applies multiple processing layers to score candidate interpretations:
   - Surface layer: Basic text patterns and common corrections
   - Recursive layer: Domain-specific knowledge and context
   - Dynamic layer: User history and feedback

4. **Playful Exploration**: Employs creative approaches to resolve ambiguities, such as word substitution, phonetic matching, and contextual inference.

5. **Humanistic Precision**: Incorporates emotional and relational context to improve interpretation accuracy.

6. **Iterative Adaptation**: Learns from past corrections and user feedback to improve future STT processing.

7. **Networked Intent Discovery**: Understands the broader conversational intent to disambiguate unclear speech.

## Implementation Components

### DolphiningSttCorrector

The `DolphiningSttCorrector` class is the core implementation of the Dolphining Framework. It provides methods for each phase of the framework and orchestrates the entire correction process.

Key methods include:

- `correct_transcript()`: Main method that applies the complete Dolphining Framework to a transcript
- `process_with_websocket_stt()`: Processes audio data through a WebSocket STT service and applies Dolphining correction
- `_generate_candidates()`: Generates multiple interpretations for an ambiguous transcript (Phase 1 & 2)
- `_score_candidates()`: Scores candidate interpretations using layered processing (Phase 3)
- `_enhance_with_context()`: Enriches scoring with emotional and relational context (Phase 5)
- `feedback_correction()`: Learns from user feedback on corrections (Phase 6)

### DolphiningSTTIntegrator

The `DolphiningSTTIntegrator` class provides integration points between the Dolphining Framework and existing STT systems. It handles callbacks for correction events and manages the interaction with STT services.

Key features include:

- Integration with NemoSTT for real-time STT processing
- Callback system for handling corrections, clarifications, and emotion detection
- WebSocket support for processing audio data through external STT services
- Domain dictionary management for improved correction accuracy

## Integration with Memory Systems

The Dolphining Framework integrates with the Enhanced Memory System and Narrative Identity features to improve correction accuracy:

1. **Conversation Context**: Uses recent conversation history to understand the context of the current transcript
2. **Emotional Context**: Analyzes the emotional tone of the conversation to improve interpretation
3. **Narrative Identity**: Leverages user preferences and communication patterns stored in the narrative identity system

## Usage Examples

### Basic Usage

```python
# Initialize the corrector
corrector = DolphiningSttCorrector(
    memory_client=memory_client,
    domain_dictionary={"Lucidia": 0.9, "neural network": 0.85}
)

# Correct a transcript
correction_result = await corrector.correct_transcript("lucydia uses neural networks")

# Access the corrected text
corrected_text = correction_result["corrected"]  # "Lucidia uses neural networks"
```

### Integration with NemoSTT

```python
# Initialize components
memory_client = EnhancedMemoryClient(...)
nemo_stt = NemoSTT(...)

# Initialize the integrator
integrator = DolphiningSTTIntegrator(
    memory_client=memory_client,
    domain_dictionary={"Lucidia": 0.9}
)

# Register callbacks
def on_correction(data):
    print(f"Correction: {data['original']} -> {data['corrected']}")

integrator.register_callback("on_correction", on_correction)

# Integrate with NemoSTT
integrator.integrate_with_nemo_stt(nemo_stt)
```

## Configuration Options

- **confidence_threshold**: Minimum confidence level required for automatic correction (default: 0.7)
- **max_candidates**: Maximum number of candidate interpretations to generate (default: 5)
- **min_similarity**: Minimum similarity threshold for fuzzy matching (default: 0.6)
- **domain_dictionary**: Dictionary of domain-specific terms and their importance weights

## Statistical Tracking

The Dolphining Framework maintains statistics on corrections made, which can be accessed via:

```python
stats = corrector.get_correction_statistics()
```

Available statistics include:
- Total corrections attempted
- Successful corrections
- Corrections accepted by users
- Corrections rejected by users
- Average confidence of corrections

## Future Enhancements

1. **Advanced Phonetic Matching**: Improve candidate generation with more sophisticated phonetic algorithms
2. **Multi-language Support**: Extend the framework to handle multiple languages
3. **Real-time Adaptation**: Adjust correction parameters in real-time based on conversation dynamics
4. **Prosody Integration**: Use prosodic features (pitch, rhythm, stress) to improve disambiguation
5. **Personalized Correction Models**: Build user-specific correction models that learn individual speech patterns
