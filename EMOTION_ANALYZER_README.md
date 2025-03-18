# Emotion Analyzer

This service provides sentiment analysis and emotional classification of text using the RoBERTa model trained on the GoEmotions dataset. It can be used to enhance memory metadata with emotional context and confidence scoring.

## Overview

The Emotion Analyzer provides two interfaces:

1. **WebSocket API** - Available on port 5007
2. **REST API** - Available on port 8007

The service analyzes text input and returns detailed emotion classifications from the GoEmotions dataset (28 emotions) as well as simplified primary emotions (joy, sadness, anger, fear, surprise, neutral).

## Model

The service uses the `roberta-base-go_emotions` model located at:

```
C:\Users\danny\OneDrive\Documents\AI_Conversations\lucid-recall-dist\lucid-recall-dist\models\roberta-base-go_emotions
```

## Docker Setup

### Building and Running

To build and run the Emotion Analyzer container:

```bash
docker-compose -f docker-compose.emotions.yml up -d
```

This will start the emotion analyzer service with both WebSocket and REST API endpoints.

### Stopping the Container

```bash
docker-compose -f docker-compose.emotions.yml down
```

## API Usage

### WebSocket API

Connect to the WebSocket server at `ws://localhost:5007` and send JSON messages with the following format:

```json
{
  "type": "analyze",
  "text": "I'm feeling really happy about this progress!",
  "threshold": 0.3
}
```

The server will respond with a JSON message like:

```json
{
  "type": "analysis_result",
  "detailed_emotions": {
    "joy": 0.92,
    "optimism": 0.85,
    "excitement": 0.76
  },
  "primary_emotions": {
    "joy": 0.92,
    "sadness": 0.0,
    "anger": 0.0,
    "fear": 0.0,
    "surprise": 0.0,
    "neutral": 0.0,
    "other": 0.0
  },
  "dominant_detailed": {
    "emotion": "joy",
    "confidence": 0.92
  },
  "dominant_primary": {
    "emotion": "joy",
    "confidence": 0.92
  },
  "input_text": "I'm feeling really happy about this progress!"
}
```

### REST API

Make a POST request to `http://localhost:8007/analyze` with a JSON body:

```json
{
  "text": "I'm feeling really happy about this progress!",
  "threshold": 0.3
}
```

The response will be in the same format as the WebSocket response (without the `type` field).

## Integration with Memory System

You can integrate this emotion analyzer with your memory system by:

1. Calling the emotion analyzer when processing new memories
2. Adding the emotional data to memory metadata
3. Using the emotional confidence scores for memory significance calculation

Example Python client code:

```python
import asyncio
import websockets
import json

async def analyze_emotion(text):
    uri = "ws://localhost:5007"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "type": "analyze",
            "text": text
        }))
        response = await websocket.recv()
        return json.loads(response)

async def main():
    result = await analyze_emotion("I'm feeling excited about this new project!")
    print(json.dumps(result, indent=2))
    
    # Example: Extract dominant emotion for memory metadata
    dominant_emotion = result["dominant_primary"]["emotion"]
    confidence = result["dominant_primary"]["confidence"]
    print(f"Dominant emotion: {dominant_emotion} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing

To test the emotion analyzer, you can use the included test script:

```bash
python test_emotion_analyzer.py
```

Or use curl for the REST API:

```bash
curl -X POST "http://localhost:8007/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text":"I am feeling very happy today!"}'
```

## Emotion Classifications

The model can detect the following emotions:

### Detailed Emotions (GoEmotions)
- admiration, amusement, anger, annoyance, approval, caring, confusion
- curiosity, desire, disappointment, disapproval, disgust, embarrassment
- excitement, fear, gratitude, grief, joy, love, nervousness
- optimism, pride, realization, relief, remorse, sadness, surprise, neutral

### Primary Emotions (Simplified)
- joy, sadness, anger, fear, surprise, neutral, other
