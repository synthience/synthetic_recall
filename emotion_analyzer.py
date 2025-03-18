import asyncio
import json
import logging
import os
import torch
import websockets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Emotion labels from the GoEmotions dataset
GO_EMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral"
]

# Primary emotions mapping (for simplicity)
PRIMARY_EMOTIONS = {
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "pride", "admiration"],
    "sadness": ["sadness", "disappointment", "grief", "remorse"],
    "anger": ["anger", "annoyance", "disapproval", "disgust"],
    "fear": ["fear", "nervousness", "embarrassment"],
    "surprise": ["surprise", "realization", "confusion"],
    "neutral": ["neutral"],
    "other": ["caring", "curiosity", "desire", "relief"]
}

# Reverse mapping from detailed to primary emotions
DETAILED_TO_PRIMARY = {}
for primary, detailed_list in PRIMARY_EMOTIONS.items():
    for emotion in detailed_list:
        DETAILED_TO_PRIMARY[emotion] = primary

class EmotionAnalyzer:
    """Emotion analyzer using the RoBERTa model trained on GoEmotions dataset"""
    
    def __init__(self, model_path="/workspace/models/roberta-base-go_emotions"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading emotion model from {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Emotion model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            raise
    
    def analyze(self, text: str, threshold: float = 0.3) -> Dict[str, Any]:
        """Analyze text for emotions
        
        Args:
            text: The text to analyze
            threshold: Confidence threshold for multi-label classification
            
        Returns:
            Dictionary with emotion analysis results
        """
        try:
            # Tokenize and get model predictions
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Convert logits to probabilities
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Get emotions above threshold
            detailed_emotions = {}
            for i, prob in enumerate(probs):
                if prob >= threshold:
                    detailed_emotions[GO_EMOTIONS_LABELS[i]] = float(prob)
            
            # If no emotions above threshold, take the highest
            if not detailed_emotions:
                max_idx = probs.argmax()
                detailed_emotions[GO_EMOTIONS_LABELS[max_idx]] = float(probs[max_idx])
            
            # Aggregate to primary emotions
            primary_emotions = {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "neutral": 0.0, "other": 0.0}
            
            for emotion, score in detailed_emotions.items():
                primary = DETAILED_TO_PRIMARY.get(emotion, "other")
                primary_emotions[primary] = max(primary_emotions[primary], score)
            
            # Get dominant emotion
            dominant_detailed = max(detailed_emotions.items(), key=lambda x: x[1])
            dominant_primary = max(primary_emotions.items(), key=lambda x: x[1])
            
            result = {
                "detailed_emotions": detailed_emotions,
                "primary_emotions": primary_emotions,
                "dominant_detailed": {
                    "emotion": dominant_detailed[0],
                    "confidence": float(dominant_detailed[1])
                },
                "dominant_primary": {
                    "emotion": dominant_primary[0],
                    "confidence": float(dominant_primary[1])
                }
            }
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            raise

# Initialize FastAPI app
app = FastAPI(title="Emotion Analyzer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the analyzer
analyzer = None

@app.on_event("startup")
async def startup_event():
    global analyzer
    analyzer = EmotionAnalyzer()

# Request and response models
class EmotionRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.3

class EmotionResponse(BaseModel):
    detailed_emotions: Dict[str, float]
    primary_emotions: Dict[str, float]
    dominant_detailed: Dict[str, Any]
    dominant_primary: Dict[str, Any]
    input_text: str

@app.post("/analyze", response_model=EmotionResponse)
async def analyze_emotions(request: EmotionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    try:
        result = analyzer.analyze(request.text, request.threshold)
        result["input_text"] = request.text
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class WebSocketServer:
    """WebSocket server for emotion analysis"""
    
    def __init__(self, host="0.0.0.0", port=5007):
        self.host = host
        self.port = port
        self.analyzer = EmotionAnalyzer()
        logger.info(f"Initialized WebSocketServer for emotion analysis")
    
    async def handle_websocket(self, websocket):
        logger.info(f"New connection from {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message type: {data.get('type', 'unknown')}")
                    
                    if data['type'] == 'analyze':
                        text = data.get('text', '')
                        threshold = data.get('threshold', 0.3)
                        
                        if not text.strip():
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': 'Empty text provided for analysis'
                            }))
                            continue
                        
                        logger.info(f"Analyzing emotions for text: {text[:50]}..." 
                                    if len(text) > 50 else f"Analyzing emotions for text: {text}")
                        
                        try:
                            result = self.analyzer.analyze(text, threshold)
                            result['type'] = 'analysis_result'
                            result['input_text'] = text[:100] + '...' if len(text) > 100 else text
                            
                            await websocket.send(json.dumps(result))
                        except Exception as e:
                            logger.error(f"Error analyzing emotions: {e}")
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': f'Error analyzing emotions: {str(e)}'
                            }))
                    
                    elif data['type'] == 'ping':
                        logger.info("Received ping, responding with pong")
                        await websocket.send(json.dumps({
                            'type': 'pong',
                            'timestamp': int(time.time() * 1000)
                        }))
                    
                    elif data['type'] == 'health_check':
                        logger.info("Received health_check request")
                        await websocket.send(json.dumps({
                            'type': 'health_check_response',
                            'status': 'ok'
                        }))
                    
                    else:
                        logger.warning(f"Unknown message type: {data.get('type', 'unknown')}")
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': f'Unknown message type: {data.get("type", "unknown")}'
                        }))
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {message}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON'
                    }))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': f'Error processing message: {str(e)}'
                    }))
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
    
    async def start(self):
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        server = await websockets.serve(self.handle_websocket, self.host, self.port)
        await server.wait_closed()

# Run both FastAPI and WebSocket server together
async def run_servers():
    # Start the WebSocket server
    websocket_server = WebSocketServer()
    websocket_task = asyncio.create_task(websocket_server.start())
    
    # Start FastAPI with uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8007)
    server = uvicorn.Server(config)
    await server.serve()
    
    # Wait for the WebSocket server to complete (though it shouldn't)
    await websocket_task

if __name__ == "__main__":
    logger.info("Starting emotion analyzer services")
    asyncio.run(run_servers())
