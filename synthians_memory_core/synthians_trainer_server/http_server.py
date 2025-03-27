import os
import tensorflow as tf
import numpy as np
import aiohttp
import asyncio
import json
from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel # Import BaseModel
from typing import List, Dict, Any, Optional
import logging
import requests
import json
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Request, Response, status, BackgroundTasks, Depends
from pydantic import BaseModel
from .synthians_trainer import SynthiansSequencePredictor, TitanMemoryConfig as TrainerConfig, SynthiansSequenceTrainer
from .surprise_detector import SurpriseDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rename app title
app = FastAPI(title="Synthians Sequence Trainer API")

# Global state for the trainer model and its memory vector
trainer_model: Optional[SynthiansSequencePredictor] = None
trainer_memory_vec: Optional[tf.Variable] = None
surprise_detector: Optional[SurpriseDetector] = None

# Memory Core API connection details
memory_core_url: Optional[str] = None

# --- Pydantic Models ---
# Update models slightly to match terminology if desired, but structure is similar

class InitConfig(TrainerConfig): # Keep using TrainerConfig structure
    memory_core_url: Optional[str] = None

class InitResponse(BaseModel): # Use BaseModel
    message: str
    config: TrainerConfig # Return the trainer's config

class TrainStepRequest(BaseModel): # Use BaseModel
    x_t: List[float]
    x_next: List[float]

class TrainStepResponse(BaseModel): # Use BaseModel
    cost: float
    predicted: List[float] # Prediction for x_{t+1}
    surprise: float # Surprise based on x_t reconstruction

class ForwardRequest(BaseModel): # Use BaseModel
    x: List[float]

class ForwardResponse(BaseModel): # Use BaseModel
    predicted: List[float] # Prediction for next step
    memory: List[float]    # The *new* internal memory state M_t
    surprise: float      # Surprise based on reconstructing input x

class SaveLoadRequest(BaseModel): # Use BaseModel
    path: str

class StatusResponse(TrainerConfig): # Inherit directly
     status: Optional[str] = None # Add status field for not-initialized case

class PredictNextEmbeddingRequest(BaseModel):
    embedding: List[float]
    previous_memory_state: Optional[List[float]] = None

class PredictNextEmbeddingResponse(BaseModel):
    predicted_embedding: List[float]
    surprise: float
    memory_state: Optional[List[float]] = None

# --- Helper Functions ---
def _get_trainer_model_and_memory():
    global trainer_model, trainer_memory_vec, surprise_detector, memory_core_url
    if trainer_model is None:
        # Initialize the predictor model
        logger.info("Initializing sequence predictor...")
        trainer_model = SynthiansSequencePredictor(config=TrainerConfig())
    
    if trainer_memory_vec is None:
        # Initialize the trainer memory vector
        logger.info("Initializing sequence trainer memory...")
        cfg = trainer_model.get_config()
        mem_dim = cfg.get('outputDim', 256) # Use 'outputDim' from config which maps to memory_dim
        initial_memory = tf.zeros([mem_dim], dtype=tf.float32)
        trainer_memory_vec = tf.Variable(initial_memory, name="trainer_memory_vec", trainable=False)
        
    if surprise_detector is None:
        # Initialize the surprise detector
        logger.info("Initializing surprise detector...")
        surprise_detector = SurpriseDetector(
            surprise_threshold=0.6,
            max_sequence_length=10,
            surprise_decay=0.9
        )
    
    # Get memory core URL from environment variable if not already set
    if memory_core_url is None:
        memory_core_url = os.environ.get("MEMORY_CORE_URL", "http://localhost:8000")
        logger.info(f"Memory Core URL set to: {memory_core_url}")
        
    return trainer_model, trainer_memory_vec, surprise_detector, memory_core_url

def _validate_vector(vec: List[float], expected_dim: int, name: str):
    if len(vec) != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid vector length for '{name}'. Expected {expected_dim}, got {len(vec)}."
        )
        
async def _fetch_from_memory_core(endpoint: str, payload: dict = None):
    if memory_core_url is None:
        raise HTTPException(status_code=400, detail="Memory Core URL not configured. Call /init with memory_core_url.")
    
    url = f"{memory_core_url}{endpoint}"
    method = "GET" if payload is None else "POST"
    
    try:
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HTTPException(status_code=response.status, detail=f"Memory Core API error: {error_text}")
                    return await response.json()
            else:  # POST
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HTTPException(status_code=response.status, detail=f"Memory Core API error: {error_text}")
                    return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to Memory Core API: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to Memory Core API: {str(e)}")

# --- API Endpoints ---

@app.post("/init", response_model=InitResponse)
async def init_trainer_model(config: InitConfig = Body(default_factory=dict)):
    global trainer_model, trainer_memory_vec, surprise_detector, memory_core_url
    logger.info(f"Initializing sequence trainer model with config: {config}")
    try:
        # Set Memory Core URL if provided
        if config.memory_core_url:
            memory_core_url = config.memory_core_url
            logger.info(f"Memory Core URL set to: {memory_core_url}")
        
        trainer_model = SynthiansSequencePredictor(config) # Instantiate new class
        cfg = trainer_model.get_config()
        mem_dim = cfg.get('outputDim', 256) # Use 'outputDim' from config which maps to memory_dim
        initial_memory = tf.zeros([mem_dim], dtype=tf.float32)
        trainer_memory_vec = tf.Variable(initial_memory, name="trainer_memory_vec", trainable=False)
        surprise_detector = SurpriseDetector(
            surprise_threshold=0.6,
            max_sequence_length=10,
            surprise_decay=0.9
        )
        logger.info(f"Sequence trainer initialized with effective config: {cfg}")
        return InitResponse(message="Sequence trainer model initialized", config=cfg)
    except Exception as e:
        logger.error(f"Failed to initialize trainer model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize trainer model: {str(e)}")

@app.post("/trainStep", response_model=TrainStepResponse)
async def train_step(data: TrainStepRequest):
    m, mem_var, _, _ = _get_trainer_model_and_memory()
    cfg = m.get_config()
    input_dim = cfg.get('inputDim', 768)

    try:
        _validate_vector(data.x_t, input_dim, "x_t")
        _validate_vector(data.x_next, input_dim, "x_next")

        x_t_tensor = tf.convert_to_tensor(data.x_t, dtype=tf.float32)
        x_next_tensor = tf.convert_to_tensor(data.x_next, dtype=tf.float32)

        # Train step updates mem_var internally
        cost_tensor = m.train_step(x_t_tensor, x_next_tensor, mem_var)

        # Get results *after* the step for the response
        # The surprise here is based on reconstructing x_t
        forward_result = m.forward(x_t_tensor, mem_var.value())

        cost_val = float(cost_tensor.numpy())
        pred_val = forward_result["predicted"].numpy().tolist() # Prediction of x_{t+1}
        sur_val = float(forward_result["surprise"].numpy()) # Reconstruction surprise of x_t

        return TrainStepResponse(cost=cost_val, predicted=pred_val, surprise=sur_val)

    except Exception as e:
        logger.error(f"Trainer train step failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Trainer train step failed: {str(e)}")

@app.post("/forward", response_model=ForwardResponse)
async def forward_pass(data: ForwardRequest):
    m, mem_var, _, _ = _get_trainer_model_and_memory()
    cfg = m.get_config()
    input_dim = cfg.get('inputDim', 768)

    try:
        _validate_vector(data.x, input_dim, "x")
        x_tensor = tf.convert_to_tensor(data.x, dtype=tf.float32)

        # Run forward pass using the current memory state (M_{t-1})
        forward_result = m.forward(x_tensor, mem_var.value())

        new_memory_tensor = forward_result["newMemory"] # This is M_t
        # Update the global trainer memory state
        mem_var.assign(new_memory_tensor)

        pred_val = forward_result["predicted"].numpy().tolist() # Prediction P_t for x_{t+1}
        mem_val = new_memory_tensor.numpy().tolist()       # Updated memory M_t
        sur_val = float(forward_result["surprise"].numpy())  # Surprise for x_t

        return ForwardResponse(predicted=pred_val, memory=mem_val, surprise=sur_val)

    except Exception as e:
        logger.error(f"Trainer forward pass failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Trainer forward pass failed: {str(e)}")

@app.post("/save")
async def save_trainer_model(data: SaveLoadRequest):
    m, _, _, _ = _get_trainer_model_and_memory()
    try:
        await m.save_model(data.path)
        return {"message": f"Trainer model saved to {data.path}"}
    except Exception as e:
        logger.error(f"Failed to save trainer model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save trainer model: {str(e)}")

@app.post("/load")
async def load_trainer_model(data: SaveLoadRequest):
    # Ensure model exists to load into. If not, init first?
    # For simplicity, assume init was called before load.
    global trainer_model, trainer_memory_vec
    if trainer_model is None:
         # Optionally, initialize with default config if loading into non-existent model
         logger.warning("Attempting to load into non-initialized model. Initializing with defaults.")
         trainer_model = SynthiansSequencePredictor()
         # Need to initialize memory_vec too
         cfg = trainer_model.get_config()
         mem_dim = cfg.get('outputDim', 256)
         trainer_memory_vec = tf.Variable(tf.zeros([mem_dim], dtype=tf.float32), name="trainer_memory_vec")


    m, mem_var, _, _ = _get_trainer_model_and_memory()
    try:
        await m.load_model(data.path)
        # Check if memory dimension changed and update trainer_memory_vec if needed
        new_cfg = m.get_config()
        new_mem_dim = new_cfg.get('outputDim', 256)
        if new_mem_dim != mem_var.shape[0]:
            logger.warning(f"Trainer memory dimension changed after load ({mem_var.shape[0]} -> {new_mem_dim}). Re-initializing memory vector.")
            mem_var.assign(tf.zeros([new_mem_dim], dtype=tf.float32))

        logger.info(f"Trainer model weights loaded from {data.path}")
        return {"message": f"Trainer model loaded from {data.path}"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Trainer model file not found: {data.path}")
    except Exception as e:
        logger.error(f"Failed to load trainer model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load trainer model: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_trainer_status():
    if trainer_model is None:
        return StatusResponse(status="No trainer model initialized")
    try:
        config = trainer_model.get_config()
        return StatusResponse(**config) # Directly return config dict
    except Exception as e:
        logger.error(f"Failed to get trainer status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get trainer status: {str(e)}")

@app.post("/predict_next_embedding", response_model=PredictNextEmbeddingResponse)
async def predict_next_embedding(request: PredictNextEmbeddingRequest):
    """Predict the next embedding based on the given input embedding.
    
    This endpoint is designed to be stateless when used with previous_memory_state.
    The Orchestrator manages state continuity by passing and receiving memory state.
    
    Args:
        request: Contains the current embedding to predict from and optional previous memory state
    """
    m, _, surprise_detector, _ = _get_trainer_model_and_memory()
    cfg = m.get_config()
    
    # Get and validate input embedding
    embedding = request.embedding
    input_dim = cfg.get("inputDim", 768)  # Default to 768-dimensional embeddings
    _validate_vector(embedding, input_dim, "Input embedding")
    
    # Convert embedding to tensor
    input_tensor = tf.convert_to_tensor([embedding], dtype=tf.float32)
    
    # If previous_memory_state provided, use it for stateless operation
    if request.previous_memory_state is not None:
        try:
            memory_tensor = tf.convert_to_tensor(request.previous_memory_state, dtype=tf.float32)
            # Forward pass with external memory state
            predicted_embedding, new_memory_tensor = m.forward(input_tensor, memory_tensor)
        except Exception as e:
            logger.error(f"Error using provided memory state: {e}", exc_info=True)
            # Fallback to default forward pass
            predicted_embedding, new_memory_tensor = m.forward(input_tensor)
    else:
        # First call, initialize with zeros
        memory_dim = cfg.get("memoryDim", 128)
        zero_memory = tf.zeros([1, memory_dim], dtype=tf.float32)
        predicted_embedding, new_memory_tensor = m.forward(input_tensor, zero_memory)
    
    # Extract the prediction (first item in batch)
    predicted = predicted_embedding[0].numpy().tolist()
    
    # Calculate surprise as 0 since we don't have actual to compare
    surprise_value = 0.0
    
    # Return prediction with new memory state
    # The orchestrator will handle passing this state to the next call
    return PredictNextEmbeddingResponse(
        predicted_embedding=predicted,
        surprise=surprise_value,
        memory_state=new_memory_tensor[0].numpy().tolist()
    )

@app.post("/analyze_surprise", response_model=Dict[str, Any])
async def analyze_surprise(request: dict):
    """Analyze the surprise between predicted and actual embeddings.
    
    This endpoint provides detailed metrics about how surprising an actual embedding
    is compared to what was predicted, useful for cognitive modeling.
    """
    _, _, surprise_detector, _ = _get_trainer_model_and_memory()
    
    # Extract embeddings from request
    predicted_embedding = request.get("predicted_embedding", [])
    actual_embedding = request.get("actual_embedding", [])
    
    if not predicted_embedding or not actual_embedding:
        raise HTTPException(status_code=400, detail="Both predicted_embedding and actual_embedding are required")
    
    # Calculate surprise metrics
    surprise_metrics = surprise_detector.calculate_surprise(
        predicted_embedding=predicted_embedding,
        actual_embedding=actual_embedding
    )
    
    # Calculate quickrecal boost
    quickrecal_boost = surprise_detector.calculate_quickrecal_boost(surprise_metrics)
    
    # Add boost to response
    surprise_metrics["quickrecal_boost"] = quickrecal_boost
    
    return surprise_metrics

# --- App startup/shutdown ---
@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down sequence trainer server")
    # Clean up resources here if needed

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)