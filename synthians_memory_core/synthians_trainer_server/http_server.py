# synthians_trainer_server/http_server.py

import os
import tensorflow as tf
import numpy as np
import aiohttp
import asyncio
import json
from fastapi import FastAPI, HTTPException, Body, Request, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import logging
import traceback # Import traceback
import datetime  # Add datetime module for timestamps

# Import the new Neural Memory module and config
from .neural_memory import NeuralMemoryModule, NeuralMemoryConfig

# Keep SurpriseDetector if needed for outer loop analysis
from .surprise_detector import SurpriseDetector
# Assume GeometryManager might be needed if surprise calculation uses it
try:
    from ..geometry_manager import GeometryManager
except ImportError:
    logger.warning("Could not import GeometryManager from synthians_memory_core. Using basic numpy ops.")
    class GeometryManager: # Dummy version
        def __init__(self, config=None): pass
        def normalize_embedding(self, vec):
            vec = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        def calculate_similarity(self, v1, v2):
             v1 = self.normalize_embedding(v1)
             v2 = self.normalize_embedding(v2)
             return np.dot(v1, v2)
        def align_vectors(self, v1, v2):
             v1, v2 = np.array(v1), np.array(v2)
             if v1.shape == v2.shape: return v1, v2
             logger.warning("Dummy GeometryManager cannot align vectors.")
             return v1, v2 # Assume they match or fail later


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Synthians Neural Memory API (Titans)")

# --- Global State ---
neural_memory: Optional[NeuralMemoryModule] = None
surprise_detector: Optional[SurpriseDetector] = None
geometry_manager: Optional[GeometryManager] = None
memory_core_url: Optional[str] = None # URL for potential outer loop callbacks

# --- Pydantic Models ---

class InitRequest(BaseModel):
    config: Optional[dict] = Field(default_factory=dict, description="Neural Memory config overrides")
    memory_core_url: Optional[str] = None
    load_path: Optional[str] = None

class InitResponse(BaseModel):
    message: str
    config: dict # Return as dict for JSON

class RetrieveRequest(BaseModel):
    query_embedding: List[float]

class RetrieveResponse(BaseModel):
    retrieved_embedding: List[float]

class UpdateMemoryRequest(BaseModel):
    input_embedding: List[float]

class UpdateMemoryResponse(BaseModel):
    status: str
    loss: Optional[float] = None
    grad_norm: Optional[float] = None

class TrainOuterRequest(BaseModel):
    input_sequence: List[List[float]]
    target_sequence: List[List[float]]

class TrainOuterResponse(BaseModel):
    average_loss: float

class SaveLoadRequest(BaseModel):
    path: str

class StatusResponse(BaseModel):
     status: str
     config: Optional[dict] = None # Return as dict

class AnalyzeSurpriseRequest(BaseModel):
    predicted_embedding: List[float]
    actual_embedding: List[float]

# --- Helper Functions ---

def get_neural_memory() -> NeuralMemoryModule:
    if neural_memory is None:
        logger.error("Neural Memory module not initialized. Call /init first.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Neural Memory module not initialized.")
    return neural_memory

def get_surprise_detector() -> SurpriseDetector:
     global surprise_detector, geometry_manager
     if surprise_detector is None:
          if geometry_manager is None:
               nm_conf = neural_memory.config if neural_memory else NeuralMemoryConfig()
               # Use get with default for safety
               gm_dim = nm_conf.get('input_dim', 768)
               geometry_manager = GeometryManager({'embedding_dim': gm_dim})
          surprise_detector = SurpriseDetector(geometry_manager=geometry_manager)
          logger.info("Initialized SurpriseDetector.")
     return surprise_detector


def _validate_vector(vec: Optional[List[float]], expected_dim: int, name: str, allow_none=False):
    """Validates vector type, length, and content."""
    if vec is None:
        if allow_none: return
        else: raise HTTPException(status_code=400, detail=f"'{name}' cannot be null.")

    if not isinstance(vec, list):
         raise HTTPException(status_code=400, detail=f"'{name}' must be a list of floats.")

    # <<< MODIFIED: Explicitly handle expected_dim == -1 >>>
    if expected_dim != -1 and len(vec) != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid vector length for '{name}'. Expected {expected_dim}, got {len(vec)}."
        )
    # Add NaN/Inf check
    try:
         # Using np.isfinite is more efficient for checking both NaN and Inf
         if not np.all(np.isfinite(vec)):
             raise HTTPException(
                  status_code=400,
                  detail=f"Invalid values (NaN/Inf) found in '{name}'."
             )
    except TypeError:
          # This might happen if vec contains non-numeric types
          raise HTTPException(
               status_code=400,
               detail=f"Invalid value types in '{name}', expected floats."
          )


# --- API Endpoints ---

@app.post("/init", response_model=InitResponse, status_code=status.HTTP_200_OK)
async def init_neural_memory(req: InitRequest):
    """Initialize the Neural Memory Module."""
    global neural_memory, memory_core_url, surprise_detector, geometry_manager
    logger.info(f"Received /init request. Config overrides: {req.config}, Load path: {req.load_path}")
    try:
        # Use .get() for safer access to potentially missing keys in Pydantic model
        mc_url = req.memory_core_url
        if mc_url:
            memory_core_url = mc_url
            logger.info(f"Memory Core URL set to: {memory_core_url}")

        # Create config, overriding defaults with request body config
        # req.config should be a dict here from Pydantic parsing
        config_data = req.config if req.config is not None else {}
        config = NeuralMemoryConfig(**config_data)
        logger.info(f"Parsed config: {dict(config)}")


        # Initialize or re-initialize
        logger.info("Creating NeuralMemoryModule instance...")
        neural_memory = NeuralMemoryModule(config=config)
        logger.info("NeuralMemoryModule instance created.")

        # Initialize shared geometry manager and surprise detector based on module's config
        # Use dictionary access here too
        geometry_manager = GeometryManager({'embedding_dim': neural_memory.config['input_dim']})
        # Reset surprise detector to use new geometry manager if re-initializing
        surprise_detector = None
        get_surprise_detector() # Initialize if not already

        loaded_ok = True
        if req.load_path:
            logger.info(f"Attempting to load state from: {req.load_path}")
            # Build model before loading
            try:
                 logger.info("Building model before loading state...")
                 _ = neural_memory(tf.zeros((1, neural_memory.config['query_dim'])))
                 logger.info("Model built successfully.")
            except Exception as build_err:
                 logger.error(f"Error explicitly building model before load: {build_err}. Load might still succeed.")

            loaded_ok = neural_memory.load_state(req.load_path)
            if not loaded_ok:
                # Fail init if loading was requested but failed
                raise HTTPException(status_code=500, detail=f"Failed to load state from {req.load_path}")

        effective_config = neural_memory.get_config_dict()
        logger.info(f"Neural Memory module initialized. Effective Config: {effective_config}")
        return InitResponse(message="Neural Memory module initialized successfully.", config=effective_config)

    except AttributeError as ae:
         # Catch the specific AttributeError related to config access during init
         logger.error(f"AttributeError during initialization: {ae}. Config object: {config}", exc_info=True)
         neural_memory = None
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                             detail=f"Initialization failed due to config access error: {ae}")
    except Exception as e:
        logger.error(f"Failed to initialize Neural Memory module: {e}", exc_info=True)
        neural_memory = None # Ensure it's None on failure
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Initialization failed: {str(e)}")

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_memory(req: RetrieveRequest):
    nm = get_neural_memory()
    try:
        _validate_vector(req.query_embedding, nm.config['query_dim'], "query_embedding")
        
        # Convert to proper tensor shape expected by the model (batch dimension)
        query_tensor = tf.convert_to_tensor([req.query_embedding], dtype=tf.float32)
        
        # Call the model with the properly shaped tensor
        retrieved_tensor = nm(query_tensor, training=False)
        
        if retrieved_tensor is None or not isinstance(retrieved_tensor, tf.Tensor):
             raise ValueError("Retrieval process returned None or invalid type")

        # Ensure we get a flat list regardless of tensor shape
        if len(tf.shape(retrieved_tensor)) > 1:
             # If we get a batch of vectors (or matrix), we want the first one
             retrieved_list = retrieved_tensor[0].numpy().tolist()
        else:
             retrieved_list = retrieved_tensor.numpy().tolist()

        return RetrieveResponse(retrieved_embedding=retrieved_list)

    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

@app.post("/update_memory", response_model=UpdateMemoryResponse)
async def update_memory(req: UpdateMemoryRequest):
    nm = get_neural_memory()
    try:
        _validate_vector(req.input_embedding, nm.config['input_dim'], "input_embedding")
        
        # Create tensor with proper batch dimension as expected by TensorFlow
        input_tensor = tf.convert_to_tensor([req.input_embedding], dtype=tf.float32)

        # Pass to update_step which now expects a batched tensor with shape [1, input_dim]
        loss_tensor, grads = nm.update_step(input_tensor)

        grad_norm = 0.0
        if grads:
             valid_grads = [g for g in grads if g is not None]
             if valid_grads:
                 # Calculate L2 norm for each valid gradient tensor and sum them
                 norms = [tf.norm(g) for g in valid_grads]
                 grad_norm = tf.reduce_sum(norms).numpy().item()

        loss_value = loss_tensor.numpy().item() if loss_tensor is not None else 0.0

        # Include timestamp in response for tracking
        timestamp = datetime.datetime.now().isoformat()
        return UpdateMemoryResponse(status="success", loss=loss_value, grad_norm=grad_norm)

    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Memory update failed: {e}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")

@app.post("/train_outer", response_model=TrainOuterResponse)
async def train_outer(req: TrainOuterRequest):
    nm = get_neural_memory()
    if not hasattr(nm, 'compiled') or not nm.compiled:
        try:
             # Make sure the optimizer is properly set
             if not hasattr(nm, 'optimizer') or nm.optimizer is None:
                 nm.optimizer = nm.outer_optimizer
             nm.compile(optimizer=nm.optimizer, loss='mse')
             logger.info("NeuralMemoryModule compiled for outer training.")
        except Exception as compile_err:
             logger.error(f"Error compiling NeuralMemoryModule: {compile_err}")
             raise HTTPException(status_code=500, detail=f"Model compilation error: {compile_err}")

    try:
        if not req.input_sequence or not req.target_sequence: raise HTTPException(status_code=400, detail="Sequences empty.")
        seq_len = len(req.input_sequence)
        if seq_len != len(req.target_sequence): raise HTTPException(status_code=400, detail="Sequence lengths mismatch.")
        if seq_len == 0: raise HTTPException(status_code=400, detail="Sequences length 0.")

        # Validate dimensions for first item in sequences
        _validate_vector(req.input_sequence[0], nm.config['input_dim'], "input_sequence[0]")
        _validate_vector(req.target_sequence[0], nm.config['value_dim'], "target_sequence[0]")

        # Convert to tensors with proper shape: [batch_size=1, seq_len, dim]
        input_seq_tensor = tf.convert_to_tensor([req.input_sequence], dtype=tf.float32)
        target_seq_tensor = tf.convert_to_tensor([req.target_sequence], dtype=tf.float32)

        # Log tensor shapes for debugging
        logger.info(f"Input sequence tensor shape: {input_seq_tensor.shape}, Target sequence tensor shape: {target_seq_tensor.shape}")
        
        # Directly call train_step with the properly shaped tensors
        metrics = nm.train_step((input_seq_tensor, target_seq_tensor))
        avg_loss = metrics.get('loss', 0.0)
        
        # Ensure we return a Python native float
        return TrainOuterResponse(average_loss=float(avg_loss))

    except HTTPException as http_exc: raise http_exc
    except tf.errors.InvalidArgumentError as tf_err:
         logger.error(f"TensorFlow argument error during outer training: {tf_err}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"TF Argument Error: {tf_err}")
    except Exception as e:
        logger.error(f"Outer training failed: {e}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Outer training error: {str(e)}")

@app.post("/save", status_code=status.HTTP_200_OK)
async def save_neural_memory_state(req: SaveLoadRequest):
    nm = get_neural_memory()
    try:
        nm.save_state(req.path)
        return {"message": f"Neural Memory state saved to {req.path}"}
    except Exception as e:
        logger.error(f"Failed to save neural memory state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save state: {str(e)}")

@app.post("/load", status_code=status.HTTP_200_OK)
async def load_neural_memory_state(req: SaveLoadRequest):
    global neural_memory, surprise_detector, geometry_manager
    try:
        # First, read the state file to examine the config without loading
        if not os.path.exists(req.path):
            raise FileNotFoundError(f"State file not found: {req.path}")
            
        with open(req.path, 'r') as f: 
            state_data = json.load(f)
            
        # Extract config from saved state
        saved_config = state_data.get("config")
        if not saved_config:
            raise ValueError("State file is missing 'config' section")
        
        # Create a properly initialized model with the saved config
        temp_nm = NeuralMemoryModule(config=saved_config)
        
        # Force build by creating dummy inputs and running a forward pass
        dummy_input = tf.zeros((1, temp_nm.config['input_dim']), dtype=tf.float32)
        dummy_query = tf.zeros((1, temp_nm.config['query_dim']), dtype=tf.float32)
        _ = temp_nm.get_projections(dummy_input)
        _ = temp_nm(dummy_query)
        
        # Now load the state into the fully initialized model with matching config
        loaded_ok = temp_nm.load_state(req.path)

        if loaded_ok:
            # Replace the global instance with our successfully loaded one
            neural_memory = temp_nm
            # Re-initialize dependent components with the loaded config
            geometry_manager = GeometryManager({'embedding_dim': neural_memory.config['input_dim']})
            surprise_detector = None  # Force re-init with new geometry manager
            get_surprise_detector()
            logger.info(f"Neural Memory state loaded from {req.path} and components re-initialized.")
            return {"message": f"Neural Memory state loaded from {req.path}"}
        else:
             raise HTTPException(status_code=500, detail=f"Failed to load state from {req.path}. Check logs.")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"State file not found: {req.path}")
    except Exception as e:
        logger.error(f"Failed to load neural memory state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load state: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_neural_memory_status():
    if neural_memory is None:
        return StatusResponse(status="Neural Memory module not initialized.")
    try:
        config_dict = neural_memory.get_config_dict()
        return StatusResponse(status="Initialized", config=config_dict)
    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.post("/analyze_surprise", response_model=Dict[str, Any])
async def analyze_surprise(request: AnalyzeSurpriseRequest):
    detector = get_surprise_detector()
    nm = get_neural_memory() # Need this for dimension info
    try:
        # Validate embeddings using input_dim from the initialized model
        _validate_vector(request.predicted_embedding, nm.config['input_dim'], "predicted_embedding")
        _validate_vector(request.actual_embedding, nm.config['input_dim'], "actual_embedding")

        surprise_metrics = detector.calculate_surprise(
            predicted_embedding=request.predicted_embedding,
            actual_embedding=request.actual_embedding
        )
        quickrecal_boost = detector.calculate_quickrecal_boost(surprise_metrics)

        response_data = surprise_metrics.copy()
        if 'delta' in response_data and isinstance(response_data['delta'], np.ndarray):
             response_data['delta'] = response_data['delta'].tolist()
        response_data["quickrecal_boost"] = quickrecal_boost

        return response_data

    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Error analyzing surprise: {e}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing surprise: {str(e)}")

# --- Health Check ---
@app.get("/health")
async def health_check():
    """Basic health check."""
    try:
         tf_version = tf.__version__
         # Perform a minimal TF computation
         tensor_sum = tf.reduce_sum(tf.constant([1.0, 2.0])).numpy()
         can_compute = abs(tensor_sum - 3.0) < 1e-6
         status_msg = "ok" if can_compute else "error_tf_compute"
    except Exception as e:
         logger.error(f"TensorFlow health check failed: {e}", exc_info=True)
         tf_version = "error"
         status_msg = f"error_tf_init: {str(e)}"

    return {
         "status": status_msg,
         "tensorflow_version": tf_version,
         "neural_memory_initialized": neural_memory is not None,
         "timestamp": datetime.datetime.utcnow().isoformat() 
     }

# --- App startup/shutdown ---
@app.on_event("startup")
async def startup_event():
     global geometry_manager
     if geometry_manager is None:
         geometry_manager = GeometryManager()
     logger.info("Synthians Neural Memory API started. Send POST to /init to initialize.")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down neural memory server.")
    # if neural_memory:
    #     try:
    #         save_path = os.environ.get("SHUTDOWN_SAVE_PATH", "/app/memory/shutdown_state.json")
    #         logger.info(f"Attempting final state save to {save_path}")
    #         neural_memory.save_state(save_path)
    #     except Exception as e:
    #         logger.error(f"Error saving state on shutdown: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    logger.info(f"Starting Synthians Neural Memory API on http://{host}:{port}")
    print(f"-> Using TensorFlow version: {tf.__version__}")
    print(f"-> Using NumPy version: {np.__version__}")
    if not np.__version__.startswith("1."):
        print("\n\n!!!! WARNING: Numpy version is not < 2.0.0. This may cause issues with TensorFlow/other libs. !!!!\n\n")

    uvicorn.run(app, host=host, port=port, log_level=log_level) # Run using the app object directly