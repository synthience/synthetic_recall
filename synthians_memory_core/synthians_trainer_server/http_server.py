# synthians_trainer_server/http_server.py

import os
import tensorflow as tf
import numpy as np
import aiohttp
import asyncio
import json
from fastapi import FastAPI, HTTPException, Body, Request, status, Response
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple, Literal
import logging
import traceback # Import traceback
import datetime  # Add datetime module for timestamps

# Import the new Neural Memory module and config
from .neural_memory import NeuralMemoryModule, NeuralMemoryConfig

# Import the new MetricsStore for cognitive flow instrumentation
from .metrics_store import MetricsStore, get_metrics_store

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

class GetProjectionsRequest(BaseModel):
    input_embedding: List[float] = Field(..., description="The raw input embedding vector")
    embedding_model: str = Field(default="unknown", example="sentence-transformers/all-mpnet-base-v2")
    projection_adapter: Optional[str] = Field(default="identity")

class GetProjectionsResponse(BaseModel):
    input_embedding_norm: float
    projection_adapter_used: str
    key_projection: List[float]
    value_projection: List[float]
    query_projection: List[float]
    projection_metadata: dict

class ClusterHotspot(BaseModel):
    cluster_id: str
    updates: int

class DiagnoseEmoLoopResponse(BaseModel):
    diagnostic_window: str
    avg_loss: float
    avg_grad_norm: float
    avg_quickrecal_boost: float
    dominant_emotions_boosted: List[str]
    emotional_entropy: float
    emotion_bias_index: float
    user_emotion_match_rate: float
    cluster_update_hotspots: List[ClusterHotspot]
    alerts: List[str]
    recommendations: List[str]

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
        # Convert to tensor format
        query_tensor = tf.convert_to_tensor([req.query_embedding], dtype=tf.float32)
        
        # Get projected query vector using NeuralMemoryModule's get_projections
        logger.info(f"Getting projections for query with shape {query_tensor.shape}")
        k_t, v_t, q_t = nm.get_projections(query_tensor)
        
        # Log debug information about shapes and config
        logger.info(f"DEBUG: Shape of k_t: {tf.shape(k_t).numpy()}, Shape of q_t: {tf.shape(q_t).numpy()}")
        logger.info(f"DEBUG: Config - query_dim={nm.config['query_dim']}, key_dim={nm.config['key_dim']}")
        
        # Check for dimension mismatch in configuration
        if nm.config['query_dim'] != nm.config['key_dim']:
            logger.warning(f"Configuration error detected! Using k_t (key projection) instead of q_t because q_t has incorrect dimensions")
            # Use k_t which is already at key_dim (128) dimensionality
            input_tensor = k_t
        else:
            # Configuration is correct, use q_t as intended
            input_tensor = q_t
            
        # Use the properly dimensioned tensor for memory retrieval
        retrieved_tensor = nm(input_tensor, training=False)
        
        if retrieved_tensor is None or not isinstance(retrieved_tensor, tf.Tensor):
             raise ValueError("Retrieval process returned None or invalid type")

        # Ensure we get a flat list regardless of tensor shape
        if len(tf.shape(retrieved_tensor)) > 1:
             # If we get a batch of vectors (or matrix), we want the first one
             retrieved_list = retrieved_tensor[0].numpy().tolist()
        else:
             retrieved_list = retrieved_tensor.numpy().tolist()
        
        # Log the retrieval metrics
        # Note: In a real implementation, we would have more metadata about the retrieved memories
        # This is a simplified version that just logs the retrieval vector
        metrics = get_metrics_store()
        
        # Extract user emotion from request metadata if available
        user_emotion = None
        if hasattr(req, "metadata") and req.metadata and "user_emotion" in req.metadata:
            user_emotion = req.metadata["user_emotion"]
            
        # For now, we don't have full memory metadata in this endpoint
        # In a more complete implementation, we would track the actual memories retrieved
        metrics.log_retrieval(
            query_embedding=req.query_embedding,
            retrieved_memories=[{"memory_id": "synthetic_memory", "embedding": retrieved_list}],
            user_emotion=user_emotion,
            metadata={
                "timestamp": datetime.datetime.now().isoformat(),
                "embedding_dim": len(retrieved_list)
            }
        )

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
        
        # Log metrics to MetricsStore for cognitive flow monitoring
        metrics = get_metrics_store()
        metrics.log_memory_update(
            input_embedding=req.input_embedding,
            loss=loss_value,
            grad_norm=grad_norm,
            # Extract emotion if available in metadata
            emotion=req.metadata.get("emotion") if hasattr(req, "metadata") and req.metadata else None,
            metadata={
                "timestamp": timestamp,
                "input_dim": len(req.input_embedding)
            }
        )

        return UpdateMemoryResponse(
            status="success",
            loss=loss_value,
            grad_norm=grad_norm
        )

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

        # Initialize geometry manager and surprise detector based on config
        geometry_manager = GeometryManager({'embedding_dim': temp_nm.config['input_dim']})
        # Reset surprise detector to use new geometry manager if re-initializing
        surprise_detector = None
        get_surprise_detector() # Initialize if not already

        # Attempt to load state into the fully initialized model with matching config
        loaded_ok = temp_nm.load_state(req.path)

        if loaded_ok:
            # Replace the global instance with our successfully loaded one
            neural_memory = temp_nm
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
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Basic health check."""
    logger.info("Health check requested.")
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

# --- Introspection and Diagnostic Endpoints ---

@app.post("/get_projections", response_model=GetProjectionsResponse, summary="Get K/V/Q Projections")
async def get_projections_endpoint(request: GetProjectionsRequest):
    """Exposes internal K, V, Q projections for a given input embedding."""
    nm = get_neural_memory()
    try:
        _validate_vector(request.input_embedding, nm.config['input_dim'], "input_embedding")
        
        # Convert to tensor format expected by NeuralMemoryModule
        input_tensor = tf.convert_to_tensor([request.input_embedding], dtype=tf.float32)  # Add batch dim
        
        # Get projections (k_t, v_t, q_t tensors)
        k_t, v_t, q_t = nm.get_projections(input_tensor)
        
        # Ensure tensors are squeezed and converted to Python lists
        k_list = tf.squeeze(k_t).numpy().tolist()
        v_list = tf.squeeze(v_t).numpy().tolist()
        q_list = tf.squeeze(q_t).numpy().tolist()
        
        # Calculate input embedding L2 norm
        input_norm = float(np.linalg.norm(np.array(request.input_embedding, dtype=np.float32)))
        
        # Get projection matrix hash (placeholder implementation)
        proj_hash = "hash_placeholder_v1"
        if hasattr(nm, 'get_projection_hash'):
            proj_hash = nm.get_projection_hash()
        else:
            # Basic placeholder hash since the method doesn't exist yet
            # In the future, implement get_projection_hash in NeuralMemoryModule
            logger.warning("get_projection_hash not implemented, using placeholder")
            
        # Prepare the response
        response = GetProjectionsResponse(
            input_embedding_norm=input_norm,
            projection_adapter_used=request.projection_adapter or "identity",
            key_projection=k_list,
            value_projection=v_list,
            query_projection=q_list,
            projection_metadata={
                "dim_key": nm.config['key_dim'],
                "dim_value": nm.config['value_dim'],
                "dim_query": nm.config['query_dim'],
                "projection_matrix_hash": proj_hash,
                "input_dim": nm.config['input_dim'],
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
        )
        return response
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"/get_projections failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting projections: {str(e)}")


@app.get("/diagnose_emoloop", response_model=DiagnoseEmoLoopResponse, summary="Diagnose Emotional Feedback Loop Health")
async def diagnose_emoloop(window: str = "last_100", emotion_filter: Optional[str] = "all", format: Optional[str] = None):
    """Returns diagnostic metrics for the surprise->QuickRecal feedback loop.
    
    Args:
        window: Time/count window to analyze ("last_100", "last_hour", "session")
        emotion_filter: Optional emotion to filter by ("all" or specific emotion)
        format: Output format ("json" or "table" for CLI-friendly ASCII table)
    """
    # Log the parameters for future reference
    logger.info(f"Received /diagnose_emoloop request: window={window}, filter={emotion_filter}, format={format}")
    
    # Get metrics from the MetricsStore instead of using placeholder data
    metrics_store = get_metrics_store()
    diagnostics = metrics_store.get_diagnostic_metrics(window=window, emotion_filter=emotion_filter)
    
    # Create response using the real metrics data
    response = DiagnoseEmoLoopResponse(
        diagnostic_window=diagnostics["diagnostic_window"],
        avg_loss=diagnostics["avg_loss"],
        avg_grad_norm=diagnostics["avg_grad_norm"],
        avg_quickrecal_boost=diagnostics["avg_quickrecal_boost"],
        dominant_emotions_boosted=diagnostics["dominant_emotions_boosted"],
        emotional_entropy=diagnostics["emotional_entropy"],
        emotion_bias_index=diagnostics["emotion_bias_index"],
        user_emotion_match_rate=diagnostics["user_emotion_match_rate"],
        cluster_update_hotspots=[ClusterHotspot(**hotspot) for hotspot in diagnostics["cluster_update_hotspots"]],
        alerts=diagnostics["alerts"],
        recommendations=diagnostics["recommendations"]
    )
    
    # Handle table format for CLI-friendly output
    if format == "table":
        return Response(
            content=metrics_store.format_diagnostics_as_table(diagnostics),
            media_type="text/plain"
        )
    
    return response

# --- App startup/shutdown ---
@app.on_event("startup")
async def startup_event():
    global neural_memory, memory_core_url, surprise_detector, geometry_manager
    logger.info("Synthians Neural Memory API starting up...")

    # --- ADD AUTO-INITIALIZATION LOGIC ---
    try:
        logger.info("Attempting auto-initialization of Neural Memory module...")
        # Use environment variables for default config or load path if needed
        default_config_dict = {
            # Set input_dim and query_dim to match Memory Core's embedding dimension (768)
            'input_dim': 768,
            'query_dim': 768,
            'hidden_dim': 768,  # Match this too for simplicity
            'output_dim': 768   # Match this for consistency
        }
        load_path = os.environ.get("NM_DEFAULT_STATE_PATH", None)
        mc_url = os.environ.get("MEMORY_CORE_URL", "http://localhost:5010") # Get MC URL if needed

        # Create default config
        config = NeuralMemoryConfig(**default_config_dict)

        # Create the module instance
        neural_memory = NeuralMemoryModule(config=config)

        # Initialize geometry manager and surprise detector based on config
        geometry_manager = GeometryManager({'embedding_dim': neural_memory.config['input_dim']})
        # Reset surprise detector to use new geometry manager if re-initializing
        surprise_detector = None
        get_surprise_detector() # Initialize if not already

        # Attempt to load state if path specified
        if load_path:
            logger.info(f"Attempting to load default state from: {load_path}")
            # Build model before loading
            try:
                _ = neural_memory(tf.zeros((1, neural_memory.config['query_dim'])))
                logger.info("Model built successfully for auto-load.")
            except Exception as build_err:
                logger.error(f"Error building model during auto-load: {build_err}")
            loaded = neural_memory.load_state(load_path)
            if loaded:
                logger.info(f"Successfully auto-loaded state from {load_path}")
            else:
                logger.warning(f"Failed to auto-load state from {load_path}. Starting with fresh state.")

        # Set Memory Core URL if available
        if mc_url:
            memory_core_url = mc_url

        logger.info("Neural Memory module auto-initialized successfully on startup.")
        logger.info(f"Effective Config: {neural_memory.get_config_dict()}")

    except Exception as e:
        logger.error(f"CRITICAL: Auto-initialization of Neural Memory failed: {e}", exc_info=True)
        # Ensure neural_memory is None if init fails
        neural_memory = None
    # --- END AUTO-INITIALIZATION LOGIC ---

    # Original message still useful as a fallback indication
    logger.info("Synthians Neural Memory API started. Send POST to /init to reinitialize if needed.")


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