import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Import TensorFlow installer before importing other modules
from synthians_memory_core.orchestrator.tf_installer import ensure_tensorflow_installed

# Attempt TensorFlow installation at module level before importing other dependencies
enforce_tf = ensure_tensorflow_installed()
if not enforce_tf:
    logging.warning("Failed to install TensorFlow. Titans variants requiring TensorFlow may not work correctly!")

from synthians_memory_core.geometry_manager import GeometryManager
from synthians_memory_core.orchestrator.context_cascade_engine import ContextCascadeEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Context Cascade Orchestrator")

# Global instance of the orchestrator
orchestrator = None

# --- Pydantic Models ---

class ProcessMemoryRequest(BaseModel):
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class SequenceEmbeddingsRequest(BaseModel):
    topic: Optional[str] = None
    limit: int = 10
    min_quickrecal_score: Optional[float] = None

class AnalyzeSurpriseRequest(BaseModel):
    predicted_embedding: List[float]
    actual_embedding: List[float]

class SetVariantRequest(BaseModel):
    variant: str
    reset_neural_memory: bool = False
    
class MetricsRequest(BaseModel):
    limit: int = 20

class CCEStatusPayload(BaseModel):
    status: str = Field(..., description="Status of the CCE service")
    uptime: str = Field(..., description="Uptime of the CCE service")
    is_processing: bool = Field(..., description="Whether the CCE service is currently processing")
    current_variant: str = Field(..., description="Current variant of the CCE service")
    dev_mode: bool = Field(..., description="Whether the CCE service is in DevMode")

# --- Helper Functions ---

def get_orchestrator():
    """Get or initialize the context cascade orchestrator."""
    global orchestrator
    if orchestrator is None:
        # Get URLs from environment variables with updated defaults
        memory_core_url = os.environ.get("MEMORY_CORE_URL", "http://localhost:5010")  # Default to localhost:5010
        neural_memory_url = os.environ.get("NEURAL_MEMORY_URL", "http://localhost:8001")
        
        # Initialize shared geometry manager
        geometry_manager = GeometryManager()
        
        # Initialize orchestrator
        orchestrator = ContextCascadeEngine(
            memory_core_url=memory_core_url,
            neural_memory_url=neural_memory_url,
            geometry_manager=geometry_manager,
            metrics_enabled=True
        )
        logger.info(f"Orchestrator initialized with Memory Core URL: {memory_core_url}, Neural Memory URL: {neural_memory_url}")
    
    return orchestrator

# --- Endpoints ---

@app.get("/")
async def root():
    """Root endpoint returning service information."""
    return {"service": "Context Cascade Orchestrator", "status": "running"}

@app.get("/health")
async def health():
    """Health check endpoint for the CCE service.
    
    Returns basic health information including service status.
    """
    orchestrator_instance = get_orchestrator()
    status_msg = "OK" if orchestrator_instance else "INITIALIZING"
    detail = "CCE service is running" if orchestrator_instance else "Orchestrator not initialized"
    
    # Calculate an estimated uptime if possible
    uptime = 0
    if orchestrator_instance and hasattr(orchestrator_instance, 'start_time'):
        uptime = time.time() - orchestrator_instance.start_time
    
    return {
        "status": status_msg,
        "detail": detail,
        "uptime": f"{uptime // 86400}d {(uptime % 86400) // 3600}h {(uptime % 3600) // 60}m" if uptime > 0 else "unknown",
        "is_processing": getattr(orchestrator_instance, 'is_processing', False),
        "current_variant": getattr(orchestrator_instance, 'current_variant', "unknown"),
        "dev_mode": os.environ.get("CCE_DEV_MODE", "false").lower() == "true"
    }

@app.get("/config")
async def get_config():
    """Get the current CCE configuration.
    
    Returns a subset of configuration parameters that are safe to expose.
    """
    orchestrator = get_orchestrator()
    
    # Return a subset of configuration parameters that are safe to expose
    config = {
        "DEFAULT_THRESHOLD": orchestrator.default_threshold if orchestrator else 0.75,
        "CURRENT_VARIANT": orchestrator.current_variant if orchestrator else "unknown",
        "AVAILABLE_VARIANTS": orchestrator.available_variants if orchestrator else [],
        "DEV_MODE": os.environ.get("CCE_DEV_MODE", "false").lower() == "true",
        "MEMORY_CORE_URL": os.environ.get("MEMORY_CORE_URL", "http://localhost:5010"),
        "NEURAL_MEMORY_URL": os.environ.get("NEURAL_MEMORY_URL", "http://localhost:8001"),
        "METRICS_ENABLED": orchestrator.metrics_enabled if orchestrator else True,
        "MAX_METRICS_HISTORY": orchestrator.max_metrics_history if orchestrator else 100
    }
    
    return config

@app.post("/process_memory")
async def process_memory(request: ProcessMemoryRequest):
    """Process a new memory through the full cognitive pipeline.
    
    This orchestrates:
    1. Store memory in Memory Core
    2. Compare with previous prediction if available
    3. Update quickrecal scores based on surprise
    4. Generate prediction for next memory
    """
    orchestrator = get_orchestrator()
    
    try:
        result = await orchestrator.process_new_input(
            content=request.content,
            embedding=request.embedding,
            metadata=request.metadata
        )
        return result
    except Exception as e:
        logger.error(f"Error processing memory: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing memory: {str(e)}")

@app.post("/get_sequence_embeddings")
async def get_sequence_embeddings(request: SequenceEmbeddingsRequest):
    """Retrieve a sequence of embeddings from Memory Core."""
    orchestrator = get_orchestrator()
    
    try:
        result = await orchestrator.get_sequence_embeddings(
            topic=request.topic,
            limit=request.limit,
            min_quickrecal_score=request.min_quickrecal_score
        )
        return result
    except Exception as e:
        logger.error(f"Error retrieving sequence embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving sequence embeddings: {str(e)}")

@app.post("/analyze_surprise")
async def analyze_surprise(request: AnalyzeSurpriseRequest):
    """Analyze surprise between predicted and actual embeddings."""
    orchestrator = get_orchestrator()
    
    try:
        # Use the surprise detector from the orchestrator
        surprise_metrics = orchestrator.surprise_detector.calculate_surprise(
            predicted_embedding=request.predicted_embedding,
            actual_embedding=request.actual_embedding
        )
        
        # Calculate quickrecal boost
        quickrecal_boost = orchestrator.surprise_detector.calculate_quickrecal_boost(surprise_metrics)
        
        # Add boost to response
        surprise_metrics["quickrecal_boost"] = quickrecal_boost
        
        return surprise_metrics
    except Exception as e:
        logger.error(f"Error analyzing surprise: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing surprise: {str(e)}")

@app.post("/set_variant")
async def set_variant(request: SetVariantRequest):
    """Set the active Titans variant at runtime. Only available in DevMode.
    
    This endpoint allows dynamic switching between TITANS variants during runtime.
    It requires the CCE_DEV_MODE environment variable to be set to "true".
    
    Args:
        request: Request body containing the variant to switch to
        
    Returns:
        Dict containing the switch result and status information
        
    Raises:
        HTTPException: If DevMode is not enabled, variant is invalid, or switching during processing
    """
    try:
        # Ensure orchestrator is initialized
        orchestrator = get_orchestrator()
        
        # Call the orchestrator's set_variant method
        result = await orchestrator.set_variant(request.variant, reset_neural_memory=request.reset_neural_memory)
        return result
    except RuntimeError as e:
        # DevMode not enabled or processing lock held
        logger.error(f"Runtime error in set_variant: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        # Invalid variant name
        logger.error(f"Value error in set_variant: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error in set_variant: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/status")
async def get_status():
    """Get the current status of the CCE service."""
    orchestrator_instance = get_orchestrator()
    
    try:
        # Calculate an estimated uptime if possible
        uptime = 0
        if orchestrator_instance and hasattr(orchestrator_instance, 'start_time'):
            uptime = time.time() - orchestrator_instance.start_time
        
        status = CCEStatusPayload(
            status="OK" if orchestrator_instance else "INITIALIZING",
            uptime=f"{uptime // 86400}d {(uptime % 86400) // 3600}h {(uptime % 3600) // 60}m" if uptime > 0 else "unknown",
            is_processing=getattr(orchestrator_instance, 'is_processing', False),
            current_variant=getattr(orchestrator_instance, 'current_variant', "unknown"),
            dev_mode=os.environ.get("CCE_DEV_MODE", "false").lower() == "true"
        )
        
        return status
    except Exception as e:
        logger.error(f"Error retrieving status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")

@app.get("/metrics/recent_cce_responses")
async def get_recent_cce_responses(request: MetricsRequest = None):
    """Retrieve recent CCE responses metrics.
    
    Returns detailed metrics about recent CCE operations, including:
    - Response timings
    - Variant selection decisions
    - LLM guidance details
    - Performance profiles
    """
    orchestrator_instance = get_orchestrator()
    
    if request is None:
        request = MetricsRequest()
    
    try:
        # CRITICAL: Add await here for the coroutine
        metrics = await orchestrator_instance.get_recent_metrics(limit=request.limit)
        return {
            "success": True,
            "metrics": metrics,
            "count": len(metrics) if metrics else 0,
            "limit": request.limit
        }
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "metrics": [],
            "count": 0
        }

# --- Startup and Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup."""
    get_orchestrator()
    logger.info("Context Cascade Orchestrator is ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Context Cascade Orchestrator")
