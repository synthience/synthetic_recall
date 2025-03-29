import os
import logging
import asyncio
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

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
