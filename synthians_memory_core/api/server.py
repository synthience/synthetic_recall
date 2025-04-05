# synthians_memory_core/api/server.py

import asyncio
import os
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Path, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
import json
from datetime import datetime
import sys
import importlib.util
import subprocess

# Import the unified memory core
from synthians_memory_core import SynthiansMemoryCore
from synthians_memory_core.custom_logger import logger
from synthians_memory_core.emotion_analyzer import EmotionAnalyzer
from synthians_memory_core.utils.transcription_feature_extractor import TranscriptionFeatureExtractor
from synthians_memory_core.interruption import InterruptionAwareMemoryHandler
from synthians_memory_core.memory_core.trainer_integration import TrainerIntegrationManager, SequenceEmbeddingsResponse, UpdateQuickRecalScoreRequest

from sentence_transformers import SentenceTransformer

# Import the new explainability routes
from synthians_memory_core.api.explainability_routes import router as explainability_router
from synthians_memory_core.api.diagnostics_routes import router as diagnostics_router

# Check for an environment variable to enable test endpoints
TEST_ENDPOINTS_ENABLED = os.environ.get("ENABLE_TEST_ENDPOINTS", "false").lower() == "true"

if TEST_ENDPOINTS_ENABLED:
    logger.warning("!!! TEST ENDPOINTS ENABLED - DO NOT USE IN PRODUCTION !!!")

# Define request/response models using Pydantic
class ProcessMemoryRequest(BaseModel):
    """Request model for processing a new memory."""
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    analyze_emotion: Optional[bool] = Field(default=True, description="Whether to analyze emotions in the content")

class ProcessMemoryResponse(BaseModel):
    """Response model for memory processing."""
    success: bool
    memory_id: Optional[str] = None
    quickrecal_score: Optional[float] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class RetrieveMemoriesRequest(BaseModel):
    query: str
    query_embedding: Optional[List[float]] = None
    top_k: int = 5
    user_emotion: Optional[Union[Dict[str, Any], str]] = None
    cognitive_load: float = 0.5
    threshold: Optional[float] = None

class RetrieveMemoriesResponse(BaseModel):
    success: bool
    memories: List[Dict[str, Any]] = []
    error: Optional[str] = None

class GenerateEmbeddingRequest(BaseModel):
    text: str

class GenerateEmbeddingResponse(BaseModel):
    success: bool
    embedding: Optional[List[float]] = None
    dimension: Optional[int] = None
    error: Optional[str] = None

class QuickRecalRequest(BaseModel):
    embedding: Optional[List[float]] = None
    text: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class QuickRecalResponse(BaseModel):
    success: bool
    quickrecal_score: Optional[float] = None
    factors: Optional[Dict[str, float]] = None
    error: Optional[str] = None

class EmotionRequest(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    success: bool
    emotions: Optional[Dict[str, float]] = None
    dominant_emotion: Optional[str] = None
    error: Optional[str] = None

class FeedbackRequest(BaseModel):
    memory_id: str
    similarity_score: float
    was_relevant: bool

class FeedbackResponse(BaseModel):
    success: bool
    new_threshold: Optional[float] = None
    error: Optional[str] = None

# Models for the transcription endpoint
class TranscriptionRequest(BaseModel):
    """Request model for processing transcription data."""
    text: str = Field(..., description="The transcribed text")
    audio_metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata about the audio source")
    embedding: Optional[List[float]] = Field(None, description="Optional pre-computed embedding for the transcription")
    memory_id: Optional[str] = Field(None, description="Optional memory ID if updating an existing memory")
    importance: Optional[float] = Field(None, description="Optional importance score for the memory (0-1)")
    force_update: bool = Field(False, description="Force update if memory ID exists")

class TranscriptionResponse(BaseModel):
    """Response model for processed transcription data."""
    success: bool = Field(..., description="Whether the operation was successful")
    memory_id: Optional[str] = Field(None, description="ID of the created/updated memory")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extracted metadata from the transcription")
    embedding: Optional[List[float]] = Field(None, description="Embedding generated for the transcription")
    error: Optional[str] = Field(None, description="Error message if operation failed")

class GetMemoryResponse(BaseModel):
    """Response model for memory retrieval."""
    success: bool
    memory: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# App lifespan for initialization/cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup app resources."""
    # Startup Logic
    logger.info("API", "Starting Synthians Memory Core API server...")
    
    # Set startup time
    app.state.startup_time = time.time()
    
    # Run GPU setup script to detect GPU and install appropriate FAISS package
    try:
        logger.info("API", "Checking for GPU availability and setting up FAISS...")
        # Get the path to gpu_setup.py
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gpu_setup_path = os.path.join(current_dir, "gpu_setup.py")
        
        if os.path.exists(gpu_setup_path):
            logger.info("API", f"Running GPU setup script from: {gpu_setup_path}")
            # Run the setup script as a subprocess
            result = subprocess.run([sys.executable, gpu_setup_path], 
                                    capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                logger.info("API", f"GPU setup completed successfully: {result.stdout.strip()}")
            else:
                logger.warning("API", f"GPU setup failed: {result.stderr.strip()}")
                logger.info("API", "Continuing with CPU-only FAISS")
        else:
            logger.warning("API", f"GPU setup script not found at {gpu_setup_path}")
    except Exception as e:
        logger.error("API", f"Error during GPU setup: {str(e)}")
        logger.info("API", "Continuing with CPU-only FAISS")
    
    # Create core instance on startup
    app.state.memory_core = SynthiansMemoryCore()
    await app.state.memory_core.initialize()
    
    # Mount routers conditionally based on feature flags
    def mount_conditional_routers():
        # Always mount essential routers
        
        # Conditionally mount explainability and diagnostics routers
        memory_core = app.state.memory_core
        # Force enable explainability for testing
        if not memory_core.config.get("ENABLE_EXPLAINABILITY", False):
            memory_core.config["ENABLE_EXPLAINABILITY"] = True
            logger.info("API", "Forcing ENABLE_EXPLAINABILITY=True for testing")
        
        if memory_core and memory_core.config.get("ENABLE_EXPLAINABILITY", False):
            # Import here to avoid circular imports
            from synthians_memory_core.api.explainability_routes import router as explainability_router
            from synthians_memory_core.api.diagnostics_routes import router as diagnostics_router
            
            logger.info("API", "Mounting explainability and diagnostics routers")
            app.include_router(explainability_router)
            app.include_router(diagnostics_router)
            logger.info("API", "Mounted explainability and diagnostics routers", {
                "routes_count": len(explainability_router.routes) + len(diagnostics_router.routes)
            })
        else:
            logger.info("API", "Explainability features are disabled", {
                "ENABLE_EXPLAINABILITY": memory_core.config.get("ENABLE_EXPLAINABILITY", False) if memory_core else False
            })
    
    mount_conditional_routers()
    
    # Initialize emotion analysis model
    try:
        logger.info("API", "Initializing emotion analyzer...")
        # Use the new EmotionAnalyzer class
        app.state.emotion_analyzer = EmotionAnalyzer()
        logger.info("API", "Emotion analyzer initialized")
    except Exception as e:
        logger.error("API", f"Failed to initialize emotion analyzer: {str(e)}")
        app.state.emotion_analyzer = None
    
    # Initialize transcription feature extractor
    try:
        logger.info("API", "Initializing transcription feature extractor...")
        # Create the extractor with the emotion_analyzer
        app.state.transcription_extractor = TranscriptionFeatureExtractor(
            emotion_analyzer=app.state.emotion_analyzer
        )
        logger.info("API", "Transcription feature extractor initialized")
    except Exception as e:
        logger.error("API", f"Failed to initialize transcription feature extractor: {str(e)}")
        app.state.transcription_extractor = None
        
    # Initialize trainer integration manager
    try:
        logger.info("API", "Initializing trainer integration manager...")
        app.state.trainer_integration = TrainerIntegrationManager(
            memory_core=app.state.memory_core
        )
        logger.info("API", "Trainer integration manager initialized")
    except Exception as e:
        logger.error("API", f"Failed to initialize trainer integration manager: {str(e)}")
        app.state.trainer_integration = None
    
    # Initialize embedding model
    try:
        model_name = os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
        logger.info("API", f"Loading embedding model: {model_name}")
        
        # Try to load the model, download if not available
        try:
            app.state.embedding_model = SentenceTransformer(model_name)
            logger.info("API", f"Embedding model {model_name} loaded successfully")
        except Exception as model_error:
            # If the model doesn't exist, it might need to be downloaded
            if "No such file or directory" in str(model_error) or "not found" in str(model_error).lower():
                logger.warning("API", f"Model {model_name} not found locally, attempting to download...")
                from sentence_transformers import util as st_util
                # Force download from Hugging Face
                app.state.embedding_model = SentenceTransformer(model_name, use_auth_token=None)
                logger.info("API", f"Successfully downloaded and loaded model {model_name}")
            else:
                # Re-raise if it's not a file-not-found error
                raise
    except Exception as e:
        logger.error("API", f"Failed to load embedding model: {str(e)}")
        app.state.embedding_model = None
    
    # Complete initialization
    logger.info("API", "Synthians Memory Core API server started")
    
    # Yield control to FastAPI
    yield
    
    # Shutdown Logic
    logger.info("API", "Shutting down Synthians Memory Core API server...")
    # Clean up resources
    try:
        if hasattr(app.state, 'memory_core'):
            await app.state.memory_core.cleanup()
    except Exception as e:
        logger.error("API", f"Error during cleanup: {str(e)}")
    
    logger.info("API", "Synthians Memory Core API server shut down")

# Create the FastAPI app with lifespan
app = FastAPI(
    title="Synthians Memory Core API",
    description="Unified API for memory, embeddings, QuickRecal, and emotion analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

# Generate embedding using the loaded model
async def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for text using the sentence transformer model."""
    if not text:
        logger.warning("generate_embedding", "Empty text provided for embedding generation")
        # Return a zero vector of appropriate dimension
        embedding_dim = app.state.memory_core.config.get('embedding_dim', 768)
        return np.zeros(embedding_dim, dtype=np.float32)
    
    try:
        # Use the embedding model from app state
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: app.state.embedding_model.encode(text)
        )
        return embedding
    except Exception as e:
        logger.error("generate_embedding", f"Error generating embedding: {str(e)}")
        # Return a zero vector as fallback
        embedding_dim = app.state.memory_core.config.get('embedding_dim', 768)
        return np.zeros(embedding_dim, dtype=np.float32)

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Synthians Memory Core API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        uptime = time.time() - app.state.startup_time
        # Use _memories instead of memories to match the updated attribute name
        memory_count = len(app.state.memory_core._memories)
        assembly_count = len(app.state.memory_core.assemblies)
        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "memory_count": memory_count,
            "assembly_count": assembly_count,
            "version": "1.0.0"  # Add version information
        }
    except Exception as e:
        logger.error("health_check", f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/stats")
async def get_stats():
    """Get system statistics.
    
    Returns system statistics including:
    - Memory count
    - Assembly count
    - Embedding dimension
    - Index health metrics
    - Recent activity
    - Runtime configuration (non-sensitive)
    - Performance metrics (if available)
    - Activation statistics (Phase 5.9)
    """
    try:
        memory_core = app.state.memory_core
        
        # Basic memory statistics
        memory_count = await memory_core.get_memory_count()
        assembly_count = await memory_core.get_assembly_count()
        
        # Get vector index status
        index_status = await memory_core.check_index_health()
        
        # Get information about activations (Phase 5.9)
        activation_stats = {}
        try:
            # Load activation stats from the persisted file if it exists
            stats_path = os.path.join(memory_core.data_dir, "stats", "assembly_activation_stats.json")
            if os.path.exists(stats_path):
                async with aiofiles.open(stats_path, "r") as f:
                    content = await f.read()
                    activation_stats = json.loads(content)
            
            # Calculate total activations and top activated assemblies
            total_activations = sum(activation_stats.values())
            
            # Get top 10 most activated assemblies
            top_activated = []
            for asm_id, count in sorted(activation_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                try:
                    assembly = await memory_core.persistence.load_assembly(asm_id)
                    name = assembly.name if assembly else "Unknown"
                    top_activated.append({
                        "assembly_id": asm_id,
                        "name": name,
                        "activation_count": count
                    })
                except Exception as e:
                    logger.warning("API", f"Error loading assembly for stats: {asm_id}", {"error": str(e)})
        except Exception as e:
            logger.warning("API", "Error loading activation stats", {"error": str(e)})
            total_activations = 0
            top_activated = []
        
        # Assemble the response
        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "memories": {
                "total_count": memory_count,
                "indexed_count": index_status.get("indexed_memory_count", 0),
                "embedding_dimension": memory_core.config.get("embedding_dim", 0)
            },
            "assemblies": {
                "total_count": assembly_count,
                "indexed_count": index_status.get("indexed_assembly_count", 0),
                "last_merge_timestamp": index_status.get("last_merge_timestamp"),
                "sync_status": index_status.get("sync_status", {}),
                "total_activations_tracked": total_activations,
                "top_activated": top_activated
            },
            "vector_index": {
                "is_healthy": index_status.get("is_healthy", False),
                "drift_count": index_status.get("drift_count", 0),
                "drift_percentage": index_status.get("drift_percentage", 0),
                "last_check_timestamp": index_status.get("last_check_timestamp")
            },
            "performance": {
                "avg_store_latency_ms": index_status.get("avg_store_latency_ms", 0),
                "avg_retrieve_latency_ms": index_status.get("avg_retrieve_latency_ms", 0),
                "avg_merge_latency_ms": index_status.get("avg_merge_latency_ms", 0)
            },
            "feature_flags": {
                "explainability_enabled": memory_core.config.get("ENABLE_EXPLAINABILITY", False),
                "assembly_pruning_enabled": memory_core.config.get("assembly_pruning_enabled", False)
            }
        }
        
        return response
    except Exception as e:
        logger.error("API", "Error retrieving system stats", {"error": str(e)}, exc_info=True)
        return {
            "success": False,
            "error": f"Failed to retrieve system statistics: {str(e)}"
        }

@app.post("/process_memory", response_model=ProcessMemoryResponse)
async def process_memory(request: ProcessMemoryRequest, background_tasks: BackgroundTasks):
    """Process and store a new memory."""
    try:
        logger.info("process_memory", "Processing new memory request")
        # Validate input
        if not request.content and not request.embedding and not request.metadata:
            raise HTTPException(status_code=400, detail="No memory content provided")
            
        # Tracking for current request (all fields start as None)
        embedding = None
        generated_text = None
        memory_id = None
        emotion_data = None
        
        # Handle case where embedding is provided but in dict format
        if request.embedding is not None:
            if isinstance(request.embedding, dict):
                logger.warning("process_memory", f"Received embedding as dict type, attempting to extract vector")
                try:
                    # Try common dict formats
                    if 'embedding' in request.embedding and isinstance(request.embedding['embedding'], list):
                        embedding = request.embedding['embedding']
                        logger.info("process_memory", "Successfully extracted embedding from dict['embedding']")
                    elif 'vector' in request.embedding and isinstance(request.embedding['vector'], list):
                        embedding = request.embedding['vector']
                        logger.info("process_memory", "Successfully extracted embedding from dict['vector']")
                    elif 'value' in request.embedding and isinstance(request.embedding['value'], list):
                        embedding = request.embedding['value']
                        logger.info("process_memory", "Successfully extracted embedding from dict['value']")
                    else:
                        keys = list(request.embedding.keys()) if hasattr(request.embedding, 'keys') else 'unknown'
                        logger.error("process_memory", f"Could not extract embedding from dict with keys: {keys}")
                        embedding = None
                except Exception as e:
                    logger.error("process_memory", f"Error extracting embedding from dict: {str(e)}")
                    embedding = None
            else:
                # Normal list embedding
                embedding = request.embedding
                
        # Step 1: Generate embedding if needed
        if request.content and (embedding is None) and hasattr(app.state, 'embedding_model'):
            try:
                # Generate embedding
                logger.info("process_memory", "Generating embedding from text")
                loop = asyncio.get_event_loop()
                embedding_list = await loop.run_in_executor(
                    None, 
                    lambda: app.state.embedding_model.encode([request.content])
                )
                # Convert numpy array to Python list to avoid array boolean issues
                if embedding_list is not None and len(embedding_list) > 0:
                    embedding = embedding_list[0].tolist()
                    logger.info("process_memory", f"Generated embedding with {len(embedding)} dimensions")
                else:
                    embedding = None
                    logger.warning("process_memory", "Failed to generate embedding - empty result")
            except Exception as embed_error:
                logger.error("process_memory", f"Embedding generation error: {str(embed_error)}")
                embedding = None
                
        # Step 2: Perform emotion analysis if requested
        if request.analyze_emotion and request.content:
            try:
                logger.info("process_memory", "Performing emotion analysis")
                
                # Use our EmotionAnalyzer directly for the analysis
                if hasattr(app.state, 'emotion_analyzer') and app.state.emotion_analyzer is not None:
                    # Use the emotion analyzer
                    logger.debug("process_memory", "Using emotion analyzer for analysis")
                    emotion_data = await app.state.emotion_analyzer.analyze(request.content)
                else:
                    # Fallback: Call the analyze_emotion endpoint
                    logger.debug("process_memory", "Using analyze_emotion endpoint fallback")
                    emotion_response = await analyze_emotion(request.content)
                    if emotion_response.success:
                        emotion_data = {
                            "emotions": emotion_response.emotions,
                            "dominant_emotion": emotion_response.dominant_emotion
                        }
                
                logger.info("process_memory", f"Emotion analysis complete: {emotion_data.get('dominant_emotion') if emotion_data else 'None'}")
            except Exception as emotion_error:
                logger.error("process_memory", f"Emotion analysis error: {str(emotion_error)}")
                # Continue without emotion data
                
        # Step 3: Process the memory through the core
        try:
            # Prepare metadata with emotion data if available
            metadata = request.metadata or {}
            
            # Add timestamp to metadata
            metadata['timestamp'] = time.time()
            
            # Add emotion data to metadata if available
            if emotion_data:
                metadata['emotional_context'] = emotion_data
            
            # If we don't have an embedding at this point but have content, create a zero-embedding
            # This is a fallback to ensure the memory core can process the request
            if (embedding is None) and request.content:
                logger.warning("process_memory", "No embedding generated or provided. Creating zero-embedding as fallback.")
                # Create a zero-embedding with the default dimension
                embedding_dim = app.state.memory_core.config.get('embedding_dim', 768)
                embedding = [0.0] * embedding_dim
            
            # Validate embedding for NaN/Inf values and handle dimension mismatches
            if embedding is not None:
                try:
                    # Check for NaN/Inf values
                    if any(not np.isfinite(val) for val in embedding):
                        logger.warning("process_memory", "Found NaN/Inf values in embedding. Replacing with zeros.")
                        embedding = [0.0 if not np.isfinite(val) else val for val in embedding]
                    
                    # Ensure correct dimensionality
                    expected_dim = app.state.memory_core.config.get('embedding_dim', 768)
                    actual_dim = len(embedding)
                    
                    if actual_dim != expected_dim:
                        logger.warning("process_memory", f"Dimension mismatch: expected {expected_dim}, got {actual_dim}. Aligning to expected dimension.")
                        if actual_dim < expected_dim:
                            # Pad with zeros if too small
                            embedding = embedding + [0.0] * (expected_dim - actual_dim)
                        else:
                            # Truncate if too large
                            embedding = embedding[:expected_dim]
                except Exception as val_error:
                    logger.error("process_memory", f"Error validating embedding: {str(val_error)}")
                    # Continue with original embedding
            
            # Call the memory core to process the memory
            logger.info("process_memory", "Calling memory core to process memory")
            
            result = await app.state.memory_core.process_new_memory(
                content=request.content,
                embedding=embedding,
                metadata=metadata
            )
            
            # CRITICAL CHECK: Handle None result explicitly
            if result is None:
                logger.error("process_memory", "Core processing failed internally (returned None)")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": "Core memory processing failed internally"}
                )
            
            memory_id = result.id
            quickrecal_score = result.quickrecal_score
            logger.info("process_memory", f"Memory processed successfully with ID: {memory_id}")
            
            # Return response with results
            return ProcessMemoryResponse(
                success=True,
                memory_id=memory_id,
                quickrecal_score=quickrecal_score,
                embedding=embedding,
                metadata=metadata
            )
            
        except Exception as core_error:
            logger.error("process_memory", f"Memory core processing error: {str(core_error)}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"Memory processing failed: {str(core_error)}"}
            )
    
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (like validation errors)
        logger.warning(f"HTTPException in process_memory: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error("process_memory", f"Process memory error: {str(e)}")
        import traceback
        logger.error("process_memory", traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Internal server error: {str(e)}"}
        )

@app.post("/retrieve_memories", response_model=RetrieveMemoriesResponse)
async def retrieve_memories(request: RetrieveMemoriesRequest):
    """Retrieve relevant memories."""
    try:
        # Add debug logging
        logger.info("retrieve_memories", f"Received request: query='{request.query}', top_k={request.top_k}, threshold={request.threshold}")
        logger.debug(f"API retrieve_memories: Received request with threshold={request.threshold} (type: {type(request.threshold)})") # Log received value with type
        
        # Convert user_emotion from dict to string if needed
        user_emotion_str = None
        if request.user_emotion:
            if isinstance(request.user_emotion, dict) and 'dominant_emotion' in request.user_emotion:
                user_emotion_str = request.user_emotion['dominant_emotion']
            elif isinstance(request.user_emotion, str):
                user_emotion_str = request.user_emotion
        
        # Retrieve memories with updated parameters - fully keyword-based to avoid positional argument confusion
        retrieve_result = await app.state.memory_core.retrieve_memories(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,  # Use threshold from request if provided
            user_emotion=user_emotion_str,
            metadata_filter=request.metadata_filter if hasattr(request, 'metadata_filter') else None,
            search_strategy=request.search_strategy if hasattr(request, 'search_strategy') else None
        )
        
        # Add detailed response debugging
        memories = retrieve_result.get('memories', [])
        logger.debug(f"API endpoint: Retrieved {len(memories)} memories from core")
        if memories:
            logger.debug(f"API endpoint: First memory ID = {memories[0].get('id')}")
        
        response = RetrieveMemoriesResponse(
            success=retrieve_result.get('success', False),
            memories=memories,
            error=retrieve_result.get('error')
        )
        
        # Final API response check
        logger.debug(f"API endpoint: Final response will contain {len(response.memories)} memories")
        
        return response
    except Exception as e:
        logger.error("retrieve_memories", f"Error: {str(e)}")
        import traceback
        logger.error("retrieve_memories", traceback.format_exc())
        return RetrieveMemoriesResponse(
            success=False,
            error=str(e)
        )

@app.post("/generate_embedding", response_model=GenerateEmbeddingResponse)
async def embedding_endpoint(request: GenerateEmbeddingRequest):
    """Generate embedding for text."""
    try:
        embedding = await generate_embedding(request.text)
        return GenerateEmbeddingResponse(
            success=True,
            embedding=embedding.tolist(),
            dimension=len(embedding)
        )
    except Exception as e:
        logger.error("generate_embedding", f"Error: {str(e)}")
        return GenerateEmbeddingResponse(
            success=False,
            error=str(e)
        )

@app.post("/calculate_quickrecal", response_model=QuickRecalResponse)
async def calculate_quickrecal(request: QuickRecalRequest):
    """Calculate QuickRecal score for an embedding or text."""
    try:
        # Generate embedding if text is provided but embedding is not
        embedding = None
        if request.embedding is None and request.text is not None:
            # Generate embedding directly
            embedding = await generate_embedding(request.text)
        elif request.embedding is not None:
            embedding = np.array(request.embedding, dtype=np.float32)
        else:
            return QuickRecalResponse(
                success=False,
                error="Either embedding or text must be provided"
            )
        
        if embedding is None:
            return QuickRecalResponse(
                success=False,
                error="Failed to generate embedding"
            )
            
        # Prepare context with text if provided
        context = request.context or {'timestamp': time.time()}
        if request.text:
            context['text'] = request.text
            
        # Calculate QuickRecal score - use synchronous method to avoid asyncio issues
        try:
            if hasattr(app.state.memory_core.quick_recal, 'calculate'):
                quickrecal_score = await app.state.memory_core.quick_recal.calculate(embedding, context=context)
            else:
                logger.warning("calculate_quickrecal", "No calculate method found, using fallback")
                quickrecal_score = 0.5  # Default fallback score
        except RuntimeError as re:
            if "asyncio.run()" in str(re):
                # Handle asyncio runtime error by using synchronous version
                logger.warning("calculate_quickrecal", f"Asyncio runtime error: {str(re)}. Using synchronous method.")
                if hasattr(app.state.memory_core.quick_recal, 'calculate_sync'):
                    quickrecal_score = app.state.memory_core.quick_recal.calculate_sync(embedding, context=context)
                else:
                    logger.error("calculate_quickrecal", "No synchronous fallback method available.")
                    quickrecal_score = 0.5  # Default fallback score
            else:
                raise re
        
        # Get factor scores if available
        factors = None
        if hasattr(app.state.memory_core.quick_recal, 'get_last_factor_scores'):
            factors = app.state.memory_core.quick_recal.get_last_factor_scores()
        
        return QuickRecalResponse(
            success=True,
            quickrecal_score=quickrecal_score,
            factors=factors
        )
    except Exception as e:
        logger.error("calculate_quickrecal", f"Error: {str(e)}")
        return QuickRecalResponse(
            success=False,
            error=str(e)
        )

@app.post("/analyze_emotion", response_model=EmotionResponse)
async def analyze_emotion(request: EmotionRequest):
    """Analyze emotional content of text."""
    try:
        # Get text from the request
        text = request.text
            
        # Ensure text is a string
        if not isinstance(text, str):
            return EmotionResponse(
                success=False,
                error="Text must be a string"
            )
        
        # Use our EmotionAnalyzer if available
        if hasattr(app.state, 'emotion_analyzer') and app.state.emotion_analyzer is not None:
            # Get analysis results from the analyzer
            result = await app.state.emotion_analyzer.analyze(text)
            
            return EmotionResponse(
                success=True,
                emotions=result.get("emotions", {}),
                dominant_emotion=result.get("dominant_emotion", "neutral")
            )
        else:
            # Fallback to keyword-based detection if analyzer isn't available
            logger.warning("analyze_emotion", "Emotion analyzer not available, using keyword fallback")
            
            # Simple keyword-based emotion detection
            emotion_keywords = {
                "joy": ["happy", "joy", "delighted", "glad", "pleased", "excited", "thrilled"],
                "sadness": ["sad", "unhappy", "depressed", "down", "miserable", "upset", "disappointed"],
                "anger": ["angry", "mad", "furious", "annoyed", "irritated", "enraged", "frustrated"],
                "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous"],
                "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned"],
                "disgust": ["disgusted", "repulsed", "revolted", "sickened"],
                "neutral": ["ok", "fine", "neutral", "average", "normal"]
            }
            
            text = text.lower()
            emotion_scores = {emotion: 0.1 for emotion in emotion_keywords}  # Base score
            
            # Simple keyword matching
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        emotion_scores[emotion] += 0.15  # Increment score for each match
            
            # Find the dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            return EmotionResponse(
                success=True,
                emotions=emotion_scores,
                dominant_emotion=dominant_emotion
            )
            
    except Exception as e:
        logger.error("analyze_emotion", f"Error analyzing emotions: {str(e)}")
        import traceback
        logger.error("analyze_emotion", traceback.format_exc())
        
        return EmotionResponse(
            success=False,
            error=str(e)
        )

@app.post("/provide_feedback", response_model=FeedbackResponse)
async def provide_feedback(request: FeedbackRequest):
    """Provide feedback on memory retrieval relevance."""
    try:
        if not app.state.memory_core.threshold_calibrator:
            return FeedbackResponse(
                success=False,
                error="Adaptive thresholding is not enabled"
            )
        
        await app.state.memory_core.provide_feedback(
            memory_id=request.memory_id,
            similarity_score=request.similarity_score,
            was_relevant=request.was_relevant
        )
        
        new_threshold = app.state.memory_core.threshold_calibrator.get_current_threshold()
        
        return FeedbackResponse(
            success=True,
            new_threshold=new_threshold
        )
    except Exception as e:
        logger.error("provide_feedback", f"Error: {str(e)}")
        return FeedbackResponse(
            success=False,
            error=str(e)
        )

@app.post("/detect_contradictions")
async def detect_contradictions(threshold: float = 0.75):
    """Detect potential causal contradictions in memories."""
    try:
        contradictions = await app.state.memory_core.detect_contradictions(threshold=threshold)
        return {
            "success": True,
            "contradictions": contradictions,
            "count": len(contradictions)
        }
    except Exception as e:
        logger.error("detect_contradictions", f"Error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/process_transcription", response_model=TranscriptionResponse)
async def process_transcription(request: TranscriptionRequest, background_tasks: BackgroundTasks):
    """Process a transcription and store it in the memory system with rich metadata."""
    try:
        logger.info("process_transcription", "Processing transcription request")
        
        # Validate input
        if not request.text or not isinstance(request.text, str) or len(request.text.strip()) == 0:
            logger.error("process_transcription", "Invalid or empty transcription text")
            return TranscriptionResponse(
                success=False,
                error="Transcription text cannot be empty"
            )
            
        # Tracking for current request
        embedding = None
        extracted_metadata = None
        memory_id = None
        
        # Step 1: Generate embedding if needed
        if request.embedding is None and hasattr(app.state, 'embedding_model'):
            try:
                logger.info("process_transcription", "Generating embedding from transcription")
                loop = asyncio.get_event_loop()
                embedding_list = await loop.run_in_executor(
                    None, 
                    lambda: app.state.embedding_model.encode([request.text])
                )
                # Convert numpy array to Python list to avoid array boolean issues
                if embedding_list is not None and len(embedding_list) > 0:
                    embedding = embedding_list[0].tolist()
                    logger.info("process_transcription", f"Generated embedding with {len(embedding)} dimensions")
                else:
                    embedding = None
                    logger.warning("process_transcription", "Failed to generate embedding - empty result")
            except Exception as embed_error:
                logger.error("process_transcription", f"Embedding generation error: {str(embed_error)}")
                # Continue with None embedding if it fails
        else:
            embedding = request.embedding
        
        # Step 2: Extract features using the TranscriptionFeatureExtractor
        if hasattr(app.state, 'transcription_extractor') and app.state.transcription_extractor is not None:
            try:
                logger.info("process_transcription", "Extracting features from transcription")
                
                # Use our extractor to get rich metadata
                audio_metadata = request.audio_metadata or {}
                extracted_metadata = await app.state.transcription_extractor.extract_features(
                    transcript=request.text,
                    meta=audio_metadata
                )
                
                logger.info("process_transcription", 
                         f"Extracted {len(extracted_metadata)} features including" +
                         f" dominant_emotion={extracted_metadata.get('dominant_emotion', 'none')}," +
                         f" keywords={len(extracted_metadata.get('keywords', []))} keywords")
            except Exception as extract_error:
                logger.error("process_transcription", f"Feature extraction error: {str(extract_error)}")
                # Continue with empty metadata if extraction fails
                extracted_metadata = {
                    "input_modality": "spoken",
                    "source": "transcription",
                    "error": str(extract_error)
                }
        else:
            logger.warning("process_transcription", "No transcription feature extractor available")
            extracted_metadata = {
                "input_modality": "spoken",
                "source": "transcription"
            }
        
        # Step 3: Process the memory through the core
        try:
            # Prepare final metadata
            metadata = extracted_metadata or {}
            
            # Set importance if provided
            if request.importance is not None:
                metadata["importance"] = max(0.0, min(1.0, request.importance))
            
            # Add timestamp to metadata
            metadata["timestamp"] = time.time()
            
            # Call memory core to process the memory
            logger.info("process_transcription", "Calling memory core to process transcription memory")
            result = await app.state.memory_core.process_memory(
                content=request.text,
                embedding=embedding,
                memory_id=request.memory_id,
                metadata=metadata,
                memory_type="transcription",
                force_update=request.force_update
            )
            
            memory_id = result.get("memory_id")
            logger.info("process_transcription", f"Transcription processed with ID: {memory_id}")
            
            # Return success response
            return TranscriptionResponse(
                success=True,
                memory_id=memory_id,
                metadata=metadata,
                embedding=embedding
            )
            
        except Exception as core_error:
            logger.error("process_transcription", f"Memory core processing error: {str(core_error)}")
            raise HTTPException(status_code=500, detail=f"Memory processing failed: {str(core_error)}")
    
    except Exception as e:
        logger.error("process_transcription", f"Process transcription error: {str(e)}")
        import traceback
        logger.error("process_transcription", traceback.format_exc())
        
        return TranscriptionResponse(
            success=False,
            error=str(e)
        )

# --- Additional Memory Management Endpoints ---

@app.get("/api/memories/{memory_id}", response_model=GetMemoryResponse, tags=["Memory Management"])
async def get_memory(memory_id: str = Path(..., title="Memory ID", description="The unique ID of the memory to retrieve")):
    """Retrieve a specific memory entry by its ID."""
    try:
        memory = await app.state.memory_core.get_memory_by_id_async(memory_id)
        
        if memory is None:
            logger.warning("API", f"Memory not found: {memory_id}")
            return GetMemoryResponse(success=False, error=f"Memory with ID '{memory_id}' not found")
        
        # Use the MemoryEntry's to_dict method for proper serialization
        memory_dict = memory.to_dict()
        
        logger.info("API", f"Retrieved memory: {memory_id}")
        return GetMemoryResponse(success=True, memory=memory_dict)
    except Exception as e:
        logger.error("API", f"Error retrieving memory: {str(e)}")
        return GetMemoryResponse(success=False, error=f"Internal error: {str(e)}")

# --- Optional: Assembly Management Endpoints (Basic for MVP) ---

@app.get("/assemblies")
async def list_assemblies():
    """List all memory assemblies."""
    try:
        assembly_info = []
        async with app.state.memory_core._lock:
            for assembly_id, assembly in app.state.memory_core.assemblies.items():
                assembly_info.append({
                    "assembly_id": assembly_id,
                    "name": assembly.name,
                    "memory_count": len(assembly.memories),
                    "last_activation": assembly.last_activation
                })
        return {
            "success": True,
            "assemblies": assembly_info,
            "count": len(assembly_info)
        }
    except Exception as e:
        logger.error("list_assemblies", f"Error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/assemblies/{assembly_id}")
async def get_assembly(assembly_id: str):
    """Get details for a specific assembly."""
    try:
        async with app.state.memory_core._lock:
            if assembly_id not in app.state.memory_core.assemblies:
                return {
                    "success": False,
                    "error": "Assembly not found"
                }
            
            assembly = app.state.memory_core.assemblies[assembly_id]
            memory_ids = list(assembly.memories)
            
            # Get memory details (limited to first 10 for brevity)
            memories = []
            for mem_id in memory_ids[:10]:
                if mem_id in app.state.memory_core._memories:
                    memory = app.state.memory_core._memories[mem_id]
                    memories.append({
                        "id": memory.id,
                        "content": memory.content,
                        "quickrecal_score": memory.quickrecal_score
                    })
            
            # Get synchronization diagnostics
            sync_diagnostics = {}
            if hasattr(assembly, "get_sync_diagnostics"):
                sync_diagnostics = assembly.get_sync_diagnostics()
            
            # Calculate if assembly is synchronized
            is_synchronized = False
            if assembly.vector_index_updated_at is not None:
                from datetime import datetime, timezone, timedelta
                now = datetime.now(timezone.utc)
                # Consider assemblies synced within the last 24 hours as synchronized
                max_allowed_drift = timedelta(hours=24) 
                is_synchronized = (now - assembly.vector_index_updated_at) < max_allowed_drift
            
            return {
                "success": True,
                "assembly_id": assembly_id,
                "name": assembly.name,
                "memory_count": len(assembly.memories),
                "last_activation": assembly.last_activation,
                "sample_memories": memories,
                "total_memories": len(memory_ids),
                # Add synchronization information
                "vector_index_updated_at": assembly.vector_index_updated_at,
                "is_synchronized": is_synchronized,
                "drift_seconds": sync_diagnostics.get("drift_seconds", None)
            }
    except Exception as e:
        logger.error("get_assembly", f"Error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# --- Trainer Integration Endpoints ---

@app.post("/api/memories/get_sequence_embeddings", response_model=SequenceEmbeddingsResponse)
async def get_sequence_embeddings(
    topic: Optional[str] = None,
    user: Optional[str] = None,
    emotion: Optional[str] = None,
    min_importance: Optional[float] = None,
    limit: int = 100,
    min_quickrecal_score: Optional[float] = None,
    start_timestamp: Optional[str] = None,
    end_timestamp: Optional[str] = None,
    sort_by: str = "timestamp"
):
    """Retrieve a sequence of memory embeddings, ordered by timestamp or quickrecal score.
    
    This endpoint enables the Trainer to obtain sequential memory embeddings
    for training its predictive models and building semantic time series.
    """
    logger.info("API", f"Retrieving sequence embeddings with topic={topic}, limit={limit}, sort_by={sort_by}")
    
    if app.state.trainer_integration is None:
        logger.error("API", "Trainer integration manager not initialized")
        raise HTTPException(status_code=500, detail="Trainer integration not available")
    
    try:
        sequence = await app.state.trainer_integration.get_sequence_embeddings(
            topic=topic,
            user=user,
            emotion=emotion,
            min_importance=min_importance,
            limit=limit,
            min_quickrecal_score=min_quickrecal_score,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            sort_by=sort_by
        )
        return sequence
    except Exception as e:
        logger.error("API", f"Error retrieving sequence embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sequence embeddings: {str(e)}")

@app.post("/api/memories/update_quickrecal_score")
async def update_quickrecal_score(request: UpdateQuickRecalScoreRequest):
    """Update a memory's quickrecal score based on surprise feedback from the Trainer.
    
    This endpoint allows the Trainer to inform the Memory Core about surprising or
    unexpected memories, which can boost their recall priority and track narrative surprise.
    
    Surprise is recorded in the memory's metadata for future reference and pattern analysis.
    """
    logger.info("API", f"Updating quickrecal score for memory {request.memory_id} with delta {request.delta}")
    
    if app.state.trainer_integration is None:
        logger.error("API", "Trainer integration manager not initialized")
        raise HTTPException(status_code=500, detail="Trainer integration not available")
    
    try:
        result = await app.state.trainer_integration.update_quickrecal_score(request)
        return result
    except Exception as e:
        logger.error("API", f"Error updating quickrecal score: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update quickrecal score: {str(e)}")

# --- Index Integrity and Repair Endpoints ---

@app.get("/check_index_integrity", response_model=Dict[str, Any])
async def check_index_integrity():
    """
    Check the integrity of the FAISS vector index and return detailed drift statistics.
    
    This endpoint verifies synchronization between the FAISS index and ID mappings,
    providing comprehensive drift metrics for monitoring system health.
    
    Returns:
        Dict containing integrity check results with drift statistics:
        - success: Whether the check completed successfully
        - is_healthy: Boolean indicating if the index is in a healthy state
        - drift_count: Number of discrepancies between index and mappings
        - drift_warning: Boolean flag if drift exceeds warning threshold
        - drift_critical: Boolean flag if drift exceeds critical threshold
        - faiss_count: Number of vectors in the FAISS index
        - mapping_count: Number of entries in ID mapping
        - error: Error message if the check failed
    """
    try:
        logger.info("Performing FAISS index integrity check")
        
        # Get stats which include drift metrics
        stats = app.state.memory_core.vector_index.get_stats()
        
        # Determine health based on drift metrics
        is_healthy = True
        if stats.get("drift_warning", False) or stats.get("drift_critical", False):
            is_healthy = False
            logger.warning(f"Index integrity check indicates unhealthy state: {stats}")
        
        return {
            "success": True,
            "is_healthy": is_healthy,
            **stats
        }
    except Exception as e:
        logger.error(f"Error checking index integrity: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/repair_index", response_model=Dict[str, Any])
async def repair_index():
    """
    Repair the FAISS vector index by synchronizing with ID mappings.
    
    This endpoint attempts to restore consistency between the FAISS index and its ID mappings
    by rebuilding the index if necessary. Part of Phase 5.8 stability improvements.
    
    Enhanced to support full re-indexing from persistence to recover from severe drift or corruption.
    This rebuilds the index by loading all memories and assemblies from storage and re-adding them.
    
    Returns:
        Dict containing repair results:
        - success: Whether the repair was successful
        - repaired: Whether any repairs were actually made
        - before_stats: Index statistics before repair
        - after_stats: Index statistics after repair
        - reindexed_count: Number of items successfully re-indexed (if applicable)
        - error: Error message if repair failed
    """
    try:
        logger.info("[VECTOR_TRACE] Starting FAISS index repair procedure with re-indexing")
        
        # Get stats before repair
        before_stats = app.state.memory_core.vector_index.get_stats()
        
        # Perform repair operation - provide persistence and geometry_manager
        persistence = getattr(app.state.memory_core, "persistence", None)
        geometry_manager = getattr(app.state.memory_core, "geometry_manager", None)
        
        # Detailed repair log
        repair_result = None
        
        if persistence and geometry_manager:
            logger.info("[VECTOR_TRACE] Persistence and GeometryManager available, performing full re-indexing repair")
            # Call repair_index with parameters
            repair_result = await app.state.memory_core.vector_index.repair_index(
                persistence=persistence,
                geometry_manager=geometry_manager
            )
        else:
            logger.warning("[VECTOR_TRACE] Persistence or GeometryManager not available, falling back to basic repair")
            # Fall back to basic repair without re-indexing
            repair_result = app.state.memory_core.vector_index.repair_index()
        
        # Get stats after repair
        after_stats = app.state.memory_core.vector_index.get_stats()
        
        # Check if the repair improved the situation
        improved = False
        if after_stats.get("drift_count", 999) < before_stats.get("drift_count", 1000):
            improved = True
            
        logger.info(f"[VECTOR_TRACE] Index repair complete. Repaired: {repair_result is not None}")
        logger.info(f"[VECTOR_TRACE] Before: {before_stats}")
        logger.info(f"[VECTOR_TRACE] After: {after_stats}")
        
        # Enhanced response with additional repair details
        response = {
            "success": True,
            "repaired": repair_result is not None and repair_result.get("success", False),
            "improved": improved,
            "before_stats": before_stats,
            "after_stats": after_stats
        }
        
        # Add repair details if available
        if isinstance(repair_result, dict):
            response.update({
                "reindexed_count": repair_result.get("reindexed_count", 0),
                "repair_details": repair_result
            })
        
        return response
    except Exception as e:
        logger.error(f"[VECTOR_TRACE] Error repairing index: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.post("/repair_vector_index_drift", response_model=Dict[str, Any])
async def repair_vector_index_drift():
    """
    Repair the vector index when drift is detected between FAISS and ID mappings.
    
    This endpoint implements the Phase 5.8 'Repair-Resilient Retrieval' feature by:
    1. Detecting discrepancies between the FAISS index and ID mappings
    2. Performing auto-repair operations to reconcile differences
    3. Saving the repaired index to disk
    
    Returns:
        Dict containing repair operation results:
        - success: Whether the repair operation completed successfully
        - is_consistent: Whether the index is now consistent after repairs
        - drift_amount: Number of discrepancies detected prior to repair
        - repair_stats: Detailed statistics about the repair operations performed
        - error: Error message if the repair failed
    """
    try:
        logger.info("Initiating vector index repair operation")
        
        # Use the new repair method we implemented
        result = await app.state.memory_core.detect_and_repair_index_drift(auto_repair=True)
        
        # Log appropriate message based on result
        if result.get("success", False):
            logger.info("Vector index repair operation completed successfully")
        else:
            logger.warning(f"Vector index repair operation failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during vector index repair: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/repair_vector_index_drift", response_model=Dict[str, Any])
async def repair_vector_index_drift_get():
    """
    Repair the vector index when drift is detected between FAISS and ID mappings (GET endpoint).
    
    This endpoint implements the Phase 5.8 'Repair-Resilient Retrieval' feature by:
    1. Detecting discrepancies between the FAISS index and ID mappings
    2. Performing auto-repair operations to reconcile differences
    3. Saving the repaired index to disk
    
    Returns:
        Dict containing repair operation results:
        - success: Whether the repair operation completed successfully
        - is_consistent: Whether the index is now consistent after repairs
        - drift_amount: Number of discrepancies detected prior to repair
        - repair_stats: Detailed statistics about the repair operations performed
        - error: Error message if the repair failed
    """
    try:
        logger.info("Initiating vector index repair operation via GET endpoint")
        
        # Use the repair method we implemented
        result = await app.state.memory_core.detect_and_repair_index_drift(auto_repair=True)
        
        # Log appropriate message based on result
        if result.get("success", False):
            logger.info("Vector index repair operation completed successfully")
        else:
            logger.warning(f"Vector index repair operation failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during vector index repair: {str(e)}")
        return {"success": False, "error": str(e)}

# --- Test Endpoints (Disabled by default) ---

if TEST_ENDPOINTS_ENABLED:
    class ConfigUpdateRequest(BaseModel):
        key: str
        value: Any

    @app.post("/dev/set_config_value", include_in_schema=False) # Hide from public schema
    async def set_config_value(request: ConfigUpdateRequest):
        if not TEST_ENDPOINTS_ENABLED:
            raise HTTPException(status_code=403, detail="Test endpoints not enabled")
        try:
            core = app.state.memory_core
            original_value = core.config.get(request.key)
            core.config[request.key] = request.value # Directly modify config
            logger.info(f"[DEV_CONFIG] Set '{request.key}' from '{original_value}' to '{request.value}'")
            # Re-log the merge threshold specifically if changed
            if request.key == 'assembly_merge_threshold':
                 logger.info(f"[DEV_CONFIG] Merge threshold updated to: {core.config.get('assembly_merge_threshold')}")
            return {"success": True, "key": request.key, "new_value": request.value, "previous_value": original_value}
        except Exception as e:
            logger.error(f"Error in /dev/set_config_value: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

# Run the server when the module is executed directly
if __name__ == "__main__":
    import os
    import uvicorn
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5010"))
    
    print(f"Starting Synthians Memory Core API server at {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)
