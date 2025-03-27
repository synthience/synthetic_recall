# synthians_memory_core/api/server.py

import asyncio
import os
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Request
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

# Optional: Import sentence_transformers for embedding generation if not moved to GeometryManager
from sentence_transformers import SentenceTransformer

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
    """Get system statistics."""
    try:
        uptime = time.time() - app.state.startup_time
        # Get vector index stats
        vector_index_stats = {
            "count": app.state.memory_core.vector_index.count(),
            "id_mappings": len(app.state.memory_core.vector_index.id_to_index),
            "index_type": app.state.memory_core.vector_index.config.get('index_type', 'Unknown')
        }
        
        return {
            "success": True,  # Add success field
            "api_server": {
                "uptime_seconds": uptime,
                "memory_count": len(app.state.memory_core._memories),
                "embedding_dim": app.state.memory_core.config.get('embedding_dim', 768),
                "geometry": app.state.memory_core.config.get('geometry', 'hyperbolic'),
                "model": os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
            },
            "memory": {
                "total_memories": len(app.state.memory_core._memories),
                "total_assemblies": len(app.state.memory_core.assemblies),
                "storage_path": app.state.memory_core.config.get('storage_path', '/app/memory/stored/synthians'),
                "threshold": app.state.memory_core.config.get('contradiction_threshold', 0.75),
            },
            "vector_index": vector_index_stats
        }
    except Exception as e:
        logger.error("get_stats", f"Error retrieving stats: {str(e)}")
        return {
            "success": False,
            "error": str(e)
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
            
            memory_id = result.id if result else None
            quickrecal_score = result.quickrecal_score if result else None
            logger.info("process_memory", f"Memory processed successfully with ID: {memory_id}")
            
            # Return response with results
            return ProcessMemoryResponse(
                success=True,
                memory_id=memory_id,
                quickrecal_score=quickrecal_score,
                metadata=metadata
            )
            
        except Exception as core_error:
            logger.error("process_memory", f"Memory core processing error: {str(core_error)}")
            raise HTTPException(status_code=500, detail=f"Memory processing failed: {str(core_error)}")
    
    except Exception as e:
        logger.error("process_memory", f"Process memory error: {str(e)}")
        import traceback
        logger.error("process_memory", traceback.format_exc())
        
        return ProcessMemoryResponse(
            success=False,
            error=str(e)
        )


@app.post("/retrieve_memories", response_model=RetrieveMemoriesResponse)
async def retrieve_memories(request: RetrieveMemoriesRequest):
    """Retrieve relevant memories."""
    try:
        # Add debug logging
        logger.info("retrieve_memories", f"Received request: query='{request.query}', top_k={request.top_k}")
        
        # Convert user_emotion from dict to string if needed
        user_emotion_str = None
        if request.user_emotion:
            if isinstance(request.user_emotion, dict) and 'dominant_emotion' in request.user_emotion:
                user_emotion_str = request.user_emotion['dominant_emotion']
            elif isinstance(request.user_emotion, str):
                user_emotion_str = request.user_emotion
        
        # Retrieve memories with updated parameters
        # Note: We no longer pass query_embedding as it's handled internally
        retrieve_result = await app.state.memory_core.retrieve_memories(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,  # Use threshold from request if provided
            user_emotion=user_emotion_str
        )
        
        return RetrieveMemoriesResponse(
            success=retrieve_result.get('success', False),
            memories=retrieve_result.get('memories', []),
            error=retrieve_result.get('error')
        )
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
            
            return {
                "success": True,
                "assembly_id": assembly_id,
                "name": assembly.name,
                "memory_count": len(assembly.memories),
                "last_activation": assembly.last_activation,
                "sample_memories": memories,
                "total_memories": len(memory_ids)
            }
    except Exception as e:
        logger.error("get_assembly", f"Error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.on_event("startup")
async def startup_db_client():
    """Initialize FastAPI app with required services."""
    # Record startup time for stats
    app.state.startup_time = time.time()
    
    # Initialize embedding model
    try:
        from sentence_transformers import SentenceTransformer
        # Use the model specified in environment or default to all-mpnet-base-v2
        model_name = os.environ.get('EMBEDDING_MODEL', 'all-mpnet-base-v2')
        logger.info("startup", f"Loading embedding model: {model_name}")
        
        # Try to load the model, download if not available
        try:
            app.state.embedding_model = SentenceTransformer(model_name)
            logger.info("startup", f"Embedding model {model_name} loaded successfully")
        except Exception as model_error:
            # If the model doesn't exist, it might need to be downloaded
            if "No such file or directory" in str(model_error) or "not found" in str(model_error).lower():
                logger.warning("startup", f"Model {model_name} not found locally, attempting to download...")
                from sentence_transformers import util as st_util
                # Force download from Hugging Face
                app.state.embedding_model = SentenceTransformer(model_name, use_auth_token=None)
                logger.info("startup", f"Successfully downloaded and loaded model {model_name}")
            else:
                # Re-raise if it's not a file-not-found error
                raise
    except Exception as e:
        logger.error("startup", f"Error loading embedding model: {str(e)}")
        raise
    
    # Initialize emotion model
    try:
        from transformers import pipeline
        
        # Check for models in both local and Docker environments
        # For local development
        local_model_path = "C:/Users/danny/OneDrive/Documents/AI_Conversations/lucid-recall-dist/lucid-recall-dist/models/roberta-base-go_emotions"
        # For Docker environment
        docker_model_path = "/app/models/roberta-base-go_emotions"
        
        # Try Docker path first, then local path
        if os.path.exists(docker_model_path):
            emotion_model_path = docker_model_path
        elif os.path.exists(local_model_path):
            emotion_model_path = local_model_path
        else:
            emotion_model_path = None
            
        if emotion_model_path:
            logger.info("startup", f"Loading emotion model from: {emotion_model_path}")
            app.state.emotion_model = pipeline("text-classification", model=emotion_model_path, return_all_scores=True)
            logger.info("startup", "Emotion model loaded successfully")
        else:
            logger.warning("startup", f"Emotion model not found in expected locations, will use fallback")
            app.state.emotion_model = None
    except Exception as e:
        logger.error("startup", f"Error loading emotion model: {str(e)}")
        app.state.emotion_model = None
    
    # Initialize SynthiansMemoryCore
    try:
        # Load configuration from environment variables
        storage_path = os.environ.get('MEMORY_STORAGE_PATH', '/app/memory/stored/synthians')
        embedding_dim = int(os.environ.get('EMBEDDING_DIM', '768'))
        geometry_type = os.environ.get('GEOMETRY_TYPE', 'hyperbolic')
        
        # Create memory core config
        memory_core_config = {
            'storage_path': storage_path,
            'embedding_dim': embedding_dim,
            'geometry': geometry_type,
            'embedding_model': app.state.embedding_model,
            'emotion_model': app.state.emotion_model  # Pass the emotion model to the memory core
        }
        
        # Initialize memory core
        logger.info("startup", "Initializing SynthiansMemoryCore", memory_core_config)
        from synthians_memory_core import SynthiansMemoryCore
        app.state.memory_core = SynthiansMemoryCore(memory_core_config)
        await app.state.memory_core.initialize()
        logger.info("startup", "SynthiansMemoryCore initialized successfully")
    except Exception as e:
        logger.error("startup", f"Error initializing SynthiansMemoryCore: {str(e)}")
        raise

# Run the server when the module is executed directly
if __name__ == "__main__":
    import os
    import uvicorn
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5010"))
    
    print(f"Starting Synthians Memory Core API server at {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)
