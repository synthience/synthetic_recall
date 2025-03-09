# api/dream_api.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Request, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
import logging
import asyncio
import time
import os
import json
from datetime import datetime, timedelta
import websockets
from websockets.exceptions import ConnectionClosed

# Import the enhanced models from the correct paths
from memory.lucidia_memory_system.core.Self.self_model import LucidiaSelfModel as SelfModel
from memory.lucidia_memory_system.core.World.world_model import LucidiaWorldModel as WorldModel
from memory.lucidia_memory_system.core.dream_processor import LucidiaDreamProcessor as DreamProcessor
from memory.lucidia_memory_system.core.knowledge_graph import LucidiaKnowledgeGraph as KnowledgeGraph
from memory.lucidia_memory_system.core.embedding_comparator import EmbeddingComparator
from server.llm_pipeline import LocalLLMPipeline
from memory.lucidia_memory_system.core.dream_structures import DreamReport, DreamFragment
from memory.lucidia_memory_system.core.reflection_engine import ReflectionEngine

router = APIRouter(prefix="/api/dream", tags=["Dream Processing"])

logger = logging.getLogger(__name__)

# In-memory store of active dream sessions
dream_sessions = {}

# Configuration values, can be overridden by environment variables
TENSOR_SERVER_URL = os.getenv('TENSOR_SERVER_URL', 'ws://nemo_sig_v3:5001')
HPC_SERVER_URL = os.getenv('HPC_SERVER_URL', 'ws://nemo_sig_v3:5005')
CONNECTION_RETRY_LIMIT = int(os.getenv('CONNECTION_RETRY_LIMIT', '5'))
CONNECTION_RETRY_DELAY = float(os.getenv('CONNECTION_RETRY_DELAY', '2.0'))
CONNECTION_TIMEOUT = float(os.getenv('CONNECTION_TIMEOUT', '10.0'))

# WebSocket connections and locks
tensor_connection = None
hpc_connection = None
tensor_lock = asyncio.Lock()
hpc_lock = asyncio.Lock()

class DreamRequest(BaseModel):
    duration_minutes: int = 30
    mode: Optional[str] = "full"  # full, consolidate, insights, optimize
    scheduled: bool = False  # Whether to schedule for later
    schedule_time: Optional[str] = None  # ISO format datetime for scheduled execution
    priority: str = "normal"  # low, normal, high
    include_self_model: bool = True  # Whether to include self-model in dream processing
    include_world_model: bool = True  # Whether to include world-model in dream processing
    
class ConsolidateRequest(BaseModel):
    target: Optional[str] = "all"  # all, redundant, low_significance
    limit: Optional[int] = 100
    min_significance: Optional[float] = 0.3
    
class OptimizeRequest(BaseModel):
    target: Optional[str] = "all"  # all, files, database
    aggressive: bool = False  # Whether to be aggressive with optimization
    
class InsightRequest(BaseModel):
    timeframe: Optional[str] = "recent"  # recent, all, week, month
    limit: Optional[int] = 20
    categories: Optional[List[str]] = None

class KnowledgeGraphRequest(BaseModel):
    concept: Optional[str] = None
    relationship: Optional[str] = None
    depth: int = 1

class SelfReflectionRequest(BaseModel):
    focus_areas: Optional[List[str]] = None  # identity, capabilities, improvement, etc.
    depth: str = "standard"  # shallow, standard, deep

class WorldModelUpdateRequest(BaseModel):
    concept: str
    definition: str
    related: Optional[List[str]] = None
    attributes: Optional[Dict[str, Any]] = None
    source: Optional[str] = None

class WorldModelQueryRequest(BaseModel):
    query: str
    max_results: int = 10
    min_relevance: float = 0.3

class ScheduleRequest(BaseModel):
    task: str  # dream, consolidate, optimize, insights, reflection
    schedule: str  # once, daily, weekly
    time: str  # ISO format time or cron expression
    parameters: Dict[str, Any] = {}

class KnowledgeAddRequest(BaseModel):
    type: str  # concept or relationship
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    relation_type: Optional[str] = None
    strength: Optional[float] = None
    
    class Config:
        # Makes the validation more lenient by allowing extra fields and coercing types
        extra = "ignore"
        validate_assignment = True

class DreamReportRequest(BaseModel):
    """Request model for generating a dream report."""
    memory_ids: Optional[List[str]] = None
    timeframe: Optional[str] = "recent"
    limit: int = 20
    domain: str = "synthien_studies"
    title: Optional[str] = None
    description: Optional[str] = None

class DreamReportRefineRequest(BaseModel):
    """Request model for refining an existing dream report."""
    report_id: str
    new_evidence_ids: Optional[List[str]] = None
    update_analysis: bool = True

class CreateTestReportRequest(BaseModel):
    title: str
    fragments: List[Dict[str, Any]]

class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    use_hypersphere: bool = True

class SimilaritySearchRequest(BaseModel):
    query: str
    top_k: int = 3
    use_hypersphere: bool = True

class TestMemoryRequest(BaseModel):
    memories: List[Dict[str, Any]]

class RefineReportRequest(BaseModel):
    report_id: str

# Dependency functions to get component instances

def get_dream_processor(request: Request):
    """Dependency to get dream processor instance."""
    return request.app.state.dream_processor

def get_memory_client(request: Request):
    """Dependency to get memory client instance."""
    return request.app.state.memory_client

def get_knowledge_graph(request: Request):
    """Dependency to get knowledge graph instance."""
    return request.app.state.knowledge_graph

def get_self_model(request: Request):
    """Dependency to get self model instance."""
    return request.app.state.self_model

def get_world_model(request: Request):
    """Dependency to get world model instance."""
    return request.app.state.world_model

def get_embedding_comparator(request: Request):
    """Dependency to get embedding comparator."""
    return request.app.state.embedding_comparator

def get_llm_service(request: Request):
    """Dependency to get LLM service instance."""
    return request.app.state.llm_service

def get_reflection_engine(request: Request):
    """Dependency to get reflection engine instance."""
    return request.app.state.reflection_engine

async def get_tensor_connection():
    """Get or create a connection to the tensor server."""
    global tensor_connection
    
    async with tensor_lock:
        # Check if connection exists and isn't closed
        if tensor_connection:
            try:
                # Test if connection is still alive with a ping
                pong_waiter = await tensor_connection.ping()
                await asyncio.wait_for(pong_waiter, timeout=2.0)
                return tensor_connection
            except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
                # Connection is dead or closed, need to create a new one
                logger.info("Tensor connection is closed or unresponsive, creating a new one")
                try:
                    await tensor_connection.close()
                except:
                    pass
                tensor_connection = None
        
        # Create new connection with retry logic
        for attempt in range(CONNECTION_RETRY_LIMIT):
            try:
                logger.info(f"Connecting to tensor server at {TENSOR_SERVER_URL} (attempt {attempt+1}/{CONNECTION_RETRY_LIMIT})")
                connection = await asyncio.wait_for(
                    websockets.connect(TENSOR_SERVER_URL, ping_interval=30.0),
                    timeout=CONNECTION_TIMEOUT
                )
                
                # Test connection with a ping
                pong_waiter = await connection.ping()
                await asyncio.wait_for(pong_waiter, timeout=5.0)
                
                # Store and return connection
                tensor_connection = connection
                logger.info("Successfully connected to tensor server")
                return connection
                
            except (asyncio.TimeoutError, ConnectionRefusedError, ConnectionError, websockets.exceptions.WebSocketException) as e:
                logger.warning(f"Failed to connect to tensor server: {e}")
                if attempt < CONNECTION_RETRY_LIMIT - 1:
                    retry_delay = CONNECTION_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {retry_delay:.2f}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to tensor server after {CONNECTION_RETRY_LIMIT} attempts")
                    raise HTTPException(status_code=503, detail="Tensor server unavailable")

async def get_hpc_connection():
    """Get or create a connection to the HPC server."""
    global hpc_connection
    
    async with hpc_lock:
        # Check if connection exists and isn't closed
        if hpc_connection:
            try:
                # Test if connection is still alive with a ping
                pong_waiter = await hpc_connection.ping()
                await asyncio.wait_for(pong_waiter, timeout=2.0)
                return hpc_connection
            except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
                # Connection is dead or closed, need to create a new one
                logger.info("HPC connection is closed or unresponsive, creating a new one")
                try:
                    await hpc_connection.close()
                except:
                    pass
                hpc_connection = None
        
        # Create new connection with retry logic
        for attempt in range(CONNECTION_RETRY_LIMIT):
            try:
                logger.info(f"Connecting to HPC server at {HPC_SERVER_URL} (attempt {attempt+1}/{CONNECTION_RETRY_LIMIT})")
                connection = await asyncio.wait_for(
                    websockets.connect(HPC_SERVER_URL, ping_interval=30.0),
                    timeout=CONNECTION_TIMEOUT
                )
                
                # Test connection with a ping
                pong_waiter = await connection.ping()
                await asyncio.wait_for(pong_waiter, timeout=5.0)
                
                # Store and return connection
                hpc_connection = connection
                logger.info("Successfully connected to HPC server")
                return connection
                
            except (asyncio.TimeoutError, ConnectionRefusedError, ConnectionError, websockets.exceptions.WebSocketException) as e:
                logger.warning(f"Failed to connect to HPC server: {e}")
                if attempt < CONNECTION_RETRY_LIMIT - 1:
                    retry_delay = CONNECTION_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {retry_delay:.2f}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to HPC server after {CONNECTION_RETRY_LIMIT} attempts")
                    raise HTTPException(status_code=503, detail="HPC server unavailable")

async def process_embedding(text: str) -> Dict[str, Any]:
    """Process text through tensor server and HPC for embedding and significance."""
    try:
        # Connect to tensor server
        tensor_conn = await get_tensor_connection()
        
        # Create a standardized request
        message_id = f"{int(time.time() * 1000)}-dream"
        tensor_payload = {
            "type": "embed",
            "text": text,
            "client_id": "dream_processor",
            "message_id": message_id,
            "timestamp": time.time()
        }
        
        # Send request and get response
        await tensor_conn.send(json.dumps(tensor_payload))
        response = await tensor_conn.recv()
        data = json.loads(response)
        
        # Extract embedding
        embedding = None
        if 'data' in data and 'embeddings' in data['data']:
            embedding = data['data']['embeddings']
        elif 'data' in data and 'embedding' in data['data']:
            embedding = data['data']['embedding']
        elif 'embeddings' in data:
            embedding = data['embeddings']
        elif 'embedding' in data:
            embedding = data['embedding']
        
        if not embedding:
            logger.error(f"Failed to extract embedding from response: {data}")
            return {"success": False, "error": "No embedding in response"}
        
        # Connect to HPC for significance
        hpc_conn = await get_hpc_connection()
        
        # Create HPC request
        hpc_message_id = f"{int(time.time() * 1000)}-dream-hpc"
        hpc_payload = {
            "type": "process",
            "embeddings": embedding,
            "client_id": "dream_processor",
            "message_id": hpc_message_id,
            "timestamp": time.time()
        }
        
        # Send request and get response
        await hpc_conn.send(json.dumps(hpc_payload))
        hpc_response = await hpc_conn.recv()
        hpc_data = json.loads(hpc_response)
        
        # Extract significance
        significance = 0.5  # Default
        if 'data' in hpc_data and 'significance' in hpc_data['data']:
            significance = hpc_data['data']['significance']
        elif 'significance' in hpc_data:
            significance = hpc_data['significance']
        
        return {
            "success": True,
            "embedding": embedding,
            "significance": significance
        }
        
    except Exception as e:
        logger.error(f"Error processing embedding: {e}")
        return {"success": False, "error": str(e)}

@router.post("/start")
async def start_dream_session(
    background_tasks: BackgroundTasks,
    request: DreamRequest,
    dream_processor: DreamProcessor = Depends(get_dream_processor)
) -> Dict[str, Any]:
    """Start a dream processing session.
    
    This will run a background task that processes memories during idle time.
    The duration parameter controls how long the dream session will run.
    """
    try:
        # Connect to servers for processing
        try:
            tensor_conn = await get_tensor_connection()
            hpc_conn = await get_hpc_connection()
            logger.info("Successfully connected to tensor and HPC servers for dream session")
        except Exception as e:
            logger.error(f"Failed to connect to servers: {e}")
            return {
                "status": "error",
                "message": f"Failed to connect to tensor/HPC servers: {str(e)}"
            }
        
        # Start dream session based on mode
        if request.mode == "full" or request.mode == "all":
            # Pass only supported parameters to the dream processor
            result = await dream_processor.schedule_dream_session(
                duration_minutes=request.duration_minutes
            )
        elif request.mode == "consolidate":
            # Run a more focused consolidation session
            background_tasks.add_task(
                dream_processor.consolidate_memories,
                time_budget_seconds=request.duration_minutes * 60
            )
            result = {
                "status": "started",
                "mode": "consolidate",
                "scheduled_duration": request.duration_minutes
            }
        elif request.mode == "insights":
            # Run a more focused insight generation session
            background_tasks.add_task(
                dream_processor.generate_insights,
                time_budget_seconds=request.duration_minutes * 60
            )
            result = {
                "status": "started",
                "mode": "insights",
                "scheduled_duration": request.duration_minutes
            }
        elif request.mode == "reflection":
            # Run a self-reflection session
            background_tasks.add_task(
                dream_processor.self_reflection,
                time_budget_seconds=request.duration_minutes * 60
            )
            result = {
                "status": "started",
                "mode": "reflection",
                "scheduled_duration": request.duration_minutes
            }
        else:
            return {
                "status": "error",
                "message": f"Unknown mode: {request.mode}"
            }
        
        # Store session information
        if "session_id" in result:
            dream_sessions[result["session_id"]] = result
        
        return result
    
    except Exception as e:
        logger.error(f"Error starting dream session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def schedule_dream_session(dream_processor, request, delay_seconds):
    """Helper function to schedule a dream session after a delay."""
    try:
        logger.info(f"Scheduling dream session to start in {delay_seconds:.2f} seconds")
        await asyncio.sleep(delay_seconds)
        
        # Get fresh connections when the scheduled time arrives
        tensor_conn = await get_tensor_connection()
        hpc_conn = await get_hpc_connection()
        
        logger.info(f"Starting scheduled dream session ({request.mode})")
        
        # Start appropriate session type
        if request.mode == "full" or request.mode == "all":
            await dream_processor.schedule_dream_session(
                duration_minutes=request.duration_minutes
            )
        elif request.mode == "consolidate":
            await dream_processor.consolidate_memories(
                time_budget_seconds=request.duration_minutes * 60
            )
        elif request.mode == "insights":
            await dream_processor.generate_insights(
                time_budget_seconds=request.duration_minutes * 60
            )
        elif request.mode == "optimize":
            await dream_processor.optimize_storage(
                time_budget_seconds=request.duration_minutes * 60
            )
        elif request.mode == "reflection":
            await dream_processor.self_reflection(
                time_budget_seconds=request.duration_minutes * 60
            )
        
        logger.info(f"Scheduled dream session ({request.mode}) started successfully")
        
    except Exception as e:
        logger.error(f"Error in scheduled dream session: {e}")

@router.get("/status")
async def get_dream_status(
    session_id: Optional[str] = None,
    dream_processor: DreamProcessor = Depends(get_dream_processor)
) -> Dict[str, Any]:
    """Get the status of dream processing.
    
    If session_id is provided, returns the status of that specific session.
    Otherwise, returns the overall dream processor status.
    """
    try:
        status = dream_processor.get_dream_status()
        
        if session_id:
            # Get specific session status
            if session_id in dream_sessions:
                return {
                    "session": dream_sessions[session_id],
                    **status
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"No dream session found with ID {session_id}"
                }
        
        # Add server connection status
        status["servers"] = {
            "tensor_server": {
                "connected": tensor_connection is not None and not tensor_connection.closed if tensor_connection else False,
                "url": TENSOR_SERVER_URL
            },
            "hpc_server": {
                "connected": hpc_connection is not None and not hpc_connection.closed if hpc_connection else False,
                "url": HPC_SERVER_URL
            }
        }
        
        return status
    
    except Exception as e:
        logger.error(f"Error getting dream status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_dream_session(
    session_id: Optional[str] = None,
    dream_processor: DreamProcessor = Depends(get_dream_processor)
) -> Dict[str, Any]:
    """Stop the current dream session or a specific session by ID."""
    try:
        result = await dream_processor.stop_dream_session()
        
        # Clean up session info if provided
        if session_id and session_id in dream_sessions:
            del dream_sessions[session_id]
        
        return result
    
    except Exception as e:
        logger.error(f"Error stopping dream session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Memory operation endpoints

@router.post("/memory/consolidate")
async def consolidate_memories(
    memory_client: Any = Depends(get_memory_client),
    dream_processor: Any = Depends(get_dream_processor)
) -> Dict[str, Any]:
    """Consolidate similar memories to reduce redundancy."""
    try:
        if dream_processor is None:
            logger.error("Dream processor is not initialized")
            raise HTTPException(status_code=500, detail="Dream processor not initialized")
            
        # Use the dream processor to consolidate memories
        results = await dream_processor.consolidate_memories(time_budget_seconds=60)
        
        return results
    except AttributeError as e:
        # Handle the specific error where memory_client doesn't have get_memories method
        if "'EnhancedMemoryClient' object has no attribute 'get_memories'" in str(e):
            logger.error(f"Memory client doesn't support required methods: {e}")
            
            # Return a successful result with zero consolidated memories
            # This prevents the error from breaking the API flow
            return {
                "status": "completed",
                "consolidated_count": 0,
                "message": "Memory consolidation is not supported in this environment"
            }
        else:
            # Re-raise other AttributeErrors
            logger.error(f"Error in memory consolidation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error in memory consolidation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory/optimize")
async def optimize_memory_storage(
    request: OptimizeRequest,
    dream_processor: DreamProcessor = Depends(get_dream_processor)
) -> Dict[str, Any]:
    """Optimize memory storage based on the specified parameters.
    
    This can improve retrieval performance and reduce storage footprint.
    """
    try:
        # Set time budget based on target
        time_budget = 180  # 3 minutes default
        
        if request.target == "all":
            time_budget = 300  # 5 minutes for full optimization
        
        result = await dream_processor.optimize_storage(
            time_budget_seconds=time_budget,
            target=request.target,
            aggressive=request.aggressive
        )
        return result
    
    except Exception as e:
        logger.error(f"Error optimizing memory storage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory/insights")
async def generate_memory_insights(
    request: InsightRequest,
    dream_processor: DreamProcessor = Depends(get_dream_processor)
) -> Dict[str, Any]:
    """Generate insights from memories based on the specified parameters.
    
    This can reveal patterns and connections not immediately obvious.
    """
    try:
        # Get server connections for the operation
        tensor_conn = await get_tensor_connection()
        hpc_conn = await get_hpc_connection()
        
        # Set time budget based on request parameters
        time_budget = 180  # 3 minutes default
        
        if request.timeframe == "all":
            time_budget = 300  # 5 minutes for all memories
        
        result = await dream_processor.generate_insights(
            time_budget_seconds=time_budget,
            timeframe=request.timeframe,
            categories=request.categories
        )
        return result
    
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/self/reflect")
async def run_self_reflection(
    request: SelfReflectionRequest,
    dream_processor: DreamProcessor = Depends(get_dream_processor),
    self_model: SelfModel = Depends(get_self_model)
) -> Dict[str, Any]:
    """Run a self-reflection session to update Lucidia's self-model."""
    try:
        # Get server connections for the operation
        tensor_conn = await get_tensor_connection()
        hpc_conn = await get_hpc_connection()
        
        # Determine time budget based on depth
        time_budget = 180  # 3 minutes for standard depth
        if request.depth == "shallow":
            time_budget = 60  # 1 minute for shallow reflection
        elif request.depth == "deep":
            time_budget = 300  # 5 minutes for deep reflection
        
        result = await dream_processor.self_reflection(
            time_budget_seconds=time_budget,
            focus_areas=request.focus_areas,
            depth=request.depth
        )
        
        # Get updated self-model
        updated_model = self_model.get_model_summary()
        
        return {
            **result,
            "self_model": updated_model
        }
    
    except Exception as e:
        logger.error(f"Error running self reflection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/self-model")
async def get_self_model_data(
    self_model: Any = Depends(get_self_model)
) -> Dict[str, Any]:
    """Retrieve data from the self model."""
    try:
        if self_model is None:
            raise HTTPException(status_code=503, detail="Self model not initialized")
            
        # Get core self model data with safe attribute access
        data = {
            "identity": getattr(self_model, "identity", {}),
            "capabilities": getattr(self_model, "capabilities", {}),
            "preferences": getattr(self_model, "preferences", {}),
            "goals": getattr(self_model, "goals", []),
        }
        
        # Safely add other attributes if they exist
        if hasattr(self_model, "limitations"):
            data["limitations"] = self_model.limitations
        else:
            data["limitations"] = []  # Default empty list if attribute doesn't exist
            
        if hasattr(self_model, "experiences"):
            data["experiences"] = self_model.experiences
        else:
            data["experiences"] = []  # Default empty list
            
        if hasattr(self_model, "version"):
            data["version"] = self_model.version
        else:
            data["version"] = "unknown"
            
        # Get statistics if available
        if hasattr(self_model, "get_stats") and callable(getattr(self_model, "get_stats")):
            data["stats"] = self_model.get_stats()
        else:
            data["stats"] = {}
            
        return data
    
    except Exception as e:
        logger.error(f"Error retrieving self-model data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge graph integrations

@router.get("/knowledge")
async def get_knowledge_graph(
    concept: Optional[str] = Query(None, description="Concept to query relationships for"),
    relationship: Optional[str] = Query(None, description="Filter by relationship type"),
    depth: int = Query(1, description="Depth of relationship traversal"),
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph)
) -> Dict[str, Any]:
    """Get knowledge graph information.
    
    If concept is provided, returns relationships for that concept.
    Otherwise, returns overall graph statistics.
    """
    try:
        if concept:
            # Query relationships for a specific concept
            result = await knowledge_graph.query_related(concept, relationship, depth)
            return {
                "status": "success",
                "concept": concept,
                "relationships": result
            }
        else:
            # Return overall graph statistics
            stats = await knowledge_graph.get_stats()
            return {
                "status": "success",
                "stats": stats
            }
    
    except Exception as e:
        logger.error(f"Error querying knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge")
async def add_to_knowledge_graph(
    request: KnowledgeAddRequest,
    knowledge_graph: Any = Depends(get_knowledge_graph)
) -> Dict[str, Any]:
    """Add a concept or relationship to the knowledge graph."""
    try:
        # Ensure knowledge graph is properly initialized
        if knowledge_graph is None:
            logger.error("Knowledge graph is not initialized")
            raise HTTPException(status_code=500, detail="Knowledge graph not initialized")
        
        # Check if knowledge_graph is a valid instance with the appropriate methods
        from memory.lucidia_memory_system.core.knowledge_graph import KnowledgeGraph
        
        if not isinstance(knowledge_graph, KnowledgeGraph):
            logger.error(f"Knowledge graph is not properly initialized. Type: {type(knowledge_graph)}")
            raise HTTPException(status_code=500, detail="Knowledge graph not properly initialized")
        
        # Now we can safely proceed with adding nodes and edges
        if request.type == "concept":
            # For concept, name is required
            if not request.name:
                raise HTTPException(status_code=400, detail="name is required for concept type")
                
            # Add a concept node
            node_id = request.id or request.name.lower().replace(" ", "_")
            
            try:
                added = await knowledge_graph.add_node(
                    node_id=node_id,
                    node_type="concept",
                    properties={
                        "name": request.name,
                        "description": request.description or "",
                        "category": request.category or "general",
                        "created": datetime.now().isoformat()
                    }
                )
                
                return {
                    "status": "success",
                    "node_id": added["id"] if isinstance(added, dict) and "id" in added else node_id,
                    "message": f"Added concept: {request.name}"
                }
            except Exception as e:
                logger.error(f"Error adding node to knowledge graph: {e}")
                raise HTTPException(status_code=500, detail=f"Error adding node: {str(e)}")
            
        elif request.type == "relationship":
            # Add a relationship between nodes
            if not request.source_id or not request.target_id:
                raise HTTPException(status_code=400, detail="source_id and target_id are required for relationships")
                
            try:
                added = await knowledge_graph.add_edge(
                    source_id=request.source_id,
                    target_id=request.target_id,
                    edge_type=request.relation_type or "related_to",
                    properties={
                        "strength": request.strength or 0.5,
                        "description": request.description or "",
                        "created": datetime.now().isoformat()
                    }
                )
                
                return {
                    "status": "success",
                    "edge_id": added["id"] if isinstance(added, dict) and "id" in added else f"{request.source_id}-{request.target_id}",
                    "message": f"Added relationship between {request.source_id} and {request.target_id}"
                }
            except Exception as e:
                logger.error(f"Error adding edge to knowledge graph: {e}")
                raise HTTPException(status_code=500, detail=f"Error adding edge: {str(e)}")
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown knowledge type: {request.type}")
    
    except Exception as e:
        logger.error(f"Error adding to knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/schedule")
async def schedule_recurring_task(
    request: ScheduleRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Schedule a recurring task such as dream processing, consolidation, etc."""
    try:
        # Validate parameters for the task
        if request.task not in ["dream", "consolidate", "optimize", "insights", "reflection"]:
            return {
                "status": "error",
                "message": f"Unknown task: {request.task}"
            }
        
        if request.schedule not in ["once", "daily", "weekly"]:
            return {
                "status": "error",
                "message": f"Unknown schedule: {request.schedule}"
            }
        
        # Parse time
        try:
            if request.schedule == "once":
                # For one-time scheduling, expect ISO format datetime
                scheduled_time = datetime.fromisoformat(request.time)
                
                # Calculate delay
                now = datetime.now()
                if scheduled_time <= now:
                    return {
                        "status": "error",
                        "message": "Scheduled time must be in the future"
                    }
                    
                delay_seconds = (scheduled_time - now).total_seconds()
                
                # Prepare task based on type
                if request.task == "dream":
                    # Create a DreamRequest from parameters
                    dream_request = DreamRequest(
                        duration_minutes=request.parameters.get("duration_minutes", 30),
                        mode=request.parameters.get("mode", "full"),
                        priority=request.parameters.get("priority", "normal")
                    )
                    
                    # Schedule task
                    from app_dependencies import dream_processor
                    background_tasks.add_task(
                        schedule_dream_session,
                        dream_processor,
                        dream_request,
                        delay_seconds
                    )
                    
                    return {
                        "status": "scheduled",
                        "task": request.task,
                        "scheduled_time": request.time,
                        "delay_seconds": delay_seconds
                    }
                
                # Add other task types as needed...
                
            else:
                # For recurring schedules, store in database or scheduler system
                # This would interface with a task scheduler like Celery, APScheduler, etc.
                
                # For this example, we just return success but don't actually schedule
                return {
                    "status": "scheduled_recurring",
                    "task": request.task,
                    "schedule": request.schedule,
                    "time": request.time,
                    "message": "Recurring task scheduled (scheduler integration required)"
                }
                
        except ValueError:
            return {
                "status": "error",
                "message": f"Invalid time format: {request.time}. Use ISO format for datetime."
            }
            
    except Exception as e:
        logger.error(f"Error scheduling task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-tensor-connection")
async def test_tensor_connection() -> Dict[str, Any]:
    """Test connection to tensor server."""
    try:
        connection = await get_tensor_connection()
        
        # Basic test message
        test_message = {
            "type": "ping",
            "timestamp": time.time(),
            "client_id": "dream_api_test"
        }
        
        await connection.send(json.dumps(test_message))
        response = await asyncio.wait_for(connection.recv(), timeout=5.0)
        
        return {
            "status": "success",
            "connected": True,
            "response": json.loads(response) if response else None
        }
        
    except Exception as e:
        logger.error(f"Error testing tensor connection: {e}")
        return {
            "status": "error",
            "connected": False,
            "error": str(e)
        }

@router.post("/test-hpc-connection")
async def test_hpc_connection() -> Dict[str, Any]:
    """Test connection to HPC server."""
    try:
        connection = await get_hpc_connection()
        
        # Basic test message
        test_message = {
            "type": "ping",
            "timestamp": time.time(),
            "client_id": "dream_api_test"
        }
        
        await connection.send(json.dumps(test_message))
        response = await asyncio.wait_for(connection.recv(), timeout=5.0)
        
        return {
            "status": "success",
            "connected": True,
            "response": json.loads(response) if response else None
        }
        
    except Exception as e:
        logger.error(f"Error testing HPC connection: {e}")
        return {
            "status": "error", 
            "connected": False,
            "error": str(e)
        }

@router.post("/process-embedding")
async def test_process_embedding(text: str) -> Dict[str, Any]:
    """Test embedding processing through tensor and HPC servers."""
    try:
        result = await process_embedding(text)
        return result
    except Exception as e:
        logger.error(f"Error processing embedding: {e}")
        return {"success": False, "error": str(e)}

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for the dream API."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "dream_api",
        "tensor_server": tensor_connection is not None and not tensor_connection.closed if tensor_connection else False,
        "hpc_server": hpc_connection is not None and not hpc_connection.closed if hpc_connection else False
    }

@router.post("/shutdown")
async def shutdown_connections() -> Dict[str, Any]:
    """Properly close all server connections."""
    global tensor_connection, hpc_connection
    
    try:
        if tensor_connection and not tensor_connection.closed:
            await tensor_connection.close()
            tensor_connection = None
            
        if hpc_connection and not hpc_connection.closed:
            await hpc_connection.close()
            hpc_connection = None
            
        return {
            "status": "success",
            "message": "All connections closed"
        }
    
    except Exception as e:
        logger.error(f"Error shutting down connections: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@router.post("/self-model/reflect", response_model=Dict[str, Any])
async def run_self_reflection(
    request: SelfReflectionRequest,
    dream_processor: DreamProcessor = Depends(get_dream_processor),
    self_model: SelfModel = Depends(get_self_model)
):
    """Run a self-reflection session to update Lucidia's self-model."""
    try:
        logger.info(f"Starting self-reflection with focus on {request.focus_areas}")
        
        # Get tensor and HPC connections for embedding generation
        tensor_conn = await get_tensor_connection()
        hpc_conn = await get_hpc_connection()
        
        # Run self-reflection
        if not request.focus_areas:
            # Default focus areas if none provided
            focus_areas = ["capabilities", "performance", "improvement"]
        else:
            focus_areas = request.focus_areas
            
        # Run reflection with the enhanced self model
        reflection_result = await self_model.reflect(focus_areas)
        
        # Return the reflection results
        return {
            "status": "success",
            "reflection_id": reflection_result.get("reflection_id", str(time.time())),
            "focus_areas": focus_areas,
            "insights": reflection_result.get("insights", {}),
            "changes": reflection_result.get("changes", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during self-reflection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Self-reflection failed: {str(e)}")

@router.get("/self-model", response_model=Dict[str, Any])
async def get_self_model_data(
    self_model: SelfModel = Depends(get_self_model)
):
    """Get Lucidia's current self-model data."""
    try:
        # Get self-model context
        context = await self_model.get_self_context("general")
        
        # Get performance metrics
        metrics = self_model.get_performance_metrics()
        
        # Return formatted self-model data
        return {
            "identity": self_model.identity,
            "capabilities": self_model.capabilities,
            "limitations": self_model.limitations,
            "values": self_model.values,
            "context": context,
            "performance_metrics": metrics,
            "last_updated": self_model.last_updated.isoformat() if hasattr(self_model, "last_updated") else None
        }
        
    except Exception as e:
        logger.error(f"Error retrieving self-model data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve self-model data: {str(e)}")

# Knowledge graph integrations

@router.post("/knowledge")
async def add_to_knowledge_graph(
    request: KnowledgeAddRequest,
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph)
) -> Dict[str, Any]:
    """Add a concept or relationship to the knowledge graph."""
    try:
        # Ensure knowledge graph is properly initialized
        if knowledge_graph is None:
            logger.error("Knowledge graph is not initialized")
            raise HTTPException(status_code=500, detail="Knowledge graph not initialized")
        
        # Check if knowledge_graph is a valid instance with the appropriate methods
        from memory.lucidia_memory_system.core.knowledge_graph import KnowledgeGraph
        
        if not isinstance(knowledge_graph, KnowledgeGraph):
            logger.error(f"Knowledge graph is not properly initialized. Type: {type(knowledge_graph)}")
            raise HTTPException(status_code=500, detail="Knowledge graph not properly initialized")
        
        # Now we can safely proceed with adding nodes and edges
        if request.type == "concept":
            # For concept, name is required
            if not request.name:
                raise HTTPException(status_code=400, detail="name is required for concept type")
                
            # Add a concept node
            node_id = request.id or request.name.lower().replace(" ", "_")
            
            try:
                added = await knowledge_graph.add_node(
                    node_id=node_id,
                    node_type="concept",
                    properties={
                        "name": request.name,
                        "description": request.description or "",
                        "category": request.category or "general",
                        "created": datetime.now().isoformat()
                    }
                )
                
                return {
                    "status": "success",
                    "node_id": added["id"] if isinstance(added, dict) and "id" in added else node_id,
                    "message": f"Added concept: {request.name}"
                }
            except Exception as e:
                logger.error(f"Error adding node to knowledge graph: {e}")
                raise HTTPException(status_code=500, detail=f"Error adding node: {str(e)}")
            
        elif request.type == "relationship":
            # Add a relationship between nodes
            if not request.source_id or not request.target_id:
                raise HTTPException(status_code=400, detail="source_id and target_id are required for relationships")
                
            try:
                added = await knowledge_graph.add_edge(
                    source_id=request.source_id,
                    target_id=request.target_id,
                    edge_type=request.relation_type or "related_to",
                    properties={
                        "strength": request.strength or 0.5,
                        "description": request.description or "",
                        "created": datetime.now().isoformat()
                    }
                )
                
                return {
                    "status": "success",
                    "edge_id": added["id"] if isinstance(added, dict) and "id" in added else f"{request.source_id}-{request.target_id}",
                    "message": f"Added relationship between {request.source_id} and {request.target_id}"
                }
            except Exception as e:
                logger.error(f"Error adding edge to knowledge graph: {e}")
                raise HTTPException(status_code=500, detail=f"Error adding edge: {str(e)}")
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown knowledge type: {request.type}")
    
    except Exception as e:
        logger.error(f"Error adding to knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/report/refine", response_model=Dict[str, Any])
async def refine_dream_report(
    request: DreamReportRefineRequest,
    background_tasks: BackgroundTasks,
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph),
    reflection_engine: ReflectionEngine = Depends(get_reflection_engine)
):
    """Refine an existing dream report with new evidence or updated analysis.
    
    This endpoint updates a dream report by incorporating new memories or evidence
    and potentially updating the analysis based on this new information.
    """
    try:
        # Check if report exists
        if not await knowledge_graph.has_node(request.report_id):
            raise HTTPException(status_code=404, detail=f"Dream report {request.report_id} not found")
        
        # Get the report from the knowledge graph
        report_data = await knowledge_graph.get_node(request.report_id)
        if report_data["type"] != "dream_report":
            raise HTTPException(status_code=400, detail=f"Node {request.report_id} is not a dream report")
        
        # Convert back to a DreamReport object
        report = DreamReport.from_dict(report_data)
        
        # Define an async wrapper function to properly await the refine_report method
        async def _refine_report_async():
            try:
                await reflection_engine.refine_report(
                    report=report,
                    new_evidence_ids=request.new_evidence_ids,
                    update_analysis=request.update_analysis
                )
                logger.info(f"Successfully refined report {request.report_id}")
            except Exception as e:
                logger.exception(f"Error in background refinement of report {request.report_id}: {e}")
        
        # Start the refinement process in the background
        background_tasks.add_task(_refine_report_async)
        
        return {
            "status": "processing",
            "message": f"Refining dream report {request.report_id}",
            "report_id": request.report_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error refining dream report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refining dream report: {str(e)}")

@router.get("/test/get_report")
async def get_test_report(
    report_id: str = None,
    request: Request = None,
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """Test endpoint for retrieving a report by ID."""
    try:
        if not report_id:
            raise HTTPException(status_code=400, detail="Report ID is required")
        
        # Get the report
        report_node = await knowledge_graph.get_node(report_id)
        if not report_node:
            raise HTTPException(status_code=404, detail=f"Report with ID {report_id} not found")
        
        # Convert the node data to a DreamReport object
        try:
            report = DreamReport.from_dict(report_node)
            # Convert back to dictionary for API response, but with properly structured data
            return report.to_dict()
        except Exception as conversion_error:
            logger.exception(f"Error converting node to DreamReport: {conversion_error}")
            # If conversion fails, return the raw node data
            return report_node
    except Exception as e:
        logger.exception(f"Error getting test report: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting test report: {str(e)}")

@router.post("/test/batch_embedding")
async def test_batch_embedding(
    request: Request,
):
    """Test endpoint for batch embedding processing with HypersphereManager."""
    try:
        # Access the app state directly to get hypersphere_manager
        hypersphere_manager = request.app.state.hypersphere_manager
        
        # Parse the request body as JSON
        data = await request.json()
        texts = data.get("texts", [])
        use_hypersphere = data.get("use_hypersphere", True)
        
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided for embedding")
        
        # Use HypersphereManager if requested and available
        if use_hypersphere and hypersphere_manager:
            logger.info(f"Using HypersphereManager to batch process {len(texts)} embeddings")
            # Process embeddings in batch
            result = await hypersphere_manager.batch_process_embeddings(texts=texts)
            return {"status": "success", "embeddings": result.get("embeddings", [])}
        else:
            # Fallback to individual processing
            logger.info(f"Processing {len(texts)} embeddings individually")
            embeddings = []
            for text in texts:
                embedding = await process_embedding(text)
                embeddings.append(embedding["embedding"])
            return {"status": "success", "embeddings": embeddings}
    except Exception as e:
        logger.error(f"Error in batch embedding processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch embeddings: {str(e)}")

@router.post("/test/add_test_memories")
async def add_test_memories(
    request: Request,
):
    """Test endpoint to add sample memories to the system for testing."""
    try:
        # Access app state directly
        knowledge_graph = request.app.state.knowledge_graph
        hypersphere_manager = request.app.state.hypersphere_manager
        
        # Get the request data directly
        data = await request.json()
        memories = data.get("memories", [])
        
        if not memories:
            raise HTTPException(status_code=400, detail="No memories provided")

        memory_ids = []
        for memory_data in memories:
            content = memory_data.get("content")
            if not content:
                continue
                
            # Process the embedding
            importance = memory_data.get("importance", 0.5)
            metadata = memory_data.get("metadata", {})
            
            # Use HypersphereManager for embedding generation
            embedding_data = await hypersphere_manager.get_embedding(text=content)
            embedding = embedding_data.get("embedding", [])
            
            # Create a new memory node
            memory_id = f"test_memory_{int(time.time() * 1000)}_{len(memory_ids)}"
            
            # Add the memory to the knowledge graph
            node_attributes = {
                "content": content,
                "embedding": embedding,
                "importance": importance,
                "created_at": time.time(),
                "type": "memory",
                "metadata": metadata
            }
            
            await knowledge_graph.add_node(memory_id, 'memory', node_attributes)
            memory_ids.append(memory_id)
        
        return {"status": "success", "memory_ids": memory_ids}
    except Exception as e:
        logger.error(f"Error adding test memories: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding test memories: {str(e)}")

@router.post("/world-model/update", response_model=Dict[str, Any])
async def update_world_model(
    request: WorldModelUpdateRequest,
    world_model: WorldModel = Depends(get_world_model)
):
    """Update the world model with new knowledge."""
    try:
        logger.info(f"Updating world model with concept: {request.concept}")
        
        # Update knowledge in the world model
        result = await world_model.update_knowledge(
            concept=request.concept,
            definition=request.definition,
            related=request.related
        )
        
        return {
            "status": "success",
            "concept": request.concept,
            "message": f"World model updated with concept: {request.concept}",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error updating world model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update world model: {str(e)}")

@router.post("/world-model/query", response_model=Dict[str, Any])
async def query_world_model(
    request: WorldModelQueryRequest,
    world_model: Any = Depends(get_world_model)
) -> Dict[str, Any]:
    """Query the world model."""
    try:
        if world_model is None:
            raise HTTPException(status_code=503, detail="World model not initialized")
        
        logger.info(f"Querying world model with: {request.query}")
        
        # Check for available parameters in the method signature
        import inspect
        process_query_params = inspect.signature(world_model.process_query).parameters
        query_args = {}
        
        # Add query parameter (this should always exist)
        query_args['query'] = request.query
        
        # Conditionally add other parameters only if they exist in the method signature
        if 'max_results' in process_query_params:
            query_args['max_results'] = request.max_results
        
        if 'min_relevance' in process_query_params:
            query_args['min_relevance'] = request.min_relevance
            
        if 'context' in process_query_params and request.context:
            query_args['context'] = request.context
        
        # Call the process_query method with only the parameters it accepts
        result = await world_model.process_query(**query_args)
        
        return {
            "status": "success",
            "query": request.query,
            "results": result.get("results", []),
            "reasoning": result.get("reasoning", ""),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error querying world model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/world-model/metrics", response_model=Dict[str, Any])
async def get_world_model_metrics(
    world_model: Any = Depends(get_world_model)
) -> Dict[str, Any]:
    """Get metrics and statistics from the world model."""
    try:
        if world_model is None:
            raise HTTPException(status_code=503, detail="World model not initialized")
        
        logger.info("Fetching world model metrics")
        
        # Get metrics from the world model
        metrics = {}
        
        # Check if the world model has a get_metrics method
        if hasattr(world_model, "get_metrics") and callable(world_model.get_metrics):
            try:
                metrics = await world_model.get_metrics()
            except Exception as e:
                logger.warning(f"Error calling get_metrics: {e}")
                # Fall back to basic metrics
                pass
        
        # If no metrics or method not available, provide basic information
        if not metrics:
            # Get concept count if available
            concept_count = 0
            if hasattr(world_model, "concepts") and isinstance(world_model.concepts, dict):
                concept_count = len(world_model.concepts)
            elif hasattr(world_model, "knowledge") and isinstance(world_model.knowledge, dict):
                concept_count = len(world_model.knowledge)
            
            metrics = {
                "concept_count": concept_count,
                "last_updated": getattr(world_model, "last_updated", datetime.now()).isoformat() if hasattr(world_model, "last_updated") else None,
                "model_version": getattr(world_model, "version", "1.0")
            }
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving world model metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve world model metrics: {str(e)}")

@router.get("/world-model/analyze", response_model=Dict[str, Any])
async def analyze_world_model(
    world_model: WorldModel = Depends(get_world_model)
):
    """Analyze the world model's concept network."""
    try:
        logger.info("Analyzing world model concept network")
        
        # Analyze the concept network
        analysis = await world_model.analyze_concept_network()
        
        return {
            "status": "success",
            "concept_count": analysis.get("concept_count", 0),
            "relationship_count": analysis.get("relationship_count", 0),
            "central_concepts": analysis.get("central_concepts", []),
            "isolated_concepts": analysis.get("isolated_concepts", []),
            "clusters": analysis.get("clusters", []),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing world model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze world model: {str(e)}")

@router.get("/llm/status", response_model=Dict[str, Any])
async def get_llm_status(
    llm_service: LocalLLMPipeline = Depends(get_llm_service)
):
    """Get the status of the LLM service."""
    try:
        status = llm_service.get_status()
        return {
            "status": "success",
            "llm_status": status
        }
    
    except Exception as e:
        logger.error(f"Error getting LLM status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get LLM status: {str(e)}")

# Dream Report endpoints

@router.post("/report/generate", response_model=Dict[str, Any])
async def generate_dream_report(
    request: DreamReportRequest,
    background_tasks: BackgroundTasks,
    dream_processor: DreamProcessor = Depends(get_dream_processor),
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph),
    reflection_engine: ReflectionEngine = Depends(get_reflection_engine)
):
    """Generate a dream report from specified memories or recent memories.
    
    This endpoint analyzes memories and creates a structured report with insights,
    questions, hypotheses, and counterfactuals.
    """
    try:
        logger.info(f"Generating dream report from {request.limit} memories")
        
        # Get memories to analyze
        if request.memory_ids:
            # Use specified memory IDs
            memories = await dream_processor.memory_client.get_memories_by_ids(request.memory_ids)
        else:
            # Get recent memories based on timeframe
            if request.timeframe == "recent":
                memories = await dream_processor.memory_client.get_recent_memories(limit=request.limit)
            elif request.timeframe == "significant":
                memories = await dream_processor.memory_client.get_significant_memories(limit=request.limit)
            elif request.timeframe == "week":
                memories = await dream_processor.memory_client.get_memories_by_timeframe(
                    start_time=datetime.now() - timedelta(days=7),
                    end_time=datetime.now(),
                    limit=request.limit
                )
            else:
                memories = await dream_processor.memory_client.get_recent_memories(limit=request.limit)
        
        if not memories:
            return {"status": "error", "message": "No memories found for analysis"}
        
        # Generate the report asynchronously
        background_tasks.add_task(
            reflection_engine.generate_report,
            memories=memories,
            domain=request.domain,
            title=request.title,
            description=request.description
        )
        
        return {
            "status": "processing",
            "message": f"Generating dream report from {len(memories)} memories",
            "memory_count": len(memories)
        }
        
    except Exception as e:
        logger.error(f"Error generating dream report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating dream report: {str(e)}")

@router.get("/report/{report_id}", response_model=Dict[str, Any])
async def get_dream_report(
    report_id: str,
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """Retrieve a dream report by its ID.
    
    Returns the full report with all fragments and analysis.
    """
    try:
        # Check if report exists in knowledge graph
        if not await knowledge_graph.has_node(report_id):
            raise HTTPException(status_code=404, detail=f"Dream report {report_id} not found")
        
        # Get the report node
        report_node = await knowledge_graph.get_node(report_id)
        if report_node["type"] != "dream_report":
            raise HTTPException(status_code=400, detail=f"Node {report_id} is not a dream report")
        
        # Get the report data
        report_data = report_node["attributes"]
        
        # Get all fragments
        fragments = {}
        fragment_ids = []
        
        # Collect all fragment IDs from the report
        for fragment_type in ["insight_ids", "question_ids", "hypothesis_ids", "counterfactual_ids"]:
            if fragment_type in report_data:
                fragment_ids.extend(report_data[fragment_type])
        
        # Fetch all fragments
        for fragment_id in fragment_ids:
            if await knowledge_graph.has_node(fragment_id):
                fragment_node = await knowledge_graph.get_node(fragment_id)
                fragments[fragment_id] = fragment_node["attributes"]
        
        # Add fragments to the response
        report_data["fragments"] = fragments
        
        # Get connected concepts
        connected_concepts = await knowledge_graph.get_connected_nodes(
            report_id,
            edge_types=["references"],
            node_types=["concept", "entity"],
            direction="outbound"
        )
        
        concept_data = {}
        for concept in connected_concepts:
            if await knowledge_graph.has_node(concept):
                concept_node = await knowledge_graph.get_node(concept)
                concept_data[concept] = concept_node["attributes"]
        
        report_data["concepts"] = concept_data
        
        return {
            "status": "success",
            "report": report_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving dream report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving dream report: {str(e)}")

@router.post("/report/refine", response_model=Dict[str, Any])
async def refine_dream_report(
    request: DreamReportRefineRequest,
    background_tasks: BackgroundTasks,
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph),
    reflection_engine: ReflectionEngine = Depends(get_reflection_engine)
):
    """Refine an existing dream report with new evidence or updated analysis.
    
    This endpoint updates a dream report by incorporating new memories or evidence
    and potentially updating the analysis based on this new information.
    """
    try:
        # Check if report exists
        if not await knowledge_graph.has_node(request.report_id):
            raise HTTPException(status_code=404, detail=f"Dream report {request.report_id} not found")
        
        # Get the report from the knowledge graph
        report_data = await knowledge_graph.get_node(request.report_id)
        if report_data["type"] != "dream_report":
            raise HTTPException(status_code=400, detail=f"Node {request.report_id} is not a dream report")
        
        # Convert back to a DreamReport object
        report = DreamReport.from_dict(report_data)
        
        # Define an async wrapper function to properly await the refine_report method
        async def _refine_report_async():
            try:
                await reflection_engine.refine_report(
                    report=report,
                    new_evidence_ids=request.new_evidence_ids,
                    update_analysis=request.update_analysis
                )
                logger.info(f"Successfully refined report {request.report_id}")
            except Exception as e:
                logger.exception(f"Error in background refinement of report {request.report_id}: {e}")
        
        # Start the refinement process in the background
        background_tasks.add_task(_refine_report_async)
        
        return {
            "status": "processing",
            "message": f"Refining dream report {request.report_id}",
            "report_id": request.report_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error refining dream report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refining dream report: {str(e)}")

@router.get("/reports", response_model=Dict[str, Any])
async def list_dream_reports(
    limit: int = Query(10, description="Maximum number of reports to return"),
    skip: int = Query(0, description="Number of reports to skip"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """List all dream reports, with optional filtering by domain.
    
    Returns a paginated list of dream reports with basic metadata.
    """
    try:
        # Get all dream report nodes
        report_nodes = await knowledge_graph.get_nodes_by_type("dream_report")
        
        # Filter by domain if specified
        if domain:
            report_nodes = [node for node in report_nodes if node.get("domain") == domain]
        
        # Sort by creation time (newest first)
        report_nodes.sort(key=lambda x: x["attributes"].get("created_at", ""), reverse=True)
        
        # Apply pagination
        paginated_nodes = report_nodes[skip:skip+limit]
        
        # Format the response
        reports = []
        for node in paginated_nodes:
            report_data = {
                "report_id": node["id"],
                "title": node["attributes"].get("title", "Untitled Report"),
                "created_at": node["attributes"].get("created_at"),
                "domain": node["attributes"].get("domain"),
                "fragment_count": sum([
                    len(node["attributes"].get("insight_ids", [])),
                    len(node["attributes"].get("question_ids", [])),
                    len(node["attributes"].get("hypothesis_ids", [])),
                    len(node["attributes"].get("counterfactual_ids", []))
                ])
            }
            reports.append(report_data)
        
        return {
            "status": "success",
            "reports": reports,
            "total": len(report_nodes),
            "limit": limit,
            "skip": skip
        }
        
    except Exception as e:
        logger.error(f"Error listing dream reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing dream reports: {str(e)}")

@router.post("/test/similarity_search")
async def test_similarity_search(request: Request):
    """Test endpoint for similarity search using HypersphereManager."""
    try:
        # Access app state directly
        knowledge_graph = request.app.state.knowledge_graph
        hypersphere_manager = request.app.state.hypersphere_manager
        embedding_comparator = request.app.state.embedding_comparator
        
        # Parse the request body as JSON
        data = await request.json()
        query = data.get("query", "")
        top_k = data.get("top_k", 3)
        use_hypersphere = data.get("use_hypersphere", True)
        
        if not query:
            raise HTTPException(status_code=400, detail="No query provided")
        
        # Generate embedding for the query
        if use_hypersphere and hypersphere_manager:
            logger.info(f"Using HypersphereManager for query embedding")
            query_embedding_data = await hypersphere_manager.get_embedding(text=query)
            query_embedding = query_embedding_data.get("embedding", [])
        else:
            # Fallback to direct embedding
            logger.info(f"Using process_embedding for query embedding")
            embedding_data = await process_embedding(query)
            query_embedding = embedding_data.get("embedding", [])
            
        if not query_embedding:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate embedding for query"
            )
            
        # Retrieve all memory nodes from knowledge graph
        memories = await knowledge_graph.get_nodes_by_type("memory")
        if not memories:
            return {"status": "success", "results": []}
            
        # Prepare results list
        results = []
        
        # Compare query embedding with each memory
        for memory_id, memory_data in memories.items():
            # Extract memory content and embedding
            content = memory_data.get("content", "")
            embedding = memory_data.get("embedding", [])
            
            if not embedding:
                continue
                
            # Calculate similarity score
            score = embedding_comparator.calculate_similarity(query_embedding, embedding)
            
            results.append({
                "memory_id": memory_id,
                "content": content,
                "score": score
            })
            
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k results
        top_results = results[:top_k] if top_k > 0 else results
        
        return {"status": "success", "results": top_results}
    except Exception as e:
        logger.exception(f"Error in similarity search: {e}")  # Changed from logger.error to logger.exception
        raise HTTPException(status_code=500, detail=f"Error performing similarity search: {str(e)}")

@router.post("/test/create_test_report")
async def create_test_report(request: Request):
    """Create a test dream report with specified fragments for testing convergence mechanisms."""
    try:
        # Access app state directly
        knowledge_graph = request.app.state.knowledge_graph
        
        # Parse the request body as JSON
        data = await request.json()
        title = data.get("title", "")
        fragments = data.get("fragments", [])
        
        if not title or not fragments:
            raise HTTPException(status_code=400, detail="Title and fragments are required")
        
        # Create fragment objects first
        insight_ids = []
        question_ids = []
        hypothesis_ids = []
        counterfactual_ids = []
        
        for fragment_data in fragments:
            content = fragment_data.get("content")
            fragment_type = fragment_data.get("type")
            confidence = fragment_data.get("confidence", 0.5)
            
            if not content or not fragment_type:
                continue
                
            # Create a fragment
            fragment = DreamFragment(
                content=content,
                fragment_type=fragment_type,
                confidence=confidence,
                source_memory_ids=[]
            )
            
            # Add fragment to knowledge graph
            await knowledge_graph.add_node(fragment.id, 'dream_fragment', fragment.to_dict())
            
            # Add to appropriate ID list based on fragment type
            if fragment_type == "insight":
                insight_ids.append(fragment.id)
            elif fragment_type == "question":
                question_ids.append(fragment.id)
            elif fragment_type == "hypothesis":
                hypothesis_ids.append(fragment.id)
            elif fragment_type == "counterfactual":
                counterfactual_ids.append(fragment.id)
        
        # Create the report
        report = DreamReport(
            title=title,
            participating_memory_ids=[],
            insight_ids=insight_ids,
            question_ids=question_ids,
            hypothesis_ids=hypothesis_ids,
            counterfactual_ids=counterfactual_ids
        )
        
        # Add report to knowledge graph
        await knowledge_graph.add_node(report.report_id, 'dream_report', report.to_dict())
        
        return {
            "status": "success",
            "report_id": report.report_id,
            "fragment_count": report.get_fragment_count()
        }
    except Exception as e:
        logger.error(f"Error creating test report: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating test report: {str(e)}")

@router.post("/test/refine_report")
async def test_refine_report(
    request: Request,
):
    """Test endpoint for refining a report and testing convergence mechanisms."""
    try:
        # Access app state directly
        knowledge_graph = request.app.state.knowledge_graph
        
        # Parse the request body as JSON
        data = await request.json()
        report_id = data.get("report_id", "")
        
        if not report_id:
            raise HTTPException(status_code=400, detail="Report ID is required")
            
        # Get the report from the knowledge graph
        report_data = await knowledge_graph.get_node(report_id)
            
        if not report_data:
            raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")
            
        # Convert back to a DreamReport object
        report = DreamReport.from_dict(report_data)
        
        # Apply some mock refinement
        # For testing, we'll just add a new insight
        new_insight = DreamFragment(
            content="This is a refined insight generated during the test.",
            fragment_type="insight",
            confidence=0.8,
            source_memory_ids=[]
        )
        
        # Add to knowledge graph - ensure node_type is provided
        success = await knowledge_graph.add_node(
            node_id=new_insight.id, 
            node_type="dream_fragment",
            attributes=new_insight.to_dict(),
            domain="general_knowledge"
        )
        
        if not success:
            logger.warning(f"Failed to add new insight node to knowledge graph")
            raise HTTPException(status_code=500, detail="Failed to add insight to knowledge graph")
        
        # Add to report
        report.insight_ids.append(new_insight.id)
        
        # Update the report in the knowledge graph
        update_success = await knowledge_graph.update_node(report.report_id, report.to_dict())
        
        if not update_success:
            logger.warning(f"Failed to update report in knowledge graph")
            raise HTTPException(status_code=500, detail="Failed to update report in knowledge graph")
        
        return {
            "status": "success",
            "report_id": report.report_id,
            "fragment_count": report.get_fragment_count(),
            "new_insight_id": new_insight.id
        }
    except Exception as e:
        logger.exception(f"Error refining test report: {e}")
        raise HTTPException(status_code=500, detail=f"Error refining test report: {str(e)}")