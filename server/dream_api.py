# api/dream_api.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Request, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import asyncio
import time
import os
import json
import uuid
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

# Configuration values (overridable by environment variables)
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


# ========= Request Models =========

class DreamRequest(BaseModel):
    duration_minutes: int = 30
    mode: Optional[str] = "full"  # full, consolidate, insights, optimize, reflection
    scheduled: bool = False
    schedule_time: Optional[str] = None
    priority: str = "normal"
    include_self_model: bool = True
    include_world_model: bool = True

class ConsolidateRequest(BaseModel):
    target: Optional[str] = "all"
    limit: Optional[int] = 100
    min_significance: Optional[float] = 0.3

class OptimizeRequest(BaseModel):
    target: Optional[str] = "all"
    aggressive: bool = False

class InsightRequest(BaseModel):
    timeframe: Optional[str] = "recent"
    limit: Optional[int] = 20
    categories: Optional[List[str]] = None

class KnowledgeGraphRequest(BaseModel):
    concept: Optional[str] = None
    relationship: Optional[str] = None
    depth: int = 1

class SelfReflectionRequest(BaseModel):
    focus_areas: Optional[List[str]] = None
    depth: str = "standard"

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
        extra = "ignore"
        validate_assignment = True

class DreamReportRequest(BaseModel):
    memory_ids: Optional[List[str]] = None
    timeframe: Optional[str] = "recent"
    limit: int = 20
    domain: str = "synthien_studies"
    title: Optional[str] = None
    description: Optional[str] = None

class DreamReportRefineRequest(BaseModel):
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


# ========= Dependency Functions =========

def get_dream_processor(request: Request):
    return request.app.state.dream_processor

def get_memory_client(request: Request):
    return request.app.state.memory_client

def get_knowledge_graph(request: Request):
    return request.app.state.knowledge_graph

def get_self_model(request: Request):
    return request.app.state.self_model

def get_world_model(request: Request):
    return request.app.state.world_model

def get_embedding_comparator(request: Request):
    return request.app.state.embedding_comparator

def get_llm_service(request: Request):
    return request.app.state.llm_service

def get_reflection_engine(request: Request):
    return request.app.state.reflection_engine


# ========= Connection Functions =========

async def get_tensor_connection():
    global tensor_connection
    async with tensor_lock:
        if tensor_connection:
            try:
                pong_waiter = await tensor_connection.ping()
                await asyncio.wait_for(pong_waiter, timeout=2.0)
                return tensor_connection
            except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
                logger.info("Tensor connection is unresponsive; reconnecting")
                try:
                    await tensor_connection.close()
                except:
                    pass
                tensor_connection = None
        for attempt in range(CONNECTION_RETRY_LIMIT):
            try:
                logger.info(f"Connecting to tensor server at {TENSOR_SERVER_URL} (attempt {attempt+1}/{CONNECTION_RETRY_LIMIT})")
                connection = await asyncio.wait_for(
                    websockets.connect(TENSOR_SERVER_URL, ping_interval=30.0),
                    timeout=CONNECTION_TIMEOUT
                )
                pong_waiter = await connection.ping()
                await asyncio.wait_for(pong_waiter, timeout=5.0)
                tensor_connection = connection
                logger.info("Successfully connected to tensor server")
                return connection
            except (asyncio.TimeoutError, ConnectionRefusedError, ConnectionError, websockets.exceptions.WebSocketException) as e:
                logger.warning(f"Failed to connect to tensor server: {e}")
                if attempt < CONNECTION_RETRY_LIMIT - 1:
                    retry_delay = CONNECTION_RETRY_DELAY * (2 ** attempt)
                    logger.info(f"Retrying in {retry_delay:.2f}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect after {CONNECTION_RETRY_LIMIT} attempts")
                    raise HTTPException(status_code=503, detail="Tensor server unavailable")

async def get_hpc_connection():
    global hpc_connection
    async with hpc_lock:
        if hpc_connection:
            try:
                pong_waiter = await hpc_connection.ping()
                await asyncio.wait_for(pong_waiter, timeout=2.0)
                return hpc_connection
            except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
                logger.info("HPC connection is unresponsive; reconnecting")
                try:
                    await hpc_connection.close()
                except:
                    pass
                hpc_connection = None
        for attempt in range(CONNECTION_RETRY_LIMIT):
            try:
                logger.info(f"Connecting to HPC server at {HPC_SERVER_URL} (attempt {attempt+1}/{CONNECTION_RETRY_LIMIT})")
                connection = await asyncio.wait_for(
                    websockets.connect(HPC_SERVER_URL, ping_interval=30.0),
                    timeout=CONNECTION_TIMEOUT
                )
                pong_waiter = await connection.ping()
                await asyncio.wait_for(pong_waiter, timeout=5.0)
                hpc_connection = connection
                logger.info("Successfully connected to HPC server")
                return connection
            except (asyncio.TimeoutError, ConnectionRefusedError, ConnectionError, websockets.exceptions.WebSocketException) as e:
                logger.warning(f"Failed to connect to HPC server: {e}")
                if attempt < CONNECTION_RETRY_LIMIT - 1:
                    retry_delay = CONNECTION_RETRY_DELAY * (2 ** attempt)
                    logger.info(f"Retrying in {retry_delay:.2f}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect after {CONNECTION_RETRY_LIMIT} attempts")
                    raise HTTPException(status_code=503, detail="HPC server unavailable")

async def process_embedding(text: str) -> Dict[str, Any]:
    try:
        tensor_conn = await get_tensor_connection()
        message_id = f"{int(time.time() * 1000)}-dream"
        tensor_payload = {
            "type": "embed",
            "text": text,
            "client_id": "dream_processor",
            "message_id": message_id,
            "timestamp": time.time()
        }
        await tensor_conn.send(json.dumps(tensor_payload))
        response = await tensor_conn.recv()
        data = json.loads(response)
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
        hpc_conn = await get_hpc_connection()
        hpc_message_id = f"{int(time.time() * 1000)}-dream-hpc"
        hpc_payload = {
            "type": "process",
            "embeddings": embedding,
            "client_id": "dream_processor",
            "message_id": hpc_message_id,
            "timestamp": time.time()
        }
        await hpc_conn.send(json.dumps(hpc_payload))
        hpc_response = await hpc_conn.recv()
        hpc_data = json.loads(hpc_response)
        significance = 0.5
        if 'data' in hpc_data and 'significance' in hpc_data['data']:
            significance = hpc_data['data']['significance']
        elif 'significance' in hpc_data:
            significance = hpc_data['significance']
        return {"success": True, "embedding": embedding, "significance": significance}
    except Exception as e:
        logger.error(f"Error processing embedding: {e}")
        return {"success": False, "error": str(e)}


# ========= Dream Processing Endpoints =========

@router.post("/start")
async def start_dream_session(
    background_tasks: BackgroundTasks,
    request: DreamRequest,
    dream_processor: DreamProcessor = Depends(get_dream_processor)
) -> Dict[str, Any]:
    try:
        try:
            await get_tensor_connection()
            await get_hpc_connection()
            logger.info("Connected to tensor and HPC servers for dream session")
        except Exception as e:
            logger.error(f"Server connection error: {e}")
            return {"status": "error", "message": f"Connection error: {str(e)}"}
        if request.mode in ["full", "all"]:
            result = await dream_processor.schedule_dream_session(duration_minutes=request.duration_minutes)
        elif request.mode == "consolidate":
            background_tasks.add_task(dream_processor.consolidate_memories, time_budget_seconds=request.duration_minutes * 60)
            result = {"status": "started", "mode": "consolidate", "scheduled_duration": request.duration_minutes}
        elif request.mode == "insights":
            background_tasks.add_task(dream_processor.generate_insights, time_budget_seconds=request.duration_minutes * 60)
            result = {"status": "started", "mode": "insights", "scheduled_duration": request.duration_minutes}
        elif request.mode == "reflection":
            background_tasks.add_task(dream_processor.self_reflection, time_budget_seconds=request.duration_minutes * 60)
            result = {"status": "started", "mode": "reflection", "scheduled_duration": request.duration_minutes}
        else:
            return {"status": "error", "message": f"Unknown mode: {request.mode}"}
        if "session_id" in result:
            dream_sessions[result["session_id"]] = result
        return result
    except Exception as e:
        logger.error(f"Error starting dream session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def schedule_dream_session(dream_processor, request, delay_seconds):
    try:
        logger.info(f"Scheduling dream session in {delay_seconds:.2f} seconds")
        await asyncio.sleep(delay_seconds)
        await get_tensor_connection()
        await get_hpc_connection()
        logger.info(f"Starting scheduled dream session ({request.mode})")
        if request.mode in ["full", "all"]:
            await dream_processor.schedule_dream_session(duration_minutes=request.duration_minutes)
        elif request.mode == "consolidate":
            await dream_processor.consolidate_memories(time_budget_seconds=request.duration_minutes * 60)
        elif request.mode == "insights":
            await dream_processor.generate_insights(time_budget_seconds=request.duration_minutes * 60)
        elif request.mode == "optimize":
            await dream_processor.optimize_storage(time_budget_seconds=request.duration_minutes * 60)
        elif request.mode == "reflection":
            await dream_processor.self_reflection(time_budget_seconds=request.duration_minutes * 60)
        logger.info(f"Scheduled dream session ({request.mode}) started successfully")
    except Exception as e:
        logger.error(f"Error in scheduled dream session: {e}")

@router.get("/status")
async def get_dream_status(
    session_id: Optional[str] = None,
    dream_processor: DreamProcessor = Depends(get_dream_processor)
) -> Dict[str, Any]:
    try:
        status = dream_processor.get_dream_status()
        if session_id:
            if session_id in dream_sessions:
                return {"session": dream_sessions[session_id], **status}
            else:
                return {"status": "not_found", "message": f"No session with ID {session_id}"}
        status["servers"] = {
            "tensor_server": {"connected": tensor_connection is not None and not tensor_connection.closed, "url": TENSOR_SERVER_URL},
            "hpc_server": {"connected": hpc_connection is not None and not hpc_connection.closed, "url": HPC_SERVER_URL}
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
    try:
        result = await dream_processor.stop_dream_session()
        if session_id and session_id in dream_sessions:
            del dream_sessions[session_id]
        return result
    except Exception as e:
        logger.error(f"Error stopping dream session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========= Memory Operation Endpoints =========

@router.post("/memory/consolidate")
async def consolidate_memories(
    memory_client: Any = Depends(get_memory_client),
    dream_processor: Any = Depends(get_dream_processor)
) -> Dict[str, Any]:
    try:
        if dream_processor is None:
            logger.error("Dream processor not initialized")
            raise HTTPException(status_code=500, detail="Dream processor not initialized")
        results = await dream_processor.consolidate_memories(time_budget_seconds=60)
        return results
    except AttributeError as e:
        if "'EnhancedMemoryClient' object has no attribute 'get_memories'" in str(e):
            logger.error(f"Memory client missing methods: {e}")
            return {"status": "completed", "consolidated_count": 0, "message": "Memory consolidation not supported"}
        else:
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
    try:
        time_budget = 180 if request.target != "all" else 300
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
    try:
        await get_tensor_connection()
        await get_hpc_connection()
        time_budget = 180 if request.timeframe != "all" else 300
        result = await dream_processor.generate_insights(
            time_budget_seconds=time_budget,
            timeframe=request.timeframe,
            categories=request.categories
        )
        return result
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========= Self-Reflection and Self-Model Endpoints =========

@router.post("/self/reflect")
async def run_self_reflection(
    request: SelfReflectionRequest,
    dream_processor: DreamProcessor = Depends(get_dream_processor),
    self_model: SelfModel = Depends(get_self_model)
) -> Dict[str, Any]:
    try:
        await get_tensor_connection()
        await get_hpc_connection()
        time_budget = 180 if request.depth == "standard" else (60 if request.depth == "shallow" else 300)
        result = await dream_processor.self_reflection(
            time_budget_seconds=time_budget,
            focus_areas=request.focus_areas,
            depth=request.depth
        )
        updated_model = self_model.get_model_summary()
        return {**result, "self_model": updated_model}
    except Exception as e:
        logger.error(f"Error running self reflection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/self_model")
async def get_self_model_data(
    self_model: SelfModel = Depends(get_self_model)
) -> Dict[str, Any]:
    try:
        context = await self_model.get_self_context("general")
        metrics = self_model.get_performance_metrics()
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
        logger.error(f"Error retrieving self-model data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========= Knowledge Graph Endpoints =========

@router.get("/knowledge")
async def get_knowledge_graph_data(
    concept: Optional[str] = Query(None, description="Concept to query relationships for"),
    relationship: Optional[str] = Query(None, description="Filter by relationship type"),
    depth: int = Query(1, description="Depth of relationship traversal"),
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph)
) -> Dict[str, Any]:
    try:
        if concept:
            result = await knowledge_graph.query_related(concept, relationship, depth)
            return {"status": "success", "concept": concept, "relationships": result}
        else:
            stats = await knowledge_graph.get_stats()
            return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error(f"Error querying knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge")
async def add_to_knowledge_graph(
    request: KnowledgeAddRequest,
    knowledge_graph: Any = Depends(get_knowledge_graph)
) -> Dict[str, Any]:
    try:
        if knowledge_graph is None:
            logger.error("Knowledge graph not initialized")
            raise HTTPException(status_code=500, detail="Knowledge graph not initialized")
        from memory.lucidia_memory_system.core.knowledge_graph import KnowledgeGraph
        if not isinstance(knowledge_graph, KnowledgeGraph):
            logger.error(f"Knowledge graph improperly initialized. Type: {type(knowledge_graph)}")
            raise HTTPException(status_code=500, detail="Knowledge graph not properly initialized")
        if request.type == "concept":
            if not request.name:
                raise HTTPException(status_code=400, detail="Name required for concept")
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
                return {"status": "success", "node_id": added.get("id", node_id),
                        "message": f"Added concept: {request.name}"}
            except Exception as e:
                logger.error(f"Error adding node: {e}")
                raise HTTPException(status_code=500, detail=f"Error adding node: {str(e)}")
        elif request.type == "relationship":
            if not request.source_id or not request.target_id:
                raise HTTPException(status_code=400, detail="source_id and target_id required")
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
                return {"status": "success", "edge_id": added.get("id", f"{request.source_id}-{request.target_id}"),
                        "message": f"Added relationship between {request.source_id} and {request.target_id}"}
            except Exception as e:
                logger.error(f"Error adding edge: {e}")
                raise HTTPException(status_code=500, detail=f"Error adding edge: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown knowledge type: {request.type}")
    except Exception as e:
        logger.error(f"Error adding to knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========= Dream Report Endpoints =========

@router.post("/report/generate", response_model=Dict[str, Any])
async def generate_dream_report(
    request: DreamReportRequest,
    background_tasks: BackgroundTasks,
    dream_processor: DreamProcessor = Depends(get_dream_processor),
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph),
    reflection_engine: ReflectionEngine = Depends(get_reflection_engine)
):
    try:
        logger.info(f"Generating dream report from {request.limit} memories")
        if request.memory_ids:
            memories = await dream_processor.memory_client.get_memories_by_ids(request.memory_ids)
        else:
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
        background_tasks.add_task(
            reflection_engine.generate_report,
            memories=memories,
            domain=request.domain,
            title=request.title,
            description=request.description
        )
        return {"status": "processing", "message": f"Generating report from {len(memories)} memories", "memory_count": len(memories)}
    except Exception as e:
        logger.error(f"Error generating dream report: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating dream report: {str(e)}")

@router.get("/report/{report_id}", response_model=Dict[str, Any])
async def get_dream_report(
    report_id: str,
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph)
):
    try:
        if not await knowledge_graph.has_node(report_id):
            raise HTTPException(status_code=404, detail=f"Dream report {report_id} not found")
        report_node = await knowledge_graph.get_node(report_id)
        node_type = report_node.get("type") or report_node.get("attributes", {}).get("type")
        if node_type != "dream_report":
            raise HTTPException(status_code=400, detail=f"Node {report_id} is not a dream report")
        report_data = report_node.get("attributes", report_node)
        fragments = {}
        fragment_ids = []
        for frag_key in ["insight_ids", "question_ids", "hypothesis_ids", "counterfactual_ids"]:
            if frag_key in report_data:
                fragment_ids.extend(report_data[frag_key])
        for fid in fragment_ids:
            if await knowledge_graph.has_node(fid):
                frag_node = await knowledge_graph.get_node(fid)
                fragments[fid] = frag_node.get("attributes", frag_node)
        report_data["fragments"] = fragments
        connected_concepts = await knowledge_graph.get_connected_nodes(
            report_id,
            edge_types=["references"],
            node_types=["concept", "entity"],
            direction="outbound"
        )
        concept_data = {}
        for concept in connected_concepts:
            if await knowledge_graph.has_node(concept):
                cnode = await knowledge_graph.get_node(concept)
                concept_data[concept] = cnode.get("attributes", cnode)
        report_data["concepts"] = concept_data
        return {"status": "success", "report": report_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving dream report: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving dream report: {str(e)}")

@router.post("/report/refine", response_model=Dict[str, Any])
async def refine_dream_report(
    request: DreamReportRefineRequest,
    background_tasks: BackgroundTasks,
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph),
    reflection_engine: ReflectionEngine = Depends(get_reflection_engine)
):
    try:
        if not await knowledge_graph.has_node(request.report_id):
            raise HTTPException(status_code=404, detail=f"Dream report {request.report_id} not found")
        report_data = await knowledge_graph.get_node(request.report_id)
        node_type = report_data.get("type") or report_data.get("attributes", {}).get("type")
        if node_type != "dream_report":
            raise HTTPException(status_code=400, detail=f"Node {request.report_id} is not a dream report")
        report = DreamReport.from_dict(report_data)
        async def _refine_report_async():
            try:
                await reflection_engine.refine_report(
                    report=report,
                    new_evidence_ids=request.new_evidence_ids,
                    update_analysis=request.update_analysis
                )
                logger.info(f"Successfully refined report {request.report_id}")
            except Exception as e:
                logger.exception(f"Error refining report {request.report_id}: {e}")
        background_tasks.add_task(_refine_report_async)
        return {"status": "processing", "message": f"Refining report {request.report_id}", "report_id": request.report_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error refining dream report: {e}")
        raise HTTPException(status_code=500, detail=f"Error refining dream report: {str(e)}")

@router.get("/reports", response_model=Dict[str, Any])
async def list_dream_reports(
    limit: int = Query(10, description="Max reports to return"),
    skip: int = Query(0, description="Reports to skip"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph)
):
    try:
        report_nodes = await knowledge_graph.get_nodes_by_type("dream_report")
        if domain:
            report_nodes = [node for node in report_nodes if node.get("domain") == domain]
        report_nodes.sort(key=lambda x: x.get("attributes", {}).get("created_at", ""), reverse=True)
        paginated = report_nodes[skip:skip+limit]
        reports = []
        for node in paginated:
            attrs = node.get("attributes", node)
            reports.append({
                "report_id": node.get("id"),
                "title": attrs.get("title", "Untitled Report"),
                "created_at": attrs.get("created_at"),
                "domain": attrs.get("domain"),
                "fragment_count": sum([
                    len(attrs.get("insight_ids", [])),
                    len(attrs.get("question_ids", [])),
                    len(attrs.get("hypothesis_ids", [])),
                    len(attrs.get("counterfactual_ids", []))
                ])
            })
        return {"status": "success", "reports": reports, "total": len(report_nodes), "limit": limit, "skip": skip}
    except Exception as e:
        logger.error(f"Error listing dream reports: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing dream reports: {str(e)}")


# ========= Test and Utility Endpoints =========

@router.post("/test/batch_embedding")
async def test_batch_embedding(request: Request):
    """
    Test endpoint for batch embedding processing with HypersphereManager.
    """
    try:
        hypersphere_manager = request.app.state.hypersphere_manager
        data = await request.json()
        texts = data.get("texts", [])
        use_hypersphere = data.get("use_hypersphere", True)
        
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if use_hypersphere and hypersphere_manager:
            logger.info(f"Using HypersphereManager to batch process {len(texts)} embeddings")
            result = await hypersphere_manager.batch_process_embeddings(texts=texts)
            return {"status": "success", "embeddings": result.get("embeddings", [])}
        else:
            embeddings = []
            for text in texts:
                res = await process_embedding(text)
                embeddings.append(res.get("embedding"))
            return {"status": "success", "embeddings": embeddings}
    except Exception as e:
        logger.exception(f"Error in batch embedding processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch embeddings: {str(e)}")

@router.post("/test/add_test_memories")
async def add_test_memories(request: Request):
    """
    Test endpoint to add sample memories.
    """
    try:
        knowledge_graph = request.app.state.knowledge_graph
        data = await request.json()
        memories = data.get("memories", [])
        if not memories:
            raise HTTPException(status_code=400, detail="No memories provided")
        memory_ids = []
        for memory in memories:
            content = memory.get("content")
            if not content:
                continue
                
            importance = memory.get("importance", 0.5)
            metadata = memory.get("metadata", {})
            res = await process_embedding(content)
            embedding = res.get("embedding", [])
            memory_id = f"test_memory_{int(time.time()*1000)}_{len(memory_ids)}"
            node_attributes = {
                "content": content,
                "embedding": embedding,
                "importance": importance,
                "created_at": time.time(),
                "type": "memory",
                "metadata": metadata
            }
            await knowledge_graph.add_node(memory_id, "memory", node_attributes)
            memory_ids.append(memory_id)
        return {"status": "success", "memory_ids": memory_ids}
    except Exception as e:
        logger.error(f"Error adding test memories: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding test memories: {str(e)}")

@router.get("/test/get_report")
async def get_test_report(report_id: str, knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph)):
    """
    Test endpoint for retrieving a report by ID.
    """
    try:
        if not await knowledge_graph.has_node(report_id):
            raise HTTPException(status_code=404, detail=f"Report with ID {report_id} not found")
        report_node = await knowledge_graph.get_node(report_id)
        node_type = report_node.get("type") or report_node.get("attributes", {}).get("type")
        if node_type != "dream_report":
            raise HTTPException(status_code=400, detail=f"Node {report_id} is not a dream report")
        report_data = report_node.get("attributes", report_node)
        return {"status": "success", "report": report_data}
    except Exception as e:
        logger.error(f"Error getting test report: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting test report: {str(e)}")

@router.post("/test/similarity_search")
async def test_similarity_search(request: Request):
    """
    Test endpoint for similarity search using the knowledge graph.
    """
    try:
        knowledge_graph = request.app.state.knowledge_graph
        embedding_comparator = request.app.state.embedding_comparator
        
        data = await request.json()
        query = data.get("query", "")
        top_k = data.get("top_k", 3)
        
        if not query:
            raise HTTPException(status_code=400, detail="No query provided")
        
        # Always use process_embedding directly since HypersphereManager has issues
        logger.info(f"Using process_embedding for query embedding")
        embedding_data = await process_embedding(query)
        query_embedding = embedding_data.get("embedding", [])
            
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding for query")
            
        # Get all memory nodes
        memories = await knowledge_graph.get_nodes_by_type("memory")
        if not memories:
            return {"status": "success", "results": []}
            
        # Calculate similarity for each memory
        results = []
        for memory_id, memory_data in memories.items():
            memory_attrs = memory_data.get("attributes", memory_data)
            content = memory_attrs.get("content", "")
            embedding = memory_attrs.get("embedding", [])
            if not embedding:
                continue
                
            score = embedding_comparator.calculate_similarity(query_embedding, embedding)
            results.append({
                "memory_id": memory_id,
                "content": content,
                "score": score
            })
            
        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k] if top_k > 0 else results
        return {"status": "success", "results": top_results}
    except Exception as e:
        logger.exception(f"Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing similarity search: {str(e)}")

@router.post("/test/create_test_report")
async def create_test_report(request: Request):
    """
    Create a test dream report with specified fragments for testing convergence mechanisms.
    """
    try:
        knowledge_graph = request.app.state.knowledge_graph
        data = await request.json()
        title = data.get("title", "")
        fragments = data.get("fragments", [])
        
        if not title or not fragments:
            raise HTTPException(status_code=400, detail="Title and fragments are required")
            
        insight_ids = []
        question_ids = []
        hypothesis_ids = []
        counterfactual_ids = []
        
        # Create fragments
        for fragment_data in fragments:
            content = fragment_data.get("content")
            fragment_type = fragment_data.get("type")
            confidence = fragment_data.get("confidence", 0.5)
            
            if not content or not fragment_type:
                continue
                
            # Generate a unique ID for this fragment
            fragment_id = f"fragment:{uuid.uuid4()}"
            
            # Create fragment attributes
            fragment_attrs = {
                "content": content,
                "type": fragment_type,
                "fragment_type": fragment_type,
                "confidence": confidence,
                "source_memory_ids": [],
                "created_at": datetime.now().isoformat()
            }
            
            # Add fragment to knowledge graph
            await knowledge_graph.add_node(fragment_id, "dream_fragment", fragment_attrs)
            
            # Collect fragment IDs by type
            if fragment_type == "insight":
                insight_ids.append(fragment_id)
            elif fragment_type == "question":
                question_ids.append(fragment_id)
            elif fragment_type == "hypothesis":
                hypothesis_ids.append(fragment_id)
            elif fragment_type == "counterfactual":
                counterfactual_ids.append(fragment_id)
        
        # Create the report
        report_id = f"report:{uuid.uuid4()}"
        report_attrs = {
            "title": title,
            "type": "dream_report",
            "participating_memory_ids": [],
            "insight_ids": insight_ids,
            "question_ids": question_ids,
            "hypothesis_ids": hypothesis_ids,
            "counterfactual_ids": counterfactual_ids,
            "created_at": datetime.now().isoformat(),
            "refinement_count": 0,
            "confidence_history": []
        }
        
        await knowledge_graph.add_node(report_id, "dream_report", report_attrs)
        
        fragment_count = len(insight_ids) + len(question_ids) + len(hypothesis_ids) + len(counterfactual_ids)
        return {
            "status": "success",
            "report_id": report_id,
            "fragment_count": fragment_count
        }
    except Exception as e:
        logger.exception(f"Error creating test report: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating test report: {str(e)}")

@router.post("/test/refine_report")
async def test_refine_report(request: Request):
    """
    Test endpoint for refining a report and testing convergence mechanisms.
    """
    try:
        knowledge_graph = request.app.state.knowledge_graph
        data = await request.json()
        report_id = data.get("report_id", "")
        
        if not report_id:
            raise HTTPException(status_code=400, detail="Report ID is required")
            
        # Get the current report
        if not await knowledge_graph.has_node(report_id):
            raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")
            
        report_node = await knowledge_graph.get_node(report_id)
        report_attrs = report_node.get("attributes", report_node)
        
        # Update refinement count
        refinement_count = report_attrs.get("refinement_count", 0) + 1
        
        # Generate a new insight as part of refinement
        fragment_id = f"fragment:{uuid.uuid4()}"
        fragment_attrs = {
            "content": f"This is a refined insight generated during refinement {refinement_count}.",
            "type": "insight",
            "fragment_type": "insight",
            "confidence": min(0.8 + (refinement_count * 0.02), 0.95),  # Increase confidence with each refinement
            "source_memory_ids": [],
            "created_at": datetime.now().isoformat()
        }
        
        # Add fragment to knowledge graph
        await knowledge_graph.add_node(fragment_id, "dream_fragment", fragment_attrs)
        
        # Update the report
        insight_ids = report_attrs.get("insight_ids", [])
        insight_ids.append(fragment_id)
        
        # Update confidence history
        confidence = min(0.7 + (refinement_count * 0.05), 0.98)  # Convergence: confidence increases but plateaus
        confidence_history = report_attrs.get("confidence_history", [])
        confidence_history.append(confidence)
        
        # Check for convergence
        status = "refined"
        reason = ""
        if refinement_count >= 5:
            status = "skipped"
            reason = "Maximum refinement count reached"
        elif refinement_count > 2 and abs(confidence - confidence_history[-2]) < 0.03:
            status = "skipped"
            reason = "Confidence convergence detected"
        
        # Update report attributes
        updated_attrs = {
            **report_attrs,
            "insight_ids": insight_ids,
            "refinement_count": refinement_count,
            "confidence_history": confidence_history,
            "last_refined_at": datetime.now().isoformat(),
            "confidence": confidence
        }
        
        await knowledge_graph.update_node(report_id, updated_attrs)
        
        return {
            "status": status,
            "report_id": report_id,
            "refinement_count": refinement_count,
            "confidence": confidence,
            "reason": reason,
            "new_fragment_id": fragment_id
        }
    except Exception as e:
        logger.exception(f"Error refining test report: {e}")
        raise HTTPException(status_code=500, detail=f"Error refining test report: {str(e)}")

# ========= Test Connection Endpoints =========

@router.post("/test/tensor_connection")
async def test_tensor_connection() -> Dict[str, Any]:
    """
    Test connection to tensor server.
    """
    try:
        connection = await get_tensor_connection()
        test_message = {"type": "ping", "timestamp": time.time(), "client_id": "dream_api_test"}
        await connection.send(json.dumps(test_message))
        response = await asyncio.wait_for(connection.recv(), timeout=5.0)
        return {"status": "success", "connected": True, "response": json.loads(response) if response else None}
    except Exception as e:
        logger.error(f"Error testing tensor connection: {e}")
        return {"status": "error", "connected": False, "error": str(e)}

@router.post("/test/hpc_connection")
async def test_hpc_connection() -> Dict[str, Any]:
    """
    Test connection to HPC server.
    """
    try:
        connection = await get_hpc_connection()
        test_message = {"type": "ping", "timestamp": time.time(), "client_id": "dream_api_test"}
        await connection.send(json.dumps(test_message))
        response = await asyncio.wait_for(connection.recv(), timeout=5.0)
        return {"status": "success", "connected": True, "response": json.loads(response) if response else None}
    except Exception as e:
        logger.error(f"Error testing HPC connection: {e}")
        return {"status": "error", "connected": False, "error": str(e)}

@router.post("/test/process_embedding")
async def test_process_embedding(text: str) -> Dict[str, Any]:
    """
    Test embedding processing.
    """
    try:
        result = await process_embedding(text)
        return result
    except Exception as e:
        logger.error(f"Error processing embedding: {e}")
        return {"success": False, "error": str(e)}

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "dream_api",
        "tensor_server": tensor_connection is not None and not tensor_connection.closed,
        "hpc_server": hpc_connection is not None and not hpc_connection.closed
    }

@router.post("/shutdown")
async def shutdown_connections() -> Dict[str, Any]:
    """
    Close all server connections.
    """
    global tensor_connection, hpc_connection
    try:
        if tensor_connection and not tensor_connection.closed:
            await tensor_connection.close()
            tensor_connection = None
        if hpc_connection and not hpc_connection.closed:
            await hpc_connection.close()
            hpc_connection = None
        return {"status": "success", "message": "All connections closed"}
    except Exception as e:
        logger.error(f"Error shutting down connections: {e}")
        return {"status": "error", "message": str(e)}
