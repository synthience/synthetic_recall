# dream_api_server.py
import asyncio
import logging
import os
import sys
import uvicorn
import json
import time
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import websockets
from websockets.exceptions import ConnectionClosed

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dream API router
from server.dream_api import router as dream_router

# Import memory system components
from memory.lucidia_memory_system.core.dream_processor import LucidiaDreamProcessor as DreamProcessor
from memory.lucidia_memory_system.core.knowledge_graph import LucidiaKnowledgeGraph as KnowledgeGraph
from memory.lucidia_memory_system.core.Self.self_model import LucidiaSelfModel as SelfModel
from memory.lucidia_memory_system.core.World.world_model import LucidiaWorldModel
from memory.lucidia_memory_system.core.embedding_comparator import EmbeddingComparator
from memory.lucidia_memory_system.core.integration import MemoryIntegration
from memory.lucidia_memory_system.core.reflection_engine import ReflectionEngine
from memory.lucidia_memory_system.core.hypersphere_dispatcher import HypersphereDispatcher
from memory.lucidia_memory_system.core.manifold_geometry import ManifoldGeometryRegistry

# Import parameter management components
from memory.lucidia_memory_system.core.parameter_manager import ParameterManager
from memory.lucidia_memory_system.core.dream_parameter_adapter import DreamParameterAdapter
from memory.lucidia_memory_system.api.dream_parameter_api import router as parameter_router, init_dream_parameter_api
from memory.lucidia_memory_system.api.parameter_api import router as general_parameter_router, init_parameter_api

# Import client and service components 
from server.memory_client import EnhancedMemoryClient
from server.llm_pipeline import LocalLLMPipeline
from server.hypersphere_manager import HypersphereManager
from voice_core.config.config import LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dream_api_server.log")
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Lucidia Dream API",
    description="API for Lucidia's Dream Processor with dynamic parameter reconfiguration and memory system integration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for dependencies and connections
dream_processor = None
knowledge_graph = None
memory_client = None
llm_service = None
self_model = None
world_model = None
embedding_comparator = None
reflection_engine = None
hypersphere_manager = None
parameter_manager = None
dream_parameter_adapter = None

# WebSocket connections and locks
tensor_connection = None
hpc_connection = None
tensor_lock = asyncio.Lock()
hpc_lock = asyncio.Lock()

# Environment variables
STORAGE_PATH = os.getenv('LUCIDIA_STORAGE_PATH', './data')
TENSOR_SERVER_URL = os.getenv('TENSOR_SERVER_URL', 'ws://nemo_sig_v3:5001')
HPC_SERVER_URL = os.getenv('HPC_SERVER_URL', 'ws://nemo_sig_v3:5005')
LLM_API_ENDPOINT = os.getenv('LLM_API_ENDPOINT', 'http://localhost:1234/v1/chat/completions')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen2.5-7b-instruct')
PING_INTERVAL = float(os.getenv('PING_INTERVAL', '30.0'))
CONNECTION_RETRY_LIMIT = int(os.getenv('CONNECTION_RETRY_LIMIT', '5'))
CONNECTION_RETRY_DELAY = float(os.getenv('CONNECTION_RETRY_DELAY', '2.0'))
CONNECTION_TIMEOUT = float(os.getenv('CONNECTION_TIMEOUT', '10.0'))
DEFAULT_MODEL_VERSION = os.getenv('DEFAULT_MODEL_VERSION', 'latest')
CONFIG_PATH = os.environ.get("LUCIDIA_CONFIG_PATH", "config/default_config.json")

# Create a patched version of WorldModel that initializes entity_importance
class PatchedWorldModel(LucidiaWorldModel):
    def __init__(self, config=None):
        # Initialize the missing attribute before parent __init__ calls _initialize_core_entities
        self.entity_importance = {}
        super().__init__(config)

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
                    raise Exception("Tensor server unavailable")

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
                    raise Exception("HPC server unavailable")

# ========= Dependency Functions =========

def get_dream_processor():
    return dream_processor

def get_knowledge_graph():
    return knowledge_graph

def get_self_model():
    return self_model

def get_world_model():
    return world_model

def get_embedding_comparator():
    return embedding_comparator

def get_reflection_engine():
    return reflection_engine

def get_hypersphere_manager():
    return hypersphere_manager

def get_parameter_manager():
    return parameter_manager

def get_dream_parameter_adapter():
    return dream_parameter_adapter

# Initialize components
async def initialize_components():
    global dream_processor, knowledge_graph, memory_client, llm_service, self_model, world_model
    global embedding_comparator, reflection_engine, hypersphere_manager, parameter_manager, dream_parameter_adapter
    
    try:
        logger.info(f"Using storage path: {STORAGE_PATH}")
        logger.info(f"Loading configuration from {CONFIG_PATH}")
        
        # Ensure storage directories exist
        os.makedirs(f"{STORAGE_PATH}/self_model", exist_ok=True)
        os.makedirs(f"{STORAGE_PATH}/world_model", exist_ok=True)
        os.makedirs(f"{STORAGE_PATH}/knowledge_graph", exist_ok=True)
        os.makedirs(f"{STORAGE_PATH}/dreams", exist_ok=True)
        os.makedirs(f"{STORAGE_PATH}/reflection", exist_ok=True)
        
        # Initialize parameter manager first
        logger.info("Initializing parameter manager...")
        parameter_manager = ParameterManager(initial_config=CONFIG_PATH)
        
        logger.info("Initializing knowledge graph...")
        knowledge_graph = KnowledgeGraph(config={
            "storage_directory": f"{STORAGE_PATH}/knowledge_graph"
        })
        
        logger.info("Initializing self model...")
        self_model = SelfModel(config={
            "storage_directory": f"{STORAGE_PATH}/self_model",
            "show_ascii": True
        })
        
        logger.info("Initializing world model...")
        world_model = PatchedWorldModel(config={
            "storage_directory": f"{STORAGE_PATH}/world_model"
        })
        
        logger.info("Initializing memory client...")
        # Connect to tensor and HPC servers running in the same Docker network
        memory_client = EnhancedMemoryClient(config={
            "tensor_server_url": TENSOR_SERVER_URL,
            "hpc_server_url": HPC_SERVER_URL,
            "ping_interval": PING_INTERVAL,
            "max_retries": CONNECTION_RETRY_LIMIT,
            "retry_delay": CONNECTION_RETRY_DELAY
        })
        await memory_client.initialize()
        
        logger.info("Initializing HypersphereManager for enhanced embedding operations...")
        hypersphere_manager = HypersphereManager(
            memory_client=memory_client,
            config={
                "max_connections": 5,
                "min_batch_size": 1,
                "max_batch_size": 32,
                "target_latency": 100,  # ms
                "default_model_version": DEFAULT_MODEL_VERSION,
                "supported_model_versions": ["latest", "v1", "v2"],
                "batch_timeout": 0.1,  # seconds
                "use_circuit_breaker": True
            }
        )
        await hypersphere_manager.initialize()
        
        logger.info("Initializing LLM service for dream processing...")
        # Create LLM configuration
        llm_config = LLMConfig(
            api_endpoint=LLM_API_ENDPOINT,
            model=LLM_MODEL,
            system_prompt="You are Lucidia's reflection system, analyzing performance and suggesting improvements during dream processing.",
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '1024')),
            timeout=30
        )
        llm_service = LocalLLMPipeline(config=llm_config)
        await llm_service.initialize()
        
        # Set memory_core in models
        self_model.memory_core = memory_client
        world_model.memory_core = memory_client
        
        # Set LLM service in models
        self_model.llm_service = llm_service
        world_model.llm_service = llm_service
        
        logger.info("Initializing embedding comparator...")
        embedding_comparator = EmbeddingComparator(hpc_client=memory_client)
        
        # Merge parameter_manager's config with additional config for dream processor
        combined_config = parameter_manager.config.copy() if parameter_manager.config else {}
        combined_config.update({
            "storage_path": f"{STORAGE_PATH}/dreams",
            "llm_service": llm_service
        })
        
        logger.info("Initializing dream processor...")
        dream_processor = DreamProcessor(
            self_model=self_model,
            world_model=world_model,
            knowledge_graph=knowledge_graph,
            config=combined_config
        )
        
        # Initialize dream parameter adapter to connect parameter manager with dream processor
        logger.info("Initializing dream parameter adapter...")
        dream_parameter_adapter = DreamParameterAdapter(dream_processor, parameter_manager)
        
        # Initialize parameter API routers
        init_parameter_api(parameter_manager)
        init_dream_parameter_api(dream_parameter_adapter)
        
        logger.info("Initializing reflection engine...")
        memory_integration = MemoryIntegration(
            config={
                "memory_client": memory_client,
                "knowledge_graph": knowledge_graph
            }
        )
        reflection_engine = ReflectionEngine(
            knowledge_graph=knowledge_graph,
            memory_integration=memory_integration,
            llm_service=llm_service,
            hypersphere_dispatcher=hypersphere_manager.dispatcher,
            config={
                "storage_path": f"{STORAGE_PATH}/reflection",
                "domain": "synthien_studies",
                "default_model_version": DEFAULT_MODEL_VERSION
            }
        )
        
        # Store components in app state for dependency injection
        app.state.memory_client = memory_client
        app.state.dream_processor = dream_processor
        app.state.self_model = self_model
        app.state.world_model = world_model
        app.state.knowledge_graph = knowledge_graph
        app.state.embedding_comparator = embedding_comparator
        app.state.llm_service = llm_service
        app.state.reflection_engine = reflection_engine
        app.state.hypersphere_manager = hypersphere_manager
        app.state.parameter_manager = parameter_manager
        app.state.dream_parameter_adapter = dream_parameter_adapter
        
        logger.info("All components initialized successfully")
        
        # Schedule a background task for continuous dream processing when idle
        asyncio.create_task(continuous_dream_processing())
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}", exc_info=True)
        raise

# Background task for continuous dream processing
async def continuous_dream_processing():
    global dream_processor, self_model, world_model
    
    # Wait for initial startup to complete
    await asyncio.sleep(60)
    
    while True:
        try:
            # Check if system is idle (no active voice sessions)
            is_idle = await check_system_idle()
            
            if is_idle and dream_processor and not dream_processor.is_dreaming:
                logger.info("System is idle, starting dream session...")
                
                # Get tensor and HPC connections
                try:
                    tensor_conn = await get_tensor_connection()
                    hpc_conn = await get_hpc_connection()
                except Exception as e:
                    logger.error(f"Error establishing connections for dream session: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying
                    continue
                
                # Perform self-reflection before dream session
                try:
                    logger.info("Performing self-model reflection...")
                    await self_model.reflect(["performance", "improvement"])
                except Exception as e:
                    logger.error(f"Error during self-reflection: {e}")
                
                # Analyze world model knowledge graph
                try:
                    logger.info("Analyzing world model knowledge graph...")
                    await world_model.analyze_concept_network()
                except Exception as e:
                    logger.error(f"Error analyzing world model: {e}")
                
                # Start a dream session with the connections
                try:
                    await dream_processor.schedule_dream_session(
                        duration_minutes=15,
                        tensor_connection=tensor_conn,
                        hpc_connection=hpc_conn,
                        priority="low"  # Use low priority for background processing
                    )
                except Exception as e:
                    logger.error(f"Error starting dream session: {e}")
                
                # Wait for dream session to complete
                while dream_processor.is_dreaming:
                    await asyncio.sleep(30)
                
                # Wait before checking again
                await asyncio.sleep(300)  # 5 minutes
            else:
                # Check again in 1 minute
                await asyncio.sleep(60)
                
        except Exception as e:
            logger.error(f"Error in continuous dream processing: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying

# Check if the system is idle (no active voice sessions)
async def check_system_idle():
    # This is a placeholder - implement actual idle detection logic
    try:
        # Check if current time is within idle hours (1 AM to 5 AM by default)
        current_hour = datetime.now().hour
        idle_hours = os.getenv('IDLE_HOURS', '1-5')
        
        # Parse idle hours range
        start_hour, end_hour = map(int, idle_hours.split('-'))
        
        # Check if current hour is within idle range
        is_idle_time = start_hour <= current_hour <= end_hour
        
        # Check for active sessions (placeholder)
        has_active_sessions = False  # Replace with actual check
        
        # Check for recent memory operations (placeholder)
        last_memory_op_time = None  # Replace with actual check
        recent_memory_activity = False  # Replace with actual check
        
        # System is idle if it's idle time and there are no active sessions or recent activity
        return is_idle_time and not has_active_sessions and not recent_memory_activity
        
    except Exception as e:
        logger.error(f"Error checking system idle state: {e}")
        # Default to not idle if there's an error checking
        return False

# Register the routers
app.include_router(dream_router)
app.include_router(parameter_router)
app.include_router(general_parameter_router)

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Dream API server...")
    await initialize_components()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Dream API server...")
    
    # Close connections
    try:
        # Shutdown reflection engine if it's running
        if reflection_engine and hasattr(reflection_engine, 'stop') and callable(reflection_engine.stop):
            await reflection_engine.stop()
            logger.info("Reflection engine stopped")
        
        # Shutdown hypersphere manager
        if hypersphere_manager and hasattr(hypersphere_manager, 'shutdown') and callable(hypersphere_manager.shutdown):
            await hypersphere_manager.shutdown()
            logger.info("Hypersphere manager shut down")
            
        # Close memory client connections
        if memory_client and hasattr(memory_client, 'close_connections') and callable(memory_client.close_connections):
            await memory_client.close_connections()
            logger.info("Memory client connections closed")
            
        # Close WebSocket connections
        global tensor_connection, hpc_connection
        if tensor_connection and not tensor_connection.closed:
            await tensor_connection.close()
            logger.info("Tensor server connection closed")
        
        if hpc_connection and not hpc_connection.closed:
            await hpc_connection.close()
            logger.info("HPC server connection closed")
            
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        
    logger.info("Dream API server shutdown complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "dream_api",
        "components": {
            "dream_processor": "initialized" if dream_processor else "not_initialized",
            "parameter_manager": "initialized" if parameter_manager else "not_initialized",
            "knowledge_graph": "initialized" if knowledge_graph else "not_initialized",
            "self_model": "initialized" if self_model else "not_initialized",
            "world_model": "initialized" if world_model else "not_initialized",
            "embedding_comparator": "initialized" if embedding_comparator else "not_initialized"
        },
        "connections": {
            "tensor_server": tensor_connection is not None and not tensor_connection.closed,
            "hpc_server": hpc_connection is not None and not hpc_connection.closed,
            "tensor_server_url": TENSOR_SERVER_URL,
            "hpc_server_url": HPC_SERVER_URL
        }
    }

# Make app_dependencies available for the router
import sys
sys.modules["app_dependencies"] = sys.modules[__name__]

# Run the server
if __name__ == "__main__":
    port = int(os.getenv("DREAM_API_PORT", "8080"))
    logger.info(f"Starting Dream API server on port {port}...")
    uvicorn.run("dream_api_server:app", host="0.0.0.0", port=port, log_level="info")