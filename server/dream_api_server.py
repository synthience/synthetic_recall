# dream_api_server.py
import asyncio
import logging
import os
import sys
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the dream API router
from server.dream_api import router as dream_router

# Import memory system components
from memory.lucidia_memory_system.core.dream_processor import LucidiaDreamProcessor as DreamProcessor
from memory.lucidia_memory_system.core.knowledge_graph import LucidiaKnowledgeGraph as KnowledgeGraph
# Import the enhanced models from the correct paths
from memory.lucidia_memory_system.core.Self.self_model import LucidiaSelfModel as SelfModel
from memory.lucidia_memory_system.core.World.world_model import LucidiaWorldModel
from memory.lucidia_memory_system.core.embedding_comparator import EmbeddingComparator
from memory.lucidia_memory_system.core.integration import MemoryIntegration
from memory.lucidia_memory_system.core.reflection_engine import ReflectionEngine
from memory.lucidia_memory_system.core.hypersphere_dispatcher import HypersphereDispatcher
from memory.lucidia_memory_system.core.manifold_geometry import ManifoldGeometryRegistry
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
app = FastAPI(title="Lucidia Dream API", description="API for dream processing in Lucidia Memory System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for dependencies
dream_processor = None
knowledge_graph = None
memory_client = None
llm_service = None
self_model = None
world_model = None  # Added world_model
embedding_comparator = None
reflection_engine = None
hypersphere_manager = None

# Environment variables
STORAGE_PATH = os.getenv('LUCIDIA_STORAGE_PATH', './data')
TENSOR_SERVER_URL = os.getenv('TENSOR_SERVER_URL', 'ws://nemo_sig_v3:5001')
HPC_SERVER_URL = os.getenv('HPC_SERVER_URL', 'ws://nemo_sig_v3:5005')
LLM_API_ENDPOINT = os.getenv('LLM_API_ENDPOINT', 'http://localhost:1234/v1/chat/completions')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen2.5-7b-instruct')
PING_INTERVAL = float(os.getenv('PING_INTERVAL', '30.0'))
CONNECTION_RETRY_LIMIT = int(os.getenv('CONNECTION_RETRY_LIMIT', '5'))
CONNECTION_RETRY_DELAY = float(os.getenv('CONNECTION_RETRY_DELAY', '2.0'))
DEFAULT_MODEL_VERSION = os.getenv('DEFAULT_MODEL_VERSION', 'latest')

# Dependency to get dream processor instance
def get_dream_processor():
    return dream_processor

# Dependency to get knowledge graph instance
def get_knowledge_graph():
    return knowledge_graph

# Dependency to get self model instance
def get_self_model():
    return self_model

# Dependency to get world model instance
def get_world_model():
    return world_model

# Dependency to get embedding comparator
def get_embedding_comparator():
    return embedding_comparator

# Dependency to get reflection engine instance
def get_reflection_engine():
    return reflection_engine

# Dependency to get hypersphere manager instance
def get_hypersphere_manager():
    return hypersphere_manager

# Create a patched version of WorldModel that initializes entity_importance
class PatchedWorldModel(LucidiaWorldModel):
    def __init__(self, config=None):
        # Initialize the missing attribute before parent __init__ calls _initialize_core_entities
        self.entity_importance = {}
        super().__init__(config)

# Initialize components
async def initialize_components():
    global dream_processor, knowledge_graph, memory_client, llm_service, self_model, world_model, embedding_comparator, reflection_engine, hypersphere_manager
    
    try:
        logger.info(f"Using storage path: {STORAGE_PATH}")
        
        # Ensure storage directories exist
        os.makedirs(f"{STORAGE_PATH}/self_model", exist_ok=True)
        os.makedirs(f"{STORAGE_PATH}/world_model", exist_ok=True)
        os.makedirs(f"{STORAGE_PATH}/knowledge_graph", exist_ok=True)
        os.makedirs(f"{STORAGE_PATH}/dreams", exist_ok=True)
        os.makedirs(f"{STORAGE_PATH}/reflection", exist_ok=True)
        
        logger.info("Initializing knowledge graph...")
        knowledge_graph = KnowledgeGraph(config={
            "storage_directory": f"{STORAGE_PATH}/knowledge_graph"
        })
        # Knowledge graph doesn't have load method - it initializes in constructor
        
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
        # Note: We don't establish the connections here, they will be established on-demand
        # by the dream_api.py get_tensor_connection() and get_hpc_connection() functions
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
        
        logger.info("Initializing dream processor...")
        dream_processor = DreamProcessor(
            self_model=self_model,
            world_model=world_model,
            knowledge_graph=knowledge_graph,
            config={
                "storage_path": f"{STORAGE_PATH}/dreams",
                "llm_service": llm_service
            }
        )
        
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
        app.state.llm_service = llm_service  # Make LLM service available
        app.state.reflection_engine = reflection_engine  # Make reflection engine available
        app.state.hypersphere_manager = hypersphere_manager  # Make hypersphere manager available
        
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
                
                # Get tensor and HPC connections from dream_api
                from server.dream_api import get_tensor_connection, get_hpc_connection
                tensor_conn = await get_tensor_connection()
                hpc_conn = await get_hpc_connection()
                
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
                await dream_processor.schedule_dream_session(
                    duration_minutes=15,
                    tensor_connection=tensor_conn,
                    hpc_connection=hpc_conn,
                    priority="low"  # Use low priority for background processing
                )
                
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
    # For example, check if there are any active LiveKit sessions
    # or if there have been any memory operations in the last 30 minutes
    
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

# Register the dream API router
app.include_router(dream_router)

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
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        
    logger.info("Dream API server shutdown complete")

# Make app_dependencies available for the router
import sys
sys.modules["app_dependencies"] = sys.modules[__name__]

# Run the server
if __name__ == "__main__":
    port = int(os.getenv("DREAM_API_PORT", "8080"))
    logger.info(f"Starting Dream API server on port {port}...")
    uvicorn.run("dream_api_server:app", host="0.0.0.0", port=port, log_level="info")
