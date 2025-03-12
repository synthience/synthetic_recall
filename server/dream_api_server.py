# dream_api_server.py
import asyncio
import logging
import os
import sys
import uvicorn
import json
import time
import torch
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import websockets
from websockets.exceptions import ConnectionClosed
from server.user_activity_tracker import UserActivityTracker
from server.memory_system import MemorySystem  # Import MemorySystem
from server.memory_bridge import MemoryBridge  # Import the new MemoryBridge
from server.model_manager import ModelManager, ModelPurpose
from server.resource_monitor import ResourceMonitor, SystemState
from server.dream_api import router as dream_router
from server.llm_pipeline import LocalLLMPipeline  # Add import for LLM Pipeline
from server.llm_manager import LLMManager  # Import the LLMManager from the dedicated module
from memory.lucidia_memory_system.core.dream_processor import LucidiaDreamProcessor as DreamProcessor
from memory.lucidia_memory_system.core.knowledge_graph import LucidiaKnowledgeGraph as KnowledgeGraph
from memory.lucidia_memory_system.core.Self.self_model import LucidiaSelfModel as SelfModel
from memory.lucidia_memory_system.core.World.world_model import LucidiaWorldModel
from memory.lucidia_memory_system.core.embedding_comparator import EmbeddingComparator
from memory.lucidia_memory_system.core.reflection_engine import ReflectionEngine
from memory.lucidia_memory_system.core.hypersphere_dispatcher import HypersphereDispatcher
from memory.lucidia_memory_system.memory_integration import MemoryIntegration
from memory.lucidia_memory_system.core.parameter_manager import ParameterManager
from memory.lucidia_memory_system.core.dream_parameter_adapter import DreamParameterAdapter
from memory.lucidia_memory_system.api.dream_parameter_api import router as parameter_router, init_dream_parameter_api
from memory.lucidia_memory_system.api.parameter_api import router as general_parameter_router, init_parameter_api

# Import client and service components 
from server.memory_client import EnhancedMemoryClient
from voice_core.config.config import LLMConfig

# Import configuration
from pathlib import Path

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
user_activity_tracker = None
memory_system = None  # Added memory_system global variable
memory_bridge = None  # Added memory_bridge global variable
model_manager = None  # Added model_manager global variable
resource_monitor = None  # Added resource_monitor global variable
llm_manager = None  # Added llm_manager global variable
rag_integration_service = None  # Added rag_integration_service global variable

# WebSocket connections and locks
tensor_connection = None
hpc_connection = None
tensor_lock = asyncio.Lock()
hpc_lock = asyncio.Lock()

# Environment variables
STORAGE_PATH = os.getenv('STORAGE_PATH', './data')
LUCIDIA_STORAGE_PATH = os.getenv('LUCIDIA_STORAGE_PATH', '/app/memory')
TENSOR_SERVER_URL = os.getenv('TENSOR_SERVER_URL', 'ws://nemo_sig_v3:5001')
HPC_SERVER_URL = os.getenv('HPC_SERVER_URL', 'ws://nemo_sig_v3:5005')
LLM_API_ENDPOINT = os.getenv('LLM_API_ENDPOINT', 'http://localhost:1234/v1/chat/completions')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen2.5-7b-instruct')
PING_INTERVAL = float(os.getenv('PING_INTERVAL', '30.0'))
CONNECTION_RETRY_LIMIT = int(os.getenv('CONNECTION_RETRY_LIMIT', '5'))
CONNECTION_RETRY_DELAY = float(os.getenv('CONNECTION_RETRY_DELAY', '2.0'))
CONNECTION_TIMEOUT = float(os.getenv('CONNECTION_TIMEOUT', '10.0'))
DEFAULT_MODEL_VERSION = os.getenv('DEFAULT_MODEL_VERSION', 'latest')
# Get the project root directory for absolute path resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use absolute path for configuration
CONFIG_PATH = os.environ.get("LUCIDIA_CONFIG_PATH", os.path.join(PROJECT_ROOT, "config", "lucidia_config.json"))

# Load configuration
settings = {}
try:
    # Check multiple possible locations for configuration
    possible_config_paths = [
        Path("config/server_config.json"),
        Path(os.path.join(PROJECT_ROOT, "config", "server_config.json")),
        Path(os.path.join(os.path.dirname(__file__), "config", "server_config.json")),
        Path(os.path.join(PROJECT_ROOT, "workspace", "config", "server_config.json"))
    ]
    
    config_path = None
    for path in possible_config_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path:
        with open(config_path, "r") as f:
            settings = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Configuration file not found in any of the standard locations. Using defaults.")
        settings = {}
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    settings = {}

# Load LM Studio configuration if available
try:
    lm_studio_config_path = Path("config/lm_studio_config.json")
    if lm_studio_config_path.exists():
        with open(lm_studio_config_path, "r") as f:
            lm_studio_config = json.load(f)
        
        # Only add the configuration if it's enabled
        if lm_studio_config.get("enabled", False):
            settings["lm_studio"] = lm_studio_config
            logger.info(f"LM Studio integration enabled with URL: {lm_studio_config.get('url')}")
        else:
            logger.info("LM Studio integration is disabled in config")
except Exception as e:
    logger.error(f"Error loading LM Studio configuration: {e}")

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

def get_user_activity_tracker():
    return UserActivityTracker.get_instance()

def get_llm_manager():
    return llm_manager

# Initialize components
async def initialize_components():
    """Initialize all system components."""
    global dream_processor, knowledge_graph, self_model, world_model, embedding_comparator, reflection_engine
    global hypersphere_manager, parameter_manager, dream_parameter_adapter, user_activity_tracker, memory_system
    global memory_bridge, model_manager, resource_monitor, llm_service, llm_manager, rag_integration_service
    
    try:
        logger.info("Starting Dream API server on port 8080...")
        
        # Ensure storage directory exists
        os.makedirs(STORAGE_PATH, exist_ok=True)
        
        # Initialize parameter manager
        logger.info("Initializing parameter manager...")
        parameter_manager = ParameterManager(initial_config=settings)
        
        # Initialize model manager with configuration
        logger.info("Initializing model manager...")
        model_manager = ModelManager(config_path=CONFIG_PATH)
        
        # Initialize resource monitor
        logger.info("Initializing resource monitor...")
        resource_monitor = ResourceMonitor.get_instance()
        
        # Initialize unified memory system
        logger.info("Initializing unified memory system...")
        memory_storage_path = os.path.join(LUCIDIA_STORAGE_PATH, "stored")
        os.makedirs(memory_storage_path, exist_ok=True)
        memory_system = MemorySystem(config={
            'storage_path': memory_storage_path,
            'embedding_dim': settings.get("memory", {}).get("embedding_dim", 1024)
        })
        
        logger.info("Initializing memory bridge...")
        memory_bridge = MemoryBridge(memory_system=memory_system)
        
        logger.info("Initializing memory client...")
        # Connect to tensor and HPC servers running in the same Docker network
        memory_client = EnhancedMemoryClient(config={
            "tensor_server_url": TENSOR_SERVER_URL,
            "hpc_server_url": HPC_SERVER_URL,
            "ping_interval": PING_INTERVAL,
            "max_retries": CONNECTION_RETRY_LIMIT,
            "retry_delay": CONNECTION_RETRY_DELAY
        }, memory_system=memory_system, memory_bridge=memory_bridge)  # Pass the memory system and bridge directly
        await memory_client.initialize()
        
        logger.info("Initializing HypersphereDispatcher for enhanced embedding operations...")
        hypersphere_manager = HypersphereDispatcher(
            tensor_server_uri=TENSOR_SERVER_URL,
            hpc_server_uri=HPC_SERVER_URL,
            max_connections=5,
            min_batch_size=4,
            max_batch_size=32,
            target_latency=0.5,
            reconnect_backoff_min=0.1,
            reconnect_backoff_max=30.0,
            reconnect_backoff_factor=2.0,
            health_check_interval=60.0
        )
        await hypersphere_manager.start()
        
        logger.info("Initializing knowledge graph...")
        knowledge_graph = KnowledgeGraph({
            "memory_client": memory_client,
            "hypersphere_manager": hypersphere_manager,
            "parameter_manager": parameter_manager
        })
        
        logger.info("Initializing world model...")
        world_model = PatchedWorldModel({
            "memory_client": memory_client,
            "knowledge_graph": knowledge_graph,
            "parameter_manager": parameter_manager
        })
        
        logger.info("Initializing self model...")
        self_model = SelfModel({
            "memory_client": memory_client,
            "knowledge_graph": knowledge_graph,
            "world_model": world_model,
            "parameter_manager": parameter_manager
        })
        
        # Initialize self model connections with knowledge graph
        knowledge_graph.self_model = self_model
        knowledge_graph.world_model = world_model
        
        # Initialize model imports after all components are available
        logger.info("Initializing knowledge graph imports from models...")
        await knowledge_graph.initialize_model_imports()
        logger.info("Knowledge graph model imports complete")
        
        logger.info("Initializing embedding comparator...")
        embedding_comparator = EmbeddingComparator(hpc_client=memory_client)
        
        logger.info("Initializing LLM service...")
        # Configure LLM service to work with the model manager
        llm_config = LLMConfig(
            api_endpoint="http://127.0.0.1:1234/v1",  # Local LLM API endpoint
            model=model_manager.active_model if model_manager else "qwen2.5-7b-instruct",
            max_tokens=2048,
            temperature=0.7
        )
        llm_service = LocalLLMPipeline(config=llm_config)
        
        logger.info("Initializing LLM Manager...")
        # Create LLM manager with config matching its expected parameters
        llm_config = {
            # Use host.docker.internal to connect to host machine from Docker container
            "api_base_url": "http://host.docker.internal:1234/v1",
            "api_key": "lm-studio",  # Standard API key for LM Studio
            "default_model": parameter_manager.config.get("lm_studio", {}).get("model", "qwen2.5-7b-instruct"),
            "local_model": True,
            "allow_simulation": True  # Keep simulation enabled as fallback
        }
        logger.info(f"LLM Manager config: {llm_config}")
        llm_manager = LLMManager(llm_config=llm_config)
        # Initialize the LLM manager
        await llm_manager.initialize()
        
        # Initialize RAG Integration Service
        logger.info("Initializing RAG Integration Service...")
        from server.rag_integration_service import RAGIntegrationService
        rag_integration_service = RAGIntegrationService(
            memory_system=memory_system,
            knowledge_graph=knowledge_graph,
            parameter_manager=parameter_manager
        )
        # Initialize the RAG integration service
        await rag_integration_service.initialize()
        logger.info("RAG Integration Service initialized successfully")
        
        # Get the LM Studio URL configuration if it exists
        lm_studio_url = settings.get("lm_studio", {}).get("url", None)
        if lm_studio_url:
            # Add the LM Studio URL to the dream processor configuration
            settings["dream_processor"]["lm_studio_url"] = lm_studio_url
            logger.info(f"LM Studio integration enabled with URL: {lm_studio_url}")
        
        logger.info("Initializing dream processor...")
        dream_processor = DreamProcessor({
            "memory_client": memory_client,
            "knowledge_graph": knowledge_graph,
            "self_model": self_model,
            "world_model": world_model,
            "reflection_engine": reflection_engine,
            "parameter_manager": parameter_manager,
            "model_manager": model_manager,  # Pass model_manager to dream processor
            "resource_monitor": resource_monitor,  # Pass resource_monitor to dream processor
            "config": settings.get("dream_processor", {}),
            "tool_providers": []  # Empty list for now, will be populated later
        })
        
        logger.info("Initializing dream parameter adapter...")
        dream_parameter_adapter = DreamParameterAdapter(
            dream_processor=dream_processor,
            parameter_manager=parameter_manager
        )
        
        # Initialize Model Context Protocol (MCP) tool providers
        logger.info("Initializing MCP tool providers...")
        from server.protocols.tool_protocol import ToolProvider
        from server.protocols.dream_tools import DreamToolProvider
        from server.protocols.counterfactual_tools import CounterfactualToolProvider
        from server.protocols.spiral_tools import SpiralToolProvider
        from server.protocols.world_model_tools import WorldModelToolProvider
        from server.protocols.model_context_tools import ModelContextToolProvider
        
        # Initialize Dream Tool Provider
        dream_tool_provider = DreamToolProvider(
            dream_processor=dream_processor,
            memory_system=memory_system,
            knowledge_graph=knowledge_graph,
            parameter_manager=parameter_manager,
            model_manager=llm_manager
        )
        
        # Initialize Counterfactual Tool Provider
        counterfactual_tool_provider = CounterfactualToolProvider(
            self_model=self_model,
            world_model=world_model,
            memory_system=memory_system,
            knowledge_graph=knowledge_graph,
            parameter_manager=parameter_manager,
            model_manager=llm_manager
        )
        
        # Initialize Spiral Tool Provider
        spiral_tool_provider = SpiralToolProvider(
            self_model=self_model,
            knowledge_graph=knowledge_graph,
            memory_system=memory_system,
            spiral_manager=self_model.spiral_phase_manager if hasattr(self_model, "spiral_phase_manager") else None,
            parameter_manager=parameter_manager,
            model_manager=llm_manager
        )
        
        # Initialize World Model Tool Provider
        world_model_tool_provider = WorldModelToolProvider(
            world_model=world_model,
            knowledge_graph=knowledge_graph,
            memory_system=memory_system,
            parameter_manager=parameter_manager,
            model_manager=llm_manager
        )
        
        # Initialize Model Context Tool Provider
        model_context_tool_provider = ModelContextToolProvider(
            self_model=self_model,
            world_model=world_model,
            knowledge_graph=knowledge_graph,
            memory_system=memory_system,
            dream_processor=dream_processor,
            spiral_manager=self_model.spiral_phase_manager if hasattr(self_model, "spiral_phase_manager") else None,
            parameter_manager=parameter_manager,
            model_manager=model_manager,
            dream_parameter_adapter=dream_parameter_adapter
        )
        
        # Register tools with Dream Processor
        dream_processor.tool_provider = dream_tool_provider
        
        # Update tool_providers list with all initialized providers
        dream_processor.tool_providers = [dream_tool_provider, counterfactual_tool_provider, spiral_tool_provider, 
                                         world_model_tool_provider, model_context_tool_provider]
        
        # Register tools with RAG service if it implements ToolProtocol
        if hasattr(rag_integration_service, "register_tool"):
            # Share key tools between providers
            for provider in [dream_tool_provider, counterfactual_tool_provider, spiral_tool_provider, 
                            world_model_tool_provider, model_context_tool_provider]:
                for tool_name, tool_info in provider.tools.items():
                    if tool_name not in rag_integration_service.tools:
                        rag_integration_service.register_tool(
                            name=tool_name,
                            function=tool_info["function"],
                            description=tool_info["schema"]["function"]["description"],
                            parameters=tool_info["schema"]["function"]["parameters"]
                        )
        
        logger.info("MCP tool providers initialized successfully")
        
        logger.info("Initializing reflection engine...")
        memory_integration = MemoryIntegration(
            config={
                "memory_core_path": os.path.join(settings.get("memory_path", "/app/memory"), "hierarchical"),
                "knowledge_graph": knowledge_graph  # Add back knowledge_graph
            }
        )
        reflection_engine = ReflectionEngine(
            knowledge_graph=knowledge_graph,
            memory_integration=memory_integration,
            llm_service=llm_manager,  # Use the LLM Manager as the LLM service
            config={
                "storage_path": f"{LUCIDIA_STORAGE_PATH}/reflection",  # Add back storage path
                "domain": settings.get("domain", "lucidia"),
                "default_model_version": settings.get("default_model_version", "1.0")
            }
        )
        
        # Initialize user activity tracker
        user_activity_tracker = UserActivityTracker.get_instance()
        logger.info("Initialized UserActivityTracker")
        
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
        app.state.user_activity_tracker = user_activity_tracker
        app.state.memory_system = memory_system  # Add memory_system to app state
        app.state.memory_bridge = memory_bridge  # Add memory_bridge to app state
        app.state.llm_manager = llm_manager  # Add llm_manager to app state
        app.state.rag_integration_service = rag_integration_service  # Add rag_integration_service to app state
        app.state.config_path = CONFIG_PATH  # Store config path for parameter persistence
        
        logger.info("All components initialized successfully")
        
        # Start background task for continuous dream processing
        asyncio.create_task(continuous_dream_processing())
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Continuous background dream processing
async def continuous_dream_processing():
    """Continuous dream processing task that runs in the background
    when the system is idle. Uses the UserActivityTracker to determine
    when it's appropriate to run background processing tasks.
    """
    global dream_processor, self_model, world_model
    
    # Wait for initialization to complete
    await asyncio.sleep(10)  # Short delay to allow system to initialize
    
    while True:
        try:
            # Check if the system is idle
            is_idle = await check_system_idle()
            
            if is_idle:
                logger.info("System is idle, starting dream processing cycle")
                
                # Set system state to dreaming in the resource monitor
                if resource_monitor:
                    resource_monitor.set_system_state(SystemState.IDLE)
                
                # Switch to dream model if we have a model manager
                if model_manager:
                    # Get the dream model from config
                    config = {}
                    try:
                        if os.path.exists(CONFIG_PATH):
                            with open(CONFIG_PATH, 'r') as f:
                                config = json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading configuration: {e}")
                    
                    dream_model = config.get("models", {}).get("dream_model", "qwen_qwq-32b")
                    logger.info(f"Switching to dream model: {dream_model}")
                    
                    # Get the recommended model for dreaming
                    recommended_model = model_manager.get_recommended_model(ModelPurpose.DREAMING)
                    if recommended_model:
                        logger.info(f"Using recommended model for dreaming: {recommended_model}")
                        # Notify the dream processor to use this model
                        if dream_processor:
                            dream_processor.set_model(recommended_model)
                
                # Use the dream processor to generate and process dreams
                if dream_processor:
                    try:
                        # Generate a dream
                        logger.info("Generating dream...")
                        dream_result = await dream_processor.generate_dream()
                        
                        if dream_result and "dream_content" in dream_result:
                            dream_id = dream_result.get("dream_id", "unknown")
                            
                            # Log detailed dream information for Docker logs
                            logger.info(f"\n==== DREAM GENERATED ====")
                            logger.info(f"Dream ID: {dream_id}")
                            logger.info(f"Title: {dream_result.get('title', 'Untitled Dream')}")
                            logger.info(f"Dream Content Length: {len(dream_result.get('dream_content', ''))} characters")
                            
                            # Log the actual dream content in manageable chunks to avoid overflowing logs
                            dream_content = dream_result.get("dream_content", "")
                            if dream_content:
                                logger.info("Dream Content:")
                                # Split long content into chunks for better log readability
                                chunk_size = 1000
                                for i in range(0, len(dream_content), chunk_size):
                                    chunk = dream_content[i:i+chunk_size]
                                    logger.info(f"[Content {i//chunk_size + 1}] {chunk}")
                            
                            # Process the dream to extract insights and update knowledge
                            logger.info("Processing dream for insights...")
                            insights = await dream_processor.process_dream(dream_result)
                            
                            if insights:
                                # Log detailed insights information
                                logger.info(f"\n==== DREAM INSIGHTS ====")
                                logger.info(f"Extracted {len(insights)} insights from dream {dream_id}")
                                
                                for i, insight in enumerate(insights):
                                    # Format the insight for logs
                                    logger.info(f"Insight {i+1}: {insight.get('content', '')}")
                                    if 'confidence' in insight:
                                        logger.info(f"Confidence: {insight.get('confidence', 0):.2f}")
                                    if 'attributes' in insight:
                                        logger.info(f"Attributes: {json.dumps(insight.get('attributes', {}), indent=2)}")
                                
                                # Update knowledge graph with insights
                                if knowledge_graph:
                                    logger.info("Updating knowledge graph with dream insights...")
                                    for insight in insights:
                                        await knowledge_graph.add_node(
                                            node_id=insight.get("node_id"),
                                            node_type=insight.get("node_type", "dream_insight"),
                                            attributes=insight.get("attributes", {})
                                        )
                            else:
                                logger.info("No insights were extracted from the dream")
                            
                            # Perform reflection on the dream and its processing
                            if reflection_engine:
                                logger.info("Performing reflection on dream processing...")
                                reflection_result = await reflection_engine.reflect_on_dream(
                                    dream_content=dream_result.get("dream_content"),
                                    insights=insights
                                )
                                
                                # Log reflection results
                                if reflection_result:
                                    logger.info(f"\n==== DREAM REFLECTION ====")
                                    logger.info(f"Reflection title: {reflection_result.get('title', 'Untitled Reflection')}")
                                    
                                    # Log fragments
                                    fragments = reflection_result.get("fragments", [])
                                    if fragments:
                                        logger.info(f"Reflection fragments: {len(fragments)}")
                                        for i, fragment in enumerate(fragments):
                                            frag_type = fragment.get("type", "unknown").capitalize()
                                            content = fragment.get("content", "")
                                            confidence = fragment.get("confidence", 0)
                                            logger.info(f"Fragment {i+1} ({frag_type}): {content} (confidence: {confidence:.2f})")
                                    
                                    logger.info("Dream reflection complete")
                                else:
                                    logger.info("No reflection was generated for the dream")
                        else:
                            logger.warning("Dream generation failed or returned invalid results")
                            
                    except Exception as e:
                        logger.error(f"Error in dream processing: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Switch back to default model
                if model_manager:
                    default_model = config.get("models", {})
                    if isinstance(default_model, dict):
                        default_model = default_model.get("default_model", "qwen2.5-7b-instruct")
                    else:
                        default_model = "qwen2.5-7b-instruct"
                    logger.info(f"Switching back to default model: {default_model}")
                    recommended_model = model_manager.get_recommended_model(ModelPurpose.GENERAL)
                    if recommended_model:
                        logger.info(f"Using recommended model for general tasks: {recommended_model}")
                
                # Reset system state
                if resource_monitor:
                    resource_monitor.set_system_state(SystemState.ACTIVE)
            
            # Wait before checking again
            # Adaptive wait time: longer when idle, shorter when active
            wait_time = 10800 if is_idle else 60  # 3 hours if idle, 1 minute if active
            logger.info(f"Waiting {wait_time} seconds before next dream processing check")
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Error in continuous dream processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await asyncio.sleep(60)  # Wait a minute before retrying

# Check if the system is idle (no active voice sessions)
async def check_system_idle():
    """Check if the system is idle based on user activity and session status
    
    Returns:
        bool: True if the system is idle and can perform background processing
    """
    try:
        # Get activity tracker
        activity_tracker = UserActivityTracker.get_instance()
        
        # Check if current time is within idle hours (1 AM to 5 AM by default)
        current_hour = datetime.now().hour
        idle_hours = os.getenv('IDLE_HOURS', '1-5')
        
        # Parse idle hours range
        start_hour, end_hour = map(int, idle_hours.split('-'))
        
        # Check if current hour is within idle range
        is_idle_time = start_hour <= current_hour <= end_hour
        
        # Check for active sessions
        has_active_sessions = activity_tracker.get_active_sessions_count() > 0
        
        # Check if user is AFK
        is_user_afk = activity_tracker.is_afk()
        
        # Check for recent memory operations (last 10 minutes)
        recent_memory_activity = activity_tracker.has_recent_activity(seconds=600)
        
        # System is idle if either:
        # 1. It's within idle hours and there are no active sessions, or
        # 2. The user has been AFK for an extended period
        # Also, ensure no recent memory activity to avoid interrupting ongoing operations
        return ((is_idle_time and not has_active_sessions) or activity_tracker.is_extended_afk()) \
               and not recent_memory_activity
    except Exception as e:
        logger.error(f"Error checking system idle state: {e}")
        # Default to not idle if there's an error checking
        return False

# Register the routers
app.include_router(dream_router)
app.include_router(parameter_router)
app.include_router(general_parameter_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    try:
        # Record API call activity
        activity_tracker = UserActivityTracker.get_instance()
        activity_tracker.record_activity(activity_type="api_call", details={"endpoint": "/health"})
        
        # Check component health
        components_status = {
            "dream_processor": dream_processor is not None,
            "knowledge_graph": knowledge_graph is not None,
            "self_model": self_model is not None,
            "world_model": world_model is not None,
            "tensor_connection": tensor_connection is not None,
            "hpc_connection": hpc_connection is not None,
            "parameter_manager": parameter_manager is not None
        }
        
        # Add user activity stats
        user_activity_stats = activity_tracker.get_activity_stats()
        
        return {
            "status": "healthy",
            "time": datetime.now().isoformat(),
            "components": components_status,
            "user_activity": user_activity_stats
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "unhealthy", "error": str(e)}

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

# Make app_dependencies available for the router
import sys
sys.modules["app_dependencies"] = sys.modules[__name__]

# Run the server
if __name__ == "__main__":
    port = int(os.getenv("DREAM_API_PORT", "8080"))
    logger.info(f"Starting Dream API server on port {port}...")
    uvicorn.run("dream_api_server:app", host="0.0.0.0", port=port, log_level="info")