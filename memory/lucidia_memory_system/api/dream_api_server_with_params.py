import os
import sys
import logging
import asyncio
import json
import time
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

# Set up paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import dream components
from memory.lucidia_memory_system.core.dream_processor import LucidiaDreamProcessor
from memory.lucidia_memory_system.core.parameter_manager import ParameterManager
from memory.lucidia_memory_system.core.dream_parameter_adapter import DreamParameterAdapter

# Import API routers
from memory.lucidia_memory_system.api.dream_api import router as dream_router
from memory.lucidia_memory_system.api.dream_parameter_api import router as parameter_router, init_dream_parameter_api
from memory.lucidia_memory_system.api.parameter_api import router as general_parameter_router, init_parameter_api

# Import memory persistence handler
from memory.storage.memory_persistence_handler import MemoryPersistenceHandler
from memory.lucidia_memory_system.core.memory_types import MemoryEntry, MemoryTypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dream_api_server")

# Create FastAPI app
app = FastAPI(
    title="Lucidia Dream API",
    description="API for Lucidia's Dream Processor with dynamic parameter reconfiguration",
    version="1.2.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
dream_processor = None
parameter_manager = None
dream_parameter_adapter = None
memory_persistence = None

async def initialize_and_save_models(self_model, world_model, memory_persistence):
    """
    Initialize and save the self_model and world_model to persistent storage.
    
    Args:
        self_model: The LucidiaSelfModel instance
        world_model: The LucidiaWorldModel instance
        memory_persistence: The MemoryPersistenceHandler instance
    """
    logger.info("Saving self_model and world_model to persistent storage")
    
    try:
        # Prepare self_model data for storage
        self_model_data = {
            'content': json.dumps({
                'identity': self_model.identity,
                'self_awareness': self_model.self_awareness,
                'core_awareness': self_model.core_awareness,
                'personality': dict(self_model.personality),
                'core_values': self_model.core_values,
                'emotional_state': self_model.emotional_state,
                'reflective_capacity': self_model.reflective_capacity,
                'version': self_model.identity.get('version', '1.0'),
                'last_update': time.time()
            }),
            'memory_type': MemoryTypes.MODEL.value,
            'metadata': {
                'self_model_version': '1.0',
                'timestamp': time.time(),
                'type': 'self_model'
            }
        }
        
        # Prepare world_model data for storage
        world_model_data = {
            'content': json.dumps({
                'reality_framework': world_model.reality_framework,
                'knowledge_domains': world_model.knowledge_domains,
                'conceptual_networks': world_model.conceptual_networks,
                'epistemological_framework': world_model.epistemological_framework,
                'belief_system': world_model.belief_system,
                'verification_methods': world_model.verification_methods,
                'causal_models': world_model.causal_models,
                'version': world_model.version,
                'last_update': time.time()
            }),
            'memory_type': MemoryTypes.MODEL.value,
            'metadata': {
                'world_model_version': world_model.version,
                'timestamp': time.time(),
                'type': 'world_model'
            }
        }
        
        # Store models in persistence
        await memory_persistence.store_memory(memory_data=self_model_data, storage_key='self_model')
        await memory_persistence.store_memory(memory_data=world_model_data, storage_key='world_model')
        
        logger.info("Successfully saved self_model and world_model to persistent storage")
        return True
    except Exception as e:
        logger.error(f"Error saving models to persistent storage: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global dream_processor, parameter_manager, dream_parameter_adapter, memory_persistence
    
    logger.info("Initializing Dream API server components")
    
    try:
        # Load configuration from environment or default
        config_path = os.environ.get("LUCIDIA_CONFIG_PATH", "config/default_config.json")
        logger.info(f"Loading configuration from {config_path}")
        
        # Initialize parameter manager
        parameter_manager = ParameterManager(initial_config=config_path)
        logger.info("Parameter manager initialized")
        
        # Initialize memory persistence handler if storage path is available
        storage_path = os.environ.get("MEMORY_STORAGE_PATH", "memory/stored")
        memory_persistence = MemoryPersistenceHandler(storage_path)
        logger.info(f"Memory persistence handler initialized at {storage_path}")
        
        # Initialize self_model and world_model if needed
        from memory.lucidia_memory_system.core.Self.self_model import LucidiaSelfModel
        from memory.lucidia_memory_system.core.World.world_model import LucidiaWorldModel
        
        # Try to load models from persistence first
        self_model_data = await memory_persistence.retrieve_memory("self_model")
        world_model_data = await memory_persistence.retrieve_memory("world_model")
        
        # Initialize models
        self_model = LucidiaSelfModel()
        world_model = LucidiaWorldModel()
        
        # Check if models were loaded from persistence
        models_loaded = self_model_data is not None and world_model_data is not None
        if not models_loaded:
            logger.info("Models not found in persistence, initializing and saving new models")
            # Save models to persistence if they don't exist
            success = await initialize_and_save_models(self_model, world_model, memory_persistence)
            if success:
                logger.info("Successfully initialized and saved models to persistence")
            else:
                logger.warning("Failed to save models to persistence, continuing with in-memory models")
        else:
            logger.info("Models found in persistence, loading model data")
            try:
                # If we have model data, update the models with it
                if self_model_data and 'content' in self_model_data:
                    self_model_content = json.loads(self_model_data['content'])
                    # Update the self model with the stored data
                    self_model.identity = self_model_content.get('identity', self_model.identity)
                    self_model.self_awareness = self_model_content.get('self_awareness', self_model.self_awareness)
                    self_model.core_awareness = self_model_content.get('core_awareness', self_model.core_awareness)
                    self_model.personality = self_model_content.get('personality', self_model.personality)
                    self_model.core_values = self_model_content.get('core_values', self_model.core_values)
                    self_model.emotional_state = self_model_content.get('emotional_state', self_model.emotional_state)
                    self_model.reflective_capacity = self_model_content.get('reflective_capacity', self_model.reflective_capacity)
                    logger.info("Self model loaded from persistence")
                
                if world_model_data and 'content' in world_model_data:
                    world_model_content = json.loads(world_model_data['content'])
                    # Update the world model with the stored data
                    world_model.reality_framework = world_model_content.get('reality_framework', world_model.reality_framework)
                    world_model.knowledge_domains = world_model_content.get('knowledge_domains', world_model.knowledge_domains)
                    world_model.conceptual_networks = world_model_content.get('conceptual_networks', world_model.conceptual_networks)
                    world_model.epistemological_framework = world_model_content.get('epistemological_framework', world_model.epistemological_framework)
                    world_model.belief_system = world_model_content.get('belief_system', world_model.belief_system)
                    world_model.verification_methods = world_model_content.get('verification_methods', world_model.verification_methods)
                    world_model.causal_models = world_model_content.get('causal_models', world_model.causal_models)
                    world_model.version = world_model_content.get('version', world_model.version)
                    logger.info("World model loaded from persistence")
            except Exception as e:
                logger.error(f"Error loading model data: {e}")
                # Re-save models in case the format was corrupted
                await initialize_and_save_models(self_model, world_model, memory_persistence)
        
        # Initialize dream processor with parameter manager's config and models
        dream_processor = LucidiaDreamProcessor(
            config=parameter_manager.config,
            self_model=self_model,
            world_model=world_model
        )
        logger.info("Dream processor initialized with models")
        
        # Connect parameter manager with dream processor through adapter
        dream_parameter_adapter = DreamParameterAdapter(dream_processor, parameter_manager)
        logger.info("Dream parameter adapter initialized")
        
        # Initialize parameter API routers
        init_parameter_api(parameter_manager)
        init_dream_parameter_api(dream_parameter_adapter)
        
        logger.info("Dream API server initialization complete")
    
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Dream API server")
    
    # Perform any necessary cleanup
    # (parameter manager and dream processor don't require special cleanup)
    
    logger.info("Shutdown complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "dream_processor": "initialized" if dream_processor else "not_initialized",
        "parameter_manager": "initialized" if parameter_manager else "not_initialized"
    }

# Include routers
app.include_router(dream_router, prefix="/api")
app.include_router(parameter_router, prefix="/api")
app.include_router(general_parameter_router, prefix="/api")

# Main function to run the server directly
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("DREAM_API_PORT", 8000))
    
    # Run the server
    uvicorn.run(
        "dream_api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
