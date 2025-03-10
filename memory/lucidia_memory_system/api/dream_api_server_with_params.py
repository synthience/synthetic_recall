import os
import sys
import logging
import asyncio
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

# Set up paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dream components
from core.dream_processor import LucidiaDreamProcessor
from core.parameter_manager import ParameterManager
from core.dream_parameter_adapter import DreamParameterAdapter

# Import API routers
from api.dream_api import router as dream_router
from api.dream_parameter_api import router as parameter_router, init_dream_parameter_api
from api.parameter_api import router as general_parameter_router, init_parameter_api

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

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global dream_processor, parameter_manager, dream_parameter_adapter
    
    logger.info("Initializing Dream API server components")
    
    try:
        # Load configuration from environment or default
        config_path = os.environ.get("LUCIDIA_CONFIG_PATH", "config/default_config.json")
        logger.info(f"Loading configuration from {config_path}")
        
        # Initialize parameter manager
        parameter_manager = ParameterManager(initial_config=config_path)
        logger.info("Parameter manager initialized")
        
        # Initialize dream processor with parameter manager's config
        dream_processor = LucidiaDreamProcessor(
            config=parameter_manager.config
        )
        logger.info("Dream processor initialized")
        
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
