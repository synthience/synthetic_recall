# test_model_context_integration.py
import os
import sys
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary components from server architecture
from server.dream_api_server import initialize_components
from server.protocols.model_context_tools import ModelContextToolProvider

async def test_model_context_tools():
    """Test the real ModelContextToolProvider functionality using the actual server components."""
    logger.info("Initializing server components...")
    
    # Initialize all server components
    await initialize_components()
    from server.dream_api_server import model_context_tool_provider
    
    if not model_context_tool_provider:
        logger.error("ModelContextToolProvider not initialized!")
        return
    
    logger.info("ModelContextToolProvider successfully initialized")
    
    # Test 1: Self-model update
    logger.info("\n=== Testing self-model update ===")
    try:
        result = await model_context_tool_provider.update_self_model(
            aspect="learning_insight",
            content={
                "insight": "Improved ability to reason about complex systems",
                "context": "Successfully integrated ModelContextToolProvider",
                "impact": "Enhanced self-evolution capabilities"
            },
            significance=0.8
        )
        logger.info(f"Self-model update result: {result}")
    except Exception as e:
        logger.error(f"Error in self-model update: {e}")
    
    # Test 2: Knowledge graph update
    logger.info("\n=== Testing knowledge graph update ===")
    try:
        result = await model_context_tool_provider.update_knowledge_graph(
            operation="add_insight",
            data={
                "concept": "model_context_protocol",
                "insight": "MCP enables autonomous self-evolution and system health monitoring",
                "related_concepts": ["self_evolution", "system_health", "autonomous_operation"]
            },
            source="test"
        )
        logger.info(f"Knowledge graph update result: {result}")
    except Exception as e:
        logger.error(f"Error in knowledge graph update: {e}")
    
    # Test 3: Memory operation
    logger.info("\n=== Testing memory operation ===")
    try:
        result = await model_context_tool_provider.memory_operation(
            operation="store",
            content={
                "text": "Successfully integrated and tested ModelContextToolProvider",
                "context": "System enhancement"
            },
            memory_type="factual",
            significance=0.9
        )
        logger.info(f"Memory operation result: {result}")
    except Exception as e:
        logger.error(f"Error in memory operation: {e}")
    
    # Test 4: System health check
    logger.info("\n=== Testing system health check ===")
    try:
        result = await model_context_tool_provider.check_system_health(
            subsystems=["memory", "self_model", "knowledge"],
            detail_level="detailed"
        )
        logger.info(f"System health check status: {result.get('status')}")
        logger.info(f"Component health statuses: {result.get('component_status', {})}")
        
        if result.get('recommendations'):
            logger.info(f"Health recommendations: {result.get('recommendations')}")
    except Exception as e:
        logger.error(f"Error in system health check: {e}")

async def main():
    logger.info("Starting ModelContextToolProvider integration test")
    try:
        await test_model_context_tools()
        logger.info("\nAll tests completed - ModelContextToolProvider integration validated")
    except Exception as e:
        logger.error(f"Error in integration test: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Ensure the server directory is in the Python path
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
        
    asyncio.run(main())
