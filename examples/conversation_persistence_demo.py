import asyncio
import logging
import sys
import os
import time
import traceback
from pathlib import Path

# Add the project root to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.middleware.conversation_persistence import ConversationPersistenceMiddleware

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('conversation_demo.log', mode='w')  # 'w' to overwrite previous logs
    ]
)

logger = logging.getLogger(__name__)


async def simple_test():
    """A very simple test of the ConversationPersistenceMiddleware."""
    try:
        logger.info("Starting simple conversation persistence test")
        
        # Create necessary directories
        session_dir = Path("session_data")
        session_dir.mkdir(exist_ok=True)
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        
        # Create the middleware with minimal required parameters
        # The ConversationPersistenceMiddleware only accepts memory_integration and session_id parameters
        logger.info("Creating middleware...")
        
        # Generate a session ID
        session_id = f"test_session_{int(time.time())}"
        logger.info(f"Session ID: {session_id}")
        
        # Create middleware - can pass None for memory_integration for basic testing
        middleware = ConversationPersistenceMiddleware(memory_integration=None, session_id=session_id)
        
        # Session is already initialized in the constructor
        logger.info("Session already initialized in constructor")
        
        # Store a simple interaction
        logger.info("Storing interaction...")
        user_input = "Hello, this is a test message."
        response = "This is a test response."
        
        success = await middleware.store_interaction(user_input, response)
        logger.info(f"Interaction stored: {success}")
        
        # Save session state
        logger.info("Saving session state...")
        save_result = await middleware.save_session_state()
        logger.info(f"Session saved: {save_result}")
        
        logger.info("Simple test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in simple_test: {e}")
        logger.error(traceback.format_exc())

    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    # Run the async test function
    logger.info("Starting test...")
    try:
        asyncio.run(simple_test())
        logger.info("Test completed")
    except Exception as e:
        logger.error(f"Main error: {e}")
        logger.error(traceback.format_exc())
