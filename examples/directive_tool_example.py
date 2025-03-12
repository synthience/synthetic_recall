"""
Example integration of the Contextual Tool Detector with the Lucidia Chat System

This example shows how to integrate the directive detection and tool execution
into the existing chat pipeline, allowing Lucidia to automatically recognize and
respond to user directives without explicit tool calls.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, List

# Set up path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Lucidia components
from memory.lucidia_memory_system.memory_integration import MemoryIntegration
from memory.lucidia_memory_system.core.Self.self_model import LucidiaSelfModel
from memory.lucidia_memory_system.core.World.world_model import LucidiaWorldModel
from memory.lucidia_memory_system.core.dream_processor import LucidiaDreamProcessor
from memory.lucidia_memory_system.core.knowledge_graph import LucidiaKnowledgeGraph

# Import directive detection components
from server.protocols.contextual_tool_detector import ContextualToolDetector, DirectivePattern
from server.protocols.dream_tools import DreamToolProvider
from server.protocols.directive_integration import DirectiveIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DirectiveExample")

class EnhancedLucidiaChat:
    """Enhanced Lucidia chat system with directive detection capabilities."""
    
    def __init__(self):
        """Initialize the enhanced Lucidia chat system."""
        self.memory_integration = None
        self.dream_processor = None
        self.knowledge_graph = None
        self.self_model = None
        self.world_model = None
        self.directive_integration = None
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all required components."""
        # Initialize memory and core components
        logger.info("Initializing memory integration...")
        self.memory_integration = MemoryIntegration()
        
        logger.info("Initializing knowledge graph...")
        self.knowledge_graph = LucidiaKnowledgeGraph()
        
        logger.info("Initializing self model...")
        self.self_model = LucidiaSelfModel()
        
        logger.info("Initializing world model...")
        self.world_model = LucidiaWorldModel()
        
        logger.info("Initializing dream processor...")
        self.dream_processor = LucidiaDreamProcessor(
            memory_integration=self.memory_integration,
            knowledge_graph=self.knowledge_graph,
            self_model=self.self_model,
            world_model=self.world_model
        )
        
        # Initialize directive integration
        logger.info("Initializing directive integration...")
        self.directive_integration = DirectiveIntegration(
            dream_processor=self.dream_processor,
            memory_system=self.memory_integration,
            knowledge_graph=self.knowledge_graph
        )
        
        # Register any additional custom directives
        self._register_custom_directives()
        
    def _register_custom_directives(self):
        """Register additional custom directives beyond the defaults."""
        # Register a custom directive for emotion analysis
        if self.directive_integration and self.directive_integration.detector:
            self.directive_integration.detector.register_directive(DirectivePattern(
                tool_name="analyze_emotion",
                patterns=[
                    r"(?:analyze|detect|recognize|identify)\s+(?:the\s+)?emotion(?:s)?\s+(?:in|from|of)\s+(.+)",
                    r"(?:how\s+do\s+I\s+feel\s+about)\s+(.+)"
                ],
                description="Analyzes emotions in text using sentiment analysis",
                parameter_extractors={
                    "text": lambda text: re.search(r"(?:in|from|of|about)\s+(.+)(?:\?|\.|$)", text).group(1)
                },
                default_params={"detailed": True},
                priority=9
            ))
            logger.info("Registered custom emotion analysis directive")
        
    async def process_message(self, message: str) -> str:
        """Process a user message with directive detection."""
        logger.info(f"Processing message: {message}")
        
        # First, check for directives
        directive_results = await self.directive_integration.process_message_for_directives(message)
        
        # Store the message in memory (simplified)
        await self.memory_integration.store_memory(message, role="user")
        
        # Retrieve relevant context from memory
        context = await self.memory_integration.get_relevant_memories(message, limit=5)
        
        # Generate response (simplified - in a real system this would use an LLM)
        response = f"Lucidia: I received your message: '{message}'. I've found {len(context)} relevant memories."
        
        # Inject directive results if any were detected and executed
        if directive_results and directive_results.get("detected_directives", 0) > 0:
            response = self.directive_integration.inject_directive_results(response, directive_results)
            
        # In a real system, we would use the LLM to generate a proper response here
        # including information from the relevant memories and directive execution results
        
        return response


async def main():
    """Run the directive integration example."""
    chat = EnhancedLucidiaChat()
    
    # Example messages to test directives
    test_messages = [
        "Hello, how are you today?",  # No directive
        "Can you initiate a dreaming cycle based on our conversation?",  # Dream cycle directive
        "What do you remember about our last conversation about AI?",  # Memory search directive
        "Reflect on your recent responses and behavior.",  # Self-reflection directive
        "Explore the knowledge graph for connections related to consciousness.",  # Knowledge graph directive
        "Generate some insights about the future of AI in healthcare.",  # Insight directive
        "Analyze the emotions in this statement: I'm really excited about this project!",  # Custom directive
    ]
    
    # Process each test message
    for message in test_messages:
        print(f"\nUser: {message}")
        response = await chat.process_message(message)
        print(response)
        await asyncio.sleep(1)  # Small delay to make output readable


if __name__ == "__main__":
    import re  # Import needed for regex in directive registration
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
