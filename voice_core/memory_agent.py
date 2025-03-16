"""Autonomous Memory Agent for Lucidia that runs alongside the Voice Agent."""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from memory_core.memory_broker import get_memory_broker, MemoryBroker
from memory_core.enhanced_memory_client import EnhancedMemoryClient
from memory.lucidia_memory_system.core.integration import MemoryIntegration
from server.tensor_server import TensorClient
from server.hpc_server import HPCClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoryAgent")

MEMORY_INTEGRATION_AVAILABLE = True
SERVERS_AVAILABLE = True

class LucidiaMemoryAgent:
    """Autonomous agent that manages all memory operations for Lucidia."""
    
    def __init__(self, 
                 tensor_server_url: str = "ws://localhost:5001",
                 hpc_server_url: str = "ws://localhost:5005",
                 config_path: Optional[str] = None):
        """Initialize the memory agent.
        
        Args:
            tensor_server_url: URL for the tensor server
            hpc_server_url: URL for the HPC server
            config_path: Path to configuration file
        """
        self.tensor_server_url = tensor_server_url
        self.hpc_server_url = hpc_server_url
        self.config_path = config_path
        self.running = False
        self.broker = None
        self.memory_client = None
        self.memory_integration = None
        self.tasks = {}
        
        # Statistics for monitoring
        self.stats = {
            "requests_processed": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }
        
        # Memory operations that this agent handles
        self.supported_operations = [
            "classify_query",
            "retrieve_memories",
            "retrieve_information",
            "store_and_retrieve",
            "store_emotional_context",
            "detect_and_store_personal_details",
            "store_transcript",
            "generate_embedding",
            "get_rag_context"
        ]
    
    async def start(self):
        """Start the memory agent."""
        if self.running:
            return
            
        logger.info("Starting Lucidia Memory Agent...")
        
        # Initialize the memory broker
        self.broker = await get_memory_broker()
        
        # Initialize memory systems
        await self._init_memory_systems()
        
        # Register operation handlers with the broker
        for operation in self.supported_operations:
            method_name = f"handle_{operation}"
            if hasattr(self, method_name):
                handler = getattr(self, method_name)
                await self.broker.register_callback(operation, handler)
        
        # Start the main processing loop
        self.running = True
        self.tasks["main_loop"] = asyncio.create_task(self._main_loop())
        
        logger.info("Lucidia Memory Agent started successfully")
    
    async def stop(self):
        """Stop the memory agent."""
        if not self.running:
            return
            
        logger.info("Stopping Lucidia Memory Agent...")
        
        # Cancel all tasks
        for task_name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Unregister callbacks
        for operation in self.supported_operations:
            await self.broker.unregister_callback(operation)
        
        # Close memory client connections
        if self.memory_client:
            await self.memory_client.close()
        
        self.running = False
        logger.info("Lucidia Memory Agent stopped")
    
    async def _init_memory_systems(self):
        """Initialize the memory systems."""
        try:
            # Initialize hierarchical memory system
            memory_config = {}  # Load from config if needed
            if MEMORY_INTEGRATION_AVAILABLE:
                self.memory_integration = MemoryIntegration(memory_config)
                logger.info("Using hierarchical memory integration")
            else:
                self.memory_integration = None
                logger.info("Hierarchical memory integration not available")
            
            # Initialize enhanced memory client
            if SERVERS_AVAILABLE:
                try:
                    self.memory_client = EnhancedMemoryClient(
                        tensor_server_url=self.tensor_server_url,
                        hpc_server_url=self.hpc_server_url,
                        memory_integration=self.memory_integration,
                        ping_interval=30.0,  # From the memory fix
                        max_retries=3,
                        retry_delay=1.5
                    )
                    logger.info("Memory client initialized with server connections")
                except Exception as e:
                    logger.error(f"Error initializing memory client with servers: {e}")
                    self.memory_client = EnhancedMemoryClient(
                        tensor_server_url=self.tensor_server_url,
                        hpc_server_url=self.hpc_server_url,
                        ping_interval=30.0  # From the memory fix
                    )
            else:
                # Create a minimal memory client that implements the required methods
                # but does not depend on external servers
                logger.info("Creating minimal memory client without server dependencies")
                self.memory_client = EnhancedMemoryClient(
                    tensor_server_url=self.tensor_server_url,
                    hpc_server_url=self.hpc_server_url,
                    ping_interval=30.0  # From the memory fix
                )
            
            logger.info("Memory systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing memory systems: {e}", exc_info=True)
            # Fall back to basic memory client without any dependencies
            logger.info("Falling back to basic memory client implementation")
            self.memory_client = EnhancedMemoryClient(
                tensor_server_url=self.tensor_server_url,
                hpc_server_url=self.hpc_server_url,
                ping_interval=30.0  # From the memory fix
            )
    
    async def _main_loop(self):
        """Main processing loop for the memory agent."""
        while self.running:
            try:
                # Periodic tasks can go here
                # e.g., memory optimization, cleaning, etc.
                
                # Health check for memory systems
                if self.memory_client:
                    health_check = await self._check_memory_health()
                    if not health_check["status"] == "ok":
                        logger.warning(f"Memory health check failed: {health_check['message']}")
                
                # Sleep to avoid tight loop
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Backoff on error
    
    async def _check_memory_health(self):
        """Check the health of memory systems."""
        try:
            # Basic connectivity check
            if hasattr(self.memory_client, 'is_connected'):
                connected = await self.memory_client.is_connected()
                if not connected:
                    return {"status": "error", "message": "Memory client disconnected"}
            
            # More health checks can be added here
            
            return {"status": "ok", "message": "Memory systems healthy"}
        
        except Exception as e:
            return {"status": "error", "message": f"Health check error: {str(e)}"}
    
    # Request handlers
    
    async def handle_classify_query(self, data):
        """Handle a classify_query request."""
        start_time = asyncio.get_event_loop().time()
        try:
            query = data.get("query", "")
            if not query:
                return {"success": False, "error": "No query provided"}
            
            query_type = await self.memory_client.classify_query(query)
            
            # Update stats
            self.stats["requests_processed"] += 1
            self.stats["successful_operations"] += 1
            
            response_time = asyncio.get_event_loop().time() - start_time
            self.stats["total_response_time"] += response_time
            self.stats["avg_response_time"] = (
                self.stats["total_response_time"] / self.stats["successful_operations"]
            )
            
            return {
                "success": True,
                "query_type": query_type,
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error in classify_query: {e}", exc_info=True)
            
            # Update error stats
            self.stats["requests_processed"] += 1
            self.stats["failed_operations"] += 1
            
            return {"success": False, "error": str(e)}
    
    async def handle_retrieve_memories(self, data):
        """Handle a retrieve_memories request."""
        start_time = asyncio.get_event_loop().time()
        try:
            query = data.get("query", "")
            limit = data.get("limit", 5)
            min_quickrecal_score = data.get("min_quickrecal_score", 0.3)
            
            if not query:
                return {"success": False, "error": "No query provided"}
            
            memories = await self.memory_client.retrieve_memories(
                query=query,
                limit=limit,
                min_quickrecal_score=min_quickrecal_score
            )
            
            # Update stats
            self.stats["requests_processed"] += 1
            self.stats["successful_operations"] += 1
            
            response_time = asyncio.get_event_loop().time() - start_time
            self.stats["total_response_time"] += response_time
            self.stats["avg_response_time"] = (
                self.stats["total_response_time"] / self.stats["successful_operations"]
            )
            
            return {
                "success": True,
                "memories": memories,
                "count": len(memories),
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_memories: {e}", exc_info=True)
            
            # Update error stats
            self.stats["requests_processed"] += 1
            self.stats["failed_operations"] += 1
            
            return {"success": False, "error": str(e)}
    
    async def handle_retrieve_information(self, data):
        """Handle a retrieve_information request."""
        start_time = asyncio.get_event_loop().time()
        try:
            query = data.get("query", "")
            context_type = data.get("context_type", "general")
            
            if not query:
                return {"success": False, "error": "No query provided"}
            
            information = await self.memory_client.retrieve_information(
                query=query,
                context_type=context_type
            )
            
            # Update stats
            self.stats["requests_processed"] += 1
            self.stats["successful_operations"] += 1
            
            response_time = asyncio.get_event_loop().time() - start_time
            self.stats["total_response_time"] += response_time
            self.stats["avg_response_time"] = (
                self.stats["total_response_time"] / self.stats["successful_operations"]
            )
            
            return {
                "success": True,
                "information": information,
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_information: {e}", exc_info=True)
            
            # Update error stats
            self.stats["requests_processed"] += 1
            self.stats["failed_operations"] += 1
            
            return {"success": False, "error": str(e)}
        
    async def handle_store_and_retrieve(self, data):
        """Handle a store_and_retrieve request."""
        start_time = asyncio.get_event_loop().time()
        try:
            content = data.get("content", "")
            query = data.get("query", "")
            memory_type = data.get("memory_type", "conversation")
            
            if not content:
                return {"success": False, "error": "No content provided"}
            
            result = await self.memory_client.store_and_retrieve(
                content=content,
                query=query,
                memory_type=memory_type
            )
            
            # Update stats
            self.stats["requests_processed"] += 1
            self.stats["successful_operations"] += 1
            
            response_time = asyncio.get_event_loop().time() - start_time
            self.stats["total_response_time"] += response_time
            self.stats["avg_response_time"] = (
                self.stats["total_response_time"] / self.stats["successful_operations"]
            )
            
            return {
                "success": True,
                "result": result,
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error in store_and_retrieve: {e}", exc_info=True)
            
            # Update error stats
            self.stats["requests_processed"] += 1
            self.stats["failed_operations"] += 1
            
            return {"success": False, "error": str(e)}
        
    async def handle_store_emotional_context(self, data):
        """Handle a store_emotional_context request."""
        start_time = asyncio.get_event_loop().time()
        try:
            user_input = data.get("user_input", "")
            emotions = data.get("emotions", {})
            timestamp = data.get("timestamp", datetime.now().isoformat())
            
            if not user_input and not emotions:
                return {"success": False, "error": "No input or emotions provided"}
            
            result = await self.memory_client.store_emotional_context(
                user_input=user_input,
                emotions=emotions,
                timestamp=timestamp
            )
            
            # Update stats
            self.stats["requests_processed"] += 1
            self.stats["successful_operations"] += 1
            
            response_time = asyncio.get_event_loop().time() - start_time
            self.stats["total_response_time"] += response_time
            self.stats["avg_response_time"] = (
                self.stats["total_response_time"] / self.stats["successful_operations"]
            )
            
            return {
                "success": True,
                "result": result,
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error in store_emotional_context: {e}", exc_info=True)
            
            # Update error stats
            self.stats["requests_processed"] += 1
            self.stats["failed_operations"] += 1
            
            return {"success": False, "error": str(e)}
        
    async def handle_detect_and_store_personal_details(self, data):
        """Handle a detect_and_store_personal_details request."""
        start_time = asyncio.get_event_loop().time()
        try:
            text = data.get("text", "")
            
            if not text:
                return {"success": False, "error": "No text provided"}
            
            details = await self.memory_client.detect_and_store_personal_details(text=text)
            
            # Update stats
            self.stats["requests_processed"] += 1
            self.stats["successful_operations"] += 1
            
            response_time = asyncio.get_event_loop().time() - start_time
            self.stats["total_response_time"] += response_time
            self.stats["avg_response_time"] = (
                self.stats["total_response_time"] / self.stats["successful_operations"]
            )
            
            return {
                "success": True,
                "details": details,
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error in detect_and_store_personal_details: {e}", exc_info=True)
            
            # Update error stats
            self.stats["requests_processed"] += 1
            self.stats["failed_operations"] += 1
            
            return {"success": False, "error": str(e)}
        
    async def handle_store_transcript(self, data):
        """Handle a store_transcript request."""
        start_time = asyncio.get_event_loop().time()
        try:
            transcript = data.get("transcript", "")
            metadata = data.get("metadata", {})
            
            if not transcript:
                return {"success": False, "error": "No transcript provided"}
            
            result = await self.memory_client.store_transcript(
                transcript=transcript,
                metadata=metadata
            )
            
            # Update stats
            self.stats["requests_processed"] += 1
            self.stats["successful_operations"] += 1
            
            response_time = asyncio.get_event_loop().time() - start_time
            self.stats["total_response_time"] += response_time
            self.stats["avg_response_time"] = (
                self.stats["total_response_time"] / self.stats["successful_operations"]
            )
            
            return {
                "success": True,
                "result": result,
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error in store_transcript: {e}", exc_info=True)
            
            # Update error stats
            self.stats["requests_processed"] += 1
            self.stats["failed_operations"] += 1
            
            return {"success": False, "error": str(e)}
        
    async def handle_generate_embedding(self, data):
        """Handle a generate_embedding request."""
        start_time = asyncio.get_event_loop().time()
        try:
            text = data.get("text", "")
            
            if not text:
                return {"success": False, "error": "No text provided"}
            
            embedding = await self.memory_client.generate_embedding(text=text)
            
            # Update stats
            self.stats["requests_processed"] += 1
            self.stats["successful_operations"] += 1
            
            response_time = asyncio.get_event_loop().time() - start_time
            self.stats["total_response_time"] += response_time
            self.stats["avg_response_time"] = (
                self.stats["total_response_time"] / self.stats["successful_operations"]
            )
            
            return {
                "success": True,
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error in generate_embedding: {e}", exc_info=True)
            
            # Update error stats
            self.stats["requests_processed"] += 1
            self.stats["failed_operations"] += 1
            
            return {"success": False, "error": str(e)}
        
    async def handle_get_rag_context(self, data):
        """Handle a get_rag_context request."""
        start_time = asyncio.get_event_loop().time()
        try:
            query = data.get("query", "")
            max_tokens = data.get("max_tokens", 1000)
            min_quickrecal_score = data.get("min_quickrecal_score", 0.3)
            
            if not query:
                return {"success": False, "error": "No query provided"}
            
            context = await self.memory_client.get_rag_context(
                query=query,
                max_tokens=max_tokens,
                min_quickrecal_score=min_quickrecal_score
            )
            
            # Update stats
            self.stats["requests_processed"] += 1
            self.stats["successful_operations"] += 1
            
            response_time = asyncio.get_event_loop().time() - start_time
            self.stats["total_response_time"] += response_time
            self.stats["avg_response_time"] = (
                self.stats["total_response_time"] / self.stats["successful_operations"]
            )
            
            return {
                "success": True,
                "context": context,
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error in get_rag_context: {e}", exc_info=True)
            
            # Update error stats
            self.stats["requests_processed"] += 1
            self.stats["failed_operations"] += 1
            
            return {"success": False, "error": str(e)}


async def main():
    """Main entry point for the memory agent."""
    memory_agent = LucidiaMemoryAgent()
    
    try:
        await memory_agent.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Memory agent interrupted")
    finally:
        await memory_agent.stop()


if __name__ == "__main__":
    asyncio.run(main())