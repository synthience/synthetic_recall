import os
import asyncio
import signal
import json
import sys
from typing import Dict, Any, List, Optional

# Import our modules
from mcp import Server, StdioServerTransport, LIST_TOOLS_REQUEST_SCHEMA, CALL_TOOL_REQUEST_SCHEMA, ErrorCode, McpError
from types import NemoAPIConfig, MemoryManagerConfig
from memory_manager import MemoryManager
from nemo_api import NemoAPI
from logger import logger

# Environment variables validation
required_env_vars = {
    "TITAN_MODEL_PATH": os.environ.get("TITAN_MODEL_PATH", ""),
    "TITAN_MAX_MEMORY_MB": os.environ.get("TITAN_MAX_MEMORY_MB", ""),
    "TITAN_PERSISTENCE_PATH": os.environ.get("TITAN_PERSISTENCE_PATH", ""),
    "TITAN_EMBEDDING_DIM": os.environ.get("TITAN_EMBEDDING_DIM", ""),
}

# Validate all required environment variables are present
for key, value in required_env_vars.items():
    if not value:
        raise Exception(f"Missing required environment variable: {key}")

logger.log("TitanMemory", "Starting with configuration", required_env_vars)

# Initialize components
nemo_api = NemoAPI(NemoAPIConfig(
    model_path=required_env_vars["TITAN_MODEL_PATH"],
    embedding_dimension=int(required_env_vars["TITAN_EMBEDDING_DIM"]),
    cutlass_config={
        "tensorCoreEnabled": True,
        "computeCapability": "8.9",
        "memoryLimit": int(required_env_vars["TITAN_MAX_MEMORY_MB"]) * 1024 * 1024,
    }
))

memory_manager = MemoryManager(MemoryManagerConfig(
    persistence_path=required_env_vars["TITAN_PERSISTENCE_PATH"],
    max_memory_mb=int(required_env_vars["TITAN_MAX_MEMORY_MB"]),
    embedding_dimension=int(required_env_vars["TITAN_EMBEDDING_DIM"])
))

logger.log("TitanMemory", "Components initialized successfully")

# Create MCP server
server = Server(
    {
        "name": "titan-memory",
        "version": "0.1.0",
    },
    {
        "capabilities": {
            "tools": {},
        },
    }
)

# Handle tool requests
async def handle_list_tools(request):
    return {
        "tools": [
            {
                "name": "store_memory",
                "description": "Store a new memory in the Titan system",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Memory content to store",
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context for the memory",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "query_memory",
                "description": "Query memories from the Titan system",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to search memories",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of results",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]
    }

async def handle_call_tool(request):
    logger.log("TitanMemory", "Received tool request", {
        "tool": request["params"]["name"],
        "args": request["params"]["arguments"],
    })
    
    try:
        if request["params"]["name"] == "store_memory":
            content = request["params"]["arguments"]["content"]
            context = request["params"]["arguments"].get("context")
            
            logger.log("TitanMemory", "Processing store_memory request", {
                "content": content,
                "context": context,
                "persistencePath": required_env_vars["TITAN_PERSISTENCE_PATH"],
                "cwd": os.getcwd(),
            })
            
            # Generate embedding
            embedding = await nemo_api.generate_embedding(content)
            logger.log("TitanMemory", "Generated embedding", {
                "contentLength": len(content),
                "embeddingLength": len(embedding),
            })
            
            # Store memory
            memory = await memory_manager.store_memory({
                "content": content,
                "embedding": embedding,
                "context": context
            })
            logger.log("TitanMemory", "Memory stored successfully", {"id": memory.id})
            
            # Convert memory object to dictionary for JSON serialization
            memory_dict = {
                "id": memory.id,
                "content": memory.content,
                "embedding": memory.embedding,
                "timestamp": memory.timestamp,
                "context": memory.context
            }
            
            return {
                "content": [{"type": "text", "text": json.dumps(memory_dict, indent=2)}]
            }
            
        elif request["params"]["name"] == "query_memory":
            query = request["params"]["arguments"]["query"]
            limit = request["params"]["arguments"].get("limit", 5)
            logger.log("TitanMemory", "Processing query_memory request", {"query": query, "limit": limit})
            
            # Generate query embedding
            query_embedding = await nemo_api.generate_embedding(query)
            logger.log("TitanMemory", "Generated query embedding", {
                "queryLength": len(query),
                "embeddingLength": len(query_embedding),
            })
            
            # Query memories
            results = await memory_manager.query_memories(query_embedding, limit)
            logger.log("TitanMemory", "Query completed", {"resultCount": len(results)})
            
            # Convert memory objects to dictionaries for JSON serialization
            results_dicts = []
            for memory in results:
                results_dicts.append({
                    "id": memory.id,
                    "content": memory.content,
                    "embedding": memory.embedding,
                    "timestamp": memory.timestamp,
                    "context": memory.context
                })
            
            return {
                "content": [{"type": "text", "text": json.dumps(results_dicts, indent=2)}]
            }
        
        else:
            logger.log("TitanMemory", "Unknown tool requested", {"tool": request["params"]["name"]})
            raise McpError(ErrorCode.METHOD_NOT_FOUND, f"Unknown tool: {request['params']['name']}")
    
    except Exception as error:
        logger.log("TitanMemory", "Tool execution failed", error)
        raise error

server.set_request_handler(LIST_TOOLS_REQUEST_SCHEMA, handle_list_tools)
server.set_request_handler(CALL_TOOL_REQUEST_SCHEMA, handle_call_tool)

# Error handling
def handle_error(error):
    logger.log("TitanMemory", "Server error", error)

server.onerror = handle_error

# Initialize components and start server
async def main():
    try:
        # Initialize NemoAPI
        logger.log("TitanMemory", "Initializing NemoAPI")
        await nemo_api.initialize()
        logger.log("TitanMemory", "NemoAPI initialization complete")
        
        # Start server
        transport = StdioServerTransport()
        await server.connect(transport)
        logger.log("TitanMemory", "Server running on stdio")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except Exception as error:
        logger.log("TitanMemory", "Failed to start server", error)
        sys.exit(1)

# Cleanup on exit
async def cleanup():
    logger.log("TitanMemory", "Received SIGINT, cleaning up")
    await memory_manager.cleanup()
    await logger.cleanup()
    await server.close()
    sys.exit(0)

if __name__ == "__main__":
    # Set up asyncio event loop
    loop = asyncio.get_event_loop()
    
    # Register signal handler for graceful shutdown
    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(cleanup()))
    
    # Run the main function
    loop.run_until_complete(main())
