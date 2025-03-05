import asyncio
import websockets
import json
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, tensor_server_url: str = "ws://localhost:5001"):
        """Initialize conversation manager with tensor server connection"""
        self.tensor_server_url = tensor_server_url
        self.conversation_history: List[Dict] = []
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.memory_context: List[str] = []
        
    async def connect(self):
        """Connect to the tensor server for memory operations"""
        try:
            self.websocket = await websockets.connect(
                self.tensor_server_url,
                ping_interval=20,
                ping_timeout=60,
                close_timeout=10
            )
            logger.info("Connected to tensor server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to tensor server: {str(e)}")
            return False
            
    async def close(self):
        """Close the websocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            
    async def add_to_memory(self, text: str, role: str):
        """Add a conversation turn to memory"""
        self.conversation_history.append({
            "role": role,
            "content": text
        })
        
        if self.websocket:
            try:
                # Match tensor server's expected message type
                message = {
                    "type": "embed",  
                    "text": text,
                    "metadata": {"role": role}
                }
                await self.websocket.send(json.dumps(message))
                response = await self.websocket.recv()
                data = json.loads(response)
                if data.get('type') != 'embeddings':
                    logger.error(f"Failed to add memory: Unexpected response type {data.get('type')}")
            except Exception as e:
                logger.error(f"Error adding to memory: {str(e)}")
                
    async def search_relevant_context(self, query: str, k: int = 3):
        """Search for relevant past context given a query"""
        if self.websocket:
            try:
                # Match tensor server's expected message type
                message = {
                    "type": "search",  
                    "text": query,
                    "limit": k
                }
                await self.websocket.send(json.dumps(message))
                response = await self.websocket.recv()
                data = json.loads(response)
                if data.get('type') == 'search_results':
                    self.memory_context = [r['text'] for r in data.get('results', [])]
                else:
                    logger.error(f"Failed to search context: Unexpected response type {data.get('type')}")
            except Exception as e:
                logger.error(f"Error searching context: {str(e)}")
                
    def get_context_for_llm(self, current_query: str) -> str:
        """Format conversation context for the LLM"""
        context = []
        
        # Add memory context if available
        if self.memory_context:
            context.append("Relevant past context:")
            context.extend(self.memory_context)
            context.append("")
            
        # Add recent conversation history
        context.append("Recent conversation:")
        for turn in self.conversation_history[-5:]:  # Last 5 turns
            role_prefix = "User:" if turn["role"] == "user" else "Assistant:"
            context.append(f"{role_prefix} {turn['content']}")
            
        # Add current query
        context.append(f"User: {current_query}")
        context.append("Assistant:")
        
        return "\n".join(context)
