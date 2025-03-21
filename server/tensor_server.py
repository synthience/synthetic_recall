"""
LUCID RECALL PROJECT


Tensor Server: Memory & Embedding Operations with Unified Memory System
"""

import asyncio
import websockets
import json
import logging
import torch
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List
# NOTE: HPCQRFlowManager import was removed to prevent circular imports
# If you need flow manager functionality, use lazy imports or dependency injection
from server.memory_system import MemorySystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 5001):
        self.host = host
        self.port = port
        self.setup_gpu()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        if torch.cuda.is_available():
            self.model.to('cuda')
        logger.info(f"Model loaded on {self.model.device}")
        
        # Initialize unified memory system
        self.memory_system = MemorySystem({
            'device': self.device,
            'storage_path': 'memory/stored'
        })
        
        # Lazily initialize HPC manager to avoid circular imports
        self._hpc_manager = None
        
    @property
    def hpc_manager(self):
        """Lazily initialize HPCQRFlowManager to avoid circular imports"""
        if self._hpc_manager is None:
            # Only import when needed
            from memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager import HPCQRFlowManager
            self._hpc_manager = HPCQRFlowManager({
                'embedding_dim': 384,
                'device': self.device
            })
        return self._hpc_manager

    def setup_gpu(self) -> None:
        if torch.cuda.is_available():
            self.device = 'cuda'
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.backends.cudnn.benchmark = True
            logger.info(f"GPU initialized: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            logger.warning("GPU not available, using CPU")

    async def get_embedding(self, text: str) -> torch.Tensor:
        """Generate an embedding for the given text using the SentenceTransformer model.
        
        Args:
            text: The text to encode
            
        Returns:
            A torch.Tensor containing the text embedding
        """
        try:
            # Generate embedding
            embedding = self.model.encode(text)
            if isinstance(embedding, list):
                embedding = torch.tensor(embedding)
            elif isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding)
            
            # Move to the correct device
            if torch.cuda.is_available():
                embedding = embedding.to('cuda')
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def add_memory(self, text: str, embedding: torch.Tensor) -> Dict[str, Any]:
        """
        Add memory with embedding and return metadata, using HPC-QR approach
        for quickrecal_score instead of old significance.
        """
        # Process through HPC
        processed_embedding, quickrecal_score = await self.hpc_manager.process_embedding(embedding)
        
        # Store in unified memory system
        memory = await self.memory_system.add_memory(
            text=text,
            embedding=processed_embedding,
            quickrecal_score=quickrecal_score
        )
        
        logger.info(f"Stored memory {memory['id']} with QuickRecal score {quickrecal_score}")
        return {
            'id': memory['id'],
            'quickrecal_score': quickrecal_score,
            'timestamp': memory['timestamp']
        }

    async def search_memories(self, query_embedding: torch.Tensor, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        # Get processed query embedding
        processed_query, _ = await self.hpc_manager.process_embedding(query_embedding)
        
        # Search using unified memory system
        results = await self.memory_system.search_memories(
            query_embedding=processed_query,
            limit=limit
        )
        
        return results

    async def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        memory_stats = await self.memory_system.get_stats()
        return {
            'gpu_memory': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            'gpu_cached': torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
            'device': self.device,
            'hpc_status': await self.hpc_manager.get_stats(),
            'memory_count': memory_stats['memory_count'],
            'storage_path': memory_stats['storage_path']
        }

    async def handle_websocket(self, websocket):
        """Handle WebSocket connections and messages."""
        try:
            logger.info(f"New connection from {websocket.remote_address}")
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message: {data}")

                    if data['type'] == 'embed':
                        # Generate embedding
                        embeddings = self.model.encode(data['text'])
                        
                        # Store memory
                        metadata = await self.add_memory(
                            data['text'], 
                            torch.tensor(embeddings)
                        )
                        
                        response = {
                            'type': 'embeddings',
                            'embeddings': embeddings.tolist(),
                            **metadata
                        }
                    
                    elif data['type'] == 'embed_only':
                        # Generate embedding without storing memory
                        embeddings = self.model.encode(data['text'])
                        
                        response = {
                            'type': 'embeddings',
                            'embeddings': embeddings.tolist(),
                            'timestamp': time.time()
                        }
                        
                    elif data['type'] == 'search':
                        # Generate query embedding
                        query_embedding = self.model.encode(data['text'])
                        
                        # Search memories
                        results = await self.search_memories(
                            torch.tensor(query_embedding),
                            limit=data.get('limit', 5)
                        )
                        
                        response = {
                            'type': 'search_results',
                            'query': data['text'],
                            'results': [{
                                'id': r['memory']['id'],
                                'text': r['memory']['text'],
                                'similarity': r['similarity'],
                                'quickrecal_score': r['memory'].get('quickrecal_score', 0.0)
                            } for r in results]
                        }
                        
                    elif data['type'] == 'stats' or data['type'] == 'get_stats':
                        response = await self.get_stats()
                        
                    else:
                        response = {
                            'type': 'error',
                            'error': f"Unknown message type: {data['type']}"
                        }
                        logger.warning(f"Unknown message type received: {data['type']}")

                    await websocket.send(json.dumps(response))
                    logger.info(f"Sent response: {response['type']}")

                except Exception as e:
                    error_msg = {
                        'type': 'error',
                        'error': str(e)
                    }
                    logger.error(f"Error processing message: {str(e)}")
                    await websocket.send(json.dumps(error_msg))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

    async def start(self):
        """Start the WebSocket server."""
        async with websockets.serve(self.handle_websocket, self.host, self.port):
            logger.info(f"Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Keep server running

class TensorClient:
    """Client for the TensorServer to handle embedding and memory operations via WebSocket."""
    
    def __init__(self, url: str = 'ws://localhost:5001', ping_interval: int = 20, ping_timeout: int = 20):
        self.url = url
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.websocket = None
        self.connected = False
        logger.info(f"Initializing TensorClient, will connect to {url}")
    
    async def connect(self):
        """Connect to the TensorServer."""
        try:
            self.websocket = await websockets.connect(
                self.url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout
            )
            self.connected = True
            logger.info(f"Connected to TensorServer at {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TensorServer: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the TensorServer."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from TensorServer")
    
    async def get_embedding(self, text: str) -> dict:
        """Get embedding for a text, storing as memory on the server."""
        if not self.connected:
            await self.connect()
        
        request = {
            'type': 'embed',
            'text': text
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return {'type': 'error', 'error': str(e)}
    
    async def search_memories(self, text: str, limit: int = 5) -> dict:
        """Search for memories similar to the given text."""
        if not self.connected:
            await self.connect()
        
        request = {
            'type': 'search',
            'text': text,
            'limit': limit
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return {'type': 'error', 'error': str(e)}
    
    async def get_stats(self) -> dict:
        """Get server statistics."""
        if not self.connected:
            await self.connect()
        
        request = {
            'type': 'stats'
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {'type': 'error', 'error': str(e)}

if __name__ == '__main__':
    server = TensorServer()
    asyncio.run(server.start())
