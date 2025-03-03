"""
LUCID RECALL PROJECT
Agent: Aurora 1.1
Date: 2/13/25
Time: 1:19 AM EST

Tensor Server: Memory & Embedding Operations with Unified Memory System
"""

import asyncio
import websockets
import json
import logging
import torch
import time
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List
from server.hpc_sig_flow_manager import HPCSIGFlowManager
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
        
        # Initialize HPC manager
        self.hpc_manager = HPCSIGFlowManager({
            'embedding_dim': 384,
            'device': self.device
        })
        
        # Initialize unified memory system
        self.memory_system = MemorySystem({
            'device': self.device,
            'embedding_dim': 384
        })
        logger.info("Initialized unified memory system")

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

    async def add_memory(self, text: str, embedding: torch.Tensor) -> Dict[str, Any]:
        """Add memory with embedding and return metadata."""
        # Process through HPC
        processed_embedding, significance = await self.hpc_manager.process_embedding(embedding)
        
        # Store in unified memory system
        memory = await self.memory_system.add_memory(
            text=text,
            embedding=processed_embedding,
            significance=significance
        )
        
        logger.info(f"Stored memory {memory['id']} with significance {significance}")
        return {
            'id': memory['id'],
            'significance': significance,
            'timestamp': memory['timestamp']
        }

    async def search_memories(self, query_embedding: torch.Tensor, limit: int = 5) -> List[Dict]:
        """Search for similar memories."""
        # Get processed query embedding
        processed_query, _ = await self.hpc_manager.process_embedding(query_embedding)
        
        # Search using unified memory system
        results = await self.memory_system.search_memories(
            query_embedding=processed_query,
            limit=limit
        )
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        memory_stats = self.memory_system.get_stats()
        stats = {
            'type': 'stats',
            'gpu_memory': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            'gpu_cached': torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
            'device': self.device,
            'hpc_status': self.hpc_manager.get_stats(),
            'memory_count': memory_stats['memory_count'],
            'storage_path': memory_stats['storage_path']
        }
        logger.info(f"Stats requested: {stats}")
        return stats

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
                            'results': [{
                                'id': r['memory']['id'],
                                'text': r['memory']['text'],
                                'similarity': r['similarity'],
                                'significance': r['memory']['significance']
                            } for r in results]
                        }
                        
                    elif data['type'] == 'stats':
                        response = self.get_stats()
                        
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
            await asyncio.Future()

if __name__ == '__main__':
    server = TensorServer()
    asyncio.run(server.start())
