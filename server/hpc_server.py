import asyncio
import websockets
import json
import logging
import torch
import time

# Import the HPCSIGFlowManager from the memory system
from memory.lucidia_memory_system.core.integration.hpc_sig_flow_manager import HPCSIGFlowManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HPCServer:
    """
    HPCServer listens on ws://0.0.0.0:5005
    Expects JSON messages:
      {
        "type": "process",
        "embeddings": [...]
      }
    or
      {
        "type": "stats"
      }
    or
      {
        "type": "get_geometry"
      }
    """
    def __init__(self, host='0.0.0.0', port=5005):
        self.host = host
        self.port = port
        # HPC manager that does hypothetical processing
        self.hpc_sig_manager = HPCSIGFlowManager({
            'embedding_dim': 384
        })
        logger.info("Initialized HPCServer with HPC-SIG manager")

    def get_stats(self):
        # Return HPC state
        return { 
            'type': 'stats',
            **self.hpc_sig_manager.get_stats()
        }

    def get_geometry(self, model_version="latest"):
        # Return hypersphere geometry information
        geometry_info = {
            'type': 'geometry',
            'model_version': model_version,
            'embedding_dim': self.hpc_sig_manager.config['embedding_dim'],
            'hypersphere_radius': 1.0,  # Normalized embeddings typically have unit radius
            'coordinate_system': 'hyperspherical',
            'timestamp': int(time.time() * 1000)  # Current timestamp in milliseconds
        }
        
        # Add additional geometry information if available from the manager
        if hasattr(self.hpc_sig_manager, 'get_geometry_info'):
            additional_info = self.hpc_sig_manager.get_geometry_info(model_version)
            geometry_info.update(additional_info)
            
        return geometry_info

    async def handle_websocket(self, websocket):
        logger.info(f"New connection from {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message: {data}")

                    if data['type'] == 'process':
                        # Perform HPC pipeline
                        embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
                        processed_embedding, significance = await self.hpc_sig_manager.process_embedding(embeddings)

                        # Example response
                        response = {
                            'type': 'processed',
                            'embeddings': processed_embedding.tolist(),
                            'significance': significance
                        }
                        logger.info(f"Sending HPC response: {response}")
                        await websocket.send(json.dumps(response))

                    elif data['type'] == 'stats':
                        stats = self.get_stats()
                        await websocket.send(json.dumps(stats))

                    elif data['type'] == 'get_geometry':
                        # Handle geometry request
                        model_version = data.get('model_version', 'latest')
                        geometry_info = self.get_geometry(model_version)
                        logger.info(f"Sending geometry information for model version {model_version}")
                        await websocket.send(json.dumps(geometry_info))
                    
                    elif data['type'] == 'ping':
                        # Handle ping - respond with pong
                        logger.info("Received ping, responding with pong")
                        await websocket.send(json.dumps({
                            'type': 'pong',
                            'timestamp': int(time.time() * 1000)
                        }))
                    
                    elif data['type'] == 'health_check':
                        # Handle health check
                        logger.info("Received health_check request")
                        await websocket.send(json.dumps({
                            'type': 'health_check_response',
                            'status': 'ok',
                            'timestamp': int(time.time() * 1000)
                        }))
                    
                    elif data['type'] == 'embedding':
                        # Process text to generate embedding
                        text = data.get('text', '')
                        source = data.get('source', 'unknown')
                        
                        if not text.strip():
                            # Empty text
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': 'Empty text provided for embedding generation'
                            }))
                            continue
                        
                        logger.info(f"Processing embedding for text from {source}: {text[:50]}..." if len(text) > 50 else f"Processing embedding for text from {source}: {text}")
                        
                        try:
                            # In a real implementation, this would use a text encoder model
                            # For now, create a mock embedding (random vector, normalized)
                            embedding_dim = self.hpc_sig_manager.config['embedding_dim']
                            mock_embedding = torch.randn(embedding_dim, dtype=torch.float32)
                            mock_embedding = torch.nn.functional.normalize(mock_embedding, p=2, dim=0)
                            
                            await websocket.send(json.dumps({
                                'type': 'embedding_result',
                                'embedding': mock_embedding.tolist(),
                                'source_text': text[:100] + '...' if len(text) > 100 else text
                            }))
                        except Exception as e:
                            logger.error(f"Error generating embedding: {e}")
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': f'Error generating embedding: {str(e)}'
                            }))
                    
                    elif data['type'] == 'stats':
                        # Process embedding for stats
                        embedding = data.get('embedding', [])
                        source = data.get('source', 'unknown')
                        
                        if not embedding or len(embedding) == 0:
                            # Invalid embedding
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': 'Invalid embedding provided for stats calculation'
                            }))
                            continue
                        
                        logger.info(f"Calculating stats for embedding from {source}")
                        
                        try:
                            # In a real implementation, this would calculate actual significance
                            # For now, return a mock significance score
                            mock_significance = 0.85  # High significance value for testing
                            
                            await websocket.send(json.dumps({
                                'type': 'stats_result',
                                'significance': mock_significance,
                                'timestamp': int(time.time() * 1000)
                            }))
                        except Exception as e:
                            logger.error(f"Error calculating stats: {e}")
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': f'Error calculating stats: {str(e)}'
                            }))
                    
                    else:
                        # Unknown message type
                        error_msg = {
                            'type': 'error',
                            'error': f"Unknown message type: {data['type']}"
                        }
                        logger.warning(f"Unknown message type: {data['type']}")
                        await websocket.send(json.dumps(error_msg))

                except Exception as e:
                    err = {'type': 'error', 'error': str(e)}
                    logger.error(f"Error handling HPC message: {str(e)}")
                    await websocket.send(json.dumps(err))

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")

        except Exception as e:
            logger.error(f"Unexpected HPC server error: {str(e)}")

    async def start(self):
        logger.info(f"Starting HPC server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handle_websocket, self.host, self.port):
            logger.info(f"HPC Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # keep running

class HPCClient:
    """Client for the HPCServer to handle hyperdimensional computing operations via WebSocket."""
    
    def __init__(self, url: str = 'ws://localhost:5005', ping_interval: int = 20, ping_timeout: int = 20):
        self.url = url
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.websocket = None
        self.connected = False
        logger.info(f"Initializing HPCClient, will connect to {url}")
    
    async def connect(self):
        """Connect to the HPCServer."""
        try:
            self.websocket = await websockets.connect(
                self.url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout
            )
            self.connected = True
            logger.info(f"Connected to HPCServer at {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to HPCServer: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the HPCServer."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from HPCServer")
    
    async def process_embeddings(self, embeddings):
        """Process embeddings through the HPC system."""
        if not self.connected:
            await self.connect()
        
        request = {
            'type': 'process',
            'embeddings': embeddings if isinstance(embeddings, list) else embeddings.tolist()
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error processing embeddings: {str(e)}")
            return {'type': 'error', 'error': str(e)}
    
    async def get_stats(self):
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

    async def get_geometry(self, model_version="latest"):
        """Get hypersphere geometry information.
        
        Args:
            model_version: Version of the model to get geometry for (default: "latest")
            
        Returns:
            Dict containing geometry information
        """
        if not self.connected:
            await self.connect()

        request = {
            "type": "get_geometry",
            "model_version": model_version
        }
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        return json.loads(response)

if __name__ == '__main__':
    server = HPCServer()
    asyncio.run(server.start())
