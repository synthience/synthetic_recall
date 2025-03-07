import asyncio
import websockets
import json
import logging
import torch

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

if __name__ == '__main__':
    server = HPCServer()
    asyncio.run(server.start())
