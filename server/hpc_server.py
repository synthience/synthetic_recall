import asyncio
import websockets
import json
import logging
import torch
import time
import numpy as np

from memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager import HPCQRFlowManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HPCServer:
    """
    HPCServer listens on ws://0.0.0.0:5005
    Expects JSON messages of various types (process, stats, etc.)
    """
    def __init__(self, host='0.0.0.0', port=5005, config_path=None):
        self.host = host
        self.port = port
        
        # Load config for HPCQRFlowManager
        import os
        import yaml
        
        # Default configuration
        hpc_config = {'embedding_dim': 384}
        
        # Try to load from quickrecal_config.yaml if no config path specified
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'quickrecal_config.yaml')
        
        # Load configuration if available
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        logger.info(f"Loaded QuickRecal config from {config_path}")
                        
                        # Extract batch_size for server if specified
                        if 'batch_size' in yaml_config:
                            hpc_config['batch_size'] = yaml_config['batch_size']
                            
                        # Extract embedding dimension if specified
                        if 'embedding_dim' in yaml_config:
                            hpc_config['embedding_dim'] = yaml_config['embedding_dim']
            except Exception as e:
                logger.warning(f"Error loading QuickRecal config: {e}")
                logger.warning("Using default configuration")
        
        # Initialize the HPCQRFlowManager with loaded config
        self.hpc_qr_manager = HPCQRFlowManager(hpc_config)
        logger.info(f"Initialized HPCServer with HPC-QR manager using config: {hpc_config}")

    def get_stats(self):
        return { 
            'type': 'stats',
            **self.hpc_qr_manager.get_stats()
        }

    def get_geometry(self, model_version="latest"):
        geometry_info = {
            'type': 'geometry',
            'model_version': model_version,
            'embedding_dim': self.hpc_qr_manager.config['embedding_dim'],
            'hypersphere_radius': 1.0,
            'coordinate_system': 'hyperspherical',
            'timestamp': int(time.time() * 1000)
        }
        return geometry_info

    async def log_score_distribution(self):
        """Log histogram of QuickRecal scores for monitoring performance"""
        try:
            # Use calculator's logging method directly
            self.hpc_qr_manager.qr_calculator.log_score_distribution()
            return {'type': 'success', 'message': 'Score distribution logged'}
        except Exception as e:
            logger.error(f"Error logging score distribution: {e}")
            return {'type': 'error', 'message': f'Error logging score distribution: {e}'}

    async def handle_websocket(self, websocket):
        logger.info(f"New connection from {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message: {data}")

                    if data['type'] == 'process':
                        embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
                        
                        # Check for enhanced processing parameters
                        include_qr_score = data.get('include_qr_score', False)
                        min_qr_threshold = data.get('min_qr_threshold', 0.0)
                        fusion_weights = data.get('fusion_weights', None)
                        
                        # Process the embedding with HPC-QR
                        processed_embedding, quickrecal_score = await self.hpc_qr_manager.process_embedding(embeddings)
                        
                        # Prepare response
                        response = {
                            'type': 'processed',
                            'processed': processed_embedding.tolist(),
                        }
                        
                        # Include QR score if requested
                        if include_qr_score:
                            response['quickrecal_score'] = float(quickrecal_score)
                            # Apply soft thresholding if requested
                            if min_qr_threshold > 0:
                                # Sigmoid-based soft threshold
                                threshold_factor = 1.0 / (1.0 + np.exp(-12 * (quickrecal_score - min_qr_threshold)))
                                response['threshold_factor'] = float(threshold_factor)
                        
                        # Send back fusion weights used, if provided
                        if fusion_weights:
                            response['fusion_weights_used'] = fusion_weights
                            
                        await websocket.send(json.dumps(response))

                    elif data['type'] == 'stats':
                        stats = self.get_stats()
                        await websocket.send(json.dumps(stats))

                    elif data['type'] == 'get_geometry':
                        model_version = data.get('model_version', 'latest')
                        geometry_info = self.get_geometry(model_version)
                        logger.info(f"Sending geometry information for model version {model_version}")
                        await websocket.send(json.dumps(geometry_info))
                    
                    elif data['type'] == 'ping':
                        logger.info("Received ping, responding with pong")
                        await websocket.send(json.dumps({
                            'type': 'pong',
                            'timestamp': int(time.time() * 1000)
                        }))
                    
                    elif data['type'] == 'health_check':
                        logger.info("Received health_check request")
                        await websocket.send(json.dumps({
                            'type': 'health_check_response',
                            'status': 'ok',
                            'timestamp': int(time.time() * 1000)
                        }))
                    
                    elif data['type'] == 'embedding':
                        text = data.get('text', '')
                        source = data.get('source', 'unknown')
                        
                        if not text.strip():
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': 'Empty text provided for embedding generation'
                            }))
                            continue
                        
                        logger.info(f"Processing embedding for text from {source}: {text[:50]}..." 
                                    if len(text) > 50 else f"Processing embedding for text from {source}: {text}")
                        
                        try:
                            # Mock embedding
                            embedding_dim = self.hpc_qr_manager.config['embedding_dim']
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
                    
                    elif data['type'] == 'process_embedding':
                        # Process embedding directly
                        try:
                            embedding = torch.tensor(data['embedding'], dtype=torch.float32)
                            result = self.hpc_qr_manager.process_embedding(embedding)
                            
                            await websocket.send(json.dumps({
                                'type': 'process_result',
                                'quickrecal_score': result['quickrecal_score'],
                                'geometry': result['geometry']
                            }))
                        except Exception as e:
                            logger.error(f"Error processing embedding: {e}")
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': f'Error processing embedding: {str(e)}'
                            }))
                    
                    elif data['type'] == 'emotional_analysis':
                        # Process emotional analysis request
                        try:
                            text = data.get('text', '')
                            if not text:
                                raise ValueError("No text provided for emotional analysis")
                                
                            # Perform emotional analysis
                            # This is a simplified implementation - in production you might use
                            # a more sophisticated emotion detection model
                            emotions = {
                                "joy": 0.0,
                                "sadness": 0.0,
                                "anger": 0.0,
                                "fear": 0.0,
                                "surprise": 0.0,
                                "disgust": 0.0,
                                "neutral": 0.0
                            }
                            
                            # Simple rule-based analysis as placeholder
                            # In production, replace with actual emotion detection model
                            if any(word in text.lower() for word in ['happy', 'glad', 'excited', 'wonderful']):
                                emotions['joy'] = 0.8
                            elif any(word in text.lower() for word in ['sad', 'unhappy', 'depressed', 'disappointed']):
                                emotions['sadness'] = 0.8
                            elif any(word in text.lower() for word in ['angry', 'mad', 'furious', 'rage']):
                                emotions['anger'] = 0.8
                            elif any(word in text.lower() for word in ['afraid', 'scared', 'worried', 'terrified']):
                                emotions['fear'] = 0.8
                            elif any(word in text.lower() for word in ['surprised', 'shocked', 'astonished']):
                                emotions['surprise'] = 0.8
                            elif any(word in text.lower() for word in ['disgusted', 'gross', 'revolting']):
                                emotions['disgust'] = 0.8
                            else:
                                emotions['neutral'] = 0.8
                                
                            # Determine dominant emotion
                            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                            
                            # Send response
                            await websocket.send(json.dumps({
                                'type': 'emotional_analysis_result',
                                'emotions': emotions,
                                'dominant_emotion': dominant_emotion,
                                'confidence': emotions[dominant_emotion],
                                'text': text[:100] + '...' if len(text) > 100 else text
                            }))
                        except Exception as e:
                            logger.error(f"Error performing emotional analysis: {e}")
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': f'Error performing emotional analysis: {str(e)}'
                            }))
                    
                    elif data['type'] == 'generate_embedding':
                        # Generate embedding
                        try:
                            text = data.get('text', '')
                            if not text:
                                raise ValueError("No text provided for embedding generation")
                            
                            # Generate embedding
                            # This is a simplified implementation - in production you might use
                            # a more sophisticated embedding generation model
                            embedding_dim = self.hpc_qr_manager.config['embedding_dim']
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
                    
                    elif data['type'] == 'log_score_distribution':
                        result = await self.log_score_distribution()
                        await websocket.send(json.dumps(result))
                    
                    else:
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
    """Client for the HPCServer to handle HPC-QR operations via WebSocket."""
    
    def __init__(self, url: str = 'ws://localhost:5005', ping_interval: int = 20, ping_timeout: int = 20):
        self.url = url
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.websocket = None
        self.connected = False
        self.search_fusion_weights = {
            'similarity_weight': 0.6,
            'quickrecal_weight': 0.4,
            'use_logarithmic_fusion': True
        }
        logger.info(f"Initializing HPCClient, will connect to {url}")
        
    async def load_config(self, config_path=None):
        """Load QuickRecal configuration from the YAML file"""
        import os
        import yaml
        
        # Try to load from quickrecal_config.yaml if no config path specified
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'quickrecal_config.yaml')
        
        # Load configuration if available
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config and 'search_fusion' in yaml_config:
                        fusion_config = yaml_config['search_fusion']
                        self.search_fusion_weights = {
                            'similarity_weight': fusion_config.get('similarity_weight', 0.6),
                            'quickrecal_weight': fusion_config.get('quickrecal_weight', 0.4),
                            'use_logarithmic_fusion': fusion_config.get('use_logarithmic_fusion', True)
                        }
                        logger.info(f"Loaded QuickRecal search fusion weights from {config_path}")
                        return True
            except Exception as e:
                logger.warning(f"Error loading QuickRecal config: {e}")
        
        return False
    
    async def connect(self):
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
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from HPCServer")
    
    async def process_embeddings(self, embeddings):
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
        if not self.connected:
            await self.connect()

        request = {
            "type": "get_geometry",
            "model_version": model_version
        }
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        return json.loads(response)

    async def analyze_emotions(self, text: str) -> dict:
        """Analyze emotions in text using the HPC server.
        
        Args:
            text: The text to analyze for emotional content
            
        Returns:
            Dictionary containing emotion analysis results
        """
        if not self.connected:
            await self.connect()
        
        request = {
            'type': 'emotional_analysis',
            'text': text
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            result = json.loads(response)
            
            if result.get('type') == 'error':
                logger.error(f"Error from server: {result.get('message') or result.get('error')}")
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing emotions: {str(e)}")
            return {
                'type': 'error', 
                'error': str(e),
                'emotions': {'neutral': 1.0},  # Default fallback
                'dominant_emotion': 'neutral'
            }

    async def process_embeddings_with_qr(self, embeddings, min_qr_threshold=0.0):
        """Process embeddings with QuickRecal and return combined scores.
        
        Args:
            embeddings: The embeddings to process
            min_qr_threshold: Minimum QR score threshold for soft filtering
            
        Returns:
            Dict with processed scores and embeddings
        """
        if not self.connected:
            await self.connect()
        
        # Convert tensor or numpy to list if needed
        if not isinstance(embeddings, list):
            embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        else:
            embeddings_list = embeddings
        
        request = {
            'type': 'process',
            'embeddings': embeddings_list,
            'include_qr_score': True,
            'min_qr_threshold': min_qr_threshold,
            'fusion_weights': self.search_fusion_weights
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error processing embeddings with QR: {str(e)}")
            return {'type': 'error', 'error': str(e)}

    async def request_score_distribution(self):
        """Request the server to log score distribution for monitoring purposes"""
        if not self.connected:
            await self.connect()
        
        request = {
            'type': 'log_score_distribution'
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error requesting score distribution: {str(e)}")
            return {'type': 'error', 'error': str(e)}

if __name__ == '__main__':
    server = HPCServer()
    asyncio.run(server.start())
