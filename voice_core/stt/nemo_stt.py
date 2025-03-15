import os
import io
import time
import asyncio
import logging
import torch
import base64
import numpy as np
import websockets
import json
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor
import uuid
import traceback
import ssl
import socket

# Import NeMo if available, or use dummy imports for development
try:
    from nemo.collections.asr.models import EncDecMultiTaskModel
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    # For development or environments without NeMo
    class EncDecMultiTaskModel:
        @staticmethod
        def from_pretrained(model_name):
            print(f"Mock loading model: {model_name}")
            return EncDecMultiTaskModel()
            
        def __init__(self):
            self.device = "cpu"
            
        def to(self, device):
            self.device = device
            return self
            
        def eval(self):
            return self
            
        async def transcribe(self, audio_signal, *args, **kwargs):
            # Mock transcription for development
            return {"text": "This is a mock transcription.", "confidence": 0.95}

class NemoSTT:
    """Modular Nemo-based STT class for speech-to-text processing.
    
    This class provides a modular interface for speech-to-text processing using NVIDIA's NeMo models.
    It can be used standalone or integrated with other services through WebSockets or callbacks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the NemoSTT engine.
        
        Args:
            config: Configuration dictionary with the following optional keys:
                - model_name: Name or path of NeMo model (default: "nvidia/canary-1b")
                - device: Device to use for computation ("cuda" or "cpu", default: auto-detect)
                - hpc_url: URL of HPC server for semantic processing (default: None)
                - enable_streaming: Enable streaming transcription (default: False)
                - max_audio_length: Maximum audio length in seconds (default: 30)
                - sample_rate: Sample rate to use for audio processing (default: 16000)
                - docker_endpoint: WebSocket endpoint for Docker STT service (default: None)
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        default_config = {
            "model_name": "nvidia/canary-1b",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "hpc_url": os.environ.get("HPC_SERVER_URL", None),
            "enable_streaming": False,
            "max_audio_length": 30,  # seconds
            "sample_rate": 16000,     # Hz
            "thread_pool_size": 2,
            "docker_endpoint": os.environ.get("NEMO_DOCKER_ENDPOINT", "ws://localhost:5002/ws/transcribe")   # Default Docker STT service endpoint
        }
        
        # Update with provided config
        self.config = default_config
        if config:
            self.config.update(config)
            
        # Initialize state variables
        self.model = None
        self.model_loaded = False
        self.hpc_client = None
        self.executor = ThreadPoolExecutor(max_workers=self.config["thread_pool_size"])
        self.docker_client = None
        self.use_docker = self.config["docker_endpoint"] is not None
        
        # Initialize callback registry
        self.callbacks = {
            "on_transcription": [],  # Called when transcription is complete
            "on_semantic": [],      # Called when semantic processing is complete
            "on_error": []          # Called when an error occurs
        }
        
    async def initialize(self):
        """Initialize the STT model and HPC client."""
        try:
            # First, check if we should use Docker STT service
            if self.use_docker:
                self.logger.info(f"Using Docker STT service at {self.config['docker_endpoint']}")
                await self._init_docker_client()
                self.logger.info("Docker STT client initialized successfully")
            # Only load local model if not using Docker and NeMo is available
            elif NEMO_AVAILABLE:
                # Load the ASR model
                self.logger.info(f"Loading ASR model: {self.config['model_name']}")
                
                if self.config['model_name'].endswith('.nemo') and os.path.isfile(self.config['model_name']):
                    # Load from local file
                    self.model = EncDecMultiTaskModel.restore_from(self.config['model_name'])
                else:
                    # Load from HuggingFace
                    self.model = EncDecMultiTaskModel.from_pretrained(self.config['model_name'])
                
                # Move to appropriate device
                self.model = self.model.to(self.config['device'])
                self.model.eval()
                self.model_loaded = True
                self.logger.info(f"ASR model loaded successfully on {self.config['device']}")
            else:
                self.logger.warning("NeMo not available and Docker endpoint not configured - using mock model")
            
            # Initialize HPC client if URL is provided
            if self.config['hpc_url']:
                self.logger.info(f"Initializing HPC client at {self.config['hpc_url']}")
                await self._init_hpc_client()
                
        except Exception as e:
            self.logger.error(f"Error initializing NemoSTT: {e}", exc_info=True)
            await self._trigger_callbacks("on_error", {"error": f"Initialization error: {str(e)}"})
            raise
            
    async def _init_docker_client(self):
        """Initialize the Docker client for STT."""
        try:
            docker_endpoint = self.config.get("docker_endpoint")
            self.logger.info(f"Initializing Docker STT client with endpoint: {docker_endpoint}")
            
            # First try the primary Docker endpoint
            self.docker_client = NemoSTTDocker(docker_endpoint, logger=self.logger)
            
            # Try connecting with the primary endpoint
            try:
                await self.docker_client.connect()
                self.logger.info(f"Successfully connected to Docker STT endpoint: {docker_endpoint}")
                return True
            except Exception as primary_error:
                self.logger.warning(f"Failed to connect to primary Docker endpoint {docker_endpoint}: {str(primary_error)}")
                
                # Try alternative endpoints if primary fails
                alternative_endpoints = [
                    # Try different port combinations
                    docker_endpoint.replace(":5002", ":5000"),
                    docker_endpoint.replace(":5002", ":8000"),
                    # Try different hostname combinations
                    docker_endpoint.replace("localhost", "127.0.0.1"),
                    # If using host.docker.internal on Windows/Mac, try alternate
                    docker_endpoint.replace("host.docker.internal", "localhost"),
                    # If the endpoint includes path, try without it
                    docker_endpoint.split("/ws")[0] if "/ws" in docker_endpoint else docker_endpoint
                ]
                
                # If a Docker container is running the STT service
                docker_host = os.environ.get("DOCKER_HOST_STT_SERVICE", None)
                if docker_host:
                    alternative_endpoints.append(f"ws://{docker_host}:5002/ws/transcribe")
                
                # Add NEMO_DOCKER_ENDPOINT_FALLBACK if defined in environment
                fallback = os.environ.get("NEMO_DOCKER_ENDPOINT_FALLBACK", None)
                if fallback:
                    alternative_endpoints.append(fallback)
                
                # Also try looking up NEMO_DOCKER_ENDPOINT environment variable
                # as final fallback in case it was updated after this class was initialized
                env_endpoint = os.environ.get("NEMO_DOCKER_ENDPOINT", None)
                if env_endpoint and env_endpoint != docker_endpoint:
                    alternative_endpoints.append(env_endpoint)
                    
                # Try each alternative endpoint
                for i, alt_endpoint in enumerate(alternative_endpoints):
                    try:
                        self.logger.info(f"Trying alternative Docker STT endpoint {i+1}/{len(alternative_endpoints)}: {alt_endpoint}")
                        
                        # Create a new client for this endpoint
                        self.docker_client = NemoSTTDocker(alt_endpoint, logger=self.logger)
                        await self.docker_client.connect()
                        
                        # Update the config with the successful endpoint for future reference
                        self.config["docker_endpoint"] = alt_endpoint
                        self.logger.info(f"Successfully connected to alternative Docker STT endpoint: {alt_endpoint}")
                        return True
                    except Exception as alt_error:
                        self.logger.warning(f"Failed to connect to alternative endpoint {alt_endpoint}: {str(alt_error)}")
                
                # If all connection attempts fail, raise the original error
                self.logger.error(f"All connection attempts to Docker STT endpoints failed. Original error: {str(primary_error)}")
                raise primary_error
        
        except Exception as e:
            self.logger.error(f"Error initializing Docker STT client: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    async def test_docker_connection(self, endpoint_url=None):
        """Test the connection to the Docker STT service.
        
        This method can be used to verify connectivity to the STT service without
        sending any actual audio data. It tries to establish a connection to the
        specified endpoint and reports on success or failure.
        
        Args:
            endpoint_url (str, optional): Specific endpoint URL to test. If not provided,
                                         the default Docker endpoint will be used.
                                         
        Returns:
            tuple: (bool, str) - Success status and a message with details
        """
        try:
            endpoint = endpoint_url or self.config.get("docker_endpoint")
            self.logger.info(f"Testing connection to Docker STT endpoint: {endpoint}")
            
            # Create a temporary client for testing
            import websocket
            ws = websocket.create_connection(endpoint, timeout=5)
            
            # Try sending a simple ping message
            ws.send(json.dumps({"type": "ping"}))
            
            # Check if we can receive a response
            try:
                response = ws.recv()
                if response:
                    self.logger.info(f"Successfully connected to {endpoint} and received response: {response}")
                    ws.close()
                    return True, f"Successfully connected to {endpoint}"
            except Exception as e:
                # If no response, but connection was successful
                self.logger.info(f"Successfully connected to {endpoint} but no response received: {str(e)}")
                ws.close()
                return True, f"Connected to {endpoint} but no response received"
                
        except Exception as e:
            self.logger.error(f"Failed to connect to {endpoint}: {str(e)}")
            return False, f"Failed to connect to Docker STT endpoint {endpoint}: {str(e)}"
            
    async def _init_hpc_client(self):
        """Initialize the HPC client for semantic processing."""
        try:
            # We'll use a simple websocket-based client for HPC communication
            # This could be replaced with a more robust implementation
            self.hpc_client = HPCClient(self.config['hpc_url'])
            await self.hpc_client.connect()
            self.logger.info("HPC client connected successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize HPC client: {e}. Semantic processing will be disabled.")
            self.hpc_client = None
            
    async def transcribe(self, audio_data: Union[bytes, np.ndarray, str], 
                        request_id: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as bytes, numpy array, or base64 string
            request_id: Optional identifier for this transcription request
            
        Returns:
            Dict with transcription results
        """
        start_time = time.time()
        
        try:
            # First try using Docker STT service if available
            if self.use_docker and self.docker_client is not None:
                self.logger.info("Transcribing using Docker STT service")
                try:
                    # Process audio data into format expected by Docker service
                    audio_bytes = await self._ensure_audio_bytes(audio_data)
                    
                    # Send to Docker service
                    result = await self.docker_client.transcribe(audio_bytes, request_id)
                    
                    if result and "error" not in result:
                        transcription = {
                            "text": result.get("text", ""),
                            "confidence": result.get("confidence", 0.95),
                            "processing_time": time.time() - start_time,
                            "request_id": request_id,
                            "source": "docker"
                        }
                        
                        self.logger.info(f"Docker transcription completed in {transcription['processing_time']:.2f}s: {transcription['text']}")
                        
                        # Trigger callbacks
                        await self._trigger_callbacks("on_transcription", transcription)
                        
                        # Process with HPC if available
                        if transcription["text"] and self.hpc_client:
                            await self._process_with_hpc(transcription)
                            
                        return transcription
                    else:
                        error_msg = result.get("error", "Unknown error") if result else "Empty result"
                        self.logger.error(f"Docker STT error: {error_msg}")
                        # Do not raise exception, let it fall through to next method
                        
                except Exception as docker_e:
                    self.logger.error(f"Error using Docker STT: {docker_e}", exc_info=True)
                    # Do not raise exception, let it fall through to next method
                    
            # Then try using local NeMo model if available
            if self.model_loaded and self.model is not None:
                self.logger.info("Transcribing using local NeMo model")
                try:
                    # Process audio data into tensor
                    audio_tensor = await self._process_audio_data(audio_data)
                    
                    # Run transcription in thread pool to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, 
                        self.run_transcribe,
                        audio_tensor
                    )
                    
                    transcription = {
                        "text": result.get("text", ""),
                        "confidence": result.get("confidence", 0.95),
                        "processing_time": time.time() - start_time,
                        "request_id": request_id,
                        "source": "local"
                    }
                    
                    self.logger.info(f"Local transcription completed in {transcription['processing_time']:.2f}s: {transcription['text']}")
                    
                    # Trigger callbacks
                    await self._trigger_callbacks("on_transcription", transcription)
                    
                    # Process with HPC if available
                    if self.hpc_client and transcription["text"]:
                        await self._process_with_hpc(transcription)
                        
                    return transcription
                except Exception as local_e:
                    self.logger.error(f"Error using local NeMo model: {local_e}", exc_info=True)
                    await self._trigger_callbacks("on_error", {"error": str(local_e), "request_id": request_id})
                    raise
                    
            # If we get here, neither Docker nor local model worked
            self.logger.warning("Using mock result because NeMo is not available")
            mock_result = {
                "text": "Error: NeMo STT model is not available. Please configure Docker STT endpoint.",
                "confidence": 0.1,  # Lowered confidence so we can tell it's a mock result
                "processing_time": time.time() - start_time,
                "request_id": request_id,
                "source": "mock"
            }
            
            # Trigger callbacks with special flag to indicate STT unavailability
            mock_result["is_stt_error"] = True  # Add flag to indicate this is an STT service error
            await self._trigger_callbacks("on_transcription", mock_result)
            return mock_result
            
        except Exception as e:
            error_msg = f"Transcription error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            await self._trigger_callbacks("on_error", {"error": error_msg, "request_id": request_id})
            return {
                "text": "Error during transcription.",
                "confidence": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "request_id": request_id
            }
            
    async def _process_audio_data(self, audio_data: Union[bytes, np.ndarray, str]) -> torch.Tensor:
        """Process different audio input formats into a tensor for the model.
        
        Args:
            audio_data: Audio data in various formats
            
        Returns:
            Audio tensor ready for the model
        """
        try:
            # Handle different input types
            if isinstance(audio_data, str) and audio_data.startswith("data:audio/"):
                # Extract base64 data from data URL
                audio_data = audio_data.split(",", 1)[1]
                
            if isinstance(audio_data, str):
                # Assume base64 encoded
                audio_bytes = base64.b64decode(audio_data)
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
            elif isinstance(audio_data, bytes):
                # Raw bytes - FIX: assign audio_bytes before using it
                audio_bytes = audio_data
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
            elif isinstance(audio_data, np.ndarray):
                # Already numpy array
                audio_np = audio_data
            else:
                raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
            
            # Log the shape and size for debugging
            self.logger.info(f"Processing {len(audio_np) if isinstance(audio_np, np.ndarray) else 'unknown'} bytes of audio data")
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_np, dtype=torch.float32)
            
            # Reshape if needed (NeMo expects [B, T])
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
                
            return audio_tensor.to(self.config['device'])
            
        except Exception as e:
            self.logger.error(f"Error processing audio data: {e}")
            raise ValueError(f"Failed to process audio data: {str(e)}")
            
    async def _ensure_audio_bytes(self, audio_data: Union[bytes, np.ndarray, str]) -> bytes:
        """Ensure audio data is in bytes format for Docker API.
        
        Args:
            audio_data: Audio data in various formats
            
        Returns:
            Audio data as bytes
        """
        if isinstance(audio_data, bytes):
            return audio_data
        elif isinstance(audio_data, str):
            # Assume base64 encoded
            try:
                return base64.b64decode(audio_data)
            except Exception as e:
                self.logger.error(f"Error decoding base64 audio: {e}")
                raise ValueError(f"Invalid base64 audio data: {str(e)}")
        elif isinstance(audio_data, np.ndarray):
            # Convert numpy array to bytes
            try:
                buffer = io.BytesIO()
                import soundfile as sf
                sf.write(buffer, audio_data, self.config["sample_rate"], format="wav")
                buffer.seek(0)
                return buffer.read()
            except Exception as e:
                self.logger.error(f"Error converting numpy array to bytes: {e}")
                raise ValueError(f"Failed to convert numpy array to bytes: {str(e)}")
        else:
            raise TypeError(f"Unsupported audio data type: {type(audio_data)}")
            
    async def _process_with_hpc(self, transcription: Dict[str, Any]):
        """Process transcription with HPC for semantic analysis.
        
        Args:
            transcription: Transcription result
        """
        try:
            if not self.hpc_client:
                return {"error": "HPC client not available"}
                
            # Get embedding
            embedding_result = await self.hpc_client.process_embedding(transcription["text"])
            
            if "error" in embedding_result:
                self.logger.error(f"Error getting embedding: {embedding_result['error']}")
                return embedding_result
                
            # Get stats
            stats_result = await self.hpc_client.get_stats(embedding_result)
            
            if "error" in stats_result:
                self.logger.error(f"Error getting stats: {stats_result['error']}")
                return stats_result
                
            # Create result object
            result = {
                "text": transcription["text"],
                "significance": stats_result.get("significance", 0.0),
                "request_id": transcription.get("request_id")
            }
            
            # Trigger callbacks
            await self._trigger_callbacks("on_semantic", result)
            
            return result
            
        except Exception as e:
            error_info = {"error": str(e), "request_id": transcription.get("request_id")}
            self.logger.error(f"Error in semantic processing: {e}")
            await self._trigger_callbacks("on_error", error_info)
            return error_info
            
    async def _trigger_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Trigger registered callbacks for an event.
        
        Args:
            event_type: Type of event (on_transcription, on_semantic, on_error)
            data: Data to pass to callbacks
        """
        for callback in self.callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")
                
    def register_callback(self, event_type: str, callback: Union[Callable[[Dict[str, Any]], None], 
                                                                Callable[[Dict[str, Any]], Awaitable[None]]]):
        """Register a callback for a specific event.
        
        Args:
            event_type: Event type (on_transcription, on_semantic, on_error)
            callback: Function to call when the event occurs
        """
        if event_type not in self.callbacks:
            self.logger.warning(f"Unknown event type: {event_type}")
            return False
            
        self.callbacks[event_type].append(callback)
        return True
        
    def unregister_callback(self, event_type: str, callback):
        """Unregister a callback.
        
        Args:
            event_type: Event type
            callback: Callback to remove
        """
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
            return True
        return False
        
    async def shutdown(self):
        """Clean up resources."""
        if self.hpc_client:
            await self.hpc_client.disconnect()
        self.executor.shutdown(wait=True)
        self.logger.info("NemoSTT shutdown complete")
        
    async def clear_buffer(self) -> None:
        """Clear any accumulated audio buffer.
        
        This method is called when we need to discard any accumulated audio data,
        such as when starting a new conversation or after processing a complete utterance.
        """
        self.logger.info("Clearing audio buffer")
        # No actual buffer to clear in this implementation since we process audio in chunks
        # and don't maintain a persistent buffer between calls.
        # This method exists for compatibility with the voice agent interface.
        return None

    async def diagnose_docker_service(self):
        """Run a comprehensive diagnostic on the Docker STT service connection.
        
        This method performs a series of checks to validate the Docker STT service:
        1. Checks if the server URL is properly formatted
        2. Attempts to connect using WebSocket
        3. Verifies if HTTP endpoint is reachable (for debugging)
        4. Tries to inspect Docker container status if possible
        
        Returns:
            dict: Diagnostic report with status and detailed messages
        """
        import socket
        import urllib.parse
        import re
        import subprocess
        
        report = {
            "status": "failed",
            "url": self.config.get("docker_endpoint", "Not configured"),
            "checks": [],
            "issues_found": [],
            "suggestions": []
        }
        
        # 1. Check URL format
        url = self.config.get("docker_endpoint", "")
        if not url:
            report["issues_found"].append("Docker endpoint URL is not configured")
            report["suggestions"].append("Set the NEMO_DOCKER_ENDPOINT environment variable")
            return report
            
        # Parse URL for validation
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ["ws", "wss"]:
                report["issues_found"].append(f"Invalid URL scheme: {parsed.scheme}, should be ws:// or wss://")
                report["suggestions"].append("Use a URL format like ws://localhost:8000/ws/transcribe")
            
            # Extract host and port
            host = parsed.netloc.split(":")[0] if ":" in parsed.netloc else parsed.netloc
            port_str = parsed.netloc.split(":")[1] if ":" in parsed.netloc else "80"
            port = int(re.search(r'^(\d+)', port_str).group(1)) if re.search(r'^(\d+)', port_str) else 80
            
            report["checks"].append({"name": "URL format", "status": "passed", "details": f"Valid WebSocket URL: {url}"})
        except Exception as e:
            report["issues_found"].append(f"URL parsing error: {str(e)}")
            report["suggestions"].append("Use a URL format like ws://localhost:8000/ws/transcribe")
            return report
            
        # 2. Check if host is reachable via socket connection
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((host, port))
            s.close()
            report["checks"].append({"name": "Host connectivity", "status": "passed", "details": f"Host {host}:{port} is reachable"})
        except Exception as e:
            report["issues_found"].append(f"Cannot connect to host {host}:{port}: {str(e)}")
            report["suggestions"].append("Verify the STT server is running and the port is correct")
            report["suggestions"].append("Check for firewall rules that might block the connection")
            
        # 3. Try to check if Docker container is running (if on same machine)
        if host in ["localhost", "127.0.0.1"]:
            try:
                # This assumes docker command is available
                self.logger.info("Checking Docker container status...")
                result = subprocess.run(["docker", "ps", "--format", "{{.Names}} ({{.Status}})"], capture_output=True, text=True, timeout=5)
                containers = result.stdout.strip().split('\n')
                
                # Look for containers with names that might be our STT server
                possible_stt_containers = [c for c in containers if any(x in c.lower() for x in ["stt", "nemo", "speech", "transcribe"])]
                
                if possible_stt_containers:
                    report["checks"].append({"name": "Docker containers", "status": "info", "details": f"Possible STT containers: {possible_stt_containers}"})
                else:
                    report["issues_found"].append("No Docker containers found that might be running the STT service")
                    report["suggestions"].append("Start the STT Docker container")
            except Exception as e:
                self.logger.warning(f"Unable to check Docker containers: {e}")
                
        # 4. Try a WebSocket connection with diagnostic info
        try:
            success, msg = await self.test_docker_connection()
            if success:
                report["status"] = "success"
                report["checks"].append({"name": "WebSocket connection", "status": "passed", "details": msg})
            else:
                report["issues_found"].append(f"WebSocket connection failed: {msg}")
                report["suggestions"].append("Make sure the STT server has the WebSocket handler enabled")
                
                # Try HTTP GET to the same endpoint to see if server responds at all
                http_url = url.replace("ws://", "http://").replace("wss://", "https://")
                try:
                    import urllib.request
                    response = urllib.request.urlopen(http_url, timeout=3)
                    report["checks"].append({
                        "name": "HTTP endpoint", 
                        "status": "info", 
                        "details": f"HTTP endpoint is reachable with status {response.status}, but WebSocket upgrade failed"
                    })
                    report["suggestions"].append("Server is running but might not support WebSockets - check server configuration")
                except Exception as http_error:
                    report["issues_found"].append(f"HTTP endpoint is also unreachable: {str(http_error)}")
                    report["suggestions"].append("The server might not be running at all. Start the STT service")
        except Exception as e:
            report["issues_found"].append(f"WebSocket test error: {str(e)}")
            
        # Return diagnostic report
        return report

    async def check_environment_variables(self):
        """Check environment variables that affect the STT service."""
        env_report = {
            "variables_found": [],
            "suggestions": []
        }
        
        # List of environment variables that might affect STT
        variables_to_check = [
            "NEMO_DOCKER_ENDPOINT",
            "NEMO_DOCKER_ENDPOINT_FALLBACK",
            "DOCKER_HOST_STT_SERVICE",
            "STT_SERVER_PORT",
            "HPC_SERVER_URL"
        ]
        
        for var in variables_to_check:
            value = os.environ.get(var)
            if value:
                env_report["variables_found"].append({"name": var, "value": value})
            else:
                env_report["suggestions"].append(f"Environment variable {var} is not set")
                
        return env_report

class HPCClient:
    """Simple WebSocket client for HPC server communication."""
    
    def __init__(self, url):
        """Initialize the HPC client.
        
        Args:
            url: WebSocket URL of the HPC server
        """
        self.url = url
        self.websocket = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
    async def connect(self):
        """Connect to the HPC server.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.websocket = await websockets.connect(self.url, ping_interval=20)
            self.connected = True
            self.logger.info(f"Connected to HPC server at {self.url}")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to HPC server: {e}")
            self.connected = False
            return False
            
    async def disconnect(self):
        """Disconnect from the HPC server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            self.logger.info("Disconnected from HPC server")
            
    async def send_message(self, message):
        """Send a message to the HPC server and get response.
        
        Args:
            message: Message to send
            
        Returns:
            Response from the server
        """
        if not self.connected:
            await self.connect()
            if not self.connected:
                return {"error": "Not connected to HPC server"}
                
        try:
            # Send message
            await self.websocket.send(json.dumps(message))
            
            # Wait for response with timeout
            response_text = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            return json.loads(response_text)
            
        except asyncio.TimeoutError:
            self.logger.error("Timeout waiting for HPC server response")
            return {"error": "Timeout waiting for response"}
        except Exception as e:
            self.logger.error(f"Error communicating with HPC server: {e}")
            self.connected = False
            return {"error": str(e)}
            
    async def process_embedding(self, text):
        """Process text to get embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            Dict with embedding result
        """
        message = {
            "type": "embedding",
            "text": text,
            "source": "nemo_stt"
        }
        
        response = await self.send_message(message)
        
        # Handle response formats
        if response and "error" not in response:
            if response.get("type") == "embedding_result" and "embedding" in response:
                return {"embedding": response["embedding"]}
            return response
        else:
            return response or {"error": "No response from HPC server"}
    
    async def get_stats(self, embedding_data):
        """Get stats for an embedding.
        
        Args:
            embedding_data: Embedding data from process_embedding
            
        Returns:
            Dict with stats result
        """
        if not embedding_data or "embedding" not in embedding_data:
            return {"error": "Invalid embedding data"}
            
        message = {
            "type": "stats",
            "embedding": embedding_data["embedding"],
            "source": "nemo_stt"
        }
        
        response = await self.send_message(message)
        
        # Handle response formats
        if response and "error" not in response:
            if response.get("type") == "stats_result" and "significance" in response:
                return {"significance": response["significance"]}
            return response
        else:
            return response or {"error": "No response from HPC server"}

class NemoSTTDocker:
    """Simple WebSocket client for Docker STT service communication."""
    
    def __init__(self, url, logger):
        """Initialize the Docker STT client.
        
        Args:
            url: WebSocket URL of the Docker STT service
            logger: Logger instance
        """
        self.url = url
        self.websocket = None
        self.connected = False
        self.logger = logger
        
    async def connect(self):
        """Establish a WebSocket connection to the STT Docker service.
        
        This method will attempt to connect to the STT Docker service using the provided URL.
        It includes retry logic and detailed error reporting to help diagnose connection issues.
        
        Raises:
            ConnectionError: If unable to connect to the STT Docker service.
        """
        if self.websocket is not None and self.websocket.open:
            self.logger.debug("WebSocket connection already established")
            return
            
        self.logger.info(f"Connecting to STT Docker service at {self.url}")
        
        # Create a context that ignores SSL certificate verification for development/debugging
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Track whether we're using ws:// or wss://
        using_ssl = self.url.startswith("wss://")
        host = self.url.split("://")[1].split("/")[0]  # Extract host:port
        
        max_retries = 3
        retry_count = 0
        last_exception = None
        
        # Multiple connection attempts with different parameters
        while retry_count < max_retries:
            try:
                # First try with the original URL
                if retry_count == 0:
                    self.logger.info(f"Connection attempt {retry_count+1}/{max_retries} to {self.url}")
                    self.websocket = await websockets.connect(self.url, ping_interval=20, 
                                                             ssl=ssl_context if using_ssl else None)
                # Second try with ping_timeout increased
                elif retry_count == 1:
                    self.logger.info(f"Connection attempt {retry_count+1}/{max_retries} with increased ping_timeout")
                    self.websocket = await websockets.connect(self.url, ping_interval=20, ping_timeout=60, 
                                                             ssl=ssl_context if using_ssl else None)
                # Third try by checking if HTTP endpoint is available
                else:
                    # Try to check if there's an HTTP endpoint available instead
                    http_url = f"http{'s' if using_ssl else ''}://{host}"
                    self.logger.info(f"Checking if HTTP endpoint is available at {http_url}")
                    
                    # Attempt to send a ping message to verify connection
                    self.websocket = await websockets.connect(self.url, ping_interval=None, 
                                                             ssl=ssl_context if using_ssl else None)
                    
                # If we get here, we're connected
                self.logger.info(f"Successfully connected to {self.url}")
                
                # Send a ping to verify the connection is working
                await self.websocket.send(json.dumps({"type": "ping"}))
                response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                self.logger.info(f"Successfully connected to {self.url} and received response: {response}")
                
                return
            except (websockets.exceptions.WebSocketException, 
                    ConnectionError, 
                    OSError, 
                    asyncio.TimeoutError) as e:
                last_exception = e
                self.logger.warning(f"Connection attempt {retry_count+1} failed: {str(e)}")
                # Close websocket if it was created but there was an error with ping
                if self.websocket and hasattr(self.websocket, 'open') and self.websocket.open:
                    await self.websocket.close()
                self.websocket = None
                retry_count += 1
                await asyncio.sleep(1.0)  # Wait before retrying
                
        # Failed after all retries
        error_message = f"Failed to connect to STT Docker service at {self.url}: {str(last_exception)}"
        self.logger.error(error_message)
        
        # Attempt to check host reachability for better diagnostics
        try:
            host, port = host.split(":")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, int(port)))
            if result == 0:
                self.logger.info(f"Host {host}:{port} is reachable, but WebSocket connection failed")
                # This indicates the server is running but not accepting WebSocket connections
                error_message += f". The host {host}:{port} is reachable, but the WebSocket handshake failed. The server may not be accepting WebSocket connections on this endpoint."
            else:
                self.logger.warning(f"Host {host}:{port} is not reachable (error code: {result})")
                # This indicates a network/firewall issue
                error_message += f". The host {host}:{port} is not reachable (error code: {result}). This may indicate a network or firewall issue."
            sock.close()
        except Exception as e:
            self.logger.warning(f"Failed to check host reachability: {e}")
            
        raise ConnectionError(error_message)
        
    async def disconnect(self):
        """Disconnect from the Docker STT service."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            self.logger.info("Disconnected from Docker STT service")
            
    async def send_message(self, message):
        """Send a message to the Docker STT service and get response.
        
        Args:
            message: Message to send
            
        Returns:
            Response from the service
        """
        if not self.connected:
            await self.connect()
            if not self.connected:
                return {"error": "Not connected to Docker STT service"}
                
        try:
            # Send message
            await self.websocket.send(json.dumps(message))
            
            # Wait for response with timeout
            response_text = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            return json.loads(response_text)
            
        except asyncio.TimeoutError:
            self.logger.error("Timeout waiting for Docker STT service response")
            return {"error": "Timeout waiting for response"}
        except Exception as e:
            self.logger.error(f"Error communicating with Docker STT service: {e}")
            self.connected = False
            return {"error": str(e)}
            
    async def transcribe(self, audio_bytes, request_id):
        """Transcribe audio data to text.
        
        Args:
            audio_bytes: Audio data as bytes
            request_id: Optional identifier for this transcription request
            
        Returns:
            Dict with transcription results
        """
        if not self.websocket:
            await self.connect()
            
        if not self.websocket:
            self.logger.error("Failed to connect to Docker STT service")
            return {"error": "Connection failed"}
        
        try:
            # Send binary audio data directly - STT_server will interpret it as
            # {"type": "audio", "audio_data": binary_data} internally
            self.logger.debug(f"Sending {len(audio_bytes)} bytes of audio data")
            await self.websocket.send(audio_bytes)
            
            # Wait for transcription response - keep receiving until we get final result
            # The Docker service may send multiple messages including status updates
            start_time = time.time()
            max_wait_time = 30.0  # Maximum time to wait for transcription
            
            while True:
                try:
                    # Set a reasonable timeout for each message
                    response_text = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                    self.logger.debug(f"Received response: {response_text}")
                    
                    try:
                        response = json.loads(response_text)
                        
                        # Check if this is a status message
                        if response.get("type") == "status":
                            self.logger.debug(f"Received status update: {response.get('message', 'No message')}")
                            # Continue waiting for the final transcription
                            continue
                            
                        # Check if this is the final transcription
                        if response.get("type") == "transcription" and "text" in response:
                            self.logger.info(f"Received final transcription: {response['text']}")
                            return {"text": response["text"], "confidence": response.get("confidence", 0.95)}
                            
                        # Handle direct text response
                        if "text" in response:
                            self.logger.info(f"Received transcription: {response['text']}")
                            return {"text": response["text"], "confidence": response.get("confidence", 0.95)}
                            
                        # Check if we've exceeded the maximum wait time
                        if time.time() - start_time > max_wait_time:
                            self.logger.warning(f"Timed out waiting for final transcription after {max_wait_time} seconds")
                            return {"error": "Timeout waiting for final transcription", "partial_response": response}
                            
                    except json.JSONDecodeError:
                        self.logger.warning(f"Received non-JSON response: {response_text}")
                        continue
                        
                except asyncio.TimeoutError:
                    self.logger.error("Timeout waiting for Docker STT service response")
                    return {"error": "Timeout waiting for response"}
                    
        except Exception as e:
            self.logger.error(f"Error during transcription with Docker STT service: {e}")
            return {"error": str(e)}
            
    async def send_audio_message(self, message):
        """Send audio data to the WebSocket with proper handling of binary data."""
        try:
            if not self.websocket:
                await self.connect()
            
            if not self.websocket:
                self.logger.error("Failed to connect to Docker STT service")
                return {"error": "Connection failed"}
                
            # Extract audio data from message to send separately if needed
            audio_data = message.pop("audio_data", None)
            
            # First send the message without the audio data
            await self.websocket.send(json.dumps({"type": message["type"]}))
            
            # Then send the binary audio data
            if audio_data:
                await self.websocket.send(audio_data)
            
            # Wait for and return the response
            try:
                response_text = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    return {"error": f"Invalid JSON response: {response_text[:100]}..."}
            except asyncio.TimeoutError:
                return {"error": "Response timeout"}
                
        except Exception as e:
            self.logger.error(f"Error sending audio message: {e}")
            return {"error": f"Send error: {str(e)}"}
