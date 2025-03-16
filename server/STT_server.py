#!/usr/bin/env python
"""
LUCID RECALL PROJECT
Speech-to-Text Server: Continuous transcription with real-time and final results
"""

import asyncio
import base64
import json
import logging
import os
import time
import traceback
import uuid
import argparse
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosedOK
from fastapi.middleware.cors import CORSMiddleware
from nemo.collections.asr.models import EncDecRNNTBPEModel
import threading
import uuid
import traceback
import base64
import io
import soundfile as sf
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("stt_server")

# Configuration
HPC_SERVER_URL = os.getenv("HPC_SERVER_URL", "hpc_server:5005")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STT_SERVER_PORT = 5002

# Initialize FastAPI app
app = FastAPI(title="STT Transcription Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
connected_clients = set()  # Set of connected client IDs
audio_buffer = {}  # Buffer to store audio chunks for each client
transcription_history = {}  # Store transcription history for each client
hpc_client = None  # HPC client for semantic processing
active_connections = {}
stt_server = None
model = None
model_loaded = False

# Function to preprocess audio data
def preprocess_audio(audio_bytes):
    """Preprocess audio data from bytes to numpy array."""
    try:
        # Try to read with soundfile
        with io.BytesIO(audio_bytes) as audio_io:
            audio_data, sample_rate = sf.read(audio_io)
            
            # Convert to float32 type to avoid type mismatch issues
            audio_data = audio_data.astype(np.float32)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate)).astype(np.float32)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1).astype(np.float32)
            
            # Normalize audio (between -1 and 1)
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
                
            logger.info(f"Preprocessed audio: shape={audio_data.shape}, dtype={audio_data.dtype}")
            return audio_data
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        logger.error(traceback.format_exc())
        raise

class STTServer:
    def __init__(self):
        self.model = None
        self.device = DEVICE
        self.sample_rate = 16000  # Expected sample rate for models
        self.setup_gpu()
        self.load_model()
        
    def setup_gpu(self) -> None:
        """Set up GPU for inference if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.backends.cudnn.benchmark = True
            logger.info(f"GPU initialized: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU not available, using CPU. This will be significantly slower.")

    def load_model(self):
        """Load the ASR model for transcription."""
        try:
            logger.info("Loading ASR model...")
            
            # Check if Canary model should be loaded
            model_path = os.getenv("ASR_MODEL_PATH", "/workspace/models/canary-1b")
            
            try:
                if os.path.exists(model_path) and not os.path.isdir(model_path):
                    # Load from local file if it exists and is not a directory
                    logger.info(f"Loading Canary model from local path: {model_path}")
                    self.model = EncDecRNNTBPEModel.restore_from(model_path)
                else:
                    # Load from pretrained source
                    logger.info("Loading Canary model from pretrained source...")
                    self.model = EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/canary-1b")
                    
                # Move model to the specified device
                self.model = self.model.to(self.device)
                logger.info("Canary ASR model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading Canary-1B model: {e}")
                logger.error(traceback.format_exc())
                self.model = None
        except Exception as e:
            logger.error(f"Error in model loading: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    async def transcribe_audio(self, audio_data, client_id=None):
        """Transcribe audio data using the Canary-1B model.
        
        Args:
            audio_data: Audio data as numpy array or base64-encoded binary data
            client_id: Client ID for tracking transcription history
        
        Returns:
            dict: Transcription result with text and metadata
        """
        if self.model is None:
            logger.error("No ASR model loaded. Cannot transcribe audio.")
            return {"error": "ASR model not loaded"}
        
        try:
            # Process the audio data based on its type
            if isinstance(audio_data, str) and audio_data.startswith("data:audio"):
                # Handle data URL format
                _, encoded = audio_data.split(",", 1)
                audio_bytes = base64.b64decode(encoded)
                # Convert to appropriate format for ASR
                audio_signal = preprocess_audio(audio_bytes)
            elif isinstance(audio_data, str):
                # Assume base64 encoded audio
                audio_bytes = base64.b64decode(audio_data)
                audio_signal = preprocess_audio(audio_bytes)
            elif isinstance(audio_data, dict) and "audio_base64" in audio_data:
                # Handle base64 encoded audio in a dict
                audio_bytes = base64.b64decode(audio_data["audio_base64"])
                audio_signal = preprocess_audio(audio_bytes)
            elif isinstance(audio_data, np.ndarray):
                # Handle numpy array directly
                audio_signal = audio_data.astype(np.float32)  # Ensure float32 type
            else:
                logger.error(f"Unsupported audio data type: {type(audio_data)}")
                return {"error": f"Unsupported audio data type: {type(audio_data)}"}
            
            start_time = time.time()
            
            # Process with Canary model
            canary_text = ""
            try:
                # Process with Canary for highest quality
                with torch.no_grad():
                    # Ensure audio is in the right format for the model
                    if len(audio_signal.shape) == 1:
                        # Reshape to [batch, time] for the model
                        audio_signal = audio_signal.reshape(1, -1)
                        # Ensure float32 type for PyTorch tensor
                        audio_signal = torch.tensor(audio_signal, dtype=torch.float32, device=self.device)
                    elif isinstance(audio_signal, np.ndarray):
                        # Ensure float32 type for PyTorch tensor
                        audio_signal = torch.tensor(audio_signal, dtype=torch.float32, device=self.device)
                        
                    # Log the tensor type for debugging
                    logger.info(f"Tensor shape: {audio_signal.shape}, type: {audio_signal.dtype}")
                    
                    # ASR processing with Canary
                    canary_text = self.model.transcribe([audio_signal])[0]
                    logger.info(f"Canary transcription: '{canary_text}'")
            except Exception as e:
                logger.error(f"Error with Canary transcription: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Store in history if client_id is provided
            if client_id and client_id in active_connections:
                if client_id not in transcription_history:
                    transcription_history[client_id] = []
                    
                # Store transcription
                transcription_history[client_id].append({
                    "text": canary_text,
                    "timestamp": time.time()
                })
            
            # Return results
            result = {
                "text": canary_text,
                "processing_time": processing_time,
                "timestamp": time.time()
            }
            
            logger.info(f"Transcription completed in {processing_time:.2f}s - Text: '{canary_text}'")
            return result
        
        except Exception as e:
            logger.error(f"Error in transcription pipeline: {e}", exc_info=True)
            return {
                "error": f"Error in transcription: {str(e)}",
                "text": ""
            }

class HPCClient:
    """Client for connecting to HPC server and processing embeddings and stats."""
    
    def __init__(self, url):
        """Initialize HPC client with server URL.
        
        Args:
            url: WebSocket URL for HPC server
        """
        self.url = url
        self.ws = None
        self.connected = False
        self.connection_lock = asyncio.Lock()
        self.loop = None  # Store the event loop used for connection
        self.pending_responses = {}  # Store pending responses for message IDs
        self.send_loop = None  # Store the event loop for sending messages

    async def connect(self):
        """Connect to HPC server."""
        async with self.connection_lock:
            if self.connected:
                logger.info("Already connected to HPC server")
                return
            
            try:
                # Store the event loop for later use
                self.loop = asyncio.get_running_loop()
                self.send_loop = self.loop  # Store the send loop
                
                # Connect to WebSocket server
                logger.info(f"Connecting to HPC server at {self.url}")
                self.ws = await websockets.connect(self.url, ping_interval=30)
                self.connected = True
                logger.info(f"Connected to HPC server at {self.url}")
                
                # Start message handling task
                self.message_task = asyncio.create_task(self._handle_messages())
            except Exception as e:
                logger.error(f"Error connecting to HPC server: {e}")
                self.connected = False  # Mark as disconnected for reconnect
                raise
    
    async def disconnect(self):
        """Disconnect from HPC server."""
        if self.ws:
            await self.ws.close()
            self.connected = False
            logger.info("Disconnected from HPC server")
        
        # Cancel message handling task
        if hasattr(self, 'message_task'):
            self.message_task.cancel()
    
    async def ensure_connected(self):
        """Ensure connection is established, reconnect if needed."""
        if not self.connected or not self.ws or self.ws.closed:
            await self.connect()
    
    async def send_message(self, message):
        """Send a message to the HPC server and wait for a response.
        
        Args:
            message (dict): Message to send
        
        Returns:
            dict: Response from the server or None if error
        """
        try:
            # Ensure we're connected
            await self.ensure_connected()
            
            if not self.ws or not self.connected:
                logger.error("Cannot send message: not connected to HPC server")
                return {"error": "Not connected to HPC server"}
            
            # Create a future to get the response
            message_id = str(uuid.uuid4())
            message["id"] = message_id
            
            # Store the future
            response_future = asyncio.Future()
            self.pending_responses[message_id] = response_future
            
            # Send message using the correct event loop
            if self.send_loop and self.send_loop != asyncio.get_running_loop():
                # Schedule in the correct loop
                await asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(
                        self._send_raw_message(json.dumps(message)),
                        self.send_loop
                    )
                )
            else:
                # Same loop or no specific loop stored
                await self._send_raw_message(json.dumps(message))
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(response_future, timeout=10.0)  # 10 second timeout
                return response
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for response to message {message_id}")
                # Remove the pending response to avoid memory leaks
                if message_id in self.pending_responses:
                    del self.pending_responses[message_id]
                return {"error": "Timeout waiting for response from HPC server"}
        
        except Exception as e:
            logger.error(f"Error communicating with HPC server: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    async def process_embedding(self, text):
        """Send text to HPC server for embedding processing.
        
        Args:
            text (str): Text to process
        
        Returns:
            dict: Embedding result or error
        """
        if not text or not text.strip():
            return {"error": "Empty text provided"}
        
        try:
            logger.info(f"Sending text for embedding: {text[:50]}..." if len(text) > 50 else f"Sending text for embedding: {text}")
            
            # Format the message with proper fields
            message = {
                "type": "embedding",
                "text": text,
                "source": "stt_server"
            }
            
            # Send message and get response
            response = await self.send_message(message)
            
            if response and "error" not in response and ("embedding" in response or "embedding_result" in response.get("type", "")):
                logger.info("Successfully received embedding from HPC server")
                # If the response has type 'embedding_result', extract embedding from the response
                if response.get("type") == "embedding_result" and "embedding" in response:
                    return {"embedding": response["embedding"]}
                return response
            else:
                error_msg = response.get("error", "Unknown error in embedding response") if response else "No response from HPC server"
                logger.error(f"Error getting embedding: {error_msg}")
                return {"error": error_msg}
        
        except Exception as e:
            logger.error(f"Exception in process_embedding: {e}")
            return {"error": str(e)}

    async def get_stats(self, embedding_data):
        """Get stats for an embedding from the HPC server.
        
        Args:
            embedding_data (dict): Embedding data returned from process_embedding
        
        Returns:
            dict: Stats result or error
        """
        try:
            if not embedding_data or "embedding" not in embedding_data:
                return {"error": "Invalid embedding data"}
            
            # Format the stats request
            message = {
                "type": "stats",
                "embedding": embedding_data["embedding"],
                "source": "stt_server"
            }
            
            # Send message and get response
            response = await self.send_message(message)
            
            if response and "error" not in response:
                # Handle stats_result type response
                if response.get("type") == "stats_result" and "significance" in response:
                    logger.info(f"Received stats from HPC server: Significance = {response.get('significance', 'N/A')}")
                    return {"significance": response.get("significance", 0.0)}
                logger.info(f"Received stats from HPC server: Significance = {response.get('significance', 'N/A')}")
                return response
            else:
                error_msg = response.get("error", "Unknown error in stats response") if response else "No response from HPC server"
                logger.error(f"Error getting stats: {error_msg}")
                return {"error": error_msg}
        
        except Exception as e:
            logger.error(f"Exception in get_stats: {e}")
            return {"error": str(e)}

    async def process_geometry(self, embedding, operation, **kwargs):
        """Process geometry operation on embedding.
        
        Args:
            embedding: Embedding vector or dict
            operation: Geometry operation name
            **kwargs: Additional operation parameters
            
        Returns:
            Dict: Operation result
        """
        # Handle both raw embedding and dict with embedding
        if isinstance(embedding, dict) and "embedding" in embedding:
            embedding_data = embedding["embedding"]
        else:
            embedding_data = embedding
        
        message = {
            "type": "geometry",
            "operation": operation,
            "embedding": embedding_data,
            **kwargs
        }
        
        return await self.send_message(message)

    async def _send_raw_message(self, message_str):
        """Send a raw message string to the WebSocket server.
        
        Args:
            message_str (str): JSON message string to send
        """
        if not self.ws or not self.connected:
            raise ConnectionError("Not connected to HPC server")
        
        try:
            await self.ws.send(message_str)
            return True
        except Exception as e:
            logger.error(f"Error sending raw message: {e}")
            self.connected = False  # Mark as disconnected for reconnect
            raise

    async def _handle_messages(self):
        """Background task to handle incoming messages from the HPC server."""
        try:
            while self.connected and self.ws:
                try:
                    # Wait for message with timeout to allow checking connection status
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30)
                    
                    # Parse message
                    try:
                        data = json.loads(message)
                        
                        # Check if it's a response to a pending request
                        message_id = data.get("id")
                        if message_id and message_id in self.pending_responses:
                            # Resolve the future with the response data
                            future = self.pending_responses.pop(message_id)
                            if not future.done():
                                future.set_result(data)
                        else:
                            # Handle other message types as needed
                            logger.info(f"Received unsolicited message: {data}")
                    except json.JSONDecodeError:
                        logger.error(f"Received invalid JSON: {message}")
            
                except asyncio.TimeoutError:
                    # Just a timeout on receive, continue
                    continue
                except Exception as e:
                    if self.connected:  # Only log if we're supposed to be connected
                        logger.error(f"Error receiving message: {e}")
                        if "connection is closed" in str(e).lower() or "websocket is closed" in str(e).lower():
                            self.connected = False
                            break
                    else:
                        # We're shutting down, exit gracefully
                        break
    
        except Exception as e:
            logger.error(f"Message handler task error: {e}")
        finally:
            # Clean up any pending responses when the handler exits
            for future in self.pending_responses.values():
                if not future.done():
                    future.set_exception(ConnectionError("WebSocket connection closed"))
            self.pending_responses.clear()
            self.connected = False

# Initialize the STT server
async def initialize_stt_server():
    global hpc_client, model, model_loaded, DEVICE
    
    # Load configuration
    logger.info("Initializing STT server...")
    
    # Set up the HPC client if URL is provided
    hpc_server_url = os.environ.get('HPC_SERVER_URL')
    if hpc_server_url:
        logger.info(f"Connecting to HPC server at {hpc_server_url}")
        hpc_client = HPCClient(hpc_server_url)
        try:
            # Connect the HPC client
            await hpc_client.connect()
            # Send a ping to test the connection
            ping_result = await hpc_client.send_message({"type": "ping"})
            if ping_result and ping_result.get("type") == "pong":
                logger.info("Successfully connected to HPC server and received pong response")
            else:
                logger.warning(f"Connected to HPC server but ping test failed: {ping_result}")
        except Exception as e:
            logger.error(f"Error connecting to HPC server: {e}")
            logger.error("HPC client will attempt to reconnect automatically when needed")
    else:
        logger.warning("No HPC_SERVER_URL provided. Semantic processing will be disabled.")
        hpc_client = None

    # Initialize the ASR model
    try:
        logger.info("Loading ASR model...")
        # Use GPU if available
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {DEVICE}")
        
        # Get model path from environment or use default
        model_name = os.environ.get('ASR_MODEL_NAME', "nvidia/canary-1b")
        logger.info(f"Loading model from: {model_name}")
        
        # Load NeMo model
        if model_name.endswith(".nemo") and os.path.isfile(model_name):
            # Load from local file
            model = EncDecRNNTBPEModel.restore_from(model_name)
        else:
            # Load from HuggingFace
            model = EncDecRNNTBPEModel.from_pretrained(model_name)
        
        # Move to appropriate device
        model = model.to(DEVICE)
        model.eval()
        model_loaded = True
        logger.info("ASR model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading ASR model: {e}")
        logger.error(traceback.format_exc())
        model = None
        model_loaded = False

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global stt_server, hpc_client
    logger.info("Shutting down STT server...")
    
    # Disconnect HPC client
    if hpc_client:
        logger.info("Disconnecting from HPC server...")
        await hpc_client.disconnect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("STT server shutdown complete")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if stt_server and stt_server.model:
        return {
            "status": "healthy",
            "device": stt_server.device,
            "model": "nvidia/canary-1b",
            "connected_clients": len(connected_clients)
        }
    else:
        raise HTTPException(status_code=503, detail="STT server not initialized")

@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    if stt_server:
        return stt_server.get_stats()
    else:
        raise HTTPException(status_code=503, detail="STT server not initialized")

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    client_id = str(uuid.uuid4())
    
    try:
        # Accept the connection
        await websocket.accept()
        
        # Store the connection after successful accept
        active_connections[client_id] = websocket
        logger.info(f"New WebSocket connection accepted: {client_id}")
        
        # For FastAPI/Starlette WebSockets, use a loop with receive()
        while True:
            try:
                # Wait for message from client
                message = await websocket.receive()
                
                # Check for disconnect messages first
                if "type" in message and message["type"] == "websocket.disconnect":
                    logger.info(f"Client disconnected: {client_id}")
                    break
                
                # Check message type and data format
                if "text" in message:
                    # Text message
                    try:
                        # Try to parse as JSON
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        # Treat as raw text message
                        data = {"type": "text", "text": message["text"]}
                elif "bytes" in message:
                    # Binary audio data
                    data = {"type": "audio", "audio_data": message["bytes"]}
                else:
                    # Unknown message format
                    await safe_send_json(websocket, {"type": "error", "message": f"Unknown message format: {message}"})
                    continue
                
                # Handle different message types
                msg_type = data.get("type", "unknown")
                
                if msg_type == "ping":
                    # Handle ping messages
                    await safe_send_json(websocket, {"type": "pong"})
                    
                elif msg_type == "audio" or msg_type == "audio_binary":
                    # Handle audio transcription
                    await safe_send_json(websocket, {"type": "status", "message": "Processing audio..."})
                    
                    # Extract audio data
                    audio_data = None
                    if msg_type == "audio":
                        if "audio_data" in data:
                            audio_data = data["audio_data"]
                        elif "audio_base64" in data:
                            audio_data = data["audio_base64"] 
                        else:
                            # Try to get from raw message
                            audio_data = message.get("bytes", message.get("text", "")) 
                    else:  # audio_binary
                        audio_data = message.get("bytes", "")  # Use binary message
                    
                    # Transcribe audio
                    transcription = await transcribe_audio(audio_data, client_id)
                    
                    # Send transcription result
                    if "error" in transcription:
                        await safe_send_json(websocket, {
                            "type": "error", 
                            "message": f"Error processing audio: {transcription['error']}"
                        })
                    else:
                        # Send transcription result
                        await safe_send_json(websocket, {
                            "type": "transcription",
                            "text": transcription["text"],
                            "confidence": transcription.get("confidence", 0.0),
                            "processing_time": transcription.get("processing_time", 0.0)
                        })
                        
                        # Process semantic data if HPC client is available and we have text
                        if hpc_client and transcription["text"].strip():
                            try:
                                # Send status message first
                                await safe_send_json(websocket, {"type": "status", "message": "Processing semantic data..."})
                                
                                # Process in a separate task to avoid blocking
                                # This will check if connection is still active before sending
                                semantic_task = asyncio.create_task(process_semantic_data(
                                    websocket, transcription["text"], client_id))
                            except Exception as e:
                                logger.error(f"Error scheduling semantic processing: {e}")
                                # Continue processing even if semantic processing fails
                
                elif msg_type == "text":
                    # Handle text messages (for chat)
                    text = data.get("text", "")
                    if text.strip():
                        # Store in history
                        if client_id not in transcription_history:
                            transcription_history[client_id] = []
                        transcription_history[client_id].append({
                            "role": "user",
                            "text": text,
                            "timestamp": time.time()
                        })
                        
                        # Process with HPC if available
                        if hpc_client and text.strip():
                            await safe_send_json(websocket, {"type": "status", "message": "Processing message..."})
                            try:
                                semantic_task = asyncio.create_task(process_semantic_data(
                                    websocket, text, client_id))
                            except Exception as e:
                                logger.error(f"Error scheduling text semantic processing: {e}")
                
                else:
                    # Unknown message type
                    await safe_send_json(websocket, {
                        "type": "error", 
                        "message": f"Unknown message type: {msg_type}"
                    })
            
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {client_id}")
                break
            except ConnectionClosedOK:
                logger.info(f"WebSocket connection closed OK: {client_id}")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                logger.error(traceback.format_exc())
                try:
                    # Safely try to send an error
                    if "connection is closed" not in str(e).lower() and "websocket is closed" not in str(e).lower():
                        await safe_send_json(websocket, {"type": "error", "message": f"Server error: {str(e)}"})
                except Exception as send_err:
                    logger.error(f"Error sending error message: {send_err}")
                    # If we can't send messages, the connection is probably broken
                    break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected during setup: {client_id}")
    except ConnectionClosedOK:
        logger.info(f"WebSocket connection closed normally: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # Clean up connection
        if client_id in active_connections:
            try:
                # Try to close the connection if it's not already closed
                if hasattr(websocket, "client_state") and websocket.client_state and websocket.client_state.name != "DISCONNECTED":
                    await websocket.close()
            except Exception:
                # Ignore errors when closing an already closed connection
                pass
            
            # Remove from active connections
            del active_connections[client_id]
        
        logger.info(f"WebSocket connection cleanup complete for client {client_id}")

async def process_semantic_data(websocket, text, client_id):
    """Process semantic data using HPC client.
    
    Args:
        websocket: WebSocket connection
        text: Text to process
        client_id: Client ID
        
    Returns:
        None
    """
    if not hpc_client:
        # HPC client not available
        error_msg = "Semantic processing unavailable - no HPC client"
        logger.warning(error_msg)
        
        # Only try to send error if connection is still active
        if client_id in active_connections and websocket == active_connections[client_id]:
            try:
                await safe_send_json(websocket, {
                    "type": "semantic_error",
                    "message": error_msg
                })
            except Exception as e:
                logger.error(f"Error sending semantic error: {e}")
        return
    
    logger.info(f"Processing semantic data for text: {text[:50]}..." if len(text) > 50 else f"Processing semantic data for text: {text}")
    
    try:
        # Make sure HPC client is connected
        if not hpc_client.connected:
            logger.info("HPC client not connected, attempting to connect...")
            connection_result = await hpc_client.connect()
            if not connection_result:
                error_msg = "Could not connect to HPC server"
                logger.error(error_msg)
                if client_id in active_connections and websocket == active_connections[client_id]:
                    try:
                        await safe_send_json(websocket, {
                            "type": "semantic_error",
                            "message": error_msg
                        })
                    except Exception as e:
                        logger.error(f"Error sending semantic error: {e}")
                return
        
        # Get embedding
        logger.info("Getting embedding...")
        embedding_result = await hpc_client.process_embedding(text)
        
        if "error" in embedding_result:
            error_msg = f"Error processing embedding: {embedding_result['error']}"
            logger.error(error_msg)
            
            # Only try to send error if connection is still active
            if client_id in active_connections and websocket == active_connections[client_id]:
                try:
                    await safe_send_json(websocket, {
                        "type": "semantic_error",
                        "message": "Could not process semantic data",
                        "details": error_msg
                    })
                except Exception as e:
                    logger.error(f"Error sending semantic error: {e}")
            return
            
        # Get stats
        logger.info("Getting stats...")
        stats_result = await hpc_client.get_stats(embedding_result)
        
        if "error" in stats_result:
            error_msg = f"Error getting stats: {stats_result['error']}"
            logger.error(error_msg)
            
            # Only try to send error if connection is still active
            if client_id in active_connections and websocket == active_connections[client_id]:
                try:
                    await safe_send_json(websocket, {
                        "type": "semantic_error",
                        "message": "Could not process semantic data",
                        "details": error_msg
                    })
                except Exception as e:
                    logger.error(f"Error sending semantic error: {e}")
            return
            
        # Send results
        logger.info(f"Sending semantic results: significance = {stats_result.get('significance', 0.0)}")
        
        # Only try to send results if connection is still active
        if client_id in active_connections and websocket == active_connections[client_id]:
            try:
                # Send semantic results
                await safe_send_json(websocket, {
                    "type": "semantic_result",
                    "significance": stats_result.get("significance", 0.0) if stats_result else 0.0,
                    "text": text
                })
                logger.info("Semantic results sent successfully")
            except Exception as e:
                logger.error(f"Error sending semantic results: {e}")
    
    except Exception as e:
        logger.error(f"Error in process_semantic_data: {e}")
        logger.error(traceback.format_exc())
        
        # Only try to send error if connection is still active
        if client_id in active_connections and websocket == active_connections[client_id]:
            try:
                await safe_send_json(websocket, {
                    "type": "semantic_error",
                    "message": f"Error processing semantic data: {str(e)}"
                })
            except Exception as send_err:
                logger.error(f"Error sending semantic error: {send_err}")

@app.get("/transcription_history/{client_id}")
async def get_transcription_history(client_id: str, limit: int = 10):
    """Get transcription history for a specific client."""
    if client_id in transcription_history:
        history = transcription_history[client_id]
        return {"history": history[-limit:] if limit > 0 else history}
    else:
        return {"history": []}

async def transcribe_audio(audio_data, client_id=None):
    """Transcribe audio data using the loaded ASR model.
    
    Args:
        audio_data: Audio data, can be raw numpy array or various formats
        client_id: Optional client ID for tracking
        
    Returns:
        Dict with transcription text and metadata
    """
    start_time = time.time()
    
    try:
        # Make sure the model is loaded
        if model is None or not model_loaded:
            logger.error("ASR model not loaded")
            return {"text": "", "error": "ASR model not loaded", "confidence": 0.0}
        
        # Process different input formats
        audio_signal = None
        
        try:
            # Handle different audio data formats
            if isinstance(audio_data, dict) and "audio_data" in audio_data:
                # Extract from dict
                audio_data = audio_data["audio_data"]
                
            # Handle base64 audio data in data URI format
            if isinstance(audio_data, str) and audio_data.startswith("data:"):
                # Extract base64 part from data URI
                _, encoded = audio_data.split(",", 1)
                audio_bytes = base64.b64decode(encoded)
                audio_signal = preprocess_audio(audio_bytes)
            elif isinstance(audio_data, str):
                # Assume base64 encoded audio
                audio_bytes = base64.b64decode(audio_data)
                audio_signal = preprocess_audio(audio_bytes)
            elif isinstance(audio_data, dict) and "audio_base64" in audio_data:
                # Handle base64 encoded audio in a dict
                audio_bytes = base64.b64decode(audio_data["audio_base64"])
                audio_signal = preprocess_audio(audio_bytes)
            elif isinstance(audio_data, np.ndarray):
                # Handle numpy array directly
                audio_signal = audio_data.astype(np.float32)  # Ensure float32 type
            else:
                # Try to process as bytes
                audio_signal = preprocess_audio(audio_data)
                
            logger.info(f"Processed audio data with shape: {audio_signal.shape}")
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"text": "", "error": f"Error preprocessing audio: {str(e)}", "confidence": 0.0}
        
        # For NeMo MultiTask models, we need to handle the transcription differently
        try:
            # Model seems to need file input rather than raw tensors
            # For MultiTaskDecoding models, let's directly use the forward pass
            # and extract the predictions
            
            # Convert to tensor and move to device
            audio_tensor = torch.tensor(audio_signal).unsqueeze(0).to(DEVICE)
            audio_len = torch.tensor([audio_signal.shape[0]], device=DEVICE)
            
            # Perform transcription without using transcribe method
            with torch.no_grad():
                # Forward pass with the model
                model_output = model.forward(
                    input_signal=audio_tensor,
                    input_signal_length=audio_len
                )
                
                # Extract text based on model output structure
                logger.info(f"Model output type: {type(model_output)}")
                logger.info(f"Model output attributes: {dir(model_output)}")
                
                # Try common output patterns in NeMo ASR models
                text = ""
                
                # Check for common output patterns
                if hasattr(model_output, 'predictions') and model_output.predictions is not None:
                    preds = model_output.predictions
                    text = preds[0] if isinstance(preds, list) else preds
                    logger.info(f"Found text in predictions: {text}")
                elif hasattr(model_output, 'text'):
                    text_output = model_output.text
                    text = text_output[0] if isinstance(text_output, list) else text_output
                    logger.info(f"Found text in text attribute: {text}")
                elif hasattr(model_output, 'transcript'):
                    transcript = model_output.transcript
                    text = transcript[0] if isinstance(transcript, list) else transcript
                    logger.info(f"Found text in transcript attribute: {text}")
                elif hasattr(model_output, 'hypotheses'):
                    hyp = model_output.hypotheses
                    if isinstance(hyp, list) and len(hyp) > 0:
                        text = hyp[0]
                    logger.info(f"Found text in hypotheses: {text}")
                else:
                    # Try to access the log_probs and decode them
                    logger.info("No direct text output found, checking for decodable outputs")
                    
                    # Try common patterns in NeMo Canary models specifically
                    if hasattr(model_output, 'encoder_output') and hasattr(model_output, 'encoded_lengths'):
                        logger.info("Found encoder outputs, attempting direct decoding")
                        
                        # Try to use the model's internal decoding methods
                        if hasattr(model, 'decoding'):
                            try:
                                # Try different decoding methods
                                logger.info(f"Decoder type: {type(model.decoding)}")
                                logger.info(f"Decoder methods: {dir(model.decoding)}")
                                
                                if hasattr(model.decoding, 'decode'):
                                    text = model.decoding.decode(model_output)[0]
                                    logger.info(f"Decoded with decode(): {text}")
                            except Exception as decode_err:
                                logger.error(f"Error during decoding: {decode_err}")

            
            # If we still didn't get text, try simple greedy decoding
            if not text and hasattr(model_output, 'encoder_output'):
                logger.info("Attempting direct greedy decoding from log probs")
                try:
                    # Use the tokenizer directly if available
                    if hasattr(model, 'tokenizer'):
                        tokenizer = model.tokenizer
                        logger.info(f"Using model tokenizer: {type(tokenizer)}")
                        
                        # Get predictions (greedy)
                        log_probs = model_output.encoder_output
                        preds = log_probs.argmax(dim=-1)
                        text = tokenizer.ids_to_text(preds[0].cpu().numpy())
                        logger.info(f"Decoded with tokenizer: {text}")
                except Exception as tok_err:
                    logger.error(f"Tokenizer decoding error: {tok_err}")

                
            # If all else fails, try accessing the asr_model_output if it exists
            if not text and hasattr(model_output, 'asr_model_output'):
                logger.info("Trying to extract from asr_model_output")
                asr_output = model_output.asr_model_output
                logger.info(f"ASR output type: {type(asr_output)}")
                logger.info(f"ASR output attributes: {dir(asr_output)}")
                
                # Check if it has predictions
                if hasattr(asr_output, 'predictions'):
                    preds = asr_output.predictions
                    text = preds[0] if isinstance(preds, list) else preds
                    logger.info(f"Found text in ASR predictions: {text}")
                
            # If still no text, save audio to temp file and try the transcribe method
            if not text and hasattr(model, 'transcribe'):
                logger.info("Attempting transcription via file path method")
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    # Write the audio data to a temporary WAV file
                    sf.write(temp_path, audio_signal, 16000, 'PCM_16')
                    
                    # Try transcribing with the file path
                    try:
                        transcription = model.transcribe([temp_path])[0]
                        text = transcription
                        logger.info(f"Transcription from file: {text}")
                    except Exception as file_err:
                        logger.error(f"Error transcribing from file: {file_err}")
                    finally:
                        # Clean up the temporary file
                        import os
                        os.unlink(temp_path)

            # If nothing worked, log it
            if not text:
                logger.warning("Could not extract transcription text from any method")
                text = ""
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"text": "", "error": f"Error during transcription: {str(e)}", "confidence": 0.0}
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Transcription completed in {processing_time:.2f} seconds: '{text}'")
        
        # Save to history if client_id provided
        if client_id and text:
            if client_id not in transcription_history:
                transcription_history[client_id] = []
            transcription_history[client_id].append({
                "text": text,
                "timestamp": time.time(),
                "confidence": 0.95  # Placeholder as we don't have real confidence
            })
        
        return {
            "text": text,
            "confidence": 0.95,  # Placeholder for confidence score
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"text": "", "error": f"Error in transcription: {str(e)}", "confidence": 0.0}

async def safe_send_json(websocket, data):
    """Safely send JSON data through a WebSocket.
    
    Args:
        websocket: WebSocket connection
        data: Data to send as JSON
    
    Returns:
        bool: True if sent successfully, False otherwise
    """
    try:
        # Check if WebSocket is closed before attempting to send
        if hasattr(websocket, "client_state") and websocket.client_state and websocket.client_state.name == "DISCONNECTED":
            return False
            
        await websocket.send_json(data)
        return True
    except (WebSocketDisconnect, ConnectionClosedOK):
        # Client disconnected, this is normal
        return False
    except Exception as e:
        if "connection is closed" in str(e).lower() or "websocket is closed" in str(e).lower():
            # Connection closed errors are normal
            return False
        # Log unexpected errors
        logger.error(f"Error sending WebSocket message: {e}")
        return False

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="STT Server for Audio Transcription")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=5002, help="Port to bind the server to")
    args = parser.parse_args()
    
    # Get event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run initialization
    logger.info("Starting STT server...")
    try:
        loop.run_until_complete(initialize_stt_server())
        
        # Start the server
        logger.info(f"Starting web server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except Exception as e:
        logger.error(f"Error starting STT server: {e}")
        import traceback
        logger.error(traceback.format_exc())