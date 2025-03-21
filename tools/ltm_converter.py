#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTM Converter - Process and embed JSON files into long-term memory

This script converts all JSON files from LTM folders into embeddings and stores them
in the long-term memory system for inference, using the HPC-QR Flow Manager for optimization.
"""

import re
import os
import sys
import json
import time
import uuid
import torch
import asyncio
import logging
import hashlib
import datetime
import requests
import argparse
import numpy as np
import random
import ijson  # Use default backend instead of YAJL
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from server.tensor_server import TensorClient
import websockets  # Added for WebSocket support

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ltm_converter")

# Import HPC-QR Flow Manager
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager import HPCQRFlowManager
from tools.gpu_resource_manager import GPUStats

# Pydantic models for message validation
class MessageContent(BaseModel):
    """Model for message content with parts"""
    parts: List[Union[str, Dict[str, str]]] = Field(default_factory=list)
    
    def get_text(self) -> str:
        """Extract text from parts"""
        parts_content = []
        for part in self.parts:
            if isinstance(part, str):
                parts_content.append(part)
            elif isinstance(part, dict) and 'text' in part:
                parts_content.append(part['text'])
        return "\n".join(parts_content)

class MessageAuthor(BaseModel):
    """Model for message author"""
    role: str = "unknown"
    name: Optional[str] = None

class Message(BaseModel):
    """Model for a message within a conversation"""
    author: Optional[MessageAuthor] = None
    content: Optional[Union[str, MessageContent]] = None
    create_time: Optional[float] = None
    
    def get_role(self) -> str:
        """Get the role from the author or default to unknown"""
        if self.author and self.author.role:
            return self.author.role
        return "unknown"
    
    def get_content(self) -> str:
        """Get the content text"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, MessageContent):
            return self.content.get_text()
        return ""

class MessageItem(BaseModel):
    """Model for a message item in the mapping"""
    message: Optional[Message] = None
    
    def get_extracted_message(self) -> Dict[str, Any]:
        """Extract a standardized message dict"""
        if not self.message:
            return {}
            
        return {
            'role': self.message.get_role(),
            'content': self.message.get_content(),
            'timestamp': self.message.create_time or time.time()
        }

class ConversationMapping(BaseModel):
    """Model for conversation mapping structure"""
    mapping: Dict[str, MessageItem] = Field(default_factory=dict)
    title: Optional[str] = None
    create_time: Optional[float] = None
    
    def extract_messages(self) -> List[Dict[str, Any]]:
        """Extract all messages from the mapping"""
        messages = []
        
        # Add title as a system message if available
        if self.title:
            messages.append({
                'role': 'system',
                'content': f"Title: {self.title}",
                'timestamp': self.create_time or time.time()
            })
        
        # Extract messages from mapping
        for msg_id, msg_item in self.mapping.items():
            if msg_item.message:
                extracted = msg_item.get_extracted_message()
                if extracted and extracted.get('content'):
                    messages.append(extracted)
        
        return messages

class StandardMessage(BaseModel):
    """Model for standard message format"""
    role: str = "unknown"
    content: str = ""
    timestamp: Optional[float] = None

class EmotionClient:
    """Client for the Emotion Analyzer service to classify text emotions using WebSockets."""
    
    def __init__(self, url: str = 'ws://host.docker.internal:5007/ws'):
        self.url = url
        self.websocket = None
        self.connected = False
        logger.info(f"Initializing EmotionClient at {url}")
    
    async def connect(self):
        """Connect to the emotion analyzer service via WebSocket."""
        try:
            # Ensure URL uses ws:// protocol
            if self.url.startswith('http'):
                self.url = 'ws' + self.url[4:]
            
            # Different endpoints for emotion analyzer
            if not self.url.endswith('/ws') and not self.url.endswith('/analyze_emotion'):
                # First try /ws endpoint
                ws_url = self.url.rstrip('/') + '/ws'
                logger.info(f"Trying to connect to emotion analyzer at {ws_url}")
                
                try:
                    self.websocket = await websockets.connect(ws_url)
                    self.url = ws_url
                    self.connected = True
                    logger.info(f"Connected to Emotion Analyzer at {self.url}")
                    return True
                except Exception as ws_error:
                    logger.warning(f"Failed to connect to /ws endpoint: {str(ws_error)}")
                    
                    # Try /analyze_emotion endpoint
                    analyze_url = self.url.rstrip('/') + '/analyze_emotion'
                    logger.info(f"Trying alternate endpoint: {analyze_url}")
                    try:
                        self.websocket = await websockets.connect(analyze_url)
                        self.url = analyze_url
                        self.connected = True
                        logger.info(f"Connected to Emotion Analyzer at {self.url}")
                        return True
                    except Exception as analyze_error:
                        logger.error(f"Failed to connect to /analyze_emotion endpoint: {str(analyze_error)}")
                        raise
            else:
                # Use the provided URL as is
                logger.info(f"Connecting to emotion analyzer at {self.url}")
                self.websocket = await websockets.connect(self.url)
                self.connected = True
                logger.info(f"Connected to Emotion Analyzer at {self.url}")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to Emotion Analyzer: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the emotion analyzer service."""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("Disconnected from emotion analyzer service")
            except Exception as e:
                logger.error(f"Error disconnecting from emotion analyzer: {str(e)}")
            finally:
                self.websocket = None
                self.connected = False
    
    async def get_emotion(self, text: str) -> dict:
        """Get emotion classification for text via WebSocket."""
        if not self.connected:
            await self.connect()
        
        if not self.connected:
            logger.warning("Emotion analyzer service not available")
            return {'emotions': {}, 'dominant_emotion': 'unknown'}
        
        try:
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text)
                
            # Handle empty text
            if not text.strip():
                return {'emotions': {}, 'dominant_emotion': 'neutral'}
                
            # Create message payload with the CORRECT format per documentation
            message = json.dumps({
                'type': 'analyze',  # Correct type according to EMOTION_ANALYZER_README.md
                'text': text
            })
            
            logger.debug(f"Sending emotion analysis request for text: {text[:50]}...")
            
            # Send the message via WebSocket
            await self.websocket.send(message)
            
            # Wait for the response
            response_text = await self.websocket.recv()
            logger.debug(f"Received raw emotion response: {response_text}")
            
            response = json.loads(response_text)
            logger.info(f"Parsed emotion response: {json.dumps(response)}")
            
            # Process the response according to the documented format
            if 'type' in response and response['type'] == 'analysis_result':
                # This is the expected response format per documentation
                primary_emotions = response.get('primary_emotions', {})
                dominant_primary = response.get('dominant_primary', {})
                
                return {
                    'emotions': primary_emotions,
                    'dominant_emotion': dominant_primary.get('emotion', 'unknown')
                }
            elif 'error' in response:
                logger.warning(f"Error from emotion analyzer: {response['error']}")
                return {'emotions': {}, 'dominant_emotion': 'unknown'}
            else:
                # Try to extract emotion data from any response format
                return self._extract_emotion_data(response)
                
        except Exception as e:
            logger.error(f"Error getting emotion classification: {str(e)}")
            return {'emotions': {}, 'dominant_emotion': 'unknown'}
    
    def _extract_emotion_data(self, response):
        """Extract emotion data from various response formats"""
        # Check for correct response format
        if 'emotions' in response and 'dominant_emotion' in response:
            return {
                'emotions': response['emotions'],
                'dominant_emotion': response['dominant_emotion']
            }
            
        # Try to adapt to different response formats
        if 'result' in response and isinstance(response['result'], dict):
            # Extract from result field
            result = response['result']
            if 'emotions' in result or 'dominant_emotion' in result:
                logger.info(f"Found emotions in result field")
                return {
                    'emotions': result.get('emotions', {}),
                    'dominant_emotion': result.get('dominant_emotion', 'unknown')
                }
        
        # Handle other possible response formats
        emotions = {}
        dominant_emotion = 'unknown'
        
        # Look for any field that might contain emotion data
        for key, value in response.items():
            if isinstance(value, dict) and any(emotion in value for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise']):
                emotions = value
                if emotions:
                    dominant_max = max(emotions.items(), key=lambda x: x[1])
                    dominant_emotion = dominant_max[0]
                break
        
        return {'emotions': emotions, 'dominant_emotion': dominant_emotion}

class LTMConverter:
    """Convert and embed JSON files into long-term memory for inference"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LTM converter"""
        # Set up configurations
        self.ltm_path = Path(config.get('ltm_path', 'memory/stored/ltm'))
        self.output_path = Path(config.get('output_path', 'memory/indexed'))
        self.batch_size = config.get('batch_size', 32)
        self.max_files = config.get('max_files', 0)  # 0 means no limit
        self.embedding_dim = config.get('embedding_dim', 384)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hpcqr_config = {
            'dynamic_scaling_factor': config.get('dynamic_scaling_factor', 0.75),
            'update_threshold': config.get('update_threshold', 0.15),
            'drift_threshold': config.get('drift_threshold', 0.05),
            'embedding_dim': self.embedding_dim
        }
        
        # Initialize the flow manager
        self.flow_manager = HPCQRFlowManager({
            'device': self.device,
            'embedding_dim': self.embedding_dim,
            **self.hpcqr_config
        })
        
        # Initialize TensorClient for embedding service
        self.embedding_client = TensorClient(url=config.get('tensor_server_url', 'ws://host.docker.internal:5001'))
        self.embedding_client_initialized = False
        
        # Initialize EmotionClient for emotion analysis
        self.emotion_client = EmotionClient(url=config.get('emotion_server_url', 'ws://host.docker.internal:5007/ws'))
        self.emotion_client_initialized = False
        
        # Track statistics
        self.stats = {
            'start_time': time.time(),
            'end_time': 0,
            'total_time': 0,
            'total_files_processed': 0,
            'total_messages_processed': 0,
            'total_embeddings_generated': 0,
            'total_memory_indexed': 0,
            'files_by_folder': {},
            'errors': [],
            'warnings': []
        }
        
        logger.info(f"Initialized LTM Converter with config: {config}")
    
    async def setup(self):
        """Set up necessary components like the tensor server and vector store"""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "embeddings"), exist_ok=True)
        
        logger.info("Setup completed successfully")
    
    async def close(self):
        """Clean up resources"""
        # Close the tensor client connection if initialized
        if hasattr(self, 'embedding_client') and self.embedding_client_initialized:
            await self.embedding_client.disconnect()
            logger.info("Disconnected from embedding service")
        
        # Close the flow manager
        if hasattr(self, 'flow_manager'):
            await self.flow_manager.close()

    async def _initialize_embedding_client(self, max_retries=2, retry_delay=1):
        """Initialize the embedding client connection
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            bool: True if connected successfully, False otherwise
        """
        if self.embedding_client_initialized:
            return True
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Connecting to embedding service (attempt {attempt + 1}/{max_retries + 1})")
                success = await self.embedding_client.connect()
                if success:
                    self.embedding_client_initialized = True
                    logger.info("Connected to embedding service successfully")
                    return True
                else:
                    logger.warning(f"Failed to connect to embedding service (attempt {attempt + 1}/{max_retries + 1})")
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Error connecting to embedding service: {str(e)}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds after error...")
                    await asyncio.sleep(retry_delay)
        
        logger.error("All connection attempts to embedding service failed")
        return False

    async def _initialize_emotion_client(self, max_retries=3, retry_delay=1):
        """
        Initialize the emotion client connection
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            bool: True if connected successfully, False otherwise
        """
        logger.info("Connecting to emotion analyzer service")
        
        for attempt in range(1, max_retries + 1):
            logger.info(f"Connecting to emotion analyzer service (attempt {attempt}/{max_retries})")
            success = await self.emotion_client.connect()
            if success:
                logger.info("Connected to emotion analyzer service successfully")
                self.emotion_client_initialized = True
                # Test the connection by sending a simple test message with correct format
                try:
                    # Use the correct message format as per EMOTION_ANALYZER_README.md
                    test_message = "Test message for emotion analysis"
                    test_result = await self.emotion_client.get_emotion(test_message)
                    logger.info(f"Test emotion analysis for '{test_message}' result: {test_result}")
                    
                    # Check if we got valid emotions back
                    if test_result.get('emotions') or test_result.get('dominant_emotion') != 'unknown':
                        logger.info("Emotion analyzer test successful!")
                        return True
                    else:
                        logger.warning("Connected to emotion analyzer but received empty or default results")
                        # Connection works but returns default values, try next attempt
                        self.emotion_client_initialized = False
                except Exception as e:
                    logger.error(f"Error in test call to emotion analyzer: {str(e)}")
                    # Connection worked but test failed, continue with next attempt
                    self.emotion_client_initialized = False
            else:
                logger.warning(f"Failed to connect to emotion analyzer service (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
        
        logger.error("All connection attempts to emotion analyzer service failed")
        logger.warning("Emotion analyzer service not available, will skip emotion analysis.")
        return False

    def _extract_messages_from_json(self, json_file: Path) -> List[Dict[str, Any]]:
        """Extract messages from a JSON file using streaming parser and schema validation"""
        try:
            # First, quickly determine the JSON structure type without loading the full file
            with open(json_file, 'r', encoding='utf-8') as f:
                # Read a small chunk to determine the structure
                first_char = f.read(1).strip()
                f.seek(0)  # Reset file position
                
                # Skip empty files
                if not first_char:
                    return []
                
                if first_char == '[':
                    # It's a list - try to process it as an array of messages or mapping objects
                    return self._process_list_json(json_file)
                elif first_char == '{':
                    # It's an object - process based on its keys
                    return self._process_dict_json(json_file)
                else:
                    logger.warning(f"Unexpected JSON structure in {json_file}")
                    return []
        except Exception as e:
            logger.error(f"Error extracting messages from {json_file}: {e}")
            return []
    
    def _process_list_json(self, json_file: Path) -> List[Dict[str, Any]]:
        """Process a JSON file that contains a list structure"""
        messages = []
        try:
            # First check if it's a simple list of standard messages
            with open(json_file, 'r', encoding='utf-8') as f:
                for prefix, event, value in ijson.parse(f):
                    # Look for the first few items to determine structure
                    if prefix.endswith('.mapping') and event == 'map_key':
                        # This is likely a list of conversation objects with mapping
                        f.seek(0)  # Reset position
                        return self._process_conversation_list(json_file)
                    
                    # If we've checked enough items, assume it's a standard message list
                    if prefix.split('.')[0] == '2':  # Checked first 3 items (0,1,2)
                        break
            
            # Process as standard message list
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)  # For small arrays, direct loading is fine
                
                for item in data:
                    try:
                        # Validate using Pydantic
                        if 'content' in item and isinstance(item.get('content'), str):
                            msg = StandardMessage(
                                role=item.get('role', 'unknown'),
                                content=item.get('content', ''),
                                timestamp=item.get('timestamp', time.time())
                            )
                            
                            if msg.content:  # Only add if there's actual content
                                messages.append({
                                    'role': msg.role,
                                    'content': msg.content,
                                    'metadata': {
                                        'source': str(json_file),
                                        'timestamp': msg.timestamp or time.time(),
                                        'folder': json_file.parent.name
                                    }
                                })
                    except Exception as e:
                        logger.debug(f"Error processing message item: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error processing list JSON {json_file}: {e}")
        
        return messages
    
    def _process_conversation_list(self, json_file: Path) -> List[Dict[str, Any]]:
        """Process a JSON file that contains a list of conversation objects with mapping"""
        all_messages = []
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)  # Load all for mapping structure
                
                for conversation in data:
                    try:
                        if 'mapping' in conversation and isinstance(conversation['mapping'], dict):
                            # Use Pydantic model
                            conv_model = ConversationMapping(
                                mapping=conversation['mapping'],
                                title=conversation.get('title'),
                                create_time=conversation.get('create_time')
                            )
                            
                            # Extract messages
                            extracted_messages = conv_model.extract_messages()
                            
                            # Add source metadata
                            for msg in extracted_messages:
                                msg['metadata'] = {
                                    'source': str(json_file),
                                    'timestamp': msg.get('timestamp', time.time()),
                                    'folder': json_file.parent.name
                                }
                            
                            all_messages.extend(extracted_messages)
                    except Exception as e:
                        logger.debug(f"Error processing conversation item: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error processing conversation list JSON {json_file}: {e}")
        
        return all_messages
    
    def _process_dict_json(self, json_file: Path) -> List[Dict[str, Any]]:
        """Process a JSON file that contains a dictionary structure"""
        try:
            # Determine the type of dictionary by checking top-level keys
            top_level_keys = set()
            with open(json_file, 'r', encoding='utf-8') as f:
                # Just get the top-level keys
                for prefix, event, value in ijson.parse(f):
                    if event == 'map_key' and '.' not in prefix:
                        top_level_keys.add(value)
                    
                    # Once we have enough keys to determine type, stop parsing
                    if len(top_level_keys) >= 5 or ('mapping' in top_level_keys):
                        break
            
            # Process based on detected keys
            if 'mapping' in top_level_keys:
                # It's a conversation with mapping
                return self._process_mapping_json(json_file)
            elif 'messages' in top_level_keys:
                # It has a messages array
                return self._process_messages_dict(json_file)
            elif 'conversations' in top_level_keys:
                # It has nested conversations
                return self._process_conversations_dict(json_file)
            elif 'content' in top_level_keys:
                # It has direct content
                return self._process_content_dict(json_file)
            else:
                logger.warning(f"Unknown dictionary structure in {json_file} with keys: {top_level_keys}")
                return []
                
        except Exception as e:
            logger.error(f"Error processing dict JSON {json_file}: {e}")
            return []
    
    def _process_mapping_json(self, json_file: Path) -> List[Dict[str, Any]]:
        """Process a JSON file with a mapping structure"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)  # Load full data for mapping
                
                # Validate with Pydantic
                conv_model = ConversationMapping(
                    mapping=data.get('mapping', {}),
                    title=data.get('title'),
                    create_time=data.get('create_time')
                )
                
                # Extract messages
                messages = conv_model.extract_messages()
                
                # Add source metadata
                for msg in messages:
                    msg['metadata'] = {
                        'source': str(json_file),
                        'timestamp': msg.get('timestamp', time.time()),
                        'folder': json_file.parent.name
                    }
                
                return messages
        except Exception as e:
            logger.error(f"Error processing mapping JSON {json_file}: {e}")
            return []
    
    def _process_messages_dict(self, json_file: Path) -> List[Dict[str, Any]]:
        """Process a JSON file with a messages field"""
        messages = []
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                # Stream the messages array elements
                prefix_path = 'messages'
                current_message = {}
                
                for prefix, event, value in ijson.parse(f):
                    if prefix.startswith(prefix_path):
                        # Extract message data
                        if prefix.endswith('.role') and event == 'string':
                            current_message['role'] = value
                        elif prefix.endswith('.content') and event == 'string':
                            current_message['content'] = value
                        elif prefix.endswith('.timestamp') and event in ('number', 'string'):
                            try:
                                current_message['timestamp'] = float(value)
                            except:
                                current_message['timestamp'] = time.time()
                        # End of an item in the array
                        elif event == 'end_map':
                            if prefix.count('.') == 1:  # Direct child of messages array
                                if 'content' in current_message and current_message['content']:
                                    # Add metadata
                                    current_message['metadata'] = {
                                        'source': str(json_file),
                                        'timestamp': current_message.get('timestamp', time.time()),
                                        'folder': json_file.parent.name
                                    }
                                    messages.append(current_message)
                                # Reset for next message
                                current_message = {}
            
            return messages
                
        except Exception as e:
            logger.error(f"Error processing messages dict JSON {json_file}: {e}")
            return []
    
    def _process_conversations_dict(self, json_file: Path) -> List[Dict[str, Any]]:
        """Process a JSON file with nested conversations"""
        all_messages = []
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)  # Load all for nested structure
                
                if 'conversations' in data and isinstance(data['conversations'], list):
                    for conv in data['conversations']:
                        if 'messages' in conv and isinstance(conv['messages'], list):
                            for msg in conv['messages']:
                                if isinstance(msg, dict) and 'content' in msg and msg['content']:
                                    processed_msg = {
                                        'role': msg.get('role', 'unknown'),
                                        'content': msg['content'],
                                        'metadata': {
                                            'source': str(json_file),
                                            'timestamp': msg.get('timestamp', time.time()),
                                            'folder': json_file.parent.name
                                        }
                                    }
                                    all_messages.append(processed_msg)
            
            return all_messages
                
        except Exception as e:
            logger.error(f"Error processing conversations dict JSON {json_file}: {e}")
            return []
    
    def _process_content_dict(self, json_file: Path) -> List[Dict[str, Any]]:
        """Process a JSON file with direct content field"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                # Stream just to get the content field
                content = None
                title = None
                
                for prefix, event, value in ijson.parse(f):
                    if prefix == 'content' and event in ('string', 'map_key'):
                        content = value
                    elif prefix == 'title' and event == 'string':
                        title = value
                    
                    # If we have both content and title (or just content), we can stop
                    if content and (title is not None):
                        break
                
                if content:
                    # Create a synthetic message
                    message = {
                        'role': 'system',
                        'content': content if isinstance(content, str) else str(content),
                        'metadata': {
                            'source': str(json_file),
                            'timestamp': time.time(),
                            'folder': json_file.parent.name
                        }
                    }
                    
                    # Add title if available
                    if title:
                        message['metadata']['title'] = title
                    
                    return [message]
                    
            return []
                
        except Exception as e:
            logger.error(f"Error processing content dict JSON {json_file}: {e}")
            return []
    
    def _prepare_message_for_embedding(self, message: Dict[str, Any]) -> str:
        """Prepare a message for embedding by formatting it appropriately"""
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        # Handle dictionary content by converting to JSON string
        if isinstance(content, dict):
            content = json.dumps(content)
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)
        
        # Format based on role for better embedding context
        if role.lower() in ['system']:
            formatted = f"SYSTEM: {content}"
        elif role.lower() in ['assistant', 'ai', 'lucidia']:
            formatted = f"ASSISTANT: {content}"
        elif role.lower() in ['user', 'human']:
            formatted = f"USER: {content}"
        else:
            formatted = content
        
        return formatted
    
    def _save_embedding(self, doc_id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        """Save an embedding and its metadata to disk"""
        try:
            # Ensure the output directory exists
            embeddings_dir = os.path.join(self.output_path, "embeddings")
            os.makedirs(embeddings_dir, exist_ok=True)
            
            # Generate filename
            filename = os.path.join(embeddings_dir, f"{doc_id}.npz")
            
            # Ensure metadata is JSON serializable (convert any non-serializable items)
            serializable_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    serializable_metadata[key] = value
                else:
                    # Convert non-serializable objects to strings
                    serializable_metadata[key] = str(value)
            
            # Save the embedding and metadata
            np.savez_compressed(
                filename,
                embedding=embedding,
                metadata=json.dumps(serializable_metadata)
            )
            
            return filename
        except Exception as e:
            logger.error(f"Error saving embedding {doc_id}: {e}")
            return ""
    
    async def process_file(self, json_file: Path) -> Union[Tuple[int, List[str]], Dict[str, Any]]:
        """Process a single JSON file and return (message_count, document_ids) or an error dict"""
        try:
            # Extract messages from JSON
            messages = self._extract_messages_from_json(json_file)
            if not messages:
                logger.warning(f"No messages extracted from {json_file}")
                return 0, []
            
            logger.info(f"Extracted {len(messages)} messages from {json_file}")
            
            # Process in batches
            document_ids = []
            for i in range(0, len(messages), self.batch_size):
                batch_messages = messages[i:i+self.batch_size]
                # Ensure we have string texts - convert dictionaries to JSON strings
                batch_texts = []
                batch_emotions = []
                
                for msg in batch_messages:
                    content = msg.get('content', '')
                    # Convert dictionary content to JSON string
                    if isinstance(content, dict):
                        content = json.dumps(content)
                    # Ensure content is a string
                    if not isinstance(content, str):
                        content = str(content)
                    batch_texts.append(content)
                
                # Process emotion in parallel if emotion client is initialized
                batch_emotions = []
                if self.emotion_client_initialized:
                    try:
                        # First log what we're doing
                        logger.info(f"Getting emotions for {len(batch_texts)} texts")
                        
                        # Process emotions one by one to avoid WebSocket issues
                        for text in batch_texts:
                            try:
                                emotion_data = await self.emotion_client.get_emotion(text)
                                batch_emotions.append(emotion_data)
                                # Log the result for each text
                                text_preview = text[:30] + '...' if len(text) > 30 else text
                                if emotion_data and emotion_data.get('emotions'):
                                    logger.info(f"Emotion for '{text_preview}': {emotion_data.get('dominant_emotion')}")
                                else:
                                    logger.warning(f"No emotion data detected for '{text_preview}'")
                            except Exception as e:
                                logger.error(f"Error getting emotion for text: {str(e)}")
                                # Add default emotion data
                                batch_emotions.append({'emotions': {}, 'dominant_emotion': 'unknown'})
                    except Exception as e:
                        logger.error(f"Error processing emotions in batch: {str(e)}")
                        # Fill with default values
                        batch_emotions = [{'emotions': {}, 'dominant_emotion': 'unknown'} for _ in batch_texts]
                else:
                    # Use default values if emotion client not initialized
                    batch_emotions = [{'emotions': {}, 'dominant_emotion': 'unknown'} for _ in batch_texts]
                    logger.warning("Using default emotion values (emotion client not initialized)")
                
                batch_embeddings = []
                
                logger.debug(f"Generating embeddings for batch of {len(batch_texts)} messages")
                
                # Make sure the embedding client is connected
                if not await self._initialize_embedding_client():
                    logger.warning(f"Cannot connect to embedding service, using fallback embeddings for {json_file}")
                    # Continue with fallback embeddings rather than returning an error
                
                # Process each text in the batch to get embeddings from the service
                for text in batch_texts:
                    try:
                        # Get embedding from tensor service
                        if self.embedding_client_initialized:
                            embedding_result = await self.embedding_client.get_embedding(text)
                            
                            if 'error' in embedding_result:
                                logger.warning(f"Error getting embedding: {embedding_result['error']}, using fallback")
                                # Use fallback embedding
                                # Ensure text is hashable
                                safe_text = str(text) if not isinstance(text, str) else text
                                text_preview = safe_text[:50] + "..." if len(safe_text) > 50 else safe_text
                                logger.debug(f"Using fallback embedding for: {text_preview}")
                                text_hash = hash(safe_text) % 10000
                                np.random.seed(text_hash)
                                embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
                                # Normalize to unit length
                                embedding = embedding / np.linalg.norm(embedding)
                                # Convert to torch tensor
                                torch_embedding = torch.tensor(embedding, device=self.device)
                            else:
                                # Extract embedding from successful result
                                raw_embedding = embedding_result.get('embeddings', [])
                                if not raw_embedding:
                                    # Ensure text is hashable
                                    safe_text = str(text) if not isinstance(text, str) else text
                                    text_preview = safe_text[:50] + "..." if len(safe_text) > 50 else safe_text
                                    logger.warning(f"Empty embedding received for: {text_preview}")
                                    # Use fallback embedding
                                    text_hash = hash(safe_text) % 10000
                                    np.random.seed(text_hash)
                                    embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
                                    # Normalize to unit length
                                    embedding = embedding / np.linalg.norm(embedding)
                                    # Convert to torch tensor
                                    torch_embedding = torch.tensor(embedding, device=self.device)
                                else:    
                                    # Convert to torch tensor
                                    torch_embedding = torch.tensor(raw_embedding, device=self.device)
                        else:
                            # Use fallback embedding
                            # Ensure text is hashable
                            safe_text = str(text) if not isinstance(text, str) else text
                            text_preview = safe_text[:50] + "..." if len(safe_text) > 50 else safe_text
                            logger.debug(f"Using fallback embedding for: {text_preview}")
                            text_hash = hash(safe_text) % 10000
                            np.random.seed(text_hash)
                            embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
                            # Normalize to unit length
                            embedding = embedding / np.linalg.norm(embedding)
                            # Convert to torch tensor
                            torch_embedding = torch.tensor(embedding, device=self.device)
                            
                        batch_embeddings.append(torch_embedding)
                    except Exception as e:
                        logger.error(f"Failed to get embedding for text: {e}")
                        # Create fallback embedding and continue
                        try:
                            # Ensure text is hashable
                            safe_text = str(text) if not isinstance(text, str) else text
                            text_hash = hash(safe_text) % 10000
                            np.random.seed(text_hash)
                            embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
                            # Normalize to unit length
                            embedding = embedding / np.linalg.norm(embedding)
                            # Convert to torch tensor
                            torch_embedding = torch.tensor(embedding, device=self.device)
                            batch_embeddings.append(torch_embedding)
                        except Exception as inner_e:
                            logger.error(f"Failed to create fallback embedding: {inner_e}")
                            # Skip this item completely
                            continue
                
                if not batch_embeddings:
                    logger.warning(f"No embeddings generated for batch in file {json_file}")
                    continue
                
                # Save embeddings and metadata to disk
                for j, (msg, embedding, emotion_data) in enumerate(zip(batch_messages, batch_embeddings, batch_emotions)):
                    # Generate a unique document ID
                    doc_id = str(uuid.uuid4())
                    
                    # Log emotion data for debugging
                    text_preview = msg.get('content', '')[:50] + '...' if len(msg.get('content', '')) > 50 else msg.get('content', '')
                    if emotion_data and emotion_data.get('emotions'):
                        logger.info(f"Emotions for '{text_preview}': {emotion_data}")
                    else:
                        logger.warning(f"No emotion data for '{text_preview}'")
                    
                    # Prepare metadata
                    metadata = {
                        'text': self._prepare_message_for_embedding(msg),
                        'source': str(json_file),
                        'folder': json_file.parent.name,
                        'role': msg.get('role', 'unknown'),
                        'timestamp': msg.get('timestamp', time.time()),
                        'emotions': emotion_data.get('emotions', {}),
                        'dominant_emotion': emotion_data.get('dominant_emotion', 'unknown')
                    }
                    
                    # Add any additional metadata from the message
                    if 'metadata' in msg and isinstance(msg['metadata'], dict):
                        for k, v in msg['metadata'].items():
                            if k not in metadata:  # Don't overwrite existing metadata
                                metadata[k] = v
                
                    # Convert the embedding to a numpy array for storage
                    numpy_embedding = embedding.cpu().numpy()
                    
                    # Save to disk
                    self._save_embedding(doc_id, numpy_embedding, metadata)
                    document_ids.append(doc_id)
                    
            logger.info(f"Saved {len(document_ids)} document embeddings from {json_file}")
            return len(messages), document_ids
            
        except Exception as e:
            logger.error(f"Error processing file {json_file}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e), 'file': str(json_file)}
    
    async def process_folder(self, folder_path: Path) -> Dict[str, Any]:
        """Process all JSON files in a folder"""
        # Find all JSON files in the folder
        json_files = list(folder_path.glob("*.json"))
        json_files.extend(folder_path.glob("*.jsonl"))  # Also look for JSONL files
        
        if not json_files:
            logger.info(f"No JSON files found in {folder_path}")
            return {
                'name': folder_path.name,
                'path': str(folder_path),
                'files_processed': 0,
                'messages_processed': 0,
                'embeddings_generated': 0,
                'document_ids': [],
                'errors': []
            }
        
        # Limit the number of files if max_files is set
        if self.max_files > 0 and len(json_files) > self.max_files:
            logger.info(f"Limiting to {self.max_files} files in {folder_path}")
            json_files = json_files[:self.max_files]
        
        # Initialize statistics
        stats = {
            'name': folder_path.name,
            'path': str(folder_path),
            'files_processed': 0,
            'messages_processed': 0, 
            'embeddings_generated': 0,
            'document_ids': [],
            'errors': []
        }
        
        # Initialize progress bar
        progress = tqdm(total=len(json_files), desc=f"Processing {folder_path.name}")
        
        # Process each file
        for json_file in json_files:
            try:
                # Process the file
                logger.debug(f"Processing file: {json_file}")
                result = await self.process_file(json_file)
                
                # Handle different return types from process_file
                if isinstance(result, tuple) and len(result) == 2:
                    message_count, document_ids = result
                    # Update statistics
                    stats['files_processed'] += 1
                    stats['messages_processed'] += message_count
                    stats['embeddings_generated'] += len(document_ids)
                    stats['document_ids'].extend(document_ids)
                elif isinstance(result, dict) and 'status' in result and result['status'] == 'error':
                    # Handle error dict
                    logger.error(f"Error processing file {json_file}: {result['error']}")
                    stats['errors'].append({
                        'file': str(json_file),
                        'error': result.get('error', 'Unknown error')
                    })
                else:
                    # Unexpected return value
                    logger.error(f"Unexpected return value from process_file: {result}")
                    stats['errors'].append({
                        'file': str(json_file),
                        'error': f"Unexpected process_file result: {result}"
                    })
                
                # Update progress bar
                progress.update(1)
                progress.set_postfix({
                    'files': stats['files_processed'],
                    'msgs': stats['messages_processed'], 
                    'embs': stats['embeddings_generated']
                })
            except Exception as e:
                logger.error(f"Error processing file {json_file}: {e}", exc_info=True)
                stats['errors'].append({
                    'file': str(json_file),
                    'error': str(e)
                })
                progress.update(1)
        
        # Close progress bar
        progress.close()
        
        # Log completion
        logger.info(f"Processed {stats['files_processed']} files from {folder_path.name}, generated {stats['embeddings_generated']} embeddings")
        
        return stats
    
    async def run(self):
        """Run the LTM conversion process"""
        try:
            logger.info("Starting LTM conversion")
            self.stats['start_time'] = time.time()
            
            # Set up necessary components
            await self.setup()
            
            # Initialize the embedding client
            embedding_service_available = await self._initialize_embedding_client()
            if not embedding_service_available:
                logger.warning("Embedding service not available, will use fallback embeddings.")
                # Add warning to stats
                if 'warnings' not in self.stats:
                    self.stats['warnings'] = []
                self.stats['warnings'].append("Embedding service not available, using fallback random embeddings")
            else:
                logger.info("Using embedding service for generating embeddings.")
            
            # Initialize the emotion client
            emotion_service_available = await self._initialize_emotion_client()
            if not emotion_service_available:
                logger.warning("Emotion analyzer service not available, will skip emotion analysis.")
                # Add warning to stats
                if 'warnings' not in self.stats:
                    self.stats['warnings'] = []
                self.stats['warnings'].append("Emotion analyzer service not available, skipping emotion analysis")
            else:
                logger.info("Using emotion analyzer service for emotion analysis.")
            
            # Find all folders in the LTM path
            folders = [f for f in self.ltm_path.iterdir() if f.is_dir()]
            if not folders:
                logger.warning(f"No folders found in {self.ltm_path}")
                self.stats['warnings'] = self.stats.get('warnings', []) + [f"No folders found in {self.ltm_path}"]
                return
                
            # Process each folder
            logger.info(f"Found {len(folders)} folders to process")
            all_document_ids = []
            for folder in folders:
                logger.info(f"Processing folder: {folder.name}")
                folder_stats = await self.process_folder(folder)
                
                # Update statistics
                self.stats['total_files_processed'] += folder_stats['files_processed']
                self.stats['total_messages_processed'] += folder_stats['messages_processed']
                self.stats['total_embeddings_generated'] += folder_stats['embeddings_generated']
                self.stats['files_by_folder'][folder.name] = folder_stats
                all_document_ids.extend(folder_stats['document_ids'])
                
                # Track errors
                if 'errors' in folder_stats and folder_stats['errors']:
                    if 'errors' not in self.stats:
                        self.stats['errors'] = []
                    self.stats['errors'].extend(folder_stats['errors'])
            
            # Clean up
            await self.close()
            
            # Run validation if embeddings were created
            if all_document_ids:
                logger.info(f"Validating {min(20, len(all_document_ids))} sample embeddings...")
                validation_results = self.validate_embeddings(all_document_ids, sample_size=20)
                
                # Add validation results to stats
                self.stats["validation"] = validation_results
                
                # Output validation summary
                valid_percent = validation_results.get("valid_percent", 0)
                logger.info(f"Validation complete: {valid_percent:.1f}% of samples are valid")
                
                if valid_percent < 50:
                    logger.warning("Less than 50% of samples passed validation. Check the report for details.")
            else:
                logger.warning("No embeddings were created. Skipping validation.")
            
            # Update final statistics
            self.stats['end_time'] = time.time()
            self.stats['total_time'] = self.stats['end_time'] - self.stats['start_time']
            self.stats['total_memory_indexed'] = len(list(Path(os.path.join(self.output_path, "embeddings")).glob('*.npz')))
            
            # Save statistics
            stats_path = os.path.join(self.output_path, f"conversion_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, default=str)
            
            logger.info(f"Conversion complete. Processed {self.stats['total_files_processed']} files with {self.stats['total_messages_processed']} messages.")
            logger.info(f"Generated {self.stats['total_embeddings_generated']} embeddings in {self.stats['total_time']:.2f} seconds.")
            logger.info(f"Statistics saved to {stats_path}")
            
            return self.stats
        
        except Exception as e:
            logger.error(f"Error running LTM conversion: {e}", exc_info=True)
            return self.stats
    
    def validate_embeddings(self, document_ids: List[str], sample_size: int = 10) -> Dict[str, Any]:
        """Validate that the generated embeddings contain proper content"""
        # Sample a subset of document IDs if there are many
        if not document_ids:
            return {
                "status": "error",
                "message": "No document IDs provided for validation",
                "total": 0,
                "samples": []
            }
        
        # Sample documents to validate (or all if fewer than sample_size)
        sample_ids = document_ids
        if len(document_ids) > sample_size:
            sample_ids = random.sample(document_ids, sample_size)
        
        embeddings_dir = os.path.join(self.output_path, "embeddings")
        samples = []
        valid_count = 0
        invalid_count = 0
        missing_count = 0
        
        for doc_id in sample_ids:
            filename = os.path.join(embeddings_dir, f"{doc_id}.npz")
            if not os.path.exists(filename):
                missing_count += 1
                samples.append({
                    "id": doc_id,
                    "status": "missing",
                    "message": f"File not found: {filename}"
                })
                continue
                
            try:
                # Load the embedding file
                with np.load(filename) as data:
                    embedding = data['embedding']
                    metadata_str = data['metadata']
                    
                    # Parse the metadata JSON
                    metadata = json.loads(metadata_str)
                    
                    # Check if the embedding is valid
                    if not isinstance(embedding, np.ndarray) or embedding.size == 0:
                        invalid_count += 1
                        samples.append({
                            "id": doc_id,
                            "status": "invalid_embedding", 
                            "message": f"Invalid embedding shape: {embedding.shape}",
                            "source": metadata.get("source", "unknown")
                        })
                        continue
                        
                    # Check if the text content is present
                    if 'text' not in metadata or not metadata['text']:
                        invalid_count += 1
                        samples.append({
                            "id": doc_id,
                            "status": "missing_content",
                            "message": "No text content in metadata",
                            "source": metadata.get("source", "unknown")
                        })
                        continue
                    
                    # Sample is valid 
                    valid_count += 1
                    samples.append({
                        "id": doc_id,
                        "status": "valid",
                        "embedding_shape": embedding.shape,
                        "content_sample": metadata['text'][:100] + "..." if len(metadata['text']) > 100 else metadata['text'],
                        "source": metadata.get("source", "unknown"),
                        "role": metadata.get("role", "unknown")
                    })
            
            except Exception as e:
                invalid_count += 1
                samples.append({
                    "id": doc_id,
                    "status": "error",
                    "message": str(e),
                    "path": filename
                })
        
        # Compute overall statistics
        total = len(sample_ids)
        valid_percent = (valid_count / total) * 100 if total > 0 else 0
        
        logger.info(f"Validation results: {valid_count}/{total} valid embeddings ({valid_percent:.1f}%)")
        
        return {
            "status": "success" if valid_count > 0 else "warning",
            "total": total,
            "valid": valid_count,
            "invalid": invalid_count,
            "missing": missing_count,
            "valid_percent": valid_percent,
            "samples": samples
        }
    
    def generate_summary_report(self):
        """Generate a markdown summary report of the conversion process"""
        report_path = os.path.join(self.output_path, f"conversion_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# Memory Conversion Summary\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write overall statistics
            f.write(f"## Overall Statistics\n\n")
            f.write(f"- **Total Processing Time**: {self.stats['total_time']:.2f} seconds\n")
            f.write(f"- **Total Folders Processed**: {len(self.stats['files_by_folder'])}\n")
            f.write(f"- **Total Files Processed**: {self.stats['total_files_processed']}\n")
            f.write(f"- **Total Messages Processed**: {self.stats['total_messages_processed']}\n")
            f.write(f"- **Total Embeddings Generated**: {self.stats['total_embeddings_generated']}\n")
            
            # Write validation results if available
            if 'validation' in self.stats:
                validation = self.stats['validation']
                f.write(f"\n## Embedding Validation Results\n\n")
                f.write(f"- **Total Samples Validated**: {validation.get('total', 0)}\n")
                f.write(f"- **Valid Embeddings**: {validation.get('valid', 0)} ({validation.get('valid_percent', 0):.1f}%)\n")
                f.write(f"- **Invalid Embeddings**: {validation.get('invalid', 0)}\n")
                f.write(f"- **Missing Embeddings**: {validation.get('missing', 0)}\n")
                
                # Write sample validation details
                if 'samples' in validation and validation['samples']:
                    f.write(f"\n### Sample Validation Details\n\n")
                    
                    # Create a table of sample results
                    f.write(f"| ID | Status | Content Sample | Source | Role |\n")
                    f.write(f"|-----|--------|----------------|--------|------|\n")
                    
                    for sample in validation['samples']:
                        status = sample.get("status", "unknown")
                        content = sample.get("content_sample", sample.get("message", ""))
                        source = sample.get("source", "unknown")
                        role = sample.get("role", "unknown")
                        doc_id = sample.get("id", "unknown")
                        
                        # Format content for markdown table
                        content = content.replace("|", "\\|").replace("\n", " ")
                        if len(content) > 50:
                            content = content[:47] + "..."
                        
                        f.write(f"| {doc_id} | {status} | {content} | {source} | {role} |\n")
            
            # Write folder-specific statistics
            f.write(f"\n## Folder Statistics\n\n")
            
            # Create a table for folder statistics
            f.write(f"| Folder | Files Processed | Messages Processed | Embeddings Generated |\n")
            f.write(f"|--------|-----------------|-------------------|---------------------|\n")
            
            for folder_name, folder_stats in self.stats['files_by_folder'].items():
                f.write(f"| {folder_name} | {folder_stats['files_processed']} | {folder_stats['messages_processed']} | {folder_stats['embeddings_generated']} |\n")
            
            # Write performance statistics
            f.write(f"\n## Performance Metrics\n\n")
            total_time = self.stats['total_time']
            total_files = self.stats['total_files_processed']
            total_msgs = self.stats['total_messages_processed']
            total_embs = self.stats['total_embeddings_generated']
            
            # Calculate performance metrics
            files_per_sec = total_files / total_time if total_time > 0 else 0
            msgs_per_sec = total_msgs / total_time if total_time > 0 else 0
            embs_per_sec = total_embs / total_time if total_time > 0 else 0
            
            f.write(f"- **Files/sec**: {files_per_sec:.2f}\n")
            f.write(f"- **Messages/sec**: {msgs_per_sec:.2f}\n")
            f.write(f"- **Embeddings/sec**: {embs_per_sec:.2f}\n")
            
            # Write configuration
            f.write(f"\n## Configuration\n\n")
            f.write(f"```python\n{json.dumps(self.config, indent=2)}\n```\n")
            
            # Write any errors or warnings
            if 'errors' in self.stats and self.stats['errors']:
                f.write(f"\n## Errors and Warnings\n\n")
                for error in self.stats['errors']:
                    f.write(f"- **{error['type']}**: {error['message']}\n")
        
        logger.info(f"Generated summary report at {report_path}")
        return report_path


async def main(emotion_server_url=None):
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="LTM Converter for embedding and indexing JSON files")
    parser.add_argument(
        "--ltm-path", 
        type=str, 
        default="memory/stored/ltm",
        help="Path to LTM folders containing JSON files"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="memory/indexed",
        help="Path to store indexed memory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum number of files to process per folder (0 = no limit)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--dynamic-scaling-factor",
        type=float,
        default=0.75,
        help="Dynamic scaling factor for shock absorption"
    )
    parser.add_argument(
        "--update-threshold",
        type=float,
        default=0.15,
        help="Update threshold for shock absorption"
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.05,
        help="Drift threshold for embedding stability"
    )
    parser.add_argument(
        "--tensor-server-url",
        type=str,
        default="ws://host.docker.internal:5001",
        help="Tensor server URL for embedding service"
    )
    parser.add_argument(
        "--emotion-server-url",
        type=str,
        default="ws://host.docker.internal:5007/ws",
        help="Emotion server URL for emotion analysis"
    )
    
    args = parser.parse_args()
    
    # Override emotion_server_url if provided
    if emotion_server_url:
        args.emotion_server_url = emotion_server_url
        
    # Set up configuration
    config = {
        'ltm_path': args.ltm_path,
        'output_path': args.output_path,
        'batch_size': args.batch_size,
        'max_files': args.max_files,
        'embedding_dim': args.embedding_dim,
        'dynamic_scaling_factor': args.dynamic_scaling_factor,
        'update_threshold': args.update_threshold,
        'drift_threshold': args.drift_threshold,
        'tensor_server_url': args.tensor_server_url,
        'emotion_server_url': args.emotion_server_url
    }
    
    converter = LTMConverter(config)
    await converter.run()
    converter.generate_summary_report()


if __name__ == "__main__":
    # Add more URLs to try for emotion analyzer
    # The service might be accessible in different ways depending on the environment
    EMOTION_ANALYZER_URLS = [
        "ws://host.docker.internal:5007",      # Base URL as in the documentation
        "ws://localhost:5007",                # Direct localhost
        "ws://emotion-analyzer:5007"          # Docker service name
    ]
    
    # Try each URL until one works or we run out of options
    async def try_emotion_analyzer_urls():
        for url in EMOTION_ANALYZER_URLS:
            try:
                print(f"Trying emotion analyzer at {url}")
                client = EmotionClient(url=url)
                if await client.connect():
                    try:
                        # Test with a simple message using the correct format
                        test_message = "Test message for emotion analysis"
                        # Construct the message directly here for testing
                        ws_message = json.dumps({
                            'type': 'analyze',  # Correct type per documentation
                            'text': test_message
                        })
                        
                        # Send message directly
                        await client.websocket.send(ws_message)
                        response_text = await client.websocket.recv()
                        response = json.loads(response_text)
                        
                        print(f"Test response: {json.dumps(response, indent=2)}")
                        
                        # Check if response has valid format
                        if 'type' in response and response['type'] == 'analysis_result':
                            print(f"Success! Emotion analyzer working at {url}")
                            await client.disconnect()
                            return url
                        else:
                            print(f"Connected but unexpected response format at {url}")
                            await client.disconnect()
                    except Exception as e:
                        print(f"Connected but test failed at {url}: {str(e)}")
                        await client.disconnect()
            except Exception as e:
                print(f"Failed to connect to {url}: {str(e)}")
        return None
    
    async def main_with_best_url():
        # Try to find the best working URL
        best_url = await try_emotion_analyzer_urls()
        if best_url:
            print(f"Using emotion analyzer at {best_url}")
            # Run the main with the best URL
            await main(emotion_server_url=best_url)
        else:
            print("No working emotion analyzer found, proceeding without emotion analysis")
            await main()
    
    # Run with URL testing
    try:
        asyncio.run(main_with_best_url())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main program: {str(e)}")
