#!/usr/bin/env python3
"""
LM Studio Client for Lucidia Reflection CLI

Provides functionality to interact with a local LLM via LM Studio's API.
"""

import aiohttp
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("LucidiaReflectionCLI")

class LMStudioClient:
    """Client for interacting with LM Studio for local LLM inference."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the LM Studio client.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.lm_studio_url = self.config.get("lm_studio_url", "http://127.0.0.1:1234")
        self.session = None
    
    async def connect(self) -> bool:
        """Connect to LM Studio and verify it's running.
        
        Returns:
            bool: True if successfully connected, False otherwise
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Test connection to LM Studio
            async with self.session.get(f"{self.lm_studio_url}/v1/models") as response:
                if response.status == 200:
                    models = await response.json()
                    logger.info(f"Connected to LM Studio. Available models: {len(models)}")
                    return True
                else:
                    logger.error(f"Failed to connect to LM Studio: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error connecting to LM Studio: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from LM Studio."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Disconnected from LM Studio")
    
    async def generate_reflection(self, memories: List[Dict[str, Any]], 
                               depth: float = 0.7, 
                               creativity: float = 0.5,
                               max_tokens: int = 2000) -> Dict[str, Any]:
        """Generate reflection based on provided memories.
        
        Args:
            memories: List of memory objects to reflect on
            depth: Reflection depth (0.0-1.0)
            creativity: Creativity level (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dict containing the reflection results
        """
        try:
            if not self.session:
                await self.connect()
            
            # Format memories for the prompt
            memory_texts = []
            for i, memory in enumerate(memories):
                try:
                    # Handle different memory formats
                    if isinstance(memory, dict):
                        content = memory.get('content', f"Memory {i+1}")
                        significance = memory.get('significance', 0)
                        created_at = memory.get('created_at', '')
                    elif isinstance(memory, str):
                        content = memory
                        significance = 0.5
                        created_at = datetime.now().isoformat()
                    else:
                        logger.warning(f"Skipping unknown memory type: {type(memory)}")
                        continue
                        
                    memory_text = f"Memory {i+1}:\nContent: {content}\nSignificance: {significance}\nCreated: {created_at}"
                    memory_texts.append(memory_text)
                except Exception as e:
                    logger.error(f"Error formatting memory {i}: {e}")
                    continue
            
            memory_text = "\n\n".join(memory_texts) if memory_texts else "No memories available"
            
            # Build the reflection prompt
            system_prompt = """You are Lucidia's reflection system, a reflective AI that analyzes memories and generates structured dream reports. 
            Your task is to analyze the provided memories, identify patterns, and generate insights, questions, hypotheses, and counterfactuals.
            Your output should be in valid JSON format that follows the specified schema."""
            
            user_prompt = f"""Review these memories and generate insights, questions, hypotheses, and counterfactuals:

{memory_text}

Generate a structured dream report with insights, questions, hypotheses, and counterfactuals based on these memories.

Reflection depth: {depth} (higher means more philosophical and abstract)
Creativity level: {creativity} (higher means more novel and unexpected connections)"""
            
            # Calculate temperature based on creativity
            temperature = 0.7 + (creativity * 0.3)  # Maps 0.0-1.0 to 0.7-1.0
            
            # Call LM Studio API with JSON Schema
            payload = {
                "model": "local-model",  # Uses the currently loaded model
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dream_report",
                        "strict": "true",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "fragments": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "content": {"type": "string"},
                                            "type": {"type": "string", "enum": ["insight", "question", "hypothesis", "counterfactual"]},
                                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                        },
                                        "required": ["content", "type", "confidence"]
                                    }
                                }
                            },
                            "required": ["title", "fragments"]
                        }
                    }
                }
            }
            
            logger.info(f"Generating reflection with depth={depth}, creativity={creativity}...")
            
            try:
                async with self.session.post(
                    f"{self.lm_studio_url}/v1/chat/completions", 
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_msg = f"LM Studio returned error: {response.status}"
                        try:
                            error_body = await response.text()
                            error_msg += f" - Response body: {error_body}"
                        except Exception as e:
                            error_msg += f" (Failed to read error body: {e})"
                        logger.error(error_msg)
                        return {"status": "error", "message": error_msg}
                    
                    result = await response.json()
                    
                    # Extract content from the response
                    choices = result.get("choices", [])
                    if not choices:
                        error_msg = "No choices returned from LLM"
                        logger.error(error_msg)
                        return {"status": "error", "message": error_msg}
                        
                    message = choices[0].get("message", {})
                    if not message:
                        error_msg = "No message in LLM response"
                        logger.error(error_msg)
                        return {"status": "error", "message": error_msg}
                        
                    content = message.get("content", "")
                    if not content:
                        error_msg = "Empty content in LLM response"
                        logger.error(error_msg)
                        return {"status": "error", "message": error_msg}
                    
                    try:
                        # Parse the JSON content
                        dream_data = json.loads(content)
                        
                        # Validate the JSON schema
                        if not isinstance(dream_data, dict):
                            error_msg = f"Invalid JSON schema: not an object: {type(dream_data)}"
                            logger.error(error_msg)
                            return {"status": "error", "message": error_msg}
                            
                        if "title" not in dream_data:
                            error_msg = "Invalid JSON schema: missing 'title'"
                            logger.error(error_msg)
                            return {"status": "error", "message": error_msg}
                            
                        if "fragments" not in dream_data:
                            error_msg = "Invalid JSON schema: missing 'fragments'"
                            logger.error(error_msg)
                            return {"status": "error", "message": error_msg}
                            
                        if not isinstance(dream_data["fragments"], list):
                            error_msg = f"Invalid JSON schema: 'fragments' is not a list: {type(dream_data['fragments'])}"
                            logger.error(error_msg)
                            return {"status": "error", "message": error_msg}
                        
                        # Create the reflection result
                        reflection_result = {
                            "status": "success",
                            "title": dream_data.get("title", "Untitled Reflection"),
                            "fragments": dream_data.get("fragments", []),
                            "metadata": {
                                "depth": depth,
                                "creativity": creativity,
                                "temperature": temperature,
                                "memory_count": len(memories)
                            }
                        }
                        
                        logger.info(f"Generated reflection with {len(reflection_result['fragments'])} fragments")
                        return reflection_result
                        
                    except json.JSONDecodeError as e:
                        error_msg = f"Failed to parse JSON from LLM response: {e}"
                        logger.error(error_msg)
                        return {"status": "error", "message": error_msg}
            except aiohttp.ClientError as e:
                error_msg = f"HTTP error when calling LM Studio: {e}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
                
        except Exception as e:
            error_msg = f"Error generating reflection: {e}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    async def evaluate_significance(self, memory_content: str) -> float:
        """Evaluate the significance of a memory using the LLM.
        
        Args:
            memory_content: The content of the memory to evaluate
            
        Returns:
            float: Significance score (0.0-1.0)
        """
        try:
            if not self.session:
                await self.connect()
            
            system_prompt = """You are an expert at evaluating the significance of memories. 
            Your task is to analyze a memory and assign it a significance score from 0.0 to 1.0,
            where 0.0 is completely insignificant and 1.0 is extremely significant.
            Consider factors like emotional impact, uniqueness, potential long-term relevance,
            and connection to core values or goals."""
            
            user_prompt = f"""Evaluate the significance of this memory on a scale from 0.0 to 1.0:

{memory_content}

Provide only a single number as your response, with no additional text."""
            
            payload = {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,  # Low temperature for more consistent scoring
                "max_tokens": 10
            }
            
            async with self.session.post(
                f"{self.lm_studio_url}/v1/chat/completions", 
                json=payload
            ) as response:
                if response.status != 200:
                    logger.error(f"LM Studio returned error: {response.status}")
                    return 0.5  # Default mid-range value
                
                result = await response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "0.5")
                
                # Extract the numeric value
                try:
                    score = float(content.strip())
                    # Ensure it's in the valid range
                    score = max(0.0, min(1.0, score))
                    return score
                except ValueError:
                    logger.warning(f"Failed to parse significance score: {content}")
                    return 0.5
                
        except Exception as e:
            logger.error(f"Error evaluating significance: {e}")
            return 0.5
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get the list of available models from LM Studio.
        
        Returns:
            List of model information dictionaries
        """
        try:
            if not self.session:
                await self.connect()
            
            async with self.session.get(f"{self.lm_studio_url}/v1/models") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get models: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
