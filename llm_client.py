#!/usr/bin/env python3
"""
LM Studio Client for Lucidia Reflection CLI

Provides functionality to interact with a local LLM via LM Studio's API.
"""

import aiohttp
import json
import logging
import asyncio
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime
import uuid
from datetime import timedelta

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
    
    async def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response from LM Studio API using the given payload.
        
        Args:
            payload: The request payload to send to LM Studio API
            
        Returns:
            Dict containing the response from LM Studio
        """
        try:
            if not self.session:
                await self.connect()
            
            logger.debug(f"Sending request to LM Studio with payload: {payload}")
            
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
                
                return {"status": "success", "response": content}
        except Exception as e:
            logger.error(f"Error generating response from LM Studio: {e}")
            return {"status": "error", "message": f"Error generating response: {e}"}

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
    
    # Contextual Tool Detection functionality
    
    async def process_with_directive_detection(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user message with directive detection and tool execution.
        
        This method detects if the user's input contains directives that should trigger
        specific tool calls, and then processes those directives before standard LLM processing.
        
        Args:
            user_input: The user's message text
            context: Additional context for processing
            
        Returns:
            Dict containing the response and any executed tool information
        """
        # Initialize response structure
        response = {
            "status": "success",
            "response": "",
            "directives_detected": False,
            "tools_executed": []
        }
        
        # Detect directives in the user input
        directives = self._detect_directives(user_input)
        
        # If directives were detected, execute the corresponding tools
        if directives:
            response["directives_detected"] = True
            for tool_name, params in directives:
                try:
                    logger.info(f"Executing tool '{tool_name}' based on detected directive")
                    tool_result = await self._execute_tool(tool_name, params)
                    response["tools_executed"].append({
                        "tool": tool_name,
                        "params": params,
                        "result": tool_result
                    })
                except Exception as e:
                    logger.error(f"Error executing tool '{tool_name}': {e}")
                    response["tools_executed"].append({
                        "tool": tool_name,
                        "params": params,
                        "error": str(e)
                    })
        
        # Generate LLM response with tool results injected
        if context is None:
            context = {}
            
        # Add tool execution results to the context
        if response["tools_executed"]:
            context["tool_results"] = response["tools_executed"]
            
        # Generate the LLM response
        llm_payload = self._build_payload_with_directive_context(user_input, context, response["tools_executed"])
        llm_response = await self.generate(llm_payload)
        
        if llm_response["status"] == "success":
            # Inject any prefix based on executed tools
            response_prefix = self._create_response_prefix(response["tools_executed"])
            response["response"] = response_prefix + llm_response["response"]
        else:
            # If LLM generation failed, just pass through the error
            response["status"] = "error"
            response["response"] = llm_response["message"]
            
        return response
    
    def _detect_directives(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Detect directives in the given text.
        
        Args:
            text: User input text to analyze
            
        Returns:
            List of (tool_name, parameters) tuples for matched directives
        """
        # List of directive patterns in the format:
        # (tool_name, [regex_patterns], parameter_extractor_function, default_params, priority)
        directive_patterns = [
            # Dream cycle directive
            ("generate_dream_cycle", 
             [r"(?:initiate|start|begin|run|execute)\s+(?:a\s+)?dream(?:ing)?\s+cycle",
              r"(?:make|let)\s+(?:lucidia\s+)?dream\s+(?:about|on)"],
             lambda text: {},
             {"time_budget": 180},
             10),
             
            # Memory search directive
            ("search_memories",
             [r"(?:find|search|look\s+for|retrieve)\s+(?:memories|memory)\s+(?:about|related\s+to|containing)\s+(.+)",
              r"(?:what\s+do\s+you\s+remember\s+about)\s+(.+)"],
             lambda text: {"query": re.search(r"(?:about|related\s+to|containing|remember\s+about)\s+(.+)(?:\?|\.|$)", text).group(1) if re.search(r"(?:about|related\s+to|containing|remember\s+about)\s+(.+)(?:\?|\.|$)", text) else ""},
             {"limit": 5},
             8),
             
            # Self-reflection directive
            ("generate_self_reflection",
             [r"(?:reflect|think)\s+(?:about|on)\s+(?:yourself|your\s+(?:thoughts|behavior|actions|responses))",
              r"(?:perform|do)\s+(?:a\s+)?self[-\s]reflection"],
             lambda text: {},
             {"depth": 0.7},
             7),
             
            # Knowledge graph exploration directive
            ("explore_knowledge_graph",
             [r"(?:explore|examine|investigate|analyze)\s+(?:the\s+)?(?:knowledge\s+graph|concept\s+network)\s+(?:for|about|related\s+to)\s+(.+)",
              r"(?:how\s+are\s+concepts\s+related\s+to)\s+(.+)"],
             lambda text: {"concept": re.search(r"(?:for|about|related\s+to|related\s+to)\s+(.+)(?:\?|\.|$)", text).group(1) if re.search(r"(?:for|about|related\s+to|related\s+to)\s+(.+)(?:\?|\.|$)", text) else ""},
             {"depth": 2},
             6),
             
            # Generate insight directive
            ("generate_insight",
             [r"(?:generate|create|provide|give\s+me)\s+(?:an\s+)?insight(?:s)?\s+(?:about|on|for|related\s+to)\s+(.+)",
              r"(?:what\s+insights\s+do\s+you\s+have\s+(?:about|on|for))\s+(.+)"],
             lambda text: {"topic": re.search(r"(?:about|on|for|related\s+to)\s+(.+)(?:\?|\.|$)", text).group(1) if re.search(r"(?:about|on|for|related\s+to)\s+(.+)(?:\?|\.|$)", text) else ""},
             {},
             5)
        ]
        
        matched_directives = []
        
        # Sort directive patterns by priority (higher values first)
        sorted_patterns = sorted(directive_patterns, key=lambda x: x[4], reverse=True)
        
        for tool_name, patterns, param_extractor, default_params, _ in sorted_patterns:
            for pattern in patterns:
                regex = re.compile(pattern, re.IGNORECASE)
                match = regex.search(text)
                if match:
                    # Extract parameters if there's a match
                    try:
                        params = default_params.copy()
                        extracted_params = param_extractor(text)
                        params.update(extracted_params)
                        matched_directives.append((tool_name, params))
                        logger.info(f"Detected directive for tool {tool_name}")
                        break  # Stop checking other patterns for this tool
                    except Exception as e:
                        logger.warning(f"Error extracting parameters for {tool_name}: {e}")
        
        return matched_directives
    
    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool based on its name and parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            
        Returns:
            Result of the tool execution
        """
        # Tool implementation mapping
        tool_implementations = {
            "generate_dream_cycle": self._tool_generate_dream_cycle,
            "search_memories": self._tool_search_memories,
            "generate_self_reflection": self._tool_generate_self_reflection,
            "explore_knowledge_graph": self._tool_explore_knowledge_graph,
            "generate_insight": self._tool_generate_insight
        }
        
        if tool_name not in tool_implementations:
            raise ValueError(f"Unknown tool: {tool_name}")
            
        return await tool_implementations[tool_name](params)
    
    async def _tool_generate_dream_cycle(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for generate_dream_cycle.
        
        This method initiates a dream cycle in the Lucidia system, leveraging the
        dream processor to consolidate memories, generate insights, and enhance
        the knowledge graph during periods of lower activity.
        
        Args:
            params: Parameters including 'time_budget' for the dream cycle duration
            
        Returns:
            Dictionary containing the dream cycle initiation status and details
        """
        time_budget = params.get("time_budget", 180)  # Default 3 minutes
        focus_topic = params.get("focus_topic", "")  # Optional focus topic
        creativity = params.get("creativity", 0.7)  # Default creativity level
        
        logger.info(f"Initiating dream cycle with time_budget: {time_budget}s, focus: '{focus_topic}', creativity: {creativity}")
        
        try:
            # Get dream processor from config
            dream_processor = self.config.get("dream_processor")
            
            if not dream_processor:
                return {"status": "error", "message": "Dream processor not available"}
                
            # Generate a unique dream ID
            dream_id = f"dream-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Initialize dream parameters
            dream_params = {
                "time_budget": time_budget,
                "dream_id": dream_id,
                "focus_topic": focus_topic,
                "creativity": creativity,
                "initiated_by": "user_directive",
                "max_memories": 50,  # Maximum memories to process
                "consolidation_threshold": 0.6  # Threshold for memory consolidation
            }
            
            # Start the dream cycle asynchronously
            # Note: This doesn't wait for completion since dream cycles can take time
            asyncio.create_task(dream_processor.run_dream_cycle(**dream_params))
            
            # In a full implementation, you might track the dream cycle status
            # and provide mechanisms to query for results later
            
            return {
                "status": "success",
                "message": "Dream cycle initiated successfully",
                "dream_id": dream_id,
                "time_budget": time_budget,
                "focus_topic": focus_topic if focus_topic else "general",
                "started_at": datetime.now().isoformat(),
                "estimated_completion": (datetime.now() + timedelta(seconds=time_budget)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error initiating dream cycle: {e}", exc_info=True)
            return {"status": "error", "message": f"Error initiating dream cycle: {str(e)}"}
    
    async def _tool_search_memories(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for search_memories.
        
        This method searches the memory system for memories relevant to the specified query.
        It leverages the memory integration component to perform semantic search with customizable
        parameters like similarity threshold and result limits.
        
        Args:
            params: Parameters including 'query' for the memory search and 'limit' for result count
            
        Returns:
            Dictionary containing the search results
        """
        query = params.get("query", "")
        limit = params.get("limit", 5)
        min_similarity = params.get("min_similarity", 0.65)  # Default threshold for relevance
        include_metadata = params.get("include_metadata", True)
        
        if not query:
            return {"status": "error", "message": "No query provided for memory search"}
            
        logger.info(f"Searching memories with query: '{query}', limit: {limit}, min_similarity: {min_similarity}")
        
        try:
            # Get memory integration from active context
            memory_integration = self.config.get("memory_integration")
            
            if not memory_integration:
                return {"status": "error", "message": "Memory integration not available"}
                
            # Retrieve memories using the memory integration component
            memories = await memory_integration.get_relevant_memories(
                query=query,
                limit=limit,
                min_similarity=min_similarity
            )
            
            # Process and format the results
            formatted_results = []
            for memory in memories:
                memory_item = {
                    "content": memory.get("content", ""),
                    "similarity": memory.get("similarity", 0.0),
                }
                
                # Include additional metadata if requested
                if include_metadata:
                    memory_item.update({
                        "created_at": memory.get("created_at", ""),
                        "significance": memory.get("significance", 0.0),
                        "memory_type": memory.get("memory_type", "episodic"),
                        "id": memory.get("id", "")
                    })
                
                formatted_results.append(memory_item)
            
            return {
                "status": "success",
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}", exc_info=True)
            return {"status": "error", "message": f"Error searching memories: {str(e)}"}
    
    async def _tool_generate_self_reflection(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for generate_self_reflection.
        
        This method generates a self-reflection using the Self Model component of the
        Lucidia system, analyzing recent interactions, memory patterns, and knowledge
        states to produce meaningful introspection.
        
        Args:
            params: Parameters including 'scope' for the reflection target and 'depth' for detail level
            
        Returns:
            Dictionary containing the generated self-reflection
        """
        scope = params.get("scope", "recent_interactions")  # Default to recent interactions
        depth = params.get("depth", 0.7)  # Default depth level (0.0-1.0)
        include_metadata = params.get("include_metadata", True)  # Whether to include process metadata
        
        valid_scopes = ["recent_interactions", "memory_patterns", "knowledge_state", "conversational_patterns", "complete"]
        if scope not in valid_scopes:
            scope = "recent_interactions"  # Default to safe option if invalid
            
        logger.info(f"Generating self-reflection with scope: '{scope}', depth: {depth}")
        
        try:
            # Get self model from config
            self_model = self.config.get("self_model")
            memory_integration = self.config.get("memory_integration")
            
            if not self_model:
                return {"status": "error", "message": "Self model not available"}
                
            # Prepare reflection parameters based on scope
            reflection_params = {
                "depth": depth,
                "scope": scope,
                "max_memories": 25 if scope in ["recent_interactions", "complete"] else 10,
                "recency_weight": 0.8 if scope == "recent_interactions" else 0.5,
                "significance_threshold": 0.4
            }
            
            # Get relevant memories for context if available
            context_memories = []
            if memory_integration:
                # Choose query based on scope
                query = "self reflection" if scope == "complete" else f"self reflection on {scope}"
                context_memories = await memory_integration.get_relevant_memories(
                    query=query,
                    limit=reflection_params["max_memories"],
                    min_similarity=0.6,
                    recency_weight=reflection_params["recency_weight"]
                )
                
            # Generate self-reflection using self model
            reflection_result = await self_model.generate_reflection(
                scope=scope,
                context_memories=context_memories,
                depth=depth,
                significance_threshold=reflection_params["significance_threshold"]
            )
            
            # Format the response
            response = {
                "status": "success",
                "scope": scope,
                "reflection": reflection_result.get("reflection", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            # Include metadata if requested
            if include_metadata and "metadata" in reflection_result:
                response["metadata"] = {
                    "memory_count": reflection_result["metadata"].get("memory_count", 0),
                    "key_concepts": reflection_result["metadata"].get("key_concepts", []),
                    "confidence": reflection_result["metadata"].get("confidence", 0.0),
                    "processing_time": reflection_result["metadata"].get("processing_time", 0.0)
                }
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating self-reflection: {e}", exc_info=True)
            return {"status": "error", "message": f"Error generating self-reflection: {str(e)}"}
    
    async def _tool_explore_knowledge_graph(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for explore_knowledge_graph.
        
        This method explores the knowledge graph from a specified concept, retrieving
        related concepts, relationships, and connections to provide a structured view
        of the knowledge landscape around the given concept.
        
        Args:
            params: Parameters including 'concept' to explore and 'max_distance' for exploration depth
            
        Returns:
            Dictionary containing the exploration results
        """
        concept = params.get("concept", "")
        max_distance = min(params.get("max_distance", 2), 3)  # Default 2, max 3 for performance
        limit = params.get("limit", 20)  # Default limit of returned connections
        min_strength = params.get("min_strength", 0.3)  # Minimum connection strength
        
        if not concept:
            return {"status": "error", "message": "No concept provided for knowledge graph exploration"}
            
        logger.info(f"Exploring knowledge graph for concept: '{concept}', max_distance: {max_distance}, limit: {limit}")
        
        try:
            # Get knowledge graph from config
            knowledge_graph = self.config.get("knowledge_graph")
            
            if not knowledge_graph:
                return {"status": "error", "message": "Knowledge graph not available"}
                
            # Explore the knowledge graph from the given concept
            exploration_results = await knowledge_graph.explore_from_concept(
                concept=concept,
                max_distance=max_distance,
                limit=limit,
                min_strength=min_strength
            )
            
            # Process and organize the results
            connections = exploration_results.get("connections", [])
            nodes = exploration_results.get("nodes", {})
            
            # Group connections by distance from the source concept
            connections_by_distance = {}
            for i in range(1, max_distance + 1):
                connections_by_distance[i] = []
                
            for connection in connections:
                distance = connection.get("distance", 1)
                if distance <= max_distance:
                    connections_by_distance[distance].append(connection)
            
            # Format the response
            return {
                "status": "success",
                "concept": concept,
                "nodes": [
                    {
                        "id": node_id,
                        "label": node_data.get("label", node_id),
                        "type": node_data.get("type", "concept"),
                        "significance": node_data.get("significance", 0.0)
                    } for node_id, node_data in nodes.items()
                ],
                "connections": connections,
                "connections_by_distance": connections_by_distance,
                "total_connections": len(connections),
                "total_nodes": len(nodes),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exploring knowledge graph: {e}", exc_info=True)
            return {"status": "error", "message": f"Error exploring knowledge graph: {str(e)}"}
    
    async def _tool_generate_insight(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for generate_insight.
        
        This method generates insights about a topic by combining information from
        the memory system and knowledge graph, then using the LLM to synthesize
        meaningful insights from the collected data.
        
        Args:
            params: Parameters including 'topic' for the insight generation
            
        Returns:
            Dictionary containing the generated insights
        """
        topic = params.get("topic", "")
        if not topic:
            return {"status": "error", "message": "No topic provided for insight generation"}
            
        logger.info(f"Generating insight about topic: '{topic}'")
        
        try:
            # Get memory integration from active context
            # This would typically be passed in during initialization or set later
            memory_integration = self.config.get("memory_integration")
            knowledge_graph = self.config.get("knowledge_graph")
            
            if not memory_integration:
                return {"status": "error", "message": "Memory integration not available"}
                
            # Step 1: Retrieve relevant memories related to the topic
            # In a real implementation, you would first convert the topic to an embedding
            # For simplicity, we'll use a text-based approach here
            memories = await memory_integration.get_relevant_memories(
                query=topic,
                limit=10,  # Retrieve up to 10 memories
                min_similarity=0.6
            )
            
            # Step 2: If knowledge graph is available, retrieve related concepts
            connections = []
            if knowledge_graph:
                connections = await knowledge_graph.get_related_concepts(
                    concept=topic,
                    max_distance=2,  # Get concepts up to 2 hops away
                    min_strength=0.4,  # Minimum connection strength
                    limit=10  # Maximum number of connections
                )
            
            # Step 3: Generate prompt for the LLM to synthesize insights
            system_prompt = """You are an insight generator for Lucidia, an AI with advanced memory and knowledge capabilities.
            Your task is to analyze the provided memories and knowledge graph connections to generate meaningful insights.
            Focus on identifying patterns, implications, contradictions, and novel perspectives.
            
            Provide 3-5 insights that are:
            1. Non-obvious and go beyond surface-level observations
            2. Supported by the provided information
            3. Relevant to the specified topic
            4. Potentially useful for future reasoning or decision-making"""
            
            # Format memories for the prompt
            memory_text = "\n\n==== RELEVANT MEMORIES ====\n\n"
            if memories:
                for i, memory in enumerate(memories):
                    memory_text += f"Memory {i+1}: {memory.get('content', '')}\n"
                    memory_text += f"Significance: {memory.get('significance', 0.0):.2f}\n\n"
            else:
                memory_text += "No relevant memories found.\n\n"
            
            # Format knowledge graph connections for the prompt
            connection_text = "\n\n==== KNOWLEDGE GRAPH CONNECTIONS ====\n\n"
            if connections:
                for i, conn in enumerate(connections):
                    connection_text += f"Connection {i+1}: {conn.get('source', topic)} → {conn.get('relationship', 'related to')} → {conn.get('target', '')}\n"
                    connection_text += f"Strength: {conn.get('strength', 0.0):.2f}\n\n"
            else:
                connection_text += "No knowledge graph connections found.\n\n"
                
            # Construct the user prompt
            user_prompt = f"""Generate insights about the topic: '{topic}'
            
            {memory_text}
            
            {connection_text}
            
            Based on these memories and knowledge connections, generate 3-5 meaningful insights about '{topic}'.
            For each insight, provide a brief explanation of why it's significant."""
            
            # Call the LLM to generate insights
            payload = {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            # Generate insights using the LLM
            result = await self.generate(payload)
            
            if result["status"] != "success":
                return {"status": "error", "message": f"Failed to generate insights: {result.get('message', 'Unknown error')}"}
                
            # Parse the insights from the LLM response
            insights_text = result["response"]
            
            # In a more structured approach, you might want to use a JSON schema response format
            # to get a more structured output, but for simplicity we'll use the raw text
            
            return {
                "status": "success",
                "topic": topic,
                "insights_text": insights_text,
                "memory_count": len(memories),
                "connection_count": len(connections),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}", exc_info=True)
            return {"status": "error", "message": f"Error generating insights: {str(e)}"}
    
    def _build_payload_with_directive_context(self, user_input: str, context: Dict[str, Any], tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a payload for the LLM that includes directive execution context.
        
        Args:
            user_input: Original user input
            context: Additional context
            tool_results: Results from executed tools
            
        Returns:
            Payload dictionary for the LLM
        """
        messages = []
        
        # Add system message
        system_message = "You are Lucidia, an AI assistant with enhanced memory capabilities. "
        
        if tool_results:
            system_message += "You have access to various tools that you can use to enhance your responses. "
            system_message += "Some tools have already been executed based on the user's request."
        
        messages.append({"role": "system", "content": system_message})
        
        # Add tool execution results as context
        if tool_results:
            tool_context = "The following tools were executed based on the user's request:\n\n"
            
            for tool_result in tool_results:
                tool_name = tool_result.get("tool", "unknown")
                result = tool_result.get("result", {})
                error = tool_result.get("error")
                
                if error:
                    tool_context += f"- Tool '{tool_name}' encountered an error: {error}\n"
                else:
                    tool_context += f"- Tool '{tool_name}' was executed successfully\n"
                    
                    # Add specific details based on the tool type
                    if tool_name == "search_memories" and "results" in result:
                        memories = result.get("results", [])
                        tool_context += f"  Found {len(memories)} memories related to '{result.get('query', '')}'\n"
                        
                        for i, memory in enumerate(memories[:3]):  # Limit to first 3 for brevity
                            tool_context += f"  Memory {i+1}: {memory.get('content', '')}\n"
                            
                    elif tool_name == "generate_dream_cycle":
                        tool_context += f"  Dream cycle initiated with ID: {result.get('dream_id', 'unknown')}\n"
                        
                    elif tool_name == "generate_self_reflection" and "reflection" in result:
                        tool_context += f"  Self-reflection: {result.get('reflection', '')}\n"
                        
                    elif tool_name == "explore_knowledge_graph" and "connections" in result:
                        connections = result.get("connections", [])
                        tool_context += f"  Found {len(connections)} connections related to '{result.get('concept', '')}'\n"
                        
                    elif tool_name == "generate_insight" and "insight" in result:
                        tool_context += f"  Insight: {result.get('insight', '')}\n"
            
            messages.append({"role": "system", "content": tool_context})
        
        # Add any other context from memory or prior conversation
        if context.get("memories"):
            memory_context = "Relevant memories from previous interactions:\n\n"
            for i, memory in enumerate(context["memories"]):
                memory_context += f"Memory {i+1}: {memory.get('content', '')}\n"
            
            messages.append({"role": "system", "content": memory_context})
        
        # Add the user's message
        messages.append({"role": "user", "content": user_input})
        
        # Build the complete payload
        return {
            "model": "local-model",  # Uses the currently loaded model
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
    
    def _create_response_prefix(self, tool_results: List[Dict[str, Any]]) -> str:
        """Create a natural language prefix for the response based on executed tools.
        
        Args:
            tool_results: Results from executed tools
            
        Returns:
            A string to prefix to the response
        """
        if not tool_results:
            return ""
            
        # Get the first successful tool result
        successful_tools = [r for r in tool_results if "error" not in r]
        if not successful_tools:
            return ""
            
        tool = successful_tools[0]["tool"]
        
        # Create appropriate prefix based on the tool type
        if tool == "generate_dream_cycle":
            return "I've initiated a dream cycle as requested. "
        elif tool == "search_memories":
            return "I've searched my memories as you asked. "
        elif tool == "generate_self_reflection":
            return "I've reflected on my recent interactions. "
        elif tool == "explore_knowledge_graph":
            return "I've explored my knowledge graph for those concepts. "
        elif tool == "generate_insight":
            return "I've generated some insights based on your request. "
        else:
            return f"I've processed your request using {tool}. "
