"""
RAG Integration Service for Lucidia with QuickRecal Support

This module implements Retrieval-Augmented Generation integration for Lucidia's
architecture, enabling knowledge retrieval during reflection and supporting
continuous self-evolution through dreaming and reflection.

The integration works at the Dream API level to ensure full access to all
Lucidia components, including the HPC-QR Flow Manager, memory persistence,
and parameter update mechanisms.
"""

import json
import logging
import os
import aiohttp
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Callable, Set
from datetime import datetime
from urllib.parse import urljoin
import traceback
import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGIntegrationService:
    """
    RAG Integration Service that connects Lucidia's components with external
    knowledge retrieval capabilities, supporting continuous self-evolution.
    
    Updated to work with the QuickRecal architecture for improved memory retrieval.
    """
    
    def __init__(self, 
                 memory_system = None,
                 knowledge_graph = None,
                 parameter_manager = None,
                 hpc_manager = None):
        """Initialize the RAG Integration Service."""
        self.memory_system = memory_system
        self.knowledge_graph = knowledge_graph
        self.parameter_manager = parameter_manager
        
        # Initialize HPC-QR Flow Manager for embedding processing if not provided
        if hpc_manager is None:
            from integration.hpc_qr_flow_manager import HPCQRFlowManager
            self.hpc_manager = HPCQRFlowManager({
                'embedding_dim': 768,  # Default dimension
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            })
        else:
            self.hpc_manager = hpc_manager
        
        # Get LM Studio configuration from parameter manager
        self.lm_studio_url = "http://host.docker.internal:1234/v1"  # Use host.docker.internal for Docker compatibility
        self.lm_studio_api_key = "lm-studio"  # Standard key for LM Studio
        
        # Get model configuration from parameter manager if available
        if parameter_manager and hasattr(parameter_manager, 'config'):
            self.lm_studio_model = parameter_manager.config.get("lm_studio", {}).get("model", "qwen2.5-7b-instruct")
        else:
            self.lm_studio_model = "qwen2.5-7b-instruct"  # Default model
        
        # HTTP client for API calls
        self.http_session = None
        self.initialized = False
        
        # Setup RAG tools
        self.tools = {}
        
        # Insight registry - tracks insights generated through RAG
        self.insights_registry = set()
        
        # RAG memory cache for quick lookup of recently retrieved information
        self.memory_cache = {}
        
        # QuickRecal thresholds for different retrieval operations
        self.retrieval_thresholds = {
            'high_quality': 0.8,    # For critical information
            'standard': 0.6,        # For normal retrieval
            'exploratory': 0.4,     # For exploratory searches
            'comprehensive': 0.2    # For broad searches
        }
        
        # Initialize with default tools
        self._register_default_tools()
        
        logger.info(f"Initialized RAG Integration Service with QuickRecal support and LM Studio at {self.lm_studio_url}")
    
    async def initialize(self):
        """Initialize required components and connections."""
        if not self.initialized:
            if self.http_session is None or self.http_session.closed:
                # Create HTTP session with appropriate headers for LLM Studio
                self.http_session = aiohttp.ClientSession(headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.lm_studio_api_key}"
                })
                logger.info(f"Initialized RAG Integration Service with LM Studio at {self.lm_studio_url}")
            
            # Verify LLM connection
            try:
                # LM Studio model endpoint doesn't include /v1 in the path
                model_endpoint = self.lm_studio_url.replace("/v1", "/models")
                async with self.http_session.get(
                    model_endpoint,
                    timeout=5
                ) as response:
                    if response.status == 200:
                        logger.info("Successfully connected to LM Studio API")
                    else:
                        logger.warning(f"LM Studio API returned unexpected status: {response.status}")
            except Exception as e:
                logger.warning(f"Could not connect to LM Studio API: {e}")
                logger.warning("RAG capabilities that require LLM will not be available")
                
            self.initialized = True
            
    async def close(self):
        """Close connections and clean up resources."""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
            logger.info("Closed aiohttp session for RAG integration")
            self.initialized = False
    
    def _register_default_tools(self):
        """Register default knowledge retrieval tools."""
        # Register Wikipedia knowledge retrieval
        self.register_tool(
            name="fetch_wikipedia_content",
            function=self.fetch_wikipedia_content,
            description="Search Wikipedia and fetch the introduction of the most relevant article. "
                       "Use this when external factual knowledge is needed that might be found "
                       "on Wikipedia. If the search query has a typo, correct it before searching.",
            parameters={
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Search query for finding the Wikipedia article",
                    },
                },
                "required": ["search_query"],
            }
        )
        
        # Register knowledge graph query tool
        self.register_tool(
            name="query_knowledge_graph",
            function=self.query_knowledge_graph,
            description="Query Lucidia's internal knowledge graph for concepts, relationships, "
                       "and stored knowledge. Use this when information likely already exists "
                       "in Lucidia's memory system.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query or concept to search for in the knowledge graph",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    },
                },
                "required": ["query"],
            }
        )
        
        # Register memory search tool
        self.register_tool(
            name="search_memories",
            function=self.search_memories,
            description="Search through Lucidia's memories to find relevant information or experiences. "
                       "Use this for retrieving personal memories or past interactions.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding related memories",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Type of memory to search for (personal, factual, relationship)",
                        "enum": ["personal", "factual", "relationship", "all"],
                        "default": "all"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of memories to return",
                        "default": 5
                    },
                },
                "required": ["query"],
            }
        )
        
        # Register insight recording tool
        self.register_tool(
            name="record_insight",
            function=self.record_insight,
            description="Record a new insight derived from reflection or knowledge retrieval. "
                       "Use this to store important realizations or connections that should be "
                       "remembered for future reference.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The insight content or realization",
                    },
                    "source": {
                        "type": "string",
                        "description": "Source of the insight (reflection, retrieved knowledge, etc.)",
                    },
                    "quickrecal_score": {
                        "type": "number",
                        "description": "QuickRecal score between 0 and 1 to indicate importance",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.75
                    },
                },
                "required": ["content", "source"],
            }
        )
        
        logger.info(f"Registered {len(self.tools)} default tools for RAG integration")
    
    def register_tool(self, name: str, function: Callable, description: str, parameters: Dict[str, Any]):
        """
        Register a new tool for the RAG integration.
        
        Args:
            name: Name of the tool
            function: The function to execute when the tool is called
            description: Description of the tool
            parameters: Parameters schema for the tool in JSON Schema format
        """
        self.tools[name] = {
            "function": function,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            }
        }
        logger.info(f"Registered tool: {name}")
    
    async def enhance_reflection(self, reflection_query: str, context: Dict[str, Any], focus_areas: Optional[List[str]] = None):
        """
        Enhance a reflection process with RAG capabilities.
        
        Args:
            reflection_query: The reflection query or topic
            context: Additional context for the reflection
            focus_areas: Specific areas to focus on
            
        Returns:
            Enhanced reflection results including insights from RAG
        """
        await self.initialize()
        
        # Initialize results structure
        results = {
            "status": "success",
            "insights": [],
            "fragments": [],
            "tool_results": []
        }
        
        # Even if LLM is unavailable, we can still retrieve information
        try:
            # Gather fragments from sources that don't require LLM
            logger.info("Retrieving memory fragments for reflection enhancement")
            memory_fragments = await self._retrieve_relevant_memories(reflection_query, focus_areas)
            if memory_fragments:
                results["fragments"].extend(memory_fragments)
                # Add a basic insight from memories
                results["insights"].append({
                    "type": "memory_based",
                    "content": f"Based on {len(memory_fragments)} relevant memories, there are patterns in how I've processed similar information before.",
                    "confidence": 0.75
                })
        except Exception as e:
            logger.error(f"Error retrieving memory fragments: {e}")
            
        try:
            # Query knowledge graph based on focus areas
            logger.info("Querying knowledge graph for relevant concepts")
            kg_fragments = await self._query_knowledge_graph(reflection_query, focus_areas)
            if kg_fragments:
                results["fragments"].extend(kg_fragments)
                # Add a basic insight from knowledge graph
                results["insights"].append({
                    "type": "concept_based",
                    "content": f"My knowledge graph contains {len(kg_fragments)} relevant concepts that inform my understanding of this area.",
                    "confidence": 0.8
                })
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            
        # Try to use LLM for enhanced insights if available
        llm_available = False
        try:
            # Verify LLM connection - use the fixed URL format as in initialize method
            model_endpoint = self.lm_studio_url.replace("/v1", "/models")
            async with self.http_session.get(
                model_endpoint,
                timeout=2
            ) as response:
                if response.status == 200:
                    llm_available = True
        except Exception:
            logger.warning("LLM is not available for enhanced reflection")

        if llm_available:
            try:
                # Build messages for the LLM
                system_message = (
                    "You are Lucidia's reflection engine assistant. Your purpose is to help Lucidia reflect "
                    "on itself, its knowledge, and its experiences to generate valuable insights and growth. "
                    "You have access to tools that can retrieve information from Wikipedia, Lucidia's knowledge graph, "
                    "and Lucidia's memories. Use these tools when relevant to enhance the reflection process."
                )
                
                if focus_areas:
                    system_message += f"\nThis reflection is focusing on: {', '.join(focus_areas)}"
                
                messages = [
                    {"role": "system", "content": system_message}
                ]
                
                # Add context if provided
                if context:
                    context_str = json.dumps(context, indent=2)
                    messages.append({
                        "role": "system", 
                        "content": f"Here is additional context for this reflection:\n{context_str}"
                    })
                
                # Add already gathered fragments to provide context
                if results["fragments"]:
                    fragments_str = "\n".join([f"- {f['type']}: {f['content']}" for f in results["fragments"]])
                    messages.append({
                        "role": "system",
                        "content": f"I've already gathered the following information:\n{fragments_str}"
                    })
                
                # Add the reflection query
                messages.append({"role": "user", "content": reflection_query})
                
                # Try to enhance reflection with the LLM
                logger.info("Using LLM to enhance reflection")
                llm_results = await self._generate_enhanced_insights(messages)
                
                # Add LLM results to our overall results
                if "insights" in llm_results:
                    results["insights"].extend(llm_results["insights"])
                if "fragments" in llm_results:
                    results["fragments"].extend(llm_results["fragments"])
                if "tool_results" in llm_results:
                    results["tool_results"].extend(llm_results["tool_results"])
                    
            except Exception as e:
                logger.error(f"Error enhancing reflection with LLM: {e}")
                results["status"] = "partial"
                results["error"] = str(e)
        else:
            # LLM is not available, so we'll use a simpler approach
            logger.info("Using simplified reflection enhancement without LLM")
            results["status"] = "partial"
            results["error"] = "LLM not available for enhanced reflection"
            
            # Synthesize basic insights from the fragments we've collected
            if results["fragments"]:
                # Add a synthetic insight summarizing the findings
                results["insights"].append({
                    "type": "synthetic",
                    "content": f"Analysis of {len(results['fragments'])} relevant information sources suggests patterns worth exploring further.",
                    "confidence": 0.65
                })
                
                # Add insights based on fragment types
                memory_count = len([f for f in results["fragments"] if f.get("type") == "memory"])
                concept_count = len([f for f in results["fragments"] if f.get("type") == "concept"])
                
                if memory_count > 0:
                    results["insights"].append({
                        "type": "memory_analysis",
                        "content": f"My memory systems contain {memory_count} relevant memories that could inform this reflection.",
                        "confidence": 0.7
                    })
                    
                if concept_count > 0:
                    results["insights"].append({
                        "type": "concept_analysis",
                        "content": f"My knowledge graph contains {concept_count} relevant concepts that may provide useful context.",
                        "confidence": 0.7
                    })
        
        # Final processing of insights to ensure quality
        if not results["insights"]:
            results["insights"].append({
                "type": "fallback",
                "content": "No specific insights could be generated at this time, but I can revisit this reflection later with improved capabilities.",
                "confidence": 0.5
            })
            
        return results
    
    async def _process_reflection_response(self, 
                                          result: Dict[str, Any], 
                                          messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process the response from the LLM during reflection, handling tool calls.
        
        Args:
            result: The API response
            messages: Current message history
            
        Returns:
            Processed reflection results
        """
        all_insights = []
        tool_results = []
        
        # Check for tool calls
        if "choices" in result and result["choices"] and "message" in result["choices"][0]:
            message = result["choices"][0]["message"]
            
            if "tool_calls" in message and message["tool_calls"]:
                # Add the assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": tool_call["id"],
                        "type": tool_call["type"],
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"]
                        }
                    } for tool_call in message["tool_calls"]]
                })
                
                # Process each tool call
                for tool_call in message["tool_calls"]:
                    if tool_call["type"] == "function":
                        function_name = tool_call["function"]["name"]
                        
                        if function_name in self.tools:
                            # Parse arguments
                            try:
                                arguments = json.loads(tool_call["function"]["arguments"])
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON arguments: {tool_call['function']['arguments']}")
                                arguments = {}
                            
                            # Execute the function
                            function = self.tools[function_name]["function"]
                            result = await function(**arguments)
                            
                            # Add result to the list
                            tool_results.append({
                                "tool": function_name,
                                "arguments": arguments,
                                "result": result
                            })
                            
                            # Add tool response message
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": json.dumps(result)
                            })
                
                # Final call to get reflections based on the tool results
                final_result = await self._get_final_reflection(messages)
                
                if "content" in final_result:
                    # Extract insights from the final response
                    insights = self._extract_insights(final_result["content"])
                    all_insights.extend(insights)
                    
                    return {
                        "status": "success",
                        "tool_results": tool_results,
                        "insights": all_insights,
                        "content": final_result["content"],
                        "fragments": self._extract_fragments(final_result["content"])
                    }
            else:
                # No tool calls, just extract insights from the direct response
                content = message.get("content", "")
                insights = self._extract_insights(content)
                all_insights.extend(insights)
                
                return {
                    "status": "success",
                    "insights": all_insights,
                    "content": content,
                    "fragments": self._extract_fragments(content)
                }
        
        # Fallback if response doesn't match expected format
        logger.warning(f"Unexpected response format from LLM: {result}")
        return {
            "status": "partial",
            "message": "Unexpected response format from LLM",
            "insights": all_insights,
            "tool_results": tool_results
        }
    
    async def _get_final_reflection(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get the final reflection after tool execution.
        
        Args:
            messages: Current message history
            
        Returns:
            Final response with reflection
        """
        try:
            # Add a system message prompting for structured reflection
            messages.append({
                "role": "system",
                "content": (
                    "Based on all the information retrieved, please provide a structured reflection "
                    "with the following sections:\n"
                    "1. Main insights and realizations\n"
                    "2. New connections between concepts\n"
                    "3. Questions for further exploration\n"
                    "4. Potential adaptations or improvements\n\n"
                    "For each insight, include a confidence level (0-1)."
                )
            })
            
            async with self.http_session.post(
                f"{self.lm_studio_url}/chat/completions",
                json={
                    "model": self.lm_studio_model,
                    "messages": messages
                }
            ) as response:
                result = await response.json()
                
                if response.status != 200:
                    logger.error(f"Error getting final reflection: {result}")
                    return {"content": "Error getting final reflection"}
                
                if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                    return result["choices"][0]["message"]
                else:
                    return {"content": "Unexpected response format for final reflection"}
                    
        except Exception as e:
            logger.error(f"Error getting final reflection: {e}")
            return {"content": f"Error: {str(e)}"}
    
    def _extract_insights(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract structured insights from reflection content.
        
        Args:
            content: The reflection content
            
        Returns:
            List of extracted insights
        """
        insights = []
        
        # Simple extraction based on common patterns
        # In a production system, this would use more sophisticated parsing
        lines = content.split('\n')
        current_type = "general"
        
        for line in lines:
            line = line.strip()
            
            # Detect section headers
            if line.lower().startswith("# insight") or line.lower().startswith("## insight"):
                current_type = "insight"
                continue
            elif line.lower().startswith("# question") or line.lower().startswith("## question"):
                current_type = "question"
                continue
            elif line.lower().startswith("# adaptation") or line.lower().startswith("## adaptation"):
                current_type = "adaptation"
                continue
            elif line.lower().startswith("# connection") or line.lower().startswith("## connection"):
                current_type = "connection"
                continue
            
            # Skip empty lines and headers
            if not line or line.startswith("#"):
                continue
                
            # Extract confidence if present
            confidence = 0.75  # Default confidence
            if "confidence:" in line.lower():
                try:
                    confidence_part = line.lower().split("confidence:")[1].strip()
                    confidence_value = float(confidence_part.split()[0])
                    confidence = min(max(confidence_value, 0), 1)  # Ensure between 0 and 1
                    # Remove the confidence part from the line
                    line = line.lower().split("confidence:")[0].strip()
                except:
                    pass
            
            insights.append({
                "type": current_type,
                "content": line,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
        
        return insights
    
    def _extract_fragments(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract structured fragments from reflection content.
        
        Args:
            content: The reflection content
            
        Returns:
            List of fragments categorized by type
        """
        fragments = []
        
        # Similar to _extract_insights but with more structure
        lines = content.split('\n')
        current_section = None
        current_fragment = {"type": "general", "content": "", "confidence": 0.75}
        
        for line in lines:
            line = line.strip()
            
            # Detect section headers
            if line.lower().startswith("# ") or line.lower().startswith("## "):
                # Save previous fragment if it has content
                if current_fragment["content"]:
                    fragments.append(current_fragment)
                
                # Start new section and fragment
                section_name = line.lstrip('#').strip().lower()
                if "insight" in section_name:
                    current_section = "insight"
                elif "question" in section_name:
                    current_section = "question"
                elif "adaptation" in section_name or "improvement" in section_name:
                    current_section = "adaptation"
                elif "connection" in section_name:
                    current_section = "connection"
                elif "hypothesis" in section_name:
                    current_section = "hypothesis"
                elif "counterfactual" in section_name:
                    current_section = "counterfactual"
                else:
                    current_section = "general"
                
                current_fragment = {
                    "type": current_section,
                    "content": "",
                    "confidence": 0.75,
                    "timestamp": datetime.now().isoformat()
                }
                continue
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for list items that might be separate fragments
            if line.startswith('-') or line.startswith('*') or (line[0].isdigit() and line[1:3] == '. '):
                # Save previous fragment if it has content
                if current_fragment["content"]:
                    fragments.append(current_fragment)
                
                # Start new fragment
                fragment_text = line[line.find(' ')+1:].strip()
                
                # Extract confidence if present
                confidence = 0.75  # Default confidence
                if "confidence:" in fragment_text.lower():
                    try:
                        confidence_part = fragment_text.lower().split("confidence:")[1].strip()
                        confidence_value = float(confidence_part.split()[0])
                        confidence = min(max(confidence_value, 0), 1)
                        # Remove the confidence part
                        fragment_text = fragment_text.split("confidence:")[0].strip()
                    except:
                        pass
                
                current_fragment = {
                    "type": current_section or "general",
                    "content": fragment_text,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }
                fragments.append(current_fragment)
                current_fragment = {
                    "type": current_section or "general",
                    "content": "",
                    "confidence": 0.75,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Add to current fragment
                if current_fragment["content"]:
                    current_fragment["content"] += " " + line
                else:
                    current_fragment["content"] = line
        
        # Add the last fragment if it has content
        if current_fragment["content"]:
            fragments.append(current_fragment)
        
        return fragments
    
    async def _integrate_rag_results(self, reflection_results: Dict[str, Any]):
        """
        Integrate RAG reflection results with Lucidia's systems.
        
        Args:
            reflection_results: Results from the reflection process
        """
        # Skip integration if there was an error
        if reflection_results.get("status") != "success":
            return
        
        # 1. Add insights to knowledge graph if available
        if self.knowledge_graph and "insights" in reflection_results:
            for insight in reflection_results["insights"]:
                try:
                    # Only add if confidence is high enough
                    if insight.get("confidence", 0) >= 0.7:
                        concept_name = f"insight:{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        await self.knowledge_graph.add_concept(
                            name=concept_name,
                            definition=insight["content"],
                            attributes={
                                "type": insight["type"],
                                "confidence": insight["confidence"],
                                "source": "rag_reflection",
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        logger.info(f"Added insight to knowledge graph: {insight['content'][:50]}...")
                except Exception as e:
                    logger.error(f"Error adding insight to knowledge graph: {e}")
        
        # 2. Store high-confidence insights in memory if available
        if self.memory_system and "insights" in reflection_results:
            for insight in reflection_results["insights"]:
                try:
                    # Only store if confidence is high enough
                    if insight.get("confidence", 0) >= 0.8:
                        memory_data = {
                            "content": insight["content"],
                            "type": "reflection_insight",
                            "quickrecal_score": insight["confidence"],  # Use confidence as QuickRecal score
                            "metadata": {
                                "insight_type": insight["type"],
                                "source": "rag_reflection",
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        
                        # Store in memory system using the new interface that supports QuickRecal
                        await self.memory_system.store(memory_data)
                        logger.info(f"Stored high-confidence insight in memory: {insight['content'][:50]}...")
                except Exception as e:
                    logger.error(f"Error storing insight in memory: {e}")
        
        # 3. Update hypersphere for concepts mentioned in reflection
        if hasattr(self, 'hypersphere_manager') and self.hypersphere_manager and "content" in reflection_results:
            try:
                # This would ideally use NLP to extract key concepts
                # For now, we'll use a simple approach
                content = reflection_results["content"]
                
                # Add to hypersphere processing queue for later processing
                await self.hypersphere_manager.queue_content_for_processing(
                    content=content,
                    source="rag_reflection",
                    priority=0.8
                )
                logger.info("Queued reflection content for hypersphere processing")
            except Exception as e:
                logger.error(f"Error updating hypersphere with reflection content: {e}")
    
    # ========== Tool Implementations ==========
    
    async def fetch_wikipedia_content(self, search_query: str):
        """
        Fetches wikipedia content for a given search query.
        
        Args:
            search_query: The search term for Wikipedia
            
        Returns:
            Dictionary containing the retrieved Wikipedia content
        """
        import urllib.parse
        import urllib.request
        
        try:
            # Search for most relevant article
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": search_query,
                "srlimit": 1,
            }

            url = f"{search_url}?{urllib.parse.urlencode(search_params)}"
            with urllib.request.urlopen(url) as response:
                search_data = json.loads(response.read().decode())

            if not search_data["query"]["search"]:
                return {
                    "status": "error",
                    "message": f"No Wikipedia article found for '{search_query}'"
                }

            # Get the normalized title from search results
            normalized_title = search_data["query"]["search"][0]["title"]

            # Now fetch the actual content with the normalized title
            content_params = {
                "action": "query",
                "format": "json",
                "titles": normalized_title,
                "prop": "extracts",
                "exintro": "true",
                "explaintext": "true",
                "redirects": 1,
            }

            url = f"{search_url}?{urllib.parse.urlencode(content_params)}"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())

            pages = data["query"]["pages"]
            page_id = list(pages.keys())[0]

            if page_id == "-1":
                return {
                    "status": "error",
                    "message": f"No Wikipedia article found for '{search_query}'"
                }

            content = pages[page_id]["extract"].strip()
            
            # Cache the result for future use
            self.memory_cache[f"wikipedia:{search_query}"] = {
                "content": content,
                "title": pages[page_id]["title"],
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "content": content,
                "title": pages[page_id]["title"],
                "source": "wikipedia"
            }

        except Exception as e:
            logger.error(f"Error fetching Wikipedia content: {e}")
            return {"status": "error", "message": str(e)}
    
    async def query_knowledge_graph(self, query: str, max_results: int = 5):
        """
        Query Lucidia's knowledge graph for concepts and relationships.
        
        Args:
            query: Query or concept to search for
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing the query results
        """
        if not self.knowledge_graph:
            return {
                "status": "error",
                "message": "Knowledge graph not available"
            }
        
        try:
            # Query the knowledge graph
            results = await self.knowledge_graph.query(
                query=query,
                max_results=max_results,
                min_relevance=0.3
            )
            
            # Format results for better readability
            formatted_results = []
            for result in results.get("results", []):
                formatted_results.append({
                    "name": result.get("name", ""),
                    "definition": result.get("definition", ""),
                    "relevance": result.get("relevance", 0),
                    "attributes": result.get("attributes", {})
                })
            
            return {
                "status": "success",
                "query": query,
                "results": formatted_results,
                "source": "knowledge_graph"
            }
            
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return {"status": "error", "message": str(e)}
    
    async def search_memories(self, query: str, memory_type: str = "all", max_results: int = 5):
        """
        Search through Lucidia's memories to find relevant information.
        Updated to use QuickRecal-based retrieval.
        
        Args:
            query: Search query for finding related memories
            memory_type: Type of memory to search for
            max_results: Maximum number of memories to return
            
        Returns:
            Dictionary containing the memory search results
        """
        if not self.memory_system:
            return {
                "status": "error",
                "message": "Memory system not available"
            }
        
        try:
            # First, get an embedding for the query using HPCQRFlowManager
            query_embedding = await self.hpc_manager.get_embedding(query)
            
            # Set up memory search options
            search_options = {
                "query_embedding": query_embedding,
                "limit": max_results,
                "min_quickrecal": self.retrieval_thresholds['standard']  # Use standard threshold
            }
            
            # Add type filter if specified
            if memory_type and memory_type != "all":
                search_options["memory_type"] = memory_type
            
            # Process the embedding through HPCQRFlowManager to get QuickRecal score
            _, _ = await self.hpc_manager.process_embedding(query_embedding)
            
            # Search memories using the enhanced memory storage interface
            memories = await self.memory_system.search(**search_options)
            
            # Format results for better readability
            formatted_memories = []
            for memory, score in memories:
                formatted_memories.append({
                    "content": memory.content,
                    "type": memory.memory_type.value if hasattr(memory, 'memory_type') else "unknown",
                    "quickrecal_score": memory.get_effective_quickrecal() 
                                        if hasattr(memory, 'get_effective_quickrecal') 
                                        else memory.quickrecal_score if hasattr(memory, 'quickrecal_score') 
                                        else 0.5,
                    "timestamp": memory.timestamp if hasattr(memory, 'timestamp') else "",
                    "similarity": score
                })
            
            return {
                "status": "success",
                "query": query,
                "results": formatted_memories,
                "source": "memory_system"
            }
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    async def record_insight(self, content: str, source: str, quickrecal_score: float = 0.75):
        """
        Record a new insight derived from reflection or knowledge retrieval.
        Updated to use QuickRecal scoring.
        
        Args:
            content: The insight content or realization
            source: Source of the insight
            quickrecal_score: QuickRecal score between 0 and 1
            
        Returns:
            Status of the recording operation
        """
        # Generate a unique ID for the insight
        insight_id = f"insight:{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Skip if we've already recorded this insight
        insight_signature = f"{content[:100]}"
        if insight_signature in self.insights_registry:
            return {
                "status": "skipped",
                "message": "Duplicate insight detected",
                "insight_id": insight_id
            }
        
        try:
            # Process the insight content through HPC-QR to get embedding and QuickRecal score
            embedding_tensor = await self.hpc_manager.get_embedding(content)
            processed_embedding, computed_quickrecal = await self.hpc_manager.process_embedding(embedding_tensor)
            
            # Use either the provided QuickRecal score or the computed one, whichever is higher
            final_quickrecal = max(quickrecal_score, float(computed_quickrecal))
            
            # Record in both knowledge graph and memory if available
            if self.knowledge_graph:
                await self.knowledge_graph.add_concept(
                    name=insight_id,
                    definition=content,
                    attributes={
                        "type": "insight",
                        "source": source,
                        "quickrecal_score": final_quickrecal,  # Use QuickRecal instead of significance
                        "timestamp": datetime.now().isoformat()
                    }
                )
                logger.info(f"Recorded insight in knowledge graph: {content[:50]}...")
            
            # Store in memory system if available
            if self.memory_system:
                memory_data = {
                    "content": content,
                    "embedding": processed_embedding.cpu().numpy() if isinstance(processed_embedding, torch.Tensor) else processed_embedding,
                    "memory_type": "insight",
                    "quickrecal_score": final_quickrecal,  # Use QuickRecal score
                    "metadata": {
                        "source": source,
                        "insight_id": insight_id,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # Use the enhanced memory storage interface
                await self.memory_system.store(**memory_data)
                logger.info(f"Recorded insight in memory system: {content[:50]}...")
            
            # Add to local registry to prevent duplicates
            self.insights_registry.add(insight_signature)
            
            return {
                "status": "success",
                "message": "Insight recorded successfully",
                "insight_id": insight_id,
                "quickrecal_score": final_quickrecal
            }
            
        except Exception as e:
            logger.error(f"Error recording insight: {e}")
            return {
                "status": "error",
                "message": str(e),
                "insight_id": insight_id
            }
    
    async def _retrieve_relevant_memories(self, query: str, focus_areas: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieve memories relevant to the reflection query using QuickRecal.
        
        Args:
            query: The reflection query
            focus_areas: Optional specific areas to focus on
            
        Returns:
            List of memory fragments
        """
        try:
            logger.info(f"Retrieving relevant memories for query: {query}")
            memory_results = []
            
            if self.memory_system:
                # Process query through HPCQRFlowManager to get embedding and QuickRecal score
                query_embedding = await self.hpc_manager.get_embedding(query)
                processed_embedding, quickrecal_score = await self.hpc_manager.process_embedding(query_embedding)
                
                # Set retrieval threshold based on importance
                if quickrecal_score > 0.8:
                    threshold = self.retrieval_thresholds['high_quality']
                elif quickrecal_score > 0.6:
                    threshold = self.retrieval_thresholds['standard']
                else:
                    threshold = self.retrieval_thresholds['exploratory']
                
                # Construct search options
                search_options = {
                    "query_embedding": processed_embedding,
                    "limit": 5,  # Reasonable limit
                    "min_quickrecal": threshold
                }
                
                # Add focus areas as filter if provided
                if focus_areas:
                    # Convert focus areas to memory types if applicable
                    memory_types = []
                    for area in focus_areas:
                        # Try to match focus area to memory types
                        if "personal" in area.lower():
                            memory_types.append("personal")
                        elif "factual" in area.lower() or "knowledge" in area.lower():
                            memory_types.append("semantic")
                        elif "procedural" in area.lower() or "skill" in area.lower():
                            memory_types.append("procedural")
                    
                    if memory_types:
                        search_options["memory_type"] = memory_types
                
                # Perform memory search
                memories = await self.memory_system.search(**search_options)
                
                # Transform results to fragments
                for memory, score in memories:
                    memory_results.append({
                        "type": "memory",
                        "content": memory.content if hasattr(memory, 'content') else str(memory),
                        "source": "memory_system",
                        "timestamp": memory.timestamp if hasattr(memory, 'timestamp') else "",
                        "quickrecal_score": memory.get_effective_quickrecal() 
                                        if hasattr(memory, 'get_effective_quickrecal') 
                                        else memory.quickrecal_score if hasattr(memory, 'quickrecal_score')
                                        else 0.5,
                        "similarity": score
                    })
                
                logger.info(f"Retrieved {len(memory_results)} relevant memories")
            else:
                logger.warning("No memory system available for memory retrieval")
                
            return memory_results
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            traceback.print_exc()
            return []
    
    async def _query_knowledge_graph(self, query: str, focus_areas: Optional[List[str]] = None) -> List[Dict]:
        """
        Query the knowledge graph for concepts relevant to the reflection.
        
        Args:
            query: The reflection query
            focus_areas: Optional specific areas to focus on
            
        Returns:
            List of concept fragments
        """
        try:
            logger.info(f"Querying knowledge graph for query: {query}")
            kg_results = []
            
            if self.knowledge_graph:
                # Get concepts related to query
                concepts = []
                
                # First try with focus areas if provided
                if focus_areas:
                    for area in focus_areas:
                        area_concepts = await self.knowledge_graph.search_concepts(area, limit=3)
                        concepts.extend(area_concepts)
                
                # Then try with the main query
                main_concepts = await self.knowledge_graph.search_concepts(query, limit=5)
                concepts.extend(main_concepts)
                
                # Remove duplicates by name
                seen = set()
                unique_concepts = []
                for c in concepts:
                    if c.get("name") not in seen:
                        seen.add(c.get("name"))
                        unique_concepts.append(c)
                
                # Transform concepts into fragments
                for concept in unique_concepts:
                    kg_results.append({
                        "type": "concept",
                        "content": concept.get("definition", concept.get("description", concept.get("name", ""))),
                        "source": "knowledge_graph",
                        "quickrecal_score": concept.get("quickrecal_score", 
                                              concept.get("significance", 0.7))  # Support both names
                    })
                
                logger.info(f"Retrieved {len(kg_results)} relevant concepts from knowledge graph")
            else:
                logger.warning("No knowledge graph available for concept retrieval")
                
            return kg_results
            
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return []
    
    async def _generate_enhanced_insights(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Generate enhanced insights using LLM with tool usage capabilities.
        
        Args:
            messages: Message history for the LLM conversation
            
        Returns:
            Dictionary of enhanced insights and related information
        """
        try:
            # Prepare tools for the API request
            tools_list = [tool["schema"] for tool in self.tools.values()]
            
            # Call the LLM with tool usage capabilities
            async with self.http_session.post(
                f"{self.lm_studio_url}/chat/completions",
                json={
                    "model": self.lm_studio_model,
                    "messages": messages,
                    "tools": tools_list,
                    "temperature": 0.7
                },
                timeout=30  # Longer timeout for complex reflections
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logger.error(f"Error from LLM API: {response.status}, {response_text}")
                    return {
                        "status": "error",
                        "message": f"Error from LLM API: {response.status}",
                        "insights": []
                    }
                
                result = await response.json()
                
                # Process the result to extract insights and process tool calls
                return await self._process_reflection_response(result, messages)
                
        except Exception as e:
            logger.error(f"Error generating enhanced insights: {e}")
            return {
                "status": "error",
                "message": str(e),
                "insights": []
            }