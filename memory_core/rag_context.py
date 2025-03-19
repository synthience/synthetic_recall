# memory_client/rag_context.py

import logging
import re
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGContextMixin:
    """
    Mixin for advanced context generation: RAG, hierarchical context, dynamic context, etc.
    """

    async def get_enhanced_rag_context(self, query: str, context_type: str = "comprehensive", max_tokens: int = 2048) -> Dict[str, Any]:
        """
        Return a rich context for RAG including data from self-model, world-model, and knowledge graph.
        
        Args:
            query: The user query to get context for
            context_type: The type of context to generate ("comprehensive", "self", "world", "memory")
            max_tokens: Maximum total tokens in the context
            
        Returns:
            Dict with 'context' string and 'metadata' about the context composition
        """
        context_parts = []
        metadata = {"sources": []}
        
        # Check if we have the Lucidia memory components
        has_lucidia_memory = hasattr(self, 'self_model') and hasattr(self, 'world_model') and hasattr(self, 'knowledge_graph')
        
        # Allocate tokens based on context type
        token_allocation = {}
        
        if context_type == "self":
            # Prioritize self-model information
            token_allocation = {
                "self_model": int(max_tokens * 0.7),
                "memory": int(max_tokens * 0.2),
                "world_model": int(max_tokens * 0.1)
            }
        elif context_type == "world":
            # Prioritize world-model information
            token_allocation = {
                "world_model": int(max_tokens * 0.7),
                "knowledge_graph": int(max_tokens * 0.2),
                "memory": int(max_tokens * 0.1)
            }
        elif context_type == "memory":
            # Prioritize memory information
            token_allocation = {
                "memory": int(max_tokens * 0.8),
                "self_model": int(max_tokens * 0.1),
                "world_model": int(max_tokens * 0.1)
            }
        else:  # "comprehensive" (default)
            # Balanced approach
            token_allocation = {
                "memory": int(max_tokens * 0.4),
                "self_model": int(max_tokens * 0.3),
                "world_model": int(max_tokens * 0.2),
                "knowledge_graph": int(max_tokens * 0.1)
            }
        
        # Add standard memory context
        try:
            memory_context = await self.get_rag_context(
                query=query,
                max_tokens=token_allocation.get("memory", int(max_tokens * 0.4))
            )
            
            if memory_context:
                context_parts.append("### Memory Context\n" + memory_context)
                metadata["sources"].append("memory")
        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}")
        
        # Add self-model context if available
        if has_lucidia_memory and token_allocation.get("self_model", 0) > 0:
            try:
                self_context = await self.self_model.get_contextual_identity(
                    query=query,
                    max_tokens=token_allocation.get("self_model", 0)
                )
                
                if self_context:
                    context_parts.append("### Self-Awareness Context\n" + self_context)
                    metadata["sources"].append("self_model")
            except Exception as e:
                logger.error(f"Error retrieving self-model context: {e}")
        
        # Add world-model context if available
        if has_lucidia_memory and token_allocation.get("world_model", 0) > 0:
            try:
                world_context = await self.world_model.get_relevant_knowledge(
                    query=query,
                    max_tokens=token_allocation.get("world_model", 0)
                )
                
                if world_context:
                    context_parts.append("### World Knowledge\n" + world_context)
                    metadata["sources"].append("world_model")
            except Exception as e:
                logger.error(f"Error retrieving world-model context: {e}")
        
        # Add knowledge graph context if available
        if has_lucidia_memory and token_allocation.get("knowledge_graph", 0) > 0:
            try:
                kg_context = await self.knowledge_graph.query_knowledge_graph(
                    query=query,
                    max_tokens=token_allocation.get("knowledge_graph", 0)
                )
                
                if kg_context:
                    context_parts.append("### Knowledge Graph\n" + kg_context)
                    metadata["sources"].append("knowledge_graph")
            except Exception as e:
                logger.error(f"Error retrieving knowledge graph context: {e}")
        
        # Combine all context parts
        combined_context = "\n\n".join(context_parts) if context_parts else ""
        
        # Add metadata
        metadata.update({
            "total_sources": len(metadata["sources"]),
            "query": query,
            "context_type": context_type,
            "timestamp": datetime.now().isoformat()
        })
        
        return {"context": combined_context, "metadata": metadata}

    async def get_hierarchical_context(self, query: str, max_tokens: int = 1024) -> str:
        return ""

    async def generate_dynamic_context(self, query: str, conversation_history: list, max_tokens: int = 1024) -> str:
        return ""

    async def boost_context_quality(self, query: str, context_text: str, feedback: Dict[str, Any] = None) -> str:
        return context_text

    async def get_rag_context(self, query: str, limit: int = 5, max_tokens: int = 1024, min_quickrecal_score: float = 0.0, min_significance: float = None) -> str:
        """
        Get memory context for RAG (Retrieval-Augmented Generation).
        
        This method retrieves relevant memories based on the query and formats them
        into a context string that can be used for RAG with an LLM.
        
        Args:
            query: The query to find relevant memories for
            limit: Maximum number of memories to include
            max_tokens: Maximum number of tokens in the context
            min_quickrecal_score: Minimum quickrecal score threshold for memories
            min_significance: Legacy parameter for backward compatibility (deprecated)
            
        Returns:
            str: Formatted memory context for RAG
        """
        try:
            if not query or not isinstance(query, str):
                logger.warning(f"Invalid query provided to get_rag_context: {type(query)}")
                return ""
                
            # For backward compatibility, use min_significance if min_quickrecal_score is not provided
            if min_significance is not None:
                logger.warning("min_significance parameter is deprecated, use min_quickrecal_score instead")
                quickrecal_threshold = min_significance
            else:
                quickrecal_threshold = min_quickrecal_score
                
            # Normalize min_quickrecal_score to ensure it's a valid float
            try:
                quickrecal_threshold = float(quickrecal_threshold)
                # Clamp to valid range
                quickrecal_threshold = max(0.0, min(1.0, quickrecal_threshold))
            except (ValueError, TypeError):
                logger.warning(f"Invalid min_quickrecal_score value: {quickrecal_threshold}, defaulting to 0.0")
                quickrecal_threshold = 0.0
                
            # First check if this is a personal detail query
            personal_detail_patterns = {
                "name": [r"what.*name", r"who am i", r"call me", r"my name", r"what.*call me"],
                "location": [r"where.*live", r"where.*from", r"my location", r"my address", r"where.*i.*live"],
                "birthday": [r"when.*born", r"my birthday", r"my birth date", r"when.*birthday", r"how old"],
                "job": [r"what.*do for (a )?living", r"my (job|profession|occupation|career|work)", r"where.*work"],
                "family": [r"my (family|wife|husband|partner|child|children|son|daughter|mother|father)"],
            }
            
            # Initialize context parts
            context_parts = []
            personal_detail_found = False
            
            # Check if query matches any personal detail patterns
            for category, patterns in personal_detail_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        logger.info(f"Personal detail query detected in RAG for category: {category}")
                        
                        # Try to get personal detail directly
                        if hasattr(self, "get_personal_detail"):
                            try:
                                value = await self.get_personal_detail(category)
                                if value:
                                    logger.info(f"Found personal detail for RAG: {category}={value}")
                                    # Add personal detail to context parts
                                    context_parts.append(f"### User Personal Information\nThe user's {category} is: {value}\n")
                                    personal_detail_found = True
                            except Exception as e:
                                logger.error(f"Error retrieving personal detail '{category}': {e}")
            
            # If not a personal detail query or no direct match found, search memories
            logger.info(f"Searching for relevant memories for query: {query} with min_quickrecal_score={quickrecal_threshold}")
            
            # Check for semantic memory matches
            memories = []
            high_quickrecal_threshold = 0.5  # Lowered from 0.7 to include more important memories
            
            # First, try to get memories with high quickrecal score
            try:
                high_sig_memories = await self.search_memory(query, limit=limit, min_quickrecal_score=high_quickrecal_threshold)
                logger.debug(f"Found {len(high_sig_memories)} high quickrecal score memories")
            except Exception as e:
                logger.error(f"Error searching for high quickrecal score memories: {e}")
                high_sig_memories = []
            
            # If we don't have enough high quickrecal score memories, get some with lower scores
            if len(high_sig_memories) < limit:
                remaining = limit - len(high_sig_memories)
                try:
                    # Use the provided min_quickrecal_score for the second search
                    low_sig_memories = await self.search_memory(query, limit=remaining, min_quickrecal_score=quickrecal_threshold)
                    logger.debug(f"Found {len(low_sig_memories)} additional memories with min_quickrecal_score={quickrecal_threshold}")
                    
                    # Filter out any duplicates
                    low_sig_memories = [m for m in low_sig_memories if m.get("id") not in [hm.get("id") for hm in high_sig_memories]]
                    memories = high_sig_memories + low_sig_memories
                except Exception as e:
                    logger.error(f"Error searching for low quickrecal score memories: {e}")
            
            # If memory search fails, try a direct content search without semantic matching
            if not memories and not personal_detail_found:
                try:
                    # Fallback to direct content search
                    if hasattr(self, "memories"):
                        # Simple text matching fallback when semantic search fails
                        query_lower = query.lower()
                        direct_matches = []
                        
                        for mem in self.memories:
                            if query_lower in mem.get("content", "").lower():
                                direct_matches.append(mem)
                        
                        if direct_matches:
                            logger.info(f"Found {len(direct_matches)} memories by direct text matching")
                            # Sort by quickrecal_score instead of significance
                            memories = sorted(direct_matches, key=lambda x: x.get("quickrecal_score", x.get("significance", 0.0)), reverse=True)[:limit]
                except Exception as e:
                    logger.error(f"Error in fallback direct search: {e}")
            
            if not memories and not personal_detail_found:
                logger.info("No relevant memories found for RAG context")
                return ""
            
            # Add memories to context parts
            if memories:
                context_parts.append("### Relevant Memories")
                
                # Sort memories by quickrecal_score (highest first)
                try:
                    sorted_memories = sorted(memories, key=lambda x: x.get("quickrecal_score", x.get("significance", 0.0)), reverse=True)
                except Exception as e:
                    logger.error(f"Error sorting memories by quickrecal_score: {e}")
                    sorted_memories = memories  # Use unsorted if sorting fails
                
                # Add memories to context
                for i, memory in enumerate(sorted_memories):
                    try:
                        content = memory.get("content", "")
                        quickrecal_score = memory.get("quickrecal_score", memory.get("significance", 0.0))
                        timestamp = memory.get("timestamp", 0)
                        
                        # Skip empty content
                        if not content or not content.strip():
                            continue
                        
                        # Format timestamp as human-readable date if available
                        date_str = ""
                        if timestamp:
                            try:
                                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                                date_str = f" ({date_str})"
                            except Exception as e:
                                logger.warning(f"Error formatting timestamp: {e}")
                        
                        # Add memory to context with quickrecal_score indicator
                        sig_indicator = "*" * int(quickrecal_score * 5)  # 0-5 stars based on quickrecal_score
                        memory_str = f"Memory {i+1}{date_str} {sig_indicator}\n{content}\n"
                        context_parts.append(memory_str)
                    except Exception as e:
                        logger.error(f"Error processing memory for context: {e}")
                        continue
            
            # Join context parts
            context = "\n".join(context_parts)
            
            # Truncate if too long (simple approach, could be more sophisticated)
            if len(context) > max_tokens * 4:  # Rough estimate of tokens to chars
                context = context[:max_tokens * 4] + "\n[Context truncated due to length]\n"
            
            logger.info(f"Generated RAG context with {len(sorted_memories) if 'sorted_memories' in locals() else 0} memories")
            return context
            
        except Exception as e:
            logger.error(f"Error generating RAG context: {e}")
            # Return empty string on error rather than propagating the exception
            return ""
