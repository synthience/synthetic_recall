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

    async def get_enhanced_rag_context(self, query: str, context_type: str = "auto", max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Return a dict with 'context' and 'metadata' for the given query.
        """
        return {"context": "", "metadata": {}}

    async def get_hierarchical_context(self, query: str, max_tokens: int = 1024) -> str:
        return ""

    async def generate_dynamic_context(self, query: str, conversation_history: list, max_tokens: int = 1024) -> str:
        return ""

    async def boost_context_quality(self, query: str, context_text: str, feedback: Dict[str, Any] = None) -> str:
        return context_text

    async def get_rag_context(self, query: str, limit: int = 5, max_tokens: int = 1024, min_significance: float = 0.0) -> str:
        """
        Get memory context for RAG (Retrieval-Augmented Generation).
        
        This method retrieves relevant memories based on the query and formats them
        into a context string that can be used for RAG with an LLM.
        
        Args:
            query: The query to find relevant memories for
            limit: Maximum number of memories to include
            max_tokens: Maximum number of tokens in the context
            min_significance: Minimum significance threshold for memories
            
        Returns:
            str: Formatted memory context for RAG
        """
        try:
            if not query or not isinstance(query, str):
                logger.warning(f"Invalid query provided to get_rag_context: {type(query)}")
                return ""
                
            # Normalize min_significance to ensure it's a valid float
            try:
                min_significance = float(min_significance)
                # Clamp to valid range
                min_significance = max(0.0, min(1.0, min_significance))
            except (ValueError, TypeError):
                logger.warning(f"Invalid min_significance value: {min_significance}, defaulting to 0.0")
                min_significance = 0.0
                
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
            logger.info(f"Searching for relevant memories for query: {query} with min_significance={min_significance}")
            
            # First, try to get memories with high significance
            high_sig_threshold = max(min_significance, 0.7)  # Use the higher of provided threshold or 0.7
            try:
                high_sig_memories = await self.search_memory(query, limit=limit, min_significance=high_sig_threshold)
                logger.debug(f"Found {len(high_sig_memories)} high significance memories")
            except Exception as e:
                logger.error(f"Error searching for high significance memories: {e}")
                high_sig_memories = []
            
            # If we don't have enough high significance memories, get some with lower significance
            memories = high_sig_memories
            if len(high_sig_memories) < limit:
                remaining = limit - len(high_sig_memories)
                try:
                    # Use the provided min_significance for the second search
                    low_sig_memories = await self.search_memory(query, limit=remaining, min_significance=min_significance)
                    logger.debug(f"Found {len(low_sig_memories)} additional memories with min_significance={min_significance}")
                    
                    # Filter out any duplicates
                    low_sig_memories = [m for m in low_sig_memories if m.get("id") not in [hm.get("id") for hm in high_sig_memories]]
                    memories = high_sig_memories + low_sig_memories
                except Exception as e:
                    logger.error(f"Error searching for low significance memories: {e}")
            
            if not memories and not personal_detail_found:
                logger.info("No relevant memories found for RAG context")
                return ""
            
            # Add memories to context parts
            if memories:
                context_parts.append("### Relevant Memories")
                
                # Sort memories by significance (highest first)
                try:
                    sorted_memories = sorted(memories, key=lambda x: x.get("significance", 0.0), reverse=True)
                except Exception as e:
                    logger.error(f"Error sorting memories by significance: {e}")
                    sorted_memories = memories  # Use unsorted if sorting fails
                
                # Add memories to context
                for i, memory in enumerate(sorted_memories):
                    try:
                        content = memory.get("content", "")
                        significance = memory.get("significance", 0.0)
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
                        
                        # Add memory to context with significance indicator
                        sig_indicator = "*" * int(significance * 5)  # 0-5 stars based on significance
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
