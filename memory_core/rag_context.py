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

    async def get_rag_context(self, query: str, limit: int = 5, max_tokens: int = 1024) -> str:
        """
        Get memory context for RAG (Retrieval-Augmented Generation).
        
        This method retrieves relevant memories based on the query and formats them
        into a context string that can be used for RAG with an LLM.
        
        Args:
            query: The query to find relevant memories for
            limit: Maximum number of memories to include
            max_tokens: Maximum number of tokens in the context
            
        Returns:
            str: Formatted memory context for RAG
        """
        try:
            # First check if this is a personal detail query
            personal_detail_patterns = {
                "name": [r"what.*name", r"who am i", r"call me", r"my name", r"what.*call me"],
                "location": [r"where.*live", r"where.*from", r"my location", r"my address", r"where.*i.*live"],
                "birthday": [r"when.*born", r"my birthday", r"my birth date", r"when.*birthday", r"how old"],
                "job": [r"what.*do for (a )?living", r"my (job|profession|occupation|career|work)", r"where.*work"],
                "family": [r"my (family|wife|husband|partner|child|children|son|daughter|mother|father)"],
            }
            
            # Check if query matches any personal detail patterns
            for category, patterns in personal_detail_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        logger.info(f"Personal detail query detected in RAG for category: {category}")
                        
                        # Try to get personal detail directly
                        if hasattr(self, "get_personal_detail"):
                            value = await self.get_personal_detail(category)
                            if value:
                                logger.info(f"Found personal detail for RAG: {category}={value}")
                                # Return a formatted context with the personal detail
                                return f"### User Personal Information\nThe user's {category} is: {value}\n\n### Relevant Memories\n"
            
            # If not a personal detail query or no direct match found, search memories
            logger.info(f"Searching for relevant memories for query: {query}")
            
            # First, try to get memories with high significance
            high_sig_memories = await self.search_memory(query, limit=limit, min_significance=0.7)
            
            # If we don't have enough high significance memories, get some with lower significance
            if len(high_sig_memories) < limit:
                remaining = limit - len(high_sig_memories)
                low_sig_memories = await self.search_memory(query, limit=remaining, min_significance=0.0)
                # Filter out any duplicates
                low_sig_memories = [m for m in low_sig_memories if m.get("id") not in [hm.get("id") for hm in high_sig_memories]]
                memories = high_sig_memories + low_sig_memories
            else:
                memories = high_sig_memories
            
            if not memories:
                logger.info("No relevant memories found for RAG context")
                return ""
            
            # Format memories into context string
            context_parts = ["### Relevant Memories"]
            
            # Sort memories by significance (highest first)
            sorted_memories = sorted(memories, key=lambda x: x.get("significance", 0.0), reverse=True)
            
            # Add memories to context
            for i, memory in enumerate(sorted_memories):
                content = memory.get("content", "")
                significance = memory.get("significance", 0.0)
                timestamp = memory.get("timestamp", 0)
                
                # Skip empty content
                if not content.strip():
                    continue
                
                # Format timestamp as human-readable date if available
                date_str = ""
                if timestamp:
                    date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                    date_str = f" ({date_str})"
                
                # Add memory to context with significance indicator
                sig_indicator = "*" * int(significance * 5)  # 0-5 stars based on significance
                memory_str = f"Memory {i+1}{date_str} {sig_indicator}\n{content}\n"
                context_parts.append(memory_str)
            
            # Join context parts
            context = "\n".join(context_parts)
            
            # Truncate if too long (simple approach, could be more sophisticated)
            if len(context) > max_tokens * 4:  # Rough estimate of tokens to chars
                context = context[:max_tokens * 4] + "\n[Context truncated due to length]\n"
            
            logger.info(f"Generated RAG context with {len(sorted_memories)} memories")
            return context
            
        except Exception as e:
            logger.error(f"Error generating RAG context: {e}")
            return ""
