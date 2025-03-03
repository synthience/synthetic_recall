import torch
import time
import logging
from typing import Dict, Any, List
from memory_index import MemoryIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatProcessor:
    def __init__(self, memory_index: MemoryIndex, config: Dict[str, Any] = None):
        """Initialize chat processor with memory integration."""
        self.config = {
            'max_memories': 5,
            'min_similarity': 0.7,
            'time_decay': 0.01,
            'significance_threshold': 0.5
        }
        if config:
            self.config.update(config)
            
        self.memory_index = memory_index

    async def retrieve_context(self, query_embedding: torch.Tensor) -> List[Dict]:
        """Retrieve relevant memories with significance and time weighting."""
        # Normalize query embedding
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.clone().detach()
            query_norm = torch.norm(query_embedding, p=2)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
        
        memories = self.memory_index.search(
            query_embedding,
            k=self.config['max_memories']
        )
        
        logger.info(f"Retrieved {len(memories)} memories before filtering")
        for i, m in enumerate(memories):
            logger.info(f"Memory {i}: similarity={m['similarity']:.3f}, content={m['memory'].get('content', 'None')}")
        
        # Filter by similarity threshold
        filtered_memories = [
            m for m in memories 
            if m['similarity'] >= self.config['min_similarity'] 
            and m['memory'].get('content')
        ]
        
        logger.info(f"Filtered to {len(filtered_memories)} memories")
        
        # Sort by significance and similarity
        filtered_memories.sort(
            key=lambda x: (x['memory']['significance'], x['similarity']),
            reverse=True
        )
        
        return filtered_memories

    async def process_chat(self, user_input: str, embedding: torch.Tensor) -> Dict[str, Any]:
        """Process chat with memory integration."""
        context = await self.retrieve_context(embedding)
        messages = []
        
        logger.info(f"Processing chat with {len(context)} context memories")
        
        # Add memory context if available
        if context:
            memory_texts = []
            for memory in context:
                if memory['memory'].get('content'):
                    memory_texts.append(
                        f"Previous Memory (Significance: {memory['memory']['significance']:.2f}, "
                        f"Similarity: {memory['similarity']:.2f}):\n"
                        f"{memory['memory']['content']}"
                    )
            
            if memory_texts:
                logger.info(f"Adding {len(memory_texts)} memories to context")
                messages.append({
                    "role": "system",
                    "content": "Relevant context:\n" + "\n\n".join(memory_texts)
                })
            else:
                logger.warning("No memory texts generated despite having context")
        else:
            logger.warning("No context memories retrieved")
        
        # Add user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Prepare response
        response = {
            "messages": messages,
            "model": "qwen2.5-7b-instruct",
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False
        }
        
        return response