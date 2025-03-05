import torch
import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemorySystem:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {
            'storage_path': Path('../memory/stored'),
            'embedding_dim': 384,
            'rebuild_threshold': 100,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **(config or {})
        }
        
        self.memories = []
        self.storage_path = Path(self.config['storage_path'])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing memories
        self._load_memories()
        logger.info(f"Initialized MemorySystem with {len(self.memories)} memories")

    def _load_memories(self):
        """Load all memories from disk."""
        self.memories = []
        try:
            for file_path in self.storage_path.glob('*.json'):
                with open(file_path, 'r') as f:
                    memory = json.load(f)
                    if isinstance(memory, dict) and 'timestamp' in memory:
                        self.memories.append(memory)
            
            # Sort by timestamp if memories exist
            if self.memories:
                self.memories.sort(key=lambda x: x.get('timestamp', 0))
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
            self.memories = []

    async def add_memory(self, text: str, embedding: torch.Tensor, 
                        significance: float = None) -> Dict[str, Any]:
        """Add memory with persistence."""
        # Normalize embedding
        embedding = self._normalize_embedding(embedding)
        
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        
        memory = {
            'id': memory_id,
            'text': text,
            'embedding': embedding.tolist(),
            'timestamp': timestamp,
            'significance': significance
        }
        
        # Add to memory list
        self.memories.append(memory)
        
        # Save to disk
        self._save_memory(memory)
        
        logger.info(f"Stored memory {memory_id} with significance {significance}")
        return memory

    async def search_memories(self, query_embedding: torch.Tensor, 
                            limit: int = 5) -> List[Dict]:
        """Search for similar memories."""
        if not self.memories:
            return []
            
        # Normalize query
        query_embedding = self._normalize_embedding(query_embedding)
        
        # Calculate similarities
        similarities = []
        for memory in self.memories:
            memory_embedding = torch.tensor(memory['embedding'], 
                                         device=self.config['device'])
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                memory_embedding.unsqueeze(0)
            )
            similarities.append({
                'memory': memory,
                'similarity': similarity.item()
            })
        
        # Sort by similarity and significance
        sorted_memories = sorted(
            similarities,
            key=lambda x: (x['similarity'] * 0.7 + 
                          (x['memory']['significance'] or 0) * 0.3),
            reverse=True
        )
        
        return sorted_memories[:limit]

    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Normalize embedding vector."""
        if isinstance(embedding, list):
            embedding = torch.tensor(embedding, device=self.config['device'])
        embedding = embedding.to(self.config['device'])
        norm = torch.norm(embedding, p=2)
        return embedding / norm if norm > 0 else embedding

    def _save_memory(self, memory: Dict[str, Any]):
        """Save memory to disk."""
        try:
            file_path = self.storage_path / f"{memory['id']}.json"
            with open(file_path, 'w') as f:
                json.dump(memory, f)
        except Exception as e:
            logger.error(f"Error saving memory {memory['id']}: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            latest_timestamp = max([m.get('timestamp', 0) for m in self.memories]) if self.memories else 0
        except Exception as e:
            logger.error(f"Error calculating latest timestamp: {str(e)}")
            latest_timestamp = 0

        return {
            'memory_count': len(self.memories),
            'device': self.config['device'],
            'storage_path': str(self.storage_path),
            'latest_timestamp': latest_timestamp
        }