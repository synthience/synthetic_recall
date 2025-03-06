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
            'storage_path': Path.cwd() / 'memory/stored',  # Use absolute path
            'embedding_dim': 384,
            'rebuild_threshold': 100,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **(config or {})
        }
        
        self.memories = []
        self.storage_path = Path(self.config['storage_path'])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Log the actual storage path being used
        logger.info(f"Memory storage path: {self.storage_path.absolute()}")
        
        # Load existing memories
        self._load_memories()
        logger.info(f"Initialized MemorySystem with {len(self.memories)} memories")

    def _load_memories(self):
        """Load all memories from disk."""
        self.memories = []
        try:
            logger.info(f"Loading memories from {self.storage_path}")
            
            # Check if directory exists
            if not self.storage_path.exists():
                logger.warning(f"Memory storage path does not exist: {self.storage_path}")
                self.storage_path.mkdir(parents=True, exist_ok=True)
                return
            
            # Count memory files
            memory_files = list(self.storage_path.glob('*.json'))
            logger.info(f"Found {len(memory_files)} memory files")
            
            # Load each file
            for file_path in memory_files:
                try:
                    with open(file_path, 'r') as f:
                        memory = json.load(f)
                        if isinstance(memory, dict) and 'timestamp' in memory:
                            # Convert embedding from list to tensor if needed
                            if 'embedding' in memory and isinstance(memory['embedding'], list):
                                memory['embedding'] = memory['embedding']  # Keep as list for now
                            
                            self.memories.append(memory)
                            logger.debug(f"Loaded memory {memory.get('id', 'unknown')} from {file_path}")
                        else:
                            logger.warning(f"Invalid memory format in {file_path}")
                except Exception as e:
                    logger.error(f"Error loading memory file {file_path}: {str(e)}")
            
            # Sort by timestamp if memories exist
            if self.memories:
                self.memories.sort(key=lambda x: x.get('timestamp', 0))
                logger.info(f"Successfully loaded {len(self.memories)} memories")
            else:
                logger.info("No valid memories found")
                
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}", exc_info=True)
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
            memory_id = memory.get('id')
            if not memory_id:
                logger.warning("Cannot save memory without an ID")
                return
                
            file_path = self.storage_path / f"{memory_id}.json"
            logger.debug(f"Saving memory {memory_id} to {file_path}")
            
            # Ensure the directory exists
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Make a copy to avoid modifying the original
            memory_copy = memory.copy()
            
            # Convert any tensor to list for JSON serialization
            if 'embedding' in memory_copy and isinstance(memory_copy['embedding'], torch.Tensor):
                memory_copy['embedding'] = memory_copy['embedding'].tolist()
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(memory_copy, f)
                
            logger.info(f"Successfully saved memory {memory_id} to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving memory {memory.get('id', 'unknown')}: {str(e)}", exc_info=True)

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