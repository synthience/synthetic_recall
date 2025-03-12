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
            'storage_path': Path('/app/memory/stored'),  # Use consistent Docker path
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
            
            # First check for memory index file
            index_path = self.storage_path / 'memory_index.json'
            memory_index = {}
            if index_path.exists():
                try:
                    with open(index_path, 'r') as f:
                        index_data = json.load(f)
                        logger.info(f"Loaded memory index with {len(index_data.get('memories', {}))} indexed memories")
                        memory_index = index_data.get('memories', {})
                        
                        # Check if index has expected format
                        if not isinstance(index_data, dict) or 'memories' not in index_data:
                            logger.warning(f"Invalid memory format in {index_path}")
                            # Create a valid memory index file
                            memory_index = {}
                except Exception as e:
                    logger.error(f"Error loading memory index: {str(e)}")
                    memory_index = {}
            else:
                logger.warning(f"Memory index file not found at {index_path}")
                memory_index = {}
            
            # Count memory files
            memory_files = list(self.storage_path.glob('*.json'))
            memory_files = [f for f in memory_files if f.name != 'memory_index.json']
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
                            
                            # Add to memory index if not already there
                            memory_id = memory.get('id')
                            if memory_id and memory_id not in memory_index:
                                memory_index[memory_id] = {
                                    'path': str(file_path.relative_to(self.storage_path)),
                                    'timestamp': memory.get('timestamp', time.time()),
                                    'significance': memory.get('significance', 0.5)
                                }
                                
                            logger.debug(f"Loaded memory {memory.get('id', 'unknown')} from {file_path}")
                        else:
                            logger.warning(f"Invalid memory format in {file_path}")
                except Exception as e:
                    logger.error(f"Error loading memory file {file_path}: {str(e)}")
            
            # Update the memory index file with any newly indexed memories
            try:
                with open(index_path, 'w') as f:
                    json.dump({
                        "memories": memory_index,
                        "last_updated": time.time(),
                        "count": len(memory_index)
                    }, f, indent=2)
                logger.info(f"Updated memory index with {len(memory_index)} entries")
            except Exception as e:
                logger.error(f"Error updating memory index: {str(e)}")
            
            # Sort by timestamp if memories exist
            if self.memories:
                self.memories.sort(key=lambda x: x.get('timestamp', 0))
                logger.info(f"Successfully loaded {len(self.memories)} memories")
                
                # Initialize memory embeddings for search
                self._initialize_memory_index()
            else:
                logger.info("No valid memories found")
                
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}", exc_info=True)
            self.memories = []
            
    def _initialize_memory_index(self):
        """Initialize the memory index for vector search."""
        from .memory_index import MemoryIndex
        
        try:
            # Create memory index with proper dimensionality
            self.memory_index = MemoryIndex(
                embedding_dim=self.config.get('embedding_dim', 384),
                rebuild_threshold=self.config.get('rebuild_threshold', 100),
            )
            
            # Add all existing memories to the index
            indexed_count = 0
            for memory in self.memories:
                if 'embedding' in memory and 'id' in memory:
                    try:
                        # Handle different embedding formats
                        if isinstance(memory['embedding'], str):
                            # Try to parse string as JSON
                            import json
                            try:
                                embedding_data = json.loads(memory['embedding'])
                                if isinstance(embedding_data, list):
                                    embedding = torch.tensor(embedding_data, dtype=torch.float32, device='cpu')
                                else:
                                    logger.warning(f"Skipping memory {memory['id']}: embedding is not a valid list after JSON parsing")
                                    continue
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping memory {memory['id']}: embedding string is not valid JSON")
                                continue
                        elif isinstance(memory['embedding'], list):
                            # Ensure all elements are numeric
                            if all(isinstance(x, (int, float)) for x in memory['embedding']):
                                embedding = torch.tensor(memory['embedding'], dtype=torch.float32, device='cpu')
                            else:
                                logger.warning(f"Skipping memory {memory['id']}: embedding list contains non-numeric values")
                                continue
                        else:
                            logger.warning(f"Skipping memory {memory['id']}: embedding has unsupported type {type(memory['embedding'])}")
                            continue
                            
                        # Now safely add to memory index
                        self.memory_index.add_memory_sync(
                            memory_id=memory['id'],
                            embedding=embedding,
                            timestamp=memory.get('timestamp', time.time()),
                            significance=memory.get('significance', 0.5),
                            content=memory.get('text', "")
                        )
                        indexed_count += 1
                    except Exception as e:
                        logger.warning(f"Error adding memory {memory.get('id', 'unknown')} to index: {str(e)}")
                    
            # Force index build after adding all memories
            if indexed_count > 0:
                self.memory_index.build_index()
                logger.info(f"Built memory index with {indexed_count} memories")
            else:
                logger.warning("No memories could be added to the index")
        except Exception as e:
            logger.error(f"Error initializing memory index: {str(e)}", exc_info=True)
            self.memory_index = None

    async def add_memory(self, text: str, embedding: torch.Tensor, 
                        significance: float = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add memory with persistence."""
        # Normalize embedding
        embedding = self._normalize_embedding(embedding)
        
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Check if this might be a relationship memory from the text content
        is_relationship_memory = False
        if metadata is None:
            metadata = {}
            
            # Check if the text is a JSON string and might contain relationship info
            if text.strip().startswith('{') and text.strip().endswith('}'):
                try:
                    # Try to parse as JSON
                    memory_json = json.loads(text)
                    if isinstance(memory_json, dict) and ('relationship' in memory_json or 'human_name' in memory_json):
                        is_relationship_memory = True
                        logger.info(f"Detected JSON relationship memory for {memory_json.get('human_name', 'unknown')}")
                        
                        # Extract important fields to metadata
                        for key in ['human_name', 'relationship', 'memory']:
                            if key in memory_json:
                                metadata[key] = memory_json[key]
                                
                        # If the memory field exists, use it as the text content
                        if 'memory' in memory_json:
                            text = memory_json['memory']
                except json.JSONDecodeError:
                    # Not valid JSON, that's okay
                    pass
        
        # Auto-enhance significance for relationship memories
        if is_relationship_memory or 'relationship' in metadata or 'human_name' in metadata:
            if significance is None or significance < 0.9:
                logger.info(f"Boosting significance for relationship memory to 0.95")
                significance = 0.95
        
        # Default significance if not provided
        if significance is None:
            significance = 0.5
        
        # Ensure embedding is in the correct format for storage
        embedding_for_storage = None
        if isinstance(embedding, torch.Tensor):
            embedding_for_storage = embedding.cpu().tolist()
        elif isinstance(embedding, list):
            # Ensure all elements are numeric
            if all(isinstance(x, (int, float)) for x in embedding):
                embedding_for_storage = embedding
            else:
                logger.warning(f"Invalid embedding format: list contains non-numeric values")
                embedding_for_storage = [0.0] * self.config.get('embedding_dim', 384)  # Default to zero vector
        else:
            logger.warning(f"Invalid embedding type: {type(embedding)}")
            embedding_for_storage = [0.0] * self.config.get('embedding_dim', 384)  # Default to zero vector
        
        memory = {
            'id': memory_id,
            'text': text,
            'embedding': embedding_for_storage,
            'timestamp': timestamp,
            'significance': significance
        }
        
        # Add metadata if present
        if metadata:
            memory['metadata'] = metadata
        
        # Add to memory list
        self.memories.append(memory)
        
        # Save to disk
        self._save_memory(memory)
        
        # Update memory index if available
        if hasattr(self, 'memory_index') and self.memory_index is not None:
            try:
                # Convert embedding back to tensor if needed
                if isinstance(embedding, list):
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device='cpu')
                else:
                    embedding_tensor = embedding.to('cpu') if embedding.device.type != 'cpu' else embedding
                    
                await self.memory_index.add_memory(
                    memory_id=memory_id,
                    embedding=embedding_tensor,
                    timestamp=timestamp,
                    significance=significance,
                    content=text
                )
                logger.debug(f"Added memory {memory_id} to index")
            except Exception as e:
                logger.error(f"Error adding memory to index: {str(e)}")
        
        logger.info(f"Stored memory {memory_id} with significance {significance}")
        return memory

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

    async def search_memories(self, query_embedding: torch.Tensor, 
                            limit: int = 5) -> List[Dict]:
        """Search for similar memories."""
        if not self.memories:
            return []
            
        # Normalize query
        query_embedding = self._normalize_embedding(query_embedding)
        
        # Use memory index if available for efficient search
        if hasattr(self, 'memory_index') and self.memory_index is not None:
            try:
                # Convert query to CPU if needed
                if query_embedding.device.type != 'cpu':
                    query_embedding = query_embedding.to('cpu')
                    
                # Use memory index for search
                results = self.memory_index.search(query_embedding, k=limit)
                
                # Check if we got results back
                if results:
                    logger.debug(f"Found {len(results)} results using memory index")
                    return results
                else:
                    logger.warning("Memory index search returned no results, falling back to direct search")
            except Exception as e:
                logger.error(f"Error using memory index for search: {str(e)}")
                # Continue with direct search as fallback
        
        # Direct search as fallback
        logger.debug("Using direct memory search")
        similarities = []
        for memory in self.memories:
            try:
                # Skip memories with invalid embeddings
                if 'embedding' not in memory:
                    continue
                    
                # Handle different embedding formats
                if isinstance(memory['embedding'], str):
                    # Try to parse string as JSON
                    try:
                        embedding_data = json.loads(memory['embedding'])
                        if isinstance(embedding_data, list):
                            memory_embedding = torch.tensor(embedding_data, dtype=torch.float32, device='cpu')
                        else:
                            continue
                    except (json.JSONDecodeError, TypeError):
                        continue
                elif isinstance(memory['embedding'], list):
                    # Check if all elements are numeric
                    if not all(isinstance(x, (int, float)) for x in memory['embedding']):
                        continue
                    memory_embedding = torch.tensor(memory['embedding'], device=self.config['device'])
                else:
                    continue
                
                # Calculate similarity
                similarity = torch.nn.functional.cosine_similarity(
                    query_embedding.to(memory_embedding.device).unsqueeze(0),
                    memory_embedding.unsqueeze(0)
                )
                
                similarities.append({
                    'memory': memory,
                    'similarity': similarity.item()
                })
            except Exception as e:
                logger.debug(f"Error calculating similarity for memory {memory.get('id', 'unknown')}: {str(e)}")
        
        # Sort by similarity and significance
        sorted_memories = sorted(
            similarities,
            key=lambda x: (x['similarity'] * 0.7 + 
                          (x['memory'].get('significance', 0) or 0) * 0.3),
            reverse=True
        )
        
        return sorted_memories[:limit]