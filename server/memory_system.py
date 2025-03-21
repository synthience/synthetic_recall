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
                        
                        if not isinstance(index_data, dict) or 'memories' not in index_data:
                            logger.warning(f"Invalid memory format in {index_path}")
                            memory_index = {}
                except Exception as e:
                    logger.error(f"Error loading memory index: {str(e)}")
                    memory_index = {}
            else:
                logger.warning(f"Memory index file not found at {index_path}")
                memory_index = {}
            
            # Load memory files
            memory_files = list(self.storage_path.glob('*.json'))
            memory_files = [f for f in memory_files if f.name != 'memory_index.json']
            logger.info(f"Found {len(memory_files)} memory files")
            
            for file_path in memory_files:
                try:
                    with open(file_path, 'r') as f:
                        memory = json.load(f)
                        if isinstance(memory, dict) and 'timestamp' in memory:
                            # Repair embedding if needed
                            if 'embedding' in memory:
                                embedding_repaired = False
                                
                                if isinstance(memory['embedding'], str):
                                    try:
                                        embedding_data = json.loads(memory['embedding'])
                                        if isinstance(embedding_data, list):
                                            memory['embedding'] = embedding_data
                                            embedding_repaired = True
                                            logger.info(f"Repaired string embedding for memory {memory.get('id','unknown')}")
                                    except json.JSONDecodeError:
                                        logger.warning(f"Invalid embedding string in memory {memory.get('id','unknown')}, setting placeholder")
                                        memory['embedding'] = [0.0] * self.config['embedding_dim']
                                        memory['has_placeholder_embedding'] = True
                                        embedding_repaired = True
                                
                                elif isinstance(memory['embedding'], list):
                                    if not all(isinstance(x, (int, float)) for x in memory['embedding']):
                                        logger.warning(f"Non-numeric values in embedding list for memory {memory.get('id','unknown')}, setting placeholder")
                                        memory['embedding'] = [0.0] * self.config['embedding_dim']
                                        memory['has_placeholder_embedding'] = True
                                        embedding_repaired = True
                                
                                elif memory['embedding'] is None:
                                    logger.warning(f"Null embedding in memory {memory.get('id','unknown')}, setting placeholder")
                                    memory['embedding'] = [0.0] * self.config['embedding_dim']
                                    memory['has_placeholder_embedding'] = True
                                    embedding_repaired = True
                                
                                if embedding_repaired:
                                    try:
                                        with open(file_path, 'w') as save_f:
                                            json.dump(memory, save_f, indent=2)
                                        logger.info(f"Saved repaired embedding for memory {memory.get('id','unknown')}")
                                    except Exception as save_error:
                                        logger.error(f"Failed to save repaired embedding: {save_error}")
                            
                            # Handle legacy significance field conversion if present
                            if 'significance' in memory and 'quickrecal_score' not in memory:
                                memory['quickrecal_score'] = memory['significance']
                                logger.debug(f"Converted legacy significance to quickrecal_score for memory {memory.get('id','unknown')}")
                                # Save the updated file with quickrecal_score
                                try:
                                    with open(file_path, 'w') as save_f:
                                        json.dump(memory, save_f, indent=2)
                                    logger.info(f"Saved updated memory with quickrecal_score for {memory.get('id','unknown')}")
                                except Exception as save_error:
                                    logger.error(f"Failed to save updated memory with quickrecal_score: {save_error}")
                            
                            # Load quickrecal_score, falling back to 0.5 if missing
                            self.memories.append({
                                'id': memory.get('id'),
                                'text': memory.get('text', ""),
                                'timestamp': memory.get('timestamp', time.time()),
                                'embedding': memory.get('embedding', []),
                                'metadata': memory.get('metadata', {}),
                                'quickrecal_score': memory.get('quickrecal_score', 0.5)
                            })
                            
                            memory_id = memory.get('id')
                            if memory_id and memory_id not in memory_index:
                                memory_index[memory_id] = {
                                    'path': str(file_path.relative_to(self.storage_path)),
                                    'timestamp': memory.get('timestamp', time.time()),
                                    'quickrecal_score': memory.get('quickrecal_score', 0.5)
                                }
                            
                            logger.debug(f"Loaded memory {memory.get('id','unknown')} from {file_path}")
                        else:
                            logger.warning(f"Invalid memory format in {file_path}")
                except Exception as e:
                    logger.error(f"Error loading memory file {file_path}: {str(e)}")
            
            # Update the memory index file
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
            
            if self.memories:
                self.memories.sort(key=lambda x: x.get('timestamp', 0))
                logger.info(f"Successfully loaded {len(self.memories)} memories")
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
            self.memory_index = MemoryIndex(
                embedding_dim=self.config.get('embedding_dim', 384),
                rebuild_threshold=self.config.get('rebuild_threshold', 100),
            )
            
            indexed_count = 0
            for memory in self.memories:
                if 'embedding' in memory and 'id' in memory:
                    try:
                        if isinstance(memory['embedding'], str):
                            import json
                            try:
                                embedding_data = json.loads(memory['embedding'])
                                if isinstance(embedding_data, list):
                                    embedding = torch.tensor(embedding_data, dtype=torch.float32, device='cpu')
                                else:
                                    logger.warning(f"Skipping memory {memory['id']}: invalid list after JSON parse")
                                    continue
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping memory {memory['id']}: invalid embedding JSON")
                                continue
                        elif isinstance(memory['embedding'], list):
                            if all(isinstance(x, (int, float)) for x in memory['embedding']):
                                embedding = torch.tensor(memory['embedding'], dtype=torch.float32, device='cpu')
                            else:
                                logger.warning(f"Skipping memory {memory['id']}: embedding list has non-numerics")
                                continue
                        else:
                            logger.warning(f"Skipping memory {memory['id']}: unsupported embedding type {type(memory['embedding'])}")
                            continue
                        
                        self.memory_index.add_memory_sync(
                            memory_id=memory['id'],
                            embedding=embedding,
                            timestamp=memory.get('timestamp', time.time()),
                            quickrecal_score=memory.get('quickrecal_score', 0.5),
                            content=memory.get('text', "")
                        )
                        indexed_count += 1
                    except Exception as e:
                        logger.warning(f"Error adding memory {memory.get('id','unknown')} to index: {str(e)}")
            
            if indexed_count > 0:
                self.memory_index.build_index()
                logger.info(f"Built memory index with {indexed_count} memories")
            else:
                logger.warning("No memories could be added to the index")
        except Exception as e:
            logger.error(f"Error initializing memory index: {str(e)}", exc_info=True)
            self.memory_index = None

    async def add_memory(self, text: str, embedding: torch.Tensor = None, text_embedding: str = None, 
                         quickrecal_score: float = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add a new memory to the system using HPC-QR quickrecal_score."""
        memory_id = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)
        
        # Handle embedding
        embedding_list = []
        if embedding is not None:
            if isinstance(embedding, torch.Tensor):
                embedding_list = embedding.cpu().tolist()
            elif isinstance(embedding, list):
                embedding_list = embedding
        
        # If text_embedding was given as an alternative approach (not used here but left as is)
        
        if quickrecal_score is None:
            quickrecal_score = 0.5
        
        if metadata is None:
            metadata = {}
        
        if not embedding_list and not text_embedding:
            embedding_list = [0.0] * self.config['embedding_dim']
            logger.warning("No embedding provided for memory, using placeholder")
        
        memory = {
            'id': memory_id,
            'text': text,
            'timestamp': timestamp,
            'embedding': embedding_list if embedding_list else text_embedding,
            'metadata': metadata,
            'quickrecal_score': quickrecal_score
        }
        
        self.memories.append(memory)
        self._save_memory(memory)
        
        if hasattr(self, 'memory_index') and self.memory_index is not None:
            try:
                if isinstance(embedding, list):
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device='cpu')
                else:
                    embedding_tensor = embedding.to('cpu') if embedding.device.type != 'cpu' else embedding
                    
                await self.memory_index.add_memory(
                    memory_id=memory_id,
                    embedding=embedding_tensor,
                    timestamp=timestamp,
                    quickrecal_score=quickrecal_score,
                    content=text
                )
                logger.debug(f"Added memory {memory_id} to index")
            except Exception as e:
                logger.error(f"Error adding memory to index: {str(e)}")
        
        logger.info(f"Stored memory {memory_id} with QuickRecal score {quickrecal_score}")
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
            
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            memory_copy = memory.copy()
            if 'embedding' in memory_copy and isinstance(memory_copy['embedding'], torch.Tensor):
                memory_copy['embedding'] = memory_copy['embedding'].tolist()
            
            # Check for emotion data and log it
            if 'metadata' in memory_copy:
                metadata = memory_copy.get('metadata', {})
                if 'emotions' in metadata or 'dominant_emotion' in metadata:
                    logger.info(f"Memory {memory_id} contains emotion data: {metadata.get('dominant_emotion', 'unknown')}")
                    logger.debug(f"Full emotion data: {metadata.get('emotions', {})}")
                else:
                    logger.warning(f"Memory {memory_id} does not contain any emotion data in metadata")
                    logger.debug(f"Metadata content: {metadata}")
            else:
                logger.warning(f"Memory {memory_id} does not have a metadata field")
            
            # Ensure metadata is JSON serializable
            try:
                # Test JSON serialization before saving
                json.dumps(memory_copy)
            except TypeError as e:
                logger.error(f"Memory {memory_id} contains non-JSON serializable data: {e}")
                # Try to fix the metadata by converting problematic types
                if 'metadata' in memory_copy:
                    fixed_metadata = {}
                    for key, value in memory_copy['metadata'].items():
                        if isinstance(value, (set, tuple)):
                            fixed_metadata[key] = list(value)
                        elif isinstance(value, complex):
                            fixed_metadata[key] = str(value)
                        elif hasattr(value, '__dict__'):
                            fixed_metadata[key] = str(value)
                        else:
                            # Try to serialize the value, if it fails, convert to string
                            try:
                                json.dumps({key: value})
                                fixed_metadata[key] = value
                            except:
                                fixed_metadata[key] = str(value)
                    
                    # Replace the original metadata with fixed version
                    memory_copy['metadata'] = fixed_metadata
                    logger.info(f"Fixed non-serializable data in memory {memory_id}")
                    
                    # Verify fix worked
                    try:
                        json.dumps(memory_copy)
                        logger.info(f"Serialization fix successful for memory {memory_id}")
                    except Exception as e2:
                        logger.error(f"Failed to fix serialization for memory {memory_id}: {e2}")
                        # Last resort: remove problematic metadata
                        if 'metadata' in memory_copy:
                            logger.warning(f"Removing metadata for memory {memory_id} due to serialization failure")
                            memory_copy['metadata'] = {}
            
            # Final verification before save - preserve emotion data if present
            if 'metadata' in memory_copy:
                metadata = memory_copy['metadata']
                # Ensure emotion data is preserved - if we have it in the original
                if 'metadata' in memory and 'emotions' in memory['metadata'] and 'emotions' not in metadata:
                    metadata['emotions'] = memory['metadata']['emotions']
                    logger.info(f"Restored emotions data for memory {memory_id}")
                
                if 'metadata' in memory and 'dominant_emotion' in memory['metadata'] and 'dominant_emotion' not in metadata:
                    metadata['dominant_emotion'] = memory['metadata']['dominant_emotion']
                    logger.info(f"Restored dominant_emotion data for memory {memory_id}")
            
            with open(file_path, 'w') as f:
                json.dump(memory_copy, f)
                
            logger.info(f"Successfully saved memory {memory_id} to {file_path}")
            
            # Verify the data was saved correctly by loading it back
            try:
                with open(file_path, 'r') as f:
                    saved_data = json.load(f)
                    
                if 'metadata' in saved_data:
                    saved_metadata = saved_data['metadata']
                    if 'emotions' in saved_metadata or 'dominant_emotion' in saved_metadata:
                        logger.info(f"Verified emotional data was saved correctly for memory {memory_id}")
                    else:
                        logger.warning(f"Emotional data verification failed for memory {memory_id}")
            except Exception as ve:
                logger.error(f"Error verifying saved data for memory {memory_id}: {ve}")
            
        except Exception as e:
            logger.error(f"Error saving memory {memory.get('id','unknown')}: {str(e)}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            latest_timestamp = max([m.get('timestamp', 0) for m in self.memories]) if self.memories else 0
            
            # Calculate average QuickRecal score
            if self.memories:
                avg_quickrecal = sum([m.get('quickrecal_score', 0.5) for m in self.memories]) / len(self.memories)
            else:
                avg_quickrecal = 0.0
                
            # Count memories by QuickRecal score range
            quickrecal_ranges = {
                "0.0-0.2": 0,
                "0.2-0.4": 0,
                "0.4-0.6": 0,
                "0.6-0.8": 0,
                "0.8-1.0": 0
            }
            
            for memory in self.memories:
                qr_score = memory.get('quickrecal_score', 0.5)
                if qr_score < 0.2:
                    quickrecal_ranges["0.0-0.2"] += 1
                elif qr_score < 0.4:
                    quickrecal_ranges["0.2-0.4"] += 1
                elif qr_score < 0.6:
                    quickrecal_ranges["0.4-0.6"] += 1
                elif qr_score < 0.8:
                    quickrecal_ranges["0.6-0.8"] += 1
                else:
                    quickrecal_ranges["0.8-1.0"] += 1
            
        except Exception as e:
            logger.error(f"Error calculating memory statistics: {str(e)}")
            latest_timestamp = 0
            avg_quickrecal = 0.0
            quickrecal_ranges = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}

        return {
            'memory_count': len(self.memories),
            'device': self.config['device'],
            'storage_path': str(self.storage_path),
            'latest_timestamp': latest_timestamp,
            'avg_quickrecal_score': avg_quickrecal,
            'quickrecal_distribution': quickrecal_ranges
        }

    async def search_memories(self, query_embedding: torch.Tensor, limit: int = 5, min_quickrecal: float = 0.0) -> List[Dict]:
        """
        Search for similar memories with QuickRecal filtering.
        
        Args:
            query_embedding: The embedding to search for
            limit: Maximum number of results to return
            min_quickrecal: Minimum QuickRecal score threshold
            
        Returns:
            List of matching memories with similarity scores
        """
        if not self.memories:
            return []
            
        query_embedding = self._normalize_embedding(query_embedding)
        
        if hasattr(self, 'memory_index') and self.memory_index is not None:
            try:
                if query_embedding.device.type != 'cpu':
                    query_embedding = query_embedding.to('cpu')
                    
                results = self.memory_index.search(
                    query_embedding, 
                    k=limit,
                    min_quickrecal=min_quickrecal  # Pass QuickRecal threshold to index
                )
                
                if results:
                    logger.debug(f"Found {len(results)} results using memory index")
                    return results
                else:
                    logger.warning("Memory index search returned no results, falling back to direct search")
            except Exception as e:
                logger.error(f"Error using memory index for search: {str(e)}")
        
        logger.debug("Using direct memory search")
        similarities = []
        for memory in self.memories:
            try:
                # Skip memories below QuickRecal threshold
                if memory.get('quickrecal_score', 0.0) < min_quickrecal:
                    continue
                    
                if 'embedding' not in memory:
                    continue
                
                if isinstance(memory['embedding'], str):
                    try:
                        embedding_data = json.loads(memory['embedding'])
                        if isinstance(embedding_data, list):
                            memory_embedding = torch.tensor(embedding_data, dtype=torch.float32, device=self.config['device'])
                        else:
                            continue
                    except (json.JSONDecodeError, TypeError):
                        continue
                elif isinstance(memory['embedding'], list):
                    if not all(isinstance(x, (int, float)) for x in memory['embedding']):
                        continue
                    memory_embedding = torch.tensor(memory['embedding'], device=self.config['device'])
                else:
                    continue
                
                similarity = torch.nn.functional.cosine_similarity(
                    query_embedding.to(memory_embedding.device).unsqueeze(0),
                    memory_embedding.unsqueeze(0)
                )
                
                similarities.append({
                    'memory': memory,
                    'similarity': similarity.item()
                })
            except Exception as e:
                logger.debug(f"Error calculating similarity for memory {memory.get('id','unknown')}: {str(e)}")
        
        # Sort by combined score: 70% similarity and 30% QuickRecal score
        sorted_memories = sorted(
            similarities,
            key=lambda x: (
                x['similarity'] * 0.7 + 
                x['memory'].get('quickrecal_score', 0.5) * 0.3
            ),
            reverse=True
        )
        
        return sorted_memories[:limit]
        
    async def update_memory_quickrecal(self, memory_id: str, new_quickrecal: float) -> bool:
        """
        Update the QuickRecal score of a memory.
        
        Args:
            memory_id: ID of the memory to update
            new_quickrecal: New QuickRecal score
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure score is in valid range
        new_quickrecal = max(0.0, min(1.0, new_quickrecal))
        
        # Find memory in memory list
        for memory in self.memories:
            if memory.get('id') == memory_id:
                memory['quickrecal_score'] = new_quickrecal
                
                # Update memory on disk
                self._save_memory(memory)
                
                # Update memory in index if available
                if hasattr(self, 'memory_index') and self.memory_index is not None:
                    try:
                        await self.memory_index.update_memory_quickrecal(memory_id, new_quickrecal)
                    except Exception as e:
                        logger.error(f"Error updating memory QuickRecal in index: {str(e)}")
                
                logger.info(f"Updated QuickRecal score for memory {memory_id} to {new_quickrecal}")
                return True
                
        logger.warning(f"Memory {memory_id} not found for QuickRecal update")
        return False