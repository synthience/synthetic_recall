"""
LUCID RECALL PROJECT

MemoryCore: Core memory system with HPC integration
"""

import torch
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional
from pathlib import Path
import time

from ..server.hpc_flow_manager import HPCFlowManager
from .memory_types import MemoryTypes, MemoryEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryCore:
    def __init__(self, config: Dict[str, Any]):
        self.config = {
            'dimension': 768,
            'max_size': 10000,
            'batch_size': 32,
            'cleanup_threshold': 0.7,
            'memory_path': Path('/workspace/memory/stored'),
            **(config or {})
        }
        
        # Initialize memory storage
        self.memories = defaultdict(list)
        self.total_memories = 0
        
        # Initialize HPC Manager
        self.hpc_manager = HPCFlowManager(config)
        
        # Performance tracking
        self.last_cleanup_time = time.time()
        self.stats = {
            'processed': 0,
            'stored': 0,
            'cleaned': 0
        }
        
        logger.info(f"Initialized MemoryCore with config: {self.config}")
        
    async def process_and_store(self, embedding: torch.Tensor, memory_type: MemoryTypes) -> bool:
        """Process embedding through HPC pipeline and store if significant"""
        try:
            # Process through HPC pipeline
            processed_embedding, significance = await self.hpc_manager.process_embedding(embedding)
            
            self.stats['processed'] += 1
            
            # Store if significant enough
            if significance > self.config['cleanup_threshold']:
                success = self._store_memory(MemoryEntry(
                    embedding=processed_embedding,
                    memory_type=memory_type,
                    significance=significance,
                    timestamp=time.time()
                ))
                
                if success:
                    self.stats['stored'] += 1
                    
                # Run cleanup if needed
                await self._maybe_cleanup()
                
                return success
                
            return False
            
        except Exception as e:
            logger.error(f"Error in process_and_store: {str(e)}")
            return False
            
    def _store_memory(self, memory: MemoryEntry) -> bool:
        """Store a memory entry in the appropriate type bucket"""
        try:
            # Check if we have room
            if self.total_memories >= self.config['max_size']:
                return False
                
            # Add to appropriate bucket
            self.memories[memory.memory_type].append(memory)
            self.total_memories += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            return False
            
    async def _maybe_cleanup(self):
        """Run cleanup if memory usage is high"""
        current_time = time.time()
        
        # Only clean up periodically
        if (current_time - self.last_cleanup_time < 3600 and  # 1 hour
            self.total_memories < self.config['max_size'] * 0.9):  # 90% full
            return
            
        await self._cleanup()
        
    async def _cleanup(self):
        """Remove least significant memories when storage is full"""
        try:
            logger.info("Starting memory cleanup...")
            
            # Sort all memories by significance
            all_memories = []
            for type_memories in self.memories.values():
                all_memories.extend(type_memories)
                
            all_memories.sort(key=lambda x: x.significance)
            
            # Remove bottom 20%
            num_to_remove = len(all_memories) // 5
            memories_to_keep = all_memories[num_to_remove:]
            
            # Reset storage
            self.memories = defaultdict(list)
            self.total_memories = 0
            
            # Re-add memories to keep
            for memory in memories_to_keep:
                self._store_memory(memory)
                
            self.stats['cleaned'] += num_to_remove
            self.last_cleanup_time = time.time()
            
            logger.info(f"Cleanup complete. Removed {num_to_remove} memories")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            
    def get_recent_memories(self, count: int = 5, memory_type: Optional[MemoryTypes] = None) -> List[MemoryEntry]:
        """Get most recent memories, optionally filtered by type"""
        try:
            if memory_type:
                memories = self.memories[memory_type]
            else:
                memories = []
                for type_memories in self.memories.values():
                    memories.extend(type_memories)
                    
            # Sort by timestamp descending
            memories.sort(key=lambda x: x.timestamp, reverse=True)
            
            return memories[:count]
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {str(e)}")
            return []
            
    def get_stats(self) -> Dict[str, Any]:
        """Get current memory system statistics"""
        return {
            'total_memories': self.total_memories,
            'memory_types': {k: len(v) for k, v in self.memories.items()},
            'processed': self.stats['processed'],
            'stored': self.stats['stored'],
            'cleaned': self.stats['cleaned'],
            'last_cleanup': self.last_cleanup_time,
            'hpc_stats': self.hpc_manager.get_stats()
        }