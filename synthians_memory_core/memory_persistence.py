# synthians_memory_core/memory_persistence.py

import os
import json
import logging
import asyncio
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
import aiofiles # Use aiofiles for async file operations
import uuid
from .memory_structures import MemoryEntry # Use the unified structure
from .custom_logger import logger # Use the shared custom logger

class MemoryPersistence:
    """Handles disk-based memory operations with robustness."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'storage_path': Path('/app/memory/stored'), # Consistent Docker path
            'backup_dir': 'backups',
            'index_filename': 'memory_index.json',
            'max_backups': 5,
            'safe_write': True, # Use atomic writes
            **(config or {})
        }
        self.storage_path = Path(self.config['storage_path'])
        self.backup_path = self.storage_path / self.config['backup_dir']
        self.index_path = self.storage_path / self.config['index_filename']
        self.memory_index: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.stats = {'saves': 0, 'loads': 0, 'deletes': 0, 'backups': 0, 'errors': 0}

        # Ensure directories exist
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.backup_path.mkdir(exist_ok=True)
        except Exception as e:
             logger.error("MemoryPersistence", "Failed to create storage directories", {"path": self.storage_path, "error": str(e)})
             raise # Initialization failure is critical

        # Load index on init
        asyncio.create_task(self._load_index()) # Load index in background

        logger.info("MemoryPersistence", "Initialized", {"storage_path": str(self.storage_path)})

    async def _load_index(self):
        """Load the memory index from disk."""
        async with self._lock:
             if not self.index_path.exists():
                 logger.info("MemoryPersistence", "Memory index file not found, starting fresh.", {"path": str(self.index_path)})
                 self.memory_index = {}
                 return

             try:
                 async with aiofiles.open(self.index_path, 'r') as f:
                     content = await f.read()
                     loaded_index = json.loads(content)
                 # Basic validation
                 if isinstance(loaded_index, dict):
                     self.memory_index = loaded_index
                     logger.info("MemoryPersistence", f"Loaded memory index with {len(self.memory_index)} entries.", {"path": str(self.index_path)})
                 else:
                      logger.error("MemoryPersistence", "Invalid index file format, starting fresh.", {"path": str(self.index_path)})
                      self.memory_index = {}
             except Exception as e:
                 logger.error("MemoryPersistence", "Error loading memory index, starting fresh.", {"path": str(self.index_path), "error": str(e)})
                 self.memory_index = {} # Start fresh on error

    async def _save_index(self):
        """Save the memory index to disk atomically."""
        async with self._lock:
             temp_path = self.index_path.with_suffix('.tmp')
             try:
                 async with aiofiles.open(temp_path, 'w') as f:
                     await f.write(json.dumps(self.memory_index, indent=2))
                 await asyncio.to_thread(os.replace, temp_path, self.index_path)
                 self.stats['last_index_update'] = time.time()
             except Exception as e:
                 logger.error("MemoryPersistence", "Error saving memory index", {"path": str(self.index_path), "error": str(e)})
                 # Attempt to remove potentially corrupted temp file
                 if await asyncio.to_thread(os.path.exists, temp_path):
                      try: await asyncio.to_thread(os.remove, temp_path)
                      except Exception: pass

    async def save_memory(self, memory: MemoryEntry) -> bool:
        """Save a single memory entry to disk."""
        try:
            # Create a unique ID if one doesn't exist
            if not hasattr(memory, 'id') or memory.id is None:
                memory.id = f"mem_{uuid.uuid4().hex[:12]}"
            
            # Ensure the storage directory exists
            memory_dir = self.storage_path 
            memory_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate a filename based on the memory ID
            file_path = memory_dir / f"{memory.id}.json"
            
            # Convert the memory to a serializable dict
            memory_dict = memory.to_dict()

            # Write the memory to disk
            async with aiofiles.open(file_path, 'w') as f:
                # Ensure complex numbers or other non-serializables are handled
                def default_serializer(obj):
                     if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                         np.int16, np.int32, np.int64, np.uint8,
                                         np.uint16, np.uint32, np.uint64)):
                         return int(obj)
                     elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                         return float(obj)
                     elif isinstance(obj, (np.ndarray,)): # Handle complex arrays if needed
                         return obj.tolist()
                     elif isinstance(obj, set):
                         return list(obj)
                     try:
                          # Fallback for other types
                          return str(obj)
                     except:
                          return "[Unserializable Object]"
                await f.write(json.dumps(memory_dict, indent=2, default=default_serializer))
            
            # Update the memory index
            self.memory_index[memory.id] = {
                'path': str(file_path.relative_to(self.storage_path)),
                'timestamp': memory.timestamp if hasattr(memory, 'timestamp') else time.time(),
                'quickrecal': memory.quickrecal_score if hasattr(memory, 'quickrecal_score') else 0.5,
                'type': 'memory'  # Default type since memory_type doesn't exist
            }
            
            # Save the memory index
            await self._save_index()
            
            self.stats['saves'] += 1
            self.stats['successful_saves'] = self.stats.get('successful_saves', 0) + 1
            return True
        except Exception as e:
            logger.error("MemoryPersistence", f"Error saving memory {getattr(memory, 'id', 'unknown')}: {str(e)}")
            self.stats['saves'] += 1
            self.stats['failed_saves'] = self.stats.get('failed_saves', 0) + 1
            return False

    async def load_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Load a single memory entry from disk."""
        async with self._lock:
             try:
                 if memory_id not in self.memory_index:
                     # Fallback: check filesystem directly (maybe index is outdated)
                     file_path = self.storage_path / f"{memory_id}.json"
                     if not await asyncio.to_thread(os.path.exists, file_path):
                          logger.warning("MemoryPersistence", f"Memory {memory_id} not found in index or filesystem.")
                          return None
                     # If found directly, update index info
                     self.memory_index[memory_id] = {'path': f"{memory_id}.json"}
                 else:
                     file_path = self.storage_path / self.memory_index[memory_id]['path']

                 # Check primary path first
                 if not await asyncio.to_thread(os.path.exists, file_path):
                      # Try backup path
                      backup_path = file_path.with_suffix('.bak')
                      if await asyncio.to_thread(os.path.exists, backup_path):
                           logger.warning("MemoryPersistence", f"Using backup file for {memory_id}", {"path": str(backup_path)})
                           file_path = backup_path
                      else:
                           logger.error("MemoryPersistence", f"Memory file not found for {memory_id}", {"path": str(file_path)})
                           # Remove from index if file is missing
                           if memory_id in self.memory_index: del self.memory_index[memory_id]
                           return None

                 async with aiofiles.open(file_path, 'r') as f:
                     content = await f.read()
                     memory_dict = json.loads(content)

                 memory = MemoryEntry.from_dict(memory_dict)
                 self.stats['loads'] = self.stats.get('loads', 0) + 1
                 self.stats['successful_loads'] = self.stats.get('successful_loads', 0) + 1
                 return memory

             except Exception as e:
                 logger.error("MemoryPersistence", f"Error loading memory {memory_id}", {"error": str(e)})
                 self.stats['loads'] = self.stats.get('loads', 0) + 1
                 self.stats['failed_loads'] = self.stats.get('failed_loads', 0) + 1
                 # Attempt recovery from backup if primary load failed
                 backup_path = self.storage_path / f"{memory_id}.json.bak"
                 if await asyncio.to_thread(os.path.exists, backup_path):
                      try:
                           logger.info("MemoryPersistence", f"Attempting recovery from backup for {memory_id}")
                           async with aiofiles.open(backup_path, 'r') as f:
                                content = await f.read()
                                memory_dict = json.loads(content)
                           memory = MemoryEntry.from_dict(memory_dict)
                           # Restore backup to primary file
                           await asyncio.to_thread(shutil.copy2, backup_path, self.storage_path / f"{memory_id}.json")
                           logger.info("MemoryPersistence", f"Successfully recovered {memory_id} from backup.")
                           return memory
                      except Exception as e_rec:
                           logger.error("MemoryPersistence", f"Backup recovery failed for {memory_id}", {"error": str(e_rec)})
                 return None

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory file from disk."""
        async with self._lock:
             try:
                 if memory_id not in self.memory_index:
                     # Check filesystem directly as fallback
                     file_path_direct = self.storage_path / f"{memory_id}.json"
                     if await asyncio.to_thread(os.path.exists, file_path_direct):
                         await asyncio.to_thread(os.remove, file_path_direct)
                         logger.info("MemoryPersistence", f"Deleted memory file directly {memory_id} (was not in index)")
                         self.stats['deletes'] = self.stats.get('deletes', 0) + 1
                         return True
                     logger.warning("MemoryPersistence", f"Memory {memory_id} not found for deletion.")
                     return False

                 file_path = self.storage_path / self.memory_index[memory_id]['path']
                 backup_path = file_path.with_suffix('.bak')

                 deleted = False
                 if await asyncio.to_thread(os.path.exists, file_path):
                     await asyncio.to_thread(os.remove, file_path)
                     deleted = True
                 if await asyncio.to_thread(os.path.exists, backup_path):
                     await asyncio.to_thread(os.remove, backup_path)
                     deleted = True # Mark deleted even if only backup existed

                 if deleted:
                     del self.memory_index[memory_id]
                     await self._save_index() # Update index after deletion
                     self.stats['deletes'] = self.stats.get('deletes', 0) + 1
                     return True
                 else:
                      # File didn't exist, remove from index anyway
                      del self.memory_index[memory_id]
                      await self._save_index()
                      return False # Indicate file wasn't actually deleted

             except Exception as e:
                 logger.error("MemoryPersistence", f"Error deleting memory {memory_id}", {"error": str(e)})
                 self.stats['errors'] = self.stats.get('errors', 0) + 1
                 return False

    async def load_all(self) -> List[MemoryEntry]:
        """Load all memories listed in the index."""
        all_memories = []
        memory_ids = list(self.memory_index.keys())
        logger.info("MemoryPersistence", f"Loading all {len(memory_ids)} memories from index.")

        # Consider batching if loading many memories
        batch_size = 100
        for i in range(0, len(memory_ids), batch_size):
             batch_ids = memory_ids[i:i+batch_size]
             load_tasks = [self.load_memory(mid) for mid in batch_ids]
             results = await asyncio.gather(*load_tasks)
             all_memories.extend(mem for mem in results if mem is not None)
             await asyncio.sleep(0.01) # Yield control between batches

        logger.info("MemoryPersistence", f"Finished loading {len(all_memories)} memories.")
        return all_memories

    async def create_backup(self) -> bool:
        """Create a timestamped backup of the memory storage."""
        async with self._lock:
             try:
                 timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                 backup_instance_path = self.backup_path / f"backup_{timestamp}"
                 # Use shutil.copytree for directory backup
                 await asyncio.to_thread(shutil.copytree, self.storage_path, backup_instance_path, ignore=shutil.ignore_patterns('backups'))
                 self.stats['last_backup'] = time.time()
                 self.stats['backup_count'] = self.stats.get('backup_count', 0) + 1
                 logger.info("MemoryPersistence", f"Created backup at {backup_instance_path}")
                 await self._prune_backups()
                 return True
             except Exception as e:
                 logger.error("MemoryPersistence", "Error creating backup", {"error": str(e)})
                 self.stats['errors'] = self.stats.get('errors', 0) + 1
                 return False

    async def _prune_backups(self):
        """Remove old backups, keeping only the most recent ones."""
        try:
             backups = sorted(
                 [d for d in self.backup_path.iterdir() if d.is_dir() and d.name.startswith('backup_')],
                 key=lambda d: d.stat().st_mtime
             )
             num_to_keep = self.config['max_backups']
             if len(backups) > num_to_keep:
                 for old_backup in backups[:-num_to_keep]:
                     await asyncio.to_thread(shutil.rmtree, old_backup)
                     logger.info("MemoryPersistence", f"Pruned old backup {old_backup.name}")
        except Exception as e:
            logger.error("MemoryPersistence", "Error pruning backups", {"error": str(e)})

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        asyncio.create_task(self._save_index()) # Ensure index is saved before getting stats
        return {
            "total_indexed_memories": len(self.memory_index),
            "last_index_update": self.stats.get('last_index_update', 0),
            "saves": self.stats.get('saves', 0),
            "successful_saves": self.stats.get('successful_saves', 0),
            "failed_saves": self.stats.get('failed_saves', 0),
            "loads": self.stats.get('loads', 0),
            "successful_loads": self.stats.get('successful_loads', 0),
            "failed_loads": self.stats.get('failed_loads', 0),
            "deletes": self.stats.get('deletes', 0),
            "backups": self.stats.get('backup_count', 0),
            "last_backup": self.stats.get('last_backup', 0),
            "errors": self.stats.get('errors', 0)
        }
