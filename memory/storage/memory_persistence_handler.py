"""
LUCID RECALL PROJECT
Memory Persistence Handler

Manages disk-based memory operations with robust error handling
and data integrity protection.
"""

import os
import json
import logging
import asyncio
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime
from ..lucidia_memory_system.core.memory_types import MemoryEntry, MemoryTypes

logger = logging.getLogger(__name__)

class MemoryPersistenceHandler:
    """
    Handles disk-based memory saving, loading, and backup operations.
    
    Features:
    - Atomic write operations to prevent data corruption
    - Automatic backups
    - Error recovery
    - Concurrent access protection
    """
    
    def __init__(self, storage_path: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the persistence handler.
        
        Args:
            storage_path: Base path for memory storage
            config: Optional configuration parameters
        """
        self.storage_path = Path(storage_path)
        self.config = {
            'auto_backup': True,                # Enable automatic backups
            'backup_interval': 86400,           # Seconds between automatic backups (1 day)
            'max_backups': 7,                   # Maximum number of backups to keep
            'backup_dir': 'backups',            # Subdirectory for backups
            'index_filename': 'memory_index.json',  # Filename for memory index
            'batch_size': 100,                  # Number of memories to process in a batch
            'safe_write': True,                 # Use atomic write operations
            'compression': False,               # Whether to compress memory files (not implemented)
            **(config or {})
        }
        
        # Create main storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create backup directory
        self.backup_path = self.storage_path / self.config['backup_dir']
        self.backup_path.mkdir(exist_ok=True)
        
        # Path to memory index
        self.index_path = self.storage_path / self.config['index_filename']
        
        # Thread safety
        self._persistence_lock = asyncio.Lock()
        
        # Memory index (memory_id -> metadata)
        self.memory_index = self._load_memory_index()
        
        # Stats
        self.stats = {
            'saves': 0,
            'successful_saves': 0,
            'failed_saves': 0,
            'loads': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'last_backup': 0,
            'last_index_update': 0,
            'backup_count': 0
        }
        
        logger.info(f"Memory persistence handler initialized at {self.storage_path}")
        
        # Check if backup is needed
        self._check_backup_needed()
    
    def _load_memory_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load memory index from disk.
        
        Returns:
            Dict mapping memory IDs to metadata
        """
        if not self.index_path.exists():
            return {}
            
        try:
            with open(self.index_path, 'r') as f:
                index = json.load(f)
            return index
        except Exception as e:
            logger.error(f"Error loading memory index: {e}")
            
            # Try to restore from backup
            backup_index_path = self.backup_path / self.config['index_filename']
            if backup_index_path.exists():
                try:
                    with open(backup_index_path, 'r') as f:
                        index = json.load(f)
                    logger.info("Restored memory index from backup")
                    return index
                except Exception as backup_error:
                    logger.error(f"Error loading backup index: {backup_error}")
            
            return {}
    
    async def _save_memory_index(self) -> bool:
        """
        Save memory index to disk with error protection.
        
        Returns:
            Success status
        """
        temp_path = self.index_path.with_suffix('.tmp')
        backup_path = self.backup_path / self.config['index_filename']
        
        try:
            # Save to temporary file first
            with open(temp_path, 'w') as f:
                json.dump(self.memory_index, f, indent=2)
                
            # Backup existing index if it exists
            if self.index_path.exists():
                try:
                    shutil.copy2(self.index_path, backup_path)
                except Exception as e:
                    logger.warning(f"Failed to backup memory index: {e}")
            
            # Atomic rename
            os.replace(temp_path, self.index_path)
            
            # Update stats
            self.stats['last_index_update'] = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving memory index: {e}")
            
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                    
            return False
    
    def _check_backup_needed(self) -> bool:
        """
        Check if backup is needed based on interval.
        
        Returns:
            True if backup is needed
        """
        if not self.config['auto_backup']:
            return False
            
        current_time = time.time()
        time_since_backup = current_time - self.stats['last_backup']
        
        if time_since_backup >= self.config['backup_interval']:
            # Schedule backup
            asyncio.create_task(self.create_backup())
            return True
            
        return False
    
    async def save_memory(self, memory: MemoryEntry) -> bool:
        """
        Save a memory to disk.
        
        Args:
            memory: Memory entry to save
            
        Returns:
            Success status
        """
        async with self._persistence_lock:
            self.stats['saves'] += 1
            
            try:
                memory_id = memory.id
                memory_type = memory.memory_type
                
                # Create type-specific directory
                type_dir = self.storage_path / memory_type.value
                type_dir.mkdir(exist_ok=True)
                
                # Determine file path
                file_path = type_dir / f"{memory_id}.json"
                temp_path = file_path.with_suffix('.tmp')
                backup_path = file_path.with_suffix('.bak')
                
                # Convert memory to dictionary
                memory_dict = memory.to_dict()
                
                # Safe write with atomic operations
                if self.config['safe_write']:
                    # Write to temp file first
                    with open(temp_path, 'w') as f:
                        json.dump(memory_dict, f, indent=2)
                        
                    # Backup existing file if it exists
                    if file_path.exists():
                        try:
                            shutil.copy2(file_path, backup_path)
                        except Exception as e:
                            logger.warning(f"Failed to backup memory file: {e}")
                    
                    # Atomic rename
                    os.replace(temp_path, file_path)
                    
                else:
                    # Direct write (not recommended)
                    with open(file_path, 'w') as f:
                        json.dump(memory_dict, f, indent=2)
                
                # Update memory index
                self.memory_index[memory_id] = {
                    'path': str(file_path),
                    'type': memory_type.value,
                    'timestamp': memory.timestamp,
                    'significance': memory.significance,
                    'access_count': memory.access_count,
                    'last_access': memory.last_access
                }
                
                # Save index periodically
                if self.stats['saves'] % 10 == 0:  # Save every 10 memories
                    await self._save_memory_index()
                
                self.stats['successful_saves'] += 1
                
                # Check if backup is needed
                self._check_backup_needed()
                
                return True
                
            except Exception as e:
                logger.error(f"Error saving memory: {e}")
                self.stats['failed_saves'] += 1
                return False
    
    async def load_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Load a memory from disk.
        
        Args:
            memory_id: ID of memory to load
            
        Returns:
            Memory entry or None if not found
        """
        self.stats['loads'] += 1
        
        try:
            # Check if memory exists in index
            if memory_id not in self.memory_index:
                logger.warning(f"Memory {memory_id} not found in index")
                return None
                
            # Get file path from index
            memory_info = self.memory_index[memory_id]
            file_path = Path(memory_info['path'])
            
            # Check if file exists
            if not file_path.exists():
                # Try backup file
                backup_path = file_path.with_suffix('.bak')
                if backup_path.exists():
                    file_path = backup_path
                else:
                    logger.warning(f"Memory file for {memory_id} not found at {file_path}")
                    return None
            
            # Read memory file
            with open(file_path, 'r') as f:
                memory_dict = json.load(f)
                
            # Create memory entry
            memory = MemoryEntry.from_dict(memory_dict)
            
            # Update access stats
            memory.record_access()
            
            # Update index
            self.memory_index[memory_id]['access_count'] = memory.access_count
            self.memory_index[memory_id]['last_access'] = memory.last_access
            
            self.stats['successful_loads'] += 1
            
            return memory
            
        except Exception as e:
            logger.error(f"Error loading memory {memory_id}: {e}")
            self.stats['failed_loads'] += 1
            return None
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from disk.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            Success status
        """
        async with self._persistence_lock:
            try:
                # Check if memory exists in index
                if memory_id not in self.memory_index:
                    logger.warning(f"Memory {memory_id} not found in index for deletion")
                    return False
                    
                # Get file path from index
                memory_info = self.memory_index[memory_id]
                file_path = Path(memory_info['path'])
                
                # Delete file if it exists
                if file_path.exists():
                    os.remove(file_path)
                
                # Also delete backup if it exists
                backup_path = file_path.with_suffix('.bak')
                if backup_path.exists():
                    os.remove(backup_path)
                    
                # Remove from index
                del self.memory_index[memory_id]
                
                # Save index
                await self._save_memory_index()
                
                return True
                
            except Exception as e:
                logger.error(f"Error deleting memory {memory_id}: {e}")
                return False
    
    async def load_all_memories(self, memory_type: Optional[MemoryTypes] = None, 
                               batch_size: Optional[int] = None) -> List[MemoryEntry]:
        """
        Load all memories of a specific type.
        
        Args:
            memory_type: Optional type to filter by
            batch_size: Optional batch size to use
            
        Returns:
            List of memory entries
        """
        # Use configured batch size if not specified
        batch_size = batch_size or self.config['batch_size']
        
        # Get memory IDs to load
        if memory_type:
            memory_ids = [
                memory_id for memory_id, info in self.memory_index.items()
                if info.get('type') == memory_type.value
            ]
        else:
            memory_ids = list(self.memory_index.keys())
            
        memories = []
        
        # Load in batches
        for i in range(0, len(memory_ids), batch_size):
            batch_ids = memory_ids[i:i+batch_size]
            
            # Load each memory in batch
            batch_loads = [self.load_memory(memory_id) for memory_id in batch_ids]
            
            # Wait for all loads to complete
            batch_results = await asyncio.gather(*batch_loads, return_exceptions=True)
            
            # Add successful loads to result
            for result in batch_results:
                if isinstance(result, MemoryEntry):
                    memories.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error in batch load: {result}")
            
            # Yield control to allow other tasks to run
            await asyncio.sleep(0)
        
        return memories
    
    async def create_backup(self) -> bool:
        """
        Create a backup of all memories.
        
        Returns:
            Success status
        """
        async with self._persistence_lock:
            try:
                # Create timestamp for backup
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_dir = self.backup_path / f"backup_{timestamp}"
                
                # Create backup directory
                backup_dir.mkdir(exist_ok=True)
                
                # Copy memory index
                shutil.copy2(self.index_path, backup_dir / self.config['index_filename'])
                
                # Copy all memory files
                for memory_type in MemoryTypes:
                    type_dir = self.storage_path / memory_type.value
                    if type_dir.exists():
                        # Create corresponding directory in backup
                        backup_type_dir = backup_dir / memory_type.value
                        backup_type_dir.mkdir(exist_ok=True)
                        
                        # Copy all files for this type
                        for file in type_dir.glob('*.json'):
                            shutil.copy2(file, backup_type_dir / file.name)
                
                # Update stats
                self.stats['last_backup'] = time.time()
                self.stats['backup_count'] += 1
                
                # Prune old backups
                await self._prune_old_backups()
                
                logger.info(f"Created memory backup at {backup_dir}")
                return True
                
            except Exception as e:
                logger.error(f"Error creating backup: {e}")
                return False
    
    async def _prune_old_backups(self) -> None:
        """Prune old backups, keeping only the most recent ones."""
        try:
            # Get all backup directories
            backup_dirs = sorted([
                d for d in self.backup_path.glob('backup_*')
                if d.is_dir()
            ], key=lambda d: d.name)
            
            # Keep only the most recent backups
            if len(backup_dirs) > self.config['max_backups']:
                for old_dir in backup_dirs[:-self.config['max_backups']]:
                    try:
                        shutil.rmtree(old_dir)
                        logger.info(f"Pruned old backup: {old_dir}")
                    except Exception as e:
                        logger.error(f"Error pruning old backup {old_dir}: {e}")
        except Exception as e:
            logger.error(f"Error pruning old backups: {e}")
    
    async def restore_from_backup(self, backup_timestamp: Optional[str] = None) -> bool:
        """
        Restore memories from a backup.
        
        Args:
            backup_timestamp: Optional specific backup to restore from
                               If None, restores from most recent backup
            
        Returns:
            Success status
        """
        async with self._persistence_lock:
            try:
                # Find backup to restore from
                if backup_timestamp:
                    backup_dir = self.backup_path / f"backup_{backup_timestamp}"
                    if not backup_dir.exists():
                        logger.error(f"Backup {backup_timestamp} not found")
                        return False
                else:
                    # Find most recent backup
                    backup_dirs = sorted([
                        d for d in self.backup_path.glob('backup_*')
                        if d.is_dir()
                    ], key=lambda d: d.name)
                    
                    if not backup_dirs:
                        logger.error("No backups found")
                        return False
                        
                    backup_dir = backup_dirs[-1]
                
                # Create backup of current state before restore
                current_backup_dir = self.backup_path / f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                current_backup_dir.mkdir(exist_ok=True)
                
                # Backup current index
                if self.index_path.exists():
                    shutil.copy2(self.index_path, current_backup_dir / self.config['index_filename'])
                
                # Backup current memory files
                for memory_type in MemoryTypes:
                    type_dir = self.storage_path / memory_type.value
                    if type_dir.exists():
                        # Create corresponding directory in backup
                        current_backup_type_dir = current_backup_dir / memory_type.value
                        current_backup_type_dir.mkdir(exist_ok=True)
                        
                        # Copy all files for this type
                        for file in type_dir.glob('*.json'):
                            shutil.copy2(file, current_backup_type_dir / file.name)
                
                # Delete current memory files
                for memory_type in MemoryTypes:
                    type_dir = self.storage_path / memory_type.value
                    if type_dir.exists():
                        shutil.rmtree(type_dir)
                        type_dir.mkdir(exist_ok=True)
                
                # Restore from backup
                # Copy index first
                backup_index_path = backup_dir / self.config['index_filename']
                if backup_index_path.exists():
                    shutil.copy2(backup_index_path, self.index_path)
                
                # Copy memory files
                for memory_type in MemoryTypes:
                    backup_type_dir = backup_dir / memory_type.value
                    if backup_type_dir.exists():
                        # Create corresponding directory
                        type_dir = self.storage_path / memory_type.value
                        type_dir.mkdir(exist_ok=True)
                        
                        # Copy all files for this type
                        for file in backup_type_dir.glob('*.json'):
                            shutil.copy2(file, type_dir / file.name)
                
                # Reload memory index
                self.memory_index = self._load_memory_index()
                
                logger.info(f"Restored from backup {backup_dir}")
                return True
                
            except Exception as e:
                logger.error(f"Error restoring from backup: {e}")
                return False
    
    async def update_memory_access(self, memory_id: str) -> bool:
        """
        Update memory access statistics without loading the full memory.
        
        Args:
            memory_id: ID of memory to update
            
        Returns:
            Success status
        """
        try:
            # Check if memory exists in index
            if memory_id not in self.memory_index:
                return False
                
            # Update access stats in index
            self.memory_index[memory_id]['access_count'] = self.memory_index[memory_id].get('access_count', 0) + 1
            self.memory_index[memory_id]['last_access'] = time.time()
            
            # Save index periodically
            if id(memory_id) % 10 == 0:  # Save approximately every 10 updates
                await self._save_memory_index()
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory access: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get persistence handler statistics."""
        # Count memories by type
        type_counts = {}
        
        for memory_info in self.memory_index.values():
            memory_type = memory_info.get('type', 'unknown')
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
            
        return {
            'total_memories': len(self.memory_index),
            'memory_types': type_counts,
            'saves': self.stats['saves'],
            'successful_saves': self.stats['successful_saves'],
            'failed_saves': self.stats['failed_saves'],
            'loads': self.stats['loads'],
            'successful_loads': self.stats['successful_loads'],
            'failed_loads': self.stats['failed_loads'],
            'last_backup': self.stats['last_backup'],
            'backup_count': self.stats['backup_count'],
            'save_success_rate': self.stats['successful_saves'] / max(1, self.stats['saves']),
            'load_success_rate': self.stats['successful_loads'] / max(1, self.stats['loads'])
        }