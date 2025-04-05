# synthians_memory_core/memory_persistence.py

import os
import sys
import json
import uuid
import time
import asyncio
import logging
import aiofiles
import shutil
from typing import Dict, List, Set, Optional, Union, Any, Tuple
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# Local imports from your codebase
from .memory_structures import MemoryEntry, MemoryAssembly
from .custom_logger import logger  # Your shared custom logger

class MemoryPersistence:
    """
    Handles disk-based memory operations with robust async I/O and index management.
    
    - Maintains a memory_index.json for quick lookups.
    - Saves/loads MemoryEntry and MemoryAssembly objects from JSON files.
    - Provides backup functionality and index self-consistency checks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'storage_path': Path('/app/memory/stored'),  # Consistent Docker path
            'backup_dir': 'backups',
            'index_filename': 'memory_index.json',
            'max_backups': 5,
            'safe_write': True,  # Use atomic writes with .tmp renaming
            **(config or {})
        }
        self.storage_path = Path(self.config['storage_path'])
        self.backup_path = self.storage_path / self.config['backup_dir']
        self.index_path = self.storage_path / self.config['index_filename']

        # In-memory index: Dict[str, Dict[str, Any]]  
        # Example entry:  
        #   "mem_1234abcd": {
        #       "path": "mem_1234abcd.json",
        #       "timestamp": "<iso-string>",
        #       "quickrecal": 0.75,
        #       "type": "memory"
        #   }
        self.memory_index: Dict[str, Dict[str, Any]] = {}

        # Async lock to protect all file/index operations
        self._lock = asyncio.Lock()
        self.stats = {
            'saves': 0, 
            'loads': 0, 
            'deletes': 0, 
            'backups': 0, 
            'errors': 0
        }
        self._initialized = False  # Flag to ensure we only load index once

        # Ensure storage directories exist
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.backup_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(
                "MemoryPersistence",
                "Failed to create storage directories",
                {"path": str(self.storage_path), "error": str(e)}
            )
            raise  # Critical error, cannot proceed

        logger.info(
            "MemoryPersistence",
            "Initialized (index load deferred)",
            {"storage_path": str(self.storage_path)}
        )

    async def initialize(self):
        """Load the memory index asynchronously (only once)."""
        if self._initialized:
            return
        logger.info("MemoryPersistence", "Initializing (loading index)...")
        await self._load_index()
        self._initialized = True
        logger.info("MemoryPersistence", "Initialization complete.")

    async def _load_index(self):
        """Load the memory index from disk (internally, with lock)."""
        async with self._lock:
            if not self.index_path.exists():
                logger.info(
                    "MemoryPersistence",
                    "Memory index file not found, starting fresh.",
                    {"path": str(self.index_path)}
                )
                self.memory_index = {}
                return

            try:
                async with aiofiles.open(self.index_path, 'r') as f:
                    content = await f.read()
                    loaded_index = await asyncio.to_thread(json.loads, content)

                if isinstance(loaded_index, dict):
                    self.memory_index = loaded_index
                    logger.info(
                        "MemoryPersistence",
                        f"Loaded memory index with {len(self.memory_index)} entries.",
                        {"path": str(self.index_path)}
                    )
                else:
                    logger.error(
                        "MemoryPersistence",
                        "Invalid index file format, starting fresh.",
                        {"path": str(self.index_path)}
                    )
                    self.memory_index = {}

            except Exception as e:
                logger.error(
                    "MemoryPersistence",
                    "Error loading memory index, starting fresh.",
                    {"path": str(self.index_path), "error": str(e)}
                )
                self.memory_index = {}  # Start fresh on error

    async def _save_index_no_lock(self) -> bool:
        """
        Save the memory index to disk atomically, without acquiring a lock.
        Caller must already hold self._lock.
        """
        try:
            logger.debug("MemoryPersistence", "Saving memory index to disk using safe_write_json")
            
            # Use the safe_write_json utility for atomic writes with directory creation
            save_success = await MemoryPersistence.safe_write_json(
                data=self.memory_index,
                target_path=self.index_path
            )
            
            if save_success:
                self.stats['last_index_update'] = time.time()
                logger.debug("MemoryPersistence", "Memory index saved successfully")
                return True
            else:
                logger.error("MemoryPersistence", "Failed to save memory index")
                return False
        except asyncio.TimeoutError:
            logger.error("MemoryPersistence", "Timeout saving memory index")
            return False
        except Exception as e:
            logger.error(
                "MemoryPersistence",
                "Error saving memory index",
                {"path": str(self.index_path), "error": str(e)}
            )
            if await asyncio.to_thread(os.path.exists, self.index_path.with_suffix('.tmp')):
                try:
                    await asyncio.to_thread(os.remove, self.index_path.with_suffix('.tmp'))
                except Exception:
                    pass
            return False

    async def _save_index(self) -> bool:
        """Acquire lock and save the memory index to disk."""
        async with self._lock:
            return await self._save_index_no_lock()

    def _save_index_sync(self) -> bool:
        """Synchronously save the memory index to disk."""
        try:
            logger.debug("MemoryPersistence", "Saving memory index to disk synchronously")
            temp_path = self.index_path.with_suffix('.tmp')

            # PHASE 5.8: Ensure parent directory exists before saving
            if not os.path.exists(os.path.dirname(temp_path)):
                logger.info("MemoryPersistence", f"Creating parent directory for index: {os.path.dirname(temp_path)}")
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)

            with open(temp_path, 'w') as f:
                f.write(json.dumps(self.memory_index, indent=2))

            shutil.move(temp_path, self.index_path)
            self.stats['last_index_update'] = time.time()
            logger.debug("MemoryPersistence", "Memory index saved successfully synchronously")
            return True

        except Exception as e:
            logger.error(
                "MemoryPersistence",
                "Error saving memory index synchronously",
                {"path": str(self.index_path), "error": str(e)}
            )
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return False

    async def save_memory(self, memory: MemoryEntry) -> bool:
        """Save a memory entry to disk and update the index.

        Args:
            memory: The memory entry to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        if memory is None:
            logger.error("MemoryPersistence", "Cannot save None memory")
            return False
        
        # Create memories directory if it doesn't exist
        memories_dir = self.storage_path / "memories"
        os.makedirs(memories_dir, exist_ok=True)
        
        # Safe filename handling for Windows compatibility
        mem_id = memory.id
        safe_mem_id = self.sanitize_id_for_filename(mem_id)
        file_path = memories_dir / f"{safe_mem_id}.json"
        
        # Convert memory to dictionary and sanitize any NaN/Inf values
        mem_dict = memory.to_dict()
        
        # Perform atomic write
        success = await MemoryPersistence.safe_write_json(
            data=mem_dict,
            target_path=file_path
        )
        
        if success:
            # Update index asynchronously
            await self._update_index(memory)
            
            # Save the updated index to disk
            await self._save_index()
            
            logger.debug("MemoryPersistence", f"Saved memory {mem_id} to {file_path}")
            return True
        else:
            logger.error("MemoryPersistence", f"Failed to save memory {mem_id}")
            return False

    async def load_memory(self, item_id: str, geometry_manager=None) -> Optional[MemoryEntry]:
        """
        Load a single memory (MemoryEntry) from disk by ID.
        Acquires lock for the load operation.
        """
        logger.debug(f"[load_memory] Acquiring lock for {item_id}")
        async with self._lock:
            item = await self._load_item_no_lock(item_id, geometry_manager)
            if item and isinstance(item, MemoryEntry):
                return item
            elif item:
                logger.warning(
                    f"[load_memory] Loaded item {item_id} but it is not a MemoryEntry (type={type(item)})"
                )
            return None

    async def load_assembly(self, assembly_id: str, geometry_manager) -> Optional[MemoryAssembly]:
        """Load a memory assembly by ID.
        
        This enhanced implementation for Phase 5.8 includes better error handling,
        schema validation, and support for the new synchronization tracking fields.
        
        Args:
            assembly_id: The ID of the assembly to load
            geometry_manager: The geometry manager for embedding validation
            
        Returns:
            The loaded MemoryAssembly object, or None if not found or error occurs
        """
        print(f"[PERSISTENCE] load_assembly START for {assembly_id}")
        if not assembly_id.startswith("asm"):
            logger.warning(f"load_assembly called with non-assembly ID prefix: {assembly_id}")
            # Attempt to load anyway, maybe index is correct
            print(f"[PERSISTENCE] load_assembly WARNING: Non-assembly ID prefix for {assembly_id}")

        # Windows-safe assembly_id for filename
        safe_assembly_id = assembly_id.replace(':', '-')
        
        # Load without lock first, as _load_item_no_lock handles file reads
        # The lock is primarily for index and file *writes*
        print(f"[PERSISTENCE] load_assembly - Calling _load_item_no_lock with safe ID for {assembly_id}...")
        # Removed safe_filename keyword argument as _load_item_no_lock does not accept it
        item = await self._load_item_no_lock(assembly_id, geometry_manager)
        print(f"[PERSISTENCE] load_assembly - _load_item_no_lock returned type: {type(item)} for {assembly_id}")
        
        if isinstance(item, MemoryAssembly):
            self.stats['assemblies_loaded'] = self.stats.get('assemblies_loaded', 0) + 1
            print(f"[PERSISTENCE] load_assembly END for {assembly_id} - SUCCESS")
            return item
        elif item is not None:
            logger.error(f"Loaded item {assembly_id} is not a MemoryAssembly, type: {type(item)}")
            print(f"[PERSISTENCE] load_assembly ERROR: Loaded item is not MemoryAssembly for {assembly_id}, type: {type(item)}")
            self.stats['failed_assembly_loads'] = self.stats.get('failed_assembly_loads', 0) + 1
            return None
        else:
            logger.warning(f"Assembly {assembly_id} not found or failed to load.")
            print(f"[PERSISTENCE] load_assembly END for {assembly_id} - FAILED (Not found or load error)")
            self.stats['failed_assembly_loads'] = self.stats.get('failed_assembly_loads', 0) + 1
            return None

    async def load_all(self, geometry_manager=None) -> List[Union[MemoryEntry, MemoryAssembly]]:
        """
        Load ALL items (memories + assemblies) from the index.
        Returns them as a list of objects.
        """
        logger.info("MemoryPersistence.load_all called.")
        if not self._initialized:
            await self.initialize()

        all_items = []
        async with self._lock:
            all_ids = list(self.memory_index.keys())
            total = len(all_ids)
            logger.info(f"Lock acquired. Found {total} items to load.")
            batch_size = 50
            loaded_count = 0

            for i in range(0, total, batch_size):
                batch_ids = all_ids[i : i + batch_size]
                load_tasks = [
                    self._load_item_no_lock(item_id, geometry_manager)
                    for item_id in batch_ids
                ]
                results = await asyncio.gather(*load_tasks, return_exceptions=True)

                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Error loading item {batch_ids[idx]} in batch: {str(result)}",
                            exc_info=True
                        )
                    elif result is not None:
                        all_items.append(result)
                        loaded_count += 1

                logger.info(f"Batch loaded: +{len(batch_ids)} IDs, total loaded so far {loaded_count}")
                await asyncio.sleep(0.01)  # small yield

            logger.info(f"Finished loading {loaded_count}/{total} items from disk.")
        return all_items

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory (MemoryEntry) from disk and remove from index."""
        async with self._lock:
            try:
                if memory_id not in self.memory_index:
                    # Possibly check direct filesystem fallback
                    file_path_direct = self.storage_path / f"{memory_id}.json"
                    if not await asyncio.to_thread(os.path.exists, file_path_direct):
                        logger.warning(
                            "MemoryPersistence",
                            f"Memory {memory_id} not found for deletion"
                        )
                        return False
                    # If found on disk but not in index, artificially fix the index
                    self.memory_index[memory_id] = {'path': f"{memory_id}.json"}

                info = self.memory_index[memory_id]
                file_path = self.storage_path / info['path']

                deleted = False
                if await asyncio.to_thread(os.path.exists, file_path):
                    await asyncio.to_thread(os.remove, file_path)
                    deleted = True
                # Check for .bak as well
                if await asyncio.to_thread(os.path.exists, file_path.with_suffix('.bak')):
                    await asyncio.to_thread(os.remove, file_path.with_suffix('.bak'))
                    deleted = True

                if deleted:
                    del self.memory_index[memory_id]
                    await self._save_index()
                    self.stats['deletes'] += 1
                    return True
                else:
                    # File not found; remove from index anyway
                    del self.memory_index[memory_id]
                    await self._save_index()
                    return False

            except Exception as e:
                logger.error(
                    "MemoryPersistence",
                    f"Error deleting memory {memory_id}",
                    {"error": str(e)}
                )
                self.stats['errors'] += 1
                return False

    async def save_assembly(self, assembly: MemoryAssembly, geometry_manager=None) -> bool:
        """Save a memory assembly to disk and update the index.

        Args:
            assembly: The memory assembly to save
            geometry_manager: Optional geometry manager for validating embeddings
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not assembly:
            logger.error("MemoryPersistence", "Cannot save assembly: invalid assembly")
            return False
        
        # Safety check for assembly_id
        if not hasattr(assembly, 'assembly_id') or not assembly.assembly_id:
            logger.error("MemoryPersistence", "Cannot save assembly: missing assembly_id")
            return False
        
        # Create assemblies directory if it doesn't exist
        assemblies_dir = self.storage_path / "assemblies"
        os.makedirs(assemblies_dir, exist_ok=True)
        
        # Safe filename handling for Windows compatibility
        assembly_id = assembly.assembly_id
        safe_assembly_id = self.sanitize_id_for_filename(assembly_id)
        file_path = assemblies_dir / f"{safe_assembly_id}.json"
        
        # Convert assembly to dictionary and sanitize any NaN/Inf values
        assembly_dict = assembly.to_dict()
        
        # Perform atomic write
        save_success = await MemoryPersistence.safe_write_json(
            data=assembly_dict,
            target_path=file_path
        )
        
        if save_success:
            # Update the memory index
            # Construct the correct relative path including the subdirectory
            correct_rel_path = str(Path("assemblies") / f"{safe_assembly_id}.json")
            await self._update_index(assembly, path=correct_rel_path, item_type="assembly")
            
            # Save the updated index to disk
            await self._save_index()
            
            logger.debug("MemoryPersistence", f"Saved assembly {assembly_id} to {file_path}")
            self.stats['saves'] = self.stats.get('saves', 0) + 1
            return True
        else:
            logger.error("MemoryPersistence", f"Failed to save assembly {assembly_id}")
            self.stats['failed_assembly_saves'] = self.stats.get('failed_assembly_saves', 0) + 1
            return False

    async def list_assemblies(self) -> List[Dict[str, Any]]:
        """List all assemblies from the index."""
        async with self._lock:
            try:
                assemblies = []
                for mem_id, info in self.memory_index.items():
                    if info.get('type') == 'assembly':
                        assemblies.append({
                            'id': mem_id,
                            'path': info.get('path', ''),
                            'timestamp': info.get('timestamp', 0)
                        })
                return assemblies
            except Exception as e:
                logger.error("MemoryPersistence", "Error listing assemblies", {"error": str(e)})
                return []

    async def delete_assembly(self, assembly_id: str) -> bool:
        """Delete an assembly file from disk and remove from index."""
        async with self._lock:
            try:
                if assembly_id not in self.memory_index or self.memory_index[assembly_id].get('type') != 'assembly':
                    logger.warning("MemoryPersistence", f"Assembly {assembly_id} not found for deletion")
                    return False

                info = self.memory_index[assembly_id]
                file_path = self.storage_path / info['path']

                if await asyncio.to_thread(os.path.exists, file_path):
                    await asyncio.to_thread(os.remove, file_path)

                del self.memory_index[assembly_id]
                await self._save_index()

                self.stats['assembly_deletes'] = self.stats.get('assembly_deletes', 0) + 1
                logger.info("MemoryPersistence", f"Deleted assembly {assembly_id}")
                return True

            except Exception as e:
                logger.error(
                    "MemoryPersistence",
                    f"Error deleting assembly {assembly_id}",
                    {"error": str(e)}
                )
                self.stats['failed_assembly_deletes'] = self.stats.get('failed_assembly_deletes', 0) + 1
                return False

    async def _load_item_no_lock(self, item_id: str, geometry_manager=None) -> Optional[Union[MemoryEntry, MemoryAssembly]]:
        """Load a memory or assembly by ID with fallback to disk search.
        This internal method is called by load_memory and load_assembly and
        assumes the lock is already held by the caller.
        
        Args:
            item_id: ID of the memory or assembly to load
            geometry_manager: Optional geometry manager for validating embeddings
            
        Returns:
            MemoryEntry or MemoryAssembly if found, None otherwise
        """
        try:
            # Check if item is in index
            if item_id in self.memory_index:
                index_entry = self.memory_index[item_id]
                item_type = index_entry.get('type')
                rel_path = index_entry.get('path')
                
                if not rel_path:
                    logger.error("MemoryPersistence", f"Invalid index entry for {item_id}: missing path")
                    return None
                
                # Load from indexed path
                file_path = self.storage_path / rel_path
                logger.debug("MemoryPersistence", f"Loading {item_type} {item_id} from indexed path: {file_path}")
                
                if not await asyncio.to_thread(os.path.exists, file_path):
                    logger.warning("MemoryPersistence", f"File not found at indexed path: {file_path}")
                    # Will try fallback paths below
                else:
                    # Load from indexed path
                    if item_type == "memory":
                        return await self._load_memory_from_file(file_path, item_id, geometry_manager)
                    elif item_type == "assembly":
                        return await self._load_assembly_from_file(file_path, item_id, geometry_manager)
            
            # Fallback: Try to find item in memories directory
            safe_item_id = self.sanitize_id_for_filename(item_id)
            memories_path = self.storage_path / "memories" / f"{safe_item_id}.json"
            logger.debug("MemoryPersistence", f"_load_item_no_lock - Checking existence of fallback path: {memories_path}")
            
            memory_exists = await asyncio.to_thread(os.path.exists, memories_path)
            logger.debug("MemoryPersistence", f"_load_item_no_lock - Fallback path exists: {memory_exists}")
            
            if memory_exists:
                # Found memory file, load it and update index
                logger.info("MemoryPersistence", f"Found {item_id} in memories directory, updating index")
                memory = await self._load_memory_from_file(memories_path, item_id, geometry_manager)
                if memory:
                    # Update index with memory file location
                    rel_path = memories_path.relative_to(self.storage_path)
                    await self._update_index(memory)
                return memory
            
            # Fallback: Try to find item in assemblies directory
            safe_item_id = self.sanitize_id_for_filename(item_id)
            assemblies_path = self.storage_path / "assemblies" / f"{safe_item_id}.json"
            logger.debug("MemoryPersistence", f"_load_item_no_lock - Checking existence of fallback path: {assemblies_path}")
            
            assembly_exists = await asyncio.to_thread(os.path.exists, assemblies_path) 
            logger.debug("MemoryPersistence", f"_load_item_no_lock - Fallback path exists: {assembly_exists}")
            
            if assembly_exists:
                # Found assembly file, load it and update index
                logger.info("MemoryPersistence", f"Found {item_id} in assemblies directory, updating index")
                assembly = await self._load_assembly_from_file(assemblies_path, item_id, geometry_manager)
                if assembly:
                    # Update index with assembly file location
                    rel_path = assemblies_path.relative_to(self.storage_path)
                    await self._update_index(assembly)
                return assembly
            
            # Item not found in index or on disk
            logger.warning("MemoryPersistence", f"Item {item_id} not found in index or on disk")
            return None
            
        except Exception as e:
            logger.error("MemoryPersistence", f"Error loading item {item_id}: {str(e)}", exc_info=True)
            return None

    async def _load_memory_from_file(self, file_path: Path, item_id: str, geometry_manager=None) -> Optional[MemoryEntry]:
        """Load a MemoryEntry from a JSON file."""
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                item_dict = await asyncio.to_thread(json.loads, content)
                instance = MemoryEntry(**item_dict)
                if 'id' not in item_dict:
                    instance.id = item_id
                if geometry_manager and instance.embedding is not None:
                    try:
                        validated = geometry_manager._validate_vector(
                            instance.embedding,
                            f"Loaded Memory Emb {item_id}"
                        )
                        if validated is None:
                            logger.warning(f"[_load_memory_from_file] Embedding validation failed for memory {item_id}, setting to None.")
                            instance.embedding = None
                        else:
                            instance.embedding = validated
                    except Exception as e_val:
                        logger.error(f"[_load_memory_from_file] Error validating embedding for memory {item_id}: {str(e_val)}")
                        instance.embedding = None
                return instance
        except Exception as e:
            logger.error(f"[_load_memory_from_file] Error loading memory {item_id}: {str(e)}", exc_info=True)
            return None

    async def _load_assembly_from_file(self, file_path: Path, item_id: str, geometry_manager=None) -> Optional[MemoryAssembly]:
        """Load a MemoryAssembly from a JSON file."""
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                item_dict = await asyncio.to_thread(json.loads, content)
                instance = MemoryAssembly.from_dict(item_dict, geometry_manager)
                if hasattr(instance, 'composite_embedding') and instance.composite_embedding is not None:
                    try:
                        validated = geometry_manager._validate_vector(
                            instance.composite_embedding,
                            f"Loaded Composite Emb for {item_id}"
                        )
                        if validated is None:
                            logger.warning(f"[_load_assembly_from_file] Composite embedding validation failed for assembly {item_id}, setting to None.")
                            instance.composite_embedding = None
                        else:
                            instance.composite_embedding = validated
                    except Exception as e_val:
                        logger.error(f"[_load_assembly_from_file] Error validating composite embedding for assembly {item_id}: {str(e_val)}")
                        instance.composite_embedding = None
                if hasattr(instance, 'hyperbolic_embedding') and instance.hyperbolic_embedding is not None:
                    try:
                        validated = geometry_manager._validate_vector(
                            instance.hyperbolic_embedding,
                            f"Loaded Hyperbolic Emb for {item_id}"
                        )
                        if validated is None:
                            logger.warning(f"[_load_assembly_from_file] Hyperbolic embedding validation failed for assembly {item_id}, setting to None.")
                            instance.hyperbolic_embedding = None
                        else:
                            instance.hyperbolic_embedding = validated
                    except Exception as e_val:
                        logger.error(f"[_load_assembly_from_file] Error validating hyperbolic embedding for assembly {item_id}: {str(e_val)}")
                        instance.hyperbolic_embedding = None
                return instance
        except Exception as e:
            logger.error(f"[_load_assembly_from_file] Error loading assembly {item_id}: {str(e)}", exc_info=True)
            return None

    async def _update_index(self, item, path=None, item_type=None):
        """Update the memory index with a memory or assembly entry.
        
        Args:
            item: MemoryEntry or MemoryAssembly to index
            path: Optional relative path override
            item_type: Optional type override ("memory" or "assembly")
        """
        try:
            # Determine item ID and type
            if hasattr(item, 'id'):
                item_id = item.id
                item_actual_type = "memory"
            elif hasattr(item, 'assembly_id'):
                item_id = item.assembly_id
                item_actual_type = "assembly"
            else:
                item_id = str(item)  # Fallback to string representation
                item_actual_type = item_type or "unknown"
                
            # Use provided type if specified
            item_type = item_type or item_actual_type
                
            # Determine relative path with sanitized filename
            if path:
                rel_path = path
            else:
                safe_id = self.sanitize_id_for_filename(item_id)
                base_dir = "assemblies" if item_type == "assembly" else "memories"
                # Use Path for cross-platform compatibility
                rel_path = str(Path(base_dir) / f"{safe_id}.json")
            
            # Create index entry
            timestamp = datetime.now(timezone.utc).isoformat()
            self.memory_index[item_id] = {
                'path': rel_path,
                'type': item_type,
                'timestamp': timestamp
            }
            
            logger.debug("MemoryPersistence", 
                       f"Updated index for {item_type} {item_id} with path {rel_path}")
            
            # If the item is a Memory Assembly, update vector_index_updated_at
            if hasattr(item, 'vector_index_updated_at') and item_type == "assembly":
                # Update vector_index_updated_at timestamp
                item.vector_index_updated_at = datetime.now(timezone.utc)
                logger.debug("MemoryPersistence", 
                            f"Updated vector_index_updated_at for assembly {item_id}")
            
            return True
        except Exception as e:
            logger.error("MemoryPersistence", f"Error updating index: {str(e)}", exc_info=True)
            return False

    async def create_backup(self) -> bool:
        """Create a full storage backup (copies entire storage_path, ignoring itself)."""
        async with self._lock:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_instance_path = self.backup_path / f"backup_{timestamp}"

                await asyncio.to_thread(
                    shutil.copytree,
                    self.storage_path,
                    backup_instance_path,
                    ignore=shutil.ignore_patterns('backups')
                )

                self.stats['last_backup'] = time.time()
                self.stats['backup_count'] = self.stats.get('backup_count', 0) + 1
                logger.info("MemoryPersistence", f"Created backup at {backup_instance_path}")

                # Prune old backups
                await self._prune_backups()
                return True

            except Exception as e:
                logger.error("MemoryPersistence", "Error creating backup", {"error": str(e)})
                self.stats['errors'] = self.stats.get('errors', 0) + 1
                return False

    async def _prune_backups(self):
        """Keep only the N most recent backups (sorted by mod time)."""
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

    async def shutdown(self):
        """Cleanup: final index save, etc."""
        logger.info("MemoryPersistence", "Shutting down persistence handler...")
        
        try:
            # First attempt - use a longer timeout for shutdown
            try:
                logger.info("MemoryPersistence", "Saving memory index during shutdown (async)")
                save_success = await asyncio.wait_for(self._save_index(), timeout=10.0)
                if save_success:
                    logger.info("MemoryPersistence", "Memory index saved successfully (async)")
                else:
                    logger.warning("MemoryPersistence", "Async index save returned False, falling back to sync")
                    raise Exception("Async save returned False")
            except asyncio.TimeoutError:
                logger.warning("MemoryPersistence", "Timeout during async index save, attempting sync")
                # Fallback to synchronous save if async times out
                try:
                    logger.info("MemoryPersistence", "Performing sync index save")
                    sync_success = self._save_index_sync()
                    if sync_success:
                        logger.info("MemoryPersistence", "Fallback synchronous index save complete")
                    else:
                        logger.error("MemoryPersistence", "Both async and sync index saves failed")
                except Exception as sync_e:
                    logger.error("MemoryPersistence", f"Error during sync fallback: {sync_e}", exc_info=True)
            except Exception as e:
                logger.error("MemoryPersistence", f"Error during shutdown index save: {e}", exc_info=True)
                # Fallback to synchronous save if async fails
                try:
                    logger.info("MemoryPersistence", "Attempting sync save after async exception")
                    sync_success = self._save_index_sync()
                    if sync_success:
                        logger.info("MemoryPersistence", "Fallback synchronous index save complete after exception")
                    else:
                        logger.error("MemoryPersistence", "Both async and sync index saves failed after exception")
                except Exception as sync_e:
                    logger.error("MemoryPersistence", f"Final sync fallback also failed: {sync_e}", exc_info=True)
            
            # Final check of persistence directories
            try:
                memories_dir = self.storage_path / "memories"
                assemblies_dir = self.storage_path / "assemblies"
                memories_exist = os.path.exists(memories_dir)
                assemblies_exist = os.path.exists(assemblies_dir)
                index_exists = os.path.exists(self.index_path)
                
                log_data = {
                    "memories_dir_exists": memories_exist,
                    "assemblies_dir_exists": assemblies_exist,
                    "index_exists": index_exists,
                    "storage_path": str(self.storage_path)
                }
                
                if memories_exist and assemblies_exist and index_exists:
                    logger.info("MemoryPersistence", "All persistence directories verified", log_data)
                else:
                    logger.warning("MemoryPersistence", "Some persistence directories missing", log_data)
            except Exception as check_e:
                logger.error("MemoryPersistence", f"Error checking persistence directories: {check_e}")
        except Exception as outer_e:
            logger.error("MemoryPersistence", f"Critical failure during shutdown: {outer_e}", exc_info=True)
        
        logger.info("MemoryPersistence", "Shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Return current persistence stats + index count (without forcing re-init)."""
        return {
            "total_indexed_items": len(self.memory_index),
            "initialized": self._initialized,
            "last_index_update": self.stats.get('last_index_update', 0),
            "last_backup": self.stats.get('last_backup', 0),
            "saves": self.stats.get('saves', 0),
            "loads": self.stats.get('loads', 0),
            "deletes": self.stats.get('deletes', 0),
            "backups": self.stats.get('backups', 0),
            "errors": self.stats.get('errors', 0)
        }

    @staticmethod
    def _default_serializer(obj):
        """Custom JSON serializer for numpy types, datetimes, sets."""
        import numpy as np
        from datetime import datetime
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, datetime):
            # Ensure timezone info is handled if present, or make naive ISO
            if obj.tzinfo:
                return obj.isoformat()
            else:
                # Or decide how to handle naive datetimes, maybe assume UTC?
                # This makes it naive ISO format
                return obj.isoformat()
        # Fallback for other types
        try:
            # Check if object is serializable by default first
            json.dumps(obj)
            return obj # If serializable, return it directly
        except TypeError:
            try:
                return str(obj) # Try string representation
            except:
                return "[Unserializable Object]" # Last resort

    @staticmethod
    async def safe_write_json(data: Any, target_path: Path, serializer=None) -> bool:
        """Atomically write data to a JSON file with proper directory creation.
        
        Args:
            data: Data to serialize to JSON
            target_path: Path object for the final destination file
            serializer: Optional custom JSON serializer function
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Generate unique temp path to prevent collisions 
        unique_suffix = f".{uuid.uuid4().hex[:8]}.tmp"
        temp_path = target_path.with_suffix(unique_suffix)
        
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(temp_path)
            if not await asyncio.to_thread(os.path.exists, parent_dir):
                logger.info("MemoryPersistence", f"Creating parent directory: {parent_dir}")
                await asyncio.to_thread(os.makedirs, parent_dir, exist_ok=True)
                
                # Verify directory was created
                dir_exists = await asyncio.to_thread(os.path.exists, parent_dir)
                if not dir_exists:
                    logger.error("MemoryPersistence", f"Failed to create directory: {parent_dir}")
                    return False
            
            # Serialize to JSON (potentially CPU-bound)
            json_data = await asyncio.to_thread(
                json.dumps, data, indent=2, 
                default=serializer or MemoryPersistence._default_serializer
            )
            
            # Write to temp file asynchronously
            async with aiofiles.open(temp_path, 'w') as f:
                await f.write(json_data)
                await f.flush() # Ensure data is written to OS buffers
            
            # Verify temp file was written successfully
            temp_exists = await asyncio.to_thread(os.path.exists, temp_path)
            temp_size = await asyncio.to_thread(os.path.getsize, temp_path) if temp_exists else -1
            
            if not temp_exists or (temp_size == 0 and len(json_data) > 0):
                logger.error("MemoryPersistence", 
                             f"Temp file write verification failed: {temp_path} (Exists: {temp_exists}, Size: {temp_size})")
                return False
                
            logger.debug("MemoryPersistence", f"Temp file written successfully (size: {temp_size}): {temp_path}")
            
            # Atomic move using shutil.move in thread
            src = str(temp_path)
            dst = str(target_path)
            logger.debug("MemoryPersistence", f"Moving temp file '{src}' to final '{dst}'")
            await asyncio.to_thread(shutil.move, src, dst)
            
            # Verify final file exists
            final_exists = await asyncio.to_thread(os.path.exists, target_path)
            if not final_exists:
                logger.error("MemoryPersistence", f"Final file does not exist after move: {target_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.error("MemoryPersistence", f"Error in safe_write_json: {str(e)}", exc_info=True)
            # Cleanup temp file if it exists
            try:
                if 'temp_path' in locals() and await asyncio.to_thread(os.path.exists, temp_path):
                    await asyncio.to_thread(os.remove, temp_path)
            except Exception as cleanup_e:
                logger.debug("MemoryPersistence", f"Error cleaning up temp file: {str(cleanup_e)}")
            return False

    @staticmethod
    def sanitize_id_for_filename(item_id: str) -> str:
        """Convert IDs to safe filenames by replacing invalid characters.
        
        This ensures IDs with characters like ':' that are invalid in Windows
        filenames are properly sanitized.
        """
        return item_id.replace(":", "-")
