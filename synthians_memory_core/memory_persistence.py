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
            logger.debug("MemoryPersistence", "Saving memory index to disk")
            temp_path = self.index_path.with_suffix('.tmp')

            async with aiofiles.open(temp_path, 'w') as f:
                await f.write(json.dumps(self.memory_index, indent=2))

            await asyncio.to_thread(shutil.move, temp_path, self.index_path)
            self.stats['last_index_update'] = time.time()
            logger.debug("MemoryPersistence", "Memory index saved successfully")
            return True

        except asyncio.TimeoutError:
            logger.error("MemoryPersistence", "Timeout saving memory index")
            return False
        except Exception as e:
            logger.error(
                "MemoryPersistence",
                "Error saving memory index",
                {"path": str(self.index_path), "error": str(e)}
            )
            if await asyncio.to_thread(os.path.exists, temp_path):
                try:
                    await asyncio.to_thread(os.remove, temp_path)
                except Exception:
                    pass
            return False

    async def _save_index(self) -> bool:
        """Acquire lock and save the memory index to disk."""
        async with self._lock:
            return await self._save_index_no_lock()

    async def save_memory(self, memory: MemoryEntry) -> bool:
        """Save a single MemoryEntry to disk and update the index."""
        try:
            # Ensure there's a running loop
            try:
                loop = asyncio.get_running_loop()
                if not loop.is_running():
                    logger.warning(
                        "MemoryPersistence",
                        f"Attempted to save memory {memory.id} with no running event loop"
                    )
                    return False
            except RuntimeError:
                # No running event loop
                logger.warning(
                    "MemoryPersistence",
                    f"Error saving memory {memory.id}: no running event loop"
                )
                return False

            async with self._lock:
                try:
                    # Ensure memory has an ID
                    if not hasattr(memory, 'id') or memory.id is None:
                        memory.id = f"mem_{uuid.uuid4().hex[:12]}"

                    # Build file path
                    file_path = self.storage_path / f"{memory.id}.json"

                    # Convert memory -> dict
                    memory_dict = memory.to_dict()

                    # Save to disk
                    async with aiofiles.open(file_path, 'w') as f:
                        json_text = json.dumps(memory_dict, indent=2, default=MemoryPersistence._default_serializer)
                        await f.write(json_text)

                    # Update index
                    self.memory_index[memory.id] = {
                        'path': str(file_path.relative_to(self.storage_path)),
                        'timestamp': memory.timestamp.isoformat()
                            if hasattr(memory.timestamp, 'isoformat')
                            else str(memory.timestamp) if hasattr(memory, 'timestamp')
                            else time.time(),
                        'quickrecal': getattr(memory, 'quickrecal_score', 0.5),
                        'type': 'memory'
                    }

                    # Save index
                    await asyncio.wait_for(self._save_index_no_lock(), timeout=5)

                    self.stats['saves'] += 1
                    self.stats['successful_saves'] = self.stats.get('successful_saves', 0) + 1
                    logger.debug("MemoryPersistence", f"Memory {memory.id} saved successfully")
                    return True

                except Exception as e:
                    logger.error(
                        "MemoryPersistence",
                        f"Error saving memory {getattr(memory, 'id', 'unknown')}: {str(e)}"
                    )
                    self.stats['saves'] += 1
                    self.stats['failed_saves'] = self.stats.get('failed_saves', 0) + 1
                    return False

        except asyncio.TimeoutError:
            logger.error(
                "MemoryPersistence",
                f"Timeout saving memory {getattr(memory, 'id', 'unknown')}"
            )
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
        item = await self._load_item_no_lock(assembly_id, geometry_manager, safe_filename=safe_assembly_id)
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
        """
        Save a single MemoryAssembly object to disk asynchronously.
        Uses an atomic write pattern (write to temp, then rename).
        Handles embedding validation and conversion if geometry_manager is provided.

        Args:
            assembly: The MemoryAssembly object to save
            geometry_manager: Optional geometry manager for handling embedding conversions
            
        Returns:
            bool: True if the save was successful, False otherwise
        """
        print(f"[PERSISTENCE] save_assembly START for {assembly.assembly_id if assembly else 'None'}")
        if not assembly or not assembly.assembly_id:
            logger.error("MemoryPersistence", "Cannot save assembly: invalid or missing ID")
            print("[PERSISTENCE] save_assembly ERROR: Invalid or missing assembly ID")
            return False

        assembly_id = assembly.assembly_id
        print(f"[PERSISTENCE] save_assembly - ID: {assembly_id}")
        try:
            # Create assemblies directory if it doesn't exist
            assembly_path = self.storage_path / 'assemblies'
            os.makedirs(assembly_path, exist_ok=True)
            
            # Windows-safe assembly_id for filename (replace : with -)
            safe_assembly_id = assembly_id.replace(':', '-')
            
            # Create file paths
            file_path = assembly_path / f"{safe_assembly_id}.json"
            temp_file_path = assembly_path / f"{safe_assembly_id}.{uuid.uuid4().hex[:8]}.tmp.json"
            
            print(f"[PERSISTENCE] save_assembly - File paths created: tmp={temp_file_path}, target={file_path}")
            
            # Convert to dict (Potentially blocking)
            try:
                print(f"[PERSISTENCE] save_assembly - Calling assembly.to_dict() for {assembly_id}...")
                assembly_dict = assembly.to_dict()
                print(f"[PERSISTENCE] save_assembly - assembly.to_dict() completed for {assembly_id}.")
            except Exception as e:
                logger.error(
                    "MemoryPersistence",
                    f"assembly.to_dict() failed for {assembly_id}: {str(e)}",
                    exc_info=True
                )
                self.stats['failed_assembly_saves'] = self.stats.get('failed_assembly_saves', 0) + 1
                print(f"[PERSISTENCE] save_assembly ERROR: to_dict failed for {assembly_id}: {e}")
                return False

            if not isinstance(assembly_dict, dict):
                logger.error(
                    "MemoryPersistence",
                    f"Cannot save assembly {assembly_id}: to_dict() did not return dict."
                )
                self.stats['failed_assembly_saves'] = self.stats.get('failed_assembly_saves', 0) + 1
                print(f"[PERSISTENCE] save_assembly ERROR: to_dict did not return dict for {assembly_id}")
                return False

            # Basic field checks
            if not assembly_dict.get("assembly_id"):
                logger.error(
                    "MemoryPersistence",
                    f"Assembly {assembly_id} missing 'assembly_id' field in its dict."
                )
                self.stats['failed_assembly_saves'] = self.stats.get('failed_assembly_saves', 0) + 1
                print(f"[PERSISTENCE] save_assembly ERROR: assembly_id missing in dict for {assembly_id}")
                return False

            # JSON serialize (Potentially blocking)
            try:
                print(f"[PERSISTENCE] save_assembly - Starting json.dumps for {assembly_id}...")
                json_data = json.dumps(assembly_dict, indent=2, default=MemoryPersistence._default_serializer)
                print(f"[PERSISTENCE] save_assembly - json.dumps completed for {assembly_id}.")
            except Exception as e:
                logger.error(
                    "MemoryPersistence",
                    f"JSON serialization error for assembly {assembly_id}",
                    {"error": str(e)},
                    exc_info=True
                )
                self.stats['failed_assembly_saves'] = self.stats.get('failed_assembly_saves', 0) + 1
                print(f"[PERSISTENCE] save_assembly ERROR: json.dumps failed for {assembly_id}: {e}")
                return False
                
            print(f"[PERSISTENCE] save_assembly - Acquiring lock for {assembly_id}...")
            async with self._lock:
                print(f"[PERSISTENCE] save_assembly - Lock acquired for {assembly_id}.")
                
                # Write to file with atomic operation pattern for reliability
                # temp_file_path = file_path.parent / f"{assembly_id}.{uuid.uuid4().hex[:8]}.tmp.json"
                print(f"[PERSISTENCE] save_assembly - Writing to temp file: {temp_file_path}")
                try:
                    async with aiofiles.open(temp_file_path, 'w') as f:
                        print(f"[PERSISTENCE] save_assembly - Temp file opened, writing content...")
                        await f.write(json_data)
                        print(f"[PERSISTENCE] save_assembly - Content written to temp file.")
                    
                    # Ensure temp file was written successfully
                    print(f"[PERSISTENCE] save_assembly - Checking temp file existence {temp_file_path}...")
                    exists = await asyncio.to_thread(os.path.exists, temp_file_path)
                    print(f"[PERSISTENCE] save_assembly - Temp file exists: {exists}")
                    if not exists:
                        logger.error(f"Temp file not created at {temp_file_path}")
                        print(f"[PERSISTENCE] save_assembly ERROR: Temp file not created at {temp_file_path}")
                        # Attempt cleanup before returning
                        if await asyncio.to_thread(os.path.exists, temp_file_path): 
                            try: await asyncio.to_thread(os.remove, temp_file_path) 
                            except: pass
                        return False
                        
                    # Use atomic rename operation (blocking call in thread)
                    print(f"[PERSISTENCE] save_assembly - Renaming temp file {temp_file_path} to {file_path}...")
                    await asyncio.to_thread(shutil.move, temp_file_path, file_path)
                    print(f"[PERSISTENCE] save_assembly - Rename completed.")
                    
                    # Update index after successful save
                    self._update_index(assembly_id, file_path.relative_to(self.storage_path), "assembly")
                    print(f"[PERSISTENCE] save_assembly - Index updated for {assembly_id}.")
                    self.stats['assemblies_saved'] = self.stats.get('assemblies_saved', 0) + 1
                    print(f"[PERSISTENCE] save_assembly - Save successful for {assembly_id}.")
                    result = True

                except Exception as e:
                    logger.error(f"Error writing assembly file: {str(e)}", exc_info=True)
                    print(f"[PERSISTENCE] save_assembly ERROR: Writing/renaming file failed for {assembly_id}: {e}")
                    # Clean up temp file if it exists
                    print(f"[PERSISTENCE] save_assembly - Cleaning up temp file {temp_file_path}...")
                    if await asyncio.to_thread(os.path.exists, temp_file_path):
                        try:
                            await asyncio.to_thread(os.remove, temp_file_path)
                            print(f"[PERSISTENCE] save_assembly - Temp file cleaned up.")
                        except Exception as rm_err:
                            print(f"[PERSISTENCE] save_assembly - Error cleaning temp file: {rm_err}")
                            pass
                    self.stats['failed_assembly_saves'] = self.stats.get('failed_assembly_saves', 0) + 1
                    result = False
                finally:
                    print(f"[PERSISTENCE] save_assembly - Releasing lock for {assembly_id}.")
            
            print(f"[PERSISTENCE] save_assembly END for {assembly_id}, Result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error saving assembly {assembly_id}: {str(e)}", exc_info=True)
            print(f"[PERSISTENCE] save_assembly UNEXPECTED ERROR for {assembly_id}: {e}")
            self.stats['errors'] = self.stats.get('errors', 0) + 1
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

    async def _load_item_no_lock(self, item_id: str, geometry_manager=None, safe_filename: str = None) -> Optional[Union[MemoryEntry, MemoryAssembly]]:
        """
        Internal helper to load EITHER a MemoryEntry or MemoryAssembly by ID.
        No lock is acquired here; caller must hold self._lock.
        """
        print(f"[PERSISTENCE] _load_item_no_lock START for {item_id}")
        from .memory_structures import MemoryEntry, MemoryAssembly

        if not item_id:
            logger.error("[_load_item_no_lock] Invalid or empty item_id")
            print(f"[PERSISTENCE] _load_item_no_lock ERROR - Empty item_id")
            return None

        try:
            print(f"[PERSISTENCE] _load_item_no_lock - Getting item info from memory_index for {item_id}")
            item_info = self.memory_index.get(item_id)
            item_type = None
            file_path = None

            print(f"[PERSISTENCE] _load_item_no_lock - Item info found: {item_info is not None}")
            
            # Determine if this is an assembly based on the ID prefix
            is_assembly = item_id.startswith("asm:")
            print(f"[PERSISTENCE] _load_item_no_lock - Is assembly: {is_assembly}")

            if item_info:
                item_type = item_info.get('type', 'memory' if not is_assembly else 'assembly')
                path_str = item_info.get('path')
                if path_str:
                    file_path = self.storage_path / path_str
                else:
                    print(f"[PERSISTENCE] _load_item_no_lock - Path missing, using fallback for {item_id}")
                    if is_assembly:
                        item_type = 'assembly'
                        if safe_filename:
                            file_path = self.storage_path / 'assemblies' / f"{safe_filename}.json"
                        else:
                            file_path = self.storage_path / 'assemblies' / f"{item_id}.json"
                    else:
                        item_type = 'memory'
                        file_path = self.storage_path / f"{item_id}.json"
            else:
                # Fallback if not found in index
                print(f"[PERSISTENCE] _load_item_no_lock - Item not in index, using fallback for {item_id}")
                if is_assembly:
                    item_type = 'assembly'
                    if safe_filename:
                        file_path = self.storage_path / 'assemblies' / f"{safe_filename}.json"
                    else:
                        file_path = self.storage_path / 'assemblies' / f"{item_id}.json"
                else:
                    item_type = 'memory'
                    file_path = self.storage_path / f"{item_id}.json"

                print(f"[PERSISTENCE] _load_item_no_lock - Checking existence of fallback path: {file_path}")
                exists = await asyncio.to_thread(os.path.exists, file_path)
                print(f"[PERSISTENCE] _load_item_no_lock - Fallback path exists: {exists}")
                if not exists:
                    logger.warning(
                        f"[_load_item_no_lock] Item {item_id} not found in index or filesystem."
                    )
                    print(f"[PERSISTENCE] _load_item_no_lock END - Item not found for {item_id}")
                    return None
                # We do not update index in fallback until after load success

            # If no file or file doesn't exist
            print(f"[PERSISTENCE] _load_item_no_lock - Checking file existence: {file_path}")
            exists = await asyncio.to_thread(os.path.exists, file_path)
            print(f"[PERSISTENCE] _load_item_no_lock - File exists: {exists}")
            if file_path is None or not exists:
                logger.warning(
                    f"[_load_item_no_lock] Could not locate file for item {item_id}, file_path: {str(file_path)}"
                )
                if item_id in self.memory_index:
                    del self.memory_index[item_id]
                print(f"[PERSISTENCE] _load_item_no_lock END - File not found for {item_id}")
                return None

            print(f"[PERSISTENCE] _load_item_no_lock - Loading {item_type} from {file_path}")
            try:
                print(f"[PERSISTENCE] _load_item_no_lock - Opening file {file_path}...")
                async with aiofiles.open(file_path, 'r') as f:
                    print(f"[PERSISTENCE] _load_item_no_lock - Reading file content...")
                    content = await f.read()
                    print(f"[PERSISTENCE] _load_item_no_lock - File content read, length: {len(content)}")
                
                print(f"[PERSISTENCE] _load_item_no_lock - Parsing JSON with asyncio.to_thread...")
                item_dict = await asyncio.to_thread(json.loads, content)
                print(f"[PERSISTENCE] _load_item_no_lock - JSON parsed successfully")
            except json.JSONDecodeError as je:
                logger.error(f"[_load_item_no_lock] JSON parsing error for {item_id}: {str(je)}")
                print(f"[PERSISTENCE] _load_item_no_lock ERROR - JSON parse error: {je}")
                return None
            except Exception as io_err:
                logger.error(f"[_load_item_no_lock] File read error for {item_id}: {str(io_err)}")
                print(f"[PERSISTENCE] _load_item_no_lock ERROR - File read error: {io_err}")
                return None

            # Distinguish between memory vs assembly
            if item_type == "assembly" or is_assembly:
                if geometry_manager is None:
                    logger.error(
                        "[_load_item_no_lock] Cannot load assembly: no geometry_manager provided."
                    )
                    print(f"[PERSISTENCE] _load_item_no_lock ERROR - No geometry_manager for assembly {item_id}")
                    return None
                # Construct MemoryAssembly
                try:
                    print(f"[PERSISTENCE] _load_item_no_lock - Creating MemoryAssembly from dict for {item_id}...")
                    instance = MemoryAssembly.from_dict(item_dict, geometry_manager)
                    print(f"[PERSISTENCE] _load_item_no_lock - MemoryAssembly created successfully for {item_id}")
                    
                    # Validate composite embedding if present
                    if hasattr(instance, 'composite_embedding') and instance.composite_embedding is not None:
                        try:
                            print(f"[PERSISTENCE] _load_item_no_lock - Validating composite embedding for {item_id}...")
                            validated = geometry_manager._validate_vector(
                                instance.composite_embedding,
                                f"Loaded Composite Emb for {item_id}"
                            )
                            print(f"[PERSISTENCE] _load_item_no_lock - Composite embedding validation result: {validated is not None}")
                            if validated is None:
                                logger.warning(f"[_load_item_no_lock] Composite embedding validation failed for assembly {item_id}, setting to None.")
                                instance.composite_embedding = None
                            else:
                                instance.composite_embedding = validated
                        except Exception as e_val:
                            logger.error(f"[_load_item_no_lock] Error validating composite embedding for assembly {item_id}: {str(e_val)}")
                            print(f"[PERSISTENCE] _load_item_no_lock ERROR - Composite embedding validation: {e_val}")
                            instance.composite_embedding = None
                            
                    # Validate hyperbolic embedding if present
                    if hasattr(instance, 'hyperbolic_embedding') and instance.hyperbolic_embedding is not None:
                        try:
                            print(f"[PERSISTENCE] _load_item_no_lock - Validating hyperbolic embedding for {item_id}...")
                            validated = geometry_manager._validate_vector(
                                instance.hyperbolic_embedding,
                                f"Loaded Hyperbolic Emb for {item_id}",
                                space="hyperbolic"
                            )
                            print(f"[PERSISTENCE] _load_item_no_lock - Hyperbolic embedding validation result: {validated is not None}")
                            if validated is None:
                                logger.warning(f"[_load_item_no_lock] Hyperbolic embedding validation failed for assembly {item_id}, setting to None.")
                                instance.hyperbolic_embedding = None
                            else:
                                instance.hyperbolic_embedding = validated
                        except Exception as e_val:
                            logger.error(f"[_load_item_no_lock] Error validating hyperbolic embedding for assembly {item_id}: {str(e_val)}")
                            print(f"[PERSISTENCE] _load_item_no_lock ERROR - Hyperbolic embedding validation: {e_val}")
                            instance.hyperbolic_embedding = None
                            
                except Exception as e:
                    logger.error(f"[_load_item_no_lock] Error constructing MemoryAssembly {item_id}: {str(e)}", exc_info=True)
                    print(f"[PERSISTENCE] _load_item_no_lock ERROR - MemoryAssembly construction: {e}")
                    return None

                # If not in index, update
                if item_id not in self.memory_index:
                    print(f"[PERSISTENCE] _load_item_no_lock - Updating index for {item_id}")
                    self._update_index(
                        item_id,
                        file_path.relative_to(self.storage_path),
                        "assembly"
                    )
                print(f"[PERSISTENCE] _load_item_no_lock END - Successfully loaded assembly {item_id}")
                return instance

            else:  # "memory" path
                try:
                    print(f"[PERSISTENCE] _load_item_no_lock - Creating MemoryEntry for {item_id}...")
                    instance = MemoryEntry(**item_dict)
                    print(f"[PERSISTENCE] _load_item_no_lock - MemoryEntry created successfully for {item_id}")
                    if 'id' not in item_dict:
                        instance.id = item_id
                except Exception as e:
                    logger.error(f"[_load_item_no_lock] Error constructing MemoryEntry {item_id}: {str(e)}", exc_info=True)
                    print(f"[PERSISTENCE] _load_item_no_lock ERROR - MemoryEntry construction: {e}")
                    return None

                # If geometry_manager is available, optionally validate embeddings
                if geometry_manager and instance.embedding is not None:
                    # If embedding is a list
                    if isinstance(instance.embedding, list):
                        try:
                            print(f"[PERSISTENCE] _load_item_no_lock - Validating memory embedding for {item_id}...")
                            validated = geometry_manager._validate_vector(
                                instance.embedding,
                                f"Loaded Memory Emb {item_id}"
                            )
                            print(f"[PERSISTENCE] _load_item_no_lock - Memory embedding validation result: {validated is not None}")
                            if validated is None:
                                logger.warning(f"[_load_item_no_lock] Embedding validation failed for memory {item_id}, setting to None.")
                                instance.embedding = None
                            else:
                                instance.embedding = validated
                        except Exception as e_val:
                            logger.error(f"[_load_item_no_lock] Error validating embedding for memory {item_id}: {str(e_val)}")
                            print(f"[PERSISTENCE] _load_item_no_lock ERROR - Memory embedding validation: {e_val}")
                            instance.embedding = None

                if item_id not in self.memory_index:
                    print(f"[PERSISTENCE] _load_item_no_lock - Updating index for memory {item_id}")
                    self._update_index(
                        item_id,
                        file_path.relative_to(self.storage_path),
                        "memory"
                    )
                print(f"[PERSISTENCE] _load_item_no_lock END - Successfully loaded memory {item_id}")
                return instance

        except Exception as e:
            logger.error(f"[_load_item_no_lock] Unexpected error loading {item_id}: {str(e)}", exc_info=True)
            print(f"[PERSISTENCE] _load_item_no_lock ERROR - Unexpected: {e}")
            return None

    def _update_index(self, item_id: str, relative_path: Path, item_type: str):
        """Update the in-memory index with minimal info (no lock)."""
        self.memory_index[item_id] = {
            'path': str(relative_path),
            'timestamp': time.time(),
            'type': item_type
        }

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
        logger.info("MemoryPersistence", "Shutting down...")
        try:
            loop = asyncio.get_running_loop()
            if not loop.is_running():
                logger.warning("MemoryPersistence", "No running event loop in shutdown")
                return
        except RuntimeError:
            logger.warning("MemoryPersistence", "No running event loop in shutdown")
            return

        await self._save_index()
        logger.info("MemoryPersistence", "Shutdown complete.")

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
