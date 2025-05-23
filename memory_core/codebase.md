# __init__.py

```py
# memory_core/__init__.py

"""Modular memory system for Lucid Recall"""

__version__ = "0.1.0"

from memory_core.enhanced_memory_client import EnhancedMemoryClient
from memory_core.memory_manager import MemoryManager

__all__ = ["EnhancedMemoryClient", "MemoryManager"]
```

# base.py

```py
# memory_core/base.py

import logging
import asyncio
import time
import os
import copy
import random
import json
import torch
import datetime
import shutil
import numpy as np
import uuid
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class BaseMemoryClient:
    """
    Base memory client that provides core memory functionality.
    
    This class handles:
    - Initialization of memory client
    - Management of background tasks
    - Memory persistence and loading
    - Connection state management
    """
    
    def __init__(self, 
                 tensor_server_url: str,
                 hpc_server_url: str,
                 session_id: Optional[str] = None,
                 user_id: Optional[str] = None,
                 storage_path: Optional[str] = None,
                 enable_persistence: bool = True,
                 memory_decay_rate: float = 0.01,
                 significance_threshold: float = 0.0,
                 ping_interval: float = 20.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 connection_timeout: float = 10.0,
                 **kwargs):
        """
        Initialize the memory client.
        
        Args:
            tensor_server_url: URL for tensor server WebSocket connection
            hpc_server_url: URL for HPC server WebSocket connection 
            session_id: Unique session identifier
            user_id: User identifier
            storage_path: Path to store persistent memories
            enable_persistence: Enable persistence of memories
            memory_decay_rate: Rate at which memory significance decays over time
            significance_threshold: Minimum significance threshold for keeping memories
            ping_interval: Interval in seconds for WebSocket ping messages
            max_retries: Maximum number of connection retry attempts
            retry_delay: Base delay between retry attempts
            connection_timeout: Timeout for connection attempts
        """
        # Core properties
        self.tensor_server_url = tensor_server_url
        self.hpc_server_url = hpc_server_url
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id or "default_user"
        
        # Connection parameters
        self.ping_interval = ping_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_timeout = connection_timeout
        
        # Storage configuration
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # Use absolute path based on the current directory
            self.storage_path = Path.cwd() / "memory/stored"
        self.persistence_enabled = enable_persistence
        self.memory_decay_rate = memory_decay_rate
        self.significance_threshold = significance_threshold
        
        # Runtime state
        self.initialized = False
        self.memories = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup storage path
        if self.persistence_enabled:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
        # Locks for async operations
        self._tensor_lock = asyncio.Lock()
        self._hpc_lock = asyncio.Lock()
        self._memory_lock = asyncio.Lock()
        
        # Current connections
        self._tensor_connection = None
        self._hpc_connection = None
        
        # Background tasks
        self._background_tasks = []
        
        logger.info(f"Initialized BaseMemoryClient with session_id={self.session_id}")
    
    async def initialize(self) -> bool:
        """
        Initialize the memory client with robust error handling and recovery.
        
        This loads memories from storage and starts background tasks
        for memory management, with comprehensive error handling and
        recovery mechanisms to ensure reliability.
        
        Returns:
            bool: Success status
        """
        try:
            # Load existing memories if persistence is enabled
            if self.persistence_enabled:
                try:
                    load_stats = await self._load_memories()
                    
                    # Log detailed statistics
                    logger.info(f"Memory loading statistics: {load_stats['loaded_count']} loaded, "
                               f"{load_stats['recovered_count']} recovered, {load_stats['failed_count']} failed")
                    
                    # Handle critical errors during loading
                    if load_stats['errors']:
                        for error in load_stats['errors'][:5]:  # Log first 5 errors
                            logger.error(f"Memory loading error: {error}")
                        
                        if len(load_stats['errors']) > 5:
                            logger.error(f"... and {len(load_stats['errors']) - 5} more errors")
                    
                    # If we had corrupted files but some memories loaded successfully, continue
                    if load_stats['corrupted_files'] and (load_stats['loaded_count'] > 0 or load_stats['recovered_count'] > 0):
                        logger.warning(f"Continuing with {load_stats['loaded_count'] + load_stats['recovered_count']} memories despite {len(load_stats['corrupted_files'])} corrupted files")
                    
                    # If we recovered memories from backups, log this success
                    if load_stats['recovered_count'] > 0:
                        logger.info(f"Successfully recovered {load_stats['recovered_count']} memories from backup files")
                        
                except Exception as e:
                    logger.error(f"Critical error loading memories: {e}", exc_info=True)
                    # Attempt to recover by recreating storage directory
                    try:
                        # Create backup of existing directory if it exists
                        if self.storage_path.exists():
                            backup_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            backup_path = self.storage_path.parent / f"{self.storage_path.name}_backup_{backup_timestamp}"
                            
                            try:
                                shutil.copytree(self.storage_path, backup_path)
                                logger.info(f"Created backup of storage directory at {backup_path}")
                            except Exception as backup_error:
                                logger.error(f"Failed to create backup of storage directory: {backup_error}")
                        
                        # Recreate storage directory
                        self.storage_path.mkdir(parents=True, exist_ok=True)
                        logger.info("Recreated storage directory after critical error")
                    except Exception as recovery_error:
                        logger.error(f"Failed to recover from memory loading error: {recovery_error}")
                        return False
            
            # Start background tasks with error handling
            try:
                self._start_background_tasks()
                logger.info("Successfully started background tasks")
            except Exception as e:
                logger.error(f"Error starting background tasks: {e}", exc_info=True)
                # Continue without background tasks - we can still function in a degraded state
                logger.warning("Continuing without background tasks - memory persistence may be affected")
            
            self.initialized = True
            logger.info(f"Memory client initialized with {len(self.memories)} memories")
            return True
            
        except Exception as e:
            logger.error(f"Unrecoverable error initializing memory client: {e}", exc_info=True)
            return False
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for memory management."""
        # Create memory management task
        memory_task = asyncio.create_task(self._memory_management_loop())
        memory_task.set_name("memory_management_loop")
        self._background_tasks.append(memory_task)
        logger.info("Started memory management background task")
        
    async def _memory_management_loop(self) -> None:
        """
        Background task for memory management.
        
        This task runs periodically to:
        - Persist memories to storage
        - Prune low-significance memories
        - Update memory significances
        """
        consecutive_failures = 0
        max_consecutive_failures = 5
        delay_base = 30  # seconds between memory management cycles
        delay_backoff_factor = 1.5  # Backoff factor for consecutive failures
        max_delay = 300  # Maximum delay (5 minutes)
        
        while True:
            try:
                # Dynamic delay based on recent failures
                current_delay = min(delay_base * (delay_backoff_factor ** consecutive_failures), max_delay)
                
                # Distribute work over time to avoid CPU spikes
                # First persist memories
                logger.debug("Running memory persistence cycle")
                persist_start = time.time()
                success_count, error_count = await self._persist_memories()
                persist_duration = time.time() - persist_start
                
                # Log performance metrics for persistence cycle
                if success_count > 0 or error_count > 0:
                    logger.info(f"Memory persistence cycle took {persist_duration:.2f}s: {success_count} succeeded, {error_count} failed")
                
                # Wait a small interval before next task to avoid CPU spiking
                await asyncio.sleep(max(0.1, current_delay * 0.2))
                
                # Then prune memories if necessary
                prune_start = time.time()
                await self._prune_memories()
                prune_duration = time.time() - prune_start
                
                if prune_duration > 1.0:
                    logger.info(f"Memory pruning took {prune_duration:.2f}s")
                
                # Reset failure counter since everything worked
                if consecutive_failures > 0:
                    consecutive_failures = 0
                    logger.info("Memory management recovered after previous failures")
                
                # Wait until next cycle (adjusted for work already done)
                elapsed = time.time() - persist_start
                next_delay = max(0.1, current_delay - elapsed)
                logger.debug(f"Next memory management cycle in {next_delay:.1f}s")
                await asyncio.sleep(next_delay)
                
            except asyncio.CancelledError:
                logger.info("Memory management loop was cancelled, exiting")
                break
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Error in memory management loop (attempt {consecutive_failures}): {e}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(f"Too many consecutive failures ({consecutive_failures}) in memory management loop, increasing delay")
                    # Don't break - keep trying but with increased delay to reduce system load
                
                # Wait with exponential backoff
                backoff_delay = min(delay_base * (delay_backoff_factor ** consecutive_failures), max_delay)
                logger.info(f"Will retry memory management in {backoff_delay:.1f}s")
                await asyncio.sleep(backoff_delay)
    
    async def _load_memories(self) -> Dict[str, Any]:
        """
        Load memories from storage with robust error handling and recovery mechanisms.
        
        Returns:
            Dict with loading statistics
        """
        # Track loading statistics
        stats = {
            "total_files": 0,
            "loaded_count": 0,
            "failed_count": 0,
            "recovered_count": 0,
            "corrupted_files": [],
            "backup_files_used": [],
            "errors": []
        }
        
        try:
            # Ensure storage directory exists
            if not self.storage_path.exists():
                logger.warning(f"Storage path does not exist: {self.storage_path}")
                self.storage_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created storage directory: {self.storage_path}")
                return stats
            
            # Get list of memory files
            memory_files = list(self.storage_path.glob("*.json"))
            backup_files = list(self.storage_path.glob("*.json.bak"))
            temp_files = list(self.storage_path.glob("*.json.tmp"))
            
            stats["total_files"] = len(memory_files)
            logger.info(f"Found {len(memory_files)} memory files, {len(backup_files)} backup files, and {len(temp_files)} temporary files in {self.storage_path}")
            
            # Clean up temporary files (these might be from interrupted writes)
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
            
            # Create a memory ID to file path mapping
            memory_id_to_file = {}
            memory_id_to_backup = {}
            
            # Map regular files
            for file_path in memory_files:
                memory_id = file_path.stem
                memory_id_to_file[memory_id] = file_path
            
            # Map backup files
            for backup_path in backup_files:
                memory_id = backup_path.stem  # This will have .json.bak suffix removed
                memory_id_to_backup[memory_id] = backup_path
            
            # Process each memory file with exponential backoff retry
            max_retries = self.max_retries if hasattr(self, 'max_retries') else 3
            base_retry_delay = self.retry_delay if hasattr(self, 'retry_delay') else 1.0
            
            # Collect successfully loaded memory IDs to avoid duplicates
            loaded_memory_ids = set()
            
            # First pass: Try to load from primary files
            for memory_id, file_path in memory_id_to_file.items():
                logger.debug(f"Loading memory {memory_id} from {file_path}")
                
                # Skip if already loaded
                if memory_id in loaded_memory_ids:
                    logger.debug(f"Memory {memory_id} already loaded, skipping")
                    continue
                
                # Implement retry logic with exponential backoff
                success = False
                last_error = None
                
                for retry in range(max_retries):
                    try:
                        # Check file integrity
                        if file_path.stat().st_size == 0:
                            logger.warning(f"Empty memory file detected: {file_path}")
                            stats["corrupted_files"].append(str(file_path))
                            break
                        
                        # Load and parse memory file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            memory = json.load(f)
                        
                        # Validate memory structure
                        required_keys = ['id', 'content', 'timestamp']
                        if not all(key in memory for key in required_keys):
                            logger.warning(f"Memory file {file_path} missing required keys")
                            stats["corrupted_files"].append(str(file_path))
                            break
                        
                        # Convert embedding from list to tensor if present
                        if 'embedding' in memory and isinstance(memory['embedding'], list):
                            try:
                                memory['embedding'] = torch.tensor(
                                    memory['embedding'], 
                                    dtype=torch.float32, 
                                    device=self.device
                                )
                            except Exception as e:
                                logger.warning(f"Failed to convert embedding to tensor for memory {memory_id}: {e}")
                                # Continue with list embedding rather than failing
                                pass
                        
                        # Add to memory collection if not already present
                        if memory_id not in loaded_memory_ids:
                            self.memories.append(memory)
                            loaded_memory_ids.add(memory_id)
                            stats["loaded_count"] += 1
                            logger.debug(f"Successfully loaded memory {memory_id}")
                        
                        # Mark as successful
                        success = True
                        break
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Corrupted memory file {file_path} (attempt {retry+1}/{max_retries}): {e}")
                        last_error = e
                        # Wait before retry with exponential backoff and jitter
                        jitter = random.uniform(0.8, 1.2)
                        await asyncio.sleep(base_retry_delay * (2 ** retry) * jitter)
                    
                    except Exception as e:
                        logger.warning(f"Error loading memory {memory_id} (attempt {retry+1}/{max_retries}): {e}")
                        last_error = e
                        # Wait before retry with exponential backoff and jitter
                        jitter = random.uniform(0.8, 1.2)
                        await asyncio.sleep(base_retry_delay * (2 ** retry) * jitter)
                
                # If primary file failed, try backup file if available
                if not success and memory_id in memory_id_to_backup:
                    backup_path = memory_id_to_backup[memory_id]
                    logger.info(f"Attempting to recover memory {memory_id} from backup file {backup_path}")
                    
                    try:
                        # Load from backup
                        with open(backup_path, 'r', encoding='utf-8') as f:
                            memory = json.load(f)
                        
                        # Validate memory structure
                        required_keys = ['id', 'content', 'timestamp']
                        if all(key in memory for key in required_keys):
                            # Convert embedding from list to tensor if present
                            if 'embedding' in memory and isinstance(memory['embedding'], list):
                                try:
                                    memory['embedding'] = torch.tensor(
                                        memory['embedding'], 
                                        dtype=torch.float32, 
                                        device=self.device
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to convert embedding to tensor for memory {memory_id} from backup: {e}")
                                    # Continue with list embedding rather than failing
                                    pass
                            
                            # Add to memory collection if not already present
                            if memory_id not in loaded_memory_ids:
                                self.memories.append(memory)
                                loaded_memory_ids.add(memory_id)
                                stats["recovered_count"] += 1
                                stats["backup_files_used"].append(str(backup_path))
                                logger.info(f"Successfully recovered memory {memory_id} from backup")
                                
                                # Restore the backup to the main file
                                try:
                                    shutil.copy2(backup_path, file_path)
                                    logger.info(f"Restored backup file to primary location: {file_path}")
                                except Exception as e:
                                    logger.error(f"Failed to restore backup to primary location: {e}")
                                
                                success = True
                        else:
                            logger.warning(f"Backup file {backup_path} missing required keys")
                    
                    except Exception as e:
                        logger.warning(f"Failed to recover memory {memory_id} from backup: {e}")
                
                # If all attempts failed, record the failure
                if not success:
                    stats["failed_count"] += 1
                    if last_error:
                        error_msg = f"Failed to load memory {memory_id}: {str(last_error)}"
                        stats["errors"].append(error_msg)
            
            # Handle corrupted files if any
            if stats["corrupted_files"]:
                logger.warning(f"Detected {len(stats['corrupted_files'])} corrupted memory files")
                # Create backup directory for corrupted files
                backup_dir = self.storage_path / "corrupted_backups"
                backup_dir.mkdir(exist_ok=True)
                
                # Move corrupted files to backup directory
                for file_path_str in stats["corrupted_files"]:
                    file_path = Path(file_path_str)
                    if file_path.exists():
                        try:
                            # Create backup with timestamp
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            backup_path = backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
                            shutil.copy2(file_path, backup_path)
                            logger.info(f"Backed up corrupted file to {backup_path}")
                        except Exception as e:
                            logger.error(f"Failed to backup corrupted file {file_path}: {e}")
                            stats["errors"].append(f"Failed to backup corrupted file {file_path}: {str(e)}")
            
            # Sort memories by timestamp (newest first)
            self.memories.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Log loading summary
            logger.info(f"Memory loading complete: {stats['loaded_count']} loaded, {stats['recovered_count']} recovered from backups, {stats['failed_count']} failed")
            
            return stats
            
        except Exception as e:
            logger.error(f"Critical error during memory loading: {e}", exc_info=True)
            stats["errors"].append(f"Critical error: {str(e)}")
            return stats
    
    async def _persist_memories(self) -> Tuple[int, int]:
        """
        Persist memories to disk with asynchronous, non-blocking I/O and robust error handling.
        Handles complex data types like NumPy arrays and PyTorch tensors.
        
        This implementation uses aiofiles for non-blocking file I/O and processes memories
        in configurable-sized batches to prevent long-running operations from blocking
        other critical tasks.
        
        Returns:
            Tuple[int, int]: A tuple containing (success_count, error_count)
        """
        if not self.persistence_enabled:
            return (0, 0)
            
        if not self.memories:
            logger.debug("No memories to persist")
            return (0, 0)
            
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
            
        # Get settings for batch processing
        # These can be configured in __init__ or derived from system resources
        batch_size = getattr(self, 'persistence_batch_size', 10)  # Default to 10 memories per batch
        
        async with self._memory_lock:
            persist_start = time.time()
            success_count = 0
            error_count = 0
            
            # Track which memories we've attempted to persist
            attempted_memories = set()
            
            # First, try to persist high-significance memories
            significance_priority_threshold = getattr(self, 'significance_priority_threshold', 0.7)
            high_sig_memories = [m for m in self.memories if m.get('significance', 0.0) > significance_priority_threshold]
            if high_sig_memories:
                logger.info(f"Prioritizing persistence of {len(high_sig_memories)} high-significance memories")
            
            # Combine high significance memories with regular memories, prioritizing high significance ones
            prioritized_memories = high_sig_memories + [m for m in self.memories if m not in high_sig_memories]
            
            # Process memories in batches
            for i in range(0, len(prioritized_memories), batch_size):
                batch = prioritized_memories[i:i+batch_size]
                batch_tasks = []
                
                for memory in batch:
                    memory_id = memory.get('id')
                    if not memory_id:
                        logger.warning("Found memory without ID, skipping persistence")
                        error_count += 1
                        continue
                        
                    # Track that we've attempted this memory
                    attempted_memories.add(memory_id)
                    
                    # Create persistence task for this memory
                    task = self._persist_single_memory(memory)
                    batch_tasks.append(task)
                
                # Wait for all batch tasks to complete
                if batch_tasks:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process results
                    for result in batch_results:
                        if isinstance(result, Exception):
                            logger.error(f"Exception during batch persistence: {result}")
                            error_count += 1
                        elif result is True:
                            success_count += 1
                        else:
                            error_count += 1
            
            # Log persistence statistics
            persist_time = time.time() - persist_start
            logger.info(f"Memory persistence completed in {persist_time:.2f}s: {success_count} succeeded, {error_count} failed")
            
            # Check for any memories that weren't attempted (this shouldn't happen, but just in case)
            missing_memories = [m.get('id') for m in self.memories if m.get('id') and m.get('id') not in attempted_memories]
            if missing_memories:
                logger.warning(f"Found {len(missing_memories)} memories that weren't attempted to be persisted: {missing_memories[:5]}")
            
            return (success_count, error_count)

    async def _persist_single_memory(self, memory: dict) -> bool:
        """
        Persist a single memory with robust error handling and retry mechanism.
        Uses asynchronous file I/O to prevent blocking the event loop.
        
        Args:
            memory: The memory to persist
            
        Returns:
            bool: True if successful, False otherwise
        """
        memory_id = memory.get('id')
        if not memory_id:
            logger.warning("Attempted to persist memory without ID")
            return False
            
        file_path = self.storage_path / f"{memory_id}.json"
        temp_file_path = self.storage_path / f"{memory_id}.json.tmp"
        backup_file_path = self.storage_path / f"{memory_id}.json.bak"
        
        # Set up retry parameters
        max_retries = self.max_retries if hasattr(self, 'max_retries') else 3
        base_retry_delay = self.retry_delay if hasattr(self, 'retry_delay') else 0.5  # seconds
        
        for retry_count in range(max_retries):
            try:
                # Create a deep copy of the memory to avoid modifying the original
                memory_copy = copy.deepcopy(memory)
                
                # Convert any NumPy arrays or PyTorch tensors to Python lists
                memory_copy = self._convert_numpy_to_python(memory_copy)
                
                # Convert to JSON string
                json_content = json.dumps(memory_copy, ensure_ascii=False, indent=2)
                
                # Try to import aiofiles for async file I/O
                try:
                    import aiofiles
                    has_aiofiles = True
                except ImportError:
                    has_aiofiles = False
                    logger.warning("aiofiles package not found, falling back to synchronous I/O")
                
                # Write to temp file using async I/O if available
                if has_aiofiles:
                    async with aiofiles.open(temp_file_path, 'w', encoding='utf-8') as f:
                        await f.write(json_content)
                else:
                    # Fallback to synchronous I/O inside executor to avoid blocking
                    await asyncio.to_thread(self._write_sync, temp_file_path, json_content)
                        
                # If the file exists, create a backup before overwriting
                if file_path.exists():
                    try:
                        # Use asyncio.to_thread to prevent blocking on file copy
                        await asyncio.to_thread(shutil.copy2, file_path, backup_file_path)
                    except Exception as e:
                        logger.warning(f"Failed to create backup for memory {memory_id}: {e}")
                
                # Rename temporary file to actual file (using to_thread to avoid blocking)
                await asyncio.to_thread(os.replace, temp_file_path, file_path)
                
                # Verify file integrity
                try:
                    # Verify file asynchronously if possible
                    if has_aiofiles:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            content = await f.read()
                            _ = json.loads(content)  # Verify it's valid JSON
                    else:
                        # Fallback to synchronous verification in a thread
                        await asyncio.to_thread(self._verify_json_file, file_path)
                    
                    # Remove backup if everything succeeded (asynchronously)
                    if backup_file_path.exists():
                        await asyncio.to_thread(os.remove, backup_file_path)
                        
                    # Success
                    return True
                        
                except json.JSONDecodeError:
                    logger.error(f"Memory file {file_path} contains invalid JSON after writing")
                    # Restore from backup if verification failed
                    if backup_file_path.exists():
                        try:
                            await asyncio.to_thread(os.replace, backup_file_path, file_path)
                            logger.info(f"Restored memory {memory_id} from backup after verification failure")
                        except Exception as e:
                            logger.error(f"Failed to restore backup for memory {memory_id}: {e}")
                    return False
                    
            except Exception as e:
                error_msg = f"Error persisting memory {memory_id} (attempt {retry_count+1}/{max_retries}): {e}"
                if retry_count < max_retries - 1:
                    logger.warning(error_msg + ", retrying...")
                    # Exponential backoff with jitter
                    backoff_time = base_retry_delay * (2 ** retry_count) * (0.5 + 0.5 * random.random())
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(error_msg + ", giving up")
                    return False
        
        # Should never reach here due to return in the loop, but just in case
        return False

    def _write_sync(self, file_path, content):
        """Synchronous file write operation to be used with asyncio.to_thread."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _verify_json_file(self, file_path):
        """Synchronous JSON file verification to be used with asyncio.to_thread."""
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)

    async def _prune_memories(self) -> None:
        """
        Prune low-significance memories based on threshold.
        """
        try:
            async with self._memory_lock:
                before_count = len(self.memories)
                
                # Filter out memories below significance threshold
                self.memories = [m for m in self.memories if m.get('significance', 0) >= self.significance_threshold]
                
                after_count = len(self.memories)
                if before_count > after_count:
                    logger.info(f"Pruned {before_count - after_count} low-significance memories")
        except Exception as e:
            logger.error(f"Error pruning memories: {e}")
    
    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Normalize an embedding vector.
        
        Args:
            embedding: The embedding tensor to normalize
            
        Returns:
            Normalized embedding tensor
        """
        if isinstance(embedding, list):
            embedding = torch.tensor(embedding, device=self.device)
        embedding = embedding.to(self.device)
        norm = torch.norm(embedding, p=2)
        return embedding / norm if norm > 0 else embedding
    
    async def cleanup(self) -> None:
        """
        Cleanup resources and ensure all memories are persisted.
        
        This method should be called before shutting down the application to ensure
        all memories are properly saved and resources are released.
        """
        logger.info("Starting memory client cleanup process")
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                logger.info(f"Cancelling background task: {task.get_name() if hasattr(task, 'get_name') else task}")
                task.cancel()
        
        # Wait for tasks to complete cancellation
        if self._background_tasks:
            try:
                # Wait with timeout to avoid hanging
                await asyncio.wait(self._background_tasks, timeout=5)
                logger.info("Background tasks cancelled")
            except Exception as e:
                logger.error(f"Error waiting for background tasks to cancel: {e}")
        
        # Force immediate persistence of all memories
        if self.persistence_enabled:
            try:
                logger.info("Forcing final memory persistence before shutdown")
                # Use a shorter timeout for the final persistence
                persistence_task = asyncio.create_task(self._persist_memories())
                try:
                    # Wait with timeout to avoid hanging during shutdown
                    await asyncio.wait_for(persistence_task, timeout=10)
                    logger.info("Final memory persistence completed successfully")
                except asyncio.TimeoutError:
                    logger.warning("Final memory persistence timed out after 10 seconds")
                except Exception as e:
                    logger.error(f"Error during final memory persistence: {e}")
                    
                # Verify persistence by checking files
                try:
                    # Count memory files in storage directory
                    memory_files = list(self.storage_path.glob("*.json"))
                    memory_count = len(self.memories)
                    logger.info(f"Verification: {len(memory_files)} memory files on disk, {memory_count} in memory")
                    
                    # Check for missing files
                    memory_ids = [m.get('id') for m in self.memories if m.get('id')]
                    file_ids = [f.stem for f in memory_files]
                    missing_ids = [mid for mid in memory_ids if mid not in file_ids]
                    
                    if missing_ids:
                        logger.warning(f"Found {len(missing_ids)} memories not persisted to disk")
                        # Emergency persistence for missing memories
                        for memory in self.memories:
                            if memory.get('id') in missing_ids:
                                try:
                                    memory_id = memory.get('id')
                                    file_path = self.storage_path / f"{memory_id}.json"
                                    memory_copy = memory.copy()
                                    
                                    # Convert tensor to list for JSON serialization
                                    if 'embedding' in memory_copy and isinstance(memory_copy['embedding'], torch.Tensor):
                                        memory_copy['embedding'] = memory_copy['embedding'].tolist()
                                    
                                    with open(file_path, 'w') as f:
                                        json.dump(memory_copy, f)
                                    logger.info(f"Emergency persistence for memory {memory_id} successful")
                                except Exception as e:
                                    logger.error(f"Emergency persistence failed for memory {memory.get('id')}: {e}")
                except Exception as e:
                    logger.error(f"Error during persistence verification: {e}")
            except Exception as e:
                logger.error(f"Error during final memory persistence: {e}", exc_info=True)
        
        # Release any other resources
        try:
            # Clear memory collections
            self.memories = []
            logger.info("Memory collections cleared")
        except Exception as e:
            logger.error(f"Error clearing memory collections: {e}")
        
        logger.info("Memory client cleanup completed")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory client statistics.
        
        Returns:
            Dict with statistics
        """
        try:
            latest_timestamp = max([m.get('timestamp', 0) for m in self.memories]) if self.memories else 0
        except Exception as e:
            logger.error(f"Error calculating latest timestamp: {e}")
            latest_timestamp = 0
            
        return {
            'memory_count': len(self.memories),
            'device': self.device,
            'storage_path': str(self.storage_path),
            'latest_timestamp': latest_timestamp,
            'initialized': self.initialized,
            'background_tasks': len(self._background_tasks)
        }
    
    async def force_persistence(self) -> Dict[str, Any]:
        """
        Force immediate persistence of all memories.
        This is used during shutdown to ensure no memories are lost.
        
        Returns:
            Dict with persistence statistics
        """
        logger.info("Forcing immediate persistence of all memories")
        
        if not self.persistence_enabled:
            logger.warning("Memory persistence is disabled, cannot force persistence")
            return {"status": "disabled", "persisted": 0, "failed": 0}
            
        if not self.memories:
            logger.info("No memories to persist")
            return {"status": "success", "persisted": 0, "failed": 0}
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Persistence statistics
        stats = {
            "status": "success",
            "persisted": 0,
            "failed": 0,
            "high_significance": 0,
            "start_time": time.time()
        }
        
        # First, prioritize high-significance memories
        high_sig_memories = [m for m in self.memories if m.get('significance', 0.0) > 0.7]
        if high_sig_memories:
            logger.info(f"Prioritizing {len(high_sig_memories)} high-significance memories for forced persistence")
            stats["high_significance"] = len(high_sig_memories)
        
        # Combine high significance memories with regular memories
        prioritized_memories = high_sig_memories + [m for m in self.memories if m not in high_sig_memories]
        
        # Use memory lock to prevent concurrent modifications
        async with self._memory_lock:
            for memory in prioritized_memories:
                memory_id = memory.get('id')
                if not memory_id:
                    logger.warning("Found memory without ID, skipping persistence")
                    stats["failed"] += 1
                    continue
                
                file_path = self.storage_path / f"{memory_id}.json"
                temp_file_path = self.storage_path / f"{memory_id}.json.tmp"
                backup_file_path = self.storage_path / f"{memory_id}.json.bak"
                
                # Set up retry parameters
                max_retries = 2  # Fewer retries during forced persistence to avoid long delays
                retry_delay = 0.2  # seconds
                
                for retry_count in range(max_retries):
                    try:
                        # Create a deep copy of the memory to avoid modifying the original
                        memory_copy = copy.deepcopy(memory)
                        
                        # Convert any NumPy arrays or PyTorch tensors to Python lists
                        memory_copy = self._convert_numpy_to_python(memory_copy)
                        
                        # Write to a temporary file first (atomic write operation)
                        with open(temp_file_path, 'w', encoding='utf-8') as f:
                            json.dump(memory_copy, f, ensure_ascii=False, indent=2)
                            
                        # If the file exists, create a backup before overwriting
                        if file_path.exists():
                            try:
                                shutil.copy2(file_path, backup_file_path)
                            except Exception as e:
                                logger.warning(f"Failed to create backup for memory {memory_id}: {e}")
                        
                        # Rename temporary file to actual file (atomic operation)
                        os.replace(temp_file_path, file_path)
                        
                        # Verify file integrity
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                _ = json.load(f)  # Just load to verify it's valid JSON
                            # If we get here, the file is valid JSON
                            stats["persisted"] += 1
                            
                            # Remove backup if everything succeeded
                            if backup_file_path.exists():
                                os.remove(backup_file_path)
                                
                            # Break out of retry loop on success
                            break
                        except json.JSONDecodeError:
                            logger.error(f"Memory file {file_path} contains invalid JSON after writing")
                            # Restore from backup if verification failed
                            if backup_file_path.exists():
                                try:
                                    os.replace(backup_file_path, file_path)
                                    logger.info(f"Restored memory {memory_id} from backup after verification failure")
                                except Exception as e:
                                    logger.error(f"Failed to restore backup for memory {memory_id}: {e}")
                            stats["failed"] += 1
                    except Exception as e:
                        error_msg = f"Error persisting memory {memory_id} (attempt {retry_count+1}/{max_retries}): {e}"
                        if retry_count < max_retries - 1:
                            logger.warning(error_msg + ", retrying...")
                            # Shorter backoff during forced persistence
                            await asyncio.sleep(retry_delay)
                        else:
                            logger.error(error_msg + ", giving up")
                            stats["failed"] += 1
        
        # Calculate persistence time
        stats["elapsed_time"] = time.time() - stats["start_time"]
        logger.info(f"Forced persistence completed in {stats['elapsed_time']:.2f}s: {stats['persisted']} succeeded, {stats['failed']} failed")
        
        return stats
    
    def _convert_numpy_to_python(self, obj):
        """
        Recursively convert NumPy arrays and PyTorch tensors to Python lists.
        Also handles other non-serializable types.
        
        Args:
            obj: The object to convert
            
        Returns:
            The converted object with all NumPy arrays and PyTorch tensors converted to lists
        """
        # Handle None
        if obj is None:
            return None
            
        # Handle NumPy arrays
        if hasattr(obj, '__module__') and obj.__module__ == 'numpy':
            if hasattr(obj, 'tolist'):
                try:
                    return obj.tolist()
                except Exception as e:
                    logger.warning(f"Error converting NumPy array to list: {e}")
                    return str(obj)
            return str(obj)
            
        # Handle PyTorch tensors
        if hasattr(obj, '__module__') and 'torch' in obj.__module__:
            if hasattr(obj, 'tolist'):
                try:
                    return obj.tolist()
                except Exception as e:
                    logger.warning(f"Error converting PyTorch tensor to list: {e}")
                    return str(obj)
            if hasattr(obj, 'detach'):
                try:
                    detached = obj.detach()
                    if hasattr(detached, 'numpy'):
                        numpy_array = detached.numpy()
                        if hasattr(numpy_array, 'tolist'):
                            return numpy_array.tolist()
                    return str(obj)
                except Exception as e:
                    logger.warning(f"Error detaching PyTorch tensor: {e}")
                    return str(obj)
            return str(obj)
            
        # Handle dictionaries
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_python(v) for k, v in obj.items()}
            
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._convert_numpy_to_python(item) for item in obj]
            
        # Handle sets
        if isinstance(obj, set):
            return [self._convert_numpy_to_python(item) for item in obj]
            
        # Handle other non-serializable types
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)
```

# cognitive.py

```py
# memory_client/cognitive.py

import logging
import math
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class CognitiveMemoryMixin:
    """
    Mixin that applies a cognitive-inspired memory approach: 
    e.g., forgetting curve, spaced repetition, etc.
    """

    def __init__(self):
        self._memory_access_counts = defaultdict(int)
        self._memory_last_access = {}
        self._memory_decay_rate = 0.05  # 5% decay per day

    async def _apply_memory_decay(self):
        """Periodically apply memory decay."""
        pass

    async def record_memory_access(self, memory_id: str):
        """Record that a memory was accessed (reinforcement)."""
        pass

    async def associate_memories(self, memory_id1: str, memory_id2: str, strength: float = 0.5) -> bool:
        """Create an association between two memories."""
        return False

    async def get_associated_memories(self, memory_id: str, min_strength: float = 0.3) -> List[Dict[str, Any]]:
        return []

```

# connectivity.py

```py
# memory_core/connectivity.py

import json
import asyncio
import websockets
import logging
from typing import Optional, Dict, Any
import time
import traceback

logger = logging.getLogger(__name__)

class ConnectivityMixin:
    """
    Mixin that handles WebSocket connectivity to the tensor and HPC servers.
    Requires self._tensor_lock, self._hpc_lock, etc. from the base class.
    """

    async def connect(self) -> bool:
        """Connect to the tensor and HPC servers."""
        if self._connected:
            return True

        logger.info("Connecting to tensor and HPC servers")
        tensor_connected = await self._connect_to_tensor_server()
        hpc_connected = await self._connect_to_hpc_server()
        
        self._connected = tensor_connected and hpc_connected
        return self._connected
    
    async def _connect_to_tensor_server(self) -> bool:
        """Connect to the tensor server with retry logic."""
        retry_count = 0
        max_retries = self.max_retries
        delay = self.retry_delay
        
        while retry_count < max_retries:
            try:
                logger.info(f"Connecting to tensor server at {self.tensor_server_url} (attempt {retry_count + 1}/{max_retries})")
                
                # Use a timeout for the connection attempt
                connection = await asyncio.wait_for(
                    websockets.connect(self.tensor_server_url, ping_interval=self.ping_interval),
                    timeout=self.connection_timeout
                )
                
                async with self._tensor_lock:
                    self._tensor_connection = connection
                    
                logger.info("Successfully connected to tensor server")
                return True
                
            except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                logger.warning(f"Failed to connect to tensor server: {e}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.error("Max retries reached for tensor server connection")
                    return False
                    
                # Exponential backoff
                wait_time = delay * (2 ** (retry_count - 1))
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Unexpected error connecting to tensor server: {e}")
                logger.error(traceback.format_exc())
                return False
    
    async def _connect_to_hpc_server(self) -> bool:
        """Connect to the HPC server with retry logic."""
        retry_count = 0
        max_retries = self.max_retries
        delay = self.retry_delay
        
        while retry_count < max_retries:
            try:
                logger.info(f"Connecting to HPC server at {self.hpc_server_url} (attempt {retry_count + 1}/{max_retries})")
                
                # Use a timeout for the connection attempt
                connection = await asyncio.wait_for(
                    websockets.connect(self.hpc_server_url, ping_interval=self.ping_interval),
                    timeout=self.connection_timeout
                )
                
                async with self._hpc_lock:
                    self._hpc_connection = connection
                    
                logger.info("Successfully connected to HPC server")
                return True
                
            except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                logger.warning(f"Failed to connect to HPC server: {e}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.error("Max retries reached for HPC server connection")
                    return False
                    
                # Exponential backoff
                wait_time = delay * (2 ** (retry_count - 1))
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Unexpected error connecting to HPC server: {e}")
                logger.error(traceback.format_exc())
                return False
    
    async def _get_tensor_connection(self) -> Optional[websockets.WebSocketClientProtocol]:
        """Get the tensor server connection, creating a new one if necessary."""
        async with self._tensor_lock:
            # Check if connection exists and is open
            if self._tensor_connection and hasattr(self._tensor_connection, 'open') and self._tensor_connection.open:
                try:
                    # Verify connection is actually responsive with a ping
                    pong_waiter = await self._tensor_connection.ping()
                    await asyncio.wait_for(pong_waiter, timeout=2.0)
                    return self._tensor_connection
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    logger.warning("Tensor connection ping failed, will create new connection")
                    # Continue to create new connection
                except Exception as e:
                    logger.warning(f"Tensor connection check failed: {e}")
                    # Continue to create new connection
                    
            # Connection closed or doesn't exist, create new one
            try:
                # Add exponential backoff for reconnection attempts
                retry_count = 0
                max_retries = 3
                base_delay = 0.5
                
                while retry_count < max_retries:
                    try:
                        logger.info("Creating new tensor server connection")
                        connection = await asyncio.wait_for(
                            websockets.connect(self.tensor_server_url, ping_interval=self.ping_interval),
                            timeout=self.connection_timeout
                        )
                        self._tensor_connection = connection
                        return connection
                    except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise
                        wait_time = base_delay * (2 ** (retry_count - 1))
                        logger.warning(f"Tensor connection attempt {retry_count} failed: {e}. Retrying in {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"Failed to create tensor connection after retries: {e}")
                return None
    
    async def _get_hpc_connection(self) -> Optional[websockets.WebSocketClientProtocol]:
        """Get the HPC server connection, creating a new one if necessary."""
        async with self._hpc_lock:
            # Check if connection exists and is open
            if self._hpc_connection and hasattr(self._hpc_connection, 'open') and self._hpc_connection.open:
                try:
                    # Verify connection is actually responsive with a ping
                    pong_waiter = await self._hpc_connection.ping()
                    await asyncio.wait_for(pong_waiter, timeout=2.0)
                    return self._hpc_connection
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    logger.warning("HPC connection ping failed, will create new connection")
                    # Continue to create new connection
                except Exception as e:
                    logger.warning(f"HPC connection check failed: {e}")
                    # Continue to create new connection
                    
            # Connection closed or doesn't exist, create new one
            try:
                # Add exponential backoff for reconnection attempts
                retry_count = 0
                max_retries = 3
                base_delay = 0.5
                
                while retry_count < max_retries:
                    try:
                        logger.info("Creating new HPC server connection")
                        connection = await asyncio.wait_for(
                            websockets.connect(self.hpc_server_url, ping_interval=self.ping_interval),
                            timeout=self.connection_timeout
                        )
                        self._hpc_connection = connection
                        return connection
                    except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise
                        wait_time = base_delay * (2 ** (retry_count - 1))
                        logger.warning(f"HPC connection attempt {retry_count} failed: {e}. Retrying in {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"Failed to create HPC connection after retries: {e}")
                return None

```

# consolidation.py

```py
# memory_client/consolidation.py

import time
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class MemoryConsolidationMixin:
    """
    Mixin for memory consolidation - grouping related memories, summarizing them, etc.
    """

    async def _consolidate_memories(self):
        """Periodically consolidate related memories."""
        try:
            logger.info("Starting memory consolidation process")
            recent_memories = await self._get_recent_memories(days=7)
            if len(recent_memories) < 5:
                logger.info("Not enough recent memories for consolidation")
                return

            clusters = await self._cluster_similar_memories(recent_memories)
            for cluster_id, cluster_mems in clusters.items():
                if len(cluster_mems) < 3:
                    continue
                summary = await self._summarize_memory_cluster(cluster_mems)
                if summary:
                    significance = max(m.get("significance", 0.5) for m in cluster_mems)
                    memory_ids = [m["id"] for m in cluster_mems if "id" in m]
                    await self.store_significant_memory(
                        text=summary,
                        memory_type="consolidated",
                        metadata={
                            "source_count": len(cluster_mems),
                            "source_ids": memory_ids
                        },
                        min_significance=min(significance + 0.1, 0.95)
                    )
            logger.info("Memory consolidation completed")
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")

    async def _get_recent_memories(self, days=7) -> List[Dict[str, Any]]:
        """Example method for retrieving recent memories from server."""
        return []

    async def _cluster_similar_memories(self, memories: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Cluster using DBSCAN or similar."""
        if not memories:
            return {}
        # Example placeholder logic
        return {}

    async def _summarize_memory_cluster(self, memories: List[Dict[str, Any]]) -> str:
        """Summarize a cluster of memory texts."""
        if not memories:
            return ""
        return "CONSOLIDATED MEMORY: ..."

```

# emotion.py

```py
# memory_core/emotion.py

import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class EmotionMixin:
    """
    Mixin that handles emotion detection and tracking in the memory system.
    Allows for detecting and storing emotional context of conversations.
    """

    def __init__(self):
        # Initialize emotion tracking
        self.emotion_tracking = {
            "current_emotion": "neutral",
            "emotion_history": [],
            "emotional_triggers": {}
        }
        # Initialize emotions collection if it doesn't exist
        if not hasattr(self, "emotions"):
            self.emotions = {}

    async def detect_emotion(self, text: str) -> str:
        """
        Detect emotion from text. Uses the HPC service for emotion analysis.
        
        Args:
            text: The text to analyze for emotion
            
        Returns:
            Detected emotion as string
        """
        try:
            connection = await self._get_hpc_connection()
            if not connection:
                logger.error("Cannot detect emotion: No HPC connection")
                return "neutral"
                
            # Create request payload
            payload = {
                "type": "emotion",
                "text": text
            }
            
            # Send request
            await connection.send(json.dumps(payload))
            
            # Get response
            response = await connection.recv()
            data = json.loads(response)
            
            if 'emotion' in data:
                emotion = data['emotion']
                
                # Update emotion tracking
                self.emotion_tracking["current_emotion"] = emotion
                self.emotion_tracking["emotion_history"].append({
                    "text": text,
                    "emotion": emotion,
                    "timestamp": self._get_timestamp()
                })
                
                # Keep history at a reasonable size
                if len(self.emotion_tracking["emotion_history"]) > 50:
                    self.emotion_tracking["emotion_history"] = self.emotion_tracking["emotion_history"][-50:]
                        
                return emotion
            else:
                logger.warning("No emotion data in response")
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return "neutral"
    
    async def detect_emotional_context(self, text: str) -> Dict[str, Any]:
        """
        Detect and analyze emotional context from text.
        This is called by the voice agent to process emotions in transcripts.
        
        Args:
            text: The text to analyze for emotional context
            
        Returns:
            Dict with emotional context information
        """
        try:
            # First detect the primary emotion
            emotion = await self.detect_emotion(text)
            
            # Default emotional data
            timestamp = self._get_timestamp()
            emotional_data = {
                "text": text,
                "emotion": emotion,
                "timestamp": timestamp,
                "sentiment": 0.0,  # Neutral by default
                "emotions": {
                    emotion: 0.7  # Default confidence
                }
            }
            
            # Try to get more detailed emotion analysis from HPC if available
            try:
                connection = await self._get_hpc_connection()
                if connection:
                    # Create detailed emotion analysis request
                    payload = {
                        "type": "emotional_analysis",
                        "text": text
                    }
                    
                    # Send request and get response
                    await connection.send(json.dumps(payload))
                    response = await connection.recv()
                    data = json.loads(response)
                    
                    # Update with more detailed information if available
                    if 'emotions' in data:
                        emotional_data["emotions"] = data['emotions']
                    
                    if 'sentiment' in data:
                        emotional_data["sentiment"] = data['sentiment']
            except Exception as e:
                logger.warning(f"Error getting detailed emotional analysis: {e}")
            
            # Store this emotion in our collection
            self.emotions[str(timestamp)] = emotional_data
            
            # Keep emotions collection at a reasonable size
            if len(self.emotions) > 100:
                # Remove oldest entries
                timestamps = sorted([float(ts) for ts in self.emotions.keys()])
                cutoff = timestamps[len(timestamps) - 100]  # Keep only newest 100
                self.emotions = {ts: data for ts, data in self.emotions.items() 
                                if float(ts) >= cutoff}
            
            # Create a complete emotional context response
            context = {
                "current_emotion": emotion,
                "sentiment": emotional_data.get("sentiment", 0.0),
                "emotions": emotional_data.get("emotions", {}),
                "timestamp": timestamp,
                "text_analyzed": text
            }
            
            logger.info(f"Detected emotional context: {emotion} with sentiment {context['sentiment']:.2f}")
            return context
            
        except Exception as e:
            logger.error(f"Error detecting emotional context: {e}", exc_info=True)
            # Return basic neutral context in case of error
            return {
                "current_emotion": "neutral",
                "sentiment": 0.0,
                "emotions": {"neutral": 1.0},
                "timestamp": self._get_timestamp(),
                "text_analyzed": text,
                "error": str(e)
            }
        
    async def get_emotional_context(self, limit: int = 5) -> Dict[str, Any]:
        """
        Get the emotional context of recent conversations.
        
        Args:
            limit: Number of recent emotions to include
            
        Returns:
            Dict with emotional context information
        """
        recent_emotions = self.emotion_tracking["emotion_history"][-limit:] if \
            self.emotion_tracking["emotion_history"] else []
            
        return {
            "current_emotion": self.emotion_tracking["current_emotion"],
            "recent_emotions": recent_emotions,
            "emotional_triggers": self.emotion_tracking["emotional_triggers"]
        }
    
    async def get_emotional_history(self, limit: int = 5) -> str:
        """
        Get a formatted string of emotional history for RAG context.
        
        Args:
            limit: Number of recent emotions to include
            
        Returns:
            Formatted string of emotional history
        """
        if not hasattr(self, "emotions") or not self.emotions:
            return ""
        
        try:
            # Sort emotions by timestamp (newest first)
            sorted_emotions = sorted(
                self.emotions.items(),
                key=lambda x: float(x[0]),
                reverse=True
            )[:limit]
            
            parts = []
            for timestamp, data in sorted_emotions:
                sentiment = data.get("sentiment", 0)
                emotion = data.get("emotion", "unknown")
                emotions_dict = data.get("emotions", {})
                
                # Format timestamp
                date_str = self._format_timestamp(float(timestamp))
                
                # Describe sentiment
                if sentiment > 0.5:
                    sentiment_desc = "very positive"
                elif sentiment > 0.1:
                    sentiment_desc = "positive"
                elif sentiment > -0.1:
                    sentiment_desc = "neutral"
                elif sentiment > -0.5:
                    sentiment_desc = "negative"
                else:
                    sentiment_desc = "very negative"
                
                # Get top emotions
                emotions_list = []
                if emotions_dict:
                    top_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                    emotions_list = [f"{e}" for e, _ in top_emotions]
                else:
                    emotions_list = [emotion]
                
                emotions_str = ", ".join(emotions_list)
                parts.append(f"• {date_str}: {sentiment_desc} ({emotions_str})")
            
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"Error formatting emotional history: {e}")
            return ""
        
    async def store_emotional_trigger(self, trigger: str, emotion: str):
        """
        Store a trigger for a specific emotion.
        
        Args:
            trigger: The text/concept that triggered the emotion
            emotion: The emotion that was triggered
        """
        if trigger and emotion:
            # Add or update the trigger
            self.emotion_tracking["emotional_triggers"][trigger] = emotion
            logger.info(f"Stored emotional trigger: {trigger} -> {emotion}")
    
    async def store_emotional_context(self, context: Dict[str, Any]):
        """
        Store emotional context data in the memory system.
        
        Args:
            context: Emotional context dictionary containing emotion data
        """
        if not context:
            logger.warning("Cannot store empty emotional context")
            return
            
        try:
            # Store timestamp if not present
            if "timestamp" not in context:
                context["timestamp"] = self._get_timestamp()
                
            # Store in emotions collection
            timestamp = str(context["timestamp"])
            self.emotions[timestamp] = context
            
            # Update current emotion tracking
            if "current_emotion" in context:
                self.emotion_tracking["current_emotion"] = context["current_emotion"]
                
            # Add to emotion history
            history_entry = {
                "emotion": context.get("current_emotion", "neutral"),
                "timestamp": context["timestamp"],
                "text": context.get("text_analyzed", ""),
                "sentiment": context.get("sentiment", 0.0)
            }
            self.emotion_tracking["emotion_history"].append(history_entry)
            
            # Keep history at a reasonable size
            if len(self.emotion_tracking["emotion_history"]) > 50:
                self.emotion_tracking["emotion_history"] = self.emotion_tracking["emotion_history"][-50:]
                    
            logger.info(f"Stored emotional context: {context.get('current_emotion')} with sentiment {context.get('sentiment', 0.0):.2f}")
            
        except Exception as e:
            logger.error(f"Error storing emotional context: {e}", exc_info=True)
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format a timestamp as a human-readable date string"""
        try:
            import datetime
            return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return f"timestamp: {timestamp}"
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        return self._get_current_timestamp() if hasattr(self, "_get_current_timestamp") else time.time()
```

# enhanced_memory_client.py

```py
# memory_core/enhanced_memory_client.py

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import re
import time
import asyncio
import json
import uuid
from memory_core.base import BaseMemoryClient
from memory_core.tools import ToolsMixin
from memory_core.emotion import EmotionMixin
from memory_core.connectivity import ConnectivityMixin
from memory_core.personal_details import PersonalDetailsMixin
from memory_core.rag_context import RAGContextMixin

# Configure logger
logger = logging.getLogger(__name__)

class EnhancedMemoryClient(BaseMemoryClient,
                           ToolsMixin,
                           EmotionMixin,
                           ConnectivityMixin,
                           PersonalDetailsMixin,
                           RAGContextMixin):
    """
    Enhanced memory client that combines all mixins to provide a complete memory system.
    
    This class integrates all the functionality from the various mixins:
    - BaseMemoryClient: Core memory functionality and initialization
    - ConnectivityMixin: WebSocket connection handling for tensor and HPC servers
    - EmotionMixin: Emotion detection and tracking
    - ToolsMixin: Memory search and embedding tools
    - PersonalDetailsMixin: Personal information extraction and storage
    - RAGContextMixin: Advanced context generation for RAG
    """
    
    def __init__(self, tensor_server_url: str, 
                 hpc_server_url: str,
                 session_id: Optional[str] = None,
                 user_id: Optional[str] = None,
                 ping_interval: float = 20.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 connection_timeout: float = 10.0,
                 **kwargs):
        """
        Initialize the enhanced memory client.
        
        Args:
            tensor_server_url: URL for tensor server WebSocket connection
            hpc_server_url: URL for HPC server WebSocket connection
            session_id: Unique session identifier
            user_id: User identifier
            ping_interval: Interval in seconds to send ping messages to servers
            max_retries: Maximum number of retries for failed connections
            retry_delay: Base delay in seconds between retries
            connection_timeout: Timeout in seconds for establishing connections
            **kwargs: Additional configuration options
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize the base class
        super().__init__(
            tensor_server_url=tensor_server_url,
            hpc_server_url=hpc_server_url,
            session_id=session_id,
            user_id=user_id,
            ping_interval=ping_interval,
            max_retries=max_retries,
            retry_delay=retry_delay,
            connection_timeout=connection_timeout,
            **kwargs
        )
        
        # Explicitly initialize all mixins
        # This is important to ensure each mixin initializes its own state
        # We need to initialize before calling __init__ on the mixins to avoid redundant initialization
        self._connected = False
        
        # Now initialize mixins
        PersonalDetailsMixin.__init__(self)
        EmotionMixin.__init__(self)
        # ConnectivityMixin.__init__(self)  # This is done implicitly since it doesn't have an __init__
        ToolsMixin.__init__(self)
        RAGContextMixin.__init__(self)
        
        # Initialize topic suppression settings
        self._topic_suppression = {
            "enabled": True,
            "suppression_time": 3600,  # Default 1 hour in seconds
            "suppressed_topics": {}
        }
        
        # User context tracking
        self._user_preferences = {}
        self._conversation_history = []
        self._max_history_items = 50
        
        logger.info(f"Initialized EnhancedMemoryClient with session_id={session_id}")
    
    async def process_message(self, text: str, role: str = "user") -> None:
        """
        Process an incoming message to extract various information.
        
        Args:
            text: The message text
            role: The role of the sender (user or assistant)
        """
        # Only process user messages for personal details
        if role == "user":
            # Process for personal details
            await self.detect_and_store_personal_details(text, role)
            
            # Process for emotions
            await self.analyze_emotions(text)
        
        # Store message in memory with default significance
        await self.store_memory(
            content=text,
            metadata={"role": role, "type": "message", "timestamp": time.time()}
        )
        
        # Add to conversation history with pruning
        self._conversation_history.append({
            "role": role,
            "content": text,
            "timestamp": time.time()
        })
        
        # Prune history if needed
        if len(self._conversation_history) > self._max_history_items:
            self._conversation_history = self._conversation_history[-self._max_history_items:]
        
        logger.debug(f"Processed {role} message: {text[:50]}...")
    
    async def get_memory_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get all memory tools formatted for the LLM.
        
        Returns:
            Dict with all available memory tools
        """
        # Get standard memory tools
        memory_tools = await self.get_memory_tools()
        
        # Add personal details tool
        personal_tool = {
            "type": "function",
            "function": {
                "name": "get_personal_details",
                "description": "Retrieve personal details about the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Optional category of personal detail to retrieve (e.g., 'name', 'location', 'birthday', 'job', 'family')"
                        }
                    }
                }
            }
        }
        
        # Add emotion tool
        emotion_tool = {
            "type": "function",
            "function": {
                "name": "get_emotional_context",
                "description": "Get the current emotional context of the conversation",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
        
        # Add topic tracking tool
        topic_tool = {
            "type": "function",
            "function": {
                "name": "track_conversation_topic",
                "description": "Track the current conversation topic to prevent repetition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic being discussed"
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance of this topic (0-1)",
                            "default": 0.7
                        }
                    },
                    "required": ["topic"]
                }
            }
        }
        
        memory_tools.append(personal_tool)
        memory_tools.append(emotion_tool)
        memory_tools.append(topic_tool)
        
        return memory_tools
    
    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route tool calls to the appropriate handlers.
        
        Args:
            tool_name: The name of the tool to call
            args: The arguments for the tool
            
        Returns:
            The result of the tool call
        """
        # Prepare arguments object with proper validation
        validated_args = self._validate_tool_args(tool_name, args)
        if "error" in validated_args:
            return {"error": validated_args["error"], "success": False}
        
        # Set up retry parameters
        max_retries = 2
        retry_count = 0
        backoff_factor = 1.5
        retry_delay = 1.0
        
        while retry_count <= max_retries:
            try:
                # Start time for performance tracking
                start_time = time.time()
                
                # Map tool names to their handlers
                result = await self._dispatch_tool_call(tool_name, validated_args)
                
                # Log performance metrics
                elapsed_time = time.time() - start_time
                logger.debug(f"Tool call {tool_name} completed in {elapsed_time:.3f}s")
                
                # Add performance metrics to result if successful
                if isinstance(result, dict) and "error" not in result:
                    result["_metadata"] = {
                        "execution_time": elapsed_time,
                        "tool_name": tool_name
                    }
                
                return result
                
            except asyncio.TimeoutError:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Tool call {tool_name} timed out after {max_retries} retries")
                    return {"error": f"Tool execution timed out", "success": False}
                
                # Exponential backoff
                wait_time = retry_delay * (backoff_factor ** retry_count)
                logger.warning(f"Tool call {tool_name} timed out, retrying in {wait_time:.2f}s (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error in tool call {tool_name}: {e}", exc_info=True)
                retry_count += 1
                
                # Determine if error is retryable
                if retry_count > max_retries or not self._is_retryable_error(e):
                    return {"error": f"Tool execution error: {str(e)}", "success": False}
                
                # Exponential backoff for retryable errors
                wait_time = retry_delay * (backoff_factor ** retry_count)
                logger.warning(f"Retrying tool call {tool_name} in {wait_time:.2f}s (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
    
    async def _dispatch_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool call to the appropriate handler method."""
        # Tool dispatcher mapping
        tool_handlers = {
            "search_memory": self.search_memory_tool,
            "store_important_memory": self.store_important_memory,
            "get_important_memories": self.get_important_memories,
            "get_personal_details": self.get_personal_details_tool,
            "get_emotional_context": self.get_emotional_context_tool,
            "track_conversation_topic": self.track_conversation_topic
        }
        
        # Get the appropriate handler
        handler = tool_handlers.get(tool_name)
        
        if not handler:
            logger.warning(f"Unknown tool call: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}", "success": False}
        
        # Handle search_memory tool specially due to its different parameter signature
        if tool_name == "search_memory":
            query = args.get("query", "")
            limit = args.get("limit", 5)
            min_significance = args.get("min_significance", 0.0)
            return await handler(query=query, max_results=limit, min_significance=min_significance)
        
        # Handle store_important_memory tool
        elif tool_name == "store_important_memory":
            content = args.get("content", "")
            significance = args.get("significance", 0.8)
            return await handler(content=content, significance=significance)
        
        # Handle get_important_memories tool
        elif tool_name == "get_important_memories":
            limit = args.get("limit", 5)
            min_significance = args.get("min_significance", 0.7)
            return await handler(limit=limit, min_significance=min_significance)
        
        # For other tools, pass args as a dictionary
        return await handler(args)
    
    def _validate_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize tool arguments."""
        if not isinstance(args, dict):
            return {"error": "Arguments must be a dictionary"}
        
        # Tool-specific validation
        if tool_name == "search_memory":
            if "query" not in args or not args["query"]:
                return {"error": "Query is required for search_memory tool"}
            
            # Sanitize limit
            if "limit" in args:
                try:
                    args["limit"] = max(1, min(int(args["limit"]), 20))  # Limit between 1-20
                except (ValueError, TypeError):
                    args["limit"] = 5  # Default if invalid
            
            # Sanitize min_significance
            if "min_significance" in args:
                try:
                    args["min_significance"] = max(0.0, min(float(args["min_significance"]), 1.0))  # Range 0-1
                except (ValueError, TypeError):
                    args["min_significance"] = 0.0  # Default if invalid
        
        elif tool_name == "store_important_memory":
            if "content" not in args or not args["content"]:
                return {"error": "Content is required for store_important_memory tool"}
            
            # Sanitize significance
            if "significance" in args:
                try:
                    args["significance"] = max(0.0, min(float(args["significance"]), 1.0))  # Range 0-1
                except (ValueError, TypeError):
                    args["significance"] = 0.8  # Default if invalid
        
        elif tool_name == "track_conversation_topic":
            if "topic" not in args or not args["topic"]:
                return {"error": "Topic is required for track_conversation_topic tool"}
            
            # Sanitize importance
            if "importance" in args:
                try:
                    args["importance"] = max(0.0, min(float(args["importance"]), 1.0))  # Range 0-1
                except (ValueError, TypeError):
                    args["importance"] = 0.7  # Default if invalid
            else:
                args["importance"] = 0.7  # Default importance
        
        return args
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        # Network-related errors are retryable
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            json.JSONDecodeError
        )
        
        # Also check error strings for network-related issues
        error_str = str(error).lower()
        network_keywords = ["connection", "timeout", "network", "socket", "unavailable"]
        
        return isinstance(error, retryable_errors) or any(keyword in error_str for keyword in network_keywords)

    async def store_transcript(self, text: str, sender: str = "user", significance: float = None, role: str = None) -> bool:
        """
        Store a transcript entry in memory.
        
        This method stores conversation transcripts with appropriate metadata
        and automatically calculates significance if not provided.
        
        Args:
            text: The transcript text to store
            sender: Who sent the message (user or assistant)
            significance: Optional pre-calculated significance value
            role: Alternative name for sender parameter (for backward compatibility)
            
        Returns:
            bool: Success status
        """
        if not text or not text.strip():
            logger.warning("Empty transcript text provided")
            return False
            
        try:
            # Handle role parameter for backward compatibility
            if role is not None and sender == "user":
                sender = role
                
            # Calculate significance if not provided
            if significance is None:
                # Use a higher base significance for user messages
                base_significance = 0.6 if sender.lower() == "user" else 0.4
                
                # Adjust based on text length (longer messages often have more content)
                length_factor = min(len(text) / 100, 0.3)  # Cap at 0.3
                
                # Check for question marks (questions are often important)
                question_factor = 0.15 if "?" in text else 0.0
                
                # Check for exclamation marks (emotional content)
                emotion_factor = 0.1 if "!" in text else 0.0
                
                # Check for personal information markers
                personal_info_markers = ["my name", "I live", "my address", "my phone", "my email", "my birthday"]
                personal_factor = 0.2 if any(marker in text.lower() for marker in personal_info_markers) else 0.0
                
                # Final significance calculation (capped at 0.95)
                significance = min(base_significance + length_factor + question_factor + emotion_factor + personal_factor, 0.95)
            
            # Create metadata
            metadata = {
                "type": "transcript",
                "sender": sender,
                "timestamp": time.time(),
                "session_id": self.session_id
            }
            
            # Store in memory system
            success = await self.store_memory(
                content=text,
                metadata=metadata,
                significance=significance
            )
            
            if success:
                logger.info(f"Stored transcript from {sender} with significance {significance:.2f}")
            else:
                logger.warning(f"Failed to store transcript from {sender}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error storing transcript: {e}")
            return False

    async def detect_and_store_personal_details(self, text: str, role: str = "user") -> bool:
        """
        Detect and store personal details from text.
        
        This method analyzes text for personal details like name, location, etc.,
        and stores them in the personal details dictionary.
        
        Args:
            text: The text to analyze
            role: The role of the speaker (user or assistant)
            
        Returns:
            bool: True if any details were detected and stored
        """
        # Only process user messages
        if role.lower() != "user":
            return False
            
        try:
            # Initialize result flag
            details_found = False
            
            # Define comprehensive patterns for different personal details using improved regex patterns
            patterns = {
                "name": [
                    r"(?:my name is|i am|i'm|call me|they call me) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})",
                    r"([A-Z][a-z]+(?: [A-Z][a-z]+){0,3}) (?:is my name|here|speaking)",
                    r"(?:name's|names) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})",
                    r"(?:known as|goes by) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})"
                ],
                "location": [
                    r"i live (?:in|at) ([\w\s,]+)",
                    r"i(?:'m| am) from ([\w\s,]+)",
                    r"my address is ([\w\s,]+)",
                    r"my location is ([\w\s,]+)",
                    r"i (?:reside|stay) (?:in|at) ([\w\s,]+)",
                    r"(?:living|residing) in ([\w\s,]+)",
                    r"(?:based in|located in) ([\w\s,]+)"
                ],
                "birthday": [
                    r"my birthday is ([\w\s,]+)",
                    r"i was born on ([\w\s,]+)",
                    r"born in ([\w\s,]+)",
                    r"my birth date is ([\w\s,]+)",
                    r"my date of birth is ([\w\s,]+)",
                    r"i was born in ([\w\s,]+)"
                ],
                "job": [
                    r"i work as (?:an?|the) ([\w\s]+)",
                    r"i am (?:an?|the) ([\w\s]+)(?: by profession| by trade)?",
                    r"my job is ([\w\s]+)",
                    r"i'm (?:an?|the) ([\w\s]+)(?: by profession| by trade)?",
                    r"my profession is ([\w\s]+)",
                    r"i (?:do|practice) ([\w\s]+)(?: for (?:a|my) living)?",
                    r"i'm (?:employed as|working as) (?:an?|the) ([\w\s]+)",
                    r"my (?:career|occupation) is (?:in|as) ([\w\s]+)"
                ],
                "email": [
                    r"my email (?:is|address is) ([\w.+-]+@[\w-]+\.[\w.-]+)",
                    r"(?:reach|contact) me at ([\w.+-]+@[\w-]+\.[\w.-]+)",
                    r"([\w.+-]+@[\w-]+\.[\w.-]+) is my email"
                ],
                "phone": [
                    r"my (?:phone|number|phone number|cell|mobile) is ((?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4})",
                    r"(?:reach|contact|call) me at ((?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4})",
                    r"((?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}) is my (?:phone|number|phone number|cell|mobile)"
                ],
                "age": [
                    r"i(?:'m| am) (\d+)(?: years old| years of age)?",
                    r"my age is (\d+)",
                    r"i turned (\d+) (?:recently|last year|this year)",
                    r"i'll be (\d+) (?:soon|next year|this year)"
                ]
            }
            
            # Family patterns need special handling
            family_patterns = {
                "spouse": [
                    r"my (wife|husband|spouse|partner) (?:is|'s) ([\w\s]+)",
                    r"i(?:'m| am) married to ([\w\s]+)",
                    r"([\w\s]+) is my (wife|husband|spouse|partner)"
                ],
                "child": [
                    r"my (son|daughter|child) (?:is|'s) ([\w\s]+)",
                    r"i have a (son|daughter|child) (?:named|called) ([\w\s]+)",
                    r"([\w\s]+) is my (son|daughter|child)"
                ],
                "parent": [
                    r"my (mother|father|mom|dad|parent) (?:is|'s) ([\w\s]+)",
                    r"([\w\s]+) is my (mother|father|mom|dad|parent)"
                ],
                "sibling": [
                    r"my (brother|sister|sibling) (?:is|'s) ([\w\s]+)",
                    r"i have a (brother|sister|sibling) (?:named|called) ([\w\s]+)",
                    r"([\w\s]+) is my (brother|sister|sibling)"
                ]
            }
            
            # Initialize family data if not already present
            if hasattr(self, "personal_details") and "family" not in self.personal_details:
                self.personal_details["family"] = {}
            
            # Process standard patterns
            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        # Take the first match and clean it
                        value = matches[0].strip().rstrip('.,:;!?')
                        
                        # Skip very short or likely invalid values
                        if len(value) < 2 or value.lower() in ["a", "an", "the", "me", "i", "my", "he", "she", "they"]:
                            continue
                            
                        # Validate based on category
                        if category == "email" and not re.match(r"[\w.+-]+@[\w-]+\.[\w.-]+", value):
                            continue
                        
                        if category == "age" and (not value.isdigit() or int(value) > 120 or int(value) < 1):
                            continue
                            
                        # Store the detail with confidence score
                        if hasattr(self, "personal_details"):
                            confidence = 0.9  # High confidence for clear pattern matches
                            
                            # Store with appropriate metadata
                            self.personal_details[category] = {
                                "value": value,
                                "confidence": confidence,
                                "timestamp": time.time(),
                                "source": "explicit_mention"
                            }
                            
                            logger.info(f"Stored personal detail: {category}={value} (confidence: {confidence:.2f})")
                            details_found = True
                        
                        # Also store as a high-significance memory
                        await self.store_memory(
                            content=f"User {category}: {value}",
                            significance=0.9,
                            metadata={
                                "type": "personal_detail",
                                "category": category,
                                "value": value,
                                "confidence": 0.9,
                                "timestamp": time.time()
                            }
                        )
                        
                        # Only process one match per category
                        break
            
            # Process family patterns (they have a nested structure)
            for relation, pattern_list in family_patterns.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            # Family patterns can return different structures based on the specific regex
                            if len(match) >= 2:
                                # Handle reversed patterns like "(name) is my (relation)"
                                if match[1] in ["wife", "husband", "spouse", "partner", "son", "daughter", 
                                               "child", "mother", "father", "mom", "dad", "parent",
                                               "brother", "sister", "sibling"]:
                                    name = match[0].strip().rstrip('.,:;!?')
                                    relation_type = match[1].lower()
                                else:
                                    relation_type = match[0].lower()
                                    name = match[1].strip().rstrip('.,:;!?')
                                
                                # Skip very short or likely invalid values
                                if len(name) < 2 or name.lower() in ["a", "an", "the", "me", "i", "my"]:
                                    continue
                                    
                                # Store in family dictionary
                                if hasattr(self, "personal_details") and "family" in self.personal_details:
                                    confidence = 0.9  # High confidence for clear pattern matches
                                    
                                    # Use a consistent structure for family entries
                                    self.personal_details["family"][relation_type] = {
                                        "name": name,
                                        "confidence": confidence,
                                        "timestamp": time.time(),
                                        "source": "explicit_mention"
                                    }
                                    
                                    logger.info(f"Stored family detail: {relation_type}={name} (confidence: {confidence:.2f})")
                                    details_found = True
                                
                                # Also store as a memory
                                await self.store_memory(
                                    content=f"User's {relation_type}: {name}",
                                    significance=0.85,
                                    metadata={
                                        "type": "personal_detail",
                                        "category": "family",
                                        "relation_type": relation_type,
                                        "value": name,
                                        "confidence": 0.9,
                                        "timestamp": time.time()
                                    }
                                )
            
            return details_found
            
        except Exception as e:
            logger.error(f"Error detecting personal details: {e}")
            return False

    async def get_rag_context(self, query: str = None, limit: int = 5, min_significance: float = 0.0, max_tokens: int = None) -> str:
        """
        Get memory context for LLM RAG (Retrieval-Augmented Generation).
        
        Enhanced version with better categorization, formatting, and relevance.
        
        Args:
            query: Optional query to filter memories
            limit: Maximum number of memories to include
            min_significance: Minimum significance threshold for memories
            max_tokens: Maximum number of tokens to include (approximate)
            
        Returns:
            str: Formatted memory context for RAG
        """
        try:
            max_tokens = max_tokens or limit * 100  # Default token limit based on memory count
            
            # Determine if this is a personal query
            is_personal_query = await self._is_personal_query(query) if query else False
            
            # Determine if this is a memory recall or history query
            is_memory_query = await self._is_memory_query(query) if query else False
            
            # Initialize context parts
            context_parts = []
            context_sections = {}
            
            # Generate different types of context based on query type
            if is_personal_query:
                # For personal queries, prioritize user information
                personal_context = await self._generate_personal_context(query)
                if personal_context:
                    context_sections["personal"] = personal_context
                    context_parts.append("### User Personal Information")
                    context_parts.append(personal_context)
            
            if is_memory_query:
                # For memory queries, provide more comprehensive memory context
                memory_limit = limit * 2  # Double the limit for memory-specific queries
                memory_context = await self._generate_memory_context(query, memory_limit, min_significance)
                if memory_context:
                    context_sections["memory"] = memory_context
                    context_parts.append("### Memory Recall")
                    context_parts.append(memory_context)
            
            # Add emotional context if appropriate (for emotional or personal queries)
            if is_personal_query or (query and any(keyword in query.lower() for keyword in ["feel", "emotion", "mood", "happy", "sad", "angry"])):
                emotional_context = await self._generate_emotional_context()
                if emotional_context:
                    context_sections["emotional"] = emotional_context
                    context_parts.append("### Recent Emotional States")
                    context_parts.append(emotional_context)
            
            # Add standard memory context if no specialized context was added or as supplement
            if not context_parts or (not is_memory_query and not is_personal_query):
                standard_context = await self._generate_standard_context(query, limit, min_significance)
                if standard_context:
                    context_sections["standard"] = standard_context
                    if not context_parts:  # If no sections yet, add a general header
                        context_parts.append("### Relevant Memory Context")
                    context_parts.append(standard_context)
            
            # Combine context parts
            context = "\n\n".join(context_parts)
            
            # Truncate if too long (approximate token count based on characters)
            char_limit = max_tokens * 4  # Rough estimate: 1 token ≈ 4 characters
            if len(context) > char_limit:
                # Try to preserve complete sections
                truncated_context = []
                current_length = 0
                
                for part in context_parts:
                    if current_length + len(part) <= char_limit:
                        truncated_context.append(part)
                        current_length += len(part) + 2  # +2 for newlines
                    else:
                        # For the last section, include as much as possible
                        if not truncated_context:  # Ensure at least one section
                            truncation_point = char_limit - current_length
                            truncated_part = part[:truncation_point] + "...\n[Context truncated due to length]"
                            truncated_context.append(truncated_part)
                        else:
                            truncated_context.append("[Additional context truncated due to length]")
                        break
                
                context = "\n\n".join(truncated_context)
            
            logger.info(f"Generated RAG context with {sum(1 for v in context_sections.values() for _ in v.split('•') if '•' in v)} memories")
            return context
            
        except Exception as e:
            logger.error(f"Error generating RAG context: {e}")
            # Return empty string on error rather than propagating the exception
            return ""
    
    async def _is_personal_query(self, query: str) -> bool:
        """Determine if a query is asking for personal information."""
        personal_keywords = [
            "my name", "who am i", "where do i live", "where am i from",
            "how old am i", "what's my age", "what's my birthday", "when was i born",
            "what do i do", "what's my job", "what's my profession", "who is my",
            "my family", "my spouse", "my partner", "my children", "my parents",
            "my email", "my phone", "my number", "my address"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in personal_keywords)
    
    async def _is_memory_query(self, query: str) -> bool:
        """Determine if a query is asking about past memories or history."""
        memory_keywords = [
            "remember", "recall", "forget", "memory", "memories", "mentioned",
            "said earlier", "talked about", "discussed", "told me about",
            "what did i say", "what did you say", "what did we discuss",
            "earlier", "before", "previously", "last time", "yesterday",
            "last week", "last month", "last year", "in the past"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in memory_keywords)
    
    async def _generate_personal_context(self, query: str) -> str:
        """Generate context about personal details."""
        if not hasattr(self, "personal_details") or not self.personal_details:
            return ""
        
        parts = []
        
        # Add clear header to indicate whose information this is
        parts.append("# USER PERSONAL INFORMATION")
        parts.append("# The following information is about the USER, not about the assistant")
        parts.append("# The assistant's name is Lucidia")
        
        # Determine specific personal detail being asked about
        personal_categories = {
            "name": ["name", "call me", "who am i"],
            "location": ["live", "from", "address", "location", "home"],
            "birthday": ["birthday", "born", "birth date", "age"],
            "job": ["job", "work", "profession", "career", "occupation"],
            "family": ["family", "spouse", "partner", "husband", "wife", "child", "children", "kid", "kids", "parent", "mother", "father"],
            "email": ["email", "e-mail", "mail"],
            "phone": ["phone", "number", "mobile", "cell"]
        }
        
        # Check if query targets a specific category
        target_category = None
        query_lower = query.lower()
        for category, keywords in personal_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                target_category = category
                break
        
        # If we have a specific target, focus on that
        if target_category:
            if target_category == "family":
                # Handle family data specially
                if "family" in self.personal_details and self.personal_details["family"]:
                    parts.append("USER's family information:")
                    for relation, data in self.personal_details["family"].items():
                        if isinstance(data, dict) and "name" in data:
                            parts.append(f"• USER's {relation}: {data['name']}")
                        else:
                            parts.append(f"• USER's {relation}: {data}")
            else:
                # Handle standard categories
                if target_category in self.personal_details:
                    detail = self.personal_details[target_category]
                    if isinstance(detail, dict) and "value" in detail:
                        parts.append(f"USER's {target_category}: {detail['value']}")
                    else:
                        parts.append(f"USER's {target_category}: {detail}")
        else:
            # No specific target, include all personal details
            for category, detail in self.personal_details.items():
                if category == "family":
                    # Handle family data specially
                    if detail:
                        parts.append("USER's family information:")
                        for relation, data in detail.items():
                            if isinstance(data, dict) and "name" in data:
                                parts.append(f"• USER's {relation}: {data['name']}")
                            else:
                                parts.append(f"• USER's {relation}: {data}")
                else:
                    # Handle standard categories
                    if isinstance(detail, dict) and "value" in detail:
                        parts.append(f"USER's {category}: {detail['value']}")
                    else:
                        parts.append(f"USER's {category}: {detail}")
        
        # Add clarity reminder at the end
        parts.append("\n# IMPORTANT: The above information is about the USER, not the assistant")
        parts.append("# The assistant's name is Lucidia\n")
        
        return "\n".join(parts)
    
    async def _generate_memory_context(self, query: str, limit: int, min_significance: float) -> str:
        """Generate context about past memories."""
        try:
            # Use higher significance threshold for memory queries
            min_significance = max(min_significance, 0.3)
            
            # Search for memories related to the query
            memories = await self.search_memory_tool(
                query=query,
                max_results=limit,
                min_significance=min_significance
            )
            
            if not memories or not memories.get("memories"):
                return ""
            
            parts = []
            results = memories["memories"]
            
            # Format each memory with timestamp and content
            for i, memory in enumerate(results):
                content = memory.get("content", "").strip()
                timestamp = memory.get("timestamp", 0)
                significance = memory.get("significance", 0.0)
                
                # Format timestamp as readable date
                date_str = ""
                if timestamp:
                    try:
                        import datetime
                        date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = f"timestamp: {timestamp}"
                
                # Add significance indicators for highly significant memories
                sig_indicator = ""
                if significance > 0.8:
                    sig_indicator = " [IMPORTANT]"
                elif significance > 0.6:
                    sig_indicator = " [Significant]"
                
                parts.append(f"• Memory from {date_str}{sig_indicator}: {content}")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"Error generating memory context: {e}")
            return ""
    
    async def _generate_emotional_context(self) -> str:
        """Generate context about emotional states."""
        try:
            # Get emotional context if available
            if hasattr(self, "get_emotional_history"):
                emotional_history = await self.get_emotional_history(limit=3)
                if emotional_history:
                    return emotional_history
            
            # Fallback to checking if we have emotions data
            if hasattr(self, "emotions") and self.emotions:
                parts = []
                
                # Sort by timestamp (newest first)
                sorted_emotions = sorted(
                    self.emotions.items(),
                    key=lambda x: float(x[0]),
                    reverse=True
                )[:3]  # Limit to 3 most recent
                
                for timestamp, data in sorted_emotions:
                    sentiment = data.get("sentiment", 0)
                    emotions = data.get("emotions", {})
                    
                    # Format timestamp
                    try:
                        import datetime
                        date_str = datetime.datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = f"timestamp: {timestamp}"
                    
                    # Determine sentiment description
                    if sentiment > 0.5:
                        sentiment_desc = "very positive"
                    elif sentiment > 0.1:
                        sentiment_desc = "positive"
                    elif sentiment > -0.1:
                        sentiment_desc = "neutral"
                    elif sentiment > -0.5:
                        sentiment_desc = "negative"
                    else:
                        sentiment_desc = "very negative"
                    
                    # Format emotions if available
                    emotion_str = ""
                    if emotions:
                        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                        emotion_str = ", ".join(f"{emotion}" for emotion, _ in top_emotions)
                        emotion_str = f" with {emotion_str}"
                    
                    parts.append(f"• {date_str}: User exhibited {sentiment_desc} sentiment{emotion_str}")
                
                return "\n".join(parts)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error generating emotional context: {e}")
            return ""
    
    async def _generate_standard_context(self, query: str, limit: int, min_significance: float) -> str:
        """Generate standard context from memories."""
        try:
            if query:
                # Search for memories related to the query
                memories = await self.search_memory_tool(
                    query=query,
                    max_results=limit,
                    min_significance=min_significance
                )
                
                if not memories or not memories.get("memories"):
                    return ""
                
                parts = []
                results = memories["memories"]
            else:
                # Get recent important memories
                memories = await self.get_important_memories(
                    limit=limit,
                    min_significance=min_significance or 0.5
                )
                
                if not memories or not memories.get("memories"):
                    return ""
                
                parts = []
                results = memories["memories"]
            
            # Format each memory with timestamp and content
            for memory in results:
                content = memory.get("content", "").strip()
                timestamp = memory.get("timestamp", 0)
                
                # Format timestamp as readable date
                date_str = ""
                if timestamp:
                    try:
                        import datetime
                        date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                        date_str = f"[{date_str}]"
                    except:
                        pass
                
                parts.append(f"• {date_str} {content}")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"Error generating standard context: {e}")
            return ""
            
    async def store_conversation(self, text: str, role: str = "assistant") -> bool:
        """
        Store a conversation message in memory.
        
        Args:
            text: The message text
            role: The role of the sender (user or assistant)
            
        Returns:
            bool: Success status
        """
        try:
            # Calculate appropriate significance based on content
            base_significance = 0.5 if role == "assistant" else 0.6
            
            # Adjust for content length
            length_factor = min(len(text) / 200, 0.2)  # Cap at 0.2
            
            # Check for question marks (questions are often important)
            question_factor = 0.1 if "?" in text else 0.0
            
            # Check for personal information indicators
            personal_indicators = ["name", "email", "phone", "address", "live", "age", "birthday", "family"]
            personal_factor = 0.2 if any(indicator in text.lower() for indicator in personal_indicators) else 0.0
            
            # Calculate final significance
            significance = min(base_significance + length_factor + question_factor + personal_factor, 0.95)
            
            # Store as a memory with conversation metadata
            return await self.store_memory(
                content=text,
                significance=significance,
                metadata={
                    "type": "conversation",
                    "role": role,
                    "session_id": self.session_id,
                    "timestamp": time.time()
                }
            )
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            return False
            
    async def mark_topic_discussed(self, topic: Union[str, List[str]], importance: float = 0.7) -> bool:
        """
        Mark a topic or list of topics as discussed in the current session.
        
        Args:
            topic: Topic or list of topics to mark as discussed
            importance: Importance of the topic (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        try:
            if isinstance(topic, str):
                topics = [topic]
            else:
                topics = topic
                
            success = True
            for t in topics:
                # Clean up topic string
                t = t.strip().lower()
                if not t:
                    continue
                
                # Store a memory indicating the topic was discussed
                result = await self.store_memory(
                    content=f"Topic '{t}' was discussed",
                    significance=importance,
                    metadata={
                        "type": "topic_discussed",
                        "topic": t,
                        "session_id": self.session_id,
                        "timestamp": time.time(),
                        "importance": importance
                    }
                )
                
                # Add to suppressed topics if enabled
                if self._topic_suppression["enabled"]:
                    expiration = time.time() + self._topic_suppression["suppression_time"]
                    self._topic_suppression["suppressed_topics"][t] = {
                        "expiration": expiration,
                        "importance": importance
                    }
                
                success = success and result
                
            return success
        except Exception as e:
            logger.error(f"Error marking topic as discussed: {e}")
            return False
    
    async def track_conversation_topic(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool implementation to track conversation topics for suppression.
        
        Args:
            args: Dictionary containing:
                - topic: The topic being discussed
                - importance: Importance of this topic (0-1)
            
        Returns:
            Dict with status information
        """
        try:
            # Extract parameters
            topic = args.get("topic", "").strip()
            importance = args.get("importance", 0.7)
            
            if not topic:
                return {"success": False, "error": "No topic provided"}
            
            # Validate importance
            try:
                importance = float(importance)
                importance = max(0.0, min(1.0, importance))
            except (ValueError, TypeError):
                importance = 0.7  # Default if invalid
            
            # Mark topic as discussed
            success = await self.mark_topic_discussed(topic, importance)
            
            if success:
                return {
                    "success": True,
                    "topic": topic,
                    "importance": importance,
                    "suppression_time": self._topic_suppression["suppression_time"] if self._topic_suppression["enabled"] else 0
                }
            else:
                return {"success": False, "error": "Failed to store topic"}
            
        except Exception as e:
            logger.error(f"Error tracking conversation topic: {e}")
            return {"success": False, "error": str(e)}
    
    async def is_topic_suppressed(self, topic: str) -> Tuple[bool, float]:
        """
        Check if a topic is currently suppressed.
        
        Args:
            topic: The topic to check
            
        Returns:
            Tuple of (is_suppressed, importance)
        """
        if not self._topic_suppression["enabled"]:
            return False, 0.0
            
        # Clean topic
        topic = topic.strip().lower()
        
        # Check if topic is suppressed
        if topic in self._topic_suppression["suppressed_topics"]:
            data = self._topic_suppression["suppressed_topics"][topic]
            expiration = data.get("expiration", 0)
            importance = data.get("importance", 0.5)
            
            # Check if suppression has expired
            if expiration > time.time():
                return True, importance
            else:
                # Remove expired suppression
                del self._topic_suppression["suppressed_topics"][topic]
        
        return False, 0.0
    
    async def configure_topic_suppression(self, enable: bool = True, suppression_time: int = None) -> None:
        """
        Configure topic suppression settings.
        
        Args:
            enable: Whether to enable topic suppression
            suppression_time: Time in seconds to suppress repetitive topics
        """
        self._topic_suppression["enabled"] = enable
        
        if suppression_time is not None:
            try:
                suppression_time = int(suppression_time)
                if suppression_time > 0:
                    self._topic_suppression["suppression_time"] = suppression_time
            except (ValueError, TypeError):
                pass
        
        # Clean up expired suppressions
        current_time = time.time()
        expired_topics = [
            topic for topic, data in self._topic_suppression["suppressed_topics"].items()
            if data.get("expiration", 0) <= current_time
        ]
        
        for topic in expired_topics:
            del self._topic_suppression["suppressed_topics"][topic]
    
    async def reset_topic_suppression(self, topic: str = None) -> None:
        """
        Reset topic suppression for a specific topic or all topics.
        
        Args:
            topic: Specific topic to reset, or None to reset all topics
        """
        if topic:
            # Reset specific topic
            topic = topic.strip().lower()
            if topic in self._topic_suppression["suppressed_topics"]:
                del self._topic_suppression["suppressed_topics"][topic]
        else:
            # Reset all topics
            self._topic_suppression["suppressed_topics"] = {}
    
    async def get_topic_suppression_status(self) -> Dict[str, Any]:
        """
        Get the current status of topic suppression.
        
        Returns:
            Dictionary containing topic suppression status information
        """
        # Clean up expired suppressions
        current_time = time.time()
        expired_topics = [
            topic for topic, data in self._topic_suppression["suppressed_topics"].items()
            if data.get("expiration", 0) <= current_time
        ]
        
        for topic in expired_topics:
            del self._topic_suppression["suppressed_topics"][topic]
        
        # Build status dictionary
        active_topics = {}
        for topic, data in self._topic_suppression["suppressed_topics"].items():
            expiration = data.get("expiration", 0)
            if expiration > current_time:
                time_left = int(expiration - current_time)
                active_topics[topic] = {
                    "time_left": time_left,
                    "importance": data.get("importance", 0.5)
                }
        
        return {
            "enabled": self._topic_suppression["enabled"],
            "suppression_time": self._topic_suppression["suppression_time"],
            "active_count": len(active_topics),
            "active_topics": active_topics
        }

    async def get_emotional_context_tool(self, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Tool implementation to get emotional context.
        
        Args:
            args: Optional arguments (unused)
            
        Returns:
            Dict with emotional context information
        """
        try:
            # Call the emotion mixin's get_emotional_context method
            context = await self.get_emotional_context()
            
            # Enhanced the response with more helpful summary
            if context:
                # Add sentiment trend analysis if we have enough history
                if len(context.get("recent_emotions", [])) >= 2:
                    emotions = context.get("recent_emotions", [])
                    if emotions:
                        # Calculate sentiment trend
                        trend = "steady"
                        sentiment_values = [e.get("sentiment", 0) for e in emotions]
                        
                        if len(sentiment_values) >= 2:
                            if sentiment_values[0] > sentiment_values[-1] + 0.2:
                                trend = "improving"
                            elif sentiment_values[0] < sentiment_values[-1] - 0.2:
                                trend = "declining"
                        
                        context["sentiment_trend"] = trend
                
                # Add dominant emotion for the entire conversation
                all_emotions = {}
                for emotion_data in context.get("recent_emotions", []):
                    for emotion, score in emotion_data.get("emotions", {}).items():
                        all_emotions[emotion] = all_emotions.get(emotion, 0) + score
                
                if all_emotions:
                    dominant_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
                    context["dominant_emotion"] = dominant_emotion
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting emotional context: {e}")
            return {
                "error": str(e),
                "current_emotion": None,
                "recent_emotions": [],
                "emotional_triggers": {}
            }

    async def get_personal_details_tool(self, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Tool implementation to get personal details.
        
        Enhanced version with better category handling and confidence scores.
        
        Args:
            args: Arguments including:
                - category: Optional category to retrieve
            
        Returns:
            Dict with personal details
        """
        try:
            # Extract category if provided
            category = None
            if args and isinstance(args, dict):
                category = args.get("category")
            
            # Initialize response
            response = {
                "found": False,
                "details": {}
            }
            
            # If we don't have personal details, return empty
            if not hasattr(self, "personal_details") or not self.personal_details:
                return response
            
            # Handle special case for name queries
            if category and category.lower() == "name":
                value = None
                confidence = 0
                
                # First check in personal details
                if "name" in self.personal_details:
                    detail = self.personal_details["name"]
                    if isinstance(detail, dict) and "value" in detail:
                        value = detail["value"]
                        confidence = detail.get("confidence", 0.9)
                    else:
                        value = detail
                        confidence = 0.9
                
                # If not found, search memories
                if not value:
                    try:
                        name_memories = await self.search_memory(
                            "user name", 
                            limit=3, 
                            min_significance=0.8
                        )
                        
                        # Extract name from memory content
                        for memory in name_memories:
                            content = memory.get("content", "")
                            
                            # Look for explicit name mentions
                            patterns = [
                                r"User name: ([A-Za-z]+(?: [A-Za-z]+){0,3})",
                                r"User's name is ([A-Za-z]+(?: [A-Za-z]+){0,3})",
                                r"([A-Za-z]+(?: [A-Za-z]+){0,3}) is my name"
                            ]
                            
                            for pattern in patterns:
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                if matches:
                                    value = matches[0].strip()
                                    confidence = 0.85
                                    
                                    # Store for future use
                                    self.personal_details["name"] = {
                                        "value": value,
                                        "confidence": confidence,
                                        "timestamp": time.time(),
                                        "source": "memory_retrieval"
                                    }
                                    break
                            
                            if value:
                                break
                    except Exception as e:
                        logger.error(f"Error searching memory for name: {e}")
                
                if value:
                    return {
                        "found": True,
                        "category": "name",
                        "value": value,
                        "confidence": confidence
                    }
            
            # Handle specific category request
            if category:
                category = category.lower()
                
                # Family category needs special handling
                if category == "family":
                    if "family" in self.personal_details and self.personal_details["family"]:
                        family_data = {}
                        for relation, data in self.personal_details["family"].items():
                            if isinstance(data, dict) and "name" in data:
                                family_data[relation] = {
                                    "name": data["name"],
                                    "confidence": data.get("confidence", 0.85)
                                }
                            else:
                                family_data[relation] = {
                                    "name": data,
                                    "confidence": 0.85
                                }
                        
                        return {
                            "found": True,
                            "category": "family",
                            "value": family_data,
                            "confidence": 0.9
                        }
                else:
                    # Standard category
                    if category in self.personal_details:
                        detail = self.personal_details[category]
                        if isinstance(detail, dict) and "value" in detail:
                            return {
                                "found": True,
                                "category": category,
                                "value": detail["value"],
                                "confidence": detail.get("confidence", 0.85)
                            }
                        else:
                            return {
                                "found": True,
                                "category": category,
                                "value": detail,
                                "confidence": 0.85
                            }
            
            # No specific category requested or not found, return all details
            formatted_details = {}
            for cat, detail in self.personal_details.items():
                if cat == "family":
                    # Format family data
                    family_data = {}
                    if detail:
                        for relation, data in detail.items():
                            if isinstance(data, dict) and "name" in data:
                                family_data[relation] = data["name"]
                            else:
                                family_data[relation] = data
                        
                        formatted_details[cat] = family_data
                else:
                    # Format standard data
                    if isinstance(detail, dict) and "value" in detail:
                        formatted_details[cat] = detail["value"]
                    else:
                        formatted_details[cat] = detail
            
            response["found"] = len(formatted_details) > 0
            response["details"] = formatted_details
            response["count"] = len(formatted_details)
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting personal details: {e}")
            return {
                "found": False,
                "error": str(e),
                "details": {}
            }

    def _get_timestamp(self) -> float:
        """
        Get current timestamp in seconds since epoch.
        
        Returns:
            float: Current timestamp
        """
        return time.time()

    async def analyze_emotions(self, text: str) -> Dict[str, Any]:
        """Analyze and record emotions from text content"""
async def detect_emotional_context(self, text: str) -> Dict[str, Any]:
    """
    Detect and analyze emotional context from text.
    This is a wrapper around the EmotionMixin functionality for the voice agent.
    
    Args:
        text: The text to analyze for emotional context
        
    Returns:
        Dict with emotional context information
    """
    # Use the EmotionMixin's method if available
    if hasattr(self, "detect_emotion") and callable(getattr(self, "detect_emotion")):
        try:
            # First, create a simple emotional context structure
            timestamp = time.time()
            emotional_context = {
                "timestamp": timestamp,
                "text": text,
                "emotions": {},
                "sentiment": 0.0,
                "emotional_state": "neutral"
            }
            
            # Try to detect emotion using the mixin method
            emotion = await self.detect_emotion(text)
            emotional_context["emotional_state"] = emotion
            emotional_context["emotions"][emotion] = 1.0
            
            # Store the emotional context for future reference
            if hasattr(self, "_emotional_history"):
                self._emotional_history.append(emotional_context)
                
                # Trim emotional history if needed
                if hasattr(self, "_max_emotional_history") and len(self._emotional_history) > self._max_emotional_history:
                    self._emotional_history = self._emotional_history[-self._max_emotional_history:]
                    
                logger.debug(f"Added to emotional history (total: {len(self._emotional_history)})")
            
            logger.info(f"Detected emotion: {emotion} for text: {text[:30]}...")
            return emotional_context
        except Exception as e:
            logger.error(f"Error in detect_emotion: {e}")
            # Fallback to empty emotional context
            return {
                "timestamp": time.time(),
                "text": text,
                "emotions": {"neutral": 1.0},
                "sentiment": 0.0,
                "emotional_state": "neutral",
                "error": str(e)
            }
    
    # Fallback if emotion detection isn't available
    return {
        "timestamp": time.time(),
        "text": text,
        "emotions": {"neutral": 1.0},
        "sentiment": 0.0,
        "emotional_state": "neutral"
    }

    async def store_memory(self, content: str, metadata: Dict[str, Any] = None, significance: float = None, importance: float = None) -> bool:
        """
        Store a new memory with semantic embedding and ensure proper persistence.
        
        Args:
            content: The memory content to store
            metadata: Additional metadata for the memory
            significance: Optional override for significance
            importance: Alternate name for significance (for backward compatibility)
            
        Returns:
            bool: Success status
        """
        if not content or not content.strip():
            logger.warning("Attempted to store empty memory content")
            return False
            
        try:
            # Handle both significance and importance parameters for backward compatibility
            if significance is None and importance is not None:
                significance = importance
            
            # Generate embedding and significance
            try:
                embedding, memory_significance = await self.process_embedding(content)
                
                # Use provided significance if available
                if significance is not None:
                    try:
                        memory_significance = float(significance)
                        # Clamp to valid range
                        memory_significance = max(0.0, min(1.0, memory_significance))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid significance value: {significance}, using calculated value: {memory_significance}")
                
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Fallback: Use zero embedding and default significance
                if hasattr(self, 'embedding_dim'):
                    embedding = torch.zeros(self.embedding_dim)
                else:
                    embedding = torch.zeros(384)  # Default embedding dimension
                memory_significance = 0.5 if significance is None else significance
            
            # Create memory object
            memory_id = str(uuid.uuid4())
            memory = {
                "id": memory_id,
                "content": content,
                "embedding": embedding,
                "timestamp": time.time(),
                "significance": memory_significance,
                "metadata": metadata or {}
            }
            
            # Add to memory list
            async with self._memory_lock:
                self.memories.append(memory)
                
            logger.info(f"Stored new memory with ID {memory_id} and significance {memory_significance:.2f}")
            
            # For high-significance memories, force immediate persistence
            if memory_significance >= 0.7 and hasattr(self, 'persistence_enabled') and self.persistence_enabled:
                logger.info(f"Forcing immediate persistence for high-significance memory: {memory_id}")
                await self._persist_single_memory(memory)
                    
            return True
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False

    async def _persist_single_memory(self, memory: Dict[str, Any]) -> bool:
        """
        Persist a single memory to disk with robust error handling.
        
        Args:
            memory: The memory object to persist
            
        Returns:
            bool: Success status
        """
        if not hasattr(self, 'persistence_enabled') or not self.persistence_enabled:
            return False
            
        if not hasattr(self, 'storage_path'):
            logger.error("No storage path configured")
            return False
            
        try:
            memory_id = memory.get('id')
            if not memory_id:
                logger.warning("Cannot persist memory without ID")
                return False
                
            # Ensure storage directory exists
            if not os.path.exists(self.storage_path):
                os.makedirs(self.storage_path, exist_ok=True)
                logger.info(f"Created storage directory: {self.storage_path}")
                
            file_path = self.storage_path / f"{memory_id}.json"
            temp_file_path = self.storage_path / f"{memory_id}.json.tmp"
            backup_file_path = self.storage_path / f"{memory_id}.json.bak"
            
            # Create a deep copy to avoid modifying the original
            memory_copy = copy.deepcopy(memory)
            
            # Convert embedding to list if it's a tensor or numpy array
            if 'embedding' in memory_copy:
                embedding_data = memory_copy['embedding']
                
                if hasattr(embedding_data, 'tolist'):
                    memory_copy['embedding'] = embedding_data.tolist()
                elif hasattr(embedding_data, 'detach'):
                    # PyTorch tensor with grad
                    memory_copy['embedding'] = embedding_data.detach().cpu().numpy().tolist()
                elif isinstance(embedding_data, np.ndarray):
                    memory_copy['embedding'] = embedding_data.tolist()
                    
            # Convert any other complex types
            memory_copy = self._convert_numpy_to_python(memory_copy)
            
            # Write to temporary file first
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(memory_copy, f, ensure_ascii=False, indent=2)
                
            # If the file exists, create a backup before overwriting
            if file_path.exists():
                try:
                    shutil.copy2(file_path, backup_file_path)
                except Exception as e:
                    logger.warning(f"Failed to create backup for memory {memory_id}: {e}")
            
            # Atomic rename
            os.replace(temp_file_path, file_path)
            
            # Verify file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    _ = json.load(f)  # Just load to verify it's valid JSON
                logger.info(f"Successfully persisted memory {memory_id}")
                
                # Remove backup if everything succeeded
                if backup_file_path.exists():
                    os.remove(backup_file_path)
                    
                return True
            except json.JSONDecodeError:
                logger.error(f"Memory file {file_path} contains invalid JSON after writing")
                # Restore from backup if verification failed
                if backup_file_path.exists():
                    try:
                        os.replace(backup_file_path, file_path)
                        logger.info(f"Restored memory {memory_id} from backup after verification failure")
                    except Exception as e:
                        logger.error(f"Failed to restore backup for memory {memory_id}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error persisting single memory: {e}")
            return False

    async def store_significant_memory(self, 
                                     text: str, 
                                     memory_type: str = "important", 
                                     metadata: Optional[Dict[str, Any]] = None,
                                     min_significance: float = 0.7) -> Dict[str, Any]:
        """
        Store a significant memory with guaranteed persistence.
        
        Args:
            text: The memory content
            memory_type: Type of memory to store
            metadata: Additional metadata
            min_significance: Minimum significance value
            
        Returns:
            Dict with memory information
        """
        if not text or not text.strip():
            logger.warning("Cannot store empty significant memory")
            return {"success": False, "error": "Empty memory content"}
            
        try:
            # Generate embedding and calculate significance
            embedding, auto_significance = await self.process_embedding(text)
            
            # Use the higher of calculated significance or minimum
            significance = max(auto_significance, min_significance)
            
            # Prepare metadata
            full_metadata = {
                "type": memory_type,
                "timestamp": time.time(),
                **(metadata or {})
            }
            
            # Store the memory
            memory_id = str(uuid.uuid4())
            memory = {
                "id": memory_id,
                "content": text,
                "embedding": embedding,
                "timestamp": time.time(),
                "significance": significance,
                "metadata": full_metadata
            }
            
            # Add to memory collection
            async with self._memory_lock:
                self.memories.append(memory)
                
            # Force immediate persistence
            success = await self._persist_single_memory(memory)
            
            if success:
                logger.info(f"Stored significant memory {memory_id} with significance {significance:.2f}")
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "significance": significance,
                    "type": memory_type
                }
            else:
                logger.error("Failed to persist significant memory")
                return {"success": False, "error": "Persistence failed"}
                
        except Exception as e:
            logger.error(f"Error storing significant memory: {e}")
            return {"success": False, "error": str(e)}

    async def store_important_memory(self, content: str = "", significance: float = 0.8) -> Dict[str, Any]:
        """
        Tool implementation to store an important memory with guaranteed persistence.
        
        Args:
            content: The memory content to store
            significance: Memory significance score (0.0-1.0)
            
        Returns:
            Dict with status
        """
        if not content:
            return {"success": False, "error": "No content provided"}
        
        try:
            # Use the store_significant_memory method
            result = await self.store_significant_memory(
                text=content,
                memory_type="important",
                min_significance=significance
            )
            
            return {
                "success": result.get("success", False),
                "memory_id": result.get("memory_id"),
                "error": result.get("error")
            }
                
        except Exception as e:
            logger.error(f"Error storing important memory: {e}")
            return {"success": False, "error": str(e)}

    async def classify_query(self, query: str) -> str:
        """Classify a user query into a specific query type to determine memory retrieval strategy.
        
        Types include:
        - recall: User is asking to recall a specific memory or conversation
        - information: User is asking for factual information
        - new_learning: User is providing new information to be stored
        - emotional: User is expressing emotions or seeking emotional support
        - clarification: User is asking for clarification on something previously said
        - task: User is requesting a specific task to be performed
        - greeting: User is greeting or making small talk
        - other: Default category for queries that don't fit elsewhere
        
        Args:
            query (str): The user's query text
            
        Returns:
            str: The classified query type
        """
        try:
            if not query or not isinstance(query, str):
                logger.warning("Empty or invalid query provided to classify_query")
                return "other"
                
            # Normalize query
            query = query.lower().strip()
            
            # Quick pattern matching for common cases
            if re.search(r'\b(remember|recall|what did (i|you) say|previous|earlier|before)\b', query):
                return "recall"
                
            if re.search(r'\b(who|what|where|when|why|how|explain|tell me about|information)\b', query):
                return "information"
                
            if re.search(r'\b(remember this|note this|save this|store this|keep this|remember that)\b', query):
                return "new_learning"
                
            if re.search(r'\b(feel|feeling|sad|happy|angry|upset|depressed|worried|anxious|excited)\b', query):
                return "emotional"
                
            if re.search(r'\b(what do you mean|clarify|explain again|confused|didn\'t understand)\b', query):
                return "clarification"
                
            if re.search(r'\b(do this|perform|execute|run|start|stop|create|make|build)\b', query):
                return "task"
                
            if re.search(r'\b(hello|hi|hey|good morning|good afternoon|good evening|how are you)\b', query):
                return "greeting"
                
            # More sophisticated classification with LLM if available
            try:
                if hasattr(self, 'llm_pipeline') and self.llm_pipeline:
                    llm_classification = await self._classify_with_llm(query)
                    if llm_classification in ["recall", "information", "new_learning", "emotional", 
                                            "clarification", "task", "greeting", "other"]:
                        logger.info(f"LLM classified query as: {llm_classification}")
                        return llm_classification
            except Exception as e:
                logger.warning(f"Error during LLM classification, falling back to heuristics: {e}")
            
            # Default to information search if nothing else matched
            return "information"
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}", exc_info=True)
            return "other"
            
    async def _classify_with_llm(self, query: str) -> str:
        """Use LLM to classify the query type more accurately.
        
        Args:
            query (str): The user query to classify
            
        Returns:
            str: The classified query type
        """
        try:
            prompt = f"""Classify the following user query into exactly one of these categories: 
            recall, information, new_learning, emotional, clarification, task, greeting, other.
            Respond with only the category name, nothing else.
            
            Query: "{query}"
            
            Classification:"""
            
            response = await self.llm_pipeline.agenerate_text(
                prompt=prompt,
                max_tokens=10,
                temperature=0.2
            )
            
            # Clean and validate response
            if response and isinstance(response, str):
                response = response.lower().strip()
                valid_types = ["recall", "information", "new_learning", "emotional", 
                              "clarification", "task", "greeting", "other"]
                if response in valid_types:
                    return response
                    
            return "information"  # Default fallback
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}", exc_info=True)
            return "information"  # Default fallback on error
    
    async def retrieve_memories(self, query: str, limit: int = 5, min_significance: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the given query using semantic search.
        
        Args:
            query: The query text to search for related memories
            limit: Maximum number of memories to return
            min_significance: Minimum significance threshold for retrieved memories
            
        Returns:
            List of dictionaries containing memory entries
        """
        try:
            logger.info(f"Retrieving memories for query: '{query[:50]}...'")
            
            # Input validation
            if not query or not isinstance(query, str):
                logger.warning("Empty or invalid query provided to retrieve_memories")
                return []
                
            if limit <= 0:
                logger.warning(f"Invalid limit parameter: {limit}, using default of 5")
                limit = 5
                
            # Check which memory system to use
            if hasattr(self, 'memory_integration') and self.memory_integration:
                # Use hierarchical memory system
                try:
                    memories = await self.memory_integration.retrieve_memories(
                        query=query,
                        limit=limit,
                        min_significance=min_significance
                    )
                    logger.info(f"Retrieved {len(memories)} memories from hierarchical memory system")
                    return memories
                except Exception as e:
                    logger.error(f"Error retrieving from hierarchical memory: {e}", exc_info=True)
                    # Fall back to standard retrieval
            
            # Standard memory retrieval using embedding similarity
            try:
                # Generate query embedding
                query_embedding = await self._generate_embedding(query)
                if not query_embedding:
                    logger.warning("Failed to generate embedding for query")
                    return []
                    
                # Retrieve memories from database
                memories = await self._retrieve_memories_by_embedding(
                    query_embedding=query_embedding,
                    limit=limit,
                    min_significance=min_significance
                )
                
                # Format results
                result = []
                for memory in memories:
                    result.append({
                        "id": memory.id,
                        "content": memory.content,
                        "timestamp": memory.timestamp.isoformat() if hasattr(memory, 'timestamp') else None,
                        "significance": memory.significance,
                        "memory_type": memory.memory_type.value if hasattr(memory, 'memory_type') else "transcript",
                        "metadata": memory.metadata
                    })
                    
                logger.info(f"Retrieved {len(result)} memories from standard memory system")
                return result
                    
            except Exception as e:
                logger.error(f"Error in standard memory retrieval: {e}", exc_info=True)
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            return []
            
    async def _retrieve_memories_by_embedding(self, query_embedding: List[float], limit: int = 5, 
                                           min_significance: float = 0.3) -> List[Any]:
        """
        Retrieve memories by comparing the query embedding against stored memory embeddings.
        
        Args:
            query_embedding: Vector embedding of the query
            limit: Maximum number of memories to return
            min_significance: Minimum significance threshold
            
        Returns:
            List of memory objects
        """
        try:
            if not hasattr(self, 'memory_db') or not self.memory_db:
                logger.warning("Memory database not initialized")
                return []
                
            # Retrieve memories from database with similarity search
            memories = await self.memory_db.get_by_vector(
                vector=query_embedding,
                limit=limit * 2,  # Get more than needed to filter by significance
                collection="memories"
            )
            
            # Filter by significance and sort by relevance
            filtered_memories = []
            for memory in memories:
                if hasattr(memory, 'significance') and memory.significance >= min_significance:
                    filtered_memories.append(memory)
                elif not hasattr(memory, 'significance'):
                    # If memory doesn't have significance attribute, include it anyway
                    filtered_memories.append(memory)
            
            # Sort by relevance and limit results
            sorted_memories = sorted(
                filtered_memories, 
                key=lambda x: x.similarity if hasattr(x, 'similarity') else 0.0,
                reverse=True
            )[:limit]
            
            return sorted_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories by embedding: {e}", exc_info=True)
            return []
    
    async def retrieve_information(self, query: str, limit: int = 5, min_significance: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve factual information relevant to the query.
        Optimized for informational queries rather than personal memories.
        
        Args:
            query: The query text
            limit: Maximum number of information pieces to retrieve
            min_significance: Minimum significance threshold
            
        Returns:
            List of dictionaries containing information entries
        """
        try:
            logger.info(f"Retrieving information for query: '{query[:50]}...'")
            
            # Input validation
            if not query or not isinstance(query, str):
                logger.warning("Empty or invalid query provided to retrieve_information")
                return []
                
            # Check which memory system to use
            if hasattr(self, 'memory_integration') and self.memory_integration:
                # Use hierarchical memory system's information retrieval
                try:
                    info_results = await self.memory_integration.retrieve_information(
                        query=query,
                        limit=limit,
                        min_significance=min_significance
                    )
                    logger.info(f"Retrieved {len(info_results)} information entries from hierarchical memory")
                    return info_results
                except Exception as e:
                    logger.error(f"Error retrieving from hierarchical information store: {e}", exc_info=True)
                    # Fall back to standard retrieval
            
            # Use RAG context if available
            if hasattr(self, 'get_rag_context'):
                try:
                    # Generate RAG context for informational queries
                    context_data = await self.get_rag_context(
                        query=query,
                        limit=limit,
                        min_significance=min_significance,
                        context_type="information"
                    )
                    
                    # Parse the context into structured information
                    if context_data and isinstance(context_data, str):
                        info_entries = self._parse_information_context(context_data)
                        if info_entries:
                            logger.info(f"Retrieved {len(info_entries)} information entries from RAG context")
                            return info_entries
                except Exception as e:
                    logger.error(f"Error retrieving RAG context for information: {e}", exc_info=True)
            
            # Fallback: Standard retrieval focused on information memory types
            try:
                # Generate query embedding
                query_embedding = await self._generate_embedding(query)
                if not query_embedding:
                    logger.warning("Failed to generate embedding for query")
                    return []
                
                # Retrieve information memory types from database
                # This uses the same method but with a filter for information memory types
                memories = await self._retrieve_memories_by_embedding(
                    query_embedding=query_embedding,
                    limit=limit * 2,  # Get more to filter
                    min_significance=min_significance
                )
                
                # Filter for information memory types and format results
                result = []
                for memory in memories:
                    # Check if memory has the appropriate type for information
                    is_info_type = False
                    if hasattr(memory, 'memory_type'):
                        info_types = ["fact", "concept", "definition", "information", "knowledge"]
                        memory_type_value = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
                        is_info_type = any(info_type in memory_type_value.lower() for info_type in info_types)
                    
                    # For standard memories, check metadata or content for clues
                    elif hasattr(memory, 'metadata') and memory.metadata:
                        if memory.metadata.get('type') in ['fact', 'information', 'knowledge']:
                            is_info_type = True
                    
                    # Include if it's an information type or if we can't determine (better to include than exclude)
                    if is_info_type or not hasattr(memory, 'memory_type'):
                        result.append({
                            "id": memory.id if hasattr(memory, 'id') else str(uuid.uuid4()),
                            "content": memory.content,
                            "timestamp": memory.timestamp.isoformat() if hasattr(memory, 'timestamp') else None,
                            "significance": getattr(memory, 'significance', 0.5),
                            "memory_type": memory.memory_type.value if hasattr(memory, 'memory_type') and hasattr(memory.memory_type, 'value') else "information",
                            "metadata": getattr(memory, 'metadata', {})
                        })
                
                logger.info(f"Retrieved {len(result)} information entries from standard memory system")
                return result[:limit]  # Limit to requested number
                
            except Exception as e:
                logger.error(f"Error in standard information retrieval: {e}", exc_info=True)
                return []
        
        except Exception as e:
            logger.error(f"Error retrieving information: {e}", exc_info=True)
            return []
            
    def _parse_information_context(self, context_data: str) -> List[Dict[str, Any]]:
        """
        Parse RAG context data into structured information entries.
        
        Args:
            context_data: The RAG context string
            
        Returns:
            List of dictionaries containing parsed information
        """
        try:
            info_entries = []
            
            # Split context into sections (each fact or piece of information)
            sections = re.split(r'\n(?=Fact|Information|Knowledge|Definition|Concept)\s*\d*:', context_data)
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                    
                # Extract content
                content = section.strip()
                
                # Generate a unique ID
                entry_id = str(uuid.uuid4())
                
                # Determine type and confidence
                entry_type = "information"
                confidence = 0.7  # Default confidence
                
                # Look for type indicators in the content
                type_match = re.search(r'^(Fact|Information|Knowledge|Definition|Concept)', content)
                if type_match:
                    entry_type = type_match.group(1).lower()
                
                # Look for confidence indicators
                confidence_match = re.search(r'confidence:\s*(\d+(\.\d+)?)', content, re.IGNORECASE)
                if confidence_match:
                    try:
                        confidence = float(confidence_match.group(1))
                        # Normalize to 0-1 range if it's expressed as percentage
                        if confidence > 1:
                            confidence /= 100
                    except ValueError:
                        pass
                
                info_entries.append({
                    "id": entry_id,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "significance": confidence,
                    "memory_type": entry_type,
                    "metadata": {
                        "source": "rag_context",
                        "confidence": confidence,
                        "index": i
                    }
                })
            
            return info_entries
            
        except Exception as e:
            logger.error(f"Error parsing information context: {e}", exc_info=True)
            return []
    
    async def store_and_retrieve(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Store new information and retrieve contextually related memories.
        This is useful for learning new information while providing context.
        
        Args:
            text: The text to store
            metadata: Optional metadata to attach to the memory
            
        Returns:
            List of related memory objects
        """
        try:
            logger.info(f"Storing and retrieving context for: '{text[:50]}...'")
            
            # Input validation
            if not text or not isinstance(text, str):
                logger.warning("Empty or invalid text provided to store_and_retrieve")
                return []
            
            # Default metadata if none provided
            if metadata is None:
                metadata = {
                    "source": "user_input",
                    "timestamp": datetime.now().isoformat(),
                    "type": "new_learning"
                }
            
            # Store the new information
            store_success = False
            if hasattr(self, 'memory_integration') and self.memory_integration:
                try:
                    # Use hierarchical memory system
                    memory_id = await self.memory_integration.store(
                        content=text,
                        metadata=metadata
                    )
                    logger.info(f"Stored new information with ID {memory_id} in hierarchical memory")
                    store_success = True
                except Exception as e:
                    logger.error(f"Error storing in hierarchical memory: {e}", exc_info=True)
                    # Fall back to base memory storage
            
            # Fallback: Use base memory storage if hierarchical failed or not available
            if not store_success:
                try:
                    memory_id = await self.store_memory(
                        text=text,
                        memory_type="user_input",
                        metadata=metadata
                    )
                    logger.info(f"Stored new information with ID {memory_id} in base memory system")
                    store_success = True
                except Exception as e:
                    logger.error(f"Error storing in base memory: {e}", exc_info=True)
            
            # Return immediately if storage failed
            if not store_success:
                logger.warning("Failed to store new information")
                return []
            
            # Retrieve related memories
            try:
                # First try to classify the text to determine retrieval strategy
                query_type = await self.classify_query(text)
                
                # Use appropriate retrieval method based on classification
                if query_type == "information":
                    related_memories = await self.retrieve_information(text, limit=3, min_significance=0.2)
                else:
                    related_memories = await self.retrieve_memories(text, limit=3, min_significance=0.2)
                
                logger.info(f"Retrieved {len(related_memories)} related memories")
                return related_memories
                
            except Exception as e:
                logger.error(f"Error retrieving related memories: {e}", exc_info=True)
                return []
            
        except Exception as e:
            logger.error(f"Error in store_and_retrieve: {e}", exc_info=True)
            return []
    
    async def store_emotional_context(self, emotional_context: Dict[str, Any]) -> bool:
        """
        Store emotional context information for tracking user emotional states over time.
        
        Args:
            emotional_context: Dictionary containing emotional context data
                - emotion: Primary emotion detected
                - intensity: Intensity score (0.0-1.0)
                - secondary_emotions: List of secondary emotions
                - text: Original text that generated this emotional context
                
        Returns:
            bool: Success status
        """
        try:
            # Input validation
            if not emotional_context or not isinstance(emotional_context, dict) or 'emotion' not in emotional_context:
                logger.warning("Invalid emotional context data provided")
                return False
            
            # Calculate significance based on emotional intensity
            significance = emotional_context.get('intensity', 0.5)
            
            # Prepare metadata
            metadata = {
                "type": "emotional_context",
                "primary_emotion": emotional_context.get('emotion'),
                "intensity": emotional_context.get('intensity', 0.5),
                "secondary_emotions": emotional_context.get('secondary_emotions', []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in emotional history for EmotionMixin
            if hasattr(self, '_emotional_history'):
                self._emotional_history.append(metadata)
                
                # Trim emotional history if needed
                if hasattr(self, '_max_emotional_history') and len(self._emotional_history) > self._max_emotional_history:
                    self._emotional_history = self._emotional_history[-self._max_emotional_history:]
                    
                logger.debug(f"Added to emotional history (total: {len(self._emotional_history)})")
            
            # Also store as a memory if significant enough
            if significance >= 0.4:  # Lower threshold to capture more emotional context
                text = emotional_context.get('text', f"User expressed {metadata['primary_emotion']} with intensity {metadata['intensity']}")
                
                store_success = False
                if hasattr(self, 'memory_integration') and self.memory_integration:
                    try:
                        # Use hierarchical memory system
                        memory_id = await self.memory_integration.store(
                            content=text,
                            metadata=metadata,
                            importance=significance
                        )
                        logger.info(f"Stored emotional context with ID {memory_id} in hierarchical memory")
                        store_success = True
                    except Exception as e:
                        logger.error(f"Error storing emotional context in hierarchical memory: {e}", exc_info=True)
                        # Fall back to base memory storage
                
                # Fallback: Use base memory storage if hierarchical failed or not available
                if not store_success:
                    try:
                        memory_id = await self.store_memory(
                            text=text,
                            memory_type="emotional",
                            significance=significance,
                            metadata=metadata
                        )
                        logger.info(f"Stored emotional context with ID {memory_id} in base memory system")
                        store_success = True
                    except Exception as e:
                        logger.error(f"Error storing emotional context in base memory: {e}", exc_info=True)
            
            logger.info(f"Stored emotional context: {metadata['primary_emotion']} (intensity: {metadata['intensity']})")
            return True
            
        except Exception as e:
            logger.error(f"Error storing emotional context: {e}", exc_info=True)
            return False
    
    async def compare_texts(self, text1: str, text2: str) -> float:
        """
        Compare two texts for semantic similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0-1.0)
        """
        try:
            # Generate embeddings for both texts
            embedding1, _ = await self.process_embedding(text1)
            embedding2, _ = await self.process_embedding(text2)
            
            if embedding1 is None or embedding2 is None:
                logger.warning("Failed to generate embeddings for similarity comparison")
                return 0.0
            
            # Calculate cosine similarity
            if isinstance(embedding1, torch.Tensor) and isinstance(embedding2, torch.Tensor):
                # Normalize embeddings
                embedding1 = embedding1 / embedding1.norm()
                embedding2 = embedding2 / embedding2.norm()
                # Calculate dot product of normalized vectors (cosine similarity)
                similarity = torch.dot(embedding1, embedding2).item()
            else:
                # Convert to numpy arrays if needed
                if not isinstance(embedding1, np.ndarray):
                    embedding1 = np.array(embedding1)
                if not isinstance(embedding2, np.ndarray):
                    embedding2 = np.array(embedding2)
                
                # Normalize embeddings
                embedding1 = embedding1 / np.linalg.norm(embedding1)
                embedding2 = embedding2 / np.linalg.norm(embedding2)
                
                # Calculate dot product (cosine similarity)
                similarity = np.dot(embedding1, embedding2)
            
            # Ensure similarity is between 0 and 1
            similarity = float(max(0.0, min(1.0, similarity)))
            return similarity
            
        except Exception as e:
            logger.error(f"Error comparing texts: {e}")
            return 0.0
```

# hierarchy.py

```py
# memory_core/hierarchy.py

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class HierarchicalMemoryMixin:
    """
    Mixin that provides hierarchical memory organization capabilities.
    
    Note: This is a stub implementation that will be fully implemented later.
    """
    
    def __init__(self):
        # Initialize hierarchical memory structures
        if not hasattr(self, "memory_hierarchies"):
            self.memory_hierarchies = {}
        
        logger.info("Initialized HierarchicalMemoryMixin (stub)")
    
    async def add_to_hierarchy(self, memory_id: str, category: str = None) -> bool:
        """
        Add a memory to the hierarchy.
        
        Args:
            memory_id: The ID of the memory to add
            category: Optional category to add the memory to
            
        Returns:
            bool: Success status
        """
        # Stub implementation
        logger.debug(f"Would add memory {memory_id} to hierarchy category {category}")
        return True
    
    async def get_category_memories(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get memories from a specific category.
        
        Args:
            category: The category to get memories from
            limit: Maximum number of memories to return
            
        Returns:
            List of memories in the category
        """
        # Stub implementation
        logger.debug(f"Would retrieve memories from category {category}")
        return []
    
    async def suggest_categories(self, query: str) -> List[str]:
        """
        Suggest relevant categories based on a query.
        
        Args:
            query: The query to suggest categories for
            
        Returns:
            List of suggested categories
        """
        # Stub implementation
        logger.debug(f"Would suggest categories for query: {query}")
        return []

```

# memory_broker.py

```py
"""Memory broker for inter-agent communication."""

import asyncio
import inspect
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union

logger = logging.getLogger(__name__)

# Singleton instance
_broker_instance = None
_broker_lock = asyncio.Lock()

class MemoryBroker:
    """Broker that facilitates communication between memory components and agents."""
    
    def __init__(self, ping_interval: float = 30.0):
        """Initialize the memory broker.
        
        Args:
            ping_interval: Interval in seconds for ping-pong health checks
        """
        self.callbacks = {}
        self.requests = asyncio.Queue()
        self.responses = {}
        self.running = False
        self.clients = set()
        self.tasks = {}
        self.ping_interval = ping_interval
        
    async def start(self):
        """Start the memory broker."""
        if self.running:
            return
            
        logger.info("Starting memory broker")
        self.running = True
        self.tasks["processor"] = asyncio.create_task(self._process_requests())
        self.tasks["ping"] = asyncio.create_task(self._ping_clients())
        logger.info("Memory broker started")
    
    async def stop(self):
        """Stop the memory broker."""
        if not self.running:
            return
            
        logger.info("Stopping memory broker")
        self.running = False
        
        # Cancel all tasks
        for task_name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clear all data
        self.callbacks = {}
        self.responses = {}
        self.clients = set()
        self.tasks = {}
        
        # Drain the queue
        while not self.requests.empty():
            try:
                self.requests.get_nowait()
                self.requests.task_done()
            except asyncio.QueueEmpty:
                break
                
        logger.info("Memory broker stopped")
    
    async def register_callback(self, operation: str, callback: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        """Register a callback for a specific operation.
        
        Args:
            operation: Name of the operation
            callback: Async function to handle the operation
        """
        if not asyncio.iscoroutinefunction(callback) and not inspect.isawaitable(callback):
            raise ValueError(f"Callback for operation {operation} must be an async function")
            
        self.callbacks[operation] = callback
        logger.info(f"Registered callback for operation: {operation}")
    
    async def unregister_callback(self, operation: str):
        """Unregister a callback for a specific operation.
        
        Args:
            operation: Name of the operation
        """
        if operation in self.callbacks:
            del self.callbacks[operation]
            logger.info(f"Unregistered callback for operation: {operation}")
    
    async def register_client(self, client_id: str = None):
        """Register a client with the broker.
        
        Args:
            client_id: Optional client ID, generated if not provided
            
        Returns:
            str: The client ID
        """
        if client_id is None:
            client_id = str(uuid.uuid4())
            
        self.clients.add(client_id)
        logger.info(f"Registered client: {client_id}")
        return client_id
    
    async def unregister_client(self, client_id: str):
        """Unregister a client from the broker.
        
        Args:
            client_id: Client ID to unregister
        """
        if client_id in self.clients:
            self.clients.remove(client_id)
            logger.info(f"Unregistered client: {client_id}")
    
    async def send_request(self, operation: str, data: Dict[str, Any], client_id: str = None) -> Dict[str, Any]:
        """Send a request to the broker and wait for a response.
        
        Args:
            operation: Name of the operation
            data: Data for the operation
            client_id: Optional client ID
            
        Returns:
            Dict[str, Any]: Response data
        """
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Register the client if not already registered
        if client_id is None:
            client_id = await self.register_client()
        elif client_id not in self.clients:
            await self.register_client(client_id)
        
        # Create a future for the response
        response_future = asyncio.get_event_loop().create_future()
        self.responses[request_id] = response_future
        
        # Prepare the request
        request = {
            "id": request_id,
            "client_id": client_id,
            "operation": operation,
            "data": data
        }
        
        # Queue the request
        await self.requests.put(request)
        
        try:
            # Wait for the response
            return await response_future
        finally:
            # Clean up
            if request_id in self.responses:
                del self.responses[request_id]
    
    async def _process_requests(self):
        """Process requests from the queue."""
        while self.running:
            try:
                # Get the next request
                request = await self.requests.get()
                
                try:
                    # Extract request details
                    request_id = request.get("id")
                    client_id = request.get("client_id")
                    operation = request.get("operation")
                    data = request.get("data", {})
                    
                    # Process the request
                    if operation in self.callbacks:
                        # Call the registered callback
                        callback = self.callbacks[operation]
                        response = await callback(data)
                    else:
                        # No callback registered for this operation
                        response = {
                            "success": False,
                            "error": f"No handler registered for operation: {operation}"
                        }
                    
                    # Set the response future
                    if request_id in self.responses:
                        self.responses[request_id].set_result(response)
                        
                except Exception as e:
                    logger.error(f"Error processing request: {e}", exc_info=True)
                    
                    # Set error response
                    if request_id in self.responses:
                        self.responses[request_id].set_result({
                            "success": False,
                            "error": str(e)
                        })
                
                finally:
                    # Mark the request as done
                    self.requests.task_done()
                    
            except asyncio.CancelledError:
                # Task was cancelled
                break
                
            except Exception as e:
                logger.error(f"Error in request processor: {e}", exc_info=True)
    
    async def _ping_clients(self):
        """Periodically ping clients to check if they're still alive."""
        while self.running:
            try:
                # Sleep for the ping interval
                await asyncio.sleep(self.ping_interval)
                
                # Ping each client (future implementation)
                # This could be used to check if clients are still responsive
                # and clean up resources for disconnected clients
                
            except asyncio.CancelledError:
                # Task was cancelled
                break
                
            except Exception as e:
                logger.error(f"Error in ping task: {e}", exc_info=True)
                # Sleep a bit to avoid tight loop on errors
                await asyncio.sleep(5)

async def get_memory_broker() -> MemoryBroker:
    """Get or create the singleton memory broker instance.
    
    Returns:
        MemoryBroker: The memory broker instance
    """
    global _broker_instance, _broker_lock
    
    async with _broker_lock:
        if _broker_instance is None or not _broker_instance.running:
            _broker_instance = MemoryBroker()
            await _broker_instance.start()
            
    return _broker_instance

```

# memory_client_proxy.py

```py
"""Memory client proxy for the voice agent to communicate with memory agent."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from memory_core.memory_broker import get_memory_broker

logger = logging.getLogger(__name__)

class MemoryClientProxy:
    """Proxy class that mimics the EnhancedMemoryClient API but forwards requests to memory agent."""
    
    def __init__(self, ping_interval: float = 30.0):
        """Initialize the memory client proxy.
        
        Args:
            ping_interval: Interval in seconds for WebSocket health checks
        """
        self.broker = None
        self.ping_interval = ping_interval
        self.client_id = None
        
    async def connect(self):
        """Connect to the memory broker."""
        self.broker = await get_memory_broker()
        self.client_id = await self.broker.register_client()
        logger.info(f"Connected to memory broker with client ID: {self.client_id}")
        
    async def close(self):
        """Close the connection to the memory broker."""
        if self.broker and self.client_id:
            await self.broker.unregister_client(self.client_id)
            logger.info(f"Disconnected from memory broker, client ID: {self.client_id}")
            
    async def classify_query(self, query: str) -> str:
        """Classify a query via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "classify_query", 
            {"query": query},
            client_id=self.client_id
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("query_type", "other")
    
    async def retrieve_memories(self, query: str, limit: int = 5, min_significance: float = 0.3) -> List[Dict[str, Any]]:
        """Retrieve memories via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "retrieve_memories", 
            {
                "query": query,
                "limit": limit,
                "min_significance": min_significance
            },
            client_id=self.client_id
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("memories", [])
        
    async def retrieve_information(self, query: str, context_type: str = "general") -> Dict[str, Any]:
        """Retrieve specific information via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "retrieve_information", 
            {
                "query": query,
                "context_type": context_type
            },
            client_id=self.client_id
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("information", {})

    async def store_and_retrieve(self, content: str, query: str = "", memory_type: str = "conversation") -> Dict[str, Any]:
        """Store content and retrieve related memories via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "store_and_retrieve", 
            {
                "content": content,
                "query": query,
                "memory_type": memory_type
            },
            client_id=self.client_id
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("result", {})

    async def store_emotional_context(self, user_input: str, emotions: Dict[str, float], timestamp: str = None) -> bool:
        """Store emotional context via the memory agent."""
        if not self.broker:
            await self.connect()
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        response = await self.broker.send_request(
            "store_emotional_context", 
            {
                "user_input": user_input,
                "emotions": emotions,
                "timestamp": timestamp
            },
            client_id=self.client_id
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("result", False)

    async def get_emotional_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve emotional history via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "get_emotional_history", 
            {
                "limit": limit
            },
            client_id=self.client_id
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("history", [])

    async def detect_and_store_personal_details(self, text: str) -> Dict[str, Any]:
        """Detect and store personal details via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "detect_and_store_personal_details", 
            {
                "text": text
            },
            client_id=self.client_id
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("details", {})

    async def store_transcript(self, transcript: str, metadata: Dict[str, Any] = None) -> bool:
        """Store a conversation transcript via the memory agent."""
        if not self.broker:
            await self.connect()
        
        if metadata is None:
            metadata = {}
            
        response = await self.broker.send_request(
            "store_transcript", 
            {
                "transcript": transcript,
                "metadata": metadata
            },
            client_id=self.client_id
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("result", False)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for text via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "generate_embedding", 
            {
                "text": text
            },
            client_id=self.client_id
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("embedding", [])

    async def get_rag_context(self, query: str, max_tokens: int = 1000, min_significance: float = 0.3) -> str:
        """Get RAG context for a query via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "get_rag_context", 
            {
                "query": query,
                "max_tokens": max_tokens,
                "min_significance": min_significance
            },
            client_id=self.client_id
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("context", "")

```

# memory_manager.py

```py
# memory_core/memory_manager.py

import logging
import asyncio
from typing import Dict, Any, Optional, List

from memory_core.enhanced_memory_client import EnhancedMemoryClient

# Add this method to the MemoryManager class in memory_manager.py

async def detect_emotional_context(self, text: str) -> Dict[str, Any]:
    """
    Detect emotional context in text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dict with emotional context information
    """
    try:
        # Just pass through to the memory client
        return await self.memory_client.detect_emotional_context(text)
    except Exception as e:
        logger.error(f"Error detecting emotional context: {e}")
        # Return default neutral context on error
        return {
            "timestamp": time.time(),
            "text": text,
            "emotions": {"neutral": 1.0},
            "sentiment": 0.0,
            "emotional_state": "neutral",
            "error": str(e)
        }

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    High-level memory system manager that provides a simplified interface 
    for interacting with the memory system.
    
    This class serves as the main entry point for applications to interact
    with the memory system, hiding the complexity of the underlying
    implementation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the memory manager.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.tensor_server_url = self.config.get("tensor_server_url", "ws://localhost:5001")
        self.hpc_server_url = self.config.get("hpc_server_url", "ws://localhost:5005")
        self.session_id = self.config.get("session_id")
        self.user_id = self.config.get("user_id")
        
        # Create memory client
        self.memory_client = EnhancedMemoryClient(
            tensor_server_url=self.tensor_server_url,
            hpc_server_url=self.hpc_server_url,
            session_id=self.session_id,
            user_id=self.user_id,
            **self.config
        )
        
        logger.info(f"Initialized MemoryManager with session_id={self.session_id}")
    
    async def initialize(self) -> bool:
        """
        Initialize the memory system.
        
        Returns:
            bool: Success status
        """
        try:
            # Initialize the memory client
            await self.memory_client.initialize()
            return True
        except Exception as e:
            logger.error(f"Error initializing memory system: {e}")
            return False
    
    async def process_message(self, text: str, role: str = "user") -> None:
        """
        Process a message and extract relevant information.
        
        Args:
            text: The message text
            role: The role of the sender (user or assistant)
        """
        await self.memory_client.process_message(text, role)
    
    async def search_memory(self, query: str, limit: int = 5, min_significance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for memories based on semantic similarity.
        
        Args:
            query: The search query
            limit: Maximum number of results
            min_significance: Minimum significance threshold
            
        Returns:
            List of matching memories
        """
        return await self.memory_client.search_memory(query, limit, min_significance)
    
    async def store_memory(self, content: str, significance: float = None) -> bool:
        """
        Store a new memory.
        
        Args:
            content: The memory content
            significance: Optional significance override
            
        Returns:
            bool: Success status
        """
        return await self.memory_client.store_memory(content, significance=significance)
    
    async def get_memory_tools(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for LLM integration.
        
        Returns:
            List of tool definitions
        """
        return await self.memory_client.get_memory_tools_for_llm()
    
    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a tool call from the LLM.
        
        Args:
            tool_name: The name of the tool to call
            args: The arguments for the tool
            
        Returns:
            The result of the tool call
        """
        return await self.memory_client.handle_tool_call(tool_name, args)
    
    async def cleanup(self) -> None:
        """
        Clean up resources and persist memories.
        """
        await self.memory_client.cleanup()
        logger.info("Memory manager cleanup complete")

```

# memory-client-proxy.py

```py
"""Memory Client Proxy for interfacing with the Memory Agent."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from memory_core.memory_broker import get_memory_broker

logger = logging.getLogger(__name__)

class MemoryClientProxy:
    """Client proxy for interfacing with the Memory Agent via the broker."""
    
    def __init__(self):
        """Initialize the memory client proxy."""
        self.broker = None
    
    async def connect(self):
        """Connect to the memory broker."""
        if self.broker is None:
            self.broker = await get_memory_broker()
            logger.info("Memory client proxy connected to broker")
    
    async def close(self):
        """Close the connection to the memory broker."""
        # No need to explicitly close the broker connection
        # It's a shared resource managed globally
        self.broker = None
        logger.info("Memory client proxy disconnected from broker")
    
    async def classify_query(self, query: str) -> str:
        """Classify a query via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "classify_query", 
            {
                "query": query
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("query_type", "unknown")
    
    async def retrieve_memories(self, query: str, limit: int = 5, min_significance: float = 0.3) -> List[Dict[str, Any]]:
        """Retrieve memories via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "retrieve_memories", 
            {
                "query": query,
                "limit": limit,
                "min_significance": min_significance
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("memories", [])
    
    async def retrieve_information(self, query: str, context_type: str = "general") -> Dict[str, Any]:
        """Retrieve specific information via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "retrieve_information", 
            {
                "query": query,
                "context_type": context_type
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("information", {})

    async def store_and_retrieve(self, content: str, query: str = "", memory_type: str = "conversation") -> Dict[str, Any]:
        """Store content and retrieve related memories via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "store_and_retrieve", 
            {
                "content": content,
                "query": query,
                "memory_type": memory_type
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("result", {})

    async def store_emotional_context(self, user_input: str, emotions: Dict[str, float], timestamp: str = None) -> bool:
        """Store emotional context via the memory agent."""
        if not self.broker:
            await self.connect()
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        response = await self.broker.send_request(
            "store_emotional_context", 
            {
                "user_input": user_input,
                "emotions": emotions,
                "timestamp": timestamp
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("result", False)

    async def get_emotional_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve emotional history via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "get_emotional_history", 
            {
                "limit": limit
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("history", [])

    async def detect_and_store_personal_details(self, text: str) -> Dict[str, Any]:
        """Detect and store personal details via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "detect_and_store_personal_details", 
            {
                "text": text
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("details", {})

    async def store_transcript(self, transcript: str, metadata: Dict[str, Any] = None) -> bool:
        """Store a conversation transcript via the memory agent."""
        if not self.broker:
            await self.connect()
        
        if metadata is None:
            metadata = {}
            
        response = await self.broker.send_request(
            "store_transcript", 
            {
                "transcript": transcript,
                "metadata": metadata
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("result", False)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for text via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "generate_embedding", 
            {
                "text": text
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("embedding", [])

    async def get_rag_context(self, query: str, max_tokens: int = 1000, min_significance: float = 0.3) -> str:
        """Get RAG context for a query via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "get_rag_context", 
            {
                "query": query,
                "max_tokens": max_tokens,
                "min_significance": min_significance
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("context", "")
```

# personal_details.py

```py
# memory_core/personal_details.py

import logging
import re
from typing import Dict, Any, List, Optional, Set
import time

logger = logging.getLogger(__name__)

class PersonalDetailsMixin:
    """
    Mixin that handles personal details extraction and storage.
    Automatically detects and stores personal information with high significance.
    """
    
    def __init__(self):
        # Initialize personal details storage if not exists
        if not hasattr(self, "personal_details"):
            self.personal_details = {}
        
        # Initialize common patterns
        self._name_patterns = [
            r"(?:my name is|i am|i'm|call me) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})",
            r"([A-Z][a-z]+(?: [A-Z][a-z]+){0,2}) (?:is my name|here)",
        ]
        
        # Known personal detail categories
        self._personal_categories = {
            "name": {"patterns": self._name_patterns, "significance": 0.9},
            "birthday": {"patterns": [r"(?:my birthday is|i was born on) (.+?)[.\n]?"], "significance": 0.85},
            "location": {"patterns": [r"(?:i live in|i'm from|i am from) (.+?)[.\n,]?"], "significance": 0.8},
            "job": {"patterns": [r"(?:i work as a|my job is|i am a) (.+?)[.\n,]?"], "significance": 0.75},
            "family": {"patterns": [r"(?:my (?:wife|husband|partner|son|daughter|child|children) (?:is|are)) (.+?)[.\n,]?"], "significance": 0.85},
        }
        
        # Initialize list of detected names
        self._detected_names: Set[str] = set()
    
    async def detect_personal_details(self, text: str) -> Dict[str, Any]:
        """
        Detect personal details in text using pattern matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict of detected personal details
        """
        found_details = {}
        
        # Check all personal categories
        for category, config in self._personal_categories.items():
            for pattern in config["patterns"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    value = matches[0].strip()
                    found_details[category] = {
                        "value": value,
                        "confidence": 0.9,  # High confidence for direct pattern matches
                        "significance": config["significance"]
                    }
                    
                    # If we found a name, add to detected names
                    if category == "name":
                        self._detected_names.add(value.lower())
        
        return found_details
    
    async def check_for_name_references(self, text: str) -> List[str]:
        """
        Check if text refers to previously detected names.
        
        Args:
            text: The text to check
            
        Returns:
            List of detected names in the text
        """
        found_names = []
        
        # Check for each detected name in the text
        for name in self._detected_names:
            # Split the name to handle first names vs. full names
            name_parts = name.split()
            
            for part in name_parts:
                # Only check parts with length > 2 to avoid false positives
                if len(part) > 2:
                    # Look for the name with word boundaries
                    pattern = r'\b' + re.escape(part) + r'\b'
                    if re.search(pattern, text, re.IGNORECASE):
                        found_names.append(name)
                        break
        
        return found_names
    
    async def store_personal_detail(self, category: str, value: str, significance: float = 0.8) -> bool:
        """
        Store a personal detail with high significance.
        
        Args:
            category: The type of personal detail (e.g., 'name', 'location')
            value: The value of the personal detail
            significance: Significance score (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        try:
            # Store in personal details dict with clear user attribution
            self.personal_details[category] = {
                "value": value,
                "timestamp": time.time(),
                "significance": significance,
                "belongs_to": "USER"  # Explicit attribution to USER
            }
            
            # Also store as a high-significance memory with clear USER attribution
            memory_content = f"USER {category}: {value}"
            await self.store_memory(
                content=memory_content,
                significance=significance,
                metadata={
                    "type": "personal_detail",
                    "category": category,
                    "value": value,
                    "belongs_to": "USER"  # Explicit attribution
                }
            )
            
            # If it's a name, add to detected names
            if category == "name":
                self._detected_names.add(value.lower())
            
            logger.info(f"Stored USER personal detail: {category}={value}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing personal detail: {e}")
            return False
    
    async def get_personal_detail(self, category: str) -> Optional[str]:
        """
        Retrieve a personal detail by category.
        
        Args:
            category: The type of personal detail to retrieve
            
        Returns:
            The value or None if not found
        """
        detail = self.personal_details.get(category)
        if detail:
            return detail.get("value")
        return None
    
    async def process_message_for_personal_details(self, text: str) -> None:
        """
        Process an incoming message to extract and store personal details.
        
        Args:
            text: The message text to process
        """
        # Detect personal details
        details = await self.detect_personal_details(text)
        
        # Store each detected detail
        for category, detail in details.items():
            await self.store_personal_detail(
                category=category,
                value=detail["value"],
                significance=detail["significance"]
            )
        
        # Check for references to known names
        name_references = await self.check_for_name_references(text)
        
        # If names were referenced, boost their significance in memory
        if name_references:
            for name in name_references:
                # Create a memory about the name reference
                memory_content = f"USER mentioned name: {name}"
                await self.store_memory(
                    content=memory_content,
                    significance=0.75,  # Slightly lower than initial detection
                    metadata={
                        "type": "name_reference",
                        "name": name,
                        "belongs_to": "USER"  # Explicit attribution
                    }
                )
    
    async def get_personal_details_tool(self, category: str = None) -> Dict[str, Any]:
        """
        Tool implementation to get personal details.
        
        Args:
            category: Optional category of personal detail to retrieve
            
        Returns:
            Dict with personal details
        """
        # Special handling for name queries
        if category and category.lower() == "name":
            # First try to get from personal details dictionary
            value = await self.get_personal_detail("name")
            
            if value:
                logger.info(f"Retrieved USER name from personal details: {value}")
                return {
                    "found": True,
                    "category": "name",
                    "value": value,
                    "confidence": 0.95,
                    "belongs_to": "USER",  # Explicit attribution
                    "note": "This is the USER's name, not the assistant's name. The assistant's name is Lucidia."
                }
            
            # If not found in personal details, try searching memory
            try:
                # Search for name introduction memories with high significance
                if hasattr(self, "search_memory"):
                    name_memories = await self.search_memory(
                        "user name", 
                        limit=3, 
                        min_significance=0.8
                    )
                    
                    # Look for explicit name statements
                    for memory in name_memories:
                        content = memory.get("content", "")
                        
                        # Check for explicit name statements
                        name_patterns = [
                            r"USER name: ([A-Za-z]+(?: [A-Za-z]+){0,2})",
                            r"USER explicitly stated their name is ([A-Za-z]+(?: [A-Za-z]+){0,2})",
                            r"The USER's name is ([A-Za-z]+(?: [A-Za-z]+){0,2})",
                            r"([A-Za-z]+(?: [A-Za-z]+){0,2}) is my name"
                        ]
                        
                        for pattern in name_patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                found_name = matches[0].strip()
                                logger.info(f"Found USER name in memory: {found_name}")
                                
                                # Store it in personal details for future reference
                                await self.store_personal_detail("name", found_name, 0.9)
                                
                                return {
                                    "found": True,
                                    "category": "name",
                                    "value": found_name,
                                    "confidence": 0.85,
                                    "source": "memory",
                                    "belongs_to": "USER",  # Explicit attribution
                                    "note": "This is the USER's name, not the assistant's name. The assistant's name is Lucidia."
                                }
            except Exception as e:
                logger.error(f"Error searching memory for name: {e}")
        
        # If specific category requested (and not handled by special cases above)
        if category:
            value = await self.get_personal_detail(category)
            return {
                "found": value is not None,
                "category": category,
                "value": value,
                "belongs_to": "USER",  # Explicit attribution
                "note": f"This is the USER's {category}, not the assistant's {category}."
            }
        
        # Return all personal details
        formatted_details = {}
        for cat, detail in self.personal_details.items():
            formatted_details[cat] = detail.get("value")
        
        return {
            "personal_details": formatted_details,
            "count": len(formatted_details),
            "belongs_to": "USER",  # Explicit attribution
            "note": "These are the USER's personal details, not the assistant's."
        }
```

# proactive.py

```py
# memory_client/proactive.py

import logging
from typing import Dict, Any, List
import time

logger = logging.getLogger(__name__)

class ProactiveRetrievalMixin:
    """
    Mixin that handles prediction of relevant memories for the conversation context.
    """

    def __init__(self):
        self._proactive_memory_context = []
        self._prediction_weights = {
            "recency": 0.3,
            "relevance": 0.5,
            "importance": 0.2
        }

    async def predict_relevant_memories(self, current_context: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Return a list of memory objects likely relevant to the current context.
        """
        # Example placeholder
        return []

    async def is_topic_repetitive(self, text: str) -> bool:
        """
        Check if text is too similar to recently discussed topics (avoid repetition).
        """
        return False

```

# rag_context.py

```py
# memory_client/rag_context.py

import logging
import re
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGContextMixin:
    """
    Mixin for advanced context generation: RAG, hierarchical context, dynamic context, etc.
    """

    async def get_enhanced_rag_context(self, query: str, context_type: str = "auto", max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Return a dict with 'context' and 'metadata' for the given query.
        """
        return {"context": "", "metadata": {}}

    async def get_hierarchical_context(self, query: str, max_tokens: int = 1024) -> str:
        return ""

    async def generate_dynamic_context(self, query: str, conversation_history: list, max_tokens: int = 1024) -> str:
        return ""

    async def boost_context_quality(self, query: str, context_text: str, feedback: Dict[str, Any] = None) -> str:
        return context_text

    async def get_rag_context(self, query: str, limit: int = 5, max_tokens: int = 1024, min_significance: float = 0.0) -> str:
        """
        Get memory context for RAG (Retrieval-Augmented Generation).
        
        This method retrieves relevant memories based on the query and formats them
        into a context string that can be used for RAG with an LLM.
        
        Args:
            query: The query to find relevant memories for
            limit: Maximum number of memories to include
            max_tokens: Maximum number of tokens in the context
            min_significance: Minimum significance threshold for memories
            
        Returns:
            str: Formatted memory context for RAG
        """
        try:
            if not query or not isinstance(query, str):
                logger.warning(f"Invalid query provided to get_rag_context: {type(query)}")
                return ""
                
            # Normalize min_significance to ensure it's a valid float
            try:
                min_significance = float(min_significance)
                # Clamp to valid range
                min_significance = max(0.0, min(1.0, min_significance))
            except (ValueError, TypeError):
                logger.warning(f"Invalid min_significance value: {min_significance}, defaulting to 0.0")
                min_significance = 0.0
                
            # First check if this is a personal detail query
            personal_detail_patterns = {
                "name": [r"what.*name", r"who am i", r"call me", r"my name", r"what.*call me"],
                "location": [r"where.*live", r"where.*from", r"my location", r"my address", r"where.*i.*live"],
                "birthday": [r"when.*born", r"my birthday", r"my birth date", r"when.*birthday", r"how old"],
                "job": [r"what.*do for (a )?living", r"my (job|profession|occupation|career|work)", r"where.*work"],
                "family": [r"my (family|wife|husband|partner|child|children|son|daughter|mother|father)"],
            }
            
            # Initialize context parts
            context_parts = []
            personal_detail_found = False
            
            # Check if query matches any personal detail patterns
            for category, patterns in personal_detail_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        logger.info(f"Personal detail query detected in RAG for category: {category}")
                        
                        # Try to get personal detail directly
                        if hasattr(self, "get_personal_detail"):
                            try:
                                value = await self.get_personal_detail(category)
                                if value:
                                    logger.info(f"Found personal detail for RAG: {category}={value}")
                                    # Add personal detail to context parts
                                    context_parts.append(f"### User Personal Information\nThe user's {category} is: {value}\n")
                                    personal_detail_found = True
                            except Exception as e:
                                logger.error(f"Error retrieving personal detail '{category}': {e}")
            
            # If not a personal detail query or no direct match found, search memories
            logger.info(f"Searching for relevant memories for query: {query} with min_significance={min_significance}")
            
            # Check for semantic memory matches
            memories = []
            high_sig_threshold = 0.5  # Lowered from 0.7 to include more important memories
            
            # First, try to get memories with high significance
            try:
                high_sig_memories = await self.search_memory(query, limit=limit, min_significance=high_sig_threshold)
                logger.debug(f"Found {len(high_sig_memories)} high significance memories")
            except Exception as e:
                logger.error(f"Error searching for high significance memories: {e}")
                high_sig_memories = []
            
            # If we don't have enough high significance memories, get some with lower significance
            if len(high_sig_memories) < limit:
                remaining = limit - len(high_sig_memories)
                try:
                    # Use the provided min_significance for the second search
                    low_sig_memories = await self.search_memory(query, limit=remaining, min_significance=min_significance)
                    logger.debug(f"Found {len(low_sig_memories)} additional memories with min_significance={min_significance}")
                    
                    # Filter out any duplicates
                    low_sig_memories = [m for m in low_sig_memories if m.get("id") not in [hm.get("id") for hm in high_sig_memories]]
                    memories = high_sig_memories + low_sig_memories
                except Exception as e:
                    logger.error(f"Error searching for low significance memories: {e}")
            
            # If memory search fails, try a direct content search without semantic matching
            if not memories and not personal_detail_found:
                try:
                    # Fallback to direct content search
                    if hasattr(self, "memories"):
                        # Simple text matching fallback when semantic search fails
                        query_lower = query.lower()
                        direct_matches = []
                        
                        for mem in self.memories:
                            if query_lower in mem.get("content", "").lower():
                                direct_matches.append(mem)
                        
                        if direct_matches:
                            logger.info(f"Found {len(direct_matches)} memories by direct text matching")
                            memories = sorted(direct_matches, key=lambda x: x.get("significance", 0.0), reverse=True)[:limit]
                except Exception as e:
                    logger.error(f"Error in fallback direct search: {e}")
            
            if not memories and not personal_detail_found:
                logger.info("No relevant memories found for RAG context")
                return ""
            
            # Add memories to context parts
            if memories:
                context_parts.append("### Relevant Memories")
                
                # Sort memories by significance (highest first)
                try:
                    sorted_memories = sorted(memories, key=lambda x: x.get("significance", 0.0), reverse=True)
                except Exception as e:
                    logger.error(f"Error sorting memories by significance: {e}")
                    sorted_memories = memories  # Use unsorted if sorting fails
                
                # Add memories to context
                for i, memory in enumerate(sorted_memories):
                    try:
                        content = memory.get("content", "")
                        significance = memory.get("significance", 0.0)
                        timestamp = memory.get("timestamp", 0)
                        
                        # Skip empty content
                        if not content or not content.strip():
                            continue
                        
                        # Format timestamp as human-readable date if available
                        date_str = ""
                        if timestamp:
                            try:
                                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                                date_str = f" ({date_str})"
                            except Exception as e:
                                logger.warning(f"Error formatting timestamp: {e}")
                        
                        # Add memory to context with significance indicator
                        sig_indicator = "*" * int(significance * 5)  # 0-5 stars based on significance
                        memory_str = f"Memory {i+1}{date_str} {sig_indicator}\n{content}\n"
                        context_parts.append(memory_str)
                    except Exception as e:
                        logger.error(f"Error processing memory for context: {e}")
                        continue
            
            # Join context parts
            context = "\n".join(context_parts)
            
            # Truncate if too long (simple approach, could be more sophisticated)
            if len(context) > max_tokens * 4:  # Rough estimate of tokens to chars
                context = context[:max_tokens * 4] + "\n[Context truncated due to length]\n"
            
            logger.info(f"Generated RAG context with {len(sorted_memories) if 'sorted_memories' in locals() else 0} memories")
            return context
            
        except Exception as e:
            logger.error(f"Error generating RAG context: {e}")
            # Return empty string on error rather than propagating the exception
            return ""

```

# tools.py

```py
# memory_core/tools.py

import logging
import numpy as np
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import re
import uuid
import torch
import asyncio
import os
import shutil
import random
import copy

logger = logging.getLogger(__name__)

class ToolsMixin:
    """
    Mixin that provides smaller utility methods: 
    - Embedding creation
    - Searching/storing memories
    - tool endpoints for retrieval
    - emotional context detection
    - personal details management
    """

    async def process_embedding(self, text: str) -> Tuple[Optional[np.ndarray], float]:
        """
        Send text to tensor server for embeddings and HPC for significance. 
        
        Args:
            text: The text to embed
            
        Returns:
            Tuple of (embedding, significance)
        """
        try:
            # Get connection (creates new one if necessary)
            connection = await self._get_tensor_connection()
            if not connection:
                logger.error("Failed to get tensor connection for embedding")
                return None, 0.0
            
            # Create a properly formatted message according to StandardWebSocketInterface
            timestamp = time.time()
            message_id = f"{int(timestamp * 1000)}-{id(self):x}"
            
            # Send embedding request with proper format
            payload = {
                "type": "embed",
                "text": text,
                "client_id": self.session_id or "unknown",
                "message_id": message_id,
                "timestamp": timestamp
            }
            
            await connection.send(json.dumps(payload))
            
            # Get embedding response
            response = await connection.recv()
            data = json.loads(response)
            logger.debug(f"Received embedding response: {data}")
            
            # Extract embedding from standardized response format
            embedding = None
            if isinstance(data, dict):
                # Check for embedding in response data structure
                if 'data' in data and 'embeddings' in data['data']:
                    embedding = np.array(data['data']['embeddings'])
                    logger.debug(f"Found embeddings in data.embeddings: shape={embedding.shape}")
                elif 'data' in data and 'embedding' in data['data']:
                    embedding = np.array(data['data']['embedding'])
                    logger.debug(f"Found embedding in data.embedding: shape={embedding.shape}")
                elif 'embeddings' in data:
                    embedding = np.array(data['embeddings'])
                    logger.debug(f"Found embeddings at root level: shape={embedding.shape}")
                elif 'embedding' in data:
                    embedding = np.array(data['embedding'])
                    logger.debug(f"Found embedding at root level: shape={embedding.shape}")
                else:
                    logger.error(f"Could not find embedding in response. Keys: {data.keys()}")
                    if 'data' in data:
                        logger.error(f"Data keys: {data['data'].keys() if isinstance(data['data'], dict) else 'data is not a dict'}")
            
            if embedding is None:
                logger.error(f"No embedding in response: {data}")
                return None, 0.0
            
            # Process embedding with HPC for significance
            hpc_connection = await self._get_hpc_connection()
            if not hpc_connection:
                logger.error("Failed to get HPC connection for significance")
                return embedding, 0.5  # Default significance
            
            # Send to HPC for significance with proper format
            hpc_timestamp = time.time()
            hpc_message_id = f"{int(hpc_timestamp * 1000)}-{id(self):x}"
            
            hpc_payload = {
                "type": "process",
                "embeddings": embedding.tolist(),
                "client_id": self.session_id or "unknown",
                "message_id": hpc_message_id,
                "timestamp": hpc_timestamp
            }
            
            await hpc_connection.send(json.dumps(hpc_payload))
            
            # Get significance
            hpc_response = await hpc_connection.recv()
            hpc_data = json.loads(hpc_response)
            
            # Extract significance from standardized response format
            significance = 0.5  # Default
            if isinstance(hpc_data, dict):
                if 'data' in hpc_data and 'significance' in hpc_data['data']:
                    significance = hpc_data['data']['significance']
                elif 'significance' in hpc_data:
                    significance = hpc_data['significance']
            
            return embedding, significance
                
        except Exception as e:
            logger.error(f"Error processing embedding: {e}")
            return None, 0.0
    
    async def search_memory(self, query: str, limit: int = 5, min_significance: float = 0.0) -> List[Dict]:
        """
        Search for memories based on semantic similarity with asynchronous processing.
        Implements a multi-tier search strategy with fallbacks and parallel processing.
        
        Args:
            query: The search query
            limit: Maximum number of results
            min_significance: Minimum significance threshold
            
        Returns:
            List of matching memories
        """
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning(f"Invalid query for search_memory: {type(query)}")
            return []

        try:
            # Track performance metrics
            start_time = time.time()
            
            # Try multiple search strategies in parallel for improved reliability and speed
            results = []
            
            # Flag to track if we've received results from tensor server
            tensor_server_results = False
            
            # 1. Try semantic search via tensor server (primary strategy)
            try:
                connection = await self._get_tensor_connection()
                if connection:
                    # Create semantic search task
                    semantic_results = await self._perform_semantic_search(
                        connection=connection,
                        query=query,
                        limit=limit,
                        min_significance=min_significance
                    )
                    
                    if semantic_results:
                        tensor_server_results = True
                        results = semantic_results
                        logger.info(f"Found {len(results)} memories via semantic search in {time.time() - start_time:.3f}s")
            except Exception as e:
                logger.error(f"Error in tensor server search: {e}")
                # Will continue to fallback methods
            
            # 2. If no results from tensor server, use local search techniques
            if not tensor_server_results and hasattr(self, "memories") and self.memories:
                logger.info("Falling back to direct text search")
                fallback_start = time.time()
                
                # Use asyncio.to_thread for potentially CPU-intensive text matching to avoid blocking
                local_results = await asyncio.to_thread(
                    self._perform_local_text_search,
                    query=query,
                    limit=limit,
                    min_significance=min_significance
                )
                
                if local_results:
                    results = local_results
                    logger.info(f"Found {len(results)} memories via fallback text search in {time.time() - fallback_start:.3f}s")
            
            # Log overall search performance
            search_time = time.time() - start_time
            if search_time > 0.1:  # Only log if search took significant time
                logger.info(f"Memory search completed in {search_time:.3f}s with {len(results)} results")
                
            return results
                
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []
    
    async def _perform_semantic_search(self, connection, query: str, limit: int, min_significance: float) -> List[Dict]:
        """
        Perform semantic search using tensor server connection.
        
        Args:
            connection: WebSocket connection to tensor server
            query: Search query
            limit: Maximum results to return
            min_significance: Minimum significance threshold
            
        Returns:
            List of matching memories
        """
        # Send search request with proper format
        timestamp = time.time()
        message_id = f"{int(timestamp * 1000)}-{id(self):x}"
        
        # Request more results than needed to allow for filtering
        request_limit = limit * 3
        
        payload = {
            "type": "search",
            "text": query,
            "limit": request_limit,
            "min_significance": max(0.0, min_significance - 0.2),  # Lower threshold for more results
            "client_id": self.session_id or "unknown",
            "message_id": message_id,
            "timestamp": timestamp
        }
        
        # Set timeout for search operation
        search_timeout = getattr(self, 'search_timeout', 2.0)  # Default 2 seconds timeout
        
        try:
            # Send query to tensor server
            await connection.send(json.dumps(payload))
            
            # Wait for response with timeout
            response = await asyncio.wait_for(connection.recv(), timeout=search_timeout)
            data = json.loads(response)
            
            # Extract results from standardized response format
            results = []
            if isinstance(data, dict):
                if data.get('type') == 'search_results' and 'results' in data:
                    results = data['results']
                elif 'data' in data and 'results' in data['data']:
                    results = data['data']['results']
                elif 'results' in data:
                    results = data['results']
            
            if results:
                # Filter by significance
                filtered_results = [
                    r for r in results 
                    if r.get('significance', 0.0) >= min_significance
                ]
                
                # Sort by similarity and limit
                sorted_results = sorted(
                    filtered_results, 
                    key=lambda x: x.get('score', 0.0), 
                    reverse=True
                )[:limit]
                
                return sorted_results
                
        except asyncio.TimeoutError:
            logger.warning(f"Tensor server search timed out after {search_timeout}s")
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            
        return []
    
    def _perform_local_text_search(self, query: str, limit: int, min_significance: float) -> List[Dict]:
        """
        Perform local text-based search on in-memory data.
        This is a CPU-bound operation that should be run in a separate thread.
        
        Args:
            query: Search query
            limit: Maximum results to return
            min_significance: Minimum significance threshold
            
        Returns:
            List of matching memories
        """
        query_lower = query.lower()
        results = []
        
        # First try exact substring match (highest confidence)
        for memory in self.memories:
            content = memory.get("content", "")
            significance = memory.get("significance", 0.0)
            
            if significance >= min_significance and content and query_lower in content.lower():
                # Create a result with the same structure as tensor server results
                results.append({
                    **memory,
                    "score": 0.9  # High score for direct matches
                })
        
        # If not enough direct matches, try word-level matching
        if len(results) < limit:
            query_words = set(query_lower.split())
            
            # Skip very short queries or single words for word-level matching
            if len(query_words) > 1:
                for memory in self.memories:
                    # Skip if already in results
                    if any(memory.get("id") == r.get("id") for r in results):
                        continue
                    
                    content = memory.get("content", "")
                    significance = memory.get("significance", 0.0)
                    
                    if significance >= min_significance and content:
                        content_words = set(content.lower().split())
                        common_words = query_words.intersection(content_words)
                        
                        if common_words:
                            # Calculate a score based on word overlap
                            match_score = len(common_words) / len(query_words)
                            if match_score >= 0.3:  # At least 30% word overlap
                                results.append({
                                    **memory,
                                    "score": match_score
                                })
        
        if results:
            # Sort by score and limit
            sorted_results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)[:limit]
            return sorted_results
            
        return []

    async def store_memory(self, content: str, metadata: Dict[str, Any] = None, significance: float = None, importance: float = None) -> bool:
        """
        Store a new memory with semantic embedding using an optimized asynchronous process.
        Features parallel processing for embedding generation and non-blocking persistence.
        
        Args:
            content: The memory content to store
            metadata: Additional metadata for the memory
            significance: Optional override for significance
            importance: Alternate name for significance (for backward compatibility)
            
        Returns:
            bool: Success status
        """
        if not content or not content.strip():
            logger.warning("Attempted to store empty memory content")
            return False
            
        try:
            # Track timing for performance monitoring
            start_time = time.time()
            
            # Handle both significance and importance parameters for backward compatibility
            if significance is None and importance is not None:
                significance = importance
            
            # Generate embedding and significance
            try:
                # Use asyncio.shield to prevent cancellation during important embedding process
                embedding_task = asyncio.shield(self.process_embedding(content))
                embedding, memory_significance = await embedding_task
                
                # Use provided significance if available
                if significance is not None:
                    try:
                        memory_significance = float(significance)
                        # Clamp to valid range
                        memory_significance = max(0.0, min(1.0, memory_significance))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid significance value: {significance}, using calculated value: {memory_significance}")
                
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Fallback: Use zero embedding and default significance
                if hasattr(self, 'embedding_dim'):
                    embedding = torch.zeros(self.embedding_dim)
                else:
                    embedding = torch.zeros(384)  # Default embedding dimension
                memory_significance = 0.5 if significance is None else significance
            
            # Create memory object
            memory_id = str(uuid.uuid4())
            memory = {
                "id": memory_id,
                "content": content,
                "embedding": embedding,
                "timestamp": time.time(),
                "significance": memory_significance,
                "metadata": metadata or {}
            }
            
            # Add to memory list - this needs to be atomic
            async with self._memory_lock:
                self.memories.append(memory)
                
            # Log memory creation time
            creation_time = time.time() - start_time
            logger.info(f"Created new memory with ID {memory_id} and significance {memory_significance:.2f} in {creation_time:.3f}s")
            
            # Use a lower threshold for immediate persistence to ensure more memories are saved promptly
            # but avoid blocking the main thread for low-significance memories
            persistence_threshold = getattr(self, 'immediate_persistence_threshold', 0.3)
            
            # Force immediate persistence for significant memories
            if memory_significance >= persistence_threshold and hasattr(self, 'persistence_enabled') and self.persistence_enabled:
                # Launch persistence as a background task that won't block this method
                # but still ensure it gets done soon
                task = asyncio.create_task(self._background_persist_memory(memory_id, memory))
                
                # Don't wait for persistence to complete - this makes the operation non-blocking
                # but we'll still log any errors via the task
                
                # Add optional callback for debugging/monitoring
                if getattr(self, 'debug_persistence', False):
                    task.add_done_callback(lambda t: self._log_persistence_result(t, memory_id))
            
            return True
                
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False
    
    async def _background_persist_memory(self, memory_id: str, memory: Dict) -> bool:
        """
        Persist a memory in the background without blocking the main operation.
        
        Args:
            memory_id: ID of the memory to persist
            memory: Memory object to persist
            
        Returns:
            bool: Success status
        """
        try:
            persist_start = time.time()
            logger.debug(f"Starting background persistence for memory {memory_id}")
            
            # Set up retry parameters
            max_retries = getattr(self, 'max_retries', 3)
            retry_delay = getattr(self, 'retry_delay', 0.5)  # seconds
            
            # Try to persist memory with retries
            for retry in range(max_retries):
                try:
                    # Check if we have a _persist_single_memory method (from BaseMemoryClient)
                    if hasattr(self, '_persist_single_memory'):
                        success = await self._persist_single_memory(memory)
                        if success:
                            persist_time = time.time() - persist_start
                            logger.info(f"Background persistence for memory {memory_id} completed in {persist_time:.3f}s")
                            return True
                    else:
                        # Fallback to manual persistence if method not available
                        # Get file path
                        file_path = self.storage_path / f"{memory_id}.json"
                        temp_file_path = self.storage_path / f"{memory_id}.json.tmp"
                        
                        # Create deep copy and convert types
                        memory_copy = copy.deepcopy(memory)
                        if hasattr(self, '_convert_numpy_to_python'):
                            memory_copy = self._convert_numpy_to_python(memory_copy)
                        
                        # Write to temp file (use thread to avoid blocking)
                        json_content = json.dumps(memory_copy, ensure_ascii=False, indent=2)
                        await asyncio.to_thread(self._write_to_file, temp_file_path, json_content)
                        
                        # Rename to final file (atomic operation, use thread to avoid blocking)
                        await asyncio.to_thread(os.replace, temp_file_path, file_path)
                        persist_time = time.time() - persist_start
                        logger.info(f"Manual background persistence for memory {memory_id} completed in {persist_time:.3f}s")
                        return True
                        
                except Exception as e:
                    last_error = e
                    # Only retry if we haven't exhausted retries
                    if retry < max_retries - 1:
                        # Exponential backoff with jitter
                        backoff_time = retry_delay * (2 ** retry) * (0.5 + 0.5 * random.random())
                        logger.warning(f"Persistence retry {retry+1}/{max_retries} for memory {memory_id} after {backoff_time:.2f}s: {e}")
                        await asyncio.sleep(backoff_time)
                    else:
                        logger.error(f"Failed to persist memory {memory_id} after {max_retries} attempts: {e}")
                        return False
            
            # Should never reach here due to return in the loop
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error in background persistence for memory {memory_id}: {e}")
            return False
    
    def _write_to_file(self, file_path, content):
        """Helper method for writing to a file from a thread."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _log_persistence_result(self, task, memory_id):
        """Helper method to log the result of a background persistence task."""
        try:
            # Extract result or exception
            if task.cancelled():
                logger.warning(f"Background persistence for memory {memory_id} was cancelled")
            elif task.exception():
                logger.error(f"Background persistence for memory {memory_id} failed with error: {task.exception()}")
            else:
                result = task.result()
                if result:
                    logger.debug(f"Background persistence for memory {memory_id} completed successfully")
                else:
                    logger.warning(f"Background persistence for memory {memory_id} reported failure")
        except Exception as e:
            logger.error(f"Error checking persistence task result for memory {memory_id}: {e}")

    async def search_memory_tool(self, query: str = "", memory_type: str = "all", max_results: int = 5, min_significance: float = 0.0, time_range: Dict = None) -> Dict[str, Any]:
        """
        Tool implementation for memory search.
        
        Args:
            query: The search query string
            memory_type: Type of memory to search (default: all)
            max_results: Maximum number of results to return (default: 5)
            min_significance: Minimum significance threshold (default: 0.0)
            time_range: Optional time range filter
            
        Returns:
            Dict with search results
        """
        if not query:
            return {"error": "No query provided", "memories": []}
        
        # Check for personal detail queries with more comprehensive patterns
        personal_detail_patterns = {
            "name": [r"what.*name", r"who am i", r"call me", r"my name", r"what.*call me", r"how.*call me", r"what.*i go by"],
            "location": [r"where.*live", r"where.*from", r"my location", r"my address", r"my home", r"where.*stay", r"where.*i.*live"],
            "birthday": [r"when.*born", r"my birthday", r"my birth date", r"when.*birthday", r"how old", r"my age"],
            "job": [r"what.*do for (a )?living", r"my (job|profession|occupation|career|work)", r"where.*work", r"what.*i do"],
            "family": [r"my (family|wife|husband|partner|child|children|son|daughter|mother|father|parent|sibling|brother|sister)"],
        }
        
        # First, try direct personal detail retrieval with higher priority
        for category, patterns in personal_detail_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    logger.info(f"Personal detail query detected for category: {category}")
                    
                    # Try to get personal detail directly first
                    if hasattr(self, "get_personal_detail"):
                        value = await self.get_personal_detail(category)
                        if value:
                            logger.info(f"Found personal detail directly: {category}={value}")
                            # Return the personal detail as a high-significance memory
                            return {
                                "memories": [
                                    {
                                        "content": f"User {category}: {value}",
                                        "significance": 0.95,
                                        "timestamp": time.time()
                                    }
                                ],
                                "count": 1
                            }
                    
                    # If direct retrieval failed, try searching memory with type filters
                    if hasattr(self, "search_memory"):
                        # First try searching for memories with metadata type related to this category
                        type_specific_results = []
                        try:
                            # Search for memories with this category in metadata
                            async with self._memory_lock:
                                type_specific_results = [
                                    memory for memory in self.memories 
                                    if memory.get("metadata", {}).get("type") == f"{category}_reference" or 
                                       memory.get("metadata", {}).get("type") == f"{category}_introduction" or
                                       memory.get("metadata", {}).get("name") == category or
                                       memory.get("metadata", {}).get("category") == category
                                ]
                            
                            # Sort by significance and recency
                            type_specific_results = sorted(
                                type_specific_results,
                                key=lambda x: (x.get("significance", 0.0), x.get("timestamp", 0)),
                                reverse=True
                            )[:max_results]
                            
                            if type_specific_results:
                                logger.info(f"Found {len(type_specific_results)} memories with metadata type related to {category}")
                                formatted_results = [
                                    {
                                        "content": r.get("content", ""),
                                        "significance": r.get("significance", 0.0),
                                        "timestamp": r.get("timestamp", 0)
                                    } for r in type_specific_results
                                ]
                                return {
                                    "memories": formatted_results,
                                    "count": len(formatted_results)
                                }
                        except Exception as e:
                            logger.error(f"Error searching for type-specific memories: {e}")
                    
                    # If we couldn't get it directly, search with higher significance threshold
                    # and add category-specific terms to the query
                    enhanced_query = f"{category} {query}"
                    min_significance = max(min_significance, 0.7)  # Raise significance threshold
                    
                    # Use the enhanced query for the search
                    query = enhanced_query
                    logger.info(f"Using enhanced query for personal detail: {enhanced_query}")
                    break
        
        # For backward compatibility, use max_results as limit
        limit = max_results
        
        # Perform the actual memory search
        results = await self.search_memory(query, limit, min_significance)
        
        # Format for LLM consumption
        formatted_results = [
            {
                "content": r.get("content", ""),
                "significance": r.get("significance", 0.0),
                "timestamp": r.get("timestamp", 0)
            } for r in results
        ]
        
        return {
            "memories": formatted_results,
            "count": len(formatted_results)
        }
    
    async def store_important_memory(self, content: str = "", significance: float = 0.8) -> Dict[str, Any]:
        """
        Tool implementation to store an important memory.
        
        Args:
            content: The memory content to store
            significance: Memory significance score (0.0-1.0)
            
        Returns:
            Dict with status
        """
        if not content:
            return {"success": False, "error": "No content provided"}
        
        try:
            # Get connection
            connection = await self._get_tensor_connection()
            if not connection:
                logger.error("Failed to get tensor connection for storing important memory")
                return {"success": False, "error": "Connection failed"}
            
            # Send store request with proper format
            timestamp = time.time()
            message_id = f"{int(timestamp * 1000)}-{id(self):x}"
            
            payload = {
                "type": "embed",
                "text": content,
                "client_id": self.session_id or "unknown",
                "message_id": message_id,
                "timestamp": timestamp
            }
            
            await connection.send(json.dumps(payload))
            
            # Get response
            response = await connection.recv()
            data = json.loads(response)
            
            # Check success status
            success = False
            if isinstance(data, dict):
                if 'data' in data and 'success' in data['data']:
                    success = data['data']['success']
                elif 'success' in data:
                    success = data['success']
                elif data.get('type') == 'embeddings':  # Consider successful embedding generation as success
                    success = True
            
            if success:
                logger.info(f"Successfully stored important memory with significance {significance}")
            else:
                logger.error("Failed to store important memory")
                
            return {"success": success}
            
        except Exception as e:
            logger.error(f"Error storing important memory: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_important_memories(self, limit: int = 5, min_significance: float = 0.7) -> Dict[str, Any]:
        """
        Tool implementation to get important memories.
        
        Args:
            limit: Maximum number of memories to return
            min_significance: Minimum significance threshold (0.0-1.0)
            
        Returns:
            Dict with important memories
        """
        try:
            # Get connection
            connection = await self._get_tensor_connection()
            if not connection:
                logger.error("Failed to get tensor connection for important memories")
                return {"memories": [], "count": 0}
            
            # Send search request with proper format
            timestamp = time.time()
            message_id = f"{int(timestamp * 1000)}-{id(self):x}"
            
            payload = {
                "type": "search",
                "min_significance": min_significance,
                "limit": limit,
                "sort_by": "significance",
                "client_id": self.session_id or "unknown",
                "message_id": message_id,
                "timestamp": timestamp
            }
            
            await connection.send(json.dumps(payload))
            
            # Get response
            response = await connection.recv()
            data = json.loads(response)
            
            # Extract results from standardized response format
            memories = []
            if isinstance(data, dict):
                if data.get('type') == 'search_results' and 'results' in data:
                    memories = data['results']
                elif 'data' in data and 'results' in data['data']:
                    memories = data['data']['results']
                elif 'results' in data:
                    memories = data['results']
            
            # Format for LLM consumption
            formatted_memories = [
                {
                    "content": mem.get("content", ""),
                    "significance": mem.get("significance", 0.0),
                    "timestamp": mem.get("timestamp", 0)
                } for mem in memories
            ]
            
            return {
                "memories": formatted_memories,
                "count": len(formatted_memories)
            }
            
        except Exception as e:
            logger.error(f"Error getting important memories: {e}")
            return {"memories": [], "count": 0}

    async def get_emotional_context(self, args: Dict = None, limit: int = 5) -> Dict[str, Any]:
        """
        Tool implementation to get emotional context information.
        
        Args:
            args: Optional arguments (unused)
            limit: Maximum number of emotions to include
            
        Returns:
            Dict with emotional context information
        """
        try:
            if not hasattr(self, "emotions") or not self.emotions:
                return {
                    "success": True,
                    "summary": "No emotional context information available yet.",
                    "emotions": {}
                }
            
            # Get recent emotions (limited by the limit parameter)
            recent_emotions = list(self.emotions.values())[-limit:] if self.emotions else []
            
            # Calculate average sentiment
            avg_sentiment = sum(e.get("sentiment", 0) for e in recent_emotions) / max(len(recent_emotions), 1)
            
            # Get dominant emotions
            all_detected = {}
            for emotion_data in recent_emotions:
                for emotion, score in emotion_data.get("emotions", {}).items():
                    if emotion not in all_detected:
                        all_detected[emotion] = []
                    all_detected[emotion].append(score)
            
            # Average the scores
            dominant_emotions = {}
            for emotion, scores in all_detected.items():
                dominant_emotions[emotion] = sum(scores) / len(scores)
            
            # Sort dominant emotions by score
            sorted_emotions = sorted(dominant_emotions.items(), key=lambda x: x[1], reverse=True)
            top_emotions = dict(sorted_emotions[:3])  # Top 3 emotions
            
            # Create emotional context summary
            if avg_sentiment > 0.6:
                sentiment_desc = "very positive"
            elif avg_sentiment > 0.2:
                sentiment_desc = "positive"
            elif avg_sentiment > -0.2:
                sentiment_desc = "neutral"
            elif avg_sentiment > -0.6:
                sentiment_desc = "negative"
            else:
                sentiment_desc = "very negative"
                
            emotion_list = ", ".join([f"{emotion}" for emotion, _ in sorted_emotions[:3]])
            summary = f"User's recent emotional state appears {sentiment_desc} with prevalent emotions of {emotion_list}."
            
            return {
                "success": True,
                "summary": summary,
                "sentiment": avg_sentiment,
                "emotions": top_emotions,
                "recent_emotions": recent_emotions
            }
            
        except Exception as e:
            logger.error(f"Error getting emotional context: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": "Failed to retrieve emotional context."
            }
    
    async def get_personal_details_tool(self, args: Dict = None, category: str = None) -> Dict[str, Any]:
        """
        Tool implementation to get personal details about the user.
        
        Args:
            args: Optional arguments (unused)
            category: Optional category of personal details to retrieve
            
        Returns:
            Dict with personal details
        """
        try:
            details = {}
            
            # Try to get from personal details cache first
            if hasattr(self, "personal_details") and self.personal_details:
                # If category is specified, only return that category
                if category and category in self.personal_details:
                    details = {category: self.personal_details[category]}
                    logger.info(f"Retrieved personal detail for category: {category}")
                else:
                    details = self.personal_details.copy()
                    logger.info(f"Retrieved {len(details)} personal details from cache")
            
            # If no details in cache, try to extract from memories
            if not details and hasattr(self, "memories") and self.memories:
                # Look for memories with personal detail metadata
                async with self._memory_lock:
                    personal_memories = [m for m in self.memories 
                                        if m.get("metadata", {}).get("type") in 
                                        ["personal_detail", "name_reference", "location_reference", 
                                         "birthday_reference", "job_reference", "family_reference"]]
                
                # Extract details from memory content
                if personal_memories:
                    logger.info(f"Found {len(personal_memories)} personal detail memories")
                    
                    # Look for name references
                    name_memories = [m for m in personal_memories 
                                   if m.get("metadata", {}).get("type") == "name_reference" or 
                                      "name" in m.get("content", "").lower()]
                    if name_memories:
                        # Sort by significance and recency
                        name_memories = sorted(name_memories, 
                                              key=lambda x: (x.get("significance", 0), x.get("timestamp", 0)), 
                                              reverse=True)
                        details["name"] = name_memories[0].get("content")
                    
                    # Look for location references
                    location_memories = [m for m in personal_memories 
                                      if m.get("metadata", {}).get("type") == "location_reference" or 
                                         "location" in m.get("content", "").lower() or 
                                         "live" in m.get("content", "").lower()]
                    if location_memories:
                        location_memories = sorted(location_memories, 
                                                 key=lambda x: (x.get("significance", 0), x.get("timestamp", 0)), 
                                                 reverse=True)
                        details["location"] = location_memories[0].get("content")
                    
                    # Add other detail types as needed
            
            # If we found details, cache them for future use
            if details and hasattr(self, "personal_details"):
                self.personal_details.update(details)
            
            return {
                "success": len(details) > 0,
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Error getting personal details: {e}")
            return {
                "success": False,
                "error": str(e),
                "details": {}
            }

    async def get_memory_tools(self) -> List[Dict[str, Any]]:
        """
        Return OpenAI-compatible function definitions for memory tools.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search for relevant memories based on semantic similarity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant memories"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of memories to return",
                                "default": 5
                            },
                            "min_significance": {
                                "type": "number",
                                "description": "Minimum significance threshold (0.0 to 1.0)",
                                "default": 0.0
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "store_important_memory",
                    "description": "Store an important memory with high significance",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The memory content to store"
                            },
                            "significance": {
                                "type": "number",
                                "description": "Memory significance (0.0 to 1.0)",
                                "default": 0.8
                            }
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_important_memories",
                    "description": "Retrieve the most important memories",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of important memories to return",
                                "default": 5
                            },
                            "min_significance": {
                                "type": "number",
                                "description": "Minimum significance threshold (0.0 to 1.0)",
                                "default": 0.7
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_emotional_context",
                    "description": "Get the current emotional context and patterns from memory",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_personal_details",
                    "description": "Get personal details about the user from memory",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

    async def compare_texts(self, text1: str, text2: str) -> float:
        """
        Compare two texts for semantic similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0-1.0)
        """
        try:
            # Generate embeddings for both texts
            embedding1, _ = await self.process_embedding(text1)
            embedding2, _ = await self.process_embedding(text2)
            
            if embedding1 is None or embedding2 is None:
                logger.warning("Failed to generate embeddings for similarity comparison")
                return 0.0
            
            # Calculate cosine similarity
            if isinstance(embedding1, torch.Tensor) and isinstance(embedding2, torch.Tensor):
                # Normalize embeddings
                embedding1 = embedding1 / embedding1.norm()
                embedding2 = embedding2 / embedding2.norm()
                # Calculate dot product of normalized vectors (cosine similarity)
                similarity = torch.dot(embedding1, embedding2).item()
            else:
                # Convert to numpy arrays if needed
                if not isinstance(embedding1, np.ndarray):
                    embedding1 = np.array(embedding1)
                if not isinstance(embedding2, np.ndarray):
                    embedding2 = np.array(embedding2)
                
                # Normalize embeddings
                embedding1 = embedding1 / np.linalg.norm(embedding1)
                embedding2 = embedding2 / np.linalg.norm(embedding2)
                
                # Calculate dot product (cosine similarity)
                similarity = np.dot(embedding1, embedding2)
            
            # Ensure similarity is between 0 and 1
            similarity = float(max(0.0, min(1.0, similarity)))
            return similarity
            
        except Exception as e:
            logger.error(f"Error comparing texts: {e}")
            return 0.0

    def _convert_numpy_to_python(self, obj):
        """
        Recursively convert numpy types to Python native types for JSON serialization.
        
        Args:
            obj: The object to convert
            
        Returns:
            The converted object with numpy types replaced by Python native types
        """
        # Handle None
        if obj is None:
            return None
            
        # Handle NumPy arrays
        if hasattr(obj, '__module__') and obj.__module__ == 'numpy':
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return str(obj)
            
        # Handle PyTorch tensors
        if hasattr(obj, '__module__') and 'torch' in obj.__module__:
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if hasattr(obj, 'detach') and hasattr(obj.detach(), 'numpy') and hasattr(obj.detach().numpy(), 'tolist'):
                return obj.detach().numpy().tolist()
            return str(obj)
            
        # Handle dictionaries
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_python(v) for k, v in obj.items()}
            
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._convert_numpy_to_python(item) for item in obj]
            
        # Handle sets
        if isinstance(obj, set):
            return [self._convert_numpy_to_python(item) for item in obj]
            
        # Handle other non-serializable types
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

```

