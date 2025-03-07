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