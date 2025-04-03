# synthians_memory_core/assembly_sync_manager.py

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import json
import os
import shutil
from collections import deque
import uuid

from .custom_logger import logger
from .memory_structures import MemoryAssembly

class AssemblySyncManager:
    """Manages the synchronization of MemoryAssembly embeddings with the vector index.
    
    This class implements a reliable retry queue for assemblies that fail to update
    in the vector index, providing stability and consistency for the Phase 5.8
    Memory Assembly integration.
    """
    
    def __init__(self, vector_index, storage_path: str = None, max_retries: int = 5):
        self.vector_index = vector_index
        self.storage_path = storage_path
        self.max_retries = max_retries
        self.pending_updates: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}
        self.last_retry_attempt: Dict[str, float] = {}
        self.update_lock = asyncio.Lock()
        self._is_running = False
        self._retry_task = None
        
        # Stats tracking
        self.total_sync_attempts = 0
        self.total_sync_successes = 0
        self.total_sync_failures = 0
        self.total_retries = 0
        
        # Configuration for retry behavior
        self.retry_backoff_base = 2.0  # Exponential backoff base
        self.initial_retry_delay = 5.0  # Initial retry delay in seconds
        self.max_retry_delay = 300.0  # Maximum retry delay (5 minutes)
        
        # Load any pending updates from disk
        self._load_pending_updates()
    
    def _get_pending_updates_path(self) -> str:
        """Get the path to the pending updates JSON file."""
        if not self.storage_path:
            return None
        return os.path.join(self.storage_path, "pending_assembly_updates.json")
    
    def _load_pending_updates(self) -> None:
        """Load pending updates from disk if they exist."""
        path = self._get_pending_updates_path()
        if not path or not os.path.exists(path):
            return
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
                self.pending_updates = data.get("pending_updates", {})
                self.retry_counts = data.get("retry_counts", {})
                self.last_retry_attempt = data.get("last_retry_attempt", {})
                
                # Convert string keys back to assembly_ids
                self.pending_updates = {str(k): v for k, v in self.pending_updates.items()}
                self.retry_counts = {str(k): v for k, v in self.retry_counts.items()}
                self.last_retry_attempt = {str(k): v for k, v in self.last_retry_attempt.items()}
                
                logger.info(f"Loaded {len(self.pending_updates)} pending assembly updates")
        except Exception as e:
            logger.error(f"Error loading pending assembly updates: {str(e)}", exc_info=True)
    
    async def _save_pending_updates(self) -> None:
        """Save pending updates to disk."""
        path = self._get_pending_updates_path()
        if not path:
            return
            
        try:
            # Create a temporary copy for serialization
            data = {
                "pending_updates": self.pending_updates,
                "retry_counts": self.retry_counts,
                "last_retry_attempt": self.last_retry_attempt,
                "saved_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Use atomic write pattern for reliability
            temp_path = f"{path}.tmp.{uuid.uuid4().hex[:8]}"
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
                
            # Rename temp file to actual file (atomic on most filesystems)
            if os.path.exists(path):
                shutil.move(temp_path, path)  # atomic replace
            else:
                os.rename(temp_path, path)
                
        except Exception as e:
            logger.error(f"Error saving pending assembly updates: {str(e)}", exc_info=True)
    
    async def queue_assembly_update(self, assembly: MemoryAssembly) -> None:
        """Queue an assembly for synchronization with the vector index.
        
        If the immediate synchronization attempt fails, the assembly will be
        added to the pending updates queue for later retry.
        
        Args:
            assembly: The MemoryAssembly to synchronize
        """
        if not assembly or not assembly.is_active:
            return
            
        assembly_id = assembly.assembly_id
        async with self.update_lock:
            # Track attempt
            self.total_sync_attempts += 1
            
            # Try immediate synchronization
            logger.debug(f"Attempting immediate synchronization for assembly {assembly_id}")
            success = await assembly.update_vector_index_async(self.vector_index)
            
            if success:
                # Success! Remove from pending if present
                self.total_sync_successes += 1
                if assembly_id in self.pending_updates:
                    del self.pending_updates[assembly_id]
                    del self.retry_counts[assembly_id]
                    del self.last_retry_attempt[assembly_id]
                    await self._save_pending_updates()
                logger.debug(f"Assembly {assembly_id} synchronized successfully")
            else:
                # Failed - add to pending updates
                self.total_sync_failures += 1
                self.pending_updates[assembly_id] = {
                    "assembly_id": assembly_id,
                    "queued_at": datetime.now(timezone.utc).isoformat(),
                    "name": assembly.name,
                    "memories_count": len(assembly.memories)
                }
                self.retry_counts[assembly_id] = 0
                self.last_retry_attempt[assembly_id] = time.time()
                await self._save_pending_updates()
                logger.warning(f"Failed to synchronize assembly {assembly_id}, added to retry queue")
                
                # Ensure retry task is running
                await self.start_retry_task()
    
    async def start_retry_task(self) -> None:
        """Start the background task that processes pending updates."""
        if self._is_running:
            return
            
        self._is_running = True
        if self._retry_task is None or self._retry_task.done():
            self._retry_task = asyncio.create_task(self._retry_loop())
            logger.info("Started assembly synchronization retry task")
    
    async def stop_retry_task(self) -> None:
        """Stop the background retry task."""
        self._is_running = False
        if self._retry_task and not self._retry_task.done():
            try:
                self._retry_task.cancel()
                await self._retry_task
            except asyncio.CancelledError:
                pass
            self._retry_task = None
            logger.info("Stopped assembly synchronization retry task")
    
    async def _retry_loop(self) -> None:
        """Background task that processes pending updates with exponential backoff."""
        try:
            while self._is_running:
                retry_candidates = []
                now = time.time()
                
                # Find assemblies eligible for retry
                async with self.update_lock:
                    for assembly_id, info in list(self.pending_updates.items()):
                        retry_count = self.retry_counts.get(assembly_id, 0)
                        last_attempt = self.last_retry_attempt.get(assembly_id, 0)
                        
                        # Calculate backoff delay for this retry
                        delay = min(
                            self.initial_retry_delay * (self.retry_backoff_base ** retry_count),
                            self.max_retry_delay
                        )
                        
                        # Check if it's time to retry
                        if now - last_attempt >= delay:
                            if retry_count < self.max_retries:
                                retry_candidates.append(assembly_id)
                            else:
                                # Max retries exceeded - log and remove
                                logger.error(
                                    f"Assembly {assembly_id} failed to synchronize after {retry_count} attempts, "
                                    f"giving up. Consider manual repair."
                                )
                                del self.pending_updates[assembly_id]
                                del self.retry_counts[assembly_id]
                                del self.last_retry_attempt[assembly_id]
                
                # Process retry candidates
                for assembly_id in retry_candidates:
                    await self._process_retry(assembly_id)
                    
                # Sleep a bit before checking again
                await asyncio.sleep(5.0)
                
        except asyncio.CancelledError:
            logger.debug("Assembly sync retry task cancelled")
        except Exception as e:
            logger.error(f"Error in assembly sync retry loop: {str(e)}", exc_info=True)
            self._is_running = False
    
    async def _process_retry(self, assembly_id: str) -> None:
        """Process a retry for a specific assembly.
        
        Args:
            assembly_id: ID of the assembly to retry synchronization
        """
        try:
            # Find the assembly in memory or storage
            if hasattr(self, "memory_manager") and self.memory_manager:
                assembly = await self.memory_manager.get_assembly_by_id(assembly_id)
            else:
                logger.warning(f"Cannot retry assembly {assembly_id}: No memory_manager available")
                return
                
            if not assembly:
                logger.warning(f"Assembly {assembly_id} not found for retry, removing from queue")
                async with self.update_lock:
                    if assembly_id in self.pending_updates:
                        del self.pending_updates[assembly_id]
                        del self.retry_counts[assembly_id]
                        del self.last_retry_attempt[assembly_id]
                        await self._save_pending_updates()
                return
                
            # Attempt synchronization
            async with self.update_lock:
                self.total_retries += 1
                self.retry_counts[assembly_id] += 1
                self.last_retry_attempt[assembly_id] = time.time()
                current_retry = self.retry_counts[assembly_id]
                
            logger.debug(f"Retry #{current_retry} for assembly {assembly_id}")
            success = await assembly.update_vector_index_async(self.vector_index)
            
            # Handle result
            async with self.update_lock:
                if success:
                    self.total_sync_successes += 1
                    logger.info(f"Retry #{current_retry} succeeded for assembly {assembly_id}")
                    del self.pending_updates[assembly_id]
                    del self.retry_counts[assembly_id]
                    del self.last_retry_attempt[assembly_id]
                else:
                    self.total_sync_failures += 1
                    logger.warning(f"Retry #{current_retry} failed for assembly {assembly_id}")
                    
                # Save updated state
                await self._save_pending_updates()
                
        except Exception as e:
            logger.error(f"Error processing retry for assembly {assembly_id}: {str(e)}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about assembly synchronization and retry queue."""
        return {
            "pending_updates_count": len(self.pending_updates),
            "total_sync_attempts": self.total_sync_attempts,
            "total_sync_successes": self.total_sync_successes,
            "total_sync_failures": self.total_sync_failures,
            "total_retries": self.total_retries,
            "is_retry_task_running": self._is_running,
            "pending_assemblies": [
                {
                    "assembly_id": assembly_id,
                    "name": info.get("name", assembly_id),
                    "retry_count": self.retry_counts.get(assembly_id, 0),
                    "queued_at": info.get("queued_at"),
                    "last_retry": datetime.fromtimestamp(self.last_retry_attempt.get(assembly_id, 0), tz=timezone.utc).isoformat() if assembly_id in self.last_retry_attempt else None
                }
                for assembly_id, info in self.pending_updates.items()
            ]
        }
