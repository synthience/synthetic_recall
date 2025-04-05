"""MergeTracker implementation for Memory Core Phase 5.9.

This module implements an append-only event logging strategy for tracking merge operations
and their cleanup status, avoiding risky file rewrites.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiofiles

from synthians_memory_core.custom_logger import get_logger

logger = get_logger(__name__)

class MergeTracker:
    """Tracks and logs assembly merge operations for historical analysis and debugging.
    
    This class implements an append-only strategy for merge event logging, where each
    significant event (merge creation, cleanup status change) is recorded as a separate
    entry in the log file.
    """
    
    def __init__(self, log_path: str, max_entries: int = 1000, max_size_mb: int = 100):
        """Initialize the MergeTracker.
        
        Args:
            log_path: Path to the merge log file
            max_entries: Maximum number of entries to keep in the log before rotation
            max_size_mb: Maximum file size in MB before rotation
        """
        self.log_path = log_path
        self.max_entries = max_entries
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        logger.info("MergeTracker", "Initialized", {
            "log_path": log_path,
            "max_entries": max_entries,
            "max_size_mb": max_size_mb
        })
    
    async def initialize(self) -> bool:
        """
        Initialize the MergeTracker, ensuring log directory exists and creating log file if needed.
        Returns True if initialization was successful, False otherwise.
        """
        try:
            log_dir = os.path.dirname(self.log_path)
            os.makedirs(log_dir, exist_ok=True)
            
            # Create an empty log file if it doesn't exist
            if not os.path.exists(self.log_path):
                async with aiofiles.open(self.log_path, "w") as f:
                    await f.write("")
                    
            logger.info("MergeTracker", "Initialized successfully", {"log_path": self.log_path})
            return True
        except Exception as e:
            logger.error("MergeTracker", "Initialization failed", {"error": str(e)})
            return False
    
    async def log_merge_creation_event(
        self,
        source_assembly_ids: List[str],
        target_assembly_id: str,
        similarity_at_merge: float,
        merge_threshold: float
    ) -> str:
        """Log a merge creation event to the append-only log.
        
        Args:
            source_assembly_ids: List of source assembly IDs involved in the merge
            target_assembly_id: ID of the assembly created by the merge
            similarity_at_merge: Similarity score that triggered the merge
            merge_threshold: Threshold used for the merge decision
            
        Returns:
            The generated merge_event_id for referencing in cleanup status updates
        """
        merge_event_id = f"merge_{uuid.uuid4()}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        event = {
            "event_type": "merge_creation",
            "merge_event_id": merge_event_id,
            "timestamp": timestamp,
            "source_assembly_ids": source_assembly_ids,
            "target_assembly_id": target_assembly_id,
            "similarity_at_merge": similarity_at_merge,
            "merge_threshold": merge_threshold
        }
        
        await self._append_event_to_log(event)
        
        logger.info("MergeTracker", "Logged merge creation event", {
            "merge_event_id": merge_event_id,
            "target_assembly_id": target_assembly_id
        })
        
        return merge_event_id
    
    async def log_cleanup_status_event(
        self, 
        merge_event_id: str, 
        new_status: str,
        error: Optional[str] = None
    ) -> None:
        """Log a cleanup status update event to the append-only log.
        
        Args:
            merge_event_id: ID of the original merge creation event to update
            new_status: New cleanup status ("completed" or "failed")
            error: Optional error details if the status is "failed"
        """
        if new_status not in ["completed", "failed"]:
            logger.warning("MergeTracker", f"Invalid cleanup status: {new_status}", {
                "merge_event_id": merge_event_id
            })
            return
        
        update_timestamp = datetime.now(timezone.utc).isoformat()
        
        event = {
            "event_type": "cleanup_status_update",
            "update_timestamp": update_timestamp,
            "target_merge_event_id": merge_event_id,
            "new_status": new_status,
            "error": error
        }
        
        await self._append_event_to_log(event)
        
        logger.info("MergeTracker", "Logged cleanup status update", {
            "merge_event_id": merge_event_id,
            "new_status": new_status,
            "has_error": error is not None
        })
    
    async def _append_event_to_log(self, event: Dict[str, Any]) -> None:
        """Append an event to the log file and handle rotation if needed.
        
        Args:
            event: The event to log
        """
        # Check if rotation is needed based on file size
        await self._check_and_rotate_log()
        
        # Append the event to the log file
        try:
            async with aiofiles.open(self.log_path, "a") as f:
                serialized = json.dumps(event)
                await f.write(serialized + "\n")
        except Exception as e:
            logger.error("MergeTracker", "Failed to write event to log", {
                "error": str(e),
                "event_type": event.get("event_type")
            }, exc_info=True)
            raise
    
    async def _check_and_rotate_log(self) -> None:
        """Check if log rotation is needed and perform it if necessary."""
        try:
            # Check file size
            if os.path.exists(self.log_path):
                size = os.path.getsize(self.log_path)
                if size >= self.max_size_bytes:
                    await self._rotate_log("size")
                    return
                
            # Check line count
            if os.path.exists(self.log_path):
                line_count = 0
                async with aiofiles.open(self.log_path, "r") as f:
                    async for _ in f:
                        line_count += 1
                
                if line_count >= self.max_entries:
                    await self._rotate_log("entry_count")
        except Exception as e:
            logger.error("MergeTracker", "Error checking for log rotation", {
                "error": str(e)
            }, exc_info=True)
    
    async def _rotate_log(self, reason: str) -> None:
        """Rotate the log file using an atomic approach.
        
        Args:
            reason: The reason for rotation ("size" or "entry_count")
        """
        if not os.path.exists(self.log_path):
            return
        
        try:
            # Generate a timestamped backup filename
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.log_path}.{timestamp}.bak"
            
            # Rename the current log file to the backup
            os.rename(self.log_path, backup_path)
            
            logger.info("MergeTracker", "Rotated merge log", {
                "reason": reason,
                "old_path": self.log_path,
                "backup_path": backup_path
            })
            
            # Create a new empty log file
            async with aiofiles.open(self.log_path, "w") as _:
                pass
        except Exception as e:
            logger.error("MergeTracker", "Failed to rotate log", {
                "error": str(e)
            }, exc_info=True)
            # Ensure the log file exists even if rotation failed
            if not os.path.exists(self.log_path):
                async with aiofiles.open(self.log_path, "w") as _:
                    pass
    
    async def read_log_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Read recent log entries from the file.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of raw log entries (not yet reconciled)
        """
        if not os.path.exists(self.log_path):
            return []
        
        try:
            entries = []
            async with aiofiles.open(self.log_path, "r") as f:
                async for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError:
                            logger.warning("MergeTracker", "Invalid JSON in log file", {"line": line[:100]})
            
            # Return the most recent entries first
            return entries[-limit:] if len(entries) > limit else entries
        except Exception as e:
            logger.error("MergeTracker", "Error reading log entries", {
                "error": str(e)
            }, exc_info=True)
            return []
    
    async def find_merge_creation_events(
        self, 
        target_assembly_id: Optional[str] = None,
        limit: int = 1
    ) -> List[Dict[str, Any]]:
        """Find merge creation events that match the specified criteria.
        
        Args:
            target_assembly_id: Optional filter for the target assembly ID
            limit: Maximum number of matching events to return
            
        Returns:
            List of matching merge creation events, newest first
        """
        all_entries = await self.read_log_entries(1000)  # Read a larger batch to filter
        
        # Filter for merge creation events
        creation_events = [e for e in all_entries if e.get("event_type") == "merge_creation"]
        
        # Apply target assembly filter if specified
        if target_assembly_id:
            creation_events = [e for e in creation_events 
                              if e.get("target_assembly_id") == target_assembly_id]
        
        # Sort by timestamp, newest first
        creation_events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        
        return creation_events[:limit]
    
    async def find_cleanup_status_updates(
        self, 
        merge_event_id: str
    ) -> List[Dict[str, Any]]:
        """Find cleanup status update events for a specific merge event ID.
        
        Args:
            merge_event_id: ID of the merge creation event to find updates for
            
        Returns:
            List of matching status update events, newest first
        """
        all_entries = await self.read_log_entries(1000)  # Read a larger batch to filter
        
        # Filter for status update events matching the target merge event ID
        status_updates = [
            e for e in all_entries 
            if e.get("event_type") == "cleanup_status_update" and 
               e.get("target_merge_event_id") == merge_event_id
        ]
        
        # Sort by timestamp, newest first
        status_updates.sort(key=lambda e: e.get("update_timestamp", ""), reverse=True)
        
        return status_updates
    
    async def reconcile_merge_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get a reconciled view of merge events with their latest status.
        
        This combines information from merge creation events with their
        corresponding latest cleanup status updates.
        
        Args:
            limit: Maximum number of reconciled events to return
            
        Returns:
            List of reconciled merge log entries matching the ReconciledMergeLogEntry model
        """
        # Find the most recent merge creation events
        creation_events = await self.find_merge_creation_events(limit=limit)
        
        reconciled_entries = []
        for creation in creation_events:
            merge_event_id = creation.get("merge_event_id")
            
            # Find the latest status update for this merge event
            status_updates = await self.find_cleanup_status_updates(merge_event_id)
            latest_status = status_updates[0] if status_updates else None
            
            # Determine the final cleanup status
            final_status = "pending"
            cleanup_timestamp = None
            cleanup_error = None
            
            if latest_status:
                final_status = latest_status.get("new_status", "pending")
                cleanup_timestamp = latest_status.get("update_timestamp")
                cleanup_error = latest_status.get("error")
            
            # Create the reconciled entry
            reconciled = {
                "merge_event_id": merge_event_id,
                "creation_timestamp": creation.get("timestamp"),
                "source_assembly_ids": creation.get("source_assembly_ids", []),
                "target_assembly_id": creation.get("target_assembly_id"),
                "similarity_at_merge": creation.get("similarity_at_merge"),
                "merge_threshold": creation.get("merge_threshold"),
                "final_cleanup_status": final_status,
                "cleanup_timestamp": cleanup_timestamp,
                "cleanup_error": cleanup_error
            }
            
            reconciled_entries.append(reconciled)
        
        return reconciled_entries
