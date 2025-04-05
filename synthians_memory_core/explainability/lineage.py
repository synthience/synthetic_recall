"""Assembly lineage tracing module.

This module provides functionality to trace the ancestry of an assembly through its merge history.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from synthians_memory_core.memory_persistence import MemoryPersistence
from synthians_memory_core.memory_structures import MemoryAssembly
from synthians_memory_core.custom_logger import get_logger
from synthians_memory_core.explainability._explain_helpers import (
    safe_load_assembly,
    get_timestamp_now
)
from synthians_memory_core.geometry_manager import GeometryManager

logger = get_logger(__name__)

async def trace_lineage(
    assembly_id: str,
    persistence: MemoryPersistence,
    geometry_manager: GeometryManager,
    max_depth: int = 10
) -> List[Dict[str, Any]]:
    """Trace the lineage of an assembly through its merge history.
    
    Args:
        assembly_id: ID of the assembly to trace lineage for
        persistence: MemoryPersistence instance for loading assemblies
        geometry_manager: GeometryManager instance for spatial queries
        max_depth: Maximum depth to trace (prevents unbounded recursion)
        
    Returns:
        List of dictionaries with lineage entries (matches LineageEntry model)
    """
    logger.debug("LineageTracer", "Tracing assembly lineage", {
        "assembly_id": assembly_id,
        "max_depth": max_depth
    })
    
    # Track visited nodes to detect cycles
    visited: Set[str] = set()
    
    # List to collect all lineage entries
    lineage_entries: List[Dict[str, Any]] = []
    
    # Flag to track if max depth was reached
    max_depth_reached = False
    cycles_detected = False
    
    async def _trace_recursively(current_id: str, depth: int) -> None:
        """Recursively trace the lineage starting from the current assembly."""
        nonlocal max_depth_reached, cycles_detected
        
        # Stop if we've reached maximum depth
        if depth > max_depth:
            max_depth_reached = True
            lineage_entries.append({
                "assembly_id": current_id,
                "name": None,  # Name is not fetched for depth-limited entries
                "depth": depth,
                "status": "depth_limit_reached",
                "created_at": None,
                "memory_count": None
            })
            return
        
        # Check for cycles
        if current_id in visited:
            cycles_detected = True
            lineage_entries.append({
                "assembly_id": current_id,
                "name": None,  # Name is not re-fetched for cycle entries
                "depth": depth,
                "status": "cycle_detected",
                "created_at": None,
                "memory_count": None
            })
            return
        
        # Mark as visited to detect cycles
        visited.add(current_id)
        
        # Load the assembly
        assembly, error = await safe_load_assembly(current_id, persistence, geometry_manager)
        
        if error or not assembly:
            lineage_entries.append({
                "assembly_id": current_id,
                "name": None,
                "depth": depth,
                "status": "not_found",
                "created_at": None,
                "memory_count": None
            })
            return
        
        # Extract the necessary information
        status = "origin"
        merged_from = getattr(assembly, "merged_from", None)
        if merged_from and isinstance(merged_from, list) and len(merged_from) > 0:
            status = "merged"
        
        # Add to lineage entries
        lineage_entries.append({
            "assembly_id": current_id,
            "name": getattr(assembly, "name", None),
            "depth": depth,
            "status": status,
            "created_at": getattr(assembly, "created_at", None),
            "memory_count": len(getattr(assembly, "memory_ids", [])) if hasattr(assembly, "memory_ids") else None
        })
        
        # Recursively trace parent assemblies if this is a merged assembly
        if merged_from and isinstance(merged_from, list):
            for parent_id in merged_from:
                await _trace_recursively(parent_id, depth + 1)
    
    # Start tracing from the target assembly
    await _trace_recursively(assembly_id, 0)
    
    # Sort entries by depth for consistent output
    lineage_entries.sort(key=lambda x: x["depth"])
    
    logger.debug("LineageTracer", "Completed lineage trace", {
        "assembly_id": assembly_id,
        "entries_count": len(lineage_entries),
        "max_depth_reached": max_depth_reached,
        "cycles_detected": cycles_detected
    })
    
    return lineage_entries
