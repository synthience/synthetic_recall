"""Helper functions for explainability module components.

This module provides common utilities used by the different explainability
components for loading data, performing calculations, and formatting responses.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from synthians_memory_core.memory_persistence import MemoryPersistence
from synthians_memory_core.memory_structures import MemoryAssembly, MemoryEntry
from synthians_memory_core.geometry_manager import GeometryManager
from synthians_memory_core.custom_logger import get_logger

logger = get_logger(__name__)

async def safe_load_assembly(
    assembly_id: str, 
    persistence: MemoryPersistence,
    geometry_manager: GeometryManager
) -> Tuple[Optional[MemoryAssembly], Optional[str]]:
    """Safely load an assembly with error handling.
    
    Args:
        assembly_id: ID of the assembly to load
        persistence: MemoryPersistence instance
        geometry_manager: GeometryManager instance
        
    Returns:
        Tuple of (assembly object or None, error message or None)
    """
    try:
        assembly = await persistence.load_assembly(assembly_id, geometry_manager)
        if not assembly:
            return None, f"Assembly '{assembly_id}' not found"
        return assembly, None
    except Exception as e:
        logger.error("explainability", f"Error loading assembly {assembly_id}", {"error": str(e)}, exc_info=True)
        return None, f"Error loading assembly: {str(e)}"

async def safe_load_memory(
    memory_id: str, 
    persistence: MemoryPersistence
) -> Tuple[Optional[MemoryEntry], Optional[str]]:
    """Safely load a memory entry with error handling.
    
    Args:
        memory_id: ID of the memory to load
        persistence: MemoryPersistence instance
        
    Returns:
        Tuple of (memory object or None, error message or None)
    """
    try:
        memory = await persistence.load_memory(memory_id)
        if not memory:
            return None, f"Memory '{memory_id}' not found"
        return memory, None
    except Exception as e:
        logger.error("explainability", f"Error loading memory {memory_id}", {"error": str(e)}, exc_info=True)
        return None, f"Error loading memory: {str(e)}"

async def get_assembly_names(
    assembly_ids: List[str],
    persistence: MemoryPersistence,
    geometry_manager: GeometryManager
) -> Dict[str, Optional[str]]:
    """Get names for multiple assemblies.
    
    Args:
        assembly_ids: List of assembly IDs
        persistence: MemoryPersistence instance
        geometry_manager: GeometryManager instance
        
    Returns:
        Dictionary mapping assembly IDs to their names (or None if not found)
    """
    result = {}
    for asm_id in assembly_ids:
        try:
            assembly, _ = await safe_load_assembly(asm_id, persistence, geometry_manager)
            result[asm_id] = assembly.name if assembly else None
        except Exception as e:
            logger.warning("explainability", f"Error fetching assembly name for {asm_id}", {"error": str(e)})
            result[asm_id] = None
    return result

async def calculate_similarity(
    memory: MemoryEntry,
    assembly: MemoryAssembly,
    geometry_manager: GeometryManager
) -> Tuple[Optional[float], Optional[str]]:
    """Calculate similarity between a memory and an assembly.
    
    Args:
        memory: Memory entry
        assembly: Memory assembly
        geometry_manager: GeometryManager instance
        
    Returns:
        Tuple of (similarity score or None, error message or None)
    """
    try:
        # Validate embeddings: Check for None explicitly
        # Use assembly.composite_embedding
        if memory.embedding is None or assembly.composite_embedding is None:
            logger.warning(
                "explainability",
                f"Missing embedding for memory {memory.id} or assembly {assembly.assembly_id}"
            )
            return None, "Missing embeddings"

        # Calculate similarity
        # Use assembly.composite_embedding
        # REMOVED await as calculate_similarity is synchronous
        similarity = geometry_manager.calculate_similarity(memory.embedding, assembly.composite_embedding)
        return similarity, None
    except Exception as e:
        logger.error(
            "explainability",
            # CORRECTED: Use assembly.assembly_id
            f"Error calculating similarity between memory {memory.id} and assembly {assembly.assembly_id}",
            {"error": str(e)},
            exc_info=True
        )
        return None, f"Error calculating similarity: {str(e)}"

def get_timestamp_now() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"

def get_simplified_assembly_state(assembly: MemoryAssembly) -> Dict[str, Any]:
    """Get a simplified state representation of an assembly.
    
    Args:
        assembly: Memory assembly
        
    Returns:
        Dictionary with simplified state
    """
    return {
        # CORRECTED: Use 'memories' attribute instead of 'memory_ids'
        "memory_count": len(assembly.memories) if assembly.memories else 0,
        "last_activation_level": getattr(assembly, "last_activation_level", None),
        "is_merged": bool(getattr(assembly, "merged_from", None)),
        "vector_index_updated": bool(getattr(assembly, "vector_index_updated_at", None)),
        "created_at": getattr(assembly, "created_at", None),
        "last_updated": getattr(assembly, "last_updated", None)
    }
