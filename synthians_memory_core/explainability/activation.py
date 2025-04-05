"""Assembly activation explanation module.

This module provides functionality to explain why a memory was or wasn't
activated as part of an assembly during retrieval operations.
"""

import logging
from typing import Any, Dict, Optional
from ..memory_structures import MemoryEntry

from synthians_memory_core.memory_persistence import MemoryPersistence
from synthians_memory_core.geometry_manager import GeometryManager
from synthians_memory_core.custom_logger import get_logger
from synthians_memory_core.explainability._explain_helpers import (
    safe_load_assembly,
    calculate_similarity,
    get_timestamp_now,
    get_simplified_assembly_state
)

logger = get_logger(__name__)

async def generate_activation_explanation(
    assembly_id: str,
    memory_id: str,
    trigger_context: Optional[str],
    persistence: MemoryPersistence,
    geometry_manager: GeometryManager,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate an explanation for why a memory was (or wasn't) activated in an assembly.
    
    Args:
        assembly_id: ID of the assembly to explain
        memory_id: ID of the memory being checked
        trigger_context: Context of what triggered the activation check (e.g., retrieval_query:xyz)
        persistence: MemoryPersistence instance
        geometry_manager: GeometryManager instance
        config: Memory Core configuration dictionary
        
    Returns:
        Dictionary with the activation explanation (matches ExplainActivationData/Empty models)
    """
    logger.debug("ActivationExplainer", "Generating activation explanation", {
        "assembly_id": assembly_id,
        "memory_id": memory_id,
        "trigger_context": trigger_context
    })
    
    # Structure for empty explanation
    empty_result = {
        "assembly_id": assembly_id,
        "target_assembly_id": assembly_id,
        "memory_id": memory_id,
        "check_timestamp": get_timestamp_now(),
        "trigger_context": trigger_context,
        "assembly_state_before_check": None,
        "calculated_similarity": None,
        "activation_threshold": None,
        "passed_threshold": None,
        "notes": None  # Will be populated with error message if needed
    }
    
    # Load assembly
    # Pass geometry_manager to safe_load_assembly
    assembly, assembly_error = await safe_load_assembly(assembly_id, persistence, geometry_manager)
    if assembly_error:
        empty_result["notes"] = assembly_error
        return empty_result
    
    # Find the specific memory within the loaded assembly
    memory = None
    if hasattr(assembly, 'memories') and assembly.memories:
        logger.debug(f"[ActivationExplainer] Searching for memory ID '{memory_id}' in assembly '{assembly_id}' containing {len(assembly.memories)} memories.")
        for mem in assembly.memories:
            if isinstance(mem, MemoryEntry):
                logger.debug(f"[ActivationExplainer] Checking loaded memory with ID: {mem.id}")
            else:
                logger.warning(f"[ActivationExplainer] Found non-MemoryEntry item in assembly.memories: {type(mem)}")
                continue # Skip non-MemoryEntry items
            
            if mem.id == memory_id:
                memory = mem
                logger.debug(f"[ActivationExplainer] Found matching memory: {mem.id}")
                break # Found it
    
    if memory is None:
        empty_result["notes"] = f"Memory '{memory_id}' not found within Assembly '{assembly_id}'"
        empty_result["assembly_state_before_check"] = get_simplified_assembly_state(assembly)
        logger.warning("ActivationExplainer", empty_result["notes"], {"assembly_id": assembly_id, "memory_id": memory_id})
        return empty_result
    
    # Ensure memory has an embedding
    if not hasattr(memory, 'embedding') or memory.embedding is None:
        empty_result["notes"] = f"Memory '{memory_id}' found but has no embedding."
        logger.warning("ActivationExplainer", empty_result["notes"], {"assembly_id": assembly_id, "memory_id": memory_id})
        return empty_result
        
    # Get activation threshold from config
    try:
        threshold = config.get("assembly_activation_threshold", 0.65)  # Default if not found
    except Exception as e:
        logger.warning("ActivationExplainer", "Error retrieving threshold from config", {"error": str(e)})
        threshold = 0.65  # Default fallback
    
    # Get simplified assembly state
    assembly_state = get_simplified_assembly_state(assembly)
    
    # Calculate similarity
    similarity, error = await calculate_similarity(memory, assembly, geometry_manager)
    
    # Prepare explanation result
    if error:
        return {
            "assembly_id": assembly_id,
            "target_assembly_id": assembly_id,
            "memory_id": memory_id,
            "check_timestamp": get_timestamp_now(),
            "trigger_context": trigger_context,
            "assembly_state_before_check": assembly_state,
            "calculated_similarity": None,
            "activation_threshold": threshold,
            "passed_threshold": False,
            "notes": f"Could not calculate similarity: {error}"
        }
    
    # Determine if passed threshold
    passed = similarity is not None and similarity >= threshold
    
    # Create explanation
    result = {
        "assembly_id": assembly_id,
        "target_assembly_id": assembly_id,
        "memory_id": memory_id,
        "check_timestamp": get_timestamp_now(),
        "trigger_context": trigger_context,
        "assembly_state_before_check": assembly_state,
        "calculated_similarity": similarity,
        "activation_threshold": threshold,
        "passed_threshold": passed,
        "notes": f"Similarity {'≥' if passed else '<'} threshold"
    }
    
    logger.debug("ActivationExplainer", "Generated activation explanation", {
        "assembly_id": assembly_id, 
        "memory_id": memory_id,
        "similarity": similarity,
        "threshold": threshold,
        "passed": passed
    })
    
    return result
