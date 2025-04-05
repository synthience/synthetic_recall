"""Assembly merge explanation module.

This module provides functionality to explain how an assembly was formed
through a merge operation, leveraging the MergeTracker's append-only log.
"""

import logging
from typing import Any, Dict, List, Optional
from ..memory_structures import MemoryAssembly 
from synthians_memory_core.memory_persistence import MemoryPersistence
from synthians_memory_core.custom_logger import get_logger
from synthians_memory_core.explainability._explain_helpers import (
    safe_load_assembly,
    get_assembly_names,
    get_timestamp_now
)
from ..geometry_manager import GeometryManager 

logger = get_logger(__name__)

async def generate_merge_explanation(
    assembly_id: str,
    merge_tracker,  # MergeTracker instance (will be fully typed once implemented)
    persistence: MemoryPersistence,
    geometry_manager: GeometryManager 
) -> Dict[str, Any]:
    """Generate an explanation for how an assembly was formed through a merge.
    
    Args:
        assembly_id: ID of the assembly to explain
        merge_tracker: MergeTracker instance for querying merge events
        persistence: MemoryPersistence instance for loading assembly data
        geometry_manager: GeometryManager instance
        
    Returns:
        Dictionary with the merge explanation (matches ExplainMergeData/Empty models)
    """
    logger.debug("MergeExplainer", "Generating merge explanation", {"assembly_id": assembly_id})
    
    # Define empty result structure
    empty_result = {
        "target_assembly_id": assembly_id,
        "notes": "Assembly was not formed by a merge or could not retrieve merge information."
    }
    
    # Load the target assembly
    assembly, error = await safe_load_assembly(assembly_id, persistence, geometry_manager)
    if error:
        empty_result["notes"] = f"Could not load target assembly: {error}"
        return empty_result
    
    # Check if this assembly was formed by a merge
    merged_from = getattr(assembly, "merged_from", None)
    if not merged_from or not isinstance(merged_from, list) or len(merged_from) == 0:
        empty_result["notes"] = "Assembly was not formed by a merge."
        return empty_result
        
    # Find the merge creation event in the log
    try:
        # Query the log for the creation event where this assembly is the target
        merge_creation_events = await merge_tracker.find_merge_creation_events(target_assembly_id=assembly_id)
        
        if not merge_creation_events:
            empty_result["notes"] = f"No merge creation event found for assembly {assembly_id} in the log."
            return empty_result
            
        # Get the most recent merge creation event (should typically be only one)
        creation_event = merge_creation_events[0]
        merge_event_id = creation_event.get("merge_event_id")
        
        # Find the latest cleanup status update for this merge event
        status_updates = await merge_tracker.find_cleanup_status_updates(merge_event_id)
        latest_status = status_updates[0] if status_updates else None
        
        # Get the names of source assemblies
        source_ids = merged_from
        source_names_dict = await get_assembly_names(source_ids, persistence, geometry_manager)
        source_names = [source_names_dict.get(sid) for sid in source_ids]
        
        # Build the reconciled cleanup details
        cleanup_status = "pending"
        cleanup_details = {}
        
        if latest_status:
            cleanup_status = latest_status.get("new_status", "pending")
            cleanup_details = {
                "timestamp": latest_status.get("update_timestamp"),
                "error": latest_status.get("error")
            }
            
        # Create the detailed result
        result = {
            "target_assembly_id": assembly_id,
            "merge_event_id": merge_event_id,
            "merge_timestamp": creation_event.get("timestamp"),
            "source_assembly_ids": source_ids,
            "source_assembly_names": source_names,
            "similarity_at_merge": creation_event.get("similarity_at_merge"),
            "threshold_at_merge": creation_event.get("merge_threshold"),
            "reconciled_cleanup_status": cleanup_status,
            "cleanup_details": cleanup_details,
            "notes": None
        }
        
        logger.debug("MergeExplainer", "Generated merge explanation", {
            "assembly_id": assembly_id,
            "merge_event_id": merge_event_id,
            "cleanup_status": cleanup_status
        })
        
        return result
        
    except Exception as e:
        logger.error("MergeExplainer", f"Error generating merge explanation for {assembly_id}", 
                     {"error": str(e)}, exc_info=True)
        empty_result["notes"] = f"Error retrieving merge information: {str(e)}"
        return empty_result
