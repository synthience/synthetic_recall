"""FastAPI routes for the explainability features of Memory Core Phase 5.9.

These routes expose the explainability functions via REST API endpoints.
"""

import logging
from typing import Any, Dict, List, Optional
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from synthians_memory_core.explainability.activation import generate_activation_explanation
from synthians_memory_core.explainability.merge import generate_merge_explanation
from synthians_memory_core.explainability.lineage import trace_lineage
from synthians_memory_core.custom_logger import get_logger

logger = get_logger(__name__)

# Define Pydantic response models as per phase_5_9_models.md

# Activation explanation models
class ExplainActivationData(BaseModel):
    assembly_id: str = Field(..., description="ID of the assembly being explained")
    memory_id: Optional[str] = Field(None, description="ID of the specific memory being checked (if provided)")
    check_timestamp: str = Field(..., description="ISO format timestamp of when this explanation was generated")
    trigger_context: Optional[str] = Field(None, description="Context of the activation check (e.g., 'retrieval_query:abc', 'assembly_update')")
    assembly_state_before_check: Optional[Dict[str, Any]] = Field(None, description="Simplified state of the assembly before check")
    calculated_similarity: Optional[float] = Field(None, description="Calculated similarity score between memory and assembly")
    activation_threshold: Optional[float] = Field(None, description="Activation threshold used for the decision")
    passed_threshold: Optional[bool] = Field(None, description="Whether the similarity met or exceeded the threshold")
    notes: Optional[str] = Field(None, description="Additional explanation notes")

class ExplainActivationEmpty(BaseModel):
    assembly_id: str = Field(..., description="ID of the assembly being explained")
    memory_id: Optional[str] = Field(None, description="ID of the specific memory being checked (if provided)")
    notes: str = Field(..., description="Explanation for why no detailed data is available")

class ExplainActivationResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    explanation: Dict[str, Any] = Field(..., description="Explanation details (either ExplainActivationData or ExplainActivationEmpty)")
    error: Optional[str] = Field(None, description="Error message if success is False")

# Merge explanation models
class ExplainMergeData(BaseModel):
    target_assembly_id: str = Field(..., description="ID of the assembly created by the merge")
    merge_event_id: Optional[str] = Field(None, description="ID of the merge_creation event in the log")
    merge_timestamp: Optional[str] = Field(None, description="ISO format timestamp of when the merge occurred")
    source_assembly_ids: List[str] = Field([], description="IDs of the source assemblies that were merged")
    source_assembly_names: Optional[List[str]] = Field(None, description="Names of the source assemblies (if available)")
    similarity_at_merge: Optional[float] = Field(None, description="Similarity score that triggered the merge")
    threshold_at_merge: Optional[float] = Field(None, description="Threshold used for the merge decision")
    reconciled_cleanup_status: Optional[str] = Field(None, description="Final cleanup status ('pending', 'completed', 'failed')")
    cleanup_details: Optional[Dict[str, Any]] = Field(None, description="Details about the cleanup status")
    notes: Optional[str] = Field(None, description="Additional explanation notes")

class ExplainMergeEmpty(BaseModel):
    target_assembly_id: str = Field(..., description="ID of the assembly checked")
    notes: str = Field("Assembly was not formed by a merge.", description="Explanation for non-merged assemblies")

class ExplainMergeResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    explanation: Dict[str, Any] = Field(..., description="Explanation details (either ExplainMergeData or ExplainMergeEmpty)")
    error: Optional[str] = Field(None, description="Error message if success is False")

# Lineage models
class LineageEntry(BaseModel):
    assembly_id: str = Field(..., description="ID of the assembly in the lineage")
    name: Optional[str] = Field(None, description="Name of the assembly")
    depth: int = Field(..., description="Depth in the lineage tree (0 = target assembly)")
    status: Optional[str] = Field(None, description="Status of this entry in the trace")
    created_at: Optional[str] = Field(None, description="ISO timestamp when this specific assembly was created")
    memory_count: Optional[int] = Field(None, description="Number of memories in this assembly")

class LineageResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    target_assembly_id: str = Field(..., description="The ID of the assembly whose lineage was traced")
    lineage: List[Dict[str, Any]] = Field([], description="List of assemblies in the lineage")
    max_depth_reached: bool = Field(False, description="Whether the tracing stopped due to reaching the max_depth limit")
    cycles_detected: bool = Field(False, description="Whether any cycles were detected during tracing")
    error: Optional[str] = Field(None, description="Error message if success is False")

# Create the router
router = APIRouter(prefix="/assemblies", tags=["explainability"])

# Simple TTL cache for lineage responses (5 minutes TTL)
@lru_cache(maxsize=100)
def get_lineage_cache_key(assembly_id: str, max_depth: int) -> str:
    """Generate a cache key for lineage requests."""
    return f"{assembly_id}_{max_depth}"

lineage_cache: Dict[str, tuple] = {}  # (response, timestamp)
LINEAGE_CACHE_TTL_SECONDS = 300  # 5 minutes

# Helper to check if explainability is enabled
def check_explainability_enabled(request: Request) -> bool:
    """Check if the explainability feature is enabled in the config."""
    memory_core = request.app.state.memory_core
    if not memory_core or not memory_core.config.get("ENABLE_EXPLAINABILITY", False):
        raise HTTPException(
            status_code=403,
            detail="Explainability features are disabled in the configuration."
        )
    return True

@router.get("/{assembly_id}/explain_activation", response_model=ExplainActivationResponse)
async def explain_activation(
    request: Request,
    assembly_id: str = Path(..., description="ID of the assembly to explain"),
    memory_id: str = Query(..., description="ID of the memory to check activation for"),
    trigger_context: Optional[str] = Query(None, description="Optional context that triggered the activation check")
):
    # Check if explainability is enabled
    check_explainability_enabled(request)
    
    memory_core = request.app.state.memory_core
    
    try:
        explanation = await generate_activation_explanation(
            assembly_id=assembly_id,
            memory_id=memory_id,
            trigger_context=trigger_context,
            persistence=memory_core.persistence,
            geometry_manager=memory_core.geometry_manager,
            config=memory_core.config
        )
        
        return {
            "success": True,
            "explanation": explanation,
            "error": None
        }
    except Exception as e:
        logger.error("API", f"Error generating activation explanation", {
            "assembly_id": assembly_id,
            "memory_id": memory_id,
            "error": str(e)
        }, exc_info=True)
        
        return {
            "success": False,
            "explanation": {
                "assembly_id": assembly_id,
                "memory_id": memory_id,
                "notes": f"Error generating explanation: {str(e)}"
            },
            "error": str(e)
        }

@router.get("/{assembly_id}/explain_merge", response_model=ExplainMergeResponse)
async def explain_merge(
    request: Request,
    assembly_id: str = Path(..., description="ID of the assembly to explain merge for")
):
    # Check if explainability is enabled
    check_explainability_enabled(request)
    
    memory_core = request.app.state.memory_core
    
    try:
        explanation = await generate_merge_explanation(
            assembly_id=assembly_id,
            merge_tracker=memory_core.merge_tracker,
            persistence=memory_core.persistence,
            geometry_manager=memory_core.geometry_manager
        )
        
        return {
            "success": True,
            "explanation": explanation,
            "error": None
        }
    except Exception as e:
        logger.error("API", f"Error generating merge explanation", {
            "assembly_id": assembly_id,
            "error": str(e)
        }, exc_info=True)
        
        return {
            "success": False,
            "explanation": {
                "target_assembly_id": assembly_id,
                "notes": f"Error generating explanation: {str(e)}"
            },
            "error": str(e)
        }

@router.get("/{assembly_id}/lineage", response_model=LineageResponse)
async def get_lineage(
    request: Request,
    assembly_id: str = Path(..., description="ID of the assembly to trace lineage for"),
    max_depth: int = Query(10, description="Maximum depth to trace (prevents unbounded recursion)")
):
    # Check if explainability is enabled
    check_explainability_enabled(request)
    
    memory_core = request.app.state.memory_core
    
    # Generate cache key
    cache_key = get_lineage_cache_key(assembly_id, max_depth)
    
    # Check cache first
    import time
    current_time = time.time()
    if cache_key in lineage_cache:
        cached_response, timestamp = lineage_cache[cache_key]
        if current_time - timestamp <= LINEAGE_CACHE_TTL_SECONDS:
            return cached_response
    
    try:
        lineage_entries = await trace_lineage(
            assembly_id=assembly_id,
            persistence=memory_core.persistence,
            geometry_manager=memory_core.geometry_manager,
            max_depth=max_depth
        )
        
        # Determine if max depth was reached or cycles were detected
        max_depth_reached = any(entry.get("status") == "depth_limit_reached" for entry in lineage_entries)
        cycles_detected = any(entry.get("status") == "cycle_detected" for entry in lineage_entries)
        
        response = {
            "success": True,
            "target_assembly_id": assembly_id,
            "lineage": lineage_entries,
            "max_depth_reached": max_depth_reached,
            "cycles_detected": cycles_detected,
            "error": None
        }
        
        # Cache the response
        lineage_cache[cache_key] = (response, current_time)
        
        return response
    except Exception as e:
        logger.error("API", f"Error tracing lineage", {
            "assembly_id": assembly_id,
            "error": str(e)
        }, exc_info=True)
        
        return {
            "success": False,
            "target_assembly_id": assembly_id,
            "lineage": [],
            "max_depth_reached": False,
            "cycles_detected": False,
            "error": str(e)
        }
