from fastapi import Depends, Request, HTTPException
from typing import Any, Dict

"""FastAPI routes for the diagnostics features of Memory Core Phase 5.9.

These routes expose diagnostics information such as merge logs and runtime configuration.
"""

import logging
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from synthians_memory_core.custom_logger import get_logger

logger = get_logger(__name__)

# Define Pydantic response models as per phase_5_9_models.md

# Merge log models
class ReconciledMergeLogEntry(BaseModel):
    merge_event_id: str = Field(..., description="Unique ID of the original merge creation event")
    creation_timestamp: str = Field(..., description="ISO timestamp when the merge was initiated")
    source_assembly_ids: List[str] = Field(..., description="IDs of the source assemblies involved")
    target_assembly_id: str = Field(..., description="ID of the assembly created by the merge")
    similarity_at_merge: Optional[float] = Field(None, description="Similarity score that triggered merge")
    merge_threshold: Optional[float] = Field(None, description="Threshold used for merge decision")
    final_cleanup_status: str = Field(..., description="The latest known cleanup status")
    cleanup_timestamp: Optional[str] = Field(None, description="ISO timestamp of the last cleanup status update")
    cleanup_error: Optional[str] = Field(None, description="Error details if the final cleanup status is 'failed'")

class MergeLogResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    reconciled_log_entries: List[ReconciledMergeLogEntry] = Field(..., description="List of recent, reconciled merge events")
    count: int = Field(..., description="Total number of reconciled merge creation events returned")
    query_limit: int = Field(..., description="The limit parameter used for the query")
    error: Optional[str] = Field(None, description="Error message if success is False")

# Runtime configuration models
class RuntimeConfigResponse(BaseModel):
    success: bool = Field(..., description="Whether the request succeeded")
    service: str = Field(..., description="Name of the service queried")
    config: Dict[str, Any] = Field(..., description="Dictionary containing only the sanitized configuration key-value pairs")
    retrieval_timestamp: str = Field(..., description="ISO timestamp when the configuration was retrieved")
    error: Optional[str] = Field(None, description="Error message if success is False")

# Create the router
router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])

# Define safe configuration keys for each service
SAFE_CONFIG_KEYS_MEMORY_CORE = [
    "embedding_dim", "geometry", "assembly_activation_threshold",
    "assembly_boost_mode", "assembly_boost_factor", "enable_explainability",
    "max_allowed_drift_seconds", "merge_log_max_entries", 
    "assembly_metrics_persist_interval", "assembly_sync_check_interval",
    "max_lineage_depth", "assembly_pruning_enabled", "assembly_pruning_interval"
]

SAFE_CONFIG_KEYS_NEURAL_MEMORY = [
    "window_size", "learning_rate", "surprise_threshold", "model_type",
    "embedding_dim", "batch_size", "momentum_decay", "momentum_window"
]

SAFE_CONFIG_KEYS_CCE = [
    "default_variant", "variant_selection_mode", "variant_selection_threshold",
    "llm_guidance_weight", "history_window_size", "guidance_integration_mode"
]

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

@router.get("/merge_log", response_model=MergeLogResponse)
async def get_merge_log(
    request: Request,
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of reconciled entries to return")
):
    # Check if explainability is enabled
    check_explainability_enabled(request)
    
    memory_core = request.app.state.memory_core
    
    try:
        # Get reconciled merge events using MergeTracker
        reconciled_entries = await memory_core.merge_tracker.reconcile_merge_events(limit=limit)
        
        return {
            "success": True,
            "reconciled_log_entries": reconciled_entries,
            "count": len(reconciled_entries),
            "query_limit": limit,
            "error": None
        }
    except Exception as e:
        logger.error("API", f"Error retrieving merge log", {
            "error": str(e),
            "limit": limit
        }, exc_info=True)
        
        return {
            "success": False,
            "reconciled_log_entries": [],
            "count": 0,
            "query_limit": limit,
            "error": str(e)
        }

@router.post("/trigger_retry_loop", status_code=200)
async def trigger_retry_loop_endpoint(
    request: Request,
    explainability_enabled: bool = Depends(check_explainability_enabled) # Ensure feature is enabled
):
    """Manually triggers the processing of the pending vector update queue."""
    # Access memory_core from app state directly
    if not hasattr(request.app.state, 'memory_core') or request.app.state.memory_core is None:
        raise HTTPException(status_code=503, detail="Memory core not available")
    memory_core = request.app.state.memory_core

    if not hasattr(memory_core, 'force_process_pending_updates'):
         raise HTTPException(status_code=501, detail="Retry loop trigger functionality not implemented in core.")

    try:
        results = await memory_core.force_process_pending_updates()
        return {
            "success": True,
            "message": f"Triggered processing of pending updates.",
            "details": results
        }
    except Exception as e:
        logger.error(f"Error during forced processing of pending updates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error during forced processing: {str(e)}")

@router.get("/runtime/config/{service_name}", response_model=RuntimeConfigResponse)
async def get_runtime_config(
    request: Request,
    service_name: str = Path(..., description="Name of the service to get config for (memory-core, neural-memory, cce)")
):
    # Check if explainability is enabled
    check_explainability_enabled(request)
    
    # Map service name to service instance and allowed keys
    service_map = {
        "memory-core": {
            "instance": getattr(request.app.state, "memory_core", None),
            "safe_keys": SAFE_CONFIG_KEYS_MEMORY_CORE
        },
        "neural-memory": {
            "instance": getattr(request.app.state, "neural_memory", None),
            "safe_keys": SAFE_CONFIG_KEYS_NEURAL_MEMORY
        },
        "cce": {
            "instance": getattr(request.app.state, "context_cascade_engine", None),
            "safe_keys": SAFE_CONFIG_KEYS_CCE
        }
    }
    
    # Check if the service exists
    if service_name not in service_map:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service_name}")
    
    service_info = service_map[service_name]
    service_instance = service_info["instance"]
    safe_keys = service_info["safe_keys"]
    
    if not service_instance:
        raise HTTPException(status_code=404, detail=f"Service {service_name} is not available")
    
    try:
        # Get full config and sanitize it
        full_config = getattr(service_instance, "config", {})
        
        # Create sanitized config with only safe keys
        sanitized_config = {k: v for k, v in full_config.items() if k in safe_keys}
        
        return {
            "success": True,
            "service": service_name,
            "config": sanitized_config,
            "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
            "error": None
        }
    except Exception as e:
        logger.error("API", f"Error retrieving runtime config", {
            "service": service_name,
            "error": str(e)
        }, exc_info=True)
        
        return {
            "success": False,
            "service": service_name,
            "config": {},
            "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }



@router.post("/trigger_retry_loop", status_code=200)
async def trigger_retry_loop_endpoint(
    request: Request,
    # memory_core: SynthiansMemoryCore = Depends(get_memory_core) # Use dependency injection if available
):
    """Manually triggers the processing of the pending vector update queue."""
    # Access memory_core from app state directly as Depends might be tricky with test client setup
    if not hasattr(request.app.state, 'memory_core') or request.app.state.memory_core is None:
        raise HTTPException(status_code=503, detail="Memory core not available")
    memory_core = request.app.state.memory_core

    if not hasattr(memory_core, 'force_process_pending_updates'):
         raise HTTPException(status_code=501, detail="Retry loop trigger functionality not implemented in core.")

    results = await memory_core.force_process_pending_updates()
    return {
        "success": True,
        "message": f"Triggered processing of pending updates.",
        "details": results
    }
