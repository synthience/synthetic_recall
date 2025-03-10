from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import uuid
from datetime import timedelta
import logging

# Router for dream parameter endpoints
router = APIRouter(prefix="/dream/parameters", tags=["dream_parameters"])

# Adapter reference (to be set during initialization)
dream_parameter_adapter = None

# Logger
logger = logging.getLogger("dream_parameter_api")

# Pydantic models
class ParameterUpdate(BaseModel):
    value: Any
    transition_period: Optional[float] = Field(None, description="Transition period in seconds")
    context: Optional[Dict] = None

class ParameterResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict] = None

@router.get("/", response_model=Dict[str, Any])
async def get_all_dream_parameters():
    """Get all dream processor parameters"""
    if dream_parameter_adapter is None:
        raise HTTPException(status_code=503, detail="Dream parameter adapter not initialized")
    
    # Get the full config from the parameter manager
    return dream_parameter_adapter.param_manager.config

@router.get("/{parameter_path:path}", response_model=Dict[str, Any])
async def get_dream_parameter(parameter_path: str):
    """Get a specific dream parameter by path"""
    if dream_parameter_adapter is None:
        raise HTTPException(status_code=503, detail="Dream parameter adapter not initialized")
    
    # Get the parameter value
    value = dream_parameter_adapter.param_manager._get_nested_value(
        dream_parameter_adapter.param_manager.config, 
        parameter_path
    )
    
    if value is None:
        raise HTTPException(status_code=404, detail=f"Parameter {parameter_path} not found")
    
    # Get metadata if available
    metadata = dream_parameter_adapter.param_manager._get_nested_value(
        dream_parameter_adapter.param_manager.parameter_metadata, 
        parameter_path, 
        {}
    )
    
    return {
        "path": parameter_path,
        "value": value,
        "metadata": metadata
    }

@router.put("/{parameter_path:path}", response_model=ParameterResponse)
async def update_dream_parameter(parameter_path: str, update: ParameterUpdate, background_tasks: BackgroundTasks):
    """Update a specific dream parameter"""
    if dream_parameter_adapter is None:
        raise HTTPException(status_code=503, detail="Dream parameter adapter not initialized")
    
    try:
        # Check if parameter exists
        current_value = dream_parameter_adapter.param_manager._get_nested_value(
            dream_parameter_adapter.param_manager.config, 
            parameter_path
        )
        
        if current_value is None:
            raise HTTPException(status_code=404, detail=f"Parameter {parameter_path} not found")
        
        # Update through the adapter
        result = dream_parameter_adapter.update_parameter(
            parameter_path,
            update.value,
            transition_period=update.transition_period
        )
        
        return ParameterResponse(
            status="success",
            message=f"Updated parameter {parameter_path}",
            data=result
        )
    
    except Exception as e:
        logger.error(f"Error updating parameter {parameter_path}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/verify", response_model=Dict[str, Any])
async def verify_parameters():
    """Verify all parameters for consistency"""
    if dream_parameter_adapter is None:
        raise HTTPException(status_code=503, detail="Dream parameter adapter not initialized")
    
    result = dream_parameter_adapter.verify_parameter_consistency()
    return result

@router.get("/status", response_model=Dict[str, Any])
async def get_parameter_status():
    """Get the status of the parameter management system"""
    if dream_parameter_adapter is None:
        raise HTTPException(status_code=503, detail="Dream parameter adapter not initialized")
    
    # Get number of locked parameters
    locked_count = 0
    locked_params = []
    
    with dream_parameter_adapter.param_manager._lock:
        for path, lock_info in dream_parameter_adapter.param_manager.parameter_locks.items():
            if lock_info["locked"]:
                locked_count += 1
                locked_params.append({
                    "path": path,
                    "holder": lock_info["holder"],
                    "reason": lock_info["reason"],
                    "expires": lock_info["expires"].isoformat() if lock_info["expires"] else None
                })
    
    # Get number of transitions in progress
    transition_count = len(dream_parameter_adapter.param_manager.transition_schedules)
    
    return {
        "status": "active",
        "locked_parameters": locked_count,
        "locked_parameter_details": locked_params,
        "transitions_in_progress": transition_count,
        "version": dream_parameter_adapter.param_manager.config.get("_version", "1.0.0")
    }

# Initialize the API with a dream parameter adapter
def init_dream_parameter_api(adapter):
    global dream_parameter_adapter
    dream_parameter_adapter = adapter
    logger.info("Dream parameter API initialized")
    return router
