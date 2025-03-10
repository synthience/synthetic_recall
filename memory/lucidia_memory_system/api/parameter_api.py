import uuid
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
from datetime import timedelta
import logging

# We'll import the parameter manager instance from the main application
# This will be set when the API is initialized
parameter_manager = None

# Setup logging
logger = logging.getLogger("parameter_api")

# Create router
router = APIRouter(prefix="/parameters", tags=["parameters"])

# Pydantic models for request/response
class ParameterUpdate(BaseModel):
    value: Any
    transition_period: Optional[float] = Field(None, description="Transition period in seconds")
    transition_function: Optional[str] = Field("linear", description="Transition function (linear, ease_in, ease_out, sigmoid, step, cubic)")
    context: Optional[Dict] = None
    user_id: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
        schema_extra = {
            "example": {
                "value": 0.75,
                "transition_period": 10.0,
                "transition_function": "sigmoid",
                "context": {"source": "user_feedback", "importance": "high"},
                "user_id": "user_123"
            }
        }

class ParameterLock(BaseModel):
    component_id: str = Field(..., description="ID of the component requesting the lock")
    duration_minutes: Optional[float] = Field(5.0, description="Lock duration in minutes")
    reason: Optional[str] = None

class ParameterResponse(BaseModel):
    status: str
    message: str
    transaction_id: Optional[str] = None
    data: Optional[Dict] = None

# API endpoints
@router.get("/config", response_model=Dict)
def get_full_config():
    """Get the complete configuration"""
    if parameter_manager is None:
        raise HTTPException(status_code=503, detail="Parameter manager not initialized")
    
    return parameter_manager.config

@router.get("/metadata", response_model=Dict)
def get_parameter_metadata():
    """Get metadata for all parameters"""
    if parameter_manager is None:
        raise HTTPException(status_code=503, detail="Parameter manager not initialized")
    
    return parameter_manager.parameter_metadata

@router.get("/value/{parameter_path:path}", response_model=Dict)
def get_parameter_value(parameter_path: str):
    """Get a specific parameter value by path"""
    if parameter_manager is None:
        raise HTTPException(status_code=503, detail="Parameter manager not initialized")
    
    value = parameter_manager._get_nested_value(parameter_manager.config, parameter_path)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Parameter {parameter_path} not found")
    
    # Get metadata if available
    metadata = parameter_manager._get_nested_value(parameter_manager.parameter_metadata, parameter_path, {})
    
    return {
        "path": parameter_path,
        "value": value,
        "metadata": metadata
    }

@router.put("/value/{parameter_path:path}", response_model=ParameterResponse)
def update_parameter(parameter_path: str, update: ParameterUpdate, background_tasks: BackgroundTasks):
    """Update a specific parameter value"""
    if parameter_manager is None:
        raise HTTPException(status_code=503, detail="Parameter manager not initialized")
    
    try:
        # Check if parameter is locked
        is_locked = False
        lock_holder = None
        with parameter_manager._lock:
            if (parameter_path in parameter_manager.parameter_locks and 
                parameter_manager.parameter_locks[parameter_path]["locked"]):
                is_locked = True
                lock_holder = parameter_manager.parameter_locks[parameter_path]["holder"]
        
        # If locked, queue the change
        if is_locked:
            result = parameter_manager.queue_parameter_change(
                parameter_path,
                update.value,
                update.user_id or "api",
                transition_period=update.transition_period
            )
            return ParameterResponse(
                status="queued",
                message=f"Parameter update queued (locked by {lock_holder})",
                data=result
            )
        
        # Otherwise, update immediately
        transition_period = None
        if update.transition_period is not None:
            transition_period = timedelta(seconds=update.transition_period)
            
        # Process the update in a background task if transition_period > 0
        if transition_period and transition_period.total_seconds() > 0:
            transaction_id = str(uuid.uuid4())
            background_tasks.add_task(
                parameter_manager.update_parameter,
                parameter_path,
                update.value,
                transition_period=transition_period,
                transition_function=update.transition_function,
                context=update.context,
                user_id=update.user_id,
                transaction_id=transaction_id
            )
            return ParameterResponse(
                status="accepted", 
                message=f"Parameter update scheduled with transition period {update.transition_period}s",
                transaction_id=transaction_id
            )
        else:
            # Immediate update
            result = parameter_manager.update_parameter(
                parameter_path,
                update.value,
                transition_period=None,
                transition_function=update.transition_function,
                context=update.context,
                user_id=update.user_id
            )
            return ParameterResponse(
                status="success", 
                message=f"Parameter {parameter_path} updated successfully",
                transaction_id=result.get("transaction_id"),
                data=result
            )
    except Exception as e:
        logger.error(f"Error updating parameter {parameter_path}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/lock/{parameter_path:path}", response_model=ParameterResponse)
def lock_parameter(parameter_path: str, lock_request: ParameterLock):
    """Lock a parameter to prevent changes"""
    if parameter_manager is None:
        raise HTTPException(status_code=503, detail="Parameter manager not initialized")
    
    try:
        duration = timedelta(minutes=lock_request.duration_minutes)
        result = parameter_manager.lock_parameter(
            parameter_path, 
            lock_request.component_id, 
            duration=duration,
            reason=lock_request.reason
        )
        
        if result["status"] == "success":
            return ParameterResponse(
                status="success",
                message=f"Parameter {parameter_path} locked successfully",
                data=result
            )
        else:
            return ParameterResponse(
                status="failed",
                message=result.get("message", "Unknown error locking parameter"),
                data=result
            )
    except Exception as e:
        logger.error(f"Error locking parameter {parameter_path}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/unlock/{parameter_path:path}", response_model=ParameterResponse)
def unlock_parameter(parameter_path: str, lock_request: ParameterLock):
    """Unlock a parameter"""
    if parameter_manager is None:
        raise HTTPException(status_code=503, detail="Parameter manager not initialized")
    
    try:
        result = parameter_manager.unlock_parameter(parameter_path, lock_request.component_id)
        
        if result["status"] == "success":
            return ParameterResponse(
                status="success",
                message=f"Parameter {parameter_path} unlocked successfully"
            )
        elif result["status"] == "not_locked":
            return ParameterResponse(
                status="warning",
                message=f"Parameter {parameter_path} was not locked"
            )
        else:
            return ParameterResponse(
                status="failed",
                message=result.get("message", "Unknown error unlocking parameter")
            )
    except Exception as e:
        logger.error(f"Error unlocking parameter {parameter_path}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/verify", response_model=Dict)
def verify_configuration():
    """Verify the entire configuration for consistency"""
    if parameter_manager is None:
        raise HTTPException(status_code=503, detail="Parameter manager not initialized")
    
    result = parameter_manager.verify_configuration_consistency()
    return result

@router.get("/history", response_model=List[Dict])
def get_parameter_history(
    path: Optional[str] = Query(None, description="Filter by parameter path"),
    limit: Optional[int] = Query(100, description="Maximum number of history items to return")
):
    """Get the history of parameter changes"""
    if parameter_manager is None:
        raise HTTPException(status_code=503, detail="Parameter manager not initialized")
    
    history = parameter_manager.change_history
    
    # Filter by path if specified
    if path:
        history = [item for item in history if item["path"] == path]
    
    # Limit the number of items
    history = history[-limit:]
    
    return history

# Initialize the API with a parameter manager instance
def init_parameter_api(manager_instance):
    global parameter_manager
    parameter_manager = manager_instance
    logger.info("Parameter API initialized with parameter manager instance")
    return router
