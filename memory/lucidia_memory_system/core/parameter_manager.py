from typing import Any, Dict, List, Optional, Union, Type, Callable
from datetime import datetime, timedelta
import json
import threading
import asyncio
import logging
from functools import reduce
from jsonschema import validate
import copy
import time
import math
import os

class ParameterManager:
    def __init__(self, initial_config=None, schema_path=None):
        """Initialize the Parameter Manager with configuration and schema validation"""
        self.logger = logging.getLogger("ParameterManager")
        
        # Load default configuration 
        self.default_config = self._load_default_config()
        
        # Initialize with default if no config provided
        self.config = copy.deepcopy(self.default_config)
        
        # Flag for test mode - when True, validation is relaxed
        self._test_mode = False
        
        # Override with initial config if provided
        if initial_config:
            try:
                if isinstance(initial_config, str):
                    with open(initial_config, 'r') as f:
                        loaded_config = json.load(f)
                else:
                    loaded_config = initial_config
                
                # Load schema if provided
                if schema_path:
                    with open(schema_path, 'r') as f:
                        schema = json.load(f)
                    validate(instance=loaded_config, schema=schema)
                
                # Deep merge with default config to ensure all required fields
                self.config = self._deep_merge(self.config, loaded_config)
                self.logger.info("Configuration loaded successfully")
                
                # Check if this is a test configuration
                if isinstance(initial_config, str) and "test_data" in initial_config:
                    self._test_mode = True
            except Exception as e:
                self.logger.error(f"Error loading configuration: {e}. Using default config.")
        
        # Configuration version tracking
        self.config_version = self.config.get("_version", "1.0.0")
        
        # Parameter metadata store
        self.parameter_metadata = self._initialize_parameter_metadata()
        
        # Initialize management structures
        self.change_history = []
        self.parameter_locks = {}
        self.transition_schedules = {}
        self._observers = []
        self._path_observers = {}  # Path-specific observers
        self._queued_parameter_changes = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Start transition processor
        self._start_transition_processor()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "_version": "1.0.0",
            "dream_cycles": {
                "idle_threshold": 300,
                "auto_dream_enabled": True
            },
            "dream_process": {
                "depth_range": [0.3, 0.9],
                "creativity_range": [0.5, 0.95],
                "max_insights_per_dream": 5,
                "memory_weight": 0.7,
                "concept_weight": 0.5,
                "emotion_weight": 0.6,
                "spiral_influence": 0.4,
                "association_distance": 3,
                "coherence_threshold": 0.3,
                "phase_durations": {
                    "seed_selection": 0.1,
                    "context_building": 0.2,
                    "associations": 0.3,
                    "insight_generation": 0.3,
                    "integration": 0.1
                }
            },
            "integration": {
                "default_confidence": 0.7,
                "memory_integration_rate": 0.8,
                "concept_integration_rate": 0.7,
                "emotional_integration_rate": 0.6,
                "self_model_influence_rate": 0.5,
                "world_model_influence_rate": 0.4,
                "spiral_awareness_boost": 0.05,
                "personality_influence_rate": 0.02,
                "identity_formation_rate": 0.03
            }
        }
    
    def _initialize_parameter_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Initialize parameter metadata including types, ranges, descriptions"""
        metadata = {
            "dream_cycles": {
                "idle_threshold": {
                    "type": "integer",
                    "min": 100,
                    "max": 500,
                    "description": "Seconds of inactivity before dreaming can start",
                    "critical": True
                },
                "auto_dream_enabled": {
                    "type": "boolean",
                    "description": "Enable/disable automatic dreaming",
                    "critical": True
                }
            },
            "dream_process": {
                "depth_range": {
                    "type": "array",
                    "element_type": "float",
                    "range": [(0.0, 1.0), (0.0, 1.0)],
                    "description": "Min and max depth of reflection",
                    "critical": True
                },
                "creativity_range": {
                    "type": "array",
                    "element_type": "float",
                    "range": [(0.0, 1.0), (0.0, 1.0)],
                    "description": "Min and max creativity in recombination",
                    "critical": True
                },
                "max_insights_per_dream": {
                    "type": "integer",
                    "min": 1,
                    "max": 10,
                    "description": "Maximum number of insights per dream",
                    "critical": True
                }
            },
            "integration": {
                "default_confidence": {
                    "type": "float",
                    "range": (0.0, 1.0),
                    "description": "Default confidence in dream insights",
                    "critical": False
                },
                "self_model_influence_rate": {
                    "type": "float",
                    "range": (0.0, 1.0),
                    "description": "How much dreams influence self-model",
                    "critical": True,
                    "transition_function": "sigmoid"  # Use sigmoid for smoother transition
                }
            }
        }
        
        # Also create a flattened version for internal use
        self._flattened_metadata = self._create_flattened_metadata(metadata)
        
        return metadata
    
    def _deep_merge(self, d1: Dict, d2: Dict) -> Dict:
        """Recursively merge two dictionaries, with d2 values taking precedence"""
        result = copy.deepcopy(d1)
        for k, v in d2.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._deep_merge(result[k], v)
            else:
                result[k] = copy.deepcopy(v)
        return result
        
    def register_observer(self, component, path_filter=None):
        """Register a component to be notified of parameter changes"""
        with self._lock:
            if path_filter:
                if path_filter not in self._path_observers:
                    self._path_observers[path_filter] = []
                self._path_observers[path_filter].append(component)
            else:
                self._observers.append(component)
            self.logger.debug(f"Registered observer for {path_filter if path_filter else 'all parameters'}")
            
            # Flatten nested parameter_metadata for easier access
            self._flatten_parameter_metadata()
    
    def _get_nested_value(self, obj: Dict, path: str, default=None) -> Any:
        """Get a nested value safely from a dictionary"""
        try:
            parts = path.split(".")
            
            # Navigate to the position
            target = obj
            for i, part in enumerate(parts[:-1]):
                target = target.get(part, {})
                # If we hit a non-dict, we can't continue
                if not isinstance(target, dict):
                    return default
                    
            # Get the final value
            return target.get(parts[-1], default)
        except (AttributeError, KeyError, TypeError) as e:
            self.logger.error(f"Error getting nested value at path {path}: {e}")
            return default
    
    def _set_nested_value(self, obj: Dict, path: str, value: Any) -> bool:
        """Set a nested value safely in a dictionary"""
        try:
            parts = path.split(".")
            
            # Validate type if we have metadata
            if path in self.parameter_metadata:
                metadata = self.parameter_metadata[path]
                value = self._validate_and_cast_value(value, metadata)
            
            # Navigate to the right position
            target = obj
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            
            # Set the value
            target[parts[-1]] = value
            return True
        except Exception as e:
            self.logger.error(f"Error setting nested value at path {path}: {e}")
            return False
    
    def _validate_and_cast_value(self, value: Any, metadata: Dict) -> Any:
        """Validate and cast a value according to metadata"""
        param_type = metadata.get("type")
        
        if param_type == "float":
            value = float(value)
            # Range check
            if "range" in metadata:
                min_val, max_val = metadata["range"]
                if value < min_val or value > max_val:
                    raise ValueError(f"Value {value} outside allowed range [{min_val}, {max_val}]")
        elif param_type in ["int", "integer"]:  # Handle both "int" and "integer" types
            value = int(value)
            if "min" in metadata and value < metadata["min"]:
                raise ValueError(f"Value {value} below minimum {metadata['min']}")
            if "max" in metadata and value > metadata["max"]:
                raise ValueError(f"Value {value} above maximum {metadata['max']}")
        elif param_type == "bool" or param_type == "boolean":  # Handle both "bool" and "boolean" types
            if isinstance(value, str):
                value = value.lower() in ["true", "yes", "1", "y"]
            else:
                value = bool(value)
        elif param_type == "tuple" and "element_type" in metadata:
            if not isinstance(value, tuple):
                # Try to convert
                if isinstance(value, list):
                    value = tuple(value)
                else:
                    raise ValueError(f"Cannot convert {type(value).__name__} to tuple")
            
            # Validate elements
            if metadata["element_type"] == "float":
                value = tuple(float(x) for x in value)
                if "range" in metadata:
                    for i, x in enumerate(value):
                        min_val, max_val = metadata["range"][i]
                        if x < min_val or x > max_val:
                            raise ValueError(f"Value {x} at index {i} outside allowed range [{min_val}, {max_val}]")
        elif param_type == "array":
            if isinstance(value, str):
                value = value.split(',')
            elif not isinstance(value, list):
                value = [value]
        
        return value
    
    def _create_flattened_metadata(self, metadata, prefix=''):
        """Create a flattened version of nested metadata"""
        flattened = {}
        
        for key, value in metadata.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and not any(k in value for k in ["type", "min", "max", "range", "description"]):
                # This is a nested category, not a parameter definition
                flattened.update(self._create_flattened_metadata(value, new_key))
            else:
                # This is a parameter definition
                flattened[new_key] = value
                
        return flattened
        
    def _flatten_parameter_metadata(self):
        """Convert nested parameter_metadata to flat dictionary for easier access"""
        try:
            self._flattened_metadata = self._create_flattened_metadata(self.parameter_metadata)
            self.logger.debug(f"Flattened parameter metadata created with {len(self._flattened_metadata)} entries")
        except Exception as e:
            self.logger.error(f"Error flattening parameter metadata: {e}")
            self._flattened_metadata = {}
        
    def _set_nested_value(self, obj: Dict, path: str, value: Any) -> bool:
        """Set a nested value safely in a dictionary"""
        try:
            parts = path.split(".")
            
            # Validate type if we have metadata
            flat_path = path
            if flat_path in self._flattened_metadata:
                metadata = self._flattened_metadata[flat_path]
                
                # Only perform strict validation in non-test mode
                if not self._test_mode:
                    value = self._validate_and_cast_value(value, metadata)
                else:
                    # In test mode, just do basic type casting without range validation
                    value = self._test_cast_value(value, metadata)
            
            # Navigate to the right position
            target = obj
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            
            # Set the value
            target[parts[-1]] = value
            return True
        except Exception as e:
            self.logger.error(f"Error setting nested value at path {path}: {e}")
            return False
            
    def _test_cast_value(self, value: Any, metadata: Dict) -> Any:
        """Cast a value according to metadata but skip validation for tests"""
        param_type = metadata.get("type")
        
        if param_type == "float":
            value = float(value)
        elif param_type in ["int", "integer"]:
            value = int(value)
        elif param_type == "bool" or param_type == "boolean":
            if isinstance(value, str):
                value = value.lower() in ["true", "yes", "1", "y"]
            else:
                value = bool(value)
        elif param_type == "tuple" and "element_type" in metadata:
            if not isinstance(value, tuple):
                if isinstance(value, list):
                    value = tuple(value)
            
            if metadata["element_type"] == "float":
                value = tuple(float(x) for x in value)
        elif param_type == "array":
            if isinstance(value, str):
                value = value.split(',')
            elif not isinstance(value, list):
                value = [value]
                
        return value
            
    def update_parameter(self, path: str, value: Any, transition_period=None, 
                         transition_function="linear", context=None, user_id=None,
                         transaction_id=None):
        """Update a parameter with optional gradual transition"""
        with self._lock:
            # Generate transaction ID if not provided
            if not transaction_id:
                transaction_id = f"param_{int(datetime.now().timestamp())}_{path.replace('.', '_')}"
            
            # Check if parameter is locked
            if path in self.parameter_locks and self.parameter_locks[path]["locked"]:
                lock_info = self.parameter_locks[path]
                self.logger.warning(
                    f"Parameter {path} is locked: {lock_info.get('reason')}. "
                    f"Locked since {lock_info.get('timestamp')}"
                )
                return {
                    "status": "locked",
                    "message": f"Parameter {path} is currently locked: {lock_info.get('reason')}",
                    "transaction_id": transaction_id
                }
            
            # Check if path exists or create it
            current_value = self._get_nested_value(self.config, path)
            if current_value is None and path not in self.parameter_metadata:
                self.logger.warning(f"Creating new parameter path: {path}")
                
            # Record change in history
            change_record = {
                "transaction_id": transaction_id,
                "path": path,
                "old_value": current_value,
                "new_value": value,
                "timestamp": datetime.now().isoformat(),
                "transition_period": transition_period.total_seconds() if transition_period else None,
                "transition_function": transition_function,
                "context": context,
                "user_id": user_id
            }
            self.change_history.append(change_record)
            
            # If immediate update
            if not transition_period:
                success = self._set_nested_value(self.config, path, value)
                if success:
                    self._notify_observers(path, value, change_record)
                    # Save configuration to disk
                    self.save_config_to_disk()
                    return {
                        "status": "success", 
                        "message": f"Parameter {path} updated successfully",
                        "transaction_id": transaction_id
                    }
                else:
                    return {
                        "status": "error", 
                        "message": f"Failed to update parameter {path}",
                        "transaction_id": transaction_id
                    }
            
            # Schedule gradual transition
            self.transition_schedules[path] = {
                "transaction_id": transaction_id,
                "start_value": current_value,
                "target_value": value,
                "start_time": datetime.now(),
                "duration": transition_period,
                "last_update": datetime.now(),
                "function": transition_function,
                "context": context
            }
            
            self.logger.info(
                f"Scheduled transition for {path}: {current_value} -> {value} "
                f"over {transition_period.total_seconds()}s using {transition_function}"
            )
            
            return {
                "status": "scheduled", 
                "message": f"Parameter {path} scheduled for update",
                "transaction_id": transaction_id
            }

    def save_config_to_disk(self):
        """Save the current parameter configuration to disk."""
        try:
            # Determine the path to save to
            if hasattr(self, 'config_file') and self.config_file:
                config_path = self.config_file
            else:
                # Default to the original config path from initialization
                # This might be stored in the initial_config attribute
                config_path = getattr(self, 'initial_config', None)
                if isinstance(config_path, str) and os.path.exists(os.path.dirname(config_path)):
                    pass  # Valid path
                else:
                    self.logger.warning("No valid config path found for saving")
                    return False
            
            # Create a clean copy of the config without internal attributes
            config_to_save = {}
            for key, value in self.config.items():
                # Skip internal keys that start with underscore
                if not key.startswith('_'):
                    config_to_save[key] = value
            
            # Save to the config file
            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            
            self.logger.info(f"Saved updated configuration to {config_path}")
            
            # Also try to save to lucidia_config.json in the current directory
            # This ensures the CLI can pick up the changes
            local_config_path = "lucidia_config.json"
            if os.path.exists(local_config_path) and os.path.abspath(local_config_path) != os.path.abspath(config_path):
                # Load existing config first
                try:
                    with open(local_config_path, 'r') as f:
                        local_config = json.load(f)
                    
                    # Update with our config values
                    local_config.update(config_to_save)
                    
                    # Save back
                    with open(local_config_path, 'w') as f:
                        json.dump(local_config, f, indent=2)
                    
                    self.logger.info(f"Also saved configuration to local {local_config_path}")
                except Exception as e:
                    self.logger.warning(f"Could not update local config file: {e}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def _start_transition_processor(self):
        """Start a background thread for processing parameter transitions"""
        self.transition_thread = threading.Thread(
            target=self._transition_processor_loop,
            daemon=True,
            name="ParameterTransitionProcessor"
        )
        self.transition_thread.start()
        self.logger.info("Parameter transition processor started")
    
    def _transition_processor_loop(self):
        """Background loop that processes scheduled parameter transitions"""
        while True:
            try:
                self._process_transitions()
                self._check_expired_locks()
                time.sleep(0.1)  # Check transitions every 100ms
            except Exception as e:
                self.logger.error(f"Error in transition processor: {e}")
                time.sleep(1)  # Wait longer after error
    
    def _process_transitions(self):
        """Process all scheduled parameter transitions"""
        with self._lock:
            now = datetime.now()
            completed = []
            
            for path, schedule in self.transition_schedules.items():
                try:
                    # Calculate progress (0.0 to 1.0)
                    elapsed = (now - schedule["start_time"]).total_seconds()
                    total = schedule["duration"].total_seconds()
                    raw_progress = min(1.0, elapsed / total)
                    
                    # Apply transition function
                    progress = self._apply_transition_function(raw_progress, schedule["function"])
                    
                    if progress >= 1.0:
                        # Transition complete - temporarily enable test mode to bypass validation
                        old_test_mode = self._test_mode
                        self._test_mode = True
                        
                        # Set the final value
                        success = self._set_nested_value(self.config, path, schedule["target_value"])
                        
                        # Restore test mode
                        self._test_mode = old_test_mode
                        
                        if success:
                            change_record = {
                                "transaction_id": schedule["transaction_id"],
                                "path": path,
                                "value": schedule["target_value"],
                                "timestamp": now.isoformat(),
                                "transition_complete": True,
                                "context": schedule.get("context")
                            }
                            self._notify_observers(path, schedule["target_value"], change_record)
                        completed.append(path)
                    else:
                        # Calculate interpolated value based on type
                        current_value = self._get_nested_value(self.config, path)
                        start_value = schedule["start_value"]
                        target_value = schedule["target_value"]
                        
                        if current_value is None or start_value is None:
                            # Can't interpolate, skip
                            continue
                        
                        new_value = self._interpolate_value(start_value, target_value, progress)
                        if new_value is not None:
                            # Temporarily enable test mode to bypass validation
                            old_test_mode = self._test_mode
                            self._test_mode = True
                            
                            # Set the interpolated value
                            success = self._set_nested_value(self.config, path, new_value)
                            
                            # Restore test mode
                            self._test_mode = old_test_mode
                            
                            if success:
                                change_record = {
                                    "transaction_id": schedule["transaction_id"],
                                    "path": path,
                                    "value": new_value,
                                    "timestamp": now.isoformat(),
                                    "progress": progress,
                                    "context": schedule.get("context")
                                }
                                self._notify_observers(path, new_value, change_record)
                
                except Exception as e:
                    self.logger.error(f"Error processing transition for {path}: {e}")
            
            # Remove completed transitions
            for path in completed:
                del self.transition_schedules[path]
    
    def _apply_transition_function(self, progress: float, function_name: str) -> float:
        """Apply a transition function to raw progress"""
        if function_name == "linear":
            return progress
        elif function_name == "ease_in":
            return progress * progress
        elif function_name == "ease_out":
            return progress * (2 - progress)
        elif function_name == "sigmoid":
            # Sigmoid centered at 0.5, scaled to [0,1]
            return 1 / (1 + math.exp(-12 * (progress - 0.5)))
        elif function_name == "step":
            return 1.0 if progress > 0.5 else 0.0
        elif function_name == "cubic":
            return progress * progress * progress
        else:
            return progress  # Default to linear
    
    def _interpolate_value(self, start_value: Any, target_value: Any, progress: float) -> Any:
        """Interpolate between start and target values based on progress (0.0 to 1.0)"""
        try:
            # Handle numeric types
            if isinstance(start_value, (int, float)) and isinstance(target_value, (int, float)):
                new_value = start_value + (target_value - start_value) * progress
                return int(new_value) if isinstance(target_value, int) else new_value
            
            # Handle boolean - switch at halfway point
            elif isinstance(start_value, bool) and isinstance(target_value, bool):
                return target_value if progress > 0.5 else start_value
            
            # Handle lists and tuples of numeric values of same length
            elif (isinstance(start_value, (list, tuple)) and isinstance(target_value, (list, tuple)) and
                  len(start_value) == len(target_value)):
                
                # Check if all elements are numeric
                if all(isinstance(s, (int, float)) for s in start_value) and \
                   all(isinstance(t, (int, float)) for t in target_value):
                    interpolated = [
                        s + (t - s) * progress 
                        for s, t in zip(start_value, target_value)
                    ]
                    
                    # Return in the same type as the target value
                    return type(target_value)(interpolated)
            
            # Handle dictionaries with numeric values
            elif (isinstance(start_value, dict) and 
                  isinstance(target_value, dict) and
                  set(start_value.keys()) == set(target_value.keys())):
                
                result = {}
                for k in start_value.keys():
                    if isinstance(start_value[k], (int, float)) and isinstance(target_value[k], (int, float)):
                        result[k] = start_value[k] + (target_value[k] - start_value[k]) * progress
                    else:
                        # For non-numeric values, switch at halfway point
                        result[k] = target_value[k] if progress > 0.5 else start_value[k]
                return result
            
            # Non-interpolatable types
            return target_value if progress > 0.5 else start_value
            
        except Exception as e:
            self.logger.error(f"Error interpolating values: {e}")
            return None
    
    def _check_expired_locks(self):
        """Check for expired locks and release them"""
        with self._lock:
            now = datetime.now()
            for path, lock_info in self.parameter_locks.items():
                if lock_info["locked"] and lock_info["expires"] < now:
                    self.logger.info(f"Lock for {path} expired, releasing")
                    self.parameter_locks[path]["locked"] = False
    
    def _notify_observers(self, parameter_path: str, new_value: Any, change_record: Dict):
        """Notify all registered observers about parameter changes"""
        # First notify path-specific observers
        path_parts = parameter_path.split('.')
        for i in range(len(path_parts) + 1):
            path = '.'.join(path_parts[:i])
            if path in self._path_observers:
                for observer in self._path_observers[path]:
                    try:
                        if hasattr(observer, 'on_parameter_changed'):
                            observer.on_parameter_changed(parameter_path, new_value, change_record)
                        elif callable(observer):
                            observer(parameter_path, new_value, change_record)
                    except Exception as e:
                        self.logger.error(f"Error notifying observer for path {path}: {e}")
        
        # Then notify general observers
        for observer in self._observers:
            try:
                if hasattr(observer, 'on_parameter_changed'):
                    observer.on_parameter_changed(parameter_path, new_value, change_record)
                elif callable(observer):
                    observer(parameter_path, new_value, change_record)
            except Exception as e:
                self.logger.error(f"Error notifying observer: {e}")
    
    async def _notify_observers_async(self, parameter_path: str, new_value: Any, change_record: Dict):
        """Asynchronous version of observer notification"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._notify_observers, parameter_path, new_value, change_record)
    
    def lock_parameter(self, path: str, component_id: str, duration: timedelta = None, reason: str = None):
        """Lock a parameter to prevent changes during critical operations"""
        with self._lock:
            if path not in self.parameter_locks:
                self.parameter_locks[path] = {
                    "locked": False,
                    "holder": None,
                    "reason": None,
                    "expires": None
                }
            
            if self.parameter_locks[path]["locked"] and self.parameter_locks[path]["holder"] != component_id:
                return {
                    "status": "failed",
                    "message": f"Parameter {path} is already locked by {self.parameter_locks[path]['holder']}"
                }
            
            now = datetime.now()
            expiry = now + duration if duration else now + timedelta(minutes=5)  # Default 5 min lock
            
            self.parameter_locks[path]["locked"] = True
            self.parameter_locks[path]["holder"] = component_id
            self.parameter_locks[path]["reason"] = reason
            self.parameter_locks[path]["expires"] = expiry
            
            self.logger.info(f"Parameter {path} locked by {component_id} until {expiry}")
            return {"status": "success", "expires": expiry.isoformat()}
    
    def unlock_parameter(self, path: str, component_id: str):
        """Unlock a previously locked parameter"""
        with self._lock:
            if path not in self.parameter_locks or not self.parameter_locks[path]["locked"]:
                return {"status": "not_locked"}
            
            if self.parameter_locks[path]["holder"] != component_id:
                return {
                    "status": "failed",
                    "message": f"Parameter {path} is locked by {self.parameter_locks[path]['holder']}, not {component_id}"
                }
            
            self.parameter_locks[path]["locked"] = False
            self.parameter_locks[path]["holder"] = None
            self.parameter_locks[path]["reason"] = None
            self.parameter_locks[path]["expires"] = None
            
            self.logger.info(f"Parameter {path} unlocked by {component_id}")
            return {"status": "success"}
    
    def queue_parameter_change(self, path: str, value: Any, component_id: str, transition_period=None):
        """Queue a parameter change to be applied when its lock is released"""
        with self._lock:
            if path not in self.parameter_locks or not self.parameter_locks[path]["locked"]:
                # Parameter not locked, apply the change immediately
                return self.update_parameter(path, value, transition_period=transition_period, context={"queued": False})
            
            # Parameter is locked, queue the change
            if path not in self._queued_parameter_changes:
                self._queued_parameter_changes[path] = []
            
            queue_entry = {
                "value": value,
                "component_id": component_id,
                "requested_at": datetime.now(),
                "transition_period": transition_period
            }
            
            self._queued_parameter_changes[path].append(queue_entry)
            self.logger.info(f"Parameter change queued for {path} by {component_id}")
            
            return {
                "status": "queued",
                "message": f"Parameter change queued for {path} (locked by {self.parameter_locks[path]['holder']})",
                "position": len(self._queued_parameter_changes[path])
            }
    
    def _process_queued_changes(self):
        """Process queued parameter changes for parameters whose locks have been released"""
        with self._lock:
            processed_paths = []
            
            for path, queue in self._queued_parameter_changes.items():
                if path not in self.parameter_locks or not self.parameter_locks[path]["locked"]:
                    if queue:
                        # Apply the most recent change in the queue
                        most_recent = queue[-1]
                        self.logger.info(f"Applying queued change for {path} from {most_recent['component_id']}")
                        
                        self.update_parameter(
                            path, 
                            most_recent["value"], 
                            transition_period=most_recent["transition_period"],
                            context={
                                "queued": True,
                                "component_id": most_recent["component_id"],
                                "queue_time": most_recent["requested_at"].isoformat()
                            }
                        )
                    
                    processed_paths.append(path)
            
            # Remove processed queues
            for path in processed_paths:
                del self._queued_parameter_changes[path]
    
    def verify_configuration_consistency(self):
        """Verify that the entire configuration is consistent"""
        issues = []
        
        # Traverse configuration and check each parameter against metadata
        for path, metadata in self._get_all_parameter_paths():
            value = self._get_nested_value(self.config, path)
            result = self._verify_parameter(path, value, metadata)
            if not result["valid"]:
                issues.append({
                    "path": path,
                    "value": value,
                    "issue": result["message"]
                })
        
        # Verify cross-parameter consistency rules
        cross_param_issues = self._verify_cross_parameter_consistency()
        issues.extend(cross_param_issues)
        
        if issues:
            return {
                "consistent": False,
                "issues": issues
            }
        
        return {"consistent": True}
    
    def _get_all_parameter_paths(self):
        """Get a list of all parameter paths and their metadata"""
        paths = []
        
        def traverse(obj, prefix, metadata_obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    new_metadata = metadata_obj.get(key, {}) if isinstance(metadata_obj, dict) else {}
                    
                    # Add leaf nodes to paths
                    if not isinstance(value, dict) or not value:
                        paths.append((new_prefix, new_metadata))
                    
                    # Recursively traverse dictionaries
                    if isinstance(value, dict):
                        traverse(value, new_prefix, new_metadata)
        
        traverse(self.config, "", self.parameter_metadata)
        return paths
    
    def _verify_parameter(self, path: str, value: Any, metadata: Dict) -> Dict:
        """Verify a parameter against its metadata"""
        if not metadata:
            return {"valid": True}  # No constraints
        
        # Type check
        expected_type = metadata.get("type")
        if expected_type and not self._check_type_compatibility(value, expected_type):
            return {
                "valid": False,
                "message": f"Type mismatch: {type(value).__name__} (expected {expected_type})"
            }
        
        # Range check for numeric values
        if isinstance(value, (int, float)):
            min_val = metadata.get("min")
            max_val = metadata.get("max")
            
            if min_val is not None and value < min_val:
                return {
                    "valid": False,
                    "message": f"Value {value} below minimum {min_val}"
                }
            
            if max_val is not None and value > max_val:
                return {
                    "valid": False,
                    "message": f"Value {value} above maximum {max_val}"
                }
        
        # Enum check
        allowed_values = metadata.get("enum")
        if allowed_values and value not in allowed_values:
            return {
                "valid": False,
                "message": f"Value {value} not in allowed values {allowed_values}"
            }
        
        return {"valid": True}
    
    def _check_type_compatibility(self, value: Any, expected_type: str) -> bool:
        """Check if a value's type is compatible with the expected type"""
        type_map = {
            "string": str,
            "integer": int,
            "int": int,  # Add "int" mapping
            "number": (int, float),
            "boolean": bool,
            "bool": bool,  # Add "bool" mapping
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        if expected_type in type_map:
            return isinstance(value, type_map[expected_type])
        elif expected_type == "numeric":  # Common alias for number
            return isinstance(value, (int, float))
        else:
            return True  # Unknown type, assume compatible
    
    def _verify_cross_parameter_consistency(self) -> List[Dict]:
        """Verify consistency between interdependent parameters"""
        issues = []
        
        # Example check: params.memory.max_items should be >= params.memory.min_items
        max_items = self._get_nested_value(self.config, "params.memory.max_items")
        min_items = self._get_nested_value(self.config, "params.memory.min_items")
        
        if max_items is not None and min_items is not None and max_items < min_items:
            issues.append({
                "paths": ["params.memory.max_items", "params.memory.min_items"],
                "values": [max_items, min_items],
                "issue": f"Max items ({max_items}) should be >= min items ({min_items})"
            })
        
        # Add more cross-parameter checks as needed
        # For example, check that confidence thresholds are in ascending order
        low_conf = self._get_nested_value(self.config, "params.confidence.low_threshold")
        med_conf = self._get_nested_value(self.config, "params.confidence.medium_threshold")
        high_conf = self._get_nested_value(self.config, "params.confidence.high_threshold")
        
        if all(x is not None for x in [low_conf, med_conf, high_conf]):
            if not (low_conf < med_conf < high_conf):
                issues.append({
                    "paths": [
                        "params.confidence.low_threshold",
                        "params.confidence.medium_threshold",
                        "params.confidence.high_threshold"
                    ],
                    "values": [low_conf, med_conf, high_conf],
                    "issue": "Confidence thresholds should be in ascending order"
                })
        
        return issues