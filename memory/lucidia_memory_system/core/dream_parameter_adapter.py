import logging
from typing import Dict, Any, Optional
from datetime import timedelta

class DreamParameterAdapter:
    """
    Adapter class that connects the DreamProcessor with the ParameterManager.
    This provides a clean interface for parameter management and change handling.
    """
    
    def __init__(self, dream_processor, parameter_manager):
        """
        Initialize the adapter with references to both components.
        
        Args:
            dream_processor: Reference to the DreamProcessor instance
            parameter_manager: Reference to the ParameterManager instance
        """
        self.logger = logging.getLogger("DreamParameterAdapter")
        self.dream_processor = dream_processor
        self.param_manager = parameter_manager
        
        # Register as an observer for parameter changes
        self.param_manager.register_observer(self)
        
        # Parameter paths that require special handling
        self.critical_parameters = [
            "dream_cycles.idle_threshold",
            "dream_cycles.auto_dream_enabled",
            "dream_process.max_insights_per_dream"
        ]
        
        # Create mappings between parameter paths and handler methods
        self.parameter_handlers = {
            "dream_cycles.idle_threshold": self._handle_idle_threshold_change,
            "dream_cycles.dream_frequency": self._handle_dream_frequency_change,
            "dream_cycles.auto_dream_enabled": self._handle_auto_dream_change,
            "dream_process.depth_range": self._handle_depth_range_change,
            "dream_process.creativity_range": self._handle_creativity_range_change,
            "integration.default_confidence": self._handle_confidence_change,
        }
        
        # Initialize parameter locks for critical operations
        self._initialize_parameter_locks()
        
        self.logger.info("Dream Parameter Adapter initialized")
    
    def _initialize_parameter_locks(self):
        """Initialize locks for critical parameters"""
        # No locks by default, add as needed during operations
        pass
    
    def on_parameter_changed(self, parameter_path: str, new_value: Any, change_record: Dict):
        """Handle parameter change notifications from ParameterManager"""
        self.logger.info(f"Parameter changed: {parameter_path} = {new_value}")
        
        # Check if we have a specific handler for this parameter
        if parameter_path in self.parameter_handlers:
            self.parameter_handlers[parameter_path](new_value, change_record)
        
        # Update the dream processor's config reference
        # This ensures the DreamProcessor always sees the latest config
        self.dream_processor.config = self.param_manager.config
    
    def _handle_idle_threshold_change(self, new_value: int, change_record: Dict):
        """Handle changes to the idle threshold parameter"""
        self.logger.info(f"Idle threshold updated to {new_value} seconds")
        # Any additional logic specific to this parameter
    
    def _handle_dream_frequency_change(self, new_value: float, change_record: Dict):
        """Handle changes to the dream frequency parameter"""
        self.logger.info(f"Dream frequency updated to {new_value}")
        # Any additional logic specific to this parameter
    
    def _handle_auto_dream_change(self, new_value: bool, change_record: Dict):
        """Handle changes to the auto dream enabled parameter"""
        status = "enabled" if new_value else "disabled"
        self.logger.info(f"Automatic dreaming {status}")
        
        # If disabling and currently dreaming, consider ending the dream
        if not new_value and self.dream_processor.is_dreaming:
            self.logger.warning("Auto-dream disabled while dreaming - considering dream termination")
            # Logic to decide whether to end dream immediately or allow it to complete
    
    def _handle_depth_range_change(self, new_value: tuple, change_record: Dict):
        """Handle changes to the dream depth range parameter"""
        self.logger.info(f"Dream depth range updated to {new_value}")
        # Validate the range is proper (min <= max)
        if new_value[0] > new_value[1]:
            self.logger.warning(f"Invalid depth range: {new_value}, min > max")
            # Consider auto-correcting or reverting
    
    def _handle_creativity_range_change(self, new_value: tuple, change_record: Dict):
        """Handle changes to the creativity range parameter"""
        self.logger.info(f"Creativity range updated to {new_value}")
        # Validate the range is proper (min <= max)
        if new_value[0] > new_value[1]:
            self.logger.warning(f"Invalid creativity range: {new_value}, min > max")
            # Consider auto-correcting or reverting
    
    def _handle_confidence_change(self, new_value: float, change_record: Dict):
        """Handle changes to the default confidence parameter"""
        self.logger.info(f"Default confidence updated to {new_value}")
        # Any additional validation or side effects
    
    def lock_critical_parameters_during_dream(self):
        """Lock critical parameters during dream processing"""
        dream_id = self.dream_processor.dream_state.get("current_dream_id", "unknown")
        duration = timedelta(minutes=30)  # Lock for max 30 minutes
        
        for param_path in self.critical_parameters:
            self.param_manager.lock_parameter(
                param_path, 
                component_id=f"dream_processor:{dream_id}",
                duration=duration,
                reason="Dream in progress - critical parameter"
            )
        
        self.logger.info(f"Locked {len(self.critical_parameters)} critical parameters during dream")
    
    def unlock_critical_parameters_after_dream(self):
        """Unlock critical parameters after dream processing"""
        dream_id = self.dream_processor.dream_state.get("current_dream_id", "unknown")
        
        for param_path in self.critical_parameters:
            self.param_manager.unlock_parameter(
                param_path, 
                component_id=f"dream_processor:{dream_id}"
            )
        
        self.logger.info("Unlocked critical parameters after dream")
    
    def update_parameter(self, path: str, value: Any, transition_period=None):
        """Convenience method to update a parameter through the parameter manager"""
        # Convert seconds to timedelta if needed
        if transition_period is not None:
            transition_period = timedelta(seconds=transition_period)
            
        return self.param_manager.update_parameter(
            path, 
            value, 
            transition_period=transition_period,
            context={"source": "dream_processor"},
            user_id="system"
        )
    
    def verify_parameter_consistency(self):
        """Verify all parameters for consistency"""
        return self.param_manager.verify_configuration_consistency()
