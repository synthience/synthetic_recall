"""Model Selector for Lucidia

Provides utilities for dynamic model selection and switching based on system state.
Serves as an interface between system components and the ModelManager.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from .model_manager import ModelManager, ModelPurpose, SystemState

logger = logging.getLogger("ModelSelector")

class ModelSelector:
    """Interface for dynamic model selection in Lucidia system"""
    
    _instance = None
    _initialized = False
    
    @classmethod
    def get_instance(cls) -> 'ModelSelector':
        """Get or create the singleton instance of ModelSelector"""
        if cls._instance is None:
            cls._instance = ModelSelector()
        return cls._instance
    
    def __init__(self):
        """Initialize the model selector"""
        if ModelSelector._initialized:
            raise RuntimeError("ModelSelector is a singleton, use get_instance() instead")
            
        self.manager = ModelManager()
        self.switch_queue = asyncio.Queue()
        self.task = None
        self.running = False
        self.llm_services = {}
        ModelSelector._initialized = True
        logger.info("ModelSelector initialized")
        
    def start(self):
        """Start the model selector background task"""
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._switch_worker())
            logger.info("Started model selector background task")
            
    def stop(self):
        """Stop the model selector background task"""
        if self.running:
            self.running = False
            if self.task and not self.task.done():
                self.task.cancel()
            logger.info("Stopped model selector background task")
    
    async def _switch_worker(self):
        """Background worker that processes model switch requests"""
        while self.running:
            try:
                # Get the next switch request from the queue
                service_id, model_name, llm_service = await self.switch_queue.get()
                
                # Attempt to switch the model
                logger.info(f"Processing model switch request for service {service_id} to {model_name}")
                success = await self.manager.switch_model(model_name, llm_service)
                
                if success:
                    logger.info(f"Successfully switched service {service_id} to model {model_name}")
                else:
                    logger.warning(f"Failed to switch service {service_id} to model {model_name}")
                    
                # Mark the task as done
                self.switch_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Model switch worker cancelled")
                break
                
            except Exception as e:
                logger.error(f"Error in model switch worker: {e}")
                # Sleep briefly to avoid high CPU usage in case of repeated errors
                await asyncio.sleep(1)
    
    def register_llm_service(self, service_id: str, llm_service) -> None:
        """Register an LLM service with the model selector
        
        Args:
            service_id: Unique identifier for the service
            llm_service: The LLM service instance
        """
        if not self.running:
            self.start()
            
        self.llm_services[service_id] = llm_service
        logger.info(f"Registered LLM service with ID: {service_id}")
        
        # Store initial model for monitoring
        current_model = llm_service.model
        if current_model:
            self.manager.active_model = current_model
    
    def unregister_llm_service(self, service_id: str) -> None:
        """Unregister an LLM service
        
        Args:
            service_id: Service ID to unregister
        """
        if service_id in self.llm_services:
            del self.llm_services[service_id]
            logger.info(f"Unregistered LLM service with ID: {service_id}")
        
        # Stop background task if no services left
        if not self.llm_services and self.running:
            self.stop()
    
    def update_system_state(self, state: Union[str, SystemState]) -> None:
        """Update the current system state
        
        Args:
            state: New system state (as string or enum)
        """
        if isinstance(state, str):
            try:
                state = SystemState(state)
            except ValueError:
                logger.warning(f"Invalid system state: {state}, using ACTIVE")
                state = SystemState.ACTIVE
                
        self.manager.update_system_state(state)
        
        # Check if we should recommend model switches
        self._check_recommended_models()
    
    def _check_recommended_models(self) -> None:
        """Check if models should be switched based on current recommendations"""
        for service_id, llm_service in self.llm_services.items():
            current_model = llm_service.model
            recommended_model = self.manager.get_recommended_model()
            
            if current_model != recommended_model:
                logger.info(f"Recommending switch for service {service_id} from {current_model} to {recommended_model}")
                # Queue the switch instead of doing it immediately
                self.switch_queue.put_nowait((service_id, recommended_model, llm_service))
    
    async def select_model_for_task(self, task_type: str, llm_service, service_id: Optional[str] = None) -> bool:
        """Select the best model for a specific task type
        
        Args:
            task_type: Type of task requiring model selection
            llm_service: LLM service to update
            service_id: Optional service ID for tracking (defaults to object id)
            
        Returns:
            Whether model was changed
        """
        # Map task type to model purpose
        purpose_map = {
            "chat": ModelPurpose.GENERAL,
            "reasoning": ModelPurpose.REASONING,
            "creative": ModelPurpose.CREATIVE,
            "analysis": ModelPurpose.ANALYSIS,
            "embedding": ModelPurpose.EMBEDDING,
            "memory": ModelPurpose.MEMORY_PROCESSING,
            "dream": ModelPurpose.DREAMING,
            "reflection": ModelPurpose.REFLECTION
        }
        
        purpose = purpose_map.get(task_type, ModelPurpose.GENERAL)
        recommended_model = self.manager.get_recommended_model(purpose)
        current_model = llm_service.model
        
        # Use object id as service_id if not provided
        if not service_id:
            service_id = id(llm_service)
            
        # Register service if not already registered
        if service_id not in self.llm_services:
            self.register_llm_service(service_id, llm_service)
        
        if current_model != recommended_model:
            logger.info(f"Task {task_type} requires model switch from {current_model} to {recommended_model}")
            # Queue the switch
            self.switch_queue.put_nowait((service_id, recommended_model, llm_service))
            return True
        
        return False
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model
        
        Args:
            model_name: Name of model (or current active model if None)
            
        Returns:
            Model information
        """
        return self.manager.get_model_info(model_name)
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models
        
        Returns:
            Dictionary of model information
        """
        return self.manager.get_all_models()
    
    def update_model_stats(self, model_name: str, stats: Dict[str, Any]) -> None:
        """Update performance statistics for a model
        
        Args:
            model_name: Name of the model
            stats: Performance statistics to update
        """
        self.manager.update_model_stats(model_name, stats)
