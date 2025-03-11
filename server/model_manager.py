"""Model Manager for Lucidia

This module implements dynamic model selection and switching based on system state,
task requirements, and resource availability.
"""

import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger("ModelManager")

class ModelPurpose(Enum):
    """Enum for different model purposes"""
    GENERAL = "general"               # General purpose interactions
    REASONING = "reasoning"           # Complex reasoning tasks
    CREATIVE = "creative"             # Creative and generative tasks
    ANALYSIS = "analysis"             # Deep analysis of information
    EMBEDDING = "embedding"           # Text embedding generation
    MEMORY_PROCESSING = "memory"      # Memory-related processing
    DREAMING = "dreaming"             # Dream generation and processing
    REFLECTION = "reflection"         # Self-reflection and improvement

@dataclass
class ModelProfile:
    """Profile for an available model"""
    name: str                          # Model identifier
    purposes: List[ModelPurpose]       # Suitable purposes
    context_length: int                # Maximum context length
    strength: float                    # Overall capability (0-1)
    speed: float                       # Relative speed (0-1)
    resource_usage: Dict[str, float]   # Resource usage metrics
    endpoint: Optional[str] = None     # Custom endpoint if different from default
    
    @property
    def efficiency(self) -> float:
        """Calculate model efficiency (strength/resource_usage)"""
        # Resource usage average (CPU, memory, etc.)
        avg_resource = sum(self.resource_usage.values()) / len(self.resource_usage) if self.resource_usage else 1.0
        return self.strength / avg_resource if avg_resource > 0 else 0

class SystemState(Enum):
    """System state indicators"""
    IDLE = "idle"                      # System is idle
    ACTIVE = "active"                  # System is actively processing
    DREAMING = "dreaming"              # System is in dreaming mode
    LOW_RESOURCES = "low_resources"    # System resources are constrained
    HIGH_RESOURCES = "high_resources"  # System has abundant resources

class ModelManager:
    """Manages dynamic model selection and switching based on system state"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the model manager
        
        Args:
            config_path: Path to configuration file with model definitions
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "config", 
            "models.json"
        )
        self.models: Dict[str, ModelProfile] = {}
        self.default_model = os.getenv('DEFAULT_MODEL', 'qwen2.5-7b-instruct')
        self.active_model = self.default_model
        self.last_switch_time = time.time()
        self.switch_cooldown = 60  # seconds between model switches
        self.system_state = SystemState.IDLE
        self.cached_llm_services = {}
        self._lock = asyncio.Lock()
        
        # Load model configurations
        self._load_model_configs()
        logger.info(f"Model Manager initialized with {len(self.models)} models")
        
    def _load_model_configs(self) -> None:
        """Load model configurations from file"""
        try:
            # Create config directory if it doesn't exist
            config_dir = os.path.dirname(self.config_path)
            os.makedirs(config_dir, exist_ok=True)
            
            # If config doesn't exist, create default config
            if not os.path.exists(self.config_path):
                logger.warning(f"Model configuration file not found at {self.config_path}, creating default")
                self._create_default_config()
                
            # Try to load configuration from file
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Validate config_data is a dictionary
                if not isinstance(config_data, dict):
                    logger.error(f"Invalid configuration format: expected dictionary, got {type(config_data)}")
                    raise ValueError("Invalid configuration format")
                    
                # Process model definitions
                models_data = config_data.get('models', {})
                if not models_data:
                    logger.warning("No models defined in configuration, using defaults")
                    self._create_default_models()
                    return
                    
                for model_name, model_data in models_data.items():
                    purposes = [ModelPurpose(p) for p in model_data.get('purposes', ['general'])]
                    
                    self.models[model_name] = ModelProfile(
                        name=model_name,
                        purposes=purposes,
                        context_length=model_data.get('context_length', 4096),
                        strength=model_data.get('strength', 0.7),
                        speed=model_data.get('speed', 0.7),
                        resource_usage=model_data.get('resource_usage', {'memory': 0.5, 'cpu': 0.5}),
                        endpoint=model_data.get('endpoint', None)
                    )
                
                # Set default model if specified in config
                if 'default_model' in config_data and config_data['default_model'] in self.models:
                    self.default_model = config_data['default_model']
                    self.active_model = self.default_model
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing model configuration: {e}")
                self._create_default_models()
            except FileNotFoundError:
                logger.error(f"Configuration file not found at {self.config_path} after creation attempt")
                self._create_default_models()
                
        except Exception as e:
            logger.error(f"Error loading model configurations: {e}")
            # Create default models if loading failed
            self._create_default_models()
    
    def _create_default_config(self) -> None:
        """Create default model configuration file"""
        default_config = {
            "models": {
                "qwen2.5-7b-instruct": {
                    "purposes": ["general", "reasoning", "creative"],
                    "context_length": 8192,
                    "strength": 0.8,
                    "speed": 0.7,
                    "resource_usage": {"memory": 0.7, "cpu": 0.7}
                },
                "phi-3.1-mini-128k-instruct": {
                    "purposes": ["general", "memory", "reflection"],
                    "context_length": 4096,
                    "strength": 0.6,
                    "speed": 0.9,
                    "resource_usage": {"memory": 0.3, "cpu": 0.3}
                },
                "gemma2-9b-it": {
                    "purposes": ["reasoning", "analysis", "dreaming"],
                    "context_length": 8192,
                    "strength": 0.85,
                    "speed": 0.6,
                    "resource_usage": {"memory": 0.8, "cpu": 0.8}
                },
                "mistral-nemo": {
                    "purposes": ["creative", "dreaming", "reflection"],
                    "context_length": 16384,
                    "strength": 0.9,
                    "speed": 0.5,
                    "resource_usage": {"memory": 0.9, "cpu": 0.9}
                },
                "all-MiniLM-L6-v2": {
                    "purposes": ["embedding"],
                    "context_length": 512,
                    "strength": 0.7,
                    "speed": 0.95,
                    "resource_usage": {"memory": 0.2, "cpu": 0.2}
                }
            },
            "default_model": "qwen2.5-7b-instruct",
            "parameters": {
                "switch_cooldown": 60
            }
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default model configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error creating default configuration: {e}")
    
    def _create_default_models(self) -> None:
        """Create default models in memory if config loading fails"""
        self.models = {
            "qwen2.5-7b-instruct": ModelProfile(
                name="qwen2.5-7b-instruct",
                purposes=[ModelPurpose.GENERAL, ModelPurpose.REASONING, ModelPurpose.CREATIVE],
                context_length=8192,
                strength=0.8,
                speed=0.7,
                resource_usage={"memory": 0.7, "cpu": 0.7}
            ),
            "phi-3.1-mini-128k-instruct": ModelProfile(
                name="phi-3.1-mini-128k-instruct",
                purposes=[ModelPurpose.GENERAL, ModelPurpose.MEMORY_PROCESSING, ModelPurpose.REFLECTION],
                context_length=4096,
                strength=0.6,
                speed=0.9,
                resource_usage={"memory": 0.3, "cpu": 0.3}
            ),
            "all-MiniLM-L6-v2": ModelProfile(
                name="all-MiniLM-L6-v2",
                purposes=[ModelPurpose.EMBEDDING],
                context_length=512,
                strength=0.7,
                speed=0.95,
                resource_usage={"memory": 0.2, "cpu": 0.2}
            )
        }
    
    def update_system_state(self, state: SystemState) -> None:
        """Update the current system state
        
        Args:
            state: New system state
        """
        if self.system_state != state:
            logger.info(f"System state changed from {self.system_state.value} to {state.value}")
            self.system_state = state
            
            # Consider switching model on state change
            recommended_model = self.get_recommended_model()
            if recommended_model != self.active_model:
                logger.info(f"Recommended model {recommended_model} differs from active model {self.active_model}")
        
    def get_recommended_model(self, purpose: Optional[ModelPurpose] = None) -> str:
        """Get the recommended model based on current state and purpose
        
        Args:
            purpose: Specific purpose for model selection
            
        Returns:
            Name of recommended model
        """
        if not self.models:
            return self.default_model
            
        # Use default purpose based on system state if not specified
        if purpose is None:
            if self.system_state == SystemState.DREAMING:
                purpose = ModelPurpose.DREAMING
            elif self.system_state == SystemState.IDLE:
                purpose = ModelPurpose.REFLECTION
            else:
                purpose = ModelPurpose.GENERAL
        
        # Filter models suitable for the purpose
        suitable_models = [
            model for model in self.models.values() 
            if purpose in model.purposes
        ]
        
        if not suitable_models:
            logger.warning(f"No models found for purpose {purpose.value}, using default")
            return self.default_model
        
        # Calculate scores based on state and requirements
        scores = {}
        for model in suitable_models:
            base_score = model.strength
            
            # Adjust score based on system state
            if self.system_state == SystemState.LOW_RESOURCES:
                # Prioritize efficiency and speed
                base_score = (base_score * 0.3) + (model.efficiency * 0.4) + (model.speed * 0.3)
            elif self.system_state == SystemState.DREAMING:
                # Prioritize strength over speed
                base_score = (base_score * 0.7) + (model.speed * 0.3)
            elif self.system_state == SystemState.IDLE:
                # Balance all factors
                base_score = (base_score * 0.5) + (model.efficiency * 0.3) + (model.speed * 0.2)
            
            scores[model.name] = base_score
        
        # Find model with highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return self.default_model
    
    async def switch_model(self, target_model: str, llm_service) -> bool:
        """Switch to the specified model
        
        Args:
            target_model: Name of model to switch to
            llm_service: LLM service instance to update
            
        Returns:
            Success status
        """
        if target_model not in self.models:
            logger.warning(f"Model {target_model} not found in available models")
            return False
            
        current_time = time.time()
        if current_time - self.last_switch_time < self.switch_cooldown:
            logger.info(f"Model switch cooling down, will retry later")
            return False
            
        async with self._lock:
            try:
                logger.info(f"Switching model from {self.active_model} to {target_model}")
                
                # Update the llm_service model
                old_model = llm_service.model
                model_profile = self.models[target_model]
                
                # Set custom endpoint if specified
                if model_profile.endpoint:
                    llm_service.base_url = model_profile.endpoint
                    
                llm_service.model = target_model
                
                # Test the new model connection
                connection_success = await llm_service.initialize()
                
                if connection_success:
                    self.active_model = target_model
                    self.last_switch_time = current_time
                    logger.info(f"Successfully switched to model {target_model}")
                    return True
                else:
                    # Revert back if connection failed
                    llm_service.model = old_model
                    logger.error(f"Failed to switch to model {target_model}, reverted to {old_model}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error switching model: {e}")
                return False
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model or active model
        
        Args:
            model_name: Name of model to get info about (or active model if None)
            
        Returns:
            Model information dictionary
        """
        target_model = model_name or self.active_model
        
        if target_model not in self.models:
            return {
                "name": target_model,
                "status": "unknown",
                "purposes": ["general"],
                "active": target_model == self.active_model
            }
            
        model = self.models[target_model]
        return {
            "name": model.name,
            "status": "available",
            "purposes": [p.value for p in model.purposes],
            "context_length": model.context_length,
            "strength": model.strength,
            "speed": model.speed,
            "efficiency": model.efficiency,
            "resource_usage": model.resource_usage,
            "active": model.name == self.active_model
        }
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models
        
        Returns:
            Dictionary of model information
        """
        return {name: self.get_model_info(name) for name in self.models}
    
    def update_model_stats(self, model_name: str, stats: Dict[str, Any]) -> None:
        """Update performance statistics for a model
        
        Args:
            model_name: Name of the model
            stats: Performance statistics to update
        """
        if model_name not in self.models:
            logger.warning(f"Cannot update stats for unknown model {model_name}")
            return
            
        # Update relevant model profile parameters based on stats
        if "response_time" in stats:
            # Update speed based on response time (inversely proportional)
            response_time = stats["response_time"]
            if response_time > 0:
                # Smooth update to avoid large fluctuations
                new_speed = min(1.0, 5.0 / response_time)  # 5 second response = 1.0 speed
                self.models[model_name].speed = (self.models[model_name].speed * 0.7) + (new_speed * 0.3)
                
        if "resource_usage" in stats:
            # Update resource usage metrics
            for resource, usage in stats["resource_usage"].items():
                if resource in self.models[model_name].resource_usage:
                    # Smooth update
                    current = self.models[model_name].resource_usage[resource]
                    self.models[model_name].resource_usage[resource] = (current * 0.7) + (usage * 0.3)
