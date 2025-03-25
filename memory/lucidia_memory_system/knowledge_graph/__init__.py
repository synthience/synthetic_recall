"""Modular Knowledge Graph Architecture for Lucidia Memory System"""

# Core system components
from .core import EventBus, ModuleRegistry, LucidiaKnowledgeGraph

# Base module framework
from .base_module import KnowledgeGraphModule

# Module implementations
from .core_graph_manager import CoreGraphManager
from .embedding_manager import EmbeddingManager
from .visualization_manager import VisualizationManager
from .dream_integration_module import DreamIntegrationModule
from .emotional_context_manager import EmotionalContextManager
from .contradiction_manager import ContradictionManager
from .maintenance_manager import MaintenanceManager
from .API_manager import APIManager
