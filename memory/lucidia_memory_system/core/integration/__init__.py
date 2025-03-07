"""
LUCID RECALL PROJECT
Memory Module

The core memory system for Lucidia with hierarchical architecture
for efficient, self-organizing memory.
"""

__version__ = "0.2.0"

# Import core components
from ..memory_types import MemoryTypes, MemoryEntry
from ..short_term_memory import ShortTermMemory
from ..long_term_memory import LongTermMemory
from ..memory_prioritization_layer import MemoryPrioritizationLayer
from ..embedding_comparator import EmbeddingComparator
from ....storage.memory_persistence_handler import MemoryPersistenceHandler
from ..memory_core import MemoryCore as EnhancedMemoryCore  # Aliasing existing core as enhanced for now
from .memory_integration import MemoryIntegration
from .updated_hpc_client import EnhancedHPCClient

# Export public API
__all__ = [
    'MemoryTypes', 
    'MemoryEntry',
    'ShortTermMemory',
    'LongTermMemory',
    'MemoryPrioritizationLayer',
    'EmbeddingComparator',
    'MemoryPersistenceHandler',
    'EnhancedMemoryCore',
    'MemoryIntegration',
    'EnhancedHPCClient'
]

def create_memory_system(config=None):
    """
    Factory function to create a pre-configured memory system.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        MemoryIntegration instance
    """
    return MemoryIntegration(config)