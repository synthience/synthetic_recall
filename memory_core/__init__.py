# memory_core/__init__.py

"""Modular memory system for Lucid Recall"""

__version__ = "0.1.0"

from memory_core.enhanced_memory_client import EnhancedMemoryClient
from memory_core.memory_manager import MemoryManager

__all__ = ["EnhancedMemoryClient", "MemoryManager"]