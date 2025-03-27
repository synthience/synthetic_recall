# synthians_memory_core/__init__.py

"""
Synthians Memory Core - A Unified, Efficient Memory System
Incorporates HPC-QuickRecal, Hyperbolic Geometry, Emotional Intelligence,
Memory Assemblies, and Adaptive Thresholds.
"""

__version__ = "1.0.0"

# Core components
from .synthians_memory_core import SynthiansMemoryCore
from .memory_structures import MemoryEntry, MemoryAssembly
from .hpc_quickrecal import UnifiedQuickRecallCalculator, QuickRecallMode, QuickRecallFactor
from .geometry_manager import GeometryManager, GeometryType
from .emotional_intelligence import EmotionalAnalyzer, EmotionalGatingService
from .memory_persistence import MemoryPersistence
from .adaptive_components import ThresholdCalibrator

__all__ = [
    "SynthiansMemoryCore",
    "MemoryEntry",
    "MemoryAssembly",
    "UnifiedQuickRecallCalculator",
    "QuickRecallMode",
    "QuickRecallFactor",
    "GeometryManager",
    "GeometryType",
    "EmotionalAnalyzer",
    "EmotionalGatingService",
    "MemoryPersistence",
    "ThresholdCalibrator",
]
