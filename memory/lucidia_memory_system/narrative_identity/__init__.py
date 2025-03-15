"""
Lucidia's Narrative Identity System

This package implements Lucidia's narrative identity system, which provides
a coherent sense of self over time through autobiographical memory, narrative
construction, and identity management.

Created by MEGAPROMPT (Daniel)
"""

from .narrative_identity import NarrativeIdentity
from .autobiographical_memory import AutobiographicalMemory
from .narrative_constructor import NarrativeTemplate, NarrativeConstructor
from .identity_manager import NarrativeIdentityManager
from .api import identity_router, add_identity_router

__all__ = [
    'NarrativeIdentity',
    'AutobiographicalMemory',
    'NarrativeTemplate',
    'NarrativeConstructor',
    'NarrativeIdentityManager',
    'identity_router',
    'add_identity_router'
]