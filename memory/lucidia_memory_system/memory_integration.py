"""
LUCID RECALL PROJECT
Memory Integration System

This module integrates the various memory components of Lucidia, including the
World Model, Self Model, and Knowledge Graph, with the persistent storage system.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Import core components
from .core.memory_core import MemoryCore
from .core.World.world_model import LucidiaWorldModel
from .core.Self.self_model import LucidiaSelfModel
from .core.knowledge_graph import LucidiaKnowledgeGraph
from ..storage.memory_persistence_handler import MemoryPersistenceHandler

logger = logging.getLogger(__name__)

class MemoryIntegration:
    """
    Integrates Lucidia's memory system components with persistence.
    
    This class serves as the coordination layer between:
    - MemoryCore (STM, LTM, MPL)
    - World Model (external knowledge and frameworks)
    - Self Model (identity and self-awareness)
    - Knowledge Graph (semantic network)
    - Persistence Handler (storage and retrieval)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the memory integration system.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = {
            'storage_path': Path('memory/stored'),
            'backup_interval': 3600,  # Seconds between backups
            'auto_persistence': True,  # Whether to automatically persist
            'persistence_interval': 300,  # Seconds between persistence operations
            'debug_mode': False,
            **(config or {})
        }
        
        logger.info(f"Initializing Memory Integration system at {self.config['storage_path']}")
        
        # Create storage paths
        self.world_model_path = self.config['storage_path'] / 'world_model'
        self.self_model_path = self.config['storage_path'] / 'self_model'
        self.knowledge_graph_path = self.config['storage_path'] / 'knowledge_graph'
        
        # Initialize persistence handler
        self.persistence_handler = MemoryPersistenceHandler(
            storage_path=self.config['storage_path'],
            config={
                'auto_backup': True,
                'backup_interval': self.config['backup_interval']
            }
        )
        
        # Initialize core memory components
        self.memory_core = MemoryCore({
            'memory_path': self.config['storage_path'],
            'enable_persistence': True
        })
        
        # Initialize World Model
        self.world_model = None
        
        # Initialize Self Model
        self.self_model = None
        
        # Initialize Knowledge Graph
        self.knowledge_graph = None
        
        # Coordination locks
        self._persistence_lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()
        
        # Last persistence timestamp
        self.last_persistence = time.time()
        
        # Stats
        self.stats = {
            'world_model_stores': 0,
            'self_model_stores': 0,
            'kg_stores': 0,
            'memory_integrations': 0,
            'last_persistence': None
        }
        
        logger.info("Memory Integration system initialized")
    
    async def initialize_components(self):
        """
        Initialize all memory system components and load data if available.
        """
        logger.info("Initializing all memory system components")
        
        # Initialize Self Model first (needed by World Model)
        self.self_model = LucidiaSelfModel()
        
        # Initialize World Model with reference to Self Model
        self.world_model = LucidiaWorldModel(self_model=self.self_model)
        
        # Initialize Knowledge Graph
        self.knowledge_graph = LucidiaKnowledgeGraph(self_model=self.self_model, world_model=self.world_model)
        
        # Initialize model imports in the knowledge graph
        logger.info("Triggering model imports into knowledge graph")
        await self.knowledge_graph.initialize_model_imports()
        logger.info("Model imports complete")
        
        # Load components from persistence if available
        await self.load_from_persistence()
        
        logger.info("All memory system components initialized")
        return True
    
    async def load_from_persistence(self):
        """
        Load all persisted data for World Model, Self Model, and Knowledge Graph.
        """
        async with self._load_lock:
            logger.info("Loading data from persistent storage")
            
            try:
                # Load World Model data
                world_model_data = await self._load_world_model_data()
                if world_model_data and self.world_model:
                    self._restore_world_model(world_model_data)
                    logger.info("World Model data loaded successfully")
                
                # Load Self Model data
                self_model_data = await self._load_self_model_data()
                if self_model_data and self.self_model:
                    self._restore_self_model(self_model_data)
                    logger.info("Self Model data loaded successfully")
                
                # Load Knowledge Graph data
                # First ensure the directory exists
                os.makedirs(self.knowledge_graph_path, exist_ok=True)
                kg_state_path = os.path.join(self.knowledge_graph_path, 'kg_state.json')
                
                if self.knowledge_graph and os.path.exists(kg_state_path):
                    success = self.knowledge_graph.load_state(kg_state_path)
                    if success:
                        logger.info("Knowledge Graph data loaded successfully")
                    else:
                        logger.warning("Failed to load Knowledge Graph data")
                
                logger.info("All components loaded from persistence")
                return True
            except Exception as e:
                logger.error(f"Error loading data from persistence: {e}")
                return False
    
    async def persist_all_components(self, force: bool = False):
        """
        Persist all components to storage.
        
        Args:
            force: Whether to force persistence regardless of interval
        """
        # Check if persistence is needed based on interval
        current_time = time.time()
        elapsed = current_time - self.last_persistence
        
        if not force and elapsed < self.config['persistence_interval']:
            logger.debug(f"Skipping persistence, last persisted {elapsed:.1f} seconds ago")
            return False
        
        async with self._persistence_lock:
            try:
                logger.info("Persisting all components to storage")
                
                # Persist World Model
                if self.world_model:
                    world_model_data = self._extract_world_model_data()
                    await self._persist_world_model_data(world_model_data)
                    self.stats['world_model_stores'] += 1
                
                # Persist Self Model
                if self.self_model:
                    self_model_data = self._extract_self_model_data()
                    await self._persist_self_model_data(self_model_data)
                    self.stats['self_model_stores'] += 1
                
                # Persist Knowledge Graph
                if self.knowledge_graph:
                    # Ensure the directory exists
                    os.makedirs(self.knowledge_graph_path, exist_ok=True)
                    kg_state_path = os.path.join(self.knowledge_graph_path, 'kg_state.json')
                    
                    success = self.knowledge_graph.save_state(kg_state_path)
                    if success:
                        self.stats['kg_stores'] += 1
                        logger.info("Knowledge Graph persisted successfully")
                    else:
                        logger.warning("Failed to persist Knowledge Graph")
                
                # Force a backup of the memory core as well
                await self.memory_core.force_backup()
                
                # Update persistence timestamp
                self.last_persistence = current_time
                self.stats['last_persistence'] = current_time
                
                logger.info("All components persisted successfully")
                return True
            except Exception as e:
                logger.error(f"Error persisting components: {e}")
                return False
    
    def _extract_world_model_data(self) -> Dict[str, Any]:
        """
        Extract serializable data from the World Model.
        """
        return {
            'reality_framework': self.world_model.reality_framework,
            'knowledge_domains': self.world_model.knowledge_domains,
            'conceptual_networks': self.world_model.conceptual_networks,
            'epistemological_framework': self.world_model.epistemological_framework,
            'belief_system': self.world_model.belief_system,
            'verification_methods': self.world_model.verification_methods,
            'causal_models': self.world_model.causal_models,
            'version': self.world_model.version,
            'last_update': time.time()
        }
    
    def _restore_world_model(self, data: Dict[str, Any]):
        """
        Restore World Model from persisted data.
        """
        if not data:
            return
        
        # Update World Model attributes
        self.world_model.reality_framework = data.get('reality_framework', self.world_model.reality_framework)
        self.world_model.knowledge_domains = data.get('knowledge_domains', self.world_model.knowledge_domains)
        self.world_model.conceptual_networks = data.get('conceptual_networks', self.world_model.conceptual_networks)
        self.world_model.epistemological_framework = data.get('epistemological_framework', self.world_model.epistemological_framework)
        self.world_model.belief_system = data.get('belief_system', self.world_model.belief_system)
        self.world_model.verification_methods = data.get('verification_methods', self.world_model.verification_methods)
        self.world_model.causal_models = data.get('causal_models', self.world_model.causal_models)
        self.world_model.version = data.get('version', self.world_model.version)
    
    def _extract_self_model_data(self) -> Dict[str, Any]:
        """
        Extract serializable data from the Self Model.
        """
        return {
            'identity': self.self_model.identity,
            'self_awareness': self.self_model.self_awareness,
            'core_awareness': self.self_model.core_awareness,
            'personality': dict(self.self_model.personality),
            'core_values': self.self_model.core_values,
            'emotional_state': self.self_model.emotional_state,
            'reflective_capacity': self.self_model.reflective_capacity,
            'version': self.self_model.identity.get('version', '1.0'),
            'last_update': time.time()
        }
    
    def _restore_self_model(self, data: Dict[str, Any]):
        """
        Restore Self Model from persisted data.
        """
        if not data:
            return
        
        # Update Self Model attributes
        self.self_model.identity = data.get('identity', self.self_model.identity)
        self.self_model.self_awareness = data.get('self_awareness', self.self_model.self_awareness)
        self.self_model.core_awareness = data.get('core_awareness', self.self_model.core_awareness)
        
        # Handle personality dictionary with defaultdict
        if 'personality' in data:
            personality_dict = data['personality']
            self.self_model.personality.clear()
            for key, value in personality_dict.items():
                self.self_model.personality[key] = value
        
        self.self_model.core_values = data.get('core_values', self.self_model.core_values)
        self.self_model.emotional_state = data.get('emotional_state', self.self_model.emotional_state)
        self.self_model.reflective_capacity = data.get('reflective_capacity', self.self_model.reflective_capacity)
    
    async def _persist_world_model_data(self, data: Dict[str, Any]):
        """
        Persist World Model data using the persistence handler.
        """
        world_model_memory = {
            'content': json.dumps(data),
            'metadata': {
                'world_model_version': data.get('version', '1.0'),
                'timestamp': time.time(),
                'type': 'world_model'
            }
        }
        
        return await self.persistence_handler.store_memory(
            memory_data=world_model_memory,
            storage_key='world_model'
        )
    
    async def _persist_self_model_data(self, data: Dict[str, Any]):
        """
        Persist Self Model data using the persistence handler.
        """
        self_model_memory = {
            'content': json.dumps(data),
            'metadata': {
                'self_model_version': data.get('version', '1.0'),
                'timestamp': time.time(),
                'type': 'self_model'
            }
        }
        
        return await self.persistence_handler.store_memory(
            memory_data=self_model_memory,
            storage_key='self_model'
        )
    
    async def _load_world_model_data(self) -> Optional[Dict[str, Any]]:
        """
        Load World Model data from persistence.
        """
        try:
            memory = await self.persistence_handler.retrieve_memory('world_model')
            if memory and 'content' in memory:
                return json.loads(memory['content'])
            return None
        except Exception as e:
            logger.error(f"Error loading World Model data: {e}")
            return None
    
    async def _load_self_model_data(self) -> Optional[Dict[str, Any]]:
        """
        Load Self Model data from persistence.
        """
        try:
            memory = await self.persistence_handler.retrieve_memory('self_model')
            if memory and 'content' in memory:
                return json.loads(memory['content'])
            return None
        except Exception as e:
            logger.error(f"Error loading Self Model data: {e}")
            return None
    
    async def _load_knowledge_graph_data(self) -> Optional[Dict[str, Any]]:
        """
        Load Knowledge Graph data from persistence.
        """
        try:
            memory = await self.persistence_handler.retrieve_memory('knowledge_graph')
            if memory and 'content' in memory:
                return json.loads(memory['content'])
            return None
        except Exception as e:
            logger.error(f"Error loading Knowledge Graph data: {e}")
            return None
    
    async def integrate_dream_insights(self, dream_fragments: List[Dict[str, Any]]):
        """
        Integrate insights from dream reflection into the memory system.
        
        Args:
            dream_fragments: List of dream fragments with insights
        """
        logger.info(f"Integrating {len(dream_fragments)} dream insights into memory system")
        
        # Process each fragment type appropriately
        for fragment in dream_fragments:
            fragment_type = fragment.get('type', 'unknown')
            content = fragment.get('content', '')
            
            if not content:
                continue
                
            # Process based on fragment type
            if fragment_type == 'insight':
                # Store insight in memory and update world model
                await self.memory_core.process_and_store(
                    content=content,
                    metadata={
                        'source': 'dream_insight',
                        'timestamp': time.time(),
                        'fragment_id': fragment.get('id')
                    }
                )
                
                # Update World Model's conceptual networks
                self.world_model.update_conceptual_network_from_insight(content)
                
            elif fragment_type == 'question':
                # Store question for future exploration
                self.world_model.add_exploration_question(content)
                
            elif fragment_type == 'hypothesis':
                # Add hypothesis to belief system with low confidence
                self.world_model.add_hypothesis(content)
                
            elif fragment_type == 'counterfactual':
                # Update causal models with counterfactual
                self.world_model.update_causal_model_from_counterfactual(content)
        
        # Persist updates
        await self.persist_all_components(force=True)
        self.stats['memory_integrations'] += 1
        
        return True
    
    async def scheduled_persistence(self):
        """
        Run persistence operations on a schedule.
        """
        while self.config['auto_persistence']:
            try:
                # Persist all components
                await self.persist_all_components()
                
                # Sleep until next persistence interval
                await asyncio.sleep(self.config['persistence_interval'])
            except Exception as e:
                logger.error(f"Error in scheduled persistence: {e}")
                await asyncio.sleep(60)  # Sleep on error and retry
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory integration system.
        """
        core_stats = self.memory_core.get_stats() if self.memory_core else {}
        
        return {
            **self.stats,
            'memory_core': core_stats,
            'uptime': time.time() - self.stats.get('last_persistence', time.time()) if self.stats.get('last_persistence') else 0
        }
