"""
LUCIDIA SYSTEM
Memory Bridge

Bridge between the flat memory system and hierarchical memory system.
Allows leveraging the capabilities of both for optimal memory management.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import os
import json
import time

# Import flat memory system
from server.memory_system import MemorySystem as FlatMemorySystem

# DO NOT import these directly to avoid circular imports
# We'll import them dynamically when needed
# from memory.lucidia_memory_system.core.memory_core import MemoryCore as HierarchicalMemoryCore
# from memory.lucidia_memory_system.core.integration.hpc_sig_flow_manager import HPCSIGFlowManager
# from memory.lucidia_memory_system.core.memory_types import MemoryTypes, MemoryEntry

logger = logging.getLogger(__name__)

class MemoryBridge:
    """
    Bridge between the flat and hierarchical memory systems.
    
    This component connects the flat memory system (server/memory_system.py) with
    the hierarchical memory system (memory/lucidia_memory_system/core/memory_core.py).
    It allows leveraging the capabilities of both systems:
    - Flat system: Simple, robust memory storage with HPC integration
    - Hierarchical system: STM, LTM, MPL layers for memory organization
    """
    
    def __init__(self, memory_system: FlatMemorySystem, config: Dict[str, Any] = None):
        """
        Initialize the memory bridge.
        
        Args:
            memory_system: The existing flat memory system
            config: Optional configuration dictionary
        """
        self.config = {
            'memory_path': Path('/app/memory/stored'),
            'embedding_dim': 384,
            'enable_migration': True,  # Auto-migrate memories between systems
            'migration_batch_size': 20, # Increased from 10 to 20 for more efficient migration
            'migration_interval': 300, # Time between migrations in seconds (5 minutes)
            'hierarchical_significance_threshold': 0.5,  # Minimum significance for hierarchical storage
            **(config or {})
        }
        
        # Store references to both memory systems
        self.flat_memory = memory_system
        
        # Import hierarchical memory dynamically to avoid circular imports
        try:
            # Dynamically import the hierarchical memory system
            from memory.lucidia_memory_system.core.memory_core import MemoryCore
            from memory.lucidia_memory_system.core.memory_types import MemoryTypes
            
            # Initialize hierarchical memory with consistent paths
            logger.info(f"Initializing hierarchical memory core with path: {self.config['memory_path']}")
            self.hierarchical_memory = MemoryCore({
                'memory_path': self.config['memory_path'],
                'embedding_dim': self.config['embedding_dim']
            })
            
            # Store the memory types for later use
            self.memory_types = MemoryTypes
            self.has_hierarchical = True
            
        except ImportError as e:
            logger.warning(f"Could not import hierarchical memory system: {e}")
            logger.warning("Running with flat memory system only")
            self.hierarchical_memory = None
            self.has_hierarchical = False
        except Exception as e:
            logger.error(f"Error initializing hierarchical memory: {e}")
            self.hierarchical_memory = None
            self.has_hierarchical = False
        
        # Track last migration time
        self.last_migration = time.time()
        
        # Initialize statistics
        self.stats = {
            'flat_memories': len(self.flat_memory.memories),
            'hierarchical_memories': {
                'stm': 0,
                'ltm': 0,
                'mpl': 0
            },
            'migrations': 0,
            'shared_memories': 0
        }
        
        # Update stats for hierarchical memory
        self._update_hierarchical_stats()
        
        logger.info(f"Memory Bridge initialized with stats: {self.stats}")
        
        # Start background tasks if enabled
        if self.config['enable_migration']:
            self._migration_task = asyncio.create_task(self._periodic_migration())
            logger.info("Started background memory migration task")
    
    async def _periodic_migration(self):
        """
        Periodically migrate memories between systems.
        """
        while True:
            try:
                # Wait for the configured interval
                await asyncio.sleep(self.config['migration_interval'])
                
                # Perform migration
                await self.migrate_memories()
                
                # Update last migration time
                self.last_migration = time.time()
                
            except asyncio.CancelledError:
                logger.info("Migration task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in periodic migration: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    async def migrate_memories(self):
        """
        Migrate memories between the flat and hierarchical systems.
        
        This ensures that significant memories from the flat system are properly
        categorized in the hierarchical system, and vice versa.
        """
        logger.info("Starting memory migration...")
        migrated = 0
        batch_size = self.config['migration_batch_size']
        
        # 1. First, ensure all flat memories are in hierarchical system
        flat_memories = self.flat_memory.memories
        logger.info(f"Checking {len(flat_memories)} flat memories for migration to hierarchical system")
        
        # Create a list of eligible memories for migration
        eligible_memories = []
        for memory in flat_memories:
            # Skip if already migrated or below threshold
            if memory.get('migrated_to_hierarchical'):
                continue
            
            # Extract content and calculate significance if not present
            text = memory.get('text', memory.get('content', ''))
            if not text:
                continue
            
            significance = memory.get('significance', memory.get('importance', 0.5))
            
            # Skip if below threshold
            if significance < self.config['hierarchical_significance_threshold']:
                continue
                    
            # Get or create embedding
            embedding = memory.get('embedding')
            
            # Check if embedding is valid
            is_valid_embedding = False
            has_placeholder = memory.get('has_placeholder_embedding', False)
            
            if embedding is not None:
                if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                    is_valid_embedding = True
                elif isinstance(embedding, str):
                    try:
                        embedding_data = json.loads(embedding)
                        if isinstance(embedding_data, list) and all(isinstance(x, (int, float)) for x in embedding_data):
                            # Update the memory with the parsed embedding
                            memory['embedding'] = embedding_data
                            is_valid_embedding = True
                    except (json.JSONDecodeError, TypeError):
                        # Invalid JSON string or other error
                        pass
            
            # Only add memories with valid embeddings
            if is_valid_embedding:
                # If this is a placeholder embedding, mark it for special handling
                priority_boost = 0
                if has_placeholder:
                    logger.info(f"Including memory {memory.get('id', 'unknown')} with placeholder embedding for migration")
                    # Lower priority for placeholder embeddings unless they are relationship memories
                    if not ('relationship' in memory.get('metadata', {}) or 'human_name' in memory.get('metadata', {})):
                        priority_boost = -0.1
                
                # Add to eligible memories
                eligible_memories.append({
                    'memory': memory,
                    'significance': significance + priority_boost
                })
            else:
                logger.warning(f"Memory {memory.get('id', 'unknown')} skipped during migration due to invalid embedding.")
        
        # Sort by significance (highest first) and limit to batch size
        eligible_memories.sort(key=lambda x: x['significance'], reverse=True)
        batch_memories = eligible_memories[:batch_size]
        
        logger.info(f"Found {len(eligible_memories)} eligible memories for migration, processing batch of {len(batch_memories)}")
        
        # Process the batch
        for item in batch_memories:
            memory = item['memory']
            try:
                # Extract necessary data
                text = memory.get('text', memory.get('content', ''))
                significance = memory.get('significance', memory.get('importance', 0.5))
                metadata = memory.get('metadata', {})
                embedded_vector = memory.get('embedding')
                
                try:
                    # Import relevant modules dynamically to avoid circular imports
                    from memory.lucidia_memory_system.core.memory_types import MemoryTypes
                    
                    # Ensure memory is forced to persist in LTM
                    memory_type = MemoryTypes.SEMANTIC  # Changed from DECLARATIVE to SEMANTIC
                    
                    # Determine memory type based on metadata
                    if 'relationship' in metadata or 'human_name' in metadata:
                        memory_type = MemoryTypes.RELATIONSHIP
                    elif 'emotional' in metadata or metadata.get('emotions'):
                        memory_type = MemoryTypes.EMOTIONAL
                    elif metadata.get('is_personal', False):
                        memory_type = MemoryTypes.PERSONAL
                    elif metadata.get('is_important', False) or significance > 0.8:
                        memory_type = MemoryTypes.IMPORTANT
                    
                    # Store in hierarchical memory with persistence flag
                    await self.hierarchical_memory.process_and_store(
                        content=text,
                        embedding=embedded_vector,
                        memory_id=memory['id'],
                        memory_type=memory_type,
                        significance=significance,
                        metadata=metadata,
                        force_ltm=True  # Force storage in LTM
                    )
                    migrated += 1
                    logger.info(f"Migrated memory {memory['id']} to hierarchical system")
                    
                    # Update migration flag in flat memory to prevent re-migration
                    memory['migrated_to_hierarchical'] = True
                    
                    # Clear placeholder flag if it exists
                    if 'has_placeholder_embedding' in memory:
                        # If we had a placeholder, we've now migrated with the actual embedding
                        logger.info(f"Clearing placeholder flag for migrated memory {memory['id']}")
                        memory.pop('has_placeholder_embedding', None)
                    
                    # Update the stored memory in flat system
                    try:
                        # Explicitly save the memory to disk to ensure the migrated flag is persisted
                        self.flat_memory._save_memory(memory)
                        logger.info(f"Updated migration status for memory {memory['id']} in flat system")
                    except Exception as save_error:
                        logger.error(f"Failed to update migration status for memory {memory['id']}: {save_error}")
                except Exception as inner_e:
                    # If we get a DECLARATIVE error, try again with SEMANTIC type
                    if "DECLARATIVE" in str(inner_e):
                        logger.warning(f"Retrying migration for memory {memory.get('id', 'unknown')} with SEMANTIC type")
                        from memory.lucidia_memory_system.core.memory_types import MemoryTypes
                        await self.hierarchical_memory.process_and_store(
                            content=text,
                            embedding=embedded_vector,
                            memory_id=memory['id'],
                            memory_type=MemoryTypes.SEMANTIC,  # Force SEMANTIC type for problematic memories
                            significance=significance,
                            metadata=metadata,
                            force_ltm=True  # Force storage in LTM
                        )
                        migrated += 1
                        logger.info(f"Successfully migrated memory {memory['id']} to hierarchical system after retry")
                        
                        # Update migration flag in flat memory to prevent re-migration
                        memory['migrated_to_hierarchical'] = True
                        # Update the stored memory in flat system
                        self.flat_memory._save_memory(memory)
                    else:
                        # Re-raise if it's not a DECLARATIVE error
                        raise inner_e
                
            except Exception as e:
                logger.error(f"Error migrating memory {memory.get('id', 'unknown')}: {e}")
        
        # 2. Check hierarchical memories that should be in flat system
        if self.has_hierarchical and hasattr(self.hierarchical_memory, 'ltm') and hasattr(self.hierarchical_memory.ltm, 'get_all_memories'):
            try:
                # Get all LTM memories
                ltm_memories = await self.hierarchical_memory.ltm.get_all_memories()
                logger.info(f"Checking {len(ltm_memories)} hierarchical memories for migration to flat system")
                
                for h_memory in ltm_memories:
                    try:
                        # Check if already in flat system
                        mem_id = h_memory.id
                        exists_in_flat = any(m.get('id') == mem_id for m in self.flat_memory.memories)
                        
                        if not exists_in_flat:
                            # Convert to flat memory format
                            flat_mem = {
                                'id': mem_id,
                                'text': h_memory.content,
                                'embedding': h_memory.embedding.tolist() if hasattr(h_memory.embedding, 'tolist') else h_memory.embedding,
                                'timestamp': time.time(),
                                'significance': h_memory.significance,
                                'metadata': {
                                    **h_memory.metadata,
                                    'source': 'hierarchical',
                                    'persisted': True
                                }
                            }
                            
                            # Save to flat system
                            self.flat_memory.memories.append(flat_mem)
                            self.flat_memory._save_memory(flat_mem)
                            migrated += 1
                            logger.info(f"Migrated hierarchical memory {mem_id} to flat system")
                    except Exception as e:
                        logger.error(f"Error processing hierarchical memory: {e}")
            except Exception as e:
                logger.error(f"Error retrieving hierarchical memories: {e}")
        
        # Update stats after migration
        self._update_flat_stats()
        self._update_hierarchical_stats()
        
        logger.info(f"Memory migration complete. Migrated {migrated} memories.")
        return migrated
    
    async def store_memory(self, content: str, significance: float = 0.5, metadata: Optional[Dict[str, Any]] = None,
                         categories: Optional[List[str]] = None, embedding = None):
        """
        Store a memory in both systems based on significance.
        
        Args:
            content: The memory content
            significance: Memory significance (0.0-1.0)
            metadata: Optional additional metadata
            categories: Optional list of categories
            embedding: Optional pre-computed embedding
            
        Returns:
            Memory ID if stored, None if rejected
        """
        # Prepare metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Prepare categories if not provided
        if categories is None:
            categories = []
            
        # Auto-detect relationship memories and boost their significance
        is_relationship_memory = False
        if 'relationship' in metadata or 'human_name' in metadata:
            is_relationship_memory = True
            logger.info(f"Detected relationship memory for {metadata.get('human_name', 'unknown')}")
            # Ensure relationship memories are highly significant
            significance = max(significance, 0.9)
            # Add relationship category if not already present
            if 'relationship' not in categories:
                categories.append('relationship')
                
        # Check if this might be a JSON-formatted relationship memory
        if not is_relationship_memory and content.strip().startswith('{') and content.strip().endswith('}'):
            try:
                # Try to parse as JSON
                memory_json = json.loads(content)
                if 'relationship' in memory_json or 'human_name' in memory_json:
                    is_relationship_memory = True
                    logger.info(f"Detected JSON relationship memory for {memory_json.get('human_name', 'unknown')}")
                    # Ensure relationship memories are highly significant
                    significance = max(significance, 0.9)
                    # Add relationship category if not already present
                    if 'relationship' not in categories:
                        categories.append('relationship')
                    # Add key fields to metadata
                    for key in ['human_name', 'relationship']:
                        if key in memory_json and key not in metadata:
                            metadata[key] = memory_json[key]
            except json.JSONDecodeError:
                # Not valid JSON, that's okay
                pass
                
        # Store in flat memory system first (always)
        flat_id = self.flat_memory.store_memory(
            content=content,
            significance=significance,
            metadata=metadata,
            embedding=embedding
        )
        
        # Store in hierarchical memory if available
        hier_id = None
        if self.has_hierarchical:
            try:
                logger.info(f"Storing memory in hierarchical system with significance {significance}")
                hier_id = await self.hierarchical_memory.process_and_store(
                    content=content,
                    memory_type=self.memory_types.EPISODIC,
                    significance=significance,
                    metadata=metadata,
                    embedding=embedding,
                    categories=categories
                )
            except Exception as e:
                logger.error(f"Error storing in hierarchical memory: {e}")
        else:
            logger.info("Hierarchical memory not available, storing only in flat memory")
        
        # Update stats
        self.stats['flat_memories'] = len(self.flat_memory.memories)
        self._update_hierarchical_stats()
        
        # Return the flat ID as the primary ID
        return flat_id
    
    async def retrieve_memories(self, query: str, limit: int = 5, min_significance: float = 0.0,
                             categories: Optional[List[str]] = None):
        """
        Retrieve memories from both systems and combine results.
        
        Prioritizes memories from hierarchical system for organization,
        but falls back to flat system if needed.
        
        Args:
            query: The search query
            limit: Maximum number of results
            min_significance: Minimum significance threshold
            categories: Optional list of categories to filter by
            
        Returns:
            Combined list of memories from both systems
        """
        results = []
        
        # First try hierarchical memory system
        if self.has_hierarchical and hasattr(self.hierarchical_memory, 'retrieve_memories'):
            try:
                hierarchical_results = await self.hierarchical_memory.retrieve_memories(
                    query=query,
                    limit=limit,
                    min_significance=min_significance,
                    categories=categories
                )
                results.extend(hierarchical_results)
            except Exception as e:
                logger.error(f"Error retrieving from hierarchical memory: {e}")
        
        # If we don't have enough results, also check flat memory system
        if len(results) < limit:
            remaining = limit - len(results)
            flat_results = self.flat_memory.search_memories(
                query=query,
                max_results=remaining,
                min_significance=min_significance
            )
            
            # Combine results, avoiding duplicates
            existing_ids = [r.get('id') for r in results]
            for mem in flat_results:
                if mem.get('id') not in existing_ids:
                    results.append(mem)
        
        # Sort by significance
        results.sort(key=lambda x: x.get('significance', 0.0), reverse=True)
        
        # Limit to requested number
        return results[:limit]
    
    async def search_memories(self, query_embedding, limit: int = 5, threshold: float = 0.0):
        """
        Search memories using a pre-computed embedding.
        
        Args:
            query_embedding: Pre-computed embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of memory results with similarity scores
        """
        logger.info(f"Searching memories with pre-computed embedding, limit={limit}, threshold={threshold}")
        results = []
        
        # First search in the hierarchical memory if available
        if self.has_hierarchical and hasattr(self.hierarchical_memory, 'search_by_embedding'):
            try:
                hierarchical_results = await self.hierarchical_memory.search_by_embedding(
                    embedding=query_embedding,
                    limit=limit,
                    threshold=threshold
                )
                results.extend(hierarchical_results)
            except Exception as e:
                logger.error(f"Error searching hierarchical memory: {e}")
        
        # If we don't have enough results, also search flat memory system
        if len(results) < limit:
            remaining = limit - len(results)
            flat_results = self.flat_memory.search_by_embedding(
                embedding=query_embedding,
                max_results=remaining,
                min_similarity=threshold
            )
            
            # Convert flat results to the same format
            for result in flat_results:
                results.append({
                    "memory": result["memory"],
                    "similarity": result["similarity"]
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit to requested number
        return results[:limit]
    
    async def process_new_memory(self, memory_data: Dict[str, Any]):
        """
        Process a newly created memory to ensure it's immediately persisted across systems.
        
        Args:
            memory_data: The memory data to process
        
        Returns:
            bool: True if successfully processed, False otherwise
        """
        try:
            logger.info(f"Processing new memory {memory_data.get('id', 'unknown')} for persistence")
            
            # Ensure memory has the required fields
            if not all(key in memory_data for key in ['id', 'text']):
                logger.warning(f"Memory missing required fields: {memory_data}")
                return False
            
            # Force immediate persistence for CLI-created memories
            if self.has_hierarchical:
                # Import necessary types dynamically to prevent circular imports
                from memory.lucidia_memory_system.core.memory_types import MemoryTypes
                
                # Determine memory type based on content and metadata
                memory_type = MemoryTypes.SEMANTIC
                metadata = memory_data.get('metadata', {})
                
                # Special case for relationship memories
                if 'relationship' in metadata or 'human_name' in metadata:
                    memory_type = MemoryTypes.RELATIONSHIP
                    # Ensure high significance for relationships
                    memory_data['significance'] = max(memory_data.get('significance', 0), 0.95)
                
                # Add persistence markers
                metadata['persisted'] = True
                metadata['immediate_storage'] = True
                metadata['source'] = 'cli_created'
                metadata['creation_time'] = time.time()
                
                # Store in hierarchical memory with persistence flag
                embedding = memory_data.get('embedding')
                if embedding:
                    await self.hierarchical_memory.process_and_store(
                        content=memory_data['text'],
                        embedding=embedding,
                        memory_id=memory_data['id'],
                        memory_type=memory_type,
                        significance=memory_data.get('significance', 0.7),  # Use higher default significance
                        metadata=metadata,
                        force_ltm=True  # Force storage in LTM for persistence
                    )
                    logger.info(f"Memory {memory_data['id']} immediately stored in hierarchical system")
                    return True
                else:
                    logger.warning(f"Cannot store memory {memory_data['id']} without embedding")
            
            return False
        except Exception as e:
            logger.error(f"Error processing new memory: {e}")
            return False
    
    def get_stats(self):
        """
        Get statistics about the memory bridge.
        
        Returns:
            Dictionary of statistics
        """
        # Update the stats before returning
        self.stats['flat_memories'] = len(self.flat_memory.memories)
        self._update_hierarchical_stats()
            
        return self.stats
    
    async def shutdown(self):
        """
        Safely shut down the memory bridge.
        """
        # Cancel background tasks
        if hasattr(self, '_migration_task'):
            self._migration_task.cancel()
            try:
                await self._migration_task
            except asyncio.CancelledError:
                pass
        
        # Shut down hierarchical memory
        if self.has_hierarchical and hasattr(self.hierarchical_memory, 'shutdown'):
            await self.hierarchical_memory.shutdown()
        
        logger.info("Memory Bridge shut down")
    
    def _update_hierarchical_stats(self):
        """
        Update the hierarchical memory statistics.
        """
        if self.has_hierarchical:
            # Safely get counts for each memory layer with proper error handling
            try:
                self.stats['hierarchical_memories']['stm'] = len(self.hierarchical_memory.stm.memories) if hasattr(self.hierarchical_memory, 'stm') and hasattr(self.hierarchical_memory.stm, 'memories') else 0
            except Exception as e:
                logger.warning(f"Error getting STM stats: {e}")
                self.stats['hierarchical_memories']['stm'] = 0
                
            try:
                self.stats['hierarchical_memories']['ltm'] = len(self.hierarchical_memory.ltm.memories) if hasattr(self.hierarchical_memory, 'ltm') and hasattr(self.hierarchical_memory.ltm, 'memories') else 0
            except Exception as e:
                logger.warning(f"Error getting LTM stats: {e}")
                self.stats['hierarchical_memories']['ltm'] = 0
                
            try:
                self.stats['hierarchical_memories']['mpl'] = len(self.hierarchical_memory.mpl.memories) if hasattr(self.hierarchical_memory, 'mpl') and hasattr(self.hierarchical_memory.mpl, 'memories') else 0
            except Exception as e:
                logger.warning(f"Error getting MPL stats: {e}")
                self.stats['hierarchical_memories']['mpl'] = 0
                
    def _update_flat_stats(self):
        """
        Update the flat memory statistics.
        """
        self.stats['flat_memories'] = len(self.flat_memory.memories)
