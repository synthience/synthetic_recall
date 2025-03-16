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

logger = logging.getLogger(__name__)

class MemoryBridge:
    """
    Bridge between the flat and hierarchical memory systems.

    This component connects the flat memory system (server/memory_system.py) with
    the hierarchical memory system (memory/lucidia_memory_system/core/memory_core.py),
    allowing both systems to be leveraged together.
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
            'enable_migration': True,   # Auto-migrate memories between systems
            'migration_batch_size': 20, # For more efficient migration
            'migration_interval': 300,  # Time between migrations in seconds (5 minutes)
            **(config or {})
        }

        # References to both memory systems
        self.flat_memory = memory_system

        # Attempt importing hierarchical memory dynamically
        try:
            from memory.lucidia_memory_system.core.memory_core import MemoryCore
            from memory.lucidia_memory_system.core.memory_types import MemoryTypes
            
            logger.info(f"Initializing hierarchical memory core with path: {self.config['memory_path']}")
            self.hierarchical_memory = MemoryCore({
                'memory_path': self.config['memory_path'],
                'embedding_dim': self.config['embedding_dim']
            })
            self.memory_types = MemoryTypes
            self.has_hierarchical = True
        except ImportError as e:
            logger.warning(f"Could not import hierarchical memory system: {e}")
            logger.warning("Running with flat memory system only.")
            self.hierarchical_memory = None
            self.has_hierarchical = False
        except Exception as e:
            logger.error(f"Error initializing hierarchical memory: {e}")
            self.hierarchical_memory = None
            self.has_hierarchical = False

        self.last_migration = time.time()

        # Initialize stats
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
        self._update_hierarchical_stats()

        logger.info(f"Memory Bridge initialized with stats: {self.stats}")

        # Start periodic migration if enabled
        if self.config.get('enable_migration', False):
            self._migration_task = asyncio.create_task(self._periodic_migration())
            logger.info("Started background memory migration task")

    async def _periodic_migration(self):
        """
        Periodically migrate memories between systems.
        """
        while True:
            try:
                await asyncio.sleep(self.config.get('migration_interval', 300))
                await self.migrate_memories()
                self.last_migration = time.time()
            except asyncio.CancelledError:
                logger.info("Migration task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in periodic migration: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    async def migrate_memories(self):
        """
        Migrate memories between the flat and hierarchical systems, without using significance.
        """
        logger.info("Starting memory migration...")
        migrated = 0
        batch_size = self.config.get('migration_batch_size', 20)

        # 1. Move memories from flat to hierarchical
        flat_memories = self.flat_memory.memories
        logger.info(f"Checking {len(flat_memories)} flat memories for migration to hierarchical system")

        # Gather memories that have not yet migrated
        # (No significance-based filtering anymore)
        not_migrated = [
            m for m in flat_memories
            if not m.get('migrated_to_hierarchical')
        ]

        # Limit batch
        batch_memories = not_migrated[:batch_size]
        logger.info(f"Found {len(not_migrated)} eligible memories, processing batch of {len(batch_memories)}")

        for memory in batch_memories:
            try:
                # Extract text/content
                text = memory.get('text', memory.get('content', ''))
                if not text:
                    continue

                # Check embedding
                embedding = memory.get('embedding')
                is_valid_embedding = False
                if embedding:
                    if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                        is_valid_embedding = True
                    elif isinstance(embedding, str):
                        try:
                            embedding_data = json.loads(embedding)
                            if (isinstance(embedding_data, list)
                                and all(isinstance(x, (int, float)) for x in embedding_data)):
                                memory['embedding'] = embedding_data
                                is_valid_embedding = True
                        except (json.JSONDecodeError, TypeError):
                            pass

                if not is_valid_embedding:
                    logger.warning(f"Memory {memory.get('id', 'unknown')} skipped: invalid embedding.")
                    continue

                if not self.has_hierarchical:
                    # If no hierarchical system, do nothing
                    continue

                from memory.lucidia_memory_system.core.memory_types import MemoryTypes

                # Basic default type
                memory_type = MemoryTypes.SEMANTIC

                metadata = memory.get('metadata', {})
                # Possibly refine type from metadata
                if 'relationship' in metadata or 'human_name' in metadata:
                    memory_type = MemoryTypes.RELATIONSHIP
                elif 'emotional' in metadata or metadata.get('emotions'):
                    memory_type = MemoryTypes.EMOTIONAL
                elif metadata.get('is_personal', False):
                    memory_type = MemoryTypes.PERSONAL
                elif metadata.get('is_important', False):
                    memory_type = MemoryTypes.IMPORTANT

                # Attempt storing in hierarchical memory
                await self.hierarchical_memory.process_and_store(
                    content=text,
                    embedding=memory['embedding'],
                    memory_id=memory['id'],
                    memory_type=memory_type,
                    metadata=metadata,
                    force_ltm=True
                )
                migrated += 1
                logger.info(f"Migrated memory {memory['id']} to hierarchical system")
                memory['migrated_to_hierarchical'] = True

                # Save updated memory in flat system
                try:
                    self.flat_memory._save_memory(memory)
                except Exception as save_error:
                    logger.error(f"Failed to update migration status for {memory['id']}: {save_error}")

            except Exception as e:
                logger.error(f"Error migrating memory {memory.get('id', 'unknown')}: {e}")

        # 2. Move memories from hierarchical to flat (if not present)
        if self.has_hierarchical and hasattr(self.hierarchical_memory, 'ltm'):
            try:
                if hasattr(self.hierarchical_memory.ltm, 'get_all_memories'):
                    ltm_memories = await self.hierarchical_memory.ltm.get_all_memories()
                    logger.info(f"Checking {len(ltm_memories)} hierarchical memories for migration to flat system")

                    for h_memory in ltm_memories:
                        mem_id = h_memory.id
                        exists_in_flat = any(m.get('id') == mem_id for m in self.flat_memory.memories)
                        if not exists_in_flat:
                            flat_mem = {
                                'id': mem_id,
                                'text': h_memory.content,
                                'embedding': (
                                    h_memory.embedding.tolist()
                                    if hasattr(h_memory.embedding, 'tolist')
                                    else h_memory.embedding
                                ),
                                'timestamp': time.time(),
                                'metadata': {
                                    **h_memory.metadata,
                                    'source': 'hierarchical',
                                    'persisted': True
                                }
                            }
                            self.flat_memory.memories.append(flat_mem)
                            self.flat_memory._save_memory(flat_mem)
                            migrated += 1
                            logger.info(f"Migrated hierarchical memory {mem_id} to flat system")
            except Exception as e:
                logger.error(f"Error retrieving hierarchical memories: {e}")

        self._update_flat_stats()
        self._update_hierarchical_stats()
        logger.info(f"Memory migration complete. Migrated {migrated} memories.")
        return migrated

    async def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        categories: Optional[List[str]] = None,
        embedding=None
    ):
        """
        Store a memory in both systems (no significance references).

        Args:
            content: The memory content
            metadata: Optional additional metadata
            categories: Optional list of categories
            embedding: Optional pre-computed embedding

        Returns:
            Memory ID if stored, otherwise None
        """
        if metadata is None:
            metadata = {}
        if categories is None:
            categories = []

        # Store in flat memory
        flat_id = self.flat_memory.store_memory(
            content=content,
            metadata=metadata,
            embedding=embedding
        )

        # Also store in hierarchical memory, if available
        hier_id = None
        if self.has_hierarchical:
            try:
                from memory.lucidia_memory_system.core.memory_types import MemoryTypes
                # Default memory type
                memory_type = MemoryTypes.EPISODIC

                # Possibly refine from metadata
                if 'relationship' in metadata or 'human_name' in metadata:
                    memory_type = MemoryTypes.RELATIONSHIP
                elif 'emotional' in metadata or metadata.get('emotions'):
                    memory_type = MemoryTypes.EMOTIONAL
                elif metadata.get('is_personal', False):
                    memory_type = MemoryTypes.PERSONAL
                elif metadata.get('is_important', False):
                    memory_type = MemoryTypes.IMPORTANT

                hier_id = await self.hierarchical_memory.process_and_store(
                    content=content,
                    memory_type=memory_type,
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

        return flat_id

    async def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        categories: Optional[List[str]] = None
    ):
        """
        Retrieve memories from both systems (no significance filtering).

        Args:
            query: The search query
            limit: Max number of results
            categories: Optional categories to filter by

        Returns:
            Combined list of memories from both systems
        """
        results = []

        # Hierarchical memory first
        if self.has_hierarchical and hasattr(self.hierarchical_memory, 'retrieve_memories'):
            try:
                hierarchical_results = await self.hierarchical_memory.retrieve_memories(
                    query=query,
                    limit=limit,
                    categories=categories
                )
                results.extend(hierarchical_results)
            except Exception as e:
                logger.error(f"Error retrieving from hierarchical memory: {e}")

        if len(results) < limit:
            remaining = limit - len(results)
            flat_results = self.flat_memory.search_memories(
                query=query,
                max_results=remaining
            )
            # Avoid duplicates
            existing_ids = {r.get('id') for r in results}
            for mem in flat_results:
                if mem.get('id') not in existing_ids:
                    results.append(mem)

        # No significance-based sorting; keep as-is or sort by recency if needed
        # Just return results up to limit
        return results[:limit]

    async def search_memories(self, query_embedding, limit: int = 5, threshold: float = 0.0):
        """
        Search memories with a pre-computed embedding (no significance logic).

        Args:
            query_embedding: The embedding vector
            limit: Max number of results
            threshold: Minimum similarity threshold

        Returns:
            List of results with similarity scores
        """
        logger.info(f"Searching memories with embedding, limit={limit}, threshold={threshold}")
        results = []

        # Hierarchical memory
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

        # Flat memory if needed
        if len(results) < limit:
            remaining = limit - len(results)
            flat_results = self.flat_memory.search_by_embedding(
                embedding=query_embedding,
                max_results=remaining,
                min_similarity=threshold
            )
            for result in flat_results:
                results.append({
                    "memory": result["memory"],
                    "similarity": result["similarity"]
                })

        # Sort by similarity only
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    async def process_new_memory(self, memory_data: Dict[str, Any]):
        """
        Process a newly created memory to ensure it's persisted in both systems (no significance).

        Args:
            memory_data: The memory data

        Returns:
            True if successfully processed, False otherwise
        """
        try:
            logger.info(f"Processing new memory {memory_data.get('id', 'unknown')} for persistence")

            if not all(k in memory_data for k in ['id', 'text']):
                logger.warning(f"Memory missing required fields: {memory_data}")
                return False

            if self.has_hierarchical:
                from memory.lucidia_memory_system.core.memory_types import MemoryTypes

                metadata = memory_data.get('metadata', {})
                memory_type = MemoryTypes.SEMANTIC

                if 'relationship' in metadata or 'human_name' in metadata:
                    memory_type = MemoryTypes.RELATIONSHIP
                elif 'emotional' in metadata or metadata.get('emotions'):
                    memory_type = MemoryTypes.EMOTIONAL
                elif metadata.get('is_personal', False):
                    memory_type = MemoryTypes.PERSONAL
                elif metadata.get('is_important', False):
                    memory_type = MemoryTypes.IMPORTANT

                metadata['persisted'] = True
                metadata['immediate_storage'] = True
                metadata['source'] = 'cli_created'
                metadata['creation_time'] = time.time()

                embedding = memory_data.get('embedding')
                if embedding:
                    await self.hierarchical_memory.process_and_store(
                        content=memory_data['text'],
                        embedding=embedding,
                        memory_id=memory_data['id'],
                        memory_type=memory_type,
                        metadata=metadata,
                        force_ltm=True
                    )
                    logger.info(f"Memory {memory_data['id']} stored in hierarchical system")
                    return True
                else:
                    logger.warning(f"Cannot store memory {memory_data['id']} without embedding")

            return False
        except Exception as e:
            logger.error(f"Error processing new memory: {e}")
            return False

    def get_stats(self):
        """
        Get current memory bridge statistics, free of significance references.
        """
        self.stats['flat_memories'] = len(self.flat_memory.memories)
        self._update_hierarchical_stats()
        return self.stats

    async def shutdown(self):
        """
        Safely shut down the memory bridge.
        """
        if hasattr(self, '_migration_task'):
            self._migration_task.cancel()
            try:
                await self._migration_task
            except asyncio.CancelledError:
                pass

        if self.has_hierarchical and hasattr(self.hierarchical_memory, 'shutdown'):
            await self.hierarchical_memory.shutdown()

        logger.info("Memory Bridge shut down")

    def _update_hierarchical_stats(self):
        """
        Update hierarchical memory statistics (no significance).
        """
        if self.has_hierarchical:
            try:
                self.stats['hierarchical_memories']['stm'] = (
                    len(self.hierarchical_memory.stm.memories)
                    if hasattr(self.hierarchical_memory, 'stm')
                       and hasattr(self.hierarchical_memory.stm, 'memories')
                    else 0
                )
            except Exception as e:
                logger.warning(f"Error getting STM stats: {e}")
                self.stats['hierarchical_memories']['stm'] = 0

            try:
                self.stats['hierarchical_memories']['ltm'] = (
                    len(self.hierarchical_memory.ltm.memories)
                    if hasattr(self.hierarchical_memory, 'ltm')
                       and hasattr(self.hierarchical_memory.ltm, 'memories')
                    else 0
                )
            except Exception as e:
                logger.warning(f"Error getting LTM stats: {e}")
                self.stats['hierarchical_memories']['ltm'] = 0

            try:
                self.stats['hierarchical_memories']['mpl'] = (
                    len(self.hierarchical_memory.mpl.memories)
                    if hasattr(self.hierarchical_memory, 'mpl')
                       and hasattr(self.hierarchical_memory.mpl, 'memories')
                    else 0
                )
            except Exception as e:
                logger.warning(f"Error getting MPL stats: {e}")
                self.stats['hierarchical_memories']['mpl'] = 0

    def _update_flat_stats(self):
        """
        Update the flat memory statistics (no significance references).
        """
        self.stats['flat_memories'] = len(self.flat_memory.memories)
