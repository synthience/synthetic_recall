
# synthians_memory_core/synthians_memory_core.py

import time
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from pathlib import Path
import random
import uuid
import json
import os
import datetime as dt
from datetime import timezone, datetime, timedelta # Ensure datetime is imported directly
import copy
import traceback # Import traceback for detailed error logging
import math
import queue # Added for the retry loop exception type
import aiofiles # For async file operations
import os # Ensure os is imported

# Import core components from this package
from .custom_logger import logger
from .memory_structures import MemoryEntry, MemoryAssembly
from .hpc_quickrecal import UnifiedQuickRecallCalculator, QuickRecallMode, QuickRecallFactor
from .geometry_manager import GeometryManager, GeometryType
from .emotional_intelligence import EmotionalGatingService
from .memory_persistence import MemoryPersistence
from .adaptive_components import ThresholdCalibrator
from .metadata_synthesizer import MetadataSynthesizer
from .emotion_analyzer import EmotionAnalyzer
from .vector_index import MemoryVectorIndex
# Import the merge tracker for Phase 5.9
from .metrics.merge_tracker import MergeTracker

# --- Add Deep Update Utility Function ---
# (Can be placed inside the class or outside)
def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modifies source in place.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            # Ensure source[key] exists and is a dict before recursing
            current_value = source.get(key)
            if isinstance(current_value, dict):
                returned = deep_update(current_value, value)
                source[key] = returned
            else:
                # If source[key] is not a dict or doesn't exist, just overwrite
                source[key] = value
    return source


class SynthiansMemoryCore:
    """
    Unified Synthians Memory Core.

    Integrates HPC-QuickRecal, Hyperbolic Geometry, Emotional Intelligence,
    Memory Assemblies, Adaptive Thresholds, and Robust Persistence
    into a lean and efficient memory system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'embedding_dim': 768,
            'geometry': 'hyperbolic', # 'euclidean', 'hyperbolic', 'spherical', 'mixed'
            'hyperbolic_curvature': -1.0,
            'storage_path': '/app/memory/stored/synthians', # Unified path
            'persistence_interval': 60.0, # Persist every minute
            'decay_interval': 3600.0, # Check decay every hour
            'prune_check_interval': 10.0, # Check if pruning needed every 10 seconds (reduced for tests)
            'max_memory_entries': 50000,
            'prune_threshold_percent': 0.9, # Prune when 90% full
            'min_quickrecal_for_ltm': 0.2, # Min score to keep after decay
            'assembly_threshold': 0.85, # Higher threshold to ensure distinct assembly formation
            'assembly_merge_threshold': 0.80, # Threshold for merging similar assemblies
            'max_assemblies_per_memory': 3,
            'adaptive_threshold_enabled': True,
            'initial_retrieval_threshold': 0.75,
            'vector_index_type': 'Cosine',  # 'L2', 'IP', 'Cosine'
            'persistence_batch_size': 100, # Batch size for persistence loop
            'check_index_on_retrieval': True, # New config option - Controls check *during* retrieval
            'check_index_periodic': True,     # New config option - Controls periodic background check
            'index_check_interval': 3600, # New config option - Interval for periodic check
            'migrate_to_idmap': True, # New config option
            'enable_assemblies': True, # CRITICAL: Explicitly enable assembly subsystem
            'enable_assembly_pruning': True, # Enable pruning of inactive assemblies
            'enable_assembly_merging': True, # Enable merging of similar assemblies
            # Retry loop config
            'vector_retry_interval_seconds': 60,
            'vector_retry_batch_size': 10,
            'max_vector_retry_attempts': 5,
            # Phase 5.9: Configuration for explainability and diagnostics
            'ENABLE_EXPLAINABILITY': False, # Default to disabled in production
            'merge_log_max_entries': 1000, # Maximum entries in merge log file
            'assembly_metrics_persist_interval': 600.0, # Seconds between saving activation stats
            'start_background_tasks_on_init': True, # New config option
            'force_skip_idmap_debug': False, # <<< ADD DEFAULT
            **(config or {})
        }

        logger.info("SynthiansMemoryCore", "Initializing...", self.config)
        self.start_time = time.time() # Record start time for uptime calculation

        # --- Core Components ---
        self.geometry_manager = GeometryManager({
            'embedding_dim': self.config['embedding_dim'],
            'geometry_type': self.config['geometry'],
            'curvature': self.config['hyperbolic_curvature']
        })

        # Pending vector update queue for failed index operations
        self._pending_vector_updates = asyncio.Queue()

        self.quick_recal = UnifiedQuickRecallCalculator({
            'embedding_dim': self.config['embedding_dim'],
            'mode': QuickRecallMode.HPC_QR, # Default to HPC-QR mode
            'geometry_type': self.config['geometry'],
            'curvature': self.config['hyperbolic_curvature']
        }, geometry_manager=self.geometry_manager) # Pass geometry manager

        # Provide the analyzer instance directly to the gating service
        self.emotional_analyzer = EmotionAnalyzer()  # Use our new robust emotion analyzer
        self.emotional_gating = EmotionalGatingService(
            emotion_analyzer=self.emotional_analyzer, # Pass the instance
            config={'emotional_weight': 0.3} # Example config
        )

        self.persistence = MemoryPersistence({'storage_path': self.config['storage_path']})

        self.threshold_calibrator = ThresholdCalibrator(
            initial_threshold=self.config['initial_retrieval_threshold']
        ) if self.config['adaptive_threshold_enabled'] else None

        # Pass geometry_manager to MetadataSynthesizer
        self.metadata_synthesizer = MetadataSynthesizer(geometry_manager=self.geometry_manager)


        # Retrieve the debug flag from config
        force_skip_idmap = self.config.get('force_skip_idmap_debug', False)
        migrate_to_idmap = self.config.get('migrate_to_idmap', True)

        # Prepare config for MemoryVectorIndex
        vector_index_config = {
            'embedding_dim': self.config['embedding_dim'],
            'storage_path': self.config['storage_path'], # Pass storage path
            'index_path': os.path.join(self.config['storage_path'], 'index'), # Construct index path
            'vector_index_type': self.config.get('vector_index_type', 'Cosine'),
            'migrate_to_idmap': migrate_to_idmap,
            'use_gpu': not migrate_to_idmap  # GPU logic still tied to migrate_to_idmap
        }

        # Initialize vector index for fast retrieval, passing the debug flag
        self.vector_index = MemoryVectorIndex(
            config=vector_index_config,
            force_skip_idmap_debug=force_skip_idmap # <<< PASS THE FLAG
        )

        # --- Memory State ---
        self._memories: Dict[str, MemoryEntry] = {} # In-memory cache/working set
        self.assemblies: Dict[str, MemoryAssembly] = {}
        self.memory_to_assemblies: Dict[str, Set[str]] = {}
        self._dirty_memories: Set[str] = set() # Track modified memory IDs for persistence

        # --- Phase 5.9: Activation and Merge Tracking ---
        self._assembly_activation_counts: Dict[str, int] = {}  # Track assembly activation counts
        self._last_activation_persist_time = time.time()  # Track when we last persisted activation stats

        # Initialize the MergeTracker for merge event logging
        merge_log_dir = os.path.join(self.config['storage_path'], 'logs')
        merge_log_file = os.path.join(merge_log_dir, 'merge_log.jsonl')

        try:
            os.makedirs(merge_log_dir, exist_ok=True)
            logger.info(f"Ensured log directory exists: {merge_log_dir}")
        except OSError as e:
            # Log the error but proceed - MergeTracker might handle it or fail later
            logger.error(f"Could not create log directory {merge_log_dir}: {e}")

        self.merge_tracker = MergeTracker(
            log_path=merge_log_file,  # Pass the string path
            max_entries=self.config.get('merge_log_max_entries', 1000),
            max_size_mb=self.config.get('merge_log_rotation_size_mb', 100) # Use .get() with default
        )

        # --- Concurrency & Tasks ---
        self._lock = asyncio.Lock()
        self._background_tasks: List[asyncio.Task] = []
        self._initialized = False
        self._shutdown_signal = asyncio.Event() # Use _shutdown_signal consistently

        logger.info("SynthiansMemoryCore", "Core components initialized.")

    async def initialize(self):
        """Initialize the memory core components asynchronously."""
        if self._initialized:
            logger.info("SynthiansMemoryCore already initialized.")
            return True

        logger.info("Initializing SynthiansMemoryCore components...")
        try:
            # Initialize Persistence first (loads the memory index file)
            if self.persistence:
                await self.persistence.initialize()
                logger.info("MemoryPersistence initialized.")
            else:
                logger.error("Persistence component is None during initialization!")
                return False  # Cannot proceed without persistence

            # Initialize Vector Index (loads FAISS index and mapping)
            if self.vector_index:
                initialized_ok = await self.vector_index.initialize()
                if not initialized_ok:
                    logger.error("Vector Index initialization failed!")
                    return False  # Fail initialization if vector index fails
                logger.info("MemoryVectorIndex initialized.")
            else:
                logger.error("Vector Index component is None during initialization!")
                return False  # Cannot proceed without vector index

            # Initialize the MergeTracker for Phase 5.9
            if hasattr(self, 'merge_tracker') and self.merge_tracker:
                await self.merge_tracker.initialize()
                logger.info("MergeTracker initialized.")

            # Load assembly activation statistics if available
            await self._load_activation_stats() # FIX: Calling restored method
            logger.info("Assembly activation stats loaded/initialized.")

            # --- PHASE 5.8.A: Vector Index Integrity Check and Auto-Repair ---
            # Mark as initialized temporarily so drift detection can run
            self._initialized = True

            # Set auto-repair based on config or environment variable
            auto_repair = os.environ.get("ENABLE_INDEX_AUTO_REPAIR", "true").lower() in ("true", "1")

            logger.info("SynthiansMemoryCore", f"Checking vector index integrity with auto-repair={auto_repair}...")
            drift_result = await self.detect_and_repair_index_drift(auto_repair=auto_repair)

            if not drift_result.get("success", False):
                if auto_repair:
                    # Auto-repair failed
                    logger.error(
                        "SynthiansMemoryCore",
                        "Vector index auto-repair failed during initialization",
                        {"details": drift_result}
                    )
                    logger.warning("SynthiansMemoryCore", "Continuing with initialization despite failed repair")
                else:
                    # Drift detected but auto-repair disabled
                    logger.warning(
                        "SynthiansMemoryCore",
                        "Vector index drift detected during initialization but auto-repair disabled",
                        {"details": drift_result}
                    )
            else:
                # Either no drift or repair was successful
                if drift_result.get("is_consistent", False):
                    logger.info("SynthiansMemoryCore", "Vector index integrity verified during initialization")
                else:
                    logger.info("SynthiansMemoryCore", "Vector index successfully repaired during initialization")
            # --- END PHASE 5.8.A ---

            # Load initial memories into cache if needed?
            # Consider loading a subset based on recency or importance if full load is too much
            # await self.load_initial_memories() # Example call

            # Check if we should start background tasks
            if self.config.get('start_background_tasks_on_init', True):
                # Start background tasks
                persistence_task = asyncio.create_task(self._persistence_loop())
                persistence_task.set_name("persistence_loop")
                self._background_tasks.append(persistence_task)
                logger.info("SynthiansMemoryCore", "Started persistence background loop")

                decay_task = asyncio.create_task(self._decay_and_pruning_loop())
                decay_task.set_name("decay_and_pruning_loop")
                self._background_tasks.append(decay_task)
                logger.info("SynthiansMemoryCore", "Started decay/pruning background loop")

                drift_task = asyncio.create_task(self._auto_repair_drift_loop())
                drift_task.set_name("auto_repair_drift_loop")
                self._background_tasks.append(drift_task)
                logger.info("SynthiansMemoryCore", "Started auto-repair drift background loop")

                retry_task = asyncio.create_task(self._vector_update_retry_loop())
                retry_task.set_name("vector_update_retry_loop")
                self._background_tasks.append(retry_task)
                logger.info("SynthiansMemoryCore", "Started vector update retry background loop")
            else:
                logger.warning("SynthiansMemoryCore", "Background tasks disabled due to start_background_tasks_on_init=False")

            # Confirm initialization is complete
            self._initialized = True
            logger.info("SynthiansMemoryCore initialization complete.")
            return True

        except AttributeError as ae: # Catch the specific error
             logger.error(f"AttributeError during SynthiansMemoryCore initialization: {ae}", exc_info=True)
             self._initialized = False
             if hasattr(self, 'vector_index') and self.vector_index: self.vector_index.state = "ERROR"
             return False
        except Exception as e:
            logger.error(f"Critical error during SynthiansMemoryCore initialization: {e}", exc_info=True)
            self._initialized = False
            if hasattr(self, 'vector_index') and self.vector_index:
                 self.vector_index.state = "ERROR"
            return False

    async def cleanup(self):
        """Clean up resources before shutdown.

        Part of Phase 5.8 stability improvements to ensure proper resource
        management during application shutdown.
        """
        logger.info("SynthiansMemoryCore", "Cleaning up resources")
        try:
            # Ensure final persistence before shutdown
            if hasattr(self, 'persistence') and self.persistence is not None:
                logger.info("SynthiansMemoryCore", "Final memory persistence before shutdown")
                if hasattr(self.persistence, 'persist_all'):
                    await self.persistence.persist_all()
                else:
                    # Fallback to our own persistence method
                    await self._persist_dirty_items()

            # --- PHASE 5.8.B: Vector Index Persistence on Cleanup ---
            # Save vector index as part of cleanup
            if hasattr(self, 'vector_index') and self.vector_index is not None:
                logger.info("SynthiansMemoryCore", "Saving vector index as part of cleanup")
                try:
                    # Add a timeout for safety
                    await asyncio.wait_for(self.vector_index.save_async(), timeout=10.0)
                    logger.info("SynthiansMemoryCore", "Vector index saved during cleanup")
                except asyncio.TimeoutError:
                    logger.warning("SynthiansMemoryCore", "Timeout waiting for vector index save during cleanup")
                except Exception as e:
                    logger.error("SynthiansMemoryCore", f"Error saving vector index during cleanup: {str(e)}", exc_info=True)
            # --- END PHASE 5.8.B ---

            # Cancel any pending tasks
            if hasattr(self, '_background_tasks'):
                for task in self._background_tasks:
                    if not task.done():
                        task_name = task.get_name() if hasattr(task, 'get_name') else 'unnamed'
                        logger.info("SynthiansMemoryCore", f"Cancelling background task {task_name}")
                        task.cancel()

            logger.info("SynthiansMemoryCore", "Cleanup completed successfully")
            return True
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error during cleanup: {str(e)}", exc_info=True)
            # We still return True as we want the shutdown to continue
            return True

    async def shutdown(self):
        """Gracefully shut down the memory core."""
        if not self._initialized:
            logger.info("SynthiansMemoryCore", "Shutdown called but not initialized.")
            return

        logger.info("SynthiansMemoryCore", "Shutting down...")
        # Signal loops to stop checking/sleeping first
        self._shutdown_signal.set()
        # Give loops a brief moment to recognize the signal
        await asyncio.sleep(0.05)

        # Cancel active tasks
        tasks_to_cancel = []
        for task in self._background_tasks:
            if task and not task.done():
                # Don't cancel if already cancelling
                if not task.cancelling():
                    task.cancel()
                    tasks_to_cancel.append(task)

        # Wait for tasks to complete cancellation
        if tasks_to_cancel:
            logger.info(f"Waiting for {len(tasks_to_cancel)} background tasks to cancel...")
            # Use return_exceptions=True so one failed task doesn't stop others
            # Wait for a reasonable time (e.g., 5 seconds) for tasks to finish cancelling
            results = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            logger.info("Background tasks cancellation completed.")
            # Check for exceptions during cancellation
            for i, result in enumerate(results):
                task_name = tasks_to_cancel[i].get_name() if hasattr(tasks_to_cancel[i], 'get_name') else f"Task-{i}"
                if isinstance(result, asyncio.CancelledError):
                    logger.debug(f"{task_name} was cancelled successfully.")
                elif isinstance(result, Exception):
                    logger.error(f"Error during cancellation of {task_name}: {result}", exc_info=result)
        else:
            logger.info("No active background tasks found to cancel.")

        # Clear the list of tasks *after* attempting cancellation
        self._background_tasks = []

        # --- Critical: Call persistence shutdown *before* resetting state ---
        # This allows persistence to do its final save using the current state
        logger.info("SynthiansMemoryCore", "Calling persistence shutdown...")
        if hasattr(self, 'persistence') and self.persistence:
            try:
                # Add a timeout for safety
                await asyncio.wait_for(self.persistence.shutdown(), timeout=5.0)
                logger.info("SynthiansMemoryCore", "Persistence shutdown completed.")
            except asyncio.TimeoutError:
                logger.warning("SynthiansMemoryCore", "Timeout waiting for persistence shutdown")
            except Exception as e:
                logger.error("SynthiansMemoryCore", f"Error during persistence shutdown: {str(e)}", exc_info=True)
        else:
             logger.warning("SynthiansMemoryCore", "Persistence object not available during shutdown.")

        # --- PHASE 5.8.A: Vector Index Persistence on Shutdown ---
        # Ensure vector index is saved before shutting down
        logger.info("SynthiansMemoryCore", "Calling vector index shutdown/save...")
        if hasattr(self, 'vector_index') and self.vector_index and self._initialized:
            try:
                # Add a timeout for safety
                await asyncio.wait_for(self.vector_index.save_async(), timeout=10.0)
                logger.info("SynthiansMemoryCore", "Vector index save on shutdown completed.")
            except asyncio.TimeoutError:
                logger.warning("SynthiansMemoryCore", "Timeout waiting for vector index save")
            except Exception as e:
                logger.error("SynthiansMemoryCore", f"Error during vector index save: {str(e)}", exc_info=True)
        else:
            logger.warning("SynthiansMemoryCore", "Vector index not available during shutdown.")
        # --- END PHASE 5.8.A ---

        # Reset state
        self._initialized = False
        # Reset shutdown signal for potential re-initialization
        self._shutdown_signal = asyncio.Event() # Recreate the event
        logger.info("SynthiansMemoryCore", "Shutdown sequence complete.")

    # --- Core Memory Operations ---

    async def process_memory(self,
                           content: Optional[str] = None,
                           embedding: Optional[Union[np.ndarray, List[float]]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """API-compatible wrapper for process_new_memory."""
        if not self._initialized: await self.initialize()

        # Call the underlying implementation
        memory = await self.process_new_memory(content=content, embedding=embedding, metadata=metadata)

        if memory:
            return {
                "success": True, # Add success flag
                "memory_id": memory.id,
                "quickrecal_score": memory.quickrecal_score,
                "embedding": memory.embedding.tolist() if memory.embedding is not None else None, # Include embedding
                "metadata": memory.metadata
            }
        else:
            return {
                "success": False, # Add success flag
                "memory_id": None,
                "quickrecal_score": None,
                "error": "Failed to process memory"
            }

    async def process_new_memory(self,
                                 content: str,
                                 embedding: Optional[Union[np.ndarray, List[float]]] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> Optional[MemoryEntry]:
        """Process and store a new memory entry."""
        if not self._initialized: await self.initialize()
        start_time = time.time()
        metadata = metadata or {}

        # 1. Validate/Generate Embedding
        if embedding is None:
            logger.info("SynthiansMemoryCore", "Generating embedding for new memory...")
            embedding = await self.generate_embedding(content) # Generate if not provided
            if embedding is None:
                logger.error("SynthiansMemoryCore", "Failed to generate embedding, cannot process memory.")
                return None

        # Handle common case where embedding is wrongly passed as a dict
        if isinstance(embedding, dict):
            logger.warning("SynthiansMemoryCore", f"Received embedding as dict type, attempting to extract vector")
            try:
                if 'embedding' in embedding and isinstance(embedding['embedding'], (list, np.ndarray)): embedding = embedding['embedding']
                elif 'vector' in embedding and isinstance(embedding['vector'], (list, np.ndarray)): embedding = embedding['vector']
                elif 'value' in embedding and isinstance(embedding['value'], (list, np.ndarray)): embedding = embedding['value']
                else: raise ValueError(f"Could not extract embedding from dict keys: {list(embedding.keys())[:5]}")
            except Exception as e:
                logger.error("SynthiansMemoryCore", f"Failed to extract embedding from dict: {str(e)}")
                return None

        validated_embedding = self.geometry_manager._validate_vector(embedding, "Input Embedding")
        if validated_embedding is None:
             logger.error("SynthiansMemoryCore", "Invalid embedding provided, cannot process memory.")
             return None
        aligned_embedding, _ = self.geometry_manager._align_vectors(validated_embedding, np.zeros(self.config['embedding_dim']))
        normalized_embedding = self.geometry_manager._normalize(aligned_embedding)

        # 2. Calculate QuickRecal Score
        context = {'timestamp': time.time(), 'metadata': metadata}
        # Include momentum buffer if available/needed by the mode
        # context['external_momentum'] = ...
        quickrecal_score = await self.quick_recal.calculate(normalized_embedding, text=content, context=context)

        # 3. Analyze Emotion only if not already provided
        emotional_context = metadata.get("emotional_context")
        if not emotional_context:
            logger.info("SynthiansMemoryCore", "Analyzing emotional context for memory")
            emotional_context = await self.emotional_analyzer.analyze(content)
            # Do not add to metadata here, let synthesizer handle it
        else:
            logger.debug("SynthiansMemoryCore", "Using precomputed emotional context from metadata")

        # 4. Generate Hyperbolic Embedding (if enabled)
        hyperbolic_embedding = None
        if self.geometry_manager.config['geometry_type'] == GeometryType.HYPERBOLIC:
            hyperbolic_embedding = self.geometry_manager._to_hyperbolic(normalized_embedding)

        # 5. Run Metadata Synthesizer
        # Pass the analyzed emotion data directly to the synthesizer
        metadata = await self.metadata_synthesizer.synthesize(
            content=content,
            embedding=normalized_embedding,
            base_metadata=metadata,
            emotion_data=emotional_context # Pass pre-analyzed data
        )

        # 6. Create Memory Entry
        memory = MemoryEntry(
            content=content,
            embedding=normalized_embedding,
            quickrecal_score=quickrecal_score,
            metadata=metadata,
            hyperbolic_embedding=hyperbolic_embedding
        )

        # Add memory ID to metadata for easier access
        memory.metadata["uuid"] = memory.id

        # 7. Store in memory and mark as dirty
        async with self._lock:
            self._memories[memory.id] = memory
            self._dirty_memories.add(memory.id) # Mark for persistence
            logger.info("SynthiansMemoryCore", f"Stored new memory {memory.id}", {"quickrecal": quickrecal_score})

        # 7.1 CRITICAL: Persist to disk and verify success
        logger.info("SynthiansMemoryCore", f"[PERSIST_CHECK] Saving memory {memory.id} to disk...")
        save_ok = await self.persistence.save_memory(memory)

        if not save_ok:
            logger.error("SynthiansMemoryCore", f"CRITICAL PERSISTENCE FAILURE for memory {memory.id}. Memory not saved to disk!")
            # Remove from cache since persistence failed
            async with self._lock:
                self._memories.pop(memory.id, None)
                self._dirty_memories.discard(memory.id)
            return None  # Signal failure to caller
        else:
            logger.info("SynthiansMemoryCore", f"[PERSIST_CHECK] Memory {memory.id} successfully saved to disk")

        # 8. Update Assemblies
        logger.info(f"[ASSEMBLY_DEBUG] Starting assembly update for memory {memory.id}")
        # Check if assemblies are actually enabled in the configuration
        assemblies_enabled = self.config.get('enable_assemblies', True)
        logger.info(f"[ASSEMBLY_DEBUG] Assembly processing enabled: {assemblies_enabled}")

        if assemblies_enabled:
            # Trace the assembly update call
            try:
                await self._update_assemblies(memory)
                logger.info(f"[ASSEMBLY_DEBUG] Assembly update completed for memory {memory.id}")
                # Verify assemblies were updated by logging count
                logger.info(f"[ASSEMBLY_DEBUG] Current assembly count: {len(self.assemblies)}")
            except Exception as e:
                logger.error(f"[ASSEMBLY_DEBUG] Error in _update_assemblies: {str(e)}", exc_info=True)
        else:
            logger.warning(f"[ASSEMBLY_DEBUG] Skipping assembly update - assemblies disabled in config")

        # 9. Add to vector index for fast retrieval
        if normalized_embedding is not None and self.vector_index is not None:
            # Only proceed with vector indexing if persistence succeeded
            if not save_ok:
                logger.error("SynthiansMemoryCore", f"Skipping vector index add for {memory.id} due to persistence failure")
                # No need to return None here, memory *was* processed, just index failed initially
            else:
                logger.debug(f"Adding memory {memory.id} to vector index...")
                added_to_index = await self.vector_index.add_async(memory.id, normalized_embedding)
                if not added_to_index:
                     # Queue failed operation for retry
                    await self._pending_vector_updates.put({
                        "operation": "add",
                        "id": memory.id,
                        "embedding": normalized_embedding.tolist(), # Store as list
                        "is_assembly": False,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "retry_count": 0
                    })
                    logger.error(f"Failed to add memory {memory.id} to vector index. Queued for retry.")
                    # Continue even if initial add fails, rely on retry loop
                else:
                    logger.debug(f"SynthiansMemoryCore", f"Added memory {memory.id} to vector index")


        proc_time = (time.time() - start_time) * 1000
        logger.debug("SynthiansMemoryCore", f"Processed new memory {memory.id}", {"time_ms": proc_time})
        return memory

    async def retrieve_memories(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        user_emotion: Optional[str] = None, # Changed to Optional[str] to match server endpoint
        metadata_filter: Optional[Dict[str, Any]] = None,
        search_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve memories based on query relevance.
        Handles potential query_embedding generation internally.
        """
        if not self._initialized: await self.initialize()
        start_time = time.time()

        # --- PHASE 5.8 - Check index integrity on each retrieval (optional) ---
        if self.config.get('check_index_on_retrieval', True):
            try:
                # FIX: Await the coroutine
                is_consistent, diagnostics = await self.vector_index.check_index_integrity()
                if not is_consistent:
                    drift_amount = abs(diagnostics.get("faiss_count", 0) - diagnostics.get("mapping_count", 0))
                    logger.warning(
                        "SynthiansMemoryCore",
                        "Vector index inconsistency detected during retrieval - ABORTING RETRIEVAL",
                        {"faiss_count": diagnostics.get("faiss_count"),
                         "id_mapping_count": diagnostics.get("mapping_count"),
                         "drift_amount": drift_amount}
                    )
                    repair_task = asyncio.create_task(self.detect_and_repair_index_drift(auto_repair=True))
                    return {
                        "success": False, "memories": [],
                        "error": f"Vector index drift detected ({drift_amount} entries). Auto-repair scheduled."
                    }
            except Exception as e:
                logger.error("SynthiansMemoryCore", f"Error checking index integrity during retrieval: {str(e)}", exc_info=True)
        # --- END PHASE 5.8 ---

        # Add diagnostic logging for parameter passing
        logger.debug(f"[retrieve_memories] START retrieve_memories: Received threshold argument = {threshold} (type: {type(threshold)})")

        query_embedding = None
        try:
            # Generate embedding for the query if necessary
            if query:
                query_embedding = await self.generate_embedding(query)
                if query_embedding is None:
                     logger.error("SynthiansMemoryCore", "Failed to generate query embedding.")
                     return {"success": False, "memories": [], "error": "Failed to generate query embedding"}
                logger.debug("SynthiansMemoryCore", "Query embedding generated")

                # Validate and normalize query embedding first
                query_embedding = self.geometry_manager._validate_vector(query_embedding, "Query Embedding")
                if query_embedding is None:
                    logger.error("SynthiansMemoryCore", "Query embedding validation failed")
                    return {"success": False, "memories": [], "error": "Invalid query embedding"}
                logger.debug(f"Validated query embedding - shape: {query_embedding.shape}")

            # Get the current threshold
            current_threshold = threshold
            if current_threshold is None and self.threshold_calibrator is not None:
                current_threshold = self.threshold_calibrator.get_current_threshold()
                logger.debug(f"Using calibrated threshold: {current_threshold:.4f}")
            elif current_threshold is None:
                current_threshold = self.config['initial_retrieval_threshold']
                logger.debug(f"Using default initial threshold: {current_threshold:.4f}")
            else:
                logger.debug(f"Using explicit threshold from request: {current_threshold:.4f}")

            # Make vector index integrity check configurable and periodic
            check_index_periodic = self.config.get('check_index_periodic', True)
            current_time = time.time()
            last_check_time = getattr(self, '_last_index_check_time', 0)
            check_interval = self.config.get('index_check_interval', 3600)

            # FIX: Use check_index_integrity (async)
            if check_index_periodic and not self.config.get('check_index_on_retrieval', True) and (current_time - last_check_time > check_interval):
                try:
                    # FIX: Await the coroutine
                    is_consistent, diagnostics = await self.vector_index.check_index_integrity()
                    self._last_index_check_time = current_time
                    logger.debug(f"Periodic Vector index status - Consistent: {is_consistent}, FAISS: {diagnostics.get('faiss_count')}, Mapping: {diagnostics.get('mapping_count')}")
                    if not is_consistent:
                        logger.warning(f"Periodic Vector index inconsistency detected! FAISS count: {diagnostics.get('faiss_count')}, Mapping count: {diagnostics.get('mapping_count')}")
                except Exception as periodic_check_err:
                     logger.error(f"Error during periodic index check: {periodic_check_err}", exc_info=True)

            # Perform the retrieval using candidate generation
            candidates, assembly_activation_scores = await self._get_candidate_memories(query_embedding, top_k * 5) # Get more candidates for filtering

            # ENHANCED: Log the raw candidates with more detail
            logger.info(f"[FAISS Results] Raw candidates count: {len(candidates)}")
            candidate_ids = [c.get('id') for c in candidates[:10]]
            logger.debug(f"First 10 candidate IDs: {candidate_ids}")

            # If no candidates found, return empty results
            if not candidates:
                logger.debug(f"No candidate memories found.")
                return {"success": True, "memories": [], "error": None}

            # Step 2: Activate assemblies based on query embedding for later boost calculation
            activated_assemblies_with_scores = []
            if query_embedding is not None:
                try:
                    activated_assemblies_with_scores = await self._activate_assemblies(query_embedding)
                    logger.debug(f"Activated {len(activated_assemblies_with_scores)} assemblies for retrieval operation")

                    # Create a lookup dictionary for quick access to activation scores
                    assembly_activation_scores = {asm.assembly_id: score for asm, score in activated_assemblies_with_scores}
                except Exception as e:
                    logger.error(f"Error during assembly activation: {e}", exc_info=True)

            # Step 3: Score and sort candidate memories
            scored_candidates = []
            if query_embedding is not None:
                logger.debug(f"Query embedding dimension: {query_embedding.shape}")
                logger.warning(f"CRITICAL DEBUG: Found {len(candidates)} raw candidates - first ID: {candidates[0].get('id') if candidates else 'None'}")

            for memory_dict in candidates:
                memory_embedding_list = memory_dict.get("embedding")
                if memory_embedding_list is not None and query_embedding is not None:
                    try:
                        # Re-convert list to numpy array
                        memory_embedding_np = np.array(memory_embedding_list, dtype=np.float32)

                        # ENHANCED: Add detailed validation logging
                        mem_id = memory_dict.get('id')
                        logger.debug(f"Processing memory {mem_id} for similarity calculation")

                        # ADDED: Explicit validation of memory embedding
                        memory_embedding_np = self.geometry_manager._validate_vector(memory_embedding_np, f"Memory {mem_id}")
                        if memory_embedding_np is None:
                            logger.warning(f"Memory {mem_id} embedding validation failed. Using zero vector.")
                            memory_embedding_np = np.zeros(self.config['embedding_dim'], dtype=np.float32)

                        # ADDED: Explicit alignment of vectors before similarity calculation
                        before_shapes = f"Before alignment - Query: {query_embedding.shape}, Memory: {memory_embedding_np.shape}"
                        logger.debug(before_shapes)

                        aligned_query, aligned_memory = self.geometry_manager._align_vectors(query_embedding, memory_embedding_np)

                        after_shapes = f"After alignment - Query: {aligned_query.shape}, Memory: {aligned_memory.shape}"
                        logger.debug(after_shapes)

                        # Check for NaN or Inf values in aligned vectors
                        if np.isnan(aligned_memory).any() or np.isinf(aligned_memory).any():
                            logger.warning(f"Memory {mem_id} aligned embedding contains NaN/Inf values. Replacing with zeros.")
                            aligned_memory = np.nan_to_num(aligned_memory, nan=0.0, posinf=0.0, neginf=0.0)

                        # Use GeometryManager to calculate similarity with aligned vectors
                        similarity = self.geometry_manager.calculate_similarity(aligned_query, aligned_memory)
                        logger.debug(f"  Calculated similarity: {similarity:.4f}")

                        memory_dict["similarity"] = similarity
                        memory_dict["relevance_score"] = similarity # Start with base similarity

                        # Apply assembly boost
                        assembly_boost = 0.0
                        max_activation = 0.0
                        boost_reason = "none"
                        associated_assembly_ids = set()
                        async with self._lock:
                            associated_assembly_ids = self.memory_to_assemblies.get(mem_id, set())

                        if associated_assembly_ids:
                            active_assemblies = []
                            for asm_id in associated_assembly_ids:
                                activation = assembly_activation_scores.get(asm_id, 0.0)
                                if activation > 0 and asm_id in self.assemblies and self.assemblies[asm_id].vector_index_updated_at:
                                    active_assemblies.append((asm_id, activation))

                            if active_assemblies:
                                max_asm_id, max_activation = max(active_assemblies, key=lambda x: x[1], default=("", 0.0))
                                boost_factor = self.config.get('assembly_boost_factor', 0.2)
                                assembly_boost = max_activation * boost_factor # Simple linear boost for now
                                boost_reason = f"linear(act:{max_activation:.2f}*f:{boost_factor:.2f})"
                                assembly_boost = min(assembly_boost, max(0.0, 1.0 - similarity)) # Clamp
                                memory_dict["relevance_score"] = min(1.0, similarity + assembly_boost)
                                logger.debug(f"Memory {mem_id}: Applied assembly boost {assembly_boost:.4f}")
                            else: boost_reason = "no_activated_sync_assemblies"
                        else: boost_reason = "no_associated_assemblies"

                        memory_dict["boost_info"] = {
                            "base_similarity": float(similarity), "assembly_boost": float(assembly_boost),
                            "max_activation": float(max_activation), "boost_reason": boost_reason
                        }
                        scored_candidates.append(memory_dict)

                    except Exception as e:
                        # Log the specific exception
                        logger.warning(f"Error calculating similarity for memory {memory_dict.get('id')}: {str(e)}")
                        logger.debug(traceback.format_exc())  # ADDED: Include stack trace for debugging
                        # Fallback: Include the memory with zero similarity rather than skipping it
                        memory_dict["similarity"] = 0.0
                        memory_dict["relevance_score"] = 0.0
                        scored_candidates.append(memory_dict)
                else:
                    # Log which specific condition failed
                    if memory_embedding_list is None:
                        logger.warning(f"Memory {memory_dict.get('id')} is missing embedding")
                    if query_embedding is None:
                        logger.warning("Query embedding is None")

                    # Even if embedding is missing, include in results with zero similarity
                    memory_dict["similarity"] = 0.0
                    memory_dict["relevance_score"] = 0.0
                    scored_candidates.append(memory_dict)

            # Sort by relevance score (which includes assembly boost)
            sorted_candidates = sorted(scored_candidates, key=lambda x: x.get("relevance_score", 0.0), reverse=True)

            # ENHANCED: Log all candidates with their scores before filtering
            logger.info(f"[Similarity Results] Found {len(sorted_candidates)} scored candidates before threshold filtering")
            logger.debug(f"Threshold filtering: Using threshold {current_threshold:.4f}")

            relevance_scores = [(c.get('id'), c.get('relevance_score', 0.0)) for c in sorted_candidates[:10]]
            logger.debug(f"Top 10 relevance scores (with boost): {relevance_scores}")

            # Apply threshold filtering based on *base similarity* (before boost)
            logger.info(f"[Threshold Filtering] Starting threshold filtering with {len(sorted_candidates)} candidates using base similarity")
            filtered_candidates = []
            candidates_filtered_out = []

            for c in sorted_candidates:
                base_similarity = c.get("similarity", 0.0) # Use base similarity for threshold
                mem_id = c.get("id", "unknown")
                if base_similarity >= current_threshold:
                    filtered_candidates.append(c)
                    logger.debug(f"Memory {mem_id} PASSED threshold with base similarity {base_similarity:.4f} >= {current_threshold:.4f}")
                else:
                    candidates_filtered_out.append((mem_id, base_similarity))
                    logger.debug(f"Memory {mem_id} FILTERED OUT with base similarity {base_similarity:.4f} < {current_threshold:.4f}")

            # Log summary of threshold filtering results
            logger.info(f"[Threshold Filtering] Kept {len(filtered_candidates)} candidates, filtered out {len(candidates_filtered_out)} candidates")

            # Log the first few filtered out candidates for debugging
            if candidates_filtered_out:
                logger.debug(f"First 5 filtered out (ID, base_similarity): {candidates_filtered_out[:5]}")

            # Step 4: Apply emotional gating
            if user_emotion and self.emotional_gating:
                logger.info(f"[Emotional Gating] Applying with user_emotion: {user_emotion}, candidates: {len(filtered_candidates)}")
                try:
                    filtered_candidates = await self.emotional_gating.gate_memories_by_context(
                        filtered_candidates, user_emotion_context=user_emotion
                    )
                    logger.info(f"[Emotional Gating] Result: {len(filtered_candidates)} candidates")
                except Exception as e:
                    logger.error(f"Error during emotional gating: {e}", exc_info=True)

            # Step 5: Apply metadata filtering
            if metadata_filter:
                logger.info(f"[Metadata Filtering] Applying filter: {metadata_filter}")
                pre_filter_count = len(filtered_candidates)
                filtered_candidates = self._filter_by_metadata(filtered_candidates, metadata_filter)
                filter_diff = pre_filter_count - len(filtered_candidates)
                logger.info(f"[Metadata Filtering] Result: {len(filtered_candidates)} candidates remain ({filter_diff} removed)")

            # Re-sort final candidates by relevance_score after all filtering
            final_sorted_candidates = sorted(filtered_candidates, key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            logger.info(f"[Final Filtering] Total candidates after all filters: {len(final_sorted_candidates)}")

            # Select top_k
            final_memories = final_sorted_candidates[:top_k]
            logger.info(f"[Results] Returning {len(final_memories)} memories (requested {top_k})")

            # Final log
            retrieval_time = (time.time() - start_time) * 1000
            logger.info("SynthiansMemoryCore", f"Retrieved {len(final_memories)} memories", {
                "top_k": top_k, "threshold": current_threshold, "user_emotion": user_emotion, "time_ms": f"{retrieval_time:.2f}"
            })
            response = {"success": True, "memories": final_memories, "error": None}
            logger.info(f"[Response] Payload stats: success={response['success']}, memories_count={len(response['memories'])}")
            return response

        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error in retrieve_memories: {str(e)}", exc_info=True)
            return {"success": False, "memories": [], "error": str(e)}

    async def _get_candidate_memories(self, query_embedding: Optional[np.ndarray], limit: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Retrieve candidate memories using assembly activation and direct vector search.

        Returns:
            Tuple containing:
            - List of candidate memories as dictionaries
            - Dictionary mapping assembly_id to activation score
        """
        if query_embedding is None:
            logger.warning("SynthiansMemoryCore", "_get_candidate_memories called with no query embedding.")
            return [], {}

        logger.debug(f"[Candidate Gen] Query embedding shape: {query_embedding.shape}")

        assembly_candidates = set()
        direct_candidates = set()
        assembly_activation_scores = {} # Store scores here

        # 1. Assembly Activation
        activated_assemblies = await self._activate_assemblies(query_embedding)
        logger.debug(f"[Candidate Gen] Got {len(activated_assemblies)} activated assemblies from _activate_assemblies")

        for assembly, score in activated_assemblies:
            asm_id = assembly.assembly_id
            assembly_activation_scores[asm_id] = score # Populate the score dict
            if hasattr(assembly, 'memories') and isinstance(assembly.memories, (set, list)):
                assembly_candidates.update(assembly.memories)
                logger.debug(f"[Candidate Gen] Added {len(assembly.memories)} memories from activated assembly {asm_id}")

        logger.info(f"[Candidate Gen] Found {len(assembly_candidates)} candidates from assembly activation.")

        # 2. Direct Vector Search
        faiss_count = self.vector_index.count()
        logger.info(f"[Candidate Gen] Vector index stats: FAISS count={faiss_count}")

        if faiss_count > 0:
            # Determine K for search, ensuring it's not more than available vectors
            k_search = min(faiss_count, limit * 2) # Search more initially
            search_results = await self.vector_index.search_async(query_embedding, k=k_search)
            logger.info(f"[Candidate Gen] FAISS search (k={k_search}) returned {len(search_results)} results")
            for memory_id, similarity in search_results:
                direct_candidates.add(memory_id)
        else:
            logger.warning(f"[Candidate Gen] FAISS index is empty! Skipping direct search.")

        # 3. Fallback Recent Memories (If needed)
        recent_fallback_count = self.config.get('recent_fallback_count', 5)
        combined_candidate_count = len(assembly_candidates.union(direct_candidates))
        if recent_fallback_count > 0 and combined_candidate_count < limit:
            async with self._lock:
                memory_ids_from_persistence = list(self.persistence.memory_index.keys())
            if memory_ids_from_persistence:
                needed_count = limit - combined_candidate_count
                recent_candidates_set = set(memory_ids_from_persistence[-min(needed_count + len(assembly_candidates), len(memory_ids_from_persistence)):]) # Get enough recent ones
                new_fallback_candidates = recent_candidates_set - assembly_candidates - direct_candidates
                if new_fallback_candidates:
                    logger.info(f"[Candidate Gen] Added {len(new_fallback_candidates)} recent memories as fallback.")
                    direct_candidates.update(new_fallback_candidates)

        # Combine all candidate IDs
        all_candidate_ids = assembly_candidates.union(direct_candidates)
        logger.info(f"[Candidate Gen] Found {len(all_candidate_ids)} total unique candidate IDs.")

        # Fetch MemoryEntry/Assembly objects as dictionaries
        final_candidates = []
        for mem_id in list(all_candidate_ids)[:limit*3]: # Limit number loaded initially
            logger.debug(f"[Candidate Gen] Attempting to load candidate ID: {mem_id}")
            memory_obj = await self.get_memory_by_id_async(mem_id)
            if memory_obj:
                mem_dict = memory_obj.to_dict()
                final_candidates.append(mem_dict)
            else:
                # Also check if it's an assembly ID that wasn't loaded via activation
                if mem_id.startswith("asm:"):
                     asm_id_only = mem_id[4:]
                     assembly_obj = await self._load_assembly(asm_id_only) # Use helper
                     if assembly_obj:
                          # We don't usually return assemblies directly, but log that we found it
                          logger.debug(f"[Candidate Gen] Found assembly {asm_id_only} corresponding to candidate ID {mem_id}, but not adding to final candidates.")
                     else:
                         logger.warning(f"[Candidate Gen] Failed to load potential assembly {asm_id_only}!")
                else:
                    logger.warning(f"[Candidate Gen] Failed to load memory {mem_id}!")

        logger.info(f"[Candidate Gen] Returning {len(final_candidates)} final candidate dictionaries for scoring/filtering.")
        return final_candidates, assembly_activation_scores


    async def _activate_assemblies(self, query_embedding: np.ndarray) -> List[Tuple[MemoryAssembly, float]]:
        """Find and activate assemblies based on query similarity.

        Returns:
            List of (assembly, similarity) tuples for activated assemblies.
        """
        if not self.vector_index:
            logger.warning("Cannot activate assemblies: vector_index is None")
            return []

        if query_embedding is None:
            logger.warning("Cannot activate assemblies: query_embedding is None")
            return []

        logger.debug(f"[Assembly Debug] Query embedding shape: {query_embedding.shape}")

        now = datetime.now(timezone.utc)
        drift_limit = self.config.get('max_allowed_drift_seconds', 86400)
        assembly_threshold = self.config.get('assembly_threshold', 0.85)
        logger.info(f"[Assembly Debug] Assembly activation threshold: {assembly_threshold}, drift_limit: {drift_limit}s")

        prefix = "asm:"
        logger.debug(f"[Assembly Debug] Searching for assemblies with prefix: {prefix}")

        try:
            stats = self.vector_index.get_stats()
            logger.debug(f"[Assembly Debug] Vector index stats: {stats}")

            faiss_count = self.vector_index.count()
            if faiss_count == 0:
                 logger.warning("[Assembly Debug] FAISS index empty, cannot search for assemblies.")
                 return []

            search_results = await self.vector_index.search_async(
                query_embedding,
                k=min(faiss_count, 200) # Search up to 200, avoid error if empty
            )
            logger.info(f"[Assembly Debug] FAISS search returned {len(search_results)} results")

            # Filter for assembly IDs
            asm_results = [(memory_id, similarity) for memory_id, similarity in search_results if memory_id.startswith(prefix)]
            logger.debug(f"[Assembly Debug] Found {len(asm_results)} potential assemblies after filtering")

            activated_assemblies = []
            max_activation_time = now - timedelta(seconds=drift_limit)

            for asm_id_with_prefix, similarity in asm_results:
                if similarity < assembly_threshold: continue # Skip below threshold

                assembly_id = asm_id_with_prefix[4:]
                assembly = await self._load_assembly(assembly_id) # Use helper

                if assembly is None:
                    logger.warning(f"[ACTIVATE_DBG] Assembly '{assembly_id}' not found after lookup. Skipping.")
                    continue

                # Check synchronization status and drift
                updated_at = assembly.vector_index_updated_at
                if updated_at is None:
                    logger.debug(f"[ACTIVATE_DBG] Skipping '{assembly_id}': Not yet synchronized.")
                    continue

                drift_seconds = (now - updated_at).total_seconds()
                if updated_at < max_activation_time:
                    logger.debug(f"[ACTIVATE_DBG] Skipping '{assembly_id}': Drift limit exceeded ({drift_seconds:.2f}s > {drift_limit}s).")
                    continue

                # All checks passed
                logger.info(f"[ACTIVATE_DBG] ACTIVATE SUCCESS for '{assembly_id}' with similarity {similarity:.4f}")
                activated_assemblies.append((assembly, similarity))
                if hasattr(assembly, 'activate'): assembly.activate(similarity)
                await self._track_assembly_activation(assembly_id)

            logger.debug(f"[Assembly Debug] Total activated assemblies: {len(activated_assemblies)}")
            return activated_assemblies

        except Exception as e:
            logger.error(f"Error during assembly activation: {str(e)}", exc_info=True)
            return []


    async def _update_assemblies(self, memory: MemoryEntry):
        """Find or create assemblies for a new memory."""
        if not self.config.get('enable_assemblies', True): return
        if memory.embedding is None: return
        validated_mem_emb = self.geometry_manager._validate_vector(memory.embedding)
        if validated_mem_emb is None: return

        suitable_assemblies = []
        best_similarity = -1.0 # Initialize to allow any positive similarity
        best_assembly_id = None
        assembly_threshold = self.config.get('assembly_threshold', 0.85)

        logger.info(f"[Assembly Update] Processing memory {memory.id}")

        async with self._lock:
             assemblies_copy = dict(self.assemblies) # Iterate over a copy

        for assembly_id, assembly in assemblies_copy.items():
             if assembly.composite_embedding is None: continue # Skip if no embedding
             try:
                  similarity = assembly.get_similarity(validated_mem_emb)
                  logger.debug(f"[Assembly Update] Sim({memory.id} -> {assembly_id}): {similarity:.4f} (Thresh: {assembly_threshold:.4f})")
                  if similarity >= assembly_threshold:
                       suitable_assemblies.append((assembly_id, similarity))
                  if similarity > best_similarity:
                       best_similarity = similarity
                       best_assembly_id = assembly_id
             except Exception as sim_err:
                  logger.error(f"Error calculating similarity between {memory.id} and {assembly_id}: {sim_err}", exc_info=True)


        suitable_assemblies.sort(key=lambda x: x[1], reverse=True)
        added_count = 0
        max_assemblies_per = self.config.get('max_assemblies_per_memory', 3)

        # Add to existing assemblies
        for assembly_id, similarity in suitable_assemblies[:max_assemblies_per]:
            async with self._lock:
                if assembly_id in self.assemblies:
                    assembly = self.assemblies[assembly_id]
                    if assembly.add_memory(memory, validated_mem_emb):
                        added_count += 1
                        self._dirty_memories.add(assembly.assembly_id)
                        self.memory_to_assemblies.setdefault(memory.id, set()).add(assembly_id)
                        logger.info(f"[Assembly Update] Added {memory.id} to EXISTING assembly {assembly_id} (Sim: {similarity:.4f})")
                        # Schedule save/index outside loop
                    else:
                        logger.warning(f"Failed add_memory {memory.id} to {assembly_id}")
                else: logger.warning(f"Assembly {assembly_id} disappeared before lock")

        # Create new assembly if needed
        create_threshold = assembly_threshold * 0.5
        if added_count == 0 and (best_similarity < create_threshold or len(assemblies_copy) == 0):
            async with self._lock:
                 if not any(asm_id in self.assemblies for asm_id in self.memory_to_assemblies.get(memory.id, set())): # Double check under lock
                     logger.info(f"[Assembly Update] Creating NEW assembly for memory {memory.id} (Best sim: {best_similarity:.4f})")
                     new_assembly = MemoryAssembly(geometry_manager=self.geometry_manager, name=f"Assembly around {memory.id[:8]}")
                     if new_assembly.add_memory(memory, validated_mem_emb):
                          if new_assembly.composite_embedding is not None:
                              self.assemblies[new_assembly.assembly_id] = new_assembly
                              self._dirty_memories.add(new_assembly.assembly_id)
                              self.memory_to_assemblies.setdefault(memory.id, set()).add(new_assembly.assembly_id)
                              added_count += 1
                              logger.info(f"[Assembly Update] Created and added {memory.id} to NEW assembly {new_assembly.assembly_id}")
                              # Schedule save/index outside loop
                          else: logger.error(f"New assembly {new_assembly.assembly_id} missing composite embedding!")
                     else: logger.error(f"Failed to add seeding memory {memory.id} to new assembly")

        # Save and index updated/new assemblies outside the main loops
        assemblies_to_process = set()
        if memory.id in self.memory_to_assemblies:
             async with self._lock: # Get associated assembly IDs safely
                  assemblies_to_process.update(self.memory_to_assemblies.get(memory.id, set()))

        for assembly_id in assemblies_to_process:
             async with self._lock: # Lock to get assembly object
                  assembly_obj = self.assemblies.get(assembly_id)
             if assembly_obj and assembly_id in self._dirty_memories: # Check if still dirty
                  logger.info(f"[Assembly Update] Saving & Indexing updated assembly {assembly_id}")
                  save_ok = await self.persistence.save_assembly(assembly_obj)
                  if save_ok:
                      await self._index_assembly_embedding(assembly_obj)
                  else:
                      logger.error(f"[Assembly Update] FAILED to save assembly {assembly_id}.")


        logger.info(f"[Assembly Update] Memory {memory.id} processed. Added to {added_count} assemblies.")


    async def detect_contradictions(self, threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Detect potential causal contradictions using embeddings."""
        contradictions = []
        async with self._lock: # Access shared _memories
            memories_list = list(self._memories.values())

        # Basic Keyword Filtering for Causal Statements (Can be improved with NLP)
        causal_keywords = ["causes", "caused", "leads to", "results in", "effect of", "affects"]
        causal_memories = [m for m in memories_list if m.embedding is not None and hasattr(m, 'content') and any(k in m.content.lower() for k in causal_keywords)]

        if len(causal_memories) < 2: return []

        logger.info("SynthiansMemoryCore", f"Checking {len(causal_memories)} causal memories for contradictions.")

        # Compare pairs (simplified N^2 comparison, can be optimized)
        compared_pairs = set()
        for i in range(len(causal_memories)):
            for j in range(i + 1, len(causal_memories)):
                mem_a = causal_memories[i]
                mem_b = causal_memories[j]

                # Calculate similarity
                similarity = self.geometry_manager.calculate_similarity(mem_a.embedding, mem_b.embedding)

                # Basic Topic Overlap Check (can be improved)
                words_a = set(mem_a.content.lower().split())
                words_b = set(mem_b.content.lower().split())
                common_words = words_a.intersection(words_b)
                overlap_ratio = len(common_words) / min(len(words_a), len(words_b)) if min(len(words_a), len(words_b)) > 0 else 0

                # Check for potential semantic opposition (basic keyword check)
                opposites = [("increase", "decrease"), ("up", "down"), ("positive", "negative"), ("high", "low")]
                has_opposite = False
                content_a_lower = mem_a.content.lower()
                content_b_lower = mem_b.content.lower()
                for w1, w2 in opposites:
                    if (w1 in content_a_lower and w2 in content_b_lower) or \
                       (w2 in content_a_lower and w1 in content_b_lower):
                        has_opposite = True
                        break

                # If high similarity, sufficient topic overlap, and potential opposition -> contradiction
                if similarity >= threshold and overlap_ratio > 0.3 and has_opposite:
                     contradictions.append({
                          "memory_a_id": mem_a.id,
                          "memory_a_content": mem_a.content,
                          "memory_b_id": mem_b.id,
                          "memory_b_content": mem_b.content,
                          "similarity": similarity,
                          "overlap_ratio": overlap_ratio
                     })

        logger.info("SynthiansMemoryCore", f"Detected {len(contradictions)} potential contradictions.")
        return contradictions


    # --- Background Tasks ---

    async def _persistence_loop(self):
        """Periodically persist changed memories and assemblies."""
        logger.info("SynthiansMemoryCore", "Persistence loop started.")
        persist_interval = self.config.get('persistence_interval', 60.0)
        try:
            while not self._shutdown_signal.is_set():
                # Wait for the configured interval OR the shutdown signal
                try:
                    await asyncio.wait_for(
                        self._shutdown_signal.wait(),
                        timeout=persist_interval
                    )
                    # If wait() finished without timeout, it means signal was set
                    logger.info("SynthiansMemoryCore", "Persistence loop: Shutdown signal received during wait.")
                    break # Exit loop if shutdown signal is set
                except asyncio.TimeoutError:
                    # Timeout occurred, time to persist
                    if not self._shutdown_signal.is_set(): # Double-check signal
                        logger.debug("SynthiansMemoryCore", "Running periodic persistence.")
                        await self._persist_dirty_items() # Persist dirty items

                        # --- PHASE 5.8.A: Vector Index Persistence ---
                        # Ensure vector index is saved periodically to prevent data loss
                        if hasattr(self, 'vector_index') and self.vector_index and self._initialized:
                            logger.debug("SynthiansMemoryCore", "Attempting periodic vector index save...")
                            try:
                                # Add a timeout for safety
                                save_success = await asyncio.wait_for(self.vector_index.save_async(), timeout=10.0)
                                if save_success:
                                    logger.debug("SynthiansMemoryCore", "Periodic vector index save successful.")
                                else:
                                    logger.warning("SynthiansMemoryCore", "Periodic vector index save failed.")
                            except asyncio.TimeoutError:
                                 logger.warning("SynthiansMemoryCore", "Timeout during periodic vector index save.")
                            except Exception as e:
                                logger.error(f"SynthiansMemoryCore: Error during periodic vector index save: {e}", exc_info=True)
                        # --- END PHASE 5.8.A ---
                except asyncio.CancelledError:
                    logger.info("SynthiansMemoryCore", "Persistence loop cancelled during wait.")
                    break # Exit loop if cancelled
        except asyncio.CancelledError:
            logger.info("SynthiansMemoryCore", "Persistence loop received cancel signal.")
        except Exception as e:
            logger.error("SynthiansMemoryCore", "Persistence loop error", {"error": str(e)}, exc_info=True)
        finally:
            logger.info("SynthiansMemoryCore", "Persistence loop stopped.")


    async def _decay_and_pruning_loop(self):
        """Periodically decay memory scores and prune/merge assemblies."""
        logger.info("SynthiansMemoryCore","Decay/Pruning/Merging loop started.") # Updated log
        decay_interval = self.config.get('decay_interval', 3600.0)
        prune_interval = self.config.get('prune_check_interval', 10.0)
        merge_interval = self.config.get('merge_check_interval', prune_interval) # Reuse prune interval
        check_interval = min(decay_interval, prune_interval, merge_interval, 5.0) # Check frequently
        last_decay_time = time.monotonic()
        last_prune_time = time.monotonic()
        last_merge_time = time.monotonic()

        try:
            while not self._shutdown_signal.is_set():
                # Wait for the configured interval
                try:
                    await asyncio.wait_for(
                        self._shutdown_signal.wait(),
                        timeout=check_interval
                    )
                    break # Shutdown signal received
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if self._shutdown_signal.is_set(): break # Check again after timeout

                    # Decay Check
                    if now - last_decay_time >= decay_interval:
                       logger.debug("Running memory decay check...")
                       # --- TODO: Add actual decay logic here ---
                       # Iterate self._memories.values() (under lock?)
                       # Decrease quickrecal_score based on time_since_last_access/update
                       # Potentially remove memories below min_quickrecal_for_ltm
                       # Remember to remove from vector index and mark dirty/delete persistence
                       last_decay_time = now

                    # Pruning Check
                    if self.config.get('enable_assembly_pruning', True) and (now - last_prune_time >= prune_interval):
                        logger.info("SynthiansMemoryCore","Running assembly pruning check.")
                        try:
                            await self._prune_if_needed()
                            last_prune_time = now
                        except Exception as prune_e:
                            logger.error("SynthiansMemoryCore","Error during pruning", {"error": str(prune_e)}, exc_info=True)

                    # Merge Check
                    if self.config.get('enable_assembly_merging', True) and (now - last_merge_time >= merge_interval):
                         logger.info("SynthiansMemoryCore", "Running assembly merging check.")
                         try:
                             await self._merge_similar_assemblies() # Ensure this exists and is awaited
                             last_merge_time = now
                         except Exception as merge_e:
                             logger.error("SynthiansMemoryCore", "Error during merging", {"error": str(merge_e)}, exc_info=True)

                except asyncio.CancelledError:
                    logger.info("SynthiansMemoryCore","Decay/Pruning/Merging loop cancelled.")
                    break
        except Exception as e:
            logger.error("SynthiansMemoryCore","Decay/Pruning/Merging loop error", {"error": str(e)}, exc_info=True)
        finally:
            logger.info("SynthiansMemoryCore","Decay/Pruning/Merging loop stopped.")

    async def _prune_if_needed(self):
        """Check if pruning is needed and perform it if enabled in config.
        This method is called periodically by the background tasks.
        """
        # Check if assembly pruning is enabled
        enable_assembly_pruning = self.config.get("enable_assembly_pruning", False)

        if not enable_assembly_pruning:
            logger.debug("[PRUNE] Assembly pruning is disabled")
            return

        try:
            # Get pruning parameters from config
            max_assemblies = self.config.get("max_assemblies", self.config.get("max_memory_entries", 50000) // 10) # Example: 1/10th of memory entries
            prune_threshold_percent = self.config.get("assembly_prune_threshold_percent", 0.8) # Use percentage

            # Check if we're above threshold for pruning
            async with self._lock: # Lock needed to access self.assemblies safely
                current_count = len(self.assemblies)

            prune_trigger_count = int(max_assemblies * prune_threshold_percent)

            if current_count < prune_trigger_count:
                logger.debug(f"[PRUNE] No pruning needed. Current: {current_count}, Trigger: {prune_trigger_count}")
                return

            logger.info(f"[PRUNE] Assembly count {current_count} exceeds threshold {prune_trigger_count}, pruning needed")

            # Find least-recently activated assemblies to prune
            assemblies_with_activation = []
            async with self._lock:
                for asm_id, asm in self.assemblies.items():
                    # Use datetime.min as fallback for sorting if last_activation is None
                    last_act = asm.last_activation if asm.last_activation else datetime.min.replace(tzinfo=timezone.utc)
                    assemblies_with_activation.append((asm_id, last_act))

            # Sort by activation time (oldest first)
            assemblies_to_prune_ids = [
                asm_id for asm_id, _ in sorted(
                    assemblies_with_activation,
                    key=lambda item: item[1] # Sort by datetime object
                )
            ]

            # Determine how many to prune (e.g., prune down to 70% of max)
            prune_down_percent = self.config.get('assembly_prune_down_percent', 0.7)
            num_to_prune = current_count - int(max_assemblies * prune_down_percent)

            if num_to_prune <= 0:
                 logger.info(f"[PRUNE] Calculated 0 assemblies to prune.")
                 return

            logger.info(f"[PRUNE] Pruning {num_to_prune} assemblies (oldest first)")

            # Remove the assemblies (call _remove_assembly which handles locking internally)
            pruned_count = 0
            for assembly_id in assemblies_to_prune_ids[:num_to_prune]:
                remove_success = await self._remove_assembly(assembly_id)
                if remove_success:
                    pruned_count += 1

            logger.info(f"[PRUNE] Pruning complete. Successfully removed: {pruned_count}.")
            # Log new count after pruning
            async with self._lock:
                 logger.info(f"[PRUNE] New assembly count: {len(self.assemblies)}")


        except Exception as e:
            logger.error(f"[PRUNE] Error during assembly pruning: {e}", exc_info=True)

    # --- Tool Interface ---

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return descriptions of available tools for LLM integration."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_memories_tool",
                    "description": "Retrieve relevant memories based on a query text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."},
                            "top_k": {"type": "integer", "description": "Max number of results.", "default": 5},
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "process_new_memory_tool",
                    "description": "Process and store a new piece of information or experience.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "The content of the memory."},
                            "metadata": {"type": "object", "description": "Optional metadata (source, type, etc.)."}
                        },
                        "required": ["content"]
                    }
                }
            },
             {
                "type": "function",
                "function": {
                    "name": "provide_retrieval_feedback_tool",
                    "description": "Provide feedback on the relevance of retrieved memories.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string", "description": "The ID of the memory being rated."},
                             "similarity_score": {"type": "number", "description": "The similarity score assigned during retrieval."},
                            "was_relevant": {"type": "boolean", "description": "True if the memory was relevant, False otherwise."}
                        },
                        "required": ["memory_id", "similarity_score", "was_relevant"]
                    }
                }
            },
             {
                "type": "function",
                "function": {
                    "name": "detect_contradictions_tool",
                    "description": "Check for potential contradictions within recent memory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                             "threshold": {"type": "number", "description": "Similarity threshold for contradiction.", "default": 0.75}
                        }
                    }
                }
            }
            # TODO: Add tools for assemblies, emotional state, etc.
        ]

    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool call from an external agent (e.g., LLM)."""
        logger.info("SynthiansMemoryCore", f"Handling tool call: {tool_name}", {"args": args})
        try:
            if tool_name == "retrieve_memories_tool":
                 query = args.get("query")
                 top_k = args.get("top_k", 5)
                 # Retrieve memories method handles embedding generation
                 response_data = await self.retrieve_memories(query=query, top_k=top_k)
                 # Return simplified dicts for LLM
                 if response_data["success"]:
                      # Use relevance_score which includes boost
                      return {"memories": [{"id": m.get("id"), "content": m.get("content"), "score": m.get("relevance_score", m.get("similarity")) } for m in response_data["memories"]]}
                 else:
                      return {"success": False, "error": response_data.get("error", "Retrieval failed")}

            elif tool_name == "process_new_memory_tool":
                 content = args.get("content")
                 metadata = args.get("metadata")
                 # Embedding generation happens in process_new_memory if needed
                 entry = await self.process_new_memory(content=content, metadata=metadata)
                 return {"success": entry is not None, "memory_id": entry.id if entry else None}

            elif tool_name == "provide_retrieval_feedback_tool":
                 memory_id = args.get("memory_id")
                 similarity_score = args.get("similarity_score")
                 was_relevant = args.get("was_relevant")
                 if self.threshold_calibrator:
                      # Need provide_feedback method if using calibrator
                      # await self.provide_feedback(memory_id, similarity_score, was_relevant)
                      logger.warning("provide_retrieval_feedback_tool called but provide_feedback method not fully implemented.")
                      return {"success": True, "message": "Feedback noted (implementation pending)."}
                 else:
                      return {"success": False, "error": "Adaptive thresholding not enabled."}

            elif tool_name == "detect_contradictions_tool":
                 threshold = args.get("threshold", 0.75)
                 contradictions = await self.detect_contradictions(threshold)
                 return {"success": True, "contradictions_found": len(contradictions), "contradictions": contradictions}

            else:
                 logger.warning("SynthiansMemoryCore", f"Unknown tool called: {tool_name}")
                 return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error handling tool call {tool_name}", {"error": str(e)}, exc_info=True)
            return {"success": False, "error": str(e)}

    # --- Helper & Placeholder Methods ---

    def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry by its ID from the cache.
           NOTE: This is synchronous and operates on the current in-memory cache.
           It does NOT acquire the async lock. Caller must manage concurrency.

        Args:
            memory_id: The unique identifier of the memory to retrieve

        Returns:
            The MemoryEntry if found in cache, None otherwise
        """
        # No lock needed here - caller (e.g., update_memory) holds the lock
        memory = self._memories.get(memory_id)
        if memory:
            logger.debug("SynthiansMemoryCore", f"Retrieved memory {memory_id} directly from cache (sync).")
        else:
            logger.warning("SynthiansMemoryCore", f"Memory {memory_id} not found in cache (sync).")
        return memory

    async def get_memory_by_id_async(self, memory_id: str) -> Optional[MemoryEntry]:
        """Asynchronously retrieve a specific memory entry by its ID, loading from disk if needed.

        Unlike the synchronous get_memory_by_id which only checks the cache, this method
        will attempt to load the memory from disk if it's not found in the cache but exists
        in the index.

        Args:
            memory_id: The unique identifier of the memory to retrieve

        Returns:
            The MemoryEntry if found in cache or successfully loaded, None otherwise
        """
        async with self._lock:
            # First check if it's already in the memory cache
            memory = self._memories.get(memory_id)
            if memory:
                logger.debug("SynthiansMemoryCore", f"Retrieved memory {memory_id} from cache.")
                # Update access stats if MemoryEntry supports them
                if hasattr(memory, 'access_count'): memory.access_count += 1
                if hasattr(memory, 'last_access_time'): memory.last_access_time = datetime.now(timezone.utc)
                return memory

            # Not in cache, check if it's in the index and try to load it
            if memory_id in self.persistence.memory_index:
                logger.debug("SynthiansMemoryCore", f"Memory {memory_id} not in cache, loading from persistence...")
                memory = await self.persistence.load_memory(memory_id)
                if memory:
                    # Add to cache
                    self._memories[memory_id] = memory
                    # Update access stats if MemoryEntry supports them
                    if hasattr(memory, 'access_count'): memory.access_count += 1
                    if hasattr(memory, 'last_access_time'): memory.last_access_time = datetime.now(timezone.utc)

                    # If this is our first time seeing this memory and we have a vector index,
                    # add it to the index if it has a valid embedding
                    # Note: This might be redundant if rebuild handles everything, but acts as a fallback
                    if memory.embedding is not None and self.vector_index is not None:
                        if memory_id not in self.vector_index.id_to_index:
                            add_ok = await self.vector_index.add_async(memory_id, memory.embedding)
                            if add_ok:
                                logger.debug("SynthiansMemoryCore", f"Added memory {memory_id} to vector index on first load.")
                            else:
                                logger.error(f"SynthiansMemoryCore", f"Failed to add memory {memory_id} to vector index on first load. Queuing.")
                                await self._pending_vector_updates.put({
                                    "operation": "add", "id": memory_id, "embedding": memory.embedding.tolist(),
                                    "is_assembly": False, "timestamp": datetime.now(timezone.utc).isoformat(), "retry_count": 0
                                })

                    logger.debug("SynthiansMemoryCore", f"Successfully loaded memory {memory_id} from persistence.")
                    return memory
                else:
                    logger.warning("SynthiansMemoryCore", f"Failed to load memory {memory_id} from persistence despite being in the index.")
                    return None
            else:
                logger.warning("SynthiansMemoryCore", f"Memory {memory_id} not found in cache or index.")
                return None

    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry with provided updates.

        Args:
            memory_id: ID of the memory to update
            updates: Dictionary of field updates

        Returns:
            bool: Whether the update was successful
        """
        if not self._initialized: await self.initialize()

        try:
            # Use the main lock to avoid race conditions during updates
            async with self._lock:
                # Look up the memory first (use async version to load if needed)
                memory = await self.get_memory_by_id_async(memory_id)
                if memory is None:
                    logger.warning(f"Cannot update non-existent memory {memory_id}")
                    return False

                # Extract metadata updates if present
                metadata_to_update = updates.pop('metadata', None)

                # Track if key fields are updated
                score_updated = False
                embedding_updated = False

                # Apply direct field updates
                for key, value in updates.items():
                    # Special handling for quickrecal_score
                    if key == "quickrecal_score":
                        try:
                            new_score_val = float(value)
                            new_score_val = max(0.0, min(1.0, new_score_val))
                            if abs(memory.quickrecal_score - new_score_val) > 1e-6:
                                 memory.quickrecal_score = new_score_val
                                 score_updated = True # Mark score as updated
                        except (ValueError, TypeError):
                            logger.warning("SynthiansMemoryCore", f"Invalid quickrecal_score value: {value}")
                            continue
                    elif key == "embedding":
                        # Validate and normalize new embedding
                        validated_emb = self.geometry_manager._validate_vector(value, f"Update Emb {memory_id}")
                        if validated_emb is not None:
                            aligned_emb, _ = self.geometry_manager._align_vectors(validated_emb, np.zeros_like(validated_emb))
                            normalized_emb = self.geometry_manager._normalize(aligned_emb)
                            # Check if it actually changed
                            if memory.embedding is None or not np.allclose(memory.embedding, normalized_emb):
                                memory.embedding = normalized_emb
                                embedding_updated = True
                        else:
                            logger.warning(f"Invalid embedding provided in update for {memory_id}, skipping embedding update.")
                    elif hasattr(memory, key):
                         setattr(memory, key, value) # Update other direct attributes
                    else:
                        logger.warning(f"Unknown/invalid field '{key}' in memory update for {memory_id}")

                # Apply metadata updates after other fields have been processed
                if metadata_to_update:
                    if memory.metadata is None:
                        memory.metadata = {}
                    # Use deep update to properly handle nested dictionaries
                    deep_update(memory.metadata, metadata_to_update)

                # Update quickrecal timestamp ONLY if the score actually changed
                if score_updated:
                    if memory.metadata is None: memory.metadata = {}
                    memory.metadata['quickrecal_updated_at'] = datetime.now(timezone.utc).isoformat()
                    logger.debug(f"quickrecal_updated_at set for memory {memory_id}")

                # Update the vector index ONLY if the embedding changed
                vector_update_success = True # Assume success unless embedding update fails
                if embedding_updated:
                    if memory.embedding is not None and self.vector_index is not None:
                        logger.debug(f"Updating vector index for memory {memory_id} due to embedding change")
                        try:
                            # Call update_entry_async which handles add/update logic
                            vector_update_success = await self.vector_index.update_entry_async(memory_id, memory.embedding)

                            if vector_update_success:
                                logger.info(f"Successfully updated vector index for memory {memory_id}.")
                            else:
                                logger.error(f"Failed vector index update for memory {memory_id}. Will be queued by update_entry_async.")
                                # update_entry_async should queue internally on failure

                        except Exception as index_update_err:
                            logger.error(f"EXCEPTION during vector index update for memory {memory_id}: {index_update_err}", exc_info=True)
                            vector_update_success = False # Ensure failure on exception
                            # Queue manually if exception wasn't handled by update_entry_async
                            await self._pending_vector_updates.put({
                                "operation": "update", "id": memory_id, "embedding": memory.embedding.tolist(),
                                "is_assembly": False, "timestamp": datetime.now(timezone.utc).isoformat(), "retry_count": 0
                            })
                    else:
                        logger.warning(f"Memory {memory_id} embedding changed, but vector index not available. Skipping index update.")

                # Mark as dirty for persistence regardless of index success (metadata/score might have changed)
                self._dirty_memories.add(memory_id)
                logger.debug(f"Memory {memory_id} updated in memory (marked dirty)")

                # Return overall success (mainly based on vector update if it happened)
                if embedding_updated and not vector_update_success:
                    logger.warning(f"Update for memory {memory_id} returning False due to vector index update failure (queued for retry).")
                    return False

                logger.info(f"Updated memory {memory_id} with {len(updates)} fields")
                return True
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error updating memory {memory_id}: {str(e)}", exc_info=True)
            return False

    def _filter_by_metadata(self, candidates: List[Dict], metadata_filter: Dict) -> List[Dict]:
        """
        Filter candidates based on metadata key-value pairs.

        Args:
            candidates: List of candidate memory dictionaries to filter
            metadata_filter: Dictionary of key-value pairs that must be present in memory metadata

        Returns:
            Filtered list of candidates that match all metadata criteria
        """
        if not metadata_filter:
            return candidates

        logger.debug(f"[_filter_by_metadata] Filtering {len(candidates)} candidates with filter: {metadata_filter}")
        filtered_results = []

        for candidate in candidates:
            metadata = candidate.get("metadata", {})
            # Skip if candidate has no metadata
            if not metadata:
                logger.debug(f"Skipping candidate {candidate.get('id')} - no metadata")
                continue

            # Check each filter criterion
            matches_all = True
            for key, value in metadata_filter.items():
                # Support for nested paths with dots (e.g., 'details.source')
                current_val = metadata
                try:
                    for part in key.split('.'):
                        current_val = current_val[part]
                except (KeyError, TypeError):
                    matches_all = False
                    break # Path doesn't exist or not traversable

                # Check the final value
                if current_val != value:
                    matches_all = False
                    break

            if matches_all:
                filtered_results.append(candidate)
                logger.debug(f"Candidate {candidate.get('id')} matched all metadata criteria")
            else:
                logger.debug(f"Candidate {candidate.get('id')} failed metadata criteria {metadata_filter}") # Log filter

        logger.debug(f"[_filter_by_metadata] Found {len(filtered_results)} candidates matching metadata criteria")
        return filtered_results

    async def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embeddings using a consistent method for all text processing."""
        # Use SentenceTransformer directly without importing server.py
        try:
            from sentence_transformers import SentenceTransformer
            # Use the same model name as server.py
            import os
            model_name = os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
            # Consider caching the model instance if performance is critical
            # Check if model is already cached in the class instance
            if not hasattr(self, '_embedding_model') or self._embedding_model_name != model_name:
                 logger.info(f"Loading embedding model: {model_name}")
                 self._embedding_model = SentenceTransformer(model_name)
                 self._embedding_model_name = model_name
            model = self._embedding_model


            # Run encode in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            embedding_list = await loop.run_in_executor(None, lambda: model.encode([text], convert_to_tensor=False))
            if embedding_list is None or len(embedding_list) == 0:
                raise ValueError("Embedding model returned empty result")

            embedding = embedding_list[0]
            return self.geometry_manager._normalize(np.array(embedding, dtype=np.float32))
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error generating embedding: {str(e)}", exc_info=True)

            # Fallback to a deterministic embedding based on text hash
            import hashlib

            # Create a deterministic embedding based on the hash of the text
            text_bytes = text.encode('utf-8')
            hash_obj = hashlib.md5(text_bytes)
            hash_digest = hash_obj.digest()

            # Convert the 16-byte digest to a list of floats
            byte_values = list(hash_digest) * (self.config['embedding_dim'] // 16 + 1)
            embedding = np.array([float(byte) / 255.0 for byte in byte_values[:self.config['embedding_dim']]], dtype=np.float32)

            logger.warning("SynthiansMemoryCore", "Using deterministic hash-based embedding generation")
            return self.geometry_manager._normalize(embedding)

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory core.

        Returns detailed stats about memories, assemblies, vector index health,
        and most importantly for Phase 5.8, stats about assembly synchronization
        status and pending vector updates.

        Returns:
            Dict[str, Any]: A dictionary of stats and metrics
        """
        try:
            # Basic counts from memory
            async with self._lock:
                memory_count_cache = len(self._memories)
                assembly_count_cache = len(self.assemblies)
                dirty_memory_count = len(self._dirty_memories)

            # Persistence counts (can be different from cache)
            memory_count_persist = 0
            assembly_count_persist = 0 # Initialize
            if hasattr(self, 'persistence') and hasattr(self.persistence, 'memory_index'):
                memory_count_persist = len(self.persistence.memory_index)
                # Attempt to count assemblies from persistence index if structured similarly
                assembly_count_persist = sum(1 for path_data in self.persistence.memory_index.values() if path_data.get("type") == "assembly")


            # Pending updates count
            pending_updates_count = self._pending_vector_updates.qsize() if hasattr(self, '_pending_vector_updates') else 0

            # Check index integrity
            index_status = {}
            index_consistent = False
            if self.vector_index:
                # FIX: Await the async call
                try:
                     is_consistent, diagnostics = await self.vector_index.check_index_integrity()
                     index_status = diagnostics # Use the diagnostics dict directly
                     index_consistent = is_consistent
                except Exception as check_err:
                     logger.error(f"Error calling vector_index.check_index_integrity: {check_err}", exc_info=True)
                     index_status = {"error": str(check_err)}


            # Get assembly synchronization status
            sync_status = {"synchronized": 0, "pending": 0, "never_synced": 0}
            assembly_sync_details = []
            if hasattr(self, 'assemblies'):
                 async with self._lock: # Lock to access assemblies and dirty set
                    for assembly_id, assembly in self.assemblies.items():
                        sync_state = "never_synced"
                        last_sync_ts = None
                        if hasattr(assembly, 'vector_index_updated_at') and assembly.vector_index_updated_at:
                            sync_state = "synchronized"
                            last_sync_ts = assembly.vector_index_updated_at.isoformat()
                        # Check if pending in queue (more accurate than just dirty)
                        # Note: Checking queue requires iterating, which is slow. Use dirty set approximation.
                        elif assembly_id in self._dirty_memories:
                             sync_state = "pending"

                        sync_status[sync_state] += 1

                        # Add to detailed list (limit to 20 for performance)
                        if len(assembly_sync_details) < 20:
                            assembly_sync_details.append({
                                "assembly_id": assembly_id,
                                "name": getattr(assembly, 'name', 'Unknown'),
                                "memory_count": len(getattr(assembly, 'memories', [])),
                                "status": sync_state,
                                "last_synced": last_sync_ts
                            })

            # Calculate uptime
            uptime_seconds = 0
            if hasattr(self, 'start_time'):
                uptime_seconds = int(time.time() - self.start_time)

            # Assemble response
            return {
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "core_stats": {
                    "memory_count_cache": memory_count_cache,
                    "memory_count_persistence": memory_count_persist,
                    "assembly_count_cache": assembly_count_cache,
                    "assembly_count_persistence": assembly_count_persist,
                    "dirty_items": dirty_memory_count,
                    "pending_vector_updates": pending_updates_count,
                    "initialized": self._initialized,
                    "vector_index_state": self.vector_index.state if self.vector_index else "N/A",
                    "uptime_seconds": uptime_seconds
                },
                "vector_index_stats": {
                    "faiss_count": index_status.get("faiss_count", 0),
                    "mapping_count": index_status.get("mapping_count", 0),
                    "persistence_count": index_status.get("persistence_count", 0), # If provided by check_index_integrity
                    "embedding_dimension": self.config.get("embedding_dim", 0),
                    "is_consistent": index_consistent,
                    "details": index_status # Include all diagnostics
                },
                "assembly_stats": {
                    "total_count": assembly_count_cache, # Use cache count
                    "sync_status": sync_status,
                    "sync_details": assembly_sync_details,
                },
                "performance_stats": {
                     # Add performance stats if tracked (e.g., avg latencies)
                },
                "feature_flags": {
                    "explainability_enabled": self.config.get("ENABLE_EXPLAINABILITY", False),
                    "assembly_pruning_enabled": self.config.get("enable_assembly_pruning", False), # Correct key
                    "assembly_merging_enabled": self.config.get("enable_assembly_merging", False), # Correct key
                }
            }
        except Exception as e:
            logger.error(f"Error generating stats: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error generating memory core statistics: {str(e)}"
            }

    async def check_core_index_integrity(self) -> Dict[str, Any]:
        """Check the integrity of the vector index and return diagnostic information.

        This method checks if the FAISS index and ID-to-index mapping are consistent.

        Returns:
            Dict with diagnostic information about the index integrity
        """
        if not self._initialized: await self.initialize()

        if not self.vector_index:
             return {"success": False, "error": "Vector index not initialized"}

        try:
            # FIX: Await the async call
            is_consistent, diagnostics = await self.vector_index.check_index_integrity()

            return {
                "success": True,
                "is_consistent": is_consistent,
                "diagnostics": diagnostics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
             logger.error(f"Error in check_core_index_integrity: {e}", exc_info=True)
             return {"success": False, "error": str(e)}

    async def repair_index(self, repair_type: str = "auto") -> Dict[str, Any]:
        """Attempt to repair integrity issues with the vector index.

        Args:
            repair_type: The type of repair to perform.
                - "auto": Automatically determine the best repair strategy
                - "recreate_mapping": Recreate the ID-to-index mapping from scratch
                - "rebuild_from_persistence": Rebuild index from persistence files (preferred)

        Returns:
            Dict with repair status and diagnostics
        """
        if not self._initialized: await self.initialize()

        if not self.vector_index:
             return {"success": False, "error": "Vector index not initialized"}

        logger.info("SynthiansMemoryCore", f"Starting index repair of type: {repair_type}")
        repair_stats = {}
        try:
            # Call the vector index's repair method
            # This method handles locking internally
            repair_stats = await self.vector_index.repair_index_async(
                 persistence=self.persistence,
                 geometry_manager=self.geometry_manager,
                 repair_mode=repair_type
            )

            if repair_stats.get("success", False):
                logger.info("SynthiansMemoryCore", f"Index repair completed successfully. Mode: {repair_stats.get('mode_used', 'N/A')}. Consistency: {repair_stats.get('consistency_after', 'N/A')}")
            else:
                logger.error("SynthiansMemoryCore", f"Index repair failed. Mode: {repair_stats.get('mode_used', 'N/A')}. Error: {repair_stats.get('error', 'Unknown')}")

            return repair_stats

        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Unexpected error during repair_index call: {e}", exc_info=True)
            return {"success": False, "error": f"Unexpected exception: {str(e)}", "details": repair_stats}

    # --- DETECT/REPAIR METHODS ---

    async def detect_and_repair_index_drift(self, auto_repair: bool = False) -> Dict[str, Any]:
        """
        Detect drift between vector index and memory persistence and optionally repair it.

        Args:
            auto_repair: If True, automatically repair detected inconsistencies

        Returns:
            Dictionary with drift detection and repair statistics
        """
        if not self._initialized:
            logger.error("SynthiansMemoryCore", "Cannot detect/repair drift: not initialized")
            return {"error": "Core not initialized", "success": False}

        logger.info("SynthiansMemoryCore", f"Checking for vector index drift (Auto-Repair: {auto_repair})...")
        result = {"success": False}

        try:
            if not self.vector_index:
                 raise ValueError("Vector index is not initialized.")

            # Get integrity status using the correct method
            # FIX: Await async call
            is_consistent, diagnostics = await self.vector_index.check_index_integrity(persistence=self.persistence)

            # --- Add check for empty index vs non-empty persistence ---
            persistence_count = diagnostics.get("persistence_count", -1)
            faiss_count = diagnostics.get("faiss_count", -1)
            mapping_count = diagnostics.get("mapping_count", -1)

            if is_consistent and persistence_count > 0 and faiss_count == 0:
                logger.warning("SynthiansMemoryCore", "Inconsistency detected: Persistence has items but vector index is empty.")
                is_consistent = False
                diagnostics["is_consistent"] = False
                diagnostics["issue"] = "persistence_mismatch_empty_faiss"
            elif is_consistent and faiss_count != mapping_count:
                 logger.warning(f"SynthiansMemoryCore: Inconsistency detected: FAISS count ({faiss_count}) != Mapping count ({mapping_count}).")
                 is_consistent = False
                 diagnostics["is_consistent"] = False
                 diagnostics["issue"] = "faiss_mapping_count_mismatch"
            # --- End check ---

            result["is_consistent"] = is_consistent
            result["diagnostics"] = diagnostics

            if is_consistent:
                logger.info("SynthiansMemoryCore", "No vector index drift detected")
                result["success"] = True
                return result

            # We detected drift
            issue = diagnostics.get("issue", "unknown")
            logger.warning(
                "SynthiansMemoryCore",
                f"Vector index inconsistency detected. Issue: {issue}. Counts: [FAISS={faiss_count}, Mapping={mapping_count}, Persistence={persistence_count}]",
                {"issue": issue, "faiss_count": faiss_count, "mapping_count": mapping_count, "persistence_count": persistence_count}
            )

            # Repair if needed and requested
            if auto_repair:
                logger.info("SynthiansMemoryCore", "Initiating auto-repair for vector index")
                try:
                    # Call the vector index's repair method (handles locking)
                    repair_stats = await self.vector_index.repair_index_async(
                        persistence=self.persistence,
                        geometry_manager=self.geometry_manager,
                        repair_mode="auto" # Use auto mode
                    )
                    result["repair_stats"] = repair_stats
                    result["success"] = repair_stats.get("success", False)
                    # Update consistency status after repair attempt
                    result["is_consistent"] = repair_stats.get("consistency_after", False)

                    if result["success"]:
                        logger.info("SynthiansMemoryCore", f"Vector index auto-repair successful (Mode: {repair_stats.get('mode_used', 'N/A')}). Consistency achieved: {result['is_consistent']}")
                    else:
                        logger.error(
                            "SynthiansMemoryCore",
                            "Vector index auto-repair failed.",
                            {"reason": repair_stats.get("error", "Unknown error"), "mode_used": repair_stats.get('mode_used', 'N/A')}
                        )
                except Exception as repair_e:
                     logger.error("SynthiansMemoryCore", f"Unexpected error during repair process: {str(repair_e)}", exc_info=True)
                     result["repair_stats"] = {"success": False, "error": f"Unexpected exception: {str(repair_e)}"}
                     result["success"] = False
            else:
                logger.warning("SynthiansMemoryCore", "Vector index drift detected but auto-repair not enabled")
                result["needs_repair"] = True

            return result

        except Exception as e:
            logger.error(
                "SynthiansMemoryCore",
                f"Error during drift detection/repair: {e}",
                exc_info=True
            )
            result["error"] = str(e)
            return result

    async def _auto_repair_drift_loop(self):
        """
        Periodically check for and repair vector index drift.

        This background task runs at configurable intervals and ensures
        that the FAISS index and ID mappings remain synchronized, preventing
        silent failures in retrieval and assembly operations.
        """
        logger.info("SynthiansMemoryCore", "Started auto-repair drift background loop")
        index_check_interval = self.config.get('index_check_interval', 3600)  # Default: hourly checks

        try:
            while not self._shutdown_signal.is_set():
                # Wait for the configured interval OR the shutdown signal
                try:
                    await asyncio.wait_for(
                        self._shutdown_signal.wait(),
                        timeout=index_check_interval
                    )
                    # If wait() finished without timeout, it means signal was set
                    logger.info("SynthiansMemoryCore","Auto-repair loop: Shutdown signal received during wait.")
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue with drift check
                    pass
                except asyncio.CancelledError:
                    logger.info("SynthiansMemoryCore","Auto-repair loop cancelled during wait.")
                    break

                # Check and repair drift
                if not self._initialized:
                    logger.debug("SynthiansMemoryCore","Auto-repair loop: Core not initialized, skipping check.")
                    continue

                try:
                    logger.info("SynthiansMemoryCore", "Running scheduled vector index drift check")
                    # Call the main drift detection method with auto_repair=True
                    result = await self.detect_and_repair_index_drift(auto_repair=True)

                    # Logging is handled within detect_and_repair_index_drift

                except asyncio.CancelledError:
                    logger.info("SynthiansMemoryCore","Auto-repair drift check cancelled.")
                    break # Exit loop if cancelled during check/repair
                except Exception as e:
                    logger.error(
                        "SynthiansMemoryCore",
                        "Error in scheduled vector index drift check",
                        {"error": str(e)}, exc_info=True
                    )
                    # Add a small delay after an error to prevent tight loops
                    await asyncio.sleep(10)

        except asyncio.CancelledError:
             logger.info("SynthiansMemoryCore","Auto-repair drift background loop received cancel signal.")
        except Exception as e:
            logger.error(
                "SynthiansMemoryCore",
                "Auto-repair drift background loop terminated with error",
                {"error": str(e)}, exc_info=True
            )
        finally:
             logger.info("SynthiansMemoryCore","Auto-repair drift background loop stopped.")


    async def _persist_dirty_items(self):
        """Persist any dirty items (memories, assemblies) to disk."""
        if not self._initialized:
            logger.warning("Cannot persist items: Memory Core not initialized")
            return

        # Get a snapshot of dirty items to process
        async with self._lock: # Lock needed to safely copy the set
            dirty_items_snapshot = set(self._dirty_memories)

        total_dirty = len(dirty_items_snapshot)

        if not dirty_items_snapshot:
            logger.debug(f"No dirty items to persist")
            return

        logger.info(f"Persisting {total_dirty} dirty items")

        # Process in batches
        batch_size = self.config.get('persistence_batch_size', 100)
        dirty_list = list(dirty_items_snapshot)
        processed = 0
        failed = 0
        items_successfully_persisted = set()

        for i in range(0, len(dirty_list), batch_size):
             # Check for shutdown signal periodically during long loops
            if self._shutdown_signal.is_set():
                logger.warning("Shutdown signal received during persistence, aborting.")
                break

            batch_ids = dirty_list[i:i+batch_size]
            batch_tasks = []

            # Create tasks for saving items in the batch
            async with self._lock: # Lock needed to access _memories and assemblies
                for item_id in batch_ids:
                    item_to_save = None
                    save_coro = None
                    is_assembly = False
                    # Determine if memory or assembly
                    # Use self.assemblies which should be more reliable under lock
                    if item_id in self.assemblies:
                        item_to_save = self.assemblies[item_id]
                        save_coro = self.persistence.save_assembly(item_to_save)
                        is_assembly = True
                    elif item_id in self._memories:
                        item_to_save = self._memories[item_id]
                        save_coro = self.persistence.save_memory(item_to_save)
                    else:
                        # Check if ID looks like an assembly ID just in case
                        if item_id.startswith("asm:"):
                             actual_asm_id = item_id[4:]
                             if actual_asm_id in self.assemblies:
                                 item_to_save = self.assemblies[actual_asm_id]
                                 save_coro = self.persistence.save_assembly(item_to_save)
                                 is_assembly = True

                    if save_coro:
                        # Wrap save operation with ID tracking
                        async def save_wrapper(id_to_save, coro):
                            try:
                                success = await coro
                                return id_to_save, success
                            except Exception as e:
                                logger.error(f"Error persisting item {id_to_save}: {e}", exc_info=True)
                                return id_to_save, False
                        batch_tasks.append(save_wrapper(item_id, save_coro))
                    elif item_id in dirty_items_snapshot: # Check original snapshot
                         # Item was marked dirty but no longer exists in memory cache
                         logger.warning(f"Item {item_id} marked dirty but not found in memory cache, removing from dirty set.")
                         items_successfully_persisted.add(item_id) # Treat as 'processed'
                         processed += 1


            # Run batch save tasks concurrently
            if batch_tasks:
                results = await asyncio.gather(*batch_tasks)
                for item_id, success in results:
                    if success:
                        items_successfully_persisted.add(item_id)
                        processed += 1
                    else:
                        logger.error(f"Failed to persist item {item_id}")
                        failed += 1

        # Update the main dirty set *after* processing all batches
        async with self._lock:
            self._dirty_memories.difference_update(items_successfully_persisted)
            remaining_dirty = len(self._dirty_memories)

        logger.info(f"Persistence complete: {processed} succeeded, {failed} failed. {remaining_dirty} items remain dirty.")


    async def _load_activation_stats(self):
        """Load assembly activation statistics from disk."""
        stats_file_path = None # Initialize
        try:
            stats_dir = os.path.join(self.config['storage_path'], "stats")
            stats_file_path = os.path.join(stats_dir, 'assembly_activation_stats.json')

            # Create stats directory if it doesn't exist
            os.makedirs(stats_dir, exist_ok=True)

            if os.path.exists(stats_file_path):
                if aiofiles: # Check if aiofiles is available
                    async with aiofiles.open(stats_file_path, "r") as f:
                        content = await f.read()
                        self._assembly_activation_counts = json.loads(content)
                else: # Fallback to synchronous read
                    # Run sync I/O in executor to avoid blocking
                    loop = asyncio.get_running_loop()
                    def read_sync():
                         with open(stats_file_path, "r") as f:
                             return json.load(f)
                    self._assembly_activation_counts = await loop.run_in_executor(None, read_sync)


                logger.info("SynthiansMemoryCore", "Loaded assembly activation statistics",
                            {"count": len(self._assembly_activation_counts)})
            else:
                self._assembly_activation_counts = {}
                logger.info("SynthiansMemoryCore", "No existing activation statistics found, starting fresh")
        except FileNotFoundError:
             self._assembly_activation_counts = {}
             logger.info("SynthiansMemoryCore", f"Activation statistics file not found at {stats_file_path}, starting fresh")
        except json.JSONDecodeError as json_err:
            logger.error("SynthiansMemoryCore", f"Error decoding assembly activation statistics JSON from {stats_file_path}: {json_err}",
                        exc_info=True)
            self._assembly_activation_counts = {} # Reset on decode error
        except Exception as e:
            logger.error("SynthiansMemoryCore", "Error loading assembly activation statistics",
                        {"error": str(e)}, exc_info=True)
            self._assembly_activation_counts = {} # Reset on other errors


    async def _persist_activation_stats(self, force: bool = False):
        """Persist assembly activation statistics to disk."""
        stats_file_path = None # Initialize
        try:
            current_time = time.time()
            persist_interval = self.config.get('assembly_metrics_persist_interval', 600.0)

            # Only persist if forced or interval has elapsed
            if not force and (current_time - self._last_activation_persist_time < persist_interval):
                return

            stats_dir = os.path.join(self.config['storage_path'], "stats")
            stats_file_path = os.path.join(stats_dir, 'assembly_activation_stats.json')

            # Create stats directory if it doesn't exist
            os.makedirs(stats_dir, exist_ok=True)

            # Write stats to file using atomic write
            temp_file_path = stats_file_path + ".tmp." + str(uuid.uuid4())[:8] # Unique temp file
            stats_json = json.dumps(self._assembly_activation_counts, indent=2)

            # Define sync write helper
            def write_sync():
                 with open(temp_file_path, "w") as f:
                     f.write(stats_json)
                 os.replace(temp_file_path, stats_file_path) # Atomic replace

            if aiofiles: # Check if aiofiles is available
                async with aiofiles.open(temp_file_path, "w") as f:
                    await f.write(stats_json)
                # Use async os replace if available, otherwise executor
                if hasattr(aiofiles.os, "replace"):
                     await aiofiles.os.replace(temp_file_path, stats_file_path)
                else:
                     loop = asyncio.get_running_loop()
                     await loop.run_in_executor(None, lambda: os.replace(temp_file_path, stats_file_path))
            else: # Fallback to sync write in executor
                 loop = asyncio.get_running_loop()
                 await loop.run_in_executor(None, write_sync)


            self._last_activation_persist_time = current_time
            logger.info("SynthiansMemoryCore", "Persisted assembly activation statistics",
                        {"count": len(self._assembly_activation_counts)})
        except Exception as e:
            logger.error("SynthiansMemoryCore", "Error persisting assembly activation statistics",
                        {"error": str(e), "path": stats_file_path}, exc_info=True)
            # Attempt to remove potentially corrupted temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass


    async def _track_assembly_activation(self, assembly_id: str):
        """Track assembly activation for diagnostics."""
        if not assembly_id:
            return

        # Increment activation count (Needs lock if accessed concurrently, but usually called from locked sections)
        # Assuming for now it's called safely or occasional race condition is acceptable for stats
        if assembly_id in self._assembly_activation_counts:
            self._assembly_activation_counts[assembly_id] += 1
        else:
            self._assembly_activation_counts[assembly_id] = 1

        # Check if we should persist activation stats (non-blocking check)
        await self._persist_activation_stats() # persist checks interval internally


    async def _index_assembly_embedding(self, assembly: MemoryAssembly) -> bool:
        """Index or update the assembly embedding in the vector index.

        This is a critical method for assembly index integrity. It ensures:
        1. Assembly vectors are correctly indexed in FAISS
        2. Failed operations are queued for retry
        3. Assembly timestamps are updated only on successful indexing
        4. Clear logging for diagnostics

        Args:
            assembly: The assembly to index

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.vector_index or self.vector_index.state != "READY":
                logger.error(f"Vector index not ready when trying to index assembly {assembly.assembly_id}")
                # Queue for retry
                await self._queue_assembly_for_retry(assembly, 'add')
                return False

            # Use composite_embedding for indexing assemblies
            if not hasattr(assembly, 'composite_embedding') or assembly.composite_embedding is None:
                logger.error(f"Assembly {assembly.assembly_id} has no composite embedding to index")
                return False

            # Prepare for vector index
            asm_id_for_index = f"asm:{assembly.assembly_id}"
            # Validate the composite embedding
            validated = self.geometry_manager._validate_vector(assembly.composite_embedding, f"Composite Emb {asm_id_for_index}")

            if validated is None:
                logger.error(f"Invalid composite embedding for assembly {assembly.assembly_id}, cannot index")
                return False

            if self.vector_index is None: # Double check after potential await
                logger.error(f"CRITICAL: Vector index became None when trying to index assembly {asm_id_for_index}")
                await self._queue_assembly_for_retry(assembly, 'add')
                return False

            logger.debug(f"Attempting index operation for assembly {asm_id_for_index}. Mapping size: {len(self.vector_index.id_to_index)}")

            success = False # Initialize success before try block
            operation = 'unknown' # Initialize operation before try block
            try:
                if asm_id_for_index in self.vector_index.id_to_index:
                    logger.debug(f"Calling update_entry_async for existing assembly {asm_id_for_index}")
                    success = await self.vector_index.update_entry_async(asm_id_for_index, validated)
                    operation = 'update'
                else:
                    logger.debug(f"Calling add_async for new assembly {asm_id_for_index}")
                    success = await self.vector_index.add_async(asm_id_for_index, validated)
                    operation = 'add'
            except Exception as index_update_err:
                logger.error(f"EXCEPTION during vector index op for assembly {asm_id_for_index}: {index_update_err}", exc_info=True)
                success = False # Ensure success is False on exception
                # Determine operation type again if possible, default to 'add' for queueing
                operation = "update" if asm_id_for_index in self.vector_index.id_to_index else "add"


            if success:
                # Set timestamp ONLY on success
                async with self._lock: # Lock to modify assembly object
                     if assembly.assembly_id in self.assemblies: # Check if still exists
                        self.assemblies[assembly.assembly_id].vector_index_updated_at = datetime.now(timezone.utc)
                        self._dirty_memories.add(assembly.assembly_id) # Mark assembly dirty for persistence
                        logger.debug(f"Updated timestamp for successfully indexed assembly {assembly.assembly_id}")
                     else:
                         logger.warning(f"Assembly {assembly.assembly_id} disappeared before timestamp update after index success.")
            else:
                logger.error(f"FAILED vector index operation '{operation}' for assembly {asm_id_for_index}. Queuing for retry.")
                # Queue the failed operation using the determined operation type
                await self._queue_assembly_for_retry(assembly, operation)


            return success
        except Exception as e:
            logger.error(f"Exception during assembly indexing for {assembly.assembly_id}: {e}", exc_info=True)
            # Queue for retry on unexpected exception
            await self._queue_assembly_for_retry(assembly, 'add') # Default to add on exception
            return False

    async def _queue_assembly_for_retry(self, assembly: MemoryAssembly, operation: str) -> None:
        """Queue a failed assembly vector operation for retry.

        Args:
            assembly: The assembly to queue
            operation: The operation type ('add', 'update', 'remove')
        """
        if not hasattr(self, '_pending_vector_updates') or self._pending_vector_updates is None:
            logger.error(f"Cannot queue assembly {assembly.assembly_id} for retry: queue not initialized")
            return

        try:
            # Prepare the update item
            asm_id_for_index = f"asm:{assembly.assembly_id}"
            embedding_list = None

            # Convert numpy embedding to list for serialization if operation needs it
            # Use composite_embedding
            if operation in ['add', 'update'] and hasattr(assembly, 'composite_embedding') and assembly.composite_embedding is not None:
                embedding_list = assembly.composite_embedding.tolist()
            elif operation in ['add', 'update']:
                 logger.warning(f"Cannot queue {operation} for assembly {assembly.assembly_id}: composite embedding is missing.")
                 return # Don't queue if embedding isn't available for add/update


            # Create update item
            update_item = {
                'operation': operation,
                'id': asm_id_for_index,
                'embedding': embedding_list, # Will be None for 'remove' or if missing
                'is_assembly': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'retry_count': 0 # Initialize retry count
            }

            # Add to queue
            await self._pending_vector_updates.put(update_item)
            logger.info(f"Queued assembly {assembly.assembly_id} for {operation} retry")

        except Exception as e:
            logger.error(f"Failed to queue assembly {assembly.assembly_id} for retry: {e}", exc_info=True)


    async def _merge_similar_assemblies(self):
        """Merge assemblies that are highly similar."""
        if not self.config.get('enable_assembly_merging', True):
            logger.debug("[MERGE] Assembly merging disabled by config.")
            return

        logger.info("[MERGE] Starting assembly merge check...")
        merge_threshold = self.config.get('assembly_merge_threshold', 0.70)
        max_merges = self.config.get('assembly_max_merges_per_run', 10)
        merges_done = 0

        logger.info(f"[MERGE] Using assembly_merge_threshold={merge_threshold:.4f}")

        async with self._lock: # Need lock to iterate and modify self.assemblies
            assembly_ids = list(self.assemblies.keys())
            checked_pairs = set()
            merged_this_run = set() # Track IDs merged in this run to avoid merging them again

            for i in range(len(assembly_ids)):
                if merges_done >= max_merges:
                    logger.info(f"[MERGE] Reached max merges ({max_merges}) for this run.")
                    break
                asm_id_a = assembly_ids[i]
                # Ensure asm_id_a exists and wasn't merged in this run
                if asm_id_a not in self.assemblies or asm_id_a in merged_this_run: continue

                asm_a = self.assemblies[asm_id_a]

                for j in range(i + 1, len(assembly_ids)):
                    if merges_done >= max_merges: break
                    asm_id_b = assembly_ids[j]
                     # Ensure asm_id_b exists and wasn't merged in this run
                    if asm_id_b not in self.assemblies or asm_id_b in merged_this_run: continue

                    # Avoid re-checking pairs
                    pair = tuple(sorted((asm_id_a, asm_id_b)))
                    if pair in checked_pairs: continue
                    checked_pairs.add(pair)

                    asm_b = self.assemblies[asm_id_b]

                    composite_a = asm_a.composite_embedding
                    composite_b = asm_b.composite_embedding

                    if composite_a is not None and composite_b is not None:
                        try:
                            aligned_a, aligned_b = self.geometry_manager._align_vectors(composite_a, composite_b)
                            if aligned_a is not None and aligned_b is not None:
                                similarity = self.geometry_manager.calculate_similarity(aligned_a, aligned_b)
                                # logger.debug(f"[MERGE_DEBUG] Comparing {asm_id_a} and {asm_id_b}: Similarity={similarity:.4f}")

                                if similarity >= merge_threshold:
                                    logger.info(f"[MERGE_TRIGGER] Threshold met ({similarity:.4f} >= {merge_threshold:.4f}) for merging {asm_id_a} and {asm_id_b}")
                                    # Determine which assembly to keep (e.g., the one with more memories or older one?)
                                    # Simple strategy: Keep B, merge A into B.
                                    keep_id, remove_id = asm_id_b, asm_id_a

                                    # --- Log Merge Event ---
                                    merge_event_id_val = None
                                    if hasattr(self, 'merge_tracker') and self.merge_tracker:
                                         try:
                                             merge_event_id_val = await self.merge_tracker.log_merge_event(
                                                 assembly_id_a=remove_id, # Source
                                                 assembly_id_b=keep_id, # Target
                                                 similarity_score=similarity
                                             )
                                         except Exception as log_err:
                                              logger.error(f"[MERGE] Failed to log merge event: {log_err}")

                                    # --- Execute Merge ---
                                    # Pass IDs directly to _execute_merge
                                    merge_success = await self._execute_merge(remove_id, keep_id, merge_event_id=merge_event_id_val)
                                    if merge_success:
                                        merges_done += 1
                                        merged_this_run.add(remove_id) # Mark the removed ID as merged
                                        logger.info(f"[MERGE] Merge successful ({remove_id} -> {keep_id}). Merges done: {merges_done}")
                                        # Break inner loop as asm_a (remove_id) no longer exists
                                        # Need to restart scan or handle modified list carefully.
                                        # For simplicity, breaking inner and continuing outer might miss some merges but is safer.
                                        break
                                    else:
                                        logger.error(f"[MERGE] Merge execution failed between {remove_id} and {keep_id}")
                                        # Log failed merge execution status if tracker ID available
                                        if merge_event_id_val and hasattr(self, 'merge_tracker') and self.merge_tracker:
                                            await self.merge_tracker.log_cleanup_status_event(merge_event_id_val, "failed", "Merge execution failed")
                            else:
                                 logger.debug(f"[MERGE_DEBUG] Alignment failed for {asm_id_a}/{asm_id_b}")
                        except Exception as e_sim:
                             logger.error(f"[MERGE_DEBUG] Error calculating similarity between {asm_id_a} and {asm_id_b}: {e_sim}", exc_info=True)
                    else:
                         logger.warning(f"[MERGE_DEBUG] Skipping comparison: Composite embedding missing for {asm_id_a} or {asm_id_b}")
                # End inner loop (j)
            # End outer loop (i)
        logger.info(f"[MERGE] Merge check completed. Merges performed in this run: {merges_done}")


    async def _execute_merge(self, asm_id_a: str, asm_id_b: str, merge_event_id: Optional[str] = None) -> bool:
        """Performs the actual merge operation (internal helper). Assumes lock is held.

        Args:
            asm_id_a: ID of the source assembly to merge from (will be removed)
            asm_id_b: Target assembly ID (may be a new assembly or one of the sources)
            merge_event_id: Optional ID from MergeTracker for tracking cleanup status

        Returns:
            bool: Whether the merge was successful
        """
        logger.warning(f"[MERGE_EXECUTE] Attempting to merge {asm_id_a} into {asm_id_b}")
        try:
            asm_a = self.assemblies.get(asm_id_a)
            asm_b = self.assemblies.get(asm_id_b)

            if not asm_a or not asm_b:
                logger.error(f"[MERGE_EXECUTE] One or both assemblies not found in memory ({asm_id_a}, {asm_id_b}). Aborting merge.")
                return False

            # Merge members (prefer keeping the one with more members or newer timestamp?)
            # Simple merge: Add all members from A into B
            members_to_add = list(asm_a.memories) # Get list before iterating/modifying
            logger.info(f"[MERGE_EXECUTE] Adding {len(members_to_add)} members from {asm_id_a} to {asm_id_b}")
            add_failures = 0
            for mem_id in members_to_add:
                # Load memory if not in cache (use async helper which handles lock internally)
                memory = await self.get_memory_by_id_async(mem_id)
                if memory and memory.embedding is not None:
                     # Validate embedding before adding
                     validated_emb = self.geometry_manager._validate_vector(memory.embedding, f"Merge Member {mem_id}")
                     if validated_emb is not None:
                         # add_memory should recalculate composite embedding
                         if not asm_b.add_memory(memory, validated_emb):
                              add_failures += 1
                              logger.warning(f"[MERGE_EXECUTE] Failed to add memory {mem_id} during merge.")
                     else:
                         add_failures += 1
                         logger.warning(f"[MERGE_EXECUTE] Invalid embedding for memory {mem_id}, cannot add during merge.")
                else:
                    add_failures += 1
                    logger.warning(f"[MERGE_EXECUTE] Memory {mem_id} not found or has no embedding, cannot add during merge.")
            if add_failures > 0:
                 logger.warning(f"[MERGE_EXECUTE] {add_failures} members failed to add during merge.")

            # Update metadata (simple concatenation for now)
            asm_b.name = f"{asm_b.name} (merged {asm_id_a[-8:]})"
            asm_b.keywords.update(asm_a.keywords)
            asm_b.tags.update(asm_a.tags)
            asm_b.merged_from.append(asm_id_a) # Record lineage
            asm_b.merged_from.extend(asm_a.merged_from) # Preserve older lineage
            asm_b.merged_from = list(set(asm_b.merged_from)) # Remove duplicates
            # Keep the newer last_activation time
            asm_b.last_activation = max(asm_a.last_activation, asm_b.last_activation) if asm_a.last_activation and asm_b.last_activation else (asm_a.last_activation or asm_b.last_activation)
            # Reset sync timestamp for the merged assembly as its embedding changed
            asm_b.vector_index_updated_at = None
            logger.info(f"[MERGE_EXECUTE] Merged metadata. New member count for {asm_id_b}: {len(asm_b.memories)}")

            # Mark merged assembly B as dirty (needs saving and re-indexing)
            self._dirty_memories.add(asm_id_b)

            # Remove assembly A from core structures (under lock)
            del self.assemblies[asm_id_a]
            self._dirty_memories.discard(asm_id_a) # Remove from dirty set if it was there
            # Update memory_to_assemblies map
            for mem_id in members_to_add: # Use the original list
                 if mem_id in self.memory_to_assemblies:
                     self.memory_to_assemblies[mem_id].discard(asm_id_a)
                     self.memory_to_assemblies[mem_id].add(asm_id_b) # Ensure mapping points to B
            logger.info(f"[MERGE_EXECUTE] Removed assembly {asm_id_a} from internal structures.")

            # Schedule cleanup and indexing task (runs outside the lock)
            asyncio.create_task(self.cleanup_and_index_after_merge(asm_id_a, asm_id_b, merge_event_id))

            logger.info(f"[MERGE_EXECUTE] Successfully merged {asm_id_a} into {asm_id_b}. Scheduled cleanup task.")
            return True
        except Exception as merge_err:
            logger.error(f"[MERGE_EXECUTE] Error during merge execution: {merge_err}", exc_info=True)
            return False

    async def cleanup_and_index_after_merge(self, asm_id_a: str, asm_id_b: str, merge_event_id: Optional[str] = None):
        """Helper to track cleanup status and log it via MergeTracker"""
        logger.info(f"[MERGE_CLEANUP] Task started for merged assembly {asm_id_b} (removing old assembly {asm_id_a})")
        # Variable to track overall success for cleanup status
        cleanup_success = True
        cleanup_error = None
        assembly_b_obj = None # To store the assembly object

        try:
            # Retrieve assembly B object (needed for saving/indexing)
            # Use lock briefly to get object reference
            async with self._lock:
                assembly_b_obj = self.assemblies.get(asm_id_b)

            if assembly_b_obj is None:
                raise ValueError(f"Merged assembly {asm_id_b} not found in memory for cleanup.")

            # Save the updated assembly B
            try:
                save_ok = await self.persistence.save_assembly(assembly_b_obj)
                if save_ok:
                     logger.info(f"[MERGE_CLEANUP] Saved merged assembly {asm_id_b}")
                     # Index the updated assembly B (use the object retrieved earlier)
                     await self._index_assembly_embedding(assembly_b_obj)
                else:
                     logger.error(f"[MERGE_CLEANUP] Failed to save merged assembly {asm_id_b}")
                     cleanup_success = False
                     cleanup_error = "Failed to save merged assembly"
            except Exception as save_err:
                logger.error(f"[MERGE_CLEANUP] Error saving/indexing merged assembly {asm_id_b}: {save_err}", exc_info=True)
                cleanup_success = False
                cleanup_error = f"Error saving/indexing: {str(save_err)}"

            # Delete the old assembly A from persistence and index
            try:
                logger.info(f"[MERGE_CLEANUP] Deleting old assembly {asm_id_a} from persistence...")
                await self.persistence.delete_assembly(asm_id_a)
                logger.info(f"[MERGE_CLEANUP] Successfully deleted old assembly {asm_id_a} from persistence")
            except Exception as del_err:
                logger.error(f"[MERGE_CLEANUP] Error deleting old assembly {asm_id_a} from persistence: {del_err}", exc_info=True)
                cleanup_success = False # Mark as failed but continue other cleanup steps
                if not cleanup_error: cleanup_error = f"Error deleting from persistence: {str(del_err)}"

            try:
                logger.info(f"[MERGE_CLEANUP] Removing old assembly {asm_id_a} from vector index...")
                # Add asm: prefix for index removal
                removed_from_index = await self.vector_index.remove_vector_async(f"asm:{asm_id_a}")
                if removed_from_index:
                    logger.info(f"[MERGE_CLEANUP] Successfully removed old assembly {asm_id_a} from vector index")
                else:
                    logger.warning(f"[MERGE_CLEANUP] Old assembly {asm_id_a} (ID: asm:{asm_id_a}) not found in vector index for removal (might be expected if already removed).")
            except Exception as idx_err:
                logger.error(f"[MERGE_CLEANUP] Error removing old assembly {asm_id_a} from vector index: {idx_err}", exc_info=True)
                cleanup_success = False
                if not cleanup_error: cleanup_error = f"Error removing from vector index: {str(idx_err)}"

        except Exception as e:
            logger.error(f"[MERGE_CLEANUP] Error during cleanup setup for {asm_id_b}: {e}", exc_info=True)
            cleanup_success = False
            cleanup_error = f"Error during cleanup: {str(e)}"

        finally:
            # Log the final cleanup status if we have a merge_event_id
            if merge_event_id and hasattr(self, 'merge_tracker') and self.merge_tracker:
                try:
                    await self.merge_tracker.log_cleanup_status_event(
                        merge_event_id=merge_event_id,
                        new_status="completed" if cleanup_success else "failed",
                        error=cleanup_error
                    )
                    logger.info(f"[MERGE_CLEANUP] Logged cleanup status: {'completed' if cleanup_success else 'failed'}")
                except Exception as log_err:
                    logger.error(f"[MERGE_CLEANUP] Failed to log cleanup status: {log_err}")

        logger.info(f"[MERGE_CLEANUP] Task completed for merged assembly {asm_id_b}")

    async def force_process_pending_updates(self) -> Dict[str, Any]:
        """Manually process all items currently in the pending vector update queue."""
        # Ensure the queue exists and is initialized
        if not hasattr(self, '_pending_vector_updates') or self._pending_vector_updates is None:
            logger.warning("Attempted to force process pending updates, but queue is not available.")
            return {"processed": 0, "failed": 0, "requeued": 0, "error": "Pending update queue not available."}

        processed_count = 0
        failed_count = 0
        requeued_count = 0
        items_to_requeue = []
        queue_size = self._pending_vector_updates.qsize()
        logger.info(f"Force processing {queue_size} pending vector updates.")

        # Process each item currently in the queue
        for _ in range(queue_size):
            update = None # Initialize update to None in case get() fails
            try:
                # Get an item, but don't wait forever
                update = await asyncio.wait_for(self._pending_vector_updates.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Queue became empty while processing
                logger.info("Pending update queue emptied during force processing.")
                break
            except asyncio.QueueEmpty: # More explicit catch
                logger.info("Pending update queue is empty.")
                break

            try: # Process the update item
                if not isinstance(update, dict) or 'operation' not in update or 'id' not in update:
                    logger.error(f"Invalid update item format in queue: {update}")
                    self._pending_vector_updates.task_done()
                    continue

                # Extract operation details
                operation = update.get('operation')
                item_id = update.get('id')
                embedding_data = update.get('embedding') # Can be None or list
                is_assembly = update.get('is_assembly', False)

                # Log the attempt
                logger.info(f"Force Processing: Retrying {operation} for {'assembly' if is_assembly else 'memory'} {item_id}")

                # Convert embedding back to numpy array if needed and possible
                embedding = None
                if embedding_data:
                    try:
                        embedding = np.array(embedding_data, dtype=np.float32)
                    except Exception as e:
                        logger.error(f"Failed to convert embedding for {item_id}: {e}. Cannot retry add/update.")
                        failed_count += 1
                        items_to_requeue.append(update) # Requeue if conversion failed
                        self._pending_vector_updates.task_done()
                        continue

                success = False
                if operation == 'add':
                    if embedding is not None:
                        success = await self.vector_index.add_async(item_id, embedding)
                    else:
                        logger.error(f"Cannot retry add operation without embedding for {item_id}")
                elif operation == 'update':
                    if embedding is not None:
                        success = await self.vector_index.update_entry_async(item_id, embedding)
                    else:
                        logger.error(f"Cannot retry update operation without embedding for {item_id}")
                elif operation == 'remove':
                    success = await self.vector_index.remove_vector_async(item_id)
                else:
                    logger.error(f"Unknown operation type '{operation}' for {item_id}")

                # Handle result
                if success:
                    logger.info(f"Force Processing: Successfully retried {operation} for {item_id}")
                    processed_count += 1

                    # For assemblies, update the timestamp on success
                    if is_assembly:
                        assembly_id = item_id[4:] if item_id.startswith("asm:") else item_id # Get actual ID
                        async with self._lock:  # Protect assembly update
                            if assembly_id in self.assemblies:
                                self.assemblies[assembly_id].vector_index_updated_at = datetime.now(timezone.utc)
                                self._dirty_memories.add(assembly_id)  # Mark for persistence
                                logger.info(f"Force Processing: Updated timestamp for assembly {assembly_id}")
                            else:
                                logger.warning(f"Force Processing: Assembly {assembly_id} not found for timestamp update.")
                else:
                    logger.warning(f"Force Processing: Failed retry of {operation} for {item_id}, will re-queue")
                    failed_count += 1
                    items_to_requeue.append(update)  # Requeue for later retry

                # Mark task as done in the queue
                self._pending_vector_updates.task_done()

            except asyncio.CancelledError:
                logger.warning("Force processing task cancelled.")
                if update: items_to_requeue.append(update) # Requeue if item was retrieved
                try: self._pending_vector_updates.task_done()
                except ValueError: pass # Ignore if already done
                raise # Re-raise cancellation

            except Exception as e:
                logger.error(f"Error force processing pending update for {update.get('id', 'N/A') if update else 'N/A'}: {e}", exc_info=True)
                failed_count += 1
                if update: items_to_requeue.append(update) # Requeue if error occurred after getting item
                try: self._pending_vector_updates.task_done()
                except ValueError: pass # Ignore if already done

        # Requeue failed items
        for item in items_to_requeue:
            await self._pending_vector_updates.put(item)
            requeued_count += 1

        if items_to_requeue:
            logger.warning(f"Force Processing: Requeued {requeued_count} failed updates.")

        return {
            "processed": processed_count,
            "failed": failed_count,
            "requeued": requeued_count
        }

    async def _vector_update_retry_loop(self) -> None:
        """Background task to process pending vector index updates from the queue.

        This is a critical stability component that handles retrying vector index operations
        (add, remove, update) that failed during regular processing. Without this loop,
        vector index failures would only be addressed during startup/repair.

        The loop runs continuously with configurable intervals, processing items from
        the _pending_vector_updates queue. Each operation is retried with full logging
        and error handling.
        """
        logger.info("Starting vector update retry background loop")
        # Use config values with defaults
        retry_interval = self.config.get('vector_retry_interval_seconds', 60)
        batch_size = self.config.get('vector_retry_batch_size', 10)
        max_retry_attempts = self.config.get('max_vector_retry_attempts', 5)
        trace_id = str(uuid.uuid4())[:8] # Unique ID for this loop instance

        # Use self._shutdown_signal consistently
        while not self._shutdown_signal.is_set():
            try:
                retry_count = 0
                # Determine max items to process in this batch
                current_queue_size = self._pending_vector_updates.qsize()
                max_retries = min(batch_size, current_queue_size)

                if max_retries > 0:
                    logger.info(f"[VectorRetry][{trace_id}] Processing up to {max_retries} pending vector operations (Queue size: {current_queue_size})")

                # Process a batch of items (up to batch_size) from the queue
                while retry_count < max_retries and not self._shutdown_signal.is_set():
                    op_data = None # Initialize in case get fails immediately
                    try:
                        # Get an item from the queue (non-blocking)
                        op_data = self._pending_vector_updates.get_nowait()
                        retry_count += 1

                        # Extract operation details more safely using .get()
                        op_type = op_data.get('operation') # Use 'operation' key consistently
                        memory_id = op_data.get('id')
                        embedding_data = op_data.get('embedding') # Can be None or list
                        is_assembly = op_data.get('is_assembly', False)
                        operation_trace = op_data.get('trace_id', 'unknown') # Original trace ID if available
                        retry_attempt = op_data.get('retry_count', 0) + 1

                        # Validate extracted data
                        if not op_type or not memory_id:
                            logger.error(f"[VectorRetry][{trace_id}] Invalid operation data structure: {op_data}")
                            self._pending_vector_updates.task_done() # Mark invalid item as done
                            continue

                        # Log the retry attempt
                        logger.info(f"[VectorRetry][{trace_id}] Attempt {retry_attempt} for {op_type} operation on {'assembly' if is_assembly else 'memory'} {memory_id} (original trace: {operation_trace})")

                        # Convert embedding back to numpy array if needed
                        embedding = None
                        if embedding_data:
                            try:
                                embedding = np.array(embedding_data, dtype=np.float32)
                            except Exception as e:
                                logger.error(f"[VectorRetry][{trace_id}] Failed to convert embedding for {memory_id}: {e}. Cannot retry add/update.")
                                # Requeue with increased retry count if possible
                                if retry_attempt < max_retry_attempts:
                                    op_data['retry_count'] = retry_attempt
                                    await self._pending_vector_updates.put(op_data)
                                else:
                                    logger.error(f"[VectorRetry][{trace_id}] Permanently failed {op_type} for {memory_id} due to embedding conversion error after {retry_attempt} attempts.")
                                self._pending_vector_updates.task_done() # Mark task done after requeue/fail
                                continue # Skip to next item

                        # Execute the appropriate operation based on type
                        success = False
                        if op_type == 'add':
                            if embedding is None:
                                logger.error(f"[VectorRetry][{trace_id}] Cannot retry add for {memory_id}: Missing embedding")
                            else:
                                logger.info(f"[VectorRetry][{trace_id}] Retrying add_async for {memory_id}")
                                success = await self.vector_index.add_async(memory_id, embedding)
                        elif op_type == 'remove':
                            logger.info(f"[VectorRetry][{trace_id}] Retrying remove_vector_async for {memory_id}")
                            success = await self.vector_index.remove_vector_async(memory_id)
                        elif op_type == 'update':
                            if embedding is None:
                                logger.error(f"[VectorRetry][{trace_id}] Cannot retry update for {memory_id}: Missing embedding")
                            else:
                                logger.info(f"[VectorRetry][{trace_id}] Retrying update_entry_async for {memory_id}")
                                success = await self.vector_index.update_entry_async(memory_id, embedding)
                        else:
                            logger.error(f"[VectorRetry][{trace_id}] Unknown operation type: {op_type} for {memory_id}")

                        # Handle the outcome
                        if success:
                            logger.info(f"[VectorRetry][{trace_id}] Successfully completed {op_type} operation for {memory_id}")

                            # For vector updates related to assemblies, update the timestamp
                            if is_assembly and op_type in ['add', 'update']:
                                try:
                                    assembly_id = memory_id[4:] if memory_id.startswith('asm:') else memory_id # Get base ID
                                    async with self._lock: # Lock for assembly update
                                        if assembly_id in self.assemblies:
                                            self.assemblies[assembly_id].vector_index_updated_at = datetime.now(timezone.utc) # Use datetime obj
                                            self._dirty_memories.add(assembly_id) # Mark assembly dirty for persistence
                                            logger.info(f"[VectorRetry][{trace_id}] Updated vector_index_updated_at for assembly {assembly_id}")
                                        else:
                                            logger.warning(f"[VectorRetry][{trace_id}] Assembly {assembly_id} not found in memory for timestamp update.")
                                except Exception as asm_err:
                                    logger.error(f"[VectorRetry][{trace_id}] Failed to update timestamp for assembly {memory_id}: {asm_err}", exc_info=True)
                        else:
                            # Operation failed again, requeue with increased retry count if below max
                            if retry_attempt < max_retry_attempts:
                                logger.warning(f"[VectorRetry][{trace_id}] Failed {op_type} for {memory_id}, requeueing (attempt {retry_attempt}/{max_retry_attempts})")
                                op_data['retry_count'] = retry_attempt
                                await self._pending_vector_updates.put(op_data)
                            else:
                                logger.error(f"[VectorRetry][{trace_id}] Permanently failed {op_type} for {memory_id} after {retry_attempt} attempts")
                                # Could add to a dead-letter queue or persistence for manual intervention

                    except asyncio.QueueEmpty: # Catch asyncio specific empty queue exception
                        # Queue is empty, nothing more to process in this batch
                        logger.debug(f"[VectorRetry][{trace_id}] Queue empty, finishing batch processing.")
                        break
                    except Exception as item_err:
                        logger.error(f"[VectorRetry][{trace_id}] Error processing vector operation for item {op_data.get('id', 'UNKNOWN') if op_data else 'UNKNOWN'}: {item_err}", exc_info=True)
                        # Decide whether to requeue on unexpected error
                        if op_data:
                             retry_attempt = op_data.get('retry_count', 0) + 1 # Recalculate attempt count
                             if retry_attempt < max_retry_attempts:
                                 op_data['retry_count'] = retry_attempt
                                 await self._pending_vector_updates.put(op_data)
                                 logger.warning(f"[VectorRetry][{trace_id}] Requeued item {op_data['id']} after unexpected error.")
                             else:
                                 logger.error(f"[VectorRetry][{trace_id}] Permanently failed item {op_data['id']} after unexpected error and max retries.")

                    finally:
                         # Ensure task_done is called even if errors occurred during processing
                         if op_data:
                             try:
                                 self._pending_vector_updates.task_done()
                             except ValueError:
                                 # Might happen if task was already marked done due to exception flow
                                 logger.debug(f"[VectorRetry][{trace_id}] task_done() called on already done task for {op_data.get('id','UNKNOWN')}")
                                 pass

                if retry_count > 0:
                    logger.info(f"[VectorRetry][{trace_id}] Completed processing batch of {retry_count} vector operations")

            except Exception as e:
                logger.error(f"[VectorRetry][{trace_id}] Unhandled error in vector update retry loop: {e}", exc_info=True)
                # Avoid tight loop on persistent errors
                await asyncio.sleep(retry_interval)

            # Wait before next processing cycle, checking shutdown signal
            try:
                # Use wait_for with the shutdown signal event
                await asyncio.wait_for(self._shutdown_signal.wait(), timeout=retry_interval)
                # If wait() returns without timeout, shutdown was signalled
                logger.info(f"[VectorRetry][{trace_id}] Shutdown signal received, exiting loop.")
                break
            except asyncio.TimeoutError:
                # This is the normal case - timeout reached, continue loop
                pass
            except asyncio.CancelledError:
                 logger.info(f"[VectorRetry][{trace_id}] Loop cancelled.")
                 break

        logger.info(f"Vector update retry background loop terminated ({trace_id})")
