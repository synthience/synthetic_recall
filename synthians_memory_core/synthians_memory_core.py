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
            'prune_check_interval': 600.0, # Check if pruning needed every 10 mins
            'max_memory_entries': 50000,
            'prune_threshold_percent': 0.9, # Prune when 90% full
            'min_quickrecal_for_ltm': 0.2, # Min score to keep after decay
            'assembly_threshold': 0.75,
            'max_assemblies_per_memory': 3,
            'adaptive_threshold_enabled': True,
            'initial_retrieval_threshold': 0.75,
            'vector_index_type': 'Cosine',  # 'L2', 'IP', 'Cosine'
            'persistence_batch_size': 100, # Batch size for persistence loop
            'check_index_on_retrieval': False, # New config option
            'index_check_interval': 3600, # New config option
            'migrate_to_idmap': True, # New config option
            **(config or {})
        }

        logger.info("SynthiansMemoryCore", "Initializing...", self.config)

        # --- Core Components ---
        self.geometry_manager = GeometryManager({
            'embedding_dim': self.config['embedding_dim'],
            'geometry_type': self.config['geometry'],
            'curvature': self.config['hyperbolic_curvature']
        })

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

        self.metadata_synthesizer = MetadataSynthesizer()  # Initialize metadata synthesizer

        # Initialize vector index for fast retrieval
        # If we're using IndexIDMap (which is the default and recommended), we need to use CPU
        # as FAISS GPU indexes don't support add_with_ids
        use_index_id_map = self.config.get('migrate_to_idmap', True)
        self.vector_index = MemoryVectorIndex({
            'embedding_dim': self.config['embedding_dim'],
            'storage_path': self.config['storage_path'],
            'index_type': self.config['vector_index_type'],
            'use_gpu': not use_index_id_map  # Use GPU only if not using IndexIDMap
        })
        
        # Check if we should migrate the index to the new IndexIDMap format
        if use_index_id_map:
            is_index_id_map = hasattr(self.vector_index.index, 'id_map')
            if not is_index_id_map:
                logger.info("Migrating vector index to use IndexIDMap for improved ID management")
                success = self.vector_index.migrate_to_idmap()
                if success:
                    logger.info("Successfully migrated vector index to IndexIDMap")
                else:
                    logger.warning("Failed to migrate vector index to IndexIDMap. Some features may not work correctly.")
        
        # --- Memory State ---
        self._memories: Dict[str, MemoryEntry] = {} # In-memory cache/working set
        self.assemblies: Dict[str, MemoryAssembly] = {}
        self.memory_to_assemblies: Dict[str, Set[str]] = {}
        self._dirty_memories: Set[str] = set() # Track modified memory IDs for persistence

        # --- Concurrency & Tasks ---
        self._lock = asyncio.Lock()
        self._background_tasks: List[asyncio.Task] = []
        self._initialized = False
        self._shutdown_signal = asyncio.Event()

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
                    # Decide if core can run without vector index (likely not)
                    return False  # Fail initialization if vector index fails
                logger.info("MemoryVectorIndex initialized.")
            else:
                logger.error("Vector Index component is None during initialization!")
                return False  # Cannot proceed without vector index

            # TODO: Load memories from persistence into cache if needed?
            # (Currently done on demand by get_memory_by_id_async)

            # TODO: Start background tasks if not disabled
            # if not os.environ.get("DISABLE_BACKGROUND", "false").lower() in ("true", "1"):
            #     self._start_background_tasks()

            self._initialized = True
            logger.info("SynthiansMemoryCore initialization complete.")
            return True

        except Exception as e:
            logger.error(f"Critical error during SynthiansMemoryCore initialization: {e}", exc_info=True)
            self._initialized = False  # Ensure it's marked as not initialized
            if hasattr(self, 'vector_index') and self.vector_index:
                 self.vector_index.state = "ERROR"  # Mark index state explicitly
            return False

    async def cleanup(self):
        """Clean up resources before shutdown.
        
        Part of Phase 5.8 stability improvements to ensure proper resource
        management during application shutdown.
        """
        logger.info("Cleaning up SynthiansMemoryCore resources")
        try:
            # Close vector index if available
            if hasattr(self, 'vector_index') and self.vector_index is not None:
                logger.info("Closing vector index")
                await self.vector_index.close() if hasattr(self.vector_index, 'close') else None
            
            # Ensure final persistence before shutdown
            if hasattr(self, 'memory_persistence') and self.memory_persistence is not None:
                logger.info("Final memory persistence before shutdown")
                await self.memory_persistence.persist_all() if hasattr(self.memory_persistence, 'persist_all') else None
            
            # Clean up assembly sync manager if available
            if hasattr(self, 'assembly_sync_manager') and self.assembly_sync_manager is not None:
                logger.info("Closing assembly sync manager")
                await self.assembly_sync_manager.shutdown() if hasattr(self.assembly_sync_manager, 'shutdown') else None
            
            # Cancel any pending tasks
            if hasattr(self, '_background_tasks'):
                for task in self._background_tasks:
                    if not task.done():
                        logger.info(f"Cancelling background task {task.get_name() if hasattr(task, 'get_name') else 'unnamed'}")
                        task.cancel()
            
            logger.info("SynthiansMemoryCore cleanup completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during SynthiansMemoryCore cleanup: {str(e)}")
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

        # Reset state
        self._initialized = False
        # Reset shutdown signal for potential re-initialization
        self._shutdown_signal = asyncio.Event()
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

        # 8. Update Assemblies
        await self._update_assemblies(memory)

        # 9. Add to vector index for fast retrieval
        if normalized_embedding is not None and self.vector_index is not None:
            logger.debug("Adding memory to vector index...")
            added_to_index = await self.vector_index.add_async(memory.id, normalized_embedding)
            if not added_to_index:
                logger.error(f"Failed to add memory {memory.id} to vector index.")
                return None
        logger.debug("SynthiansMemoryCore", f"Added memory {memory.id} to vector index")

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
                # TEMPORARILY set threshold to 0.0 for debugging the '0 memories' issue
                # Will revert to self.config['initial_retrieval_threshold'] once issue is resolved
                current_threshold = 0.0  # DEBUG: Lowered to 0.0 to see if any memories pass
                logger.warning(f"[DEBUG MODE] Using debug threshold of {current_threshold} to diagnose '0 memories' issue")
            else:
                logger.debug(f"Using explicit threshold from request: {current_threshold:.4f}")

            # Make vector index integrity check configurable and periodic
            check_index = self.config.get('check_index_on_retrieval', False)
            current_time = time.time()
            last_check_time = getattr(self, '_last_index_check_time', 0)
            check_interval = self.config.get('index_check_interval', 3600)  # Default: check once per hour
            
            if check_index or (current_time - last_check_time > check_interval):
                is_consistent, diagnostics = self.vector_index.verify_index_integrity()
                self._last_index_check_time = current_time
                logger.debug(f"Vector index status - Consistent: {is_consistent}, FAISS: {diagnostics.get('faiss_count')}, Mapping: {diagnostics.get('mapping_count')}")
                
                # Warn if inconsistency detected
                if not is_consistent:
                    logger.warning(f"Vector index inconsistency detected! FAISS count: {diagnostics.get('faiss_count')}, Mapping count: {diagnostics.get('mapping_count')}")

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
                    logger.error(f"Error during assembly activation: {e}")
            
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
                        memory_dict["relevance_score"] = similarity
                        
                        # ADDED: Calculate and apply assembly boost (Phase 5.8)
                        assembly_boost = 0.0
                        max_activation = 0.0
                        boost_reason = "none"
                        mem_id = memory_dict.get("id")
                        associated_assembly_ids = set()
                        
                        # Get the assemblies associated with this memory
                        async with self._lock:  # Need lock to access memory_to_assemblies safely
                            mem_id_lower = mem_id.lower() if isinstance(mem_id, str) else mem_id
                            associated_assembly_ids = self.memory_to_assemblies.get(mem_id, set())
                            # Try lowercase version if not found
                            if not associated_assembly_ids and mem_id != mem_id_lower:
                                associated_assembly_ids = self.memory_to_assemblies.get(mem_id_lower, set())
                                if associated_assembly_ids:
                                    logger.debug(f"Found assemblies using lowercase memory ID: {mem_id_lower}")
                        
                        # Enhanced debug logging
                        logger.debug(f"Memory {mem_id} is associated with assemblies: {associated_assembly_ids}")
                        logger.debug(f"Available activation scores: {assembly_activation_scores}")
                        
                        # Use the pre-calculated assembly activation scores from earlier
                        # Remove incorrect line that tried to redefine assembly_activation_scores locally
                        
                        if associated_assembly_ids:
                            # Find max activation score from the activated assemblies
                            active_assemblies = []
                            for asm_id in associated_assembly_ids:
                                activation = assembly_activation_scores.get(asm_id, 0.0)
                                logger.debug(f"Assembly {asm_id} activation: {activation}")
                                if activation > 0:
                                    # Check if assembly is synchronized with vector index
                                    if asm_id in self.assemblies and self.assemblies[asm_id].vector_index_updated_at:
                                        active_assemblies.append((asm_id, activation))
                                        logger.debug(f"Adding assembly {asm_id} with activation {activation} to active_assemblies")
                                    else:
                                        logger.debug(f"Assembly {asm_id} not synchronized, skipping boost")
                            
                            if active_assemblies:
                                # Find max activation among synchronized assemblies
                                max_asm_id, max_activation = max(active_assemblies, key=lambda x: x[1], default=("", 0.0))
                                logger.debug(f"Max activation for memory {mem_id}: {max_activation} from assembly {max_asm_id}")
                                
                                # Calculate boost based on configuration
                                boost_mode = self.config.get('assembly_boost_mode', 'linear')
                                boost_factor = self.config.get('assembly_boost_factor', 0.2)
                                
                                if boost_mode == "linear":
                                    assembly_boost = max_activation * boost_factor
                                    boost_reason = f"linear(act:{max_activation:.2f}*f:{boost_factor:.2f})"
                                elif boost_mode == "multiplicative":
                                    assembly_boost = similarity * max_activation * boost_factor
                                    boost_reason = f"multiplicative(sim:{similarity:.2f}*act:{max_activation:.2f}*f:{boost_factor:.2f})"
                                else:
                                    # Default additive behavior
                                    assembly_boost = max_activation * boost_factor
                                    boost_reason = f"default(act:{max_activation:.2f}*f:{boost_factor:.2f})"
                                
                                # Clamp boost to prevent exceeding 1.0 total score
                                assembly_boost = min(assembly_boost, max(0.0, 1.0 - similarity))
                                
                                # Update relevance score with boost
                                memory_dict["relevance_score"] = min(1.0, similarity + assembly_boost)
                                logger.debug(f"Memory {mem_id}: Applied assembly boost {assembly_boost:.4f} from assembly {max_asm_id} (activation: {max_activation:.4f})")
                            else:
                                boost_reason = "no_activated_assemblies"
                        else:
                            boost_reason = "no_associated_assemblies"
                        
                        # Store boost information in the memory dictionary
                        memory_dict["boost_info"] = {
                            "base_similarity": float(similarity),
                            "assembly_boost": float(assembly_boost),
                            "max_activation": float(max_activation),
                            "boost_reason": boost_reason
                        }
                        
                        scored_candidates.append(memory_dict)
                        logger.debug(f"Memory {mem_id}: similarity={similarity:.4f}")
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
            
            # Sort by similarity score (descending)
            sorted_candidates = sorted(scored_candidates, key=lambda x: x.get("similarity", 0.0), reverse=True)

            # ENHANCED: Log all candidates with their scores before filtering
            logger.info(f"[Similarity Results] Found {len(sorted_candidates)} scored candidates before threshold filtering")
            logger.debug(f"Threshold filtering: Using threshold {current_threshold:.4f}")
            
            similarities = [(c.get('id'), c.get('similarity', 0.0)) for c in sorted_candidates[:10]]
            logger.debug(f"Top 10 similarities: {similarities}")
            
            # Apply threshold filtering
            logger.info(f"[Threshold Filtering] Starting threshold filtering with {len(sorted_candidates)} candidates")
            filtered_candidates = []
            candidates_filtered_out = []
            
            threshold_to_use = threshold if threshold is not None else self.threshold_calibrator.current_threshold if self.threshold_calibrator else self.config.get('initial_retrieval_threshold', 0.75)
            
            for c in sorted_candidates:
                similarity = c.get("similarity", 0.0)
                mem_id = c.get("id", "unknown")
                if similarity >= threshold_to_use:
                    filtered_candidates.append(c)
                    logger.debug(f"Memory {mem_id} PASSED threshold with similarity {similarity:.4f} >= {threshold_to_use:.4f}")
                else:
                    candidates_filtered_out.append((mem_id, similarity))
                    logger.debug(f"Memory {mem_id} FILTERED OUT with similarity {similarity:.4f} < {threshold_to_use:.4f}")
            
            # Log summary of threshold filtering results
            logger.info(f"[Threshold Filtering] Kept {len(filtered_candidates)} candidates, filtered out {len(candidates_filtered_out)} candidates")
            
            # Log the first few filtered out candidates for debugging
            if candidates_filtered_out:
                logger.debug(f"First 5 filtered out (ID, similarity): {candidates_filtered_out[:5]}")

            # Step 4: Apply emotional gating if requested
            if user_emotion and self.emotional_gating:
                logger.info(f"[Emotional Gating] Applying with user_emotion: {user_emotion}, candidates: {len(filtered_candidates)}") 
                try:
                    filtered_candidates = await self.emotional_gating.gate_memories_by_context(
                        filtered_candidates, user_emotion_context=user_emotion
                    )
                    logger.info(f"[Emotional Gating] Result: {len(filtered_candidates)} candidates")
                except Exception as e:
                    logger.error(f"Error during emotional gating: {e}")
                    # Continue with original filtered candidates if gating fails
            
            # Step 5: Apply metadata filtering if requested
            if metadata_filter:
                logger.info(f"[Metadata Filtering] Applying filter: {metadata_filter}") 
                pre_filter_count = len(filtered_candidates)
                
                filtered_candidates = self._filter_by_metadata(filtered_candidates, metadata_filter)
                
                post_filter_count = len(filtered_candidates)
                filter_diff = pre_filter_count - post_filter_count
                logger.info(f"[Metadata Filtering] Result: {post_filter_count} candidates remain ({filter_diff} removed)") 
                
                # Log the metadata of the remaining candidates
                if filtered_candidates:
                    # Get the first candidate's metadata keys for reference
                    first_meta_keys = list(filtered_candidates[0].get("metadata", {}).keys())[:5]  # First 5 keys
                    logger.debug(f"[Post-Metadata Filtering] First candidate metadata keys: {first_meta_keys}")
            else:
                logger.debug("[Metadata Filtering] Skipped (no metadata filter provided)")

            # *** ENHANCED POST-FILTERING LOG ***
            logger.info(f"[Final Filtering] Total filtered candidates: {len(filtered_candidates)}")
            if filtered_candidates:
                final_top_ids = [c.get('id') for c in filtered_candidates[:5]]
                logger.info(f"[Final Filtering] Top 5 candidate IDs after all filtering: {final_top_ids}")
            else:
                logger.warning("[Final Filtering] No candidates remain after all filtering steps")

            # Return top_k results (simplify slicing)
            if len(filtered_candidates) >= top_k:
                final_memories = filtered_candidates[:top_k]
                logger.info(f"[Results] Returning {top_k} memories out of {len(filtered_candidates)} filtered candidates")
            else:
                final_memories = filtered_candidates.copy() # Take all if fewer than top_k, and make a copy to be safe
                logger.info(f"[Results] Returning all {len(final_memories)} filtered candidates (fewer than requested {top_k})")

            # *** ENHANCED FINAL CHECK ***
            if final_memories:
                final_ids = [mem.get('id') for mem in final_memories]
                final_scores = [mem.get('similarity', 0.0) for mem in final_memories]
                logger.info(f"[Results] Final memory IDs: {final_ids}")
                logger.info(f"[Results] Final similarity scores: {final_scores}")
            else:
                logger.warning("[Results] No memories to return!")

            retrieval_time = (time.time() - start_time) * 1000
            # Log the length again, just before returning
            logger.info("SynthiansMemoryCore", f"Retrieved {len(final_memories)} memories", {
                "top_k": top_k, "threshold": current_threshold, "user_emotion": user_emotion, "time_ms": retrieval_time
            })
            
            # DIRECT DEBUG: Log full response payload length
            response = {"success": True, "memories": final_memories, "error": None}
            logger.info(f"[Response] Payload stats: success={response['success']}, memories_count={len(response['memories'])}")
            
            return response

        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error in retrieve_memories: {str(e)}")
            logger.error(traceback.format_exc())
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

        # Log the query embedding stats for debugging
        if query_embedding is not None:
            logger.debug(f"[Candidate Gen] Query embedding shape: {query_embedding.shape}, sum: {np.sum(query_embedding):.4f}, mean: {np.mean(query_embedding):.4f}")
            if np.isnan(query_embedding).any() or np.isinf(query_embedding).any():
                logger.warning(f"[Candidate Gen] WARNING: Query embedding contains NaN/Inf values!")

        assembly_candidates = set()
        direct_candidates = set()
        
        # Create a dictionary to track assembly activation scores
        assembly_activation_scores = {}

        # 1. Assembly Activation
        activated_assemblies = await self._activate_assemblies(query_embedding)
        
        # Enhanced logging to debug assembly activation
        logger.debug(f"[Candidate Gen] Got {len(activated_assemblies)} activated assemblies from _activate_assemblies")
        for assembly, score in activated_assemblies:
            logger.debug(f"[Candidate Gen] Activated assembly: {assembly.assembly_id if hasattr(assembly, 'assembly_id') else 'unknown'}, score={score:.4f}")
        
        # Store activation scores in the dictionary
        for assembly, activation_score in activated_assemblies:
            # Use assembly_id attribute instead of id
            if hasattr(assembly, 'assembly_id'):  # Safety check
                assembly_activation_scores[assembly.assembly_id] = activation_score
                logger.debug(f"Stored activation score {activation_score} for assembly {assembly.assembly_id}")
        
        # Use top 5 assemblies for candidate generation
        for assembly, activation_score in activated_assemblies[:5]:
            if activation_score > 0.01:  # Lower activation threshold to ensure assemblies are used
                if hasattr(assembly, 'memories') and assembly.memories:
                    assembly_candidates.update(assembly.memories)
                    logger.debug(f"[Candidate Gen] Added {len(assembly.memories)} memories from assembly {assembly.assembly_id}")
                else:
                    logger.warning(f"[Candidate Gen] Assembly {assembly.assembly_id if hasattr(assembly, 'assembly_id') else 'unknown'} has no memories or memories attribute is missing")
        
        logger.info(f"[Candidate Gen] Found {len(assembly_candidates)} candidates from assembly activation")
        
        # 2. Direct Vector Search using FAISS Index
        search_threshold = 0.0  # Set to zero to get all candidates regardless of similarity
        faiss_count = self.vector_index.count()
        id_mapping_count = len(self.vector_index.id_to_index) if hasattr(self.vector_index, 'id_to_index') else 0
        
        logger.info(f"[Candidate Gen] Vector index stats: FAISS count={faiss_count}, ID mapping count={id_mapping_count}")
        
        # Check if index is empty
        if faiss_count == 0:
            logger.warning(f"[Candidate Gen] FAISS index is empty! Check memory creation and indexing.")
        
        search_results = await self.vector_index.search_async(query_embedding, k=min(limit, max(faiss_count, 1)))
        
        logger.info(f"[Candidate Gen] FAISS search returned {len(search_results)} results")
        
        # Log detailed search results
        if search_results:
            top_results = search_results[:5] if len(search_results) > 5 else search_results
            result_details = [f"({mem_id}, {sim:.4f})" for mem_id, sim in top_results]
            logger.info(f"[Candidate Gen] Top FAISS results: {', '.join(result_details)}")
        else:
            logger.warning(f"[Candidate Gen] FAISS search returned ZERO results! Check indexing.")
            
        for memory_id, similarity in search_results:
            direct_candidates.add(memory_id)

        # 3. Get the most recently added memories as fallback
        # This ensures we always have candidates even if similarity search fails
        async with self._lock:
            # Get IDs of memories in our persistence index
            memory_ids = list(self.persistence.memory_index.keys())
            logger.info(f"[Candidate Gen] Persistence index has {len(memory_ids)} memories total")
            
        # Take the most recent ones if we have any
        if memory_ids and len(direct_candidates) == 0:
            # Sort by creation time if available, otherwise just take the last few
            recent_candidates = set(memory_ids[-min(5, len(memory_ids)):])  # Get the last 5 memories
            logger.info(f"[Candidate Gen] Added {len(recent_candidates)} recent memories as fallback candidates: {list(recent_candidates)}")
            direct_candidates.update(recent_candidates)
        elif len(memory_ids) == 0:
            logger.warning(f"[Candidate Gen] Persistence index is EMPTY! No memories have been created.")

        # Combine candidates
        all_candidate_ids = assembly_candidates.union(direct_candidates)
        logger.info(f"[Candidate Gen] Found {len(all_candidate_ids)} total candidate IDs: {list(all_candidate_ids)[:10]}")

        # Fetch MemoryEntry objects as dictionaries
        final_candidates = []
        for mem_id in all_candidate_ids:
            # Log before attempting to load
            logger.debug(f"[Candidate Gen] Attempting to load memory with ID: {mem_id}")
            # Use our new async method to get the memory from disk if not in cache
            memory = await self.get_memory_by_id_async(mem_id)
            if memory:
                # Make sure to convert memory to dict before returning
                mem_dict = memory.to_dict()
                final_candidates.append(mem_dict)
                logger.debug(f"[Candidate Gen] Successfully loaded memory {mem_id}: content_len={len(mem_dict.get('content', ''))}, embedding_shape={memory.embedding.shape if memory.embedding is not None else 'None'}")
            else:
                logger.warning(f"[Candidate Gen] Failed to load memory {mem_id}! Check persistence storage.")

        # Always ensure we return at least some candidates for scoring/filtering
        if len(final_candidates) == 0:
            logger.warning("[Candidate Gen] No candidates found after loading! This will result in empty retrieval results.")
            # Log vector index statistics to help debug
            is_consistent, diagnostics = self.vector_index.verify_index_integrity()
            logger.warning(f"[Candidate Gen] Vector index diagnostics: consistent={is_consistent}, {diagnostics}")
            
            # Check storage files
            import os
            if hasattr(self.persistence, 'storage_path'):
                storage_files = os.listdir(self.persistence.storage_path) if os.path.exists(self.persistence.storage_path) else []
                logger.warning(f"[Candidate Gen] Storage directory contents: {storage_files[:10]}")
                
                # Check for FAISS index file
                faiss_path = os.path.join(self.persistence.storage_path, 'faiss_index.bin')
                mapping_path = os.path.join(self.persistence.storage_path, 'id_to_index_mapping.json')
                logger.warning(f"[Candidate Gen] FAISS index file exists: {os.path.exists(faiss_path)}")
                logger.warning(f"[Candidate Gen] ID mapping file exists: {os.path.exists(mapping_path)}")

        logger.info(f"[Candidate Gen] Returning {len(final_candidates)} final candidates for scoring/filtering")
        # Return both the candidates and activation scores
        return final_candidates[:limit * 2], assembly_activation_scores

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
            
        # Add detailed debug logging for the query embedding
        logger.debug(f"[Assembly Debug] Query embedding shape: {query_embedding.shape}, norm: {np.linalg.norm(query_embedding)}")
        logger.debug(f"[Assembly Debug] Query embedding snippet: {query_embedding[:5]}")
        
        # Fix: Use dictionary access instead of attribute access
        now = datetime.now(timezone.utc)
        drift_limit = self.config.get('max_allowed_drift_seconds', 3600)  # Default 1 hour if not specified
        assembly_threshold = self.config.get('assembly_threshold', 0.1)  # Default threshold if not specified
        logger.debug(f"[Assembly Debug] Assembly activation threshold: {assembly_threshold}")
            
        # Search the vector index for assembly vectors
        prefix = "asm:"
        logger.debug(f"[Assembly Debug] Searching for assemblies with prefix: {prefix}")
        
        try:
            # Logging the current state of vector index to verify assemblies were added
            stats = self.vector_index.get_stats()
            logger.debug(f"[Assembly Debug] Vector index stats: {stats}")
            
            # FIXED: Remove id_prefix parameter, search all vectors and filter results afterward
            search_results = await self.vector_index.search_async(
                query_embedding, 
                k=200  # Larger value to ensure we find all relevant assemblies after filtering
            )
            
            # Post-search filtering for assemblies (ids starting with prefix)
            asm_results = []
            for memory_id, similarity in search_results:
                if memory_id.startswith(prefix):
                    asm_results.append((memory_id, similarity))
            
            logger.debug(f"[Assembly Debug] Found {len(asm_results)} potential assemblies after filtering")
            
            # Filter and keep only IDs that start with assembly prefix
            asm_results = [r for r in search_results if r[0].startswith("asm:")]
            logger.debug(f"Found {len(asm_results)} assembly IDs in search results")
            
            # Debug: show available assemblies
            logger.debug(f"[ACTIVATE_DBG] Available assemblies in dictionary: {list(self.assemblies.keys())}")

            activated_assemblies = []
            max_activation_time = now - timedelta(seconds=drift_limit)

            for asm_id_with_prefix, similarity in asm_results:
                logger.debug(f"[ACTIVATE_DBG] Examining result: ID='{asm_id_with_prefix}', Sim={similarity:.4f}") # Log raw result

                # Extract the actual assembly ID (remove "asm:" prefix)
                assembly_id = asm_id_with_prefix[4:] if asm_id_with_prefix.startswith("asm:") else asm_id_with_prefix
                logger.debug(f"[ACTIVATE_DBG] Extracted assembly_id: '{assembly_id}'") # Log extracted ID

                # Check if assembly exists in the core's dictionary
                assembly_present_in_dict = assembly_id in self.assemblies
                logger.debug(f"[ACTIVATE_DBG] Assembly '{assembly_id}' present in self.assemblies? {assembly_present_in_dict}") # Log lookup result

                # Skip results below threshold
                local_assembly_threshold = self.config.get('assembly_threshold')
                if similarity < local_assembly_threshold:
                    logger.debug(f"[ACTIVATE_DBG] Skipping '{assembly_id}': similarity {similarity:.6f} below threshold {local_assembly_threshold}")
                    continue

                # FIXED: Get assembly from self.assemblies instead of persistence.get_assembly 
                assembly = self.assemblies.get(assembly_id)
                if assembly is None:
                    logger.warning(f"[ACTIVATE_DBG] Assembly '{assembly_id}' lookup returned None. Skipping.")
                    continue

                logger.debug(f"[ACTIVATE_DBG] Found assembly object: Name='{assembly.name}', ID='{assembly.assembly_id}'")

                # Check if the assembly is synchronized with the vector index
                enable_sync = self.config.get('enable_assembly_sync', True)  # Default to True if not specified
                if not enable_sync:
                    logger.debug(f"[ACTIVATE_DBG] Sync check disabled for '{assembly_id}'.")
                    # Synchronization is disabled, treat all assemblies as valid
                    activated_assemblies.append((assembly, similarity))
                    logger.debug(f"[ACTIVATE_DBG] Activated '{assembly_id}' (Sync Disabled)")
                    continue

                # Check synchronization status
                updated_at = assembly.vector_index_updated_at
                logger.debug(f"[ACTIVATE_DBG] Checking sync for '{assembly_id}': updated_at={updated_at}") # Log timestamp
                if assembly.vector_index_updated_at is None:
                    logger.debug(f"[ACTIVATE_DBG] Skipping '{assembly_id}': updated_at is None.")
                    continue

                # Check for embedding drift
                drift_seconds = (now - updated_at).total_seconds()
                logger.debug(f"[ACTIVATE_DBG] Checking drift for '{assembly_id}': drift={drift_seconds:.2f}s, limit={drift_limit}s") # Log drift
                if assembly.vector_index_updated_at < max_activation_time:
                    logger.debug(f"[ACTIVATE_DBG] Skipping '{assembly_id}': Drift limit exceeded.")
                    continue

                # All checks passed, add to activated assemblies
                logger.info(f"[ACTIVATE_DBG] ACTIVATE SUCCESS for '{assembly_id}'")
                activated_assemblies.append((assembly, similarity))
                logger.debug(f"Activated assembly {assembly_id} with similarity {similarity}")
                
            # Log final activation count
            logger.debug(f"[Assembly Debug] Total activated assemblies: {len(activated_assemblies)}")
            
            # Return the list of (assembly, similarity) tuples
            return activated_assemblies
                
        except Exception as e:
            logger.error(f"Error during assembly activation: {str(e)}", exc_info=True)
            return []

    async def _update_assemblies(self, memory: MemoryEntry):
        """Find or create assemblies for a new memory."""
        if memory.embedding is None: return

        suitable_assemblies = []
        best_similarity = 0.0
        best_assembly_id = None

        async with self._lock: # Access shared self.assemblies
             for assembly_id, assembly in self.assemblies.items():
                  similarity = assembly.get_similarity(memory.embedding)
                  if similarity >= self.config['assembly_threshold']:
                       suitable_assemblies.append((assembly_id, similarity))
                  if similarity > best_similarity:
                       best_similarity = similarity
                       best_assembly_id = assembly_id

        # Sort suitable assemblies by similarity
        suitable_assemblies.sort(key=lambda x: x[1], reverse=True)

        # Add memory to best matching assemblies (up to max limit)
        added_count = 0
        assemblies_updated = set()
        for assembly_id, _ in suitable_assemblies[:self.config['max_assemblies_per_memory']]:
            async with self._lock: # Lock for modifying assembly
                 if assembly_id in self.assemblies:
                     assembly = self.assemblies[assembly_id]
                     if assembly.add_memory(memory):
                          added_count += 1
                          assemblies_updated.add(assembly_id)
                          # Update memory_to_assemblies mapping
                          if memory.id not in self.memory_to_assemblies:
                               self.memory_to_assemblies[memory.id] = set()
                          self.memory_to_assemblies[memory.id].add(assembly_id)
                          self._dirty_memories.add(assembly.assembly_id) # Mark assembly as dirty

        # If no suitable assembly found, consider creating a new one
        if added_count == 0 and best_similarity > self.config['assembly_threshold'] * 0.5: # Threshold to create new
             async with self._lock: # Lock for creating new assembly
                 # Double check if a suitable assembly was created concurrently
                 assembly_exists = False
                 for asm_id in self.memory_to_assemblies.get(memory.id, set()):
                      if asm_id in self.assemblies: assembly_exists = True; break

                 if not assembly_exists:
                     logger.info("SynthiansMemoryCore", f"Creating new assembly seeded by memory {memory.id[:8]}")
                     new_assembly = MemoryAssembly(geometry_manager=self.geometry_manager, name=f"Assembly around {memory.id[:8]}")
                     if new_assembly.add_memory(memory):
                          self.assemblies[new_assembly.assembly_id] = new_assembly
                          assemblies_updated.add(new_assembly.assembly_id)
                          # Update mapping
                          if memory.id not in self.memory_to_assemblies:
                               self.memory_to_assemblies[memory.id] = set()
                          self.memory_to_assemblies[memory.id].add(new_assembly.assembly_id)
                          added_count += 1
                          self._dirty_memories.add(new_assembly.assembly_id) # Mark assembly as dirty

        if added_count > 0:
             logger.debug("SynthiansMemoryCore", f"Updated {added_count} assemblies for memory {memory.id}", {"assemblies": list(assemblies_updated)})

    async def provide_feedback(self, memory_id: str, similarity_score: float, was_relevant: bool):
        """Provide feedback to the threshold calibrator."""
        if self.threshold_calibrator:
            self.threshold_calibrator.record_feedback(similarity_score, was_relevant)
            logger.debug("SynthiansMemoryCore", "Recorded feedback", {"memory_id": memory_id, "score": similarity_score, "relevant": was_relevant})

    async def detect_contradictions(self, threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Detect potential causal contradictions using embeddings."""
        contradictions = []
        async with self._lock: # Access shared _memories
            memories_list = list(self._memories.values())

        # Basic Keyword Filtering for Causal Statements (Can be improved with NLP)
        causal_keywords = ["causes", "caused", "leads to", "results in", "effect of", "affects"]
        causal_memories = [m for m in memories_list if m.embedding is not None and any(k in m.content.lower() for k in causal_keywords)]

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
        """Periodically persist changed memories."""
        logger.info("SynthiansMemoryCore","Persistence loop started.")
        persist_interval = self.config.get('persistence_interval', 60.0)
        try:
            while not self._shutdown_signal.is_set():
                try:
                    # Wait for the configured interval OR the shutdown signal
                    await asyncio.wait_for(
                        self._shutdown_signal.wait(),
                        timeout=persist_interval
                    )
                    # If wait() finished without timeout, it means signal was set
                    logger.info("SynthiansMemoryCore","Persistence loop: Shutdown signal received during wait.")
                    break # Exit loop if shutdown signal is set
                except asyncio.TimeoutError:
                    # Timeout occurred, time to persist
                    if not self._shutdown_signal.is_set(): # Double-check signal
                        logger.debug("SynthiansMemoryCore", "Running periodic persistence.")
                        await self._persist_dirty_items() # Persist dirty items
                except asyncio.CancelledError:
                    logger.info("SynthiansMemoryCore","Persistence loop cancelled during wait.")
                    break # Exit loop if cancelled
        except asyncio.CancelledError:
            logger.info("SynthiansMemoryCore","Persistence loop received cancel signal.")
        except Exception as e:
            logger.error("SynthiansMemoryCore","Persistence loop error", {"error": str(e)}, exc_info=True)
        finally:
            # Remove final save attempt to avoid 'no running event loop' errors
            # The main shutdown method should handle any critical final saves
            logger.info("SynthiansMemoryCore","Persistence loop stopped.")

    async def _decay_and_pruning_loop(self):
        """Periodically decay memory scores and prune old/irrelevant memories."""
        logger.info("SynthiansMemoryCore","Decay/Pruning loop started.")
        decay_interval = self.config.get('decay_interval', 3600.0)
        prune_interval = self.config.get('prune_check_interval', 600.0)
        # Determine the shortest interval to check the shutdown signal more frequently
        check_interval = min(decay_interval, prune_interval, 5.0) # Check at least every 5s
        last_decay_time = time.monotonic()
        last_prune_time = time.monotonic()
        try:
            while not self._shutdown_signal.is_set():
                try:
                    # Wait for the check interval OR the shutdown signal
                    await asyncio.wait_for(
                        self._shutdown_signal.wait(),
                        timeout=check_interval
                    )
                    # If wait() finished without timeout, it means signal was set
                    logger.info("SynthiansMemoryCore","Decay/Pruning loop: Shutdown signal received during wait.")
                    break # Exit loop if shutdown signal is set
                except asyncio.TimeoutError:
                    # Timeout occurred, check if it's time for decay or pruning
                    now = time.monotonic()
                    if not self._shutdown_signal.is_set():
                        # Decay Check
                        if now - last_decay_time >= decay_interval:
                            logger.info("SynthiansMemoryCore","Running memory decay check.")
                            try:
                                await self._apply_decay()
                                last_decay_time = now
                            except Exception as decay_e:
                                logger.error("SynthiansMemoryCore","Error during decay application", {"error": str(decay_e)})
                        # Pruning Check (can happen more often than decay)
                        if now - last_prune_time >= prune_interval:
                             logger.debug("SynthiansMemoryCore","Running pruning check.")
                             try:
                                 await self._prune_if_needed()
                                 last_prune_time = now
                             except Exception as prune_e:
                                 logger.error("SynthiansMemoryCore","Error during pruning check", {"error": str(prune_e)})
                except asyncio.CancelledError:
                    logger.info("SynthiansMemoryCore","Decay/Pruning loop cancelled during wait.")
                    break # Exit loop if cancelled
        except asyncio.CancelledError:
            logger.info("SynthiansMemoryCore","Decay/Pruning loop received cancel signal.")
        except Exception as e:
            logger.error("SynthiansMemoryCore","Decay/Pruning loop error", {"error": str(e)}, exc_info=True)
        finally:
            logger.info("SynthiansMemoryCore","Decay/Pruning loop stopped.")

    async def _persist_dirty_items(self):
        """Persist all items marked as dirty."""
        # Get a copy of dirty IDs and clear the set under the lock
        async with self._lock:
            if not self._dirty_memories:
                logger.debug("SynthiansMemoryCore", "No dirty items to persist.")
                # Save index periodically even if no memories changed? Maybe not needed if index saved on add/delete.
                # await self.persistence._save_index_no_lock() # Save index under lock
                # vector_index_save_needed = ... # Logic to check if vector index needs saving
                # if vector_index_save_needed: self.vector_index.save() # Save outside lock?
                return

            logger.info(f"Persisting {len(self._dirty_memories)} dirty items...")
            items_to_save = list(self._dirty_memories)
            self._dirty_memories.clear() # Clear the set *after* copying

        # --- Perform saving outside the main core lock ---
        persist_count = 0
        persist_errors = 0
        batch_size = self.config.get('persistence_batch_size', 100)

        for i in range(0, len(items_to_save), batch_size):
            batch_ids = items_to_save[i:i+batch_size]
            save_tasks = []
            items_in_batch = {}

            # Get copies of items under lock first
            async with self._lock:
                 for item_id in batch_ids:
                     item = None
                     if item_id.startswith("mem_") and item_id in self._memories:
                         item = copy.deepcopy(self._memories[item_id])
                         item_type = "memory"
                     elif item_id.startswith("asm_") and item_id in self.assemblies:
                         item = copy.deepcopy(self.assemblies[item_id])
                         item_type = "assembly"

                     if item:
                         items_in_batch[item_id] = (item, item_type)
                     else:
                          logger.warning(f"Dirty item {item_id} not found in cache for persistence.")

            # Now create save tasks outside the lock
            for item_id, (item, item_type) in items_in_batch.items():
                 if item_type == "memory":
                      save_tasks.append(self.persistence.save_memory(item))
                 elif item_type == "assembly":
                      save_tasks.append(self.persistence.save_assembly(item))

            # Run tasks for the batch
            if save_tasks:
                 results = await asyncio.gather(*save_tasks, return_exceptions=True)
                 for result, item_id in zip(results, items_in_batch.keys()):
                      if isinstance(result, Exception) or result is False:
                           logger.error(f"Error persisting dirty item {item_id}", {"error": str(result)})
                           persist_errors += 1
                           # Optionally re-add to dirty set for next attempt?
                           # async with self._lock: self._dirty_memories.add(item_id)
                      else:
                           persist_count += 1

            # Check for shutdown signal between batches
            if self._shutdown_signal.is_set():
                logger.info("Persistence interrupted by shutdown signal.")
                break

        logger.info(f"Periodic persistence completed: Saved {persist_count} dirty items with {persist_errors} errors.")

        # Save index and vector index after processing dirty items
        await self.persistence._save_index() # Use the method with the lock
        if hasattr(self, 'vector_index') and self.vector_index:
            try:
                self.vector_index.save() # This might block briefly
            except Exception as e:
                 logger.error("Failed to save vector index during periodic persistence.", {"error": str(e)})

    async def _apply_decay(self):
        """Apply decay to QuickRecal scores."""
        async with self._lock:
             # No actual score modification needed, just update metadata if desired
             logger.info("SynthiansMemoryCore", f"Decay check completed for {len(self._memories)} memories (no scores changed).")


    async def _prune_if_needed(self):
        """Prune memories if storage limit is exceeded."""
        async with self._lock:
             current_size = len(self._memories)
             max_size = self.config['max_memory_entries']
             prune_threshold = int(max_size * self.config['prune_threshold_percent'])

             if current_size <= prune_threshold:
                  return # No pruning needed

             logger.info("SynthiansMemoryCore", f"Memory usage ({current_size}/{max_size}) exceeds threshold ({prune_threshold}). Starting pruning.")
             num_to_prune = current_size - int(max_size * 0.85) # Prune down to 85%

             # Get memories sorted by effective QuickRecal score (lowest first)
             scored_memories = [(mem.id, mem.get_effective_quickrecal(self.config['time_decay_rate'])) for mem in self._memories.values()]
             scored_memories.sort(key=lambda x: x[1])

             pruned_ids = []
             for mem_id, score in scored_memories[:num_to_prune]:
                 if score < self.config['min_quickrecal_for_ltm']:
                      pruned_ids.append(mem_id)

             if not pruned_ids:
                  logger.info("SynthiansMemoryCore", "No memories met pruning criteria.")
                  return

             logger.info(f"Identified {len(pruned_ids)} memories for pruning.")
             # Perform deletion outside the main iteration
             pruned_count = 0
             ids_to_remove_from_index = []
             for mem_id in pruned_ids:
                 if mem_id in self._memories:
                      del self._memories[mem_id]
                      ids_to_remove_from_index.append(mem_id) # Mark for vector index removal

                      # Also remove from assemblies mapping
                      if mem_id in self.memory_to_assemblies:
                           for asm_id in list(self.memory_to_assemblies[mem_id]): # Iterate over copy
                                if asm_id in self.assemblies:
                                     self.assemblies[asm_id].memories.discard(mem_id)
                                     self._dirty_memories.add(asm_id) # Mark assembly dirty
                           del self.memory_to_assemblies[mem_id]

                      # Delete from persistence - call async method
                      deleted_persistence = await self.persistence.delete_memory(mem_id)
                      if deleted_persistence:
                          pruned_count += 1
                      else:
                           logger.warning(f"Failed to delete memory {mem_id} from persistence.")

             # Remove from vector index if needed
             if ids_to_remove_from_index and self.vector_index is not None:
                  for mem_id in ids_to_remove_from_index:
                       try:
                            removed = await self.vector_index.remove_vector_async(mem_id)
                            if not removed:
                                 logger.warning(f"Could not remove vector for {mem_id} during pruning (not found in index).")
                       except Exception as e:
                            logger.error(f"Error removing vector for {mem_id} during pruning: {e}")

             logger.info("SynthiansMemoryCore", f"Pruned {pruned_count} memories.")

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
                      return {"memories": [{"id": m.get("id"), "content": m.get("content"), "score": m.get("final_score", m.get("relevance_score", m.get("similarity"))) } for m in response_data["memories"]]}
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
                      await self.provide_feedback(memory_id, similarity_score, was_relevant)
                      return {"success": True, "message": "Feedback recorded."}
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
            logger.error("SynthiansMemoryCore", f"Error handling tool call {tool_name}", {"error": str(e)})
            return {"success": False, "error": str(e)}

    # --- Helper & Placeholder Methods ---

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
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
                memory.access_count += 1
                memory.last_access_time = datetime.now(timezone.utc) # Convert to datetime
                return memory
                
            # Not in cache, check if it's in the index and try to load it
            if memory_id in self.persistence.memory_index:
                logger.debug("SynthiansMemoryCore", f"Memory {memory_id} not in cache, loading from persistence...")
                memory = await self.persistence.load_memory(memory_id)
                if memory:
                    # Add to cache
                    self._memories[memory_id] = memory
                    memory.access_count += 1
                    memory.last_access_time = datetime.now(timezone.utc) # Convert to datetime
                    
                    # If this is our first time seeing this memory and we have a vector index,
                    # add it to the index if it has a valid embedding
                    if memory.embedding is not None and self.vector_index is not None:
                        self.vector_index.add(memory_id, memory.embedding)
                        logger.debug("SynthiansMemoryCore", f"Added memory {memory_id} to vector index on first load.")
                    
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
                # Look up the memory first
                memory = self._get_memory_by_id(memory_id)
                if memory is None:
                    logger.warning(f"Cannot update non-existent memory {memory_id}")
                    return False
                
                # Extract metadata updates if present
                metadata_to_update = updates.pop('metadata', None)
                
                # Track if quickrecal score is updated for timestamp update
                score_updated = False
                
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
                    elif hasattr(memory, key):
                         setattr(memory, key, value) # Update other direct attributes
                    else:
                        logger.warning(f"Unknown/invalid field '{key}' in memory update")

                # Apply metadata updates after other fields have been processed
                if metadata_to_update:
                    if memory.metadata is None:
                        memory.metadata = {}
                    # Use deep update to properly handle nested dictionaries
                    deep_update(memory.metadata, metadata_to_update)

                # Update quickrecal timestamp ONLY if the score actually changed in THIS update call
                if score_updated:
                    if memory.metadata is None: memory.metadata = {}
                    memory.metadata['quickrecal_updated_at'] = datetime.now(timezone.utc).isoformat()
                    logger.debug(f"quickrecal_updated_at set for memory {memory_id}")

                # Update the vector index with the memory's embedding
                vector_update_success = True  # Assume success initially
                if memory.embedding is not None and self.vector_index is not None:
                    logger.debug(f"Updating vector index for memory {memory_id}")
                    try:
                        updated_index = await self.vector_index.update_entry_async(memory_id, memory.embedding)
                        if not updated_index:
                            logger.error(f"CRITICAL: Failed to update vector index for memory {memory_id} during memory update.")
                            vector_update_success = False  # Mark failure
                    except Exception as e:
                        logger.error(f"CRITICAL: Exception updating vector index for {memory_id}: {e}", exc_info=True)
                        vector_update_success = False

                # Mark as dirty for persistence
                self._dirty_memories.add(memory_id)
                logger.debug(f"Memory {memory_id} updated in memory (marked dirty)")
                
                # Return success based on vector index update
                if not vector_update_success:
                    logger.warning(f"Update for memory {memory_id} returning False due to vector index update failure.")
                    return False

                logger.info(f"Updated memory {memory_id} with {len(updates)} fields (marked dirty for persistence)")
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
                if '.' in key:
                    path_parts = key.split('.')
                    current_obj = metadata
                    # Navigate through the nested structure
                    for part in path_parts[:-1]:
                        if part not in current_obj or not isinstance(current_obj[part], dict):
                            matches_all = False
                            break
                        current_obj = current_obj[part]
                    
                    # Check the final value
                    if matches_all and (path_parts[-1] not in current_obj or current_obj[path_parts[-1]] != value):
                        matches_all = False
                # Simple direct key match        
                elif key not in metadata or metadata[key] != value:
                    matches_all = False
                    break
                    
            if matches_all:
                filtered_results.append(candidate)
                logger.debug(f"Candidate {candidate.get('id')} matched all metadata criteria")
            else:
                logger.debug(f"Candidate {candidate.get('id')} failed metadata criteria")
                
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
            model = SentenceTransformer(model_name)

            logger.info("SynthiansMemoryCore", f"Using embedding model {model_name}")
            # Run encode in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            embedding_list = await loop.run_in_executor(None, lambda: model.encode([text], convert_to_tensor=False))
            if embedding_list is None or len(embedding_list) == 0:
                raise ValueError("Embedding model returned empty result")

            embedding = embedding_list[0]
            return self.geometry_manager._normalize(np.array(embedding, dtype=np.float32))
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error generating embedding: {str(e)}")

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

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        # Run get_stats synchronously as it doesn't involve async operations directly
        persistence_stats = self.persistence.get_stats()
        quick_recal_stats = self.quick_recal.get_stats()
        threshold_stats = self.threshold_calibrator.get_statistics() if self.threshold_calibrator else {}
        vector_index_stats = self.vector_index.get_stats() if hasattr(self.vector_index, 'get_stats') else {"count": self.vector_index.count(), "id_mappings": len(self.vector_index.id_to_index)}


        return {
            "core_stats": {
                "total_memories": len(self._memories),
                "total_assemblies": len(self.assemblies),
                "dirty_memories": len(self._dirty_memories),
                "initialized": self._initialized,
            },
            "persistence_stats": persistence_stats,
            "quick_recal_stats": quick_recal_stats,
            "threshold_stats": threshold_stats,
            "vector_index_stats": vector_index_stats
        }

    def process_memory_sync(self, content: str, embedding: Optional[np.ndarray] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           emotion_data: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], None]:
        """
Process a new memory synchronously without using asyncio.run().

This is a synchronous version of process_new_memory that avoids potential asyncio.run() issues.

Args:
    content: The text content of the memory
    embedding: Vector representation of the content (optional)
    metadata: Base metadata for the memory entry (optional)
    emotion_data: Pre-computed emotion analysis results (optional)
        """
        try:
            logger.info("SynthiansMemoryCore", "Processing memory synchronously")

            # Create a new memory entry
            memory_id = f"mem_{uuid.uuid4().hex[:12]}"  # More consistent ID format
            timestamp = metadata.get('timestamp', time.time()) if metadata else time.time()

            # Ensure metadata is a dictionary
            metadata = metadata or {}
            metadata['timestamp'] = timestamp

            # Use provided embedding or generate from content
            if embedding is None:
                logger.warning("SynthiansMemoryCore", "Sync processing requires embedding, using zeros")
                embedding = np.zeros(self.config.get('embedding_dim', 768), dtype=np.float32)

            # Validate/normalize embedding
            validated_embedding = self.geometry_manager._validate_vector(embedding, "Input Embedding")
            if validated_embedding is None:
                 logger.error("SynthiansMemoryCore", "Invalid embedding provided (sync).")
                 return None
            aligned_embedding, _ = self.geometry_manager._align_vectors(validated_embedding, np.zeros(self.config['embedding_dim']))
            normalized_embedding = self.geometry_manager._normalize(aligned_embedding)

            # If emotion_data is not provided but we have an emotion analyzer, try to generate it
            if emotion_data is None and self.emotional_analyzer is not None:
                # Needs a sync version of analyze
                logger.warning("SynthiansMemoryCore", "Sync emotion analysis not implemented")

            # Enhance metadata using the MetadataSynthesizer
            enhanced_metadata = metadata
            if self.metadata_synthesizer is not None:
                try:
                    enhanced_metadata = self.metadata_synthesizer.synthesize_sync(
                        content=content,
                        embedding=normalized_embedding,
                        base_metadata=metadata,
                        emotion_data=emotion_data
                    )
                    logger.info("SynthiansMemoryCore", f"Enhanced metadata for memory {memory_id} (sync)")
                except Exception as e:
                    logger.error("SynthiansMemoryCore", f"Error enhancing metadata (sync): {str(e)}")

            # Calculate QuickRecal score
            quickrecal_score = 0.5  # Default value
            if self.quick_recal is not None:
                try:
                    context = {'text': content, 'timestamp': timestamp}
                    if enhanced_metadata: context.update(enhanced_metadata)
                    quickrecal_score = self.quick_recal.calculate_sync(normalized_embedding, context=context)
                    logger.info("SynthiansMemoryCore", f"Calculated QuickRecal score (sync): {quickrecal_score}")
                except Exception as e:
                    logger.error("SynthiansMemoryCore", f"Error calculating QuickRecal score (sync): {str(e)}")

            # Create memory object (using MemoryEntry directly)
            memory_entry_obj = MemoryEntry(
                id=memory_id,
                content=content,
                embedding=normalized_embedding,
                metadata=enhanced_metadata,
                quickrecal_score=quickrecal_score,
                timestamp=datetime.fromtimestamp(timestamp, timezone.utc) # Convert to datetime
            )

            # Add memory ID to metadata for easier access
            memory_entry_obj.metadata["uuid"] = memory_entry_obj.id

            # Store memory directly
            self._memories[memory_id] = memory_entry_obj
            self._dirty_memories.add(memory_id) # Mark as dirty for next persistence cycle
            logger.info("SynthiansMemoryCore", f"Memory {memory_id} stored in memory (sync)")

            # Persistence is handled by the background loop

            # Return a dictionary representation
            return memory_entry_obj.to_dict()
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error processing memory synchronously: {str(e)}")
            return None

    async def check_index_integrity(self) -> Dict[str, Any]:
        """Check the integrity of the vector index and return diagnostic information.
        
        This method checks if the FAISS index and ID-to-index mapping are consistent.
        
        Returns:
            Dict with diagnostic information about the index integrity
        """
        if not self._initialized: await self.initialize()
        
        async with self._lock: # We need the lock to ensure thread safety
            is_consistent, diagnostics = self.vector_index.verify_index_integrity()
            
            return {
                "success": True,
                "is_consistent": is_consistent,
                "diagnostics": diagnostics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def repair_index(self, repair_type: str = "auto") -> Dict[str, Any]:
        """Attempt to repair integrity issues with the vector index.
        
        Args:
            repair_type: The type of repair to perform.
                - "auto": Automatically determine the best repair strategy
                - "recreate_mapping": Recreate the ID-to-index mapping from scratch
                - "rebuild": Completely rebuild the index (not fully implemented)
                
        Returns:
            Dict with repair status and diagnostics
        """
        if not self._initialized: await self.initialize()
        
        async with self._lock:
            logger.info("SynthiansMemoryCore", f"Starting index repair of type: {repair_type}")
            
            # Check initial integrity state
            is_consistent_before, diagnostics_before = self.vector_index.verify_index_integrity()
            
            # If already consistent and not a forced rebuild, we can consider this a success
            if is_consistent_before and repair_type != "rebuild":
                logger.info("SynthiansMemoryCore", "Index is already consistent, no repair needed.")
                return {
                    "success": True,
                    "message": "Index is already consistent, no repair needed.",
                    "diagnostics_before": diagnostics_before,
                    "diagnostics_after": diagnostics_before,
                    "is_consistent": True
                }
            
            # Check current implementation and migrate if needed
            is_index_id_map = hasattr(self.vector_index.index, 'id_map')
            if not is_index_id_map:
                logger.info("Migrating vector index to use IndexIDMap for improved ID management")
                success = self.vector_index.migrate_to_idmap()
                if success:
                    logger.info("Successfully migrated vector index to IndexIDMap")
                else:
                    logger.warning("Failed to migrate vector index to IndexIDMap. Some features may not work correctly.")
            else:
                logger.info("Vector index is already using IndexIDMap")
            
            # Determine repair strategy
            if repair_type == "auto":
                # Choose the best repair strategy based on diagnostics
                faiss_count = self.vector_index.count()
                id_mapping_count = len(self.vector_index.id_to_index)
                
                if id_mapping_count == 0 and faiss_count > 0:
                    repair_type = "recreate_mapping"
                    logger.info("SynthiansMemoryCore", "Auto-selected 'recreate_mapping' repair strategy")
                elif id_mapping_count > faiss_count:
                    # Prune excess mappings
                    repair_type = "recreate_mapping"
                    logger.info("SynthiansMemoryCore", "Auto-selected 'recreate_mapping' to handle excess mappings")
                else:
                    # In other cases, we don't have a good automated solution yet
                    repair_type = "recreate_mapping"  # Default to recreate_mapping for now
                    logger.warning("SynthiansMemoryCore", "No optimal repair strategy determined, defaulting to 'recreate_mapping'")
            
            # Execute repair
            if repair_type == "recreate_mapping":
                success = self.vector_index.recreate_mapping()
            elif repair_type == "rebuild":
                logger.warning("SynthiansMemoryCore", "Full rebuild requires original embeddings which aren't stored. Falling back to recreate_mapping.")
                success = self.vector_index.recreate_mapping()
            else:
                logger.error("SynthiansMemoryCore", f"Unsupported repair_type: {repair_type}")
                success = False
            
            # Check integrity after repair
            is_consistent_after, diagnostics_after = self.vector_index.verify_index_integrity()
            
            # Determine overall success: either repair succeeded or the index is now consistent
            overall_success = success or is_consistent_after
            
            if overall_success:
                logger.info("SynthiansMemoryCore", f"Index repair of type '{repair_type}' completed successfully. Consistency: {is_consistent_after}")
            else:
                logger.error("SynthiansMemoryCore", f"Index repair of type '{repair_type}' failed. Consistency: {is_consistent_after}")
                
            return {
                "success": overall_success,
                "repair_type": repair_type,
                "diagnostics_before": diagnostics_before,
                "diagnostics_after": diagnostics_after,
                "is_consistent": is_consistent_after
            }