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
from datetime import timezone, datetime # Ensure datetime is imported directly
import copy
import traceback # Import traceback for detailed error logging

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
        else:
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
        self.vector_index = MemoryVectorIndex({
            'embedding_dim': self.config['embedding_dim'],
            'storage_path': self.config['storage_path'],
            'index_type': self.config['vector_index_type']
        })

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
        """Load persisted state and start background tasks."""
        if self._initialized: return True
        logger.info("SynthiansMemoryCore", "Starting initialization...")
        async with self._lock:
            # Initialize persistence first to ensure memory index is loaded
            await self.persistence.initialize()
            logger.info("SynthiansMemoryCore", "Persistence layer initialized.")

            # Load memories and assemblies from persistence
            loaded_memories = await self.persistence.load_all()
            for mem in loaded_memories:
                self._memories[mem.id] = mem
            logger.info("SynthiansMemoryCore", f"Loaded {len(self._memories)} memories from persistence.")

            # Load assemblies and memory_to_assemblies mapping
            assembly_list = await self.persistence.list_assemblies()
            loaded_assemblies_count = 0

            # Initialize memory_to_assemblies mapping
            self.memory_to_assemblies = {memory_id: set() for memory_id in self._memories.keys()}

            # Load each assembly
            for assembly_info in assembly_list:
                assembly_id = assembly_info.get("id")
                if assembly_id:
                    assembly = await self.persistence.load_assembly(assembly_id, self.geometry_manager)
                    if assembly:
                        self.assemblies[assembly_id] = assembly
                        loaded_assemblies_count += 1

                        # Update memory_to_assemblies mapping
                        for memory_id in assembly.memories:
                            if memory_id in self.memory_to_assemblies:
                                self.memory_to_assemblies[memory_id].add(assembly_id)
                            else:
                                # Create mapping entry if memory not in cache
                                self.memory_to_assemblies[memory_id] = {assembly_id}

            logger.info("SynthiansMemoryCore", f"Loaded {loaded_assemblies_count} assemblies from persistence.")

            # Load the vector index
            index_loaded = self.vector_index.load()

            # If index wasn't found, build it from loaded memories
            if not index_loaded and self._memories:
                logger.info("SynthiansMemoryCore", "Building vector index from loaded memories...")
                for mem_id, memory in self._memories.items():
                    if memory.embedding is not None:
                        self.vector_index.add(mem_id, memory.embedding)
                # Save the newly built index
                self.vector_index.save()
                logger.info("SynthiansMemoryCore", f"Built and saved vector index with {len(self._memories)} entries")

            # Start background tasks only if intervals are > 0
            if self.config['persistence_interval'] > 0:
                 self._background_tasks.append(asyncio.create_task(self._persistence_loop(), name=f"PersistenceLoop_{id(self)}"))
            else:
                logger.warning("SynthiansMemoryCore", "Persistence loop disabled (interval <= 0)")

            if self.config['decay_interval'] > 0 or self.config['prune_check_interval'] > 0:
                self._background_tasks.append(asyncio.create_task(self._decay_and_pruning_loop(), name=f"DecayPruningLoop_{id(self)}"))
            else:
                logger.warning("SynthiansMemoryCore", "Decay/Pruning loop disabled (intervals <= 0)")

            self._initialized = True
            logger.info("SynthiansMemoryCore", "Initialization complete. Background tasks started.")
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
        self.vector_index.add(memory.id, normalized_embedding)
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

        query_embedding = None
        try:
            # Generate embedding for the query if necessary
            if query:
                query_embedding = await self.generate_embedding(query)
                if query_embedding is None:
                     logger.error("SynthiansMemoryCore", "Failed to generate query embedding.")
                     return {"success": False, "memories": [], "error": "Failed to generate query embedding"}
                logger.debug("SynthiansMemoryCore", "Query embedding generated")

            # Get the current threshold
            current_threshold = threshold
            if current_threshold is None and self.threshold_calibrator is not None:
                current_threshold = self.threshold_calibrator.get_current_threshold()
                logger.debug(f"Using calibrated threshold: {current_threshold:.4f}")
            elif current_threshold is None:
                current_threshold = self.config['initial_retrieval_threshold'] # Use default if None provided and no calibrator
                logger.debug(f"Using default initial threshold: {current_threshold:.4f}")
            else:
                logger.debug(f"Using explicit threshold: {current_threshold:.4f}")

            # Perform the retrieval using candidate generation
            candidates = await self._get_candidate_memories(query_embedding, top_k * 2) # Get more candidates for filtering

            # Score and filter candidates
            if not candidates:
                logger.info("SynthiansMemoryCore", "No candidate memories found.")
                return {"success": True, "memories": [], "error": None}

            scored_candidates = []
            for memory_dict in candidates:
                memory_embedding = memory_dict.get("embedding")
                if memory_embedding is not None and query_embedding is not None:
                    # Ensure embedding is a list before converting to array
                    if isinstance(memory_embedding, list):
                        memory_embedding_np = np.array(memory_embedding, dtype=np.float32)
                    else:
                        # Skip if embedding is not in expected format
                        logger.warning(f"Skipping memory {memory_dict.get('id')} due to unexpected embedding format: {type(memory_embedding)}")
                        continue

                    similarity = self.geometry_manager.calculate_similarity(query_embedding, memory_embedding_np)
                    memory_dict["similarity"] = similarity # Use 'similarity' to match client expectations
                    memory_dict["relevance_score"] = similarity # Keep internal name too
                    scored_candidates.append(memory_dict)
                else:
                    logger.debug(f"Skipping candidate {memory_dict.get('id')} due to missing embedding.")

            # Sort by similarity score (descending)
            scored_candidates.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

            # Apply threshold filtering
            filtered_candidates = [c for c in scored_candidates if c.get("similarity", 0.0) >= current_threshold]

            # Apply emotional gating if requested
            if user_emotion and self.emotional_gating:
                # Construct user_emotion dict for gating service
                user_emotion_dict = {"dominant_emotion": user_emotion} # Simulate expected input for gating service
                filtered_candidates = await self.emotional_gating.gate_memories(
                    filtered_candidates, user_emotion_dict
                )
                # Re-sort based on 'final_score' if gating was applied
                filtered_candidates.sort(key=lambda x: x.get("final_score", x.get("similarity", 0.0)), reverse=True)


            # Apply metadata filtering if requested (basic implementation)
            if metadata_filter:
                 filtered_candidates = self._filter_by_metadata(filtered_candidates, metadata_filter)

            # Return top_k results
            final_memories = filtered_candidates[:top_k]

            retrieval_time = (time.time() - start_time) * 1000
            logger.info("SynthiansMemoryCore", f"Retrieved {len(final_memories)} memories", {
                "top_k": top_k, "threshold": current_threshold, "user_emotion": user_emotion, "time_ms": retrieval_time
            })
            return {"success": True, "memories": final_memories, "error": None}

        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error in retrieve_memories: {str(e)}")
            logger.error(traceback.format_exc())
            return {"success": False, "memories": [], "error": str(e)}

    async def _get_candidate_memories(self, query_embedding: Optional[np.ndarray], limit: int) -> List[Dict[str, Any]]:
        """Retrieve candidate memories using assembly activation and direct vector search."""
        if query_embedding is None:
            logger.warning("SynthiansMemoryCore", "_get_candidate_memories called with no query embedding.")
            return []

        assembly_candidates = set()
        direct_candidates = set()

        # 1. Assembly Activation
        activated_assemblies = await self._activate_assemblies(query_embedding)
        for assembly, activation_score in activated_assemblies[:5]: # Consider top 5 assemblies
            if activation_score > 0.2: # Lower activation threshold
                assembly_candidates.update(assembly.memories)

        # 2. Direct Vector Search using FAISS Index
        search_threshold = 0.05 # Use a low threshold to get enough candidates
        search_results = self.vector_index.search(query_embedding, k=limit, threshold=search_threshold)
        for memory_id, _ in search_results:
            direct_candidates.add(memory_id)

        # Combine candidates
        all_candidate_ids = assembly_candidates.union(direct_candidates)
        logger.debug(f"Found {len(all_candidate_ids)} total candidate IDs.")

        # Fetch MemoryEntry objects as dictionaries
        final_candidates = []
        async with self._lock:
             for mem_id in all_candidate_ids:
                 if mem_id in self._memories:
                      # Make sure to convert memory to dict before returning
                      final_candidates.append(self._memories[mem_id].to_dict())

        return final_candidates[:limit * 2] # Return more initially for scoring/filtering


    async def _activate_assemblies(self, query_embedding: np.ndarray) -> List[Tuple[MemoryAssembly, float]]:
        """Find and activate assemblies based on query similarity."""
        activated = []
        async with self._lock: # Accessing shared self.assemblies
            for assembly_id, assembly in self.assemblies.items():
                 similarity = assembly.get_similarity(query_embedding)
                 if similarity >= self.config['assembly_threshold'] * 0.8: # Lower threshold for activation
                      assembly.activate(similarity)
                      activated.append((assembly, similarity))
        # Sort by activation score
        activated.sort(key=lambda x: x[1], reverse=True)
        return activated

    async def _update_assemblies(self, memory: MemoryEntry):
        """Find or create assemblies for a new memory."""
        if memory.embedding is None: return

        suitable_assemblies = []
        best_similarity = 0.0
        best_assembly_id = None

        async with self._lock: # Accessing shared self.assemblies
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

             # Remove from vector index if needed (Requires remove method)
             # if ids_to_remove_from_index:
             #     # removed_count = self.vector_index.remove(ids_to_remove_from_index)
             #     logger.warning(f"Vector index remove not implemented. Cannot remove {len(ids_to_remove_from_index)} pruned IDs.")


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

    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a memory entry with provided updates.
        Performs a deep merge for metadata updates.

        Args:
            memory_id: ID of the memory to update
            updates: Dictionary of fields to update and their new values

        Returns:
            True if the update succeeded, False otherwise
        """
        try:
            logger.debug(f"Updating memory {memory_id} - acquiring lock")

            # Step 1: Get and update the memory while holding the lock
            try:
                async with asyncio.timeout(5):  # 5 second timeout
                    async with self._lock: # Use the actual async lock
                        logger.debug(f"Lock acquired for memory {memory_id}")
                        # Get the memory (use synchronous version since we already hold the lock)
                        memory = self.get_memory_by_id(memory_id)
                        if not memory:
                            logger.warning(f"Cannot update memory {memory_id}: Not found")
                            return False

                        # Store metadata update separately to apply after all direct attributes
                        metadata_to_update = None
                        score_updated = False

                        # Update the memory fields
                        for key, value in updates.items():
                            if key == "metadata" and isinstance(value, dict):
                                # Store metadata updates to apply them after direct attribute updates
                                metadata_to_update = value
                                continue # Process metadata last

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

                        # Mark as dirty for persistence
                        self._dirty_memories.add(memory_id)
                        logger.debug(f"Memory {memory_id} updated in memory (marked dirty), releasing lock")
            except asyncio.TimeoutError:
                logger.error(f"Timeout while acquiring or using lock for memory {memory_id}")
                return False

            # The memory is marked as dirty, the persistence loop will handle saving it.
            logger.info(f"Updated memory {memory_id} with {len(updates)} fields (marked dirty for persistence)")
            return True

        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error updating memory {memory_id}: {str(e)}", exc_info=True)
            return False


    def _filter_by_metadata(self, memories: List[Dict], metadata_filter: Dict) -> List[Dict]:
        """Filter memories based on metadata key-value pairs."""
        if not metadata_filter: return memories
        filtered = []
        for mem in memories:
            metadata = mem.get("metadata", {})
            match = True
            for key, value in metadata_filter.items():
                 if metadata.get(key) != value:
                      match = False
                      break
            if match:
                 filtered.append(mem)
        return filtered

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