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

            # Pass momentum buffer to calculator if needed
            # self.quick_recal.set_external_momentum(...)

            # Start background tasks
            self._background_tasks.append(asyncio.create_task(self._persistence_loop()))
            self._background_tasks.append(asyncio.create_task(self._decay_and_pruning_loop()))

            self._initialized = True
            logger.info("SynthiansMemoryCore", "Initialization complete. Background tasks started.")
        return True

    async def shutdown(self):
        """Gracefully shut down the memory core."""
        logger.info("SynthiansMemoryCore", "Shutting down...")
        self._shutdown_signal.set()
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        # Wait for tasks to finish cancellation
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        # Final persistence flush
        await self._persist_all_managed_memories()
        logger.info("SynthiansMemoryCore", "Shutdown complete.")

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
                "memory_id": memory.id,
                "quickrecal_score": memory.quickrecal_score,
                "metadata": memory.metadata
            }
        else:
            return {
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
            # embedding = await self._generate_embedding(content) # Assumed external
            logger.warning("SynthiansMemoryCore", "process_new_memory called without embedding, skipping.")
            return None # Cannot proceed without embedding
        
        # Handle common case where embedding is wrongly passed as a dict
        if isinstance(embedding, dict):
            logger.warning("SynthiansMemoryCore", f"Received embedding as dict type, attempting to extract vector")
            try:
                # Try common dict formats seen in the wild
                if 'embedding' in embedding and isinstance(embedding['embedding'], (list, np.ndarray)):
                    embedding = embedding['embedding']
                    logger.info("SynthiansMemoryCore", "Successfully extracted embedding from dict['embedding']") 
                elif 'vector' in embedding and isinstance(embedding['vector'], (list, np.ndarray)):
                    embedding = embedding['vector']
                    logger.info("SynthiansMemoryCore", "Successfully extracted embedding from dict['vector']")
                elif 'value' in embedding and isinstance(embedding['value'], (list, np.ndarray)):
                    embedding = embedding['value']
                    logger.info("SynthiansMemoryCore", "Successfully extracted embedding from dict['value']")
                else:
                    logger.error("SynthiansMemoryCore", f"Could not extract embedding from dict: {list(embedding.keys())[:5]}")
                    return None
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
            metadata["emotional_context"] = emotional_context
        else:
            logger.debug("SynthiansMemoryCore", "Using precomputed emotional context from metadata")

        # 4. Generate Hyperbolic Embedding (if enabled)
        hyperbolic_embedding = None
        if self.geometry_manager.config['geometry_type'] == GeometryType.HYPERBOLIC:
            hyperbolic_embedding = self.geometry_manager._to_hyperbolic(normalized_embedding)

        # 5. Run Metadata Synthesizer
        metadata = await self.metadata_synthesizer.synthesize(
            content=content,
            embedding=normalized_embedding,
            base_metadata=metadata,
            emotion_data=emotional_context
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

        # 7. Store in memory and persistence
        async with self._lock:
            self._memories[memory.id] = memory
            stored = await self.persistence.save_memory(memory)
            if stored:
                 logger.info("SynthiansMemoryCore", f"Stored new memory {memory.id}", {"quickrecal": quickrecal_score})
            else:
                 # Rollback if persistence failed
                 del self._memories[memory.id]
                 logger.error("SynthiansMemoryCore", f"Failed to persist memory {memory.id}, rolling back.")
                 return None

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
        user_emotion: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        search_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve memories based on query relevance.
        """
        
        query_embedding = None
        try:
            # Generate embedding for the query if necessary
            if query:
                query_embedding = await self.generate_embedding(query)
                logger.debug("SynthiansMemoryCore", "Query embedding generated", {
                    "query": query,
                    "has_embedding": query_embedding is not None,
                })
            
            # Get the current threshold (use provided or default)
            current_threshold = threshold
            if current_threshold is None and self.threshold_calibrator is not None:
                current_threshold = self.threshold_calibrator.get_current_threshold()
                logger.debug("SynthiansMemoryCore", f"Using calibrated threshold: {current_threshold:.4f}")
            else:
                logger.debug("SynthiansMemoryCore", f"Using explicit threshold: {current_threshold}")
            
            logger.debug("SynthiansMemoryCore", "Memory retrieval parameters", {
                "query": query,
                "has_embedding": query_embedding is not None,
                "threshold": current_threshold,
                "user_emotion": user_emotion,
                "top_k": top_k,
                "metadata_filter": metadata_filter
            })
            
            # Perform the retrieval
            candidates = await self._get_candidate_memories(query_embedding, top_k * 2)
            logger.debug("SynthiansMemoryCore", f"Found {len(candidates)} candidate memories")
            
            # Score and filter candidates
            if candidates:
                scored_candidates = []
                for memory_dict in candidates:
                    memory_embedding = memory_dict.get("embedding")
                    if memory_embedding is not None and query_embedding is not None:
                        # Calculate similarity score
                        similarity = self.geometry_manager.calculate_similarity(query_embedding, memory_embedding)
                        memory_dict["similarity"] = similarity
                        scored_candidates.append(memory_dict)
                    else:
                        # Skip memories without embeddings
                        continue
                
                # Sort by similarity score (descending)
                scored_candidates.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                
                # Log similarity scores for debugging
                if scored_candidates:
                    score_info = [{
                        "id": cand.get("id", "")[-8:],  # Last 8 chars of ID
                        "score": round(cand.get("similarity", 0), 4),
                        "metadata": {k: v for k, v in cand.get("metadata", {}).items() 
                                     if k in ["source", "test_type"]}
                    } for cand in scored_candidates[:5]]
                    logger.debug("SynthiansMemoryCore", "Top candidate scores", score_info)
                
                # Apply threshold filtering
                if current_threshold is not None:
                    before_threshold = len(scored_candidates)
                    scored_candidates = [c for c in scored_candidates 
                                       if c.get("similarity", 0) >= current_threshold]
                    logger.debug("SynthiansMemoryCore", "After threshold filtering", {
                        "before": before_threshold,
                        "after": len(scored_candidates),
                        "threshold": current_threshold
                    })
                
                # Apply emotional gating if requested
                if user_emotion and hasattr(self, 'emotional_gating') and self.emotional_gating is not None:
                    before_gating = len(scored_candidates)
                    scored_candidates = await self.emotional_gating.gate_memories(
                        scored_candidates, user_emotion
                    )
                    logger.debug("SynthiansMemoryCore", "After emotional gating", {
                        "before": before_gating,
                        "after": len(scored_candidates),
                        "user_emotion": user_emotion
                    })
                
                # Apply metadata filtering if requested
                if metadata_filter and len(scored_candidates) > 0:
                    before_metadata = len(scored_candidates)
                    scored_candidates = self._filter_by_metadata(scored_candidates, metadata_filter)
                    logger.debug("SynthiansMemoryCore", "After metadata filtering", {
                        "before": before_metadata,
                        "after": len(scored_candidates),
                        "filter": metadata_filter
                    })
                
                # Format and return results (taking top_k)
                top_candidates = scored_candidates[:top_k] if len(scored_candidates) > top_k else scored_candidates
                
                result = {
                    "success": True,
                    "memories": top_candidates,
                    "error": None
                }
                
                return result
            
            # Fall through if no candidates found
            logger.warning("SynthiansMemoryCore", "No candidate memories found for query", {"query": query})
            return {"success": True, "memories": [], "error": None}
            
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error in retrieve_memories: {str(e)}")
            import traceback
            logger.error("SynthiansMemoryCore", traceback.format_exc())
            return {"success": False, "memories": [], "error": str(e)}

    async def _get_candidate_memories(self, query_embedding: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """Retrieve candidate memories using assembly activation and direct vector search."""
        assembly_candidates = set()
        direct_candidates = set()

        # 1. Assembly Activation
        activated_assemblies = await self._activate_assemblies(query_embedding)
        for assembly, activation_score in activated_assemblies[:5]: # Consider top 5 assemblies
            # Lower activation threshold from 0.3 to 0.2 for better retrieval
            if activation_score > 0.2: # Lower activation threshold
                assembly_candidates.update(assembly.memories)
                logger.debug("SynthiansMemoryCore", f"Assembly activated: {len(assembly.memories)} memories, score: {activation_score:.4f}")

        # 2. Direct Vector Search using FAISS Index
        # Validate query embedding for NaN/Inf values
        if query_embedding is not None and (np.isnan(query_embedding).any() or np.isinf(query_embedding).any()):
            logger.warning("SynthiansMemoryCore", "Query embedding has NaN or Inf values")
            # Replace with zeros or return empty list
            return []
            
        # Perform vector search using the FAISS index - USE A MUCH LOWER THRESHOLD
        # Lower threshold from 0.3 to 0.05 for better recall sensitivity
        search_threshold = 0.05  # Significantly lowered threshold
        search_results = self.vector_index.search(query_embedding, k=limit, threshold=search_threshold)
        
        logger.info("SynthiansMemoryCore", f"Vector search with threshold {search_threshold} returned {len(search_results)} results")
        
        # Add direct search candidates from vector index
        for memory_id, similarity in search_results:
            direct_candidates.add(memory_id)
            logger.info("SynthiansMemoryCore", f"Memory {memory_id} similarity: {similarity:.4f} (from vector index)")

        # Combine candidates
        all_candidate_ids = assembly_candidates.union(direct_candidates)
        
        logger.info("SynthiansMemoryCore", f"Found {len(all_candidate_ids)} total candidate memories")

        # Fetch MemoryEntry objects
        final_candidates = []
        async with self._lock:
             for mem_id in all_candidate_ids:
                 if mem_id in self._memories:
                      final_candidates.append(self._memories[mem_id].to_dict())

        return final_candidates[:limit] # Limit the final candidate list

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
        while not self._shutdown_signal.is_set():
            await asyncio.sleep(self.config['persistence_interval'])
            logger.debug("SynthiansMemoryCore", "Running periodic persistence.")
            await self._persist_all_managed_memories()

    async def _decay_and_pruning_loop(self):
        """Periodically decay memory scores and prune old/irrelevant memories."""
        while not self._shutdown_signal.is_set():
            # Decay check interval
            await asyncio.sleep(self.config['decay_interval'])
            logger.info("SynthiansMemoryCore", "Running memory decay check.")
            await self._apply_decay()

            # Pruning check interval (more frequent)
            await asyncio.sleep(self.config['prune_check_interval'] - self.config['decay_interval'] % self.config['prune_check_interval'])
            logger.debug("SynthiansMemoryCore", "Running pruning check.")
            await self._prune_if_needed()


    async def _persist_all_managed_memories(self):
        """Persist all memories currently managed (in self._memories)."""
        try:
            if not self._memories:
                logger.info("SynthiansMemoryCore", "No memories to persist.")
                return
                
            async with self._lock:
                count = 0
                for memory_id, memory in self._memories.items():
                    stored = await self.persistence.save_memory(memory)
                    if stored:
                        count += 1
                logger.info("SynthiansMemoryCore", f"Persisted {count} memories.")
                
                # Save the vector index to ensure ID mappings persist
                if self.vector_index.count() > 0:
                    vector_index_saved = self.vector_index.save()
                    logger.info("SynthiansMemoryCore", f"Vector index saved: {vector_index_saved} with {self.vector_index.count()} vectors and {len(self.vector_index.id_to_index)} id mappings")
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error persisting memories: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    async def _apply_decay(self):
        """Apply decay to QuickRecal scores."""
        async with self._lock:
             modified_ids = []
             for memory_id, memory in self._memories.items():
                 effective_score = memory.get_effective_quickrecal()
                 # Store the effective score back, but don't overwrite original quickrecal
                 memory.metadata['effective_quickrecal'] = effective_score
                 modified_ids.append(memory_id) # Mark for potential persistence update

             # Persist modified memories (optional, could be done in main persistence loop)
             # for mem_id in modified_ids:
             #     await self.persistence.save_memory(self._memories[mem_id])
             logger.info("SynthiansMemoryCore", f"Applied decay to {len(modified_ids)} memories.")


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
             scored_memories = [(mem.id, mem.get_effective_quickrecal()) for mem in self._memories.values()]
             scored_memories.sort(key=lambda x: x[1])

             pruned_count = 0
             for mem_id, score in scored_memories[:num_to_prune]:
                 if score < self.config['min_quickrecal_for_ltm']:
                      if mem_id in self._memories:
                           del self._memories[mem_id]
                           # Also remove from assemblies mapping
                           if mem_id in self.memory_to_assemblies:
                                for asm_id in self.memory_to_assemblies[mem_id]:
                                     if asm_id in self.assemblies:
                                          # Assembly removal logic would be more complex, involving recalculation
                                          # self.assemblies[asm_id].remove_memory(mem_id)
                                          pass # Simplified for now
                                del self.memory_to_assemblies[mem_id]
                           # Delete from persistence
                           await self.persistence.delete_memory(mem_id)
                           pruned_count += 1

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
                 # Assume query embedding generation happens here or is passed in context
                 # query_embedding = await self._generate_embedding(query) # Placeholder
                 # For now, we rely on the retrieve_memories method to handle text query
                 memories = await self.retrieve_memories(query=query, top_k=top_k)
                 # Return simplified dicts for LLM
                 return {"memories": [{"id": m.get("id"), "content": m.get("content"), "score": m.get("final_score", m.get("relevance_score"))} for m in memories]}

            elif tool_name == "process_new_memory_tool":
                 content = args.get("content")
                 metadata = args.get("metadata")
                 # Embedding generation would happen here
                 # embedding = await self._generate_embedding(content) # Placeholder
                 # For now, store without embedding if not provided
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
            embedding = model.encode([text], convert_to_tensor=False)[0]
            return self.geometry_manager._normalize(np.array(embedding, dtype=np.float32))
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error generating embedding: {str(e)}")
            
            # Fallback to a deterministic embedding based on text hash
            # This ensures same text always gets same embedding
            import hashlib
            
            # Create a deterministic embedding based on the hash of the text
            text_bytes = text.encode('utf-8')
            hash_obj = hashlib.md5(text_bytes)
            hash_digest = hash_obj.digest()
            
            # Convert the 16-byte digest to a list of floats
            # Repeating it to fill the embedding dimension
            byte_values = list(hash_digest) * (self.config['embedding_dim'] // 16 + 1)
            
            # Create a normalized embedding vector
            embedding = np.array([float(byte) / 255.0 for byte in byte_values[:self.config['embedding_dim']]], dtype=np.float32)
            
            logger.warning("SynthiansMemoryCore", "Using deterministic hash-based embedding generation")
            return self.geometry_manager._normalize(embedding)

    async def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry by its ID.
        
        Args:
            memory_id: The unique identifier of the memory to retrieve
            
        Returns:
            The MemoryEntry if found, None otherwise
        """
        async with self._lock:
            return self._memories.get(memory_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        persistence_stats = asyncio.run(self.persistence.get_stats()) # Run sync in this context
        quick_recal_stats = self.quick_recal.get_stats()
        threshold_stats = self.threshold_calibrator.get_statistics() if self.threshold_calibrator else {}

        return {
            "core_stats": {
                "total_memories": len(self._memories),
                "total_assemblies": len(self.assemblies),
                "initialized": self._initialized,
            },
            "persistence_stats": persistence_stats,
            "quick_recal_stats": quick_recal_stats,
            "threshold_stats": threshold_stats
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
            if embedding is None and self.geometry_manager is not None:
                try:
                    embedding = self.geometry_manager.process_text(content)
                    logger.info("SynthiansMemoryCore", f"Generated embedding for memory {memory_id}")
                except Exception as e:
                    logger.error("SynthiansMemoryCore", f"Error generating embedding: {str(e)}")
                    # Create a fallback embedding of zeros
                    embedding = np.zeros(self.config.get('embedding_dim', 768), dtype=np.float32)
            
            # If emotion_data is not provided but we have an emotion analyzer, try to generate it
            if emotion_data is None and self.emotion_analyzer is not None:
                try:
                    # Use the synchronous version for consistency
                    emotion_result = self.emotion_analyzer.analyze_sync(content)
                    if emotion_result and emotion_result.get('success', False):
                        emotion_data = emotion_result
                        logger.info("SynthiansMemoryCore", f"Generated emotion data for memory {memory_id}")
                except Exception as e:
                    logger.warning("SynthiansMemoryCore", f"Error analyzing emotions: {str(e)}")
            
            # Enhance metadata using the MetadataSynthesizer
            enhanced_metadata = metadata
            if self.metadata_synthesizer is not None:
                try:
                    # Use the synchronous version of metadata synthesis
                    enhanced_metadata = self.metadata_synthesizer.synthesize_sync(
                        content=content,
                        embedding=embedding,
                        base_metadata=metadata,
                        emotion_data=emotion_data
                    )
                    logger.info("SynthiansMemoryCore", f"Enhanced metadata for memory {memory_id}")
                except Exception as e:
                    logger.error("SynthiansMemoryCore", f"Error enhancing metadata: {str(e)}")
            
            # Calculate QuickRecal score
            quickrecal_score = 0.5  # Default value
            if self.quick_recal is not None and embedding is not None:
                try:
                    # Use synchronous version to avoid asyncio issues
                    context = {'text': content, 'timestamp': timestamp}
                    if enhanced_metadata:
                        context.update(enhanced_metadata)
                    
                    if hasattr(self.quick_recal, 'calculate_sync'):
                        quickrecal_score = self.quick_recal.calculate_sync(embedding, context=context)
                        logger.info("SynthiansMemoryCore", f"Calculated QuickRecal score: {quickrecal_score}")
                    else:
                        logger.warning("SynthiansMemoryCore", "No synchronous QuickRecal calculate method available")
                except Exception as e:
                    logger.error("SynthiansMemoryCore", f"Error calculating QuickRecal score: {str(e)}")
            
            # Create memory object
            memory_entry = {
                'id': memory_id,
                'content': content,
                'embedding': embedding.tolist() if embedding is not None else None,
                'metadata': enhanced_metadata,  # Use the enhanced metadata
                'quickrecal_score': quickrecal_score,
                'created_at': timestamp,
                'updated_at': timestamp,
                'access_count': 0,
                'last_accessed': timestamp
            }
            
            # Store memory directly without queueing to avoid potential deadlocks
            self._memories[memory_id] = memory_entry
            logger.info("SynthiansMemoryCore", f"Memory {memory_id} stored in memory")
            
            # Only queue if persistence is enabled and queue is not full
            try:
                if self._persistence and not self._memory_queue.full():
                    self._memory_queue.put_nowait((memory_id, memory_entry))
                    logger.info("SynthiansMemoryCore", f"Memory {memory_id} queued for persistence")
            except Exception as queue_err:
                logger.error("SynthiansMemoryCore", f"Failed to queue memory: {str(queue_err)}")
                # Memory is still in _memories, just not persisted
            
            # Return success
            return memory_entry
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error processing memory synchronously: {str(e)}")
            return None

    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry with provided updates.
        
        Args:
            memory_id: ID of the memory to update
            updates: Dictionary of fields to update and their new values
            
        Returns:
            True if the update succeeded, False otherwise
        """
        try:
            async with self._lock:
                # Get the memory
                memory = self._memories.get(memory_id)
                if not memory:
                    logger.warning("SynthiansMemoryCore", f"Cannot update memory {memory_id}: Not found")
                    return False
                
                # Update the memory fields
                for key, value in updates.items():
                    if hasattr(memory, key):
                        setattr(memory, key, value)
                    elif key == "metadata" and isinstance(value, dict):
                        # Special handling for metadata to merge rather than replace
                        if memory.metadata is None:
                            memory.metadata = {}
                        memory.metadata.update(value)
                    else:
                        logger.warning("SynthiansMemoryCore", f"Unknown field '{key}' in memory update")
                
                # Update quickrecal timestamp if quickrecal score was changed
                if "quickrecal_score" in updates:
                    memory.quickrecal_updated = dt.datetime.utcnow()
                
                # If this memory is in the vector index, update it there as well
                if memory.embedding is not None and memory_id in self.vector_index.id_to_index:
                    self.vector_index.update_entry(memory_id, memory.embedding)
                
                # Schedule persistence update
                await self.persistence.save_memory(memory)
                
                logger.info("SynthiansMemoryCore", f"Updated memory {memory_id} with {len(updates)} fields")
                return True
                
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error updating memory {memory_id}: {str(e)}")
            return False
