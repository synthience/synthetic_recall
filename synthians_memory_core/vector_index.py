# synthians_memory_core/vector_index.py

import logging
import os
import asyncio # Import asyncio
import threading # Keep for potential sync locks if needed elsewhere
import time
import numpy as np
import faiss
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import hashlib
import uuid
import traceback # Import traceback

# Try importing aiofiles, but don't make it a hard requirement
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    logging.getLogger(__name__).warning("aiofiles library not found. File operations will be synchronous.")

logger = logging.getLogger(__name__)

# Dynamic FAISS import with fallback installation capability
# (Keep the dynamic import logic as is)
try:
    import faiss
    logger.info("FAISS import successful")
    try:
        res = faiss.StandardGpuResources()
        logger.info("FAISS GPU support available")
    except Exception as e:
        logger.warning(f"FAISS GPU support not available: {e}")
except ImportError:
    logger.warning("FAISS not found, attempting to install")
    try:
        import subprocess
        import sys
        try:
            nvidia_smi_output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
            logger.info("NVIDIA GPU detected, installing FAISS with GPU support")
            result = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "faiss-gpu"], check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.info("No NVIDIA GPU detected, installing FAISS CPU version")
            result = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "faiss-cpu>=1.7.4"], check=True) # Ensure recent CPU version

        import faiss
        logger.info("FAISS installed and imported successfully")
    except Exception as e:
        logger.error(f"Failed to install FAISS: {e}")
        raise

class MemoryVectorIndex:
    """A vector index for storing and retrieving memory embeddings."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector index."""
        self.config = config
        self.embedding_dim = config.get('embedding_dim', 768)
        self.storage_path = config.get('storage_path', './faiss_index')
        os.makedirs(self.storage_path, exist_ok=True) # Ensure storage path exists
        self.index_type = config.get('index_type', 'L2')
        self.use_gpu = config.get('use_gpu', False)
        self.gpu_timeout_seconds = config.get('gpu_timeout_seconds', 10)
        self.id_to_index: Dict[str, int] = {}  # Maps memory IDs (str) to their FAISS numeric IDs (int)
        self.is_using_gpu = False
        self._lock = asyncio.Lock() # Added lock for async operations

        # Initialize index - default to IndexIDMap
        self._initialize_index(use_id_map=self.config.get('migrate_to_idmap', True))

    def _initialize_index(self, force_cpu=False, use_id_map=True):
        """Initialize the FAISS index for the vector store."""
        try:
            logger.info(f"Initializing FAISS index: dim={self.embedding_dim}, type={self.index_type}, use_id_map={use_id_map}")
            if self.index_type.upper() == 'L2':
                base_index = faiss.IndexFlatL2(self.embedding_dim)
            elif self.index_type.upper() in ['IP', 'COSINE']:
                base_index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                logger.warning(f"Unsupported index_type '{self.index_type}'. Defaulting to L2.")
                self.index_type = 'L2'
                base_index = faiss.IndexFlatL2(self.embedding_dim)

            self.is_using_gpu = False # Reset GPU flag

            # Attempt GPU usage only if requested, not forced CPU, and GPU support exists
            if self.use_gpu and not force_cpu and hasattr(faiss, 'StandardGpuResources'):
                # Check if we actually need GPU (e.g., IDMap forces CPU)
                if use_id_map:
                    logger.warning("IndexIDMap requested, which is incompatible with GPU indexes. Forcing CPU for base index.")
                else:
                    # Try to initialize GPU
                    try:
                        self.gpu_resources = faiss.StandardGpuResources()
                        base_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, base_index)
                        self.is_using_gpu = True
                        logger.info(f"Using GPU FAISS index (Device 0)")
                    except Exception as e:
                        logger.warning(f"Failed to initialize GPU index: {e}. Falling back to CPU.")
                        # Re-create CPU base index if GPU init failed
                        if self.index_type.upper() == 'L2':
                            base_index = faiss.IndexFlatL2(self.embedding_dim)
                        else:
                            base_index = faiss.IndexFlatIP(self.embedding_dim)

            # Wrap with IndexIDMap if requested and available
            if use_id_map and hasattr(faiss, 'IndexIDMap'):
                # Ensure base index is on CPU for IDMap
                if self.is_using_gpu:
                    logger.warning("Cannot use IndexIDMap with GPU index. Reverting base index to CPU.")
                    base_index = faiss.index_gpu_to_cpu(base_index)
                    self.is_using_gpu = False # Mark as not using GPU anymore
                self.index = faiss.IndexIDMap(base_index)
                logger.info(f"Created IndexIDMap wrapping {self.index_type} base index.")
            elif use_id_map:
                 logger.error("faiss.IndexIDMap not available in this build. Cannot use ID mapping.")
                 self.index = base_index # Use base index as fallback
            else:
                 self.index = base_index # Not using IDMap
                 logger.info(f"Using base {self.index_type} index without ID mapping.")

            return True
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}", exc_info=True)
            self.index = None # Set index to None on critical failure
            return False

    async def _backup_id_mapping(self) -> bool:
        """Backup the ID mapping to a JSON file asynchronously."""
        # This operation modifies a shared resource (mapping file), lock should be acquired by caller
        mapping_path = os.path.join(self.storage_path, 'faiss_index.bin.mapping.json')
        try:
            # Create a serializable copy
            serializable_mapping = {str(k): int(v) if isinstance(v, np.integer) else v
                                    for k, v in self.id_to_index.items()}

            if AIOFILES_AVAILABLE:
                async with aiofiles.open(mapping_path, 'w') as f:
                    await f.write(json.dumps(serializable_mapping, indent=2))
            else:
                # Synchronous fallback
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._backup_id_mapping_sync_helper, mapping_path, serializable_mapping)

            logger.debug(f"Backed up {len(serializable_mapping)} ID mappings to {mapping_path}")
            return True
        except Exception as e:
            logger.error(f"Error backing up ID mapping: {e}")
            return False

    def _backup_id_mapping_sync_helper(self, path, mapping):
        """Synchronous helper for file writing."""
        with open(path, 'w') as f:
            json.dump(mapping, f, indent=2)

    def _backup_id_mapping_sync(self) -> bool:
        """Synchronous backup of the ID mapping."""
        # Called by synchronous `save` method
        mapping_path = os.path.join(self.storage_path, 'faiss_index.bin.mapping.json')
        try:
            serializable_mapping = {str(k): int(v) if isinstance(v, np.integer) else v
                                    for k, v in self.id_to_index.items()}
            self._backup_id_mapping_sync_helper(mapping_path, serializable_mapping)
            return True
        except Exception as e:
            logger.error(f"Error backing up ID mapping (sync): {e}")
            return False

    async def add(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Add a memory vector to the index asynchronously."""
        if self.index is None:
             logger.error(f"Cannot add memory {memory_id}: Index not initialized.")
             return False
        if not hasattr(self.index, 'add_with_ids'):
            logger.error(f"Cannot add memory {memory_id}: Index does not support 'add_with_ids'. Initialize with use_id_map=True.")
            return False

        async with self._lock: # Acquire lock for modifying index and mapping
            try:
                embedding_validated = self._validate_embedding(embedding)
                if embedding_validated is None:
                    logger.warning(f"Invalid embedding for memory {memory_id}, skipping add")
                    return False

                if len(embedding_validated.shape) == 1:
                    embedding_validated = embedding_validated.reshape(1, -1)

                numeric_id = self._get_numeric_id(memory_id)
                ids_array = np.array([numeric_id], dtype=np.int64)

                loop = asyncio.get_running_loop()
                # FAISS add is typically CPU-bound or involves GPU transfer, run in executor
                await loop.run_in_executor(None, self.index.add_with_ids, embedding_validated, ids_array)

                self.id_to_index[memory_id] = numeric_id
                backup_success = await self._backup_id_mapping() # Await the async backup

                if not backup_success:
                    logger.warning(f"Failed to backup ID mapping after adding {memory_id}")

                logger.debug(f"Added vector for memory ID {memory_id} (Numeric ID: {numeric_id})")
                return True

            except Exception as e:
                logger.error(f"Error adding memory {memory_id} to index: {e}", exc_info=True)
                return False

    async def remove_vector(self, memory_id: str) -> bool:
        """Remove a vector by its memory ID asynchronously."""
        if self.index is None:
             logger.error(f"Cannot remove memory {memory_id}: Index not initialized.")
             return False
        if not hasattr(self.index, 'remove_ids'):
             logger.error("Remove_vector called, but index does not support remove_ids.")
             return False # Cannot proceed if index doesn't support removal by ID

        async with self._lock: # Acquire lock
            try:
                numeric_id = self.id_to_index.get(memory_id)
                if numeric_id is None:
                    logger.warning(f"Cannot remove vector for {memory_id}: ID not found in mapping.")
                    return False # ID wasn't mapped, nothing to remove

                ids_to_remove = np.array([numeric_id], dtype=np.int64)

                loop = asyncio.get_running_loop()
                # FAISS remove is typically CPU-bound, run in executor
                num_removed = await loop.run_in_executor(None, self.index.remove_ids, ids_to_remove)

                if num_removed > 0:
                    del self.id_to_index[memory_id]
                    backup_success = await self._backup_id_mapping()
                    if not backup_success:
                         logger.warning(f"Failed to backup ID mapping after removing {memory_id}")
                    logger.debug(f"Removed vector for memory ID {memory_id}")
                    return True
                else:
                    logger.warning(f"Vector for {memory_id} (numeric ID {numeric_id}) not found in FAISS index for removal, but removing from mapping.")
                    if memory_id in self.id_to_index:
                         del self.id_to_index[memory_id]
                         await self._backup_id_mapping() # Await the async backup
                    return False # Indicate vector wasn't actually in FAISS index

            except Exception as e:
                logger.error(f"Error removing vector for {memory_id}: {e}", exc_info=True)
                return False

    async def update_entry(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Update the embedding for an existing memory ID asynchronously."""
        # Locks are handled by remove_vector and add
        try:
            validated_embedding = self._validate_embedding(embedding)
            if validated_embedding is None:
                logger.warning(f"Invalid embedding for memory {memory_id}, skipping update")
                return False

            # Check mapping first (no lock needed for read, but remove/add use lock)
            if memory_id not in self.id_to_index:
                 logger.warning(f"Cannot update vector for {memory_id}: ID not found in mapping.")
                 return False

            # Remove the existing vector first
            removed = await self.remove_vector(memory_id)
            if not removed:
                logger.warning(f"Failed to remove existing vector for {memory_id} during update, attempting to add anyway")

            # Add the updated vector
            added = await self.add(memory_id, validated_embedding)
            if not added:
                logger.error(f"Failed to add updated vector for {memory_id} after removal attempt.")
                return False

            logger.debug(f"Successfully updated vector for memory ID {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating vector for {memory_id}: {e}", exc_info=True)
            return False

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search the index for similar embeddings. (Synchronous)"""
        if self.index is None:
            logger.error("Search failed: Index not initialized.")
            return []
        try:
            validated_query = self._validate_embedding(query_embedding)
            if validated_query is None: return []

            current_count = self.count()
            if current_count == 0: return []
            k = min(k, current_count)
            if k <= 0: return []

            # Normalize query for cosine/IP if needed
            if self.index_type.upper() in ['IP', 'COSINE']:
                 norm = np.linalg.norm(validated_query)
                 if norm > 1e-6: validated_query = validated_query / norm

            query_vector_faiss = validated_query.reshape(1, -1)

            # Perform search (synchronous FAISS call)
            distances, numeric_ids = self.index.search(query_vector_faiss, k)

            results = []
            if len(numeric_ids) > 0 and len(distances) > 0:
                valid_ids_indices = [(idx, i) for i, idx in enumerate(numeric_ids[0]) if idx >= 0]
                numeric_to_memory_id = {v: k for k, v in self.id_to_index.items()} # Build reverse map inside

                for numeric_id, index_in_results in valid_ids_indices:
                    dist = distances[0][index_in_results]
                    similarity = 0.0
                    if self.index_type.upper() == 'L2':
                        similarity = 1.0 / (1.0 + float(dist)) # Simple inverse distance
                    elif self.index_type.upper() == 'IP':
                        similarity = float(dist) # Inner product IS similarity (if vectors normalized)
                    elif self.index_type.upper() == 'COSINE':
                         # FAISS IP index on normalized vectors gives cosine similarity directly
                         similarity = float(dist)

                    memory_id = numeric_to_memory_id.get(int(numeric_id))
                    if memory_id is not None:
                        results.append((memory_id, similarity))
                    else:
                        logger.warning(f"No memory ID found for numeric FAISS ID {numeric_id}")

            results.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"FAISS search returning {len(results)} candidates")
            return results
        except Exception as e:
            logger.error(f"Error searching index: {e}", exc_info=True)
            return []

    def _validate_embedding(self, embedding: Union[np.ndarray, list, tuple]) -> Optional[np.ndarray]:
        """Validate and align embedding vector."""
        try:
            if embedding is None: return None
            if isinstance(embedding, dict): return None # Catch dict error

            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)

            if embedding.size == 0: return None
            if len(embedding.shape) > 1:
                if len(embedding.shape) == 2 and embedding.shape[0] == 1: embedding = embedding.flatten()
                else: return None

            if np.isnan(embedding).any() or np.isinf(embedding).any():
                logger.warning("Embedding contains NaN/Inf values. Replacing with zeros.")
                embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0) # More robust replace

            if len(embedding) != self.embedding_dim:
                logger.warning(f"Aligning embedding dim: expected {self.embedding_dim}, got {len(embedding)}")
                if len(embedding) < self.embedding_dim:
                    embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
                else:
                    embedding = embedding[:self.embedding_dim]

            # Ensure float32 for FAISS compatibility
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error validating embedding: {e}", exc_info=True)
            return None

    def count(self) -> int:
        """Get the number of embeddings in the index."""
        try:
            index_count = self.index.ntotal if self.index and hasattr(self.index, 'ntotal') else 0
            mapping_count = len(self.id_to_index)

            # Only log warning if counts mismatch AND we are using IndexIDMap (where they should match)
            if index_count != mapping_count and hasattr(self.index, 'id_map'):
                logger.warning(f"Vector index potential inconsistency! FAISS count: {index_count}, Mapping count: {mapping_count}")

            return index_count
        except Exception as e:
            logger.error(f"Error getting index count: {e}")
            return 0

    def reset(self) -> bool:
        """Reset the index, removing all embeddings."""
        try:
            logger.info("Resetting FAISS index and ID mapping.")
            # Determine if current index uses IDMap before resetting
            use_id_map = hasattr(self.index, 'id_map') if self.index else True # Default to true if index is None
            self._initialize_index(use_id_map=use_id_map)
            self.id_to_index = {}
            self._backup_id_mapping_sync() # Backup empty mapping synchronously
            return True
        except Exception as e:
            logger.error(f"Error resetting index: {e}")
            return False

    def save(self, filepath: Optional[str] = None) -> bool:
        """Save the index to disk. (Synchronous)"""
        if self.index is None:
            logger.error("Cannot save: Index not initialized.")
            return False
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            if filepath is None:
                filepath = os.path.join(self.storage_path, 'faiss_index.bin')

            index_to_save = self.index
            if self.is_using_gpu:
                try:
                    index_to_save = faiss.index_gpu_to_cpu(self.index)
                except Exception as e:
                    logger.warning(f"Could not extract CPU index from GPU: {e}")

            faiss.write_index(index_to_save, filepath)
            save_map_ok = self._backup_id_mapping_sync()

            if not save_map_ok:
                 logger.warning(f"Index saved to {filepath}, but failed to save mapping file.")

            logger.info(f"Saved index to {filepath} with {self.count()} vectors")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}", exc_info=True)
            return False

    def load(self, filepath: Optional[str] = None) -> bool:
        """Load the index from disk. (Synchronous)"""
        try:
            if filepath is None:
                filepath = os.path.join(self.storage_path, 'faiss_index.bin')
            mapping_path = filepath + '.mapping.json'

            if not os.path.exists(filepath):
                logger.warning(f"Index file not found: {filepath}. Initializing empty index.")
                # Initialize empty index, respecting IDMap setting
                return self._initialize_index(use_id_map=self.config.get('migrate_to_idmap', True))

            logger.info(f"Loading FAISS index from {filepath}")
            loaded_cpu_index = faiss.read_index(filepath)
            is_index_id_map = hasattr(loaded_cpu_index, 'id_map')
            logger.info(f"Loaded index type: {type(loaded_cpu_index).__name__}, Is IDMap: {is_index_id_map}, NTotal: {loaded_cpu_index.ntotal}")

            # Decide whether to move to GPU
            if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
                if not is_index_id_map:
                    try:
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, 0, loaded_cpu_index)
                        self.is_using_gpu = True
                        logger.info(f"Successfully moved loaded index to GPU, ntotal={self.index.ntotal}")
                    except Exception as e:
                        logger.error(f"Failed to move loaded index to GPU: {e}. Using CPU.")
                        self.index = loaded_cpu_index
                        self.is_using_gpu = False
                else:
                    logger.info("Keeping loaded IndexIDMap on CPU.")
                    self.index = loaded_cpu_index
                    self.is_using_gpu = False
            else:
                self.index = loaded_cpu_index
                self.is_using_gpu = False
                logger.info(f"Using loaded CPU index, ntotal={self.index.ntotal}")

            # Load mapping
            self.id_to_index = {}
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, 'r') as f:
                        mapping_data = json.load(f)
                    if isinstance(mapping_data, dict):
                        # Convert keys back to str, values to int
                        self.id_to_index = {str(k): int(v) for k, v in mapping_data.items() if isinstance(v, (int, str)) and str(v).isdigit()}
                        logger.info(f"Loaded {len(self.id_to_index)} ID mappings from {mapping_path}")
                    else:
                         logger.warning(f"Invalid mapping file format: {mapping_path}")
                except Exception as e:
                    logger.error(f"Error loading mapping file {mapping_path}: {e}")
            else:
                 logger.warning(f"Mapping file not found: {mapping_path}. Mapping is empty.")

            # --- CRITICAL: Rebuild mapping if IndexIDMap and mapping is empty/mismatched ---
            if is_index_id_map and self.index.ntotal > 0 and (len(self.id_to_index) == 0 or self.index.ntotal != len(self.id_to_index)):
                logger.warning(f"Rebuilding id_to_index mapping from IndexIDMap content (FAISS: {self.index.ntotal}, Mapping: {len(self.id_to_index)}).")
                # This requires iterating through the index IDs, which can be slow for large indices
                # FAISS Python API doesn't provide a direct way to get all IDs from IndexIDMap efficiently without reconstruction
                # Option 1: If we have the original string IDs somewhere (e.g., persistence index) - Preferred
                # Option 2: Reconstruct vectors and potentially match? Very slow.
                # Option 3: Store mapping within FAISS (not standard)?
                # For now, we rely on the mapping file as the primary source for string IDs.
                # If mapping file is bad, `recreate_mapping` is needed.
                logger.warning("Automatic mapping rebuild from IndexIDMap content is not implemented. Run repair if needed.")


            # Final consistency check
            if self.index.ntotal != len(self.id_to_index):
                 logger.warning(f"Inconsistency after load: FAISS has {self.index.ntotal}, Mapping has {len(self.id_to_index)}. Consider repair.")

            logger.info("Index load completed.")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}", exc_info=True)
            self._initialize_index(use_id_map=self.config.get('migrate_to_idmap', True)) # Re-init empty on error
            self.id_to_index = {}
            return False

    def verify_index_integrity(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify the integrity of the index and the ID mapping. (Synchronous)"""
        # (Implementation remains the same)
        try:
            diagnostics = { "faiss_count": 0, "id_mapping_count": 0, "is_index_id_map": False,
                            "index_implementation": "Unknown", "is_consistent": False,
                            "backup_mapping_exists": False, "backup_mapping_count": 0 }
            if self.index is None: return False, {**diagnostics, "error": "Index is None"}

            index_type = type(self.index).__name__
            diagnostics["index_implementation"] = index_type
            is_index_id_map = hasattr(self.index, 'id_map')
            diagnostics["is_index_id_map"] = is_index_id_map
            faiss_count = self.count() # Uses internal count method
            diagnostics["faiss_count"] = faiss_count
            id_mapping_count = len(self.id_to_index)
            diagnostics["id_mapping_count"] = id_mapping_count

            is_consistent = (faiss_count == id_mapping_count)

            # Check backup only if inconsistent
            if not is_consistent:
                mapping_path = os.path.join(self.storage_path, 'faiss_index.bin.mapping.json')
                if os.path.exists(mapping_path):
                    diagnostics["backup_mapping_exists"] = True
                    try:
                        with open(mapping_path, 'r') as f: mapping_data = json.load(f)
                        if isinstance(mapping_data, dict): diagnostics["backup_mapping_count"] = len(mapping_data)
                    except Exception as e: logger.error(f"Error checking backup mapping: {e}")

            diagnostics["is_consistent"] = is_consistent
            return is_consistent, diagnostics
        except Exception as e:
            logger.error(f"Error verifying index integrity: {e}")
            return False, {"error": str(e)}

    # (migrate_to_idmap, recreate_mapping, destroy_index should ideally remain synchronous
    # or be refactored to not rely on async context if called outside event loops)

    def _get_numeric_id(self, memory_id: str) -> int:
        """Generate a consistent 64-bit numeric ID from a string ID."""
        return int(hashlib.md5(memory_id.encode()).hexdigest(), 16) % (2**63 - 1)

    # --- Add get_stats method ---
    def get_stats(self) -> Dict[str, Any]:
         """Get statistics about the vector index."""
         return {
             "count": self.count(),
             "id_mappings": len(self.id_to_index),
             "embedding_dim": self.embedding_dim,
             "index_type": self.index_type,
             "is_gpu": self.is_using_gpu,
             "is_id_map": hasattr(self.index, 'id_map') if self.index else False,
         }