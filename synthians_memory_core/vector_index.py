import logging
import os
import threading
import time
import numpy as np
import faiss
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import hashlib
import uuid

logger = logging.getLogger(__name__)

# Dynamic FAISS import with fallback installation capability
try:
    import faiss
    logger.info("FAISS import successful")
    # Explicitly check for GPU support
    try:
        res = faiss.StandardGpuResources()
        logger.info("FAISS GPU support available")
    except Exception as e:
        logger.warning(f"FAISS GPU support not available: {e}")
except ImportError:
    logger.warning("FAISS not found, attempting to install")
    try:
        import subprocess
        
        # Try to detect GPU and install appropriate version
        try:
            # First check if CUDA is available via nvidia-smi
            nvidia_smi_output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
            # If we get here, nvidia-smi worked, so install GPU version
            logger.info("NVIDIA GPU detected, installing FAISS with GPU support")
            result = subprocess.run(["pip", "install", "faiss-gpu"], check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            # No GPU available, install CPU version
            logger.info("No NVIDIA GPU detected, installing FAISS CPU version")
            result = subprocess.run(["pip", "install", "faiss-cpu"], check=True)
        
        # Now try importing again
        import faiss
        logger.info("FAISS installed and imported successfully")
    except Exception as e:
        logger.error(f"Failed to install FAISS: {e}")
        raise

class MemoryVectorIndex:
    """A vector index for storing and retrieving memory embeddings."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector index.
        
        Args:
            config: A dictionary containing configuration options for the vector index.
                - embedding_dim: The dimension of the embeddings to store (int)
                - storage_path: The path to store the index (str)
                - index_type: The type of index to use (str, e.g., 'L2' or 'IP')
                - use_gpu: Whether to use GPU for the index (bool, default: False)
                - gpu_timeout_seconds: Seconds to wait for GPU init before fallback (int, default: 10)
        """
        self.config = config
        self.embedding_dim = config.get('embedding_dim', 768)
        self.storage_path = config.get('storage_path', './faiss_index')
        self.index_type = config.get('index_type', 'L2')
        self.use_gpu = config.get('use_gpu', False)
        self.gpu_timeout_seconds = config.get('gpu_timeout_seconds', 10)
        self.id_to_index = {}  # Maps memory IDs to their indices in the FAISS index
        self.is_using_gpu = False  # Will be set to True if GPU init succeeds
        
        # Initialize the index based on the configuration
        self._initialize_index()

    def _initialize_index(self, force_cpu=False, use_id_map=False):
        """Initialize the FAISS index for the vector store.
        
        Args:
            force_cpu (bool): If True, forces CPU usage even if GPU is requested (for incompatible index types)
            use_id_map (bool): If True, use IndexIDMap for the index
        """
        try:
            # Create a flat index for L2 or IP distance
            if self.index_type.upper() == 'L2':
                base_index = faiss.IndexFlatL2(self.embedding_dim)
            else: # Inner Product or Cosine
                base_index = faiss.IndexFlatIP(self.embedding_dim)
                
            # For GPU usage, try to create a GPU version of the index
            # IMPORTANT: FAISS GPU indexes don't support add_with_ids, so we need CPU for IndexIDMap
            if self.use_gpu and not force_cpu:
                try:
                    # Create GPU resources
                    if not hasattr(faiss, 'StandardGpuResources'):
                        logger.warning("GPU FAISS not available. Falling back to CPU.")
                        self.is_using_gpu = False
                    else:
                        self.gpu_resources = faiss.StandardGpuResources()
                        # Convert the index to a GPU index
                        base_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, base_index)
                        self.is_using_gpu = True
                        logger.info(f"Using GPU FAISS index")
                except Exception as e:
                    logger.warning(f"Failed to initialize GPU index: {str(e)}. Falling back to CPU.")
                    self.is_using_gpu = False
                    # Re-create the base index since the conversion may have failed
                    if self.index_type.upper() == 'L2':
                        base_index = faiss.IndexFlatL2(self.embedding_dim)
                    else: # Inner Product or Cosine
                        base_index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self.is_using_gpu = False
                
            # Wrap the base index with an IndexIDMap to handle custom IDs
            # NOTE: ID map is incompatible with GPU indexes, so we need to use CPU
            if use_id_map and hasattr(faiss, 'IndexIDMap'):
                # If we're using GPU but need IndexIDMap, fall back to CPU
                if self.is_using_gpu:
                    logger.warning("IndexIDMap is incompatible with GPU indexes. Falling back to CPU.")
                    # Re-create the base index on CPU
                    if self.index_type.upper() == 'L2':
                        base_index = faiss.IndexFlatL2(self.embedding_dim)
                    else: # Inner Product or Cosine
                        base_index = faiss.IndexFlatIP(self.embedding_dim)
                    self.is_using_gpu = False
                
                self.index = faiss.IndexIDMap(base_index)
                logger.info(f"Created IndexIDMap with {self.index_type} base index, dimension {self.embedding_dim}")
            else:
                # Fallback if IndexIDMap is not available
                self.index = base_index
                logger.warning("IndexIDMap is not available. Using base index instead.")
                
            return True
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}")
            return False

    def add(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Add a memory vector to the index.
        
        Args:
            memory_id: The unique ID of the memory
            embedding: The embedding vector
            
        Returns:
            bool: Whether the add was successful
        """
        try:
            # Validate the embedding
            if not self._validate_embedding(embedding):
                logger.warning(f"Invalid embedding for memory {memory_id}, skipping")
                return False
                
            # Ensure embedding has correct shape
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
                
            # Generate a numeric ID for this memory if needed
            numeric_id = self._get_numeric_id(memory_id)
            
            # Different add approach based on index type
            if hasattr(self.index, 'add_with_ids'):
                # If using IndexIDMap
                try:
                    self.index.add_with_ids(embedding, np.array([numeric_id]))
                    self.id_to_index[memory_id] = numeric_id
                    # Backup id mapping after each add for better recovery
                    self._backup_id_mapping()
                    return True
                except Exception as e:
                    logger.error(f"Failed to add with IDs: {str(e)}")
                    # Fall back to standard add method
                    pass
                    
            # If not using IDMap or if add_with_ids failed
            if not hasattr(self.index, 'add_with_ids'):
                # Standard add approach
                index_before = self.count()
                self.index.add(embedding)
                if self.count() > index_before:
                    self.id_to_index[memory_id] = index_before  # First new index
                    # Backup id mapping after each add for better recovery
                    self._backup_id_mapping()
                    return True
                else:
                    logger.warning(f"Failed to add embedding for memory {memory_id}")
            
            return False
                
        except Exception as e:
            logger.error(f"Error adding memory to index: {str(e)}")
            return False

    def _backup_id_mapping(self) -> bool:
        """Backup the ID mapping to a JSON file for recovery purposes.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            mapping_path = os.path.join(self.storage_path, 'faiss_index.bin' + '.mapping.json')
            
            # Create a serializable copy of the mapping
            serializable_mapping = {}
            for k, v in self.id_to_index.items():
                # Convert any non-string keys to strings for JSON serializability
                key = str(k)
                # Convert any special numeric types to standard Python types
                if isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                    value = int(v)
                else:
                    value = v
                serializable_mapping[key] = value
            
            # Write the mapping to a file
            with open(mapping_path, 'w') as f:
                json.dump(serializable_mapping, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error backing up ID mapping: {str(e)}")
            return False

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search the index for similar embeddings.

        Args:
            query_embedding: The query embedding
            k: The number of results to return

        Returns:
            List[Tuple[str, float]]: A list of memory IDs and similarity scores in descending order of similarity
        """
        try:
            # Validate the query embedding
            validated_query = self._validate_embedding(query_embedding)
            if validated_query is None:
                logger.error("Invalid query embedding")
                return []
                
            # Log the state of the index and mappings
            current_count = self.count()
            logger.debug(f"Searching index with {current_count} vectors and {len(self.id_to_index)} id mappings")
            
            if current_count == 0:
                logger.warning("Search called on empty index")
                return []
                
            # Ensure k is not larger than the number of items in the index
            k = min(k, current_count)
            if k <= 0:
                logger.warning(f"Search k value adjusted to 0 or less ({k}), returning empty list.")
                return []
                
            # Different search approach based on index type
            if hasattr(self.index, 'search_and_reconstruct'):
                # For IndexIDMap, we get IDs directly
                distances, numeric_ids, _ = self.index.search_and_reconstruct(np.array([validated_query], dtype=np.float32), k)
                
                # Log raw results for debugging
                logger.debug(f"FAISS raw results - distances: {distances}, IDs: {numeric_ids}")
                
                # Convert the results to memory IDs and scores
                results = []
                if len(numeric_ids) > 0 and len(distances) > 0:  # Check if search returned anything
                    # Ensure indices are flattened and valid
                    valid_ids = [idx for idx in numeric_ids[0] if idx >= 0]  # Filter out -1 indices
                    valid_distances = [distances[0][i] for i, idx in enumerate(numeric_ids[0]) if idx >= 0]  # Filter corresponding distances
                    
                    logger.debug(f"Valid IDs after filtering -1: {valid_ids}")
                    
                    # Create reverse mapping from numeric_id to memory_id
                    numeric_to_memory_id = {v: k for k, v in self.id_to_index.items()}
                    
                    for i, numeric_id in enumerate(valid_ids):
                        dist = valid_distances[i]
                        
                        # Convert distances to similarity scores based on index type
                        if self.index_type.upper() == 'L2':
                            similarity = float(np.exp(-dist))  # Convert distance to similarity
                        else:  # IP or Cosine (already similarities)
                            similarity = float(dist)
                        
                        # Find the memory ID for this numeric ID
                        memory_id = numeric_to_memory_id.get(int(numeric_id))
                        
                        if memory_id is not None:
                            results.append((memory_id, similarity))
                            logger.debug(f"Candidate: ID={memory_id}, NumericID={numeric_id}, Similarity={similarity:.4f}")
                        else:
                            # Enhanced diagnostics for missing mappings
                            logger.warning(f"No memory ID found for numeric ID {numeric_id}")
            else:
                # Fallback for non-IDMap indices - use legacy approach
                distances, indices = self.index.search(np.array([validated_query], dtype=np.float32), k)
                
                # Convert the results to memory IDs and scores using the old mapping approach
                index_to_id = {idx: mid for mid, idx in self.id_to_index.items()}
                results = []
                
                if len(indices) > 0 and len(distances) > 0:  # Check if search returned anything
                    valid_indices = [idx for idx in indices[0] if idx >= 0]  # Filter out -1 indices
                    valid_distances = [distances[0][i] for i, idx in enumerate(indices[0]) if idx >= 0]  # Filter corresponding distances
                    
                    for i, idx in enumerate(valid_indices):
                        dist = valid_distances[i]
                        
                        # Convert distances to similarity scores based on index type
                        if self.index_type.upper() == 'L2':
                            similarity = float(np.exp(-dist))  # Convert distance to similarity
                        else:  # IP or Cosine (already similarities)
                            similarity = float(dist)
                        
                        # Find the memory ID for this index using the reverse mapping
                        memory_id = index_to_id.get(int(idx))
                        
                        if memory_id is not None:
                            results.append((memory_id, similarity))
                            logger.debug(f"Legacy candidate: ID={memory_id}, Index={idx}, Similarity={similarity:.4f}")
                        else:
                            logger.warning(f"No memory ID found for index {idx}")
            
            logger.info(f"FAISS search returning {len(results)} raw candidates")
            return results
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _validate_embedding(self, embedding: Union[np.ndarray, list, tuple]) -> Optional[np.ndarray]:
        """Validate and normalize an embedding vector.
        
        This handles several common issues:
        1. Converts lists/tuples to numpy arrays
        2. Ensures the embedding is 1D
        3. Checks for NaN or Inf values
        4. Ensures the embedding has the correct dimension
        
        Args:
            embedding: The embedding vector to validate
            
        Returns:
            np.ndarray: A validated embedding vector, or None if invalid
        """
        try:
            # Handle case where embedding is a dict (common error)
            if isinstance(embedding, dict):
                logger.error("Embedding is a dict, not a vector. You may have passed a structured payload instead.")
                return None
                
            # Convert to numpy array if not already
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            
            # Ensure embedding is 1D
            if len(embedding.shape) > 1:
                # If it's a 2D array with only one row, flatten it
                if len(embedding.shape) == 2 and embedding.shape[0] == 1:
                    embedding = embedding.flatten()
                else:
                    logger.error(f"Expected 1D embedding, got shape {embedding.shape}")
                    return None
            
            # Check for NaN or Inf values
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                logger.warning("Embedding contains NaN or Inf values. Replacing with zeros.")
                embedding = np.where(np.isnan(embedding) | np.isinf(embedding), 0.0, embedding)
            
            # Check dimension
            if len(embedding) != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
                # Resize the embedding to match expected dimension
                if len(embedding) < self.embedding_dim:
                    # Pad with zeros
                    padding = np.zeros(self.embedding_dim - len(embedding), dtype=np.float32)
                    embedding = np.concatenate([embedding, padding])
                else:
                    # Truncate
                    embedding = embedding[:self.embedding_dim]
            
            # Ensure dtype is float32 for FAISS
            embedding = embedding.astype(np.float32)
            
            return embedding
        except Exception as e:
            logger.error(f"Error validating embedding: {str(e)}")
            return None

    def count(self) -> int:
        """Get the number of embeddings in the index.
        
        Returns:
            int: The number of embeddings in the index
        """
        try:
            index_count = self.index.ntotal if hasattr(self.index, 'ntotal') else 0
            mapping_count = len(self.id_to_index) if hasattr(self, 'id_to_index') else 0
            
            # Check for inconsistencies
            if index_count != mapping_count:
                logger.warning(f"Vector index inconsistency detected! FAISS count: {index_count}, Mapping count: {mapping_count}")
                
            return index_count
        except Exception as e:
            logger.error(f"Error getting index count: {str(e)}")
            return 0

    def reset(self) -> bool:
        """Reset the index, removing all embeddings.
        
        Returns:
            bool: True if the index was reset successfully, False otherwise
        """
        try:
            # Re-initialize the index
            self._initialize_index()
            self.id_to_index = {}
            return True
        except Exception as e:
            logger.error(f"Error resetting index: {str(e)}")
            return False

    def save(self, filepath: Optional[str] = None) -> bool:
        """Save the index to disk.
        
        When using IndexIDMap, we save the entire index including the ID mappings in a single file.
        
        Args:
            filepath: The filepath to save the index to. If None, use the storage_path.
            
        Returns:
            bool: True if the index was saved successfully, False otherwise
        """
        try:
            # Create storage directory if it doesn't exist
            os.makedirs(self.storage_path, exist_ok=True)
            
            if filepath is None:
                filepath = os.path.join(self.storage_path, 'faiss_index.bin')
            
            # Prepare the index for saving
            index_to_save = self.index
            
            # If using GPU, extract CPU index first
            if self.is_using_gpu:
                try:
                    index_to_save = faiss.index_gpu_to_cpu(self.index)
                    logger.info("Successfully converted GPU index to CPU for saving")
                except Exception as e:
                    logger.warning(f"Could not extract CPU index from GPU index: {e}. Saving with default method.")
            
            # Save the FAISS index
            faiss.write_index(index_to_save, filepath)
            
            # Check if we need to save the id_to_index mapping separately
            # With IndexIDMap this might not be necessary, but we keep it for backward compatibility
            # and as a safety backup
            mapping_path = filepath + '.mapping.json'
            try:
                with open(mapping_path, 'w') as f:
                    # Convert any non-string keys to strings for JSON serialization
                    mapping_serializable = {str(k): v for k, v in self.id_to_index.items()}
                    json.dump(mapping_serializable, f)
                logger.info(f"Saved backup ID mapping to {mapping_path} with {len(self.id_to_index)} entries")
            except Exception as map_e:
                logger.warning(f"Failed to save ID mapping: {map_e}. Index saved but mapping may be lost.")
            
            logger.info(f"Successfully saved index to {filepath} with {self.count()} vectors")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def load(self, filepath: Optional[str] = None) -> bool:
        """Load the index from disk.
        
        Args:
            filepath: The filepath to load the index from. If None, use the storage_path.
            
        Returns:
            bool: True if the index was loaded successfully, False otherwise
        """
        try:
            if filepath is None:
                filepath = os.path.join(self.storage_path, 'faiss_index.bin')
                mapping_path = filepath + '.mapping.json'
            else:
                # If custom filepath, derive mapping path by adding .mapping.json extension
                mapping_path = filepath + '.mapping.json'
            
            if not os.path.exists(filepath):
                logger.warning(f"Index file not found at {filepath}. Starting fresh.")
                self._initialize_index()  # Initialize an empty index
                self.id_to_index = {}
                return False  # Indicate load didn't happen, but state is clean
            
            if os.path.isdir(filepath):
                logger.error(f"Expected a file but got a directory: {filepath}")
                return False
                
            # --- Load the index data from disk ---
            logger.info(f"Loading FAISS index data from {filepath}")
            loaded_cpu_index = faiss.read_index(filepath)
            logger.info(f"Successfully loaded CPU index data, ntotal={loaded_cpu_index.ntotal}")

            # --- Check if the loaded index uses IndexIDMap ---
            is_index_id_map = hasattr(loaded_cpu_index, 'id_map')
            logger.info(f"Loaded index is{'not' if not is_index_id_map else ''} an IndexIDMap")
            
            # --- Handle the loaded index (CPU or GPU) ---
            if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
                logger.info("Attempting to move loaded index data to GPU...")
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, loaded_cpu_index)
                    self.is_using_gpu = True
                    logger.info(f"Successfully moved loaded index to GPU, ntotal={self.index.ntotal}")
                except Exception as e:
                    logger.error(f"Failed to move loaded index to GPU: {e}. Falling back to CPU.")
                    self.index = loaded_cpu_index  # Fallback to the loaded CPU index
                    self.is_using_gpu = False
            else:
                # If not using GPU, assign the loaded CPU index directly
                self.index = loaded_cpu_index
                self.is_using_gpu = False
                logger.info(f"Using loaded CPU index, ntotal={self.index.ntotal}")
            
            # --- Attempt to load or rebuild the ID-to-index mapping ---
            self.id_to_index = {}  # Reset mapping before loading
            
            # If the index is an IndexIDMap, we can extract IDs directly
            if is_index_id_map:
                # For IndexIDMap, we need to rebuild the id_to_index from the index itself
                # This will be done later in a full rebuild if needed
                logger.info("Loaded index is an IndexIDMap, will extract IDs directly for operations")
            
            # Optionally load the backup mapping file (even for IndexIDMap as it has string->numeric mapping)
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, 'r') as f:
                        mapping_data = json.load(f)
                        
                    if isinstance(mapping_data, dict):
                        # Convert string keys back to their original type if needed
                        self.id_to_index = {k: int(v) if isinstance(v, str) and v.isdigit() else v 
                                           for k, v in mapping_data.items()}
                        logger.info(f"Successfully loaded {len(self.id_to_index)} memory mappings from {mapping_path}")
                except Exception as map_e:
                    logger.warning(f"Error loading mapping file {mapping_path}: {map_e}. May need to rebuild mapping.")
            else:
                logger.warning(f"Mapping file not found at {mapping_path}. Will rely on IndexIDMap internal mapping if available.")
            
            # For consistency checking and backup purposes
            if not is_index_id_map and (self.index.ntotal != len(self.id_to_index)):
                logger.warning(f"Mismatch after load: FAISS index has {self.index.ntotal} vectors, mapping has {len(self.id_to_index)} entries.")
            
            logger.info("Index load process completed.")
            return True
        except Exception as e:
            logger.error(f"General error loading index: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Reset index on general error
            self._initialize_index()
            self.id_to_index = {}
            return False

    def verify_index_integrity(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify the integrity of the index and the ID mapping.
        
        This method performs a thorough check of the index to identify any inconsistencies.
        
        Returns:
            Tuple[bool, Dict]: A tuple containing a boolean indicating whether the index is consistent,
                              and a dictionary with diagnostic information
        """
        try:
            # Initialize diagnostics
            diagnostics = {
                "faiss_count": 0,
                "id_mapping_count": 0,
                "is_index_id_map": False,
                "index_implementation": "Unknown",
                "is_consistent": False,
                "backup_mapping_exists": False,
                "backup_mapping_count": 0
            }
            
            # Check if the index is empty
            if self.index is None:
                return False, {**diagnostics, "error": "Index is None"}
                
            # Get the index type
            index_type = type(self.index).__name__
            diagnostics["index_implementation"] = index_type
            
            # Check if the index is an IndexIDMap
            is_index_id_map = hasattr(self.index, 'id_map')
            diagnostics["is_index_id_map"] = is_index_id_map
            
            # Count the number of vectors in the index
            faiss_count = self.count()
            diagnostics["faiss_count"] = faiss_count
            
            # Count the number of ID mappings
            id_mapping_count = len(self.id_to_index)
            diagnostics["id_mapping_count"] = id_mapping_count
            
            # Check if the ID mapping count matches the FAISS count
            if faiss_count != id_mapping_count:
                logger.warning(f"Vector index inconsistency detected! FAISS count: {faiss_count}, Mapping count: {id_mapping_count}")
                
                # Try to recover from backup mapping file if available
                mapping_path = os.path.join(self.storage_path, 'faiss_index.bin' + '.mapping.json')
                if os.path.exists(mapping_path):
                    diagnostics["backup_mapping_exists"] = True
                    try:
                        with open(mapping_path, 'r') as f:
                            mapping_data = json.load(f)
                        
                        if isinstance(mapping_data, dict):
                            diagnostics["backup_mapping_count"] = len(mapping_data)
                            
                            if id_mapping_count == 0 and len(mapping_data) > 0:
                                logger.warning("Empty ID mapping detected with available backup - recommend running repair_index")
                    except Exception as e:
                        logger.error(f"Error checking backup mapping file: {str(e)}")
            
            # The index is consistent if the counts match
            is_consistent = (faiss_count == id_mapping_count) or (is_index_id_map and id_mapping_count > 0)
            diagnostics["is_consistent"] = is_consistent
            
            return is_consistent, diagnostics
        
        except Exception as e:
            logger.error(f"Error verifying index integrity: {str(e)}")
            return False, {"error": str(e)}

    def migrate_to_idmap(self, force_cpu: bool = True) -> bool:
        """Migrate from a standard index to an IndexIDMap index.
        
        This method extracts all vectors from the current index,
        creates a new IndexIDMap, and adds all vectors with their IDs.
        
        Args:
            force_cpu: Whether to force CPU usage during migration
            
        Returns:
            bool: True if migration was successful, False otherwise
        """
        try:
            # Check if we already have an IndexIDMap
            if hasattr(self.index, 'id_map'):
                logger.info("Index is already using IndexIDMap, no migration needed")
                return True
                
            logger.info(f"Starting migration to IndexIDMap (force_cpu={force_cpu})")
            
            # Save the current index and mappings
            old_index = self.index
            old_id_to_index = self.id_to_index.copy()  # Copy to avoid modifying during iteration
            
            # Get the current vector count
            original_count = old_index.ntotal
            logger.info(f"Index contains {original_count} vectors before migration")
            
            if original_count == 0:
                logger.info("Empty index, creating fresh IndexIDMap")
                self._initialize_index(force_cpu=force_cpu, use_id_map=True)
                return True
                
            # Check mapping consistency
            if len(old_id_to_index) != original_count:
                logger.warning(f"Inconsistent ID mapping during migration: {len(old_id_to_index)} mappings for {original_count} vectors")
                # Continue anyway as migration might actually help fix this
                
            # Create a list to hold all the vectors and their IDs
            vectors = []
            ids = []
            id_mapping = {}
            next_id = 0
            
            # Special case: If we have vectors but no ID mapping, we need a special approach
            if original_count > 0 and len(old_id_to_index) == 0:
                logger.info("Using sequential extraction for index with no ID mappings")
                try:
                    # Extract vectors directly using a sequential approach
                    memory_ids = []
                    # Try to find memory files to get real memory IDs
                    memory_path = os.path.join(os.path.dirname(self.storage_path), 'memories')
                    
                    # Fallback paths if standard path doesn't exist
                    if not os.path.exists(memory_path):
                        alt_paths = [
                            os.path.join(self.storage_path, 'memories'),
                            os.path.join(os.path.dirname(os.path.dirname(self.storage_path)), 'memories')
                        ]
                        for path in alt_paths:
                            if os.path.exists(path):
                                memory_path = path
                                logger.info(f"Found memories directory at: {memory_path}")
                                break
                    
                    # If memory directory found, read memory IDs from files
                    if os.path.exists(memory_path):
                        for root, _, files in os.walk(memory_path):
                            for file in files:
                                if file.endswith('.json') and file.startswith('mem_'):
                                    memory_id = file.split('.')[0]  # Remove .json extension
                                    memory_ids.append(memory_id)
                        logger.info(f"Found {len(memory_ids)} memory files")
                    
                    # If we have memory_ids from files and they match the count, use them
                    if len(memory_ids) >= original_count:
                        logger.info(f"Using real memory IDs from files for extraction: {len(memory_ids)} available")
                        memory_ids = memory_ids[:original_count]  # Limit to number of vectors
                    else:
                        # Generate synthetic memory IDs if we couldn't find them or counts don't match
                        logger.warning(f"Generating synthetic memory IDs for {original_count} vectors (found only {len(memory_ids)} real IDs)")
                        memory_ids = [f"mem_{uuid.uuid4().hex[:12]}" for _ in range(original_count)]
                    
                    # Extract vectors using index.reconstruct with sequential indices
                    for i in range(original_count):
                        try:
                            memory_id = memory_ids[i]
                            numeric_id = self._get_numeric_id(memory_id)
                            
                            # Extract vector - approach depends on index type
                            if hasattr(old_index, 'reconstruct'):
                                vector = old_index.reconstruct(i)
                                vectors.append(vector)
                                ids.append(numeric_id)
                                id_mapping[memory_id] = numeric_id
                                next_id += 1
                            elif hasattr(old_index, 'xb'):
                                vector = old_index.xb[i * old_index.d: (i + 1) * old_index.d].reshape(1, -1)
                                vectors.append(vector.reshape(-1))
                                ids.append(numeric_id)
                                id_mapping[memory_id] = numeric_id
                                next_id += 1
                        except Exception as e:
                            logger.warning(f"Error extracting vector at index {i}: {str(e)}")
                    
                    logger.info(f"Extracted {len(vectors)} vectors using sequential extraction")
                except Exception as e:
                    logger.error(f"Error during sequential extraction: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # If no vectors extracted yet and we have ID mappings, try the standard approaches
            if not vectors and len(old_id_to_index) > 0:
                # Approach 1: If the old index allows reconstruction
                if hasattr(old_index, 'reconstruct'):
                    logger.info("Using vector reconstruction from original index")
                    try:
                        # For indices that support reconstruction
                        for memory_id, idx in old_id_to_index.items():
                            vector = old_index.reconstruct(idx)
                            
                            # Generate a consistent numeric ID for this memory_id
                            numeric_id = self._get_numeric_id(memory_id)
                            
                            vectors.append(vector)
                            ids.append(numeric_id)
                            id_mapping[memory_id] = numeric_id
                            next_id += 1
                    except Exception as e:
                        logger.error(f"Error during vector reconstruction: {str(e)}")
                        return False
                # Approach 2: For CPU indices, directly access the storage
                elif not force_cpu and hasattr(old_index, 'xb'):
                    logger.info("Using direct vector access from CPU index")
                    try:
                        # For CPU indices, we can directly access the vectors
                        for memory_id, idx in old_id_to_index.items():
                            if 0 <= idx < old_index.ntotal:
                                vector = old_index.xb[idx * old_index.d: (idx + 1) * old_index.d].reshape(1, -1)
                                
                                # Generate a consistent numeric ID for this memory_id
                                numeric_id = self._get_numeric_id(memory_id)
                                
                                vectors.append(vector.reshape(-1))
                                ids.append(numeric_id)
                                id_mapping[memory_id] = numeric_id
                                next_id += 1
                            else:
                                logger.warning(f"Invalid index {idx} for memory ID {memory_id}, skipping")
                    except Exception as e:
                        logger.error(f"Error during direct vector access: {str(e)}")
                        return False
                else:
                    logger.error("Cannot extract vectors from current index type for migration")
                    return False
                    
            # Check if we successfully extracted vectors
            if not vectors:
                logger.error("Failed to extract any vectors for migration")
                return False
                
            logger.info(f"Extracted {len(vectors)} vectors for migration")
            
            # Create a new IndexIDMap with CPU backend (GPU doesn't support add_with_ids)
            self._initialize_index(force_cpu=True, use_id_map=True)
            
            # Add all vectors with their IDs
            if vectors and ids:
                # Convert to numpy arrays
                vectors_array = np.vstack(vectors).astype(np.float32)
                ids_array = np.array(ids, dtype=np.int64)
                
                # Add to the new index
                self.index.add_with_ids(vectors_array, ids_array)
                
                # Update the ID mapping
                self.id_to_index = id_mapping
                
                # Verify the migration was successful
                if self.index.ntotal != len(vectors):
                    logger.error(f"Migration verification failed: expected {len(vectors)} vectors, got {self.index.ntotal}")
                    return False
                    
                # Backup ID mapping after successful migration
                self._backup_id_mapping()
                
                logger.info(f"Successfully migrated {self.index.ntotal} vectors to IndexIDMap")
                return True
            else:
                logger.error("No vectors to migrate")
                return False
                
        except Exception as e:
            logger.error(f"Error migrating to IndexIDMap: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def recreate_mapping(self) -> bool:
        """Recreate the ID mapping from persistent storage.
        
        This is useful when the index is intact but the ID mapping is lost or corrupted.
        It attempts to reconstruct the ID mappings by:
        1. Reading the backup mapping file if available
        2. Reading memory IDs from persistence layer
        3. Generating consistent numeric IDs for all memories
        
        Returns:
            bool: True if reconstruction was successful, False otherwise
        """
        try:
            logger.info("Starting ID mapping reconstruction...")
            
            # First check if we're using IndexIDMap
            if not hasattr(self.index, 'id_map'):
                logger.warning("Index is not using IndexIDMap, migrating first...")
                success = self.migrate_to_idmap()
                if not success:
                    logger.error("Migration to IndexIDMap failed, cannot recreate mapping.")
                    return False
            
            # 1. Try to read the backup mapping file first
            mapping_path = os.path.join(self.storage_path, 'faiss_index.bin' + '.mapping.json')
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, 'r') as f:
                        mapping_data = json.load(f)
                        
                    if isinstance(mapping_data, dict):
                        # Convert string keys to appropriate types if needed
                        self.id_to_index = {k: int(v) if isinstance(v, str) and v.isdigit() else v 
                                           for k, v in mapping_data.items()}
                        logger.info(f"Loaded {len(self.id_to_index)} ID mappings from backup file")
                        return True
                    else:
                        logger.warning("Mapping file has invalid format, cannot use it for reconstruction.")
                except Exception as e:
                    logger.error(f"Error reading mapping file: {str(e)}")
            else:
                logger.warning("No mapping backup file found, will try to reconstruct from memory directories.")
                
            # If we have no mappings at this point, we need to rebuild from scratch
            if len(self.id_to_index) == 0:
                # 2. Try to reconstruct from memory directories
                logger.info("Attempting to reconstruct ID mapping from memory directories...")
                memory_path = os.path.join(self.storage_path, 'memories')
                
                if not os.path.exists(memory_path):
                    # Look one level up
                    parent_path = os.path.dirname(self.storage_path)
                    potential_memory_path = os.path.join(parent_path, 'memories')
                    if os.path.exists(potential_memory_path):
                        memory_path = potential_memory_path
                    else:
                        # Search for memories directory
                        for root, dirs, _ in os.walk(os.path.dirname(self.storage_path)):
                            if 'memories' in dirs:
                                memory_path = os.path.join(root, 'memories')
                                logger.info(f"Found memories directory at {memory_path}")
                                break
                
                if os.path.exists(memory_path):
                    try:
                        # Scan memory directory for memory files
                        memory_ids = []
                        for root, _, files in os.walk(memory_path):
                            for file in files:
                                if file.endswith('.json') and file.startswith('mem_'):
                                    memory_id = file.split('.')[0]  # Remove .json extension
                                    memory_ids.append(memory_id)
                        
                        logger.info(f"Found {len(memory_ids)} memory files")
                        
                        # Generate numeric IDs for all memory IDs
                        total_count = self.index.ntotal
                        logger.info(f"FAISS index contains {total_count} vectors")
                        
                        # Two strategies - try both for best results:
                        # 1. Generate consistent IDs for all memories and update mapping
                        new_mapping = {}
                        for memory_id in memory_ids:
                            numeric_id = self._get_numeric_id(memory_id)
                            new_mapping[memory_id] = numeric_id
                        
                        # Only update if we found a reasonable number of memories
                        # and not wildly more than the vectors in the index
                        if 0 < len(new_mapping) <= total_count * 1.5:  # Allow some buffer
                            self.id_to_index = new_mapping
                            logger.info(f"Reconstructed {len(self.id_to_index)} ID mappings from memory files")
                            
                            # Backup the reconstructed mapping
                            self._backup_id_mapping()
                            return True
                        else:
                            logger.warning(f"Memory count ({len(new_mapping)}) doesn't match index count ({total_count})")
                    except Exception as e:
                        logger.error(f"Error scanning memory directory: {str(e)}")
                else:
                    logger.warning(f"Memories directory not found at {memory_path}, cannot reconstruct from files")
                    
                # 3. Last-resort fallback: Auto-generate sequential mappings
                if total_count > 0:
                    logger.warning("Using last-resort fallback: Generating sequential ID mappings")
                    # This is a complete guess but might recover some functionality
                    # Generate UUIDs for the vectors in the index
                    for i in range(total_count):
                        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
                        self.id_to_index[memory_id] = i
                    
                    logger.warning(f"Generated {len(self.id_to_index)} sequential mappings - CAUTION: these are not the original IDs")
                    # Backup these mappings for future use
                    self._backup_id_mapping()
                    return True
            
            return len(self.id_to_index) > 0
            
        except Exception as e:
            logger.error(f"Error recreating mapping: {str(e)}")
            return False
            
    def destroy_index(self) -> bool:
        """Completely remove the index from storage and memory.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Clear in-memory structures
            self.id_to_index = {}
            
            # Reset the index
            if self.index_type.upper() == 'L2':
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            else:  # IP or Cosine
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                
            # If the index file exists, remove it
            index_path = os.path.join(self.storage_path, 'faiss_index.bin')
            if os.path.exists(index_path):
                os.remove(index_path)
                
            # Also remove mapping file if it exists
            mapping_path = os.path.join(self.storage_path, 'faiss_index.bin' + '.mapping.json')
            if os.path.exists(mapping_path):
                os.remove(mapping_path)
                
            logger.info("Index successfully destroyed and reset")
            return True
            
        except Exception as e:
            logger.error(f"Error destroying index: {str(e)}")
            return False

    def _get_numeric_id(self, memory_id: str) -> int:
        """Generate a consistent numeric ID from a memory_id string.
        
        This ensures that each memory_id always maps to the same numeric ID,
        which is important for index consistency across restarts.
        
        Args:
            memory_id: The string memory ID
            
        Returns:
            int: A numeric ID derived from the memory_id
        """
        # Convert memory_id to a numeric ID using a hash function
        # We use a large prime to reduce collision chances
        # Note: we mask to 63 bits to avoid int64 overflow issues
        numeric_id = int(hashlib.md5(memory_id.encode()).hexdigest(), 16) % (2**63-1)
        return numeric_id
