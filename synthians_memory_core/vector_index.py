import logging
import os
import threading
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

# Dynamic FAISS import with fallback installation capability
try:
    import faiss
    logger.info("FAISS import successful")
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
        logger.info("FAISS successfully installed and imported")
    except Exception as e:
        logger.error(f"Failed to install FAISS: {str(e)}")
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

    def _initialize_index(self):
        """Initialize the FAISS index based on the configuration."""
        # Create CPU index first - always needed as a fallback and for initialization
        if self.index_type.upper() == 'L2':
            self.cpu_index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created L2 CPU index with dimension {self.embedding_dim}")
        elif self.index_type.upper() == 'IP' or self.index_type.upper() == 'COSINE':
            self.cpu_index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info(f"Created IP CPU index with dimension {self.embedding_dim}")
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # If GPU usage is requested, try to create a GPU index with timeout protection
        self.index = self.cpu_index  # Default to CPU index
        
        if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
            self._initialize_gpu_index_with_timeout()
        else:
            if self.use_gpu:
                logger.warning("GPU requested but FAISS was not built with GPU support. Using CPU index.")
            logger.info("Using CPU FAISS index")

    def _initialize_gpu_index_with_timeout(self):
        """Initialize GPU index with a timeout to prevent indefinite hanging."""
        # This will hold the result of GPU initialization
        result = {"success": False, "error": None, "index": None}
        
        # Define the initialization function to run in a separate thread
        def init_gpu():
            try:
                logger.info("Moving FAISS index to GPU 0...")
                
                # Create GPU resources with safe memory configuration
                res = faiss.StandardGpuResources()
                
                # Configure lower temp memory to avoid CUDA OOM issues
                try:
                    # This is a safer approach as it uses less GPU memory
                    # 64MB is typically sufficient for most operations, adjust as needed
                    res.setTempMemory(64 * 1024 * 1024)  # 64 MB, much safer than default
                    logger.info("Set FAISS GPU temp memory to 64 MB")
                except Exception as e:
                    logger.warning(f"Could not set GPU temp memory: {e}. Will use default.")
                
                # Transfer index to GPU
                gpu_index = faiss.index_cpu_to_gpu(res, 0, self.cpu_index)
                
                # Store result
                result["success"] = True
                result["index"] = gpu_index
                logger.info("GPU index successfully created")
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)
                logger.warning(f"GPU index creation failed: {e}")
        
        # Create and start the thread
        init_thread = threading.Thread(target=init_gpu)
        init_thread.daemon = True  # Allow the thread to be killed when the main thread exits
        init_thread.start()
        
        # Wait for the thread with timeout
        init_thread.join(timeout=self.gpu_timeout_seconds)
        
        # Check the result
        if init_thread.is_alive():
            # Thread is still running after timeout
            logger.warning(f"GPU initialization timed out after {self.gpu_timeout_seconds} seconds. Falling back to CPU.")
            return  # Keep using CPU index
        
        if result["success"] and result["index"] is not None:
            self.index = result["index"]
            self.is_using_gpu = True
            logger.info("Successfully initialized GPU index")
        else:
            error_msg = result["error"] if result["error"] else "Unknown error"
            logger.warning(f"Failed to initialize GPU index: {error_msg}. Falling back to CPU index.")

    def add(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Add a memory embedding to the index.
        
        Args:
            memory_id: A unique identifier for the memory
            embedding: The embedding vector for the memory
            
        Returns:
            bool: True if the memory was added successfully, False otherwise
        """
        try:
            # Validate the embedding
            embedding = self._validate_embedding(embedding)
            if embedding is None:
                logger.error(f"Invalid embedding for memory {memory_id}")
                return False
            
            # Get the next available index
            idx = len(self.id_to_index)
            
            # Add the embedding to the index
            self.index.add(np.array([embedding], dtype=np.float32))
            
            # Map the memory ID to its index
            self.id_to_index[memory_id] = idx
            
            return True
        except Exception as e:
            logger.error(f"Error adding memory {memory_id} to index: {str(e)}")
            return False

    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Search for the k nearest memories to the query embedding.
        
        Args:
            query_embedding: The query embedding vector
            k: The number of nearest neighbors to return
            threshold: Minimum similarity score threshold (for L2, this is actually a maximum distance)
            
        Returns:
            List[Tuple[str, float]]: A list of tuples containing the memory ID and similarity score
        """
        try:
            # Validate the query embedding
            query_embedding = self._validate_embedding(query_embedding)
            if query_embedding is None:
                logger.error("Invalid query embedding")
                return []
            
            # Create a reverse mapping for more efficient lookups
            index_to_id = {idx: mid for mid, idx in self.id_to_index.items()}
            
            # Log the state of the index and mappings
            logger.info(f"Searching index with {self.count()} vectors and {len(self.id_to_index)} id mappings")
            if self.count() == 0:
                logger.warning("Search called on empty index")
                return []
                
            # Ensure k is not larger than the number of items in the index
            k = min(k, self.count())
            if k == 0:
                return []
            
            # Search the index
            # For L2 distance, smaller values are better (closer)
            # For IP, larger values are better (more similar)
            distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
            
            logger.info(f"FAISS search returned {len(indices[0])} results with threshold {threshold}")
            
            # Convert the results to memory IDs and scores
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0:  # Invalid index, skip
                    continue
                    
                # Convert distances to similarity scores based on index type
                if self.index_type.upper() == 'L2':
                    # For L2, we want to penalize large distances
                    # Convert to a similarity score by using exp(-dist)
                    # This gives a score between 0 and 1, with 1 being an exact match
                    similarity = np.exp(-dist)
                    
                    # Apply threshold (for L2, lower distance is better, so check if similarity is high enough)
                    if threshold > 0 and similarity < threshold:
                        logger.debug(f"Filtered result: similarity {similarity:.4f} < threshold {threshold}")
                        continue
                else:
                    # For IP/Cosine, higher is better
                    similarity = dist
                    
                    # Apply threshold (for IP, higher is better)
                    if threshold > 0 and similarity < threshold:
                        logger.debug(f"Filtered result: similarity {similarity:.4f} < threshold {threshold}")
                        continue
                
                # Find the memory ID for this index using the reverse mapping
                memory_id = index_to_id.get(int(idx))
                
                if memory_id is not None:
                    results.append((memory_id, float(similarity)))
                    logger.debug(f"Found memory {memory_id} with similarity {similarity:.4f}")
                else:
                    logger.warning(f"No memory ID found for index {idx}")
            
            logger.info(f"Returning {len(results)} results after filtering")
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
        return self.index.ntotal

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
                mapping_path = os.path.join(self.storage_path, 'id_to_index_mapping.json')
            else:
                # If custom filepath, derive mapping path by adding .mapping.json extension
                mapping_path = filepath + '.mapping.json'
            
            # Save the FAISS index
            if self.is_using_gpu:
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, filepath)
                except Exception as e:
                    logger.warning(f"Could not extract CPU index from GPU index: {e}. Saving with default method.")
                    faiss.write_index(self.index, filepath)
            else:
                faiss.write_index(self.index, filepath)
            
            # Save the ID-to-index mapping
            import json
            with open(mapping_path, 'w') as f:
                # Convert any non-string keys to strings for JSON serialization
                mapping_serializable = {str(k): v for k, v in self.id_to_index.items()}
                json.dump(mapping_serializable, f)
            
            logger.info(f"Successfully saved index to {filepath} with {len(self.id_to_index)} memory mappings")
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
                mapping_path = os.path.join(self.storage_path, 'id_to_index_mapping.json')
            else:
                # If custom filepath, derive mapping path by adding .mapping.json extension
                mapping_path = filepath + '.mapping.json'
            
            if not os.path.exists(filepath):
                logger.warning(f"Index file not found at {filepath}")
                return False
            
            if os.path.isdir(filepath):
                logger.error(f"Expected a file but got a directory: {filepath}")
                return False
                
            # Load the index
            self.cpu_index = faiss.read_index(filepath)
            
            # Move to GPU if requested and supported
            if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
                self._initialize_gpu_index_with_timeout()
            else:
                self.index = self.cpu_index
            
            # Load the ID-to-index mapping
            import json
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mapping_data = json.load(f)
                    # Convert string keys back to their original type if needed
                    self.id_to_index = {k: int(v) for k, v in mapping_data.items()}
                logger.info(f"Successfully loaded {len(self.id_to_index)} memory mappings from {mapping_path}")
            else:
                logger.warning(f"Mapping file not found at {mapping_path}, memory retrieval may not work properly")
            
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
