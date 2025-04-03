"""
Embedding validation and alignment utilities for Synthians Memory Core.
Contains robust validation functions to ensure vectors are properly validated
before being used in critical operations.
"""

import numpy as np
import torch
import logging
from typing import Optional, Tuple, Union, List, Dict, Any

logger = logging.getLogger(__name__)

def validate_embedding(
    vector: Union[np.ndarray, List[float], torch.Tensor, None], 
    context_name: str = "Unknown",
    target_dim: int = 768
) -> Optional[np.ndarray]:
    """Thoroughly validate embeddings to prevent NaN/Inf values or dimension mismatches.
    
    Args:
        vector: The vector to validate
        context_name: Description of where this vector is used (for better error logs)
        target_dim: Expected dimension of the embedding
        
    Returns:
        Validated numpy array or None if validation fails
    """
    # Handle None case
    if vector is None:
        logger.warning(f"[VALIDATE] {context_name}: Received None vector")
        return None
        
    # Convert to numpy array if needed
    if isinstance(vector, list):
        try:
            vector = np.array(vector, dtype=np.float32)
        except Exception as e:
            logger.error(f"[VALIDATE] {context_name}: Failed to convert list to array: {e}")
            return None
    elif isinstance(vector, torch.Tensor):
        try:
            vector = vector.detach().cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error(f"[VALIDATE] {context_name}: Failed to convert tensor to array: {e}")
            return None
    elif not isinstance(vector, np.ndarray):
        logger.error(f"[VALIDATE] {context_name}: Unsupported vector type: {type(vector)}")
        return None
        
    # Check for NaN/Inf values
    if np.isnan(vector).any() or np.isinf(vector).any():
        logger.warning(f"[VALIDATE] {context_name}: Vector contains NaN or Inf values")
        return None
        
    # Check dimension
    if len(vector.shape) == 0:
        logger.warning(f"[VALIDATE] {context_name}: Vector has no dimensions")
        return None
        
    # Handle dimension mismatch
    if vector.shape[0] != target_dim:
        logger.warning(f"[VALIDATE] {context_name}: Dimension mismatch - got {vector.shape[0]}, expected {target_dim}")
        
        # Adjust dimensions if needed
        if vector.shape[0] > target_dim:
            vector = vector[:target_dim]  # Truncate
        else:
            # Pad with zeros
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:vector.shape[0]] = vector
            vector = padded
            
    return vector

def align_vectors(
    vec_a: np.ndarray, 
    vec_b: np.ndarray, 
    target_dim: int = 768
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Align two vectors to the same dimension.
    
    Args:
        vec_a: First vector
        vec_b: Second vector
        target_dim: Target dimension for both vectors
        
    Returns:
        Tuple of aligned vectors, or (None, None) if alignment fails
    """
    # Validate both vectors
    vec_a = validate_embedding(vec_a, "Vector A", target_dim)
    vec_b = validate_embedding(vec_b, "Vector B", target_dim)
    
    if vec_a is None or vec_b is None:
        return None, None
        
    return vec_a, vec_b

def safe_normalize(vector: np.ndarray) -> np.ndarray:
    """Safely normalize a vector to unit length.
    
    Args:
        vector: Vector to normalize
        
    Returns:
        Normalized vector or zero vector if normalization fails
    """
    if vector is None:
        return np.zeros(768, dtype=np.float32)
        
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector
        
    return vector / norm

def safe_calculate_similarity(vec_a: np.ndarray, vec_b: np.ndarray, target_dim: int = 768) -> float:
    """Safely calculate cosine similarity between two vectors.
    
    Args:
        vec_a: First vector
        vec_b: Second vector
        target_dim: Target dimension for alignment
        
    Returns:
        Cosine similarity in range [-1, 1] or 0.0 if calculation fails
    """
    # Align and validate
    vec_a, vec_b = align_vectors(vec_a, vec_b, target_dim)
    if vec_a is None or vec_b is None:
        return 0.0
        
    # Calculate similarity
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
        
    dot_product = np.dot(vec_a, vec_b)
    similarity = dot_product / (norm_a * norm_b)
    
    return float(np.clip(similarity, -1.0, 1.0))