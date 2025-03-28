# synthians_memory_core/geometry_manager.py

import numpy as np
import torch
import math
from enum import Enum
from typing import Optional, Tuple, List, Union, Dict, Any

from .custom_logger import logger # Use the shared custom logger

class GeometryType(Enum):
    EUCLIDEAN = "euclidean"
    HYPERBOLIC = "hyperbolic"
    SPHERICAL = "spherical"
    MIXED = "mixed"

class GeometryManager:
    """Centralized handling of embedding geometry, transformations, and calculations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'embedding_dim': 768,
            'geometry_type': GeometryType.EUCLIDEAN,
            'curvature': -1.0, # Relevant for Hyperbolic/Spherical
            'alignment_strategy': 'truncate', # or 'pad' or 'project'
            'normalization_enabled': True,
             **(config or {})
        }
        # Ensure geometry_type is enum
        if isinstance(self.config['geometry_type'], str):
            try:
                self.config['geometry_type'] = GeometryType(self.config['geometry_type'].lower())
            except ValueError:
                 logger.warning("GeometryManager", f"Invalid geometry type {self.config['geometry_type']}, defaulting to EUCLIDEAN.")
                 self.config['geometry_type'] = GeometryType.EUCLIDEAN

        # Warning counters
        self.dim_mismatch_warnings = 0
        self.max_dim_mismatch_warnings = 10
        self.nan_inf_warnings = 0
        self.max_nan_inf_warnings = 10

        logger.info("GeometryManager", "Initialized", self.config)

    def _validate_vector(self, vector: Union[np.ndarray, List[float], torch.Tensor], name: str = "Vector") -> Optional[np.ndarray]:
        """Validate and convert vector to numpy array."""
        if vector is None:
            logger.warning("GeometryManager", f"{name} is None")
            return None

        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
        elif isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy().astype(np.float32)
        elif not isinstance(vector, np.ndarray):
            logger.warning("GeometryManager", f"Unsupported vector type {type(vector)} for {name}, attempting conversion.")
            try:
                vector = np.array(vector, dtype=np.float32)
            except Exception as e:
                 logger.error("GeometryManager", f"Failed to convert {name} to numpy array", {"error": str(e)})
                 return None

        # Check for NaN/Inf
        if np.isnan(vector).any() or np.isinf(vector).any():
            if self.nan_inf_warnings < self.max_nan_inf_warnings:
                 logger.warning("GeometryManager", f"{name} contains NaN or Inf values. Replacing with zeros.")
                 self.nan_inf_warnings += 1
                 if self.nan_inf_warnings == self.max_nan_inf_warnings:
                      logger.warning("GeometryManager", "Max NaN/Inf warnings reached, suppressing further warnings.")
            return np.zeros_like(vector) # Replace invalid vector with zeros

        return vector

    def align_vectors(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two vectors to the configured embedding dimension."""
        # Validate inputs
        vec_a = self._validate_vector(vec_a, "Vector A")
        if vec_a is None:
            vec_a = np.zeros(self.config['embedding_dim'], dtype=np.float32)
            
        vec_b = self._validate_vector(vec_b, "Vector B")
        if vec_b is None:
            vec_b = np.zeros(self.config['embedding_dim'], dtype=np.float32)
            
        target_dim = self.config['embedding_dim']
        dim_a = vec_a.shape[0]
        dim_b = vec_b.shape[0]

        aligned_a = vec_a
        aligned_b = vec_b

        strategy = self.config['alignment_strategy']

        if dim_a != target_dim:
            if self.dim_mismatch_warnings < self.max_dim_mismatch_warnings:
                 logger.warning("GeometryManager", f"Vector A dimension mismatch: got {dim_a}, expected {target_dim}. Applying strategy: {strategy}")
                 self.dim_mismatch_warnings += 1
                 if self.dim_mismatch_warnings == self.max_dim_mismatch_warnings:
                      logger.warning("GeometryManager", "Max dimension mismatch warnings reached.")

            if strategy == 'pad':
                if dim_a < target_dim:
                    aligned_a = np.pad(vec_a, (0, target_dim - dim_a))
                else: # Truncate if padding isn't the strategy and dim > target
                    aligned_a = vec_a[:target_dim]
            elif strategy == 'truncate':
                if dim_a > target_dim:
                    aligned_a = vec_a[:target_dim]
                else: # Pad if truncating isn't the strategy and dim < target
                     aligned_a = np.pad(vec_a, (0, target_dim - dim_a))
            # Add 'project' strategy later if needed
            else: # Default to truncate/pad based on relative size
                if dim_a > target_dim: aligned_a = vec_a[:target_dim]
                else: aligned_a = np.pad(vec_a, (0, target_dim - dim_a))


        if dim_b != target_dim:
             if self.dim_mismatch_warnings < self.max_dim_mismatch_warnings:
                 logger.warning("GeometryManager", f"Vector B dimension mismatch: got {dim_b}, expected {target_dim}. Applying strategy: {strategy}")
                 # No warning count increment here, handled by vec_a check

             if strategy == 'pad':
                if dim_b < target_dim:
                    aligned_b = np.pad(vec_b, (0, target_dim - dim_b))
                else: aligned_b = vec_b[:target_dim]
             elif strategy == 'truncate':
                 if dim_b > target_dim: aligned_b = vec_b[:target_dim]
                 else: aligned_b = np.pad(vec_b, (0, target_dim - dim_b))
             else: # Default
                 if dim_b > target_dim: aligned_b = vec_b[:target_dim]
                 else: aligned_b = np.pad(vec_b, (0, target_dim - dim_b))

        return aligned_a, aligned_b

    def _align_vectors(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Backward compatibility method that forwards to align_vectors.
        
        Several components are calling this method with underscore, but the implementation
        is named 'align_vectors' (without underscore). This method ensures backward compatibility.
        """
        return self.align_vectors(vec_a, vec_b)

    def normalize_embedding(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        # Ensure input is numpy array
        vector = self._validate_vector(vector, "Vector to Normalize")
        if vector is None:
            # Return zero vector of appropriate dimension if validation failed
            return np.zeros(self.config.get('embedding_dim', 768), dtype=np.float32)

        if not self.config['normalization_enabled']:
             return vector
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            logger.debug("GeometryManager", "normalize_embedding received zero vector, returning as is.")
            return vector
        return vector / norm

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Backward compatibility method that forwards to normalize_embedding.
        
        Several components are calling this method with underscore, but the implementation
        is named 'normalize_embedding' (without underscore). This method ensures backward compatibility.
        """
        # Ensure vector is numpy array before calling
        validated_vector = self._validate_vector(vector, "Vector for _normalize")
        if validated_vector is None:
            # Return zero vector if validation fails
            return np.zeros(self.config.get('embedding_dim', 768), dtype=np.float32)
        return self.normalize_embedding(validated_vector)

    def _to_hyperbolic(self, euclidean_vector: np.ndarray) -> np.ndarray:
        """Project Euclidean vector to Poincaré ball."""
        norm = np.linalg.norm(euclidean_vector)
        if norm == 0: return euclidean_vector
        curvature = abs(self.config['curvature']) # Ensure positive for scaling
        if curvature == 0: curvature = 1.0 # Avoid division by zero if Euclidean is accidentally chosen
        # Adjusted scaling: tanh maps [0, inf) -> [0, 1)
        scale_factor = np.tanh(norm / 2.0) # Removed curvature influence here, seems standard
        hyperbolic_vector = (euclidean_vector / norm) * scale_factor
        # Ensure norm is strictly less than 1
        hyp_norm = np.linalg.norm(hyperbolic_vector)
        if hyp_norm >= 1.0:
            hyperbolic_vector = hyperbolic_vector * (0.99999 / hyp_norm)
        return hyperbolic_vector

    def _from_hyperbolic(self, hyperbolic_vector: np.ndarray) -> np.ndarray:
        """Project Poincaré ball vector back to Euclidean."""
        norm = np.linalg.norm(hyperbolic_vector)
        if norm >= 1.0:
            logger.warning("GeometryManager", "Hyperbolic vector norm >= 1, cannot project back accurately.", {"norm": norm})
            # Project onto the boundary and then back
            norm = 0.99999
            hyperbolic_vector = (hyperbolic_vector / np.linalg.norm(hyperbolic_vector)) * norm
        if norm == 0: return hyperbolic_vector
        # Inverse of tanh is arctanh
        original_norm_approx = np.arctanh(norm) * 2.0 # Approximation without curvature
        return (hyperbolic_vector / norm) * original_norm_approx

    def euclidean_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate Euclidean distance."""
        aligned_a, aligned_b = self.align_vectors(vec_a, vec_b)
        return np.linalg.norm(aligned_a - aligned_b)

    def hyperbolic_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate Hyperbolic (Poincaré) distance."""
        aligned_a, aligned_b = self.align_vectors(vec_a, vec_b)
        norm_a_sq = np.sum(aligned_a**2)
        norm_b_sq = np.sum(aligned_b**2)

        # Ensure vectors are strictly inside the unit ball
        if norm_a_sq >= 1.0: aligned_a = aligned_a * (0.99999 / np.sqrt(norm_a_sq)); norm_a_sq=np.sum(aligned_a**2)
        if norm_b_sq >= 1.0: aligned_b = aligned_b * (0.99999 / np.sqrt(norm_b_sq)); norm_b_sq=np.sum(aligned_b**2)

        euclidean_dist_sq = np.sum((aligned_a - aligned_b)**2)
        denominator = (1 - norm_a_sq) * (1 - norm_b_sq)

        if denominator < 1e-15: # Prevent division by zero or extreme values
            # If denominator is tiny, points are near boundary. If points are also close, distance is small. If far, distance is large.
            if euclidean_dist_sq < 1e-9: return 0.0
            else: return np.inf # Effectively infinite distance

        argument = 1 + (2 * euclidean_dist_sq / denominator)

        # Clamp argument to handle potential floating point issues near 1.0
        argument = max(1.0, argument)

        # Calculate distance with curvature
        curvature = abs(self.config['curvature'])
        if curvature <= 1e-9: curvature = 1.0 # Treat 0 curvature as Euclidean-like case within arccosh framework
        distance = np.arccosh(argument) / np.sqrt(curvature)

        return float(distance)

    def spherical_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate Spherical distance (angle)."""
        aligned_a, aligned_b = self.align_vectors(vec_a, vec_b)
        norm_a = np.linalg.norm(aligned_a)
        norm_b = np.linalg.norm(aligned_b)
        if norm_a < 1e-9 or norm_b < 1e-9: return np.pi # Max distance if one vector is zero
        cos_angle = np.dot(aligned_a, aligned_b) / (norm_a * norm_b)
        # Clamp to valid range for arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def mixed_distance(self, vec_a: np.ndarray, vec_b: np.ndarray, weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> float:
        """Calculate a weighted mixed distance."""
        euc_dist = self.euclidean_distance(vec_a, vec_b)
        hyp_dist = self.hyperbolic_distance(self._to_hyperbolic(vec_a), self._to_hyperbolic(vec_b))
        sph_dist = self.spherical_distance(vec_a, vec_b)
        # Normalize distances before combining (rough normalization)
        # Max Euclidean dist is 2, max spherical is pi
        euc_norm = euc_dist / 2.0
        sph_norm = sph_dist / np.pi
        # Hyperbolic distance can be large, use exp(-dist) for similarity-like scaling
        hyp_norm = np.exp(-hyp_dist * 0.5) # Scaled exponential decay

        # Combine weighted distances (treating hyp_norm as similarity, so use 1-hyp_norm)
        mixed_dist = weights[0] * euc_norm + weights[1] * (1.0 - hyp_norm) + weights[2] * sph_norm
        return float(mixed_dist)

    def calculate_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate distance based on configured geometry."""
        vec_a = self._validate_vector(vec_a, "Vector A")
        vec_b = self._validate_vector(vec_b, "Vector B")
        if vec_a is None or vec_b is None: return np.inf # Return infinite distance if validation failed

        geom_type = self.config['geometry_type']
        if geom_type == GeometryType.EUCLIDEAN:
            return self.euclidean_distance(vec_a, vec_b)
        elif geom_type == GeometryType.HYPERBOLIC:
            # Assume vectors are Euclidean, project them first
            hyp_a = self._to_hyperbolic(vec_a)
            hyp_b = self._to_hyperbolic(vec_b)
            return self.hyperbolic_distance(hyp_a, hyp_b)
        elif geom_type == GeometryType.SPHERICAL:
            return self.spherical_distance(vec_a, vec_b)
        elif geom_type == GeometryType.MIXED:
            return self.mixed_distance(vec_a, vec_b)
        else:
            logger.warning("GeometryManager", f"Unknown geometry type {geom_type}, using Euclidean.")
            return self.euclidean_distance(vec_a, vec_b)

    def calculate_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate similarity between two vectors based on the configured geometry type.
        
        Returns cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)
        """
        # Validate inputs
        vec_a = self._validate_vector(vec_a, "Vector A for similarity")
        if vec_a is None:
            return 0.0
            
        vec_b = self._validate_vector(vec_b, "Vector B for similarity")
        if vec_b is None:
            return 0.0
            
        # Align vectors to same dimension
        aligned_a, aligned_b = self.align_vectors(vec_a, vec_b)
        
        # Normalize both vectors
        norm_a = np.linalg.norm(aligned_a)
        norm_b = np.linalg.norm(aligned_b)
        
        # Handle zero vectors
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        
        # Calculate cosine similarity
        norm_a_inv = 1.0 / norm_a
        norm_b_inv = 1.0 / norm_b
        dot_product = np.dot(aligned_a, aligned_b)
        similarity = dot_product * norm_a_inv * norm_b_inv
        
        # Ensure result is in valid range [-1.0, 1.0]
        return float(np.clip(similarity, -1.0, 1.0))

    def transform_to_geometry(self, vector: np.ndarray) -> np.ndarray:
        """Transform a vector into the configured geometry space (e.g., Poincaré ball)."""
        vector = self._validate_vector(vector, "Input Vector")
        if vector is None: return np.zeros(self.config['embedding_dim'])

        geom_type = self.config['geometry_type']
        if geom_type == GeometryType.HYPERBOLIC:
            return self._to_hyperbolic(vector)
        elif geom_type == GeometryType.SPHERICAL:
            # Project onto unit sphere (normalize)
            return self.normalize_embedding(vector)
        else: # Euclidean or Mixed (no specific projection needed for Euclidean part)
            return vector

    def transform_from_geometry(self, vector: np.ndarray) -> np.ndarray:
        """Transform a vector from the configured geometry space back to Euclidean."""
        vector = self._validate_vector(vector, "Input Vector")
        if vector is None: return np.zeros(self.config['embedding_dim'])

        geom_type = self.config['geometry_type']
        if geom_type == GeometryType.HYPERBOLIC:
            return self._from_hyperbolic(vector)
        else: # Spherical, Euclidean, Mixed - assume normalization or no transformation needed
            return vector
