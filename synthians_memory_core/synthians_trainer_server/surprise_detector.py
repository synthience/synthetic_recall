import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from synthians_memory_core.geometry_manager import GeometryManager, GeometryType

logger = logging.getLogger(__name__)

class SurpriseDetector:
    """Detects surprising patterns in embedding sequences.
    
    This class analyzes semantic shifts in embeddings to identify moments that
    break the expected narrative flow, enabling the system to recognize pattern
    discontinuities and meaningful context shifts.
    """
    
    def __init__(self, 
                 geometry_manager: Optional[GeometryManager] = None,
                 surprise_threshold: float = 0.6,
                 max_sequence_length: int = 10,
                 surprise_decay: float = 0.9):
        """Initialize the surprise detector.
        
        Args:
            geometry_manager: Shared GeometryManager instance for consistent vector operations
            surprise_threshold: Threshold above which an embedding is considered surprising (0-1)
            max_sequence_length: Maximum number of recent embeddings to track
            surprise_decay: Decay factor for historical surprise (0-1)
        """
        # Use provided GeometryManager or create a default one
        self.geometry_manager = geometry_manager or GeometryManager()
        self.surprise_threshold = surprise_threshold
        self.max_sequence_length = max_sequence_length
        self.surprise_decay = surprise_decay
        
        # Internal memory of recent embeddings
        self.recent_embeddings: List[np.ndarray] = []
        self.recent_surprises: List[float] = []
        
        # Adaptive threshold tracking
        self.min_surprise_seen = 1.0
        self.max_surprise_seen = 0.0
        self.surprise_history: List[float] = []
        
        logger.info(f"SurpriseDetector initialized with geometry type: {self.geometry_manager.config['geometry_type']}")
    
    def _standardize_embedding(self, embedding: Union[List[float], np.ndarray]) -> np.ndarray:
        """Standardize an embedding to a normalized numpy array.
        
        Args:
            embedding: Input embedding
            
        Returns:
            Normalized numpy array
        """
        # Delegate to GeometryManager for consistent handling across the system
        return self.geometry_manager.normalize_embedding(embedding)
    
    def calculate_surprise(self, 
                           predicted_embedding: Union[List[float], np.ndarray],
                           actual_embedding: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """Calculate surprise between predicted and actual embeddings.
        
        Args:
            predicted_embedding: The embedding predicted by the trainer
            actual_embedding: The actual embedding observed
            
        Returns:
            Dictionary with surprise metrics
        """
        # Standardize inputs using GeometryManager
        pred_vec = self.geometry_manager.normalize_embedding(predicted_embedding)
        actual_vec = self.geometry_manager.normalize_embedding(actual_embedding)
        
        # Calculate similarity using GeometryManager
        similarity = self.geometry_manager.calculate_similarity(pred_vec, actual_vec)
        
        # Calculate surprise (1 - cosine similarity, rescaled to 0-1)
        cosine_surprise = (1.0 - similarity) / 2.0
        
        # Calculate delta vector (using GeometryManager for any needed alignment)
        aligned_pred, aligned_actual = self.geometry_manager.align_vectors(pred_vec, actual_vec)
        delta_vec = aligned_actual - aligned_pred
        delta_norm = float(np.linalg.norm(delta_vec))
        
        # Calculate context shift by comparing to recent embeddings
        context_surprise = 0.0
        if len(self.recent_embeddings) > 0:
            # Calculate average similarity to recent embeddings using GeometryManager
            similarities = [self.geometry_manager.calculate_similarity(actual_vec, e) for e in self.recent_embeddings]
            avg_similarity = sum(similarities) / len(similarities)
            context_surprise = (1.0 - avg_similarity) / 2.0
        
        # Combine surprise metrics (weighted average)
        prediction_weight = 0.7  # Weight for prediction error
        context_weight = 0.3     # Weight for context shift
        
        total_surprise = (prediction_weight * cosine_surprise + 
                          context_weight * context_surprise)
        
        # Update surprise history
        self.surprise_history.append(total_surprise)
        if len(self.surprise_history) > 100:  # Keep history manageable
            self.surprise_history = self.surprise_history[-100:]
            
        # Update min/max tracking for adaptive thresholds
        self.min_surprise_seen = min(self.min_surprise_seen, total_surprise)
        self.max_surprise_seen = max(self.max_surprise_seen, total_surprise)
        
        # Update recent embeddings memory
        self.recent_embeddings.append(actual_vec)
        if len(self.recent_embeddings) > self.max_sequence_length:
            self.recent_embeddings = self.recent_embeddings[-self.max_sequence_length:]
            
        # Update recent surprises
        self.recent_surprises.append(total_surprise)
        if len(self.recent_surprises) > self.max_sequence_length:
            self.recent_surprises = self.recent_surprises[-self.max_sequence_length:]
        
        # Calculate adaptive threshold
        if len(self.surprise_history) >= 10:
            mean_surprise = np.mean(self.surprise_history)
            std_surprise = np.std(self.surprise_history)
            adaptive_threshold = mean_surprise + std_surprise
        else:
            adaptive_threshold = self.surprise_threshold
            
        # Determine if this is surprising
        is_surprising = total_surprise > adaptive_threshold
        
        # Calculate surprise volatility (how much does surprise vary?)
        if len(self.recent_surprises) >= 3:
            volatility = float(np.std(self.recent_surprises))
        else:
            volatility = 0.0
            
        return {
            "surprise": float(total_surprise),
            "cosine_surprise": float(cosine_surprise),
            "context_surprise": float(context_surprise),
            "delta_norm": delta_norm,
            "is_surprising": is_surprising,
            "adaptive_threshold": float(adaptive_threshold),
            "volatility": float(volatility),
            "delta": delta_vec.tolist()
        }
    
    def calculate_quickrecal_boost(self, surprise_metrics: Dict[str, Any]) -> float:
        """Calculate how much to boost a memory's quickrecal score based on surprise.
        
        Args:
            surprise_metrics: Output from calculate_surprise method
            
        Returns:
            QuickRecal score boost (0-1 range)
        """
        # Extract metrics
        total_surprise = surprise_metrics["surprise"]
        is_surprising = surprise_metrics["is_surprising"]
        volatility = surprise_metrics["volatility"]
        
        # Base multiplier depends on whether it's actually surprising
        if not is_surprising:
            return 0.0
            
        # Scale boost based on how surprising it is
        # Apply sigmoid scaling to make boost more aggressive for very surprising items
        def sigmoid(x):
            return 1 / (1 + np.exp(-10 * (x - 0.5)))
            
        # Apply sigmoid scaling to boost (0.5-1.0 range becomes steeper)
        scaled_surprise = sigmoid(total_surprise)
        
        # Incorporate volatility - higher volatility increases the boost
        # Max volatility boost is 1.5x
        volatility_multiplier = 1.0 + (volatility * 0.5)
        
        # Calculate final boost (max 0.5 adjustment to quickrecal)
        boost = scaled_surprise * volatility_multiplier * 0.5
        
        # Ensure boost is in 0-0.5 range (we don't want to boost by more than 0.5)
        return float(min(0.5, max(0.0, boost)))
