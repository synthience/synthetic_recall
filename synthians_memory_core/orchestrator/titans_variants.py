#!/usr/bin/env python

from enum import Enum
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING
import threading

# Use TYPE_CHECKING for type hints that won't be evaluated at runtime
if TYPE_CHECKING:
    import tensorflow as tf

# Lazy-load TensorFlow to avoid NumPy incompatibility issues during startup
_tf = None
_tf_lock = threading.Lock()

def _get_tf():
    """Lazy-load TensorFlow only when needed to avoid early NumPy conflicts"""
    global _tf
    if _tf is None:
        with _tf_lock:
            # Double-check after acquiring lock (thread-safe singleton pattern)
            if _tf is None:
                import tensorflow as tf
                _tf = tf
    return _tf

logger = logging.getLogger(__name__)

class TitansVariantType(str, Enum):
    """Enumeration of Titans architecture variants."""
    NONE = "NONE"  # No attention mechanism, base Neural Memory
    MAC = "MAC"    # Memory-Attended Computation
    MAG = "MAG"    # Memory-Attended Gates
    MAL = "MAL"    # Memory-Augmented Learning


class TitansVariantConfig(dict):
    """Configuration for Titans architecture variants."""
    def __init__(self, *args, **kwargs):
        defaults = {
            "variant": TitansVariantType.NONE.value,
            "attention_num_heads": 4,
            "attention_key_dim": 32,  # per head
            "attention_dropout": 0.0,
            "attention_use_layer_norm": True,
            "attention_use_residual": True,
            "max_context_length": 50,
            "max_dim_mismatch_warnings": 10,
        }
        
        config = defaults.copy()
        # Apply kwargs first
        config.update(kwargs)
        # Then apply dict from args if provided
        if args and isinstance(args[0], dict):
            config.update(args[0])
            
        super().__init__(config)
        
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'TitansVariantConfig' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


class TitansVariantBase:
    """Base class for all Titans architecture variants."""
    
    def __init__(self, attention_config: Optional[Dict[str, Any]] = None):
        """Initialize the base Titans variant.
        
        Args:
            attention_config: Optional configuration dictionary for attention parameters.
        """
        self.config = attention_config or {}
        self.variant_type = TitansVariantType.NONE
        self.sequence_context = None
    
    def set_sequence_context(self, sequence_context) -> None:
        """Set the sequence context manager for historical attention context.
        
        Args:
            sequence_context: SequenceContextManager instance to use for context history.
        """
        self.sequence_context = sequence_context
        
    def process_input(self, memory_id: str, x_t: np.ndarray, k_t: np.ndarray, 
                      v_t: np.ndarray, q_t: np.ndarray, y_t: np.ndarray) -> Dict[str, Any]:
        """Process input through the variant's logic.
        
        Args:
            memory_id: ID of the current memory being processed
            x_t: Original input embedding
            k_t: Key projection
            v_t: Value projection
            q_t: Query projection
            y_t: Retrieved embedding from Neural Memory
            
        Returns:
            Dict containing variant-specific outputs and metrics
        """
        # Base implementation just returns empty dict
        return {}


class MACVariant(TitansVariantBase):
    """Memory-Attended Computation (MAC) variant.
    
    Enhances memory retrieval by attending over historical memory outputs.
    Flow: q_t -> M -> y_t -> Attend(q_t, K_hist, Y_hist) -> attended_y_t
    """
    
    def __init__(
        self, 
        config: Optional[Union[TitansVariantConfig, Dict]] = None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.name = "MAC"
        self.variant_type = TitansVariantType.MAC
        
        # Initialize attention module for this variant
        attention_config = {
            "num_heads": self.config.get("attention_num_heads", 4),
            "key_dim": self.config.get("attention_key_dim", 32),
            "dropout": self.config.get("attention_dropout", 0.0),
            # The following parameters are not supported in this TF version
            # "use_layer_norm": self.config.get("attention_use_layer_norm", True),
            # "use_residual": self.config.get("attention_use_residual", True),
            "max_dim_mismatch_warnings": self.config.get("max_dim_mismatch_warnings", 10),
        }
        self.attention_module = _get_tf().keras.layers.MultiHeadAttention(
            num_heads=attention_config["num_heads"],
            key_dim=attention_config["key_dim"],
            dropout=attention_config["dropout"],
            # Removed unsupported parameters
            # use_layer_norm=attention_config["use_layer_norm"],
            # use_residual=attention_config["use_residual"],
            name="MAC_Attention"
        )
        
        logger.info(f"Initialized MAC variant with {attention_config['num_heads']} attention heads")

    def process_input(
        self,
        memory_id: str,
        x_t: np.ndarray, 
        k_t: np.ndarray,
        v_t: np.ndarray,
        q_t: np.ndarray,
        y_t: np.ndarray,
    ) -> Dict[str, Any]:
        """Implement MAC variant logic.
        
        1. Store context tuple (timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)
        2. Retrieve recent history pairs (k_i, y_i) from sequence_context
        3. Calculate attended output using attention module: attended_y_t = AttentionModule(q_t, K_hist, Y_hist)
        4. Return attended_y_t for use by downstream components
        """
        # First store the context
        self.store_context(memory_id, x_t, k_t, v_t, q_t, y_t)
        
        # Get historical contexts for attention
        if len(self.sequence_context) < 2:
            # Not enough context for attention, return original output
            logger.info("MAC: Not enough context for attention, using original output")
            return {
                "memory_id": memory_id,
                "attended_output": y_t,  # No change
                "metrics": self.attention_module.get_metrics() if self.attention_module else {},
            }
        
        # Get historical keys and memory outputs
        k_hist, y_hist = self.sequence_context.get_recent_ky_pairs(count=len(self.sequence_context) - 1)  # Exclude current
        
        # If using TensorFlow backend for attention:
        # Convert numpy arrays to tensors for TF operations
        q_tensor = _get_tf().convert_to_tensor(q_t, dtype='float32')
        if len(q_tensor.shape) == 1:
            q_tensor = _get_tf().expand_dims(q_tensor, 0)  # Add batch dimension
            
        k_hist_tensor = _get_tf().convert_to_tensor(k_hist, dtype='float32')
        if len(k_hist_tensor.shape) == 2:  # [seq_len, key_dim]
            k_hist_tensor = _get_tf().expand_dims(k_hist_tensor, 0)  # Add batch dimension [1, seq_len, key_dim]
            
        y_hist_tensor = _get_tf().convert_to_tensor(y_hist, dtype='float32')
        if len(y_hist_tensor.shape) == 2:  # [seq_len, value_dim]
            y_hist_tensor = _get_tf().expand_dims(y_hist_tensor, 0)  # Add batch dimension [1, seq_len, value_dim]
        
        # Apply attention mechanism
        try:
            attended_y_tensor = self.attention_module(
                query=q_tensor,
                key=k_hist_tensor,
                value=y_hist_tensor,
                training=False,
            )
            
            # Convert back to numpy for consistency
            attended_y = attended_y_tensor.numpy()
            if len(attended_y.shape) > 1:
                attended_y = attended_y[0]  # Remove batch dimension
                
            logger.info(f"MAC: Applied attention to {len(k_hist)} historical memories")
            
            return {
                "memory_id": memory_id,
                "attended_output": attended_y,
                "original_output": y_t,  # Keep original for comparison
                "metrics": self.attention_module.get_metrics(),
            }
            
        except Exception as e:
            logger.error(f"MAC attention failed: {e}")
            # Fallback to original output
            return {
                "memory_id": memory_id,
                "attended_output": y_t,  # Fallback to original
                "error": str(e),
                "metrics": {},
            }


class MAGVariant(TitansVariantBase):
    """Memory-Attended Gates (MAG) variant.
    
    Modifies gate values (alpha, theta, eta) for the neural memory update
    by attending over historical key projections.
    
    Flow: 
    1. q_t -> Attend(q_t, K_hist, K_hist) -> attention_output
    2. attention_output -> Gate Projections -> (alpha_t, theta_t, eta_t)
    3. Update memory with calculated gates
    """
    
    def __init__(
        self, 
        config: Optional[Union[TitansVariantConfig, Dict]] = None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.name = "MAG"
        self.variant_type = TitansVariantType.MAG
        
        # Initialize attention module for this variant
        attention_config = {
            "num_heads": self.config.get("attention_num_heads", 4),
            "key_dim": self.config.get("attention_key_dim", 32),
            "dropout": self.config.get("attention_dropout", 0.0),
            "max_dim_mismatch_warnings": self.config.get("max_dim_mismatch_warnings", 10),
        }
        self.attention_module = _get_tf().keras.layers.MultiHeadAttention(
            num_heads=attention_config["num_heads"],
            key_dim=attention_config["key_dim"],
            dropout=attention_config["dropout"],
            name="MAG_Attention"
        )
        
        # Gate projection layers (to be initialized when dimensions are known)
        self.attention_to_gates = None
        
        logger.info(f"Initialized MAG variant with {attention_config['num_heads']} attention heads")
    
    def init_gate_projections(self, attention_dim: int):
        """Initialize gate projection layers.
        
        These layers project the attention output to scalar gate values.
        
        Args:
            attention_dim: Dimension of the attention output
        """
        self.attention_to_alpha = _get_tf().keras.layers.Dense(1, name="att_alpha_proj")
        self.attention_to_theta = _get_tf().keras.layers.Dense(1, name="att_theta_proj")
        self.attention_to_eta = _get_tf().keras.layers.Dense(1, name="att_eta_proj")
        
        # Build the layers with a dummy input to ensure variables are created
        dummy_input = _get_tf().zeros([1, attention_dim], dtype='float32')
        self.attention_to_alpha(dummy_input)
        self.attention_to_theta(dummy_input)
        self.attention_to_eta(dummy_input)
        
        logger.info(f"MAG: Initialized gate projection layers with attention_dim={attention_dim}")

    def calculate_gates_from_attention(self, attention_output: 'tf.Tensor') -> Tuple[float, float, float]:
        """Calculate gate values from attention output.
        
        Args:
            attention_output: Output tensor from attention mechanism
            
        Returns:
            Tuple of (alpha, theta, eta) gate values
        """
        if self.attention_to_alpha is None:
            # Initialize gate projections if not already done
            self.init_gate_projections(attention_output.shape[-1])
        
        # Project attention output to gate logits
        alpha_logit = self.attention_to_alpha(attention_output)
        theta_logit = self.attention_to_theta(attention_output)
        eta_logit = self.attention_to_eta(attention_output)
        
        # Apply sigmoid activation and convert to scalar values
        alpha = _get_tf().sigmoid(alpha_logit).numpy().item()
        theta = _get_tf().sigmoid(theta_logit).numpy().item()
        eta = _get_tf().sigmoid(eta_logit).numpy().item()
        
        return alpha, theta, eta

    def process_input(
        self,
        memory_id: str,
        x_t: np.ndarray, 
        k_t: np.ndarray,
        v_t: np.ndarray,
        q_t: np.ndarray,
        y_t: np.ndarray,
    ) -> Dict[str, Any]:
        """Implement MAG variant logic.
        
        1. Store context tuple (timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)
        2. Retrieve recent history keys from sequence_context
        3. Calculate attention output using K_hist
        4. Project attention output to gate values (alpha, theta, eta)
        5. Return calculated gates for use in neural memory update
        """
        # First store the context
        self.store_context(memory_id, x_t, k_t, v_t, q_t, y_t)
        
        # Get historical contexts for attention
        if len(self.sequence_context) < 2:
            # Not enough context for attention, return default gates
            logger.info("MAG: Not enough context for attention, using default gates")
            return {
                "memory_id": memory_id,
                "attended_output": y_t,  # No change to y_t
                "alpha": None,  # Let neural memory use default
                "theta": None,  # Let neural memory use default
                "eta": None,    # Let neural memory use default
                "metrics": {},
            }
        
        # Get historical keys
        k_hist = self.sequence_context.get_recent_keys(count=len(self.sequence_context) - 1)  # Exclude current
        
        # Convert numpy arrays to tensors for TF operations
        q_tensor = _get_tf().convert_to_tensor(q_t, dtype='float32')
        if len(q_tensor.shape) == 1:
            q_tensor = _get_tf().expand_dims(q_tensor, 0)  # Add batch dimension
            
        k_hist_tensor = _get_tf().convert_to_tensor(k_hist, dtype='float32')
        if len(k_hist_tensor.shape) == 2:  # [seq_len, key_dim]
            k_hist_tensor = _get_tf().expand_dims(k_hist_tensor, 0)  # Add batch dimension [1, seq_len, key_dim]
        
        # Apply attention mechanism
        try:
            attention_output = self.attention_module(
                query=q_tensor,
                key=k_hist_tensor,
                value=k_hist_tensor,  # Self-attention on historical keys
                training=False,
            )
            
            # Calculate gates from attention output
            alpha, theta, eta = self.calculate_gates_from_attention(attention_output)
            
            logger.info(f"MAG: Calculated gates from attention: alpha={alpha:.4f}, theta={theta:.4f}, eta={eta:.4f}")
            
            return {
                "memory_id": memory_id,
                "attended_output": y_t,  # No change to y_t
                "alpha": alpha,
                "theta": theta,
                "eta": eta,
                "metrics": self.attention_module.get_metrics(),
            }
            
        except Exception as e:
            logger.error(f"MAG attention failed: {e}")
            # Fallback to default gates
            return {
                "memory_id": memory_id,
                "attended_output": y_t,
                "alpha": None,
                "theta": None,
                "eta": None,
                "error": str(e),
                "metrics": {},
            }


class MALVariant(TitansVariantBase):
    """Memory-Augmented Learning (MAL) variant.
    
    Modifies value projection for neural memory update by attending over
    historical value projections.
    
    Flow: 
    1. q_t, K_hist, V_hist -> Attend(q_t, K_hist, V_hist) -> attended_v_t
    2. Combine attended_v_t with v_t -> v_prime_t
    3. Update memory with k_t and v_prime_t
    """
    
    def __init__(
        self, 
        config: Optional[Union[TitansVariantConfig, Dict]] = None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.name = "MAL"
        self.variant_type = TitansVariantType.MAL
        
        # Initialize attention module for this variant
        attention_config = {
            "num_heads": self.config.get("attention_num_heads", 4),
            "key_dim": self.config.get("attention_key_dim", 32),
            "dropout": self.config.get("attention_dropout", 0.0),
            "max_dim_mismatch_warnings": self.config.get("max_dim_mismatch_warnings", 10),
        }
        self.attention_module = _get_tf().keras.layers.MultiHeadAttention(
            num_heads=attention_config["num_heads"],
            key_dim=attention_config["key_dim"],
            dropout=attention_config["dropout"],
            name="MAL_Attention"
        )
        
        # Gating layers for combining attended and current values (initialized when dimensions are known)
        self.v_prime_gate = None
        self.v_prime_projector = None
        
        logger.info(f"Initialized MAL variant with {attention_config['num_heads']} attention heads")
    
    def init_value_projection_layers(self, value_dim: int):
        """Initialize value projection and gating layers.
        
        Args:
            value_dim: Dimension of the value vectors
        """
        self.v_prime_gate = _get_tf().keras.layers.Dense(1, activation='sigmoid', name="v_prime_gate")
        self.v_prime_projector = _get_tf().keras.layers.Dense(value_dim, activation='tanh', name="v_prime_projector")
        
        # Build the layers with dummy inputs to ensure variables are created
        dummy_input = _get_tf().zeros([1, value_dim * 2], dtype='float32')  # Concatenated dimension
        self.v_prime_gate(dummy_input)
        
        dummy_input2 = _get_tf().zeros([1, value_dim], dtype='float32')
        self.v_prime_projector(dummy_input2)
        
        logger.info(f"MAL: Initialized value projection layers with value_dim={value_dim}")

    def process_input(
        self,
        memory_id: str,
        x_t: np.ndarray, 
        k_t: np.ndarray,
        v_t: np.ndarray,
        q_t: np.ndarray,
        y_t: np.ndarray,
    ) -> Dict[str, Any]:
        """Implement MAL variant logic.
        
        1. Store context tuple (timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)
        2. Retrieve recent history pairs (k_i, v_i) from sequence_context
        3. Calculate attended value using attention module: attended_v_t = AttentionModule(q_t, K_hist, V_hist)
        4. Combine attended_v_t with current v_t using gating mechanism
        5. Return v_prime_t for neural memory update
        """
        # First store the context
        self.store_context(memory_id, x_t, k_t, v_t, q_t, y_t)
        
        # Get historical contexts for attention
        if len(self.sequence_context) < 2:
            # Not enough context for attention, return original values
            logger.info("MAL: Not enough context for attention, using original value projection")
            return {
                "memory_id": memory_id,
                "attended_output": y_t,  # No change
                "v_prime": v_t,  # No change
                "metrics": {},
            }
        
        # Get historical keys and values
        k_hist, v_hist = self.sequence_context.get_recent_kv_pairs(count=len(self.sequence_context) - 1)  # Exclude current
        
        # Convert numpy arrays to tensors for TF operations
        q_tensor = _get_tf().convert_to_tensor(q_t, dtype='float32')
        if len(q_tensor.shape) == 1:
            q_tensor = _get_tf().expand_dims(q_tensor, 0)  # Add batch dimension
            
        k_hist_tensor = _get_tf().convert_to_tensor(k_hist, dtype='float32')
        if len(k_hist_tensor.shape) == 2:  # [seq_len, key_dim]
            k_hist_tensor = _get_tf().expand_dims(k_hist_tensor, 0)  # Add batch dimension [1, seq_len, key_dim]
            
        v_hist_tensor = _get_tf().convert_to_tensor(v_hist, dtype='float32')
        if len(v_hist_tensor.shape) == 2:  # [seq_len, value_dim]
            v_hist_tensor = _get_tf().expand_dims(v_hist_tensor, 0)  # Add batch dimension [1, seq_len, value_dim]
        
        v_tensor = _get_tf().convert_to_tensor(v_t, dtype='float32')
        if len(v_tensor.shape) == 1:
            v_tensor = _get_tf().expand_dims(v_tensor, 0)  # Add batch dimension
        
        # Apply attention mechanism
        try:
            attended_v_tensor = self.attention_module(
                query=q_tensor,
                key=k_hist_tensor,
                value=v_hist_tensor,
                training=False,
            )
            
            # Combine attended and current values
            if self.v_prime_gate is None:
                # Initialize projection layers if not already done
                self.init_value_projection_layers(v_tensor.shape[-1])
            
            # Concatenate vectors for gating
            concat_v = _get_tf().concat([v_tensor, attended_v_tensor], axis=-1)
            
            # Calculate gate value
            gate = self.v_prime_gate(concat_v)
            
            # Combine original and attended values
            v_prime_tensor = gate * v_tensor + (1 - gate) * attended_v_tensor
            
            # Final projection
            v_prime_tensor = self.v_prime_projector(v_prime_tensor)
            
            # Convert back to numpy
            v_prime = v_prime_tensor.numpy()
            if len(v_prime.shape) > 1:
                v_prime = v_prime[0]  # Remove batch dimension
            
            logger.info(f"MAL: Generated augmented value projection from {len(k_hist)} historical values")
            
            return {
                "memory_id": memory_id,
                "attended_output": y_t,  # No change in output
                "v_prime": v_prime,  # Augmented value projection
                "original_v": v_t,  # Original for comparison
                "metrics": self.attention_module.get_metrics(),
            }
            
        except Exception as e:
            logger.error(f"MAL attention failed: {e}")
            # Fallback to original value projection
            return {
                "memory_id": memory_id,
                "attended_output": y_t,
                "v_prime": v_t,  # Fallback to original
                "error": str(e),
                "metrics": {},
            }


def create_titans_variant(variant_type: TitansVariantType, attention_config: Optional[Dict[str, Any]] = None) -> TitansVariantBase:
    """Factory function to create a Titans variant instance based on type.
    
    Args:
        variant_type: Type of variant to create (MAC, MAG, MAL, or NONE)
        attention_config: Configuration dictionary for attention parameters
        
    Returns:
        An instance of the requested variant type
    """
    if variant_type == TitansVariantType.NONE:
        return TitansVariantBase(attention_config)
    elif variant_type == TitansVariantType.MAC:
        return MACVariant(attention_config)
    elif variant_type == TitansVariantType.MAG:
        return MAGVariant(attention_config)
    elif variant_type == TitansVariantType.MAL:
        return MALVariant(attention_config)
    else:
        raise ValueError(f"Unknown variant type: {variant_type}")
