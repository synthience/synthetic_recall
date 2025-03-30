#!/usr/bin/env python

from enum import Enum
import logging
import sys
import threading
import time
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING

# Set recursion limit higher to handle potential deep call stacks
sys.setrecursionlimit(5000)

# Configure logger
logger = logging.getLogger(__name__)

# Use TYPE_CHECKING for type hints that won't be evaluated at runtime
if TYPE_CHECKING:
    import tensorflow as tf
    import numpy as np
else:
    # Placeholders for module imports that will be lazily loaded
    tf = None
    np = None

# Lazy-load TensorFlow to avoid NumPy incompatibility issues during startup
_tf = None
_tf_lock = threading.Lock()

def _get_tf():
    """Lazy-load TensorFlow only when needed.
    
    Returns:
        The tensorflow module if successfully loaded, None otherwise.
    """
    global tf
    if tf is None:
        try:
            import tensorflow as tensorflow_module
            tf = tensorflow_module
            logger.debug(f"Successfully imported TensorFlow version {tf.__version__}")
        except ImportError as e:
            logger.error(f"Error importing TensorFlow: {e}")
            return None
    return tf

def _get_numpy():
    """Lazy-load NumPy only when needed.
    
    Returns:
        The numpy module if successfully loaded, None otherwise.
    """
    global np
    if np is None:
        try:
            import numpy as numpy_module
            np = numpy_module
            logger.debug(f"Successfully imported NumPy version {np.__version__}")
        except ImportError as e:
            logger.error(f"Error importing NumPy: {e}")
            try:
                # Try direct import as fallback
                import numpy
                np = numpy
                logger.warning(f"Successfully imported NumPy via fallback, version {np.__version__}")
            except ImportError as e2:
                logger.error(f"Direct NumPy import also failed: {e2}")
                return None
    return np

def init_variants_module():
    """Initialize the variants module by setting up lazy imports.
    
    This function configures the module to use lazy loading for TensorFlow and NumPy
    to avoid import-time recursion issues that can occur when these libraries
    are imported during class definition.
    """
    global tf, np
    
    # Don't do anything if we're in TYPE_CHECKING mode
    if TYPE_CHECKING:
        return
        
    # Set placeholders to None initially
    tf = None
    np = None
    
    logger.info("Titans variants module initialized with lazy loading")

# Call the initialization function at import time
init_variants_module()

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
        # Initialize with defaults first, then override with provided values
        super().__init__(defaults)
        
        # Update with any positional dict args
        for arg in args:
            if isinstance(arg, dict):
                self.update(arg)
        
        # Update with any keyword args
        self.update(kwargs)


class TitansVariantBase:
    """Base class for all Titans architecture variants."""
    
    def __init__(self, config: Optional[Union[TitansVariantConfig, Dict]] = None, **kwargs):
        """Initialize the base Titans variant.
        
        Args:
            config: Optional configuration dictionary for attention parameters.
        """
        if isinstance(config, dict) or config is None: 
            self.config = TitansVariantConfig(**(config or {}))
        elif isinstance(config, TitansVariantConfig): 
            self.config = config
        else: 
            raise TypeError("config must be a dict or TitansVariantConfig")
            
        self.variant_type = TitansVariantType.NONE
        self.name = "NONE"
        self.sequence_context = None
        self.neural_memory_url = None
        self.api_client = None
    
    def set_sequence_context(self, sequence_context):
        """Set the sequence context manager for historical attention context.
        
        Args:
            sequence_context: SequenceContextManager instance to use for context history.
        """
        self.sequence_context = sequence_context
        logger.info(f"{self.name}: Sequence context manager set, max_length={sequence_context.max_length}")
    
    def set_neural_memory_url(self, neural_memory_url: str) -> None:
        """Set the Neural Memory server URL and initialize API client.
        
        Args:
            neural_memory_url: URL to the Neural Memory server
        """
        self.neural_memory_url = neural_memory_url
        
        # Initialize the API client for making requests to Neural Memory server
        try:
            # Try importing from direct path first
            try:
                from synthians_memory_core.synthians_trainer_server.api_client import NeuralMemoryClient
            except ImportError:
                # Try fallback import paths
                try:
                    from synthians_trainer_server.api_client import NeuralMemoryClient
                except ImportError:
                    # Final fallback - create a simple HTTP client if all else fails
                    import aiohttp
                    
                    class SimpleNeuralMemoryClient:
                        def __init__(self, base_url):
                            self.base_url = base_url
                            self.session = None
                            
                        async def _ensure_session(self):
                            if self.session is None or self.session.closed:
                                self.session = aiohttp.ClientSession()
                            return self.session
                                
                        async def post(self, endpoint, json=None):
                            session = await self._ensure_session()
                            async with session.post(f"{self.base_url}{endpoint}", json=json) as response:
                                return await response.json()
                                
                    NeuralMemoryClient = SimpleNeuralMemoryClient
                    logger.warning(f"Using fallback SimpleNeuralMemoryClient for {self.name} variant")
            
            self.api_client = NeuralMemoryClient(base_url=neural_memory_url)
            logger.info(f"Initialized API client for {self.name} variant with Neural Memory URL: {neural_memory_url}")
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}", exc_info=True)
    
    def store_context(self, memory_id: str, x_t: Any, k_t: Any, 
                    v_t: Any, q_t: Any, y_t: Any) -> None:
        """Store context tuple in the sequence context manager.
        
        This helper method adds the current context to the sequence context manager,
        which is used by all variant implementations to track historical context.
        
        Args:
            memory_id: ID of the memory being processed
            x_t: Original input embedding
            k_t: Key projection
            v_t: Value projection
            q_t: Query projection
            y_t: Retrieved embedding from Neural Memory
        """
        if self.sequence_context is None:
            logger.warning(f"Cannot store context: sequence_context is not set for {self.name} variant")
            return
            
        self.sequence_context.add_context(memory_id, x_t, k_t, v_t, q_t, y_t)
    
    async def process_input(self, memory_id: str, x_t: Any, k_t: Any, 
                      v_t: Any, q_t: Any, y_t: Any) -> Dict[str, Any]:
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
        # Store the current context
        try:
            # Convert to numpy arrays if needed
            np = _get_numpy()
            if np is None:
                logger.warning(f"{self.name}: NumPy not available, skipping context storage")
            else:
                # Convert inputs to numpy arrays for the sequence context
                x_t_np = np.asarray(x_t, dtype=np.float32) if not isinstance(x_t, np.ndarray) else x_t
                k_t_np = np.asarray(k_t, dtype=np.float32) if not isinstance(k_t, np.ndarray) else k_t
                v_t_np = np.asarray(v_t, dtype=np.float32) if not isinstance(v_t, np.ndarray) else v_t
                q_t_np = np.asarray(q_t, dtype=np.float32) if not isinstance(q_t, np.ndarray) else q_t
                y_t_np = np.asarray(y_t, dtype=np.float32) if not isinstance(y_t, np.ndarray) else y_t
                
                self.store_context(memory_id, x_t_np, k_t_np, v_t_np, q_t_np, y_t_np)
        except Exception as e:
            logger.error(f"{self.name}: Error storing context: {e}", exc_info=True)
            # Continue processing even if context storage fails
        
        # Base implementation just returns y_t unchanged
        return {
            "memory_id": memory_id,
            "attended_output": y_t,
            "metrics": {}
        }


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
        
        # Store attention config for lazy initialization
        self._attention_config = {
            "num_heads": self.config.get("attention_num_heads", 4),
            "key_dim": self.config.get("attention_key_dim", 32),
            "dropout": self.config.get("attention_dropout", 0.0),
            "max_dim_mismatch_warnings": self.config.get("max_dim_mismatch_warnings", 10),
        }
        
        # Defer creation of attention module to avoid import-time recursion
        self._attention_initialized = False
        self.attention_module = None
        
        logger.info(f"Initialized MAC variant with config for {self._attention_config['num_heads']} attention heads")
    
    def _initialize_attention(self):
        """Lazily initialize the attention module to avoid import-time recursion"""
        if self._attention_initialized:
            return
            
        try:
            tf = _get_tf()
            if tf is None:
                logger.error("MAC: Failed to initialize attention module - TensorFlow not available")
                return
                
            self.attention_module = tf.keras.layers.MultiHeadAttention(
                num_heads=self._attention_config["num_heads"],
                key_dim=self._attention_config["key_dim"],
                dropout=self._attention_config["dropout"],
                name="MAC_Attention"
            )
            self._attention_initialized = True
            logger.info("MAC: Successfully initialized attention module")
        except Exception as e:
            logger.error(f"MAC: Error initializing attention module: {e}", exc_info=True)

    async def process_input(
        self,
        memory_id: str,
        x_t: Any, 
        k_t: Any,
        v_t: Any,
        q_t: Any,
        y_t: Any,
    ) -> Dict[str, Any]:
        """Implement MAC variant logic.
        
        1. Store context tuple (timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)
        2. Retrieve recent history pairs (k_i, y_i) from sequence_context
        3. Calculate attended output using attention module: attended_y_t = AttentionModule(q_t, K_hist, Y_hist)
        4. Return attended_y_t for use by downstream components
        """
        # Store the current context
        try:
            # Convert to numpy arrays if needed
            np = _get_numpy()
            if np is None:
                logger.warning("MAC: NumPy not available, cannot store context")
                return {
                    "memory_id": memory_id,
                    "attended_output": y_t,  # No change
                    "metrics": {"error": "numpy_not_available"},
                }
                
            # Convert inputs to numpy arrays for the sequence context
            x_t_np = np.asarray(x_t, dtype=np.float32) if not isinstance(x_t, np.ndarray) else x_t
            k_t_np = np.asarray(k_t, dtype=np.float32) if not isinstance(k_t, np.ndarray) else k_t
            v_t_np = np.asarray(v_t, dtype=np.float32) if not isinstance(v_t, np.ndarray) else v_t
            q_t_np = np.asarray(q_t, dtype=np.float32) if not isinstance(q_t, np.ndarray) else q_t
            y_t_np = np.asarray(y_t, dtype=np.float32) if not isinstance(y_t, np.ndarray) else y_t
            
            self.store_context(memory_id, x_t_np, k_t_np, v_t_np, q_t_np, y_t_np)
        except Exception as e:
            logger.error(f"MAC: Error storing context: {e}")
            # Continue processing even if context storage fails
        
        # If no context history or context too short
        if not self.sequence_context or len(self.sequence_context) < 2:
            logger.info("MAC: Not enough context for attention, using original output")
            return {
                "memory_id": memory_id,
                "attended_output": y_t,  # No change
                "metrics": {},
            }
        
        # Initialize attention module if not already done
        self._initialize_attention()
        
        # If attention module initialization failed, use original output
        if not self._attention_initialized or self.attention_module is None:
            logger.warning("MAC: Attention module not available, using original output")
            return {
                "memory_id": memory_id,
                "attended_output": y_t,  # No change
                "metrics": {"attention_error": "module_not_initialized"},
            }
        
        # Get historical keys and memory outputs
        k_hist, y_hist = self.sequence_context.get_recent_ky_pairs(count=len(self.sequence_context) - 1)  # Exclude current
        
        try:
            # Get TensorFlow only when needed
            tf = _get_tf()
            if tf is None:
                logger.warning("MAC: TensorFlow not available, using original output")
                return {
                    "memory_id": memory_id,
                    "attended_output": y_t,  # No change
                    "metrics": {"attention_error": "tensorflow_not_available"},
                }
                
            # Convert inputs to tensors
            q_tensor = tf.convert_to_tensor(q_t, dtype='float32')
            if len(q_tensor.shape) == 1:
                q_tensor = tf.expand_dims(q_tensor, 0)  # Add batch dimension
                
            k_hist_tensor = tf.convert_to_tensor(k_hist, dtype='float32')
            if len(k_hist_tensor.shape) == 2:  # [seq_len, key_dim]
                k_hist_tensor = tf.expand_dims(k_hist_tensor, 0)  # Add batch dimension [1, seq_len, key_dim]
                
            y_hist_tensor = tf.convert_to_tensor(y_hist, dtype='float32')
            if len(y_hist_tensor.shape) == 2:  # [seq_len, value_dim]
                y_hist_tensor = tf.expand_dims(y_hist_tensor, 0)  # Add batch dimension [1, seq_len, value_dim]
            
            # Apply attention
            attended_output_tensor = self.attention_module(
                query=q_tensor,
                key=k_hist_tensor,
                value=y_hist_tensor,
                training=False,
            )
            
            # Convert back to numpy array
            attended_output = attended_output_tensor.numpy()
            
            # Remove batch dimension if it exists
            if len(attended_output.shape) > 1 and attended_output.shape[0] == 1:
                attended_output = attended_output[0]
                
            logger.info(f"MAC: Generated attended output using {len(k_hist)} historical values")
            
            return {
                "memory_id": memory_id,
                "attended_output": attended_output,
                "metrics": {
                    "attention_applied": True,
                    "history_size": len(k_hist),
                    "original_shape": str(y_t.shape) if hasattr(y_t, 'shape') else "unknown",
                    "attended_shape": str(attended_output.shape) if hasattr(attended_output, 'shape') else "unknown"
                },
            }
            
        except Exception as e:
            logger.error(f"MAC: Error applying attention: {e}", exc_info=True)
            # Fallback to original output
            return {
                "memory_id": memory_id,
                "attended_output": y_t,  # No change
                "metrics": {"attention_error": str(e)},
            }


class MAGVariant(TitansVariantBase):
    """Memory-Attended Gates (MAG) variant.
    
    Modifies gate values (alpha, theta, eta) for the neural memory update
    by attending over historical key projections.
    
    Flow: 
    1. q_t -> Attend(q_t, K_hist, K_hist) -> attention_output
    2. Call Neural Memory's /calculate_gates endpoint with attention output
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
        
        # Lazily initialize the TensorFlow components to avoid recursion
        self._attention_initialized = False
        self._attention_config = attention_config
        self.attention_module = None
        
        logger.info(f"MAG: Initialized with config for {attention_config['num_heads']} attention heads")
        
    def _initialize_attention(self):
        """Lazily initialize the attention module to avoid import-time recursion"""
        if self._attention_initialized:
            return
            
        tf = _get_tf()
        if tf is None:
            logger.error("MAG: Failed to initialize attention - TensorFlow not available")
            return
            
        try:
            logger.info("MAG: Initializing TensorFlow attention module")
            self.attention_module = tf.keras.layers.MultiHeadAttention(
                num_heads=self._attention_config["num_heads"],
                key_dim=self._attention_config["key_dim"],
                dropout=self._attention_config["dropout"],
                name="MAG_Attention"
            )
            self._attention_initialized = True
            logger.info("MAG: Successfully initialized attention module")
        except Exception as e:
            logger.error(f"MAG: Error initializing attention module: {e}", exc_info=True)
    
    async def process_input(
            self,
            memory_id: str,
            x_t: Any, 
            k_t: Any,
            v_t: Any,
            q_t: Any,
            y_t: Any,
        ) -> Dict[str, Any]:
        """Implement MAG variant logic.
        
        1. Store context tuple (timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)
        2. Calculate attention gates based on history (alpha, theta, eta)
        3. Return gates for use by neural memory during update step
        """
        # Store the current context
        try:
            # Convert to numpy arrays if needed
            np = _get_numpy()
            if np is None:
                logger.warning("MAG: NumPy not available, cannot store context")
                return {
                    "memory_id": memory_id,
                    "attended_output": y_t,  # No change
                    "metrics": {"error": "numpy_not_available"},
                }
                
            # Convert inputs to numpy arrays for the sequence context
            x_t_np = np.asarray(x_t, dtype=np.float32) if not isinstance(x_t, np.ndarray) else x_t
            k_t_np = np.asarray(k_t, dtype=np.float32) if not isinstance(k_t, np.ndarray) else k_t
            v_t_np = np.asarray(v_t, dtype=np.float32) if not isinstance(v_t, np.ndarray) else v_t
            q_t_np = np.asarray(q_t, dtype=np.float32) if not isinstance(q_t, np.ndarray) else q_t
            y_t_np = np.asarray(y_t, dtype=np.float32) if not isinstance(y_t, np.ndarray) else y_t
            
            self.store_context(memory_id, x_t_np, k_t_np, v_t_np, q_t_np, y_t_np)
        except Exception as e:
            logger.error(f"MAG: Error storing context: {e}")
            # Continue processing even if context storage fails
        
        # Check if we have enough context for attention
        if not self.sequence_context or len(self.sequence_context) < 2:
            logger.info("MAG: Not enough context for attention, using default gates")
            return {
                "memory_id": memory_id,
                "gates": None,  # No external gates
                "metrics": {}
            }
            
        # Initialize attention module if not already done
        self._initialize_attention()
        
        # If attention module initialization failed, use original output
        if not self._attention_initialized or not self.attention_module:
            logger.error("MAG: Attention module not initialized, cannot process input")
            return {
                "memory_id": memory_id,
                "gates": None,
                "error": "Attention module not initialized",
                "metrics": {}
            }
        
        # Get historical keys for attention
        k_hist, _ = self.sequence_context.get_recent_kv_pairs(count=len(self.sequence_context) - 1)  # Exclude current
        
        try:
            # Get TensorFlow
            tf = _get_tf()
            if tf is None:
                logger.error("MAG: TensorFlow not available, using default gates")
                return {
                    "memory_id": memory_id,
                    "gates": None,
                    "error": "TensorFlow not available",
                    "metrics": {}
                }
            
            # Convert numpy arrays to tensors for TF operations
            q_tensor = tf.convert_to_tensor(q_t, dtype='float32')
            if len(q_tensor.shape) == 1:
                q_tensor = tf.expand_dims(q_tensor, 0)  # Add batch dimension
                
            k_hist_tensor = tf.convert_to_tensor(k_hist, dtype='float32')
            if len(k_hist_tensor.shape) == 2:  # [seq_len, key_dim]
                k_hist_tensor = tf.expand_dims(k_hist_tensor, 0)  # Add batch dimension
            
            # Apply self-attention mechanism (keys as values)
            attention_output = self.attention_module(
                query=q_tensor,
                key=k_hist_tensor,
                value=k_hist_tensor,  # Self-attention uses keys as values
                training=False,
            )
            
            # Convert back to numpy for API call
            attention_output_np = attention_output.numpy()
            if len(attention_output_np.shape) > 1:
                attention_output_np = attention_output_np[0]  # Remove batch dimension
                
            logger.info(f"MAG: Applied attention to {len(k_hist)} historical memories")
            
            # Call Neural Memory's /calculate_gates endpoint with attention output
            if not self.neural_memory_url:
                logger.error("MAG: Neural Memory URL not set")
                return {
                    "memory_id": memory_id,
                    "gates": None,
                    "error": "Neural Memory URL not set",
                    "metrics": {}
                }
                
            try:
                gate_response = self._make_request(
                    "calculate_gates",
                    payload={
                        "attention_output": attention_output_np.tolist()
                    }
                )
                
                if not gate_response or "gates" not in gate_response:
                    logger.error(f"MAG: Failed to get gates from Neural Memory: {gate_response}")
                    return {
                        "memory_id": memory_id,
                        "gates": None,
                        "error": "Failed to get gates",
                        "metrics": {}
                    }
                    
                logger.info(f"MAG: Got gates from Neural Memory: {gate_response['gates']}")
                
                # Return gates for use in neural memory update
                return {
                    "memory_id": memory_id,
                    "gates": gate_response["gates"],
                    "attention_output": attention_output_np.tolist(),
                    "metrics": {}
                }
                
            except Exception as e:
                logger.error(f"MAG: Error calling Neural Memory: {e}")
                return {
                    "memory_id": memory_id,
                    "gates": None,
                    "error": f"Error calling Neural Memory: {e}",
                    "metrics": {}
                }
            
        except Exception as e:
            logger.error(f"MAG: Error calculating gates: {e}", exc_info=True)
            return {
                "memory_id": memory_id,
                "gates": None,
                "error": str(e),
                "metrics": {}
            }
            
    def _make_request(self, endpoint: str, payload: Dict = None) -> Dict:
        """Make a request to the Neural Memory API"""
        import aiohttp
        import json
        
        if not self.neural_memory_url:
            logger.error("Neural Memory URL not set")
            return {"error": "Neural Memory URL not set"}
            
        url = f"{self.neural_memory_url}/{endpoint}"
        try:
            import requests
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                logger.error(f"Error from Neural Memory: {response.status_code}, {response.text}")
                return {"error": f"Neural Memory API error: {response.status_code}"}
                
            return response.json()
        except Exception as e:
            logger.error(f"Error calling Neural Memory: {e}")
            return {"error": str(e)}


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
        
        # Lazily initialize the TensorFlow components to avoid recursion
        self._attention_initialized = False
        self._attention_config = attention_config
        self.attention_module = None
        
        # Gating layers for combining attended and current values (initialized when dimensions are known)
        self.v_prime_gate = None
        self.v_prime_projector = None
        
        logger.info(f"Initialized MAL variant with config for {attention_config['num_heads']} attention heads")
        
    def _initialize_attention(self):
        """Lazily initialize the attention module to avoid import-time recursion"""
        if self._attention_initialized:
            return
            
        tf = _get_tf()
        if tf is None:
            logger.error("MAL: Failed to initialize attention - TensorFlow not available")
            return
            
        try:
            logger.info("MAL: Initializing TensorFlow attention module")
            self.attention_module = tf.keras.layers.MultiHeadAttention(
                num_heads=self._attention_config["num_heads"],
                key_dim=self._attention_config["key_dim"],
                dropout=self._attention_config["dropout"],
                name="MAL_Attention"
            )
            self._attention_initialized = True
            logger.info("MAL: Successfully initialized attention module")
        except Exception as e:
            logger.error(f"MAL: Error initializing attention module: {e}", exc_info=True)
    
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

    def calculate_v_prime(self, q_t: Any, v_t: Any, k_hist: List[Any], v_hist: List[Any]) -> Dict[str, Any]:
        """Calculate modified value projection using attention over historical values.
        
        This method is specifically called by the ContextCascadeEngine._apply_variant_pre_update
        method to get a modified value projection for use in the Neural Memory update.
        
        Args:
            q_t: Query projection for the current input
            v_t: Original value projection for the current input
            k_hist: Historical key projections to attend over
            v_hist: Historical value projections to attend over
            
        Returns:
            Dict containing v_prime_t (modified value projection) and metrics
        """
        if not k_hist or not v_hist:
            logger.warning("MAL: No historical data available for attention. Returning original v_t.")
            return {"v_prime_t": v_t, "metrics": {"attended": False, "reason": "no_history"}}
        
        # Ensure the attention module is initialized
        if not self._attention_initialized:
            self._initialize_attention()
            
        # If attention initialization failed, fall back to original values
        if not self._attention_initialized or self.attention_module is None:
            logger.warning("MAL: Attention module not available. Returning original v_t.")
            return {"v_prime_t": v_t, "metrics": {"attended": False, "reason": "no_attention_module"}}
            
        # Get TensorFlow only when needed
        try:
            tf = _get_tf()
            if tf is None:
                logger.warning("MAL: TensorFlow not available. Returning original v_t.")
                return {"v_prime_t": v_t, "metrics": {"attended": False, "reason": "no_tensorflow"}}
            
            # Convert numpy arrays to tensors for TF operations
            q_tensor = tf.convert_to_tensor(q_t, dtype='float32')
            if len(q_tensor.shape) == 1:
                q_tensor = tf.expand_dims(q_tensor, 0)  # Add batch dimension
                
            k_hist_tensor = tf.convert_to_tensor(k_hist, dtype='float32')
            if len(k_hist_tensor.shape) == 2:  # [seq_len, key_dim]
                k_hist_tensor = tf.expand_dims(k_hist_tensor, 0)  # Add batch dimension [1, seq_len, key_dim]
                
            v_hist_tensor = tf.convert_to_tensor(v_hist, dtype='float32')
            if len(v_hist_tensor.shape) == 2:  # [seq_len, value_dim]
                v_hist_tensor = tf.expand_dims(v_hist_tensor, 0)  # Add batch dimension [1, seq_len, value_dim]
            
            v_tensor = tf.convert_to_tensor(v_t, dtype='float32')
            if len(v_tensor.shape) == 1:
                v_tensor = tf.expand_dims(v_tensor, 0)  # Add batch dimension
            
            # Apply attention mechanism
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
            concat_v = tf.concat([v_tensor, attended_v_tensor], axis=-1)
            
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
                "v_prime_t": v_prime,  # Augmented value projection
                "metrics": {}
            }
            
        except Exception as e:
            logger.error(f"MAL calculate_v_prime failed: {str(e)}", exc_info=True)
            # Fallback to original value projection
            return {
                "v_prime_t": v_t,  # Fallback to original
                "metrics": {}
            }

    async def process_input(
        self,
        memory_id: str,
        x_t: Any, 
        k_t: Any,
        v_t: Any,
        q_t: Any,
        y_t: Any,
    ) -> Dict[str, Any]:
        """Implement MAL variant logic.
        
        1. Store context tuple (timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)
        2. Retrieve recent history v/k pairs from sequence_context
        3. Calculate attended v_prime using attention module
        4. Return v_prime for use by Neural Memory during update
        """
        # Store the current context
        try:
            # Convert to numpy arrays if needed
            np = _get_numpy()
            if np is None:
                logger.warning("MAL: NumPy not available, cannot store context")
                return {
                    "memory_id": memory_id,
                    "attended_output": y_t,  # No change
                    "v_prime": v_t,  # Original value projection as fallback
                    "original_v": v_t,
                    "metrics": {"error": "numpy_not_available"},
                }
                
            # Convert inputs to numpy arrays for the sequence context
            x_t_np = np.asarray(x_t, dtype=np.float32) if not isinstance(x_t, np.ndarray) else x_t
            k_t_np = np.asarray(k_t, dtype=np.float32) if not isinstance(k_t, np.ndarray) else k_t
            v_t_np = np.asarray(v_t, dtype=np.float32) if not isinstance(v_t, np.ndarray) else v_t
            q_t_np = np.asarray(q_t, dtype=np.float32) if not isinstance(q_t, np.ndarray) else q_t
            y_t_np = np.asarray(y_t, dtype=np.float32) if not isinstance(y_t, np.ndarray) else y_t
            
            self.store_context(memory_id, x_t_np, k_t_np, v_t_np, q_t_np, y_t_np)
        except Exception as e:
            logger.error(f"MAL: Error storing context: {e}")
            # Continue processing even if context storage fails
        
        # Get historical contexts for attention
        if len(self.sequence_context) < 2:
            # Not enough context for attention, return original values
            logger.info("MAL: Not enough context for attention, using original value projection")
            return {
                "memory_id": memory_id,
                "attended_output": y_t,  # No change in output
                "v_prime": v_t,  # No change
                "original_v": v_t,
                "metrics": {},
            }
        
        # Get historical keys and values
        k_hist, v_hist = self.sequence_context.get_recent_kv_pairs(count=len(self.sequence_context) - 1)  # Exclude current
        
        # Initialize attention module if not already done
        self._initialize_attention()
        
        # Convert numpy arrays to tensors for TF operations
        q_tensor = _get_tf().convert_to_tensor(q_t, dtype='float32')
        if len(q_tensor.shape) == 1:
            q_tensor = _get_tf().expand_dims(q_tensor, 0)  # Add batch dimension
            
        k_hist_tensor = _get_tf().convert_to_tensor(k_hist, dtype='float32')
        if len(k_hist_tensor.shape) == 2:  # [seq_len, key_dim]
            k_hist_tensor = _get_tf().expand_dims(k_hist_tensor, 0)  # Add batch dimension
            
        v_hist_tensor = _get_tf().convert_to_tensor(v_hist, dtype='float32')
        if len(v_hist_tensor.shape) == 2:  # [seq_len, value_dim]
            v_hist_tensor = _get_tf().expand_dims(v_hist_tensor, 0)  # Add batch dimension
        
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
                "metrics": {},
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
    logger.info(f"Creating Titans variant of type: {variant_type}")
    
    try:
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
    except Exception as e:
        logger.error(f"Error creating variant {variant_type}: {e}", exc_info=True)
        # Return the base variant as a fallback
        return TitansVariantBase(attention_config)
