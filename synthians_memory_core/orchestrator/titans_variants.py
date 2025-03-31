#!/usr/bin/env python

from enum import Enum
import logging
import sys
import threading
import time
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING
import datetime

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
    """Get TensorFlow module with error handling.
    
    Returns:
        TensorFlow module or None if not available
    """
    try:
        # Try importing with increased recursion limit to avoid the circular import issue
        import sys
        default_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)  # Temporarily increase the recursion limit
        
        try:
            import tensorflow as tensorflow_module
            return tensorflow_module
        finally:
            # Always restore the original recursion limit
            sys.setrecursionlimit(default_limit)
    except Exception as e:
        logger.error(f"Error importing TensorFlow: {e}")
        return None

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
        
        # Convert inputs to NumPy arrays before adding to context
        try:
            np = _get_numpy()
            if np is None:
                logger.warning(f"{self.name}: NumPy not available, skipping context storage")
                return  # Exit if numpy cannot be loaded

            # Convert ALL inputs to numpy arrays robustly *before* adding
            # Use empty arrays as fallbacks if conversion fails
            try:
                x_t_np = np.asarray(x_t, dtype=np.float32) if x_t is not None else np.zeros(1, dtype=np.float32)
            except Exception as e:
                logger.warning(f"{self.name}: Error converting x_t to numpy array: {e}, using zeros")
                x_t_np = np.zeros(1, dtype=np.float32)
                
            try:
                k_t_np = np.asarray(k_t, dtype=np.float32) if k_t is not None else np.zeros(1, dtype=np.float32)
            except Exception as e:
                logger.warning(f"{self.name}: Error converting k_t to numpy array: {e}, using zeros")
                k_t_np = np.zeros(1, dtype=np.float32)
                
            try:
                v_t_np = np.asarray(v_t, dtype=np.float32) if v_t is not None else np.zeros(1, dtype=np.float32)
            except Exception as e:
                logger.warning(f"{self.name}: Error converting v_t to numpy array: {e}, using zeros")
                v_t_np = np.zeros(1, dtype=np.float32)
                
            try:
                q_t_np = np.asarray(q_t, dtype=np.float32) if q_t is not None else np.zeros(1, dtype=np.float32)
            except Exception as e:
                logger.warning(f"{self.name}: Error converting q_t to numpy array: {e}, using zeros")
                q_t_np = np.zeros(1, dtype=np.float32)
                
            try:
                y_t_np = np.asarray(y_t, dtype=np.float32) if y_t is not None else np.zeros(1, dtype=np.float32)
            except Exception as e:
                logger.warning(f"{self.name}: Error converting y_t to numpy array: {e}, using zeros")
                y_t_np = np.zeros(1, dtype=np.float32)

            # Now call add_context with guaranteed numpy arrays
            self.sequence_context.add_context(memory_id, x_t_np, k_t_np, v_t_np, q_t_np, y_t_np)
            logger.debug(f"{self.name}: Successfully stored context for memory {memory_id} (context size: {len(self.sequence_context)})")
            
        except Exception as e:
            logger.error(f"{self.name}: Error storing context: {e}", exc_info=True)
            # We don't re-raise the error as we want to continue processing even if context storage fails
    
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
        2. Apply attention to retrieved embedding y_t using historical context
        3. Return modified y_t for use by CCE
        
        Args:
            memory_id: ID of the memory being processed
            x_t: Original input embedding
            k_t: Key projection
            v_t: Value projection
            q_t: Query projection
            y_t: Retrieved embedding from NM
            
        Returns:
            Dict with:
                - 'y_t_final': Modified output (may be identical to input)
                - 'metrics': Dictionary with attention metrics
                - 'success': Boolean indicating if processing succeeded
        """
        # Initialize metrics with required fields
        metrics = {
            "attention_applied": False,
            "attended_output_generated": False,
            "history_size_used": 0,
            "fallback_mode": False,
        }
            
        # Store context in history - using the base class method to ensure consistency
        # First ensure we have numpy arrays (done directly in the base class method)
        self.store_context(memory_id, x_t, k_t, v_t, q_t, y_t)
        
        # Abort early if the output is None or invalid
        if y_t is None:
            logger.error(f"MAC: Invalid y_t (None) for memory {memory_id}")
            metrics["error"] = "Invalid y_t (None)"
            return {"y_t_final": y_t, "metrics": metrics, "success": False}
        
        # Make sure inputs are NumPy arrays
        try:
            x_t = self._ensure_numpy(x_t)
            k_t = self._ensure_numpy(k_t)
            v_t = self._ensure_numpy(v_t) 
            q_t = self._ensure_numpy(q_t)
            y_t = self._ensure_numpy(y_t)
        except Exception as e:
            logger.error(f"MAC: Error converting inputs to NumPy arrays: {e}")
            metrics["error"] = f"Error converting inputs: {str(e)}"
            return {"y_t_final": y_t, "metrics": metrics, "success": False}
        
        try:
            # Get historical context using synchronous method
            try:
                ky_pairs = self.sequence_context.get_recent_ky_pairs(max_pairs=20) 
                # Note: Removed await since this should be synchronous
            except AttributeError:
                # Fallback if method doesn't exist
                logger.warning("MAC: get_recent_ky_pairs not available, trying get_history")
                history = self.sequence_context.get_history()
                if not history:
                    ky_pairs = []
                else:
                    # Extract k,y pairs from history
                    ky_pairs = []
                    for entry in history:
                        if len(entry) >= 6:  # Ensure we have enough elements
                            # Typical format is (timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)
                            # We need k_t (index 3) and y_t (index 6)
                            k = entry[3] if len(entry) > 3 else None
                            y = entry[6] if len(entry) > 6 else None
                            if k is not None and y is not None:
                                ky_pairs.append((k, y))
            
            metrics["history_size_used"] = len(ky_pairs)
            
            # If history is empty, return original y_t
            if not ky_pairs:
                logger.info("MAC: No historical context available, using original output")
                metrics["fallback_mode"] = True
                return {"y_t_final": y_t, "metrics": metrics, "success": True}
                
            # Initialize attention if needed
            if not self._initialize_attention():
                logger.warning("MAC: Attention module not initialized, using original output")
                metrics["error"] = self._attention_error or "Attention initialization failed"
                metrics["fallback_mode"] = True
                return {"y_t_final": y_t, "metrics": metrics, "success": True}  # Return success=True with fallback
            
            # Get TensorFlow and apply attention
            tf = _get_tf()
            if tf is None:
                logger.error("MAC: TensorFlow not available for attention")
                metrics["error"] = "TensorFlow not available"
                metrics["fallback_mode"] = True
                return {"y_t_final": y_t, "metrics": metrics, "success": True}  # Return success=True with fallback
            
            # Convert k,y pairs to tensors
            k_history = np.vstack([pair[0] for pair in ky_pairs])
            y_history = np.vstack([pair[1] for pair in ky_pairs])
            
            # Ensure consistent shape with q_t (add batch dimension)
            q_t_reshaped = q_t.reshape(1, -1) 
            k_hist_tensor = tf.convert_to_tensor(k_history, dtype=tf.float32)
            y_hist_tensor = tf.convert_to_tensor(y_history, dtype=tf.float32)
            
            # Apply attention mechanism
            attended_output = self.attention_module(
                query=q_t_reshaped,
                key=k_hist_tensor,
                value=y_hist_tensor
            )
            
            # Convert back to numpy and remove batch dimension
            attended_output_np = attended_output.numpy().flatten()
            
            # Update metrics
            metrics["attention_applied"] = True
            metrics["attended_output_generated"] = True
            
            return {"y_t_final": attended_output_np, "metrics": metrics, "success": True}
            
        except Exception as e:
            logger.error(f"MAC: Error in attention processing: {str(e)}")
            # Ensure metrics includes the required fields even in error state
            metrics["error"] = f"Error in attention processing: {str(e)}"
            metrics["fallback_mode"] = True
            return {"y_t_final": y_t, "metrics": metrics, "success": False}

    def _ensure_numpy(self, x):
        """Ensure input is a NumPy array"""
        try:
            return np.asarray(x, dtype=np.float32)
        except Exception as e:
            logger.error(f"MAC: Error converting input to NumPy array: {e}")
            return x


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
        self._attention_lock = threading.Lock()
        self._attention_error = None
        
        logger.info(f"MAG: Initialized with config for {attention_config['num_heads']} attention heads")
        
    def _initialize_attention(self):
        """Initialize TensorFlow attention module.
        
        This is done lazily to minimize startup time and memory usage.
        """
        # Only initialize once
        if self._attention_initialized:
            return True
            
        with self._attention_lock:
            if self._attention_initialized:
                return True
                
            try:
                # Get TensorFlow with robust error handling
                tf = _get_tf()
                if tf is None:
                    logger.error("Could not import TensorFlow, attention will be unavailable")
                    self._attention_error = "TensorFlow import failed"
                    return False
                    
                # Get TensorFlow version
                tf_version = tf.__version__
                logger.debug(f"Using TensorFlow {tf_version} for attention")
                
                # Check if MultiHeadAttention exists
                if not hasattr(tf.keras.layers, 'MultiHeadAttention'):
                    logger.error("TensorFlow version does not support MultiHeadAttention")  
                    self._attention_error = "TensorFlow version does not support MultiHeadAttention"
                    return False
                    
                # Create attention module - with comprehensive error handling
                try:
                    # Define attention parameters
                    num_heads = 4
                    key_dim = 32
                    self.attention_module = tf.keras.layers.MultiHeadAttention(
                        num_heads=num_heads, 
                        key_dim=key_dim,
                        dropout=0.1
                    )
                    self._attention_initialized = True
                    return True
                except Exception as e:
                    logger.error(f"Error creating MultiHeadAttention: {e}")
                    self._attention_error = f"Error creating MultiHeadAttention: {e}"
                    return False
                    
            except RecursionError as re:
                logger.error(f"RecursionError during TensorFlow initialization: {re}")
                self._attention_error = f"RecursionError during TensorFlow initialization"
                return False
            except Exception as e:
                logger.error(f"Error initializing attention module: {e}")
                self._attention_error = f"Error initializing attention: {e}"
                return False

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
        2. Retrieve historical key projections for attention
        3. Calculate attention gates based on history (alpha, theta, eta)
        4. Return gates for use by neural memory during update step
        
        Args:
            memory_id: ID of the memory being processed
            x_t: Input embedding
            k_t: Key projection
            v_t: Value projection
            q_t: Query projection
            y_t: Output embedding
            
        Returns:
            Dictionary with gates and metrics for use by neural memory during update
        """
        # Initialize metrics dictionary
        metrics = {}
        metrics["gate_calculation_attempted"] = True
        
        # First, store this context tuple in history
        self.store_context(memory_id, x_t, k_t, v_t, q_t, y_t)
        
        # Then, retrieve historical key projections for attention
        keys = self.sequence_context.get_recent_keys()
        
        if not keys:
            logger.warning("MAG: No history available for attention, skipping gate calculation")
            metrics["gate_calculation_success"] = False
            metrics["error"] = "No history available for attention"
            metrics["history_size_used"] = 0
            return {"success": False, "gates": None, "metrics": metrics}
        
        # Record the size of history used for attention - directly use the keys list
        metrics["history_size_used"] = len(keys)
        
        try:
            # Lazy initialization of attention
            if not self._initialize_attention():
                logger.warning("MAG: Attention module not initialized, skipping gate calculation")
                metrics["gate_calculation_success"] = False
                metrics["error"] = self._attention_error
                return {"success": False, "gates": None, "metrics": metrics}
            
            # Convert to TensorFlow tensors if not already (avoiding lazy import)
            tf = _get_tf()
            if tf is None:
                logger.error("MAG: Failed to import TensorFlow for attention calculation")
                metrics["gate_calculation_success"] = False
                metrics["error"] = "Failed to import TensorFlow for attention calculation"
                return {"success": False, "gates": None, "metrics": metrics}
                
            # Convert q_t and k_hist to appropriate tensors
            try:
                q_t_tf = tf.convert_to_tensor(q_t, dtype=tf.float32)
                if len(q_t_tf.shape) == 1:
                    q_t_tf = tf.expand_dims(q_t_tf, 0)  # Add batch dimension
                    
                k_hist_tf = tf.convert_to_tensor(keys, dtype=tf.float32)
                if len(k_hist_tf.shape) == 2:  # [seq_len, key_dim]
                    k_hist_tf = tf.expand_dims(k_hist_tf, 0)  # Add batch dimension [1, seq_len, key_dim]
            except Exception as e:
                logger.error(f"MAG: Error converting inputs to tensors: {e}")
                metrics["gate_calculation_success"] = False
                metrics["error"] = f"Error converting inputs to tensors: {str(e)}"
                return {"success": False, "gates": None, "metrics": metrics}
            
            # Calculate attention between q_t and historical k_t values
            # Returns attended_k which is a weighted combination of k_hist values
            attended_output = self.attention_module(
                query=q_t_tf,           # [1, D]  
                key=k_hist_tf,           # [1, N, D]
                value=k_hist_tf,         # Use k_hist as values too [1, N, D]
                return_attention_scores=False
            )
            
            # Convert the attended output to numpy for API call
            attention_output_np = attended_output.numpy()
            
            # Record the attention norm in metrics
            metrics["attention_norm"] = float(np.linalg.norm(attention_output_np))
            
            # Use attended_k to calculate gates via API call
            api_response = self._make_request(
                "/calculate_gates",
                {"attention_output": attention_output_np.squeeze().tolist()}
            )
            
            if api_response.get("success", False):
                gates = {
                    "alpha": api_response.get("alpha"),
                    "theta": api_response.get("theta"),
                    "eta": api_response.get("eta")
                }
                metrics["gate_calculation_success"] = True
                metrics["calculated_gates"] = gates.copy()
                
                logger.info(f"MAG: Successfully calculated gates: alpha={gates['alpha']}, theta={gates['theta']}, eta={gates['eta']}")
                return {"success": True, "gates": gates, "metrics": metrics}
            else:
                error_msg = api_response.get("error", "Unknown error in gate calculation")
                logger.error(f"MAG: Error calculating gates: {error_msg}")
                metrics["gate_calculation_success"] = False
                metrics["error"] = error_msg
                return {"success": False, "gates": None, "metrics": metrics}
                
        except Exception as e:
            logger.error(f"MAG: Error in process_input: {e}")
            metrics["gate_calculation_success"] = False
            metrics["error"] = f"Error in MAG process_input: {str(e)}"
            return {"success": False, "gates": None, "metrics": metrics}
    
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
        # Initialize metrics dictionary
        metrics = {}
        metrics["v_prime_calculation_attempted"] = True
        metrics["history_size_used"] = len(k_hist) if k_hist else 0
        
        if not k_hist or not v_hist:
            logger.warning("MAL: No historical data available for attention. Returning original v_t.")
            metrics["v_prime_calculation_success"] = False
            metrics["error"] = "No historical data available for attention"
            return {"success": False, "v_prime_t": v_t, "metrics": metrics}
        
        # Ensure the attention module is initialized
        if not self._attention_initialized:
            self._initialize_attention()
            
        # If attention initialization failed, fall back to original values
        if not self._attention_initialized or self.attention_module is None:
            logger.warning("MAL: Attention module not available. Returning original v_t.")
            metrics["v_prime_calculation_success"] = False
            metrics["error"] = "Attention module not available"
            return {"success": False, "v_prime_t": v_t, "metrics": metrics}
            
        # Get TensorFlow only when needed
        try:
            tf = _get_tf()
            np = _get_numpy()
            if tf is None or np is None:
                logger.warning("MAL: TensorFlow or NumPy not available. Returning original v_t.")
                metrics["v_prime_calculation_success"] = False
                metrics["error"] = "TensorFlow or NumPy not available"
                return {"success": False, "v_prime_t": v_t, "metrics": metrics}
            
            # Convert numpy arrays to tensors for TF operations
            q_tensor = tf.convert_to_tensor(q_t, dtype='float32')
            if len(q_tensor.shape) == 1:
                q_tensor = tf.expand_dims(q_tensor, 0)  # Add batch dimension
                
            k_hist_tensor = tf.convert_to_tensor(k_hist, dtype='float32')
            if len(k_hist_tensor.shape) == 2:  # [seq_len, key_dim]
                k_hist_tensor = tf.expand_dims(k_hist_tensor, 0)  # Add batch dimension
            
            v_hist_tensor = tf.convert_to_tensor(v_hist, dtype='float32')
            if len(v_hist_tensor.shape) == 2:  # [seq_len, value_dim]
                v_hist_tensor = tf.expand_dims(v_hist_tensor, 0)  # Add batch dimension
        
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
            
            # Record attention norm in metrics
            attended_v_np = attended_v_tensor.numpy()
            metrics["attention_norm"] = float(np.linalg.norm(attended_v_np))
            
            # Combine attended and current values
            if self.v_prime_gate is None:
                # Initialize projection layers if not already done
                self.init_value_projection_layers(v_tensor.shape[-1])
            
            # Concatenate vectors for gating
            concat_v = tf.concat([v_tensor, attended_v_tensor], axis=-1)
            
            # Calculate gate value
            gate = self.v_prime_gate(concat_v)
            
            # Record the gate value (blend ratio) in metrics
            metrics["gated_blend_ratio"] = float(tf.squeeze(gate).numpy())
            
            # Combine original and attended values
            v_prime_tensor = gate * v_tensor + (1 - gate) * attended_v_tensor
            
            # Final projection
            v_prime_tensor = self.v_prime_projector(v_prime_tensor)
            
            # Convert back to numpy
            v_prime_np = v_prime_tensor.numpy()
            if len(v_prime_np.shape) > 1:
                v_prime_np = v_prime_np[0]  # Remove batch dimension
            
            # Calculate the difference norm between v_t and v_prime_t
            v_t_np = np.asarray(v_t, dtype=np.float32) if not isinstance(v_t, np.ndarray) else v_t
            metrics["v_prime_diff_norm"] = float(np.linalg.norm(v_prime_np - v_t_np))
            
            logger.info(f"MAL: Generated augmented value projection from {len(k_hist)} historical values")
            
            # Mark calculation as successful
            metrics["v_prime_calculation_success"] = True
            
            return {
                "success": True,
                "v_prime_t": v_prime_np,  # Augmented value projection
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"MAL calculate_v_prime failed: {str(e)}", exc_info=True)
            # Fallback to original value projection
            metrics["v_prime_calculation_success"] = False
            metrics["error"] = f"Error in MAL calculate_v_prime: {str(e)}"
            return {
                "success": False,
                "v_prime_t": v_t,  # Fallback to original
                "metrics": metrics
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
