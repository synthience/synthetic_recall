# synthians_trainer_server/neural_memory.py

import tensorflow as tf
import numpy as np
import json
import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING
from enum import Enum # Import Enum
import datetime

# Ensure TensorFlow uses float32 by default
tf.keras.backend.set_floatx('float32')
logger = logging.getLogger(__name__)

# --- Configuration Class ---
class NeuralMemoryConfig(dict):
    """Configuration for the NeuralMemoryModule."""
    def __init__(self, *args, **kwargs):
        defaults = {
            "input_dim": 768,
            "key_dim": 128,
            "value_dim": 768,
            "query_dim": 128,
            "memory_hidden_dims": [512],
            "gate_hidden_dims": [64],
            "alpha_init": -2.0,
            "theta_init": -3.0, # Controls inner loop LR
            "eta_init": 2.0,
            "outer_learning_rate": 1e-4,
            "use_complex_gates": False
        }
        config = defaults.copy()
        # Apply kwargs first
        config.update(kwargs)
        # Then apply dict from args if provided
        if args and isinstance(args[0], dict):
            config.update(args[0])

        super().__init__(config)
        # Ensure integer dimensions after all updates
        for key in ["input_dim", "key_dim", "value_dim", "query_dim"]:
            if key in self: self[key] = int(self[key])
        if "memory_hidden_dims" in self:
            self["memory_hidden_dims"] = [int(d) for d in self["memory_hidden_dims"]]
        if "gate_hidden_dims" in self:
            self["gate_hidden_dims"] = [int(d) for d in self["gate_hidden_dims"]]

    # Allow attribute access (though we avoid relying on it internally now)
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'NeuralMemoryConfig' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


# --- Core Memory MLP ---
class MemoryMLP(tf.keras.layers.Layer):
    """The core MLP model (M) used for associative memory."""
    def __init__(self, key_dim, value_dim, hidden_dims, name="MemoryMLP", **kwargs):
        super().__init__(name=name, **kwargs)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.hidden_dims = [int(d) for d in hidden_dims]
        
        # Create layers in __init__ as instance attributes so they're properly tracked
        self.hidden_layers = []
        for i, units in enumerate(self.hidden_dims):
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    units, 
                    activation='relu',
                    name=f"mem_hidden_{i+1}"
                )
            )
        
        # Output Layer
        self.output_layer = tf.keras.layers.Dense(self.value_dim, name="mem_output")

    def build(self, input_shape):
        # input_shape is expected to be [batch_size, key_dim]
        shape = tf.TensorShape(input_shape)
        last_dim = shape[-1]
        if last_dim is None:
             raise ValueError(f"Input dimension must be defined for {self.name}. Received shape: {input_shape}")
        if last_dim != self.key_dim:
             logger.warning(f"{self.name} input shape last dim {last_dim} != config key_dim {self.key_dim}. Ensure config matches data.")

        # Build all layers with explicit input shapes
        current_shape = shape
        for layer in self.hidden_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)
            
        # Build output layer
        self.output_layer.build(current_shape)
        
        # Call super build to ensure proper tracking
        super().build(input_shape)
        logger.info(f"{self.name} built successfully with input shape {input_shape}. Found {len(self.trainable_variables)} trainable vars.")

    def call(self, inputs, training=None):
        x = inputs
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        # Pass through output layer
        return self.output_layer(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"key_dim": self.key_dim, "value_dim": self.value_dim, "hidden_dims": self.hidden_dims})
        return config

# --- Neural Memory Module ---
class NeuralMemoryModule(tf.keras.Model):
    """
    Implements the Titans Neural Memory module that learns at test time.
    Inherits from tf.keras.Model for easier weight management and saving.
    """
    def __init__(self, config: Optional[Union[NeuralMemoryConfig, Dict]] = None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(config, dict) or config is None: self.config = NeuralMemoryConfig(**(config or {}))
        elif isinstance(config, NeuralMemoryConfig): self.config = config
        else: raise TypeError("config must be a dict or NeuralMemoryConfig")

        logger.info(f"Initializing NeuralMemoryModule with config: {dict(self.config)}")

        # --- Outer Loop Parameters ---
        initializer_outer = tf.keras.initializers.GlorotUniform()
        key_dim, value_dim, query_dim, input_dim = self.config['key_dim'], self.config['value_dim'], self.config['query_dim'], self.config['input_dim']

        self.WK_layer = tf.keras.layers.Dense(key_dim, name="WK_proj", use_bias=False, kernel_initializer=initializer_outer)
        self.WV_layer = tf.keras.layers.Dense(value_dim, name="WV_proj", use_bias=False, kernel_initializer=initializer_outer)
        self.WQ_layer = tf.keras.layers.Dense(query_dim, name="WQ_proj", use_bias=False, kernel_initializer=initializer_outer)

        if not self.config.get('use_complex_gates', False):
            self.alpha_logit = tf.Variable(tf.constant(self.config['alpha_init'], dtype=tf.float32), name="alpha_logit", trainable=True)
            self.theta_logit = tf.Variable(tf.constant(self.config['theta_init'], dtype=tf.float32), name="theta_logit", trainable=True)
            self.eta_logit = tf.Variable(tf.constant(self.config['eta_init'], dtype=tf.float32), name="eta_logit", trainable=True)
            self._gate_params = [self.alpha_logit, self.theta_logit, self.eta_logit]
        else:
            logger.warning("Complex gates not implemented, using simple scalar gates.")
            self.alpha_logit = tf.Variable(tf.constant(self.config['alpha_init'], dtype=tf.float32), name="alpha_logit", trainable=True)
            self.theta_logit = tf.Variable(tf.constant(self.config['theta_init'], dtype=tf.float32), name="theta_logit", trainable=True)
            self.eta_logit = tf.Variable(tf.constant(self.config['eta_init'], dtype=tf.float32), name="eta_logit", trainable=True)
            self._gate_params = [self.alpha_logit, self.theta_logit, self.eta_logit]

        # --- Inner Loop Parameters (Memory Model M) ---
        self.memory_mlp = MemoryMLP(
            key_dim=key_dim, value_dim=value_dim, hidden_dims=self.config['memory_hidden_dims'], name="MemoryMLP"
        )
        # --- Force build with a defined input shape ---
        # Create a dummy input tensor with batch size 1 and correct key_dim
        dummy_mlp_input = tf.TensorSpec(shape=[1, key_dim], dtype=tf.float32)
        # Build the MLP now
        self.memory_mlp.build(dummy_mlp_input.shape)
        # Verify build
        if not self.memory_mlp.built:
             logger.error("MemoryMLP failed to build during init!")
        self._inner_trainable_variables = self.memory_mlp.trainable_variables
        logger.info(f"MemoryMLP built. Trainable variables: {len(self._inner_trainable_variables)}")
        if not self._inner_trainable_variables: logger.error("MemoryMLP has NO trainable variables!")

        # --- Momentum State ---
        self.momentum_state = [
            tf.Variable(tf.zeros_like(var), trainable=False, name=f"momentum_{i}")
            for i, var in enumerate(self._inner_trainable_variables)
        ]
        logger.info(f"Momentum state variables created: {len(self.momentum_state)}")

        # --- Optimizer for Outer Loop ---
        self.outer_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['outer_learning_rate'])

        # Build projection layers
        self.WK_layer.build(input_shape=(None, input_dim))
        self.WV_layer.build(input_shape=(None, input_dim))
        self.WQ_layer.build(input_shape=(None, input_dim))
        logger.info("Projection layers built.")


    @property
    def inner_trainable_variables(self):
        return self.memory_mlp.trainable_variables

    @property
    def outer_trainable_variables(self):
         return self.WK_layer.trainable_variables + \
                self.WV_layer.trainable_variables + \
                self.WQ_layer.trainable_variables + \
                self._gate_params

    def get_projections(self, x_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Calculate key, value, query projections from input."""
        # Convert to tensor and ensure correct dtype
        x_t = tf.convert_to_tensor(x_t, dtype=tf.float32)
        original_shape = tf.shape(x_t)
        
        # Handle various input shapes
        if len(original_shape) == 1:
            # Single vector -> add batch dimension
            x_t = tf.expand_dims(x_t, 0)
            logger.debug(f"Reshaped input from {original_shape} to {tf.shape(x_t)}")
        
        # Verify input dimension
        if tf.shape(x_t)[-1] != self.config['input_dim']:
            logger.warning(f"Input dimension mismatch: expected {self.config['input_dim']}, got {tf.shape(x_t)[-1]}")
            raise ValueError(f"Input dimension mismatch: expected {self.config['input_dim']}, got {tf.shape(x_t)[-1]}")
            
        # Apply projections
        k_t = self.WK_layer(x_t)
        v_t = self.WV_layer(x_t)
        q_t = self.WQ_layer(x_t)
        return k_t, v_t, q_t
    
    def get_projection_hash(self) -> str:
        """Returns a hash representation of the current projection matrices.
        
        This is used for introspection and cognitive tracing, allowing the system to
        detect when projection matrices have changed between different runs or sessions.
        
        Returns:
            str: A hash string representing the current state of WK, WV, WQ matrices
        """
        # In a full implementation, this would calculate a hash based on the actual weight values
        # For now, return a placeholder with timestamp to make it somewhat unique
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"proj_matrix_hash_{timestamp}"

    def call(self, q_t: tf.Tensor, training=False) -> tf.Tensor:
        """Retrieve value from memory given query q_t (inference only)."""
        # Dynamic shape handling - reshape input if needed
        q_t = tf.convert_to_tensor(q_t, dtype=tf.float32)
        original_shape = tf.shape(q_t)
        
        # Handle various input shapes
        if len(original_shape) == 1:
            # Single vector -> add batch dimension
            q_t = tf.expand_dims(q_t, 0)
            logger.debug(f"Reshaped input from {original_shape} to {tf.shape(q_t)}")
        
        # Config sanity check - key_dim and query_dim should match
        if self.config['query_dim'] != self.config['key_dim']:
            logger.error(f"CONFIG ERROR: query_dim ({self.config['query_dim']}) != key_dim ({self.config['key_dim']})")
            # This is a configuration error that should be fixed
            # For now, use key_dim as the source of truth for validation
            expected_dim = self.config['key_dim']
            logger.warning(f"Using key_dim={expected_dim} as expected dimension for memory_mlp input")
        else:
            expected_dim = self.config['key_dim']  # Either key_dim or query_dim (they should be equal)
            
        # Verify input dimensions match key_dim (what memory_mlp expects)
        actual_dim = tf.shape(q_t)[-1]
        if actual_dim != expected_dim:
            logger.error(f"Input dimension mismatch: expected key_dim={expected_dim}, got {actual_dim}")
            # This is a runtime error
            raise ValueError(f"Input dimension mismatch: memory_mlp expects key_dim={expected_dim}, got {actual_dim}")
            
        retrieved_value = self.memory_mlp(q_t, training=training)
        return retrieved_value

    # Inner loop update step - NO @tf.function for now
    def update_step(self, x_t: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        # Ensure proper input shape with batch dimension
        if len(tf.shape(x_t)) == 1: x_t = tf.expand_dims(x_t, axis=0)
        tf.debugging.assert_equal(tf.shape(x_t)[0], 1, message="update_step expects batch size 1")
        tf.debugging.assert_equal(tf.shape(x_t)[-1], self.config['input_dim'], message="Input dimension mismatch")

        # Get current gate values
        alpha_t = tf.sigmoid(self.alpha_logit)
        theta_t = tf.sigmoid(self.theta_logit)
        eta_t = tf.sigmoid(self.eta_logit)

        # Calculate projections
        k_t, v_t, _ = self.get_projections(x_t)

        # Access inner trainable variables directly from the memory_mlp
        inner_vars = self.memory_mlp.trainable_variables
        if not inner_vars:
            logger.error("No inner trainable variables found in update_step!")
            # Try force-building the MLP again to ensure variables are created
            dummy_key = tf.zeros((1, self.config['key_dim']), dtype=tf.float32)
            _ = self.memory_mlp(dummy_key)  # Force model execution
            inner_vars = self.memory_mlp.trainable_variables
            if not inner_vars:
                return tf.constant(0.0), []
            else:
                logger.info(f"Successfully rebuilt memory MLP. Found {len(inner_vars)} trainable variables.")

        # Ensure momentum state matches inner variables
        if len(self.momentum_state) != len(inner_vars):
            logger.warning("Rebuilding momentum state due to variable mismatch.")
            self.momentum_state = [tf.Variable(tf.zeros_like(var), trainable=False, name=f"momentum_{i}") 
                                 for i, var in enumerate(inner_vars)]

        # Compute gradient with explicit watch on inner variables
        with tf.GradientTape() as tape:
            # Tape automatically watches trainable variables
            predicted_v_t = self.memory_mlp(k_t, training=True)
            loss = 0.5 * tf.reduce_sum(tf.square(predicted_v_t - v_t))

        grads = tape.gradient(loss, inner_vars)

        # Verify we have valid gradients
        valid_grads_indices = [i for i, g in enumerate(grads) if g is not None]
        if len(valid_grads_indices) != len(inner_vars): 
            logger.warning(f"Found {len(inner_vars) - len(valid_grads_indices)} None gradients in inner loop.")

        # Compute momentum updates
        for i in valid_grads_indices:
            grad = grads[i]
            s_var = self.momentum_state[i]
            # Update momentum state
            s_new = eta_t * s_var - theta_t * grad
            s_var.assign(s_new)

        # Apply weight updates
        for i in valid_grads_indices:
            s_t = self.momentum_state[i]
            m_var = inner_vars[i]
            # Update memory weights
            m_new = (1.0 - alpha_t) * m_var + s_t
            m_var.assign(m_new)

        # Return loss and gradients for tracking
        return loss, grads


    def train_step(self, data):
        input_sequence, target_sequence = data
        
        # Ensure memory_mlp has trainable variables
        if not self.memory_mlp.trainable_variables:
            logger.warning("No trainable variables in memory_mlp during train_step. Attempting to rebuild...")
            dummy_key = tf.zeros((1, self.config['key_dim']), dtype=tf.float32)
            _ = self.memory_mlp(dummy_key)  # Force model execution
        
        # Store initial state
        initial_memory_weights = [tf.identity(v) for v in self.memory_mlp.trainable_variables]
        initial_momentum_state = [tf.identity(s) for s in self.momentum_state]
        
        # Get sequence dimensions
        batch_size = tf.shape(input_sequence)[0]
        seq_len = tf.shape(input_sequence)[1]
        total_outer_loss = tf.constant(0.0, dtype=tf.float32)

        # Get outer trainable variables to track
        outer_vars = self.outer_trainable_variables # Get current list

        with tf.GradientTape() as tape:
            # Explicitly watch outer variables
            for var in outer_vars:
                tape.watch(var)

            # Reset inner memory and momentum state
            for i, var in enumerate(self.memory_mlp.trainable_variables):
                var.assign(tf.zeros_like(var))
            for i, s_var in enumerate(self.momentum_state):
                s_var.assign(tf.zeros_like(s_var))

            # Process sequence
            for t in tf.range(seq_len):
                x_t_batch = input_sequence[:, t, :]
                target_t_batch = target_sequence[:, t, :]

                # Generate predictions (use projection layers - outer params)
                _, _, q_t_batch = self.get_projections(x_t_batch)
                retrieved_y_t_batch = self(q_t_batch, training=False) # Uses memory_mlp - inner params

                # Compute loss against target
                tf.debugging.assert_equal(tf.shape(retrieved_y_t_batch)[-1], tf.shape(target_t_batch)[-1], 
                                          message="Outer loss target dim mismatch")
                step_loss = tf.reduce_mean(tf.square(retrieved_y_t_batch - target_t_batch))
                total_outer_loss += step_loss

                # Inner update loop - process one example at a time for now
                # This is inefficient for batch>1 but ensures correct updates
                for b in tf.range(batch_size):
                    x_t = tf.expand_dims(x_t_batch[b], axis=0)
                    _, _ = self.update_step(x_t)  # Apply inner loop update

        # Check validity of outer vars
        valid_outer_vars = [v for v in outer_vars if v is not None]
        if len(valid_outer_vars) < len(outer_vars):
            logger.warning(f"Found {len(outer_vars) - len(valid_outer_vars)} None variables in outer_vars!")
        
        # Calculate outer gradients
        outer_grads = tape.gradient(total_outer_loss, valid_outer_vars)
        
        # Check for None gradients in outer loop
        none_grads = sum(1 for g in outer_grads if g is None)
        if none_grads > 0:
            logger.warning(f"Found {none_grads} None gradients in outer loop.")

        # Apply outer gradients
        non_none_grads = []
        non_none_vars = []
        for i, (grad, var) in enumerate(zip(outer_grads, valid_outer_vars)):
            if grad is not None:
                non_none_grads.append(grad)
                non_none_vars.append(var)
        
        # Apply valid gradients only
        if non_none_grads:
            self.outer_optimizer.apply_gradients(zip(non_none_grads, non_none_vars))
        
        # Restore original memory state
        for i, var in enumerate(self.memory_mlp.trainable_variables):
            if i < len(initial_memory_weights):
                var.assign(initial_memory_weights[i])
                
        for i, s_var in enumerate(self.momentum_state):
            if i < len(initial_momentum_state):
                s_var.assign(initial_momentum_state[i])

        return {"loss": total_outer_loss / tf.cast(seq_len, dtype=tf.float32)}

    # --- Persistence ---
    def save_state(self, path: str) -> None:
        if path.startswith("file://"): path = path[7:]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            state = {
                "config": self.get_config_dict(),
                "inner_weights": {v.name: v.numpy().tolist() for v in self.inner_trainable_variables},
                "outer_weights": {v.name: v.numpy().tolist() for v in self.outer_trainable_variables},
                "momentum_state": {s.name: s.numpy().tolist() for s in self.momentum_state},
                "timestamp": datetime.datetime.now().isoformat(),
            }
            
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Neural Memory state saved to {path}")
        except Exception as e:
            logger.error(f"Error saving Neural Memory state: {e}", exc_info=True)
            raise

    def load_state(self, path: str) -> bool:
        if path.startswith("file://"): path = path[7:]
        if not os.path.exists(path): 
            logger.error(f"State file not found: {path}")
            return False

        logger.info(f"Loading Neural Memory state from {path}")
        try:
            with open(path, 'r') as f: 
                state = json.load(f)
                
            loaded_config_dict = state.get("config")
            if not loaded_config_dict: 
                logger.error("State missing 'config'")
                return False

            # Check if we need to re-initialize with the loaded config
            current_config_dict = self.get_config_dict()
            config_changed = current_config_dict != loaded_config_dict
            if config_changed:
                logger.warning(f"Loaded config differs from current config")
                # We don't attempt to rebuild the model here - that needs to be done externally
                # Just log a warning that configs don't match

            # Load inner weights (memory model)
            inner_weights_loaded = state.get("inner_weights", {})
            inner_vars_dict = {v.name: v for v in self.inner_trainable_variables}
            loaded_count = 0
            for name, loaded_list in inner_weights_loaded.items():
                if name in inner_vars_dict:
                    var = inner_vars_dict[name]
                    loaded_val = tf.convert_to_tensor(loaded_list, dtype=tf.float32)
                    if var.shape == loaded_val.shape:
                        var.assign(loaded_val)
                        loaded_count += 1
                    else: 
                        logger.error(f"Shape mismatch loading inner var {name}: {var.shape} vs {loaded_val.shape}")
                else: 
                    logger.warning(f"Inner var {name} not in current model.")
            logger.info(f"Loaded {loaded_count} inner weights.")

            # Load outer weights (projection layers)
            outer_weights_loaded = state.get("outer_weights", {})
            outer_vars_dict = {v.name: v for v in self.outer_trainable_variables}
            loaded_count = 0
            for name, loaded_list in outer_weights_loaded.items():
                if name in outer_vars_dict:
                    var = outer_vars_dict[name]
                    loaded_val = tf.convert_to_tensor(loaded_list, dtype=tf.float32)
                    if var.shape == loaded_val.shape:
                        var.assign(loaded_val)
                        loaded_count += 1
                    else: 
                        logger.error(f"Shape mismatch loading outer var {name}: {var.shape} vs {loaded_val.shape}")
                else: 
                    logger.warning(f"Outer var {name} not in current model.")
            logger.info(f"Loaded {loaded_count} outer weights.")

            # Load momentum state
            momentum_loaded = state.get("momentum_state", {})
            # Rebuild momentum state if needed (without calling assign directly)
            if len(self.momentum_state) != len(self.inner_trainable_variables):
                logger.warning("Momentum state size doesn't match inner vars. Creating new state.")
                # Create new momentum variables without assigning to self yet
                new_momentum = []
                for i, var in enumerate(self.inner_trainable_variables):
                    new_momentum.append(tf.Variable(tf.zeros_like(var), trainable=False, name=f"momentum_{i}"))
                # Now replace the list (safer than assigning individual vars)
                self.momentum_state = new_momentum

            loaded_count = 0
            mom_vars_dict = {v.name: v for v in self.momentum_state}
            for name, loaded_list in momentum_loaded.items():
                if name in mom_vars_dict:
                    var = mom_vars_dict[name]
                    loaded_val = tf.convert_to_tensor(loaded_list, dtype=tf.float32)
                    if var.shape == loaded_val.shape:
                        var.assign(loaded_val)
                        loaded_count += 1
                    else: 
                        logger.error(f"Shape mismatch loading momentum var {name}: {var.shape} vs {loaded_val.shape}")
                else: 
                    logger.warning(f"Momentum var {name} not in current model.")
            logger.info(f"Loaded {loaded_count} momentum states.")

            logger.info(f"Neural Memory state successfully loaded from {path}")
            return True
            
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON state file {path}: {e}")
             return False
        except Exception as e:
            logger.error(f"Error loading Neural Memory state: {e}", exc_info=True)
            return False

    def get_config_dict(self) -> Dict:
         """Return config as a serializable dict."""
         # Convert Enum members to strings if necessary
         serializable_config = {}
         for k, v in self.config.items():
              serializable_config[k] = v.value if isinstance(v, Enum) else v
         return serializable_config