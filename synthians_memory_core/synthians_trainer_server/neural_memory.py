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
        
        # Initialize gate projection layers for MAG variant (used by calculate_gates)
        self.attention_to_alpha = tf.keras.layers.Dense(1, name="attention_to_alpha")
        self.attention_to_theta = tf.keras.layers.Dense(1, name="attention_to_theta")
        self.attention_to_eta = tf.keras.layers.Dense(1, name="attention_to_eta")
        
        # Storage for last computed gate values
        self.last_applied_gates = {}

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

        # Build gate projection layers
        self.attention_to_alpha.build(input_shape=(None, query_dim))
        self.attention_to_theta.build(input_shape=(None, query_dim))
        self.attention_to_eta.build(input_shape=(None, query_dim))
        logger.info("Gate projection layers built.")

    @property
    def inner_trainable_variables(self):
        return self.memory_mlp.trainable_variables

    @property
    def outer_trainable_variables(self):
         return self.WK_layer.trainable_variables + \
                self.WV_layer.trainable_variables + \
                self.WQ_layer.trainable_variables + \
                self.attention_to_alpha.trainable_variables + \
                self.attention_to_theta.trainable_variables + \
                self.attention_to_eta.trainable_variables + \
                self._gate_params

    def get_projections(self, x_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Calculate key, value, query projections from input.
        
        Args:
            x_t: Input tensor with shape [batch_size, input_dim]
            
        Returns:
            Tuple of (key_projection, value_projection, query_projection)
        """
        x_t = tf.convert_to_tensor(x_t, dtype=tf.float32)
        
        # Ensure input has right shape
        if len(tf.shape(x_t)) == 1:
            # Add batch dimension if missing
            x_t = tf.expand_dims(x_t, 0)
            
        # Get projections
        k_t = self.WK_layer(x_t)  # [batch_size, key_dim]
        v_t = self.WV_layer(x_t)  # [batch_size, value_dim]
        q_t = self.WQ_layer(x_t)  # [batch_size, query_dim]
        
        return k_t, v_t, q_t
    
    def calculate_gates(self, attention_output) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Calculate gate values from attention output for MAG variant.
        
        Args:
            attention_output: Output tensor from attention mechanism
            
        Returns:
            Tuple of (alpha_t, theta_t, eta_t) gate values
        """
        # Default gates (fallback if computation fails)
        alpha_logit = self.alpha_logit
        theta_logit = self.theta_logit
        eta_logit = self.eta_logit
        
        try:
            # Ensure attention_output has the right shape
            attention_output = tf.convert_to_tensor(attention_output, dtype=tf.float32)
            if len(tf.shape(attention_output)) == 1:
                attention_output = tf.expand_dims(attention_output, 0)
            
            # Project attention output to gate logits using dedicated layers
            alpha_logit = self.attention_to_alpha(attention_output)
            theta_logit = self.attention_to_theta(attention_output)
            eta_logit = self.attention_to_eta(attention_output)
            
            # Remove the extra dimensions
            alpha_logit = tf.squeeze(alpha_logit)
            theta_logit = tf.squeeze(theta_logit)
            eta_logit = tf.squeeze(eta_logit)
            
            logger.debug(f"Calculated gate logits from attention: alpha={alpha_logit.numpy()}, theta={theta_logit.numpy()}, eta={eta_logit.numpy()}")
            
        except Exception as e:
            logger.warning(f"Error calculating gates from attention output: {e}. Using default gates.")
        
        # Apply sigmoid to get gate values in [0,1] range
        alpha_t = tf.nn.sigmoid(alpha_logit)  # Forget rate
        theta_t = tf.nn.sigmoid(theta_logit)  # Inner learning rate
        eta_t = tf.nn.sigmoid(eta_logit)      # Momentum
        
        return alpha_t, theta_t, eta_t

    def __call__(self, q_t: tf.Tensor, training=False):
        """Retrieve value from memory given query q_t (inference only)."""
        # Ensure q_t has correct shape with batch dimension
        q_t = tf.convert_to_tensor(q_t, dtype=tf.float32)
        if len(tf.shape(q_t)) == 1:
            q_t = tf.expand_dims(q_t, 0)  # Add batch dim
            
        return self.memory_mlp(q_t, training=training)

    def update_step(self, x_t: tf.Tensor, 
                   external_k_t: Optional[tf.Tensor] = None,
                   external_v_t: Optional[tf.Tensor] = None,
                   external_alpha_t: Optional[float] = None,
                   external_theta_t: Optional[float] = None,
                   external_eta_t: Optional[float] = None) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Update memory weights based on input x_t.
        
        Args:
            x_t: Input tensor with shape [batch_size, input_dim]
            external_k_t: Optional external key projection (MAL variant)
            external_v_t: Optional external value projection (MAL variant)
            external_alpha_t: Optional external alpha gate (MAG variant - single value)
            external_theta_t: Optional external theta gate (MAG variant - single value)
            external_eta_t: Optional external eta gate (MAG variant - single value)
            
        Returns:
            Tuple of (loss, gradients)
        """
        # Ensure x_t has correct shape with batch dimension
        x_t = tf.convert_to_tensor(x_t, dtype=tf.float32)
        if len(tf.shape(x_t)) == 1:
            x_t = tf.expand_dims(x_t, 0)  # Add batch dim
        
        # Get projections if not provided externally
        if external_k_t is None or external_v_t is None:
            k_t, v_t, _ = self.get_projections(x_t)
            k_t = external_k_t if external_k_t is not None else k_t
            v_t = external_v_t if external_v_t is not None else v_t
        else:
            # Both projections provided externally
            k_t, v_t = external_k_t, external_v_t
        
        # Determine gate values - use externals if provided, otherwise use defaults
        alpha_t = tf.convert_to_tensor(external_alpha_t, dtype=tf.float32) if external_alpha_t is not None else tf.nn.sigmoid(self.alpha_logit)  # Forget rate
        theta_t = tf.convert_to_tensor(external_theta_t, dtype=tf.float32) if external_theta_t is not None else tf.nn.sigmoid(self.theta_logit)  # Inner learning rate
        eta_t = tf.convert_to_tensor(external_eta_t, dtype=tf.float32) if external_eta_t is not None else tf.nn.sigmoid(self.eta_logit)      # Momentum
        
        # Log gate values for debugging
        logger.debug(f"Applied gate values - alpha_t: {float(alpha_t.numpy()) if hasattr(alpha_t, 'numpy') else float(alpha_t)}, "
                  f"theta_t: {float(theta_t.numpy()) if hasattr(theta_t, 'numpy') else float(theta_t)}, "
                  f"eta_t: {float(eta_t.numpy()) if hasattr(eta_t, 'numpy') else float(eta_t)}")
        
        # Store the applied gates for downstream monitoring
        self.last_applied_gates = {
            "alpha_t": float(alpha_t.numpy()) if hasattr(alpha_t, 'numpy') else float(alpha_t),
            "theta_t": float(theta_t.numpy()) if hasattr(theta_t, 'numpy') else float(theta_t),
            "eta_t": float(eta_t.numpy()) if hasattr(eta_t, 'numpy') else float(eta_t)
        }
        
        # --- Gradient Calculation using GradientTape ---
        inner_vars = self.inner_trainable_variables
        with tf.GradientTape() as tape:
            # Forward pass through memory MLP
            predicted_v_t = self.memory_mlp(k_t, training=True) # Use k_t here
            # Calculate loss using potentially modified v_t (from MAL or original)
            loss = 0.5 * tf.reduce_mean(tf.square(predicted_v_t - v_t))

        grads = tape.gradient(loss, inner_vars)
        # --- End Gradient Calculation ---

        # --- Momentum and Weight Updates ---
        valid_grads_indices = [i for i, g in enumerate(grads) if g is not None]
        if len(valid_grads_indices) != len(inner_vars):
            logger.warning(f"Found {len(inner_vars) - len(valid_grads_indices)} None gradients in inner loop.")

        for i in valid_grads_indices:
            grad = grads[i]
            s_var = self.momentum_state[i]
            # Update momentum state
            s_new = eta_t * s_var - theta_t * grad
            s_var.assign(s_new)

        for i in valid_grads_indices:
            s_t = self.momentum_state[i]
            m_var = inner_vars[i]
            # Update memory weights
            m_new = (1.0 - alpha_t) * m_var + s_t
            m_var.assign(m_new)
        # --- End Updates ---

        return loss, grads # Return original grads list (may contain None)

    # Inner loop update step - NO @tf.function for now
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