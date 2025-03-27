import tensorflow as tf
import json
import os
import numpy as np
from typing import Dict, Any, Optional

# Assuming types.py might be reused or adapted. For simplicity, define locally if needed.
# from .types import TitanMemoryConfig, ForwardResult
class TitanMemoryConfig(dict): pass # Placeholder type
class ForwardResult(dict): pass # Placeholder type

# Ensure TensorFlow uses float32 by default
tf.keras.backend.set_floatx('float32')

class SynthiansSequencePredictor:
    """
    A neural network model for predicting the next element in a sequence of embeddings,
    maintaining an internal state (memory vector).

    This model complements the SynthiansMemoryCore by focusing on temporal dynamics
    rather than storage and retrieval of individual memories.
    """
    def __init__(self, config: Optional[TitanMemoryConfig] = None):
        if config is None:
            config = {}

        # Configuration remains similar, focusing on network dimensions and learning
        self.input_dim: int = config.get('inputDim', 768) # Match embedding dim typically
        self.hidden_dim: int = config.get('hiddenDim', 128) # Internal hidden layer size
        # Renamed 'outputDim' to 'memory_dim' internally for clarity
        self.memory_dim: int = config.get('outputDim', 256) # Dimension of the internal state vector

        # The predictor's output dimension matches the input dimension (predicting next embedding)
        self.predictor_output_dim: int = self.input_dim

        # Full output of the network includes memory update AND prediction
        self.full_internal_output_dim: int = self.memory_dim + self.predictor_output_dim

        self.learning_rate: float = config.get('learningRate', 1e-3)
        self.use_manifold: bool = config.get('useManifold', False)
        self.forget_gate_init: float = config.get('forgetGateInit', 0.01)
        self.max_step_size: float = config.get('maxStepSize', 0.1)
        self.tangent_epsilon: float = config.get('tangentEpsilon', 1e-8)

        # --- Trainable Parameters ---
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)

        # Layer 1: Input (current embedding + previous memory state) -> Hidden
        self.W1 = tf.Variable(initializer(shape=[self.input_dim + self.memory_dim, self.hidden_dim]), name="W1")
        self.b1 = tf.Variable(tf.zeros([self.hidden_dim]), name="b1")

        # Layer 2: Hidden -> Output (new memory state + predicted next embedding)
        self.W2 = tf.Variable(initializer(shape=[self.hidden_dim, self.full_internal_output_dim]), name="W2")
        self.b2 = tf.Variable(tf.zeros([self.full_internal_output_dim]), name="b2")

        # Forget gate (scalar) - controls how much of previous memory is kept
        self.forget_gate = tf.Variable(tf.constant(self.forget_gate_init, dtype=tf.float32), name="forgetGate")

        self.trainable_variables = [self.W1, self.b1, self.W2, self.b2, self.forget_gate]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32), # Current embedding x_t
        tf.TensorSpec(shape=[None], dtype=tf.float32)  # Previous memory state M_{t-1}
    ])
    def forward(self, x: tf.Tensor, memory: tf.Tensor) -> ForwardResult:
        """
        Performs a forward pass to predict the next embedding and update the memory state.

        Args:
            x: The current input embedding (shape: [input_dim]).
            memory: The previous memory state vector (shape: [memory_dim]).

        Returns:
            A dictionary containing:
                - predicted: The predicted next embedding (shape: [input_dim]).
                - newMemory: The updated memory state vector (shape: [memory_dim]).
                - surprise: A scalar indicating prediction error (MSE).
        """
        tf.debugging.assert_equal(tf.shape(x)[0], self.input_dim, message="Input dimension mismatch")
        tf.debugging.assert_equal(tf.shape(memory)[0], self.memory_dim, message="Memory dimension mismatch")

        # Gate the previous memory state
        forget_val = tf.nn.sigmoid(self.forget_gate) # Ensure 0-1 range
        gated_memory = tf.multiply(memory, tf.subtract(1.0, forget_val))

        # Combine current input and gated memory
        combined = tf.concat([x, gated_memory], axis=0)

        # MLP forward pass
        hidden1 = tf.nn.relu(tf.matmul(tf.expand_dims(combined, 0), self.W1) + self.b1)
        out_logits = tf.matmul(hidden1, self.W2) + self.b2
        out = tf.squeeze(out_logits, axis=0)

        # Split output into new memory state (M_t) and prediction (P_t for x_{t+1})
        new_memory = tf.slice(out, [0], [self.memory_dim])
        predicted = tf.slice(out, [self.memory_dim], [self.predictor_output_dim]) # predictor_output_dim == input_dim

        # Calculate surprise (MSE between predicted and *current input* x)
        # Note: This 'surprise' measures how well the *current* input could be reconstructed,
        # which differs slightly from a surprise metric based on predicting the *next* input.
        # The training loss focuses on predicting the *next* input.
        diff = tf.subtract(predicted, x)
        surprise = tf.reduce_mean(tf.square(diff))

        return {"predicted": predicted, "newMemory": new_memory, "surprise": surprise}

    # Manifold step remains conceptually the same, operating on internal memory vectors
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def manifold_step(self, base: tf.Tensor, velocity: tf.Tensor) -> tf.Tensor:
        """Applies a step on the manifold (sphere) if use_manifold is True."""
        # (Implementation remains the same as in the original model.py)
        if not self.use_manifold:
            return tf.add(base, velocity)

        base_norm = tf.norm(base)
        safe_base = base / (base_norm + 1e-12)
        dot = tf.reduce_sum(safe_base * velocity)
        radial = safe_base * dot
        tangent = velocity - radial
        t_norm = tf.norm(tangent)

        if t_norm < self.tangent_epsilon:
            return safe_base

        step_size = tf.minimum(t_norm, tf.constant(self.max_step_size, dtype=tf.float32))
        direction = tangent / t_norm
        cos_step = tf.cos(step_size)
        sin_step = tf.sin(step_size)
        new_param = safe_base * cos_step + direction * sin_step
        new_param_norm = tf.norm(new_param)
        return new_param / (new_param_norm + 1e-12)


    # Training step needs careful adaptation for tf.function
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32), # x_t
        tf.TensorSpec(shape=[None], dtype=tf.float32), # x_next (target)
        tf.TensorSpec(shape=[None], dtype=tf.float32)  # memory_state (M_{t-1})
    ])
    def _calculate_loss_and_grads(self, x_t, x_next, memory_state):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)

            # Run forward pass: Predicts P_t (for x_{t+1}) and M_t from x_t and M_{t-1}
            forward_result = self.forward(x_t, memory_state)
            predicted = forward_result["predicted"] # This is the prediction for x_{t+1}
            surprise = forward_result["surprise"]   # This is based on reconstructing x_t

            # Loss = MSE between predicted (P_t) and actual next state (x_{t+1})
            diff_next = tf.subtract(predicted, x_next)
            mse_loss = tf.reduce_mean(tf.square(diff_next))

            # Optional: Add surprise penalty (how well x_t was reconstructed)
            surprise_penalty_weight = tf.constant(0.01, dtype=tf.float32)
            total_loss = tf.add(mse_loss, tf.multiply(surprise_penalty_weight, surprise))

        grads = tape.gradient(total_loss, self.trainable_variables)
        # Return the forward result needed to update memory state outside tf.function
        return total_loss, grads, forward_result["newMemory"]

    def train_step(self, x_t: tf.Tensor, x_next: tf.Tensor, memory_state_var: tf.Variable) -> tf.Tensor:
        """Performs one training step. Updates memory_state_var."""
        tf.debugging.assert_equal(tf.shape(x_t)[0], self.input_dim, message="x_t dimension mismatch")
        tf.debugging.assert_equal(tf.shape(x_next)[0], self.input_dim, message="x_next dimension mismatch")
        tf.debugging.assert_equal(tf.shape(memory_state_var)[0], self.memory_dim, message="Memory dimension mismatch")

        total_loss, grads, new_memory_tensor = self._calculate_loss_and_grads(
            x_t, tf.identity(x_next), memory_state_var.value()
        )

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # --- Update Memory State ---
        # Optionally apply manifold step to the new memory state update
        # The 'velocity' could be the change from the old memory state
        if self.use_manifold:
            # Example: Velocity is the difference between new and old memory states
            memory_velocity = new_memory_tensor - memory_state_var.value()
            updated_memory = self.manifold_step(memory_state_var.value(), memory_velocity)
            memory_state_var.assign(updated_memory)
        else:
            memory_state_var.assign(new_memory_tensor) # Standard update

        return total_loss

    # Persistence methods remain the same, saving/loading the *trainer's* weights
    async def save_model(self, path: str) -> None:
        if path.startswith("file://"): path = path[7:]
        weights_dict = {var.name.split(':')[0]: var.numpy().tolist() for var in self.trainable_variables}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Use standard sync file IO for simplicity within async handler
        with open(path, 'w') as f: json.dump(weights_dict, f)
        print(f"Trainer model weights saved to {path}")

    async def load_model(self, path: str) -> None:
        if path.startswith("file://"): path = path[7:]
        if not os.path.exists(path): raise FileNotFoundError(f"Trainer model file not found: {path}")
        with open(path, 'r') as f: content = f.read()
        weights_dict = json.loads(content)
        loaded_vars = set()
        for var in self.trainable_variables:
            var_name = var.name.split(':')[0]
            if var_name in weights_dict:
                loaded_value = tf.convert_to_tensor(weights_dict[var_name], dtype=tf.float32)
                if var.shape != loaded_value.shape:
                    raise ValueError(f"Shape mismatch for {var_name}: expected {var.shape}, got {loaded_value.shape}")
                var.assign(loaded_value)
                loaded_vars.add(var_name)
            else: print(f"Warning: Variable {var_name} not found in loaded weights.")
        print(f"Trainer model weights loaded from {path}")


    def get_config(self) -> TitanMemoryConfig:
        """Returns the trainer model configuration."""
        return {
            "inputDim": self.input_dim,
            "hiddenDim": self.hidden_dim,
            "outputDim": self.memory_dim, # Keep original name for compatibility
            "learningRate": self.learning_rate,
            "useManifold": self.use_manifold,
            "forgetGateInit": self.forget_gate_init,
            "maxStepSize": self.max_step_size,
            "tangentEpsilon": self.tangent_epsilon
        }

    def get_weights(self) -> Dict[str, Any]:
        """Returns the trainer model weights as lists."""
        return {var.name.split(':')[0]: var.numpy().tolist() for var in self.trainable_variables}