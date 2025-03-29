import tensorflow as tf
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any, Union
from collections import deque
import time

logger = logging.getLogger(__name__)

class MultiHeadAttentionModule(tf.keras.layers.Layer):
    """
    Attention module for Titans architecture variants (MAC, MAG, MAL).

    This module wraps TensorFlow's MultiHeadAttention with additional features:
    - Dimension validation and standardization
    - Optional residual connections and layer normalization
    - Basic attention metrics tracking
    """

    def __init__(
        self,
        num_heads: int = 8,
        key_dim: int = 64, # Dimension per head
        value_dim: Optional[int] = None, # Dimension per head for value
        dropout: float = 0.0,
        use_bias: bool = True,
        output_shape: Optional[int] = None, # Total output dimension, defaults to query input dim
        attention_axes: Optional[Tuple[int, ...]] = None,
        kernel_initializer: str = "glorot_uniform",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        max_dim_mismatch_warnings: int = 10,
        name="MultiHeadAttentionModule",
        **kwargs
    ):
        """
        Initialize the attention module.

        Args:
            num_heads: Number of attention heads.
            key_dim: Size of each attention head for query and key.
            value_dim: Size of each attention head for value (defaults to key_dim).
            dropout: Dropout rate for attention weights.
            use_bias: Whether to use bias in projections.
            output_shape: Optional output dimension (defaults to query input shape).
            attention_axes: Axes over which attention is applied.
            kernel_initializer: Initializer for kernel weights.
            use_layer_norm: Whether to apply layer normalization to output.
            use_residual: Whether to use residual connections.
            max_dim_mismatch_warnings: Max dimension mismatch warnings to log.
        """
        super().__init__(name=name, **kwargs)

        self.num_heads = num_heads
        self.key_dim_per_head = key_dim # Renamed for clarity
        self.value_dim_per_head = value_dim or key_dim # Renamed for clarity
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.dim_mismatch_warnings = 0
        self.max_dim_mismatch_warnings = max_dim_mismatch_warnings

        # TensorFlow's MHA layer expects the size per head
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim_per_head, # Pass per-head dim
            value_dim=self.value_dim_per_head, # Pass per-head dim
            dropout=dropout,
            use_bias=use_bias,
            output_shape=output_shape, # Total output dimension
            attention_axes=attention_axes,
            kernel_initializer=kernel_initializer,
            name="MHA_layer"
        )

        # Layer normalization for stability
        if use_layer_norm:
            self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="MHA_LayerNorm")

        # For metrics tracking
        self._attention_scores_history = deque(maxlen=100) # Keep limited history
        self.metrics_tracker = { # Use dict for metrics
            "mean_attention_score": tf.Variable(0.0, trainable=False, dtype=tf.float32),
            "attention_entropy": tf.Variable(0.0, trainable=False, dtype=tf.float32),
            "attention_sparsity": tf.Variable(0.0, trainable=False, dtype=tf.float32)
        }
        logger.info(f"{self.name} initialized.")

    def _validate_and_standardize(self, tensor: tf.Tensor, expected_total_dim: int, name: str) -> tf.Tensor:
        """Validate tensor for NaN/Inf and standardize dimension if necessary."""
        if tensor is None:
            logger.error(f"Input tensor '{name}' is None.")
            # Return a zero tensor of the expected shape if possible, otherwise raise error
            # Note: Determining batch/sequence dimensions might be tricky here if they vary.
            # For simplicity, assuming batch=1, seq=1 if shape info is lost.
            # A more robust approach might involve passing expected shape.
            shape = [1, 1, expected_total_dim] # Placeholder shape
            return tf.zeros(shape, dtype=tf.float32)

        # Check for NaN/Inf values
        if tf.reduce_any(tf.math.is_nan(tensor)) or tf.reduce_any(tf.math.is_inf(tensor)):
            if self.dim_mismatch_warnings < self.max_dim_mismatch_warnings: # Reuse counter for this warning type
                 logger.warning(f"Invalid values (NaN/Inf) detected in '{name}'. Replacing with zeros.")
                 self.dim_mismatch_warnings += 1
                 if self.dim_mismatch_warnings == self.max_dim_mismatch_warnings:
                     logger.warning("Max NaN/Inf/Dim warnings reached.")
            return tf.zeros_like(tensor)

        # Check and standardize dimension
        actual_dim = tf.shape(tensor)[-1]
        if actual_dim != expected_total_dim:
            if self.dim_mismatch_warnings < self.max_dim_mismatch_warnings:
                 logger.warning(f"Dimension mismatch for '{name}': Expected {expected_total_dim}, got {actual_dim}. Applying strategy.")
                 # Increment logic is shared with NaN/Inf check above

            # Apply strategy (padding or truncation)
            if actual_dim < expected_total_dim:
                paddings = [[0, 0]] * (len(tf.shape(tensor)) - 1) + [[0, expected_total_dim - actual_dim]]
                tensor = tf.pad(tensor, paddings)
            else:
                tensor = tensor[..., :expected_total_dim]

        return tensor

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: Optional[tf.Tensor] = None, # Value can default to Key
        attention_mask: Optional[tf.Tensor] = None,
        return_attention_scores: bool = False,
        training: bool = False,
        use_causal_mask: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Apply attention mechanism.

        Args:
            query: Query tensor (shape: [Batch, TargetSeq, QueryDim]).
            key: Key tensor (shape: [Batch, SourceSeq, KeyDim]).
            value: Value tensor (shape: [Batch, SourceSeq, ValueDim]). Defaults to `key`.
            attention_mask: Optional mask tensor.
            return_attention_scores: Whether to return attention scores.
            training: Whether in training mode.
            use_causal_mask: Whether to use causal masking.

        Returns:
            Attention output tensor or tuple of (output, attention_scores).
        """
        if value is None:
            value = key # Default value to key if not provided

        # --- Determine Expected TOTAL Dimensions ---
        # MHA layer expects inputs compatible with its internal projections.
        # The total query dim should align with the query input to MHA.
        # The total key/value dims should align with the key/value inputs to MHA.
        # These dimensions are usually the embedding dimensions BEFORE head splitting.
        # We assume the caller provides tensors with these total dimensions.
        # Let's infer expected total dims from the first non-None tensor's shape
        # This is a simplification; a more robust way might be needed if inputs vary wildly.
        if query is not None: expected_q_dim = tf.shape(query)[-1]
        else: raise ValueError("Query tensor cannot be None")

        if key is not None: expected_k_dim = tf.shape(key)[-1]
        else: raise ValueError("Key tensor cannot be None")

        if value is not None: expected_v_dim = tf.shape(value)[-1]
        else: raise ValueError("Value tensor cannot be None") # Should not happen due to default above

        # Validate and standardize inputs based on inferred expected TOTAL dimensions
        query = self._validate_and_standardize(query, expected_q_dim, "query")
        key = self._validate_and_standardize(key, expected_k_dim, "key")
        value = self._validate_and_standardize(value, expected_v_dim, "value")

        # Store original query for residual connection if needed
        residual_query = query

        # Apply attention
        attn_output, attention_scores = self.mha(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            return_attention_scores=True, # Always get scores for metrics
            training=training,
            use_causal_mask=use_causal_mask
        )

        # Track metrics
        if not training: # Only track metrics during inference? Or always? Let's track always for now.
            self._update_attention_metrics(attention_scores)

        # Apply dropout (MHA layer already includes dropout)

        # Apply residual connection BEFORE layer norm (common practice)
        if self.use_residual:
             # Ensure shapes match for residual connection
             if residual_query.shape == attn_output.shape:
                 attn_output = attn_output + residual_query
             else:
                 logger.warning(f"Cannot apply residual connection due to shape mismatch: "
                                f"Query shape {residual_query.shape}, Output shape {attn_output.shape}")

        # Apply layer normalization
        if self.use_layer_norm:
            attn_output = self.layer_norm(attn_output, training=training)

        # --- Placeholder for persisting internal attention vectors ---
        # Example: if needed, one could save attention_scores or intermediate MHA outputs here
        # self.persist_attention_details(attention_scores, ...)

        if return_attention_scores:
            return attn_output, attention_scores
        return attn_output

    def _update_attention_metrics(self, attention_scores: tf.Tensor) -> None:
        """Update attention metrics based on attention scores."""
        if attention_scores is None: return

        try:
            # Calculate metrics using TensorFlow operations
            mean_score = tf.reduce_mean(attention_scores)

            epsilon = tf.keras.backend.epsilon() # Use Keras epsilon for stability
            log_scores = tf.math.log(attention_scores + epsilon)
            entropy = -tf.reduce_sum(attention_scores * log_scores, axis=-1)
            mean_entropy = tf.reduce_mean(entropy)

            sparsity_threshold = 0.01
            sparsity = tf.reduce_mean(tf.cast(tf.less(attention_scores, sparsity_threshold), tf.float32))

            # Update metric variables
            self.metrics_tracker["mean_attention_score"].assign(mean_score)
            self.metrics_tracker["attention_entropy"].assign(mean_entropy)
            self.metrics_tracker["attention_sparsity"].assign(sparsity)

            # Optional: Add scores to history deque (consider performance impact)
            # self._attention_scores_history.append(attention_scores.numpy())

        except Exception as e:
            logger.error(f"Error updating attention metrics: {e}", exc_info=False)


    def get_metrics(self) -> Dict[str, float]:
        """Get current attention metrics."""
        # Return numpy values from tf.Variables
        return {k: v.numpy().item() for k, v in self.metrics_tracker.items()}

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim_per_head, # Save per-head dim
            "value_dim": self.value_dim_per_head, # Save per-head dim
            "dropout": self.dropout,
            "use_bias": self.mha.use_bias,
            "output_shape": self.mha.output_shape,
            "attention_axes": self.mha.attention_axes,
            "kernel_initializer": tf.keras.initializers.serialize(self.mha.kernel_initializer),
            "use_layer_norm": self.use_layer_norm,
            "use_residual": self.use_residual,
            "max_dim_mismatch_warnings": self.max_dim_mismatch_warnings
        })
        return config
