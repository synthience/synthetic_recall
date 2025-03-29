import time
import logging
from collections import deque
from typing import Deque, Tuple, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Define the structure of the context tuple for clarity
ContextTuple = Tuple[float, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)

class SequenceContextManager:
    """
    Manages a deque-based context buffer for storing attention-related embeddings and projections.

    Maintains a fixed-length history of (timestamp, memory_id, x_t, k_t, v_t, q_t, y_t) tuples
    for attention calculations in Titans architecture variants.
    """

    def __init__(self, max_length: int = 50):
        """
        Initialize the sequence context manager.

        Args:
            max_length: Maximum number of context tuples to store.
        """
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        self.max_length = max_length
        self._context_buffer: Deque[ContextTuple] = deque(maxlen=max_length)
        logger.info(f"SequenceContextManager initialized with max_length={max_length}")

    def add_context(
        self,
        memory_id: str,
        x_t: np.ndarray,
        k_t: np.ndarray,
        v_t: np.ndarray,
        q_t: np.ndarray,
        y_t: np.ndarray, # Output from NeuralMemory.call
        timestamp: Optional[float] = None
    ) -> None:
        """
        Add a new context element (tuple) to the buffer.

        Args:
            memory_id: Identifier for the memory entry.
            x_t: Input embedding (np.ndarray).
            k_t: Key projection (np.ndarray).
            v_t: Value projection (np.ndarray).
            q_t: Query projection (np.ndarray).
            y_t: Neural memory output embedding (np.ndarray).
            timestamp: Optional timestamp (defaults to current time).
        """
        ts = timestamp if timestamp is not None else time.time()

        # Basic validation of inputs
        if not all(isinstance(arr, np.ndarray) for arr in [x_t, k_t, v_t, q_t, y_t]):
            logger.error("Invalid input type for context tuple. All embeddings/projections must be numpy arrays.")
            # Decide how to handle: raise error or skip adding? Let's skip for robustness.
            return

        context_tuple: ContextTuple = (ts, memory_id, x_t, k_t, v_t, q_t, y_t)
        self._context_buffer.append(context_tuple)
        logger.debug(f"Added context for memory {memory_id} to buffer (size: {len(self._context_buffer)})")

    def update_last_context(self, y_t: np.ndarray) -> bool:
        """Update the most recent context entry with the y_t value.
        
        This is useful when y_t is not available at the time of initial context creation,
        such as when we need to add context before Neural Memory retrieval but only get
        the y_t value after retrieval.
        
        Args:
            y_t: The retrieved embedding (output from Neural Memory)
            
        Returns:
            True if update was successful, False otherwise
        """
        if not len(self._context_buffer):
            logger.warning("Cannot update last context: buffer is empty")
            return False
            
        if not isinstance(y_t, np.ndarray):
            logger.error("Invalid y_t type for context update. Must be numpy array.")
            return False
            
        # Get the last context tuple
        last_tuple = self._context_buffer[-1]
        
        # Create a new tuple with the updated y_t
        updated_tuple = (
            last_tuple[0],  # timestamp
            last_tuple[1],  # memory_id
            last_tuple[2],  # x_t
            last_tuple[3],  # k_t
            last_tuple[4],  # v_t
            last_tuple[5],  # q_t
            y_t             # updated y_t
        )
        
        # Replace the last tuple
        self._context_buffer[-1] = updated_tuple
        logger.debug(f"Updated last context entry for memory {last_tuple[1]} with y_t")
        return True

    def get_recent_history(self, count: Optional[int] = None) -> List[ContextTuple]:
        """Get the most recent context tuples."""
        num_items = count if count is not None else len(self._context_buffer)
        num_items = min(num_items, len(self._context_buffer)) # Don't request more than available
        if num_items <= 0:
            return []
        # Return a list slice of the deque
        return list(self._context_buffer)[-num_items:]

    def get_recent_keys(self, count: Optional[int] = None) -> List[np.ndarray]:
        """Get the most recent key projections (k_t)."""
        history = self.get_recent_history(count)
        return [item[3] for item in history] # Index 3 is k_t

    def get_recent_values(self, count: Optional[int] = None) -> List[np.ndarray]:
        """Get the most recent value projections (v_t)."""
        history = self.get_recent_history(count)
        return [item[4] for item in history] # Index 4 is v_t

    def get_recent_queries(self, count: Optional[int] = None) -> List[np.ndarray]:
        """Get the most recent query projections (q_t)."""
        history = self.get_recent_history(count)
        return [item[5] for item in history] # Index 5 is q_t

    def get_recent_outputs(self, count: Optional[int] = None) -> List[np.ndarray]:
        """Get the most recent neural memory outputs (y_t)."""
        history = self.get_recent_history(count)
        return [item[6] for item in history] # Index 6 is y_t

    def get_recent_kv_pairs(self, count: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Convenience method to get recent (Key, Value) pairs for attention."""
        history = self.get_recent_history(count)
        keys = [item[3] for item in history]
        values = [item[4] for item in history]
        return keys, values

    def get_recent_ky_pairs(self, count: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Convenience method to get recent (Key, Output) pairs for MAC attention."""
        history = self.get_recent_history(count)
        keys = [item[3] for item in history]
        outputs = [item[6] for item in history]
        return keys, outputs

    def __len__(self) -> int:
        """Return the current number of items in the buffer."""
        return len(self._context_buffer)

    def clear(self) -> None:
        """Clear the context buffer."""
        self._context_buffer.clear()
        logger.info("SequenceContextManager buffer cleared.")
