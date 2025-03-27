# synthians_memory_core/adaptive_components.py

import time
import math
from collections import deque
from typing import Dict, Any, Optional

from .custom_logger import logger # Use the shared custom logger

class ThresholdCalibrator:
    """Dynamically calibrates similarity thresholds based on feedback."""

    def __init__(self, initial_threshold: float = 0.75, learning_rate: float = 0.05, window_size: int = 50):
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.feedback_history = deque(maxlen=window_size)
        self.stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} # Added tn for completeness
        logger.info("ThresholdCalibrator", "Initialized", {"initial": initial_threshold, "lr": learning_rate, "window": window_size})

    def record_feedback(self, similarity_score: float, was_relevant: bool):
        """Record feedback for a retrieved memory."""
        is_above_threshold = similarity_score >= self.threshold

        self.feedback_history.append({
            "score": similarity_score,
            "relevant": was_relevant,
            "predicted_relevant": is_above_threshold,
            "threshold_at_time": self.threshold
        })

        # Update stats based on prediction vs actual relevance
        if is_above_threshold:
            if was_relevant: self.stats['tp'] += 1
            else: self.stats['fp'] += 1
        else:
            if was_relevant: self.stats['fn'] += 1
            else: self.stats['tn'] += 1 # Correctly predicted irrelevant

        # Adjust threshold immediately based on this feedback
        self.adjust_threshold()

    def adjust_threshold(self) -> float:
        """Adjust the similarity threshold based on recent feedback."""
        if len(self.feedback_history) < 10: # Need minimum feedback
            return self.threshold

        # Calculate Precision and Recall from recent history (last N items)
        recent_feedback = list(self.feedback_history)
        recent_tp = sum(1 for f in recent_feedback if f["predicted_relevant"] and f["relevant"])
        recent_fp = sum(1 for f in recent_feedback if f["predicted_relevant"] and not f["relevant"])
        recent_fn = sum(1 for f in recent_feedback if not f["predicted_relevant"] and f["relevant"])

        precision = recent_tp / max(1, recent_tp + recent_fp)
        recall = recent_tp / max(1, recent_tp + recent_fn)

        adjustment = 0.0
        # If precision is low (too many irrelevant items retrieved), increase threshold
        if precision < 0.6 and recall > 0.5: # Avoid penalizing if recall is also low
            adjustment = self.learning_rate * (1.0 - precision) # Stronger increase for lower precision
        # If recall is low (too many relevant items missed), decrease threshold
        elif recall < 0.6 and precision > 0.5: # Avoid penalizing if precision is also low
             adjustment = -self.learning_rate * (1.0 - recall) # Stronger decrease for lower recall

        # Apply adjustment with diminishing returns near bounds
        current_threshold = self.threshold
        if adjustment > 0:
            # Less adjustment as we approach 1.0
            adjustment *= (1.0 - current_threshold)
        else:
             # Less adjustment as we approach 0.0
             adjustment *= current_threshold

        new_threshold = current_threshold + adjustment
        new_threshold = max(0.1, min(0.95, new_threshold)) # Keep within reasonable bounds

        if abs(new_threshold - self.threshold) > 0.001:
            logger.info("ThresholdCalibrator", f"Adjusted threshold: {self.threshold:.3f} -> {new_threshold:.3f}",
                        {"adjustment": adjustment, "precision": precision, "recall": recall})
            self.threshold = new_threshold

        return self.threshold

    def get_current_threshold(self) -> float:
        """Return the current similarity threshold."""
        return self.threshold

    def get_statistics(self) -> dict:
        """Return statistics about calibration performance."""
        total = self.stats['tp'] + self.stats['fp'] + self.stats['fn'] + self.stats['tn']
        precision = self.stats['tp'] / max(1, self.stats['tp'] + self.stats['fp'])
        recall = self.stats['tp'] / max(1, self.stats['tp'] + self.stats['fn'])
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        return {
            "threshold": self.threshold,
            "feedback_count": len(self.feedback_history),
            "true_positives": self.stats['tp'],
            "false_positives": self.stats['fp'],
            "false_negatives": self.stats['fn'],
            "true_negatives": self.stats['tn'],
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

# Note: AdaptiveBatchScheduler might be overkill if batching is handled externally
# or if the primary interaction pattern doesn't benefit significantly from adaptive batching.
# Keeping ThresholdCalibrator as it's directly related to retrieval relevance.
