"""
Confidence management for the Lucidia memory system.

Provides mechanisms for properly bounded confidence adjustments with natural
decay and recovery tendencies to prevent extreme values and oscillations.
"""

import time
import math
import logging
from typing import Dict, Any, Optional, Tuple


class BoundedConfidenceManager:
    """Manages confidence values with proper boundary constraints."""
    
    def __init__(self, min_confidence=0.0, max_confidence=1.0, 
                 decay_rate=0.01, recovery_rate=0.005):
        """Initialize the confidence manager with configurable parameters.
        
        Args:
            min_confidence: Minimum allowable confidence value
            max_confidence: Maximum allowable confidence value
            decay_rate: Rate at which high confidence naturally decays (per day)
            recovery_rate: Rate at which low confidence naturally recovers (per day)
        """
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.decay_rate = decay_rate  # Natural decay over time
        self.recovery_rate = recovery_rate  # Natural recovery over time
        self.logger = logging.getLogger(__name__)
        
    def apply_confidence_adjustment(self, current_confidence: float, adjustment: float,
                                    reason: str = "", time_since_last_update: float = 0) -> float:
        """Apply bounded confidence adjustment with natural decay/recovery.
        
        Args:
            current_confidence: Current confidence value (0.0-1.0)
            adjustment: The raw adjustment to apply (-1.0 to 1.0)
            reason: Reason for adjustment (for logging)
            time_since_last_update: Seconds since last confidence update
            
        Returns:
            New confidence value within bounds
        """
        # Apply natural decay/recovery based on time since last update
        days_since_update = time_since_last_update / 86400  # Convert to days
        
        if current_confidence > 0.5:
            # Higher confidence decays naturally
            time_factor = self.decay_rate * days_since_update
            natural_adjustment = -min(time_factor, 0.1)  # Cap at 0.1 per update
        else:
            # Lower confidence recovers naturally
            time_factor = self.recovery_rate * days_since_update
            natural_adjustment = min(time_factor, 0.05)  # Cap at 0.05 per update
            
        # Apply diminishing returns for adjustments near boundaries
        if adjustment > 0 and current_confidence > 0.8:
            # Diminish positive adjustments when already confident
            adjustment *= (1 - (current_confidence - 0.8) * 5)
        elif adjustment < 0 and current_confidence < 0.2:
            # Diminish negative adjustments when already low confidence
            adjustment *= (1 - (0.2 - current_confidence) * 5)
            
        # Apply combined adjustment
        new_confidence = current_confidence + adjustment + natural_adjustment
        
        # Ensure boundaries
        bounded_confidence = max(self.min_confidence, min(self.max_confidence, new_confidence))
        
        if abs(bounded_confidence - current_confidence) > 0.01:  # Only log non-trivial changes
            self.logger.debug(f"Confidence adjustment: {current_confidence:.2f} -> {bounded_confidence:.2f} "
                            f"(raw={adjustment:.2f}, natural={natural_adjustment:.2f}, reason='{reason}')")
            
        return bounded_confidence
    
    def calculate_confidence_from_evidence(self, evidence_items: list, 
                                          baseline_confidence: float = 0.5) -> float:
        """Calculate a confidence value from multiple evidence items.
        
        Args:
            evidence_items: List of dictionaries with 'confidence' and 'weight' keys
            baseline_confidence: Default confidence if no evidence provided
            
        Returns:
            Weighted confidence value
        """
        if not evidence_items:
            return baseline_confidence
            
        total_weight = sum(item.get('weight', 1.0) for item in evidence_items)
        if total_weight == 0:
            return baseline_confidence
            
        weighted_sum = sum(item.get('confidence', 0.5) * item.get('weight', 1.0) 
                          for item in evidence_items)
        
        # Calculate weighted average
        raw_confidence = weighted_sum / total_weight
        
        # Ensure boundaries
        return max(self.min_confidence, min(self.max_confidence, raw_confidence))
    
    def merge_confidence_values(self, values: list, weights: list = None) -> float:
        """Merge multiple confidence values with optional weights.
        
        Args:
            values: List of confidence values to merge
            weights: Optional list of weights for each value
            
        Returns:
            Merged confidence value
        """
        if not values:
            return 0.5  # Default neutral confidence
            
        if weights is None:
            # Equal weights
            weights = [1.0] * len(values)
        elif len(weights) != len(values):
            self.logger.warning(f"Confidence merge received {len(values)} values but {len(weights)} weights")
            # Extend or truncate weights to match values
            if len(weights) < len(values):
                weights.extend([1.0] * (len(values) - len(weights)))
            else:
                weights = weights[:len(values)]
                
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5  # Default if weights sum to zero
            
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        raw_confidence = weighted_sum / total_weight
        
        # Ensure boundaries
        return max(self.min_confidence, min(self.max_confidence, raw_confidence))
