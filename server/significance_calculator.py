"""
LUCID RECALL PROJECT
Agent: Aurora 1.1
Date: 2/13/25
Time: 5:50 AM EST

Significance Calculator: Memory Importance Evaluation
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SignificanceCalculator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {
            'time_weight': 0.4,    # aT weight
            'info_weight': 0.3,    # bI weight
            'pattern_weight': 0.3,  # cP weight
            'decay_rate': 0.1,
            'history_window': 1000,
            'batch_size': 32,
            **(config or {})
        }
        
        self.significance_history = []
        self.start_time = time.time()
        
        logger.info(f"Initialized SignificanceCalculator with config: {self.config}")

    def calculate(self, embedding: torch.Tensor, context: List[torch.Tensor]) -> float:
        """
        Calculate significance using the equation:
        Significance = ∫(aT + bI + cP) × f(t) × g(past) × ψ(x,t) dt
        """
        try:
            # Calculate components
            temporal = self._temporal_component()  # T
            info = self._information_component(embedding, context)  # I
            pattern = self._predictive_component(embedding, context)  # P
            
            # Combine with weights (a, b, c)
            weighted_sum = (
                self.config['time_weight'] * temporal +
                self.config['info_weight'] * info +
                self.config['pattern_weight'] * pattern
            )
            
            # Apply time evolution f(t)
            time_evolution = self._time_evolution()
            
            # Apply historical context g(past)
            historical = self._historical_context()
            
            # Apply state function ψ(x,t)
            state = self._state_function(embedding)
            
            # Calculate final significance
            significance = weighted_sum * time_evolution * historical * state
            
            # Track history
            self.significance_history.append(significance.item())
            if len(self.significance_history) > self.config['history_window']:
                self.significance_history = self.significance_history[-self.config['history_window']:]
            
            return significance.item()
            
        except Exception as e:
            logger.error(f"Error calculating significance: {str(e)}")
            return 0.0

    def _information_component(self, embedding: torch.Tensor, context: List[torch.Tensor]) -> torch.Tensor:
        """Calculate information density/uniqueness"""
        if not context:
            return torch.tensor(1.0, device=embedding.device)
            
        with torch.no_grad():
            context_tensor = torch.stack(context).to(embedding.device)
            similarities = torch.matmul(embedding, context_tensor.T)
            uniqueness = 1.0 - torch.mean(similarities)
            return torch.clamp(uniqueness, 0.0, 1.0)

    def _predictive_component(self, embedding: torch.Tensor, context: List[torch.Tensor]) -> torch.Tensor:
        """Calculate predictive value"""
        if not context:
            return torch.tensor(0.5, device=embedding.device)
            
        with torch.no_grad():
            recent = context[-10:]  # Use last 10 memories
            context_tensor = torch.stack(recent).to(embedding.device)
            pred_value = torch.matmul(embedding, context_tensor.T)
            return torch.clamp(torch.sigmoid(pred_value.mean()), 0.0, 1.0)

    def _temporal_component(self) -> torch.Tensor:
        """Calculate temporal relevance"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        return torch.tensor(np.exp(-self.config['decay_rate'] * elapsed))

    def _time_evolution(self) -> float:
        """Time evolution function f(t)"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        return np.sin(2 * np.pi * elapsed / (24 * 3600)) * 0.5 + 0.5  # Daily cycle

    def _historical_context(self) -> float:
        """Historical context function g(past)"""
        if not self.significance_history:
            return 1.0
        recent_significance = np.mean(self.significance_history[-10:])
        return np.clip(recent_significance, 0.1, 1.0)

    def _state_function(self, embedding: torch.Tensor) -> float:
        """State function ψ(x,t)"""
        # Use embedding magnitude as state indicator
        with torch.no_grad():
            magnitude = torch.norm(embedding)
            return torch.clamp(torch.sigmoid(magnitude), 0.1, 1.0).item()

    def get_stats(self) -> Dict[str, Any]:
        """Get calculator statistics"""
        stats = {
            'history_size': len(self.significance_history),
            'uptime': time.time() - self.start_time
        }
        
        if self.significance_history:
            stats.update({
                'mean': np.mean(self.significance_history),
                'std': np.std(self.significance_history),
                'min': np.min(self.significance_history),
                'max': np.max(self.significance_history)
            })
            
        return stats
