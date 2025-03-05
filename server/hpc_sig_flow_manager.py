"""
LUCID RECALL PROJECT
Agent: Aurora 1.1
Date: 2/13/25
Time: 12:08 AM EST

HPC-SIG Flow Manager: Handles hypersphere processing chain and significance calculation
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HPCSIGFlowManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = {
            'chunk_size': 384,  # Match embedding dimension
            'embedding_dim': 768,
            'batch_size': 32,
            'momentum': 0.9,
            'diversity_threshold': 0.7,
            'surprise_threshold': 0.8,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **(config or {})
        }
        
        self.momentum_buffer = None
        self.current_batch = []
        self.batch_timestamps = []
        
        logger.info(f"Initialized HPCSIGFlowManager with config: {self.config}")
        
    async def process_embedding(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Process a single embedding through the HPC pipeline"""
        with torch.no_grad():
            # Log input shape
            logger.info(f"Input embedding shape: {embedding.shape}")
            
            # Move to correct device
            embedding = embedding.to(self.config['device'])
            
            # Ensure correct shape
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()[:384]
                logger.info(f"Flattened embedding shape: {embedding.shape}")
            
            # Project to unit hypersphere
            norm = torch.norm(embedding, p=2, dim=-1, keepdim=True)
            normalized = embedding / (norm + 1e-8)
            logger.info(f"Normalized embedding shape: {normalized.shape}")
            
            # Calculate surprise if we have momentum
            surprise_score = 0.0
            if self.momentum_buffer is not None:
                surprise_score = self._compute_surprise(normalized)
                logger.info(f"Calculated surprise score: {surprise_score}")
                
                # Apply shock absorber if surprise is high
                if surprise_score > self.config['surprise_threshold']:
                    normalized = self._apply_shock_absorber(normalized)
                    logger.info("Applied shock absorber")
            
            # Update momentum buffer
            self._update_momentum(normalized)
            
            # Calculate significance score
            significance = self._calculate_significance(normalized, surprise_score)
            logger.info(f"Calculated significance score: {significance}")
            
            return normalized, significance
            
    def _compute_surprise(self, embedding: torch.Tensor) -> float:
        """Calculate surprise score based on momentum buffer"""
        if self.momentum_buffer is None:
            return 0.0
            
        similarity = torch.matmul(embedding, self.momentum_buffer.T)
        return 1.0 - torch.mean(similarity).item()
        
    def _apply_shock_absorber(self, embedding: torch.Tensor) -> torch.Tensor:
        """Smooth out high-surprise embeddings"""
        if self.momentum_buffer is None:
            return embedding
            
        alpha = 1.0 - self.config['momentum']
        absorbed = alpha * embedding + (1 - alpha) * self.momentum_buffer[-1:]
        
        # Re-normalize
        norm = torch.norm(absorbed, p=2, dim=-1, keepdim=True)
        return absorbed / (norm + 1e-8)
        
    def _update_momentum(self, embedding: torch.Tensor):
        """Update momentum buffer with new embedding"""
        if self.momentum_buffer is None:
            self.momentum_buffer = embedding
        else:
            combined = torch.cat([self.momentum_buffer, embedding])
            self.momentum_buffer = combined[-self.config['chunk_size']:]
            
    def _calculate_significance(self, embedding: torch.Tensor, surprise: float) -> float:
        """Calculate significance score for memory storage"""
        magnitude = torch.norm(embedding).item()
        
        if self.momentum_buffer is not None:
            diversity = 1.0 - torch.max(torch.matmul(embedding, self.momentum_buffer.T)).item()
        else:
            diversity = 1.0
            
        # Combine factors (weights can be tuned)
        significance = (
            0.4 * surprise +
            0.3 * magnitude +
            0.3 * diversity
        )
        
        return significance
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current state statistics"""
        return {
            'has_momentum': self.momentum_buffer is not None,
            'momentum_size': len(self.momentum_buffer) if self.momentum_buffer is not None else 0,
            'device': self.config['device']
        }
