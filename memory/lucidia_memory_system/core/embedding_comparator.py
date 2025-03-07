"""
LUCID RECALL PROJECT
Embedding Comparator

Provides standardized interfaces for generating embeddings
and comparing their similarity across memory components.
"""

import torch
import logging
import asyncio
from typing import Dict, Any, Optional, Union, List
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingComparator:
    """
    Provides standardized methods for embedding generation and comparison.
    
    This class serves as an interface layer for the HPC system, allowing
    different components to generate and compare embeddings consistently.
    """
    
    def __init__(self, hpc_client, embedding_dim: int = 384):
        """
        Initialize the embedding comparator.
        
        Args:
            hpc_client: HPC client for embedding generation
            embedding_dim: Embedding dimension
        """
        self.hpc_client = hpc_client
        self.embedding_dim = embedding_dim
        self._embedding_cache = {}
        self._cache_limit = 1000
        self._lock = asyncio.Lock()
        
        # Performance tracking
        self.stats = {
            'embeddings_generated': 0,
            'embeddings_normalized': 0,
            'comparisons_made': 0,
            'cache_hits': 0
        }
        
        logger.info(f"Initialized EmbeddingComparator with dim={embedding_dim}")
    
    async def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Generate embedding for text with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding tensor or None on failure
        """
        # Check cache first
        cache_key = text.strip()
        if cache_key in self._embedding_cache:
            self.stats['cache_hits'] += 1
            return self._embedding_cache[cache_key]
        
        try:
            # Get embedding through HPC client
            embedding = await self.hpc_client.get_embedding(text)
            
            if embedding is None:
                logger.warning(f"Failed to generate embedding for text: {text[:50]}...")
                return None
            
            # Normalize embedding if needed
            embedding = self._normalize_embedding(embedding)
            
            # Cache the embedding
            async with self._lock:
                self._embedding_cache[cache_key] = embedding
                
                # Prune cache if needed
                if len(self._embedding_cache) > self._cache_limit:
                    # Remove oldest (first) item
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]
            
            self.stats['embeddings_generated'] += 1
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def _normalize_embedding(self, embedding: Union[torch.Tensor, np.ndarray, List[float]]) -> torch.Tensor:
        """
        Normalize embedding to unit vector.
        
        Args:
            embedding: Embedding to normalize
            
        Returns:
            Normalized embedding tensor
        """
        try:
            # Convert to torch tensor if not already
            if not isinstance(embedding, torch.Tensor):
                if isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding).float()
                elif isinstance(embedding, list):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported embedding type: {type(embedding)}")
            
            # Ensure correct shape
            if len(embedding.shape) > 1 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            
            # Compute L2 norm
            norm = torch.norm(embedding, p=2)
            
            # Normalize if norm is non-zero
            if norm > 0:
                normalized = embedding / norm
            else:
                # If norm is zero, return original to avoid NaN
                normalized = embedding
                
            self.stats['embeddings_normalized'] += 1
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing embedding: {e}")
            # Return original embedding as fallback
            if isinstance(embedding, torch.Tensor):
                return embedding
            elif isinstance(embedding, np.ndarray):
                return torch.from_numpy(embedding).float()
            elif isinstance(embedding, list):
                return torch.tensor(embedding, dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported embedding type: {type(embedding)}")
    
    async def compare(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compare two embeddings and return similarity score.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0.0-1.0)
        """
        try:
            # Normalize embeddings if necessary
            embedding1 = self._normalize_embedding(embedding1)
            embedding2 = self._normalize_embedding(embedding2)
            
            # Ensure correct shapes for dot product
            if len(embedding1.shape) > 1:
                embedding1 = embedding1.squeeze()
            if len(embedding2.shape) > 1:
                embedding2 = embedding2.squeeze()
                
            # Cosine similarity (dot product of normalized vectors)
            similarity = torch.dot(embedding1, embedding2).item()
            
            # Ensure result is in valid range
            similarity = max(0.0, min(1.0, similarity))
            
            self.stats['comparisons_made'] += 1
            return similarity
            
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return 0.0
    
    async def batch_compare(self, query_embedding: torch.Tensor, 
                          embeddings: List[torch.Tensor]) -> List[float]:
        """
        Compare query embedding against multiple embeddings.
        
        Args:
            query_embedding: Query embedding
            embeddings: List of embeddings to compare against
            
        Returns:
            List of similarity scores (0.0-1.0)
        """
        try:
            # Normalize query embedding
            query_embedding = self._normalize_embedding(query_embedding)
            
            # Calculate similarities for each embedding
            similarities = []
            for emb in embeddings:
                similarity = await self.compare(query_embedding, emb)
                similarities.append(similarity)
                
            self.stats['comparisons_made'] += len(embeddings)
            return similarities
            
        except Exception as e:
            logger.error(f"Error in batch comparison: {e}")
            return [0.0] * len(embeddings)
    
    async def clear_cache(self) -> None:
        """Clear the embedding cache."""
        async with self._lock:
            self._embedding_cache.clear()
            logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comparator statistics."""
        return {
            'embeddings_generated': self.stats['embeddings_generated'],
            'embeddings_normalized': self.stats['embeddings_normalized'],
            'comparisons_made': self.stats['comparisons_made'],
            'cache_hits': self.stats['cache_hits'],
            'cache_size': len(self._embedding_cache),
            'cache_limit': self._cache_limit,
            'cache_utilization': len(self._embedding_cache) / self._cache_limit
        }