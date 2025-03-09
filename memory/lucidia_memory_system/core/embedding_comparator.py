"""
LUCID RECALL PROJECT
Embedding Comparator

Provides standardized interfaces for generating embeddings
and comparing their similarity across memory components.
"""

# Try to import torch, but create a fallback if it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using NumPy fallback for embedding operations.")

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
    
    async def get_embedding(self, text: str) -> Optional[Union[np.ndarray, 'torch.Tensor']]:
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
    
    def _normalize_embedding(self, embedding: Union[List[float], np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Normalize embedding to unit vector.
        
        Args:
            embedding: Embedding to normalize
            
        Returns:
            Normalized embedding tensor or ndarray
        """
        self.stats['embeddings_normalized'] += 1
        
        # Convert to appropriate type based on torch availability
        if TORCH_AVAILABLE:
            if not isinstance(embedding, torch.Tensor):
                if isinstance(embedding, list):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                elif isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding).float()
                    
            # Normalize using PyTorch
            norm = torch.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        else:
            # Fallback to NumPy implementation
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
                
            # Normalize using NumPy
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
    
    async def compare(self, embedding1: Union[np.ndarray, 'torch.Tensor'], embedding2: Union[np.ndarray, 'torch.Tensor']) -> float:
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
            if TORCH_AVAILABLE:
                similarity = torch.dot(embedding1, embedding2).item()
            else:
                similarity = np.dot(embedding1, embedding2)
            
            # Ensure result is in valid range
            similarity = max(0.0, min(1.0, similarity))
            
            self.stats['comparisons_made'] += 1
            return similarity
            
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return 0.0
    
    async def batch_compare(self, query_embedding: Union[np.ndarray, 'torch.Tensor'], 
                          embeddings: List[Union[np.ndarray, 'torch.Tensor']]) -> List[float]:
        """
        Compare query embedding against multiple embeddings.
        
        Args:
            query_embedding: Query embedding
            embeddings: List of embeddings to compare against
            
        Returns:
            List of similarity scores (0.0-1.0)
        """
        try:
            results = []
            
            # Optimize batch computation based on available library
            if TORCH_AVAILABLE and isinstance(query_embedding, torch.Tensor):
                # Ensure all embeddings are torch tensors
                tensor_embeddings = []
                for emb in embeddings:
                    if isinstance(emb, np.ndarray):
                        tensor_embeddings.append(torch.from_numpy(emb).float())
                    elif isinstance(emb, list):
                        tensor_embeddings.append(torch.tensor(emb, dtype=torch.float32))
                    else:  # already a torch tensor
                        tensor_embeddings.append(emb)
                
                # Stack embeddings for batch operation
                if tensor_embeddings:
                    stacked = torch.stack(tensor_embeddings)
                    # Compute dot product for all embeddings at once
                    similarities = torch.matmul(stacked, query_embedding).tolist()
                    
                    # Ensure results are in valid range [0, 1]
                    results = [max(0.0, min(1.0, sim)) for sim in similarities]
                
            else:  # NumPy fallback
                # Ensure query_embedding is numpy array
                if TORCH_AVAILABLE and isinstance(query_embedding, torch.Tensor):
                    query_np = query_embedding.cpu().numpy()
                else:
                    query_np = query_embedding if isinstance(query_embedding, np.ndarray) else np.array(query_embedding)
                
                # Process each embedding individually
                for emb in embeddings:
                    if TORCH_AVAILABLE and isinstance(emb, torch.Tensor):
                        emb_np = emb.cpu().numpy()
                    elif isinstance(emb, list):
                        emb_np = np.array(emb)
                    else:  # already numpy
                        emb_np = emb
                    
                    similarity = np.dot(query_np, emb_np)
                    # Ensure result is in valid range
                    similarity = max(0.0, min(1.0, similarity))
                    results.append(similarity)
            
            self.stats['comparisons_made'] += len(results)
            return results
            
        except Exception as e:
            logger.error(f"Error in batch compare: {e}")
            # Return zeros as fallback
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