"""
LUCID RECALL PROJECT

HPC-QR Flow Manager: High-Performance Computing with QuickRecal approach for memory embeddings
Offers geometry-aware, causal/contextual, and self-organization-based memory scoring
"""

import logging
import asyncio
import torch
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class HPCQRFlowManager:
    """High-Performance Computing QuickRecal Flow Manager for memory embeddings
    
    Manages embedding processing using QuickRecal (QR) approach that unifies:
    - Geometry-Aware Distance (Riemannian/mixed curvature metrics)
    - Causal/Contextual Novelty
    - Self-Organization Divergence
    - Redundancy/Overlap penalization
    
    Optimized for asynchronous, non-blocking operations with parallel processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = {
            'chunk_size': 384,  # Match embedding dimension
            'embedding_dim': 768,
            'batch_size': 32,
            # HPC-QR weighting parameters
            'alpha': 0.4,  # Weight for geometry-aware distance
            'beta': 0.3,   # Weight for causal/contextual novelty
            'gamma': 0.2,  # Weight for self-organization divergence
            'delta': 0.1,  # Weight for redundancy/overlap penalty
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_threads': 4,  # Maximum number of threads for parallel processing
            'retry_attempts': 3,  # Number of retry attempts for failed operations
            'retry_backoff': 0.5,  # Base backoff time (seconds) for retries
            'timeout': 5.0,  # Default timeout for async operations
            **(config or {})
        }
        
        self.momentum_buffer = None
        self.current_batch = []
        self.batch_timestamps = []
        
        # Thread pool for CPU-intensive operations
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config['max_threads'])
        
        # Processing statistics
        self._stats = {
            'processed_count': 0,
            'error_count': 0,
            'retry_count': 0,
            'avg_processing_time': 0.0,
            'last_error': None,
            'total_processing_time': 0.0
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"Initialized HPCQRFlowManager with config: {self.config}")
    
    async def process_embedding(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Process a single embedding through the HPC-QR pipeline asynchronously
        
        Args:
            embedding: Input embedding tensor
            
        Returns:
            Tuple of (processed_embedding, quickrecal_score)
            
        Raises:
            TimeoutError: If processing exceeds configured timeout
            RuntimeError: If processing fails after all retry attempts
        """
        start_time = time.time()
        attempt = 0
        last_error = None
        
        while attempt < self.config['retry_attempts']:
            try:
                # Use asyncio.wait_for to add timeout
                result = await asyncio.wait_for(
                    self._process_embedding_internal(embedding),
                    timeout=self.config['timeout']
                )
                
                # Update statistics
                async with self._lock:
                    self._stats['processed_count'] += 1
                    proc_time = time.time() - start_time
                    self._stats['total_processing_time'] += proc_time
                    self._stats['avg_processing_time'] = (
                        self._stats['total_processing_time'] / self._stats['processed_count']
                    )
                
                return result
                
            except Exception as e:
                attempt += 1
                last_error = str(e)
                
                # Update error statistics
                async with self._lock:
                    self._stats['error_count'] += 1
                    self._stats['last_error'] = last_error
                    self._stats['retry_count'] += 1
                
                if attempt >= self.config['retry_attempts']:
                    logger.error(f"Failed to process embedding after {attempt} attempts: {last_error}")
                    raise RuntimeError(f"Failed to process embedding: {last_error}")
                
                # Exponential backoff with jitter
                backoff = self.config['retry_backoff'] * (2 ** (attempt - 1))
                backoff *= (0.5 + 0.5 * torch.rand(1).item())  # Add jitter (50-100% of backoff)
                
                logger.warning(f"Embedding processing error (attempt {attempt}): {e}. Retrying in {backoff:.2f}s")
                await asyncio.sleep(backoff)
        
        # This should not be reached due to the raise above, but just in case
        raise RuntimeError(f"Failed to process embedding: {last_error}")
    
    async def _process_embedding_internal(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Internal implementation of embedding processing with QuickRecal HPC-QR approach"""
        # Wrap CPU-intensive operations in run_in_executor
        loop = asyncio.get_event_loop()
        
        try:
            # Move to correct device and preprocess - non-blocking
            preprocess_future = loop.run_in_executor(
                self._thread_pool,
                self._preprocess_embedding,
                embedding
            )
            normalized = await preprocess_future
            
            # Calculate HPC-QR score
            qr_future = loop.run_in_executor(
                self._thread_pool,
                self._compute_hpc_qr,
                normalized
            )
            
            # Update momentum buffer in parallel
            await self._update_momentum_async(normalized)
            
            # Get the QuickRecal score
            quickrecal_score = await qr_future
            logger.info(f"Calculated QuickRecal score: {quickrecal_score}")
            
            return normalized, quickrecal_score
            
        except Exception as e:
            logger.error(f"Error in _process_embedding_internal: {e}")
            raise
    
    def _preprocess_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Preprocess embedding (CPU-bound operation)"""
        with torch.no_grad():
            # Move to correct device
            embedding = embedding.to(self.config['device'])
            
            # Ensure correct shape and size
            if len(embedding.shape) > 1:
                # Handle 2D tensor (batch x features)
                if embedding.shape[1] < self.config['chunk_size']:
                    # Pad if smaller than chunk_size
                    padded = torch.zeros((embedding.shape[0], self.config['chunk_size']), 
                                        device=embedding.device, dtype=embedding.dtype)
                    padded[:, :embedding.shape[1]] = embedding
                    embedding = padded
                else:
                    # Truncate if larger than chunk_size
                    embedding = embedding[:, :self.config['chunk_size']]
                
                # Flatten if necessary for further processing
                embedding = embedding.reshape(-1)[:self.config['chunk_size']]
            else:
                # Handle 1D tensor
                if embedding.shape[0] < self.config['chunk_size']:
                    # Pad if smaller than chunk_size
                    padded = torch.zeros(self.config['chunk_size'], 
                                        device=embedding.device, dtype=embedding.dtype)
                    padded[:embedding.shape[0]] = embedding
                    embedding = padded
                else:
                    # Truncate if larger than chunk_size
                    embedding = embedding[:self.config['chunk_size']]
            
            # Project to unit hypersphere
            norm = torch.norm(embedding, p=2, dim=-1, keepdim=True)
            normalized = embedding / (norm + 1e-8)
            
            return normalized
    
    def _compute_hpc_qr(self, embedding: torch.Tensor) -> float:
        """
        Calculate QuickRecal HPC-QR score for memory prioritization.
        
        HPC_QR(x) = alpha * R_geom(x)    # Geometry-aware distance
                  + beta  * C_novel(x)   # Causal/contextual novelty
                  + gamma * S_org(x)     # Self-organization divergence
                  - delta * O_red(x)     # Redundancy/overlap penalty
        """
        with torch.no_grad():
            alpha = self.config['alpha']
            beta = self.config['beta']
            gamma = self.config['gamma']
            delta = self.config['delta']
            
            # Initialize with default values for cold-start scenarios
            # Default to midrange values when we have no history to compare against
            r_geom = 0.5  # Default geometric distance - midrange
            c_novel = 0.7  # Default novelty - high for first entries
            s_org = 0.5    # Default self-organization - midrange
            o_red = 0.0    # Default redundancy - none for first entries
            
            if self.momentum_buffer is not None and len(self.momentum_buffer) > 0:
                # 1) R_geom: Geometry-aware Riemannian distance calculation
                # Compute the mean of the momentum buffer as a center
                center = torch.mean(self.momentum_buffer, dim=0)
                
                # Enhanced implementation: Use a mixed curvature approach combining
                # both Euclidean and hyperbolic-like metrics for better representation
                # of hierarchical and non-hierarchical relationships
                
                # Basic cosine distance forms our first component
                cos_dist = 1.0 - torch.cosine_similarity(embedding, center, dim=0)
                
                # Hyperbolic-inspired component (approximated)
                # In a true hyperbolic space, we'd compute:
                # d_h(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
                norm_diff = torch.norm(embedding - center, p=2)
                hyperbolic_factor = 2.0 * norm_diff / (1.0 + norm_diff)
                
                # Combine both distance metrics
                r_geom = float(0.7 * cos_dist + 0.3 * hyperbolic_factor)
                
                # 2) C_novel: Causal/contextual novelty
                # Consider surprise based on the entire momentum buffer distribution
                similarities = torch.matmul(embedding, self.momentum_buffer.T)
                max_sim = torch.max(similarities).item()
                mean_sim = torch.mean(similarities).item()
                std_sim = torch.std(similarities).item() + 1e-5  # avoid division by zero
                
                # Compute surprise as deviation from expected similarity
                z_score = abs(max_sim - mean_sim) / std_sim
                
                # Calculate surprise factor properly with tensor inputs
                exp_input = torch.tensor(-z_score, device=embedding.device)
                surprise_factor = 2.0 / (1.0 + torch.exp(exp_input).item())
                
                # Traditional novelty (difference from most recent)
                last_emb = self.momentum_buffer[-1] if len(self.momentum_buffer) > 1 else self.momentum_buffer[0]
                recency_diff = float(torch.norm(embedding - last_emb, p=2))
                
                # Combine distributional surprise with recency difference
                c_novel = 0.6 * surprise_factor + 0.4 * recency_diff
                
                # 3) S_org: Self-organization divergence 
                # Enhanced measurement of how much the memory organization would need to adapt
                # Compute pairwise distances
                distances = torch.cdist(embedding.unsqueeze(0), self.momentum_buffer, p=2)[0]
                
                # Identify nearest neighbors (k=3 or fewer if buffer is smaller)
                k = min(3, len(distances))
                _, indices = torch.topk(distances, k, largest=False)
                neighbors = self.momentum_buffer[indices]
                
                # Calculate how much this embedding would "stretch" the local topology
                if k > 1:
                    # Original neighbor distances to each other
                    original_dists = torch.cdist(neighbors, neighbors, p=2)
                    # New distances if embedding is added to the neighborhood
                    with_new = torch.cat([neighbors, embedding.unsqueeze(0)], dim=0)
                    new_dists = torch.cdist(with_new, with_new, p=2)
                    # Calculate topology stretch as the increase in average distance
                    orig_avg = torch.mean(original_dists).item()
                    new_avg = torch.mean(new_dists).item()
                    topology_stretch = new_avg / (orig_avg + 1e-8) - 1.0
                    
                    # Scale to a reasonable range
                    s_org = float(min(1.0, max(0.0, topology_stretch)))
                else:
                    # With only one neighbor, use normalized distance
                    s_org = float(distances[indices[0]] / (torch.norm(neighbors[0]) + 1e-8))
                
                # 4) O_red: Redundancy/overlap with existing memories
                similarities = torch.matmul(embedding, self.momentum_buffer.T)
                max_sim = torch.max(similarities).item()
                
                # Apply exponential scaling to increase penalty for highly similar items
                if max_sim > 0.8:  # Only penalize for high similarity
                    similarity_sharpness = 5.0
                    
                    # Ensure we're using torch.exp with tensor input
                    exp_input = torch.tensor(-similarity_sharpness * (1.0 - max_sim), device=embedding.device)
                    o_red = 1.0 - torch.exp(exp_input).item()
                else:
                    o_red = 0.0  # No significant redundancy detected
            
            # Calculate the unified HPC-QR score
            hpc_qr_score = (alpha * r_geom + 
                           beta * c_novel + 
                           gamma * s_org - 
                           delta * o_red)
            
            # Apply sigmoid scaling to ensure balanced score distribution
            import math
            # Simple sigmoid function to produce a more balanced score distribution
            sigmoid = lambda x: 1.0 / (1.0 + math.exp(-6.0 * (x - 0.5)))
            hpc_qr_score = sigmoid(hpc_qr_score)
            
            # Ensure the score is in a normalized range
            hpc_qr_score = max(0.1, min(1.0, hpc_qr_score))  # Minimum 0.1 to avoid zeros
            
            logger.debug(f"HPC-QR components: r_geom={r_geom:.3f}, c_novel={c_novel:.3f}, s_org={s_org:.3f}, o_red={o_red:.3f}")
            return float(hpc_qr_score)
    
    async def _update_momentum_async(self, embedding: torch.Tensor):
        """Update momentum buffer with new embedding (thread-safe)"""
        async with self._lock:
            if self.momentum_buffer is None:
                self.momentum_buffer = embedding.unsqueeze(0)  # [1, dim]
            else:
                # Stack the new embedding as a new row
                if len(self.momentum_buffer.shape) == 1:
                    # Convert 1D tensor to 2D for first append
                    self.momentum_buffer = torch.stack([self.momentum_buffer, embedding])
                else:
                    combined = torch.cat([self.momentum_buffer, embedding.unsqueeze(0)], dim=0)
                    # Keep only the last 'chunk_size' embeddings to limit memory use
                    if combined.shape[0] > self.config['chunk_size']:
                        combined = combined[-self.config['chunk_size']:]
                    self.momentum_buffer = combined
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current state statistics"""
        stats = {
            'has_momentum': self.momentum_buffer is not None,
            'momentum_size': 0 if self.momentum_buffer is None else (
                self.momentum_buffer.shape[0] if len(self.momentum_buffer.shape) > 1 else 1
            ),
            'device': self.config['device'],
            'processed_count': self._stats['processed_count'],
            'error_count': self._stats['error_count'],
            'retry_count': self._stats['retry_count'],
            'avg_processing_time': self._stats['avg_processing_time'],
            'last_error': self._stats['last_error'],
        }
        return stats
    
    async def get_embedding(self, text: str) -> torch.Tensor:
        """
        Generate an embedding vector from text content by calling the TensorServer.
        
        Args:
            text: The text content to generate an embedding for
            
        Returns:
            Tensor embedding representation of the text
        
        Raises:
            RuntimeError: If embedding generation fails after retries
        """
        from server.tensor_server import TensorClient
        
        # Use TensorClient to get embedding from TensorServer
        try:
            # Lazy initialization of the client
            if not hasattr(self, '_tensor_client'):
                self._tensor_client = TensorClient()
                await self._tensor_client.connect()
                
            # Get embedding from TensorServer
            result = await self._tensor_client.get_embedding(text)
            
            if not result or 'embeddings' not in result:
                raise RuntimeError(f"Failed to get embedding from TensorServer: {result}")
                
            # Convert to tensor and ensure it's on the right device
            embedding = torch.tensor(result['embeddings'], dtype=torch.float32, device=self.config['device'])
            
            # Ensure correct shape
            if len(embedding.shape) > 1:
                embedding = embedding.squeeze()
                
            # Normalize if needed
            norm = torch.norm(embedding, p=2)
            if abs(norm - 1.0) > 1e-5:  # Not already normalized
                embedding = embedding / norm
                
            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}") from e
    
    async def close(self):
        """Clean up resources used by this manager"""
        logger.info("Shutting down HPCQRFlowManager...")
        self._thread_pool.shutdown(wait=True)
        logger.info("HPCQRFlowManager shutdown complete")