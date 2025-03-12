"""
LUCID RECALL PROJECT
Agent: Aurora 1.1
Date: 2/13/25
Time: 12:08 AM EST

HPC-SIG Flow Manager: Handles hypersphere processing chain and significance calculation
"""

import logging
import asyncio
import torch
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class HPCSIGFlowManager:
    """High-Performance Computing SIG Flow Manager for memory embeddings
    
    Manages embedding processing, significance calculation, and memory flow.
    Optimized for asynchronous, non-blocking operations with parallel processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = {
            'chunk_size': 384,  # Match embedding dimension
            'embedding_dim': 768,
            'batch_size': 32,
            'momentum': 0.9,
            'diversity_threshold': 0.7,
            'surprise_threshold': 0.8,
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
        
        logger.info(f"Initialized HPCSIGFlowManager with config: {self.config}")
    
    async def process_embedding(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Process a single embedding through the HPC pipeline asynchronously
        
        Args:
            embedding: Input embedding tensor
            
        Returns:
            Tuple of (processed_embedding, significance_score)
            
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
        """Internal implementation of embedding processing with parallel components"""
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
            
            # All the following can happen in parallel
            # Calculate surprise if we have momentum
            surprise_score = 0.0
            surprise_future = None
            shock_absorber_future = None
            
            async with self._lock:
                if self.momentum_buffer is not None:
                    # Calculate surprise score
                    surprise_future = loop.run_in_executor(
                        self._thread_pool,
                        self._compute_surprise,
                        normalized
                    )
            
            # Wait for surprise calculation to complete if initiated
            if surprise_future:
                surprise_score = await surprise_future
                logger.info(f"Calculated surprise score: {surprise_score}")
                
                # Apply shock absorber if surprise is high
                if surprise_score > self.config['surprise_threshold']:
                    shock_absorber_future = loop.run_in_executor(
                        self._thread_pool,
                        self._apply_shock_absorber,
                        normalized
                    )
                    normalized = await shock_absorber_future
                    logger.info("Applied shock absorber")
            
            # Update momentum buffer
            await self._update_momentum_async(normalized)
            
            # Calculate significance score
            significance_future = loop.run_in_executor(
                self._thread_pool,
                self._calculate_significance,
                normalized,
                surprise_score
            )
            significance = await significance_future
            logger.info(f"Calculated significance score: {significance}")
            
            return normalized, significance
            
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
    
    def _compute_surprise(self, embedding: torch.Tensor) -> float:
        """Calculate surprise score based on momentum buffer (CPU-bound)"""
        with torch.no_grad():
            if self.momentum_buffer is None:
                return 0.0
                
            similarity = torch.matmul(embedding, self.momentum_buffer.T)
            return 1.0 - torch.mean(similarity).item()
    
    def _apply_shock_absorber(self, embedding: torch.Tensor) -> torch.Tensor:
        """Smooth out high-surprise embeddings (CPU-bound)"""
        with torch.no_grad():
            if self.momentum_buffer is None:
                return embedding
                
            alpha = 1.0 - self.config['momentum']
            absorbed = alpha * embedding + (1 - alpha) * self.momentum_buffer[-1:]
            
            # Re-normalize
            norm = torch.norm(absorbed, p=2, dim=-1, keepdim=True)
            return absorbed / (norm + 1e-8)
    
    async def _update_momentum_async(self, embedding: torch.Tensor):
        """Update momentum buffer with new embedding (thread-safe)"""
        async with self._lock:
            if self.momentum_buffer is None:
                self.momentum_buffer = embedding
            else:
                combined = torch.cat([self.momentum_buffer, embedding])
                self.momentum_buffer = combined[-self.config['chunk_size']:]
    
    def _calculate_significance(self, embedding: torch.Tensor, surprise: float) -> float:
        """Calculate significance score for memory storage (CPU-bound)"""
        with torch.no_grad():
            # Thread-safely access momentum buffer
            momentum_copy = None
            if self.momentum_buffer is not None:
                momentum_copy = self.momentum_buffer.clone()
            
            magnitude = torch.norm(embedding).item()
            
            if momentum_copy is not None:
                # Use .mT for matrix transpose which supports batched matrices correctly
                # or use permute() as an alternative for any tensor dimensions
                if momentum_copy.dim() == 2:
                    similarity = torch.matmul(embedding, momentum_copy.T)
                else:
                    # Using the recommended approach for >2 dimensions
                    similarity = torch.matmul(embedding, momentum_copy.permute(*torch.arange(momentum_copy.ndim - 1, -1, -1)))
                diversity = 1.0 - torch.max(similarity).item()
            else:
                diversity = 1.0
                
            # Enhanced significance calculation with importance weights
            # Higher weight assigned to surprise for better context retention
            significance = (
                0.5 * surprise +  # Increased weight for surprise
                0.2 * magnitude +
                0.3 * diversity
            )
            
            # Special handling for potential personal information
            # Higher significance for content that might contain personal details
            if surprise > 0.7 and diversity > 0.6:
                significance = max(significance, 0.75)  # Ensure high significance
            
            return significance
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current state statistics"""
        stats = {
            'has_momentum': self.momentum_buffer is not None,
            'momentum_size': len(self.momentum_buffer) if self.momentum_buffer is not None else 0,
            'device': self.config['device'],
            'processed_count': self._stats['processed_count'],
            'error_count': self._stats['error_count'],
            'retry_count': self._stats['retry_count'],
            'avg_processing_time': self._stats['avg_processing_time'],
            'last_error': self._stats['last_error'],
        }
        
        return stats
    
    async def close(self):
        """Clean up resources used by this manager"""
        self._thread_pool.shutdown(wait=True)
