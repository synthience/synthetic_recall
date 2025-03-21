"""
LUCID RECALL PROJECT

HPC-QR Flow Manager: High-Performance Computing with QuickRecal approach for memory embeddings
Offers geometry-aware, causal/contextual, and self-organization-based memory scoring
"""

import asyncio
import time
import logging
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import gc
import os

# Import UnifiedQuickRecallCalculator for the calculation
from server.qr_calculator import UnifiedQuickRecallCalculator, QuickRecallMode, QuickRecallFactor

logger = logging.getLogger(__name__)

# Avoid import-time TensorClient usage to prevent circular imports
_tensor_client_instance = None

def get_tensor_client(**kwargs):
    """
    Factory function to get a TensorClient instance, lazily loaded
    to avoid circular imports.
    """
    global _tensor_client_instance
    if _tensor_client_instance is None:
        # Lazy import to avoid circular reference
        from server.tensor_server import TensorClient
        _tensor_client_instance = TensorClient(**kwargs)
    return _tensor_client_instance

async def get_tensor_client_async():
    """
    Get a TensorClient instance, initializing lazily.
    This async version allows for awaiting the connection if needed.
    
    Returns:
        TensorClient instance
    """
    from server.tensor_client import TensorClient
    
    # Create client if needed
    client = TensorClient()
    
    # Ensure connection
    if not client.is_connected():
        await client.connect()
        
    return client

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
            # Shock absorption parameters
            'max_momentum_size': 10000,  # Maximum size of momentum buffer
            'shock_absorption_enabled': True,  # Enable/disable shock absorption
            'dynamic_scaling_factor': 0.5,  # Scaling factor for adaptive shock absorption
            'update_threshold': 0.05,  # Threshold for determining when to update embeddings
            'drift_threshold': 0.3,  # Threshold for embedding drift triggering self-healing
            **(config or {})
        }
        
        self.momentum_buffer = None
        self.current_batch = []
        self.batch_timestamps = []
        self.surprise_scores = []  # Store surprise scores for batch processing
        
        # Thread pool for CPU-intensive operations
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config['max_threads'])
        
        # Processing statistics
        self._stats = {
            'processed_count': 0,
            'error_count': 0,
            'retry_count': 0,
            'avg_processing_time': 0.0,
            'last_error': None,
            'total_processing_time': 0.0,
            'momentum_buffer_size': 0,
            'last_batch_size': 0,
            'last_batch_time': 0.0,
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Initialize UnifiedQuickRecallCalculator with proper configuration
        self._init_calculator()
        
        logger.info(f"Initialized HPCQRFlowManager with config: {self.config}")
        logger.info(f"Using UnifiedQuickRecallCalculator with mode: {self.qr_calculator.config['mode']}")
    
    def _init_calculator(self):
        """
        Initialize the UnifiedQuickRecallCalculator instance with appropriate configuration
        """
        from server.qr_calculator import UnifiedQuickRecallCalculator, QuickRecallMode, QuickRecallFactor
        import os
        import yaml
        
        # First try to load the QuickRecal config from file
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'config', 'quickrecal_config.yaml')
        
        # Set up calculator config with defaults
        calculator_config = {
            'embedding_dim': self.config.get('embedding_dim', 768),
            'chunk_size': self.config.get('chunk_size', 768),
            'mode': self.config.get('calculator_mode', 'hpc_qr'),
            'device': self.config.get('device', 'cpu'),
            'alpha': self.config.get('alpha', 0.4),
            'beta': self.config.get('beta', 0.3),
            'gamma': self.config.get('gamma', 0.2),
            'delta': self.config.get('delta', 0.1),
            'som_grid_size': (10, 10),  # Default SOM grid size
            'geometry': 'mixed',  # Default geometry for hyperbolic calculations
            'curvature': -1.0     # Default curvature for hyperbolic space
        }
        
        # Try to load config from file
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    logger.info(f"Loaded QuickRecal config from {config_path}")
                    
                    # Map config keys to calculator config
                    if yaml_config:
                        # Core settings
                        calculator_config['mode'] = yaml_config.get('mode', calculator_config['mode'])
                        
                        # Score normalization settings
                        calculator_config['use_zscore_normalization'] = yaml_config.get('use_zscore_normalization', False)
                        calculator_config['z_score_window_size'] = yaml_config.get('z_score_window_size', 100)
                        calculator_config['history_window'] = yaml_config.get('history_window', 100)
                        
                        # HPC-QR parameters
                        calculator_config['alpha'] = yaml_config.get('alpha', calculator_config['alpha'])
                        calculator_config['beta'] = yaml_config.get('beta', calculator_config['beta'])
                        calculator_config['gamma'] = yaml_config.get('gamma', calculator_config['gamma'])
                        calculator_config['delta'] = yaml_config.get('delta', calculator_config['delta'])
                        
                        # Threshold settings
                        calculator_config['novelty_threshold'] = yaml_config.get('novelty_threshold', 0.7)
                        calculator_config['time_decay_rate'] = yaml_config.get('time_decay_rate', 0.1)
                        calculator_config['min_time_decay'] = yaml_config.get('min_time_decay', 0.1)
                        
                        # Debug settings
                        calculator_config['debug'] = yaml_config.get('debug', False)
                        
                        # Update local class config with fusion weights from yaml
                        if 'search_fusion' in yaml_config:
                            fusion_config = yaml_config['search_fusion']
                            self.search_fusion_weights = {
                                'similarity_weight': fusion_config.get('similarity_weight', 0.5),
                                'quickrecal_weight': fusion_config.get('quickrecal_weight', 0.5),
                                'use_logarithmic_fusion': fusion_config.get('use_logarithmic_fusion', True)
                            }
                            
                        # Set factor weights from yaml
                        if 'factor_weights' in yaml_config:
                            factor_weights_map = {
                                'embedding_similarity': QuickRecallFactor.RELEVANCE,
                                'emotion_weight': QuickRecallFactor.EMOTION,
                                'surprise_factor': QuickRecallFactor.SURPRISE,
                                'novelty': QuickRecallFactor.DIVERSITY,
                                'time_decay': QuickRecallFactor.RECENCY
                            }
                            
                            # Create QuickRecall factor weights dict from yaml config
                            yaml_weights = yaml_config['factor_weights']
                            factor_weights = {}
                            
                            for yaml_key, factor in factor_weights_map.items():
                                if yaml_key in yaml_weights:
                                    factor_weights[factor] = yaml_weights[yaml_key]
                            
                            # Set all other factors to 0
                            for factor in QuickRecallFactor:
                                if factor not in factor_weights:
                                    factor_weights[factor] = 0.0
                            
                            # Cache the factor weights for later use
                            self.factor_weights = factor_weights
        except Exception as e:
            logger.warning(f"Error loading QuickRecal config: {e}")
            logger.warning("Using default QuickRecal configuration")
        
        # Convert string mode to enum if needed
        if isinstance(calculator_config['mode'], str):
            mode_str = calculator_config['mode'].upper()
            calculator_config['mode'] = getattr(QuickRecallMode, mode_str, QuickRecallMode.HPC_QR)
        
        # Initialize calculator
        self.qr_calculator = UnifiedQuickRecallCalculator(calculator_config)
        
        # If we loaded factor weights from config file, apply them
        if hasattr(self, 'factor_weights'):
            self.qr_calculator.set_factor_weights(self.factor_weights)
            logger.info(f"Applied factor weights from config file: {self.factor_weights}")
        # Otherwise use default factor weights for HPC_QR mode
        elif calculator_config['mode'] == QuickRecallMode.HPC_QR:
            # Core HPC factors get most of the weight
            core_weights = {
                QuickRecallFactor.R_GEOMETRY: 0.25,
                QuickRecallFactor.CAUSAL_NOVELTY: 0.2,
                QuickRecallFactor.SELF_ORG: 0.125,
                QuickRecallFactor.OVERLAP: 0.1
            }
            
            # Additional factors
            additional_weights = {
                QuickRecallFactor.SURPRISE: 0.0625,
                QuickRecallFactor.DIVERSITY: 0.03125,
                QuickRecallFactor.EMOTION: 0.0625,
                QuickRecallFactor.RECENCY: 0.0625,
                QuickRecallFactor.IMPORTANCE: 0.0625,
                QuickRecallFactor.PERSONAL: 0.0625,
                QuickRecallFactor.COHERENCE: 0.03125,
                QuickRecallFactor.INFORMATION: 0.03125,
                QuickRecallFactor.RELEVANCE: 0.03125,
                QuickRecallFactor.USER_ATTENTION: 0.0  # Disabled by default
            }
            
            # Combine weights
            factor_weights = {**core_weights, **additional_weights}
            
            # Update calculator factor weights
            self.qr_calculator.set_factor_weights(factor_weights)
        
        # Log initialization
        logger.info(f"Using UnifiedQuickRecallCalculator with mode: {self.qr_calculator.config['mode']}")
    
    async def process_embedding(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Process a single embedding through the HPC-QR pipeline asynchronously
        
        Args:
            embedding: Raw embedding tensor to process
            
        Returns:
            Tuple of (processed_embedding, qr_score)
        """
        # Validate and preprocess the embedding
        normalized = self._preprocess_embedding(embedding)
        
        # Check for malformed embedding (dimension = 1 case)
        expected_dim = self.config.get('embedding_dim', 768)
        actual_dim = normalized.shape[0] if normalized is not None else 0
        
        if normalized is None or actual_dim == 1 or (actual_dim < 10 and expected_dim >= 384):
            # Log the issue
            if normalized is None:
                logger.warning("Received None embedding - replacing with penalty vector")
            else:
                logger.warning(f"Malformed embedding detected (dim={actual_dim}) - replacing with penalty vector")
            
            # Create a penalty vector
            if torch.is_tensor(embedding):
                device = embedding.device
            else:
                device = torch.device('cpu')
                
            # Generate a consistent penalty vector with low QR score potential
            penalty_vector = torch.normal(0, 0.01, size=(expected_dim,), device=device)
            normalized = penalty_vector / torch.norm(penalty_vector).clamp(min=1e-9)
            
            # Use a fixed low score for malformed embeddings
            score = 0.1
            context = {"timestamp": time.time(), "qr_cap": 0.1, "is_penalty": True}
        else:
            # Calculate score with validated embedding
            context = {"timestamp": time.time()}
            score = await self.qr_calculator.calculate(normalized, context=context)
        
        # Apply shock absorption if enabled and not a penalty vector
        if self.config['shock_absorption_enabled'] and not context.get('is_penalty', False):
            adjusted = await self._apply_shock_absorber(normalized, score)
        else:
            adjusted = normalized
        
        # Only update momentum with valid embeddings above threshold
        if not context.get('is_penalty', False) and score > 0.3:
            await self._update_momentum_async(normalized)
        
        return adjusted, score
    
    async def process_embedding_batch(self, embeddings: List[torch.Tensor]) -> List[Tuple[torch.Tensor, float]]:
        """Process a batch of embeddings through the HPCQR pipeline"""
        if not embeddings:
            return []
        
        start_time = time.time()
        batch_size = len(embeddings)
        self._stats['last_batch_size'] = batch_size
        
        try:
            # 1. Preprocess all embeddings
            loop = asyncio.get_event_loop()
            normalized_tensors = []
            for emb in embeddings:
                # Run preprocessing in thread pool
                preproc_future = loop.run_in_executor(
                    self._thread_pool,
                    self._preprocess_embedding,
                    emb
                )
                normalized = await preproc_future
                normalized_tensors.append(normalized)
            
            # 2. Calculate scores using calculator
            # Create context with batch information
            context = {
                "timestamp": time.time(),
                "batch_size": batch_size,
                "device": str(embeddings[0].device),
                "processing_start": start_time,
            }
            
            # Process sequentially for now (can be optimized later if needed)
            results = []
            for i, norm_tensor in enumerate(normalized_tensors):
                try:
                    # Pass context with position information
                    batch_context = context.copy()
                    batch_context["batch_position"] = i
                    
                    # Calculate score using calculator
                    score = await self.qr_calculator.calculate(norm_tensor, context=batch_context)
                    
                    # Apply shock absorption if enabled
                    if self.config['shock_absorption_enabled']:
                        adjusted = await self._apply_shock_absorber(norm_tensor, score)
                    else:
                        adjusted = norm_tensor
                        
                    results.append((adjusted, score))
                    
                    # Update momentum buffer (only for selected embeddings)
                    if score > 0.3:  # Only update for embeddings with sufficient score
                        await self._update_momentum_async(norm_tensor)
                        
                except Exception as e:
                    logger.error(f"Error processing embedding {i} in batch: {str(e)}")
                    # On error, return original with default score
                    results.append((embeddings[i], 0.5))
                    self._stats['error_count'] += 1
            
            # Update statistics
            self._stats['processed_count'] += batch_size
            self._stats['last_batch_time'] = time.time() - start_time
            self._stats['total_processing_time'] += self._stats['last_batch_time']
            self._stats['avg_processing_time'] = self._stats['total_processing_time'] / self._stats['processed_count']
            
            return results
            
        except Exception as e:
            logger.error(f"Error in process_embedding_batch: {str(e)}")
            self._stats['error_count'] += 1
            self._stats['last_error'] = str(e)
            
            # Fallback: return original embeddings with default scores
            return [(emb, 0.5) for emb in embeddings]
    
    async def _process_embedding_internal(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Internal implementation of embedding processing with QuickRecal HPC-QR approach"""
        # Get event loop for thread pool execution
        loop = asyncio.get_event_loop()
        
        # Run preprocessing in thread pool
        preprocess_future = loop.run_in_executor(
            self._thread_pool,
            self._preprocess_embedding,
            embedding
        )
        normalized = await preprocess_future
        
        # Check for malformed embedding (dimension = 1 case)
        expected_dim = self.config.get('embedding_dim', 768)
        actual_dim = normalized.shape[0] if normalized is not None else 0
        
        if normalized is None or actual_dim == 1 or (actual_dim < 10 and expected_dim >= 384):
            # Log the issue
            if normalized is None:
                logger.warning("Received None embedding in _process_embedding_internal - using penalty vector")
            else:
                logger.warning(f"Malformed embedding detected (dim={actual_dim}) in _process_embedding_internal - using penalty vector")
            
            # Create a penalty vector
            if torch.is_tensor(embedding):
                device = embedding.device
            else:
                device = torch.device('cpu')
                
            # Generate a consistent penalty vector with low QR score potential
            penalty_vector = torch.normal(0, 0.01, size=(expected_dim,), device=device)
            normalized = penalty_vector / torch.norm(penalty_vector).clamp(min=1e-9)
            
            # Use a fixed low score for malformed embeddings
            score = 0.1
            return normalized, score
        
        # Calculate HPC-QR score using UnifiedQuickRecallCalculator
        context = {"timestamp": time.time(), "device": str(embedding.device)}
        score = await self.qr_calculator.calculate(normalized, context=context)
        
        # Apply shock absorber if enabled
        if self.config['shock_absorption_enabled']:
            adjusted = await self._apply_shock_absorber(normalized, score)
        else:
            adjusted = normalized
        
        # Only update momentum buffer if score is high enough 
        # (to avoid updating with malformed or low-quality embeddings)
        if score > 0.3:
            await self._update_momentum_async(normalized)
        
        return adjusted, score
    
    async def _apply_shock_absorber(self, embedding: torch.Tensor, surprise_score: float) -> torch.Tensor:
        """
        Apply shock absorption to an embedding based on surprise score.
        
        Shock absorption dampens the embedding in the direction of the momentum buffer
        based on the surprise score, preventing abrupt changes while maintaining 
        hypersphere integrity (unit norm).
        
        Args:
            embedding: The normalized embedding tensor
            surprise_score: Value indicating how surprising/novel the embedding is
            
        Returns:
            Shock-absorbed normalized embedding
        """
        if not self.config['shock_absorption_enabled'] or self.momentum_buffer is None or len(self.momentum_buffer) == 0:
            return embedding
            
        with torch.no_grad():
            try:
                # Get dynamic scaling factor for adaptive shock absorption
                dynamic_scaling_factor = self.config['dynamic_scaling_factor']
                
                # Calculate damping factor based on surprise score
                # Higher surprise = less damping (to preserve novel information)
                # Lower surprise = more damping (to prevent oscillations)
                damping_factor = torch.exp(torch.tensor(-surprise_score / dynamic_scaling_factor, device=embedding.device))
                
                # Calculate the momentum direction (mean of buffer)
                momentum_center = torch.mean(self.momentum_buffer, dim=0)
                
                # Ensure dimension compatibility
                if momentum_center.shape[0] != embedding.shape[0]:
                    logger.warning(f"Dimension mismatch: momentum_center dim {momentum_center.shape[0]} "
                                 f"!= embedding dim {embedding.shape[0]}. Adjusting momentum center.")
                    if momentum_center.shape[0] < embedding.shape[0]:
                        # Pad momentum center if too small
                        padded_center = torch.zeros(embedding.shape[0], device=embedding.device)
                        padded_center[:momentum_center.shape[0]] = momentum_center
                        momentum_center = padded_center
                    else:
                        # Truncate momentum center if too large
                        momentum_center = momentum_center[:embedding.shape[0]]
                
                # Apply directional damping
                # - Move embedding toward momentum center for low surprise items
                # - Preserve embedding direction for high surprise items
                adjusted_embedding = embedding * (1 - damping_factor) + momentum_center * damping_factor
                
                # Re-normalize to maintain hypersphere integrity
                adjusted_embedding = adjusted_embedding / torch.norm(adjusted_embedding, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                
                # Check if the update is significant enough
                update_distance = torch.norm(adjusted_embedding - embedding)
                if update_distance < self.config['update_threshold']:
                    return embedding  # Return original if change is minimal
                
                # Log damping factor statistics if debug logging is enabled
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Applied damping factor: {damping_factor:.4f} (surprise score: {surprise_score:.4f})")
                    
                # Maintain unit norm
                return adjusted_embedding
                
            except Exception as e:
                logger.warning(f"Shock absorption failed: {e}. Falling back to original embedding.")
                return embedding  # Fall back to original embedding on error
    
    async def _apply_batch_shock_absorber(self, embeddings: torch.Tensor, surprise_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply shock absorption to a batch of embeddings based on their surprise scores.
        Optimized for parallel processing of multiple embeddings at once.
        
        Args:
            embeddings: Batch of normalized embeddings [batch_size, embedding_dim]
            surprise_scores: Batch of surprise scores [batch_size]
            
        Returns:
            Batch of shock-absorbed normalized embeddings
        """
        if not self.config['shock_absorption_enabled'] or self.momentum_buffer is None or len(self.momentum_buffer) == 0:
            return embeddings
            
        with torch.no_grad():
            try:
                device = embeddings.device
                batch_size = embeddings.shape[0]
                dtype = embeddings.dtype
                
                # Use mixed precision for faster computation when available
                if device.type == 'cuda' and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        dynamic_scaling_factor = self.config['dynamic_scaling_factor']
                        
                        # Calculate batch damping factors from surprise scores in one operation
                        # Higher surprise (score) = less damping to preserve novel information
                        # Lower surprise = more damping to prevent oscillations
                        damping_factors = torch.exp(-surprise_scores / dynamic_scaling_factor).to(device)
                        
                        # Reshape for broadcasting [batch_size, 1]
                        damping_factors = damping_factors.view(-1, 1)
                        
                        # Calculate momentum center once for all embeddings
                        momentum_center = torch.mean(self.momentum_buffer, dim=0)
                        
                        # Ensure dimension compatibility
                        if momentum_center.shape[0] != embeddings.shape[1]:
                            logger.warning(f"Dimension mismatch in shock absorber: momentum_center dim {momentum_center.shape[0]} "
                                         f"!= embeddings dim {embeddings.shape[1]}. Adjusting momentum center.")
                            if momentum_center.shape[0] < embeddings.shape[1]:
                                # Pad momentum center if too small
                                padded_center = torch.zeros(embeddings.shape[1], device=device)
                                padded_center[:momentum_center.shape[0]] = momentum_center
                                momentum_center = padded_center
                            else:
                                # Truncate momentum center if too large
                                momentum_center = momentum_center[:embeddings.shape[1]]
                        
                        # Expand momentum center to match batch dimensions for vectorized operations
                        # This avoids loop-based processing
                        momentum_batch = momentum_center.expand(batch_size, -1)
                        
                        # Apply directional damping to all embeddings at once
                        # This is a fully vectorized operation that runs efficiently on GPU
                        adjusted_embeddings = embeddings * (1 - damping_factors) + momentum_batch * damping_factors
                        
                        # Re-normalize to maintain hypersphere integrity
                        # Using keepdim=True to maintain tensor dimensions for broadcasting
                        norms = torch.norm(adjusted_embeddings, p=2, dim=1, keepdim=True)
                        adjusted_embeddings = adjusted_embeddings / norms.clamp(min=1e-8)
                        
                        # Vectorized check for which updates are significant enough
                        # Compute distances between original and adjusted embeddings
                        update_distances = torch.norm(adjusted_embeddings - embeddings, dim=1)
                        
                        # Create a binary mask for significant updates (True where distance >= threshold)
                        # This operation is also fully vectorized
                        mask = update_distances >= self.config['update_threshold']
                        
                        # Use the mask to selectively update embeddings
                        # Only embeddings with significant changes are updated
                        # This is memory-efficient as it modifies the tensor in-place
                        result = embeddings.clone()
                        if mask.any():
                            result[mask] = adjusted_embeddings[mask]
                        
                        # Log statistics about the batch processing
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Shock absorption - Damping factors: Mean={damping_factors.mean().item():.4f}, "
                                       f"Min={damping_factors.min().item():.4f}, Max={damping_factors.max().item():.4f}, "
                                       f"Applied to {mask.sum().item()}/{batch_size} embeddings")
                else:
                    # CPU fallback implementation
                    dynamic_scaling_factor = self.config['dynamic_scaling_factor']
                    update_threshold = self.config['update_threshold']
                    
                    # Calculate batch damping factors from surprise scores [batch_size]
                    damping_factors = torch.exp(-surprise_scores / dynamic_scaling_factor)
                    
                    # Reshape for broadcasting [batch_size, 1]
                    damping_factors = damping_factors.view(-1, 1)
                    
                    # Calculate momentum center once for efficiency
                    momentum_center = torch.mean(self.momentum_buffer, dim=0)
                    
                    # Ensure dimension compatibility
                    if momentum_center.shape[0] != embeddings.shape[1]:
                        logger.warning(f"Dimension mismatch in shock absorber: momentum_center dim {momentum_center.shape[0]} "
                                     f"!= embeddings dim {embeddings.shape[1]}. Adjusting momentum center.")
                        if momentum_center.shape[0] < embeddings.shape[1]:
                            # Pad momentum center if too small
                            padded_center = torch.zeros(embeddings.shape[1], device=embeddings.device)
                            padded_center[:momentum_center.shape[0]] = momentum_center
                            momentum_center = padded_center
                        else:
                            # Truncate momentum center if too large
                            momentum_center = momentum_center[:embeddings.shape[1]]
                    
                    # Expand momentum center to match batch dimensions [batch_size, embedding_dim]
                    momentum_batch = momentum_center.expand(batch_size, -1)
                    
                    # Apply directional damping to all embeddings at once
                    adjusted_embeddings = embeddings * (1 - damping_factors) + momentum_batch * damping_factors
                    
                    # Re-normalize to maintain hypersphere integrity
                    norms = torch.norm(adjusted_embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-8)
                    adjusted_embeddings = adjusted_embeddings / norms
                    
                    # Check which updates are significant enough
                    update_distances = torch.norm(adjusted_embeddings - embeddings, dim=1)
                    mask = update_distances >= update_threshold
                    
                    # Only update embeddings that changed significantly
                    result = embeddings.clone()
                    result[mask] = adjusted_embeddings[mask]
                
                return result
                
            except Exception as e:
                logger.warning(f"Batch shock absorption failed: {e}. Falling back to original embeddings.")
                return embeddings  # Fall back to original embeddings on error
    
    async def _update_momentum_async(self, embedding: torch.Tensor):
        """Update the momentum buffer with the new embedding asynchronously"""
        async with self._lock:
            try:
                device = embedding.device
                # Initialize momentum buffer if not done yet
                if self.momentum_buffer is None:
                    logger.info(f"Initializing momentum buffer on device {device}")
                    self.momentum_buffer = embedding.unsqueeze(0)
                else:
                    # Ensure consistent device usage
                    if self.momentum_buffer.device != device:
                        logger.debug(f"Moving momentum buffer from {self.momentum_buffer.device} to {device}")
                        self.momentum_buffer = self.momentum_buffer.to(device)
                    
                    # Append to momentum_buffer, ensuring dimension compatibility
                    if self.momentum_buffer.shape[1] != embedding.shape[0]:
                        logger.warning(f"Dimension mismatch: momentum_buffer dim {self.momentum_buffer.shape[1]} " 
                                      f"!= embedding dim {embedding.shape[0]}. Adjusting embedding.")
                        if self.momentum_buffer.shape[1] < embedding.shape[0]:
                            # Truncate embedding if buffer dim is smaller
                            resized_embedding = embedding[:self.momentum_buffer.shape[1]]
                        else:
                            # Pad embedding if buffer dim is larger
                            resized_embedding = torch.zeros(self.momentum_buffer.shape[1], device=device)
                            resized_embedding[:embedding.shape[0]] = embedding
                        
                        embedding = resized_embedding
                    
                    # Add embedding to momentum buffer
                    self.momentum_buffer = torch.cat((self.momentum_buffer, embedding.unsqueeze(0)), dim=0)
                
                # Limit size of momentum buffer
                buffer_size = len(self.momentum_buffer)
                if buffer_size > self.config['max_momentum_size']:
                    # Keep most recent entries
                    logger.debug(f"Truncating momentum buffer from {buffer_size} to {self.config['max_momentum_size']}")
                    self.momentum_buffer = self.momentum_buffer[-self.config['max_momentum_size']:]
                
                # Update statistics
                self._stats['momentum_buffer_size'] = len(self.momentum_buffer)
                
                # Synchronize momentum buffer with UnifiedQuickRecallCalculator
                try:
                    self.qr_calculator.set_external_momentum(self.momentum_buffer)
                except Exception as e:
                    logger.error(f"Error syncing momentum buffer with calculator: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error updating momentum buffer: {str(e)}")
    
    def _preprocess_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Preprocess embedding - GPU-optimized version with input validation"""
        # Validate input
        if embedding is None:
            logger.warning("Received None embedding in _preprocess_embedding")
            return None
            
        # Handle dimension of 1 or other invalid shapes
        if not torch.is_tensor(embedding):
            try:
                logger.warning(f"Non-tensor embedding received in _preprocess_embedding: {type(embedding)}")
                # Try conversion but return None if fails
                embedding = torch.tensor(embedding, dtype=torch.float32) 
            except Exception as e:
                logger.error(f"Failed to convert embedding to tensor: {str(e)}")
                return None
                
        # Handle scalar or empty tensor
        if embedding.numel() <= 1:
            logger.warning(f"Invalid embedding with {embedding.numel()} elements")
            return None
            
        with torch.no_grad():
            # Move to correct device immediately
            embedding = embedding.to(self.config['device'], non_blocking=True)
            
            # Use mixed precision where available
            if self.config['device'] == 'cuda' and torch.cuda.is_available():
                # Enable auto mixed precision for faster computation
                with torch.cuda.amp.autocast():
                    # Handle shape and size adjustment
                    if len(embedding.shape) > 1:
                        # Handle 2D tensor (batch x features)
                        if embedding.shape[1] < self.config['chunk_size']:
                            # Pad if smaller than chunk_size - on GPU
                            padded = torch.zeros((embedding.shape[0], self.config['chunk_size']),
                                            device=embedding.device, dtype=embedding.dtype)
                            padded[:, :embedding.shape[1]] = embedding
                            embedding = padded
                        else:
                            # Truncate if larger than chunk_size - on GPU
                            embedding = embedding[:, :self.config['chunk_size']]
                        
                        # Handle batch dimension correctly
                        if embedding.shape[0] == 1:
                            embedding = embedding.reshape(-1)
                    else:
                        # Handle 1D tensor
                        if embedding.shape[0] < self.config['chunk_size']:
                            # Pad if smaller than chunk_size - on GPU
                            padded = torch.zeros(self.config['chunk_size'], 
                                            device=embedding.device, 
                                            dtype=embedding.dtype)
                            padded[:embedding.shape[0]] = embedding
                            embedding = padded
                        else:
                            # Truncate if larger than chunk_size - on GPU
                            embedding = embedding[:self.config['chunk_size']]
            else:
                # Standard precision for CPU
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
                    if embedding.shape[0] == 1:
                        embedding = embedding.reshape(-1)
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
            
            # Check for NaN or Inf values
            if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                logger.warning("Embedding contains NaN or Inf values - replacing with penalty vector")
                # Return None to trigger penalty vector creation in process_embedding
                return None
            
            # Use torch.nn.functional for optimized normalization on GPU
            try:
                normalized = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                return normalized
            except Exception as e:
                logger.error(f"Normalization failed: {str(e)}")
                return None
    
    def _normalize_embedding(self, embedding: Union[torch.Tensor, np.ndarray, List[float]]) -> torch.Tensor:
        """
        Normalize embedding to unit length and ensure proper dimensions.
        Handles conversion between different embedding types and dimension mismatches.
        
        Args:
            embedding: The embedding to normalize (torch.Tensor, numpy array, or list)
            
        Returns:
            Normalized embedding with proper dimensions as a torch.Tensor
        """
        # Get expected dimension
        expected_dim = self.config.get('embedding_dim', 768)
        
        # Convert list to tensor if needed
        if isinstance(embedding, list):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        # Convert numpy array to tensor if needed
        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        # Ensure we're working with a torch.Tensor
        if not isinstance(embedding, torch.Tensor):
            raise ValueError(f"Unsupported embedding type: {type(embedding)}")
        
        # Move to the correct device
        device = self.config.get('device', 'cpu')
        embedding = embedding.to(device)
        
        # Flatten if needed
        if len(embedding.shape) > 1:
            embedding = embedding.squeeze()
            # If it's still multi-dimensional, take the first vector
            if len(embedding.shape) > 1:
                embedding = embedding[0]
        
        # Current dimension
        actual_dim = embedding.shape[0]
        
        # Handle dimension mismatch
        if actual_dim != expected_dim:
            if actual_dim < expected_dim:
                # Pad with zeros
                padding = torch.zeros(expected_dim - actual_dim, 
                                    device=embedding.device, 
                                    dtype=embedding.dtype)
                embedding = torch.cat([embedding, padding])
                logger.debug(f"Padded embedding from {actual_dim} to {expected_dim}")
            else:
                # Truncate
                embedding = embedding[:expected_dim]
                logger.debug(f"Truncated embedding from {actual_dim} to {expected_dim}")
        
        # Normalize to unit length
        norm = torch.norm(embedding)
        if norm > 1e-9:
            embedding = embedding / norm
        else:
            # Generate a random unit vector if norm is too close to zero
            logger.warning("Zero-norm embedding detected, using random unit vector")
            random_vector = torch.randn(expected_dim, device=embedding.device)
            embedding = random_vector / torch.norm(random_vector)
        
        return embedding
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get current state statistics about the HPCQRFlowManager.
        
        Returns:
            Dictionary of statistics about the current state
        """
        # Get calculator stats if available
        try:
            calculator_stats = self.qr_calculator.get_stats()
        except Exception as e:
            logger.error(f"Failed to get calculator stats: {str(e)}")
            calculator_stats = {"error": str(e)}
        
        # Merge internal stats with calculator stats (avoid name collisions)
        stats = {
            **self._stats,
            "calculator": calculator_stats,
            "momentum_buffer_size": self._stats["momentum_buffer_size"],
            "configuration": {
                "alpha": self.config["alpha"],
                "beta": self.config["beta"],
                "gamma": self.config["gamma"],
                "delta": self.config["delta"],
                "device": self.config["device"],
                "max_momentum_size": self.config["max_momentum_size"],
                "shock_absorption_enabled": self.config["shock_absorption_enabled"],
                "calculator_mode": self.qr_calculator.config["mode"].value,
            }
        }
        
        return stats
    
    async def get_embedding(self, text: str) -> torch.Tensor:
        """
        Generate an embedding vector from text content by calling the TensorServer.
        Ensures the embedding is normalized and has the correct dimensions.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            Normalized embedding tensor with proper dimensions
        """
        tensor_server = await self._get_tensor_server_instance()
        
        if tensor_server is None:
            logger.error("Failed to get tensor server instance")
            return None
        
        # Get embedding from tensor server
        embedding = await tensor_server.get_embedding(text)
        
        # Ensure we have a tensor with the right properties
        if embedding is None:
            raise ValueError("TensorServer returned None embedding")
                
        # Convert to tensor if it's not already
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
                
        # Move to the right device
        device = self.config.get('device', 'cpu')
        embedding = embedding.to(device)
        
        # Normalize and handle dimension mismatches
        embedding = self._normalize_embedding(embedding)
        
        return embedding
    
    async def _get_tensor_server_instance(self):
        """
        Get an instance of the TensorServer client.
        
        Returns:
            TensorClient instance or None if unavailable
        """
        try:
            # Use direct import to avoid circular dependency issues
            from server.tensor_server import TensorServer
            return TensorServer()
        except Exception as e:
            logger.error(f"Failed to get tensor client: {str(e)}")
            return None
    
    async def close(self):
        """Clean up resources used by this manager"""
        logger.info("Shutting down HPCQRFlowManager...")
        self._thread_pool.shutdown(wait=True)
        logger.info("HPCQRFlowManager shutdown complete")
    
    def update_calculator_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the configuration of the UnifiedQuickRecallCalculator
        
        Args:
            new_config: New configuration values
        """
        from server.qr_calculator import QuickRecallMode, QuickRecallFactor
        
        # Update HPCQRFlowManager config first
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
        
        # Update calculator config
        calculator_updates = {}
        
        # Map core parameters
        for param in ['alpha', 'beta', 'gamma', 'delta', 'embedding_dim', 'device']:
            if param in new_config:
                calculator_updates[param] = new_config[param]
        
        # Handle mode change
        if 'calculator_mode' in new_config:
            mode_value = new_config['calculator_mode']
            # Convert string to enum if needed
            if isinstance(mode_value, str):
                mode_str = mode_value.upper()
                mode_value = getattr(QuickRecallMode, mode_str, QuickRecallMode.HPC_QR)
            calculator_updates['mode'] = mode_value
        
        # Update factor weights if specified
        if 'factor_weights' in new_config and isinstance(new_config['factor_weights'], dict):
            factor_weights = {}
            
            # Convert string keys to enum if needed
            for factor_key, weight in new_config['factor_weights'].items():
                if isinstance(factor_key, str):
                    # Try to match by enum value first (e.g. 'r_geometry')
                    matched = False
                    for factor in QuickRecallFactor:
                        if factor.value == factor_key:
                            factor_weights[factor] = weight
                            matched = True
                            break
                    
                    # If not matched, try by enum name (e.g. 'R_GEOMETRY')
                    if not matched:
                        factor_enum = getattr(QuickRecallFactor, factor_key.upper(), None)
                        if factor_enum is not None:
                            factor_weights[factor_enum] = weight
                else:
                    # Already an enum
                    factor_weights[factor_key] = weight
            
            # Apply the weights to the calculator
            if factor_weights:
                self.qr_calculator.set_factor_weights(factor_weights)
        
        # Apply configuration updates to calculator
        if calculator_updates:
            for key, value in calculator_updates.items():
                if key in self.qr_calculator.config:
                    self.qr_calculator.config[key] = value
        
        logger.info(f"Updated calculator configuration: {calculator_updates}")
    
    async def benchmark_modes(self, test_embeddings, batch_sizes=None, modes_to_test=None):
        """
        Compare calculation speed across different calculator modes and batch sizes.
        
        Args:
            test_embeddings: List of embeddings to use for benchmarking
            batch_sizes: List of batch sizes to test (default: [1, 8, 32, 64])
            modes_to_test: List of QuickRecallMode values to test
            
        Returns:
            Dictionary with benchmark results per mode and batch size including response time
            distribution and throughput metrics
        """
        if modes_to_test is None:
            modes_to_test = [QuickRecallMode.STANDARD, QuickRecallMode.EFFICIENT, 
                            QuickRecallMode.PRECISE, QuickRecallMode.HPC_QR]
        
        if batch_sizes is None:
            batch_sizes = [1, 8, 32, 64]
        
        results = {}
        
        for mode in modes_to_test:
            # Save current mode
            current_mode = self.qr_calculator.config['mode']
            
            # Set to test mode
            if isinstance(mode, str):
                try:
                    mode = QuickRecallMode(mode)
                except ValueError:
                    logger.warning(f"Invalid mode: {mode}, skipping")
                    continue
            
            self.qr_calculator.config['mode'] = mode
            mode_name = mode.value
            results[mode_name] = {}
            
            for batch_size in batch_sizes:
                batch_results = {
                    'response_times_ms': [],
                    'scores': []
                }
                
                # Run benchmark with single embeddings first
                if batch_size == 1:
                    start_time = time.time()
                    for emb in test_embeddings:
                        # Track individual response times
                        item_start = time.time()
                        context = {"timestamp": time.time(), "device": str(emb.device)}
                        score = await self.qr_calculator.calculate(emb, context=context)
                        item_elapsed = (time.time() - item_start) * 1000
                        batch_results['response_times_ms'].append(item_elapsed)
                        batch_results['scores'].append(score)
                    
                    total_elapsed = time.time() - start_time
                
                # Then batch processing
                else:
                    start_time = time.time()
                    
                    # Split into batches of the specified size
                    batches = [test_embeddings[i:i+batch_size] for i in range(0, len(test_embeddings), batch_size)]
                    
                    for batch in batches:
                        # Track batch response time
                        batch_start = time.time()
                        results_batch = await self.process_embedding_batch(batch)
                        batch_elapsed = (time.time() - batch_start) * 1000
                        
                        # Record scores and estimate per-item times
                        for _, score in results_batch:
                            batch_results['scores'].append(score)
                        
                        # Record the batch time divided by items as an estimate
                        per_item_time = batch_elapsed / len(batch)
                        batch_results['response_times_ms'].extend([per_item_time] * len(batch))
                    
                    total_elapsed = time.time() - start_time
                
                # Calculate statistics
                response_times = np.array(batch_results['response_times_ms'])
                throughput = len(test_embeddings) / total_elapsed  # embeddings per second
                
                # Store comprehensive metrics
                results[mode_name][f'batch_{batch_size}'] = {
                    'throughput_eps': throughput,
                    'time_total_ms': total_elapsed * 1000,
                    'time_per_embedding_avg_ms': np.mean(response_times),
                    'time_per_embedding_median_ms': np.median(response_times),
                    'time_per_embedding_p95_ms': np.percentile(response_times, 95),
                    'time_per_embedding_p99_ms': np.percentile(response_times, 99),
                    'time_per_embedding_std_ms': np.std(response_times),
                    'avg_score': np.mean(batch_results['scores']),
                    'score_std': np.std(batch_results['scores']),
                    'embeddings_processed': len(test_embeddings)
                }
            
            # Restore original mode
            self.qr_calculator.config['mode'] = current_mode
        
        return results
        
    def get_detailed_performance_stats(self):
        """
        Get detailed performance statistics including CPU and GPU utilization.
        
        Returns:
            Dictionary with detailed system performance metrics
        """
        stats = {
            'timestamp': time.time(),
            'cpu': {},
            'gpu': {},
            'memory': {}
        }
        
        # Get CPU statistics
        try:
            import psutil
            stats['cpu']['usage_percent'] = psutil.cpu_percent(interval=0.1)
            stats['cpu']['count'] = psutil.cpu_count()
            stats['cpu']['count_logical'] = psutil.cpu_count(logical=True)
            
            # Memory usage
            memory = psutil.virtual_memory()
            stats['memory']['total_gb'] = memory.total / (1024 * 1024 * 1024)
            stats['memory']['available_gb'] = memory.available / (1024 * 1024 * 1024)
            stats['memory']['used_gb'] = memory.used / (1024 * 1024 * 1024)
            stats['memory']['percent'] = memory.percent
        except ImportError:
            stats['cpu']['error'] = "psutil module not available. Install with: pip install psutil"
        
        # Get GPU statistics
        if torch.cuda.is_available():
            stats['gpu']['count'] = torch.cuda.device_count()
            stats['gpu']['current_device'] = torch.cuda.current_device()
            stats['gpu']['memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats['gpu']['memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
            stats['gpu']['max_memory_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            # Per-device stats
            stats['gpu']['devices'] = {}
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    stats['gpu']['devices'][i] = {
                        'name': torch.cuda.get_device_name(i),
                        'memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                        'memory_reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                        'memory_cached_mb': torch.cuda.memory_cached() / (1024 * 1024) if hasattr(torch.cuda, 'memory_cached') else 0
                    }
            
            # Try to get GPU utilization with pynvml if available
            try:
                import pynvml
                pynvml.nvmlInit()
                for i in range(torch.cuda.device_count()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats['gpu']['devices'][i]['gpu_utilization_percent'] = utilization.gpu
                    stats['gpu']['devices'][i]['memory_utilization_percent'] = utilization.memory
                pynvml.nvmlShutdown()
            except (ImportError, Exception) as e:
                stats['gpu']['utilization_error'] = f"pynvml error: {str(e)}"
        
        return stats
        
    async def run_profiling(self, test_embeddings, output_file=None, profile_calculator=True, profile_manager=True):
        """
        Run profiling on components of the processing pipeline.
        
        Args:
            test_embeddings: List of embeddings to profile with
            output_file: Optional file to save profile data to (.prof extension recommended)
            profile_calculator: Whether to profile the calculator component
            profile_manager: Whether to profile the flow manager processes
            
        Returns:
            Profile statistics object
        """
        import cProfile
        import pstats
        import io
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        if profile_calculator:
            logger.info("Profiling calculator component...")
            for emb in test_embeddings:
                context = {"timestamp": time.time(), "device": str(emb.device)}
                await self.qr_calculator.calculate(emb, context=context)
        
        if profile_manager:
            logger.info("Profiling flow manager processing...")
            for emb in test_embeddings:
                await self.process_embedding(emb)
                
            # Also test batch processing
            batch_size = min(len(test_embeddings), 16)
            await self.process_embedding_batch(test_embeddings[:batch_size])
        
        profiler.disable()
        
        # Process and display/save results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Print top 30 time-consuming functions
        profile_text = s.getvalue()
        
        if output_file:
            ps.dump_stats(output_file)
            logger.info(f"Profile data saved to {output_file}")
        
        # Print the profile summary to the log
        for line in profile_text.split('\n')[:40]:  # Print first 40 lines
            logger.info(line)
            
        return ps
        
    async def process_large_embedding_batch(self, embeddings, max_chunk_size=100, progress_callback=None):
        """
        Process very large batches by chunking to manage memory usage.
        
        Args:
            embeddings: List of embeddings to process
            max_chunk_size: Maximum size of each processing chunk
            progress_callback: Optional callback function(processed_count, total_count) 
                              for reporting progress
            
        Returns:
            List of tuples (adjusted_embedding, score) for all embeddings processed
        """
        all_results = []
        chunks = [embeddings[i:i + max_chunk_size] for i in range(0, len(embeddings), max_chunk_size)]
        
        total_processed = 0
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} embeddings")
            results = await self.process_embedding_batch(chunk)
            all_results.extend(results)
            
            total_processed += len(chunk)
            if progress_callback:
                try:
                    progress_callback(total_processed, len(embeddings))
                except Exception as e:
                    logger.warning(f"Error in progress callback: {e}")
            
            # Force garbage collection between chunks
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Short pause to allow other tasks to run
            await asyncio.sleep(0.01)
        
        return all_results
    
    def validate_configuration(self, config=None, validate_calculator=True):
        """
        Validate configuration and warn about potentially problematic settings.
        
        Args:
            config: Configuration dict to validate (uses current config if None)
            validate_calculator: Whether to validate calculator config too
            
        Returns:
            Dictionary with validation results including warnings and recommendations
        """
        if config is None:
            config = self.config
            
        validation = {
            "valid": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Validate device settings
        if config.get('device') == 'cuda' and not torch.cuda.is_available():
            validation["valid"] = False
            validation["warnings"].append("Configuration specifies CUDA device but CUDA is not available")
            validation["recommendations"].append("Change device to 'cpu' or install CUDA")
        
        # Validate batch size
        if config.get('batch_size', 0) > 64 and config.get('device') == 'cpu':
            validation["warnings"].append(f"Large batch size ({config['batch_size']}) on CPU may cause performance issues")
            validation["recommendations"].append("Reduce batch size to 16-32 for CPU processing")
        
        # Validate embedding dimension
        if config.get('embedding_dim', 0) <= 0:
            validation["valid"] = False
            validation["warnings"].append("Invalid embedding dimension")
            validation["recommendations"].append("Set a positive embedding dimension value")
        
        # Validate momentum buffer size
        if config.get('max_momentum_size', 0) < 10:
            validation["warnings"].append("Small momentum buffer may affect scoring quality")
            validation["recommendations"].append("Increase max_momentum_size to at least 50 for better results")
        
        # Validate calculator configuration
        if validate_calculator and hasattr(self, 'qr_calculator'):
            calculator_config = self.qr_calculator.config
            
            # Check factor weights
            if calculator_config.get('mode') == QuickRecallMode.CUSTOM:
                if not self.qr_calculator.factor_weights or sum(self.qr_calculator.factor_weights.values()) != 1.0:
                    validation["warnings"].append("Custom mode is set but factor weights don't sum to 1.0")
                    validation["recommendations"].append("Adjust factor weights to sum to 1.0")
            
            # Check calculator's embedding dimension matches flow manager's
            if calculator_config.get('embedding_dim') != config.get('embedding_dim'):
                validation["valid"] = False
                validation["warnings"].append("Embedding dimensions in calculator and flow manager don't match")
                validation["recommendations"].append("Synchronize embedding dimensions across both components")
        
        return validation
        
    async def warm_up(self, num_samples=10):
        """
        Warm up the pipeline with random embeddings to initialize and optimize components.
        
        Args:
            num_samples: Number of random samples to use for warming up
            
        Returns:
            Dictionary with warm-up statistics
        """
        logger.info(f"Warming up HPCQRFlowManager with {num_samples} samples...")
        
        # Create random embeddings for warm-up
        dim = self.config['embedding_dim']
        device_str = self.config['device']
        device = torch.device(device_str)
        
        warm_up_embeddings = [torch.randn(dim, device=device) for _ in range(num_samples)]
        
        # Process in single and batch mode
        start_time = time.time()
        
        # Single processing
        for emb in warm_up_embeddings[:3]:
            await self.process_embedding(emb)
            
        # Batch processing
        await self.process_embedding_batch(warm_up_embeddings)
        
        # Force compile optimization if using PyTorch 2.0+
        if hasattr(torch, 'compile') and device_str == 'cuda':
            try:
                logger.info("Attempting to use torch.compile for optimization...")
                sample_emb = warm_up_embeddings[0]
                
                # Try to optimize the calculator's core calculation
                @torch.compile
                def optimized_calculate(embedding):
                    return self.qr_calculator._calculate_component_scores(embedding)
                
                # Run the compiled function
                _ = optimized_calculate(sample_emb)
                logger.info("Successfully used torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile optimization failed: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"Warm-up complete in {elapsed:.2f}s")
        
        return {
            "warm_up_time": elapsed,
            "samples": num_samples,
            "device": device_str
        }
    
    async def process_text_and_embedding(self, text: str, embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Process text and/or embedding through the HPC-QR pipeline.
        Handles dimension mismatches between embeddings by using the qr_calculator's normalize_embedding method.
        
        Args:
            text: The text to process
            embedding: Optional pre-computed embedding (if None, will be generated from text)
            
        Returns:
            Dict containing the QR calculation results
        """
        tensor_server = await self._get_tensor_server_instance()
        
        if embedding is None and text:
            try:
                # Generate embedding from text
                embedding = await self.get_embedding(text)
            except Exception as e:
                logger.error(f"Error generating embedding from text: {e}")
                # Fall back to a random normalized embedding if we can't generate one
                embedding_dim = self.config.get('embedding_dim', 768)
                random_embedding = torch.randn(embedding_dim)
                embedding = random_embedding / torch.norm(random_embedding)
        
        # Make sure we have an embedding to work with
        if embedding is None:
            logger.warning("No embedding provided or generated in process_text_and_embedding")
            embedding_dim = self.config.get('embedding_dim', 768)
            random_embedding = torch.randn(embedding_dim)
            embedding = random_embedding / torch.norm(random_embedding)
        
        # Check embedding dimensions
        expected_dim = self.config.get('embedding_dim', 768)
        actual_dim = embedding.shape[0]
        if actual_dim != expected_dim:
            logger.debug(f"Embedding dimension mismatch in process_text_and_embedding: expected {expected_dim}, got {actual_dim}")
            # The calculator will handle the dimension mismatch internally
        
        # Create the context dictionary for the calculator
        context = {
            'text': text,
            'tensor_server': tensor_server,
            'config': self.config,
        }
        
        # Get the appropriate calculator reference 
        calculator = self.calculator if hasattr(self, 'calculator') else self.qr_calculator
        
        # Calculate QR scores - pass the embedding as the first parameter and text as the second
        qr_score = await calculator.calculate(
            embedding,  # first parameter is embedding_or_text
            text,       # second parameter is text
            context     # third parameter is context
        )
        
        # Create results dictionary
        results = {
            'embedding': embedding,
            'qr_score': qr_score
        }
        
        return results