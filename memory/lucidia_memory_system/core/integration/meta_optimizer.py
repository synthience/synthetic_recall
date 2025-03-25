"""
Meta-Adaptive Parameter Optimizer for Lucid Recall

This module implements a Bayesian Optimization approach to dynamically tune
the hyperparameters of the HPCQRFlowManager based on observed performance metrics.
"""

import logging
import torch
import numpy as np
import time
import asyncio
from typing import Dict, Any, List, Tuple, Optional, Callable
import threading
from collections import deque

# For Bayesian Optimization
try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_model
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from torch.quasirandom import SobolEngine
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    logging.warning("BoTorch not available. Falling back to simpler optimization strategies.")

logger = logging.getLogger(__name__)

class MetaAdaptiveOptimizer:
    """
    Meta-Adaptive Parameter Optimizer using Bayesian Optimization
    
    Dynamically tunes the hyperparameters of the HPCQRFlowManager based on
    observed performance metrics and changing data characteristics.
    """
    
    def __init__(self, hpc_manager, config: Dict[str, Any] = None):
        """
        Initialize the meta-adaptive optimizer.
        
        Args:
            hpc_manager: Reference to the HPCQRFlowManager instance
            config: Configuration dictionary for the optimizer
        """
        self.hpc_manager = hpc_manager
        self.config = {
            'enabled': True,
            'optimization_interval': 1000,  # Update hyperparameters every N processed embeddings
            'exploration_rate': 0.2,        # Initial exploration rate (decreases over time)
            'learning_rate': 0.05,          # Learning rate for simple gradient-based updates
            'window_size': 50,              # Number of observations to keep for optimization
            'warm_start_trials': 10,        # Number of random trials before starting optimization
            'min_improvement': 0.01,        # Minimum improvement to accept a parameter change
            'max_trials': 100,              # Maximum optimization trials per update cycle
            'parameter_bounds': {
                'alpha': (0.1, 0.6),        # Weight for geometry-aware distance
                'beta': (0.1, 0.5),         # Weight for causal/contextual novelty
                'gamma': (0.1, 0.4),        # Weight for self-organization divergence
                'delta': (0.05, 0.2),       # Weight for redundancy/overlap penalty
                'dynamic_scaling_factor': (0.1, 1.0),  # Scaling factor for shock absorption
            },
            'use_bayesian_opt': BOTORCH_AVAILABLE,  # Use Bayesian optimization if available
            'metrics_to_optimize': ['retrieval_accuracy', 'drift_stability', 'diversity'],
            'metric_weights': {
                'retrieval_accuracy': 0.5,    # Weight for retrieval accuracy in optimization
                'drift_stability': 0.3,       # Weight for drift stability in optimization
                'diversity': 0.2,             # Weight for embedding diversity in optimization
            },
            **(config or {})
        }
        
        # Parameter history and performance metrics
        self.parameter_history = []
        self.performance_metrics = deque(maxlen=self.config['window_size'])
        
        # Current best parameters and score
        self.best_parameters = None
        self.best_score = -float('inf')
        
        # Initialize metrics tracking
        self.metrics = {
            'retrieval_accuracy': 0.0,
            'diversity': 0.0,
            'drift_stability': 1.0,
            'processing_speed': 0.0,
            'memory_efficiency': 0.0,
        }
        
        # Trial counter
        self.trials = 0
        self.processed_count_last_update = 0
        
        # For simple gradient-based optimization when BoTorch is not available
        self.gradient_estimates = {
            'alpha': 0.0,
            'beta': 0.0,
            'gamma': 0.0,
            'delta': 0.0,
            'dynamic_scaling_factor': 0.0,
        }
        
        # Initialize lock for thread safety
        self._lock = asyncio.Lock()
        
        # Optimization thread/task
        self._optimization_task = None
        self._stop_event = threading.Event()
        
        logger.info(f"Initialized MetaAdaptiveOptimizer with config: {self.config}")
        
        # Store initial parameters as baseline
        self._store_initial_parameters()
    
    def _store_initial_parameters(self):
        """Store the initial parameters from the HPC manager as baseline"""
        self.initial_parameters = {
            'alpha': self.hpc_manager.config['alpha'],
            'beta': self.hpc_manager.config['beta'],
            'gamma': self.hpc_manager.config['gamma'],
            'delta': self.hpc_manager.config['delta'],
            'dynamic_scaling_factor': self.hpc_manager.config['dynamic_scaling_factor'],
        }
        
        # Set current best parameters to initial parameters
        self.best_parameters = self.initial_parameters.copy()
    
    async def start(self):
        """Start the optimization process asynchronously"""
        if not self.config['enabled']:
            logger.info("Meta-adaptive optimization is disabled")
            return
        
        logger.info("Starting meta-adaptive parameter optimization")
        
        # Create and start the optimization task
        self._optimization_task = asyncio.create_task(self._optimization_loop())
    
    async def stop(self):
        """Stop the optimization process"""
        if self._optimization_task is not None:
            self._stop_event.set()
            await self._optimization_task
            self._optimization_task = None
            logger.info("Meta-adaptive optimization stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop that periodically updates parameters"""
        while not self._stop_event.is_set():
            try:
                # Check if it's time to update parameters
                current_processed = self.hpc_manager._stats['processed_count']
                if (current_processed - self.processed_count_last_update) >= self.config['optimization_interval']:
                    await self._update_parameters()
                    self.processed_count_last_update = current_processed
                
                # Sleep to avoid excessive CPU usage
                await asyncio.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(30.0)  # Sleep longer on error
    
    async def _update_parameters(self):
        """Update the hyperparameters based on collected performance metrics"""
        async with self._lock:
            try:
                # Calculate current performance
                current_score = self._calculate_performance_score()
                
                # Store current parameters and performance
                current_parameters = {
                    'alpha': self.hpc_manager.config['alpha'],
                    'beta': self.hpc_manager.config['beta'],
                    'gamma': self.hpc_manager.config['gamma'],
                    'delta': self.hpc_manager.config['delta'],
                    'dynamic_scaling_factor': self.hpc_manager.config['dynamic_scaling_factor'],
                }
                
                self.parameter_history.append((current_parameters, current_score))
                self.performance_metrics.append((current_parameters, current_score, self.metrics.copy()))
                
                # Update best parameters if current score is better
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_parameters = current_parameters.copy()
                
                # Determine if we have enough data to start optimization
                if len(self.parameter_history) < self.config['warm_start_trials']:
                    # Not enough history yet, use exploration strategy
                    new_parameters = self._exploration_strategy()
                elif self.config['use_bayesian_opt'] and BOTORCH_AVAILABLE:
                    # Use Bayesian optimization
                    new_parameters = await self._bayesian_optimization()
                else:
                    # Use simpler gradient-based strategy
                    new_parameters = self._gradient_strategy()
                
                # Apply the new parameters to the HPC manager
                logger.info(f"Updating parameters: {new_parameters}")
                for param, value in new_parameters.items():
                    self.hpc_manager.config[param] = value
                
                # Increment trials counter
                self.trials += 1
                
                # Log performance improvement
                logger.info(f"Parameter update complete. Current score: {current_score:.4f}, Best score: {self.best_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error updating parameters: {e}")
    
    def _calculate_performance_score(self) -> float:
        """
        Calculate a unified performance score from multiple metrics
        
        Returns:
            A single scalar performance score (higher is better)
        """
        weighted_score = 0.0
        
        # Apply weights to individual metrics
        for metric_name, weight in self.config['metric_weights'].items():
            if metric_name in self.metrics:
                weighted_score += self.metrics[metric_name] * weight
        
        return weighted_score
    
    def _exploration_strategy(self) -> Dict[str, float]:
        """
        Generate random parameter values for exploration
        
        Returns:
            Dictionary of new parameter values
        """
        new_params = {}
        
        # Random exploration within bounds
        for param, (min_val, max_val) in self.config['parameter_bounds'].items():
            # More exploration early, less exploration later
            exploration_factor = self.config['exploration_rate'] * (1.0 - min(0.9, self.trials / 100.0))
            
            # Mix between current value and random values
            current_value = self.hpc_manager.config[param]
            random_value = min_val + (max_val - min_val) * torch.rand(1).item()
            
            # Weighted combination of current and random
            new_value = (1 - exploration_factor) * current_value + exploration_factor * random_value
            
            # Ensure we stay within bounds
            new_params[param] = max(min_val, min(max_val, new_value))
        
        return new_params
    
    async def _bayesian_optimization(self) -> Dict[str, float]:
        """
        Perform Bayesian optimization to find optimal hyperparameters
        
        Returns:
            Dictionary of optimized parameter values
        """
        if not BOTORCH_AVAILABLE:
            logger.warning("BoTorch not available. Falling back to gradient strategy.")
            return self._gradient_strategy()
        
        try:
            # Convert parameter history to torch tensors
            X = []  # Parameter vectors
            Y = []  # Performance scores
            
            for params, score in self.parameter_history[-self.config['window_size']:]:
                # Extract parameter values in a consistent order
                param_vector = [
                    params['alpha'],
                    params['beta'],
                    params['gamma'],
                    params['delta'],
                    params['dynamic_scaling_factor']
                ]
                X.append(param_vector)
                Y.append([score])
            
            X_tensor = torch.tensor(X, dtype=torch.float64)
            Y_tensor = torch.tensor(Y, dtype=torch.float64)
            
            # Normalize X to [0, 1] for better GP performance
            bounds = torch.tensor([
                self.config['parameter_bounds']['alpha'],
                self.config['parameter_bounds']['beta'],
                self.config['parameter_bounds']['gamma'],
                self.config['parameter_bounds']['delta'],
                self.config['parameter_bounds']['dynamic_scaling_factor']
            ], dtype=torch.float64)
            
            X_normalized = (X_tensor - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
            
            # Initialize GP model
            gp = SingleTaskGP(X_normalized, Y_tensor)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            
            # Fit GP model
            fit_gpytorch_model(mll)
            
            # Set up acquisition function (Expected Improvement)
            best_y = Y_tensor.max().item()
            EI = ExpectedImprovement(gp, best_f=best_y)
            
            # Generate candidates using Sobol sequences for better coverage
            sobol = SobolEngine(dimension=5, scramble=True)
            candidates = sobol.draw(256)
            
            # Optimize acquisition function
            candidate, acq_value = optimize_acqf(
                EI,
                bounds=torch.tensor([[0.0] * 5, [1.0] * 5], dtype=torch.float64),
                q=1,
                num_restarts=10,
                raw_samples=candidates
            )
            
            # Denormalize the candidate to original parameter space
            optimal_params = candidate[0] * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            
            # Convert to parameter dictionary
            new_params = {
                'alpha': optimal_params[0].item(),
                'beta': optimal_params[1].item(),
                'gamma': optimal_params[2].item(),
                'delta': optimal_params[3].item(),
                'dynamic_scaling_factor': optimal_params[4].item()
            }
            
            return new_params
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return self._gradient_strategy()  # Fall back to simpler method
    
    def _gradient_strategy(self) -> Dict[str, float]:
        """
        Simple gradient-based optimization strategy for hyperparameter tuning
        when Bayesian optimization is not available
        
        Returns:
            Dictionary of updated parameter values
        """
        new_params = {}
        
        if len(self.parameter_history) < 2:
            return self._exploration_strategy()
        
        # Get current and previous parameters and scores
        current_params, current_score = self.parameter_history[-1]
        prev_params, prev_score = self.parameter_history[-2]
        
        # Basic gradient calculation
        score_diff = current_score - prev_score
        
        # Update gradient estimates for each parameter
        for param in self.gradient_estimates.keys():
            if param in current_params and param in prev_params:
                param_diff = current_params[param] - prev_params[param]
                
                # Avoid division by zero
                if abs(param_diff) > 1e-10:
                    # Approximate gradient: change in score / change in parameter
                    gradient = score_diff / param_diff
                    
                    # Update running gradient estimate with exponential moving average
                    self.gradient_estimates[param] = 0.7 * self.gradient_estimates[param] + 0.3 * gradient
        
        # Apply gradient updates with learning rate
        for param, (min_val, max_val) in self.config['parameter_bounds'].items():
            # Get current parameter value
            current_value = current_params[param]
            
            # Apply update in direction of positive gradient
            if abs(self.gradient_estimates[param]) > 1e-6:  # Only update if gradient is significant
                step_size = self.config['learning_rate'] * self.gradient_estimates[param]
                
                # Add some noise for exploration
                noise = 0.01 * (min_val - max_val) * (torch.rand(1).item() - 0.5)
                new_value = current_value + step_size + noise
                
                # Ensure we stay within bounds
                new_params[param] = max(min_val, min(max_val, new_value))
            else:
                # If gradient is small, add small random perturbation
                perturbation = 0.05 * (min_val - max_val) * (torch.rand(1).item() - 0.5)
                new_value = current_value + perturbation
                new_params[param] = max(min_val, min(max_val, new_value))
        
        return new_params
    
    async def update_metrics(self, new_metrics: Dict[str, float]):
        """
        Update the performance metrics based on observed data
        
        Args:
            new_metrics: Dictionary of updated metrics
        """
        async with self._lock:
            for metric, value in new_metrics.items():
                if metric in self.metrics:
                    # Exponential moving average to smooth metrics
                    self.metrics[metric] = 0.9 * self.metrics[metric] + 0.1 * value
    
    def estimate_retrieval_accuracy(self, query_results, ground_truth=None) -> float:
        """
        Estimate retrieval accuracy based on query results
        
        Args:
            query_results: Results from a query operation
            ground_truth: Optional ground truth for direct comparison
            
        Returns:
            Estimated retrieval accuracy score [0-1]
        """
        # If we have ground truth, we can calculate precision/recall
        if ground_truth is not None:
            # Simple precision calculation
            if len(query_results) > 0:
                correct = sum(1 for item in query_results if item in ground_truth)
                precision = correct / len(query_results)
                return precision
            return 0.0
        
        # Without ground truth, use similarity clustering as a proxy
        # High similarity within results may indicate coherent retrieval
        if not query_results or len(query_results) < 2:
            return 0.5  # Neutral score for insufficient data
        
        try:
            # Extract embeddings from results
            embeddings = [result[0] for result in query_results if isinstance(result, tuple) and len(result) > 0]
            
            if len(embeddings) < 2:
                return 0.5
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = torch.nn.functional.cosine_similarity(
                        embeddings[i].unsqueeze(0), 
                        embeddings[j].unsqueeze(0)
                    ).item()
                    similarities.append(sim)
            
            # Average similarity as a proxy for coherence
            avg_similarity = sum(similarities) / len(similarities)
            
            # Convert to a score in [0,1]
            # We want similarity to be high but not too high (which would indicate redundancy)
            # Ideal range might be 0.5-0.8 for diverse but related results
            if avg_similarity < 0.3:
                # Too dissimilar, likely poor retrieval
                return 0.3 + avg_similarity
            elif avg_similarity > 0.9:
                # Too similar, likely redundant retrieval
                return 1.3 - avg_similarity
            else:
                # Good balance of similarity and diversity
                return 0.85
        except Exception as e:
            logger.error(f"Error estimating retrieval accuracy: {e}")
            return 0.5
    
    def calculate_embedding_diversity(self) -> float:
        """
        Calculate the diversity of embeddings in the momentum buffer
        
        Returns:
            Diversity score [0-1], higher means more diverse
        """
        if self.hpc_manager.momentum_buffer is None or len(self.hpc_manager.momentum_buffer) < 2:
            return 0.5  # Default value for insufficient data
        
        try:
            # Sample embeddings if there are too many
            momentum_buffer = self.hpc_manager.momentum_buffer
            if len(momentum_buffer) > 100:
                indices = torch.randperm(len(momentum_buffer))[:100]
                sample_embeddings = momentum_buffer[indices]
            else:
                sample_embeddings = momentum_buffer
            
            # Calculate pairwise distances
            # Use cosine distance (1 - cosine similarity)
            n = sample_embeddings.shape[0]
            
            # For efficiency, calculate using batched operations
            # Normalize first (if not already normalized)
            normalized = torch.nn.functional.normalize(sample_embeddings, p=2, dim=1)
            
            # Calculate similarity matrix
            similarity_matrix = torch.mm(normalized, normalized.t())
            
            # Mask out the diagonal (self-similarity = 1)
            mask = torch.eye(n, device=similarity_matrix.device)
            masked_similarities = similarity_matrix * (1 - mask)
            
            # Average similarity (excluding self-similarities)
            total_similarity = masked_similarities.sum().item()
            avg_similarity = total_similarity / (n * (n - 1))
            
            # Convert to diversity score (1 - avg_similarity)
            diversity = 1.0 - avg_similarity
            
            return diversity
            
        except Exception as e:
            logger.error(f"Error calculating embedding diversity: {e}")
            return 0.5
    
    def calculate_drift_stability(self) -> float:
        """
        Calculate the stability of the embedding space over time
        
        Returns:
            Stability score [0-1], higher means more stable
        """
        if self.hpc_manager.momentum_buffer is None or len(self.hpc_manager.momentum_buffer) < 10:
            return 1.0  # Default value for insufficient data
        
        try:
            buffer = self.hpc_manager.momentum_buffer
            
            # Split the buffer into older and newer portions
            split_point = len(buffer) // 2
            older_embeddings = buffer[:split_point]
            newer_embeddings = buffer[split_point:]
            
            # Calculate centers of each set
            older_center = torch.mean(older_embeddings, dim=0)
            newer_center = torch.mean(newer_embeddings, dim=0)
            
            # Calculate drift as distance between centers
            drift = torch.norm(newer_center - older_center).item()
            
            # Convert to stability score (inverse of drift)
            # Normalize to [0,1] scale with exponential decay
            stability = torch.exp(-drift * 5.0).item()
            
            return stability
            
        except Exception as e:
            logger.error(f"Error calculating drift stability: {e}")
            return 1.0
    
    async def analyze_query_results(self, query, results, ground_truth=None):
        """
        Analyze query results and update metrics
        
        Args:
            query: The query that produced the results
            results: The retrieval results
            ground_truth: Optional ground truth for evaluation
        """
        metrics_update = {}
        
        # Estimate retrieval accuracy
        accuracy = self.estimate_retrieval_accuracy(results, ground_truth)
        metrics_update['retrieval_accuracy'] = accuracy
        
        # Calculate embedding diversity
        diversity = self.calculate_embedding_diversity()
        metrics_update['diversity'] = diversity
        
        # Calculate drift stability
        stability = self.calculate_drift_stability()
        metrics_update['drift_stability'] = stability
        
        # Update metrics
        await self.update_metrics(metrics_update)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current state statistics about the optimizer
        
        Returns:
            Dictionary with various performance metrics and state information
        """
        stats = {
            'enabled': self.config['enabled'],
            'trials': self.trials,
            'current_metrics': self.metrics.copy(),
            'best_score': self.best_score,
            'best_parameters': self.best_parameters.copy() if self.best_parameters else None,
            'current_parameters': {
                'alpha': self.hpc_manager.config['alpha'],
                'beta': self.hpc_manager.config['beta'],
                'gamma': self.hpc_manager.config['gamma'],
                'delta': self.hpc_manager.config['delta'],
                'dynamic_scaling_factor': self.hpc_manager.config['dynamic_scaling_factor'],
            },
            'initial_parameters': self.initial_parameters.copy(),
            'optimization_method': 'bayesian' if self.config['use_bayesian_opt'] and BOTORCH_AVAILABLE else 'gradient',
        }
        
        return stats
