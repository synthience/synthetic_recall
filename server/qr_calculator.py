"""
LUCID RECALL PROJECT
Unified Quick Recall Calculator 

Agent: Lucidia 1.1
Date: 05/03/25
Time: 4:43 PM EST

A standardized quick recall (HPC-QR) calculator for consistent memory importance
assessment across all memory system components.
"""

import os
import math
import logging
import json
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import aiohttp
import scipy.stats as stats
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable, Coroutine
from enum import Enum
from scipy.spatial.distance import cdist
from scipy.special import expit, logsumexp
from scipy import stats
import networkx as nx

logger = logging.getLogger(__name__)


def safe_run(coro_func, *args, **kwargs):
    """Safely run an async coroutine in any context.
    
    This function handles the "This event loop is already running" error by detecting
    whether there's an existing event loop, and deciding whether to use asyncio.run() 
    or a different approach for an already running loop.
    
    Args:
        coro_func: The coroutine function to run
        *args: Arguments to pass to the coroutine function
        **kwargs: Keyword arguments to pass to the coroutine function
        
    Returns:
        The result of the coroutine
    """
    # Create the coroutine object
    coro = coro_func(*args, **kwargs)
    
    try:
        # Try to get the current running loop
        loop = asyncio.get_running_loop()
        
        # We're inside a running event loop, so we need to handle this differently
        # We can't use run_until_complete when a loop is already running
        # Let's use a thread to run the coroutine in a separate event loop
        import threading
        import concurrent.futures
        
        # Function to run in a separate thread
        def run_in_new_loop(coro):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        
        # Use a ThreadPoolExecutor to run the function in a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop, coro)
            return future.result()
            
    except RuntimeError:
        # No loop running, create a new one with asyncio.run
        return asyncio.run(coro)


class QuickRecallMode(Enum):
    """Operating modes for HPC-QR calculation approach."""
    STANDARD = "standard"       # Balanced approach for general use
    PRECISE = "precise"         # More detailed HPC-QR for thorough analysis
    EFFICIENT = "efficient"     # Simplified HPC-QR for speed
    EMOTIONAL = "emotional"     # Prioritizes emotional signals
    INFORMATIONAL = "informational"  # Prioritizes info density
    PERSONAL = "personal"       # Prioritizes personal relevance
    CUSTOM = "custom"           # Uses custom HPC-QR factor weights
    HPC_QR = "hpc_qr"           # Direct replacement for HPCQRFlowManager
    MINIMAL = "minimal"         # Very simple HPC-QR approach (basic recency + a few factors)
    CLASSIC_QR = "classic_qr"   # Original QuickRecall approach prior to HPC-QR

class QuickRecallFactor(Enum):
    """
    Factors that contribute to HPC-QR. Each factor can represent geometry, novelty,
    emotional signals, redundancy, etc.
    """
    # Traditional elements (from older significance approach)
    SURPRISE = "surprise"
    DIVERSITY = "diversity"
    EMOTION = "emotion"
    RECENCY = "recency"
    IMPORTANCE = "importance"
    PERSONAL = "personal"
    COHERENCE = "coherence"
    INFORMATION = "information"
    RELEVANCE = "relevance"
    USER_ATTENTION = "user_attention"

    # HPC-QR-specific components:
    R_GEOMETRY = "r_geometry"         # geometry-based distance (e.g., hyperbolic)
    CAUSAL_NOVELTY = "causal_novel"   # how surprising under a causal model
    SELF_ORG = "self_org"            # measure of self-organizing reconfiguration
    OVERLAP = "overlap"              # redundancy with existing memory

class GeometryType(Enum):
    """Types of geometry used for embedding space calculations."""
    EUCLIDEAN = "euclidean"
    HYPERBOLIC = "hyperbolic"
    SPHERICAL = "spherical"
    MIXED = "mixed"  # Combination (if implementing piecewise curvature)

class UnifiedQuickRecallCalculator:
    """
    Unified Quick Recall (HPC-QR) calculator for consistent assessment of memory importance.

    This class merges multiple factors (geometry, novelty, emotional signals, redundancy, etc.)
    into a single HPC-QR score. It is a direct replacement for older significance-based logic.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Quick Recall calculator.

        Args:
            config: Configuration options including:
                - mode (QuickRecallMode)
                - factor_weights: custom HPC-QR factor weights
                - time_decay_rate: rate at which HPC-QR decays over time
                - novelty_threshold: threshold for surprise/novelty
                - min_qr_score: minimum HPC-QR score
                - max_qr_score: maximum HPC-QR score
                - adaptive_thresholds: whether to adapt thresholds based on data
                - history_window: number of samples to keep for adaptive thresholds
                - geometry_type: GeometryType for embedding space calculations
                - curvature: curvature parameter for non-Euclidean geometries
                - embedding_dim: dimension of embedding vectors
                - causal_graph: optional pre-built causal graph (networkx.DiGraph)
                - som_grid_size: size of self-organizing map grid
                - som_learning_rate: learning rate for SOM updates
                - som_sigma: neighborhood parameter for SOM
                - personal_information_keywords, emotional_keywords, etc.
        """
        self.config = {
            'mode': QuickRecallMode.STANDARD,
            'factor_weights': {},
            'time_decay_rate': 0.15,      # Slightly increased from 0.1
            'novelty_threshold': 0.45,    # Lowered from 0.7
            'min_qr_score': 0.0,
            'max_qr_score': 1.0,
            'adaptive_thresholds': True,
            'history_window': 500,        # Increased from 100
            
            # Z-score normalization settings
            'use_zscore_normalization': True,
            'z_score_window_size': 100,
            
            # Geometry configuration
            'geometry_type': GeometryType.EUCLIDEAN,
            'curvature': -1.0,   # Negative for hyperbolic, 0 for Euclidean, positive for spherical
            'embedding_dim': 768,

            # Causal model parameters
            'causal_prior_strength': 0.1,
            'causal_learning_rate': 0.01,
            'causal_edge_threshold': 0.3,
            'causal_max_parents': 5,

            # Self-organizing map parameters
            'som_grid_size': (10, 10),
            'som_learning_rate': 0.1,
            'som_sigma': 2.0,
            'som_decay_rate': 0.01,

            # Overlap/redundancy parameters
            'overlap_threshold': 0.8,
            'overlap_method': 'max_similarity',  # e.g., 'max_similarity', 'avg_similarity', 'adaptive'

            # Additional text-based analysis configs
            'personal_information_keywords': [
                'name', 'address', 'phone', 'email', 'birthday', 'age', 'family',
                'friend', 'password', 'account', 'credit', 'social security',
                'ssn', 'identification', 'id card', 'passport', 'license'
            ],
            'emotional_keywords': [
                'happy', 'sad', 'angry', 'excited', 'love', 'hate', 'scared',
                'anxious', 'proud', 'disappointed', 'hope', 'fear', 'joy',
                'grief', 'frustration', 'satisfaction', 'worry', 'relief'
            ],
            'emotional_intensifiers': [
                'very', 'extremely', 'incredibly', 'absolutely', 'completely',
                'deeply', 'profoundly', 'utterly', 'greatly', 'intensely'
            ],
            'informational_prefixes': [
                'fact:', 'important:', 'remember:', 'note:', 'key point:',
                'critical:', 'essential:', 'reminder:', 'don\'t forget:'
            ],
            'connection_words': [
                'therefore', 'thus', 'because', 'since', 'so', 'as a result', 
                'consequently', 'furthermore', 'moreover', 'in addition',
                'however', 'nevertheless', 'conversely', 'in contrast', 
                'alternatively', 'although', 'despite', 'regardless'
            ],

            # HPC-QR alpha/beta/gamma/delta weighting parameters
            'alpha': 0.35,  # geometry (slightly reduced from 0.4)
            'beta': 0.35,   # causal novelty (increased from 0.3)
            'gamma': 0.2,   # self-org
            'delta': 0.1,   # overlap penalty

            # Defaults if geometry/novelty calculations not available
            'default_geometry_score': 0.3,
            'default_novelty': 0.5,
            'default_overlap': 0.3,
            
            # Set lower time decay minimum value
            'min_time_decay': 0.02,  # Lowered from 0.1

            **(config or {})
        }

        # Initialize HPC-QR factor weights
        self._init_factor_weights()

        # Ensure mode is QuickRecallMode
        if isinstance(self.config['mode'], str):
            try:
                self.config['mode'] = QuickRecallMode(self.config['mode'].lower())
            except ValueError:
                logger.warning(f"Invalid mode: {self.config['mode']}, using STANDARD")
                self.config['mode'] = QuickRecallMode.STANDARD

        # Ensure geometry_type is a GeometryType
        if isinstance(self.config['geometry_type'], str):
            try:
                self.config['geometry_type'] = GeometryType(self.config['geometry_type'].lower())
            except ValueError:
                logger.warning(f"Invalid geometry type: {self.config['geometry_type']}, using EUCLIDEAN")
                self.config['geometry_type'] = GeometryType.EUCLIDEAN

        # Initialize statistical history
        self.history = {
            'calculated_qr': [],
            'factor_values': {fct: [] for fct in QuickRecallFactor},
            'timestamps': []
        }

        # Initialize causal graph
        if 'causal_graph' not in self.config or self.config['causal_graph'] is None:
            self.causal_graph = nx.DiGraph()
            self.causal_weights = {}  # Maps (parent, child) -> weight
        else:
            self.causal_graph = self.config['causal_graph']
            self.causal_weights = {
                (u, v): self.causal_graph[u][v].get('weight', 0.5) 
                for u, v in self.causal_graph.edges()
            }

        # Initialize SOM (Self-Organizing Map)
        self._init_som()

        # Initialize prototype vectors (for geometry calculations)
        self.prototypes = {}

        # Tracking stats
        self.total_calculations = 0
        self.start_time = time.time()
        self.last_calculation_time = 0

        logger.info(f"Initialized UnifiedQuickRecallCalculator with mode: {self.config['mode'].value}")
        logger.info(f"Using geometry: {self.config['geometry_type'].value} with curvature: {self.config['curvature']}")

        # External momentum (e.g. from HPCQRFlowManager)
        self.external_momentum = None

        # Initialize logging counters for dimension mismatch warnings
        self.dim_mismatch_warnings = 0
        self.max_dim_mismatch_warnings = 10  # Max number of warnings to show
        self.dim_mismatch_logged = False  # Flag to indicate if general warning was logged

    def _init_factor_weights(self) -> None:
        """
        Initialize HPC-QR factor weights based on the selected mode.
        """
        standard_weights = {
            QuickRecallFactor.SURPRISE: 0.10,
            QuickRecallFactor.DIVERSITY: 0.05,
            QuickRecallFactor.EMOTION: 0.10,
            QuickRecallFactor.RECENCY: 0.10,
            QuickRecallFactor.IMPORTANCE: 0.10,
            QuickRecallFactor.PERSONAL: 0.10,
            QuickRecallFactor.COHERENCE: 0.05,
            QuickRecallFactor.INFORMATION: 0.05,
            QuickRecallFactor.RELEVANCE: 0.05,
            QuickRecallFactor.USER_ATTENTION: 0.00,
            # HPC-QR-specific components with some default weighting
            QuickRecallFactor.R_GEOMETRY: 0.15,
            QuickRecallFactor.CAUSAL_NOVELTY: 0.10,
            QuickRecallFactor.SELF_ORG: 0.05,
            QuickRecallFactor.OVERLAP: 0.00  # Overlap is often treated as a penalty
        }

        mode_weights = {
            QuickRecallMode.PRECISE: {
                QuickRecallFactor.SURPRISE: 0.05,
                QuickRecallFactor.DIVERSITY: 0.05,
                QuickRecallFactor.EMOTION: 0.10,
                QuickRecallFactor.RECENCY: 0.05,
                QuickRecallFactor.IMPORTANCE: 0.10,
                QuickRecallFactor.PERSONAL: 0.10,
                QuickRecallFactor.COHERENCE: 0.05,
                QuickRecallFactor.INFORMATION: 0.05,
                QuickRecallFactor.RELEVANCE: 0.05,
                QuickRecallFactor.USER_ATTENTION: 0.00,
                QuickRecallFactor.R_GEOMETRY: 0.20,
                QuickRecallFactor.CAUSAL_NOVELTY: 0.15,
                QuickRecallFactor.SELF_ORG: 0.05,
                QuickRecallFactor.OVERLAP: 0.00
            },
            QuickRecallMode.EFFICIENT: {
                QuickRecallFactor.SURPRISE: 0.15,
                QuickRecallFactor.DIVERSITY: 0.10,
                QuickRecallFactor.EMOTION: 0.00,
                QuickRecallFactor.RECENCY: 0.15,
                QuickRecallFactor.IMPORTANCE: 0.15,
                QuickRecallFactor.PERSONAL: 0.10,
                QuickRecallFactor.COHERENCE: 0.00,
                QuickRecallFactor.INFORMATION: 0.00,
                QuickRecallFactor.RELEVANCE: 0.05,
                QuickRecallFactor.USER_ATTENTION: 0.00,
                QuickRecallFactor.R_GEOMETRY: 0.20,
                QuickRecallFactor.CAUSAL_NOVELTY: 0.10,
                QuickRecallFactor.SELF_ORG: 0.00,
                QuickRecallFactor.OVERLAP: 0.00
            },
            QuickRecallMode.EMOTIONAL: {
                QuickRecallFactor.SURPRISE: 0.05,
                QuickRecallFactor.DIVERSITY: 0.05,
                QuickRecallFactor.EMOTION: 0.30,
                QuickRecallFactor.RECENCY: 0.05,
                QuickRecallFactor.IMPORTANCE: 0.05,
                QuickRecallFactor.PERSONAL: 0.15,
                QuickRecallFactor.COHERENCE: 0.00,
                QuickRecallFactor.INFORMATION: 0.00,
                QuickRecallFactor.RELEVANCE: 0.05,
                QuickRecallFactor.USER_ATTENTION: 0.00,
                QuickRecallFactor.R_GEOMETRY: 0.15,
                QuickRecallFactor.CAUSAL_NOVELTY: 0.10,
                QuickRecallFactor.SELF_ORG: 0.05,
                QuickRecallFactor.OVERLAP: 0.00
            },
            QuickRecallMode.INFORMATIONAL: {
                QuickRecallFactor.SURPRISE: 0.05,
                QuickRecallFactor.DIVERSITY: 0.10,
                QuickRecallFactor.EMOTION: 0.00,
                QuickRecallFactor.RECENCY: 0.05,
                QuickRecallFactor.IMPORTANCE: 0.20,
                QuickRecallFactor.PERSONAL: 0.00,
                QuickRecallFactor.COHERENCE: 0.10,
                QuickRecallFactor.INFORMATION: 0.15,
                QuickRecallFactor.RELEVANCE: 0.05,
                QuickRecallFactor.USER_ATTENTION: 0.00,
                QuickRecallFactor.R_GEOMETRY: 0.15,
                QuickRecallFactor.CAUSAL_NOVELTY: 0.15,
                QuickRecallFactor.SELF_ORG: 0.00,
                QuickRecallFactor.OVERLAP: 0.00
            },
            QuickRecallMode.PERSONAL: {
                QuickRecallFactor.SURPRISE: 0.05,
                QuickRecallFactor.DIVERSITY: 0.05,
                QuickRecallFactor.EMOTION: 0.10,
                QuickRecallFactor.RECENCY: 0.05,
                QuickRecallFactor.IMPORTANCE: 0.05,
                QuickRecallFactor.PERSONAL: 0.35,
                QuickRecallFactor.COHERENCE: 0.00,
                QuickRecallFactor.INFORMATION: 0.00,
                QuickRecallFactor.RELEVANCE: 0.05,
                QuickRecallFactor.USER_ATTENTION: 0.00,
                QuickRecallFactor.R_GEOMETRY: 0.15,
                QuickRecallFactor.CAUSAL_NOVELTY: 0.10,
                QuickRecallFactor.SELF_ORG: 0.05,
                QuickRecallFactor.OVERLAP: 0.00
            },
            QuickRecallMode.HPC_QR: {
                QuickRecallFactor.RELEVANCE: 0.4,
                QuickRecallFactor.EMOTION: 0.25,
                QuickRecallFactor.SURPRISE: 0.15,
                QuickRecallFactor.DIVERSITY: 0.1,
                QuickRecallFactor.RECENCY: 0.1,
                QuickRecallFactor.R_GEOMETRY: 0.0,
                QuickRecallFactor.CAUSAL_NOVELTY: 0.0,
                QuickRecallFactor.SELF_ORG: 0.0,
                QuickRecallFactor.OVERLAP: 0.0
            }
        }

        mode = self.config['mode']
        if isinstance(mode, str):
            try:
                mode = QuickRecallMode(mode.lower())
            except ValueError:
                mode = QuickRecallMode.STANDARD

        # Base is standard
        base_weights = standard_weights.copy()
        # Overwrite with mode-specific if we have it
        if mode in mode_weights:
            base_weights.update(mode_weights[mode])

        # If "CUSTOM" and user-supplied factor_weights, override those
        if mode == QuickRecallMode.CUSTOM and self.config['factor_weights']:
            base_weights.update(self.config['factor_weights'])

        # Store final HPC-QR factor weights
        self.factor_weights = base_weights

        # Normalize them so that sum of factors (except OVERLAP) <= 1.0
        total_weight = sum(w for f, w in self.factor_weights.items() if f != QuickRecallFactor.OVERLAP)
        if total_weight > 0:
            for fct in self.factor_weights:
                if fct != QuickRecallFactor.OVERLAP:
                    self.factor_weights[fct] = self.factor_weights[fct] / total_weight

    def _init_som(self) -> None:
        """Initialize Self-Organizing Map."""
        grid_size = self.config['som_grid_size']
        dim = self.config['embedding_dim']

        # Create grid of neurons
        self.som_grid = np.random.randn(grid_size[0], grid_size[1], dim) * 0.1

        # Initialize learning parameters
        self.som_learning_rate = self.config['som_learning_rate']
        self.som_sigma = self.config['som_sigma']
        self.som_iterations = 0

        # Keep track of activations
        self.som_activations = np.zeros(grid_size)

        logger.info(f"Initialized SOM with grid size {grid_size} and embedding dim {dim}")

    async def calculate(
        self, 
        embedding_or_text: Union[str, np.ndarray, torch.Tensor], 
        text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate the composite significance/QR score for a memory.
        
        Args:
            embedding_or_text: Either a string or an embedding vector
            text: Optional original text, if embedding_or_text is an embedding
            context: Optional context information to help with scoring
            
        Returns:
            A normalized score between 0 and 1 that represents significance
        """
        context = context or {}
        
        # If embedding_or_text is an embedding and text is provided, add text to context
        if not isinstance(embedding_or_text, str) and text is not None:
            context['text'] = text
            
        # Determine calculation mode
        mode = self.config['mode']
        
        # Calculate based on mode
        if mode == QuickRecallMode.MINIMAL:
            result = await self._calculate_minimal(embedding_or_text, context)
        elif mode == QuickRecallMode.CLASSIC_QR:
            result = await self._calculate_classic_qr(embedding_or_text, context)
        else:
            # Use HPC-QR for all other modes
            result = await self._calculate_hpc_qr(embedding_or_text, context)
        
        # Periodically log score distribution for analysis (every 100 calculations)
        if (self.total_calculations % 100 == 0 and self.total_calculations > 0 and 
                len(self.history['calculated_qr']) >= 50):
            self.log_score_distribution()
            
        return result

    def _calculate_minimal(self, 
                           embedding_or_text: Union[str, np.ndarray, torch.Tensor], 
                           context: Dict[str, Any]) -> float:
        """
        Simple HPC-QR calculation mode, focusing primarily on recency (time decay)
        and a couple of quick text-based signals (personal, importance).
        """
        # Recency factor
        recency = self._calculate_recency(context)

        # If text is provided, check personal + importance
        if isinstance(embedding_or_text, str):
            text = embedding_or_text
            personal = self._calculate_personal(text, context)
            importance = self._calculate_importance(text, context)
            score = (recency + personal + importance) / 3.0
        else:
            score = recency

        # Optionally multiply by time decay factor
        if 'timestamp' in context:
            time_decay = self._calculate_time_decay(context)
            score *= time_decay

        # Clamp
        return min(1.0, max(0.0, score))

    def _calculate_classic_qr(self, 
                              embedding_or_text: Union[str, np.ndarray, torch.Tensor], 
                              context: Dict[str, Any]) -> float:
        """
        Classic QuickRecall approach from older significance-based logic:
        - Weighted combination of recency, surprise, diversity, importance.
        """
        recency = self._calculate_recency(context)
        importance = 0.0
        if isinstance(embedding_or_text, str):
            text = embedding_or_text
            importance = self._calculate_importance(text, context)

        # If we have an embedding, do quick surprise/diversity
        surprise_val = 0.0
        diversity_val = 0.0
        if not isinstance(embedding_or_text, str):
            emb = self._normalize_embedding(embedding_or_text)
            surprise_val = self._calculate_surprise_sync(emb, context)
            diversity_val = self._calculate_diversity_sync(emb, context)

        # Simple weighting
        # Adjust these as desired to emulate your "classic" distribution
        w_recency = 0.3
        w_surprise = 0.25
        w_diversity = 0.25
        w_importance = 0.2

        score = (w_recency * recency) + \
                (w_surprise * surprise_val) + \
                (w_diversity * diversity_val) + \
                (w_importance * importance)

        # Optionally apply time decay
        if 'timestamp' in context:
            time_decay = self._calculate_time_decay(context)
            score *= time_decay

        return min(1.0, max(0.0, score))

    def _calculate_surprise_sync(self, 
                                 emb: np.ndarray, 
                                 context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate surprise by comparing to recent entries.
        Synchronous version that handles dimension mismatches.
        """
        try:
            if self.external_momentum is None or len(self.external_momentum) == 0:
                return 0.5

            # Get the expected dimension from our config
            expected_dim = self.config['embedding_dim']
            actual_dim = emb.shape[0]
            
            sims = []
            for emb_mem in self.external_momentum:
                # Handle dimension mismatches when comparing
                emb_mem_dim = emb_mem.shape[0] if isinstance(emb_mem, np.ndarray) else emb_mem.size
                
                # If dimensions don't match, adjust for comparison
                if emb_mem_dim != actual_dim:
                    # Make copies to avoid modifying originals
                    emb_copy = emb.copy()
                    if isinstance(emb_mem, torch.Tensor):
                        emb_mem_copy = emb_mem.cpu().detach().numpy().copy()
                    else:
                        emb_mem_copy = emb_mem.copy()
                    
                    # Resize to smaller dimension for comparison
                    min_dim = min(emb_mem_dim, actual_dim)
                    emb_copy = emb_copy[:min_dim]
                    emb_mem_copy = emb_mem_copy[:min_dim]
                    
                    # Calculate similarity with adjusted dimensions
                    sim = np.dot(emb_copy, emb_mem_copy)
                else:
                    # Standard similarity calculation when dimensions match
                    if isinstance(emb_mem, torch.Tensor):
                        sim = float(torch.dot(torch.from_numpy(emb), emb_mem.cpu()))
                    else:
                        sim = np.dot(emb, emb_mem)
                sims.append(sim)

            # Take the maximum similarity as a measure of redundancy (opposite of surprise)
            max_sim = max(sims) if sims else 0.0
            # Calculate surprise as inverse of similarity
            surprise = 1.0 - max_sim
            
            # Apply threshold for "novel enough"
            novelty_threshold = self.config.get('novelty_threshold', 0.3)
            if surprise > novelty_threshold:
                # Scale up surprises above threshold to enhance novel information
                normalized_surprise = min(1.0, 0.5 + (surprise - novelty_threshold) * 2.0)
            else:
                # Scale down low surprise values
                normalized_surprise = min(0.5, surprise * (0.5 / novelty_threshold))
                
            return float(normalized_surprise)
            
        except Exception as e:
            logger.error(f"Error in surprise calculation: {e}")
            return 0.5

    def _calculate_diversity_sync(self, emb: np.ndarray, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate diversity by comparing to entire history of embeddings.
        Handles dimension mismatches between embeddings.
        """
        try:
            if self.external_momentum is None or len(self.external_momentum) < 5:
                return 0.5  # Default when not enough history
            
            # Get embedding dimensions
            actual_dim = emb.shape[0]
            
            # Calculate average similarity to all entries in external_momentum
            all_sims = []
            for mem_emb in self.external_momentum:
                # Convert pytorch tensor to numpy if needed
                if isinstance(mem_emb, torch.Tensor):
                    mem_emb = mem_emb.detach().cpu().numpy()
                
                # Handle dimension mismatch
                mem_dim = mem_emb.shape[0]
                if mem_dim != actual_dim:
                    # Make a copy of both embeddings to adjust dimensions
                    emb_copy = emb.copy()
                    mem_emb_copy = mem_emb.copy()
                    
                    # Resize to common dimension for comparison
                    min_dim = min(mem_dim, actual_dim)
                    emb_copy = emb_copy[:min_dim]
                    mem_emb_copy = mem_emb_copy[:min_dim]
                    
                    # Calculate similarity with resized embeddings
                    sim = np.dot(emb_copy, mem_emb_copy)
                else:
                    # Standard comparison when dimensions match
                    sim = np.dot(emb, mem_emb)
                
                all_sims.append(sim)
            
            if not all_sims:
                return 0.5
            
            # Average similarity (redundancy)
            avg_sim = np.mean(all_sims)
            
            # Convert to diversity score (inverse of redundancy)
            diversity_score = 1.0 - avg_sim
            
            # Apply scaling based on the size of buffer to encourage diversity
            # with larger history
            buffer_size = len(self.external_momentum)
            if buffer_size > 20:
                # Boost diversity slightly for large buffers
                diversity_score = min(1.0, diversity_score * 1.1)
            
            return float(min(1.0, max(0.0, diversity_score)))
        except Exception as e:
            logger.error(f"Error in diversity calculation: {e}")
            return 0.5

    async def _calculate_surprise(self, 
                                  emb: np.ndarray, 
                                  context: Optional[Dict[str, Any]] = None
                                 ) -> float:
        """
        Calculate how 'surprising' or novel the new embedding is, given recent momentum.
        Safely handles dimension mismatches.
        """
        try:
            # Ensure context is a dictionary
            context = context or {}

            # Prefer context-based external_momentum
            external = context.get('external_momentum', self.external_momentum)
            if external is None or (hasattr(external, '__len__') and len(external) < 1):
                return 0.5  # Default

            # We'll look at up to the last 3-5 embeddings
            recent = external[-3:] if hasattr(external, '__len__') and len(external) > 3 else external

            # If we have at least 2-3 embeddings, we can do better predictions
            if hasattr(recent, '__len__') and len(recent) >= 2:
                # Align the last two embeddings for linear prediction
                b, c, _ = self._align_vectors_for_comparison(recent[-1], recent[-2])
                predicted_direction = b - c  # Direction of momentum
                predicted_embedding = b + predicted_direction  # Extrapolated prediction

                # Normalize predicted embedding
                norm = np.linalg.norm(predicted_embedding)
                if norm > 1e-9:
                    predicted_embedding = predicted_embedding / norm

                # Align the current embedding with the predicted embedding
                aligned_emb, aligned_pred, _ = self._align_vectors_for_comparison(emb, predicted_embedding)
                
                # Calculate similarity between actual and predicted
                similarity = np.dot(aligned_emb, aligned_pred)
                # Convert similarity [-1,1] => [0,1]
                surprise_score = 1.0 - ((similarity + 1.0) / 2.0)
            else:
                # Fallback: direct similarity to nearest neighbor
                sims = []
                for mem_emb in recent:
                    aligned_a, aligned_b, _ = self._align_vectors_for_comparison(emb, mem_emb)
                    sims.append(np.dot(aligned_a, aligned_b))
                
                if not sims:
                    return 0.5
                max_sim = max(sims)
                surprise_score = 1.0 - ((max_sim + 1.0) / 2.0)
            
            # Apply threshold-based scaling
            novelty_threshold = self.config.get('novelty_threshold', 0.3)
            if surprise_score > novelty_threshold:
                # Scale up surprises above threshold
                return min(1.0, surprise_score * 1.5)
            else:
                # Scale down low surprise values
                return min(0.5, surprise_score)
                
        except Exception as e:
            logger.error(f"Error calculating surprise: {e}")
            return 0.5

    async def _calculate_diversity(self, 
                                   emb: np.ndarray, 
                                   context: Optional[Dict[str, Any]] = None
                                  ) -> float:
        """
        Calculate diversity factor, which is how 'different' this embedding is from external_momentum.
        Safely handles dimension mismatches.
        """
        try:
            # Ensure context is a dictionary
            context = context or {}
            
            ext_momentum = context.get('external_momentum', self.external_momentum)
            if ext_momentum is None or not hasattr(ext_momentum, '__len__') or len(ext_momentum) == 0:
                return 0.5

            # We'll look at up to the last 3-5 embeddings
            recent = ext_momentum[-3:] if hasattr(ext_momentum, '__len__') and len(ext_momentum) > 3 else ext_momentum

            # If we have at least 2-3 embeddings, we can do better predictions
            if hasattr(recent, '__len__') and len(recent) >= 2:
                # Align the last two embeddings for linear prediction
                b, c, _ = self._align_vectors_for_comparison(recent[-1], recent[-2])
                predicted_direction = b - c  # Direction of momentum
                predicted_embedding = b + predicted_direction  # Extrapolated prediction

                # Normalize predicted embedding
                norm = np.linalg.norm(predicted_embedding)
                if norm > 1e-9:
                    predicted_embedding = predicted_embedding / norm

                # Align the current embedding with the predicted embedding
                aligned_emb, aligned_pred, _ = self._align_vectors_for_comparison(emb, predicted_embedding)
                
                # Calculate similarity between actual and predicted
                similarity = np.dot(aligned_emb, aligned_pred)
                # Convert similarity [-1,1] => [0,1]
                diversity_score = 1.0 - ((similarity + 1.0) / 2.0)
            else:
                # Fallback: direct similarity to nearest neighbor
                sims = []
                for mem_emb in recent:
                    aligned_a, aligned_b, _ = self._align_vectors_for_comparison(emb, mem_emb)
                    sims.append(np.dot(aligned_a, aligned_b))
                
                if not sims:
                    return 0.5
                max_sim = max(sims)
                diversity_score = 1.0 - ((max_sim + 1.0) / 2.0)
            
            # Apply threshold-based scaling
            novelty_threshold = self.config.get('novelty_threshold', 0.3)
            if diversity_score > novelty_threshold:
                # Scale up diversity above threshold
                return min(1.0, diversity_score * 1.5)
            else:
                # Scale down low diversity values
                return min(0.5, diversity_score)
                
        except Exception as e:
            logger.error(f"Error in diversity calculation: {e}")
            return 0.5

    async def _calculate_emotion(self, 
                                 embedding_or_text: Union[str, np.ndarray, torch.Tensor], 
                                 context: Optional[Dict[str, Any]] = None
                                ) -> float:
        """
        Calculate emotion factor using text or embedding.
        Uses emotion-analyzer API if available, with fallback to simpler calculations.
        Handles dimension mismatches safely.
        """
        try:
            # If we have text, use it directly with emotion analyzer
            if isinstance(embedding_or_text, str) and embedding_or_text.strip():
                text = embedding_or_text
                emotion_score = await self._analyze_emotion_from_text(text, context)
                if emotion_score is not None:
                    return emotion_score

            # If we have an embedding (or if text analysis failed), use embedding-based method
            if not isinstance(embedding_or_text, str):
                # Normalize embedding to ensure correct dimensions
                embedding = self._normalize_embedding(embedding_or_text)
                
                # Try to get text from context if available
                text = context.get('text', '') if context else ''
                if text.strip():
                    # Try emotion analyzer first
                    emotion_score = await self._analyze_emotion_from_text(text, context)
                    if emotion_score is not None:
                        return emotion_score
                
                # Fallback: Use embedding-based emotion estimation
                return await self._calculate_emotion_from_embedding(embedding)
            
            # Default fallback
            return 0.5
            
        except Exception as e:
            logger.error(f"Error in emotion calculation: {e}")
            return 0.5

    async def _analyze_emotion_from_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        Analyze emotional content of text using the emotion-analyzer service.
        Returns a value between 0.0 and 1.0 indicating emotional intensity.
        Returns None if analysis fails or is not available.
        """
        try:
            import websockets
            import json
            import asyncio
            
            # Use the emotion-analyzer service running on port 5007 with WebSockets
            emotion_analyzer_url = "ws://localhost:5007/analyze_emotion"
            
            # Prepare request
            request_data = {"text": text}
            
            async with websockets.connect(emotion_analyzer_url) as websocket:
                # Send the request
                await websocket.send(json.dumps(request_data))
                
                # Wait for response with a timeout
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                result = json.loads(response)
                
                # Extract emotion score from the response
                emotion_score = result.get('emotion_score', 0.5)
                return float(emotion_score)
                        
        except Exception as e:
            logger.warning(f"Error connecting to emotion analyzer service: {e}")
            # Fallback default value
            return 0.5

    def _calculate_importance(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate the 'importance' factor from text and/or context.
        Could be refined via model-based approach, but here we use:
          - explicit context signals (e.g. 'importance_score')
          - presence of 'informational_prefixes' 
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        # Check explicit context
        explicit_importance = context.get('importance_score', 0.0)

        # Count how many 'informational_prefixes' appear
        count_prefix = sum(text_lower.count(pref.lower()) for pref in self.config['informational_prefixes'])
        # Scale factor
        prefix_factor = min(1.0, count_prefix / 3.0)

        # Combine them
        importance_score = max(explicit_importance, prefix_factor)
        return float(min(1.0, max(0.0, importance_score)))

    def _calculate_personal(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate the 'personal' factor based on presence of personal info or
        first-person references in text.
        """
        if not text:
            return 0.0

        text_lower = text.lower()

        # Check for personal info keywords
        personal_keywords = self.config['personal_information_keywords']
        personal_count = sum(1 for kw in personal_keywords if kw in text_lower)
        base_personal = min(1.0, personal_count / 2.0)

        # Check for pronouns: 'i ', ' my ', ' we ', etc.
        first_person_pronouns = ['i ', ' my ', ' me ', 'mine ', ' we ', ' our ', ' us ']
        pronoun_hits = sum(text_lower.count(p) for p in first_person_pronouns)
        pronoun_factor = min(1.0, pronoun_hits / 5.0)

        # Combine
        personal_score = max(base_personal, pronoun_factor)

        # Also check explicit signals from context
        explicit_personal = context.get('personal_significance', 0.0)
        personal_score = max(personal_score, explicit_personal)

        return float(min(1.0, max(0.0, personal_score)))

    def _calculate_information(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate 'information' factor based on text length and structural markers.
        """
        if not text:
            return 0.0

        tokens = text.split()
        length_factor = min(1.0, len(tokens) / 100.0)  # cap at 100 tokens

        # Check presence of bullet points or numeric data
        bullet_points = text.count('\n- ') + text.count('\n* ')
        bullet_factor = min(0.3, bullet_points / 5.0)

        import re
        numeric_count = len(re.findall(r'\d+', text))
        numeric_factor = min(0.3, numeric_count / 5.0)

        info_score = length_factor + bullet_factor + numeric_factor
        return float(min(1.0, info_score))

    def _calculate_coherence(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate coherence factor based on sentence structure and connectives.
        """
        if not text:
            return 0.5

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.5

        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        if len(lengths) > 1:
            variance = sum((l - avg_len)**2 for l in lengths) / len(lengths)
            std_dev = variance ** 0.5
            length_coherence = max(0.0, 1.0 - (std_dev / (avg_len+1e-9)))
        else:
            length_coherence = 0.5

        # Check for connectives
        connection_words = self.config['connection_words']
        text_lower = text.lower()
        connective_hits = sum(text_lower.count(cw) for cw in connection_words)
        connective_factor = min(1.0, connective_hits / 5.0)

        return float(0.5 * length_coherence + 0.5 * connective_factor)

    def _calculate_recency(self, context: Dict[str, Any]) -> float:
        """
        Calculate recency factor based on how old the memory is.
        Exponential decay by day.
        """
        timestamp = context.get('timestamp', time.time())
        elapsed = time.time() - timestamp
        elapsed_days = elapsed / 86400.0
        decay_rate = self.config['time_decay_rate']
        recency = np.exp(-decay_rate * elapsed_days)
        return float(recency)

    def _calculate_relevance(self, context: Dict[str, Any]) -> float:
        """
        Calculate relevance factor from explicit context (query_similarity, etc.).
        """
        if 'relevance' in context:
            return float(min(1.0, max(0.0, context['relevance'])))
        if 'query_similarity' in context:
            return float(min(1.0, max(0.0, context['query_similarity'])))
        return 0.5

    def _calculate_user_attention(self, context: Dict[str, Any]) -> float:
        """
        Calculate user_attention factor from user interaction signals.
        """
        interaction = context.get('user_interaction', {})
        dwell_time = interaction.get('dwell_time', 0.0)
        explicit_focus = interaction.get('explicit_focus', 0.0)
        # Simple combination: dwell_time scaled, then compare with explicit focus
        attention_score = max(min(1.0, dwell_time / 20.0), explicit_focus)
        return float(min(1.0, max(0.0, attention_score)))

    def _calculate_time_decay(self, context: Dict[str, Any]) -> float:
        """
        Exponential time decay factor, clamped at a minimum of 0.1.
        """
        timestamp = context.get('timestamp', time.time())
        elapsed_days = (time.time() - timestamp) / 86400.0
        decay_rate = self.config['time_decay_rate']
        decay_factor = np.exp(-decay_rate * elapsed_days)
        return float(max(0.1, decay_factor))

    async def _calculate_r_geometry(self, emb: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Calculate geometry-based factor by comparing 'emb' to the momentum center.
        Safely handles dimension mismatches.
        """
        try:
            # Ensure context is a dictionary
            context = context or {}
            
            ext_momentum = context.get('external_momentum', self.external_momentum)
            if ext_momentum is None or (hasattr(ext_momentum, '__len__') and len(ext_momentum) < 3):
                return 0.5

            # Use last 10 for center
            subset = ext_momentum[-10:] if hasattr(ext_momentum, '__len__') and len(ext_momentum) > 10 else ext_momentum
            aligned_list = []
            aligned_emb = None
            
            for mem_emb in subset:
                # Align each momentum vector with 'emb' so that all share the same dimension
                a, b, _ = self._align_vectors_for_comparison(mem_emb, emb)
                # We'll store the 'a' portion for the center calculation
                aligned_list.append(a)
                # Keep track of the aligned embedding (b)
                if aligned_emb is None:
                    aligned_emb = b

            # Now aligned_list are all the same dimension as 'aligned_emb'
            momentum_center = np.mean(aligned_list, axis=0)

            # Cosine similarity
            norm_emb = np.linalg.norm(aligned_emb)
            norm_ctr = np.linalg.norm(momentum_center)
            if norm_emb < 1e-9 or norm_ctr < 1e-9:
                return 0.5

            cos_sim = np.dot(aligned_emb, momentum_center) / (norm_emb * norm_ctr)
            # Convert [-1,1] => [0,1]
            geo_score = (cos_sim + 1.0) / 2.0

            return float(min(1.0, max(0.0, geo_score)))
        except Exception as e:
            logger.error(f"Error calculating r_geometry: {e}")
            return 0.5

    async def _calculate_causal_novelty(self, emb: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Calculate novelty based on similarity to recently observed embeddings for causal changes.
        Safely handles dimension mismatches.
        """
        try:
            # Ensure context is a dictionary
            context = context or {}
            
            if self.external_momentum is None or (hasattr(self.external_momentum, '__len__') and len(self.external_momentum) == 0):
                return self.config['default_novelty']

            # Prepare momentum data for iteration
            if hasattr(self.external_momentum, '__len__'):
                # It's a list or array with length
                window_size = min(10, len(self.external_momentum))
                subset = self.external_momentum[-window_size:]
            else:
                # It's a single item without length
                subset = [self.external_momentum]

            sims = []
            for mem_emb in subset:
                aligned_a, aligned_b, _ = self._align_vectors_for_comparison(emb, mem_emb)
                norm_a = np.linalg.norm(aligned_a)
                norm_b = np.linalg.norm(aligned_b)
                
                if norm_a > 0 and norm_b > 0:
                    cos_sim = np.dot(aligned_a, aligned_b) / (norm_a * norm_b)
                    sims.append(cos_sim)
                else:
                    sims.append(0.0)
            
            if not sims:
                return self.config['default_novelty']

            avg_sim = np.mean(sims)
            novelty = 1.0 - avg_sim
            return float(min(1.0, max(0.0, novelty)))
        except Exception as e:
            logger.error(f"Error calculating causal novelty: {e}")
            return self.config.get('default_novelty', 0.5)

    async def _calculate_self_organization(self, emb: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Calculate how much the new embedding might disrupt or re-organize 
        the existing momentum pattern.
        """
        try:
            # Ensure context is a dictionary
            context = context or {}
            
            if self.external_momentum is None or not hasattr(self.external_momentum, '__len__') or len(self.external_momentum) == 0:
                return 0.5
                
            # Process the momentum data based on its type
            momentum_data = self.external_momentum
            
            # Calculate momentum center
            if isinstance(momentum_data, torch.Tensor):
                momentum_center = momentum_data.detach().cpu().numpy()
                if len(momentum_center.shape) > 1:
                    # Handle multi-dimensional tensor
                    momentum_center = np.mean(momentum_center, axis=0)
            elif isinstance(momentum_data, list):
                # Handle list of embeddings
                if len(momentum_data) == 1:
                    momentum_center = momentum_data[0]
                else:
                    # Convert list elements to numpy arrays if needed
                    numpy_arrays = []
                    for item in momentum_data:
                        if isinstance(item, torch.Tensor):
                            numpy_arrays.append(item.detach().cpu().numpy())
                        else:
                            numpy_arrays.append(item)
                    
                    # Find a common dimension for all arrays
                    common_dim = min(arr.shape[0] for arr in numpy_arrays if hasattr(arr, 'shape'))
                    
                    # Align all arrays to common dimension
                    aligned_arrays = []
                    for arr in numpy_arrays:
                        if arr.shape[0] > common_dim:
                            aligned_arrays.append(arr[:common_dim])
                        elif arr.shape[0] < common_dim:
                            padded = np.zeros(common_dim)
                            padded[:arr.shape[0]] = arr
                            aligned_arrays.append(padded)
                        else:
                            aligned_arrays.append(arr)
                    
                    # Calculate mean of aligned arrays
                    momentum_center = np.mean(aligned_arrays, axis=0)
            else:
                # Handle direct numpy array
                momentum_center = momentum_data
            
            # Align dimensions for comparison
            aligned_emb, aligned_center, _ = self._align_vectors_for_comparison(emb, momentum_center)
            
            distance = np.linalg.norm(aligned_emb - aligned_center)
            
            # Convert to self-org score: higher distance => higher disruption => higher S-org
            return float(min(1.0, distance / 2.0))
        except Exception as e:
            logger.error(f"Error calculating self-organization: {e}")
            return 0.5

    async def _calculate_overlap(self, emb: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Calculate redundancy/overlap factor. Higher => more similar to existing data => 
        less 'novel' in HPC-QR sense.
        Safely handles dimension mismatches.
        """
        try:
            # Ensure context is a dictionary
            context = context or {}
            
            # Check if external_momentum exists and has items
            if self.external_momentum is None or not hasattr(self.external_momentum, '__len__') or len(self.external_momentum) == 0:
                return self.config.get('default_overlap', 0.3)
            
            # Prepare the momentum data properly based on type
            momentum_data = self.external_momentum
            
            # Calculate best similarity across all embeddings in momentum
            best_sim = -1.0
            for mem_emb in momentum_data:
                # Align vectors for comparison to handle dimension mismatches
                aligned_emb, aligned_mem, _ = self._align_vectors_for_comparison(emb, mem_emb)
                
                # Ensure vectors have non-zero norms
                norm1 = np.linalg.norm(aligned_emb)
                norm2 = np.linalg.norm(aligned_mem)
                
                if norm1 < 1e-9 or norm2 < 1e-9:
                    continue
                    
                # Calculate cosine similarity with aligned vectors
                cos_sim = np.dot(aligned_emb, aligned_mem) / (norm1 * norm2)
                if cos_sim > best_sim:
                    best_sim = cos_sim

            if best_sim < 0.0:
                return 0.0
            return float(min(1.0, max(0.0, best_sim)))

        except Exception as e:
            logger.error(f"Error calculating overlap: {e}")
            return self.config.get('default_overlap', 0.3)

    async def _som_disruption(self, emb: np.ndarray) -> float:
        """
        If we want a fallback measure of disruption using the SOM:
        - Find the BMU distance from the average or from a typical node.
        """
        try:
            bmu_row, bmu_col = self._find_som_bmu(emb)
            bmu_vector = self.som_grid[bmu_row, bmu_col]
            
            # Align dimensions for comparison
            aligned_emb, aligned_bmu, _ = self._align_vectors_for_comparison(emb, bmu_vector)
            
            dist = np.linalg.norm(aligned_emb - aligned_bmu)
            
            # Convert to self-org score: higher distance => higher disruption => higher S-org
            return float(min(1.0, dist / 2.0))
        except Exception as e:
            logger.error(f"SOM disruption error: {e}")
            return 0.5

    async def _update_context_with_external_momentum(self, context: Dict[str, Any]) -> None:
        """
        Update context with external momentum buffer if available
        """
        if 'external_momentum' not in context and self.external_momentum is not None:
            try:
                # Copy external momentum for calculations
                context['external_momentum'] = self.external_momentum
                
                # Log any dimension mismatches
                if len(self.external_momentum) > 0:
                    if isinstance(self.external_momentum[0], np.ndarray):
                        dim = self.external_momentum[0].shape[0]
                        expected_dim = self.config.get('embedding_dim', dim)
                        
                        if dim != expected_dim:
                            logger.warning(f"External momentum dimension ({dim}) doesn't match calculator embedding_dim ({expected_dim})")
                            logger.info("Comparisons will be adjusted using padding/truncation")
                    elif isinstance(self.external_momentum[0], torch.Tensor):
                        dim = self.external_momentum[0].shape[0]
                        expected_dim = self.config.get('embedding_dim', dim)
                        
                        if dim != expected_dim:
                            logger.warning(f"External momentum tensor dimension ({dim}) doesn't match calculator embedding_dim ({expected_dim})")
                            logger.info("Comparisons will be adjusted using padding/truncation")
            except Exception as e:
                logger.warning(f"Error updating context with external momentum: {e}")

    def _find_som_bmu(self, emb: np.ndarray) -> Tuple[int,int]:
        """
        Find the Best Matching Unit (BMU) in the SOM grid for the embedding.
        """
        grid_shape = self.som_grid.shape
        reshaped = self.som_grid.reshape((-1, grid_shape[2]))
        dists = np.linalg.norm(reshaped - emb, axis=1)
        bmu_idx = np.argmin(dists)
        row = bmu_idx // grid_shape[1]
        col = bmu_idx % grid_shape[1]
        return (row, col)

    async def _update_som(self, emb: np.ndarray) -> None:
        """
        Update the SOM with the new embedding. (Basic incremental SOM update.)
        """
        self.som_iterations += 1
        max_iter = 10000  # for decay scheduling
        t = self.som_iterations

        lr_0 = self.som_learning_rate
        sigma_0 = self.som_sigma
        lr_t = lr_0 * np.exp(-t / max_iter)
        sigma_t = sigma_0 * np.exp(-t / max_iter)

        bmu_pos = self._find_som_bmu(emb)
        grid_size = self.som_grid.shape[:2]

        for r in range(grid_size[0]):
            for c in range(grid_size[1]):
                dist_sq = (r - bmu_pos[0])**2 + (c - bmu_pos[1])**2
                # Gaussian neighborhood
                if dist_sq < (sigma_t**2) * 4:  
                    influence = np.exp(-dist_sq / (2 * sigma_t**2))
                    self.som_grid[r, c] += lr_t * influence * (emb - self.som_grid[r, c])

    async def _update_causal_graph(self, concept_id: str, emb: np.ndarray, context: Dict[str, Any]) -> None:
        """
        Update the causal graph with new edges or weights based on the new embedding.
        This is a simplified placeholder approach.
        """
        for node in self.causal_graph.nodes:
            if node == concept_id:
                continue
            node_emb = self.causal_graph.nodes[node].get('embedding')
            if node_emb is None:
                continue

            node_emb = self._normalize_embedding(node_emb)
            sim = float(np.dot(emb, node_emb))

            if sim > self.config['causal_edge_threshold']:
                # Add or update edge
                if not self.causal_graph.has_edge(node, concept_id):
                    self.causal_graph.add_edge(node, concept_id, weight=sim)
                else:
                    old_weight = self.causal_graph[node][concept_id].get('weight', 0.5)
                    new_weight = (old_weight + sim) / 2.0
                    self.causal_graph[node][concept_id]['weight'] = new_weight

        # Store embedding on the concept node
        self.causal_graph.nodes[concept_id]['embedding'] = emb

    async def _update_prototype(self, category: str, emb: np.ndarray) -> None:
        """
        Update or create a category prototype vector.
        """
        if category not in self.prototypes:
            self.prototypes[category] = emb
            return

        old_proto = self.prototypes[category]
        new_proto = 0.9 * old_proto + 0.1 * emb
        self.prototypes[category] = self._normalize_embedding(new_proto)

    def _update_history(self, score: float, factor_values: Dict[QuickRecallFactor, float]) -> None:
        """
        Update HPC-QR score history for adaptive thresholding.
        """
        self.history['calculated_qr'].append(score)
        self.history['timestamps'].append(time.time())
        for fct, val in factor_values.items():
            if fct in self.history['factor_values']:
                self.history['factor_values'][fct].append(val)

        # Trim if over history_window
        hw = self.config['history_window']
        if len(self.history['calculated_qr']) > hw:
            self.history['calculated_qr'] = self.history['calculated_qr'][-hw:]
            self.history['timestamps'] = self.history['timestamps'][-hw:]
            for fct in self.history['factor_values']:
                if len(self.history['factor_values'][fct]) > hw:
                    self.history['factor_values'][fct] = self.history['factor_values'][fct][-hw:]

    def _sigmoid_scale(self, x: float) -> float:
        """
        Scale x via a sigmoid so that HPC-QR remains in [0, 1].
        """
        return float(expit(2.0 * (x - 0.5)))

    def _normalize_embedding(self, 
                             embedding: Union[np.ndarray, torch.Tensor, List[float]]
                            ) -> np.ndarray:
        """
        Normalize an embedding to the configured self.config['embedding_dim'].
        If actual_dim < expected_dim, we pad with zeros. If actual_dim > expected_dim, we truncate.
        Then we L2-normalize the final vector.
        """
        if embedding is None:
            logger.warning("Received None embedding in _normalize_embedding")
            return np.zeros(self.config['embedding_dim'], dtype=np.float32)

        # Convert list to np if needed
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        # Convert torch to numpy
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        # Flatten if needed
        if len(embedding.shape) > 1:
            embedding = embedding.flatten()

        expected_dim = self.config.get('embedding_dim', 768)
        actual_dim = embedding.shape[0]

        if actual_dim < expected_dim:
            # Pad
            pad_len = expected_dim - actual_dim
            embedding = np.concatenate([embedding, np.zeros(pad_len, dtype=np.float32)])
            logger.debug(f"Padded embedding from {actual_dim} to {expected_dim}")
        elif actual_dim > expected_dim:
            # Truncate
            embedding = embedding[:expected_dim]
            logger.debug(f"Truncated embedding from {actual_dim} to {expected_dim}")

        # L2-normalize
        norm_val = np.linalg.norm(embedding)
        if norm_val < 1e-9:
            # If zero or near-zero
            logger.warning("Zero-norm embedding, substituting random unit vector")
            random_vec = np.random.randn(expected_dim).astype(np.float32)
            embedding = random_vec / np.linalg.norm(random_vec)
        else:
            embedding = embedding / norm_val

        return embedding

    def _align_vectors_for_comparison(
        self, 
        vec_a: Union[np.ndarray, torch.Tensor], 
        vec_b: Union[np.ndarray, torch.Tensor],
        log_warnings: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Safely align two vectors to the same dimension for comparison operations.
        Returns the aligned vectors and their common dimension.
        
        Args:
            vec_a: First vector (numpy array or torch.Tensor)
            vec_b: Second vector (numpy array or torch.Tensor)
            log_warnings: Whether to log dimension mismatch warnings
        
        Returns:
            (aligned_vec_a, aligned_vec_b, common_dim):
                aligned_vec_a -> numpy array with shape [common_dim]
                aligned_vec_b -> numpy array with shape [common_dim]
                common_dim -> int dimension used for alignment
        """
        # Convert torch tensors to numpy arrays if needed
        if isinstance(vec_a, torch.Tensor):
            vec_a = vec_a.detach().cpu().numpy()
        if isinstance(vec_b, torch.Tensor):
            vec_b = vec_b.detach().cpu().numpy()
            
        # Ensure they are 1D arrays
        vec_a = np.ravel(vec_a)
        vec_b = np.ravel(vec_b)
        
        dim_a = vec_a.shape[0]
        dim_b = vec_b.shape[0]
        
        # If dimensions match, return as is
        if dim_a == dim_b:
            return vec_a, vec_b, dim_a
            
        # Dimensions don't match, need to align
        expected_dim = self.config.get('embedding_dim', max(dim_a, dim_b))
        
        # Control logging frequency to avoid excessive warnings
        should_log = log_warnings and self.dim_mismatch_warnings < self.max_dim_mismatch_warnings
        
        # Log warning about dimension mismatch (only for the first few occurrences)
        if should_log:
            self.dim_mismatch_warnings += 1
            
            if dim_a != expected_dim:
                logger.warning(f"Vector A dimension ({dim_a}) doesn't match expected dimension ({expected_dim})")
            
            if dim_b != expected_dim:
                logger.warning(f"Vector B dimension ({dim_b}) doesn't match expected dimension ({expected_dim})")
                
            # Only log adjustment info message for first few warnings
            logger.info("Comparisons will be adjusted using padding/truncation")
            
            # Log a message when reaching the warning limit
            if self.dim_mismatch_warnings >= self.max_dim_mismatch_warnings and not self.dim_mismatch_logged:
                logger.warning(f"Suppressing further dimension mismatch warnings after {self.max_dim_mismatch_warnings} occurrences")
                self.dim_mismatch_logged = True
        
        # Find common dimension to use (can be different logic based on preferences)
        # Option 1: Use the minimum dimension
        common_dim = min(dim_a, dim_b)
        
        # Option 2: Use the calculator's configured embedding_dim if specified
        if 'embedding_dim' in self.config:
            # If both dimensions are smaller than config, use the larger of the two
            if dim_a < self.config['embedding_dim'] and dim_b < self.config['embedding_dim']:
                common_dim = max(dim_a, dim_b)
            # Otherwise use the configured dimension
            else:
                common_dim = self.config['embedding_dim']
        
        # Align vec_a to common_dim
        if dim_a > common_dim:
            # Truncate
            aligned_a = vec_a[:common_dim]
        elif dim_a < common_dim:
            # Pad with zeros
            aligned_a = np.zeros(common_dim)
            aligned_a[:dim_a] = vec_a
        else:
            aligned_a = vec_a
            
        # Align vec_b to common_dim
        if dim_b > common_dim:
            # Truncate
            aligned_b = vec_b[:common_dim]
        elif dim_b < common_dim:
            # Pad with zeros
            aligned_b = np.zeros(common_dim)
            aligned_b[:dim_b] = vec_b
        else:
            aligned_b = vec_b
            
        return aligned_a, aligned_b, common_dim

    async def _calculate_hpc_qr(
        self, 
        embedding_or_text: Union[str, np.ndarray, torch.Tensor], 
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate the HPC-QR (High Performance Computing Quick Recall) score.
        This is the most comprehensive QR calculation mode, using advanced 
        geometric, causal, and self-organizing metrics plus text/context factors.
        """
        start_time = time.time()
        context = context or {}

        try:
            # Convert embedding if needed
            if isinstance(embedding_or_text, torch.Tensor):
                # Keep device info if relevant
                if 'device' not in context and hasattr(embedding_or_text, 'device'):
                    context['device'] = str(embedding_or_text.device)
                embedding = embedding_or_text.detach().cpu().numpy()
            elif isinstance(embedding_or_text, str):
                text = embedding_or_text
                embedding = None
            else:
                embedding = embedding_or_text

            emb_array = self._normalize_embedding(embedding) if embedding is not None else None

            # Validate embedding
            emb_array, is_penalty, qr_cap = self.validate_embedding(emb_array)

            # Make sure context references external momentum, if available
            await self._update_context_with_external_momentum(context)

            factor_values = {}

            # Text-based factors
            if isinstance(embedding_or_text, str):
                text = embedding_or_text
                factor_values[QuickRecallFactor.EMOTION] = await self._calculate_emotion(text, context)
                factor_values[QuickRecallFactor.IMPORTANCE] = self._calculate_importance(text, context)
                factor_values[QuickRecallFactor.PERSONAL] = self._calculate_personal(text, context)
                factor_values[QuickRecallFactor.INFORMATION] = self._calculate_information(text, context)
                factor_values[QuickRecallFactor.COHERENCE] = self._calculate_coherence(text, context)
            else:
                factor_values[QuickRecallFactor.EMOTION] = 0.0
                factor_values[QuickRecallFactor.IMPORTANCE] = 0.0
                factor_values[QuickRecallFactor.PERSONAL] = 0.0
                factor_values[QuickRecallFactor.INFORMATION] = 0.0
                factor_values[QuickRecallFactor.COHERENCE] = 0.0

            # Embedding-based novelty factors
            if emb_array is not None:
                factor_values[QuickRecallFactor.SURPRISE] = await self._calculate_surprise(emb_array, context)
                factor_values[QuickRecallFactor.DIVERSITY] = await self._calculate_diversity(emb_array, context)
            else:
                factor_values[QuickRecallFactor.SURPRISE] = 0.0
                factor_values[QuickRecallFactor.DIVERSITY] = 0.0

            # Context-based factors
            factor_values[QuickRecallFactor.RECENCY] = self._calculate_recency(context)
            factor_values[QuickRecallFactor.RELEVANCE] = self._calculate_relevance(context)
            factor_values[QuickRecallFactor.USER_ATTENTION] = self._calculate_user_attention(context)

            # HPC-QR-specific geometry / causal / self-org / overlap
            if emb_array is not None:
                factor_values[QuickRecallFactor.R_GEOMETRY] = await self._calculate_r_geometry(emb_array, context)
                factor_values[QuickRecallFactor.CAUSAL_NOVELTY] = await self._calculate_causal_novelty(emb_array, context)
                factor_values[QuickRecallFactor.SELF_ORG] = await self._calculate_self_organization(emb_array, context)
                factor_values[QuickRecallFactor.OVERLAP] = await self._calculate_overlap(emb_array, context)
            else:
                factor_values[QuickRecallFactor.R_GEOMETRY] = 0.0
                factor_values[QuickRecallFactor.CAUSAL_NOVELTY] = 0.0
                factor_values[QuickRecallFactor.SELF_ORG] = 0.0
                factor_values[QuickRecallFactor.OVERLAP] = 0.0

            # Weighted sum approach, or fallback HPC-QR formula
            weighted_sum = 0.0
            weights_sum = 0.0
            debug_weights = {}

            for factor, value in factor_values.items():
                weight = self.factor_weights.get(factor, 0.0)
                weighted_sum += weight * value
                weights_sum += weight
                debug_weights[factor.value] = (weight, value, weight * value)

            if weights_sum < 1e-9:
                # Fallback HPC-QR formula
                hpc_qr_score = (
                    self.config.get('alpha', 0.4) * factor_values[QuickRecallFactor.R_GEOMETRY] +
                    self.config.get('beta', 0.3) * factor_values[QuickRecallFactor.CAUSAL_NOVELTY] +
                    self.config.get('gamma', 0.2) * factor_values[QuickRecallFactor.SELF_ORG] -
                    self.config.get('delta', 0.1) * factor_values[QuickRecallFactor.OVERLAP]
                )
            else:
                hpc_qr_score = weighted_sum / weights_sum

            # Apply time decay
            if 'timestamp' in context:
                time_decay = self._calculate_time_decay(context)
                hpc_qr_score *= time_decay

            # If context has 'adjustment_factor', apply
            if 'adjustment_factor' in context:
                hpc_qr_score *= context['adjustment_factor']

            # Get penalty info from context if available
            is_penalty = context.get('is_penalty', False)
            qr_cap = context.get('qr_cap', 1.0)

            # Apply penalty for malformed embeddings
            if is_penalty:
                hpc_qr_score = min(hpc_qr_score, qr_cap)

            # Clamp
            hpc_qr_score = min(1.0, max(0.0, hpc_qr_score))

            # Apply Z-score normalization if enabled
            if self.config.get('use_zscore_normalization', False):
                hpc_qr_score = self._normalize_score_with_zscore(hpc_qr_score)

            # Update history for adaptive thresholds
            if self.config.get('adaptive_thresholds', False):
                self._update_history(hpc_qr_score, factor_values)

            self.total_calculations += 1
            self.last_calculation_time = time.time()
            calc_time = (self.last_calculation_time - start_time) * 1000

            if self.config.get('debug', False):
                logger.debug(f"HPC-QR Score: {hpc_qr_score:.4f} (calculated in {calc_time:.2f} ms, weights: {debug_weights})")
            else:
                logger.debug(f"HPC-QR Score: {hpc_qr_score:.4f} (calculated in {calc_time:.2f} ms)")

            # Update internal models if not in evaluation_only
            if emb_array is not None and not context.get('evaluation_only', False):
                try:
                    await self._update_som(emb_array)
                    if 'concept_id' in context:
                        await self._update_causal_graph(context['concept_id'], emb_array, context)
                    if 'category' in context:
                        await self._update_prototype(context['category'], emb_array)
                except Exception as e:
                    logger.warning(f"Error updating models (non-critical): {e}")

            return float(hpc_qr_score)

        except Exception as e:
            logger.error(f"Error calculating HPC-QR: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.5

    def log_score_distribution(self) -> None:
        """
        Log a histogram of the recent QR score distribution for analysis.
        
        This is useful for visualizing the impact of normalization and weight adjustments.
        """
        if not self.history['calculated_qr']:
            logger.info("No QR scores available for distribution analysis.")
            return
            
        recent_scores = self.history['calculated_qr'][-self.config.get('z_score_window_size', 100):]
        if len(recent_scores) < 10:
            logger.info(f"Not enough scores for distribution analysis. Only {len(recent_scores)} available.")
            return
            
        # Simple text-based histogram
        try:
            bins = 10
            hist, bin_edges = np.histogram(recent_scores, bins=bins, range=(0, 1))
            max_count = max(hist)
            scale = 40 / max_count if max_count > 0 else 1  # Scale for terminal width
            
            logger.info(f"\nQR Score Distribution (last {len(recent_scores)} scores):")
            logger.info("-" * 60)
            
            for i in range(bins):
                bar_len = int(hist[i] * scale)
                bar = "█" * bar_len
                logger.info(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}: {bar} ({hist[i]})")
                
            logger.info("-" * 60)
            logger.info(f"Mean: {np.mean(recent_scores):.4f}, StdDev: {np.std(recent_scores):.4f}")
            logger.info(f"Min: {min(recent_scores):.4f}, Max: {max(recent_scores):.4f}")
            logger.info(f"25th: {np.percentile(recent_scores, 25):.4f}, "
                       f"50th: {np.percentile(recent_scores, 50):.4f}, "
                       f"75th: {np.percentile(recent_scores, 75):.4f}")
            logger.info("-" * 60)
        except Exception as e:
            logger.warning(f"Error generating score distribution histogram: {e}")

    def _normalize_score_with_zscore(self, score: float) -> float:
        """
        Normalize a score using Z-score normalization based on historical score distribution.
        
        This helps compress high-end scores and enhance sensitivity in the middle range.
        
        Args:
            score: The raw score to normalize
            
        Returns:
            Normalized score using Z-scoring and sigmoid transformation
        """
        # Collect the window of recent scores for normalization
        window_size = self.config.get('z_score_window_size', 100)
        recent_scores = self.history['calculated_qr'][-window_size:] if self.history['calculated_qr'] else [0.5]
        
        if len(recent_scores) < 5:  # Not enough data for meaningful Z-score
            return score
            
        # Calculate mean and standard deviation
        mean = np.mean(recent_scores)
        std = np.std(recent_scores)
        
        # Avoid division by zero
        if std < 1e-6:
            return score
            
        # Calculate Z-score
        z_score = (score - mean) / std
        
        # Apply sigmoid transformation to map back to [0, 1] range
        # Using a scaled sigmoid to ensure good utilization of the range
        normalized_score = 1.0 / (1.0 + np.exp(-z_score * 1.5))
        
        # Log the transformation details for debugging
        if self.config.get('debug', False):
            logger.debug(f"Z-score normalization: {score:.4f} -> Z={z_score:.4f} -> {normalized_score:.4f} "
                         f"(window mean={mean:.4f}, std={std:.4f})")
            
        return normalized_score

    def update_adaptive_thresholds(self) -> None:
        """
        Optionally adjust HPC-QR thresholds based on historical data (e.g., novelty_threshold).
        """
        surprise_vals = self.history['factor_values'][QuickRecallFactor.SURPRISE]
        if len(surprise_vals) > 10:
            new_thresh = float(np.percentile(surprise_vals, 70))
            self.config['novelty_threshold'] = new_thresh
            logger.debug(f"Adaptive novelty_threshold updated to {new_thresh:.2f}")

    def set_mode(self, mode: Union[str, QuickRecallMode]) -> None:
        """
        Dynamically switch HPC-QR mode.
        """
        if isinstance(mode, str):
            try:
                mode = QuickRecallMode(mode.lower())
            except ValueError:
                logger.warning(f"Invalid mode: {mode}, using STANDARD")
                mode = QuickRecallMode.STANDARD
        self.config['mode'] = mode
        self._init_factor_weights()
        logger.info(f"HPC-QR mode set to: {mode.value}")

    def set_external_momentum(self, momentum_buffer: Optional[Union[List[np.ndarray], np.ndarray, torch.Tensor]] = None) -> None:
        """
        Set a reference to an external momentum buffer.
        """
        self.external_momentum = momentum_buffer
        
        if momentum_buffer is not None:
            # Log some information about the momentum buffer
            if isinstance(momentum_buffer, list):
                logger.info(f"External momentum buffer set with {len(momentum_buffer)} entries")
                
                # Check first embedding dimension if available
                if len(momentum_buffer) > 0 and hasattr(momentum_buffer[0], 'shape'):
                    first_dim = momentum_buffer[0].shape[0]
                    expected_dim = self.config.get('embedding_dim', first_dim)
                    
                    if first_dim != expected_dim:
                        # Only log warning if under threshold
                        if self.dim_mismatch_warnings < self.max_dim_mismatch_warnings:
                            logger.warning(f"External momentum dimension ({first_dim}) doesn't match calculator embedding_dim ({expected_dim})")
                            self.dim_mismatch_warnings += 1
                            
                            # If this is the last warning we'll show, add a summary message
                            if self.dim_mismatch_warnings >= self.max_dim_mismatch_warnings and not self.dim_mismatch_logged:
                                logger.warning(f"Suppressing further dimension mismatch warnings after {self.max_dim_mismatch_warnings} occurrences")
                                self.dim_mismatch_logged = True
                        
                        # Always log that comparisons will be adjusted
                        logger.info("Comparisons will be adjusted using padding/truncation")
            elif isinstance(momentum_buffer, torch.Tensor):
                shape_str = "x".join(str(dim) for dim in momentum_buffer.shape)
                logger.info(f"External momentum tensor set with shape {shape_str}")
                
                # Check embedding dimension against calculator's expected dimension
                emb_dim = momentum_buffer.shape[-1] if len(momentum_buffer.shape) > 1 else momentum_buffer.shape[0]
                expected_dim = self.config.get('embedding_dim', emb_dim)
                
                if emb_dim != expected_dim:
                    # Only log warning if under threshold
                    if self.dim_mismatch_warnings < self.max_dim_mismatch_warnings:
                        logger.warning(f"External momentum tensor dimension ({emb_dim}) doesn't match calculator embedding_dim ({expected_dim})")
                        self.dim_mismatch_warnings += 1
                        
                        # If this is the last warning we'll show, add a summary message
                        if self.dim_mismatch_warnings >= self.max_dim_mismatch_warnings and not self.dim_mismatch_logged:
                            logger.warning(f"Suppressing further dimension mismatch warnings after {self.max_dim_mismatch_warnings} occurrences")
                            self.dim_mismatch_logged = True
                    
                    # Always log that comparisons will be adjusted
                    logger.info("Comparisons will be adjusted using padding/truncation")
            elif isinstance(momentum_buffer, np.ndarray):
                shape_str = "x".join(str(dim) for dim in momentum_buffer.shape)
                logger.info(f"External momentum array set with shape {shape_str}")
                
                # Check embedding dimension
                emb_dim = momentum_buffer.shape[-1] if len(momentum_buffer.shape) > 1 else momentum_buffer.shape[0]
                expected_dim = self.config.get('embedding_dim', emb_dim)
                
                if emb_dim != expected_dim:
                    # Only log warning if under threshold
                    if self.dim_mismatch_warnings < self.max_dim_mismatch_warnings:
                        logger.warning(f"External momentum dimension ({emb_dim}) doesn't match calculator embedding_dim ({expected_dim})")
                        self.dim_mismatch_warnings += 1
                        
                        # If this is the last warning we'll show, add a summary message
                        if self.dim_mismatch_warnings >= self.max_dim_mismatch_warnings and not self.dim_mismatch_logged:
                            logger.warning(f"Suppressing further dimension mismatch warnings after {self.max_dim_mismatch_warnings} occurrences")
                            self.dim_mismatch_logged = True
                    
                    # Always log that comparisons will be adjusted
                    logger.info("Comparisons will be adjusted using padding/truncation")
        else:
            logger.info("External momentum buffer cleared")

    def set_factor_weights_from_strings(self, weights: Dict[Union[str, QuickRecallFactor], float]) -> None:
        """
        Manually override HPC-QR factor weights using string keys. Automatically sets mode to CUSTOM.
        """
        custom = {}
        for key, val in weights.items():
            if isinstance(key, str):
                try:
                    fct_enum = QuickRecallFactor[key.upper()]
                    custom[fct_enum] = val
                except KeyError:
                    logger.warning(f"Unknown factor: {key}, ignoring...")
            else:
                custom[key] = val

        self.config['factor_weights'] = custom
        self.config['mode'] = QuickRecallMode.CUSTOM
        self._init_factor_weights()
        logger.info("Custom HPC-QR factor weights applied from strings.")

    def set_factor_weights(self, factor_weights: Dict[QuickRecallFactor, float]) -> None:
        """
        Set custom weights for HPC-QR factors.
        """
        if not factor_weights:
            return

        # Update the factor weights
        for factor, weight in factor_weights.items():
            if factor in self.factor_weights:
                self.factor_weights[factor] = weight

        # Normalize the weights
        total_weight = sum(w for f, w in self.factor_weights.items() if f != QuickRecallFactor.OVERLAP)
        if total_weight > 0:
            for fct in self.factor_weights:
                if fct != QuickRecallFactor.OVERLAP:
                    self.factor_weights[fct] = self.factor_weights[fct] / total_weight

        logger.info(f"Updated factor weights: {self.factor_weights}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve HPC-QR calculator statistics.
        """
        qr_vals = self.history['calculated_qr']
        avg_qr = float(np.mean(qr_vals)) if qr_vals else 0.0

        factor_stats = {}
        for fct in QuickRecallFactor:
            vals = self.history['factor_values'].get(fct, [])
            if vals:
                factor_stats[fct.value] = {
                    'average': float(np.mean(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'weight': self.factor_weights.get(fct, 0.0)
                }

        return {
            'mode': self.config['mode'].value,
            'total_calculations': self.total_calculations,
            'avg_qr_score': avg_qr,
            'novelty_threshold': self.config['novelty_threshold'],
            'time_decay_rate': self.config['time_decay_rate'],
            'adaptive_thresholds': self.config['adaptive_thresholds'],
            'history_size': len(qr_vals),
            'factors': factor_stats
        }

    def validate_embedding(self, embedding: Union[np.ndarray, torch.Tensor, List[float]]) -> Tuple[np.ndarray, bool, float]:
        """
        Validate an embedding before processing. Checks for malformed embeddings (None, 
        dimension=1, etc.) and returns a valid embedding for processing with a flag 
        indicating if it's a penalty vector.
        
        Args:
            embedding: The input embedding to validate
            
        Returns:
            Tuple containing:
            - validated_embedding: A valid embedding (original or penalty)
            - is_penalty: Flag indicating if a penalty vector was used
            - qr_cap: Maximum QR score this embedding should receive (1.0 for valid, lower for penalties)
        """
        expected_dim = self.config.get('embedding_dim', 768)
        
        # Default values for valid embedding
        is_penalty = False
        qr_cap = 1.0
        
        # Case 1: None embedding
        if embedding is None:
            logger.warning("Received None embedding - using penalty vector with QR cap of 0.1")
            penalty_vector = np.random.normal(0, 0.01, expected_dim).astype(np.float32)
            embedding = penalty_vector / max(np.linalg.norm(penalty_vector), 1e-9)
            is_penalty = True
            qr_cap = 0.1
            return embedding, is_penalty, qr_cap
        
        # Convert list to np if needed
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        # Convert torch to numpy
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        # Flatten if needed
        if len(embedding.shape) > 1:
            embedding = embedding.flatten()

        actual_dim = embedding.shape[0]

        # Case 2: Dimension of 1 (severe malformation)
        if actual_dim == 1:
            logger.warning(f"Malformed embedding detected (dim={actual_dim}) - using penalty vector with QR cap of 0.1")
            penalty_vector = np.random.normal(0, 0.01, expected_dim).astype(np.float32)
            embedding = penalty_vector / max(np.linalg.norm(penalty_vector), 1e-9)
            is_penalty = True
            qr_cap = 0.1
            return embedding, is_penalty, qr_cap
            
        # Case 3: Extremely small dimension (likely malformed)
        if actual_dim < 10 and expected_dim >= 384:
            logger.warning(f"Suspicious embedding dimension (dim={actual_dim}) - using penalty vector with QR cap of 0.3")
            penalty_vector = np.random.normal(0, 0.01, expected_dim).astype(np.float32) 
            embedding = penalty_vector / max(np.linalg.norm(penalty_vector), 1e-9)
            is_penalty = True
            qr_cap = 0.3
            return embedding, is_penalty, qr_cap
            
        # Case 4: Zero norm embedding
        norm_val = np.linalg.norm(embedding)
        if norm_val < 1e-9:
            logger.warning("Zero-norm embedding - using penalty vector with QR cap of 0.2")
            penalty_vector = np.random.normal(0, 0.01, expected_dim).astype(np.float32)
            embedding = penalty_vector / max(np.linalg.norm(penalty_vector), 1e-9)
            is_penalty = True
            qr_cap = 0.2
            return embedding, is_penalty, qr_cap
            
        # Case 5: NaN or Inf values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            logger.warning("Embedding contains NaN or Inf values - using penalty vector with QR cap of 0.1")
            penalty_vector = np.random.normal(0, 0.01, expected_dim).astype(np.float32)
            embedding = penalty_vector / max(np.linalg.norm(penalty_vector), 1e-9)
            is_penalty = True
            qr_cap = 0.1
            return embedding, is_penalty, qr_cap
            
        # All validation passed, return original embedding with normalization
        return embedding, is_penalty, qr_cap
