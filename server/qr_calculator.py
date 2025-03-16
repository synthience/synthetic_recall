"""
LUCID RECALL PROJECT
Unified Quick Recall Calculator 

Agent: Lucidia 1.1
Date: 05/03/25
Time: 4:43 PM EST

A standardized quick recall (HPC-QR) calculator for consistent memory importance
assessment across all memory system components.
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from enum import Enum
from scipy.spatial.distance import cdist
from scipy.special import expit, logsumexp
import networkx as nx

logger = logging.getLogger(__name__)

class QuickRecallMode(Enum):
    """Operating modes for HPC-QR calculation approach."""
    STANDARD = "standard"       # Balanced approach for general use
    PRECISE = "precise"         # More detailed HPC-QR for thorough analysis
    EFFICIENT = "efficient"     # Simplified HPC-QR for speed
    EMOTIONAL = "emotional"     # Prioritizes emotional signals
    INFORMATIONAL = "informational"  # Prioritizes info density
    PERSONAL = "personal"       # Prioritizes personal relevance
    CUSTOM = "custom"           # Uses custom HPC-QR factor weights

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
    R_GEOMETRY = "r_geometry"        # geometry-based distance (e.g., hyperbolic)
    CAUSAL_NOVELTY = "causal_novel"  # how surprising under a causal model
    SELF_ORG = "self_org"            # measure of self-organizing reconfiguration
    OVERLAP = "overlap"              # redundancy with existing memory

class GeometryType(Enum):
    """Types of geometry used for embedding space calculations."""
    EUCLIDEAN = "euclidean"    
    HYPERBOLIC = "hyperbolic"  
    SPHERICAL = "spherical"    
    MIXED = "mixed"            # Combination (if implementing piecewise curvature)

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
            'time_decay_rate': 0.1,
            'novelty_threshold': 0.7,
            'min_qr_score': 0.0,
            'max_qr_score': 1.0,
            'adaptive_thresholds': True,
            'history_window': 1000,
            
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
            'alpha': 0.4,  # geometry
            'beta': 0.3,   # causal novelty
            'gamma': 0.2,  # self-org
            'delta': 0.1,  # overlap penalty
            
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
            self.causal_weights = {(u, v): self.causal_graph[u][v].get('weight', 0.5) 
                                   for u, v in self.causal_graph.edges()}
        
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
            QuickRecallFactor.OVERLAP: 0.00  # Overlap is subtracted externally
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
        embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
        text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate the HPC-QR (Quick Recall) score for a memory.

        Args:
            embedding: Vector representation of memory content
            text: Text content of memory
            context: Additional contextual information

        Returns:
            HPC-QR score between [min_qr_score, max_qr_score]
        """
        start_time = time.time()
        context = context or {}

        try:
            # Process the embedding if provided
            if embedding is not None:
                emb_array = self._normalize_embedding(embedding)
            else:
                emb_array = None

            # 1) Calculate HPC-QR factors
            factor_values = {}

            # Text-based factors
            if text:
                factor_values[QuickRecallFactor.EMOTION] = self._calculate_emotion(text, context)
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

            # 2) Traditional weighted sum (excluding overlap)
            weighted_sum = 0.0
            for factor, value in factor_values.items():
                if factor == QuickRecallFactor.OVERLAP:
                    continue  # We'll handle overlap below
                w = self.factor_weights.get(factor, 0.0)
                weighted_sum += w * value

            # 3) HPC-QR formula: alpha*R_geom + beta*C_novel + gamma*S_org - delta*Overlap
            hpc_qr_core = (
                self.config['alpha'] * factor_values[QuickRecallFactor.R_GEOMETRY] +
                self.config['beta'] * factor_values[QuickRecallFactor.CAUSAL_NOVELTY] +
                self.config['gamma'] * factor_values[QuickRecallFactor.SELF_ORG] -
                self.config['delta'] * factor_values[QuickRecallFactor.OVERLAP]
            )

            # 4) Blend them
            combined_score = 0.6 * weighted_sum + 0.4 * hpc_qr_core

            # 5) Time decay
            time_decay = self._calculate_time_decay(context)
            final_score = combined_score * time_decay

            # 6) Sigmoid transform (0-1 range)
            final_score = self._sigmoid_scale(final_score)

            # 7) Clamp
            final_score = max(self.config['min_qr_score'], 
                              min(self.config['max_qr_score'], final_score))

            # 8) Update history
            if self.config['adaptive_thresholds']:
                self._update_history(final_score, factor_values)

            # Track stats
            self.total_calculations += 1
            self.last_calculation_time = time.time()
            calc_time = (self.last_calculation_time - start_time)*1000

            logger.debug(f"HPC-QR Score: {final_score:.4f} (calculated in {calc_time:.2f} ms)")

            # Update internal models (SOM, causal graph, prototypes) unless in eval-only mode
            if emb_array is not None and not context.get('evaluation_only', False):
                await self._update_som(emb_array)
                if 'concept_id' in context:
                    await self._update_causal_graph(context['concept_id'], emb_array, context)
                if 'category' in context:
                    await self._update_prototype(context['category'], emb_array)

            return float(final_score)

        except Exception as e:
            logger.error(f"Error calculating HPC-QR: {e}", exc_info=True)
            return 0.5  # fallback


    def _calculate_personal(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate the 'personal' factor based on presence of personal info or
        first-person references in text.
        
        Args:
            text: Text content
            context: Additional contextual info
        """
        text_lower = text.lower()

        # Check for personal info keywords
        personal_keywords = self.config['personal_information_keywords']
        personal_count = sum(1 for kw in personal_keywords if kw in text_lower)
        # Cap or scale
        base_personal = min(1.0, personal_count / 2.0)

        # Check for pronouns: 'i ', 'my ', 'our ', etc.
        first_person_pronouns = ['i ', ' my ', ' me ', 'mine ', ' we ', ' our ', ' us ']
        pronoun_hits = sum(text_lower.count(p) for p in first_person_pronouns)
        pronoun_factor = min(1.0, pronoun_hits / 5.0)

        # Combine
        personal_score = max(base_personal, pronoun_factor)
        
        # Also check explicit signals from context
        explicit_personal = context.get('personal_significance', 0.0)
        personal_score = max(personal_score, explicit_personal)

        return min(1.0, personal_score)

    def _calculate_information(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate information factor based on text length and structural markers.
        
        Args:
            text: The text content
            context: Additional context
        """
        tokens = text.split()
        length_factor = min(1.0, len(tokens) / 100.0)  # cap at 100 tokens

        # Check presence of numeric or bullet points
        bullet_points = text.count('\n- ') + text.count('\n* ')
        bullet_factor = min(0.3, bullet_points / 5.0)

        # Check for numeric data
        import re
        numeric_count = len(re.findall(r'\d+', text))
        numeric_factor = min(0.3, numeric_count / 5.0)

        info_score = length_factor + bullet_factor + numeric_factor
        return min(1.0, info_score)

    def _calculate_coherence(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate coherence factor based on sentence structure and connectives.
        
        Args:
            text: The text content
            context: Additional context
        """
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.5

        # Basic measure: average sentence length standard deviation
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
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

        return 0.5 * length_coherence + 0.5 * connective_factor

    def _calculate_recency(self, context: Dict[str, Any]) -> float:
        """
        Calculate recency factor based on how old the memory is.
        
        Args:
            context: Additional context with 'timestamp'
        """
        timestamp = context.get('timestamp', time.time())
        elapsed = time.time() - timestamp
        # Convert to days
        elapsed_days = elapsed / 86400.0

        # Exponential decay
        decay_rate = self.config['time_decay_rate']
        recency = np.exp(-decay_rate * elapsed_days)
        return float(recency)

    def _calculate_relevance(self, context: Dict[str, Any]) -> float:
        """
        Calculate relevance factor from explicit context (query_similarity, etc.).
        
        Args:
            context: Additional context
        """
        if 'relevance' in context:
            return min(1.0, max(0.0, context['relevance']))
        if 'query_similarity' in context:
            return min(1.0, max(0.0, context['query_similarity']))
        return 0.5

    def _calculate_user_attention(self, context: Dict[str, Any]) -> float:
        """
        Calculate user_attention factor from user interaction signals.
        
        Args:
            context: Additional context
        """
        interaction = context.get('user_interaction', {})
        dwell_time = interaction.get('dwell_time', 0.0)
        explicit_focus = interaction.get('explicit_focus', 0.0)
        # Simple combination
        attention_score = max( min(1.0, dwell_time / 20.0), explicit_focus )
        return attention_score

    async def _calculate_r_geometry(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Calculate geometry-based distance factor (R_GEOMETRY).
        E.g., hyperbolic distance from a 'center' or from a relevant prototype.
        """
        geom_type = self.config['geometry_type']
        if geom_type == GeometryType.EUCLIDEAN:
            return self._euclidean_geometry_factor(embedding, context)
        elif geom_type == GeometryType.HYPERBOLIC:
            return self._hyperbolic_geometry_factor(embedding, context)
        elif geom_type == GeometryType.SPHERICAL:
            return self._spherical_geometry_factor(embedding, context)
        else:  # MIXED or fallback
            # For demonstration, combine Euclidean + a small hyperbolic factor
            euc = self._euclidean_geometry_factor(embedding, context)
            hyp = self._hyperbolic_geometry_factor(embedding, context)
            return 0.5 * euc + 0.5 * hyp

    def _euclidean_geometry_factor(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Example: distance from some 'center' or a stored prototype.
        A higher distance => higher geometry factor.
        """
        center_emb = context.get('geometry_center')
        if center_emb is None:
            return 0.0

        center_emb = self._normalize_embedding(center_emb)
        dist = np.linalg.norm(embedding - center_emb)
        # Scale into [0,1] with a simple logistic
        return float(1.0 / (1.0 + np.exp(-dist + 2.0)))  # shift/scale as needed

    def _hyperbolic_geometry_factor(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Example hyperbolic distance using a negative curvature (K < 0).
        dK(u,v) = (2 / sqrt(|K|)) * asinh( sqrt(|K|) * ||u - v|| / 2 )
        """
        center_emb = context.get('geometry_center')
        if center_emb is None:
            return 0.0

        k = self.config['curvature']
        if k >= 0.0:
            k = -1.0  # force negative curvature if not set

        center_emb = self._normalize_embedding(center_emb)
        diff = np.linalg.norm(embedding - center_emb)
        abs_k = abs(k)
        dist = (2.0 / np.sqrt(abs_k)) * np.arcsinh((np.sqrt(abs_k) * diff) / 2.0)

        # Scale or clamp
        # You can convert it to [0,1] via e.g. 1 - exp(-dist)
        return float(1.0 - np.exp(-dist))

    def _spherical_geometry_factor(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Example spherical distance using positive curvature (K > 0).
        For simplicity, treat it similarly to measuring arc length on the unit sphere.
        """
        center_emb = context.get('geometry_center')
        if center_emb is None:
            return 0.0

        k = self.config['curvature']
        if k <= 0.0:
            k = 1.0  # default to 1.0 if not set

        center_emb = self._normalize_embedding(center_emb)
        dot_val = np.clip(np.dot(embedding, center_emb), -1.0, 1.0)
        angle = np.arccos(dot_val)  # angle in radians
        # arc length ~ radius * angle, radius = 1/sqrt(k)
        radius = 1.0 / np.sqrt(k)
        dist = radius * angle
        return float(1.0 - np.exp(-dist))

    async def _calculate_causal_novelty(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Evaluate how 'novel' the embedding is under a causal model (DAG).
        For demonstration, we do a placeholder that checks if the embedding
        strongly correlates with existing causal nodes.
        """
        if not self.causal_graph.nodes:
            return 0.0

        # In a real system, you'd do a do()-operation or measure surprise 
        # in a Bayesian network. We'll do a simplistic correlation check:
        node_embeddings = [
            (n, self.causal_graph.nodes[n].get('embedding', None))
            for n in self.causal_graph.nodes
        ]
        similarities = []
        for n_id, node_emb in node_embeddings:
            if node_emb is None:
                continue
            node_arr = self._normalize_embedding(node_emb)
            sim = np.dot(embedding, node_arr)
            similarities.append(sim)
        if not similarities:
            return 0.0

        # If the embedding is quite different from existing nodes, we treat it as novel
        avg_sim = np.mean(similarities)
        # scale to [0,1]
        novelty = max(0.0, 1.0 - avg_sim)
        return novelty

    async def _calculate_self_organization(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Measure how strongly the SOM (self-organizing map) might need to adapt
        to incorporate 'embedding'. 
        """
        # We'll measure distance to the BMU (best matching unit) 
        # and scale that as a factor
        bmu_pos = self._find_som_bmu(embedding)
        bmu_vec = self.som_grid[bmu_pos[0], bmu_pos[1]]
        dist = np.linalg.norm(embedding - bmu_vec)
        # The further from BMU, the more the SOM must reorganize
        # scale
        return float(1.0 - np.exp(-dist))

    async def _calculate_overlap(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Overlap factor - a measure of redundancy with existing memory. 
        HPC-QR typically subtracts overlap. High overlap => bigger penalty.
        """
        if "embedding_history" not in context or not context["embedding_history"]:
            return 0.0

        max_sim = 0.0
        sims = []
        for hist_emb in context["embedding_history"]:
            hist = self._normalize_embedding(hist_emb)
            sim = float(np.dot(embedding, hist))
            sims.append(sim)
            if sim > max_sim:
                max_sim = sim

        method = self.config['overlap_method']
        if method == 'avg_similarity':
            overlap_val = np.mean(sims) if sims else 0.0
        else:  # 'max_similarity' or fallback
            overlap_val = max_sim

        return min(1.0, overlap_val)

    def _find_som_bmu(self, emb: np.ndarray) -> Tuple[int,int]:
        """
        Find the Best Matching Unit (BMU) in the SOM grid for the embedding.
        """
        # Flatten grid to (M*N) x dim
        grid_shape = self.som_grid.shape
        reshaped = self.som_grid.reshape((-1, grid_shape[2]))
        # Compute distances
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
        max_iter = 10000
        t = self.som_iterations

        # Decay learning rate and sigma
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
        This is a simplified placeholder showing how you might approach it.
        """
        # For each existing node, measure similarity. If above threshold, link them.
        for node in self.causal_graph.nodes:
            if node == concept_id:
                continue
            node_emb = self.causal_graph.nodes[node].get('embedding')
            if node_emb is None:
                continue
            
            node_emb = self._normalize_embedding(node_emb)
            sim = float(np.dot(emb, node_emb))
            
            if sim > self.config['causal_edge_threshold']:
                # Add or update edge weight
                if not self.causal_graph.has_edge(node, concept_id):
                    self.causal_graph.add_edge(node, concept_id, weight=sim)
                else:
                    old_weight = self.causal_graph[node][concept_id].get('weight', 0.5)
                    new_weight = (old_weight + sim) / 2.0
                    self.causal_graph[node][concept_id]['weight'] = new_weight

        # Also store embedding on the concept node
        self.causal_graph.nodes[concept_id]['embedding'] = emb

    async def _update_prototype(self, category: str, emb: np.ndarray) -> None:
        """
        Update or create a category prototype vector.
        """
        if category not in self.prototypes:
            self.prototypes[category] = emb
            return

        old_proto = self.prototypes[category]
        new_proto = 0.9 * old_proto + 0.1 * emb  # simple moving average
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
        # You can adjust the slope or center as needed
        return float(expit(2.0 * (x - 0.5)))  # logistic around 0.5

    def _normalize_embedding(self, x: Union[np.ndarray, torch.Tensor, List[float]]) -> np.ndarray:
        """
        Utility to convert to np array and L2-normalize.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, list):
            x = np.array(x, dtype=np.float32)

        norm = np.linalg.norm(x)
        if norm > 1e-9:
            return x / norm
        return x

    def _calculate_time_decay(self, context: Dict[str, Any]) -> float:
        """
        Exponential time decay factor.
        """
        timestamp = context.get('timestamp', time.time())
        elapsed_days = (time.time() - timestamp) / 86400.0
        decay_rate = self.config['time_decay_rate']
        decay_factor = np.exp(-decay_rate * elapsed_days)
        return max(0.1, decay_factor)  # clamp at 0.1

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

    def set_factor_weights(self, weights: Dict[Union[str, QuickRecallFactor], float]) -> None:
        """
        Manually override HPC-QR factor weights. Automatically sets mode to CUSTOM.
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
        logger.info("Custom HPC-QR factor weights applied.")

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
