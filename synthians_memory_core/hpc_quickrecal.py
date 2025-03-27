# synthians_memory_core/hpc_quickrecal.py

import os
import math
import logging
import json
import time
import asyncio
import traceback
import numpy as np
import torch
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union

from .geometry_manager import GeometryManager, GeometryType # Import from the unified manager
from .custom_logger import logger # Use the shared custom logger

# Renamed from FactorKeys for clarity
class QuickRecallFactor(Enum):
    RECENCY = "recency"
    EMOTION = "emotion"
    EXTENDED_EMOTION = "extended_emotion" # For buffer-based emotion
    RELEVANCE = "relevance" # e.g., similarity to query
    OVERLAP = "overlap" # Redundancy penalty
    R_GEOMETRY = "r_geometry" # Geometric novelty/distance
    CAUSAL_NOVELTY = "causal_novelty" # Surprise based on causal model/prediction
    SELF_ORG = "self_org" # Based on SOM or similar clustering
    IMPORTANCE = "importance" # Explicitly assigned importance
    PERSONAL = "personal" # Related to user's personal info
    SURPRISE = "surprise" # General novelty or unexpectedness
    DIVERSITY = "diversity" # Difference from other recent memories
    COHERENCE = "coherence" # Logical consistency with existing knowledge
    INFORMATION = "information" # Information density or value

class QuickRecallMode(Enum):
    STANDARD = "standard"
    HPC_QR = "hpc_qr" # Original HPC-QR formula using alpha, beta, etc.
    MINIMAL = "minimal" # Basic recency, relevance, emotion
    CUSTOM = "custom" # User-defined weights

class UnifiedQuickRecallCalculator:
    """Unified calculator for memory importance using HPC-QR principles."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, geometry_manager: Optional[GeometryManager] = None):
        self.config = {
            'mode': QuickRecallMode.STANDARD,
            'factor_weights': {},
            'time_decay_rate': 0.1,
            'novelty_threshold': 0.45,
            'min_qr_score': 0.0,
            'max_qr_score': 1.0,
            'history_window': 100,
            'embedding_dim': 768,
             # HPC-QR specific weights (used in HPC_QR mode or as fallback)
            'alpha': 0.35, 'beta': 0.35, 'gamma': 0.2, 'delta': 0.1,
             # Factor configs
            'personal_keywords': ['my name', 'i live', 'my birthday', 'my job', 'my family'],
            'emotion_intensifiers': ['very', 'really', 'extremely', 'so'],
             **(config or {})
        }
        self.geometry_manager = geometry_manager or GeometryManager(self.config) # Use provided or create new
        self._init_factor_weights()
        self.history = {'calculated_qr': [], 'timestamps': [], 'factor_values': {f: [] for f in QuickRecallFactor}}
        self.total_calculations = 0
        logger.info("UnifiedQuickRecallCalculator", f"Initialized with mode: {self.config['mode'].value}")

    def _init_factor_weights(self):
        """Initialize weights based on mode."""
        default_weights = {f: 0.1 for f in QuickRecallFactor if f != QuickRecallFactor.OVERLAP} # Default equal weights
        default_weights[QuickRecallFactor.OVERLAP] = -0.1 # Overlap is a penalty

        mode = self.config['mode']
        if mode == QuickRecallMode.STANDARD:
            self.factor_weights = {
                QuickRecallFactor.RELEVANCE: 0.25, QuickRecallFactor.RECENCY: 0.15,
                QuickRecallFactor.EMOTION: 0.15, QuickRecallFactor.IMPORTANCE: 0.1,
                QuickRecallFactor.PERSONAL: 0.1, QuickRecallFactor.SURPRISE: 0.1,
                QuickRecallFactor.DIVERSITY: 0.05, QuickRecallFactor.COHERENCE: 0.05,
                QuickRecallFactor.INFORMATION: 0.05, QuickRecallFactor.OVERLAP: -0.1,
                 # Include HPC-QR factors with small default weights
                QuickRecallFactor.R_GEOMETRY: 0.0, QuickRecallFactor.CAUSAL_NOVELTY: 0.0,
                QuickRecallFactor.SELF_ORG: 0.0
            }
        elif mode == QuickRecallMode.MINIMAL:
             self.factor_weights = {
                QuickRecallFactor.RECENCY: 0.4, QuickRecallFactor.RELEVANCE: 0.4,
                QuickRecallFactor.EMOTION: 0.2, QuickRecallFactor.OVERLAP: -0.1
            }
        elif mode == QuickRecallMode.CUSTOM:
            # Ensure all factors are present, use defaults if missing
            user_weights = self.config.get('factor_weights', {})
            self.factor_weights = default_weights.copy()
            for factor, weight in user_weights.items():
                 if isinstance(factor, str): factor = QuickRecallFactor(factor.lower())
                 if factor in self.factor_weights: self.factor_weights[factor] = weight
        else: # Default to standard weights if mode is unrecognized (including HPC_QR for now)
            self.factor_weights = default_weights.copy()

        # Normalize weights (excluding overlap penalty)
        positive_weight_sum = sum(w for f, w in self.factor_weights.items() if f != QuickRecallFactor.OVERLAP and w > 0)
        if positive_weight_sum > 0 and abs(positive_weight_sum - 1.0) > 1e-6 :
             scale = 1.0 / positive_weight_sum
             for f in self.factor_weights:
                  if f != QuickRecallFactor.OVERLAP and self.factor_weights[f] > 0:
                       self.factor_weights[f] *= scale
        logger.debug("UnifiedQuickRecallCalculator", f"Initialized factor weights for mode {mode.value}", self.factor_weights)

    async def calculate(
        self,
        embedding_or_text: Union[str, np.ndarray, torch.Tensor, List[float]],
        text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate the composite QuickRecal score."""
        start_time = time.time()
        context = context or {}

        # --- Prepare Embedding ---
        embedding = None
        if isinstance(embedding_or_text, str):
            text_content = embedding_or_text
            # Generate embedding if needed (consider moving this outside for performance)
            # embedding = await self._generate_embedding(text_content) # Assume embedding gen is handled externally or mocked
            logger.debug("UnifiedQuickRecallCalculator", "Calculating score based on text, embedding generation assumed external.")
        else:
            embedding = self.geometry_manager._validate_vector(embedding_or_text, "Input Embedding")
            text_content = text or context.get("text", "") # Get text if available

        if embedding is not None:
             embedding = self.geometry_manager._normalize(embedding) # Ensure normalized

        # --- Calculate Factors ---
        factor_values = {}
        tasks = []

        # Helper to run potentially async factor calculations
        async def calculate_factor(factor, func, *args):
             try:
                 # Check if the function is a coroutine function or returns a coroutine
                 if asyncio.iscoroutinefunction(func):
                     val = await func(*args)
                 else:
                     # Regular function, don't await
                     val = func(*args)
                 factor_values[factor] = float(np.clip(val, 0.0, 1.0))
             except Exception as e:
                 logger.error("UnifiedQuickRecallCalculator", f"Error calculating factor {factor.value}", {"error": str(e)})
                 factor_values[factor] = 0.0 # Default on error

        # Context-based factors (fast)
        # Use the sync versions directly since they're quick
        factor_values[QuickRecallFactor.RECENCY] = self._calculate_recency(context)
        factor_values[QuickRecallFactor.RELEVANCE] = self._calculate_relevance(context)
        factor_values[QuickRecallFactor.IMPORTANCE] = self._calculate_importance(text_content, context)
        factor_values[QuickRecallFactor.PERSONAL] = self._calculate_personal(text_content, context)

        # Text-based factors (potentially slower)
        if text_content:
             tasks.append(calculate_factor(QuickRecallFactor.EMOTION, self._calculate_emotion, text_content, context))
             tasks.append(calculate_factor(QuickRecallFactor.INFORMATION, self._calculate_information, text_content, context))
             tasks.append(calculate_factor(QuickRecallFactor.COHERENCE, self._calculate_coherence, text_content, context))
        else:
             factor_values[QuickRecallFactor.EMOTION] = 0.0
             factor_values[QuickRecallFactor.INFORMATION] = 0.0
             factor_values[QuickRecallFactor.COHERENCE] = 0.0

        # Embedding-based factors (potentially slowest)
        if embedding is not None:
             # Use external momentum if provided
             external_momentum = context.get('external_momentum', None)
             tasks.append(calculate_factor(QuickRecallFactor.SURPRISE, self._calculate_surprise, embedding, external_momentum))
             tasks.append(calculate_factor(QuickRecallFactor.DIVERSITY, self._calculate_diversity, embedding, external_momentum))
             tasks.append(calculate_factor(QuickRecallFactor.OVERLAP, self._calculate_overlap, embedding, external_momentum))
             # HPC-QR specific factors
             tasks.append(calculate_factor(QuickRecallFactor.R_GEOMETRY, self._calculate_r_geometry, embedding, external_momentum))
             tasks.append(calculate_factor(QuickRecallFactor.CAUSAL_NOVELTY, self._calculate_causal_novelty, embedding, context)) # Causal needs context
             tasks.append(calculate_factor(QuickRecallFactor.SELF_ORG, self._calculate_self_org, embedding, context)) # SOM needs context (or internal SOM state)
        else:
             factor_values[QuickRecallFactor.SURPRISE] = 0.5
             factor_values[QuickRecallFactor.DIVERSITY] = 0.5
             factor_values[QuickRecallFactor.OVERLAP] = 0.0
             factor_values[QuickRecallFactor.R_GEOMETRY] = 0.5
             factor_values[QuickRecallFactor.CAUSAL_NOVELTY] = 0.5
             factor_values[QuickRecallFactor.SELF_ORG] = 0.5

        # Run potentially async calculations
        if tasks:
            await asyncio.gather(*tasks)

        # --- Combine Factors ---
        final_score = 0.0
        if self.config['mode'] == QuickRecallMode.HPC_QR:
            # Use the original alpha, beta, gamma, delta formula
            final_score = (
                self.config['alpha'] * factor_values.get(QuickRecallFactor.R_GEOMETRY, 0.0) +
                self.config['beta'] * factor_values.get(QuickRecallFactor.CAUSAL_NOVELTY, 0.0) +
                self.config['gamma'] * factor_values.get(QuickRecallFactor.SELF_ORG, 0.0) -
                self.config['delta'] * factor_values.get(QuickRecallFactor.OVERLAP, 0.0)
            )
            # Add other factors with small weights if needed, or keep it pure HPC-QR
            final_score += 0.05 * factor_values.get(QuickRecallFactor.RECENCY, 0.0)
            final_score += 0.05 * factor_values.get(QuickRecallFactor.EMOTION, 0.0)

        else:
            # Use weighted sum based on mode/custom weights
            for factor, value in factor_values.items():
                weight = self.factor_weights.get(factor, 0.0)
                # Overlap is a penalty
                if factor == QuickRecallFactor.OVERLAP:
                    final_score -= abs(weight) * value
                else:
                    final_score += weight * value

        # Apply time decay
        time_decay = self._calculate_time_decay(context)
        final_score *= time_decay

        # Clamp score
        final_score = float(np.clip(final_score, self.config['min_qr_score'], self.config['max_qr_score']))

        # Update history and stats
        self._update_history(final_score, factor_values)
        self.total_calculations += 1
        calculation_time = (time.time() - start_time) * 1000
        logger.debug("UnifiedQuickRecallCalculator", f"Score calculated: {final_score:.4f}", {"time_ms": calculation_time, "mode": self.config['mode'].value, "factors": {f.value: v for f,v in factor_values.items()}})

        return final_score

    def calculate_sync(
        self,
        embedding_or_text: Union[str, np.ndarray, torch.Tensor, List[float]],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Synchronous version of calculate for use in environments where asyncio.run() causes issues."""
        start_time = time.time()
        context = context or {}

        # --- Prepare Embedding ---
        embedding = None
        if isinstance(embedding_or_text, str):
            text_content = embedding_or_text
            logger.debug("UnifiedQuickRecallCalculator", "Calculating score based on text only in sync mode.")
        else:
            embedding = self.geometry_manager._validate_vector(embedding_or_text, "Input Embedding")
            text_content = context.get("text", "") # Get text if available

        if embedding is not None:
            embedding = self.geometry_manager._normalize(embedding) # Ensure normalized

        # --- Calculate Factors ---
        factor_values = {}

        # Context-based factors (fast)
        factor_values[QuickRecallFactor.RECENCY] = self._calculate_recency(context)
        factor_values[QuickRecallFactor.RELEVANCE] = self._calculate_relevance(context)
        factor_values[QuickRecallFactor.IMPORTANCE] = self._calculate_importance(text_content, context)
        factor_values[QuickRecallFactor.PERSONAL] = self._calculate_personal(text_content, context)

        # Text-based factors
        if text_content:
            # Use synchronous versions or set defaults
            try:
                factor_values[QuickRecallFactor.EMOTION] = self._calculate_emotion_sync(text_content, context)
            except:
                factor_values[QuickRecallFactor.EMOTION] = 0.0
                
            factor_values[QuickRecallFactor.INFORMATION] = 0.5  # Default value
            factor_values[QuickRecallFactor.COHERENCE] = 0.5    # Default value
        else:
            factor_values[QuickRecallFactor.EMOTION] = 0.0
            factor_values[QuickRecallFactor.INFORMATION] = 0.0
            factor_values[QuickRecallFactor.COHERENCE] = 0.0

        # Embedding-based factors
        if embedding is not None:
            # Use external momentum if provided
            external_momentum = context.get('external_momentum', None)
            factor_values[QuickRecallFactor.SURPRISE] = self._calculate_surprise_sync(embedding, external_momentum)
            factor_values[QuickRecallFactor.DIVERSITY] = self._calculate_diversity_sync(embedding, external_momentum)
            factor_values[QuickRecallFactor.OVERLAP] = self._calculate_overlap_sync(embedding, external_momentum)
            # HPC-QR specific factors
            factor_values[QuickRecallFactor.R_GEOMETRY] = self._calculate_r_geometry_sync(embedding, external_momentum)
            factor_values[QuickRecallFactor.CAUSAL_NOVELTY] = self._calculate_causal_novelty_sync(embedding, context)
            factor_values[QuickRecallFactor.SELF_ORG] = self._calculate_self_org_sync(embedding, context)
        else:
            factor_values[QuickRecallFactor.SURPRISE] = 0.5
            factor_values[QuickRecallFactor.DIVERSITY] = 0.5
            factor_values[QuickRecallFactor.OVERLAP] = 0.0
            factor_values[QuickRecallFactor.R_GEOMETRY] = 0.5
            factor_values[QuickRecallFactor.CAUSAL_NOVELTY] = 0.5
            factor_values[QuickRecallFactor.SELF_ORG] = 0.5

        # --- Combine Factors ---
        final_score = 0.0
        if self.config['mode'] == QuickRecallMode.HPC_QR:
            # Use the original alpha, beta, gamma, delta formula
            final_score = (
                self.config['alpha'] * factor_values.get(QuickRecallFactor.R_GEOMETRY, 0.0) +
                self.config['beta'] * factor_values.get(QuickRecallFactor.CAUSAL_NOVELTY, 0.0) +
                self.config['gamma'] * factor_values.get(QuickRecallFactor.SELF_ORG, 0.0) -
                self.config['delta'] * factor_values.get(QuickRecallFactor.OVERLAP, 0.0)
            )
            # Add other factors with small weights
            final_score += 0.05 * factor_values.get(QuickRecallFactor.RECENCY, 0.0)
            final_score += 0.05 * factor_values.get(QuickRecallFactor.EMOTION, 0.0)
        else:
            # Use weighted sum based on mode/custom weights
            for factor, value in factor_values.items():
                weight = self.factor_weights.get(factor, 0.0)
                # Overlap is a penalty
                if factor == QuickRecallFactor.OVERLAP:
                    final_score -= abs(weight) * value
                else:
                    final_score += weight * value

        # Apply time decay
        time_decay = self._calculate_time_decay(context)
        final_score *= time_decay

        # Clamp score
        final_score = float(np.clip(final_score, self.config['min_qr_score'], self.config['max_qr_score']))

        # Update history and stats
        self._update_history(final_score, factor_values)
        self.total_calculations += 1
        calculation_time = (time.time() - start_time) * 1000
        logger.debug("UnifiedQuickRecallCalculator", f"Score calculated (sync): {final_score:.4f}", {"time_ms": calculation_time, "mode": self.config['mode'].value})

        return final_score
        
    # Synchronous versions of the async calculation methods
    def _calculate_emotion_sync(self, text: str, context: Dict[str, Any]) -> float:
        # Simple fallback implementation
        return 0.5
        
    def _calculate_surprise_sync(self, embedding: np.ndarray, external_momentum=None) -> float:
        # Handle the dimension mismatch with safe alignment
        try:
            # Similar implementation to async version but synchronous
            if self.momentum_buffer is None or len(self.momentum_buffer) == 0:
                return 0.5  # Default value when no history
                
            # Calculate vector distance, handle dimension mismatch
            distances = []
            for vec in self.momentum_buffer:
                try:
                    # Use existing alignment functionality
                    aligned_vec, aligned_embedding = self._align_vectors_for_comparison(vec, embedding, log_warnings=False)
                    dist = self.geometry_manager.calculate_distance(aligned_vec, aligned_embedding)
                    distances.append(dist)
                except Exception:
                    distances.append(0.5)  # Default on error
                    
            if not distances:
                return 0.5
                
            # Calculate surprise based on minimum distance (most similar)
            min_dist = min(distances)
            surprise = min_dist / self.config.get('surprise_normalization', 2.0)
            return float(np.clip(surprise, 0.0, 1.0))
        except Exception as e:
            logger.warning("UnifiedQuickRecallCalculator", f"Error in surprise calc: {str(e)}")
            return 0.5
    
    def _calculate_diversity_sync(self, embedding: np.ndarray, external_momentum=None) -> float:
        # Simple implementation that handles dimension mismatches
        try:
            return self._calculate_surprise_sync(embedding, external_momentum) * 0.8  # Simplified
        except Exception:
            return 0.5
    
    def _calculate_overlap_sync(self, embedding: np.ndarray, external_momentum=None) -> float:
        # Simple implementation
        return 0.0  # Default no overlap
    
    def _calculate_r_geometry_sync(self, embedding: np.ndarray, external_momentum=None) -> float:
        # Simple implementation
        return 0.6  # Default moderate geometric novelty
    
    def _calculate_causal_novelty_sync(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
        # Simple implementation
        return 0.5  # Default causal novelty
    
    def _calculate_self_org_sync(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
        # Simple implementation
        return 0.5  # Default self-organization

    # --- Factor Calculation Methods ---
    # (Implementations adapted from your previous code, using GeometryManager where needed)

    def _calculate_recency(self, context: Dict[str, Any]) -> float:
        timestamp = context.get('timestamp', time.time())
        age_seconds = time.time() - timestamp
        # Exponential decay with a half-life of ~3 days
        decay_factor = np.exp(-age_seconds / (3 * 86400))
        return float(decay_factor)

    def _calculate_relevance(self, context: Dict[str, Any]) -> float:
        # Relevance might come from an external source (e.g., query similarity)
        return float(context.get('relevance', context.get('similarity', 0.5)))

    def _calculate_importance(self, text: str, context: Dict[str, Any]) -> float:
        # Explicit importance or keyword-based
        explicit_importance = context.get('importance', context.get('significance', 0.0))
        if text:
             keywords = ['important', 'remember', 'critical', 'key', 'significant']
             keyword_score = sum(1 for k in keywords if k in text.lower()) / 3.0
             return float(np.clip(max(explicit_importance, keyword_score), 0.0, 1.0))
        return float(explicit_importance)

    def _calculate_personal(self, text: str, context: Dict[str, Any]) -> float:
        # Check for personal keywords
        if text:
            count = sum(1 for k in self.config.get('personal_keywords', []) if k in text.lower())
            return float(np.clip(count / 3.0, 0.0, 1.0))
        return 0.0

    async def _calculate_emotion(self, text: str, context: Dict[str, Any]) -> float:
         # Reuse context's emotion data if available, otherwise analyze
         if 'emotion_data' in context and context['emotion_data']:
             intensity = context['emotion_data'].get('intensity', 0.0) # Assumes intensity 0-1
             valence_abs = abs(context['emotion_data'].get('sentiment_value', 0.0)) # Assumes valence -1 to 1
             return float(np.clip((intensity + valence_abs) / 2.0, 0.0, 1.0))
         # Placeholder: Simple keyword analysis if no analyzer
         if text:
              count = sum(1 for k in self.config.get('emotional_keywords', []) if k in text.lower())
              intensity = sum(1 for k in self.config.get('emotion_intensifiers', []) if k in text.lower())
              return float(np.clip((count + intensity) / 5.0, 0.0, 1.0))
         return 0.0

    async def _calculate_surprise(self, embedding: np.ndarray, external_momentum) -> float:
        # Novelty compared to recent memories (momentum)
        if external_momentum is None or len(external_momentum) == 0: return 0.5
        similarities = []
        for mem_emb in external_momentum[-5:]: # Compare with last 5
             sim = self.geometry_manager.calculate_similarity(embedding, mem_emb)
             similarities.append(sim)
        max_sim = max(similarities) if similarities else 0.0
        surprise = 1.0 - max_sim # Higher surprise if less similar to recent items
        return float(np.clip(surprise, 0.0, 1.0))

    async def _calculate_diversity(self, embedding: np.ndarray, external_momentum) -> float:
         # Novelty compared to the entire buffer (or a sample)
         if external_momentum is None or len(external_momentum) < 2: return 0.5
         # Sample if buffer is large
         sample_size = min(50, len(external_momentum))
         indices = np.random.choice(len(external_momentum), sample_size, replace=False)
         sample_momentum = [external_momentum[i] for i in indices]

         similarities = []
         for mem_emb in sample_momentum:
              sim = self.geometry_manager.calculate_similarity(embedding, mem_emb)
              similarities.append(sim)
         avg_sim = np.mean(similarities) if similarities else 0.0
         diversity = 1.0 - avg_sim # Higher diversity if less similar on average
         return float(np.clip(diversity, 0.0, 1.0))

    async def _calculate_overlap(self, embedding: np.ndarray, external_momentum) -> float:
         # Similar to surprise, but focused on maximum similarity as redundancy measure
         if external_momentum is None or len(external_momentum) == 0: return 0.0
         similarities = []
         for mem_emb in external_momentum[-10:]: # Check against more recent items for overlap
              sim = self.geometry_manager.calculate_similarity(embedding, mem_emb)
              similarities.append(sim)
         max_sim = max(similarities) if similarities else 0.0
         # Overlap is directly related to max similarity
         return float(np.clip(max_sim, 0.0, 1.0))

    async def _calculate_r_geometry(self, embedding: np.ndarray, external_momentum) -> float:
         # Distance from the center of the momentum buffer
         if external_momentum is None or len(external_momentum) < 3: return 0.5
         # Calculate centroid
         aligned_embeddings = []
         target_dim = self.config['embedding_dim']
         for emb in external_momentum:
              validated = self.geometry_manager._validate_vector(emb)
              if validated is not None:
                   aligned, _ = self.geometry_manager._align_vectors(validated, np.zeros(target_dim))
                   aligned_embeddings.append(aligned)

         if not aligned_embeddings: return 0.5
         centroid = np.mean(aligned_embeddings, axis=0)
         # Calculate distance from embedding to centroid
         distance = self.geometry_manager.calculate_distance(embedding, centroid)
         # Convert distance to score (larger distance = more novel = higher score)
         # Use exponential decay on distance
         geometry_score = np.exp(-distance * 0.5) # Adjust scaling factor as needed
         return float(np.clip(geometry_score, 0.0, 1.0))

    async def _calculate_causal_novelty(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
         # Placeholder - Requires a causal model
         # Simulates prediction based on context and compares with actual embedding
         predicted_embedding = embedding + np.random.randn(*embedding.shape) * 0.1 # Simulate slight prediction error
         novelty = 1.0 - self.geometry_manager.calculate_similarity(embedding, predicted_embedding)
         return float(np.clip(novelty, 0.0, 1.0))

    async def _calculate_self_org(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
         # Placeholder - Requires SOM or similar structure
         # Simulates finding BMU distance
         distance = np.random.rand() * 2.0 # Random distance 0-2
         self_org_score = np.exp(-distance * 0.5)
         return float(np.clip(self_org_score, 0.0, 1.0))

    def _calculate_information(self, text: str, context: Dict[str, Any]) -> float:
         # Simple length and keyword based information score
         if not text: return 0.0
         length_score = np.clip(len(text.split()) / 100.0, 0.0, 1.0)
         # Add bonus for specific informational keywords if needed
         return float(length_score)

    def _calculate_coherence(self, text: str, context: Dict[str, Any]) -> float:
         # Placeholder - Requires more advanced NLP or context checking
         # Simulate based on sentence structure (longer sentences might be more coherent)
         if not text: return 0.5
         sentences = [s for s in text.split('.') if s.strip()]
         if not sentences: return 0.5
         avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
         coherence_score = np.clip(avg_len / 30.0, 0.0, 1.0) # Assuming avg sentence length target is ~30 words
         return float(coherence_score)

    def _calculate_user_attention(self, context: Dict[str, Any]) -> float:
         # Placeholder - Requires input from UI/interaction layer
         return float(context.get('user_attention', 0.0))

    def _calculate_time_decay(self, context: Dict[str, Any]) -> float:
        """Exponential time decay, clamped at min_time_decay."""
        timestamp = context.get('timestamp', time.time())
        elapsed_days = (time.time() - timestamp) / 86400.0
        decay_factor = np.exp(-self.config['time_decay_rate'] * elapsed_days)
        return float(max(self.config.get('min_time_decay', 0.02), decay_factor))

    def _update_history(self, score: float, factor_values: Dict[QuickRecallFactor, float]):
        """Update score history."""
        self.history['calculated_qr'].append(score)
        self.history['timestamps'].append(time.time())
        for factor, value in factor_values.items():
            self.history['factor_values'][factor].append(value)

        # Trim history
        hw = self.config['history_window']
        if len(self.history['calculated_qr']) > hw:
            self.history['calculated_qr'].pop(0)
            self.history['timestamps'].pop(0)
            for factor in self.history['factor_values']:
                if len(self.history['factor_values'][factor]) > hw:
                    self.history['factor_values'][factor].pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve calculator statistics."""
        qr_scores = self.history['calculated_qr']
        factor_stats = {}
        for factor, values in self.history['factor_values'].items():
             if values:
                  factor_stats[factor.value] = {
                       'average': float(np.mean(values)),
                       'stddev': float(np.std(values)),
                       'weight': self.factor_weights.get(factor, 0.0)
                  }

        return {
            'mode': self.config['mode'].value,
            'total_calculations': self.total_calculations,
            'avg_qr_score': float(np.mean(qr_scores)) if qr_scores else 0.0,
            'std_qr_score': float(np.std(qr_scores)) if qr_scores else 0.0,
            'history_size': len(qr_scores),
            'factors': factor_stats
        }
