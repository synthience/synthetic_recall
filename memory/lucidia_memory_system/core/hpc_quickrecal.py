"""
HPC-QuickRecal Module

Implements the enhanced memory significance system with hyperbolic geometry,
causal drift detection, and emotional gating as outlined in the blueprint.
"""

import asyncio
import logging
import collections
import time
import math
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum, auto

logger = logging.getLogger(__name__)

# Standardize factor keys with an Enum for consistency
class FactorKeys(Enum):
    RECENCY = auto()
    EMOTION = auto()
    EXTENDED_EMOTION = auto()
    RELEVANCE = auto()
    OVERLAP = auto()
    R_GEOMETRY = auto()
    CAUSAL_NOVELTY = auto()
    SELF_ORG = auto()
    
    def __str__(self):
        return self.name

class EmotionalBuffer:
    """
    Implements the Emotional Buffer component from the blueprint.
    Tracks both short-term emotional spikes and repeated mild emotional triggers.
    """
    
    def __init__(self, maxlen: int = 5):
        """Initialize the emotional buffer."""
        self.buffer = collections.deque(maxlen=maxlen)
        self.spike_threshold = 0.7  # For immediate flagging
        self.mild_threshold = 0.3   # For repeated mild emotions
        self.repetition_threshold = 3  # How many times mild emotions need to appear
    
    def add(self, emotion_score: float, timestamp: float = None) -> None:
        """
        Add an emotion score to the buffer.
        
        Args:
            emotion_score: Emotion intensity score (0.0-1.0)
            timestamp: Optional timestamp for the emotion
        """
        if timestamp is None:
            timestamp = time.time()
            
        self.buffer.append({
            "score": emotion_score,
            "timestamp": timestamp
        })
    
    def check_emotional_gating(self, current_score: float) -> Tuple[bool, str]:
        """
        Check if current emotion should trigger advanced HPC-QR.
        
        Args:
            current_score: Current emotion score to check
            
        Returns:
            Tuple of (should_trigger, reason)
        """
        # Check for short-term emotional spike
        if current_score >= self.spike_threshold:
            return True, "short_term_emotional_spike"
            
        # Add current emotion to buffer for future checks
        self.add(current_score)
        
        # Check for repeated mild emotional hits
        mild_emotions = [item for item in self.buffer 
                        if item["score"] >= self.mild_threshold]
                        
        if len(mild_emotions) >= self.repetition_threshold:
            # Calculate recency-weighted average to prioritize recent emotions
            recent_window = 300  # 5 minutes in seconds
            now = time.time()
            
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for item in mild_emotions:
                age = now - item["timestamp"]
                weight = math.exp(-age / recent_window)  # Exponential decay
                weighted_sum += item["score"] * weight
                weight_sum += weight
            
            if weight_sum > 0:
                weighted_avg = weighted_sum / weight_sum
                if weighted_avg >= self.mild_threshold:
                    return True, "repeated_mild_emotions"
        
        return False, ""

    def calculate_extended_emotion(self, current_score: float) -> float:
        """
        Calculate extended emotion based on buffer history and current score.
        
        Args:
            current_score: Current emotion score
            
        Returns:
            Extended emotion score incorporating buffer history
        """
        # Add current score to buffer
        self.add(current_score)
        
        if len(self.buffer) <= 1:
            return current_score
        
        # Calculate recency-weighted average
        recent_window = 300  # 5 minutes in seconds
        now = time.time()
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for item in self.buffer:
            age = now - item["timestamp"]
            # Exponential decay with recency bias
            weight = math.exp(-age / recent_window)
            weighted_sum += item["score"] * weight
            weight_sum += weight
        
        if weight_sum > 0:
            extended_score = weighted_sum / weight_sum
            
            # Boost extended score if there's a pattern of consistent emotions
            if len(self.buffer) >= 3:
                consistency = self._calculate_emotional_consistency()
                extended_score = extended_score * (1.0 + 0.5 * consistency)
            
            return min(1.0, extended_score)
        
        return current_score
    
    def _calculate_emotional_consistency(self) -> float:
        """
        Calculate how consistent the emotions in the buffer are.
        
        Returns:
            Consistency score (0.0-1.0)
        """
        if len(self.buffer) < 2:
            return 0.0
        
        # Calculate standard deviation of scores
        scores = [item["score"] for item in self.buffer]
        std_dev = np.std(scores)
        
        # Lower std_dev means higher consistency
        consistency = max(0.0, 1.0 - std_dev * 2.0)  # Scale so std_dev of 0.5 → 0 consistency
        
        return consistency


class CausalDriftTracker:
    """
    Implements the Causal Drift Tracker component from the blueprint.
    Tracks repeated small contradictions that add up over time.
    """
    
    def __init__(self, drift_threshold: int = 3):
        """Initialize the causal drift tracker."""
        self.tracker = {}  # Maps memory_id to drift data
        self.drift_threshold = drift_threshold
    
    def update_causal_drift(self, memory_id: str, 
                           new_contradiction: bool = False, 
                           contradiction_strength: float = 0.1) -> float:
        """
        Update the causal drift for a memory.
        
        Args:
            memory_id: ID of the memory to update
            new_contradiction: Whether a new contradiction was detected
            contradiction_strength: Strength of the contradiction (0.0-1.0)
            
        Returns:
            Current causal drift value
        """
        # Initialize if not present
        if memory_id not in self.tracker:
            self.tracker[memory_id] = {
                "repeated_surprise_hits": 0,
                "drift_value": 0.0,
                "last_update": time.time(),
                "contradiction_history": []
            }
        
        # Update with new contradiction if any
        if new_contradiction:
            self.tracker[memory_id]["repeated_surprise_hits"] += 1
            self.tracker[memory_id]["last_update"] = time.time()
            self.tracker[memory_id]["contradiction_history"].append({
                "timestamp": time.time(),
                "strength": contradiction_strength
            })
            
            # Update drift value - stronger contradictions contribute more
            self.tracker[memory_id]["drift_value"] += contradiction_strength
        
        return self.tracker[memory_id]["drift_value"]
    
    def get_drift_factor(self, memory_id: str) -> float:
        """
        Get the normalized drift factor for a memory.
        
        Args:
            memory_id: ID of the memory to check
            
        Returns:
            Normalized drift factor (0.0-1.0)
        """
        if memory_id not in self.tracker:
            return 0.0
        
        # Calculate normalized drift based on repeated hits
        hits = self.tracker[memory_id]["repeated_surprise_hits"]
        normalized_drift = min(1.0, hits / self.drift_threshold)
        
        # Combine with drift value
        drift_value = self.tracker[memory_id]["drift_value"]
        
        return max(normalized_drift, min(1.0, drift_value))
    
    def check_drift_gating(self, memory_id: str) -> Tuple[bool, float]:
        """
        Check if memory's drift should trigger advanced HPC-QR.
        
        Args:
            memory_id: ID of the memory to check
            
        Returns:
            Tuple of (should_trigger, drift_value)
        """
        drift = self.get_drift_factor(memory_id)
        
        # Get repeat counts
        repeat_count = 0
        if memory_id in self.tracker:
            repeat_count = self.tracker[memory_id]["repeated_surprise_hits"]
        
        # Check if hits exceed threshold
        if repeat_count >= self.drift_threshold:
            return True, drift
            
        # Check if drift value is high
        if drift >= 0.7:  # High drift threshold
            return True, drift
            
        return False, drift
    
    def get_contradiction_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Get the contradiction history for a memory.
        
        Args:
            memory_id: ID of the memory to check
            
        Returns:
            List of contradiction events
        """
        if memory_id not in self.tracker:
            return []
        
        return self.tracker[memory_id]["contradiction_history"]


class HPCQuickRecal:
    """
    Main HPC-QuickRecal implementation with tiered processing
    and hyperboloid geometry as described in the blueprint.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the HPC-QuickRecal system."""
        self.config = {
            "embedding_dim": 768,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "som_grid_size": (10, 10),
            "som_learning_rate": 0.1,
            "som_sigma": 2.0,
            "geometry_type": "mixed",  # euclidean, hyperbolic, spherical, mixed
            "curvature": -1.0,  # Negative for hyperbolic
            "alpha": 0.35,  # geometry weighting
            "beta": 0.35,   # causal novelty weighting
            "gamma": 0.2,   # self-org weighting
            "delta": 0.1,   # overlap penalty weighting
            "drift_threshold": 3,
            "emotional_buffer_size": 5,
            "overshadow_threshold": 0.3,
            "shock_absorption_rate": 0.3,  # Rate of embedding adjustment (0.0-1.0)
            "momentum_buffer_max_size": 50,  # Max size of external momentum buffer
            "momentum_decay_half_life": 3600,  # Half-life of momentum items in seconds (1 hour)
            **(config or {})
        }
        
        # Initialize components
        self.emotional_buffer = EmotionalBuffer(
            maxlen=self.config["emotional_buffer_size"]
        )
        
        self.causal_drift = CausalDriftTracker(
            drift_threshold=self.config["drift_threshold"]
        )
        
        # Initialize SOM
        self._init_som()
        
        # Initialize prototypes vector for manifold geometry
        self.prototypes = {}
        
        # External momentum buffer for similarity comparisons
        self.external_momentum = []
        
        # Add timestamp to each momentum item for decay
        self.momentum_timestamps = []
        
        # Track when SOM was last fully updated
        self.last_som_full_update = time.time()
        
        logger.info(f"Initialized HPC-QuickRecal with geometry: {self.config['geometry_type']}")

    def _init_som(self) -> None:
        """Initialize Self-Organizing Map (SOM)."""
        grid_size = self.config["som_grid_size"]
        dim = self.config["embedding_dim"]
        
        # Create grid of neurons
        self.som_grid = torch.randn(grid_size[0], grid_size[1], dim) * 0.1
        if self.config["device"] == "cuda" and torch.cuda.is_available():
            self.som_grid = self.som_grid.cuda()
            
        # Initialize learning parameters
        self.som_learning_rate = self.config["som_learning_rate"]
        self.som_sigma = self.config["som_sigma"]
        self.som_iterations = 0
        
        # Keep track of activations
        self.som_activations = torch.zeros(grid_size)
        if self.config["device"] == "cuda" and torch.cuda.is_available():
            self.som_activations = self.som_activations.cuda()
            
        logger.info(f"Initialized SOM with grid size {grid_size} and embedding dim {dim}")
    
    async def process_embedding(self, 
                              embedding: torch.Tensor, 
                              context: Dict[str, Any] = None) -> Tuple[torch.Tensor, float]:
        """
        Process an embedding through the HPC-QuickRecal pipeline.
        
        This is the main entry point for processing embeddings that implements
        both the Minimal and Advanced passes from the blueprint.
        
        Args:
            embedding: The embedding to process
            context: Additional context for processing
            
        Returns:
            Tuple of (processed_embedding, quickrecal_score)
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Move to correct device
            device = self.config["device"]
            if torch.cuda.is_available() and device == "cuda":
                embedding = embedding.cuda()
            else:
                embedding = embedding.cpu()
                
            # Ensure normalized embedding
            embedding = self._normalize_embedding(embedding)
            
            # Check for advanced HPC-QR gating
            use_advanced = False
            gating_reason = ""
            
            # Check emotional gating if available in context
            if "emotion_score" in context:
                emotion_gating, emotion_reason = self.emotional_buffer.check_emotional_gating(
                    context["emotion_score"]
                )
                if emotion_gating:
                    use_advanced = True
                    gating_reason = f"emotion_gating:{emotion_reason}"
            
            # Check causal drift gating if memory_id available
            if "memory_id" in context:
                drift_gating, drift_value = self.causal_drift.check_drift_gating(
                    context["memory_id"]
                )
                if drift_gating:
                    use_advanced = True
                    gating_reason = f"drift_gating:{drift_value:.2f}"
                    
            # Store gating information in context
            context["advanced_gating"] = use_advanced
            context["gating_reason"] = gating_reason
            
            # Run appropriate pass based on gating
            if use_advanced:
                logger.debug(f"Using advanced HPC-QR pass. Reason: {gating_reason}")
                quickrecal_score, adjusted_embedding, factor_contributions = await self._advanced_pass(
                    embedding, context
                )
            else:
                logger.debug("Using minimal HPC-QR pass")
                quickrecal_score, adjusted_embedding, factor_contributions = await self._minimal_pass(
                    embedding, context
                )
                
            # Store factor contributions in context
            context["factor_contributions"] = factor_contributions
            
            # Generate explanation if requested
            if context.get("generate_explanation", False):
                explanation = self._generate_explanation(factor_contributions, quickrecal_score)
                context["explanation"] = explanation
                
            # Check for overshadowing
            if quickrecal_score < self.config["overshadow_threshold"]:
                context["overshadowed"] = True
                context["overshadow_reason"] = "Low HPC-QR"
                
                # Generate overshadow explanation
                if context.get("generate_explanation", False):
                    overshadow_explanation = self._generate_overshadow_explanation(
                        factor_contributions, quickrecal_score
                    )
                    context["overshadow_explanation"] = overshadow_explanation
            
            # Update momentum buffer with this embedding
            self._update_momentum_buffer(adjusted_embedding)
            
            # Log performance
            duration_ms = (time.time() - start_time) * 1000
            logger.debug(f"HPC-QR processing took {duration_ms:.2f}ms, score: {quickrecal_score:.4f}")
            
            return adjusted_embedding, quickrecal_score
            
        except Exception as e:
            logger.error(f"Error in HPC-QuickRecal processing: {e}")
            # Return original embedding with default score on error
            return embedding, 0.5
    
    async def _minimal_pass(self, 
                           embedding: torch.Tensor, 
                           context: Dict[str, Any]) -> Tuple[float, torch.Tensor, Dict[str, float]]:
        """
        Perform minimal HPC-QR pass for efficiency.
        
        Implements the "Minimal HPC-QR" from the blueprint:
        - Recency
        - Overlap
        - Basic Relevance
        - Quick Emotional Check
        
        Args:
            embedding: The embedding to process
            context: Processing context
            
        Returns:
            Tuple of (quickrecal_score, adjusted_embedding, factor_contributions)
        """
        factor_contributions = {}
        
        # Recency factor
        recency = self._calculate_recency(context)
        factor_contributions[str(FactorKeys.RECENCY)] = recency
        
        # Quick emotional check
        emotion = 0.0
        if "emotion_score" in context:
            emotion = context["emotion_score"]
        factor_contributions[str(FactorKeys.EMOTION)] = emotion
        
        # Basic relevance (can be provided in context)
        relevance = context.get("relevance", 0.5)
        factor_contributions[str(FactorKeys.RELEVANCE)] = relevance
        
        # Quick overlap check
        overlap = await self._calculate_quick_overlap(embedding, context)
        factor_contributions[str(FactorKeys.OVERLAP)] = overlap
        
        # Calculate minimal HPC-QR score
        # Weighted sum of positive factors, penalized by overlap
        quickrecal_score = (
            0.4 * recency +
            0.3 * emotion +
            0.3 * relevance
        ) * (1.0 - overlap * 0.5)  # Partial overlap penalty
        
        # Adjust embedding minimally (identity transform)
        adjusted_embedding = embedding
        
        return quickrecal_score, adjusted_embedding, factor_contributions
    
    async def _advanced_pass(self, 
                            embedding: torch.Tensor, 
                            context: Dict[str, Any]) -> Tuple[float, torch.Tensor, Dict[str, float]]:
        """
        Perform advanced HPC-QR pass for important memories.
        
        Implements the "Advanced HPC-QR" from the blueprint:
        - All minimal pass factors
        - R_GEOMETRY (hyperbolic distance)
        - CAUSAL_NOVELTY (with repeated surprise tracking)
        - SELF_ORG (SOM-based organization)
        - EXTENDED_EMOTION (using emotional buffer)
        
        Args:
            embedding: The embedding to process
            context: Processing context
            
        Returns:
            Tuple of (quickrecal_score, adjusted_embedding, factor_contributions)
        """
        # First calculate minimal factors
        minimal_score, _, minimal_factors = await self._minimal_pass(embedding, context)
        factor_contributions = minimal_factors
        
        # R_GEOMETRY (hyperbolic/mixed distance)
        geometry = await self._calculate_r_geometry(embedding, context)
        factor_contributions[str(FactorKeys.R_GEOMETRY)] = geometry
        
        # CAUSAL_NOVELTY (with drift tracking)
        causal_novelty = await self._calculate_causal_novelty(embedding, context)
        factor_contributions[str(FactorKeys.CAUSAL_NOVELTY)] = causal_novelty
        
        # Update causal drift if memory_id available
        if "memory_id" in context:
            # Use causal_novelty as contradiction strength
            # Higher = more surprising = stronger contradiction
            drift_value = self.causal_drift.update_causal_drift(
                memory_id=context["memory_id"],
                new_contradiction=(causal_novelty > 0.6),  # Only significant surprises
                contradiction_strength=causal_novelty * 0.5  # Scale down
            )
            context["causal_drift"] = drift_value
        
        # SELF_ORG (SOM-based organization factor)
        self_org = await self._calculate_self_org(embedding, context)
        factor_contributions[str(FactorKeys.SELF_ORG)] = self_org
        
        # EXTENDED_EMOTION (using emotional buffer state)
        extended_emotion = minimal_factors[str(FactorKeys.EMOTION)]  # Start with basic emotion
        if "emotion_score" in context:
            # Enrich with emotional context
            extended_emotion = await self._calculate_extended_emotion(
                context["emotion_score"], context
            )
        factor_contributions[str(FactorKeys.EXTENDED_EMOTION)] = extended_emotion
        
        # Apply HPC-QR formula from blueprint
        # Use alpha/beta/gamma/delta weighting parameters
        alpha = self.config["alpha"]
        beta = self.config["beta"]
        gamma = self.config["gamma"]
        delta = self.config["delta"]
        
        quickrecal_score = (
            alpha * geometry +
            beta * causal_novelty +
            gamma * self_org +
            0.2 * factor_contributions[str(FactorKeys.RECENCY)] +
            0.2 * extended_emotion +
            0.1 * factor_contributions[str(FactorKeys.RELEVANCE)] -
            delta * factor_contributions[str(FactorKeys.OVERLAP)]  # Subtraction as penalty
        )
        
        # Ensure score is in valid range
        quickrecal_score = max(0.0, min(1.0, quickrecal_score))
        
        # Update SOM with embedding
        await self._update_som_light(embedding)
        
        # Apply shock absorption to smooth embedding changes
        adjusted_embedding = await self._apply_shock_absorption(
            embedding, quickrecal_score, context
        )
        
        return quickrecal_score, adjusted_embedding, factor_contributions
    
    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Normalize embedding to unit length and handle dimension mismatches.
        
        Args:
            embedding: Raw embedding tensor
            
        Returns:
            Normalized embedding with proper dimensions
        """
        # Handle dimension mismatch
        expected_dim = self.config["embedding_dim"]
        actual_dim = embedding.shape[0]
        
        if actual_dim != expected_dim:
            # Create new embedding with correct dimensions
            if actual_dim < expected_dim:
                # Pad with zeros
                if isinstance(embedding, torch.Tensor):
                    device = embedding.device
                    padded = torch.zeros(expected_dim, device=device)
                    padded[:actual_dim] = embedding
                    embedding = padded
                else:
                    # Handle numpy arrays
                    padded = np.zeros(expected_dim)
                    padded[:actual_dim] = embedding
                    embedding = torch.tensor(padded, device=self.config["device"])
            else:
                # Truncate
                embedding = embedding[:expected_dim]
        
        # Normalize to unit length
        norm = torch.norm(embedding)
        if norm > 1e-9:
            normalized = embedding / norm
        else:
            # Handle zero vector with random unit vector
            random_vec = torch.randn_like(embedding)
            normalized = random_vec / torch.norm(random_vec)
            
        return normalized
    
    def _calculate_recency(self, context: Dict[str, Any]) -> float:
        """Calculate recency factor based on timestamp."""
        # Get timestamp, default to current time
        timestamp = context.get("timestamp", time.time())
        
        # Calculate age in days
        age_days = (time.time() - timestamp) / (60 * 60 * 24)
        
        # Apply exponential decay
        decay_rate = 0.1  # Adjustable
        recency = math.exp(-decay_rate * age_days)
        
        return float(recency)
    
    async def _calculate_quick_overlap(self, 
                                     embedding: torch.Tensor, 
                                     context: Dict[str, Any]) -> float:
        """
        Calculate quick overlap with existing memories.
        
        Args:
            embedding: The embedding to check
            context: Processing context
            
        Returns:
            Overlap score (0.0-1.0), higher means more overlap/redundancy
        """
        # Use external momentum if available
        if not self.external_momentum or len(self.external_momentum) == 0:
            return 0.0
            
        # Calculate similarity to most similar memory
        max_sim = -1.0
        
        # Convert to CPU for calculations if needed
        cpu_embedding = embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
        
        for mem_emb in self.external_momentum:
            # Handle different types
            if isinstance(mem_emb, torch.Tensor):
                mem_cpu = mem_emb.cpu().numpy()
            else:
                mem_cpu = mem_emb
                
            # Dimension check and adjustment
            if len(mem_cpu) != len(cpu_embedding):
                # Use smaller dimension
                min_dim = min(len(mem_cpu), len(cpu_embedding))
                mem_cpu = mem_cpu[:min_dim]
                cpu_embedding_adj = cpu_embedding[:min_dim]
            else:
                cpu_embedding_adj = cpu_embedding
                
            # Calculate cosine similarity
            similarity = np.dot(cpu_embedding_adj, mem_cpu) / (
                np.linalg.norm(cpu_embedding_adj) * np.linalg.norm(mem_cpu)
            )
            
            if similarity > max_sim:
                max_sim = similarity
        
        # Convert to overlap score (higher similarity = higher overlap)
        overlap = max(0.0, max_sim)
        
        return float(overlap)
    
    async def _calculate_r_geometry(self, 
                                  embedding: torch.Tensor, 
                                  context: Dict[str, Any]) -> float:
        """
        Calculate geometry factor using hyperbolic or mixed distance.
        
        Implements the "Manifold Geometry Upgrade" from the blueprint:
        Uses hyperbolic distance in advanced pass.
        
        Args:
            embedding: The embedding to process
            context: Processing context
            
        Returns:
            Geometry factor (0.0-1.0)
        """
        if not self.external_momentum or len(self.external_momentum) < 3:
            return 0.5  # Default when not enough reference data
        
        # Calculate center of momentum buffer
        momentum_center = None
        
        if isinstance(self.external_momentum, list):
            # Convert all to same format and device for averaging
            converted = []
            for mem in self.external_momentum[-10:]:  # Use latest 10
                if isinstance(mem, torch.Tensor):
                    converted.append(mem.cpu().numpy())
                else:
                    converted.append(mem)
            
            # Get common dimension
            common_dim = min(len(e) for e in converted)
            
            # Truncate to common dimension
            aligned = [e[:common_dim] for e in converted]
            
            # Calculate center
            momentum_center = np.mean(aligned, axis=0)
        else:
            # Single tensor
            momentum_center = self.external_momentum.cpu().numpy()
        
        # Calculate distance based on geometry type
        if self.config["geometry_type"] == "hyperbolic":
            # Hyperbolic distance calculation
            distance = self._hyperbolic_distance(
                embedding.cpu().numpy()[:common_dim],
                momentum_center,
                curvature=self.config["curvature"]
            )
        elif self.config["geometry_type"] == "spherical":
            # Spherical distance calculation
            distance = self._spherical_distance(
                embedding.cpu().numpy()[:common_dim],
                momentum_center
            )
        elif self.config["geometry_type"] == "mixed":
            # Mixed distance (weighted combination)
            euc_dist = np.linalg.norm(embedding.cpu().numpy()[:common_dim] - momentum_center)
            hyp_dist = self._hyperbolic_distance(
                embedding.cpu().numpy()[:common_dim],
                momentum_center,
                curvature=self.config["curvature"]
            )
            
            # Weighted combination
            distance = 0.5 * euc_dist + 0.5 * hyp_dist
        else:
            # Default to Euclidean
            distance = np.linalg.norm(embedding.cpu().numpy()[:common_dim] - momentum_center)
        
        # Convert to geometry factor
        # Higher distance = higher novelty = higher factor
        # Scale to 0-1 range with sigmoid
        geometry_factor = 1.0 / (1.0 + np.exp(-distance + 2.0))
        
        return float(geometry_factor)
    
    def _hyperbolic_distance(self, 
                            x: np.ndarray, 
                            y: np.ndarray, 
                            curvature: float = -1.0) -> float:
        """
        Calculate hyperbolic distance between two points.
        
        Args:
            x: First point
            y: Second point
            curvature: Hyperbolic space curvature (negative)
            
        Returns:
            Hyperbolic distance
        """
        # Ensure curvature is negative for hyperbolic space
        c = min(-1e-6, curvature)
        
        # Map to Poincaré ball
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        
        # Safety threshold (points must be inside unit ball)
        if x_norm >= 1.0 or y_norm >= 1.0:
            return float('inf')
        
        # Calculate terms in hyperbolic distance formula
        xy_dist_sq = np.linalg.norm(x - y) ** 2
        denom1 = 1 - c * x_norm ** 2
        denom2 = 1 - c * y_norm ** 2
        
        # Hyperbolic distance formula
        numerator = 2 * xy_dist_sq
        denominator = denom1 * denom2
        
        # Avoid division by zero
        if denominator <= 0:
            return float('inf')
        
        # Calculate distance
        dist = np.arccosh(1 + (numerator / denominator))
        
        return float(dist)
    
    def _spherical_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate spherical distance (angle) between two points.
        
        Args:
            x: First point
            y: Second point
            
        Returns:
            Spherical distance (angle in radians)
        """
        # Normalize to unit sphere
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        
        if x_norm < 1e-9 or y_norm < 1e-9:
            return float('inf')
            
        x_unit = x / x_norm
        y_unit = y / y_norm
        
        # Compute dot product (cosine of angle)
        cos_angle = np.dot(x_unit, y_unit)
        
        # Clamp to valid range
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # Calculate spherical distance (angle)
        angle = np.arccos(cos_angle)
        
        return float(angle)
    
    async def _calculate_causal_novelty(self, 
                                      embedding: torch.Tensor, 
                                      context: Dict[str, Any]) -> float:
        """
        Calculate causal novelty factor for advanced pass.
        
        Implements the "Causal Drift Detection" from the blueprint:
        Repeated small contradictions/surprises add up.
        
        Args:
            embedding: The embedding to process
            context: Processing context
            
        Returns:
            Causal novelty factor (0.0-1.0)
        """
        if not self.external_momentum or len(self.external_momentum) < 3:
            return 0.5  # Default when not enough history
        
        try:
            # Use the most recent 3 memories to establish trend
            recent = self.external_momentum[-3:]
            
            # Convert all to numpy for consistency
            if isinstance(recent[0], torch.Tensor):
                trend = [mem.cpu().numpy() for mem in recent]
            else:
                trend = recent
                
            # Get smallest common dimension
            common_dim = min(len(e) for e in trend)
            trend = [e[:common_dim] for e in trend]
            
            # Get embedding with matching dimension
            emb_np = embedding.cpu().numpy()[:common_dim]
            
            # Calculate expected direction from previous memories
            # For simplicity, use latest two to predict trend
            direction = trend[-1] - trend[-2]
            
            # Normalize direction
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-9:
                direction = direction / dir_norm
            
            # Predict next position based on trend
            predicted = trend[-1] + direction
            
            # Normalize predicted
            pred_norm = np.linalg.norm(predicted)
            if pred_norm > 1e-9:
                predicted = predicted / pred_norm
            
            # Calculate similarity between actual and predicted
            similarity = np.dot(emb_np, predicted)
            
            # Convert to novelty (inverse of similarity)
            novelty = (1.0 - similarity) / 2.0  # Map [-1,1] to [0,1]
            
            # Apply threshold scaling
            novelty_threshold = 0.45  # From blueprint
            if novelty > novelty_threshold:
                # Boost novelty above threshold
                scaled_novelty = 0.5 + (novelty - novelty_threshold) * 2.0
                novelty = min(1.0, scaled_novelty)
            else:
                # Dampen novelty below threshold
                scaled_novelty = novelty * (0.5 / novelty_threshold)
                novelty = min(0.5, scaled_novelty)
            
            return float(novelty)
        except Exception as e:
            logger.error(f"Error calculating causal novelty: {e}")
            return 0.5
    
    async def _calculate_self_org(self, 
                                embedding: torch.Tensor, 
                                context: Dict[str, Any]) -> float:
        """
        Calculate self-organization factor for advanced pass.
        
        Implements the "Self-Organization & SOM" from the blueprint.
        
        Args:
            embedding: The embedding to process
            context: Processing context
            
        Returns:
            Self-organization factor (0.0-1.0)
        """
        # Find best matching unit (BMU) in SOM grid
        bmu_pos, bmu_dist = self._find_som_bmu(embedding)
        
        # Calculate self-organization factor based on BMU distance
        # Higher distance = more self-organization potential
        self_org_factor = min(1.0, bmu_dist / 2.0)
        
        return float(self_org_factor)
    
    def _find_som_bmu(self, embedding: torch.Tensor) -> Tuple[Tuple[int, int], float]:
        """
        Find Best Matching Unit (BMU) in SOM grid.
        
        Args:
            embedding: Input embedding
            
        Returns:
            Tuple of ((row, col), distance)
        """
        # Move to appropriate device
        if embedding.device != self.som_grid.device:
            embedding = embedding.to(self.som_grid.device)
        
        # Calculate distances to all SOM units
        grid_size = self.config["som_grid_size"]
        distances = torch.zeros(grid_size)
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Compute distance between input and SOM unit
                unit = self.som_grid[i, j]
                dist = torch.norm(embedding - unit)
                distances[i, j] = dist
        
        # Find BMU (minimum distance)
        min_idx = torch.argmin(distances.view(-1))
        row = min_idx // grid_size[1]
        col = min_idx % grid_size[1]
        
        # Get minimum distance
        min_dist = distances[row, col].item()
        
        return (row.item(), col.item()), min_dist
    
    async def _update_som_light(self, embedding: torch.Tensor) -> None:
        """
        Perform a light update to the SOM with a single embedding.
        
        This is a fast, incremental update vs. the full SOM update
        which would process a batch of embeddings and take more time.
        
        Args:
            embedding: The embedding to use for updating
        """
        # Find BMU
        bmu_pos, _ = self._find_som_bmu(embedding)
        
        # Update SOM grid
        # Use decreasing learning rate and neighborhood size over time
        self.som_iterations += 1
        current_lr = self.som_learning_rate * math.exp(-self.som_iterations / 1000)
        current_sigma = self.som_sigma * math.exp(-self.som_iterations / 1000)
        
        # Update weights
        grid_size = self.config["som_grid_size"]
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Calculate grid distance to BMU
                grid_dist = math.sqrt((i - bmu_pos[0])**2 + (j - bmu_pos[1])**2)
                
                # Calculate neighborhood function (Gaussian)
                if grid_dist <= 3 * current_sigma:  # Optimization: skip far units
                    neighborhood = math.exp(-(grid_dist**2) / (2 * current_sigma**2))
                    
                    # Update weights
                    unit = self.som_grid[i, j]
                    delta = current_lr * neighborhood * (embedding - unit)
                    self.som_grid[i, j] = unit + delta
                    
                    # Optionally track activation
                    self.som_activations[i, j] += 0.1 * neighborhood
        
        # Check if it's time for a full SOM update
        now = time.time()
        if (now - self.last_som_full_update) > 3600 * 24:  # 24 hours
            # Schedule a full SOM update
            asyncio.create_task(self._update_som_full())
    
    async def _update_som_full(self) -> None:
        """
        Perform a full update to the SOM using the entire momentum buffer.
        
        This is a more computationally intensive operation that runs
        periodically or when triggered by specific events.
        """
        logger.info("Starting full SOM update")
        start_time = time.time()
        
        # Use momentum buffer as training data
        if not self.external_momentum or len(self.external_momentum) < 10:
            logger.info("Not enough data for full SOM update")
            return
        
        # Convert all embeddings to tensors on the correct device
        embeddings = []
        for mem in self.external_momentum:
            if isinstance(mem, torch.Tensor):
                emb = mem.to(self.som_grid.device)
            else:
                emb = torch.tensor(mem, device=self.som_grid.device)
            embeddings.append(emb)
        
        # Ensure all embeddings have the same dimension
        dim = min(e.shape[0] for e in embeddings)
        embeddings = [e[:dim] for e in embeddings]
        
        # Shuffle the data
        indices = torch.randperm(len(embeddings))
        
        # Reset SOM learning parameters
        learning_rate = self.config["som_learning_rate"]
        sigma = self.config["som_sigma"]
        
        # Number of training iterations
        iterations = min(len(embeddings) * 3, 1000)
        
        # Full batch SOM training
        for it in range(iterations):
            # Get random embedding
            idx = indices[it % len(indices)]
            emb = embeddings[idx]
            
            # Find BMU
            bmu_pos, _ = self._find_som_bmu(emb)
            
            # Calculate learning rate and sigma for this iteration
            iter_fraction = it / iterations
            current_lr = learning_rate * (1.0 - iter_fraction)
            current_sigma = sigma * (1.0 - iter_fraction)
            
            # Update weights in batch
            grid_size = self.config["som_grid_size"]
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    # Calculate grid distance to BMU
                    grid_dist = math.sqrt((i - bmu_pos[0])**2 + (j - bmu_pos[1])**2)
                    
                    # Calculate neighborhood function (Gaussian)
                    if grid_dist <= 3 * current_sigma:  # Skip far units
                        neighborhood = math.exp(-(grid_dist**2) / (2 * current_sigma**2))
                        
                        # Update weights
                        unit = self.som_grid[i, j]
                        delta = current_lr * neighborhood * (emb - unit)
                        self.som_grid[i, j] = unit + delta
            
            # Yield control occasionally to prevent blocking
            if it % 100 == 0:
                await asyncio.sleep(0)
        
        # Update last full update timestamp
        self.last_som_full_update = time.time()
        
        # Reset activations
        self.som_activations.fill_(0)
        
        duration = time.time() - start_time
        logger.info(f"Full SOM update completed in {duration:.2f} seconds")
    
    async def _calculate_extended_emotion(self, 
                                       emotion_score: float, 
                                       context: Dict[str, Any]) -> float:
        """
        Calculate extended emotion factor using emotional buffer history.
        
        Args:
            emotion_score: Current emotion score
            context: Processing context
            
        Returns:
            Extended emotion factor (0.0-1.0)
        """
        # Use EmotionalBuffer to calculate extended emotion
        extended_score = self.emotional_buffer.calculate_extended_emotion(emotion_score)
        
        # Get memory_id if available for persistent tracking
        memory_id = context.get("memory_id")
        if memory_id and extended_score > self.emotional_buffer.mild_threshold:
            # If no 'memory_repeated_emotion' in context, initialize it
            if 'memory_repeated_emotion' not in context:
                context['memory_repeated_emotion'] = {}
            
            # Increment repeated emotion hits for this memory
            if memory_id not in context['memory_repeated_emotion']:
                context['memory_repeated_emotion'][memory_id] = 1
            else:
                context['memory_repeated_emotion'][memory_id] += 1
                
            # Boost extended score based on repeated hits
            repeated_hits = context['memory_repeated_emotion'][memory_id]
            repeated_boost = min(0.3, 0.1 * repeated_hits)
            extended_score = min(1.0, extended_score + repeated_boost)
        
        return extended_score
    
    async def _apply_shock_absorption(self, 
                                    embedding: torch.Tensor, 
                                    quickrecal_score: float,
                                    context: Dict[str, Any]) -> torch.Tensor:
        """
        Apply shock absorption to smooth embedding changes.
        
        Higher HPC-QR scores allow more embedding adjustment,
        while lower scores dampen changes to prevent noise.
        
        Args:
            embedding: Original embedding
            quickrecal_score: The HPC-QR score
            context: Processing context
            
        Returns:
            Adjusted embedding with shock absorption
        """
        # If no momentum buffer, return original embedding
        if not self.external_momentum or len(self.external_momentum) == 0:
            return embedding
        
        # Get most recent embedding from momentum buffer
        recent_embedding = self.external_momentum[-1]
        if isinstance(recent_embedding, np.ndarray):
            recent_embedding = torch.tensor(
                recent_embedding, device=embedding.device
            )
        elif recent_embedding.device != embedding.device:
            recent_embedding = recent_embedding.to(embedding.device)
        
        # Adjust dimensions if needed
        if recent_embedding.shape[0] != embedding.shape[0]:
            min_dim = min(recent_embedding.shape[0], embedding.shape[0])
            recent_embedding = recent_embedding[:min_dim]
            embedding_adj = embedding[:min_dim]
        else:
            embedding_adj = embedding
        
        # Calculate adjustment rate based on HPC-QR score
        # Higher score = more adjustment allowed
        base_rate = self.config["shock_absorption_rate"]
        adjustment_rate = base_rate * (0.5 + 0.5 * quickrecal_score)
        
        # Apply weighted adjustment
        adjusted = (
            (1.0 - adjustment_rate) * recent_embedding + 
            adjustment_rate * embedding_adj
        )
        
        # Re-normalize
        adjusted = adjusted / torch.norm(adjusted)
        
        # If dimensions were adjusted, pad back to original size
        if adjusted.shape[0] < embedding.shape[0]:
            padded = torch.zeros_like(embedding)
            padded[:adjusted.shape[0]] = adjusted
            adjusted = padded
        
        return adjusted
    
    def _update_momentum_buffer(self, embedding: torch.Tensor) -> None:
        """
        Update external momentum buffer with new embedding.
        
        Implements decay and pruning to keep the buffer size manageable.
        
        Args:
            embedding: New embedding to add to buffer
        """
        # Convert to numpy for consistent storage
        if isinstance(embedding, torch.Tensor):
            emb_np = embedding.cpu().numpy()
        else:
            emb_np = embedding
        
        # Add to buffer with timestamp
        self.external_momentum.append(emb_np)
        self.momentum_timestamps.append(time.time())
        
        # Prune buffer if it exceeds maximum size
        max_size = self.config["momentum_buffer_max_size"]
        if len(self.external_momentum) > max_size:
            # Calculate decay weights based on timestamps
            now = time.time()
            half_life = self.config["momentum_decay_half_life"]
            weights = []
            
            for ts in self.momentum_timestamps:
                age = now - ts
                # Exponential decay
                weight = math.exp(-age * math.log(2) / half_life)
                weights.append(weight)
            
            # Create list of (weight, index) pairs and sort by weight
            weight_idx = [(w, i) for i, w in enumerate(weights)]
            weight_idx.sort(reverse=True)
            
            # Keep top half and anything with a weight above 0.5
            # This preserves both recent and significant items
            keep_indices = set()
            for i, (w, idx) in enumerate(weight_idx):
                if i < max_size // 2 or w > 0.5:
                    keep_indices.add(idx)
            
            # Create new buffer with only kept items
            new_momentum = []
            new_timestamps = []
            
            for i in range(len(self.external_momentum)):
                if i in keep_indices:
                    new_momentum.append(self.external_momentum[i])
                    new_timestamps.append(self.momentum_timestamps[i])
            
            # Update buffer
            self.external_momentum = new_momentum
            self.momentum_timestamps = new_timestamps
    
    def _generate_explanation(self, 
                            factor_contributions: Dict[str, float],
                            quickrecal_score: float) -> str:
        """
        Generate a human-readable explanation of the HPC-QR score.
        
        Args:
            factor_contributions: Dictionary of factor names to their values
            quickrecal_score: Final HPC-QR score
            
        Returns:
            Human-readable explanation string
        """
        # Sort factors by contribution value
        sorted_factors = sorted(
            factor_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Format each factor with its value
        factor_strs = [
            f"{factor} ({value:.2f})"
            for factor, value in sorted_factors
            if value > 0.1  # Only include significant factors
        ]
        
        # Create explanation based on HPC-QR score
        if quickrecal_score >= 0.7:
            significance = "high"
        elif quickrecal_score >= 0.4:
            significance = "moderate"
        else:
            significance = "low"
            
        # Format the factors list
        if factor_strs:
            factor_list = ", ".join(factor_strs)
            explanation = (
                f"Memory has {significance} significance (HPC-QR: {quickrecal_score:.2f}) "
                f"based on: {factor_list}."
            )
        else:
            explanation = (
                f"Memory has {significance} significance (HPC-QR: {quickrecal_score:.2f})."
            )
            
        # Add additional insights if available
        if (str(FactorKeys.CAUSAL_NOVELTY) in factor_contributions and
                factor_contributions[str(FactorKeys.CAUSAL_NOVELTY)] > 0.6):
            explanation += " This memory contains surprising or contradictory information."
            
        if (str(FactorKeys.EXTENDED_EMOTION) in factor_contributions and
                factor_contributions[str(FactorKeys.EXTENDED_EMOTION)] > 0.6):
            explanation += " This memory has strong emotional significance."
            
        if (str(FactorKeys.OVERLAP) in factor_contributions and
                factor_contributions[str(FactorKeys.OVERLAP)] > 0.7):
            explanation += " This memory significantly overlaps with existing memories."
        
        return explanation
    
    def _generate_overshadow_explanation(self, 
                                      factor_contributions: Dict[str, float],
                                      quickrecal_score: float) -> str:
        """
        Generate an explanation for why a memory was overshadowed.
        
        Args:
            factor_contributions: Dictionary of factor names to their values
            quickrecal_score: Final HPC-QR score
            
        Returns:
            Human-readable explanation string
        """
        # Find the main reasons for overshadowing
        reasons = []
        
        # Check for high overlap
        if (str(FactorKeys.OVERLAP) in factor_contributions and
                factor_contributions[str(FactorKeys.OVERLAP)] > 0.5):
            reasons.append(f"high overlap with existing memories ({factor_contributions[str(FactorKeys.OVERLAP)]:.2f})")
            
        # Check for low emotion
        if (str(FactorKeys.EMOTION) in factor_contributions and
                factor_contributions[str(FactorKeys.EMOTION)] < 0.3):
            reasons.append(f"low emotional significance ({factor_contributions[str(FactorKeys.EMOTION)]:.2f})")
            
        # Check for low recency
        if (str(FactorKeys.RECENCY) in factor_contributions and
                factor_contributions[str(FactorKeys.RECENCY)] < 0.3):
            reasons.append(f"not recent ({factor_contributions[str(FactorKeys.RECENCY)]:.2f})")
            
        # Check for low relevance
        if (str(FactorKeys.RELEVANCE) in factor_contributions and
                factor_contributions[str(FactorKeys.RELEVANCE)] < 0.3):
            reasons.append(f"low relevance ({factor_contributions[str(FactorKeys.RELEVANCE)]:.2f})")
            
        # Advanced factors
        if str(FactorKeys.R_GEOMETRY) in factor_contributions:
            if factor_contributions[str(FactorKeys.R_GEOMETRY)] < 0.3:
                reasons.append(f"low geometry significance ({factor_contributions[str(FactorKeys.R_GEOMETRY)]:.2f})")
                
        if str(FactorKeys.CAUSAL_NOVELTY) in factor_contributions:
            if factor_contributions[str(FactorKeys.CAUSAL_NOVELTY)] < 0.3:
                reasons.append(f"low causal novelty ({factor_contributions[str(FactorKeys.CAUSAL_NOVELTY)]:.2f})")
                
        if str(FactorKeys.SELF_ORG) in factor_contributions:
            if factor_contributions[str(FactorKeys.SELF_ORG)] < 0.3:
                reasons.append(f"low self-organization significance ({factor_contributions[str(FactorKeys.SELF_ORG)]:.2f})")
        
        # Format the explanation
        if reasons:
            reason_list = ", ".join(reasons)
            explanation = (
                f"Memory overshadowed (HPC-QR: {quickrecal_score:.2f}) due to: {reason_list}."
            )
        else:
            explanation = (
                f"Memory overshadowed (HPC-QR: {quickrecal_score:.2f}) due to low overall significance."
            )
        
        return explanation
    
    async def run_periodic_tasks(self) -> None:
        """
        Run periodic maintenance tasks like SOM full updates and momentum buffer pruning.
        
        This can be scheduled to run daily or weekly.
        """
        logger.info("Running periodic HPC-QuickRecal maintenance tasks")
        
        # Run a full SOM update
        await self._update_som_full()
        
        # Prune momentum buffer more aggressively
        now = time.time()
        half_life = self.config["momentum_decay_half_life"]
        cutoff_time = now - half_life * 3  # Keep only items within 3 half-lives
        
        # Filter momentum buffer
        new_momentum = []
        new_timestamps = []
        
        for i, ts in enumerate(self.momentum_timestamps):
            if ts >= cutoff_time:
                new_momentum.append(self.external_momentum[i])
                new_timestamps.append(ts)
        
        self.external_momentum = new_momentum
        self.momentum_timestamps = new_timestamps
        
        logger.info(f"Periodic tasks completed. Momentum buffer size: {len(self.external_momentum)}")