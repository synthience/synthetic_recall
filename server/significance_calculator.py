"""
LUCID RECALL PROJECT
Unified Significance Calculator

Agent: Lucidia 1.1
Date: 05/03/25
Time: 4:43 PM EST
A standardized significance calculator for consistent memory importance
assessment across all memory system components.
"""

import time
import logging
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class SignificanceMode(Enum):
    """Operating modes for significance calculation."""
    STANDARD = "standard"       # Balanced approach for general use
    PRECISE = "precise"         # More elaborate calculation for higher quality
    EFFICIENT = "efficient"     # Simplified calculation for speed
    EMOTIONAL = "emotional"     # Prioritizes emotional content
    INFORMATIONAL = "informational"  # Prioritizes information density
    PERSONAL = "personal"       # Prioritizes personal relevance
    CUSTOM = "custom"           # Uses custom weights

class SignificanceComponent(Enum):
    """Components that contribute to significance calculation."""
    SURPRISE = "surprise"             # Novelty of information
    DIVERSITY = "diversity"           # Uniqueness compared to existing memories
    EMOTION = "emotion"               # Emotional intensity
    RECENCY = "recency"               # Temporal relevance
    IMPORTANCE = "importance"         # Explicit importance markers
    PERSONAL = "personal"             # Personal information relevance
    COHERENCE = "coherence"           # Logical consistency
    INFORMATION = "information"       # Information density
    RELEVANCE = "relevance"           # Contextual relevance
    USER_ATTENTION = "user_attention" # User engagement signals

class UnifiedSignificanceCalculator:
    """
    Unified significance calculator for consistent assessment of memory importance.
    
    This class provides a standardized approach to calculating memory significance
    that can be used across all memory system components. It supports multiple
    modes and can be customized with different weights for different factors.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize significance calculator.
        
        Args:
            config: Configuration options including:
                - mode: SignificanceMode for calculation approach
                - component_weights: Custom weights for different components
                - time_decay_rate: Rate at which significance decays over time
                - surprise_threshold: Threshold for considering something surprising
                - min_significance: Minimum significance value
                - max_significance: Maximum significance value
                - adaptive_thresholds: Whether to adjust thresholds based on data
                - history_window: Number of samples to keep for adaptive thresholds
        """
        self.config = {
            'mode': SignificanceMode.STANDARD,
            'component_weights': {},
            'time_decay_rate': 0.1,
            'surprise_threshold': 0.7,
            'min_significance': 0.0,
            'max_significance': 1.0,
            'adaptive_thresholds': True,
            'history_window': 1000,
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
            'informational_prefixes': [
                'fact:', 'important:', 'remember:', 'note:', 'key point:',
                'critical:', 'essential:', 'reminder:', 'don\'t forget:'
            ],
            **(config or {})
        }
        
        # Initialize component weights based on mode
        self._init_component_weights()
        
        # Set mode - convert from string if needed
        if isinstance(self.config['mode'], str):
            try:
                self.config['mode'] = SignificanceMode(self.config['mode'].lower())
            except ValueError:
                logger.warning(f"Invalid mode: {self.config['mode']}, using STANDARD")
                self.config['mode'] = SignificanceMode.STANDARD
                
        # History for adaptive thresholds
        self.history = {
            'calculated_significance': [],
            'component_values': {comp: [] for comp in SignificanceComponent},
            'timestamps': []
        }
        
        # Tracking
        self.total_calculations = 0
        self.start_time = time.time()
        self.last_calculation_time = 0
        
        logger.info(f"Initialized UnifiedSignificanceCalculator with mode: {self.config['mode'].value}")
        
    def _init_component_weights(self) -> None:
        """Initialize component weights based on selected mode."""
        # Default weights for STANDARD mode
        standard_weights = {
            SignificanceComponent.SURPRISE: 0.20,
            SignificanceComponent.DIVERSITY: 0.15,
            SignificanceComponent.EMOTION: 0.15,
            SignificanceComponent.RECENCY: 0.10,
            SignificanceComponent.IMPORTANCE: 0.15,
            SignificanceComponent.PERSONAL: 0.15,
            SignificanceComponent.COHERENCE: 0.05,
            SignificanceComponent.INFORMATION: 0.05,
            SignificanceComponent.RELEVANCE: 0.05,
            SignificanceComponent.USER_ATTENTION: 0.00  # Not used by default
        }
        
        # Mode-specific weights
        mode_weights = {
            SignificanceMode.PRECISE: {
                # More thorough analysis with all components
                SignificanceComponent.SURPRISE: 0.15,
                SignificanceComponent.DIVERSITY: 0.15,
                SignificanceComponent.EMOTION: 0.15,
                SignificanceComponent.RECENCY: 0.05,
                SignificanceComponent.IMPORTANCE: 0.15,
                SignificanceComponent.PERSONAL: 0.15,
                SignificanceComponent.COHERENCE: 0.10,
                SignificanceComponent.INFORMATION: 0.05,
                SignificanceComponent.RELEVANCE: 0.05,
                SignificanceComponent.USER_ATTENTION: 0.00
            },
            SignificanceMode.EFFICIENT: {
                # Focus on most important factors for speed
                SignificanceComponent.SURPRISE: 0.25,
                SignificanceComponent.DIVERSITY: 0.20,
                SignificanceComponent.EMOTION: 0.00,
                SignificanceComponent.RECENCY: 0.15,
                SignificanceComponent.IMPORTANCE: 0.20,
                SignificanceComponent.PERSONAL: 0.20,
                SignificanceComponent.COHERENCE: 0.00,
                SignificanceComponent.INFORMATION: 0.00,
                SignificanceComponent.RELEVANCE: 0.00,
                SignificanceComponent.USER_ATTENTION: 0.00
            },
            SignificanceMode.EMOTIONAL: {
                # Prioritize emotional content
                SignificanceComponent.SURPRISE: 0.15,
                SignificanceComponent.DIVERSITY: 0.10,
                SignificanceComponent.EMOTION: 0.35,
                SignificanceComponent.RECENCY: 0.05,
                SignificanceComponent.IMPORTANCE: 0.10,
                SignificanceComponent.PERSONAL: 0.15,
                SignificanceComponent.COHERENCE: 0.00,
                SignificanceComponent.INFORMATION: 0.05,
                SignificanceComponent.RELEVANCE: 0.05,
                SignificanceComponent.USER_ATTENTION: 0.00
            },
            SignificanceMode.INFORMATIONAL: {
                # Prioritize information density
                SignificanceComponent.SURPRISE: 0.20,
                SignificanceComponent.DIVERSITY: 0.15,
                SignificanceComponent.EMOTION: 0.05,
                SignificanceComponent.RECENCY: 0.05,
                SignificanceComponent.IMPORTANCE: 0.20,
                SignificanceComponent.PERSONAL: 0.05,
                SignificanceComponent.COHERENCE: 0.10,
                SignificanceComponent.INFORMATION: 0.15,
                SignificanceComponent.RELEVANCE: 0.05,
                SignificanceComponent.USER_ATTENTION: 0.00
            },
            SignificanceMode.PERSONAL: {
                # Prioritize personal information
                SignificanceComponent.SURPRISE: 0.10,
                SignificanceComponent.DIVERSITY: 0.10,
                SignificanceComponent.EMOTION: 0.10,
                SignificanceComponent.RECENCY: 0.05,
                SignificanceComponent.IMPORTANCE: 0.10,
                SignificanceComponent.PERSONAL: 0.40,
                SignificanceComponent.COHERENCE: 0.05,
                SignificanceComponent.INFORMATION: 0.05,
                SignificanceComponent.RELEVANCE: 0.05,
                SignificanceComponent.USER_ATTENTION: 0.00
            }
        }
        
        # Set weights based on mode
        mode = self.config['mode']
        if isinstance(mode, str):
            try:
                mode = SignificanceMode(mode.lower())
            except ValueError:
                mode = SignificanceMode.STANDARD
        
        weights = mode_weights.get(mode, standard_weights)
        
        # Override with custom weights if provided
        if mode == SignificanceMode.CUSTOM and self.config['component_weights']:
            weights.update(self.config['component_weights'])
        
        # Store final weights
        self.component_weights = weights
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.component_weights = {comp: weight / total_weight for comp, weight in weights.items()}
    
    async def calculate(self, 
                      embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
                      text: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate significance score for a memory.
        
        Args:
            embedding: Vector representation of memory content
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Significance score between 0.0 and 1.0
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Calculate component values
            component_values = {}
            
            # 1. Calculate surprise component if embedding provided
            if embedding is not None:
                component_values[SignificanceComponent.SURPRISE] = await self._calculate_surprise(embedding, context)
                component_values[SignificanceComponent.DIVERSITY] = await self._calculate_diversity(embedding, context)
            else:
                component_values[SignificanceComponent.SURPRISE] = 0.0
                component_values[SignificanceComponent.DIVERSITY] = 0.0
                
            # 2. Calculate text-based components if text provided
            if text:
                component_values[SignificanceComponent.EMOTION] = self._calculate_emotion(text, context)
                component_values[SignificanceComponent.IMPORTANCE] = self._calculate_importance(text, context)
                component_values[SignificanceComponent.PERSONAL] = self._calculate_personal(text, context)
                component_values[SignificanceComponent.INFORMATION] = self._calculate_information(text, context)
                component_values[SignificanceComponent.COHERENCE] = self._calculate_coherence(text, context)
            else:
                component_values[SignificanceComponent.EMOTION] = 0.0
                component_values[SignificanceComponent.IMPORTANCE] = 0.0
                component_values[SignificanceComponent.PERSONAL] = 0.0
                component_values[SignificanceComponent.INFORMATION] = 0.0
                component_values[SignificanceComponent.COHERENCE] = 0.0
                
            # 3. Calculate context-based components
            component_values[SignificanceComponent.RECENCY] = self._calculate_recency(context)
            component_values[SignificanceComponent.RELEVANCE] = self._calculate_relevance(context)
            component_values[SignificanceComponent.USER_ATTENTION] = self._calculate_user_attention(context)
            
            # 4. Apply weights to components
            weighted_sum = 0.0
            for component, value in component_values.items():
                weight = self.component_weights.get(component, 0.0)
                weighted_sum += weight * value
                
            # 5. Apply time decay
            time_decay = self._calculate_time_decay(context)
            significance = weighted_sum * time_decay
            
            # 6. Apply sigmoid function to ensure value is between 0-1
            significance = 1.0 / (1.0 + np.exp(-5.0 * (significance - 0.5)))
            
            # 7. Clamp to configured min/max range
            significance = max(self.config['min_significance'], 
                             min(self.config['max_significance'], significance))
            
            # 8. Update history for adaptive thresholds
            if self.config['adaptive_thresholds']:
                self._update_history(significance, component_values)
            
            # Update tracking
            self.total_calculations += 1
            self.last_calculation_time = time.time()
            
            logger.debug(f"Calculated significance: {significance:.4f} in {(time.time() - start_time)*1000:.2f}ms")
            
            return float(significance)
            
        except Exception as e:
            logger.error(f"Error calculating significance: {e}")
            # Return default value on error
            return 0.5
    
    async def _calculate_surprise(self, 
                                embedding: Union[np.ndarray, torch.Tensor, List[float]], 
                                context: Dict[str, Any]) -> float:
        """
        Calculate surprise component.
        
        Args:
            embedding: Vector representation of memory content
            context: Additional contextual information
            
        Returns:
            Surprise score between 0.0 and 1.0
        """
        # If history available in context, use it
        if "embedding_history" in context and context["embedding_history"]:
            history = context["embedding_history"]
            
            # Process embedding
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            elif isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
                
            # Ensure embedding is normalized
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            # Calculate similarities to history
            similarities = []
            for hist_emb in history:
                if isinstance(hist_emb, torch.Tensor):
                    hist_emb = hist_emb.detach().cpu().numpy()
                elif isinstance(hist_emb, list):
                    hist_emb = np.array(hist_emb, dtype=np.float32)
                    
                # Ensure history embedding is normalized
                hist_norm = np.linalg.norm(hist_emb)
                if hist_norm > 0:
                    hist_emb = hist_emb / hist_norm
                    
                # Calculate cosine similarity
                similarity = np.dot(embedding, hist_emb)
                similarities.append(similarity)
                
            if similarities:
                # Higher surprise = lower similarity
                avg_similarity = np.mean(similarities)
                surprise = 1.0 - avg_similarity
                
                # Adjust based on threshold
                surprise = max(0.0, (surprise - self.config['surprise_threshold'])) / (1.0 - self.config['surprise_threshold'])
                return min(1.0, surprise)
                
        # Default surprise if no history or not enough context
        return 0.5
    
    async def _calculate_diversity(self, 
                                 embedding: Union[np.ndarray, torch.Tensor, List[float]], 
                                 context: Dict[str, Any]) -> float:
        """
        Calculate diversity component.
        
        Args:
            embedding: Vector representation of memory content
            context: Additional contextual information
            
        Returns:
            Diversity score between 0.0 and 1.0
        """
        # Similar to surprise but focuses on maximum similarity rather than average
        if "embedding_history" in context and context["embedding_history"]:
            history = context["embedding_history"]
            
            # Process embedding
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            elif isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
                
            # Ensure embedding is normalized
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            # Calculate similarities to history
            similarities = []
            for hist_emb in history:
                if isinstance(hist_emb, torch.Tensor):
                    hist_emb = hist_emb.detach().cpu().numpy()
                elif isinstance(hist_emb, list):
                    hist_emb = np.array(hist_emb, dtype=np.float32)
                    
                # Ensure history embedding is normalized
                hist_norm = np.linalg.norm(hist_emb)
                if hist_norm > 0:
                    hist_emb = hist_emb / hist_norm
                    
                # Calculate cosine similarity
                similarity = np.dot(embedding, hist_emb)
                similarities.append(similarity)
                
            if similarities:
                # Higher diversity = lower maximum similarity
                max_similarity = max(similarities)
                diversity = 1.0 - max_similarity
                
                return min(1.0, diversity)
                
        # Default diversity if no history or not enough context
        return 0.5
    
    def _calculate_emotion(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate emotion component based on text content.
        
        Args:
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Emotion score between 0.0 and 1.0
        """
        # Check for emotional keywords
        emotional_keywords = self.config['emotional_keywords']
        
        # Count emotional keywords in text
        text_lower = text.lower()
        emotion_count = sum(1 for keyword in emotional_keywords if keyword in text_lower)
        
        # Normalize count
        normalized_count = min(1.0, emotion_count / 5.0)  # Cap at 5 emotional keywords
        
        # Check for explicit emotion markers in context
        emotion_markers = context.get('emotion_markers', {})
        explicit_emotion = emotion_markers.get('intensity', 0.0)
        
        # Combine text-based and explicit emotion
        emotion_score = max(normalized_count, explicit_emotion)
        
        return emotion_score
    
    def _calculate_importance(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate importance component based on importance markers.
        
        Args:
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        # Check for informational prefixes like "important:" or "note:"
        text_lower = text.lower()
        prefix_matches = any(text_lower.startswith(prefix) for prefix in self.config['informational_prefixes'])
        
        # Check for explicit importance in context
        explicit_importance = context.get('importance', 0.0)
        
        # Check for imperative verbs that suggest importance
        imperative_markers = ['must', 'should', 'need to', 'have to', 'remember', 'don\'t forget']
        imperative_match = any(marker in text_lower for marker in imperative_markers)
        
        # Calculate importance score
        importance_score = max(
            float(prefix_matches) * 0.8,
            explicit_importance,
            float(imperative_match) * 0.6
        )
        
        return importance_score
    
    def _calculate_personal(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate personal component based on personal information content.
        
        Args:
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Personal score between 0.0 and 1.0
        """
        # Check for personal information keywords
        personal_keywords = self.config['personal_information_keywords']
        
        # Count personal keywords in text
        text_lower = text.lower()
        personal_count = sum(1 for keyword in personal_keywords if keyword in text_lower)
        
        # Normalize count
        normalized_count = min(1.0, personal_count / 3.0)  # Cap at 3 personal keywords
        
        # Check for first-person pronouns which indicate personal information
        first_person_pronouns = ['i ', 'me ', 'my ', 'mine ', 'we ', 'us ', 'our ', 'ours ']
        pronoun_count = sum(text_lower.count(pronoun) for pronoun in first_person_pronouns)
        pronoun_score = min(1.0, pronoun_count / 10.0)  # Cap at 10 pronouns
        
        # Check for explicit personal markers in context
        explicit_personal = context.get('personal_relevance', 0.0)
        
        # Combine scores
        personal_score = max(normalized_count, pronoun_score * 0.7, explicit_personal)
        
        return personal_score
    
    def _calculate_information(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate information density component.
        
        Args:
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Information score between 0.0 and 1.0
        """
        # Basic estimation of information density based on text length
        # Longer text tends to contain more information, but with diminishing returns
        token_count = len(text.split())
        normalized_length = min(1.0, token_count / 100.0)  # Cap at 100 tokens
        
        # Modify based on structural elements like lists or formatting
        list_items = text.count('\n- ')
        has_lists = list_items > 0
        list_bonus = min(0.3, list_items / 10.0)  # Bonus for structured lists, cap at 0.3
        
        # Check for numerical content (dates, quantities, etc.)
        import re
        numerical_content = len(re.findall(r'\d+', text))
        numerical_score = min(0.3, numerical_content / 10.0)  # Cap at 0.3
        
        # Combine scores
        information_score = normalized_length + (list_bonus if has_lists else 0) + numerical_score
        
        return min(1.0, information_score)
    
    def _calculate_coherence(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate coherence component based on text structure.
        
        Args:
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Coherence score between 0.0 and 1.0
        """
        # Simple coherence estimation based on sentence structure
        # More sophisticated NLP would be better but beyond scope
        
        # Check sentence length distribution (extreme variation suggests incoherence)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.5  # Default for empty text
            
        # Calculate sentence length stats
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        
        # Large variation in sentence length suggests less coherence
        if len(sentence_lengths) > 1:
            variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
            std_dev = variance ** 0.5
            length_coherence = max(0.0, 1.0 - (std_dev / avg_length))
        else:
            length_coherence = 0.5  # Default for single sentence
            
        # Check for connection words that suggest coherent structure
        connection_words = ['therefore', 'thus', 'because', 'since', 'so', 'as a result', 
                          'consequently', 'furthermore', 'moreover', 'in addition']
        text_lower = text.lower()
        connection_count = sum(text_lower.count(word) for word in connection_words)
        connection_score = min(0.5, connection_count / 5.0)  # Cap at 0.5
        
        # Combine scores
        coherence_score = 0.5 * length_coherence + 0.5 * connection_score
        
        return coherence_score
    
    def _calculate_recency(self, context: Dict[str, Any]) -> float:
        """
        Calculate recency component based on time.
        
        Args:
            context: Additional contextual information
            
        Returns:
            Recency score between 0.0 and 1.0
        """
        # Get timestamp from context or use current time
        timestamp = context.get('timestamp', time.time())
        current_time = time.time()
        
        # Calculate time elapsed
        elapsed_seconds = max(0, current_time - timestamp)
        
        # Convert to days
        elapsed_days = elapsed_seconds / (24 * 3600)
        
        # Apply exponential decay
        decay_rate = self.config['time_decay_rate']
        recency = np.exp(-decay_rate * elapsed_days)
        
        return recency
    
    def _calculate_relevance(self, context: Dict[str, Any]) -> float:
        """
        Calculate relevance component based on context.
        
        Args:
            context: Additional contextual information
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Use explicit relevance if provided
        if 'relevance' in context:
            return min(1.0, max(0.0, context['relevance']))
            
        # Use similarity to query if provided
        if 'query_similarity' in context:
            return min(1.0, max(0.0, context['query_similarity']))
            
        # Default relevance
        return 0.5
    
    def _calculate_user_attention(self, context: Dict[str, Any]) -> float:
        """
        Calculate user attention component based on interaction signals.
        
        Args:
            context: Additional contextual information
            
        Returns:
            User attention score between 0.0 and 1.0
        """
        # Check for user interaction markers
        interaction_markers = context.get('user_interaction', {})
        
        # Different types of attention signals
        explicit_focus = interaction_markers.get('explicit_focus', 0.0)
        repeat_count = interaction_markers.get('repeat_count', 0)
        dwell_time = interaction_markers.get('dwell_time', 0.0)
        
        # Normalize signals
        normalized_repeat = min(1.0, repeat_count / 3.0)  # Cap at 3 repeats
        normalized_dwell = min(1.0, dwell_time / 30.0)    # Cap at 30 seconds
        
        # Combine signals
        attention_score = max(explicit_focus, normalized_repeat, normalized_dwell)
        
        return attention_score
    
    def _calculate_time_decay(self, context: Dict[str, Any]) -> float:
        """
        Calculate time decay factor.
        
        Args:
            context: Additional contextual information
            
        Returns:
            Time decay factor between 0.0 and 1.0
        """
        # Get timestamp from context or use current time
        timestamp = context.get('timestamp', time.time())
        current_time = time.time()
        
        # For very recent memories, don't apply decay
        if current_time - timestamp < 60:  # Less than a minute old
            return 1.0
            
        # For older memories, apply configurable decay
        elapsed_days = (current_time - timestamp) / (24 * 3600)
        
        # Different modes have different decay profiles
        if self.config['mode'] == SignificanceMode.PRECISE:
            # Slower decay for precise mode
            decay_rate = self.config['time_decay_rate'] * 0.5
        elif self.config['mode'] == SignificanceMode.EFFICIENT:
            # Faster decay for efficient mode
            decay_rate = self.config['time_decay_rate'] * 1.5
        else:
            # Standard decay rate
            decay_rate = self.config['time_decay_rate']
            
        # Apply exponential decay
        decay_factor = np.exp(-decay_rate * elapsed_days)
        
        # Ensure minimum decay factor based on memory importance
        min_decay = context.get('min_decay_factor', 0.1)
        decay_factor = max(min_decay, decay_factor)
        
        return decay_factor
    
    def _update_history(self, significance: float, component_values: Dict[SignificanceComponent, float]) -> None:
        """
        Update history for adaptive thresholds.
        
        Args:
            significance: Calculated significance score
            component_values: Individual component values
        """
        # Add to history
        self.history['calculated_significance'].append(significance)
        self.history['timestamps'].append(time.time())
        
        for component, value in component_values.items():
            if component in self.history['component_values']:
                self.history['component_values'][component].append(value)
                
        # Trim history if needed
        if len(self.history['calculated_significance']) > self.config['history_window']:
            self.history['calculated_significance'] = self.history['calculated_significance'][-self.config['history_window']:]
            self.history['timestamps'] = self.history['timestamps'][-self.config['history_window']:]
            
            for component in self.history['component_values']:
                if len(self.history['component_values'][component]) > self.config['history_window']:
                    self.history['component_values'][component] = self.history['component_values'][component][-self.config['history_window']:]
    
    def update_adaptive_thresholds(self) -> None:
        """Update adaptive thresholds based on history."""
        if not self.history['calculated_significance']:
            return
            
        # Calculate significance distribution
        significance_values = np.array(self.history['calculated_significance'])
        
        # Adjust surprise threshold based on historical distribution
        if SignificanceComponent.SURPRISE in self.history['component_values'] and len(self.history['component_values'][SignificanceComponent.SURPRISE]) > 10:
            surprise_values = np.array(self.history['component_values'][SignificanceComponent.SURPRISE])
            
            # Set threshold at 70th percentile
            self.config['surprise_threshold'] = np.percentile(surprise_values, 70)
            
        logger.debug(f"Updated adaptive thresholds: surprise_threshold={self.config['surprise_threshold']:.2f}")
    
    def set_mode(self, mode: Union[str, SignificanceMode]) -> None:
        """
        Set the calculation mode.
        
        Args:
            mode: New mode as string or SignificanceMode enum
        """
        if isinstance(mode, str):
            try:
                mode = SignificanceMode(mode.lower())
            except ValueError:
                logger.warning(f"Invalid mode: {mode}, using STANDARD")
                mode = SignificanceMode.STANDARD
        
        self.config['mode'] = mode
        self._init_component_weights()
        logger.info(f"Significance calculator mode set to: {mode.value}")
    
    def set_component_weights(self, weights: Dict[Union[str, SignificanceComponent], float]) -> None:
        """
        Set custom component weights.
        
        Args:
            weights: Dictionary mapping components to weights
        """
        # Convert string keys to SignificanceComponent enum
        component_weights = {}
        for key, value in weights.items():
            if isinstance(key, str):
                try:
                    component = SignificanceComponent[key.upper()]
                    component_weights[component] = value
                except KeyError:
                    logger.warning(f"Unknown component: {key}, ignoring")
            else:
                component_weights[key] = value
        
        # Set custom weights
        self.config['component_weights'] = component_weights
        
        # Set mode to CUSTOM
        self.config['mode'] = SignificanceMode.CUSTOM
        
        # Reinitialize weights
        self._init_component_weights()
        
        logger.info("Custom component weights set")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get significance calculator statistics."""
        # Calculate average significance and component values
        avg_significance = np.mean(self.history['calculated_significance']) if self.history['calculated_significance'] else 0.0
        
        component_stats = {}
        for component in SignificanceComponent:
            values = self.history['component_values'].get(component, [])
            if values:
                component_stats[component.value] = {
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'weight': self.component_weights.get(component, 0.0)
                }
        
        return {
            'mode': self.config['mode'].value,
            'total_calculations': self.total_calculations,
            'avg_significance': avg_significance,
            'surprise_threshold': self.config['surprise_threshold'],
            'time_decay_rate': self.config['time_decay_rate'],
            'adaptive_thresholds': self.config['adaptive_thresholds'],
            'history_size': len(self.history['calculated_significance']),
            'components': component_stats
        }