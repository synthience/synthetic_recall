import time
import datetime
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import json

from .custom_logger import logger

# Define the current metadata schema version
METADATA_SCHEMA_VERSION = "1.0.0"

class MetadataSynthesizer:
    """
    Enriches memory entries with synthesized metadata derived from content analysis,
    embedding characteristics, and contextual information.
    
    This class serves as a modular pipeline for extracting, computing, and assembling
    metadata fields that add semantic richness to memory entries beyond their raw content.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the MetadataSynthesizer with configuration options.
        
        Args:
            config: Configuration dictionary for customizing metadata synthesis behavior
        """
        self.config = config or {}
        self.metadata_processors = [
            self._process_base_metadata,   # Always process base metadata first (versioning, etc)
            self._process_temporal_metadata,
            self._process_emotional_metadata,
            self._process_cognitive_metadata,
            self._process_embedding_metadata,
            self._process_identifiers_and_basic_stats  # Add identifiers and basic stats processor
        ]
        logger.info("MetadataSynthesizer", "Initialized with processors")
    
    async def synthesize(self, 
                   content: str, 
                   embedding: Optional[np.ndarray] = None,
                   base_metadata: Optional[Dict[str, Any]] = None,
                   emotion_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synthesize rich metadata from content, embedding, and optional existing metadata.
        
        Args:
            content: The text content of the memory
            embedding: Vector representation of the content (optional)
            base_metadata: Existing metadata to build upon (optional)
            emotion_data: Pre-computed emotion analysis results (optional)
            
        Returns:
            Enriched metadata dictionary with synthesized fields
        """
        # Start with base metadata or empty dict
        metadata = base_metadata or {}
        
        # Track original fields to identify what we've added
        original_keys = set(metadata.keys())
        
        # Process through each metadata processor
        context = {
            'content': content,
            'embedding': embedding,
            'emotion_data': emotion_data,
            'original_metadata': base_metadata
        }
        
        # Run all processors
        for processor in self.metadata_processors:
            try:
                processor_result = processor(metadata, context)
                
                # Handle both synchronous and asynchronous processor results
                if processor_result and hasattr(processor_result, '__await__'):
                    metadata = await processor_result
            except Exception as e:
                logger.error("MetadataSynthesizer", f"Error in processor {processor.__name__}: {str(e)}")
        
        # Log what was added
        added_keys = set(metadata.keys()) - original_keys
        logger.info("MetadataSynthesizer", f"Added metadata fields: {list(added_keys)}")
        
        return metadata
    
    def synthesize_sync(self, 
                   content: str, 
                   embedding: Optional[np.ndarray] = None,
                   base_metadata: Optional[Dict[str, Any]] = None,
                   emotion_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synchronous version of synthesize for contexts where async cannot be used.
        
        Args:
            content: The text content of the memory
            embedding: Vector representation of the content (optional)
            base_metadata: Existing metadata to build upon (optional)
            emotion_data: Pre-computed emotion analysis results (optional)
            
        Returns:
            Enriched metadata dictionary with synthesized fields
        """
        # Start with base metadata or empty dict
        metadata = base_metadata or {}
        
        # Track original fields to identify what we've added
        original_keys = set(metadata.keys())
        
        # Process through each metadata processor
        context = {
            'content': content,
            'embedding': embedding,
            'emotion_data': emotion_data,
            'original_metadata': base_metadata
        }
        
        # Run all processors (synchronously)
        for processor in self.metadata_processors:
            try:
                processor_result = processor(metadata, context)
                
                # Since we're in sync mode, we skip any async processors
                if processor_result and not hasattr(processor_result, '__await__'):
                    metadata = processor_result
            except Exception as e:
                logger.error("MetadataSynthesizer", f"Error in processor {processor.__name__}: {str(e)}")
        
        # Log what was added
        added_keys = set(metadata.keys()) - original_keys
        logger.info("MetadataSynthesizer", f"Added metadata fields: {list(added_keys)}")
        
        return metadata
    
    def _process_base_metadata(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add base metadata fields including:
        - metadata_schema_version
        - creation_time
        """
        # Add metadata schema version
        metadata['metadata_schema_version'] = METADATA_SCHEMA_VERSION
        
        # Add creation time
        metadata['creation_time'] = time.time()
        
        return metadata
    
    def _process_temporal_metadata(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add time-related metadata including:
        - timestamp (if not already present)
        - time_of_day (morning, afternoon, evening, night)
        - day_of_week
        - is_weekend
        """
        # Ensure timestamp exists and is a float
        if 'timestamp' not in metadata:
            metadata['timestamp'] = float(time.time())
        else:
            # Ensure timestamp is a float to avoid serialization issues
            try:
                metadata['timestamp'] = float(metadata['timestamp'])
            except (ValueError, TypeError):
                logger.warning("MetadataSynthesizer", f"Invalid timestamp format {metadata['timestamp']}, using current time")
                metadata['timestamp'] = float(time.time())
            
        # Convert timestamp to datetime
        dt = datetime.datetime.fromtimestamp(metadata['timestamp'])
        
        # Add ISO-formatted timestamp for convenience (guarantees serialization compatibility)
        metadata['timestamp_iso'] = dt.isoformat()
        
        # Add temporal markers
        hour = dt.hour
        if 5 <= hour < 12:
            time_of_day = 'morning'
        elif 12 <= hour < 17:
            time_of_day = 'afternoon'
        elif 17 <= hour < 22:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'
            
        # Add temporal metadata
        metadata['time_of_day'] = time_of_day
        metadata['day_of_week'] = dt.strftime('%A').lower()
        metadata['is_weekend'] = dt.weekday() >= 5  # 5 = Saturday, 6 = Sunday
        metadata['month'] = dt.strftime('%B').lower()
        metadata['year'] = dt.year
        
        # Debug log the temporal metadata
        logger.debug("MetadataSynthesizer", "Temporal metadata processed", {
            'timestamp': metadata.get('timestamp'),
            'time_of_day': metadata.get('time_of_day'),
            'day_of_week': metadata.get('day_of_week'),
            'is_weekend': metadata.get('is_weekend')
        })
        
        return metadata
    
    def _process_emotional_metadata(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add emotion-related metadata including:
        - dominant_emotion
        - sentiment_value
        - emotional_intensity
        """
        # Use pre-computed emotion data if available
        emotion_data = context.get('emotion_data')
        
        if emotion_data and isinstance(emotion_data, dict):
            # Extract emotions from the expected location in the emotion_data
            emotions = emotion_data.get('emotions', {})
            
            if isinstance(emotions, dict) and emotions:
                # Ensure we're not duplicating emotion data
                if 'emotions' in metadata:
                    # If emotions already exists in metadata, avoid nesting
                    # Just log it and use the one from emotion_data
                    logger.debug("MetadataSynthesizer", "Emotions already present in metadata, overwriting")
                
                # Copy relevant emotion data to metadata
                if emotions.get('dominant_emotion') is not None:
                    metadata['dominant_emotion'] = emotions.get('dominant_emotion')
                elif 'dominant_emotion' in emotion_data:
                    metadata['dominant_emotion'] = emotion_data.get('dominant_emotion')
                    
                if emotions.get('sentiment_value') is not None:
                    sentiment = emotions.get('sentiment_value')
                    metadata['sentiment_value'] = float(sentiment) # Ensure it's a float
                    # Add a simple polarity label
                    if sentiment > 0.2:
                        metadata['sentiment_polarity'] = 'positive'
                    elif sentiment < -0.2:
                        metadata['sentiment_polarity'] = 'negative'
                    else:
                        metadata['sentiment_polarity'] = 'neutral'
                
                if emotions.get('intensity') is not None:
                    metadata['emotional_intensity'] = float(emotions.get('intensity', 0.5)) # Ensure it's a float
        
        # Ensure mandatory emotional fields are present with safe default values
        if 'dominant_emotion' not in metadata:
            metadata['dominant_emotion'] = 'neutral'  # Default
        
        if 'sentiment_polarity' not in metadata:
            metadata['sentiment_polarity'] = 'neutral' # Default
            
        if 'sentiment_value' not in metadata:
            metadata['sentiment_value'] = 0.0  # Default neutral sentiment
            
        if 'emotional_intensity' not in metadata:
            metadata['emotional_intensity'] = 0.5  # Default (medium intensity)
            
        # Debug log the final emotional metadata
        logger.debug("MetadataSynthesizer", "Emotional metadata processed", {
            'dominant_emotion': metadata.get('dominant_emotion'),
            'sentiment_polarity': metadata.get('sentiment_polarity'),
            'emotional_intensity': metadata.get('emotional_intensity')
        })
            
        return metadata
    
    def _process_cognitive_metadata(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add cognitive-related metadata including:
        - complexity_estimate
        - word_count
        - cognitive_load_estimate
        """
        content = context.get('content', '')
        
        # Simple metrics based on content
        word_count = len(content.split())
        metadata['word_count'] = word_count
        
        # Estimate complexity (very simple heuristic)
        avg_word_length = sum(len(word) for word in content.split()) / max(1, word_count)
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        sentence_count = max(1, sentence_count)  # Avoid division by zero
        
        words_per_sentence = word_count / sentence_count
        
        # Simplified complexity score (0-1 range)
        complexity = min(1.0, ((avg_word_length / 10) + (words_per_sentence / 25)) / 2)
        metadata['complexity_estimate'] = float(complexity)
        
        # Cognitive load is a factor of complexity and length
        cognitive_load = min(1.0, (complexity * 0.7) + (min(1.0, word_count / 500) * 0.3))
        metadata['cognitive_load_estimate'] = float(cognitive_load)
        
        return metadata
    
    def _process_embedding_metadata(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from embedding characteristics:
        - embedding_norm
        - embedding_sparsity
        - embedding_dim
        - embedding_valid
        """
        embedding = context.get('embedding')
        
        if embedding is not None:
            # Extract embedding characteristics
            try:
                # First validate the embedding
                embedding, is_valid = self._validate_embedding(embedding)
                metadata['embedding_valid'] = is_valid
                
                # Calculate embedding norm (magnitude)
                embedding_norm = float(np.linalg.norm(embedding))
                metadata['embedding_norm'] = embedding_norm
                
                # Calculate sparsity (percent of near-zero values)
                near_zero = np.abs(embedding) < 0.01
                sparsity = float(np.mean(near_zero))
                metadata['embedding_sparsity'] = sparsity
                
                # Store embedding dimension
                metadata['embedding_dim'] = embedding.shape[0]
                
                # Log the embedding metadata
                logger.debug("MetadataSynthesizer", "Embedding metadata processed", {
                    'valid': metadata.get('embedding_valid'),
                    'norm': metadata.get('embedding_norm'),
                    'sparsity': metadata.get('embedding_sparsity'),
                    'dim': metadata.get('embedding_dim')
                })
            except Exception as e:
                logger.warning("MetadataSynthesizer", f"Error processing embedding metadata: {str(e)}")
                metadata['embedding_valid'] = False
        else:
            # No embedding available
            metadata['embedding_valid'] = False
        
        return metadata
        
    def _validate_embedding(self, embedding: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Validate an embedding vector and replace with zeros if invalid.
        
        Args:
            embedding: The embedding vector to validate
            
        Returns:
            Tuple of (possibly_fixed_embedding, is_valid)
        """
        # Check for None
        if embedding is None:
            return np.zeros(768), False
            
        # Convert to numpy array if not already
        if not isinstance(embedding, np.ndarray):
            try:
                embedding = np.array(embedding, dtype=np.float32)
            except Exception:
                return np.zeros(768), False
        
        # Check for NaN or Inf values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            logger.warning("MetadataSynthesizer", "Embedding contains NaN or Inf values, replacing with zeros")
            # Create a zero vector of the same shape
            return np.zeros_like(embedding), False
            
        # Check if the vector is all zeros
        if np.all(embedding == 0):
            return embedding, False
            
        return embedding, True
        
    def _align_vectors_for_comparison(self, vec1: np.ndarray, vec2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two vectors to the same dimension for comparison operations.
        Will pad the smaller vector with zeros or truncate the larger one.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Tuple of (aligned_vec1, aligned_vec2)
        """
        # Make sure both are numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1, dtype=np.float32)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2, dtype=np.float32)
            
        # Get dimensions
        dim1 = vec1.shape[0]
        dim2 = vec2.shape[0]
        
        # If dimensions match, no alignment needed
        if dim1 == dim2:
            return vec1, vec2
            
        # Need to align dimensions
        if dim1 < dim2:
            # Pad vec1 with zeros
            aligned_vec1 = np.zeros(dim2, dtype=np.float32)
            aligned_vec1[:dim1] = vec1
            return aligned_vec1, vec2
        else:
            # Pad vec2 with zeros
            aligned_vec2 = np.zeros(dim1, dtype=np.float32)
            aligned_vec2[:dim2] = vec2
            return vec1, aligned_vec2

    def _process_identifiers_and_basic_stats(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds memory ID (uuid) and content length if available.
        This should run after base metadata and before final memory entry creation.
        """
        content = context.get('content', '')
        
        # Add length (raw character count)
        if 'length' not in metadata:
            metadata['length'] = len(content)
        
        # NOTE: 'uuid' (aka memory_id) must be passed externally if you want it included,
        # or you can let the core insert it *after* memory creation if needed.
        
        return metadata
