import time
import datetime
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import json

from .custom_logger import logger
from .geometry_manager import GeometryManager

# Define the current metadata schema version
METADATA_SCHEMA_VERSION = "1.0.0"

class MetadataSynthesizer:
    """
    Enriches memory entries with synthesized metadata derived from content analysis,
    embedding characteristics, and contextual information.
    
    This class serves as a modular pipeline for extracting, computing, and assembling
    metadata fields that add semantic richness to memory entries beyond their raw content.
    """
    
    def __init__(self, config: Dict[str, Any] = None, geometry_manager: Optional[GeometryManager] = None):
        """
        Initialize the MetadataSynthesizer with configuration options.
        
        Args:
            config: Configuration dictionary for customizing metadata synthesis behavior
            geometry_manager: Instance of GeometryManager for embedding validation/alignment
        """
        self.config = config or {}
        if geometry_manager is None:
            logger.warning("MetadataSynthesizer", "GeometryManager not provided, creating default.")
            self.geometry_manager = GeometryManager()
        else:
            self.geometry_manager = geometry_manager
        
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
        metadata = base_metadata or {}
        
        original_keys = set(metadata.keys())
        
        context = {
            'content': content,
            'embedding': embedding,
            'emotion_data': emotion_data,
            'original_metadata': base_metadata
        }
        
        for processor in self.metadata_processors:
            try:
                processor_result = processor(metadata, context)
                
                if processor_result and hasattr(processor_result, '__await__'):
                    metadata = await processor_result
            except Exception as e:
                logger.error("MetadataSynthesizer", f"Error in processor {processor.__name__}: {str(e)}")
        
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
        metadata = base_metadata or {}
        
        original_keys = set(metadata.keys())
        
        context = {
            'content': content,
            'embedding': embedding,
            'emotion_data': emotion_data,
            'original_metadata': base_metadata
        }
        
        for processor in self.metadata_processors:
            try:
                processor_result = processor(metadata, context)
                
                if processor_result and not hasattr(processor_result, '__await__'):
                    metadata = processor_result
            except Exception as e:
                logger.error("MetadataSynthesizer", f"Error in processor {processor.__name__}: {str(e)}")
        
        added_keys = set(metadata.keys()) - original_keys
        logger.info("MetadataSynthesizer", f"Added metadata fields: {list(added_keys)}")
        
        return metadata
    
    def _process_base_metadata(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add base metadata fields including:
        - metadata_schema_version
        - creation_time
        """
        metadata['metadata_schema_version'] = METADATA_SCHEMA_VERSION
        
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
        if 'timestamp' not in metadata:
            metadata['timestamp'] = float(time.time())
        else:
            try:
                metadata['timestamp'] = float(metadata['timestamp'])
            except (ValueError, TypeError):
                logger.warning("MetadataSynthesizer", f"Invalid timestamp format {metadata['timestamp']}, using current time")
                metadata['timestamp'] = float(time.time())
            
        dt = datetime.datetime.fromtimestamp(metadata['timestamp'])
        
        metadata['timestamp_iso'] = dt.isoformat()
        
        hour = dt.hour
        if 5 <= hour < 12:
            time_of_day = 'morning'
        elif 12 <= hour < 17:
            time_of_day = 'afternoon'
        elif 17 <= hour < 22:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'
            
        metadata['time_of_day'] = time_of_day
        metadata['day_of_week'] = dt.strftime('%A').lower()
        metadata['is_weekend'] = dt.weekday() >= 5  # 5 = Saturday, 6 = Sunday
        metadata['month'] = dt.strftime('%B').lower()
        metadata['year'] = dt.year
        
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
        emotion_data = context.get('emotion_data')
        
        if emotion_data and isinstance(emotion_data, dict):
            emotions = emotion_data.get('emotions', {})
            
            if isinstance(emotions, dict) and emotions:
                if 'emotions' in metadata:
                    logger.debug("MetadataSynthesizer", "Emotions already present in metadata, overwriting")
                
                if emotions.get('dominant_emotion') is not None:
                    metadata['dominant_emotion'] = emotions.get('dominant_emotion')
                elif 'dominant_emotion' in emotion_data:
                    metadata['dominant_emotion'] = emotion_data.get('dominant_emotion')
                    
                if emotions.get('sentiment_value') is not None:
                    sentiment = emotions.get('sentiment_value')
                    metadata['sentiment_value'] = float(sentiment) 
                    if sentiment > 0.2:
                        metadata['sentiment_polarity'] = 'positive'
                    elif sentiment < -0.2:
                        metadata['sentiment_polarity'] = 'negative'
                    else:
                        metadata['sentiment_polarity'] = 'neutral'
                
                if emotions.get('intensity') is not None:
                    metadata['emotional_intensity'] = float(emotions.get('intensity', 0.5)) 
        
        if 'dominant_emotion' not in metadata:
            metadata['dominant_emotion'] = 'neutral'  
        
        if 'sentiment_polarity' not in metadata:
            metadata['sentiment_polarity'] = 'neutral' 
            
        if 'sentiment_value' not in metadata:
            metadata['sentiment_value'] = 0.0  
            
        if 'emotional_intensity' not in metadata:
            metadata['emotional_intensity'] = 0.5  
            
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
        
        word_count = len(content.split())
        metadata['word_count'] = word_count
        
        avg_word_length = sum(len(word) for word in content.split()) / max(1, word_count)
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        sentence_count = max(1, sentence_count)  
        
        words_per_sentence = word_count / sentence_count
        
        complexity = min(1.0, ((avg_word_length / 10) + (words_per_sentence / 25)) / 2)
        metadata['complexity_estimate'] = float(complexity)
        
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
            try:
                # Use the correct validation method
                validated_embedding = self.geometry_manager._validate_vector(embedding, "Embedding for Metadata")
                is_valid = validated_embedding is not None
                metadata['embedding_valid'] = is_valid
                
                # Use the validated embedding if available, otherwise fall back to original
                embedding_to_use = validated_embedding if is_valid else embedding
                
                embedding_norm = float(np.linalg.norm(embedding_to_use))
                metadata['embedding_norm'] = embedding_norm
                
                near_zero = np.abs(embedding_to_use) < 0.01
                sparsity = float(np.mean(near_zero))
                metadata['embedding_sparsity'] = sparsity
                
                metadata['embedding_dim'] = embedding_to_use.shape[0]
                
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
            metadata['embedding_valid'] = False
        
        return metadata
        
    def _process_identifiers_and_basic_stats(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds memory ID (uuid) and content length if available.
        This should run after base metadata and before final memory entry creation.
        """
        content = context.get('content', '')
        
        if 'length' not in metadata:
            metadata['length'] = len(content)
        
        return metadata
