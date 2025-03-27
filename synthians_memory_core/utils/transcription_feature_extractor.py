import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio

from ..custom_logger import logger

class TranscriptionFeatureExtractor:
    """
    Extracts emotion and semantic features from transcribed voice input.
    Uses an emotion analyzer and optional keyword extractor to enrich transcription metadata.
    
    This class is designed to work with the EmotionAnalyzer and KeyBERT, but can be
    used with any compatible analyzers that follow the same interface.
    """

    def __init__(self, emotion_analyzer, keyword_extractor=None, config: Optional[Dict] = None):
        self.emotion_analyzer = emotion_analyzer  # EmotionAnalyzer instance
        self.keyword_extractor = keyword_extractor  # KeyBERT or similar
        self.config = config or {}
        
        # Default configuration with fallbacks
        self.top_n_keywords = self.config.get('top_n_keywords', 5)
        self.min_keyword_score = self.config.get('min_keyword_score', 0.3)
        self.include_ngrams = self.config.get('include_ngrams', True)
        
        logger.info("TranscriptionFeatureExtractor", "Initialized with" + 
                   f" emotion_analyzer={emotion_analyzer is not None}" +
                   f" keyword_extractor={keyword_extractor is not None}")
        
        # Lazy-load KeyBERT if not provided but needed
        self._keybert = None
    
    async def extract_features(self, transcript: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract features from a transcript and return them as metadata.
        
        Args:
            transcript: The text transcript to analyze
            meta: Optional metadata about the audio (duration, etc.)
            
        Returns:
            A dictionary of extracted features suitable for metadata
        """
        if not transcript or not isinstance(transcript, str) or len(transcript.strip()) == 0:
            logger.warning("TranscriptionFeatureExtractor", "Empty or invalid transcript provided")
            return {"input_modality": "spoken", "source": "transcription", "error": "Empty transcript"}
        
        metadata = {}
        
        # Tag basic information about the input
        metadata["input_modality"] = "spoken"
        metadata["source"] = "transcription"
        metadata["word_count"] = len(transcript.split())
        
        # 1. Emotion Analysis
        emotion_features = await self._extract_emotion_features(transcript)
        if emotion_features:
            metadata.update(emotion_features)
        
        # 2. Keyword Extraction
        keyword_features = await self._extract_keyword_features(transcript)
        if keyword_features:
            metadata.update(keyword_features)
            
        # 3. Speech Metadata
        if meta:
            speech_features = self._extract_speech_features(transcript, meta)
            if speech_features:
                metadata.update(speech_features)
        
        logger.info("TranscriptionFeatureExtractor", 
                   f"Extracted {len(metadata)} features from transcript")
        return metadata
    
    async def _extract_emotion_features(self, text: str) -> Dict[str, Any]:
        """
        Extract emotion features using the emotion analyzer.
        """
        features = {}
        
        if self.emotion_analyzer is None:
            logger.warning("TranscriptionFeatureExtractor", "No emotion analyzer available")
            return features
        
        try:
            # Use our emotion analyzer to get emotion data
            emotion = await self.emotion_analyzer.analyze(text)
            
            # Extract the core emotional features
            features["dominant_emotion"] = emotion.get("dominant_emotion", "neutral")
            features["emotions"] = emotion.get("emotions", {})
            
            # Calculate derived features
            if "emotions" in emotion and emotion["emotions"]:
                # Get intensity (highest emotion score)
                features["intensity"] = max(emotion["emotions"].values())
                
                # Calculate sentiment value (-1 to 1 scale)
                pos_emotions = ["joy", "happiness", "excitement", "love", "optimism", "admiration"]
                neg_emotions = ["sadness", "anger", "fear", "disgust", "disappointment"]
                
                sentiment = 0.0
                for emotion_name, score in emotion["emotions"].items():
                    if emotion_name in pos_emotions:
                        sentiment += score
                    elif emotion_name in neg_emotions:
                        sentiment -= score
                
                # Normalize to [-1, 1]
                features["sentiment_value"] = max(min(sentiment, 1.0), -1.0)
            else:
                features["intensity"] = 0.5
                features["sentiment_value"] = 0.0
            
            # Create emotional_context for compatibility with other systems
            features["emotional_context"] = {
                "dominant_emotion": features["dominant_emotion"],
                "emotions": features["emotions"],
                "intensity": features["intensity"],
                "sentiment_value": features["sentiment_value"]
            }
            
        except Exception as e:
            logger.error("TranscriptionFeatureExtractor", f"Error in emotion analysis: {str(e)}")
            features["dominant_emotion"] = "neutral"
            features["intensity"] = 0.5
            features["sentiment_value"] = 0.0
        
        return features
    
    async def _extract_keyword_features(self, text: str) -> Dict[str, Any]:
        """
        Extract keyword features using KeyBERT or a similar keyword extractor.
        Lazy-loads KeyBERT if needed and not provided.
        """
        features = {}
        
        # Ensure we have a keyword extractor
        if self.keyword_extractor is None:
            # Try to lazy-load KeyBERT if possible
            if self._keybert is None:
                try:
                    loop = asyncio.get_event_loop()
                    self._keybert = await loop.run_in_executor(None, self._load_keybert)
                    if self._keybert is None:
                        logger.warning("TranscriptionFeatureExtractor", "Failed to load KeyBERT")
                        return features
                except Exception as e:
                    logger.error("TranscriptionFeatureExtractor", f"Error loading KeyBERT: {str(e)}")
                    return features
            
            # Use the lazy-loaded KeyBERT
            self.keyword_extractor = self._keybert
        
        # Extract keywords if we have an extractor
        if self.keyword_extractor:
            try:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                keywords = await loop.run_in_executor(
                    None, 
                    lambda: self.keyword_extractor.extract_keywords(
                        text, 
                        top_n=self.top_n_keywords,
                        keyphrase_ngram_range=(1, 3) if self.include_ngrams else (1, 1),
                        stop_words='english',
                        use_mmr=True,
                        diversity=0.7
                    )
                )
                
                # Filter by minimum score
                keywords = [(kw, score) for kw, score in keywords if score >= self.min_keyword_score]
                
                # Save as separate lists for keywords and scores
                features["keywords"] = [kw for kw, _ in keywords]
                features["keyword_scores"] = {kw: score for kw, score in keywords}
                
                # Also save as topic tags for compatibility
                features["topic_tags"] = features["keywords"][:3] if len(features["keywords"]) > 3 else features["keywords"]
                
            except Exception as e:
                logger.error("TranscriptionFeatureExtractor", f"Error extracting keywords: {str(e)}")
                features["keywords"] = []
                features["topic_tags"] = []
        
        return features
    
    def _extract_speech_features(self, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features related to speech patterns from metadata.
        """
        features = {}
        
        # Extract duration and calculate speaking rate
        duration = meta.get("duration_sec", None)
        if duration is not None and duration > 0:
            word_count = len(text.split())
            features["speaking_rate"] = round(word_count / duration, 2)  # words per second
            features["duration_sec"] = round(duration, 2)
        
        # Add interruption metadata if available
        features["user_interruptions"] = meta.get("user_interruptions", 0)
        features["was_interrupted"] = meta.get("was_interrupted", False)
        
        # Add timestamps if available
        if "interruption_timestamps" in meta and isinstance(meta["interruption_timestamps"], list):
            features["interruption_timestamps"] = meta["interruption_timestamps"]
            
        # Add conversation flow metrics
        if features["was_interrupted"]:
            # Flag for reflection triggers during retrieval
            features["requires_reflection"] = True
            
            # Add analysis of interruption severity
            if features["user_interruptions"] > 5:
                features["interruption_severity"] = "high"
            elif features["user_interruptions"] > 2:
                features["interruption_severity"] = "medium"
            else:
                features["interruption_severity"] = "low"
        
        # Add other speech-related metadata if available
        for key in ["speaker_id", "confidence", "language", "timestamp", "session_id"]:
            if key in meta:
                features[key] = meta[key]
        
        return features
    
    def _load_keybert(self):
        """
        Attempt to lazy-load KeyBERT if it's available.
        Returns None if KeyBERT can't be loaded.
        """
        try:
            from keybert import KeyBERT
            logger.info("TranscriptionFeatureExtractor", "Lazy-loading KeyBERT")
            return KeyBERT()
        except ImportError:
            logger.warning("TranscriptionFeatureExtractor", 
                         "KeyBERT not installed. Install with: pip install keybert")
            return None
