import asyncio
import os
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

from .custom_logger import logger

class EmotionAnalyzer:
    """
    Handles emotion analysis using a dual-mode approach:
    1. Primary: RoBERTa-based GoEmotions transformer model
    2. Fallback: Lightweight keyword-based approach
    
    Ensures consistent emotion detection structure regardless of the mode used.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the EmotionAnalyzer with a transformer model if available.
        
        Args:
            model_path: Path to the emotion model, if None will check for environment variable
            device: Device to use for inference (cuda, cpu). If None, will auto-detect.
        """
        # Auto-detect device if not specified
        if device is None:
            # Check for CUDA availability at runtime - default to CPU if not available
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("EmotionAnalyzer", f"Auto-detected device: {self.device}")
            except ImportError:
                self.device = "cpu"
                logger.info("EmotionAnalyzer", "Torch not available, defaulting to CPU device")
        else:
            self.device = device
            
        # Model path can come from multiple sources with increasing precedence:
        # 1. Default path relative to the project
        # 2. Environment variable EMOTION_MODEL_PATH
        # 3. Explicitly provided model_path parameter
        default_paths = [
            "models/roberta-base-go_emotions",  # Default relative path
            "/app/models/emotion",             # Common Docker mount point
            "/data/models/emotion",            # Alternative Docker volume
        ]
        
        # Determine the model path with proper precedence
        env_path = os.environ.get("EMOTION_MODEL_PATH")
        self.model_path = model_path or env_path or next((p for p in default_paths if os.path.exists(p)), default_paths[0])
        logger.info("EmotionAnalyzer", f"Using model path: {self.model_path}")
        
        # Model will be loaded on first use, not during initialization
        self.model = None
        self.model_loaded = False
        self.model_load_attempted = False
        
        # Track analysis stats
        self.stats = {
            "primary_calls": 0,
            "fallback_calls": 0,
            "errors": 0,
            "avg_time_ms": 0,
            "total_calls": 0
        }
    
    def _initialize_model(self):
        """
        Load the transformer-based emotion model if available.
        Returns True if model loaded successfully, False otherwise.
        """
        # Skip if we've already attempted to load and failed
        if self.model_loaded:
            return True
            
        if self.model_load_attempted and not self.model_loaded:
            logger.debug("EmotionAnalyzer", "Previous model load attempt failed, using fallback")
            return False
            
        self.model_load_attempted = True
        
        try:
            # Only import transformers if we're actually going to use it
            from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            # Check both if the path exists AND if it contains expected model files
            path_exists = os.path.exists(self.model_path)
            model_files_exist = False
            
            if path_exists:
                # Check for key files that indicate a Hugging Face model
                expected_files = ['config.json', 'pytorch_model.bin']
                model_files_exist = any(os.path.exists(os.path.join(self.model_path, f)) for f in expected_files)
                
            # Log what we found about the model path
            if path_exists and model_files_exist:
                logger.info("EmotionAnalyzer", f"Found model files at {self.model_path}")
            elif path_exists:
                logger.warning("EmotionAnalyzer", f"Path {self.model_path} exists but doesn't contain model files")
            else:
                logger.warning("EmotionAnalyzer", f"Model path {self.model_path} does not exist")
            
            # If model files exist locally, use them; otherwise try to download from Hugging Face Hub
            if path_exists and model_files_exist:
                # Load from local path
                logger.info("EmotionAnalyzer", f"Loading local model from {self.model_path}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_path, local_files_only=True)
            else:
                # Try to download model from Hugging Face Hub
                try:
                    logger.info("EmotionAnalyzer", "Local model not found, downloading from Hugging Face Hub")
                    # Use a fallback model ID - GoEmotions on Hugging Face
                    model_id = "joeddav/distilbert-base-uncased-go-emotions-student"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForSequenceClassification.from_pretrained(model_id)
                    
                    # Save the model to the specified path for future use
                    if path_exists:
                        logger.info("EmotionAnalyzer", f"Saving downloaded model to {self.model_path}")
                        model.save_pretrained(self.model_path)
                        tokenizer.save_pretrained(self.model_path)
                except Exception as download_error:
                    logger.error("EmotionAnalyzer", f"Error downloading model: {str(download_error)}")
                    return False
            
            # Create the pipeline with the loaded model
            self.model = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
                top_k=None  # Return all emotion scores
            )
            
            self.model_loaded = True
            logger.info("EmotionAnalyzer", "Emotion model loaded successfully")
            return True
            
        except Exception as e:
            logger.error("EmotionAnalyzer", f"Error loading emotion model: {str(e)}")
            self.model = None
            self.model_loaded = False
            self.stats["errors"] += 1
            return False
    
    async def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotions in the given text.
        Attempts to use the transformer model first, and falls back to keyword analysis if needed.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing emotions and the dominant emotion
        """
        start_time = time.time()
        
        try:
            # Try to load the model on first use if not already loaded
            if not self.model and not self.model_load_attempted:
                logger.info("EmotionAnalyzer", "First-time model loading during analyze call")
                model_loaded = self._initialize_model()
                if model_loaded:
                    logger.info("EmotionAnalyzer", "Successfully loaded model on first use")
                else:
                    logger.warning("EmotionAnalyzer", "Failed to load model on first use, falling back to keywords")
            
            # Attempt primary analysis if model is available
            if self.model is not None:
                logger.debug("EmotionAnalyzer", "Using transformer-based analysis")
                result = await self._analyze_with_transformer(text)
                self.stats["primary_calls"] += 1
            else:
                # Fall back to keyword analysis
                logger.debug("EmotionAnalyzer", "Using keyword-based analysis fallback")
                result = await self._analyze_with_keywords(text)
                self.stats["fallback_calls"] += 1
            
            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["avg_time_ms"] = (
                (self.stats["avg_time_ms"] * (self.stats["primary_calls"] + self.stats["fallback_calls"] - 1) + elapsed_ms) /
                (self.stats["primary_calls"] + self.stats["fallback_calls"])
            )
            self.stats["total_calls"] += 1
            
            return result
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error("EmotionAnalyzer", f"Error in emotion analysis: {str(e)}")
            self.stats["errors"] += 1
            
            # Always return a valid response, even in case of errors
            return {
                "dominant_emotion": "neutral",
                "emotions": {"neutral": 1.0},
                "error": str(e)
            }
    
    async def _analyze_with_transformer(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotions using the transformer model.
        """
        # Execute the model in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(None, lambda: self.model(text))
        
        # Convert the transformer output format to our expected format
        # The model returns a list of dictionaries with 'label' and 'score'
        emotion_results = {}
        for result_list in raw_results:
            for item in result_list:
                label = item['label']
                score = float(item['score'])  # Ensure score is float
                emotion_results[label] = score
        
        # Find the dominant emotion based on score
        if emotion_results:
            dominant_emotion = max(emotion_results.items(), key=lambda x: x[1])[0]
        else:
            dominant_emotion = "neutral"
            emotion_results["neutral"] = 0.5
        
        return {
            "emotions": emotion_results,
            "dominant_emotion": dominant_emotion
        }
    
    async def _analyze_with_keywords(self, text: str) -> Dict[str, Any]:
        """
        Fallback emotion analysis using keyword matching.
        Much less accurate but works without any models.
        """
        # Simple keyword-based emotion detection
        emotion_keywords = {
            "joy": ["happy", "joy", "delighted", "glad", "pleased", "excited", "thrilled"],
            "sadness": ["sad", "unhappy", "depressed", "down", "miserable", "upset", "disappointed"],
            "anger": ["angry", "mad", "furious", "annoyed", "irritated", "enraged", "frustrated"],
            "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned"],
            "disgust": ["disgusted", "repulsed", "revolted", "sickened"],
            "neutral": ["ok", "fine", "neutral", "average", "normal"]
        }
        
        text = text.lower()
        emotion_scores = {emotion: 0.1 for emotion in emotion_keywords}  # Base score
        
        # Simple keyword matching
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotion_scores[emotion] += 0.15  # Increment score for each match
        
        # Normalize scores
        max_score = max(emotion_scores.values())
        if max_score > 0.1:  # If we found any matches
            for emotion in emotion_scores:
                emotion_scores[emotion] = min(emotion_scores[emotion] / max_score, 1.0)
        else:
            # If no matches, default to neutral
            emotion_scores["neutral"] = 0.5
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "emotions": emotion_scores,
            "dominant_emotion": dominant_emotion
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for the emotion analyzer.
        """
        total_calls = self.stats["primary_calls"] + self.stats["fallback_calls"]
        
        return {
            "total_calls": self.stats["total_calls"],
            "primary_calls": self.stats["primary_calls"],
            "fallback_calls": self.stats["fallback_calls"],
            "primary_percentage": (self.stats["primary_calls"] / max(total_calls, 1)) * 100,
            "fallback_percentage": (self.stats["fallback_calls"] / max(total_calls, 1)) * 100,
            "errors": self.stats["errors"],
            "avg_time_ms": round(self.stats["avg_time_ms"], 2),
            "model_loaded": self.model_loaded,
            "model_path": self.model_path
        }
