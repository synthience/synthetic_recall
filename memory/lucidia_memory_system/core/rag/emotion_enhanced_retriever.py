"""
Emotion-Enhanced RAG Retriever

Enhances RAG retrieval by incorporating emotional context in the retrieval process.
Allows for emotionally-aware document ranking and retrieval.
"""

import logging
import asyncio
import json
import websockets
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

class EmotionEnhancedRetriever:
    """Enhances RAG retrieval with emotional context awareness.
    
    Connects to the emotion analyzer service via WebSocket to analyze query and document text
    and incorporates emotional context in the retrieval and ranking process.
    """
    
    def __init__(self, url: str = "ws://localhost:5007", 
                 base_retriever = None,
                 emotion_weight: float = 0.3,
                 emotion_threshold: float = 0.3):
        """Initialize the EmotionEnhancedRetriever.
        
        Args:
            url: WebSocket URL for the emotion analyzer service
            base_retriever: Base retriever to enhance
            emotion_weight: Weight of emotional similarity in ranking (0-1)
            emotion_threshold: Confidence threshold for emotions
        """
        self.url = url
        self.base_retriever = base_retriever
        self.emotion_weight = emotion_weight
        self.emotion_threshold = emotion_threshold
        self.websocket = None
        self.connected = False
        self.emotion_cache = {}  # Cache for emotion analysis results
        logger.info(f"Initialized EmotionEnhancedRetriever, will connect to {url}")
        
    async def connect(self):
        """Connect to the emotion analyzer service."""
        try:
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            logger.info(f"Connected to emotion analyzer service at {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to emotion analyzer: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the emotion analyzer service."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from emotion analyzer service")
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for emotional content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion analysis results
        """
        # Check cache first
        if text in self.emotion_cache:
            return self.emotion_cache[text]
            
        if not self.connected:
            success = await self.connect()
            if not success:
                logger.error("Cannot analyze text: not connected to emotion analyzer")
                return {}
                
        try:
            # Send analysis request
            await self.websocket.send(json.dumps({
                "type": "analyze",
                "text": text,
                "threshold": self.emotion_threshold
            }))
            
            # Get response
            response = await self.websocket.recv()
            result = json.loads(response)
            
            if result.get("type") == "error":
                logger.error(f"Error analyzing text: {result.get('message')}")
                return {}
            
            # Cache result
            self.emotion_cache[text] = result
            return result
        except Exception as e:
            logger.error(f"Error in analyze_text: {e}")
            # Try to reconnect
            self.connected = False
            return {}
    
    def _emotion_vector_from_analysis(self, analysis: Dict[str, Any]) -> np.ndarray:
        """Convert emotion analysis to a vector.
        
        Args:
            analysis: Emotion analysis result
            
        Returns:
            Numpy array representing emotion vector
        """
        # Use primary emotions as a vector
        primary_emotions = analysis.get("primary_emotions", {})
        
        # Create vector [joy, sadness, anger, fear, surprise, neutral, other]
        vector = np.zeros(7)
        
        if primary_emotions:
            vector[0] = primary_emotions.get("joy", 0.0)
            vector[1] = primary_emotions.get("sadness", 0.0)
            vector[2] = primary_emotions.get("anger", 0.0)
            vector[3] = primary_emotions.get("fear", 0.0)
            vector[4] = primary_emotions.get("surprise", 0.0)
            vector[5] = primary_emotions.get("neutral", 0.0)
            vector[6] = primary_emotions.get("other", 0.0)
            
        return vector
    
    def _calculate_emotion_similarity(self, query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
        """Calculate cosine similarity between emotion vectors.
        
        Args:
            query_vector: Query emotion vector
            doc_vector: Document emotion vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Avoid division by zero
        query_norm = np.linalg.norm(query_vector)
        doc_norm = np.linalg.norm(doc_vector)
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
            
        cosine_sim = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
        return float(cosine_sim)
    
    async def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve documents with emotional context awareness.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments for the base retriever
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.base_retriever:
            logger.error("No base retriever provided")
            return []
            
        # First, use the base retriever to get relevant documents
        base_results = await self.base_retriever.retrieve(query, 
                                                          top_k=min(top_k * 2, 20),  # Retrieve more docs for reranking
                                                          **kwargs)
        
        if not base_results:
            return []
            
        # Analyze query for emotions
        query_emotion = await self.analyze_text(query)
        if not query_emotion:
            # If emotion analysis fails, fall back to base retriever
            return base_results[:top_k]
            
        query_emotion_vector = self._emotion_vector_from_analysis(query_emotion)
        
        # Analyze and rerank documents
        enhanced_results = []
        
        for doc in base_results:
            doc_text = doc.get("text", "")
            if not doc_text:
                # If no text, use original score
                enhanced_results.append(doc)
                continue
                
            # Analyze document text
            doc_emotion = await self.analyze_text(doc_text)
            if not doc_emotion:
                # If emotion analysis fails, use original score
                enhanced_results.append(doc)
                continue
                
            doc_emotion_vector = self._emotion_vector_from_analysis(doc_emotion)
            
            # Calculate emotional similarity
            emotion_sim = self._calculate_emotion_similarity(query_emotion_vector, doc_emotion_vector)
            
            # Combine with original score
            original_score = doc.get("score", 0.0)
            combined_score = (1 - self.emotion_weight) * original_score + self.emotion_weight * emotion_sim
            
            # Create enhanced document
            enhanced_doc = doc.copy()
            enhanced_doc["score"] = combined_score
            enhanced_doc["original_score"] = original_score
            enhanced_doc["emotion_similarity"] = emotion_sim
            enhanced_doc["emotions"] = doc_emotion.get("primary_emotions", {})
            enhanced_doc["dominant_emotion"] = doc_emotion.get("dominant_primary", {}).get("emotion")
            
            enhanced_results.append(enhanced_doc)
            
        # Sort by combined score
        enhanced_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Return top_k
        return enhanced_results[:top_k]
    
    async def retrieve_by_emotion(self, query: str, target_emotion: str, 
                              top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve documents that match a specific emotion.
        
        Args:
            query: Query string
            target_emotion: Target emotion to match
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments for the base retriever
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.base_retriever:
            logger.error("No base retriever provided")
            return []
            
        # First, use the base retriever to get relevant documents
        base_results = await self.base_retriever.retrieve(query, 
                                                          top_k=min(top_k * 3, 30),  # Retrieve more docs for filtering
                                                          **kwargs)
        
        if not base_results:
            return []
            
        # Define target emotion vector (one-hot vector for the target emotion)
        target_emotion_vector = np.zeros(7)
        emotion_indices = {"joy": 0, "sadness": 1, "anger": 2, "fear": 3, "surprise": 4, "neutral": 5, "other": 6}
        
        target_idx = emotion_indices.get(target_emotion.lower(), 6)  # Default to "other"
        target_emotion_vector[target_idx] = 1.0
        
        # Analyze and filter documents
        emotion_matched_results = []
        
        for doc in base_results:
            doc_text = doc.get("text", "")
            if not doc_text:
                continue
                
            # Analyze document text
            doc_emotion = await self.analyze_text(doc_text)
            if not doc_emotion:
                continue
                
            doc_emotion_vector = self._emotion_vector_from_analysis(doc_emotion)
            
            # Calculate emotional similarity to target
            emotion_sim = self._calculate_emotion_similarity(target_emotion_vector, doc_emotion_vector)
            
            # Only include if emotional similarity is significant
            if emotion_sim > 0.1:
                # Create enhanced document
                enhanced_doc = doc.copy()
                enhanced_doc["score"] = emotion_sim  # Use emotion similarity as score
                enhanced_doc["original_score"] = doc.get("score", 0.0)
                enhanced_doc["emotion_similarity"] = emotion_sim
                enhanced_doc["emotions"] = doc_emotion.get("primary_emotions", {})
                enhanced_doc["dominant_emotion"] = doc_emotion.get("dominant_primary", {}).get("emotion")
                
                emotion_matched_results.append(enhanced_doc)
            
        # Sort by emotion similarity
        emotion_matched_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Return top_k
        return emotion_matched_results[:top_k]
    
    async def analyze_query_emotion(self, query: str) -> Dict[str, Any]:
        """Analyze the emotional content of a query.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with emotion analysis results
        """
        return await self.analyze_text(query)
    
    async def get_emotional_distribution(self, documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get the emotional distribution across a set of documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary with emotion distribution
        """
        # Initialize distribution
        distribution = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "neutral": 0.0,
            "other": 0.0
        }
        
        if not documents:
            return distribution
            
        count = 0
        
        for doc in documents:
            # Check if emotions already in document
            if "emotions" in doc:
                emotions = doc["emotions"]
                for emotion, score in emotions.items():
                    if emotion in distribution:
                        distribution[emotion] += score
                count += 1
                continue
                
            # Otherwise, analyze document text
            doc_text = doc.get("text", "")
            if not doc_text:
                continue
                
            doc_emotion = await self.analyze_text(doc_text)
            if not doc_emotion:
                continue
                
            primary_emotions = doc_emotion.get("primary_emotions", {})
            for emotion, score in primary_emotions.items():
                if emotion in distribution:
                    distribution[emotion] += score
                    
            count += 1
            
        # Normalize
        if count > 0:
            for emotion in distribution:
                distribution[emotion] /= count
                
        return distribution
