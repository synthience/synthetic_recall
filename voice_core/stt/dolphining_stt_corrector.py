import asyncio
import logging
import json
import base64
import re
import time
import difflib
import random
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, Awaitable
import websockets

logger = logging.getLogger(__name__)

class DolphiningSttCorrector:
    """Implements the Dolphining Framework for corrective STT processing.
    
    This class provides methods to correct STT transcripts using the seven phases
    of the Dolphining Framework:
    1. Dive into Ambiguity - Acknowledging and exploring ambiguity in transcripts
    2. Overlapping Realities - Maintaining multiple interpretations concurrently
    3. Layered Processing - Using multiple processing layers (surface, recursive, dynamic)
    4. Playful Exploration - Creative approaches to resolving ambiguities
    5. Humanistic Precision - Using emotional and relational context
    6. Iterative Adaptation - Learning from past corrections
    7. Networked Intent Discovery - Understanding broader conversational intent
    """
    
    def __init__(self, 
                 memory_client,
                 domain_dictionary: Optional[Dict[str, float]] = None, 
                 confidence_threshold: float = 0.7,
                 max_candidates: int = 5,
                 min_similarity: float = 0.6):
        """Initialize the Dolphining STT corrector.
        
        Args:
            memory_client: EnhancedMemoryClient instance for memory and emotion access
            domain_dictionary: Dictionary of domain-specific terms and their importance weights
            confidence_threshold: Threshold for automatic correction without user clarification
            max_candidates: Maximum number of candidate interpretations to consider
            min_similarity: Minimum similarity threshold for fuzzy matches
        """
        self.memory_client = memory_client
        self.domain_dictionary = domain_dictionary or {}
        self.confidence_threshold = confidence_threshold
        self.max_candidates = max_candidates
        self.min_similarity = min_similarity
        
        # Statistics for iterative adaptation
        self.correction_stats = {
            "total_processed": 0,
            "corrections_made": 0,
            "automatic_corrections": 0,
            "clarifications_needed": 0,
            "correction_patterns": {}
        }
        
        # Cache for previously seen transcripts and their corrections
        self.correction_cache = {}
        
    async def correct_transcript(self, transcript: str) -> Dict[str, Any]:
        """Apply the complete Dolphining Framework to correct a transcript.
        
        This method orchestrates all seven phases of the Dolphining Framework
        to produce the best possible correction of the input transcript.
        
        Args:
            transcript: Raw STT transcript text to correct
            
        Returns:
            Dictionary with correction results including:
            - original: Original transcript
            - corrected: Corrected transcript
            - candidates: List of alternative candidates considered
            - changed: Boolean indicating if a correction was made
            - confidence: Confidence score for the correction
            - needs_clarification: Boolean indicating if user clarification is needed
            - clarification_options: List of options for user clarification if needed
        """
        # Update statistics
        self.correction_stats["total_processed"] += 1
        
        # Check cache first for exact matches to avoid redundant processing
        if transcript in self.correction_cache:
            cached_result = self.correction_cache[transcript]
            logger.info(f"Using cached correction: {transcript} -> {cached_result['corrected']}")
            return cached_result
        
        # Phase 1: Dive into Ambiguity - Generate potential interpretations
        candidates = await self._generate_candidates(transcript)
        
        # Phase 2: Overlapping Realities - Keep multiple interpretations
        # (Already handled by maintaining the candidates list)
        
        # Phase 3: Layered Processing - Score candidates using multiple layers
        scored_candidates = await self._score_candidates(transcript, candidates)
        
        # Phase 4 & 5: Playful Exploration & Humanistic Precision
        # Use emotion and memory context to refine candidate selection
        enhanced_candidates = await self._enhance_with_context(transcript, scored_candidates)
        
        # Sort by final score and get top candidates
        sorted_candidates = sorted(enhanced_candidates, key=lambda x: x['score'], reverse=True)
        top_candidate = sorted_candidates[0] if sorted_candidates else {'text': transcript, 'score': 1.0}
        
        # Prepare the result
        changed = top_candidate['text'] != transcript
        confidence = top_candidate['score']
        needs_clarification = changed and confidence < self.confidence_threshold
        
        # Phase 6: Iterative Adaptation - Update correction statistics
        if changed:
            self.correction_stats["corrections_made"] += 1
            if needs_clarification:
                self.correction_stats["clarifications_needed"] += 1
            else:
                self.correction_stats["automatic_corrections"] += 1
                
            # Record pattern for future reference
            pattern_key = f"{transcript} -> {top_candidate['text']}"
            self.correction_stats["correction_patterns"][pattern_key] = \
                self.correction_stats["correction_patterns"].get(pattern_key, 0) + 1
        
        # Prepare clarification options if needed
        clarification_options = [c['text'] for c in sorted_candidates[:3]] if needs_clarification else []
        
        # Create result object
        result = {
            "original": transcript,
            "corrected": top_candidate['text'],
            "candidates": [c['text'] for c in sorted_candidates[:5]],
            "changed": changed,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "clarification_options": clarification_options,
            "reasoning": top_candidate.get('reasoning', 'No specific reasoning provided.')
        }
        
        # Cache the result
        self.correction_cache[transcript] = result
        
        # Log the correction if made
        if changed:
            logger.info(f"Dolphining correction: '{transcript}' -> '{top_candidate['text']}' (confidence: {confidence:.2f})")
            if needs_clarification:
                logger.info(f"Clarification needed, options: {clarification_options}")
                
        return result
        
    async def _generate_candidates(self, transcript: str) -> List[Dict[str, Any]]:
        """Phase 1: Dive into Ambiguity - Generate candidate interpretations.
        
        This method identifies potential ambiguities in the transcript and 
        generates alternative interpretations.
        
        Args:
            transcript: Raw transcript text
            
        Returns:
            List of candidate dictionaries, each with 'text' and initial 'score'
        """
        candidates = [{'text': transcript, 'score': 1.0, 'source': 'original'}]
        
        # Always include the original transcript as a candidate
        candidates = [{'text': transcript, 'score': 1.0, 'source': 'original'}]
        
        # 1. Domain dictionary matching
        # Check if any domain terms might be misspelled in the transcript
        for domain_term, importance in self.domain_dictionary.items():
            # Simple case-insensitive search
            if domain_term.lower() in transcript.lower():
                continue  # Term already present, no need to correct
                
            # Check for potential misspellings or similar terms
            for word in transcript.split():
                # Skip very short words
                if len(word) <= 2:
                    continue
                    
                # Use difflib to compute string similarity
                similarity = difflib.SequenceMatcher(None, word.lower(), domain_term.lower()).ratio()
                
                if similarity > self.min_similarity:
                    # Create a corrected version by replacing the word
                    corrected = transcript.replace(word, domain_term)
                    # Initial score based on similarity and domain term importance
                    score = similarity * importance
                    candidates.append({
                        'text': corrected,
                        'score': score,
                        'source': 'domain_dictionary',
                        'replaced': word,
                        'replacement': domain_term,
                        'similarity': similarity,
                        'reasoning': f"Domain term '{domain_term}' is similar to '{word}' (similarity: {similarity:.2f})"
                    })
        
        # 2. Common misheard words (could be expanded with a more comprehensive list)
        common_mishearings = {
            'loose idea': 'Lucidia',
            'lucid idea': 'Lucidia',
            'lose ideas': 'Lucidia',
            'lucia': 'Lucidia'
        }
        
        for misheard, correction in common_mishearings.items():
            if misheard.lower() in transcript.lower():
                corrected = re.sub(misheard, correction, transcript, flags=re.IGNORECASE)
                candidates.append({
                    'text': corrected,
                    'score': 0.8,  # Initial score for common mishearings
                    'source': 'common_mishearing',
                    'replaced': misheard,
                    'replacement': correction,
                    'reasoning': f"Common mishearing: '{misheard}' -> '{correction}'"
                })
        
        # 3. Check for other potential mishearings based on phonetic similarity
        # [This would require a more sophisticated phonetic algorithm, simplified here]
        
        # Limit number of candidates
        return candidates[:self.max_candidates] if len(candidates) > self.max_candidates else candidates
    
    async def _score_candidates(self, original: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phase 3: Layered Processing - Score candidates using multiple layers.
        
        This method applies different processing layers to score each candidate:
        - Surface layer: Basic linguistic checks
        - Recursive layer: Memory context relevance
        - Dynamic layer: Alignment with current conversation
        
        Args:
            original: Original transcript text
            candidates: List of candidate dictionaries
            
        Returns:
            Updated list of candidates with refined scores
        """
        # Make a copy to avoid modifying the original list
        scored_candidates = candidates.copy()
        
        # Surface Layer: Check for linguistic patterns, word frequencies, etc.
        # This could be expanded with more sophisticated linguistic analysis
        for candidate in scored_candidates:
            # Slightly favor corrections that are proper nouns (starting with capital letters)
            if candidate['source'] != 'original':
                if 'replacement' in candidate and candidate['replacement'][0].isupper():
                    candidate['score'] *= 1.1
                    candidate['reasoning'] = (candidate.get('reasoning', '') +
                                             "; Proper noun correction favored")
        
        # Recursive Layer: Check relevance to memory context
        try:
            # Try to get memory context relevant to the candidates
            context_query = original
            memory_context = await self.memory_client.generate_rag_context(
                context_query, max_memories=5, include_knowledge_graph=True)
            
            # Check if any domain terms appear in the memory context
            if memory_context:
                for candidate in scored_candidates:
                    if 'replacement' in candidate:
                        # If the replacement term appears in memory context, boost score
                        if candidate['replacement'].lower() in memory_context.lower():
                            candidate['score'] *= 1.2
                            candidate['reasoning'] = (candidate.get('reasoning', '') +
                                                     "; Term appears in memory context")
        except Exception as e:
            logger.warning(f"Error accessing memory context: {e}")
        
        # Dynamic Layer: Check alignment with current conversation emotion/sentiment
        try:
            # Get emotional context if available
            emotional_context = await self.memory_client.get_emotional_context()
            
            if emotional_context and 'current_emotion' in emotional_context:
                current_emotion = emotional_context['current_emotion']
                
                # If current emotion is positive (joy, admiration, etc.), 
                # slightly boost corrections that have positive associations
                positive_emotions = ['joy', 'admiration', 'amusement', 'approval']
                if current_emotion in positive_emotions:
                    for candidate in scored_candidates:
                        if candidate['source'] == 'domain_dictionary' and 'replacement' in candidate:
                            # Get importance from domain dictionary
                            importance = self.domain_dictionary.get(candidate['replacement'], 0.5)
                            # Boost based on importance and emotional context
                            emotion_boost = 1.0 + (importance * 0.2)
                            candidate['score'] *= emotion_boost
                            candidate['reasoning'] = (candidate.get('reasoning', '') +
                                                     f"; Boosted by positive emotion {current_emotion}")
        except Exception as e:
            logger.warning(f"Error accessing emotional context: {e}")
            
        return scored_candidates
    
    async def _enhance_with_context(self, original: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phases 4 & 5: Playful Exploration & Humanistic Precision.
        
        This method uses emotion, memory, and creative approaches to 
        refine candidate selection.
        
        Args:
            original: Original transcript text
            candidates: List of candidate dictionaries with initial scores
            
        Returns:
            Enhanced list of candidates with updated scores
        """
        # Make a copy to avoid modifying the original list
        enhanced_candidates = candidates.copy()
        
        # Try to detect emotion in the original text
        try:
            # If roberta-base-go_emotions is available through our client
            emotion_result = await self.memory_client.detect_emotional_context(original)
            
            # Get the sentiment (positive/negative) and adjust scores
            sentiment = emotion_result.get('sentiment', 0)
            
            # Adjust based on sentiment - favor brand/domain terms more when sentiment is positive
            for candidate in enhanced_candidates:
                if candidate['source'] == 'domain_dictionary' or candidate['source'] == 'common_mishearing':
                    # If sentiment is positive, boost domain term corrections
                    if sentiment > 0.3:  # Moderately positive
                        sentiment_boost = 1.0 + (sentiment * 0.3)  # Up to 30% boost for very positive
                        candidate['score'] *= sentiment_boost
                        candidate['reasoning'] = (candidate.get('reasoning', '') +
                                                f"; Boosted by positive sentiment ({sentiment:.2f})")
        except Exception as e:
            logger.warning(f"Error analyzing emotion: {e}")
        
        # Phase 7: Networked Intent Discovery
        # Check if other recent conversations have similar patterns
        try:
            # Check correction history for patterns
            for pattern, count in self.correction_stats["correction_patterns"].items():
                if pattern.startswith(original):
                    # This exact correction has been made before
                    correction = pattern.split(' -> ')[1]
                    
                    # Find the candidate with this correction
                    for candidate in enhanced_candidates:
                        if candidate['text'] == correction:
                            # Boost based on how many times this correction has been made
                            history_boost = min(1.0 + (count * 0.05), 1.3)  # Cap at 30% boost
                            candidate['score'] *= history_boost
                            candidate['reasoning'] = (candidate.get('reasoning', '') +
                                                     f"; Seen {count} times before")
        except Exception as e:
            logger.warning(f"Error checking correction history: {e}")
            
        return enhanced_candidates
    
    async def process_with_websocket_stt(self, audio_bytes: bytes, stt_url: str = "ws://stt_transcription:5002/ws/transcribe") -> Dict[str, Any]:
        """Process audio with WebSocket STT service, then apply Dolphining correction.
        
        Args:
            audio_bytes: Audio data as bytes
            stt_url: WebSocket URL for STT service
            
        Returns:
            Dictionary with correction results
        """
        try:
            # Call the STT service via WebSocket
            raw_transcript = await self._call_websocket_stt(audio_bytes, stt_url)
            
            # Apply Dolphining correction to the transcript
            correction_result = await self.correct_transcript(raw_transcript)
            
            # Add the raw transcript to the result
            correction_result["raw_transcript"] = raw_transcript
            
            return correction_result
            
        except Exception as e:
            logger.error(f"Error in process_with_websocket_stt: {e}")
            # Return a basic error result
            return {
                "original": "",
                "corrected": "",
                "candidates": [],
                "changed": False,
                "confidence": 0.0,
                "needs_clarification": False,
                "clarification_options": [],
                "error": str(e)
            }
    
    async def _call_websocket_stt(self, audio_bytes: bytes, stt_url: str) -> str:
        """Call the STT service via WebSocket.
        
        Args:
            audio_bytes: Audio data as bytes
            stt_url: WebSocket URL for STT service
            
        Returns:
            Raw transcript text from STT service
        """
        try:
            uri = stt_url
            async with websockets.connect(uri) as ws:
                # Prepare the audio data
                payload = {
                    "command": "transcribe",
                    "audio_data": base64.b64encode(audio_bytes).decode("utf-8")
                }
                
                # Send the audio data
                await ws.send(json.dumps(payload))
                
                # Get the transcription result
                raw_response = await ws.recv()
                response = json.loads(raw_response)
                
                # Extract the transcription text
                transcription = response.get("transcription", "")
                
                return transcription
                
        except Exception as e:
            logger.error(f"Error in _call_websocket_stt: {e}")
            raise
    
    def update_domain_dictionary(self, new_terms: Dict[str, float]):
        """Update the domain dictionary with new terms.
        
        Args:
            new_terms: Dictionary of new terms and their importance weights
        """
        self.domain_dictionary.update(new_terms)
        logger.info(f"Updated domain dictionary with {len(new_terms)} new terms")
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get statistics about corrections made by the Dolphining system.
        
        Returns:
            Dictionary with correction statistics
        """
        return self.correction_stats
    
    def feedback_correction(self, original: str, correction: str, accepted: bool):
        """Provide feedback on a correction for iterative adaptation.
        
        This allows the system to learn from user feedback on corrections.
        
        Args:
            original: Original transcript text
            correction: Corrected text
            accepted: Whether the correction was accepted by the user
        """
        if accepted:
            # If the correction was accepted, reinforce this pattern
            pattern_key = f"{original} -> {correction}"
            self.correction_stats["correction_patterns"][pattern_key] = \
                self.correction_stats["correction_patterns"].get(pattern_key, 0) + 2  # Double reinforcement
            
            # Add to cache for immediate use
            self.correction_cache[original] = {
                "original": original,
                "corrected": correction,
                "candidates": [correction],
                "changed": original != correction,
                "confidence": 1.0,  # High confidence since user confirmed
                "needs_clarification": False
            }
            
            logger.info(f"Correction feedback: '{original}' -> '{correction}' (accepted)")
        else:
            # If rejected, remove from patterns or decrease confidence
            pattern_key = f"{original} -> {correction}"
            if pattern_key in self.correction_stats["correction_patterns"]:
                if self.correction_stats["correction_patterns"][pattern_key] > 1:
                    self.correction_stats["correction_patterns"][pattern_key] -= 1
                else:
                    del self.correction_stats["correction_patterns"][pattern_key]
            
            # Remove from cache
            if original in self.correction_cache:
                del self.correction_cache[original]
                
            logger.info(f"Correction feedback: '{original}' -> '{correction}' (rejected)")
