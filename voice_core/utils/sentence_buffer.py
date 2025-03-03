from __future__ import annotations
import re
import time
from typing import Optional, List, Dict, Any, Deque
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)

class SentenceBuffer:
    """
    Enhanced SentenceBuffer that manages partial transcriptions and sentence chunking 
    for more natural conversation flow with improved text normalization and context awareness.
    """
    
    def __init__(self, 
                 max_buffer_time: float = 5.0,
                 min_words_for_chunk: int = 3,
                 end_of_sentence_timeout: float = 1.0,
                 max_history_size: int = 10,
                 confidence_threshold: float = 0.7):
        """
        Initialize the sentence buffer with configurable parameters.
        
        Args:
            max_buffer_time: Maximum time (in seconds) to buffer text before forcing processing
            min_words_for_chunk: Minimum number of words required to process a chunk
            end_of_sentence_timeout: Time (in seconds) after which to consider a sentence complete
            max_history_size: Maximum number of processed sentences to keep in history
            confidence_threshold: Minimum confidence score for accepting transcripts
        """
        self.buffer = []
        self.last_update_time = 0
        self.max_buffer_time = max_buffer_time
        self.min_words_for_chunk = min_words_for_chunk
        self.end_of_sentence_timeout = end_of_sentence_timeout
        self.confidence_threshold = confidence_threshold
        
        # Enhanced sentence boundary detection
        self.sentence_endings = re.compile(r'[.!?][\s"\')\]]?$|$')
        self.question_pattern = re.compile(r'\b(who|what|when|where|why|how|is|are|was|were|will|do|does|did|can|could|would|should|may|might)\b', re.IGNORECASE)
        
        # Track processed sentences for context
        self.processed_history: Deque[Dict[str, Any]] = deque(maxlen=max_history_size)
        
        # Performance metrics
        self.metrics = {
            "chunks_processed": 0,
            "sentences_completed": 0,
            "avg_sentence_length": 0,
            "total_processing_time": 0
        }
        
        # Additional filler words and hesitation sounds
        self.fillers = {
            'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'so', 'basically',
            'actually', 'literally', 'well', 'right', 'okay', 'hmm', 'mmm'
        }
        
        # Common speech recognition errors to correct
        self.common_corrections = {
            "i'm gonna": "I'm going to",
            "i gotta": "I've got to",
            "wanna": "want to",
            "kinda": "kind of",
            "lemme": "let me",
            "gimme": "give me",
            "dunno": "don't know"
        }
        
        self.logger = logging.getLogger(__name__)
        
    def add_transcript(self, text: str, confidence: float = 1.0) -> Optional[str]:
        """
        Add a new transcript chunk and return a complete sentence if available.
        
        Args:
            text: The transcript text to add
            confidence: Confidence score (0-1) for this transcript
            
        Returns:
            Completed sentence if available, None otherwise
        """
        start_time = time.time()
        current_time = start_time
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            self.logger.debug(f"Transcript below confidence threshold: {confidence:.2f} < {self.confidence_threshold:.2f}")
            return None
        
        # Clean and normalize the text
        text = text.strip().lower()
        if not text:
            return None
            
        # Check if this is a repeat of the last chunk
        if self.buffer and text == self.buffer[-1]['text']:
            self.logger.debug("Duplicate transcript chunk detected, skipping")
            return None
            
        # Add new chunk to buffer
        self.buffer.append({
            'text': text,
            'timestamp': current_time,
            'confidence': confidence
        })
        
        self.metrics["chunks_processed"] += 1
        self.last_update_time = current_time
        
        # Try to form a complete sentence
        result = self._process_buffer(current_time)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.metrics["total_processing_time"] += processing_time
        self.logger.debug(f"Transcript processing time: {processing_time:.3f}s")
        
        return result
    
    def _process_buffer(self, current_time: float) -> Optional[str]:
        """
        Process buffer to find complete sentences with enhanced detection rules.
        
        Args:
            current_time: Current time for timeout calculation
            
        Returns:
            Completed sentence if available, None otherwise
        """
        if not self.buffer:
            return None
            
        # Join all chunks
        full_text = ' '.join(chunk['text'] for chunk in self.buffer)
        words = full_text.split()
        
        # Calculate average confidence
        avg_confidence = sum(chunk.get('confidence', 1.0) for chunk in self.buffer) / len(self.buffer)
        
        # Enhanced conditions for processing the buffer
        should_process = (
            # Natural sentence ending
            bool(self.sentence_endings.search(full_text)) or
            
            # Question detection (more likely to be a complete thought)
            bool(self.question_pattern.search(full_text) and len(words) >= 4) or
            
            # Enough words and time gap
            (len(words) >= self.min_words_for_chunk and 
             current_time - self.buffer[0]['timestamp'] > self.end_of_sentence_timeout) or
            
            # Buffer timeout
            (current_time - self.buffer[0]['timestamp'] > self.max_buffer_time) or
            
            # High confidence and sufficient length
            (avg_confidence > 0.9 and len(words) >= self.min_words_for_chunk * 2)
        )
        
        if should_process:
            # Clean up the text
            result = self._clean_text(full_text)
            
            # Add to processed history
            self.processed_history.append({
                'text': result,
                'timestamp': current_time,
                'word_count': len(result.split()),
                'confidence': avg_confidence,
                'chunks': len(self.buffer)
            })
            
            # Update metrics
            self.metrics["sentences_completed"] += 1
            total_words = sum(len(entry['text'].split()) for entry in self.processed_history)
            if self.metrics["sentences_completed"] > 0:
                self.metrics["avg_sentence_length"] = total_words / self.metrics["sentences_completed"]
            
            # Clear the buffer for next sentence
            self.buffer.clear()
            return result
            
        return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize the transcribed text with enhanced processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        # Original text for logging
        original = text
        
        # Remove filler words and hesitation sounds
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Skip filler words
            if word.lower() in self.fillers:
                continue
                
            # Apply common corrections
            corrected = False
            for error, correction in self.common_corrections.items():
                if word.lower() == error or f"{word.lower()} " == error:
                    if not cleaned_words:  # If first word, capitalize correction
                        cleaned_words.append(correction)
                    else:
                        cleaned_words.append(correction.lower())
                    corrected = True
                    break
                    
            if not corrected:
                cleaned_words.append(word)
        
        # Join words and ensure proper spacing around punctuation
        text = ' '.join(cleaned_words)
        text = re.sub(r'\s+([.,!?:;])', r'\1', text)
        
        # Add sentence ending if missing
        if not re.search(r'[.!?]$', text):
            # Add question mark if it looks like a question
            if self.question_pattern.search(text):
                text += '?'
            else:
                text += '.'
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
            
        if original != text:
            self.logger.debug(f"Text cleaned: '{original}' → '{text}'")
            
        return text
    
    def clear(self) -> None:
        """Clear the buffer and reset processing state."""
        self.buffer.clear()
        self.last_update_time = 0
        
    def get_partial_transcript(self) -> str:
        """
        Get the current partial transcript without clearing the buffer.
        
        Returns:
            Current partial transcript as a single string
        """
        if not self.buffer:
            return ""
        return ' '.join(chunk['text'] for chunk in self.buffer)
    
    def get_context(self, max_sentences: int = 3) -> str:
        """
        Get recent conversation context from processed history.
        
        Args:
            max_sentences: Maximum number of recent sentences to include
            
        Returns:
            Recent conversation context as a string
        """
        context = [entry['text'] for entry in list(self.processed_history)[-max_sentences:]]
        return " ".join(context)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the sentence buffer.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate average processing time
        if self.metrics["chunks_processed"] > 0:
            avg_processing_time = self.metrics["total_processing_time"] / self.metrics["chunks_processed"]
        else:
            avg_processing_time = 0
            
        return {
            **self.metrics,
            "buffer_size": len(self.buffer),
            "history_size": len(self.processed_history),
            "avg_processing_time": avg_processing_time
        }
    
    def to_json(self) -> str:
        """
        Convert current buffer state to JSON for debugging or UI display.
        
        Returns:
            JSON representation of current buffer state
        """
        state = {
            "buffer": self.buffer,
            "history": list(self.processed_history),
            "metrics": self.get_metrics(),
            "partial": self.get_partial_transcript()
        }
        return json.dumps(state, indent=2)
    
    def __len__(self) -> int:
        """Return the number of chunks in the buffer."""
        return len(self.buffer)
    
    def __bool__(self) -> bool:
        """Return True if the buffer has content."""
        return bool(self.buffer)